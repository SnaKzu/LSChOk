"""Backend Flask para Reconocimiento de Lenguaje de Señas.

Adaptado a:
- Etiquetas dinámicas desde labels.json (mismo orden que entrenamiento)
- Secuencias de 30 frames
- Extracción de landmarks consistente (si landmarks_normalization está disponible)
"""

import os
import sys
import json
import time
import base64
import logging
import numpy as np
import cv2
from datetime import datetime
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from collections import deque

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear Flask app
app = Flask(__name__, 
            static_folder='../frontend',
            static_url_path='',
            template_folder='../frontend')

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'senas-railway-key-2024')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///lsp_users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)

# Importar modelos y configurar DB
from models import db, User, Prediction, init_db

# Inicializar base de datos
db.init_app(app)

# Configurar Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Por favor inicia sesión para acceder a esta página.'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Configurar SocketIO
socketio = SocketIO(app, 
                    cors_allowed_origins="*",
                    async_mode='threading',
                    logger=False,
                    engineio_logger=False,
                    ping_timeout=60,
                    ping_interval=25)

# Variables globales del modelo
modelo = None
mp_hands = None
mp_drawing = None
hands = None
PALABRAS = []  # label_ids (ordenadas) para el modelo
LABELS_MAP = {}  # label_id -> etiqueta humana
FRAMES_POR_SECUENCIA = 30
ML_ENABLED = False

# --- Anti falsos positivos (igual que InferenciaSeñas.py) ---
CONFIDENCE_THRESHOLD = 0.90
MARGIN_THRESHOLD = 0.15
STABLE_PRED_FRAMES = 8
DISPLAY_HOLD_SECONDS = 0.8

REQUIRE_MOTION = True
MOTION_WINDOW = 12
MIN_MOTION = 0.004
MOTION_ARM_SECONDS = 0.6

# Normalización de landmarks (opcional, si existe en el repo)
NORMALIZATION_AVAILABLE = False
try:
    from landmarks_normalization import frame_from_mediapipe_results

    NORMALIZATION_AVAILABLE = True
except Exception:
    frame_from_mediapipe_results = None  # type: ignore
    NORMALIZATION_AVAILABLE = False


def _get_data_root() -> str:
    """Resolve dataset root for labels.json.

    Priority:
    1) env var LSCH_DATA_DIR
    2) repo_root/ModeloML/Scripts/DataLS
    3) web-app/DataLS
    """
    env = os.environ.get("LSCH_DATA_DIR")
    if env:
        return env

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidate = os.path.join(repo_root, "ModeloML", "Scripts", "DataLS")
    if os.path.isdir(candidate):
        return candidate

    web_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(web_root, "DataLS")


def _load_labels() -> tuple[list[str], dict[str, str]]:
    """Load labels.json and return sorted label IDs + mapping to human labels."""
    data_root = _get_data_root()
    labels_path = os.path.join(data_root, "labels.json")
    if not os.path.exists(labels_path):
        return [], {}

    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if not isinstance(labels, dict) or not labels:
            return [], {}
        label_ids = sorted(labels.keys())
        return label_ids, {str(k): str(v) for k, v in labels.items()}
    except Exception as e:
        logger.warning(f"labels.json inválido: {labels_path} ({e})")
        return [], {}

# Sesiones de usuarios
user_sessions = {}

class UserSession:
    """Sesión de usuario con buffer de frames"""
    def __init__(self, session_id):
        self.session_id = session_id
        self.buffer = deque(maxlen=FRAMES_POR_SECUENCIA)
        self.last_prediction = None
        self.prediction_count = 0
        self.start_time = datetime.now()
        self.frame_counter = 0  # Contador secuencial
        self.last_process_time = 0  # Último tiempo de procesamiento
        self.min_frame_interval = 0.033  # ~30 FPS

        # Estado para estabilidad temporal (gating)
        self.pred_history = deque(maxlen=STABLE_PRED_FRAMES)
        self.last_shown_at = 0.0

        # "Arming" por movimiento: si hubo movimiento reciente, permitimos clasificar
        # incluso si en los últimos frames el movimiento baja (fin de seña).
        self.motion_armed_until = 0.0
        
    def add_frame(self, frame_data):
        """Agregar frame al buffer"""
        self.buffer.append(frame_data)
        self.frame_counter += 1
        
    def clear_buffer(self):
        """Limpiar buffer"""
        self.buffer.clear()
        self.frame_counter = 0
        self.pred_history.clear()
        self.last_shown_at = 0.0
        self.motion_armed_until = 0.0
        
    def should_process_frame(self):
        """Verificar si debe procesar el siguiente frame (rate limiting)"""
        current_time = time.time()
        if current_time - self.last_process_time >= self.min_frame_interval:
            self.last_process_time = current_time
            return True
        return False
        
    def get_buffer_size(self):
        """Obtener tamaño actual del buffer"""
        return len(self.buffer)

def init_ml():
    """Inicializar modelo y MediaPipe"""
    global modelo, mp_hands, mp_drawing, hands, ML_ENABLED, PALABRAS, LABELS_MAP
    
    logger.info("Iniciando MediaPipe y modelo LSTM...")
    
    try:
        # Importar MediaPipe
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        
        logger.info("MediaPipe importado correctamente")
        
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1,
        )
        logger.info("MediaPipe Hands inicializado")
        
        # Cargar labels dinámicas (si existe labels.json)
        label_ids, labels_map = _load_labels()
        if label_ids:
            PALABRAS = label_ids
            LABELS_MAP = labels_map
            logger.info(f"Labels cargadas: {len(PALABRAS)}")
        else:
            # Fallback histórico (por si no hay labels.json en deploy)
            PALABRAS = ["DIEGO", "GRACIAS", "HOLA", "MI_NOMBRE", "NOS_VEMOS"]
            LABELS_MAP = {w: w for w in PALABRAS}
            logger.warning("labels.json no encontrado; usando vocabulario hardcodeado")

        # Cargar modelo LSTM
        try:
            import tensorflow as tf
            load_model = tf.keras.models.load_model
            
            logger.info("TensorFlow importado correctamente")
            
            # Buscar modelo en diferentes ubicaciones
            possible_paths = [
                # Prefer final model if present
                'modelo_señas_final.h5',
                'modelo_señas_best.h5',
                '../modelo_señas_final.h5',
                '../modelo_señas_best.h5',
                '../../modelo_señas_final.h5',
                '../../modelo_señas_best.h5',
                os.path.join(os.path.dirname(__file__), '..', '..', 'web-app', 'modelo_señas_final.h5'),
                os.path.join(os.path.dirname(__file__), '..', '..', 'web-app', 'modelo_señas_best.h5'),
                os.path.join(os.path.dirname(__file__), '..', '..', 'modelo_señas_final.h5'),
                os.path.join(os.path.dirname(__file__), '..', '..', 'modelo_señas_best.h5'),
            ]
            
            modelo_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    modelo_path = path
                    break
            
            if modelo_path:
                modelo = load_model(modelo_path)
                ML_ENABLED = True
                logger.info(f"Modelo LSTM cargado desde: {modelo_path}")
                # Validar consistencia entre salida del modelo y labels cargadas
                try:
                    output_units = int(getattr(modelo, "output_shape", [None, 0])[-1])
                except Exception:
                    output_units = 0

                if output_units and len(PALABRAS) != output_units:
                    logger.warning(
                        "Mismatch modelo/labels: "
                        f"modelo tiene {output_units} clases, labels tienen {len(PALABRAS)}. "
                        "Usando etiquetas sintéticas para evitar fallo de predicción."
                    )
                    PALABRAS = [f"CLASS_{i}" for i in range(output_units)]
                    LABELS_MAP = {lid: lid for lid in PALABRAS}

                logger.info(f"Vocabulario ({len(PALABRAS)}): {', '.join(PALABRAS)}")
                logger.info(f"Normalización landmarks: {'ON' if NORMALIZATION_AVAILABLE else 'OFF'}")
            else:
                logger.warning("Modelo no encontrado en rutas esperadas")
                logger.info("Ubicaciones buscadas:")
                for path in possible_paths:
                    logger.info(f"  - {os.path.abspath(path)}")
                ML_ENABLED = False
                
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            ML_ENABLED = False
            
    except ImportError as e:
        logger.error(f"Error importando dependencias MediaPipe: {e}")
        ML_ENABLED = False
    except Exception as e:
        logger.error(f"Error inesperado en init_ml: {e}")
        ML_ENABLED = False
    
    logger.info(f"Inicialización completa - MediaPipe: {hands is not None}, ML: {ML_ENABLED}")


def _label_id_for_index(i: int) -> str:
    if 0 <= i < len(PALABRAS):
        return str(PALABRAS[i])
    return f"CLASS_{i}"


def _compute_motion(sequence_30x126: np.ndarray) -> float:
    """Compute motion metric like InferenciaSeñas.py.

    sequence_30x126: (T, 126)
    """
    if not REQUIRE_MOTION:
        return 1.0

    window = min(MOTION_WINDOW, int(sequence_30x126.shape[0]))
    if window < 2:
        return 0.0
    seq_w = sequence_30x126[-window:, :]
    return float(np.mean(np.abs(np.diff(seq_w, axis=0))))

def extract_landmarks(frame_bgr):
    """Extraer landmarks de un frame BGR"""
    if not hands:
        logger.error("MediaPipe Hands no inicializado")
        return None
    
    try:
        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        logger.debug(f"Frame convertido a RGB: {frame_rgb.shape}")
        
        # Procesar sin timestamp para evitar errores de MediaPipe
        results = hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            logger.debug("MediaPipe no encontró landmarks")
            return None
        
        logger.debug(f"Detectadas {len(results.multi_hand_landmarks)} mano(s)")

        # Prefer: extracción normalizada consistente con entrenamiento
        if frame_from_mediapipe_results is not None:
            frame_data = frame_from_mediapipe_results(results, rotate_palm=True)
            return frame_data

        # Fallback: extracción cruda
        frame_data = [0.0] * 126
        idx = 0
        for hand_landmarks in results.multi_hand_landmarks[:2]:  # Máximo 2 manos
            for landmark in hand_landmarks.landmark:
                frame_data[idx] = landmark.x
                frame_data[idx + 1] = landmark.y
                frame_data[idx + 2] = landmark.z
                idx += 3
        return frame_data
        
    except Exception as e:
        logger.error(f"Error en extract_landmarks: {e}")
        return None

def predict_from_sequence(sequence_buffer):
    """Hacer predicción desde buffer de secuencia"""
    if not ML_ENABLED or not modelo:
        return None
    
    if len(sequence_buffer) < FRAMES_POR_SECUENCIA:
        return None
    
    try:
        # Convertir a array numpy (tomar los últimos N frames)
        secuencia = np.array(list(sequence_buffer)[-FRAMES_POR_SECUENCIA:], dtype=np.float32)
        
        # Expandir dimensiones para batch
        secuencia = np.expand_dims(secuencia, axis=0)
        
        # Predecir
        predicciones = modelo.predict(secuencia, verbose=0)[0]
        predicciones = np.asarray(predicciones, dtype=np.float32)
        n_classes = int(predicciones.shape[0])
        idx_prediccion = int(np.argmax(predicciones))
        confianza = float(predicciones[idx_prediccion])

        label_id = _label_id_for_index(idx_prediccion)
        label_human = LABELS_MAP.get(label_id, label_id)

        all_predictions = {
            _label_id_for_index(i): float(predicciones[i] * 100) for i in range(n_classes)
        }
        
        return {
            'word': label_human,
            'word_id': label_id,
            'confidence': confianza * 100,
            'all_predictions': all_predictions,
        }
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return None


def predict_from_sequence_gated(session: UserSession):
    """Predicción robusta (igual que InferenciaSeñas.py).

    Devuelve dict de predicción SOLO cuando hay estabilidad temporal y pasa gating.
    """
    if not ML_ENABLED or not modelo:
        return None
    if session.get_buffer_size() < FRAMES_POR_SECUENCIA:
        return None

    try:
        seq = np.array(list(session.buffer)[-FRAMES_POR_SECUENCIA:], dtype=np.float32)
        secuencia = np.expand_dims(seq, axis=0)  # (1, 30, 126)

        pred = modelo.predict(secuencia, verbose=0)[0]
        pred = np.asarray(pred, dtype=np.float32)
        idx = int(np.argmax(pred))
        conf = float(pred[idx])

        # Margin gate
        if pred.shape[0] >= 2:
            top2 = float(np.partition(pred, -2)[-2])
        else:
            top2 = 0.0
        margin = float(conf - top2)

        # Motion gate (con hysteresis): el movimiento arma la clasificación por un rato.
        motion = _compute_motion(seq)
        now = time.monotonic()
        if REQUIRE_MOTION and motion >= MIN_MOTION:
            session.motion_armed_until = max(session.motion_armed_until, now + MOTION_ARM_SECONDS)

        motion_ok = (not REQUIRE_MOTION) or (now <= session.motion_armed_until)

        accepted = (
            conf >= CONFIDENCE_THRESHOLD
            and margin >= MARGIN_THRESHOLD
            and motion_ok
        )

        session.pred_history.append(idx if accepted else None)

        if len(session.pred_history) == STABLE_PRED_FRAMES and all(
            p is not None and p == session.pred_history[0] for p in session.pred_history
        ):
            label_id = _label_id_for_index(idx)
            label_human = LABELS_MAP.get(label_id, label_id)
            session.last_shown_at = time.monotonic()

            all_predictions = {
                _label_id_for_index(i): float(pred[i] * 100) for i in range(int(pred.shape[0]))
            }

            return {
                'word': label_human,
                'word_id': label_id,
                'confidence': conf * 100,
                'all_predictions': all_predictions,
                'debug': {
                    'margin': margin,
                    'motion': motion,
                    'motion_ok': motion_ok,
                },
            }

        # Hold: mientras esté dentro del hold, no forzar limpieza
        if session.last_shown_at and (time.monotonic() - session.last_shown_at) <= DISPLAY_HOLD_SECONDS:
            return None

        return None
    except Exception as e:
        logger.error(f"Error en predicción (gated): {e}")
        return None

# ==================== RUTAS HTTP ====================

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/demo')
def demo():
    """Página demo"""
    return render_template('demo.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Página de login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember') == 'on'
        
        if not username or not password:
            flash('Por favor completa todos los campos', 'error')
            return render_template('login.html')
        
        # Buscar usuario por username o email
        user = User.query.filter(
            (User.username == username) | (User.email == username)
        ).first()
        
        if user and user.check_password(password):
            if not user.is_active:
                flash('Tu cuenta está desactivada. Contacta al soporte.', 'error')
                return render_template('login.html')
            
            login_user(user, remember=remember)
            user.update_last_login()
            
            flash(f'¡Bienvenido de vuelta, {user.full_name or user.username}!', 'success')
            
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Usuario o contraseña incorrectos', 'error')
            return render_template('login.html')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Página de registro"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        password_confirm = request.form.get('password_confirm', '')
        
        # Validaciones
        if not all([full_name, username, email, password, password_confirm]):
            flash('Por favor completa todos los campos', 'error')
            return render_template('register.html')
        
        if password != password_confirm:
            flash('Las contraseñas no coinciden', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('La contraseña debe tener al menos 6 caracteres', 'error')
            return render_template('register.html')
        
        if len(username) < 3 or len(username) > 20:
            flash('El usuario debe tener entre 3 y 20 caracteres', 'error')
            return render_template('register.html')
        
        if '@' not in email:
            flash('Por favor ingresa un email válido', 'error')
            return render_template('register.html')
        
        # Verificar si usuario o email ya existen
        if User.query.filter_by(username=username).first():
            flash('El nombre de usuario ya está en uso', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('El email ya está registrado', 'error')
            return render_template('register.html')
        
        # Crear nuevo usuario
        new_user = User(
            full_name=full_name,
            username=username,
            email=email
        )
        new_user.set_password(password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            
            flash(f'¡Cuenta creada exitosamente! Bienvenido, {full_name}', 'success')
            
            # Auto-login después del registro
            login_user(new_user)
            new_user.update_last_login()
            
            return redirect(url_for('dashboard'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error al crear usuario: {e}")
            flash('Error al crear la cuenta. Por favor intenta nuevamente.', 'error')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    """Cerrar sesión"""
    logout_user()
    flash('Sesión cerrada correctamente', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard de usuario"""
    stats = current_user.get_stats()
    return render_template('dashboard.html', user=current_user, stats=stats)

@app.route('/health')
def health():
    """Health check para Railway"""
    return jsonify({
        'status': 'ok',
        'ml_enabled': ML_ENABLED,
        'model_loaded': modelo is not None,
        'vocabulary': PALABRAS,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health')
def api_health():
    """Health check API"""
    return jsonify({
        'status': 'ok',
        'ml_enabled': ML_ENABLED,
        'model_loaded': modelo is not None,
        'vocabulary_size': len(PALABRAS),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/vocabulary')
def api_vocabulary():
    """Obtener vocabulario disponible"""
    vocabulary_map = {lid: LABELS_MAP.get(lid, lid) for lid in PALABRAS}
    return jsonify({
        'vocabulary': vocabulary_map,
        'vocabulary_list': list(vocabulary_map.values()),
        'label_ids': PALABRAS,
        'size': len(PALABRAS),
        'timestamp': datetime.now().isoformat()
    })

# ==================== WEBSOCKET EVENTS ====================

@socketio.on('connect')
def handle_connect(auth=None):
    """Cliente conectado"""
    session_id = request.sid  # request viene de Flask
    user_sessions[session_id] = UserSession(session_id)
    
    logger.info(f"Cliente conectado: {session_id}")
    
    vocabulary_map = {lid: LABELS_MAP.get(lid, lid) for lid in PALABRAS}

    emit('connected', {
        'session_id': session_id,
        'ml_enabled': ML_ENABLED,
        'model_loaded': modelo is not None,
        'vocabulary': vocabulary_map,
        'vocabulary_list': list(vocabulary_map.values()),
        'label_ids': PALABRAS,
        'frames_required': FRAMES_POR_SECUENCIA,
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect(reason=None):
    """Cliente desconectado"""
    session_id = request.sid
    if session_id in user_sessions:
        del user_sessions[session_id]
    logger.info(f"Cliente desconectado: {session_id} (reason: {reason})")

@socketio.on('process_frame')
def handle_process_frame(data):
    """Procesar frame desde cliente"""
    session_id = request.sid
    
    if session_id not in user_sessions:
        emit('error', {'message': 'Sesión no encontrada'})
        return
    
    session = user_sessions[session_id]
    
    # Rate limiting: ignorar frames que llegan muy rápido
    if not session.should_process_frame():
        return
    
    try:
        # Decodificar imagen base64
        frame_data = data.get('frame', '')
        if not frame_data.startswith('data:image'):
            emit('error', {'message': 'Formato de imagen inválido'})
            return
        
        # Extraer base64
        img_base64 = frame_data.split(',')[1]
        img_bytes = base64.b64decode(img_base64)
        
        # Convertir a imagen OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame_bgr is None:
            logger.error(f"Error decodificando imagen")
            emit('error', {'message': 'Error decodificando imagen'})
            return
        
        logger.debug(
            f"Frame recibido: shape={frame_bgr.shape}, dtype={frame_bgr.dtype}, size={len(img_bytes)} bytes"
        )
        
        # Extraer landmarks
        landmarks = extract_landmarks(frame_bgr)
        
        if landmarks is not None:
            logger.debug(f"Landmarks detectados: {len(landmarks)} valores")
        else:
            logger.debug(f"No se detectaron manos - Frame shape: {frame_bgr.shape}")
        
        if landmarks:
            # Agregar al buffer
            session.add_frame(landmarks)
            
            logger.debug(f"Manos detectadas - Buffer: {session.get_buffer_size()}/{FRAMES_POR_SECUENCIA}")
            
            # Emitir estado
            emit('frame_processed', {
                'buffer_size': session.get_buffer_size(),
                'max_size': FRAMES_POR_SECUENCIA,
                'hands_detected': True,
                'ready_to_predict': session.get_buffer_size() >= FRAMES_POR_SECUENCIA
            })
            
            # Intentar predicción si tenemos suficientes frames
            if session.get_buffer_size() >= FRAMES_POR_SECUENCIA:
                prediction = predict_from_sequence_gated(session)

                if prediction:
                    # Evitar predicciones repetidas
                    if session.last_prediction != prediction['word_id']:
                        session.last_prediction = prediction['word_id']
                        session.prediction_count += 1

                        emit('prediction', {
                            'word': prediction['word'],
                            'word_id': prediction.get('word_id'),
                            'confidence': round(float(prediction['confidence']), 2),
                            'all_predictions': prediction['all_predictions'],
                            'method': 'LSTM_TensorFlow_GATED',
                            'timestamp': datetime.now().isoformat(),
                            'prediction_number': session.prediction_count
                        })

                        logger.info(
                            f"Predicción: {prediction['word_id']} -> {prediction['word']} "
                            f"({float(prediction['confidence']):.1f}%)"
                        )

                        # Reiniciar ventana para que no sea necesario pausar la cámara
                        session.clear_buffer()
        else:
            # Sin manos detectadas, limpiar buffer
            session.clear_buffer()
            session.last_prediction = None
            emit('frame_processed', {
                'buffer_size': 0,
                'max_size': FRAMES_POR_SECUENCIA,
                'hands_detected': False,
                'ready_to_predict': False
            })
    
    except Exception as e:
        logger.error(f"Error procesando frame: {e}")
        emit('error', {'message': f'Error procesando frame: {str(e)}'})

@socketio.on('clear_history')
def handle_clear_history():
    """Limpiar historial de predicciones"""
    session_id = request.sid
    
    if session_id in user_sessions:
        session = user_sessions[session_id]
        session.clear_buffer()
        session.last_prediction = None
        
        emit('history_cleared', {
            'message': 'Historial limpiado',
            'timestamp': datetime.now().isoformat()
        })
        logger.info(f"🧹 Historial limpiado para: {session_id}")

@socketio.on('get_stats')
def handle_get_stats():
    """Obtener estadísticas de la sesión"""
    session_id = request.sid
    
    if session_id in user_sessions:
        session = user_sessions[session_id]
        emit('stats', {
            'predictions_count': session.prediction_count,
            'buffer_size': session.get_buffer_size(),
            'session_duration': str(datetime.now() - session.start_time)
        })

# ==================== INICIALIZACIÓN ====================

# Inicializar ML al cargar el módulo (para gunicorn)
logger.info("Inicializando sistema de reconocimiento de señas...")
init_ml()

if ML_ENABLED:
    logger.info("Sistema ML activo")
else:
    logger.warning("Sistema funcionando sin ML")

# Crear tablas de base de datos si no existen
with app.app_context():
    try:
        db.create_all()
        logger.info("Base de datos inicializada")
        
        # Crear usuario demo si no existe
        if not User.query.filter_by(username='demo').first():
            demo_user = User(
                username='demo',
                email='demo@lsch.com',
                full_name='Usuario Demo'
            )
            demo_user.set_password('demo123')
            demo_user.email_verified = True
            
            db.session.add(demo_user)
            db.session.commit()
            logger.info("Usuario demo creado: demo / demo123")
    except Exception as e:
        logger.error(f"Error al inicializar base de datos: {e}")

if __name__ == '__main__':
    # Obtener puerto de Railway o usar 5000
    PORT = int(os.environ.get("PORT", 5000))
    HOST = "0.0.0.0"
    
    logger.info(f"Servidor iniciando en {HOST}:{PORT}")
    
    # Iniciar servidor
    socketio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True)
