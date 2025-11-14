"""
Backend Flask para Reconocimiento de Lenguaje de Señas
Integración del modelo LSTM entrenado (DIEGO, GRACIAS, HOLA, MI_NOMBRE, NOS_VEMOS)
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
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
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
CORS(app)

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
PALABRAS = ["DIEGO", "GRACIAS", "HOLA", "MI_NOMBRE", "NOS_VEMOS"]
FRAMES_POR_SECUENCIA = 60
ML_ENABLED = False

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
        self.min_frame_interval = 0.05  # Mínimo 50ms entre frames (20 FPS)
        
    def add_frame(self, frame_data):
        """Agregar frame al buffer"""
        self.buffer.append(frame_data)
        self.frame_counter += 1
        
    def clear_buffer(self):
        """Limpiar buffer"""
        self.buffer.clear()
        self.frame_counter = 0
        
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
    global modelo, mp_hands, mp_drawing, hands, ML_ENABLED
    
    try:
        # Importar MediaPipe
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        logger.info("MediaPipe inicializado con baja confianza para mejor detección")
        
        # Cargar modelo LSTM
        try:
            from tensorflow.keras.models import load_model
            
            # Buscar modelo en diferentes ubicaciones
            possible_paths = [
                'modelo_señas_best.h5',
                '../modelo_señas_best.h5',
                '../../modelo_señas_best.h5',
                '../../Tensorflow/modelo_señas_best.h5',
                os.path.join(os.path.dirname(__file__), '..', '..', 'Tensorflow', 'modelo_señas_best.h5'),
                os.path.join(os.path.dirname(__file__), '..', '..', 'modelo_señas_best.h5')
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
                logger.info(f"Vocabulario: {', '.join(PALABRAS)}")
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
        logger.error(f"Error importando dependencias: {e}")
        ML_ENABLED = False

def extract_landmarks(frame_bgr):
    """Extraer landmarks de un frame BGR"""
    if not hands:
        return None
    
    # Convertir BGR a RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Procesar sin timestamp para evitar errores de MediaPipe
    # MediaPipe requiere timestamps monotónicos, pero no son necesarios
    # para detección de manos en modo no-streaming
    results = hands.process(frame_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    # Extraer landmarks (siempre 126 valores)
    frame_data = [0.0] * 126
    idx = 0
    
    for hand_landmarks in results.multi_hand_landmarks[:2]:  # Máximo 2 manos
        for landmark in hand_landmarks.landmark:
            frame_data[idx] = landmark.x
            frame_data[idx + 1] = landmark.y
            frame_data[idx + 2] = landmark.z
            idx += 3
    
    return frame_data

def predict_from_sequence(sequence_buffer):
    """Hacer predicción desde buffer de secuencia"""
    if not ML_ENABLED or not modelo:
        return None
    
    if len(sequence_buffer) < 30:  # Mínimo 30 frames
        return None
    
    try:
        # Convertir a array numpy
        secuencia = np.array(list(sequence_buffer))
        
        # Aplicar padding si es necesario
        if len(secuencia) < FRAMES_POR_SECUENCIA:
            padding = np.zeros((FRAMES_POR_SECUENCIA - len(secuencia), 126))
            secuencia = np.vstack([secuencia, padding])
        
        # Expandir dimensiones para batch
        secuencia = np.expand_dims(secuencia, axis=0)
        
        # Predecir
        predicciones = modelo.predict(secuencia, verbose=0)[0]
        idx_prediccion = np.argmax(predicciones)
        confianza = float(predicciones[idx_prediccion])
        
        return {
            'word': PALABRAS[idx_prediccion],
            'confidence': confianza * 100,
            'all_predictions': {PALABRAS[i]: float(predicciones[i] * 100) for i in range(len(PALABRAS))}
        }
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
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
    return jsonify({
        'vocabulary': PALABRAS,
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
    
    emit('connected', {
        'session_id': session_id,
        'ml_enabled': ML_ENABLED,
        'model_loaded': modelo is not None,
        'vocabulary': PALABRAS,
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
        
        logger.debug(f"📸 Frame recibido: {frame_bgr.shape}")
        
        # Extraer landmarks
        landmarks = extract_landmarks(frame_bgr)
        
        logger.debug(f"👋 Landmarks detectados: {landmarks is not None}")
        
        if landmarks:
            # Agregar al buffer
            session.add_frame(landmarks)
            
            logger.info(f"✅ Manos detectadas - Buffer: {session.get_buffer_size()}/{FRAMES_POR_SECUENCIA}")
            
            # Emitir estado
            emit('frame_processed', {
                'buffer_size': session.get_buffer_size(),
                'max_size': FRAMES_POR_SECUENCIA,
                'hands_detected': True,
                'ready_to_predict': session.get_buffer_size() >= 30
            })
            
            # Intentar predicción si tenemos suficientes frames
            if session.get_buffer_size() >= 30:
                prediction = predict_from_sequence(session.buffer)
                
                if prediction and prediction['confidence'] > 60:  # Umbral de confianza
                    # Evitar predicciones repetidas
                    if session.last_prediction != prediction['word']:
                        session.last_prediction = prediction['word']
                        session.prediction_count += 1
                        
                        emit('prediction', {
                            'word': prediction['word'],
                            'confidence': round(prediction['confidence'], 2),
                            'all_predictions': prediction['all_predictions'],
                            'method': 'LSTM_TensorFlow',
                            'timestamp': datetime.now().isoformat(),
                            'prediction_number': session.prediction_count
                        })
                        
                        logger.info(f"🎯 Predicción: {prediction['word']} ({prediction['confidence']:.1f}%)")
        else:
            logger.warning(f"⚠️ No se detectaron manos en el frame")
            # Sin manos detectadas, limpiar buffer
            session.clear_buffer()
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

if __name__ == '__main__':
    # Inicializar ML
    logger.info("Inicializando sistema de reconocimiento de señas...")
    init_ml()
    
    if ML_ENABLED:
        logger.info("Sistema ML activo")
    else:
        logger.info("Sistema funcionando sin ML")
    
    # Obtener puerto de Railway o usar 5000
    PORT = int(os.environ.get("PORT", 5000))
    HOST = "0.0.0.0"
    
    logger.info(f"Servidor iniciando en {HOST}:{PORT}")
    
    # Iniciar servidor
    socketio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True)
