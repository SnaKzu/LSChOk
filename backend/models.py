"""
Modelos de Base de Datos para LSCh Web Application
PostgreSQL + SQLAlchemy + Flask-Login
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import uuid

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """Modelo de Usuario para autenticación y perfil"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(50), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # Información básica
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    full_name = db.Column(db.String(200), nullable=True)
    
    # Autenticación
    password_hash = db.Column(db.String(255), nullable=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # Configuración de cuenta
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    email_verified = db.Column(db.Boolean, default=False, nullable=False)
    
    # Relaciones
    predictions = db.relationship('Prediction', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Establecer contraseña hasheada"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verificar contraseña"""
        return check_password_hash(self.password_hash, password)
    
    def update_last_login(self):
        """Actualizar timestamp del último login"""
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    def get_stats(self):
        """Obtener estadísticas del usuario"""
        total_predictions = self.predictions.count()
        if total_predictions == 0:
            return {
                'total_predictions': 0,
                'avg_confidence': 0.0,
                'unique_words': 0,
                'top_words': [],
                'recent_predictions': []
            }
        
        # Confianza promedio
        predictions = self.predictions.all()
        avg_confidence = sum(p.confidence for p in predictions) / len(predictions)
        
        # Palabras únicas y más usadas
        word_counts = {}
        for pred in predictions:
            word_counts[pred.word] = word_counts.get(pred.word, 0) + 1
        
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Predicciones recientes
        recent_predictions = self.predictions.order_by(Prediction.timestamp.desc()).limit(10).all()
        
        return {
            'total_predictions': total_predictions,
            'avg_confidence': avg_confidence,
            'unique_words': len(word_counts),
            'top_words': top_words,
            'recent_predictions': recent_predictions
        }
    
    def to_dict(self):
        """Convertir a diccionario para JSON"""
        return {
            'id': self.id,
            'public_id': self.public_id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active
        }
    
    def __repr__(self):
        return f'<User {self.username}>'


class Prediction(db.Model):
    """Modelo para predicciones de señas LSCh"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(50), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # Relación con usuario
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Datos de la predicción
    word = db.Column(db.String(100), nullable=False, index=True)
    word_id = db.Column(db.String(100), nullable=False, index=True)  # ID técnico de la seña
    confidence = db.Column(db.Float, nullable=False, index=True)
    
    # Metadatos de la sesión
    session_id = db.Column(db.String(100), nullable=True, index=True)
    frame_count = db.Column(db.Integer, nullable=True)
    processing_time = db.Column(db.Float, nullable=True)  # Tiempo en procesar en ms
    
    # Datos técnicos (opcional)
    model_version = db.Column(db.String(50), nullable=True)
    keypoints_quality = db.Column(db.Float, nullable=True)  # Calidad de los keypoints detectados
    
    # Timestamps
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    def to_dict(self):
        """Convertir a diccionario para JSON"""
        return {
            'id': self.id,
            'public_id': self.public_id,
            'word': self.word,
            'word_id': self.word_id,
            'confidence': self.confidence,
            'session_id': self.session_id,
            'frame_count': self.frame_count,
            'processing_time': self.processing_time,
            'model_version': self.model_version,
            'timestamp': self.timestamp.isoformat()
        }
    
    def __repr__(self):
        return f'<Prediction {self.word} ({self.confidence:.2f})>'


# Funciones de utilidad para inicialización
def create_tables():
    """Crear todas las tablas en la base de datos"""
    db.create_all()
    print("✅ Tablas de base de datos creadas")

def init_db(app):
    """Inicializar base de datos con la app Flask"""
    db.init_app(app)
    
    with app.app_context():
        create_tables()
        
        # Crear usuario demo por defecto si no existe
        if not User.query.filter_by(username='demo').first():
            demo_user = User(
                username='demo',
                email='demo@lsch.com',
                full_name='Usuario Demo LSCh'
            )
            demo_user.set_password('demo123')  
            demo_user.email_verified = True
            
            db.session.add(demo_user)
            db.session.commit()
            print("✅ Usuario demo creado: demo / demo123")

def get_system_stats():
    """Obtener estadísticas generales del sistema"""
    total_users = User.query.count()
    total_predictions = Prediction.query.count()
    
    # Promedio de confianza general
    avg_confidence = db.session.query(db.func.avg(Prediction.confidence)).scalar() or 0.0
    
    # Palabra más popular
    top_word = db.session.query(
        Prediction.word, 
        db.func.count(Prediction.word)
    ).group_by(Prediction.word).order_by(db.func.count(Prediction.word).desc()).first()
    
    return {
        'total_users': total_users,
        'total_predictions': total_predictions,
        'avg_confidence': float(avg_confidence),
        'top_word': top_word[0] if top_word else None,
        'top_word_count': top_word[1] if top_word else 0
    }
