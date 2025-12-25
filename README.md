# Sistema de Reconocimiento de Lenguaje de Señas Chilenas (LSCh)

Sistema web de reconocimiento en tiempo real de lenguaje de señas chilenas usando LSTM y MediaPipe.

## Características

- Reconocimiento en tiempo real (vocabulario dinámico)
- Modelo LSTM con 90.67% de precisión
- Interfaz web interactiva
- WebSocket para comunicación en tiempo real
- Procesamiento con MediaPipe Hands

## Vocabulario

El vocabulario ya no está hardcodeado.

- Se carga dinámicamente desde `labels.json` (mismo orden que en el entrenamiento).
- Si `labels.json` no existe en el servidor, se usa un vocabulario de respaldo para la demo.

## Instalación

### Requisitos
- Python 3.10+
- Webcam

### Pasos

1. **Clonar repositorio**
```bash
git clone https://github.com/SnaKzu/LSChOk.git
cd LSChOk
```

2. **Crear entorno virtual**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Ejecutar servidor**
```bash
cd web-app
python backend/app_senas.py
```

5. **Abrir en navegador**
```
http://localhost:5000/demo
```

## Arquitectura

### Backend
- **Flask + SocketIO**: Servidor web y WebSocket
- **TensorFlow/Keras**: Modelo LSTM
- **MediaPipe**: Detección de manos y landmarks
- **OpenCV**: Procesamiento de imágenes

### Frontend
- **HTML5 + JavaScript**: Interfaz de usuario
- **Socket.IO**: Comunicación en tiempo real
- **Canvas API**: Captura y procesamiento de video

### Modelo LSTM
- **Entrada**: secuencias de 30 frames (126 features por frame)
- **Arquitectura**: LSTM(64) → LSTM(64) → Dense(32) → Dense(N)
- **Parámetros**: 84,000
- **Precisión**: 90.67%
- **Dataset**: 500 secuencias (100 por palabra)

## Flujo de Datos

```
Cliente → WebSocket → Servidor
  ↓                      ↓
Video                MediaPipe (landmarks)
  ↓                      ↓
Base64               Buffer (30 frames)
                         ↓
                    LSTM Model
                         ↓
                    Predicción
                         ↓
Cliente ← WebSocket ← Servidor
```

## Interfaz

La aplicación incluye:
- Panel de video en vivo
- Indicadores de detección de manos
- Contador de buffer de frames
- Predicción actual con % de confianza
- Historial de predicciones
- Estadísticas de sesión

## Configuración

### Variables de Entorno (Producción)
```
PORT=5000
SECRET_KEY=<tu-clave-secura>
PYTHONUNBUFFERED=1
```

## Uso

1. **Activar Cámara**: Click en "Activar Cámara"
2. **Permitir Acceso**: Aceptar permisos de webcam
3. **Realizar Señas**: Ejecutar señas del vocabulario LSCh
4. **Ver Resultados**: Las predicciones aparecen automáticamente

## Desarrollo

### Estructura del Proyecto
```
web-app/
├── backend/
│   └── app_senas.py         # Servidor Flask + LSTM
├── frontend/
│   ├── demo.html            # Interfaz principal
│   ├── css/
│   │   ├── styles.css
│   │   └── demo.css
│   └── js/
│       ├── websocket.js
│       └── demo.js
├── modelo_señas_best.h5     # Modelo entrenado
└── requirements.txt
```

### Scripts Adicionales
- `CapturarSecuencias.py`: Captura videos para entrenamiento
- `EntrenarModeloSeñas.py`: Entrena el modelo LSTM
- `InferenciaSeñas.py`: Prueba local del modelo

## Despliegue

### Railway
1. Conectar repositorio a Railway
2. Configurar variables de entorno
3. Railway detectará `requirements.txt` automáticamente
4. El servidor iniciará en el puerto configurado

### Heroku / Render
Similar a Railway, asegurarse de:
- Configurar `Procfile` si es necesario
- Establecer `PORT` como variable de entorno
- Verificar que `modelo_señas_best.h5` esté incluido

## Performance

- **FPS Frontend**: 30
- **FPS Backend**: ~20 (rate limiting)
- **Latencia**: <100ms por predicción
- **Confianza promedio**: 95%
- **Uso RAM**: ~1.2 GB

## Notas

### Advertencia MediaPipe
```
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
```
Esta advertencia es normal en Windows y no afecta el funcionamiento.

### Rate Limiting
El sistema limita el procesamiento a 20 FPS para evitar sobrecarga y errores de timestamp en MediaPipe.

## Contribuir

Las contribuciones son bienvenidas! Por favor:
1. Fork el proyecto
2. Crear una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abrir un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT.

## Autor

**Diego Canales** - [SnaKzu](https://github.com/SnaKzu)

## Agradecimientos

- MediaPipe por la detección de manos
- TensorFlow/Keras por el framework de ML
- Flask-SocketIO por la comunicación en tiempo real
- Comunidad LSCh por el vocabulario

---

**Hecho con ❤️ para la comunidad sorda chilena**
