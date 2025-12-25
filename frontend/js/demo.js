// ================================================
// LSP Recognition - Demo Page JavaScript v2.0
// ================================================

console.log('🚀 LSP Demo v2.0 Loading...');

class LSPDemo {
    constructor() {
        console.log('LSPDemo Constructor - v2.0');
        this.socket = null;
        this.videoElement = document.getElementById('videoElement');
        this.canvasElement = document.getElementById('canvasElement');
        this.ctx = this.canvasElement.getContext('2d');
        this.stream = null;
        this.isProcessing = false;
        this.isPaused = false;
        this.sessionStartTime = null;
        this.predictions = [];
        this.frameCount = 0;
        this.framesSent = 0;
        this.lastFrameTime = 0;
        this.frameInterval = 33; // ~30 FPS
        
        this.init();
    }
    
    init() {
        console.log('LSPDemo Init - v2.0');
        this.setupElements();
        this.setupEventListeners();
        this.initWebSocket();
        this.showInstructionsModal();
    }
    
    setupElements() {
        this.startBtn = document.getElementById('startBtn');
        this.pauseBtn = document.getElementById('pauseBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.startOverlay = document.getElementById('startOverlay');
        this.statusBadge = document.getElementById('statusBadge');
        this.recordingIndicator = document.getElementById('recordingIndicator');
        this.frameCounterEl = document.getElementById('frameCount');
        this.currentPredictionEl = document.querySelector('.prediction-word');
        this.confidenceFillEl = document.getElementById('confidenceFill');
        this.confidenceValueEl = document.getElementById('confidenceValue');
        this.sentenceContainer = document.getElementById('sentenceContainer');
        this.totalPredictionsEl = document.getElementById('totalPredictions');
        this.sessionTimeEl = document.getElementById('sessionTime');
        this.avgConfidenceEl = document.getElementById('avgConfidence');
        this.connectionStatus = document.getElementById('connectionStatus');
        this.statusIcon = document.getElementById('statusIcon');
        this.statusText = document.getElementById('statusText');
    }
    
    setupEventListeners() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.pauseBtn.addEventListener('click', () => this.togglePause());
        this.clearBtn.addEventListener('click', () => this.clearHistory());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
        
        // Modal controls
        const modal = document.getElementById('instructionsModal');
        const closeModal = document.getElementById('closeModal');
        const gotItBtn = document.getElementById('gotItBtn');
        
        closeModal.addEventListener('click', () => modal.classList.remove('active'));
        gotItBtn.addEventListener('click', () => modal.classList.remove('active'));
    }
    
    showInstructionsModal() {
        const modal = document.getElementById('instructionsModal');
        setTimeout(() => {
            modal.classList.add('active');
        }, 500);
    }
    
    async initWebSocket() {
        console.log('🔌 initWebSocket - v2.0');
        this.updateConnectionStatus('connecting', 'Conectando al servidor...');
        
        // Limpiar cualquier conexión anterior
        this.cleanupSocket();
        
        // Conectar directamente con Socket.IO (más confiable)
        console.log('Conectando directamente con Socket.IO...');
        this.initWebSocketDirect();
    }
    
    cleanupSocket() {
        console.log('🧹 Cleaning up previous socket connections');
        
        // Limpiar socket directo
        if (this.socket && typeof this.socket.disconnect === 'function') {
            this.socket.disconnect();
            this.socket = null;
        }
    }
    
    initWebSocketDirect() {
        console.log('Connecting directly with Socket.IO');
        
        // Determinar URL
        const socketUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
            ? 'http://127.0.0.1:5000' 
            : window.location.origin;
        
        console.log('Connecting to:', socketUrl);
        
        this.socket = io(socketUrl, {
            transports: ['websocket', 'polling'],
            upgrade: true,
            timeout: 20000,
            forceNew: true
        });
        
        this.socket.on('connect', () => {
            console.log('Socket.IO connected!');
            this.updateConnectionStatus('connected', 'Conectado');
            this.showNotification('Conectado al servidor', 'success');
        });

        // Handshake de servidor (incluye estado de modelo y frames requeridos)
        this.socket.on('connected', (data) => {
            console.log('Server handshake:', data);
            const framesRequired = (data && data.frames_required) ? data.frames_required : 30;
            if (data && data.model_loaded === false) {
                this.updateConnectionStatus('error', 'Conectado, pero sin modelo');
                this.showNotification('El servidor está conectado pero el modelo no está cargado.', 'error');
            } else {
                this.updateConnectionStatus('connected', `Conectado - ${framesRequired} frames`);
            }
        });
        
        this.socket.on('disconnect', (reason) => {
            console.log('Socket.IO disconnected:', reason);
            this.updateConnectionStatus('disconnected', 'Desconectado del servidor');
        });
        
        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            this.updateConnectionStatus('error', 'Error de conexión');
        });
        
        // Eventos del sistema LSTM
        this.socket.on('frame_processed', (data) => {
            if (data && data.buffer_size !== undefined && data.buffer_size % 30 === 0) {
                console.log('frame_processed:', data);
            }
            this.handleFrameProcessed(data);
        });
        
        this.socket.on('prediction', (data) => {
            this.handlePrediction(data);
        });
        
        this.socket.on('stats', (data) => {
            this.handleStats(data);
        });
        
        this.socket.on('history_cleared', (data) => {
            this.showNotification(data.message || 'Historial limpiado', 'success');
        });
        
        this.socket.on('error', (data) => {
            console.error('⚠️ Server error:', data.message);
            this.showNotification('Error: ' + data.message, 'error');
        });
    }
    
    updateConnectionStatus(status, text) {
        this.statusIcon.className = `fas fa-circle ${status}`;
        this.statusText.textContent = text;
    }
    
    async startCamera() {
        try {
            // Solicitar permisos de cámara con configuración optimizada
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640, max: 1280 },
                    height: { ideal: 480, max: 720 },
                    facingMode: 'user',
                    frameRate: { ideal: 30, max: 30 }
                },
                audio: false
            });
            
            console.log('Cámara iniciada:', {
                width: this.stream.getVideoTracks()[0].getSettings().width,
                height: this.stream.getVideoTracks()[0].getSettings().height,
                frameRate: this.stream.getVideoTracks()[0].getSettings().frameRate
            });
            
            this.videoElement.srcObject = this.stream;
            this.videoElement.onloadedmetadata = () => {
                this.canvasElement.width = this.videoElement.videoWidth;
                this.canvasElement.height = this.videoElement.videoHeight;
            };
            
            this.startOverlay.classList.add('hidden');
            this.isProcessing = true;
            this.sessionStartTime = Date.now();
            
            this.startBtn.disabled = true;
            this.pauseBtn.disabled = false;
            this.stopBtn.disabled = false;
            
            this.processFrames();
            this.startSessionTimer();
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            alert('No se pudo acceder a la cámara. Por favor, verifica los permisos.');
        }
    }
    
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.isProcessing = false;
        this.isPaused = false;
        this.startOverlay.classList.remove('hidden');
        
        this.startBtn.disabled = false;
        this.pauseBtn.disabled = true;
        this.stopBtn.disabled = true;
        
        if (this.sessionTimer) {
            clearInterval(this.sessionTimer);
        }
        
        // Resetear contadores
        this.framesSent = 0;
        this.lastFrameTime = 0;
        
        console.log('Cámara detenida');
    }
    
    togglePause() {
        this.isPaused = !this.isPaused;
        
        if (this.isPaused) {
            this.pauseBtn.innerHTML = '<i class="fas fa-play"></i> Reanudar';
        } else {
            this.pauseBtn.innerHTML = '<i class="fas fa-pause"></i> Pausar';
        }
    }
    
    async processFrames() {
        if (!this.isProcessing) return;
        
        const now = Date.now();
        const elapsed = now - this.lastFrameTime;
        
        // Throttling: solo procesar si ha pasado suficiente tiempo
        if (!this.isPaused && this.videoElement.readyState === 4 && elapsed >= this.frameInterval) {
            this.lastFrameTime = now;
            
            try {
                // Capturar frame del video
                this.ctx.drawImage(
                    this.videoElement, 
                    0, 0, 
                    this.canvasElement.width, 
                    this.canvasElement.height
                );
                
                // Convertir a base64 JPEG con calidad 85%
                const frameData = this.canvasElement.toDataURL('image/jpeg', 0.85);
                
                // Enviar al servidor si está conectado
                if (this.socket && this.socket.connected) {
                    this.socket.emit('process_frame', { 
                        frame: frameData,
                        timestamp: now
                    });
                    this.framesSent++;
                    
                    // Actualizar contador visual cada 10 frames
                    if (this.framesSent % 10 === 0) {
                        console.log(`Frames enviados: ${this.framesSent}`);
                    }
                } else {
                    console.warn('Socket desconectado, frame no enviado');
                }
            } catch (error) {
                console.error('Error capturando frame:', error);
            }
        }
        
        // Continuar procesando
        requestAnimationFrame(() => this.processFrames());
    }
    
    handleStatusUpdate(data) {
        this.frameCount = data.frame_count || 0;
        this.frameCounterEl.textContent = this.frameCount;
        
        if (data.recording) {
            this.recordingIndicator.classList.add('active');
            this.statusBadge.innerHTML = `
                <i class="fas fa-hand-paper"></i>
                <span>Manos detectadas (${this.frameCount})</span>
            `;
        } else {
            this.recordingIndicator.classList.remove('active');
            if (data.has_hands) {
                this.statusBadge.innerHTML = `
                    <i class="fas fa-hand-paper"></i>
                    <span>Manos detectadas</span>
                `;
            } else {
                this.statusBadge.innerHTML = `
                    <i class="fas fa-hand-paper"></i>
                    <span>Esperando manos...</span>
                `;
            }
        }
    }
    
    handlePrediction(data) {
        console.log('Prediction received:', data);
        
        // Actualizar predicción actual
        this.currentPredictionEl.textContent = data.word.toUpperCase();
        
        const confidence = Math.round(data.confidence);
        this.confidenceFillEl.style.width = `${confidence}%`;
        this.confidenceValueEl.textContent = `${confidence}%`;
        
        // Añadir a historial con información detallada
        this.addToSentence(data.word.toUpperCase(), confidence);
        
        // Guardar estadística
        this.predictions.push({
            word: data.word,
            confidence: data.confidence / 100,
            timestamp: Date.now()
        });
        
        // Actualizar estadísticas
        this.updateStatistics();
        
        // Reproducir sonido (opcional)
        this.playSuccessSound();
        
        // Animación de feedback
        this.animatePrediction();
        
        // Notificación
        this.showNotification(`${data.word.toUpperCase()} (${confidence}%)`, 'success');
    }
    
    addToSentence(word, confidence) {
        // Remover mensaje vacío si existe
        const emptyMessage = this.sentenceContainer.querySelector('.empty-message');
        if (emptyMessage) {
            emptyMessage.remove();
        }
        
        // Crear elemento
        const item = document.createElement('div');
        item.className = 'sentence-item';
        item.innerHTML = `
            <span class="sentence-word">${word}</span>
            <span class="sentence-confidence">${confidence}%</span>
        `;
        
        // Añadir al principio
        this.sentenceContainer.insertBefore(item, this.sentenceContainer.firstChild);
        
        // Limitar a últimas 10 predicciones
        const items = this.sentenceContainer.querySelectorAll('.sentence-item');
        if (items.length > 10) {
            items[items.length - 1].remove();
        }
    }
    
    clearHistory() {
        this.sentenceContainer.innerHTML = `
            <p class="empty-message">
                <i class="fas fa-info-circle"></i>
                Realiza señas para empezar
            </p>
        `;
        
        this.currentPredictionEl.textContent = '---';
        this.confidenceFillEl.style.width = '0%';
        this.confidenceValueEl.textContent = '0%';
        
        // Usar el nuevo evento clear_history
        if (this.socket) {
            this.socket.emit('clear_history');
        }
        
        this.showNotification('Limpiando historial...', 'info');
    }
    
    // Nuevas funciones para el sistema LSTM
    
    handleStats(data) {
        console.log('Stats received:', data);
        
        // Actualizar información del buffer
        if (data.buffer_size !== undefined) {
            this.frameCounterEl.textContent = data.buffer_size;
        }
        
        // Mostrar información adicional si está disponible
        if (data.total_predictions !== undefined) {
            this.totalPredictionsEl.textContent = data.total_predictions;
        }
    }
    
    handleFrameProcessed(data) {
        // Actualizar contador de frames
        if (data.buffer_size !== undefined) {
            this.frameCounterEl.textContent = data.buffer_size;
        }

        const maxSize = data.max_size || 30;
        
        // Actualizar indicadores visuales según detección de manos
        if (data.hands_detected) {
            this.recordingIndicator.classList.add('active');
            this.statusBadge.innerHTML = `
                <i class="fas fa-hand-paper"></i>
                <span>Manos detectadas (Buffer: ${data.buffer_size}/${maxSize})</span>
            `;
            
            // Mostrar progreso de buffer
            const progress = Math.round((data.buffer_size / maxSize) * 100);
            this.statusBadge.style.background = `linear-gradient(90deg, #10b981 ${progress}%, #6b7280 ${progress}%)`;
        } else {
            this.recordingIndicator.classList.remove('active');
            this.statusBadge.innerHTML = `
                <i class="fas fa-hand-paper"></i>
                <span>Esperando manos...</span>
            `;
            this.statusBadge.style.background = '';
        }
    }
    
    getConfidenceClass(confidence) {
        if (confidence >= 85) return 'high-confidence';
        if (confidence >= 70) return 'medium-confidence';
        return 'low-confidence';
    }
    
    updateStatistics() {
        // Total predicciones
        this.totalPredictionsEl.textContent = this.predictions.length;
        
        // Confianza promedio
        if (this.predictions.length > 0) {
            const avgConf = this.predictions.reduce((sum, p) => sum + p.confidence, 0) / this.predictions.length;
            this.avgConfidenceEl.textContent = Math.round(avgConf * 100) + '%';
        }
    }
    
    startSessionTimer() {
        this.sessionTimer = setInterval(() => {
            if (!this.sessionStartTime) return;
            
            const elapsed = Date.now() - this.sessionStartTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            
            this.sessionTimeEl.textContent = 
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
    }
    
    animatePrediction() {
        const predictionEl = document.querySelector('.current-prediction');
        predictionEl.style.animation = 'none';
        setTimeout(() => {
            predictionEl.style.animation = 'pulse 0.5s ease-out';
        }, 10);
    }
    
    playSuccessSound() {
        // Crear un tono simple
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.frequency.value = 800;
            oscillator.type = 'sine';
            
            gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
            
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.2);
        } catch (error) {
            // Silently fail if Web Audio API is not supported
        }
    }
    
    showNotification(message, type = 'info') {
        // Crear notificación toast (simple)
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: ${type === 'error' ? '#ef4444' : '#10b981'};
            color: white;
            padding: 15px 25px;
            border-radius: 10px;
            font-weight: 600;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            z-index: 10000;
            animation: slideInRight 0.3s ease-out;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// Animaciones CSS adicionales
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideOutRight {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100px);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
`;
document.head.appendChild(style);

// Inicializar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    // Verificar dependencias antes de inicializar
    if (typeof io === 'undefined') {
        console.error('Socket.IO no está disponible');
        alert('Error: Socket.IO no se cargó correctamente. Recarga la página.');
        return;
    }
    
    console.log('Socket.IO verificado, inicializando demo...');
    window.lspDemo = new LSPDemo();
});

// Manejar cierre de página
window.addEventListener('beforeunload', () => {
    if (window.lspDemo && window.lspDemo.stream) {
        window.lspDemo.stopCamera();
    }
    if (window.lspDemo && window.lspDemo.socket) {
        window.lspDemo.socket.disconnect();
    }
});
