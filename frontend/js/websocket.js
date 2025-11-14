// ================================================
// WebSocket Configuration for LSCh Demo v2.0
// Auto-detects environment (localhost vs Railway)
// ================================================

console.log('🔧 WebSocket Manager v2.0 Loading...');

class WebSocketManager {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        
        this.callbacks = {
            connect: [],
            disconnect: [],
            prediction: [],
            frame_processed: [],
            stats: [],
            history_cleared: [],
            error: []
        };
    }
    
    connect() {
        return new Promise((resolve, reject) => {
            // Determinar URL del WebSocket
            const socketUrl = this.getSocketUrl();
            console.log('🔌 Connecting to WebSocket v2.0:', socketUrl);
            console.log('🌍 Current location:', {
                hostname: window.location.hostname,
                protocol: window.location.protocol,
                port: window.location.port,
                origin: window.location.origin
            });
            
            // Verificar que Socket.IO esté disponible
            if (typeof io === 'undefined') {
                const error = new Error('Socket.IO library not loaded');
                console.error('❌ Socket.IO no está disponible');
                reject(error);
                return;
            }
            
            console.log('✅ Socket.IO version:', io.version || 'unknown');
            
            // Verificar si hay múltiples instancias
            if (window.io && window.io !== io) {
                console.warn('⚠️ Multiple Socket.IO instances detected');
            }
            
            // Configuración de Socket.IO
            const socketConfig = {
                transports: ['websocket', 'polling'],
                upgrade: true,
                timeout: 20000,
                forceNew: true,
                reconnection: true,
                reconnectionAttempts: this.maxReconnectAttempts,
                reconnectionDelay: this.reconnectDelay,
                reconnectionDelayMax: 5000
            };
            
            console.log('⚙️ Socket.IO config:', socketConfig);
            
            // Desconectar socket anterior si existe
            if (this.socket) {
                console.log('🔄 Cleaning up previous socket');
                this.socket.disconnect();
                this.socket = null;
            }
            
            this.socket = io(socketUrl, socketConfig);
            
            // Event handlers
            this.socket.on('connect', () => {
                console.log('✅ WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.trigger('connect', { connected: true });
                resolve(this.socket);
            });
            
            this.socket.on('disconnect', (reason) => {
                console.log('❌ WebSocket disconnected:', reason);
                this.isConnected = false;
                this.trigger('disconnect', { reason });
                
                // Auto-reconnect si no fue intencional
                if (reason === 'io server disconnect') {
                    this.socket.connect();
                }
            });
            
            this.socket.on('connect_error', (error) => {
                console.error('🚫 WebSocket connection error:', error);
                this.reconnectAttempts++;
                
                if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                    console.error('💥 Max reconnection attempts reached');
                    reject(error);
                } else {
                    console.log(`🔄 Reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
                }
                
                this.trigger('error', { error, attempts: this.reconnectAttempts });
            });
            
            // Eventos específicos de la aplicación
            this.socket.on('connected', (data) => {
                console.log('📡 Server handshake:', data);
            });
            
            this.socket.on('frame_processed', (data) => {
                this.trigger('frame_processed', data);
            });
            
            this.socket.on('prediction', (data) => {
                this.trigger('prediction', data);
            });
            
            this.socket.on('stats', (data) => {
                this.trigger('stats', data);
            });
            
            this.socket.on('history_cleared', (data) => {
                this.trigger('history_cleared', data);
            });
            
            this.socket.on('error', (data) => {
                console.error('⚠️ Server error:', data);
                this.trigger('error', data);
            });
            
            // Timeout si no conecta (aumentado a 20 segundos)
            setTimeout(() => {
                if (!this.isConnected) {
                    console.warn('⏰ Connection timeout, rejecting...');
                    reject(new Error('WebSocket connection timeout'));
                }
            }, 20000);
        });
    }
    
    getSocketUrl() {
        const { hostname, protocol, port } = window.location;
        
        // Desarrollo local
        if (hostname === 'localhost' || hostname === '127.0.0.1') {
            return `http://localhost:5000`;
        }
        
        // Railway o producción
        if (protocol === 'https:') {
            return `https://${hostname}`;
        } else {
            return `http://${hostname}${port ? ':' + port : ''}`;
        }
    }
    
    emit(event, data) {
        if (this.socket && this.isConnected) {
            this.socket.emit(event, data);
            return true;
        } else {
            console.warn('⚠️ Cannot emit - WebSocket not connected');
            return false;
        }
    }
    
    on(event, callback) {
        if (!this.callbacks[event]) {
            this.callbacks[event] = [];
        }
        this.callbacks[event].push(callback);
    }
    
    off(event, callback) {
        if (this.callbacks[event]) {
            const index = this.callbacks[event].indexOf(callback);
            if (index > -1) {
                this.callbacks[event].splice(index, 1);
            }
        }
    }
    
    trigger(event, data) {
        if (this.callbacks[event]) {
            this.callbacks[event].forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in ${event} callback:`, error);
                }
            });
        }
    }
    
    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
            this.isConnected = false;
        }
    }
    
    getConnectionInfo() {
        return {
            connected: this.isConnected,
            url: this.getSocketUrl(),
            transport: this.socket?.io?.engine?.transport?.name,
            attempts: this.reconnectAttempts
        };
    }
}

// Instancia global
window.wsManager = new WebSocketManager();

// Debug info
console.log('🔧 WebSocket Manager initialized');
console.log('🌐 Target URL:', window.wsManager.getSocketUrl());
console.log('📍 Environment:', window.location.hostname === 'localhost' ? 'Development' : 'Production');