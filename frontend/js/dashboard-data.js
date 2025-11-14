// LSCh Dashboard JavaScript - Dynamic Data Loading
// Carga datos reales desde la API PostgreSQL

document.addEventListener('DOMContentLoaded', function() {
    // Verificar si estamos en la página del dashboard
    if (window.location.pathname.includes('/dashboard')) {
        loadDashboardData();
    }
});

async function loadDashboardData() {
    try {
        console.log('🔄 Cargando datos del dashboard...');
        
        // Cargar estadísticas del usuario
        const response = await fetch('/api/user/stats');
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('📊 Datos recibidos:', data);
        
        // Actualizar estadísticas principales
        updateMainStats(data.stats);
        
        // Actualizar información del usuario
        updateUserInfo(data);
        
        // Actualizar gráfico de palabras más usadas
        updateTopWords(data.stats.top_words);
        
        // Actualizar actividad reciente
        updateRecentActivity(data.stats.recent_predictions);
        
        console.log('✅ Dashboard actualizado con datos reales');
        
    } catch (error) {
        console.error('❌ Error cargando datos del dashboard:', error);
        showError('Error al cargar datos del dashboard');
    }
}

function updateMainStats(stats) {
    // Actualizar predicciones totales
    const totalPredictionsEl = document.getElementById('totalPredictions');
    if (totalPredictionsEl) {
        animateNumber(totalPredictionsEl, stats.total_predictions);
    }
    
    // Actualizar confianza promedio
    const avgConfidenceEl = document.getElementById('avgConfidence');
    if (avgConfidenceEl) {
        const percentage = Math.round(stats.avg_confidence * 100);
        avgConfidenceEl.textContent = `${percentage}%`;
    }
    
    // Actualizar palabras únicas
    const uniqueWordsEl = document.getElementById('uniqueWords');
    if (uniqueWordsEl) {
        animateNumber(uniqueWordsEl, stats.unique_words);
    }
    
    // Actualizar mensaje de bienvenida
    const welcomeMessageEl = document.getElementById('welcomeMessage');
    if (welcomeMessageEl && stats.total_predictions > 0) {
        welcomeMessageEl.innerHTML = `¡Bienvenido de vuelta! <span style="color: var(--primary);">${stats.total_predictions}</span> predicciones realizadas`;
    }
}

function updateUserInfo(data) {
    // Actualizar nombre de usuario en el menú
    const usernameEl = document.getElementById('username');
    if (usernameEl && data.username) {
        usernameEl.textContent = data.username;
    }
    
    // Actualizar información en la sección de cuenta
    const userInfoEl = document.getElementById('userInfo');
    if (userInfoEl && data.username) {
        userInfoEl.textContent = data.username;
    }
}

function updateTopWords(topWords) {
    const topWordsContainer = document.getElementById('topWordsList');
    if (!topWordsContainer || !topWords || topWords.length === 0) {
        // Mostrar mensaje de vacío
        if (topWordsContainer) {
            topWordsContainer.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-hand-paper"></i>
                    <p>Aún no has realizado predicciones</p>
                    <a href="/demo" class="btn-secondary">Comenzar Ahora</a>
                </div>
            `;
        }
        return;
    }
    
    const maxCount = topWords[0][1]; // Mayor número de predicciones
    
    topWordsContainer.innerHTML = topWords.map((wordData, index) => {
        const [word, count] = wordData;
        const percentage = Math.round((count / maxCount) * 100);
        
        return `
            <div class="top-word-item">
                <div class="word-rank">${index + 1}</div>
                <div class="word-info">
                    <span class="word-name">${word}</span>
                    <div class="word-bar">
                        <div class="word-fill" data-width="${percentage}"></div>
                    </div>
                </div>
                <div class="word-count">${count}</div>
            </div>
        `;
    }).join('');
    
    // Animar las barras después de un pequeño delay
    setTimeout(() => {
        document.querySelectorAll('.word-fill').forEach(el => {
            const width = el.dataset.width;
            el.style.width = width + '%';
        });
    }, 100);
}

function updateRecentActivity(recentPredictions) {
    const activityContainer = document.getElementById('activityList');
    if (!activityContainer) return;
    
    if (!recentPredictions || recentPredictions.length === 0) {
        activityContainer.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-history"></i>
                <p>No hay actividad reciente</p>
            </div>
        `;
        return;
    }
    
    activityContainer.innerHTML = recentPredictions.map(prediction => {
        const date = new Date(prediction.timestamp);
        const formattedDate = date.toLocaleDateString('es-ES') + ' ' + 
                             date.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' });
        
        const confidence = Math.round(prediction.confidence * 100);
        const confidenceClass = confidence >= 90 ? 'high' : 
                               confidence >= 70 ? 'medium' : 'low';
        
        return `
            <div class="activity-item">
                <div class="activity-icon">
                    <i class="fas fa-hand-paper"></i>
                </div>
                <div class="activity-content">
                    <div class="activity-title">${prediction.word}</div>
                    <div class="activity-time">
                        <i class="fas fa-clock"></i>
                        ${formattedDate}
                    </div>
                </div>
                <div class="activity-confidence">
                    <div class="confidence-badge ${confidenceClass}" data-confidence="${prediction.confidence}">
                        ${confidence}%
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function animateNumber(element, targetNumber) {
    const startNumber = 0;
    const duration = 1000; // 1 segundo
    const startTime = Date.now();
    
    function updateNumber() {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function (ease out)
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const currentNumber = Math.round(startNumber + (targetNumber - startNumber) * easeOut);
        
        element.textContent = currentNumber;
        
        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        }
    }
    
    requestAnimationFrame(updateNumber);
}

function showError(message) {
    // Crear notificación de error
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-error dashboard-alert';
    errorDiv.innerHTML = `
        <i class="fas fa-exclamation-circle"></i>
        <span>${message}</span>
        <button class="alert-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Añadir al container del dashboard
    const container = document.querySelector('.dashboard-container .container');
    if (container) {
        container.insertBefore(errorDiv, container.firstChild);
        
        // Auto-remove después de 5 segundos
        setTimeout(() => {
            if (errorDiv.parentElement) {
                errorDiv.remove();
            }
        }, 5000);
    }
}

// Actualizar datos cada 30 segundos si estamos en el dashboard
if (window.location.pathname.includes('/dashboard')) {
    setInterval(() => {
        console.log('🔄 Actualizando datos del dashboard...');
        loadDashboardData();
    }, 30000);
}