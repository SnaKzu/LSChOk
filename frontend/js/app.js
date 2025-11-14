// ================================================
// LSP Recognition - Main JavaScript
// ================================================

// Smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Mobile menu toggle
const hamburger = document.querySelector('.hamburger');
const navMenu = document.querySelector('.nav-menu');

if (hamburger) {
    hamburger.addEventListener('click', () => {
        navMenu.classList.toggle('active');
        hamburger.classList.toggle('active');
    });
}

// Close menu when clicking nav links
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
        navMenu.classList.remove('active');
        hamburger.classList.remove('active');
    });
});

// Navbar scroll effect
let lastScroll = 0;
const navbar = document.querySelector('.navbar');

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;
    
    if (currentScroll > 100) {
        navbar.style.padding = '10px 0';
        navbar.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.padding = '20px 0';
        navbar.style.boxShadow = '0 1px 2px 0 rgba(0, 0, 0, 0.05)';
    }
    
    lastScroll = currentScroll;
});

// Active nav link based on scroll position
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-link');

window.addEventListener('scroll', () => {
    let current = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        
        if (window.pageYOffset >= sectionTop - 200) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// Counter animation for stats
function animateCounter(element, target, duration = 2000) {
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;
    
    const updateCounter = () => {
        current += increment;
        
        if (current < target) {
            element.textContent = Math.ceil(current);
            requestAnimationFrame(updateCounter);
        } else {
            element.textContent = target;
        }
    };
    
    updateCounter();
}

// Intersection Observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
            
            // Animate counters when stats section is visible
            if (entry.target.classList.contains('hero-stats')) {
                const counters = entry.target.querySelectorAll('.stat-number');
                counters.forEach(counter => {
                    const target = parseInt(counter.getAttribute('data-target'));
                    animateCounter(counter, target);
                });
            }
            
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe elements for animation
document.querySelectorAll('.feature-card, .tech-card, .vocabulary-card, .hero-stats').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
    observer.observe(el);
});

// Fetch vocabulary from API
async function loadVocabulary() {
    const vocabularyGrid = document.getElementById('vocabulary-list');
    
    try {
        const response = await fetch('/api/vocabulary');
        const data = await response.json();
        
        // Clear skeleton loaders
        vocabularyGrid.innerHTML = '';
        
        // Icon mapping for words
        const iconMap = {
            'hola': 'fa-hand-wave',
            'adios': 'fa-hand-peace',
            'gracias': 'fa-heart',
            'por_favor': 'fa-hands-praying',
            'buenos_dias': 'fa-sun',
            'buenas_tardes': 'fa-cloud-sun',
            'buenas_noches': 'fa-moon',
            'bien': 'fa-thumbs-up',
            'mal': 'fa-thumbs-down',
            'mas_o_menos': 'fa-hand-point-right',
            'como_estas': 'fa-hand-holding-heart',
            'me_ayudas': 'fa-hands-helping',
            'disculpa': 'fa-hand-holding',
            'default': 'fa-hand-paper'
        };
        
        // Create vocabulary cards
        Object.entries(data.vocabulary).forEach(([wordId, wordLabel]) => {
            const card = document.createElement('div');
            card.className = 'vocabulary-card';
            
            const iconClass = iconMap[wordId] || iconMap['default'];
            
            card.innerHTML = `
                <div class="vocabulary-icon">
                    <i class="fas ${iconClass}"></i>
                </div>
                <div class="vocabulary-word">${wordLabel}</div>
                <div class="vocabulary-id">${wordId.replace(/_/g, ' ')}</div>
            `;
            
            vocabularyGrid.appendChild(card);
            
            // Observe for animation
            card.style.opacity = '0';
            card.style.transform = 'scale(0.9)';
            card.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
            observer.observe(card);
        });
        
    } catch (error) {
        console.error('Error loading vocabulary:', error);
        vocabularyGrid.innerHTML = `
            <div class="error-message" style="grid-column: 1 / -1; text-align: center; color: var(--danger);">
                <i class="fas fa-exclamation-circle" style="font-size: 3rem; margin-bottom: 15px;"></i>
                <p>Error cargando vocabulario. Asegúrate de que el servidor esté ejecutándose.</p>
            </div>
        `;
    }
}

// Check server health
async function checkServerHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        console.log('Server Status:', data);
        
        if (!data.model_loaded) {
            console.warn('Warning: Model not loaded on server');
        }
        
        return data;
    } catch (error) {
        console.error('Server health check failed:', error);
        return null;
    }
}

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    // Load vocabulary
    loadVocabulary();
    
    // Check server health
    checkServerHealth();
    
    // Add loading animation to buttons
    document.querySelectorAll('.btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            if (!this.classList.contains('loading')) {
                this.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    this.style.transform = '';
                }, 100);
            }
        });
    });
    
    // Easter egg: Konami code
    let konamiCode = [];
    const konamiSequence = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];
    
    document.addEventListener('keydown', (e) => {
        konamiCode.push(e.key);
        konamiCode = konamiCode.slice(-10);
        
        if (konamiCode.join(',') === konamiSequence.join(',')) {
            document.body.style.animation = 'rainbow 2s linear infinite';
            setTimeout(() => {
                document.body.style.animation = '';
            }, 5000);
        }
    });
});

// Rainbow animation for easter egg
const style = document.createElement('style');
style.textContent = `
    @keyframes rainbow {
        0% { filter: hue-rotate(0deg); }
        100% { filter: hue-rotate(360deg); }
    }
`;
document.head.appendChild(style);

// Performance monitoring
if ('performance' in window) {
    window.addEventListener('load', () => {
        const perfData = performance.timing;
        const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
        console.log(`Page loaded in ${pageLoadTime}ms`);
    });
}

// Service Worker registration (for PWA - optional)
if ('serviceWorker' in navigator) {
    // Uncomment to enable PWA functionality
    /*
    navigator.serviceWorker.register('/sw.js')
        .then(reg => console.log('Service Worker registered:', reg))
        .catch(err => console.log('Service Worker registration failed:', err));
    */
}

// Export functions for use in other scripts
window.LSP = {
    checkServerHealth,
    loadVocabulary,
    animateCounter
};
