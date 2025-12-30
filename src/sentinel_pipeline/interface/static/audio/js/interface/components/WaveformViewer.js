/**
 * WaveformViewer Component
 */
export class WaveformViewer {
    constructor(canvasId, dbIndicatorId) {
        this.canvas = document.getElementById(canvasId);
        this.dbIndicator = document.getElementById(dbIndicatorId);
        this.ctx = null;
        this.animationId = null;
        this.mode = 'idle';
        
        this.init();
    }

    init() {
        if (!this.canvas) return;
        
        this.ctx = this.canvas.getContext('2d');
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.startAnimation();
    }

    resize() {
        if (this.canvas) {
            this.canvas.width = this.canvas.offsetWidth;
            this.canvas.height = this.canvas.offsetHeight;
        }
    }

    setMode(mode) {
        this.mode = mode;
        this.updateDBIndicator();
    }

    updateDBIndicator() {
        if (!this.dbIndicator) return;

        const dbValues = {
            'scream': { value: '-2.1 dB (PEAK)', color: '#ef4444' },
            'speech': { value: '-14.5 dB', color: '#14b8a6' },
            'wind': { value: '-18.2 dB', color: '#9ca3af' },
            'idle': { value: '-inf dB', color: '#6b7280' }
        };

        const db = dbValues[this.mode] || dbValues.idle;
        this.dbIndicator.textContent = db.value;
        this.dbIndicator.style.color = db.color;
    }

    startAnimation() {
        const draw = () => {
            if (!this.ctx || !this.canvas) return;

            this.ctx.fillStyle = '#111827';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

            if (this.mode !== 'idle') {
                this.ctx.strokeStyle = this.getColor();
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();

                const centerY = this.canvas.height / 2;
                const amplitude = this.getAmplitude();
                const frequency = this.getFrequency();

                for (let x = 0; x < this.canvas.width; x++) {
                    const noise = this.mode === 'wind' ? Math.random() * 10 : Math.random() * 5;
                    const y = centerY + Math.sin(x * frequency + Date.now() / 100) * amplitude + noise;
                    this.ctx.lineTo(x, y);
                }
                this.ctx.stroke();
            }

            this.animationId = requestAnimationFrame(draw);
        };

        draw();
    }

    getColor() {
        const colors = {
            'scream': '#ef4444',
            'speech': '#14b8a6',
            'wind': '#9ca3af',
            'idle': '#374151'
        };
        return colors[this.mode] || colors.idle;
    }

    getAmplitude() {
        const amplitudes = {
            'scream': 50,
            'speech': 20,
            'wind': 30,
            'idle': 2
        };
        return amplitudes[this.mode] || amplitudes.idle;
    }

    getFrequency() {
        const frequencies = {
            'scream': 0.2,
            'speech': 0.1,
            'wind': 0.02,
            'idle': 0.05
        };
        return frequencies[this.mode] || frequencies.idle;
    }

    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
}

