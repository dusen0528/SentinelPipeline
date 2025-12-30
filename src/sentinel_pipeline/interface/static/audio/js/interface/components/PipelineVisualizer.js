/**
 * PipelineVisualizer Component
 */
import { Renderer } from '../../presentation/Renderer.js';

export class PipelineVisualizer {
    constructor() {
        this.nodes = [1, 2, 3, 5];
        this.particle = document.getElementById('particle');
    }

    reset() {
        this.nodes.forEach(id => this.setNodeStatus(id, 'idle'));
        this.hideParticle();
    }

    setNodeStatus(nodeId, status) {
        const node = document.getElementById(`node-${nodeId}`);
        if (!node) return;

        const badge = node.querySelector('.status-badge');
        if (!badge) return;

        // Remove all status classes
        node.classList.remove('active', 'alert', 'opacity-50');
        badge.className = 'status-badge text-[10px] font-bold bg-gray-100 text-gray-400 px-2.5 py-1 rounded-full border border-gray-200';
        Renderer.setTextContent(badge, 'IDLE');

        if (status === 'processing') {
            node.classList.add('active');
            badge.className = 'status-badge text-[10px] font-bold bg-brand-bg text-brand px-2.5 py-1 rounded-full border border-brand-light/30 animate-pulse';
            Renderer.setTextContent(badge, 'PROCESSING');
        } else if (status === 'alert') {
            node.classList.add('alert');
            badge.className = 'status-badge text-[10px] font-bold bg-red-100 text-red-600 px-2.5 py-1 rounded-full border border-red-200';
            Renderer.setTextContent(badge, 'ALERT');
        } else if (status === 'pass') {
            badge.className = 'status-badge text-[10px] font-bold bg-green-50 text-green-600 px-2.5 py-1 rounded-full border border-green-200';
            Renderer.setTextContent(badge, 'PASS');
        } else if (status === 'drop') {
            node.classList.add('opacity-50');
            badge.className = 'status-badge text-[10px] font-bold bg-gray-200 text-gray-500 px-2.5 py-1 rounded-full';
            Renderer.setTextContent(badge, 'DROP');
        }
    }

    moveParticle(nodeId) {
        const node = document.getElementById(`node-${nodeId}`);
        if (!node || !this.particle) return;

        const container = document.getElementById('pipeline-container');
        const nodeRect = node.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();

        const top = nodeRect.top - containerRect.top + nodeRect.height / 2 - 5;
        const left = nodeRect.left - containerRect.left + nodeRect.width / 2 - 5;

        this.particle.style.top = `${top}px`;
        this.particle.style.left = `${left}px`;
        this.particle.style.opacity = '1';
    }

    hideParticle() {
        if (this.particle) {
            this.particle.style.opacity = '0';
        }
    }

    setParticleColor(color) {
        if (this.particle) {
            this.particle.style.backgroundColor = color;
        }
    }

    showSTTOutput(text, isAlert = false) {
        const sttBox = document.getElementById('stt-output-box');
        const sttText = document.getElementById('stt-text');
        
        if (sttBox && sttText) {
            sttBox.classList.remove('hidden');
            Renderer.setTextContent(sttText, text);
            
            if (isAlert) {
                sttText.className = 'text-xs text-red-400 font-mono font-bold';
            } else {
                sttText.className = 'text-xs text-gray-300 font-mono';
            }
        }
    }

    hideSTTOutput() {
        const sttBox = document.getElementById('stt-output-box');
        if (sttBox) {
            sttBox.classList.add('hidden');
        }
    }
}

