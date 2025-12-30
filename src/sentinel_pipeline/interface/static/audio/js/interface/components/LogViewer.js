/**
 * LogViewer Component
 */
import { Renderer } from '../../presentation/Renderer.js';

export class LogViewer {
    constructor(containerId, maxLogs = 20) {
        this.container = document.getElementById(containerId);
        this.maxLogs = maxLogs;
    }

    addLog(title, value, type = 'default') {
        const logCard = this.createLogCard(title, value, type);
        this.container.insertBefore(logCard, this.container.firstChild);

        // Keep only maxLogs
        while (this.container.children.length > this.maxLogs) {
            this.container.removeChild(this.container.lastChild);
        }
    }

    createLogCard(title, value, type) {
        const colors = {
            default: { border: 'border-l-gray-300', text: 'text-gray-800' },
            brand: { border: 'border-l-brand', text: 'text-brand' },
            alert: { border: 'border-l-red-500', text: 'text-red-600' },
            success: { border: 'border-l-green-500', text: 'text-green-600' }
        };

        const color = colors[type] || colors.default;
        const card = Renderer.createElement('div', 
            `bg-white border border-gray-100 border-l-4 p-3 rounded shadow-sm ${color.border}`);

        const header = Renderer.createElement('div', 'flex justify-between items-center mb-1');
        const titleSpan = Renderer.createElement('span', 'text-[10px] font-bold text-gray-400 uppercase');
        Renderer.setTextContent(titleSpan, title);
        
        const timeSpan = Renderer.createElement('span', 'text-[10px] font-mono text-gray-300');
        Renderer.setTextContent(timeSpan, new Date().toLocaleTimeString());

        Renderer.appendChild(header, titleSpan);
        Renderer.appendChild(header, timeSpan);

        const valueDiv = Renderer.createElement('div', `text-sm font-mono font-bold ${color.text}`);
        Renderer.setTextContent(valueDiv, value);

        Renderer.appendChild(card, header);
        Renderer.appendChild(card, valueDiv);

        return card;
    }

    clear() {
        Renderer.removeChildren(this.container);
    }
}

