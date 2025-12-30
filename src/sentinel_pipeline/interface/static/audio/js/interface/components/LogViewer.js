/**
 * LogViewer Component
 */
import { Renderer } from '../../presentation/Renderer.js';

export class LogViewer {
    constructor(containerId, maxLogs = 20) {
        this.container = document.getElementById(containerId);
        this.maxLogs = maxLogs;
        this.allLogs = []; // 모든 로그 저장
        this.filteredNodeId = null; // 현재 필터링된 노드 ID
    }

    addLog(title, value, type = 'default', nodeId = null) {
        const logCard = this.createLogCard(title, value, type);
        logCard.dataset.nodeId = nodeId || ''; // 노드 ID 저장
        
        // 모든 로그 배열에 추가
        this.allLogs.unshift({
            title,
            value,
            type,
            nodeId: nodeId || null,
            element: logCard
        });
        
        // Keep only maxLogs in array
        if (this.allLogs.length > this.maxLogs) {
            const removed = this.allLogs.pop();
            if (removed.element.parentNode) {
                removed.element.parentNode.removeChild(removed.element);
            }
        }
        
        // 필터링 적용
        this.applyFilter();
    }
    
    filterByNode(nodeId) {
        this.filteredNodeId = nodeId;
        this.applyFilter();
    }
    
    clearFilter() {
        this.filteredNodeId = null;
        this.applyFilter();
    }
    
    applyFilter() {
        // 기존 로그 모두 제거
        Renderer.removeChildren(this.container);
        
        // 필터링된 로그만 표시
        const logsToShow = this.filteredNodeId 
            ? this.allLogs.filter(log => log.nodeId === this.filteredNodeId)
            : this.allLogs;
        
        logsToShow.forEach(log => {
            this.container.appendChild(log.element);
        });
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
        this.allLogs = [];
        this.filteredNodeId = null;
    }
}

