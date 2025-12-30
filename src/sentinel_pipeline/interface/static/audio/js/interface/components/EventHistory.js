/**
 * EventHistory Component
 */
import { Renderer } from '../../presentation/Renderer.js';

export class EventHistory {
    constructor(containerId, maxEvents = 10) {
        this.container = document.getElementById(containerId);
        this.maxEvents = maxEvents;
    }

    addEvent(event) {
        const eventCard = this.createEventCard(event);
        this.container.insertBefore(eventCard, this.container.firstChild);

        // Keep only maxEvents
        while (this.container.children.length > this.maxEvents) {
            this.container.removeChild(this.container.lastChild);
        }
    }

    getAllEvents() {
        // 차트를 위한 이벤트 목록 반환
        const events = [];
        for (let i = 0; i < this.container.children.length; i++) {
            const card = this.container.children[i];
            // 카드에서 이벤트 데이터 추출 (필요시)
        }
        return events;
    }

    createEventCard(event) {
        const card = Renderer.createElement('div', 'bg-red-50 border border-red-200 rounded-lg p-3');

        const header = Renderer.createElement('div', 'flex justify-between items-start mb-1');
        const typeSpan = Renderer.createElement('span', 'text-xs font-bold text-red-600');
        Renderer.setTextContent(typeSpan, event.type);
        
        const timeSpan = Renderer.createElement('span', 'text-[10px] font-mono text-gray-400');
        const timestamp = event.timestamp ? new Date(event.timestamp) : new Date();
        Renderer.setTextContent(timeSpan, timestamp.toLocaleTimeString());

        Renderer.appendChild(header, typeSpan);
        Renderer.appendChild(header, timeSpan);

        const dataDiv = Renderer.createElement('div', 'text-xs text-gray-700 font-mono');
        Renderer.setTextContent(dataDiv, JSON.stringify(event.data, null, 2));

        Renderer.appendChild(card, header);
        Renderer.appendChild(card, dataDiv);

        return card;
    }

    clear() {
        Renderer.removeChildren(this.container);
    }
}

