/**
 * StreamList Component
 */
import { Renderer } from '../../presentation/Renderer.js';

export class StreamList {
    constructor(containerId, onStreamSelect, onStreamDelete) {
        this.container = document.getElementById(containerId);
        this.onStreamSelect = onStreamSelect;
        this.onStreamDelete = onStreamDelete;
        this.selectedStreamId = null;
    }

    render(streams) {
        Renderer.removeChildren(this.container);

        streams.forEach(stream => {
            const item = this.createStreamItem(stream);
            this.container.appendChild(item);
        });
    }

    createStreamItem(stream) {
        const item = Renderer.createElement('div', 'stream-item p-3 cursor-pointer');
        
        if (stream.streamId === this.selectedStreamId) {
            item.classList.add('active');
        }

        item.addEventListener('click', () => {
            this.selectStream(stream.streamId);
            if (this.onStreamSelect) {
                this.onStreamSelect(stream);
            }
        });

        const wrapper = Renderer.createElement('div', 'flex justify-between items-start');
        
        const leftSection = Renderer.createElement('div', 'flex-1');
        const streamId = Renderer.createElement('div', 'text-sm font-bold text-gray-900');
        Renderer.setTextContent(streamId, stream.streamId);
        
        const streamUrl = Renderer.createElement('div', 'text-xs text-gray-500 font-mono mt-1 truncate');
        Renderer.setTextContent(streamUrl, stream.rtspUrl);
        
        Renderer.appendChild(leftSection, streamId);
        Renderer.appendChild(leftSection, streamUrl);

        const statusBadge = this.createStatusBadge(stream.status);

        Renderer.appendChild(wrapper, leftSection);
        Renderer.appendChild(wrapper, statusBadge);
        Renderer.appendChild(item, wrapper);

        return item;
    }

    createStatusBadge(status) {
        const colorMap = {
            'active': 'px-2 py-1 text-[10px] font-bold rounded-full bg-green-50 text-green-600 border border-green-200',
            'error': 'px-2 py-1 text-[10px] font-bold rounded-full bg-red-50 text-red-600 border border-red-200',
            'inactive': 'px-2 py-1 text-[10px] font-bold rounded-full bg-gray-50 text-gray-600 border border-gray-200'
        };
        
        const className = colorMap[status] || colorMap.inactive;
        const badge = Renderer.createElement('span', className);
        Renderer.setTextContent(badge, status.toUpperCase());
        return badge;
    }

    getStatusColor(status) {
        const colorMap = {
            'active': 'green',
            'error': 'red',
            'inactive': 'gray'
        };
        return colorMap[status] || 'gray';
    }

    selectStream(streamId) {
        // Remove active class from all items
        this.container.querySelectorAll('.stream-item').forEach(item => {
            item.classList.remove('active');
        });

        // Add active class to selected item
        const items = Array.from(this.container.querySelectorAll('.stream-item'));
        items.forEach(item => {
            const streamIdElement = item.querySelector('.text-sm.font-bold.text-gray-900');
            if (streamIdElement && streamIdElement.textContent.trim() === streamId) {
                item.classList.add('active');
            }
        });

        this.selectedStreamId = streamId;
    }

    updateStreamStatus(streamId, status) {
        const items = Array.from(this.container.querySelectorAll('.stream-item'));
        items.forEach(item => {
            const streamIdElement = item.querySelector('.text-sm.font-bold.text-gray-900');
            if (streamIdElement && streamIdElement.textContent.trim() === streamId) {
                const statusBadge = item.querySelector('.px-2.py-1');
                if (statusBadge) {
                    const colorMap = {
                        'active': 'px-2 py-1 text-[10px] font-bold rounded-full bg-green-50 text-green-600 border border-green-200',
                        'error': 'px-2 py-1 text-[10px] font-bold rounded-full bg-red-50 text-red-600 border border-red-200',
                        'inactive': 'px-2 py-1 text-[10px] font-bold rounded-full bg-gray-50 text-gray-600 border border-gray-200'
                    };
                    statusBadge.className = colorMap[status] || colorMap.inactive;
                    Renderer.setTextContent(statusBadge, status.toUpperCase());
                }
            }
        });
    }
}

