import { BaseComponent } from '../../common/js/BaseComponent.js';

export class StreamCard extends BaseComponent {
    constructor(stream) {
        super(); // No element ID, we create the element
        this.stream = stream;
        this.element = document.createElement('div');
    }

    render() {
        const s = this.stream;
        const status = (s.status || 'IDLE').toLowerCase();
        const isRunning = status === 'running';
        const isError = status === 'error';
        
        const badgeColor = isRunning ? 'bg-green-50 text-green-700 border-green-200' : 
                           isError ? 'bg-red-50 text-red-700 border-red-200' : 'bg-gray-50 text-gray-500 border-gray-200';
        const icon = isRunning ? 'fiber_manual_record' : isError ? 'error' : 'stop';

        this.element.className = "bg-white rounded-2xl shadow-card border border-gray-100 flex flex-col group transition-all duration-300 hover:shadow-lg";
        this.element.innerHTML = `
            <div class="p-5 border-b border-gray-50 flex justify-between items-start">
                <div>
                    <h3 class="font-bold text-gray-900 text-base">${s.stream_id}</h3>
                    <p class="text-xs text-gray-400 font-mono mt-1">ID: ${s.stream_id}</p>
                </div>
                <div class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold border shadow-sm ${badgeColor}">
                    <span class="material-symbols-outlined text-[12px] ${isRunning ? 'animate-pulse' : ''}" style="font-size: 12px;">${icon}</span>
                    ${status.toUpperCase()}
                </div>
            </div>
            
            <div class="p-4 bg-gray-50/30 flex-1">
                 <!-- Metrics -->
                 <div class="grid grid-cols-2 gap-3 mb-4">
                    <div class="p-2.5 rounded-xl bg-white border border-gray-100 shadow-sm flex flex-col items-center">
                        <span class="text-[9px] text-gray-400 font-bold uppercase tracking-wider mb-1">Events</span>
                        <span class="font-mono text-lg font-bold text-gray-900">${s.event_count || 0}</span>
                    </div>
                    <div class="p-2.5 rounded-xl bg-white border border-gray-100 shadow-sm flex flex-col items-center">
                        <span class="text-[9px] text-gray-400 font-bold uppercase tracking-wider mb-1">Decibels</span>
                        <span class="font-mono text-lg font-bold text-teal-600">${s.db_level || '-'} dB</span>
                    </div>
                 </div>

                 <!-- URL Info -->
                 <div class="space-y-2">
                    <div class="flex items-center gap-2 text-xs text-gray-500">
                        <span class="w-1.5 h-1.5 rounded-full bg-blue-400"></span>
                        <span class="font-bold">Input:</span>
                        <span class="font-mono truncate flex-1">${s.input_url || 'Mic'}</span>
                    </div>
                 </div>
            </div>

            <!-- Controls -->
            <div class="p-4 border-t border-gray-100 bg-gray-50/50 flex gap-2">
                ${isRunning 
                    ? `<button onclick="audioApp.stopStream('${s.stream_id}')" class="flex-1 bg-white hover:bg-red-50 text-red-600 border border-gray-200 py-2 rounded-xl text-sm font-bold transition shadow-sm">Stop</button>` 
                    : `<button onclick="audioApp.startStream('${s.stream_id}')" class="flex-1 bg-teal-600 hover:bg-teal-700 text-white border border-transparent py-2 rounded-xl text-sm font-bold transition shadow-sm">Start</button>`
                }
                <button onclick="audioApp.deleteStream('${s.stream_id}')" class="w-10 flex items-center justify-center text-gray-400 hover:text-red-600 bg-white border border-transparent hover:border-gray-200 rounded-xl transition"><span class="material-symbols-outlined text-sm">delete</span></button>
            </div>
        `;
        return this.element;
    }
}
