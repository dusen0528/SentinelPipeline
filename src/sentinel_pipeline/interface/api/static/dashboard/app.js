// --- Configuration ---
const USE_MOCK_DATA = false; // 실제 API 사용
const API_BASE = '/api';

// --- Alarm System ---
let alarms = []; // 알람 목록
let previousStreamStates = {}; // 이전 스트림 상태 추적
let alarmModal = null;

// --- Global State ---
let latestDashboardData = null;

// --- API Functions ---
const api = {
    getDashboardData: async () => {
        if (USE_MOCK_DATA) {
            // Mock data for the new combined structure
            return {
                system: { cpu_percent: 45, memory_total_mb: 8192, memory_used_mb: 3000, memory_percent: 36.6 },
                process: { cpu_percent: 15, memory_mb: 512 },
                streams: [
                    { stream_id: "cam_01", status: "RUNNING", is_healthy: true, input_url: "rtsp://192.168.1.101/main", output_url: "rtsp://srv:8554/cam_01-blur", fps: 24.1, avg_latency_ms: 50, frame_count: 1000, event_count: 10, error_count: 0, last_error: null, config: { anonymize_method: 'pixelate', blur_strength: 10 } },
                    { stream_id: "cam_02", status: "RUNNING", is_healthy: true, input_url: "rtsp://192.168.1.102/lobby", output_url: "rtsp://srv:8554/cam_02-blur", fps: 30.0, avg_latency_ms: 30, frame_count: 2000, event_count: 20, error_count: 0, last_error: null, config: { anonymize_method: 'blur', blur_strength: 20 } },
                    { stream_id: "cam_03", status: "STOPPED", is_healthy: false, input_url: "rtsp://192.168.1.103/park", output_url: "rtsp://srv:8554/cam_03-blur", fps: 0, avg_latency_ms: 0, frame_count: 0, event_count: 0, error_count: 0, last_error: null, config: { anonymize_method: 'solid', blur_strength: 0 } },
                    { stream_id: "cam_04", status: "ERROR", is_healthy: false, input_url: "rtsp://192.168.1.104/ware", output_url: "rtsp://srv:8554/cam_04-blur", fps: 0, avg_latency_ms: 0, frame_count: 0, event_count: 0, error_count: 1, last_error: "Connection refused", config: { anonymize_method: 'pixelate', blur_strength: 15 } },
                    { stream_id: "cam_05", status: "RUNNING", is_healthy: true, input_url: "rtsp://192.168.1.105/elv", output_url: "rtsp://srv:8554/cam_05-blur", fps: 15.0, avg_latency_ms: 70, frame_count: 500, event_count: 5, error_count: 0, last_error: null, config: { anonymize_method: 'mosaic', blur_strength: 12 } },
                ],
                modules: { FaceBlurModule: { faces_detected: 7, total_processed: 100, error_count: 0, timeout_count: 0 } }
            };
        }
        try {
            const res = await fetch(`${API_BASE}/dashboard/stats`);
            if (!res.ok) {
                const errText = await res.text();
                throw new Error(errText || `HTTP ${res.status}`);
            }
            const data = await res.json();
            latestDashboardData = data;
            return data;
        } catch (error) {
            console.error('Failed to fetch dashboard data:', error);
            throw error;
        }
    },

    controlStream: async (id, action) => {
        if (action === 'start') {
            if (!latestDashboardData) {
                throw new Error("Dashboard data not loaded yet.");
            }
            const stream = latestDashboardData.streams.find(s => s.stream_id === id);
            if (!stream || !stream.config || !stream.config.rtsp_url) {
                throw new Error('Stream config not found; please edit stream to set rtsp_url');
            }
            const payload = {
                rtsp_url: stream.config.rtsp_url,
                output_url: stream.config.output_url || undefined,
                max_fps: stream.config.max_fps,
                downscale: stream.config.downscale,
            };
            const res = await fetch(`${API_BASE}/streams/${encodeURIComponent(id)}/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!res.ok) {
                const errorText = await res.text();
                throw new Error(errorText || `HTTP ${res.status}`);
            }
            return await res.json();
        } else {
             // stop, restart
            const res = await fetch(`${API_BASE}/streams/${encodeURIComponent(id)}/${action}`, { 
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            });
            if (!res.ok) {
                const errorData = await res.json().catch(() => ({ message: `HTTP ${res.status}` }));
                throw new Error(errorData.message || `HTTP ${res.status}`);
            }
            return await res.json();
        }
    },
    deleteStream: async (id) => {
        // DELETE 엔드포인트 사용
        try {
            const res = await fetch(`${API_BASE}/streams/${encodeURIComponent(id)}`, { 
                method: 'DELETE',
                headers: {'Content-Type': 'application/json'}
            });
            if (!res.ok) {
                const errorData = await res.json().catch(() => ({ message: `HTTP ${res.status}` }));
                throw new Error(errorData.message || `HTTP ${res.status}`);
            }
            return await res.json();
        } catch (error) {
            console.error('Failed to delete stream:', error);
            throw error;
        }
    },
    saveStream: async (data) => {
        // 실제 API는 start 엔드포인트 사용
        // stream_id는 name 필드를 사용하거나 자동 생성
        const streamId = data.id || data.name || `stream_${Date.now()}`;
        const maxFpsInput = document.getElementById('input-max-fps');
        const downscaleInput = document.getElementById('input-downscale');
        const skipIntervalInput = document.getElementById('input-skip-interval');
        
        const maxFps = maxFpsInput ? parseInt(maxFpsInput.value) || DEFAULT_VALUES.max_fps : DEFAULT_VALUES.max_fps;
        const downscale = downscaleInput ? parseFloat(downscaleInput.value) || DEFAULT_VALUES.downscale : DEFAULT_VALUES.downscale;
        const skipInterval = skipIntervalInput ? parseInt(skipIntervalInput.value) || DEFAULT_VALUES.skip_interval : DEFAULT_VALUES.skip_interval;
        
        const payload = {
            rtsp_url: data.input_url,
            output_url: data.output_url && data.output_url !== 'Auto-generated' ? data.output_url : undefined,
            max_fps: maxFps,
            downscale: downscale
        };
        
        try {
            // 스트림 시작
            const res = await fetch(`${API_BASE}/streams/${encodeURIComponent(streamId)}/start`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            if (!res.ok) {
                const errorText = await res.text();
                throw new Error(errorText || `HTTP ${res.status}`);
            }
            const result = await res.json();
            
            // skip_interval은 모듈 설정이므로 별도로 업데이트
            if (skipInterval !== DEFAULT_VALUES.skip_interval) {
                try {
                    await fetch(`${API_BASE}/config/modules/FaceBlurModule`, {
                        method: 'PATCH',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            options: {
                                skip_interval: skipInterval
                            }
                        })
                    });
                } catch (configError) {
                    console.warn('Failed to update skip_interval:', configError);
                    // skip_interval 업데이트 실패해도 스트림 시작은 성공으로 처리
                }
            }
            
            return result;
        } catch (error) {
            console.error('Failed to save stream:', error);
            throw error;
        }
    }
};

// --- UI Rendering Functions ---
function getStatusBadge(status) {
    const styles = {
        running: 'bg-green-50 text-green-700 border-green-200',
        starting: 'bg-yellow-50 text-yellow-700 border-yellow-200',
        reconnecting: 'bg-orange-50 text-orange-700 border-orange-200',
        stopping: 'bg-gray-50 text-gray-500 border-gray-200',
        stopped: 'bg-gray-50 text-gray-500 border-gray-200',
        error: 'bg-red-50 text-red-700 border-red-200',
        idle: 'bg-gray-50 text-gray-400 border-gray-200'
    };
    const icons = { 
        running: 'fiber_manual_record', 
        starting: 'sync', 
        reconnecting: 'autorenew', 
        stopping: 'stop', 
        stopped: 'stop', 
        error: 'error',
        idle: 'pause_circle'
    };
    const label = status === 'running' ? 'LIVE' : status.toUpperCase();
    const style = styles[status] || styles.stopped;
    const icon = icons[status] || icons.stopped;
    
    return `
    <div class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold border shadow-sm ${style}">
        <span class="material-symbols-outlined text-[12px] ${status === 'running' ? 'animate-pulse text-green-600' : ''} ${status === 'starting' || status === 'reconnecting' ? 'animate-spin' : ''}" style="font-size: 12px;">${icon}</span>
        ${label}
    </div>`;
}

function renderStreamCard(s) {
    const status = (s.status || 'IDLE').toLowerCase();
    const isRunning = status === 'running';
    const isError = status === 'error';
    const isStarting = status === 'starting' || status === 'reconnecting';
    const errorMessage = s.last_error || '';
    
    // Config may not exist on newly created streams from some endpoints
    const config = s.config || {};
    const blurSettings = config.blur_settings || { anonymize_method: 'pixelate', blur_strength: 31 };

    return `
    <div class="bg-white rounded-2xl shadow-card hover:shadow-lg hover:ring-2 hover:ring-brand-light/20 border border-gray-100 transition-all duration-300 flex flex-col group animate-fade-in cursor-pointer" data-stream-id="${s.stream_id}" data-status="${status}" onclick="openChartModal('${s.stream_id}', '${s.stream_id}')">
        <div class="p-5 border-b border-gray-50 flex justify-between items-start">
            <div class="flex-1 min-w-0 pr-3">
                <h3 class="font-bold text-gray-900 text-base truncate" title="${s.stream_id}">${s.stream_id}</h3>
                <p class="text-xs text-gray-400 font-mono mt-1 tracking-tight truncate">ID: ${s.stream_id}</p>
                ${(isError || errorMessage) ? `<p class="text-xs text-red-600 mt-1 truncate error-msg" title="${errorMessage}">⚠️ ${errorMessage || 'Unknown Error'}</p>` : ''}
            </div>
            <div class="flex-shrink-0" data-status-badge>${getStatusBadge(status)}</div>
        </div>
        <div class="p-4 grid grid-cols-3 gap-3 bg-gray-50/30">
            <div class="p-2.5 rounded-xl bg-white border border-gray-100 shadow-sm flex flex-col items-center">
                <span class="text-[9px] text-gray-400 font-bold uppercase tracking-wider mb-1">FPS</span>
                <span class="font-mono text-lg font-bold ${isRunning ? 'text-gray-900' : 'text-gray-300'}" data-metric="fps" data-color-running="text-gray-900">${isRunning ? s.fps.toFixed(1) : '-'}</span>
            </div>
            <div class="p-2.5 rounded-xl bg-white border border-gray-100 shadow-sm flex flex-col items-center">
                <span class="text-[9px] text-gray-400 font-bold uppercase tracking-wider mb-1">Latency</span>
                <span class="font-mono text-lg font-bold ${isRunning ? 'text-brand' : 'text-gray-300'}" data-metric="latency" data-color-running="text-brand">${isRunning ? s.avg_latency_ms.toFixed(0) : '-'}</span>
            </div>
            <div class="p-2.5 rounded-xl bg-white border border-gray-100 shadow-sm flex flex-col items-center">
                <span class="text-[9px] text-gray-400 font-bold uppercase tracking-wider mb-1">Errors</span>
                <span class="font-mono text-lg font-bold ${isRunning && s.error_count > 0 ? 'text-red-600' : 'text-gray-300'}" data-metric="errors" data-color-running="text-red-600">${isRunning ? s.error_count : '-'}</span>
            </div>
        </div>
        <div class="px-5 py-4 space-y-3 flex-1 url-container">
            <div class="flex items-center justify-between group/url cursor-pointer hover:bg-gray-50 p-2 -mx-2 rounded-lg transition" onclick="event.stopPropagation(); window.copyToClipboard(this.dataset.url)" data-url-type="input" data-url="${s.input_url || ''}">
                <div class="flex items-center gap-2"><span class="w-1.5 h-1.5 rounded-full bg-blue-400"></span><span class="text-xs text-gray-500 font-bold">Input</span></div>
                <div class="flex items-center gap-1.5 max-w-[140px]"><span class="text-xs text-gray-400 font-mono truncate url-text">${s.input_url || 'N/A'}</span><span class="material-symbols-outlined text-[12px] text-gray-300 group-hover/url:text-brand transition">content_copy</span></div>
            </div>
            <div class="flex items-center justify-between group/url cursor-pointer hover:bg-gray-50 p-2 -mx-2 rounded-lg transition" onclick="event.stopPropagation(); window.copyToClipboard(this.dataset.url)" data-url-type="output" data-url="${s.output_url || ''}">
                 <div class="flex items-center gap-2"><span class="w-1.5 h-1.5 rounded-full bg-brand-light"></span><span class="text-xs text-gray-500 font-bold">Output</span></div>
                <div class="flex items-center gap-1.5 max-w-[140px]"><span class="text-xs text-gray-400 font-mono truncate url-text">${s.output_url || 'N/A'}</span><span class="material-symbols-outlined text-[12px] text-gray-300 group-hover/url:text-brand transition">content_copy</span></div>
            </div>
            <div class="pt-2 flex items-center gap-2">
                <span class="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded-md font-medium border border-gray-200 capitalize flex items-center gap-1"><span class="material-symbols-outlined text-[12px]">enhanced_encryption</span>${(blurSettings.anonymize_method || 'blur')}</span>
                <span class="text-[10px] text-gray-400 font-bold">STR: ${blurSettings.blur_strength}</span>
            </div>
        </div>
        <div class="p-4 border-t border-gray-100 bg-gray-50/50 flex justify-between items-center gap-3 rounded-b-2xl btn-container" onclick="event.stopPropagation()">
            <div class="flex-1">
                ${status === 'running' 
                    ? `<button onclick="handleAction('${s.stream_id}', 'stop')" class="w-full flex items-center justify-center gap-1.5 bg-white hover:bg-red-50 text-red-600 border border-gray-200 hover:border-red-200 py-2 rounded-xl text-sm font-bold transition shadow-sm active:scale-95"><span class="material-symbols-outlined text-sm">stop_circle</span>Stop</button>` 
                    : (status === 'error' || status === 'stopped')
                    ? `<button onclick="handleRetry('${s.stream_id}')" class="w-full flex items-center justify-center gap-1.5 bg-orange-500 hover:bg-orange-600 text-white py-2 rounded-xl text-sm font-bold transition shadow-md active:scale-95"><span class="material-symbols-outlined text-sm">refresh</span>Retry</button>`
                    : isStarting
                    ? `<button disabled class="w-full flex items-center justify-center gap-1.5 bg-gray-100 text-gray-400 py-2 rounded-xl text-sm font-bold cursor-not-allowed"><span class="material-symbols-outlined text-sm animate-spin">sync</span>Starting...</button>`
                    : `<button onclick="handleAction('${s.stream_id}', 'start')" class="w-full flex items-center justify-center gap-1.5 bg-brand text-white hover:bg-brand-hover py-2 rounded-xl text-sm font-bold transition shadow-md shadow-teal-700/10 active:scale-95"><span class="material-symbols-outlined text-sm">play_circle</span>Start</button>`
                }
            </div>
            <div class="flex gap-1">
                 <button onclick="editStream('${s.stream_id}')" class="w-9 h-9 flex items-center justify-center text-gray-400 hover:text-brand hover:bg-white border border-transparent hover:border-gray-200 rounded-lg transition shadow-none hover:shadow-sm"><span class="material-symbols-outlined text-sm">settings</span></button>
                 <button onclick="deleteStream('${s.stream_id}')" class="w-9 h-9 flex items-center justify-center text-gray-400 hover:text-red-600 hover:bg-white border border-transparent hover:border-gray-200 rounded-lg transition shadow-none hover:shadow-sm"><span class="material-symbols-outlined text-sm">delete</span></button>
            </div>
        </div>
    </div>`;
}

// No need to store previous data - we always use partial updates

async function renderApp() {
    const data = await api.getDashboardData();
    const { system, process: proc, streams, modules } = data;

    // --- Update Header Stats ---
    const runningStreams = streams.filter(s => s.status.toLowerCase() === 'running');
    const totalFps = runningStreams.reduce((sum, s) => sum + (s.fps || 0), 0);
    const averageFps = runningStreams.length > 0 ? totalFps / runningStreams.length : 0;
    
    document.getElementById('stat-total').innerText = streams.length;
    document.getElementById('stat-running').innerText = runningStreams.length;
    
    // CPU Info
    const systemCpu = Math.min(Math.max(system.cpu_percent || 0, 0), 100);
    document.getElementById('stat-cpu').innerText = systemCpu.toFixed(1) + '%';
    document.getElementById('bar-cpu').style.width = systemCpu + '%';
    const processCpuEl = document.getElementById('stat-process-cpu');
    if (processCpuEl) {
        processCpuEl.innerText = proc.cpu_percent.toFixed(1) + '%';
    }
    
    // Memory Info
    const memText = system.memory_total_mb > 0 
        ? `${Math.round(system.memory_used_mb)} / ${Math.round(system.memory_total_mb)} MB`
        : `${Math.round(system.memory_used_mb)} MB`;
    document.getElementById('stat-mem').innerText = memText;
    const memPercent = system.memory_percent || 0;
    document.getElementById('bar-mem').style.width = Math.min(memPercent, 100) + '%';
    const processMemEl = document.getElementById('stat-process-mem');
    if (processMemEl) {
        processMemEl.innerText = Math.round(proc.memory_mb || 0) + ' MB';
    }
    const memPercentEl = document.getElementById('stat-mem-percent');
    if (memPercentEl) {
        memPercentEl.innerText = memPercent.toFixed(1) + '%';
    }
    
    document.getElementById('stat-fps').innerText = averageFps.toFixed(1);

    // --- Update Stream Grid ---
    const grid = document.getElementById('stream-grid');
    if (streams.length === 0) {
        if (grid.children.length === 0 || !grid.querySelector('.col-span-full')) {
            grid.innerHTML = `<div class="col-span-full py-16 flex flex-col items-center justify-center text-gray-400 bg-white rounded-2xl border border-dashed border-gray-200 shadow-sm"><div class="p-4 bg-gray-50 rounded-full mb-3"><span class="material-symbols-outlined text-4xl text-gray-300">videocam_off</span></div><p class="font-bold text-gray-600 text-lg">No streams configured</p><button onclick="openModal()" class="mt-6 text-brand hover:text-brand-hover font-bold text-sm bg-brand-bg px-4 py-2 rounded-lg border border-brand/20 transition">Add Stream Now</button></div>`;
        }
    } else {
        const existingCardIds = Array.from(grid.querySelectorAll('[data-stream-id]')).map(card => card.getAttribute('data-stream-id'));
        const newStreamIds = streams.map(s => s.stream_id);
        const hasNewStreams = newStreamIds.some(id => !existingCardIds.includes(id));
        const hasRemovedStreams = existingCardIds.some(id => !newStreamIds.includes(id));
        
        if (grid.children.length === 0 || hasNewStreams || hasRemovedStreams) {
            grid.innerHTML = streams.map(renderStreamCard).join('');
        } else {
            updateStreamCards(streams, modules);
        }
        
        checkStreamStatusChanges(streams);
        
        if (currentStreamId) {
            const selectedStream = streams.find(s => s.stream_id === currentStreamId);
            if (selectedStream) {
                updateChartData(selectedStream, modules);
            }
        }
    }
}

// Update existing stream cards without full re-render
function updateStreamCards(streams, modules) {
    streams.forEach(stream => {
        const card = document.querySelector(`[data-stream-id="${stream.stream_id}"]`);
        if (card) {
            const status = (stream.status || 'IDLE').toLowerCase();
            const isRunning = status === 'running';
            const isError = status === 'error';
            const isStarting = status === 'starting' || status === 'reconnecting';
            const errorMessage = stream.last_error || '';
            
            // Update metrics
            const updateMetric = (metric, value, format = val => val) => {
                const el = card.querySelector(`[data-metric="${metric}"]`);
                if (el) {
                    el.textContent = isRunning ? format(value) : '-';
                    const colorClass = isRunning ? (el.dataset.colorRunning || 'text-gray-900') : 'text-gray-300';
                    el.className = `font-mono text-lg font-bold ${colorClass}`;
                }
            };
            
            updateMetric('fps', stream.fps, v => v.toFixed(1));
            updateMetric('latency', stream.avg_latency_ms, v => v.toFixed(0));
            updateMetric('errors', stream.error_count);

            // Update URLs and copy data
            const inputContainer = card.querySelector('[data-url-type="input"]');
            const outputContainer = card.querySelector('[data-url-type="output"]');
            if (inputContainer) {
                inputContainer.dataset.url = stream.input_url || '';
                const textEl = inputContainer.querySelector('.url-text');
                if (textEl) textEl.textContent = stream.input_url || 'N/A';
            }
            if (outputContainer) {
                outputContainer.dataset.url = stream.output_url || '';
                const textEl = outputContainer.querySelector('.url-text');
                if (textEl) textEl.textContent = stream.output_url || 'N/A';
            }

            // Update status badge and button if status changed
            const statusBadgeContainer = card.querySelector('[data-status-badge]');
            if (statusBadgeContainer) {
                const currentStatus = card.getAttribute('data-status');
                if (currentStatus !== status) {
                    statusBadgeContainer.innerHTML = getStatusBadge(status);
                    card.setAttribute('data-status', status);
                    
                    const buttonContainer = card.querySelector('.btn-container .flex-1');
                    if (buttonContainer) {
                        if (status === 'running') {
                            buttonContainer.innerHTML = `<button onclick="handleAction('${stream.stream_id}', 'stop')" class="w-full flex items-center justify-center gap-1.5 bg-white hover:bg-red-50 text-red-600 border border-gray-200 hover:border-red-200 py-2 rounded-xl text-sm font-bold transition shadow-sm active:scale-95"><span class="material-symbols-outlined text-sm">stop_circle</span>Stop</button>`;
                        } else if (status === 'error' || status === 'stopped') {
                            buttonContainer.innerHTML = `<button onclick="handleRetry('${stream.stream_id}')" class="w-full flex items-center justify-center gap-1.5 bg-orange-500 hover:bg-orange-600 text-white py-2 rounded-xl text-sm font-bold transition shadow-md active:scale-95"><span class="material-symbols-outlined text-sm">refresh</span>Retry</button>`;
                        } else if (isStarting) {
                            buttonContainer.innerHTML = `<button disabled class="w-full flex items-center justify-center gap-1.5 bg-gray-100 text-gray-400 py-2 rounded-xl text-sm font-bold cursor-not-allowed"><span class="material-symbols-outlined text-sm animate-spin">sync</span>Starting...</button>`;
                        } else {
                            buttonContainer.innerHTML = `<button onclick="handleAction('${stream.stream_id}', 'start')" class="w-full flex items-center justify-center gap-1.5 bg-brand text-white hover:bg-brand-hover py-2 rounded-xl text-sm font-bold transition shadow-md shadow-teal-700/10 active:scale-95"><span class="material-symbols-outlined text-sm">play_circle</span>Start</button>`;
                        }
                    }
                }
            }

            // Always update error message if it exists
            const nameEl = card.querySelector('h3');
            const existingErrorEl = card.querySelector('.error-msg');
            if ((isError || errorMessage) && nameEl) {
                const displayMsg = errorMessage || (isError ? 'Unknown Error' : '');
                if (displayMsg) {
                    if (!existingErrorEl) {
                        const errorEl = document.createElement('p');
                        errorEl.className = 'text-xs text-red-600 mt-1 truncate error-msg';
                        errorEl.title = displayMsg;
                        errorEl.textContent = `⚠️ ${displayMsg}`;
                        nameEl.parentElement.appendChild(errorEl);
                    } else {
                        existingErrorEl.textContent = `⚠️ ${displayMsg}`;
                        existingErrorEl.title = displayMsg;
                    }
                } else if (existingErrorEl) {
                    existingErrorEl.remove();
                }
            } else if (existingErrorEl) {
                existingErrorEl.remove();
            }
        }
    });
}

// --- Action Handlers ---
async function handleAction(id, action) {
    try { 
        await api.controlStream(id, action); 
        showToast(`Stream ${action === 'start' ? 'Started' : 'Stopped'}`, 'success'); 
        renderApp();
    } 
    catch(e) { 
        const errorMsg = e.message || 'Action failed';
        showToast(errorMsg, 'error'); 
    }
}

async function handleRetry(id) {
    try {
        showToast('스트림 재시도 중...', 'success');
        await api.controlStream(id, 'restart');
        showToast('스트림 재시도 완료', 'success');
        renderApp();
    } catch(e) {
        const errorMsg = e.message || '재시도 실패';
        showToast(errorMsg, 'error');
    }
}

async function deleteStream(id) { 
    if(confirm('Delete stream?')) { 
        await api.deleteStream(id); 
        showToast('Removed', 'success'); 
        renderApp(); 
    } 
}

// --- Modal Functions ---
const modal = document.getElementById('stream-modal');
const form = document.getElementById('stream-form');

// Default values from config.json
const DEFAULT_VALUES = {
    anonymize_method: 'pixelate',  // FaceBlurModule.options.anonymize_method
    blur_strength: 31,              // FaceBlurModule.options.blur_strength
    confidence: 0.15,               // FaceBlurModule.options.confidence_threshold
    max_fps: 30,                    // global.max_fps (YOLO 성능을 위해 30으로 증가)
    downscale: 1.0,                 // global.downscale (원본 해상도 유지)
    skip_interval: 1                // FaceBlurModule.options.skip_interval (모든 프레임 처리)
};

async function openModal(id = null) {
    modal.classList.remove('hidden');
    if(id) {
        document.getElementById('modal-title').innerText = 'Edit Stream';
        if (!latestDashboardData) {
            showToast('Dashboard data not loaded yet.', 'error');
            return;
        }
        const stream = latestDashboardData.streams.find(s => s.stream_id === id);
        if (stream) {
            const config = stream.config || {};
            document.getElementById('edit-id').value = stream.stream_id;
            document.getElementById('input-name').value = stream.stream_id;
            document.getElementById('input-source').value = (config.rtsp_url || '').replace('rtsp://', '');
            document.getElementById('input-output').value = config.output_url || '';
            document.getElementById('input-max-fps').value = config.max_fps || DEFAULT_VALUES.max_fps;
            document.getElementById('input-downscale').value = config.downscale || DEFAULT_VALUES.downscale;
            // These are module-level settings, so we use defaults or global config if available
            document.getElementById('input-method').value = DEFAULT_VALUES.anonymize_method;
            document.getElementById('input-conf').value = DEFAULT_VALUES.confidence;
            document.getElementById('input-strength').value = DEFAULT_VALUES.blur_strength;
            document.getElementById('strength-val').innerText = DEFAULT_VALUES.blur_strength;
            const skipIntervalInput = document.getElementById('input-skip-interval');
            const skipIntervalVal = document.getElementById('skip-interval-val');
            if (skipIntervalInput) {
                skipIntervalInput.value = DEFAULT_VALUES.skip_interval;
                if (skipIntervalVal) skipIntervalVal.innerText = DEFAULT_VALUES.skip_interval;
            }
        } else {
            showToast(`Stream with ID ${id} not found`, 'error');
            closeModal();
        }
    } else {
        // New stream logic remains the same
        document.getElementById('modal-title').innerText = 'New Stream';
        form.reset();
        document.getElementById('edit-id').value = '';
        // Set default values
        document.getElementById('input-method').value = DEFAULT_VALUES.anonymize_method;
        document.getElementById('input-conf').value = DEFAULT_VALUES.confidence;
        document.getElementById('input-strength').value = DEFAULT_VALUES.blur_strength;
        document.getElementById('strength-val').innerText = DEFAULT_VALUES.blur_strength;
        const maxFpsInput = document.getElementById('input-max-fps');
        const downscaleInput = document.getElementById('input-downscale');
        const skipIntervalInput = document.getElementById('input-skip-interval');
        const skipIntervalVal = document.getElementById('skip-interval-val');
        if (maxFpsInput) maxFpsInput.value = DEFAULT_VALUES.max_fps;
        if (downscaleInput) downscaleInput.value = DEFAULT_VALUES.downscale;
        if (skipIntervalInput) {
            skipIntervalInput.value = DEFAULT_VALUES.skip_interval;
            if (skipIntervalVal) skipIntervalVal.innerText = DEFAULT_VALUES.skip_interval;
        }
    }
}

function closeModal() { 
    modal.classList.add('hidden'); 
}

// --- Form Event Listeners ---
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
        const data = {
            id: document.getElementById('edit-id').value || null,
            name: document.getElementById('input-name').value,
            input_url: 'rtsp://' + document.getElementById('input-source').value,
            output_url: document.getElementById('input-output').value || 'Auto-generated',
            blur_settings: { 
                anonymize_method: document.getElementById('input-method').value, 
                blur_strength: parseInt(document.getElementById('input-strength').value) 
            }
        };
        await api.saveStream(data);
        showToast('Stream saved successfully', 'success');
        closeModal();
        renderApp();
    } catch (error) {
        console.error('Save stream error:', error);
        showToast(`Failed to save: ${error.message}`, 'error');
    }
});

document.getElementById('input-strength').addEventListener('input', (e) => { 
    document.getElementById('strength-val').innerText = e.target.value; 
});

const skipIntervalInput = document.getElementById('input-skip-interval');
if (skipIntervalInput) {
    skipIntervalInput.addEventListener('input', (e) => { 
        document.getElementById('skip-interval-val').innerText = e.target.value; 
    });
}

// --- Chart Modal Functions ---
let chartModal = null;
let fpsChart = null;
let facesChart = null;
let cpuChart = null;
let moduleChart = null;
let currentStreamId = null;
let chartUpdateInterval = null;
let chartData = {
    fps: { labels: [], data: [] },
    faces: { labels: [], data: [] },
    cpu: { labels: [], data: [] },
    module: { labels: [], success: [], errors: [], timeouts: [] }
};
const MAX_CHART_POINTS = 50;

function initCharts() {
    if (typeof Chart === 'undefined') return;
    
    const fpsCtx = document.getElementById('fps-chart');
    const facesCtx = document.getElementById('faces-chart');
    const cpuCtx = document.getElementById('cpu-chart');
    const moduleCtx = document.getElementById('module-chart');
    
    if (fpsCtx) {
        fpsChart = new Chart(fpsCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'FPS',
                    data: [],
                    borderColor: '#0f766e',
                    backgroundColor: 'rgba(15, 118, 110, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 2,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    y: { beginAtZero: true }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
    
    if (facesCtx) {
        facesChart = new Chart(facesCtx, {
      type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Faces',
                    data: [],
                    borderColor: '#14b8a6',
                    backgroundColor: 'rgba(20, 184, 166, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 2,
                    borderWidth: 2
                }]
            },
      options: {
                responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
                    y: { beginAtZero: true }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
    
    if (cpuCtx) {
        cpuChart = new Chart(cpuCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'CPU %',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 2,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    y: { beginAtZero: true, max: 100 }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
    
  if (moduleCtx) {
    moduleChart = new Chart(moduleCtx, {
      type: 'line',
      data: {
                labels: [],
        datasets: [
                    {
                        label: 'Success',
                        data: [],
                        borderColor: '#34d399',
                        backgroundColor: 'rgba(52, 211, 153, 0.1)',
                        tension: 0.4,
                        fill: false,
                        pointRadius: 2,
                        borderWidth: 2
                    },
                    {
                        label: 'Errors',
                        data: [],
                        borderColor: '#f87171',
                        backgroundColor: 'rgba(248, 113, 113, 0.1)',
                        tension: 0.4,
                        fill: false,
                        pointRadius: 2,
                        borderWidth: 2
                    },
                    {
                        label: 'Timeouts',
                        data: [],
                        borderColor: '#f5a524',
                        backgroundColor: 'rgba(245, 165, 36, 0.1)',
                        tension: 0.4,
                        fill: false,
                        pointRadius: 2,
                        borderWidth: 2
                    }
                ]
      },
      options: {
                responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
                    y: { beginAtZero: true }
                },
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
    });
  }
}

function formatTime(tsSec) {
    const d = tsSec ? new Date(tsSec * 1000) : new Date();
    return d.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function updateCharts(streamData, moduleStats) {
    if (!fpsChart || !facesChart || !cpuChart || !moduleChart) return;
    
    const now = formatTime();
    
    // FPS Chart
    chartData.fps.labels.push(now);
    chartData.fps.data.push(streamData.fps || 0);
    if (chartData.fps.labels.length > MAX_CHART_POINTS) {
        chartData.fps.labels.shift();
        chartData.fps.data.shift();
    }
    fpsChart.data.labels = chartData.fps.labels;
    fpsChart.data.datasets[0].data = chartData.fps.data;
    fpsChart.update('none');
    
    // Faces Chart
    chartData.faces.labels.push(now);
    chartData.faces.data.push(streamData.faces_detected || 0);
    if (chartData.faces.labels.length > MAX_CHART_POINTS) {
        chartData.faces.labels.shift();
        chartData.faces.data.shift();
    }
    facesChart.data.labels = chartData.faces.labels;
    facesChart.data.datasets[0].data = chartData.faces.data;
    facesChart.update('none');
    
    // CPU Chart (0-100%로 제한)
    chartData.cpu.labels.push(now);
    const cpuValue = Math.min(Math.max(streamData.cpu_usage || 0, 0), 100);
    chartData.cpu.data.push(cpuValue);
    if (chartData.cpu.labels.length > MAX_CHART_POINTS) {
        chartData.cpu.labels.shift();
        chartData.cpu.data.shift();
    }
    cpuChart.data.labels = chartData.cpu.labels;
    cpuChart.data.datasets[0].data = chartData.cpu.data;
    cpuChart.update('none');
    
    // Module Chart
    if (moduleStats) {
        let success = 0, errors = 0, timeouts = 0;
        Object.values(moduleStats).forEach((stats) => {
            success += stats.total_processed || 0;
            errors += stats.error_count || 0;
            timeouts += stats.timeout_count || 0;
        });
        
        chartData.module.labels.push(now);
        chartData.module.success.push(success);
        chartData.module.errors.push(errors);
        chartData.module.timeouts.push(timeouts);
        if (chartData.module.labels.length > MAX_CHART_POINTS) {
            chartData.module.labels.shift();
            chartData.module.success.shift();
            chartData.module.errors.shift();
            chartData.module.timeouts.shift();
        }
        moduleChart.data.labels = chartData.module.labels;
        moduleChart.data.datasets[0].data = chartData.module.success;
        moduleChart.data.datasets[1].data = chartData.module.errors;
        moduleChart.data.datasets[2].data = chartData.module.timeouts;
        moduleChart.update('none');
    }
}

async function openChartModal(streamId, streamName) {
    currentStreamId = streamId;
    chartModal = document.getElementById('chart-modal');
    document.getElementById('chart-modal-title').textContent = `${streamName} Analytics`;
    document.getElementById('chart-modal-subtitle').textContent = `Stream ID: ${streamId}`;
    
    // 차트 초기화
    if (!fpsChart) {
        initCharts();
    }
    
    // 데이터 초기화
    chartData = {
        fps: { labels: [], data: [] },
        faces: { labels: [], data: [] },
        cpu: { labels: [], data: [] },
        module: { labels: [], success: [], errors: [], timeouts: [] },
        lastData: null // 마지막 데이터 저장 (변경 감지용)
    };
    
    // 모달 표시
    chartModal.classList.remove('hidden');
    
    // 초기 데이터 로드
    const { streams, modules, process: proc } = latestDashboardData;
    const selectedStream = streams.find(s => s.stream_id === streamId);
    if (selectedStream) {
        updateChartData(selectedStream, modules, proc);
    }
    
    // 주기적으로 차트 업데이트 (2초마다)
    if (chartUpdateInterval) {
        clearInterval(chartUpdateInterval);
    }
    chartUpdateInterval = setInterval(() => {
        if (currentStreamId && chartModal && !chartModal.classList.contains('hidden')) {
            const { streams, modules, process: proc } = latestDashboardData;
            const selectedStream = streams.find(s => s.stream_id === currentStreamId);
            if (selectedStream) {
                updateChartData(selectedStream, modules, proc);
            }
        }
    }, 2000);
}

async function updateChartData(stream, modules, proc) {
    if (!stream || !modules || !proc) return;
    
    // 스트림 상태 확인 (ERROR 또는 STOPPED 상태일 때는 차트 업데이트 안 함)
    const streamStatus = (stream.status || '').toLowerCase();
    if (streamStatus === 'error' || streamStatus === 'stopped') {
        return; // ERROR 또는 STOPPED 상태면 차트 업데이트 중단
    }
    
    const facesDetected = modules?.FaceBlurModule?.faces_detected || 0;
    
    // 데이터 준비 (CPU는 0-100%로 제한)
    const newData = {
        fps: stream.fps || 0,
        faces_detected: facesDetected,
        cpu_usage: Math.min(Math.max(proc.cpu_percent || 0, 0), 100)
    };
    
    // 항상 차트 업데이트 (시간이 지나면서 데이터가 추가되어야 함)
    chartData.lastData = newData;
    updateCharts(newData, modules);
}

function closeChartModal() {
    if (chartModal) {
        chartModal.classList.add('hidden');
    }
    currentStreamId = null;
    // 차트 업데이트 인터벌 정리
    if (chartUpdateInterval) {
        clearInterval(chartUpdateInterval);
        chartUpdateInterval = null;
    }
    // 마지막 데이터 초기화
    if (chartData) {
        chartData.lastData = null;
    }
    chartData.lastData = null;
}

// 모달 외부 클릭 시 닫기
document.addEventListener('DOMContentLoaded', () => {
    chartModal = document.getElementById('chart-modal');
    if (chartModal) {
        chartModal.addEventListener('click', (e) => {
            if (e.target === chartModal) {
                closeChartModal();
            }
        });
    }
});

// --- Utility Functions ---
window.copyToClipboard = function(text) { 
    if (!text || text === 'N/A' || text === 'undefined') {
        showToast('No URL to copy', 'error');
        return;
    }
    
    console.log('Attempting to copy:', text);
    
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
            showToast('Copied to clipboard', 'success');
        }).catch(err => {
            console.error('Failed to copy via navigator.clipboard:', err);
            fallbackCopyToClipboard(text);
        });
    } else {
        fallbackCopyToClipboard(text);
    }
};

function fallbackCopyToClipboard(text) {
    try {
        const textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.position = "fixed";
        textArea.style.left = "-9999px";
        textArea.style.top = "0";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        const successful = document.execCommand('copy');
        document.body.removeChild(textArea);
        
        if (successful) {
            showToast('Copied to clipboard', 'success');
        } else {
            showToast('Failed to copy', 'error');
        }
    } catch (err) {
        console.error('Fallback copy failed:', err);
        showToast('Failed to copy', 'error');
    }
}

function showToast(msg, type = 'success') {
    const container = document.getElementById('toast-container');
    const el = document.createElement('div');
    el.className = `${type === 'success' ? 'bg-gray-900 text-white' : 'bg-red-500 text-white'} px-4 py-3 rounded-xl shadow-xl flex items-center gap-3 min-w-[200px] transform translate-y-2 opacity-0 transition-all duration-300`;
    el.innerHTML = `<span class="material-symbols-outlined text-sm font-bold bg-white/20 p-1 rounded-full">${type === 'success' ? 'check' : 'priority_high'}</span><span class="text-sm font-bold">${msg}</span>`;
    container.appendChild(el);
    requestAnimationFrame(() => el.classList.remove('translate-y-2', 'opacity-0'));
    setTimeout(() => { 
        el.classList.add('translate-y-2', 'opacity-0'); 
        setTimeout(() => el.remove(), 300); 
    }, 3000);
}

// --- Alarm System Functions ---
function checkStreamStatusChanges(streams) {
    streams.forEach(stream => {
        const prevState = previousStreamStates[stream.stream_id];
        const currentStatus = stream.status.toLowerCase();
        
        // 상태 변경 감지
        if (prevState && prevState.status !== currentStatus) {
            // ERROR 또는 LIVE(running) 상태로 변경된 경우 알람 생성
            if (currentStatus === 'error' || currentStatus === 'running') {
                const alarmType = currentStatus === 'error' ? 'error' : 'success';
                const message = currentStatus === 'error' 
                    ? `스트림 "${stream.stream_id}"이(가) 오류 상태로 변경되었습니다.`
                    : `스트림 "${stream.stream_id}"이(가) 라이브 상태로 시작되었습니다.`;
                
                addAlarm({
                    streamId: stream.stream_id,
                    streamName: stream.stream_id,
                    type: alarmType,
                    message: message,
                    timestamp: Date.now()
                });
                
                // 토스트 메시지 표시
                showToast(message, alarmType);
            }
        }
        
        // 현재 상태 저장
        previousStreamStates[stream.stream_id] = {
            status: currentStatus,
            name: stream.stream_id
        };
    });
}

function addAlarm(alarm) {
    const alarmId = `alarm_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    alarms.unshift({
        id: alarmId,
        ...alarm,
        read: false
    });
    
    // 최대 100개까지만 유지
    if (alarms.length > 100) {
        alarms = alarms.slice(0, 100);
    }
    
    updateAlarmBadge();
    const alarmModal = document.getElementById('alarm-modal');
    if (alarmModal && !alarmModal.classList.contains('hidden')) {
        renderAlarmList();
    }
}

function updateAlarmBadge() {
    const badge = document.getElementById('alarm-badge');
    const unreadCount = alarms.filter(a => !a.read).length;
    if (badge) {
        if (unreadCount > 0) {
            badge.classList.remove('hidden');
            const badgeText = unreadCount > 99 ? '99+' : unreadCount.toString();
            badge.innerHTML = badgeText;
            badge.className = 'absolute -top-1 -right-1 min-w-[18px] h-[18px] flex items-center justify-center text-[10px] font-bold text-white bg-red-500 rounded-full border-2 border-white px-1';
        } else {
            badge.classList.add('hidden');
        }
    }
}

function renderAlarmList() {
    const alarmList = document.getElementById('alarm-list');
    if (!alarmList) return;
    
    if (alarms.length === 0) {
        alarmList.innerHTML = `
            <div class="p-8 text-center text-gray-400">
                <span class="material-symbols-outlined text-4xl mb-2">notifications_off</span>
                <p class="text-sm font-medium">알람이 없습니다</p>
            </div>
        `;
    return;
  }

    alarmList.innerHTML = alarms.map(alarm => {
        const icon = alarm.type === 'error' ? 'error' : 'check_circle';
        const color = alarm.type === 'error' ? 'text-red-600 bg-red-50 border-red-200' : 'text-green-600 bg-green-50 border-green-200';
        const time = new Date(alarm.timestamp).toLocaleString('ko-KR', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
        
        return `
            <div class="p-4 border-b border-gray-100 hover:bg-gray-50 transition ${alarm.read ? 'opacity-60' : ''}" data-alarm-id="${alarm.id}">
                <div class="flex items-start gap-3">
                    <div class="flex-shrink-0 w-10 h-10 rounded-full ${color} flex items-center justify-center border">
                        <span class="material-symbols-outlined text-lg">${icon}</span>
                    </div>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm font-bold text-gray-900 mb-1">${alarm.streamName}</p>
                        <p class="text-xs text-gray-600 mb-2">${alarm.message}</p>
                        <div class="flex items-center justify-between">
                            <span class="text-[10px] text-gray-400 font-mono">${time}</span>
                            <button onclick="removeAlarm('${alarm.id}')" class="text-gray-400 hover:text-red-600 transition p-1">
                                <span class="material-symbols-outlined text-sm">close</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function toggleAlarmModal() {
    const alarmModal = document.getElementById('alarm-modal');
    if (alarmModal) {
        alarmModal.classList.toggle('hidden');
        if (!alarmModal.classList.contains('hidden')) {
            renderAlarmList();
            // 모든 알람을 읽음 처리
            alarms.forEach(alarm => alarm.read = true);
            updateAlarmBadge();
        }
    }
}

function removeAlarm(alarmId) {
    alarms = alarms.filter(a => a.id !== alarmId);
    updateAlarmBadge();
    renderAlarmList();
}

function clearAllAlarms() {
    if (confirm('모든 알람을 삭제하시겠습니까?')) {
        alarms = [];
        updateAlarmBadge();
        renderAlarmList();
    }
}

// Global functions
window.toggleAlarmModal = toggleAlarmModal;
window.removeAlarm = removeAlarm;
window.clearAllAlarms = clearAllAlarms;

// 모달 외부 클릭 시 닫기
document.addEventListener('DOMContentLoaded', () => {
    const alarmModal = document.getElementById('alarm-modal');
    const alarmButton = document.getElementById('alarm-button');
    if (alarmModal && alarmButton) {
        document.addEventListener('click', (e) => {
            if (!alarmModal.classList.contains('hidden')) {
                if (!alarmModal.contains(e.target) && !alarmButton.contains(e.target)) {
                    alarmModal.classList.add('hidden');
                }
            }
        });
    }
});

// --- Global Functions (for onclick handlers) ---
window.editStream = (id) => openModal(id);

// --- Search Functionality ---
document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const searchTerm = e.target.value.toLowerCase();
            const cards = document.querySelectorAll('#stream-grid > div');
            cards.forEach(card => {
                const name = card.querySelector('h3')?.textContent.toLowerCase() || '';
                const id = card.querySelector('p')?.textContent.toLowerCase() || '';
                if (name.includes(searchTerm) || id.includes(searchTerm)) {
                    card.style.display = '';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    }
});

// --- Initialize App ---
renderApp();
// Update every 2 seconds, but only re-render when data actually changes
setInterval(renderApp, 2000);
