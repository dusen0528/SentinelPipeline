// --- Configuration ---
const USE_MOCK_DATA = false; // 실제 API 사용
const API_BASE = '/api';

// --- Mock Data ---
const generateMockStats = () => ({
    total_streams: 12,
    running_streams: 9,
    total_cpu_usage: 42 + Math.random() * 5,
    total_memory_mb: 2048 + Math.random() * 200,
    average_fps: 24.5 + Math.random() * 2,
});

const mockStreams = [
    { id: "cam_01", name: "Main Entrance", input_url: "rtsp://192.168.1.101/main", output_url: "rtsp://srv:8554/cam_01-blur", status: "running", fps: 24.1, faces_detected: 2, cpu_usage: 12.5, blur_settings: { anonymize_method: 'pixelate', blur_strength: 10 } },
    { id: "cam_02", name: "Lobby North", input_url: "rtsp://192.168.1.102/lobby", output_url: "rtsp://srv:8554/cam_02-blur", status: "running", fps: 30.0, faces_detected: 5, cpu_usage: 15.2, blur_settings: { anonymize_method: 'blur', blur_strength: 20 } },
    { id: "cam_03", name: "Parking Lot B", input_url: "rtsp://192.168.1.103/park", output_url: "rtsp://srv:8554/cam_03-blur", status: "stopped", fps: 0, faces_detected: 0, cpu_usage: 0, blur_settings: { anonymize_method: 'solid', blur_strength: 0 } },
    { id: "cam_04", name: "Warehouse Interior", input_url: "rtsp://192.168.1.104/ware", output_url: "rtsp://srv:8554/cam_04-blur", status: "error", fps: 0, faces_detected: 0, cpu_usage: 0, blur_settings: { anonymize_method: 'pixelate', blur_strength: 15 } },
    { id: "cam_05", name: "Elevator Hall", input_url: "rtsp://192.168.1.105/elv", output_url: "rtsp://srv:8554/cam_05-blur", status: "running", fps: 15.0, faces_detected: 1, cpu_usage: 8.2, blur_settings: { anonymize_method: 'mosaic', blur_strength: 12 } },
];

// --- API Functions ---
const api = {
    getStats: async () => {
        if (USE_MOCK_DATA) {
            return generateMockStats();
        }
        try {
            // 스트림 통계와 시스템 통계를 병렬로 가져오기
            const [streamsRes, systemRes] = await Promise.all([
                fetch(`${API_BASE}/streams`),
                fetch(`${API_BASE}/stats/system`)
            ]);
            
            const streamsData = await streamsRes.json();
            const streams = streamsData.data || [];
            const running = streams.filter(s => s.status === 'RUNNING' || s.status === 'running');
            const totalFps = streams.reduce((sum, s) => sum + (s.fps || 0), 0);
            
            let systemStats = {
                system_cpu: 0,
                system_memory_total: 0,
                system_memory_used: 0,
                system_memory_percent: 0,
                process_cpu: 0,
                process_memory: 0
            };
            
            if (systemRes.ok) {
                const systemData = await systemRes.json();
                if (systemData.success && systemData.data) {
                    systemStats = {
                        system_cpu: systemData.data.system.cpu_percent || 0,
                        system_memory_total: systemData.data.system.memory_total_mb || 0,
                        system_memory_used: systemData.data.system.memory_used_mb || 0,
                        system_memory_percent: systemData.data.system.memory_percent || 0,
                        process_cpu: systemData.data.process.cpu_percent || 0,
                        process_memory: systemData.data.process.memory_mb || 0
                    };
                }
            }
            
            return {
                total_streams: streams.length,
                running_streams: running.length,
                total_cpu_usage: systemStats.system_cpu,
                total_memory_mb: systemStats.system_memory_used,
                total_memory_total_mb: systemStats.system_memory_total,
                total_memory_percent: systemStats.system_memory_percent,
                process_cpu_usage: systemStats.process_cpu,
                process_memory_mb: systemStats.process_memory,
                average_fps: streams.length > 0 ? totalFps / streams.length : 0,
            };
        } catch (error) {
            console.error('Failed to fetch stats:', error);
            return {
                total_streams: 0,
                running_streams: 0,
                total_cpu_usage: 0,
                total_memory_mb: 0,
                total_memory_total_mb: 0,
                total_memory_percent: 0,
                process_cpu_usage: 0,
                process_memory_mb: 0,
                average_fps: 0,
            };
        }
    },
    getStreams: async () => {
        if (USE_MOCK_DATA) {
            return mockStreams.map(s => {
                if (s.status === 'running') {
                    s.fps = Math.max(0, 24 + (Math.random() * 4 - 2));
                    s.faces_detected = Math.max(0, s.faces_detected + (Math.random() > 0.8 ? (Math.random() > 0.5 ? 1 : -1) : 0));
                    s.cpu_usage = 10 + Math.random() * 5;
                }
                return s;
            });
        }
        try {
            const res = await fetch(`${API_BASE}/streams`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            const streams = data.data || [];
            
            // 각 스트림의 상세 정보를 병렬로 가져오기
            const streamDetails = await Promise.all(
                streams.map(async (stream) => {
                    try {
                        const detailRes = await fetch(`${API_BASE}/streams/${stream.stream_id}`);
                        if (!detailRes.ok) {
                            console.warn(`Failed to fetch details for ${stream.stream_id}`);
                            return null;
                        }
                        const detailData = await detailRes.json();
                        return detailData.data || null;
                    } catch (error) {
                        console.warn(`Error fetching details for ${stream.stream_id}:`, error);
                        return null;
                    }
                })
            );
            
            // API 응답을 프론트엔드 형식으로 변환
            return streams.map((stream, index) => {
                const detail = streamDetails[index];
                const config = detail?.config || {};
                const status = stream.status.toLowerCase(); // RUNNING -> running
                
                // FaceBlurModule 설정 추출 (기본값 사용)
                const blurSettings = {
                    anonymize_method: 'pixelate',
                    blur_strength: 31
                };
                
                return {
                    id: stream.stream_id,
                    name: stream.stream_id, // stream_id를 이름으로 사용
                    input_url: config.rtsp_url || '',
                    output_url: config.output_url || '',
                    status: status,
                    fps: stream.fps || 0,
                    faces_detected: 0, // 모듈 통계는 별도로 가져와야 함
                    cpu_usage: 0, // 시스템 메트릭은 별도로 가져와야 함
                    blur_settings: blurSettings
                };
            });
        } catch (error) {
            console.error('Failed to fetch streams:', error);
            return [];
        }
    },
    controlStream: async (id, action) => {
        if (USE_MOCK_DATA) {
            const s = mockStreams.find(x => x.id === id);
            if (s) {
                if (action === 'start') s.status = 'starting';
                setTimeout(() => { s.status = action === 'start' ? 'running' : 'stopped'; renderApp(); }, 600);
            }
            return { success: true };
        }
        try {
            if (action === 'start') {
                // 시작 시에는 기존 스트림 설정을 불러와서 payload로 전달해야 422를 피함
                const detailRes = await fetch(`${API_BASE}/streams/${encodeURIComponent(id)}`);
                if (!detailRes.ok) {
                    const errText = await detailRes.text();
                    throw new Error(errText || `Failed to fetch stream config (HTTP ${detailRes.status})`);
                }
                const detail = await detailRes.json();
                const cfg = detail?.data?.config;
                if (!cfg || !cfg.rtsp_url) {
                    throw new Error('Stream config not found; please edit stream to set rtsp_url');
                }
                const payload = {
                    rtsp_url: cfg.rtsp_url,
                    output_url: cfg.output_url || undefined,
                    max_fps: cfg.max_fps,
                    downscale: cfg.downscale,
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
                const res = await fetch(`${API_BASE}/streams/${encodeURIComponent(id)}/${action}`, { 
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                });
                if (!res.ok) {
                    const errorText = await res.text();
                    throw new Error(errorText || `HTTP ${res.status}`);
                }
                return await res.json();
            }
        } catch (error) {
            console.error(`Failed to ${action} stream:`, error);
            throw error;
        }
    },
    deleteStream: async (id) => {
        if (USE_MOCK_DATA) {
            const idx = mockStreams.findIndex(x => x.id === id);
            if (idx > -1) mockStreams.splice(idx, 1);
            return { success: true };
        }
        // 실제 API는 stop을 호출 (DELETE 엔드포인트 없음)
        try {
            const res = await fetch(`${API_BASE}/streams/${encodeURIComponent(id)}/stop`, { 
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            });
            if (!res.ok) {
                const errorText = await res.text();
                throw new Error(errorText || `HTTP ${res.status}`);
            }
            return await res.json();
        } catch (error) {
            console.error('Failed to delete stream:', error);
            throw error;
        }
    },
    saveStream: async (data) => {
        if (USE_MOCK_DATA) {
            if (data.id) {
                const idx = mockStreams.findIndex(x => x.id === data.id);
                if (idx > -1) mockStreams[idx] = { ...mockStreams[idx], ...data };
            } else {
                data.id = `cam_${Date.now()}`;
                data.status = 'stopped';
                data.fps = 0; data.faces_detected = 0; data.cpu_usage = 0;
                mockStreams.push(data);
            }
            return { success: true };
        }
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
        stopped: 'bg-gray-50 text-gray-500 border-gray-200',
        error: 'bg-red-50 text-red-700 border-red-200'
    };
    const icons = { running: 'fiber_manual_record', starting: 'sync', stopped: 'stop', error: 'error' };
    const label = status === 'running' ? 'LIVE' : status.toUpperCase();
    return `
    <div class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold border shadow-sm ${styles[status] || styles.stopped}">
        <span class="material-symbols-outlined text-[12px] ${status === 'running' ? 'animate-pulse text-green-600' : ''} ${status === 'starting' ? 'animate-spin' : ''}" style="font-size: 12px;">${icons[status]}</span>
        ${label}
    </div>`;
}

function renderStreamCard(s) {
    const isRunning = s.status === 'running';
    return `
    <div class="bg-white rounded-2xl shadow-card hover:shadow-lg hover:ring-2 hover:ring-brand-light/20 border border-gray-100 transition-all duration-300 flex flex-col group animate-fade-in" data-stream-id="${s.id}" data-status="${s.status}">
        <div class="p-5 border-b border-gray-50 flex justify-between items-start">
            <div class="flex-1 min-w-0 pr-3">
                <h3 class="font-bold text-gray-900 text-base truncate" title="${s.name}">${s.name}</h3>
                <p class="text-xs text-gray-400 font-mono mt-1 tracking-tight truncate">ID: ${s.id}</p>
            </div>
            <div class="flex-shrink-0" data-status-badge>${getStatusBadge(s.status)}</div>
        </div>
        <div class="p-4 grid grid-cols-3 gap-3 bg-gray-50/30">
            <div class="p-2.5 rounded-xl bg-white border border-gray-100 shadow-sm flex flex-col items-center">
                <span class="text-[9px] text-gray-400 font-bold uppercase tracking-wider mb-1">FPS</span>
                <span class="font-mono text-lg font-bold ${isRunning ? 'text-gray-900' : 'text-gray-300'}" data-metric="fps">${isRunning ? s.fps.toFixed(1) : '-'}</span>
            </div>
            <div class="p-2.5 rounded-xl bg-white border border-gray-100 shadow-sm flex flex-col items-center">
                <span class="text-[9px] text-gray-400 font-bold uppercase tracking-wider mb-1">Faces</span>
                <span class="font-mono text-lg font-bold ${isRunning ? 'text-brand' : 'text-gray-300'}" data-metric="faces">${isRunning ? s.faces_detected : '-'}</span>
            </div>
            <div class="p-2.5 rounded-xl bg-white border border-gray-100 shadow-sm flex flex-col items-center">
                <span class="text-[9px] text-gray-400 font-bold uppercase tracking-wider mb-1">CPU</span>
                <span class="font-mono text-lg font-bold ${isRunning ? 'text-blue-600' : 'text-gray-300'}" data-metric="cpu">${isRunning ? Math.round(s.cpu_usage) + '%' : '-'}</span>
            </div>
        </div>
        <div class="px-5 py-4 space-y-3 flex-1">
            <div class="flex items-center justify-between group/url cursor-pointer hover:bg-gray-50 p-2 -mx-2 rounded-lg transition" onclick="copyToClipboard('${s.input_url}')">
                <div class="flex items-center gap-2"><span class="w-1.5 h-1.5 rounded-full bg-blue-400"></span><span class="text-xs text-gray-500 font-bold">Input</span></div>
                <div class="flex items-center gap-1.5 max-w-[140px]"><span class="text-xs text-gray-400 font-mono truncate">${s.input_url}</span><span class="material-symbols-outlined text-[12px] text-gray-300 group-hover/url:text-brand transition">content_copy</span></div>
            </div>
            <div class="flex items-center justify-between group/url cursor-pointer hover:bg-gray-50 p-2 -mx-2 rounded-lg transition" onclick="copyToClipboard('${s.output_url}')">
                 <div class="flex items-center gap-2"><span class="w-1.5 h-1.5 rounded-full bg-brand-light"></span><span class="text-xs text-gray-500 font-bold">Output</span></div>
                <div class="flex items-center gap-1.5 max-w-[140px]"><span class="text-xs text-gray-400 font-mono truncate">${s.output_url}</span><span class="material-symbols-outlined text-[12px] text-gray-300 group-hover/url:text-brand transition">content_copy</span></div>
            </div>
            <div class="pt-2 flex items-center gap-2">
                <span class="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded-md font-medium border border-gray-200 capitalize flex items-center gap-1"><span class="material-symbols-outlined text-[12px]">enhanced_encryption</span>${s.blur_settings.anonymize_method}</span>
                <span class="text-[10px] text-gray-400 font-bold">STR: ${s.blur_settings.blur_strength}</span>
            </div>
        </div>
        <div class="p-4 border-t border-gray-100 bg-gray-50/50 flex justify-between items-center gap-3 rounded-b-2xl">
            <div class="flex-1">
                ${s.status === 'running' 
                    ? `<button onclick="handleAction('${s.id}', 'stop')" class="w-full flex items-center justify-center gap-1.5 bg-white hover:bg-red-50 text-red-600 border border-gray-200 hover:border-red-200 py-2 rounded-xl text-sm font-bold transition shadow-sm active:scale-95"><span class="material-symbols-outlined text-sm">stop_circle</span>Stop</button>` 
                    : `<button onclick="handleAction('${s.id}', 'start')" class="w-full flex items-center justify-center gap-1.5 bg-brand text-white hover:bg-brand-hover py-2 rounded-xl text-sm font-bold transition shadow-md shadow-teal-700/10 active:scale-95"><span class="material-symbols-outlined text-sm">play_circle</span>Start</button>`
                }
            </div>
            <div class="flex gap-1">
                 <button onclick="editStream('${s.id}')" class="w-9 h-9 flex items-center justify-center text-gray-400 hover:text-brand hover:bg-white border border-transparent hover:border-gray-200 rounded-lg transition shadow-none hover:shadow-sm"><span class="material-symbols-outlined text-sm">settings</span></button>
                 <button onclick="deleteStream('${s.id}')" class="w-9 h-9 flex items-center justify-center text-gray-400 hover:text-red-600 hover:bg-white border border-transparent hover:border-gray-200 rounded-lg transition shadow-none hover:shadow-sm"><span class="material-symbols-outlined text-sm">delete</span></button>
            </div>
        </div>
    </div>`;
}

// No need to store previous data - we always use partial updates

async function renderApp() {
    const stats = await api.getStats();
    document.getElementById('stat-total').innerText = stats.total_streams;
    document.getElementById('stat-running').innerText = stats.running_streams;
    
    // CPU 정보 업데이트
    document.getElementById('stat-cpu').innerText = stats.total_cpu_usage.toFixed(1) + '%';
    document.getElementById('bar-cpu').style.width = Math.min(stats.total_cpu_usage, 100) + '%';
    const processCpuEl = document.getElementById('stat-process-cpu');
    if (processCpuEl) {
        processCpuEl.innerText = (stats.process_cpu_usage || 0).toFixed(1) + '%';
    }
    
    // Memory 정보 업데이트
    const memText = stats.total_memory_total_mb > 0 
        ? `${Math.round(stats.total_memory_mb)} / ${Math.round(stats.total_memory_total_mb)} MB`
        : `${Math.round(stats.total_memory_mb)} MB`;
    document.getElementById('stat-mem').innerText = memText;
    const memPercent = stats.total_memory_percent || (stats.total_memory_total_mb > 0 
        ? (stats.total_memory_mb / stats.total_memory_total_mb) * 100 
        : 0);
    document.getElementById('bar-mem').style.width = Math.min(memPercent, 100) + '%';
    const processMemEl = document.getElementById('stat-process-mem');
    if (processMemEl) {
        processMemEl.innerText = Math.round(stats.process_memory_mb || 0) + ' MB';
    }
    const memPercentEl = document.getElementById('stat-mem-percent');
    if (memPercentEl) {
        memPercentEl.innerText = memPercent.toFixed(1) + '%';
    }
    
    document.getElementById('stat-fps').innerText = stats.average_fps.toFixed(1);

    const streams = await api.getStreams();
    const grid = document.getElementById('stream-grid');
    
    if (streams.length === 0) {
        // Empty state - only update if needed
        if (grid.children.length === 0 || !grid.querySelector('.col-span-full')) {
            grid.innerHTML = `<div class="col-span-full py-16 flex flex-col items-center justify-center text-gray-400 bg-white rounded-2xl border border-dashed border-gray-200 shadow-sm"><div class="p-4 bg-gray-50 rounded-full mb-3"><span class="material-symbols-outlined text-4xl text-gray-300">videocam_off</span></div><p class="font-bold text-gray-600 text-lg">No streams configured</p><button onclick="openModal()" class="mt-6 text-brand hover:text-brand-hover font-bold text-sm bg-brand-bg px-4 py-2 rounded-lg border border-brand/20 transition">Add Stream Now</button></div>`;
        }
    } else {
        // Check if we need to create cards (new streams added or cards don't exist)
        const existingCardIds = Array.from(grid.querySelectorAll('[data-stream-id]')).map(card => card.getAttribute('data-stream-id'));
        const newStreamIds = streams.map(s => s.id);
        const hasNewStreams = newStreamIds.some(id => !existingCardIds.includes(id));
        const hasRemovedStreams = existingCardIds.some(id => !newStreamIds.includes(id));
        
        // Only re-render if structure changed (new/removed streams)
        if (grid.children.length === 0 || hasNewStreams || hasRemovedStreams) {
            grid.innerHTML = streams.map(renderStreamCard).join('');
        } else {
            // Always update existing cards without re-rendering (no flicker)
            updateStreamCards(streams);
        }
    }
}

// Update existing stream cards without full re-render
function updateStreamCards(streams) {
    streams.forEach(stream => {
        const card = document.querySelector(`[data-stream-id="${stream.id}"]`);
        if (card) {
            const isRunning = stream.status === 'running';
            
            // Update FPS
            const fpsEl = card.querySelector('[data-metric="fps"]');
            if (fpsEl) {
                fpsEl.textContent = isRunning ? stream.fps.toFixed(1) : '-';
                fpsEl.className = `font-mono text-lg font-bold ${isRunning ? 'text-gray-900' : 'text-gray-300'}`;
            }
            
            // Update Faces
            const facesEl = card.querySelector('[data-metric="faces"]');
            if (facesEl) {
                facesEl.textContent = isRunning ? stream.faces_detected : '-';
                facesEl.className = `font-mono text-lg font-bold ${isRunning ? 'text-brand' : 'text-gray-300'}`;
            }
            
            // Update CPU
            const cpuEl = card.querySelector('[data-metric="cpu"]');
            if (cpuEl) {
                cpuEl.textContent = isRunning ? Math.round(stream.cpu_usage) + '%' : '-';
                cpuEl.className = `font-mono text-lg font-bold ${isRunning ? 'text-blue-600' : 'text-gray-300'}`;
            }
            
            // Update status badge if status changed
            const statusBadgeContainer = card.querySelector('[data-status-badge]');
            if (statusBadgeContainer) {
                const currentStatus = card.getAttribute('data-status');
                if (currentStatus !== stream.status) {
                    statusBadgeContainer.innerHTML = getStatusBadge(stream.status);
                    card.setAttribute('data-status', stream.status);
                    
                    // Update button
                    const buttonContainer = card.querySelector('.p-4.border-t');
                    if (buttonContainer) {
                        const buttonDiv = buttonContainer.querySelector('.flex-1');
                        if (buttonDiv) {
                            buttonDiv.innerHTML = stream.status === 'running' 
                                ? `<button onclick="handleAction('${stream.id}', 'stop')" class="w-full flex items-center justify-center gap-1.5 bg-white hover:bg-red-50 text-red-600 border border-gray-200 hover:border-red-200 py-2 rounded-xl text-sm font-bold transition shadow-sm active:scale-95"><span class="material-symbols-outlined text-sm">stop_circle</span>Stop</button>` 
                                : `<button onclick="handleAction('${stream.id}', 'start')" class="w-full flex items-center justify-center gap-1.5 bg-brand text-white hover:bg-brand-hover py-2 rounded-xl text-sm font-bold transition shadow-md shadow-teal-700/10 active:scale-95"><span class="material-symbols-outlined text-sm">play_circle</span>Start</button>`;
                        }
                    }
                }
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
        showToast('Action failed', 'error'); 
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
        // 실제 API에서 스트림 정보 가져오기
        if (USE_MOCK_DATA) {
            const s = mockStreams.find(x => x.id === id);
            if(s) {
                document.getElementById('edit-id').value = s.id;
                document.getElementById('input-name').value = s.name;
                document.getElementById('input-source').value = s.input_url.replace('rtsp://', '');
                document.getElementById('input-output').value = s.output_url;
                document.getElementById('input-method').value = s.blur_settings.anonymize_method;
                document.getElementById('input-conf').value = s.blur_settings.confidence || DEFAULT_VALUES.confidence;
                document.getElementById('input-strength').value = s.blur_settings.blur_strength;
                document.getElementById('strength-val').innerText = s.blur_settings.blur_strength;
            }
        } else {
            try {
                const res = await fetch(`${API_BASE}/streams/${encodeURIComponent(id)}`);
                if (res.ok) {
                    const data = await res.json();
                    const stream = data.data;
                    const config = stream.config || {};
                    document.getElementById('edit-id').value = stream.stream_id;
                    document.getElementById('input-name').value = stream.stream_id;
                    document.getElementById('input-source').value = (config.rtsp_url || '').replace('rtsp://', '');
                    document.getElementById('input-output').value = config.output_url || '';
                    document.getElementById('input-max-fps').value = config.max_fps || DEFAULT_VALUES.max_fps;
                    document.getElementById('input-downscale').value = config.downscale || DEFAULT_VALUES.downscale;
                    document.getElementById('input-method').value = DEFAULT_VALUES.anonymize_method;
                    document.getElementById('input-conf').value = DEFAULT_VALUES.confidence;
                    document.getElementById('input-strength').value = DEFAULT_VALUES.blur_strength;
                    document.getElementById('strength-val').innerText = DEFAULT_VALUES.blur_strength;
                    // skip_interval은 모듈 설정에서 가져와야 함 (현재는 기본값 사용)
                    const skipIntervalInput = document.getElementById('input-skip-interval');
                    const skipIntervalVal = document.getElementById('skip-interval-val');
                    if (skipIntervalInput) {
                        skipIntervalInput.value = DEFAULT_VALUES.skip_interval;
                        if (skipIntervalVal) skipIntervalVal.innerText = DEFAULT_VALUES.skip_interval;
                    }
                }
            } catch (error) {
                console.error('Failed to load stream details:', error);
                showToast('Failed to load stream details', 'error');
            }
        }
    } else {
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

// --- Utility Functions ---
function copyToClipboard(text) { 
    navigator.clipboard.writeText(text); 
    showToast('Copied', 'success'); 
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
