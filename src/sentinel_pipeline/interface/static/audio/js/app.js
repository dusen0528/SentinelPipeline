/**
 * Main Application Entry Point
 */
import { StreamService } from './application/StreamService.js';
import { EventService } from './application/EventService.js';
import { EventMapper } from './application/EventMapper.js';
// Use Shared Infrastructure
import { ApiClient } from '../../common/js/ApiClient.js';
import { WebSocketClient } from '../../common/js/WebSocketClient.js';

import { StreamList } from './interface/components/StreamList.js';
import { PipelineVisualizer } from './interface/components/PipelineVisualizer.js';
import { LogViewer } from './interface/components/LogViewer.js';
import { EventHistory } from './interface/components/EventHistory.js';
import { EventChart } from './interface/components/EventChart.js';
import { WaveformViewer } from './interface/components/WaveformViewer.js';
import { StreamHandler } from './interface/handlers/StreamHandler.js';
import { PipelineStatus } from './domain/PipelineStatus.js';
import { Event } from './domain/Event.js';

import { KpiDashboard } from './interface/components/KpiDashboard.js';

class App {
    constructor() {
        // Initialize Shared ApiClient with base URL for audio API
        this.apiClient = new ApiClient('/api/audio');
        this.streamService = new StreamService(this.apiClient);
        this.eventService = new EventService(''); // EventService usually uses WS or EventBus, assuming empty init is fine for now
        this.wsClient = null;
        
        this.selectedStreamId = null;
        this.pipelineStatus = null;
        
        // Components
        this.streamList = null;
        this.pipelineVisualizer = null;
        this.logViewer = null;
        this.eventHistory = null;
        this.waveformViewer = null;
        this.streamHandler = null;
        this.kpiDashboard = null;
    }

    async init() {
        // Initialize components
        this.streamList = new StreamList('stream-list', 
            (stream) => this.selectStream(stream),
            () => this.deleteStream()
        );
        
        this.pipelineVisualizer = new PipelineVisualizer();
        this.logViewer = new LogViewer('inspector-logs');
        this.eventHistory = new EventHistory('event-history');
        this.eventChart = new EventChart('event-chart');
        this.waveformViewer = new WaveformViewer('waveCanvas', 'db-indicator');
        this.kpiDashboard = new KpiDashboard();
        this.kpiDashboard.init();
        
        // Node click handlers
        this.setupNodeClickHandlers();
        
        this.streamHandler = new StreamHandler(
            this.streamService,
            (stream) => this.selectStream(stream),
            () => this.deleteStream(),
            async () => {
                await this.loadStreams();
                this.updateActiveCount();
            }
        );

        // Setup handlers
        this.streamHandler.setupForm('stream-form');
        this.streamHandler.setupDeleteButton('delete-stream-btn');

        // Load initial data
        await this.loadStreams();
        this.updateActiveCount();

        // Setup WebSocket
        this.setupWebSocket();
    }

    setupNodeClickHandlers() {
        const nodeIds = ['node-1', 'node-2', 'node-3', 'node-5'];
        nodeIds.forEach(nodeId => {
            const node = document.getElementById(nodeId);
            if (node) {
                node.addEventListener('click', () => {
                    nodeIds.forEach(id => {
                        const n = document.getElementById(id);
                        if (n) n.classList.remove('ring-2', 'ring-brand', 'ring-offset-2');
                    });
                    
                    node.classList.add('ring-2', 'ring-brand', 'ring-offset-2');
                    this.eventChart.selectNode(nodeId);
                });
            }
        });
    }

    async loadStreams() {
        try {
            const streams = await this.streamService.loadStreams();
            this.streamList.render(streams);
        } catch (error) {
            console.error('Failed to load streams:', error);
            // Don't crash app on load failure
        }
    }

    selectStream(stream) {
        this.selectedStreamId = stream.streamId;
        this.pipelineStatus = new PipelineStatus(stream.streamId);
        
        this.streamList.selectStream(stream.streamId);
        this.showPipelineView(stream);
        this.pipelineVisualizer.reset();
        this.logViewer.clear();
        this.eventHistory.clear();
        this.eventChart.clear();
        
        ['node-1', 'node-2', 'node-3', 'node-5'].forEach(nodeId => {
            const node = document.getElementById(nodeId);
            if (node) node.classList.remove('ring-2', 'ring-brand', 'ring-offset-2');
        });
    }

    showPipelineView(stream) {
        document.getElementById('empty-state').classList.add('hidden');
        document.getElementById('pipeline-view').classList.remove('hidden');
        
        document.getElementById('stream-name').textContent = stream.streamId;
        document.getElementById('stream-url').textContent = stream.rtspUrl || (stream.useMicrophone ? 'Microphone' : 'N/A');
        
        const statusElement = document.getElementById('stream-status');
        statusElement.textContent = (stream.status || 'inactive').toUpperCase();
        const statusClassMap = {
            'active': 'px-3 py-1 text-xs font-bold rounded-full bg-green-50 text-green-600 border border-green-200',
            'error': 'px-3 py-1 text-xs font-bold rounded-full bg-red-50 text-red-600 border border-red-200',
            'inactive': 'px-3 py-1 text-xs font-bold rounded-full bg-gray-50 text-gray-600 border border-gray-200'
        };
        statusElement.className = statusClassMap[stream.status] || statusClassMap.inactive;
    }

    async deleteStream() {
        if (!this.selectedStreamId) return;
        
        if (!confirm(`Delete stream ${this.selectedStreamId}?`)) return;
        
        try {
            await this.streamService.deleteStream(this.selectedStreamId);
            this.selectedStreamId = null;
            this.pipelineStatus = null;
            
            await this.loadStreams();
            this.updateActiveCount();
            
            document.getElementById('empty-state').classList.remove('hidden');
            document.getElementById('pipeline-view').classList.add('hidden');
        } catch (error) {
            alert('Failed to delete stream: ' + error.message);
        }
    }

    updateActiveCount() {
        const count = this.streamService.getStreamCount();
        const countEl = document.getElementById('active-count');
        if (countEl) countEl.textContent = count;
    }

    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // Adjust WS path as per backend config (usually /ws/admin or /api/ws)
        // Assuming /ws based on original code, but might need adjustment to /api/ws/admin depending on routes
        const wsUrl = `${protocol}//${window.location.host}/api/ws/admin`; 
        
        // Use Shared WebSocketClient
        // Constructor: url, onMessage, onOpen, onClose
        this.wsClient = new WebSocketClient(
            wsUrl,
            (data) => this.handleStreamUpdate(data),
            () => this.logViewer.addLog('System', 'WebSocket connected', 'info'),
            () => this.logViewer.addLog('System', 'WebSocket disconnected', 'error')
        );
        this.wsClient.connect();
    }

    async handleStreamUpdate(data) {
        // KPI Dashboard updates
        if (this.kpiDashboard) this.kpiDashboard.update(data);

        // Stream update (registration/deletion)
        if (data.type === 'stream_update') {
            const wasDeleted = data.action === 'deleted' && data.stream_id === this.selectedStreamId;
            
            await this.loadStreams();
            this.updateActiveCount();
            
            // 삭제된 스트림이 현재 선택된 스트림이면 UI 숨기기
            if (wasDeleted || (data.stream_id === this.selectedStreamId && !this.streamService.getStream(this.selectedStreamId))) {
                this.selectedStreamId = null;
                this.pipelineStatus = null;
                document.getElementById('empty-state').classList.remove('hidden');
                document.getElementById('pipeline-view').classList.add('hidden');
            }
            return;
        }

        // Pipeline status updates
        if (data.type === 'pipeline_status') {
            if (!this.selectedStreamId || data.stream_id !== this.selectedStreamId) return;

            if (data.node_id) {
                const status = data.status === 'alert' ? 'alert' : 'processing';
                this.pipelineVisualizer.setNodeStatus(data.node_id, status);
                
                const nodeId = `node-${data.node_id}`;
                const event = new Event('PIPELINE_STATUS', data.stream_id, {
                    confidence: data.confidence || 0,
                    status: data.status
                });
                this.eventChart.addEvent(event, nodeId);
                
                if (data.node_id === 2) {
                    this.logViewer.addLog(
                        'Scream Detection',
                        `Confidence: ${(data.confidence * 100).toFixed(1)}%`,
                        data.scream_detected ? 'alert' : 'info'
                    );
                }
            }

            if (data.scream_detected) {
                const event = new Event('SCREAM_DETECTED', data.stream_id, {
                    confidence: data.confidence
                });
                this.eventHistory.addEvent(event);
                this.eventChart.addEvent(event, 'node-2');
                this.eventService.sendEvent(event);
            }
            return;
        }

        // Scream detected event
        if (data.type === 'scream_detected') {
            if (!this.selectedStreamId || data.stream_id !== this.selectedStreamId) return;

            this.pipelineVisualizer.setNodeStatus(2, 'alert');
            const event = new Event('SCREAM_DETECTED', data.stream_id, {
                confidence: data.confidence
            });
            this.eventHistory.addEvent(event);
            this.eventChart.addEvent(event, 'node-2');
            this.eventService.sendEvent(event);
            this.logViewer.addLog(
                'Scream Detection',
                `Scream detected! Confidence: ${(data.confidence * 100).toFixed(1)}%`,
                'alert'
            );
            return;
        }

        // STT result
        if (data.type === 'stt_result') {
            if (!this.selectedStreamId || data.stream_id !== this.selectedStreamId) return;

            const sttResult = data.stt_result;
            const riskResult = sttResult.risk_analysis;

            this.pipelineVisualizer.setNodeStatus(4, 'processing');
            this.pipelineVisualizer.setNodeStatus(5, riskResult.is_dangerous ? 'alert' : 'processing');

            if (sttResult.text) {
                this.pipelineVisualizer.showSTTOutput(sttResult.text, riskResult.is_dangerous);
                this.logViewer.addLog(
                    'STT',
                    `Transcription: ${sttResult.text}`,
                    riskResult.is_dangerous ? 'alert' : 'success'
                );
            }

            if (riskResult.is_dangerous) {
                const event = EventMapper.fromRiskAnalysis(
                    riskResult,
                    data.stream_id,
                    sttResult.text
                );
                if (event) {
                    this.eventHistory.addEvent(event);
                    this.eventChart.addEvent(event, 'node-5');
                    this.eventService.sendEvent(event);
                    this.logViewer.addLog(
                        'Risk Analysis',
                        `🚨 Dangerous keyword detected: ${riskResult.keyword}`,
                        'alert'
                    );
                }
            } else {
                const event = new Event('STT_RESULT', data.stream_id, {
                    text: sttResult.text,
                    confidence: 0
                });
                this.eventChart.addEvent(event, 'node-5');
            }
            return;
        }

        // Event detected
        if (data.type === 'event_detected') {
            if (!this.selectedStreamId || data.stream_id !== this.selectedStreamId) return;

            const event = EventMapper.fromRiskAnalysis(
                {
                    is_dangerous: true,
                    event_type: data.event_type,
                    keyword: data.data.keyword,
                    confidence: data.data.confidence,
                    original_text: data.data.text
                },
                data.stream_id,
                data.data.text
            );
            if (event) {
                this.eventHistory.addEvent(event);
                this.eventChart.addEvent(event, 'node-5');
                this.eventService.sendEvent(event);
                this.pipelineVisualizer.setNodeStatus(5, 'alert');
                this.pipelineVisualizer.showSTTOutput(data.data.text, true);
                this.logViewer.addLog(
                    'Risk Analysis',
                    `Dangerous keyword detected: ${data.data.keyword}`,
                    'alert'
                );
            }
            return;
        }

        if (data.type === 'connected') {
            this.logViewer.addLog('System', 'WebSocket connected', 'success');
            if (data.streams) {
                this.loadStreams();
            }
            return;
        }
    }
}

// Initialize app when DOM is ready
window.addEventListener('DOMContentLoaded', () => {
    const app = new App();
    app.init();
});

