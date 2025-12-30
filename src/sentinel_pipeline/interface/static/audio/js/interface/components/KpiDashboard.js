export class KpiDashboard {
    constructor() {
        this.kpiCardContainer = document.getElementById('kpi-cards');
        this.stats = {
            totalEvents: 0,
            screamEvents: 0,
            dangerKeywordEvents: 0,
            totalChunks: 0, // 전체 처리 청크 수
            screamChunks: 0, // 비명 포함 청크 수
            eventsByType: {},
            // Latency 추적
            latencies: [], // 처리 지연 시간 배열 (ms)
            lastEventTimestamp: null, // 마지막 이벤트 타임스탬프
            // Drop Rate 추적
            expectedChunks: 0, // 예상 청크 수 (시간 기반)
            processedChunks: 0, // 실제 처리된 청크 수
            startTime: null // 추적 시작 시간
        };
        this.kpis = {
            screamRate: { name: 'Scream Rate', value: '0.00%', unit: '' },
            avgLatency: { name: 'Avg Latency', value: 'N/A', unit: 'ms' },
            dropRate: { name: 'Drop Rate', value: 'N/A', unit: '%' },
            totalEvents: { name: 'Total Events', value: 0, unit: '' },
        };
    }

    init() {
        this.render();
    }

    update(event) {
        console.log('[KPI Dashboard] Received event:', event); // For debugging
        if (!event || !event.type) return;

        const now = Date.now();
        
        // 시작 시간 설정 (첫 이벤트)
        if (this.stats.startTime === null) {
            this.stats.startTime = now;
        }

        // "Scream Rate" 계산을 위한 로직
        // pipeline_status는 모든 청크 처리 시 발생하므로 이를 기준으로 계산
        if (event.type === 'pipeline_status') {
            this.stats.totalChunks++;
            this.stats.processedChunks++;
            
            // scream_detected 속성 확인 (boolean 또는 truthy 값)
            if (event.scream_detected === true || event.scream_detected === 1) {
                this.stats.screamChunks++;
            }
            
            // Latency 계산: 이벤트 timestamp와 현재 시간의 차이
            if (event.timestamp) {
                const latency = now - event.timestamp;
                if (latency > 0 && latency < 10000) { // 0~10초 범위만 유효한 지연으로 간주
                    this.stats.latencies.push(latency);
                    // 최근 100개만 유지 (메모리 관리)
                    if (this.stats.latencies.length > 100) {
                        this.stats.latencies.shift();
                    }
                }
            }
        }

        // "Total Events" 계산을 위한 로직 (실제 감지된 이벤트만 카운트)
        if (event.type === 'scream_detected') {
            this.stats.totalEvents++;
            this.stats.screamEvents++;
            
            // Latency 계산
            if (event.timestamp) {
                const latency = now - event.timestamp;
                if (latency > 0 && latency < 10000) {
                    this.stats.latencies.push(latency);
                    if (this.stats.latencies.length > 100) {
                        this.stats.latencies.shift();
                    }
                }
            }
        } else if (event.type === 'event_detected') {
            this.stats.totalEvents++;
            if (event.data && event.data.risk_type) {
                this.stats.dangerKeywordEvents++;
                const riskType = event.data.risk_type;
                this.stats.eventsByType[riskType] = (this.stats.eventsByType[riskType] || 0) + 1;
            }
            
            // Latency 계산
            if (event.timestamp) {
                const latency = now - event.timestamp;
                if (latency > 0 && latency < 10000) {
                    this.stats.latencies.push(latency);
                    if (this.stats.latencies.length > 100) {
                        this.stats.latencies.shift();
                    }
                }
            }
        }

        // 예상 청크 수 계산 (시간 기반: 약 0.5초마다 청크가 들어온다고 가정)
        if (this.stats.startTime) {
            const elapsedSeconds = (now - this.stats.startTime) / 1000;
            // 0.5초마다 청크가 들어온다고 가정 (2초 청크, 0.5초 hop)
            this.stats.expectedChunks = Math.floor(elapsedSeconds / 0.5);
        }

        this.calculateKpis();
        this.render();
    }

    calculateKpis() {
        // Scream Rate: (비명 청크 / 전체 청크) * 100
        const screamRate = this.stats.totalChunks > 0 ? (this.stats.screamChunks / this.stats.totalChunks) * 100 : 0;
        this.kpis.screamRate.value = screamRate.toFixed(2) + '%';
        
        // Avg Latency: 평균 지연 시간 계산
        if (this.stats.latencies.length > 0) {
            const avgLatency = this.stats.latencies.reduce((sum, lat) => sum + lat, 0) / this.stats.latencies.length;
            this.kpis.avgLatency.value = Math.round(avgLatency);
        } else {
            this.kpis.avgLatency.value = 'N/A';
        }
        
        // Drop Rate: (예상 청크 - 처리된 청크) / 예상 청크 * 100
        if (this.stats.expectedChunks > 0) {
            const droppedChunks = Math.max(0, this.stats.expectedChunks - this.stats.processedChunks);
            const dropRate = (droppedChunks / this.stats.expectedChunks) * 100;
            this.kpis.dropRate.value = dropRate.toFixed(2) + '%';
        } else {
            this.kpis.dropRate.value = '0.00%';
        }
        
        // Total Events: 실제 감지된 주요 이벤트 수
        this.kpis.totalEvents.value = this.stats.totalEvents;
    }

    render() {
        if (!this.kpiCardContainer) return;
        
        this.kpiCardContainer.innerHTML = ''; // Clear existing cards
        
        Object.values(this.kpis).forEach(kpi => {
            const card = this.createKpiCard(kpi);
            this.kpiCardContainer.appendChild(card);
        });
    }

    createKpiCard(kpi) {
        const card = document.createElement('div');
        card.className = 'bg-white p-4 rounded-lg shadow-sm border border-gray-200';
        
        const nameEl = document.createElement('h3');
        nameEl.className = 'text-xs text-gray-500 font-bold uppercase tracking-wide';
        nameEl.textContent = kpi.name;
        
        const valueEl = document.createElement('p');
        valueEl.className = 'text-2xl font-bold text-gray-900 mt-1';
        valueEl.textContent = kpi.value;
        
        if (kpi.unit) {
            const unitEl = document.createElement('span');
            unitEl.className = 'text-sm font-medium text-gray-400 ml-1';
            unitEl.textContent = kpi.unit;
            valueEl.appendChild(unitEl);
        }

        card.appendChild(nameEl);
        card.appendChild(valueEl);
        
        return card;
    }
}
