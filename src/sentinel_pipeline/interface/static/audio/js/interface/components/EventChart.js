/**
 * EventChart Component - 노드별 이벤트 차트 시각화
 */
export class EventChart {
    constructor(canvasId, maxDataPoints = 50) {
        this.canvas = document.getElementById(canvasId);
        this.maxDataPoints = maxDataPoints;
        this.chart = null;
        this.eventData = {
            'node-1': [], // HPF
            'node-2': [], // Scream Detection
            'node-3': [], // VAD
            'node-5': [], // STT
        };
        this.selectedNode = null;
        this.init();
    }

    init() {
        if (!this.canvas) return;

        const ctx = this.canvas.getContext('2d');
        
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            font: {
                                size: 10
                            }
                        }
                    },
                    tooltip: {
                        enabled: true
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            font: {
                                size: 9
                            },
                            maxRotation: 45,
                            minRotation: 0
                        },
                        title: {
                            display: true,
                            text: 'Time',
                            font: {
                                size: 10
                            }
                        }
                    },
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        ticks: {
                            font: {
                                size: 9
                            },
                            stepSize: 0.1
                        },
                        title: {
                            display: true,
                            text: 'Confidence / Count',
                            font: {
                                size: 10
                            }
                        }
                    }
                },
                animation: {
                    duration: 300
                }
            }
        });
    }

    addEvent(event, nodeId) {
        if (!nodeId || !this.eventData[nodeId]) return;

        // Event 객체에서 timestamp 추출
        let timestamp;
        if (event.timestamp) {
            timestamp = typeof event.timestamp === 'number' ? event.timestamp : new Date(event.timestamp).getTime();
        } else {
            timestamp = Date.now();
        }
        
        const time = new Date(timestamp).toLocaleTimeString();
        
        // 이벤트 데이터 추가
        this.eventData[nodeId].push({
            time,
            timestamp,
            confidence: event.data?.confidence || 0,
            type: event.type
        });

        // 최대 데이터 포인트 제한
        if (this.eventData[nodeId].length > this.maxDataPoints) {
            this.eventData[nodeId].shift();
        }

        // 선택된 노드가 있으면 차트 업데이트
        if (this.selectedNode === nodeId) {
            this.updateChart(nodeId);
        }
    }

    selectNode(nodeId) {
        this.selectedNode = nodeId;
        this.updateChart(nodeId);
    }

    updateChart(nodeId) {
        if (!this.chart || !nodeId || !this.eventData[nodeId]) return;

        const data = this.eventData[nodeId];
        
        if (data.length === 0) {
            this.chart.data.labels = [];
            this.chart.data.datasets = [];
            this.chart.update();
            return;
        }

        // 노드별 차트 설정
        const nodeConfig = {
            'node-1': {
                label: 'HPF Processing',
                color: 'rgb(59, 130, 246)', // blue
                metric: 'count'
            },
            'node-2': {
                label: 'Scream Detection Confidence',
                color: 'rgb(239, 68, 68)', // red
                metric: 'confidence'
            },
            'node-3': {
                label: 'VAD Processing',
                color: 'rgb(16, 185, 129)', // emerald
                metric: 'count'
            },
            'node-5': {
                label: 'STT Risk Analysis',
                color: 'rgb(139, 92, 246)', // indigo
                metric: 'confidence'
            }
        };

        const config = nodeConfig[nodeId] || {
            label: 'Events',
            color: 'rgb(107, 114, 128)',
            metric: 'count'
        };

        // 데이터 준비
        const labels = data.map(d => d.time);
        const values = data.map(d => {
            if (config.metric === 'confidence') {
                return d.confidence;
            } else {
                // count: 시간대별 이벤트 수
                return 1;
            }
        });

        // 누적 카운트 계산 (count 메트릭인 경우)
        let cumulativeCount = 0;
        const processedValues = config.metric === 'count' 
            ? values.map(v => {
                cumulativeCount += v;
                return cumulativeCount;
            })
            : values;

        this.chart.data.labels = labels;
        this.chart.data.datasets = [{
            label: config.label,
            data: processedValues,
            borderColor: config.color,
            backgroundColor: config.color + '20',
            fill: true,
            tension: 0.4,
            pointRadius: 3,
            pointHoverRadius: 5
        }];

        this.chart.update();
    }

    clear() {
        Object.keys(this.eventData).forEach(key => {
            this.eventData[key] = [];
        });
        if (this.chart) {
            this.chart.data.labels = [];
            this.chart.data.datasets = [];
            this.chart.update();
        }
    }
}

