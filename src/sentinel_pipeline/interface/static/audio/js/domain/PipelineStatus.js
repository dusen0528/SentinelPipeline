/**
 * Pipeline Status Domain Entity
 */
export class PipelineStatus {
    constructor(streamId) {
        this.streamId = streamId;
        this.nodes = {
            1: { status: 'idle', lastUpdate: null },
            2: { status: 'idle', lastUpdate: null },
            3: { status: 'idle', lastUpdate: null },
            5: { status: 'idle', lastUpdate: null }
        };
        this.currentPhase = null;
        this.lastEvent = null;
    }

    updateNode(nodeId, status) {
        if (this.nodes[nodeId]) {
            this.nodes[nodeId].status = status;
            this.nodes[nodeId].lastUpdate = Date.now();
        }
    }

    setPhase(phase) {
        this.currentPhase = phase;
    }

    setLastEvent(event) {
        this.lastEvent = event;
    }

    reset() {
        Object.keys(this.nodes).forEach(nodeId => {
            this.nodes[nodeId].status = 'idle';
            this.nodes[nodeId].lastUpdate = null;
        });
        this.currentPhase = null;
        this.lastEvent = null;
    }
}

