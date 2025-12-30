/**
 * Event Domain Entity
 */
export class Event {
    constructor(type, streamId, data = {}, timestamp = null) {
        this.type = type;
        this.streamId = streamId;
        this.data = data;
        this.timestamp = timestamp || Date.now();
    }

    toJSON() {
        return {
            event_type: this.type,
            stream_id: this.streamId,
            data: this.data,
            timestamp: this.timestamp
        };
    }

    static fromJSON(data) {
        return new Event(
            data.event_type,
            data.stream_id,
            data.data,
            data.timestamp
        );
    }
}

