/**
 * Stream Domain Entity
 */
export class Stream {
    constructor(streamId, rtspUrl, credentials = null, status = 'active', useMicrophone = false) {
        this.streamId = streamId;
        this.rtspUrl = rtspUrl;
        this.credentials = credentials;
        this.status = status;
        this.useMicrophone = useMicrophone;
        this.createdAt = new Date();
        this.lastUpdate = new Date();
    }

    updateStatus(status) {
        this.status = status;
        this.lastUpdate = new Date();
    }

    toJSON() {
        return {
            stream_id: this.streamId,
            rtsp_url: this.rtspUrl,
            credentials: this.credentials,
            status: this.status,
            created_at: this.createdAt.toISOString(),
            last_update: this.lastUpdate.toISOString()
        };
    }

    static fromJSON(data) {
        const stream = new Stream(
            data.stream_id,
            data.rtsp_url,
            data.credentials,
            data.status || 'active',
            data.use_microphone || false
        );
        if (data.created_at) {
            stream.createdAt = new Date(data.created_at);
        }
        if (data.last_update) {
            stream.lastUpdate = new Date(data.last_update);
        }
        return stream;
    }
}

