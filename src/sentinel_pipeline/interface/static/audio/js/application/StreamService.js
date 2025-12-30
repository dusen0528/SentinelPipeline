/**
 * Stream Service - Application Layer
 */
import { Stream } from '../domain/Stream.js';

export class StreamService {
    constructor(apiClient) {
        if (!apiClient) throw new Error("ApiClient is required");
        this.apiClient = apiClient;
        this.streams = new Map();
    }

    async loadStreams() {
        try {
            // Using generic get method (baseUrl already includes /api/audio)
            const data = await this.apiClient.get('/streams');
            this.streams.clear();
            
            // Handle array response
            const streamList = Array.isArray(data) ? data : (data.streams || []);

            streamList.forEach(streamData => {
                const stream = Stream.fromJSON(streamData);
                this.streams.set(stream.streamId, stream);
            });
            
            return Array.from(this.streams.values());
        } catch (error) {
            console.error('Failed to load streams:', error);
            throw error;
        }
    }

    async registerStream(streamId, rtspUrl, useMicrophone = false, micDeviceIndex = null) {
        try {
            const streamData = {
                stream_id: streamId,
                rtsp_url: useMicrophone ? null : rtspUrl,
                use_microphone: useMicrophone,
                mic_device_index: micDeviceIndex,
                sample_rate: 16000,
                scream_threshold: 0.8,
                stt_enabled: true,
                stt_model_size: "base"
            };
            
            // Using generic post method (baseUrl already includes /api/audio)
            const response = await this.apiClient.post('/streams', streamData);
            
            // API 응답을 사용하여 스트림 객체 생성
            const stream = Stream.fromJSON(response);
            this.streams.set(streamId, stream);
            
            return stream;
        } catch (error) {
            console.error('Failed to register stream:', error);
            // 에러 메시지를 더 명확하게 전달
            const errorMessage = error.response?.data?.detail || error.message || 'Unknown error';
            throw new Error(errorMessage);
        }
    }

    async deleteStream(streamId) {
        try {
            // Using generic delete method (baseUrl already includes /api/audio)
            await this.apiClient.delete(`/streams/${streamId}`);
            this.streams.delete(streamId);
            return true;
        } catch (error) {
            console.error('Failed to delete stream:', error);
            throw error;
        }
    }

    getStream(streamId) {
        return this.streams.get(streamId);
    }

    getAllStreams() {
        return Array.from(this.streams.values());
    }

    getStreamCount() {
        return this.streams.size;
    }
}


