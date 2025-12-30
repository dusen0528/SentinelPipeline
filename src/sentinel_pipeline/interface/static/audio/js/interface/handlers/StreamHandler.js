/**
 * StreamHandler - Event Handlers for Stream Management
 */
export class StreamHandler {
    constructor(streamService, onStreamSelect, onStreamDelete, onStreamRegistered) {
        this.streamService = streamService;
        this.onStreamSelect = onStreamSelect;
        this.onStreamDelete = onStreamDelete;
        this.onStreamRegistered = onStreamRegistered;
    }

    setupForm(formId) {
        const form = document.getElementById(formId);
        if (!form) return;

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.handleSubmit(form);
        });
    }

    async handleSubmit(form) {
        const streamId = form.querySelector('#stream-id').value;
        const rtspUrl = form.querySelector('#rtsp-url').value;
        const useMicrophone = form.querySelector('#use-microphone').checked;

        // 마이크 사용 시 RTSP URL 검증 스킵
        if (!useMicrophone && !rtspUrl) {
            alert('Please enter RTSP URL or select "Use Microphone"');
            return;
        }

        // Stream ID 검증
        if (!streamId || streamId.trim() === '') {
            alert('Please enter a Stream ID');
            return;
        }

        try {
            // 마이크 사용 시 디바이스 인덱스는 null (기본 디바이스 사용)
            await this.streamService.registerStream(streamId, rtspUrl, useMicrophone, null);
            form.reset();
            
            // 리스트 갱신을 위한 콜백 호출
            if (this.onStreamRegistered) {
                await this.onStreamRegistered();
            }
            
            // 등록된 스트림 선택
            if (this.onStreamSelect) {
                const stream = this.streamService.getStream(streamId);
                if (stream) {
                    this.onStreamSelect(stream);
                }
            }
        } catch (error) {
            const errorMsg = error.message || 'Failed to register stream';
            alert(`Failed to register stream: ${errorMsg}`);
            console.error('Stream registration error:', error);
        }
    }

    setupDeleteButton(buttonId) {
        const button = document.getElementById(buttonId);
        if (!button) return;

        button.addEventListener('click', async () => {
            await this.handleDelete();
        });
    }

    async handleDelete() {
        // Get current selected stream from context
        if (this.onStreamDelete) {
            await this.onStreamDelete();
        }
    }
}

