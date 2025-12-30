export class WebSocketClient {
    constructor(url, onMessage, onOpen, onClose) {
        this.url = url;
        this.onMessage = onMessage;
        this.onOpen = onOpen;
        this.onClose = onClose;
        this.socket = null;
        this.reconnectInterval = 3000;
        this.shouldReconnect = true;
    }

    connect() {
        console.log(`Connecting to WebSocket: ${this.url}`);
        this.socket = new WebSocket(this.url);

        this.socket.onopen = () => {
            console.log('WebSocket Connected');
            if (this.onOpen) this.onOpen();
        };

        this.socket.onmessage = async (event) => {
            try {
                const data = JSON.parse(event.data);
                if (this.onMessage) {
                    const result = this.onMessage(data);
                    // async 함수인 경우 await
                    if (result instanceof Promise) {
                        await result;
                    }
                }
            } catch (e) {
                console.error('WebSocket message parse error:', e);
            }
        };

        this.socket.onclose = () => {
            console.log('WebSocket Disconnected');
            if (this.onClose) this.onClose();
            if (this.shouldReconnect) {
                setTimeout(() => this.connect(), this.reconnectInterval);
            }
        };

        this.socket.onerror = (error) => {
            console.error('WebSocket Error:', error);
            this.socket.close();
        };
    }

    send(data) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(data));
        } else {
            console.warn('WebSocket is not open. Cannot send message.');
        }
    }

    disconnect() {
        this.shouldReconnect = false;
        if (this.socket) {
            this.socket.close();
        }
    }
}
