/**
 * Event Service - Application Layer
 */
import { Event } from '../domain/Event.js';

export class EventService {
    constructor(webhookUrl = '') {
        this.webhookUrl = webhookUrl;
        this.eventHistory = [];
        this.maxHistorySize = 100;
    }

    async sendEvent(event) {
        // Add to history
        this.addToHistory(event);

        // Send to external webhook if configured
        if (this.webhookUrl) {
            try {
                await fetch(this.webhookUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(event.toJSON())
                });
            } catch (error) {
                console.error('Failed to send event to webhook:', error);
            }
        }
    }

    addToHistory(event) {
        this.eventHistory.unshift(event);
        if (this.eventHistory.length > this.maxHistorySize) {
            this.eventHistory.pop();
        }
    }

    getEventHistory(limit = 10) {
        return this.eventHistory.slice(0, limit);
    }

    clearHistory() {
        this.eventHistory = [];
    }
}

