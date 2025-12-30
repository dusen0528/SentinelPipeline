export class BaseComponent {
    constructor(elementId) {
        this.element = document.getElementById(elementId);
        if (!this.element) {
            console.warn(`Element with ID '${elementId}' not found.`);
        }
    }

    render(data) {
        throw new Error('Render method must be implemented');
    }

    show() {
        if (this.element) this.element.classList.remove('hidden');
    }

    hide() {
        if (this.element) this.element.classList.add('hidden');
    }
}
