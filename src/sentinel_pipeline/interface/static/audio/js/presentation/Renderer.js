/**
 * Renderer - Presentation Layer
 * DOM 조작을 담당하는 유틸리티 클래스
 */
export class Renderer {
    static createElement(tag, className = '', attributes = {}) {
        const element = document.createElement(tag);
        if (className) {
            element.className = className;
        }
        Object.entries(attributes).forEach(([key, value]) => {
            element.setAttribute(key, value);
        });
        return element;
    }

    static createTextNode(text) {
        return document.createTextNode(text);
    }

    static appendChild(parent, child) {
        parent.appendChild(child);
        return parent;
    }

    static removeChildren(element) {
        while (element.firstChild) {
            element.removeChild(element.firstChild);
        }
    }

    static setTextContent(element, text) {
        element.textContent = text;
    }

    static toggleClass(element, className, condition) {
        if (condition) {
            element.classList.add(className);
        } else {
            element.classList.remove(className);
        }
    }

    static setAttribute(element, name, value) {
        element.setAttribute(name, value);
    }

    static removeAttribute(element, name) {
        element.removeAttribute(name);
    }
}

