/**
 * Syntax Highlighting (Placeholder)
 *
 * Future implementation for prompt syntax highlighting:
 * - Color-code [scheduling:syntax:when]
 * - Color-code (emphasis:1.2)
 * - Color-code [A|B|C] alternation
 * - Custom textarea with overlay (complex)
 */

import { app } from "../../../scripts/app.js";

// Syntax patterns for highlighting
export const SYNTAX_PATTERNS = {
  // Emphasis: (text:1.2), (text), [text]
  emphasis: /\(([^():]+)(?::(\d+\.?\d*))?\)/g,
  deemphasis: /\[([^\[\]|:]+)\]/g,

  // Scheduling: [from:to:when]
  scheduling: /\[([^:\[\]]+):([^:\[\]]+):(\d+\.?\d*)\]/g,
  addAt: /\[([^:\[\]]+):(\d+\.?\d*)\]/g,
  removeAt: /\[([^:\[\]]+)::(\d+\.?\d*)\]/g,

  // Alternation: [A|B|C]
  alternation: /\[([^\[\]]+\|[^\[\]]+)\]/g,

  // BREAK keyword
  breakKeyword: /\bBREAK\b/g,
};

// Color scheme for syntax highlighting
export const SYNTAX_COLORS = {
  emphasis: "#f39c12", // Orange
  deemphasis: "#9b59b6", // Purple
  scheduling: "#3498db", // Blue
  alternation: "#2ecc71", // Green
  breakKeyword: "#e74c3c", // Red
  number: "#1abc9c", // Teal
};

/**
 * Parse prompt text and identify syntax elements
 * @param {string} text - The prompt text
 * @returns {Array} Array of syntax elements with positions
 */
export function parseSyntaxElements(text) {
  const elements = [];

  // TODO: Implement proper parsing
  // This is complex because we need to handle nesting

  return elements;
}

/**
 * Create highlighted HTML from prompt text
 * @param {string} text - The prompt text
 * @returns {string} HTML with syntax highlighting
 */
export function createHighlightedHtml(text) {
  // TODO: Implement syntax highlighting
  // This requires careful handling of nested syntax
  return escapeHtml(text);
}

/**
 * Escape HTML characters
 * @param {string} str - String to escape
 * @returns {string} Escaped string
 */
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/**
 * Create an overlay layer for syntax highlighting
 * This creates a mirror div behind the textarea that shows colored syntax
 * @param {HTMLTextAreaElement} textarea - The textarea to highlight
 * @returns {Object} Object with overlay element and update function
 */
export function createSyntaxOverlay(textarea) {
  // Create container
  const container = document.createElement("div");
  container.className = "a1111-syntax-overlay-container";
  container.style.cssText = `
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    pointer-events: none;
    overflow: hidden;
  `;

  // Create mirror div
  const mirror = document.createElement("div");
  mirror.className = "a1111-syntax-mirror";
  mirror.style.cssText = `
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    white-space: pre-wrap;
    word-wrap: break-word;
    pointer-events: none;
    color: transparent;
    overflow: hidden;
  `;

  container.appendChild(mirror);

  // Update function
  const update = () => {
    const html = createHighlightedHtml(textarea.value);
    mirror.innerHTML = html;
  };

  // Sync scroll
  const syncScroll = () => {
    mirror.scrollTop = textarea.scrollTop;
    mirror.scrollLeft = textarea.scrollLeft;
  };

  return {
    container,
    mirror,
    update,
    syncScroll,
  };
}

// Extension registration placeholder
// Uncomment when implementing syntax highlighting
/*
app.registerExtension({
  name: "A1111PromptNode.SyntaxHighlight",

  async nodeCreated(node) {
    if (
      node.comfyClass !== "A1111Prompt" &&
      node.comfyClass !== "A1111PromptNegative"
    )
      return;

    const textWidget = node.widgets?.find((w) => w.name === "text");
    if (!textWidget?.inputEl) return;

    const textarea = textWidget.inputEl;
    const { container, update, syncScroll } = createSyntaxOverlay(textarea);
    
    // Insert overlay
    textarea.parentNode.style.position = "relative";
    textarea.parentNode.insertBefore(container, textarea);
    
    // Copy styles from textarea
    const computed = window.getComputedStyle(textarea);
    container.firstChild.style.font = computed.font;
    container.firstChild.style.padding = computed.padding;
    container.firstChild.style.lineHeight = computed.lineHeight;
    
    // Add event listeners
    textarea.addEventListener("input", update);
    textarea.addEventListener("scroll", syncScroll);
    
    // Initial update
    update();
  },
});
*/
