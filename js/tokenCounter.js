/**
 * Token Counter UI for A1111 Prompt Node
 *
 * Displays token counts per 77-token sequence in the node header.
 * Shows BREAK segments distinctly: "6/75 | 1/75" means 2 sequences.
 * Updates in real-time as the user types.
 */

import { app } from "../../../scripts/app.js";

// Color palette for chunks (alternating colors)
export const CHUNK_COLORS = [
  "#e74c3c",
  "#3498db",
  "#2ecc71",
  "#f39c12",
  "#9b59b6",
  "#1abc9c",
  "#e67e22",
  "#34495e",
];

/**
 * Token Tooltip Helpers
 * Creates a floating tooltip showing detailed token information
 */
let tooltipElement = null;
let backdropElement = null;

export function createTooltipElement() {
  if (tooltipElement) return tooltipElement;

  // Create backdrop (invisible overlay to catch outside clicks)
  backdropElement = document.createElement("div");
  backdropElement.className = "a1111-token-tooltip-backdrop";
  backdropElement.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 9999;
    display: none;
  `;
  backdropElement.onclick = hideTokenTooltip;
  document.body.appendChild(backdropElement);

  // Create tooltip
  tooltipElement = document.createElement("div");
  tooltipElement.className = "a1111-token-tooltip";
  tooltipElement.style.cssText = `
    position: fixed;
    z-index: 10000;
    background: #1a1a2e;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 12px;
    font-family: monospace;
    font-size: 12px;
    color: #eee;
    max-width: 500px;
    max-height: 400px;
    overflow-y: auto;
    display: none;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
  `;
  document.body.appendChild(tooltipElement);
  return tooltipElement;
}

export function showTokenTooltip(node, mouseEvent, graphCanvas) {
  const tooltip = createTooltipElement();
  const info = node._tokenInfo;

  if (!info?.tokens || !info?.stats) {
    hideTokenTooltip();
    return;
  }

  // Store reference for copy functionality
  const tokenIds = info.tokens.filter((t) => t.id !== null).map((t) => t.id);

  // Build tooltip content
  let html = `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
      <strong style="font-size: 14px;">Token Info</strong>
      <button id="a1111-tooltip-close" style="
        background: none;
        border: none;
        color: #888;
        font-size: 18px;
        cursor: pointer;
        padding: 0 4px;
      ">&times;</button>
    </div>
    <div style="display: flex; gap: 20px; margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #444;">
      <div><strong>Tokens</strong><br><span style="font-size: 18px; color: #3498db;">${info.stats.total_tokens}</span></div>
      <div><strong>Characters</strong><br><span style="font-size: 18px; color: #2ecc71;">${info.stats.characters}</span></div>
      <div><strong>Words</strong><br><span style="font-size: 18px; color: #f39c12;">${info.stats.words}</span></div>
      <div><strong>Chunks</strong><br><span style="font-size: 18px; color: #9b59b6;">${info.stats.chunks}</span></div>
    </div>
  `;

  // Colored tokens section
  html += `<div style="margin-bottom: 8px;"><strong>Tokens</strong></div>`;
  html += `<div style="max-height: 150px; overflow-y: auto; margin-bottom: 10px; padding: 8px; background: #0d0d1a; border-radius: 4px; line-height: 1.8;">`;

  for (const token of info.tokens) {
    if (token.is_break) {
      html += `<span style="
        display: inline-block;
        padding: 2px 6px;
        margin: 2px;
        background: #3498db;
        color: white;
        border-radius: 3px;
        font-weight: bold;
      ">BREAK</span>`;
    } else {
      const color = CHUNK_COLORS[token.chunk % CHUNK_COLORS.length];
      // Handle </w> end-of-word suffix
      let displayText = token.text;
      let endOfWordHtml = "";
      if (displayText.endsWith("</w>")) {
        displayText = displayText.slice(0, -4);
        endOfWordHtml = `<span style="opacity: 0.4; font-size: 10px;">&lt;/w&gt;</span>`;
      }
      const escapedText = escapeHtml(displayText);
      html += `<span style="
        display: inline-block;
        padding: 2px 4px;
        margin: 1px;
        background: ${color}33;
        border: 1px solid ${color};
        border-radius: 3px;
        color: ${color};
      ">${escapedText}${endOfWordHtml}</span>`;
    }
  }
  html += `</div>`;

  // Token IDs section with copy button
  html += `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
      <strong>Input IDs</strong>
      <button id="a1111-tooltip-copy" style="
        background: #333;
        border: 1px solid #555;
        color: #aaa;
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 4px;
        cursor: pointer;
      ">Copy IDs</button>
    </div>
  `;
  html += `<div id="a1111-token-ids" style="max-height: 80px; overflow-y: auto; padding: 8px; background: #0d0d1a; border-radius: 4px; font-size: 11px; color: #888; word-break: break-all;">`;
  html += tokenIds.join(", ");
  html += `</div>`;

  tooltip.innerHTML = html;
  tooltip.style.display = "block";

  // Position tooltip near click but keep on screen
  const padding = 15;
  let x = mouseEvent.clientX + padding;
  let y = mouseEvent.clientY + padding;

  // Adjust if would go off screen
  const rect = tooltip.getBoundingClientRect();
  if (x + rect.width > window.innerWidth) {
    x = mouseEvent.clientX - rect.width - padding;
  }
  if (y + rect.height > window.innerHeight) {
    y = mouseEvent.clientY - rect.height - padding;
  }

  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;

  // Add event listeners for buttons
  document.getElementById("a1111-tooltip-close").onclick = hideTokenTooltip;
  document.getElementById("a1111-tooltip-copy").onclick = () => {
    navigator.clipboard.writeText(tokenIds.join(", ")).then(() => {
      const btn = document.getElementById("a1111-tooltip-copy");
      btn.textContent = "Copied!";
      btn.style.color = "#2ecc71";
      setTimeout(() => {
        btn.textContent = "Copy IDs";
        btn.style.color = "#aaa";
      }, 1500);
    });
  };

  // Show backdrop
  if (backdropElement) {
    backdropElement.style.display = "block";
  }
}

export function hideTokenTooltip() {
  if (tooltipElement) {
    tooltipElement.style.display = "none";
  }
  if (backdropElement) {
    backdropElement.style.display = "none";
  }
}

// Helper to escape HTML
export function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

export function getTooltipElement() {
  return tooltipElement;
}
