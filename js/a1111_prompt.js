/**
 * A1111 Prompt Node Frontend Extension
 *
 * Main entry point for all A1111 Prompt Node UI functionality.
 * Imports and uses modules from:
 * - tokenCounter.js - Token counting and tooltip display
 * - autocomplete.js - Tag autocomplete (placeholder)
 * - syntaxHighlight.js - Syntax highlighting (placeholder)
 */

import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Import from tokenCounter module
import {
  showTokenTooltip,
  hideTokenTooltip,
  escapeHtml,
  getTooltipElement,
} from "./tokenCounter.js";

/**
 * Show Text Extension
 *
 * Displays the incoming prompt text (e.g., from TIPO) in a readonly widget
 * after execution, so users can see what prompt was actually encoded.
 */
app.registerExtension({
  name: "A1111PromptNode.ShowText",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "A1111Prompt") {
      /**
       * Populate the node with a readonly text widget showing the prompt
       */
      function populate(text) {
        // Find or create the display widget
        let displayWidget = this.widgets?.find(
          (w) => w.name === "_prompt_display"
        );

        // Get the actual text value (could be an array)
        const textValue = Array.isArray(text) ? text[0] : text;

        if (!textValue) return;

        if (!displayWidget) {
          // Create a new readonly widget to show the prompt
          const widgetResult = ComfyWidgets["STRING"](
            this,
            "_prompt_display",
            ["STRING", { multiline: true }],
            app
          );
          displayWidget = widgetResult.widget;
          displayWidget.inputEl.readOnly = true;
          displayWidget.inputEl.style.opacity = "0.7";
          displayWidget.inputEl.style.fontStyle = "italic";
          displayWidget.inputEl.placeholder =
            "(Prompt will appear after execution)";
          // Mark it as a display-only widget
          displayWidget.serialize = false;
        }

        displayWidget.value = textValue;

        // Resize node to fit the new widget content
        requestAnimationFrame(() => {
          const sz = this.computeSize();
          if (sz[0] < this.size[0]) sz[0] = this.size[0];
          if (sz[1] < this.size[1]) sz[1] = this.size[1];
          this.onResize?.(sz);
          app.graph.setDirtyCanvas(true, false);
        });
      }

      // Hook into onExecuted to display the prompt after execution
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        if (message?.text) {
          populate.call(this, message.text);
        }
      };

      // Store widget values during configure for workflow reload
      const VALUES = Symbol();
      const configure = nodeType.prototype.configure;
      nodeType.prototype.configure = function () {
        this[VALUES] = arguments[0]?.widgets_values;
        return configure?.apply(this, arguments);
      };

      // Restore display widget on workflow load
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        onConfigure?.apply(this, arguments);
        const widgets_values = this[VALUES];
        // Look for stored prompt display value
        if (widgets_values?.length > 1) {
          requestAnimationFrame(() => {
            // The display widget value might be stored after the main text widget
            const displayValue = widgets_values.find(
              (v) =>
                typeof v === "string" &&
                v.length > 0 &&
                this.widgets?.find((w) => w.name === "text")?.value !== v
            );
            if (displayValue) {
              populate.call(this, [displayValue]);
            }
          });
        }
      };
    }
  },
});

/**
 * Token Counter Extension
 *
 * Displays token counts per 77-token sequence in the node header.
 * Shows BREAK segments distinctly: "6/75 | 1/75" means 2 sequences.
 * Updates in real-time as the user types.
 */
app.registerExtension({
  name: "A1111PromptNode.TokenCounter",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (
      nodeData.name !== "A1111Prompt" &&
      nodeData.name !== "A1111PromptNegative"
    )
      return;

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      if (onDrawForeground) onDrawForeground.apply(this, arguments);

      // Only show if sequences are available (not null/unavailable)
      if (
        this._tokenInfo?.sequences &&
        Array.isArray(this._tokenInfo.sequences)
      ) {
        const seqs = this._tokenInfo.sequences;

        // Calculate total usable tokens (sum of per-sequence counts)
        const totalUsable = seqs.reduce((sum, s) => sum + s, 0);

        // Build display formats from longest to shortest
        const fullText =
          seqs.map((s) => `${s}/75`).join(" | ") + ` (${totalUsable})`;
        const shortText = seqs.join("|") + ` (${totalUsable})`;
        const tinyText = `${totalUsable} tok`;

        // Calculate total tokens for color (each sequence is 77 with start/end)
        const totalTokens = seqs.length * 77;

        // Determine color based on total token count
        // Yellow: 300+ tokens (4+ chunks) - getting long
        // Red: 450+ tokens (6+ chunks) - very long, may impact quality/memory
        let color = "#888"; // Default gray
        if (totalTokens > 450) {
          color = "#e74c3c"; // Red
        } else if (totalTokens > 300) {
          color = "#f39c12"; // Yellow/Orange
        }

        ctx.save();
        ctx.font = "11px monospace";
        ctx.fillStyle = color;
        ctx.textAlign = "right";

        // Available width (leave margin for title on left)
        const availableWidth = this.size[0] - 80;

        // Pick the longest format that fits
        let displayText = tinyText;
        if (ctx.measureText(fullText).width <= availableWidth) {
          displayText = fullText;
        } else if (ctx.measureText(shortText).width <= availableWidth) {
          displayText = shortText;
        }

        // Store bounds for click detection
        const textWidth = ctx.measureText(displayText).width;
        this._tokenCounterBounds = {
          x: this.size[0] - 10 - textWidth,
          y: -20,
          width: textWidth + 10,
          height: 20,
        };

        // Draw background highlight if hovered (indicates clickable)
        if (this._tokenCounterHovered) {
          ctx.fillStyle = "#333143"; // Subtle cyan/blue
          ctx.beginPath();
          ctx.roundRect(
            this.size[0] - 14 - textWidth,
            -16,
            textWidth + 8,
            14,
            4
          );
          ctx.fill();
          ctx.fillStyle = color; // Restore text color
        }

        ctx.fillText(displayText, this.size[0] - 10, -6);

        ctx.restore();
      }
    };

    // Track hover state for visual feedback
    const onMouseMove = nodeType.prototype.onMouseMove;
    nodeType.prototype.onMouseMove = function (e, localPos, graphCanvas) {
      if (onMouseMove) onMouseMove.apply(this, arguments);

      const bounds = this._tokenCounterBounds;
      if (bounds && this._tokenInfo?.tokens) {
        const isOver =
          localPos[0] >= bounds.x &&
          localPos[0] <= bounds.x + bounds.width &&
          localPos[1] >= bounds.y &&
          localPos[1] <= bounds.y + bounds.height;

        if (isOver !== this._tokenCounterHovered) {
          this._tokenCounterHovered = isOver;
          this.setDirtyCanvas(true, false);

          // Change cursor
          if (graphCanvas?.canvas) {
            graphCanvas.canvas.style.cursor = isOver ? "pointer" : "";
          }
        }
      }
    };

    // Click handler for tooltip (replaces hover)
    const onMouseDown = nodeType.prototype.onMouseDown;
    nodeType.prototype.onMouseDown = function (e, localPos, graphCanvas) {
      const bounds = this._tokenCounterBounds;

      // Check if click is on token counter area
      if (bounds && this._tokenInfo?.tokens) {
        const isOver =
          localPos[0] >= bounds.x &&
          localPos[0] <= bounds.x + bounds.width &&
          localPos[1] >= bounds.y &&
          localPos[1] <= bounds.y + bounds.height;

        if (isOver) {
          // Clear hover state when clicking
          this._tokenCounterHovered = false;
          this.setDirtyCanvas(true, false);

          // Toggle tooltip
          const tooltip = getTooltipElement();
          if (tooltip?.style.display === "block") {
            hideTokenTooltip();
          } else {
            showTokenTooltip(this, e, graphCanvas);
          }
          return true; // Consume the click
        }
      }

      // Call original handler
      if (onMouseDown) return onMouseDown.apply(this, arguments);
    };
  },

  async nodeCreated(node) {
    if (
      node.comfyClass !== "A1111Prompt" &&
      node.comfyClass !== "A1111PromptNegative"
    )
      return;

    const textWidget = node.widgets?.find((w) => w.name === "text");
    if (!textWidget) return;

    node._tokenInfo = { sequences: [0], boundaries: [], estimated: true };

    // Create boundary marker overlay
    let overlayContainer = null;
    let mirrorDiv = null;

    const createOverlay = () => {
      if (!textWidget.inputEl) return;

      const textarea = textWidget.inputEl;

      // Create container for the overlay
      overlayContainer = document.createElement("div");
      overlayContainer.className = "a1111-boundary-overlay-container";
      overlayContainer.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        pointer-events: none;
        overflow: hidden;
      `;

      // Create mirror div that replicates textarea styling
      mirrorDiv = document.createElement("div");
      mirrorDiv.className = "a1111-boundary-mirror";
      mirrorDiv.style.cssText = `
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

      overlayContainer.appendChild(mirrorDiv);

      // Insert overlay before textarea so it appears behind
      textarea.parentNode.style.position = "relative";
      textarea.parentNode.insertBefore(overlayContainer, textarea);

      // Copy textarea styles to mirror
      const copyStyles = () => {
        const computed = window.getComputedStyle(textarea);
        mirrorDiv.style.font = computed.font;
        mirrorDiv.style.padding = computed.padding;
        mirrorDiv.style.border = computed.border;
        mirrorDiv.style.borderColor = "transparent";
        mirrorDiv.style.lineHeight = computed.lineHeight;
        mirrorDiv.style.letterSpacing = computed.letterSpacing;
        mirrorDiv.style.width = computed.width;
      };

      copyStyles();

      // Sync scroll position
      textarea.addEventListener("scroll", () => {
        mirrorDiv.scrollTop = textarea.scrollTop;
        mirrorDiv.scrollLeft = textarea.scrollLeft;
      });
    };

    const updateBoundaryMarkers = () => {
      if (!mirrorDiv || !textWidget.inputEl) return;

      const text = textWidget.inputEl.value || "";
      const boundaries = node._tokenInfo?.boundaries || [];

      if (boundaries.length === 0) {
        mirrorDiv.textContent = text;
        return;
      }

      // Build HTML with boundary markers
      let html = "";
      let lastPos = 0;

      // Sort boundaries by position
      const sortedBoundaries = [...boundaries].sort(
        (a, b) => a.char_pos - b.char_pos
      );

      for (const boundary of sortedBoundaries) {
        const pos = Math.min(boundary.char_pos, text.length);
        if (pos <= lastPos) continue;

        // Add text before boundary
        html += escapeHtml(text.slice(lastPos, pos));

        // Add boundary marker
        const markerColor =
          boundary.type === "break"
            ? "#3498db" // Blue for BREAK
            : "#e67e22"; // Orange for chunk boundary

        html += `<span style="
          display: inline-block;
          width: 2px;
          height: 1.2em;
          background: ${markerColor};
          vertical-align: middle;
          margin: 0 1px;
          border-radius: 1px;
          opacity: 0.7;
        "></span>`;

        lastPos = pos;
      }

      // Add remaining text
      html += escapeHtml(text.slice(lastPos));

      mirrorDiv.innerHTML = html;
    };

    let timeout;
    const updateTokenCount = async (text) => {
      clearTimeout(timeout);
      timeout = setTimeout(async () => {
        try {
          const response = await fetch("/a1111_prompt/tokenize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
          });
          node._tokenInfo = await response.json();
          node.setDirtyCanvas(true, false);

          // Update boundary markers
          requestAnimationFrame(updateBoundaryMarkers);
        } catch (e) {
          // Silently ignore errors, just keep old count
        }
      }, 300); // 300ms debounce
    };

    // Hook into widget callback for text changes
    const origCallback = textWidget.callback;
    textWidget.callback = function (value) {
      origCallback?.apply(this, arguments);
      updateTokenCount(value);
    };

    // Create overlay when the textarea is available
    const waitForTextarea = () => {
      if (textWidget.inputEl) {
        createOverlay();
        updateTokenCount(textWidget.value || "");
      } else {
        requestAnimationFrame(waitForTextarea);
      }
    };
    requestAnimationFrame(waitForTextarea);
  },
});
