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
    if (nodeData.name === "A1111Prompt" || nodeData.name === "A1111PromptNegative") {
      function resizeNode(node) {
        requestAnimationFrame(() => {
          const sz = node.computeSize();
          if (sz[0] < node.size[0]) sz[0] = node.size[0];
          node.onResize?.(sz);
          app.graph.setDirtyCanvas(true, false);
        });
      }

      function getCurrentText(node) {
        const textWidget = node.widgets?.find((w) => w.name === "text");
        return textWidget?.inputEl?.value ?? textWidget?.value ?? "";
      }

      const PREVIEW_PROPERTY = "a1111_resolved_prompt";
      const PREVIEW_EXPANDED_PROPERTY = "a1111_resolved_prompt_expanded";
      const PREVIEW_SOURCE_PROPERTY = "a1111_resolved_prompt_source";
      const PREVIEW_WIDGET_HEIGHT = 120;
      const HIDDEN_WIDGET_HEIGHT = -4;

      function getPreviewText(node) {
        return node.properties?.[PREVIEW_PROPERTY] || "";
      }

      function getPreviewSource(node) {
        return node.properties?.[PREVIEW_SOURCE_PROPERTY] || "";
      }

      function isPreviewExpanded(node) {
        return Boolean(node.properties?.[PREVIEW_EXPANDED_PROPERTY]);
      }

      function setPreviewExpanded(node, expanded) {
        node.properties ??= {};
        node.properties[PREVIEW_EXPANDED_PROPERTY] = Boolean(expanded);
      }

      function setPreviewText(node, value, sourceText = "") {
        node.properties ??= {};
        if (value) {
          node.properties[PREVIEW_PROPERTY] = value;
          node.properties[PREVIEW_SOURCE_PROPERTY] = sourceText;
        } else {
          delete node.properties[PREVIEW_PROPERTY];
          delete node.properties[PREVIEW_EXPANDED_PROPERTY];
          delete node.properties[PREVIEW_SOURCE_PROPERTY];
        }
      }

      function getSavedPreviewValue(node, info) {
        const previewFromProperties =
          info?.properties?.[PREVIEW_PROPERTY] ??
          node.properties?.[PREVIEW_PROPERTY] ??
          "";
        if (previewFromProperties) return previewFromProperties;

        const widgetValues = info?.widgets_values ?? node.widgets_values;
        if (!Array.isArray(widgetValues)) return "";

        // Compatibility with workflows saved while _prompt_display was serialized
        // as an extra widget value.
        return widgetValues.length > 3 ? widgetValues[3] || "" : "";
      }

      function getSavedPreviewSource(node, info) {
        return (
          info?.properties?.[PREVIEW_SOURCE_PROPERTY] ??
          node.properties?.[PREVIEW_SOURCE_PROPERTY] ??
          getCurrentText(node)
        );
      }

      function setWidgetHidden(widget, hidden) {
        if (!widget) return;
        if (!widget._a1111OriginalComputeSize) {
          widget._a1111OriginalComputeSize = widget.computeSize;
        }

        if (hidden) {
          widget.computeSize = () => [0, HIDDEN_WIDGET_HEIGHT];
          if (widget.inputEl) widget.inputEl.style.display = "none";
        } else {
          widget.computeSize = widget._a1111OriginalComputeSize;
          if (widget.inputEl) widget.inputEl.style.display = "block";
        }
      }

      function setPreviewWidgetHidden(widget, hidden) {
        if (!widget) return;
        if (hidden) {
          widget.computeSize = () => [0, HIDDEN_WIDGET_HEIGHT];
          if (widget.inputEl) widget.inputEl.style.display = "none";
          return;
        }

        widget.computeSize = () => [0, PREVIEW_WIDGET_HEIGHT];
        if (widget.inputEl) {
          widget.inputEl.style.display = "block";
          widget.inputEl.style.height = `${PREVIEW_WIDGET_HEIGHT - 16}px`;
          widget.inputEl.style.boxSizing = "border-box";
          widget.inputEl.style.border = "1px solid var(--border-color, #4b4664)";
          widget.inputEl.style.borderRadius = "4px";
        }
      }

      function ensureToggleWidget(node) {
        let toggleWidget = node.widgets?.find(
          (w) => w._a1111PreviewToggle || w.name === "_prompt_display_toggle"
        );
        if (!toggleWidget) {
          toggleWidget = node.addWidget(
            "button",
            "resolved prompt",
            "Show",
            () => {}
          );
          toggleWidget.serialize = false;
        }
        toggleWidget._a1111PreviewToggle = true;
        toggleWidget.serialize = false;
        toggleWidget.type = "button";
        toggleWidget.callback = () => {
          setPreviewExpanded(node, !isPreviewExpanded(node));
          renderPreview(node);
        };
        return toggleWidget;
      }

      function ensureDisplayWidget(node) {
        let displayWidget = node.widgets?.find(
          (w) => w.name === "_prompt_display"
        );

        if (!displayWidget) {
          const widgetResult = ComfyWidgets["STRING"](
            node,
            "_prompt_display",
            ["STRING", { multiline: true }],
            app
          );
          displayWidget = widgetResult.widget;
        }

        displayWidget.serialize = false;
        if (displayWidget.inputEl) {
          displayWidget.inputEl.readOnly = true;
          displayWidget.inputEl.style.opacity = "0.7";
          displayWidget.inputEl.style.fontStyle = "italic";
          displayWidget.inputEl.placeholder =
            "(Prompt will appear after execution)";
        }

        return displayWidget;
      }

      function hideDisplayWidget(node, displayWidget, clearValue = true) {
        displayWidget.type = "converted-widget";
        setPreviewWidgetHidden(displayWidget, true);
        if (clearValue) displayWidget.value = "";
        resizeNode(node);
      }

      function showDisplayWidget(node, displayWidget, textValue) {
        displayWidget.type = "customtext";
        setPreviewWidgetHidden(displayWidget, false);
        displayWidget.value = textValue;
        resizeNode(node);
      }

      function renderPreview(node) {
        const previewText = getPreviewText(node);
        const currentText = getCurrentText(node);
        const toggleWidget = node.widgets?.find(
          (w) => w._a1111PreviewToggle || w.name === "_prompt_display_toggle"
        );
        const displayWidget = node.widgets?.find(
          (w) => w.name === "_prompt_display"
        );

        if (!previewText || previewText === currentText) {
          if (previewText === currentText) setPreviewText(node, "");
          if (toggleWidget) setWidgetHidden(toggleWidget, true);
          if (displayWidget) hideDisplayWidget(node, displayWidget, true);
          resizeNode(node);
          return;
        }

        const visibleToggle = ensureToggleWidget(node);
        const isExpanded = isPreviewExpanded(node);
        visibleToggle.label = isExpanded
          ? "Hide"
          : "Show";
        visibleToggle.value = visibleToggle.label;
        setWidgetHidden(visibleToggle, false);

        if (isExpanded) {
          showDisplayWidget(node, ensureDisplayWidget(node), previewText);
        } else if (displayWidget) {
          hideDisplayWidget(node, displayWidget, false);
        }

        resizeNode(node);
      }

      /**
       * Populate the node with a readonly text widget showing the prompt.
       */
      function populate(text) {
        const textValue = Array.isArray(text) ? text[0] : text;
        const currentText = getCurrentText(this);

        // Only show if the effective prompt is different from the input text.
        // This avoids clutter when no expansion (TIPO, etc.) is happening.
        if (!textValue || textValue === currentText) {
          setPreviewText(this, "");
          renderPreview(this);
          return;
        }

        setPreviewText(this, textValue, currentText);
        renderPreview(this);
      }

      function clearPreviewOnTextEdit(node) {
        const textWidget = node.widgets?.find((w) => w.name === "text");
        const textInput = textWidget?.inputEl;
        if (!textInput || textInput.dataset.a1111PromptPreviewClear === "true") {
          return;
        }

        textInput.dataset.a1111PromptPreviewClear = "true";
        textInput.addEventListener("input", () => {
          const sourceText = getPreviewSource(node);
          if (sourceText && getCurrentText(node) === sourceText) {
            renderPreview(node);
            return;
          }
          if (getPreviewText(node)) {
            setPreviewText(node, "");
            renderPreview(node);
          }
        });
      }

      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);
        requestAnimationFrame(() => clearPreviewOnTextEdit(this));
      };

      // Hook into onExecuted to display the prompt after execution
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        populate.call(this, message?.text ?? "");
      };

      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        onConfigure?.apply(this, arguments);
        requestAnimationFrame(() => {
          clearPreviewOnTextEdit(this);
          const savedPreview = getSavedPreviewValue(this, info);
          if (savedPreview) setPreviewText(this, savedPreview, getSavedPreviewSource(this, info));
          renderPreview(this);
        });
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

    // Add cleanup on node removal
    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (onRemoved) onRemoved.apply(this, arguments);

      if (this._boundaryTokenTimeout) {
        clearTimeout(this._boundaryTokenTimeout);
        this._boundaryTokenTimeout = null;
      }

      const textarea = this.widgets?.find((w) => w.name === "text")?.inputEl;
      if (textarea && this._boundaryInputHandler) {
        textarea.removeEventListener("input", this._boundaryInputHandler);
        this._boundaryInputHandler = null;
      }

      if (textarea && this._boundaryFocusHandler) {
        textarea.removeEventListener("focus", this._boundaryFocusHandler);
        this._boundaryFocusHandler = null;
      }

      if (textarea && this._boundaryScrollHandler) {
        textarea.removeEventListener("scroll", this._boundaryScrollHandler);
        this._boundaryScrollHandler = null;
      }

      if (this._boundaryObserver) {
        this._boundaryObserver.disconnect();
        this._boundaryObserver = null;
      }

      if (this._overlayContainer && this._overlayContainer.parentNode) {
        this._overlayContainer.parentNode.removeChild(this._overlayContainer);
        this._overlayContainer = null;
      }
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
    let activeTextarea = null;
    let tokenUpdateTimeout = null;
    let handleInput = null;
    let handleFocus = null;

    const cleanupOverlay = () => {
      if (node._boundaryBoundTextarea && node._boundaryInputHandler) {
        node._boundaryBoundTextarea.removeEventListener(
          "input",
          node._boundaryInputHandler
        );
      }

      if (node._boundaryBoundTextarea && node._boundaryFocusHandler) {
        node._boundaryBoundTextarea.removeEventListener(
          "focus",
          node._boundaryFocusHandler
        );
      }

      if (activeTextarea && node._boundaryScrollHandler) {
        activeTextarea.removeEventListener("scroll", node._boundaryScrollHandler);
        node._boundaryScrollHandler = null;
      }

      node._boundaryBoundTextarea = null;
      node._boundaryInputHandler = null;
      node._boundaryFocusHandler = null;

      if (node._boundaryObserver) {
        node._boundaryObserver.disconnect();
        node._boundaryObserver = null;
      }

      if (overlayContainer?.parentNode) {
        overlayContainer.parentNode.removeChild(overlayContainer);
      }

      overlayContainer = null;
      mirrorDiv = null;
      activeTextarea = null;
      node._overlayContainer = null;
    };

    const bindTextarea = (textarea) => {
      if (!textarea || node._boundaryBoundTextarea === textarea) return;

      if (node._boundaryBoundTextarea && node._boundaryInputHandler) {
        node._boundaryBoundTextarea.removeEventListener(
          "input",
          node._boundaryInputHandler
        );
      }

      if (node._boundaryBoundTextarea && node._boundaryFocusHandler) {
        node._boundaryBoundTextarea.removeEventListener(
          "focus",
          node._boundaryFocusHandler
        );
      }

      if (!handleInput) {
        handleInput = () => {
          updateTokenCount(textWidget.inputEl?.value || "");
          requestAnimationFrame(updateBoundaryMarkers);
        };
      }

      if (!handleFocus) {
        handleFocus = () => {
          ensureOverlay();
          requestAnimationFrame(updateBoundaryMarkers);
        };
      }

      textarea.addEventListener("input", handleInput);
      textarea.addEventListener("focus", handleFocus);
      node._boundaryInputHandler = handleInput;
      node._boundaryFocusHandler = handleFocus;
      node._boundaryBoundTextarea = textarea;
    };

    const ensureOverlay = () => {
      const textarea = textWidget.inputEl;
      if (!textarea) return false;

      if (
        textarea !== activeTextarea ||
        !overlayContainer ||
        !overlayContainer.isConnected ||
        !mirrorDiv
      ) {
        createOverlay();
      }

      bindTextarea(textarea);

      return Boolean(mirrorDiv);
    };

    const createOverlay = () => {
      if (!textWidget.inputEl) return;

      const textarea = textWidget.inputEl;
      cleanupOverlay();
      activeTextarea = textarea;

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
      if (!textarea.parentNode) return;
      textarea.parentNode.style.position = "relative";
      textarea.parentNode.insertBefore(overlayContainer, textarea);
      node._overlayContainer = overlayContainer;

      // Copy textarea styles to mirror
      const copyStyles = () => {
        const computed = window.getComputedStyle(textarea);
        
        // Critical layout properties
        mirrorDiv.style.boxSizing = computed.boxSizing;
        mirrorDiv.style.width = computed.width;
        mirrorDiv.style.height = computed.height;
        mirrorDiv.style.padding = computed.padding;
        mirrorDiv.style.border = computed.border;
        mirrorDiv.style.borderColor = "transparent"; // Keep invisible
        
        // Font properties - copy individually to be safe
        mirrorDiv.style.fontFamily = computed.fontFamily;
        mirrorDiv.style.fontSize = computed.fontSize;
        mirrorDiv.style.fontWeight = computed.fontWeight;
        mirrorDiv.style.letterSpacing = computed.letterSpacing;
        mirrorDiv.style.lineHeight = computed.lineHeight;
        
        // Text wrapping properties
        mirrorDiv.style.whiteSpace = computed.whiteSpace;
        mirrorDiv.style.wordWrap = computed.wordWrap;
        mirrorDiv.style.wordBreak = computed.wordBreak;
        mirrorDiv.style.overflowWrap = computed.overflowWrap;
        
        // Handle scrollbar width difference matching
        // If textarea has a visible scrollbar, force one on mirror so wrapping matches
        if (textarea.scrollHeight > textarea.clientHeight) {
             mirrorDiv.style.overflowY = "scroll";
        } else {
             mirrorDiv.style.overflowY = "hidden";
        }
      };

      copyStyles();

      // Sync scroll position
      const syncScroll = () => {
        if (!mirrorDiv) return;
        mirrorDiv.scrollTop = textarea.scrollTop;
        mirrorDiv.scrollLeft = textarea.scrollLeft;
      };
      textarea.addEventListener("scroll", syncScroll);
      node._boundaryScrollHandler = syncScroll;

      // Use ResizeObserver to keep mirror synced with textarea size
      const observer = new ResizeObserver(() => {
        copyStyles();
        // Force re-sync of scroll after resize as content might reflow
        syncScroll();
      });
      observer.observe(textarea);
      node._boundaryObserver = observer;
    };

    const updateBoundaryMarkers = () => {
      if (!ensureOverlay()) return;

      const text = textWidget.inputEl.value || "";
      const boundaries = node._tokenInfo?.boundaries || [];

      if (boundaries.length === 0) {
        mirrorDiv.textContent = text;
        return;
      }

      // Build HTML with boundary markers
      let html = "";
      let lastPos = 0;

      // Sort boundaries by position. Estimated wildcard boundaries can map
      // multiple chunk cuts to the same source position, so equal positions
      // must still render separate markers.
      const sortedBoundaries = [...boundaries]
        .filter((boundary) => Number.isFinite(boundary?.char_pos))
        .sort((a, b) => a.char_pos - b.char_pos);
      const markersAtPosition = new Map();

      for (const boundary of sortedBoundaries) {
        const pos = Math.min(boundary.char_pos, text.length);
        if (pos < lastPos) continue;

        // Add text before boundary
        if (pos > lastPos) {
          html += escapeHtml(text.slice(lastPos, pos));
        }

        // Add boundary marker
        const markerColor =
          boundary.type === "break"
            ? "#3498db" // Blue for BREAK
            : "#e67e22"; // Orange for chunk boundary
        const markerOpacity = boundary.estimated ? 0.45 : 0.7;
        const markerOffset = markersAtPosition.get(pos) || 0;
        markersAtPosition.set(pos, markerOffset + 1);

        html += `<span style="
          display: inline-block;
          width: 0;
          height: 1.2em;
          vertical-align: middle;
          position: relative;
          overflow: visible;
          margin: 0;
        "><span style="
          display: block;
          position: absolute;
          left: ${-1 + markerOffset * 3}px;
          top: 0;
          width: 2px;
          height: 100%;
          background: ${markerColor};
          border-radius: 1px;
          opacity: ${markerOpacity};
        "></span></span>`;

        lastPos = pos;
      }

      // Add remaining text
      html += escapeHtml(text.slice(lastPos));

      mirrorDiv.innerHTML = html;
    };

    const updateTokenCount = async (text) => {
      clearTimeout(tokenUpdateTimeout);
      tokenUpdateTimeout = setTimeout(async () => {
        try {
          const response = await fetch("/a1111_prompt/tokenize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
          });
          
          if (!response.ok) {
            console.warn(`[A1111 Prompt] Tokenization API error: ${response.status}`);
            return;
          }
          
          node._tokenInfo = await response.json();
          node.setDirtyCanvas(true, false);

          // Update boundary markers
          requestAnimationFrame(() => {
            updateBoundaryMarkers();
          });
        } catch (e) {
          // Silently ignore errors, just keep old count
        }
      }, 300); // 300ms debounce
      node._boundaryTokenTimeout = tokenUpdateTimeout;
    };

    // Hook into widget callback for text changes
    const origCallback = textWidget.callback;
    textWidget.callback = function (value) {
      origCallback?.apply(this, arguments);
      updateTokenCount(
        typeof value === "string" ? value : textWidget.inputEl?.value || ""
      );
    };

    // Create overlay when the textarea is available
    const waitForTextarea = () => {
      if (textWidget.inputEl) {
        ensureOverlay();
        updateTokenCount(textWidget.inputEl.value || textWidget.value || "");
      } else {
        requestAnimationFrame(waitForTextarea);
      }
    };
    requestAnimationFrame(waitForTextarea);
  },
});
