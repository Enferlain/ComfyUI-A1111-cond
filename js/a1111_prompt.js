import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

/**
 * A1111 Prompt Node Frontend Extension
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
          (w) => w.name === "_prompt_display",
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
            app,
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
                this.widgets?.find((w) => w.name === "text")?.value !== v,
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
      if (this._tokenInfo?.sequences && Array.isArray(this._tokenInfo.sequences)) {
        const seqs = this._tokenInfo.sequences;
        const text = seqs.map((s) => `${s}/75`).join(" | ");

        ctx.save();
        ctx.font = "11px monospace";
        ctx.fillStyle = "#888";
        ctx.textAlign = "right";
        ctx.fillText(text, this.size[0] - 10, -6);
        ctx.restore();
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

    node._tokenInfo = { sequences: [0], estimated: true };

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

    // Initial count
    updateTokenCount(textWidget.value || "");
  },
});
