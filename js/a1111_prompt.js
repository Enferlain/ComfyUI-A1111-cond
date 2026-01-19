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
