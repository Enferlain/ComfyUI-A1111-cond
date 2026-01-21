/**
 * Tag Autocomplete UI (Placeholder)
 *
 * Future implementation for tag autocomplete functionality:
 * - Popup when typing (like A1111's tag autocomplete)
 * - Support Danbooru/e621 tag databases
 * - Custom tag lists (user-defined)
 * - Show tag frequency/popularity
 */

import { app } from "../../../scripts/app.js";

// Placeholder for autocomplete popup element
let autocompletePopup = null;

/**
 * Create the autocomplete popup element
 * @returns {HTMLElement} The popup element
 */
export function createAutocompletePopup() {
  if (autocompletePopup) return autocompletePopup;

  autocompletePopup = document.createElement("div");
  autocompletePopup.className = "a1111-autocomplete-popup";
  autocompletePopup.style.cssText = `
    position: fixed;
    z-index: 10000;
    background: #1a1a2e;
    border: 1px solid #444;
    border-radius: 8px;
    max-width: 300px;
    max-height: 300px;
    overflow-y: auto;
    display: none;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
  `;
  document.body.appendChild(autocompletePopup);
  return autocompletePopup;
}

/**
 * Show autocomplete suggestions
 * @param {Array} suggestions - Array of tag suggestions
 * @param {HTMLElement} textarea - The textarea element
 * @param {number} cursorPosition - Current cursor position
 */
export function showAutocompleteSuggestions(
  suggestions,
  textarea,
  cursorPosition
) {
  // TODO: Implement this
  console.log("Autocomplete not yet implemented");
}

/**
 * Hide the autocomplete popup
 */
export function hideAutocompletePopup() {
  if (autocompletePopup) {
    autocompletePopup.style.display = "none";
  }
}

/**
 * Fetch tag suggestions from the API
 * @param {string} query - The search query
 * @param {number} limit - Maximum number of results
 * @returns {Promise<Array>} Array of tag suggestions
 */
export async function fetchTagSuggestions(query, limit = 20) {
  // TODO: Implement API call when backend is ready
  return [];
}

/**
 * Get the current word being typed at cursor position
 * @param {string} text - The full text
 * @param {number} cursorPos - Current cursor position
 * @returns {Object} Object with word, start, and end positions
 */
export function getCurrentWord(text, cursorPos) {
  // Find word boundaries
  let start = cursorPos;
  let end = cursorPos;

  // Move start back to word boundary
  while (start > 0 && !/[\s,()]/.test(text[start - 1])) {
    start--;
  }

  // Move end forward to word boundary
  while (end < text.length && !/[\s,()]/.test(text[end])) {
    end++;
  }

  return {
    word: text.slice(start, end),
    start,
    end,
  };
}

// Extension registration placeholder
// Uncomment when implementing autocomplete functionality
/*
app.registerExtension({
  name: "A1111PromptNode.Autocomplete",

  async nodeCreated(node) {
    if (
      node.comfyClass !== "A1111Prompt" &&
      node.comfyClass !== "A1111PromptNegative"
    )
      return;

    const textWidget = node.widgets?.find((w) => w.name === "text");
    if (!textWidget?.inputEl) return;

    const textarea = textWidget.inputEl;
    
    // Add input event listener for autocomplete
    textarea.addEventListener("input", async (e) => {
      const cursorPos = textarea.selectionStart;
      const { word } = getCurrentWord(textarea.value, cursorPos);
      
      if (word.length >= 2) {
        const suggestions = await fetchTagSuggestions(word);
        if (suggestions.length > 0) {
          showAutocompleteSuggestions(suggestions, textarea, cursorPos);
        } else {
          hideAutocompletePopup();
        }
      } else {
        hideAutocompletePopup();
      }
    });

    // Hide popup on blur
    textarea.addEventListener("blur", () => {
      setTimeout(hideAutocompletePopup, 200);
    });
  },
});
*/
