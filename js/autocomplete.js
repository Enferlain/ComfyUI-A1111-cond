/**
 * Tag Autocomplete UI
 *
 * Provides A1111-style tag autocomplete functionality:
 * - Popup when typing (like A1111's tag autocomplete)
 * - Support Danbooru/e621 tag databases
 * - Keyboard navigation (up/down/tab/enter/escape)
 * - Mouse click selection
 * - Alias support and display
 */

import { app } from "../../../scripts/app.js";

// Global state
let autocompletePopup = null;
let currentTextarea = null;
let currentWordInfo = null;
let selectedIndex = -1;
let currentResults = [];
let debounceTimeout = null;

// Frequency tracking
const FREQUENCY_STORAGE_KEY = "a1111_tag_frequency";
let tagFrequency = {}; // tag_name -> usage_count

/**
 * Load tag frequency data from localStorage
 */
function loadTagFrequency() {
  try {
    const stored = localStorage.getItem(FREQUENCY_STORAGE_KEY);
    if (stored) {
      tagFrequency = JSON.parse(stored);
    }
  } catch (e) {
    console.warn("Failed to load tag frequency:", e);
    tagFrequency = {};
  }
}

/**
 * Save tag frequency data to localStorage
 */
function saveTagFrequency() {
  try {
    localStorage.setItem(FREQUENCY_STORAGE_KEY, JSON.stringify(tagFrequency));
  } catch (e) {
    console.warn("Failed to save tag frequency:", e);
  }
}

/**
 * Increment usage count for a tag
 * @param {string} tagName - The tag name
 */
function incrementTagFrequency(tagName) {
  tagFrequency[tagName] = (tagFrequency[tagName] || 0) + 1;
  saveTagFrequency();
}

/**
 * Get usage count for a tag
 * @param {string} tagName - The tag name
 * @returns {number} Usage count
 */
function getTagFrequency(tagName) {
  return tagFrequency[tagName] || 0;
}

/**
 * Calculate frequency boost score
 * Uses logarithmic scaling to prevent over-boosting
 * @param {number} frequency - Usage count
 * @returns {number} Boost score
 */
function calculateFrequencyBoost(frequency) {
  if (frequency === 0) return 0;
  // Logarithmic boost: log2(frequency + 1)
  // This gives: 1 use = 1.0, 3 uses = 2.0, 7 uses = 3.0, 15 uses = 4.0
  return Math.log2(frequency + 1);
}

// Load frequency data on startup
loadTagFrequency();

// Tag type colors (matching backend TAG_TYPES)
const TAG_COLORS = {
  0: "#87ceeb", // general - lightblue
  1: "#cd5c5c", // artist - indianred
  3: "#ee82ee", // copyright - violet
  4: "#90ee90", // character - lightgreen
  5: "#ffa500", // meta - orange
};

/**
 * Create the autocomplete popup element
 * @returns {HTMLElement} The popup element
 */
export function createAutocompletePopup() {
  if (autocompletePopup) return autocompletePopup;

  autocompletePopup = document.createElement("div");
  autocompletePopup.className = "a1111-autocomplete-popup";
  
  // Use ComfyUI theme variables for consistent theming
  autocompletePopup.style.cssText = `
    position: fixed;
    z-index: 10000;
    background: var(--comfy-menu-bg, #353535);
    border: 2px solid var(--border-color, #555);
    border-radius: 6px;
    max-width: 350px;
    max-height: 300px;
    overflow-y: auto;
    display: none;
    box-shadow: 0 8px 24px rgba(0,0,0,0.8), 0 0 0 1px rgba(255,255,255,0.1);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 13px;
    color: var(--fg-color, #fff);
  `;

  // Add scrollbar styling that respects theme
  const style = document.createElement("style");
  style.textContent = `
    .a1111-autocomplete-popup::-webkit-scrollbar {
      width: 8px;
    }
    .a1111-autocomplete-popup::-webkit-scrollbar-track {
      background: var(--comfy-input-bg, #222);
    }
    .a1111-autocomplete-popup::-webkit-scrollbar-thumb {
      background: var(--border-color, #555);
      border-radius: 4px;
    }
    .a1111-autocomplete-popup::-webkit-scrollbar-thumb:hover {
      background: var(--content-hover-bg, #666);
    }
  `;
  document.head.appendChild(style);

  document.body.appendChild(autocompletePopup);
  return autocompletePopup;
}

/**
 * Get cursor coordinates in the viewport
 * @param {HTMLTextAreaElement} textarea - The textarea element
 * @param {number} cursorPos - Cursor position in text
 * @returns {Object} Object with x, y coordinates
 */
function getCaretCoordinates(textarea, cursorPos) {
  // Create a mirror div to measure text position
  const mirror = document.createElement("div");
  const computed = window.getComputedStyle(textarea);
  
  // Copy textarea styles to mirror
  mirror.style.cssText = `
    position: absolute;
    top: -9999px;
    left: -9999px;
    white-space: pre-wrap;
    word-wrap: break-word;
    visibility: hidden;
  `;
  
  // Copy all relevant styles
  const properties = [
    'font-family', 'font-size', 'font-weight', 'font-style',
    'letter-spacing', 'line-height', 'padding', 'border',
    'width', 'height'
  ];
  
  properties.forEach(prop => {
    mirror.style[prop] = computed[prop];
  });
  
  document.body.appendChild(mirror);
  
  // Set text up to cursor position
  const textBeforeCursor = textarea.value.substring(0, cursorPos);
  mirror.textContent = textBeforeCursor;
  
  // Add a span to measure cursor position
  const span = document.createElement("span");
  span.textContent = "|";
  mirror.appendChild(span);
  
  // Get textarea position
  const textareaRect = textarea.getBoundingClientRect();
  const spanRect = span.getBoundingClientRect();
  const mirrorRect = mirror.getBoundingClientRect();
  
  // Calculate position relative to textarea
  const x = textareaRect.left + (spanRect.left - mirrorRect.left) - textarea.scrollLeft;
  const y = textareaRect.top + (spanRect.top - mirrorRect.top) - textarea.scrollTop;
  
  document.body.removeChild(mirror);
  
  return { x, y };
}

/**
 * Show autocomplete suggestions
 * @param {Array} suggestions - Array of tag suggestions
 * @param {HTMLElement} textarea - The textarea element
 * @param {Object} wordInfo - Current word information
 */
export function showAutocompleteSuggestions(suggestions, textarea, wordInfo) {
  if (!suggestions || suggestions.length === 0) {
    hideAutocompletePopup();
    return;
  }

  const popup = createAutocompletePopup();
  
  // Apply frequency sorting to suggestions
  const sortedSuggestions = sortByFrequency(suggestions);
  
  currentResults = sortedSuggestions;
  selectedIndex = -1;
  currentTextarea = textarea;
  currentWordInfo = wordInfo;

  // Clear previous content
  popup.innerHTML = "";

  // Create result items
  sortedSuggestions.forEach((tag, index) => {
    const item = document.createElement("div");
    item.className = "autocomplete-item";
    item.style.cssText = `
      padding: 8px 12px;
      cursor: pointer;
      border-bottom: 1px solid var(--border-color, #444);
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: background-color 0.15s ease;
    `;

    // Tag name and alias info
    const nameSpan = document.createElement("span");
    nameSpan.style.cssText = `
      color: ${TAG_COLORS[tag.type] || "var(--fg-color, #ccc)"};
      font-weight: 500;
      flex: 1;
    `;
    
    let displayName = tag.name;
    if (tag.matched_alias) {
      displayName = `${tag.matched_alias} → ${tag.name}`;
      nameSpan.style.fontStyle = "italic";
    }
    nameSpan.textContent = displayName;

    // Frequency indicator (if used before)
    const frequency = getTagFrequency(tag.name);
    if (frequency > 0) {
      const freqSpan = document.createElement("span");
      freqSpan.style.cssText = `
        color: #4a9eff;
        font-size: 10px;
        margin-left: 6px;
        padding: 2px 4px;
        background: rgba(74, 158, 255, 0.15);
        border-radius: 3px;
      `;
      freqSpan.textContent = `★${frequency}`;
      freqSpan.title = `Used ${frequency} time${frequency > 1 ? 's' : ''}`;
      nameSpan.appendChild(freqSpan);
    }

    // Post count
    const countSpan = document.createElement("span");
    countSpan.style.cssText = `
      color: var(--descrip-text, #888);
      font-size: 11px;
      margin-left: 8px;
    `;
    countSpan.textContent = formatCount(tag.count);

    item.appendChild(nameSpan);
    item.appendChild(countSpan);

    // Click handler
    item.addEventListener("click", () => {
      insertTag(index);
    });

    // Hover handler
    item.addEventListener("mouseenter", () => {
      setSelectedIndex(index);
    });

    popup.appendChild(item);
  });

  // Position popup near cursor
  const coords = getCaretCoordinates(textarea, wordInfo.start);
  popup.style.left = `${coords.x}px`;
  popup.style.top = `${coords.y + 20}px`; // Offset below cursor

  // Ensure popup stays in viewport
  const rect = popup.getBoundingClientRect();
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;

  if (rect.right > viewportWidth) {
    popup.style.left = `${viewportWidth - rect.width - 10}px`;
  }
  if (rect.bottom > viewportHeight) {
    popup.style.top = `${coords.y - rect.height - 5}px`; // Show above cursor
  }

  popup.style.display = "block";
  setSelectedIndex(0); // Select first item by default
}

/**
 * Sort suggestions by frequency and relevance
 * @param {Array} suggestions - Array of tag suggestions
 * @returns {Array} Sorted suggestions
 */
function sortByFrequency(suggestions) {
  return suggestions.map(tag => {
    const frequency = getTagFrequency(tag.name);
    const frequencyBoost = calculateFrequencyBoost(frequency);
    
    // Combined score: base score (from backend) + frequency boost
    // Backend already provides relevance score via ordering
    // We add frequency boost to prioritize frequently used tags
    return {
      ...tag,
      _frequency: frequency,
      _frequencyBoost: frequencyBoost
    };
  }).sort((a, b) => {
    // Sort by frequency boost first (higher is better)
    const freqDiff = b._frequencyBoost - a._frequencyBoost;
    if (Math.abs(freqDiff) > 0.5) { // Only prioritize if significant difference
      return freqDiff;
    }
    
    // Then by post count (backend relevance)
    return b.count - a.count;
  });
}

/**
 * Hide the autocomplete popup
 */
export function hideAutocompletePopup() {
  if (autocompletePopup) {
    autocompletePopup.style.display = "none";
    selectedIndex = -1;
    currentResults = [];
    currentTextarea = null;
    currentWordInfo = null;
  }
}

/**
 * Set the selected index and update visual selection
 * @param {number} index - Index to select
 */
function setSelectedIndex(index) {
  if (!autocompletePopup || currentResults.length === 0) return;

  // Remove previous selection
  const items = autocompletePopup.querySelectorAll(".autocomplete-item");
  items.forEach(item => {
    item.style.backgroundColor = "";
  });

  // Set new selection
  selectedIndex = Math.max(0, Math.min(index, currentResults.length - 1));
  if (items[selectedIndex]) {
    // Use theme variable for hover background
    items[selectedIndex].style.backgroundColor = "var(--interface-panel-hover-surface, var(--content-hover-bg, #404040))";
    
    // Scroll into view if needed
    items[selectedIndex].scrollIntoView({
      block: "nearest",
      behavior: "smooth"
    });
  }
}

/**
 * Insert the selected tag into the textarea
 * @param {number} index - Index of tag to insert (optional, uses selectedIndex if not provided)
 */
function insertTag(index = selectedIndex) {
  if (!currentTextarea || !currentWordInfo || index < 0 || index >= currentResults.length) {
    return;
  }

  const tag = currentResults[index];
  const textarea = currentTextarea;
  const wordInfo = currentWordInfo;

  // Track tag usage for frequency sorting
  incrementTagFrequency(tag.name);

  // Get current text and cursor position
  const text = textarea.value;
  const cursorPos = textarea.selectionStart;

  // Replace the current word with the selected tag
  let tagName = tag.name;
  
  // Convert underscores to spaces (configurable behavior)
  tagName = tagName.replace(/_/g, " ");
  
  // Escape parentheses in tag names (for A1111 compatibility)
  tagName = tagName.replace(/\(/g, "\\(").replace(/\)/g, "\\)");

  // Build new text
  const beforeWord = text.substring(0, wordInfo.start);
  const afterWord = text.substring(wordInfo.end);
  
  // Always add comma and space after tag completion
  // This prevents autocomplete from triggering on the just-completed tag
  const suffix = ", ";

  const newText = beforeWord + tagName + suffix + afterWord;
  const newCursorPos = beforeWord.length + tagName.length + suffix.length;

  // Hide popup BEFORE updating textarea to prevent re-triggering
  hideAutocompletePopup();

  // Update textarea
  textarea.value = newText;
  textarea.setSelectionRange(newCursorPos, newCursorPos);

  // Trigger change event for ComfyUI (but popup is already hidden)
  textarea.dispatchEvent(new Event("input", { bubbles: true }));

  // Focus back to textarea
  textarea.focus();
}

/**
 * Handle keyboard navigation in the popup
 * @param {KeyboardEvent} e - The keyboard event
 * @returns {boolean} True if event was handled
 */
function handleKeyboardNavigation(e) {
  if (!autocompletePopup || autocompletePopup.style.display === "none") {
    return false;
  }

  switch (e.key) {
    case "ArrowDown":
      e.preventDefault();
      setSelectedIndex(selectedIndex + 1);
      return true;

    case "ArrowUp":
      e.preventDefault();
      setSelectedIndex(selectedIndex - 1);
      return true;

    case "Tab":
    case "Enter":
      e.preventDefault();
      insertTag();
      return true;

    case "Escape":
      e.preventDefault();
      hideAutocompletePopup();
      return true;

    default:
      return false;
  }
}

/**
 * Format post count for display
 * @param {number} count - Post count
 * @returns {string} Formatted count
 */
function formatCount(count) {
  if (count >= 1000000) {
    return `${(count / 1000000).toFixed(1)}M`;
  } else if (count >= 1000) {
    return `${(count / 1000).toFixed(1)}k`;
  }
  return count.toString();
}

/**
 * Fetch tag suggestions from the API
 * @param {string} query - The search query
 * @param {number} limit - Maximum number of results
 * @returns {Promise<Array>} Array of tag suggestions
 */
export async function fetchTagSuggestions(query, limit = 20) {
  if (!query || query.length < 2) return [];

  try {
    const response = await fetch("/a1111_prompt/autocomplete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        limit: limit,
        search_aliases: true,
        extra_files: ["extra-quality-tags.csv"] // Include quality tags
      }),
    });

    if (!response.ok) {
      console.warn("Autocomplete API error:", response.status);
      return [];
    }

    const data = await response.json();
    return data.results || [];
  } catch (error) {
    console.warn("Autocomplete fetch error:", error);
    return [];
  }
}

/**
 * Reset tag frequency data
 * Useful for clearing usage statistics
 */
export function resetTagFrequency() {
  if (confirm("Reset all tag frequency data? This will clear your usage statistics.")) {
    tagFrequency = {};
    saveTagFrequency();
    console.log("[Autocomplete] Tag frequency data reset");
  }
}

/**
 * Export tag frequency data for backup
 * @returns {string} JSON string of frequency data
 */
export function exportTagFrequency() {
  return JSON.stringify(tagFrequency, null, 2);
}

/**
 * Get statistics about tag usage
 * @returns {Object} Statistics object
 */
export function getFrequencyStats() {
  const entries = Object.entries(tagFrequency);
  const totalTags = entries.length;
  const totalUses = entries.reduce((sum, [_, count]) => sum + count, 0);
  const topTags = entries
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([tag, count]) => ({ tag, count }));
  
  return {
    totalTags,
    totalUses,
    topTags
  };
}

// Expose utility functions to window for console access
if (typeof window !== 'undefined') {
  window.A1111Autocomplete = {
    resetFrequency: resetTagFrequency,
    exportFrequency: exportTagFrequency,
    getStats: getFrequencyStats
  };
}

/**
 * Get the current word being typed at cursor position
 * @param {string} text - The full text
 * @param {number} cursorPos - Current cursor position
 * @returns {Object} Object with word, start, and end positions
 */
export function getCurrentWord(text, cursorPos) {
  // Find word boundaries - tags are separated by commas, spaces, or parentheses
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
    word: text.slice(start, end).trim(),
    start,
    end,
  };
}

// Extension registration
app.registerExtension({
  name: "A1111PromptNode.Autocomplete",

  async nodeCreated(node) {
    if (
      node.comfyClass !== "A1111Prompt" &&
      node.comfyClass !== "A1111PromptNegative"
    )
      return;

    const textWidget = node.widgets?.find((w) => w.name === "text");
    if (!textWidget) return;

    // Wait for textarea to be available
    const waitForTextarea = () => {
      if (!textWidget.inputEl) {
        requestAnimationFrame(waitForTextarea);
        return;
      }

      const textarea = textWidget.inputEl;

      // Add input event listener for autocomplete
      textarea.addEventListener("input", async (e) => {
        // Clear existing debounce
        if (debounceTimeout) {
          clearTimeout(debounceTimeout);
        }

        // Debounce the search to avoid excessive API calls
        debounceTimeout = setTimeout(async () => {
          const cursorPos = textarea.selectionStart;
          const wordInfo = getCurrentWord(textarea.value, cursorPos);

          if (wordInfo.word.length >= 2) {
            const suggestions = await fetchTagSuggestions(wordInfo.word);
            if (suggestions.length > 0) {
              showAutocompleteSuggestions(suggestions, textarea, wordInfo);
            } else {
              hideAutocompletePopup();
            }
          } else {
            hideAutocompletePopup();
          }
        }, 100); // 100ms debounce - faster response
      });

      // Add keydown event listener for navigation
      textarea.addEventListener("keydown", (e) => {
        if (handleKeyboardNavigation(e)) {
          // Event was handled by autocomplete
          return;
        }
      });

      // Hide popup on blur (with delay to allow clicks)
      textarea.addEventListener("blur", () => {
        setTimeout(() => {
          hideAutocompletePopup();
        }, 200);
      });

      // Hide popup when clicking outside
      document.addEventListener("click", (e) => {
        if (autocompletePopup && 
            !autocompletePopup.contains(e.target) && 
            e.target !== textarea) {
          hideAutocompletePopup();
        }
      });
    };

    requestAnimationFrame(waitForTextarea);
  },
});
