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
    width: max-content;
    min-width: 250px;
    max-width: 800px;
    max-height: 400px;
    overflow-y: auto;
    overflow-x: hidden;
    display: none;
    padding: 4px 0;
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
    .autocomplete-item:hover {
      background-color: rgba(255, 255, 255, 0.05);
    }
    .autocomplete-item.selected {
      background-color: rgba(74, 158, 255, 0.25) !important;
      border-left: 3px solid #4a9eff;
      padding-left: 9px !important;
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
// From https://github.com/component/textarea-caret-position
const CARET_PROPERTIES = [
  'direction', 'boxSizing', 'width', 'height', 'overflowX', 'overflowY',
  'borderTopWidth', 'borderRightWidth', 'borderBottomWidth', 'borderLeftWidth', 'borderStyle',
  'paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft',
  'fontStyle', 'fontVariant', 'fontWeight', 'fontStretch', 'fontSize', 'fontSizeAdjust', 'lineHeight', 'fontFamily',
  'textAlign', 'textTransform', 'textIndent', 'textDecoration',
  'letterSpacing', 'wordSpacing', 'tabSize', 'MozTabSize',
  'whiteSpace', 'wordBreak', 'wordWrap', 'overflowWrap'
];

/**
 * Get cursor coordinates in the viewport
 * @param {HTMLTextAreaElement} element - The textarea element
 * @param {number} position - Cursor position in text
 * @returns {Object} Object with x, y, height coordinates
 */
/**
 * Get cursor coordinates in the viewport
 * @param {HTMLTextAreaElement} element - The textarea element
 * @param {number} position - Cursor position in text
 * @returns {Object} Object with x, y, height coordinates
 */
function getCaretCoordinates(element, position) {
  if (typeof window === 'undefined') return { x: 0, y: 0, height: 0 };

  // Create a mirror div to replicate the textarea's style in 1:1 layout space
  const div = document.createElement('div');
  div.id = 'input-textarea-caret-position-mirror-div';
  document.body.appendChild(div);

  const style = div.style;
  const computed = window.getComputedStyle(element);

  // Transfer all layout-affecting properties
  CARET_PROPERTIES.forEach(prop => {
    style[prop] = computed[prop];
  });

  // Reset positioning for the mirror
  style.position = 'absolute';
  style.visibility = 'hidden';
  style.pointerEvents = 'none';
  style.top = '0';
  style.left = '0';

  // Force scrollbar behavior to match accurately for width calculations
  if (element.scrollHeight > element.clientHeight) {
    style.overflowY = 'scroll';
  } else {
    style.overflowY = 'hidden';
  }

  // Replicate text content up to the caret
  div.textContent = element.value.substring(0, position);
  
  const span = document.createElement('span');
  span.textContent = element.value.substring(position) || '.';
  div.appendChild(span);

  // Get layout-local coordinates
  // offsetTop/Left are relative to the div's border box
  const layoutX = span.offsetLeft;
  const layoutY = span.offsetTop;
  const layoutHeight = parseInt(computed.lineHeight) || 20;

  // Cleanup mirror
  document.body.removeChild(div);

  // Calculate scaling factor between layout and viewport
  // This is crucial for ComfyUI's zoom/pan
  const rect = element.getBoundingClientRect();
  
  // Use offsetWidth as the layout width reference
  // If offsetWidth is 0 (not visible), fallback to 1 to avoid division by zero
  const layoutWidth = element.offsetWidth || parseFloat(computed.width) || 1;
  const scale = rect.width / layoutWidth;

  // Map local layout coordinates to viewport coordinates
  // We account for: 
  // 1. Textarea's viewport position (rect.left/top)
  // 2. Local caret offset in layout units (layoutX/Y)
  // 3. Current scroll position (element.scrollLeft/Top)
  // 4. Transform scale (scale)
  // 5. Borders (computed border width)
  
  const borderLeft = parseInt(computed.borderLeftWidth) || 0;
  const borderTop = parseInt(computed.borderTopWidth) || 0;

  return {
    x: rect.left + (layoutX + borderLeft - element.scrollLeft) * scale,
    y: rect.top + (layoutY + borderTop - element.scrollTop) * scale,
    height: layoutHeight * scale
  };
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
  const sortedSuggestions = sortByFrequency(suggestions);
  
  currentResults = sortedSuggestions;
  selectedIndex = -1;
  currentTextarea = textarea;
  currentWordInfo = wordInfo;

  popup.innerHTML = "";

  sortedSuggestions.forEach((tag, index) => {
    const item = document.createElement("div");
    item.className = "autocomplete-item";
    item.style.cssText = `
      padding: 8px 16px;
      cursor: pointer;
      border-bottom: 1px solid var(--border-color, #444);
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: background-color 0.15s ease;
      gap: 20px;
    `;

    const nameSpan = document.createElement("span");
    nameSpan.style.cssText = `
      color: ${TAG_COLORS[tag.type] || "var(--fg-color, #ccc)"};
      font-weight: 500;
      flex: 1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    `;
    
    let displayName = tag.name;
    if (tag.matched_alias) {
      displayName = `${tag.matched_alias} → ${tag.name}`;
      nameSpan.style.fontStyle = "italic";
    }
    nameSpan.textContent = displayName;

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
      nameSpan.appendChild(freqSpan);
    }

    const countSpan = document.createElement("span");
    countSpan.style.cssText = `
      color: var(--descrip-text, #888);
      font-size: 11px;
      margin-left: 8px;
    `;
    countSpan.textContent = formatCount(tag.count);

    item.appendChild(nameSpan);
    item.appendChild(countSpan);
    item.addEventListener("click", () => insertTag(index));
    popup.appendChild(item);
  });

  // Calculate position using robust caret logic
  const coords = getCaretCoordinates(textarea, wordInfo.start);
  const popupHeight = Math.min(300, currentResults.length * 40); // Estimate
  const viewportHeight = window.innerHeight;
  
  // Decide whether to show above or below
  let top = coords.y + coords.height + 5;
  if (top + popupHeight > viewportHeight - 10) {
    // Show above if it doesn't fit below
    top = coords.y - popupHeight - 5;
  }

  popup.style.left = `${Math.max(10, Math.min(coords.x, window.innerWidth - 360))}px`;
  popup.style.top = `${top}px`;
  popup.style.display = "block";
  setSelectedIndex(-1);
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
    item.classList.remove("selected");
  });

  // Set new selection, allowing -1 for "no selection"
  selectedIndex = Math.max(-1, Math.min(index, currentResults.length - 1));
  
  if (selectedIndex >= 0 && items[selectedIndex]) {
    items[selectedIndex].classList.add("selected");
    
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

  incrementTagFrequency(tag.name);

  const text = textarea.value;
  let tagName = tag.name;
  
  // Configurable A1111 formatting (usually on by default)
  tagName = tagName.replace(/_/g, " ");
  tagName = tagName.replace(/\(/g, "\\(").replace(/\)/g, "\\)");

  const beforeWord = text.substring(0, wordInfo.start);
  const afterWord = text.substring(wordInfo.end);
  
  // Smart separator logic from reference
  // Only add if not already present
  let suffix = "";
  const matchAfter = afterWord.match(/^\s*[,:]/);
  if (!matchAfter) {
    suffix = ", ";
  } else if (!afterWord.startsWith(", ")) {
    // If it has a comma but no space, maybe add space
    if (afterWord.startsWith(",") && !afterWord.startsWith(", ")) {
      // Just ensure space exists after
    }
  }

  const newText = beforeWord + tagName + suffix + afterWord;
  const newCursorPos = beforeWord.length + tagName.length + suffix.length;

  hideAutocompletePopup();

  textarea.value = newText;
  textarea.setSelectionRange(newCursorPos, newCursorPos);
  textarea.dispatchEvent(new Event("input", { bubbles: true }));
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

  const length = currentResults.length;

  switch (e.key) {
    case "ArrowDown":
      e.preventDefault();
      if (selectedIndex === -1) {
        setSelectedIndex(0);
      } else {
        setSelectedIndex((selectedIndex + 1) % length);
      }
      return true;

    case "ArrowUp":
      e.preventDefault();
      if (selectedIndex === -1) {
        setSelectedIndex(length - 1);
      } else {
        setSelectedIndex((selectedIndex - 1 + length) % length);
      }
      return true;

    case "Tab":
    case "Enter":
      if (selectedIndex === -1) {
        // No selection, allow default behavior (like newline)
        hideAutocompletePopup();
        return false;
      }
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
  // Find word boundaries - tags are separated by commas, spaces, parentheses, colons, brackets, etc.
  // Matching reference regex logic: [^\s,|<>():\[\]]
  let start = cursorPos;
  let end = cursorPos;

  const isSeparator = (char) => /[\s,()|<>:\[\]]/.test(char);

  // Move start back to word boundary
  while (start > 0 && !isSeparator(text[start - 1])) {
    start--;
  }

  // Move end forward to word boundary
  while (end < text.length && !isSeparator(text[end])) {
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
