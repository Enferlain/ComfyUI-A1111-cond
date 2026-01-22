# Tag Autocomplete Implementation Plan

This document outlines the work required to implement A1111-style tag autocomplete for the A1111 Prompt Node in ComfyUI, based on analysis of the [a1111-sd-webui-tagcomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete) extension.

## Executive Summary

The A1111 tag autocomplete is a mature, feature-rich extension (~68KB main JS, ~43KB Python backend). A full port would be substantial work. This plan proposes a **phased approach** starting with core functionality and progressively adding features.

---

## Phase 1: Core Tag Completion (MVP)

**Goal:** Basic tag autocomplete with Danbooru/e621 tags  
**Estimated Effort:** 3-5 days

### 1.1 Backend: Tag Data Loading

**File:** `api/autocomplete.py`

- [ ] **CSV Parser**: Parse tag CSV files with format `name,type,postCount,"aliases"`
- [ ] **Tag Types** (for color coding):
  - `0` = General (blue)
  - `1` = Artist (red)
  - `3` = Copyright (violet)
  - `4` = Character (green)
  - `5` = Meta (orange)
- [ ] **In-memory Tag Index**: Load tags into searchable data structure
- [ ] **Search API Endpoint**: `POST /a1111_prompt/autocomplete`
  ```python
  # Request: {"query": "1gir", "limit": 20}
  # Response: {"results": [{"name": "1girl", "type": 0, "count": 4114588, "aliases": ["1girls", "sole_female"]}]}
  ```
- [ ] **Lazy Loading**: Only load tags when first search is triggered (not on startup)

### 1.2 Frontend: Popup UI

**File:** `js/autocomplete.js`

- [ ] **Trigger Detection**: Detect when user is typing a tag (after comma, start of text, after `(` etc.)
- [ ] **Current Word Extraction**: Get the word being typed at cursor position
- [ ] **Popup Creation**: Create floating results div near cursor
  ```javascript
  // Use getCaretCoordinates() to position popup at cursor
  ```
- [ ] **Results Display**: Show tag name, count, and category color
- [ ] **Keyboard Navigation**:
  - `↑/↓` - Move selection
  - `Tab/Enter` - Accept selection
  - `Escape` - Close popup
- [ ] **Click Selection**: Click on result to insert
- [ ] **Debouncing**: 100-300ms delay to prevent excessive API calls

### 1.3 Tag Insertion Logic

- [ ] **Replace current word** with selected tag
- [ ] **Underscore to space**: Convert `1_girl` → `1 girl` (configurable)
- [ ] **Escape parentheses**: `name_(artist)` → `name_\(artist\)` (configurable)
- [ ] **Append comma + space** after insertion (configurable)

### 1.4 Data Files

**Directory:** `data/tags/`

- [ ] **Bundle default tag files**:
  - `danbooru.csv` (~4MB, top 100k tags by post count)
  - `e621.csv` (~3MB, alternative for e621-trained models)
- [ ] **Document CSV format** for users adding custom tags

---

## Phase 2: Alias & Translation Support ✅

**Goal:** Search by aliases and translations  
**Status:** COMPLETED

### 2.1 Alias Search ✅

- [x] **Parse aliases from CSV** (4th column, comma-separated in quotes)
- [x] **Include aliases in search**: Match `sole_female` when searching for `1girl`
- [x] **Display alias mapping**: Show `sole_female → 1girl`
- [x] **Insert canonical tag**, not alias

### 2.2 Translation Support

- [ ] **Translation file format**: CSV with `<english_tag>,<translated_tag>`
- [ ] **Search by translation**: Type in native language, find English tag
- [ ] **Show translation in results**: `1girl (一个女孩)`

---

## Phase 3: Advanced Features

**Goal:** Quality-of-life improvements  
**Estimated Effort:** 3-5 days

### 3.1 Frequency Sorting (High Priority) ✅

- [x] **Track tag usage** locally (localStorage)
- [x] **Sort frequently used tags higher** in results
- [x] **Logarithmic boost function**: Prevents over-boosting common tags
- [x] **Visual indicator**: ★ badge shows usage count
- [x] **Utility functions**: Reset, export, and view stats via console
- [x] **Persistent storage**: Survives browser restarts

### 3.2 Extra File (Custom Tags) ✅

- [x] **Support multiple tag files**: Load main + extra files
- [x] **Auto-load quality tags**: `extra-quality-tags.csv` loaded by default
- [x] **Merge results**: Combines tags from all sources
- [x] **Duplicate prevention**: Same tag won't appear twice

### 3.3 Chants (Prompt Presets)

- [ ] **JSON format**:
  ```json
  {
    "name": "HighQuality",
    "terms": "Best,Quality",
    "content": "(masterpiece, best quality, highres)",
    "color": 5
  }
  ```
- [ ] **Trigger character** (e.g., `@@` or custom): Show available chants
- [ ] **Insert full preset** on selection

### 3.4 Wiki Links

- [ ] **Optional `?` button** next to tags
- [ ] **Link to Danbooru/e621 wiki** for tag examples

### 3.5 Configuration UI

- [ ] **Settings panel** in ComfyUI for:
  - Tag file selection
  - Insert behavior (commas, spaces, underscore replacement)
  - Result count
  - Hotkey mapping
  - Color customization

---

## Phase 4: Wildcard Completion (Requires Parser Support)

**Goal:** Complete `__wildcard__` syntax  
**Estimated Effort:** 2-3 days  
**Blocked by:** Wildcard support needs to be added to the parser first

### 4.1 Trigger System

- [ ] **`__` trigger**: Start showing wildcard files after `__`
- [ ] **Two-stage completion**:
  1. First `__` shows available wildcard files
  2. Selecting a file shows contents (possible replacements)

### 4.2 Data Collection

- [ ] **Scan `data/wildcards/`** for `.txt` files
- [ ] **Parse wildcard files**: One option per line
- [ ] **Support nested folders**: `hair/colors/light.txt` → `__hair/colors/light__`

### 4.3 Insertion

- [ ] **File selection**: Insert `__filename__` for random selection
- [ ] **Direct selection**: Insert specific option from file contents

---

## Technical Architecture

### Backend Structure

```
api/
├── __init__.py
├── tokenize.py              # (existing) Token counter
└── autocomplete.py          # Tag autocomplete API
    ├── TagDatabase          # In-memory tag index
    ├── search_tags()        # Fuzzy prefix search
    ├── load_csv()           # CSV parser
    └── register_endpoints() # FastAPI/aiohttp routes

data/
├── tags/
│   ├── danbooru.csv         # Main tag database
│   ├── e621.csv             # Alternative database
│   ├── extra-quality-tags.csv
│   └── translations/        # Translation files
└── wildcards/
    ├── README.md
    └── *.txt                # User wildcard files
```

### Frontend Structure

```
js/
├── a1111_prompt.js          # (existing) Main extension entry
├── tokenCounter.js          # (existing) Token tooltip
├── autocomplete.js          # Autocomplete UI
│   ├── AutocompletePopup    # Popup DOM management
│   ├── ResultsList          # Results rendering
│   ├── KeyboardNav          # Arrow key navigation
│   └── insertTag()          # Text insertion logic
├── caretPosition.js         # Cursor position tracking
└── syntaxHighlight.js       # (placeholder) Syntax highlighting
```

### Data Flow

```
User Types → Debounce → Extract Word → API Call → Filter Results → Update Popup
     ↓                                                                  ↓
  Key Press ←←←←←←←←← Navigate ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←← Render List
     ↓
Tab/Enter → Insert Selected → Close Popup → Continue Typing
```

---

## CSV Tag Format Reference

```csv
<name>,<type>,<postCount>,"<aliases>"
1girl,0,4114588,"1girls,sole_female"
solo,0,3426446,"female_solo,solo_female"
highres,5,3008413,"high_res,high_resolution,hires"
```

| Column    | Description                              |
| --------- | ---------------------------------------- |
| name      | Tag text (underscores for spaces)        |
| type      | Category number (see below)              |
| postCount | Usage frequency (for sorting)            |
| aliases   | Comma-separated alternatives (in quotes) |

### Tag Types (Danbooru)

| Value | Category  | Color (Dark/Light)   |
| ----- | --------- | -------------------- |
| 0     | General   | lightblue/dodgerblue |
| 1     | Artist    | indianred/firebrick  |
| 3     | Copyright | violet/darkorchid    |
| 4     | Character | lightgreen/darkgreen |
| 5     | Meta      | orange/darkorange    |

---

## Implementation Priority

| Phase                  | Priority    | Effort   | Value                           | Status      |
| ---------------------- | ----------- | -------- | ------------------------------- | ----------- |
| Phase 1 (Core)         | 🔥 Critical | 3-5 days | High - Basic functionality      | ✅ Complete |
| Phase 2 (Aliases)      | ⚡ High     | 2-3 days | Medium - Better discoverability | ✅ Complete |
| Phase 3 (Advanced)     | 💡 Medium   | 3-5 days | Medium - QoL improvements       | 📋 Next     |
| Phase 4 (Wildcards)    | 💡 Low      | 2-3 days | Low - Blocked by parser support | ⏸️ Blocked  |

**Total Estimated Effort:** 10-15 days for full implementation

**Current Status:** Phase 1 & 2 complete! Autocomplete is fully functional with alias support.

> **Note:** Phase 4 (Wildcards) requires wildcard support to be added to the parser first.
> Phase 3 (Advanced Features) is now the next priority, especially frequency sorting.

---

## Key Differences from A1111

1. **No Gradio**: ComfyUI uses LiteGraph, so integration is different
2. **No shared.opts**: Settings storage needs custom solution
3. **File serving**: May need different approach for loading CSV/preview images
4. **Textareas**: ComfyUI widget textareas vs Gradio textareas
5. **Extension loading**: JS module loading differs from A1111
6. **No Extra Networks**: ComfyUI uses dedicated nodes for LoRAs/embeddings, not inline syntax

---

## Files to Port/Reference

| A1111 File                   | Purpose                     | Port Priority           |
| ---------------------------- | --------------------------- | ----------------------- |
| `tagAutocomplete.js` (68KB)  | Main logic                  | Extract core patterns   |
| `_caretPosition.js`          | Cursor position calculation | Port directly           |
| `_utils.js`                  | CSV parsing, debounce       | Port utilities          |
| `__globals.js`               | Shared state                | Adapt for our structure |
| `tag_autocomplete_helper.py` | Backend API                 | Reference for endpoints |
| `danbooru.csv`               | Tag database                | Include as data file    |

---

## Testing Checklist

- [ ] Basic word completion after comma
- [ ] Completion at start of prompt
- [ ] Completion inside parentheses
- [ ] Keyboard navigation (up/down/tab/enter/escape)
- [ ] Mouse click selection
- [ ] Underscore replacement
- [ ] Parenthesis escaping
- [ ] Alias search and insertion
- [ ] Wildcard completion with `__`
- [ ] Performance with large tag database (100k+ tags)
- [ ] Memory usage monitoring
- [ ] Dark/light mode colors

---

## Open Questions

1. **Settings Storage**: Where to store user preferences? (localStorage vs ComfyUI config)
2. **Tag File Distribution**: Bundle CSVs or download on first use? (Size: ~4MB each)
3. **Integration Point**: How to attach to ComfyUI textarea widgets?

---

## Next Steps

1. **Start Phase 1.1**: Create `TagDatabase` class with CSV loading
2. **Create API endpoint**: `/a1111_prompt/autocomplete`
3. **Build minimal popup**: Just show results, no styling
4. **Iterate on UX**: Add keyboard nav, styling, cursor tracking
5. **Add configuration**: User preferences for behavior

---

_Document created: 2026-01-21_  
_Based on analysis of a1111-sd-webui-tagcomplete v2025.x_
