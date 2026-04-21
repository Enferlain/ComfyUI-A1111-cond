# A1111 Prompt Node - TODO / Roadmap

## ✅ Recently Completed

### Hook System Refactor (NEW!)

- [x] Replaced MODEL input requirement with TransformerOptionsHook
- [x] Auto-detect step count from sampler's sample_sigmas
- [x] Simplified workflow: CLIP → Node → Sampler (no MODEL needed)
- [x] Step-based syntax now scales to actual sampler steps
- [x] Updated documentation and created migration guide

### Maintenance & Quality (2026-02-02)

- [x] **Memory Leak Fixes**:
  - [x] Implemented `onRemoved` cleanup for token counter and autocomplete popups
  - [x] Added `try...finally` for defensive DOM cleanup in coordinate calculations
  - [x] Localized autocomplete blur timeouts to node instances
- [x] **Memory Management**:
  - [x] Implemented instance-level prompt encoding cache
  - [x] Enforced cache limits during execution to prevent single-run spikes
  - [x] Added explicit tensor cleanup in hooks
- [x] **Performance**: Optimized BFS graph traversal using `collections.deque`
- [x] **API Robustness**:
  - [x] Added normalization for `extra_files` parameters
  - [x] Implemented status code checking for all frontend API calls
- [x] **Code Quality**: Removed dead code, optimized log strings, and addressed linter warnings
- [x] **Documentation**: Fixed README formatting/lints and documented hidden inputs

### Token Counter & Autocomplete

- [x] Token counter in node header (`45/75 | 32/75`)
- [x] BREAK-aware sequence counting
- [x] Real tokenization (no estimation)
- [x] Warning colors for long prompts (yellow 300+, red 450+ tokens)
- [x] Visual boundary markers (orange bars at 75-token boundaries)
- [x] BREAK position markers (blue bars)
- [x] Tag autocomplete with Danbooru/e621 databases
- [x] Frequency sorting with usage tracking
- [x] Theme support (respects ComfyUI color scheme)

### Wildcards (2026-04-21)

- [x] `__wildcard__` syntax expansion
- [x] Nested wildcards
- [x] Dynamic prompt syntax support inside wildcard content (`{a|b}`, `{1-2$$a|b|c}`, weighted picks)
- [x] Wildcard autocomplete when typing `__`
- [x] Wildcard folder + file suggestions with second-stage content drill-down
- [x] Wildcard prompt spacing cleanup after optional/empty expansions

---

## 🔥 High Priority

### Tag Autocomplete ✅

- [x] Autocomplete popup when typing (like A1111's tag autocomplete)
- [x] Support Danbooru/e621 tag databases
- [x] Show tag frequency/popularity
- [x] Keyboard navigation (↑/↓/Tab/Enter/Escape)
- [x] Mouse click selection
- [x] Alias support and display
- [x] Color-coded tags by type
- [x] Smart tag insertion with comma handling
- [x] Parenthesis escaping for A1111 compatibility
- [x] Frequency sorting with usage tracking
- [x] Multiple tag file support (main + extra files)
- [x] Auto-load quality tags
- [x] Theme support (respects ComfyUI color scheme)
- [x] Custom tag lists (user-defined) - partially done (CSV support)
- [ ] Configuration UI for tag file selection
- [ ] Chants/prompt presets (prompt library/bookmarks)
- [ ] Wiki links for tag documentation
- [ ] Source-aware autocomplete results
  - Load multiple tag sources without flattening away source identity
  - Preserve per-source aliases instead of using one global alias map
  - Show prepended source badge/icon in the popup (e.g. Danbooru, e621)
  - Handle duplicate canonical tags by showing one result per source when needed
- [ ] Check if usage count is recorded properly
- [ ] Visual distinction between different databases (e.g., Danbooru vs e621 tags)
  - Consider: badge/icon, subtle background color, or source indicator
  - Useful when loading multiple databases simultaneously
  - Should be subtle to not clutter the UI

### Wildcard Support

- [x] `__wildcard__` syntax expansion
- [x] Nested wildcards
- [ ] Wildcard file browser/picker
  - Current state: autocomplete can browse wildcard folders/files and drill into file contents
- [ ] Preview what wildcards will expand to
- [ ] Toggle for static wildcards (same seed same wildcard result)
- [ ] Toggle for history for what wildcards have been used so you don't get the same one too soon

---

## 📊 Token Counter Enhancements

### Visual Boundaries

- [x] Show where 75-token boundary falls inside the text
- [x] Subtle visual marker (vertical bar overlay)
- [x] API returns character positions of boundaries

### Display Improvements

- [x] Show total tokens: `45/75 | 32/75 (77 total)`
- [x] Warning colors for long prompts (yellow 300+, red 450+ tokens)
- [x] Tooltip with detailed breakdown (click to open)

---

## ✨ Syntax Features

### Syntax Highlighting (low priority)

- [ ] Color-code `[scheduling:syntax:when]`
- [ ] Color-code `(emphasis:1.2)`
- [ ] Color-code `[A|B|C]` alternation
- [ ] Custom textarea with overlay (complex)

### Embedding Support (low priority)

- [ ] Warn if `embedding:name` doesn't exist
- [ ] Autocomplete for available embeddings
- [ ] Show embedding token count

### Schedule Preview (low priority)

- [ ] Tooltip showing what prompt looks like at different steps
- [ ] Interactive slider to preview step-by-step changes
- [ ] Useful for `[from:to:when]` and `[A|B]` alternation

---

## 🛠️ Quality of Life

### Prompt Library

- [ ] Save/load prompt snippets
- [ ] Favorites/categories
- [ ] Quick insert from library

### Negative Prompt Node

- [x] Dedicated negative prompt variant
- [x] Shared syntax highlighting/autocomplete

### Prompt Macros

- [ ] Define reusable prompt fragments
- [ ] `{macro_name}` expansion

### Technical Debt & Refinement

- [ ] Standardize null-checking patterns across all JS files (prefer `?.`)
- [x] Improve blur handling race condition in autocomplete (timeout/clearTimeout pattern)
- [ ] Add unit tests for graph traversal logic (`_get_downstream_steps`)
- [ ] Add JSDoc comments to all public functions in `autocomplete.js`
- [ ] Standardize error response formatting across all backend API endpoints
- [ ] Implement LRU (Least Recently Used) caching policy in `nodes/prompt_node.py` (improve current "clear full" strategy)
- [ ] Add query validation (max length) to autocomplete API endpoint
- [ ] Refine CUDA cache clearing (smarter fragmentation detection / configurable threshold)
- [ ] Modularize `js/autocomplete.js` (separate coordinates, sorting, and UI logic)
- [ ] Complete CSS variable conversion for all UI elements (Themes support)
- [ ] Add rate limiting/throttling to autocomplete API endpoint
- [ ] Refactor `model_function_wrapper` chaining into a reusable utility

---

## ✅ Code Organization (Completed)

### File Structure Refactor

- [x] Split `nodes.py` → `nodes/prompt_node.py` + `nodes/negative_node.py`
- [x] Split `parser.py` → `parser/grammar.py` + `parser/scheduler.py` + `parser/wildcards.py`
- [x] Split `api.py` → `api/tokenize.py` + `api/autocomplete.py`
- [x] Split `a1111_prompt.js` → Added `tokenCounter.js` + `autocomplete.js` + `syntaxHighlight.js`
- [x] Created `data/tags/` and `data/wildcards/` directories

```
A1111_Prompt_Node/
├── __init__.py
├── nodes/
│   ├── __init__.py
│   ├── prompt_node.py      # Main node ✓
│   └── negative_node.py    # Negative variant ✓
├── parser/
│   ├── __init__.py
│   ├── grammar.py          # Lark grammar ✓
│   ├── scheduler.py        # Step scheduling ✓
│   └── wildcards.py        # Wildcard expansion (placeholder) ✓
├── api/
│   ├── __init__.py
│   ├── tokenize.py         # Token counter endpoint ✓
│   └── autocomplete.py     # Tag autocomplete endpoint (placeholder) ✓
├── js/
│   ├── a1111_prompt.js     # Main extension ✓
│   ├── tokenCounter.js     # Token counter UI ✓
│   ├── autocomplete.js     # Autocomplete UI (placeholder) ✓
│   └── syntaxHighlight.js  # Syntax highlighting (placeholder) ✓
├── data/
│   ├── tags/               # Tag databases ✓
│   └── wildcards/          # Default wildcards ✓
└── hooks.py                # ComfyUI hooks ✓
```

> **Note:** Old files (`nodes.py`, `parser.py`, `api.py`) can be safely deleted after verifying the new structure works.

---

## 💡 Ideas (Maybe Later)

- [ ] LoRA weight syntax `<lora:name:weight>`
- [ ] Regional prompting support
- [ ] Prompt diff viewer (compare two prompts)
- [ ] Import from A1111 PNG metadata
- [ ] Export to A1111-compatible format
- [ ] Prompt history (undo/redo)
- [ ] Multi-line prompt editor (full-screen mode)
- [ ] Wildcard/tags having fast/performant thumbnails on hover in autocomplete.js (needs autocompelte.js package split)
