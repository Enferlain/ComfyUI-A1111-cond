# A1111 Prompt Node - TODO / Roadmap

## ✅ Completed

- [x] Token counter in node header (`45/75 | 32/75`)
- [x] BREAK-aware sequence counting
- [x] Real tokenization (no estimation)
- [x] Warning colors for long prompts (yellow 300+, red 450+ tokens)
- [x] Visual boundary markers (orange bars at 75-token boundaries)
- [x] BREAK position markers (blue bars)

---

## 🔥 High Priority

### Tag Autocomplete

- [ ] Autocomplete popup when typing (like A1111's tag autocomplete)
- [ ] Support Danbooru/e621 tag databases
- [ ] Custom tag lists (user-defined)
- [ ] Show tag frequency/popularity

### Wildcard Support

- [ ] `__wildcard__` syntax expansion
- [ ] Nested wildcards
- [ ] Wildcard file browser/picker
- [ ] Preview what wildcards will expand to

---

## 📊 Token Counter Enhancements

### Visual Boundaries

- [x] Show where 75-token boundary falls inside the text
- [x] Subtle visual marker (vertical bar overlay)
- [x] API returns character positions of boundaries

### Display Improvements

- [ ] Show total tokens: `45/75 | 32/75 (77 total)`
- [x] Warning colors for long prompts (yellow 300+, red 450+ tokens)
- [ ] Tooltip with detailed breakdown on hover

---

## ✨ Syntax Features

### Syntax Highlighting

- [ ] Color-code `[scheduling:syntax:when]`
- [ ] Color-code `(emphasis:1.2)`
- [ ] Color-code `[A|B|C]` alternation
- [ ] Custom textarea with overlay (complex)

### Embedding Support

- [ ] Warn if `embedding:name` doesn't exist
- [ ] Autocomplete for available embeddings
- [ ] Show embedding token count

### Schedule Preview

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

- [ ] Dedicated negative prompt variant
- [ ] Shared syntax highlighting/autocomplete

### Prompt Macros

- [ ] Define reusable prompt fragments
- [ ] `{macro_name}` expansion

---

## 📁 Code Organization

### File Structure Refactor

```
A1111_Prompt_Node/
├── __init__.py
├── nodes/
│   ├── __init__.py
│   ├── prompt_node.py      # Main node
│   └── negative_node.py    # Negative variant
├── parser/
│   ├── __init__.py
│   ├── grammar.py          # Lark grammar
│   ├── scheduler.py        # Step scheduling
│   └── wildcards.py        # Wildcard expansion
├── api/
│   ├── __init__.py
│   ├── tokenize.py         # Token counter endpoint
│   └── autocomplete.py     # Tag autocomplete endpoint
├── js/
│   ├── a1111_prompt.js     # Main extension
│   ├── tokenCounter.js     # Token counter UI
│   ├── autocomplete.js     # Autocomplete UI
│   └── syntaxHighlight.js  # Syntax highlighting
├── data/
│   ├── tags/               # Tag databases
│   └── wildcards/          # Default wildcards
└── hooks.py                # ComfyUI hooks
```

---

## 💡 Ideas (Maybe Later)

- [ ] LoRA weight syntax `<lora:name:weight>`
- [ ] Regional prompting support
- [ ] Prompt diff viewer (compare two prompts)
- [ ] Import from A1111 PNG metadata
- [ ] Export to A1111-compatible format
- [ ] Prompt history (undo/redo)
- [ ] Multi-line prompt editor (full-screen mode)
