# Everyday Usage Issue Findings

Investigation date: 2026-06-15

Scope: usage issues reported from day-to-day ComfyUI use. This file records findings and likely fix directions only; no behavior changes are included here.

## Summary

The remembered issues cluster around four areas:

- Frontend lifecycle/state resilience in `js/autocomplete.js` and `js/a1111_prompt.js`.
- Autocomplete latency and ranking in `js/autocomplete.js` plus `/a1111_prompt/autocomplete`.
- Token counter limitations in `api/tokenize.py` and the boundary overlay in `js/a1111_prompt.js`.
- Wildcard expansion metadata/UX gaps between the Python node result and ComfyUI's saved workflow metadata.

## 1. Custom node JS sometimes stops until page refresh

Observed behavior:

- Autocomplete dropdowns stop working.
- Sequence boundary markers stop working.
- Effective prompt preview stops working until refreshing the page.

Likely causes:

- The frontend extensions keep several module-global variables for autocomplete state: `autocompletePopup`, `currentTextarea`, `currentWordInfo`, `selectedIndex`, and `currentResults` in `js/autocomplete.js`.
- `hideAutocompletePopup()` clears `currentTextarea` and `currentWordInfo` globally. If a stale request resolves after focus/node changes, state can be reset or pointed at the wrong textarea.
- `scheduleAutocomplete()` uses a per-node `requestVersion`, but the popup and selected result state are global, so multiple A1111 nodes can still compete for the same popup state.
- Both autocomplete and token boundary overlays depend on `textWidget.inputEl` staying valid. ComfyUI can recreate widget DOM, and the code has a rebinding path for token boundaries but autocomplete only binds once after `requestAnimationFrame(waitForTextarea)`.
- `js/a1111_prompt.js` creates and destroys overlay DOM. `ensureOverlay()` can recreate it when disconnected, but only after relevant calls. If an exception happens while rendering marker HTML or positioning the popup, there is no extension-level recovery path.

Evidence:

- `js/autocomplete.js`: global state at lines 14-19.
- `js/autocomplete.js`: one-time textarea binding starts around lines 881-1014.
- `js/autocomplete.js`: global popup reset in `hideAutocompletePopup()` around lines 541-548.
- `js/a1111_prompt.js`: overlay lifecycle around lines 324-361 and `ensureOverlay()` around lines 401-417.
- `CHANGELOG.md` notes an earlier frontend stability fix for duplicate `nodeCreated` hooks, which suggests this area has already been a source of breakage.

Suggested later fix:

- Move autocomplete state into a per-textarea/per-node controller object and let the popup be only a view.
- On every focus/input, verify the bound textarea is still connected and still belongs to the current node.
- Add guarded error logging around autocomplete render, caret measurement, marker render, and prompt preview render.
- Consider a small self-healing path: if popup/overlay DOM is missing, stale, or disconnected, rebuild and rebind instead of silently doing nothing.

## 2. Dropdowns feel sluggish

Observed behavior:

- Dropdown is intended to appear after the second keystroke, but sometimes feels one beat behind.

Likely causes:

- Autocomplete has a fixed 100 ms debounce before even starting the fetch.
- The request then waits for `/a1111_prompt/autocomplete`.
- For wildcard searches, frontend raises the requested limit to at least 200. Backend wildcard search scans/scores entries in Python over all wildcard entries.
- For tag search, backend `TagDatabase.search()` linearly scans all loaded tags and aliases on each request. With large merged tag files this can feel slower than a prefix index.
- The frontend discards stale results if the word has changed, which is correct, but with debounce plus network latency the visible popup can lag behind fast typing.

Evidence:

- `js/autocomplete.js`: `scheduleAutocomplete(delay = 100)` around lines 893-944.
- `js/autocomplete.js`: `fetchTagSuggestions()` starts around line 698 and requests at least 200 results for wildcard modes around line 702.
- `api/autocomplete.py`: `TagDatabase.search()` scans `self._tags` and `_alias_map`.
- `api/autocomplete.py`: `WildcardDatabase.search()` scans `_entries`.

Suggested later fix:

- Reduce debounce for local typing, e.g. 40-60 ms, or issue immediately on the first eligible query and debounce subsequent edits.
- Cache recent query responses client-side by mode/query/content wildcard.
- Add a backend prefix index for tags and wildcard names, or at least separate prefix results from contains results to avoid full scans before first paint.
- Render cached/stale prefix results immediately while the fresh request is in flight.

## 3. Frequency stars do not always bring frequent items to the top

Observed behavior:

- Frequency is recorded and displayed, but frequent items do not always rank first for the entered letters.

Likely cause:

- This is currently expected from the sorting model. Relevance outranks usage. Exact matches sort above prefix matches, prefix matches above contains matches, and only then usage score is applied.
- Usage score is logarithmic plus a small recency boost, so large post-count differences can still dominate when usage scores are close.
- Usage keying uses `tag.name`. For alias matches, selecting via an alias increments the canonical tag name, not the alias text. That is usually sensible, but it can feel wrong when typing the alias again.
- Wildcard content entries use the content line as `tag.name`; wildcard files use the path as `tag.name`. Frequency is therefore path-sensitive for wildcard files and not leaf-name-sensitive.

Evidence:

- `js/autocomplete.js`: `getUsageScore()` uses `Math.log2(count + 1)` plus recency around lines 108-120.
- `js/autocomplete.js`: `sortByFrequency()` sorts by relevance first, usage second, post count third around lines 506-535.
- `js/autocomplete.js`: `incrementTagFrequency(tag.name)` on insert around line 591.
- `CHANGELOG.md` states frequency sorting intentionally keeps match quality ahead of usage history.

Suggested later fix:

- Decide intended behavior:
  - "Usage always wins within the current filtered list", or
  - "Usage only breaks ties within relevance buckets" (current behavior).
- If usage should feel stronger, move usage above post count and possibly above relevance for prefix/contains matches.
- Store alias usage and canonical usage separately, or record both.
- For wildcard files, consider recording both full path and basename so leaf-name usage can influence nested wildcard ranking.

## 4. Token counter does not count maximum possible wildcard tokens

Observed behavior:

- A selected wildcard's token count reflects the literal wildcard syntax, not the longest possible expansion.

Likely cause:

- The live token counter endpoint only strips A1111 emphasis/scheduling syntax. It does not inspect wildcard files.
- `parser/wildcards.py` has `get_wildcard_options()` and `expand_wildcards()`, but tokenization does not call either.
- `expand_wildcards()` chooses a random option, which is wrong for "maximum possible tokens". Token counter would need a deterministic worst-case wildcard analysis.
- Dynamic prompt syntax inside wildcard files can contain ranges and weighted choices, which complicates worst-case counting further.

Evidence:

- `api/tokenize.py`: `strip_a1111_syntax()` handles emphasis, scheduling, and alternation, not wildcards.
- `api/tokenize.py`: `tokenize_prompt()` tokenizes `clean_segment` after stripping syntax.
- `parser/wildcards.py`: `expand_wildcards()` uses `chooser.choice(options)`.
- `parser/wildcards.py`: `get_wildcard_options()` returns the raw choices from the wildcard file.

Suggested later fix:

- Add a deterministic "wildcard max expansion" helper separate from `expand_wildcards()`.
- For each `__wildcard__`, load options and choose the option with the highest token count after recursively resolving nested wildcards up to the existing depth limit.
- For dynamic prompt choices, choose the maximum quantity and longest/highest-token options.
- Return token response metadata that marks counts as "max wildcard estimate" vs actual random expansion.

## 5. Fulfilled wildcard prompt is not reflected in saved image metadata

Observed behavior:

- The node preview shows the expanded prompt, but saved metadata shows the wildcard syntax that was typed.

Likely cause:

- Python node returns `{"ui": {"text": [expanded_text]}}`, and frontend `onExecuted` uses that to populate `_prompt_display`.
- The serialized workflow/input metadata still uses the original `text` widget value.
- `_prompt_display` is explicitly set `serialize = false`, so it is only a UI preview and will not be included in the saved workflow metadata.

Evidence:

- `nodes/prompt_node.py`: expands wildcards at line 88 and returns `ui.text` around lines 275-278 and line 419.
- `nodes/negative_node.py`: same pattern around lines 63 and 112-115.
- `js/a1111_prompt.js`: effective prompt display is populated from `onExecuted(message?.text)` around lines 95-100.
- `js/a1111_prompt.js`: `_prompt_display` has `serialize = false` around line 84.

Suggested later fix:

- Decide how metadata should expose both values:
  - Preserve original workflow input for reproducibility and add an additional `a1111_resolved_prompt` field.
  - Or optionally overwrite image metadata prompt text with the resolved prompt while keeping workflow unchanged.
- Investigate ComfyUI metadata hooks/custom node APIs for adding extra PNG info from a node execution.
- Avoid relying on `_prompt_display` for metadata; it is intentionally UI-only.

## 6. Wildcards show/insert full paths instead of only selected leaf name

### This is probably fine as it is

Observed behavior:

- Selecting a nested wildcard inserts `__/thing/here/a__` instead of `__a__`.

Likely cause:

- Backend wildcard entries use the relative path as `name`.
- `WildcardEntry.to_dict()` sets `completion` to `__{self.name}__`, so nested files intentionally complete to their full relative path.
- The UI renders nested paths with colored path segments, but insertion still uses `tag.completion || tag.name`.

Evidence:

- `api/autocomplete.py`: `WildcardEntry.to_dict()` sets `completion` at line 135.
- `api/autocomplete.py`: wildcard file names come from `relative.as_posix()` around lines 346-349.
- `js/autocomplete.js`: wildcard display uses `appendWildcardPathLabel()` around lines 136-155 and 437-440.
- `js/autocomplete.js`: insertion uses `tag.completion || tag.name` around line 596.

Important caveat:

- Full paths are needed for disambiguation if multiple wildcard files share the same basename.
- The Python resolver can find wildcard files by normalized full path and also falls back to scanning, but leaf-only insertion may become ambiguous.

Suggested later fix:

- Add separate backend fields: `name` full path, `leaf_name`, `completion`, and maybe `display_path`.
- Offer a setting or insertion mode:
  - Insert full path for unambiguous reproducibility.
  - Insert leaf name when unique.
  - Insert leaf name with a warning/fallback when duplicates exist.

## 7. Sequence end marker slices words/tokens at boundaries

Observed behavior:

- Boundary marker can appear inside a word/token instead of clearly showing whether the word goes before or after the sequence break.

Likely causes:

- Backend boundary positions are based on the word that triggers overflow, but the frontend marker renderer inserts a zero-width span at the raw character position.
- If the position is inside a tokenized word, escaped syntax, punctuation, or text transformed by `strip_a1111_syntax()`, the mirror HTML visually splits the word.
- The current code does not render "word belongs to previous/next chunk" as a semantic marker. It only inserts a marker at a character offset.
- Recent backend changes map stripped words back to original positions, which should help syntax-stripped prompts, but the UI still renders by slicing raw text at `char_pos`.

Evidence:

- `api/tokenize.py`: boundary is set to `original_word_start` when a word would exceed the chunk around lines 233-244.
- `js/a1111_prompt.js`: `updateBoundaryMarkers()` slices raw text at `boundary.char_pos` and inserts a zero-width marker around lines 537-576.
- `api/tokenize.py`: token counting is word-by-word, while the actual tokenizer can split one visible word into multiple tokens.

Suggested later fix:

- Return richer boundary data from `/a1111_prompt/tokenize`, e.g. `word_start`, `word_end`, `overflow_word`, `placed_in_chunk`, and `previous_chunk`.
- Render a marker before the whole word that moves to the next chunk instead of slicing by raw character offset.
- Optionally highlight the overflow word rather than inserting an inline zero-width marker.
- For extremely long single words that exceed 75 tokens by themselves, show a distinct "single word spans chunks" state.

## Cross-cutting follow-up ideas

- Add a small frontend debug flag that logs extension binding, textarea rebinding, popup open/close, request version drops, token fetch errors, and overlay recreation.
- Add browser-side tests or a lightweight Playwright smoke test for:
  - Type two letters and see autocomplete.
  - Select a wildcard file/content.
  - Type enough tokens to force a boundary.
  - Execute a node and verify prompt preview appears.
- Split `js/autocomplete.js` into smaller controller/view/storage modules before deeper fixes. The current file mixes persistence, scoring, positioning, rendering, network calls, and node lifecycle state.

## 8. Emphasis syntax changes tokenization on Qwen-family encoders (2026-09-17)

Observed behavior:

- On the Anima text encoder, writing `a (cat:1.3) sitting` produces a different token id sequence than `a cat sitting` even with all weights neutralized: the tagged word tokenizes as a standalone fragment.
- Verified with the bundled `qwen_3_06b_base` TE via `research/te_harness_anima.py`: cond seq 7 (plain) vs 8 (weighted), and direct tokenizer comparison shows `cat` inside text → id 8251 (`Ġcat`) but standalone `cat` → id 4616 (`cat`).

Verified cause:

- Comfy's `token_weights()` (comfy/sd1_clip.py) splits the prompt at emphasis boundaries and tokenizes each fragment separately. `AnimaTokenizer` (comfy/text_encoders/anima.py) does NOT pass `disable_weights=True` to its qwen3_06b branch (weights are reset to 1.0 afterwards), so the qwen branch goes through fragment-split tokenization.
- GPT-style BPE vocabularies (Qwen) encode word starts with a `Ġ` marker and add no prefix on standalone calls, so a mid-sentence fragment loses its word-start marker. Verified per-family behavior:
  - SD1.5 CLIP (`</w>` word-end markers): fragment path and single-pass tokenization produce identical ids. No effect.
  - T5 sentencepiece (dummy `▁` prefix added per tokenizer call): standalone `cat` → `▁cat`, matching in-context form. Effectively lossless. This is also why the Anima `t5xxl_weights` channel aligns exactly with tagged tokens — alignment currently relies on the tokenizer's dummy-prefix behavior.
  - Qwen BPE: not lossless; ids change for tagged words. Any GPT-style BPE TE without a dummy prefix behaves the same.
- A1111 reference behavior: `parse_prompt_attention` (vendored copy in `context_files_from_a1111/parsing.py`) tokenizes the full text once and applies weights by character ranges, so emphasis never changes tokenization there.

Impact assessment:

- SD1.5/SDXL: none (lossless there).
- Anima: real semantic difference — the encoded cond differs from an unweighted encode even before weights are considered. Whether this is visible in output is untested.

Suggested later fix (not decided):

- Option A (A1111 parity): single-pass tokenization with weights derived from character ranges; requires reworking how weights are attached, and interacts with `embedding:` handling.
- Option B (targeted): on word-start-marker vocabularies, tokenize fragments with an explicit leading word-start marker (or tokenize with a leading space and reconcile the extra token) so fragment ids match in-context ids.
- Either way, decide whether to preserve current Comfy behavior for CLIP-family (identical output either way) and only special-case BPE-start-marker vocabularies.

## 9. `api/tokenize.py` crashes any headless import (2026-09-17)

Observed behavior:

- Importing the node package outside a running ComfyUI server (CI, unit tests, research scripts) fails with `AttributeError: 'NoneType' object has no attribute 'instance'` raised from `api/tokenize.py` line 301.

Verified cause:

- `api/tokenize.py` uses an unguarded module-level route decorator: `@server.PromptServer.instance.routes.post("/a1111_prompt/tokenize")`. `PromptServer.instance` only exists once the server is live.
- `api/autocomplete.py` handles the same situation correctly (`if _HAS_SERVER and PromptServer:` guard around registration).

Suggested later fix:

- Wrap the route registration in `api/tokenize.py` the same way `api/autocomplete.py` does, so the module imports cleanly without a server. No behavior change in a live server.
