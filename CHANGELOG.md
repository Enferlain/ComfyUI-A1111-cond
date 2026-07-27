# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Rules:
- Use proper sub titles "Added", "Changed", "Removed" and "Fixed"
- Keep proper track of days for where entries should go
- Be concise but mention all changes without necessarily detailing each one

## [Unreleased] - 2026-07-25

### Fixed

- Fixed the resolved prompt toggle occasionally rendering outside the node and becoming unclickable by using ComfyUI's current widget visibility and layout APIs.
- Kept the last resolved prompt preview visible while editing the prompt, replacing it only after the next execution.
- Made wildcard autocomplete deterministic by loading the complete path manifest once and filtering it locally, eliminating delayed and incomplete cached-prefix results.
- Reused the startup wildcard path catalog during prompt expansion, eliminating repeated full-tree scans for concise unique wildcard names.

## [Unreleased] - 2026-07-23

### Fixed

- Fixed resolved prompt preview state resetting after each generation; the preview now stays expanded or collapsed according to the user's last toggle.
- Hardened resolved prompt preview rendering so preview widget failures do not break autocomplete or the rest of the node frontend.

## [Unreleased] - 2026-06-15

### Added

- **Autocomplete Response Cache**: Added a client-side cache for repeated and prefix-refined autocomplete queries.
- **Autocomplete Prefix Index**: Added backend prefix indexing for tag names and aliases to speed up large tag database searches.
- **Wildcard Token Estimates**: Added deterministic longest-option wildcard and dynamic prompt expansion for live token counting.
- **Prompt PNG Metadata**: Added an `a1111_prompt_node` PNG metadata entry that records each A1111 prompt node's original and resolved prompt by node id.

### Changed

- **Autocomplete Responsiveness**: Reduced autocomplete debounce from 100 ms to 50 ms and shows cached prefix matches while fresh results load.
- **Autocomplete Ranking**: Frequent and recently selected entries now have stronger ranking influence while exact matches remain prioritized.
- **Effective Prompt Restore**: The resolved prompt preview is now stored with the workflow and shown behind a compact toggle so dragged-back images can reveal the prompt that was actually encoded without shrinking the main prompt field.
- **Wildcard Completion Text**: Nested wildcard autocomplete now inserts the leaf name when it is unambiguous, falling back to the full path only for duplicate leaf names.
- **Autocomplete Warm-Up**: The frontend now starts loading the tag database before the first typed autocomplete request, and contains searches use lazy candidate indexing to preserve substring results without repeated full-database scans.

### Fixed

- Fixed autocomplete lifecycle rebinding when ComfyUI recreates prompt textarea DOM.
- Fixed autocomplete API coercion for unusual query, limit, boolean, file, and request-body values.
- Fixed token boundary positions after A1111 syntax stripping so markers align better with original prompt text.
- Fixed token counter route registration/import resilience after adding wildcard token estimates.
- Fixed literal `__wildcard__` text being counted instead of the maximum estimated wildcard expansion in the token counter.
- Fixed wildcard-only parser imports requiring Lark when scheduler parsing is not being used.
- Fixed dragged-back generated images not showing the resolved prompt preview by writing restore data into the embedded workflow metadata during node execution.
- Fixed resolved prompt preview collapse/expand layout so the preview stays inside the node bounds after toggling.
- Fixed token sequence boundary markers disappearing for wildcard-estimated counts and later sequence cutoffs.
- Hardened resolved prompt preview toggles so restored or rebuilt widgets keep a working click handler and stray input events do not collapse an unchanged preview.
- Fixed scheduled prompt padding being written back into the global encoded prompt cache.

## [Unreleased] - 2026-04-22

### Added

- **Wildcard Expansion**: Added A1111-style `__wildcard__` expansion for prompt and negative prompt nodes, including nested wildcard resolution from `data/wildcards`.
- **Dynamic Prompt Support in Wildcards**: Added brace-style choice expansion inside wildcard content, including simple choices (`{a|b}`), ranged picks (`{1-2$$a|b|c}`), and weighted picks (`{20%a|b}`).
- **Wildcard Autocomplete**: Added wildcard-aware autocomplete when typing `__`, with support for wildcard files, inferred folders, and second-stage file-content suggestions.
- **Wildcard Documentation**: Updated README, TODO, and changelog notes to cover wildcard expansion, autocomplete flow, and current completion status.

### Changed

- **Expanded Prompt Preview**: Effective prompt output now reflects resolved wildcard expansion so executed prompt text is visible after generation.
- **Wildcard Browsing UX**: Wildcard autocomplete now keeps the popup open after selecting a wildcard file and drills into either descendant entries or file contents.
- **Wildcard Result Rendering**: Improved wildcard result styling in the autocomplete popup to make nested paths easier to scan.
- **Prompt Cleanup**: Wildcard and dynamic prompt expansion now normalizes leftover spacing from optional empty branches so final prompts do not accumulate doubled spaces or messy comma spacing.
- **Autocomplete Ranking**: Frequency sorting now uses a blended relevance model that keeps exact/prefix/contains match quality ahead of usage history, with logarithmic frequency scaling and a light recency boost.

### Fixed

- Fixed wildcard autocomplete result limits so large wildcard packs return substantially more matching entries.
- Fixed wildcard expansion to resolve a1111-style wildcard packs that combine nested wildcard files with dynamic prompt syntax.
- Fixed wildcard autocomplete usage stars being display-only; usage history now actually influences wildcard result ordering.

## [Unreleased] - 2026-04-03

### Added

- **API Normalization**: The `autocomplete` API now robustly handles the `extra_files` parameter, coercing strings to lists and providing sensible defaults.
- **Memory Management Refinements**:
  - **Bounded Caching**: Implemented instance-level prompt encoding cache with strict enforcement during execution to prevent unbounded growth.
  - **Internal Cleanup**: Added explicit tensor deletion in `hooks.py` to prevent accumulation.
  - **Leak Prevention**: Refactored frontend extensions to properly disconnect `ResizeObserver` instances and remove global click listeners upon node removal.

### Fixed

- **Hotfixes & Stability**:
  - **Backend**: Restored missing `transformers_dict` initialization in `hooks.py` to fix `AttributeError` crash.
  - **Frontend**: Merged duplicate `nodeCreated` hooks in `autocomplete.js` to restore extension functionality.

### Changed

- **Performance Optimization**:
  - **Shared Caching**: Implemented a structural result cache in `hooks.py` that persists across hook clones, critical for 2nd order samplers (DPM++ 2M, etc.).
  - **GPU Pre-Allocation**: Moved all embeddings to the GPU during the encoding phase to eliminate device-transfer latency.
  - **Zero-Allocation Hot-Path**: Reduced Python overhead to the absolute minimum for standard and scheduled runs.
- **Log Management**: Silenced all verbose console logs (encoding progress, graph traversal, and hook execution) behind the `debug` flag.
- **Stability & Determinism**: Fixed a regression in the identity check logic that caused intermittent "skipped swaps" and ensured perfect image consistency across all sampler types.
- **Code Quality & PR Feedback**:
  - **Exception Handling**: Moved verbose step-detection errors to a dedicated `StepCountRequiredError` class.
  - **Memory Optimization**: Limited tensor padding to only the prompts used in the current generation, preventing global cache bloat.
  - **Schedule Precision**: Implemented linear index scaling in `hooks.py` to ensure percentage-based schedules remain accurate even when sampler step counts vary from defaults.
  - **Cache Key Robustness**: Updated encoding cache to use composite keys (CLIP identity + normalization), ensuring embeddings are never reused in incompatible contexts.
  - **Improved Cache Management**: Refactored cache clearing to happen before execution loops, preventing data loss during multi-prompt runs.
  - **UI Robustness**: Improved autocomplete error handling to correctly reset internal selection state upon backend failure.

## [1.0.0] - 2026-02-02

### 🎉 Launch!

Stable release on the ComfyUI Registry with full parity verification and security hardening.

### Added

- **Comprehensive Test Suite**: 5 new test categories verifying the scheduler, hooks, node math, tokenizer, and API security.
- **GitHub Registry CI**: Added automated publication workflow for ComfyUI Registry.
- **Project Metadata**: Full `pyproject.toml` and `requirements.txt` integration for standardized distribution.

### Security

- **API Hardening**: Implemented a 255-character query length limit for the `autocomplete` API to prevent long-query DOS attempts.

### Changed

- **Verified A1111 Parity**:
  - Verified `BREAK` isolation logic ensures zero concept bleeding between segments.
  - Confirmed `_apply_direct_scaling` math exactly match A1111 emphasis (multiplication vs interpolation).
  - Verified `normalization` accurately implements A1111 "Mean Rescaling" (Norm vs No Norm).
- **Hooks Scaling**: Improved hook sigma-to-step conversion for better accuracy in low-step (2-4 steps) generations.

### Refinements

- **Memory Management**:
  - Bounded caching for prompt encoding to prevent memory leak on high-volume runs.
  - Explicit tensor cleanup in hook executions.
  - Frontend lifecycle management for `ResizeObserver` and global listeners.
- **Performance**:
  - GPU pre-allocation for per-step embeddings.
  - Shared result caching for cloned hooks in 2nd order samplers.
  - Zero-allocation hot-path for sampling loop swapping.

## [Unreleased] - 2026-01-31

### Added

- Extended JS frontend support to the Negative Prompt Node:
  - **Autocomplete**: Enabled tag autocomplete for negative prompts.
  - **Token Counter**: Header display and detailed tooltip now show for negative nodes.
  - **Sequence Visualizers**: Added blue `BREAK` markers and orange 75-token chunk markers to the negative prompt textarea.

### Changed

- **Smart "Effective Prompt" Visibility**: The expanded prompt view now automatically hides itself when it's identical to the user-typed prompt (e.g., when no TIPO expansion or wildcards are active).
- **Refined Autocomplete Behavior**: The autocomplete popup now suppresses itself when deleting separator characters (comma, space, brackets) via Backspace, preventing unnecessary popups on already "finished" tags.

### Fixed

- Fixed Negative Prompt node missing UI feedback functionality that was present in the positive node.

---

## [Unreleased] - Pre 2026/1/31 changes

### 🎉 Major Changes

#### Removed MODEL Input Requirement

The node now uses ComfyUI's `TransformerOptionsHook` system instead of requiring a MODEL input. This makes workflows simpler and more intuitive.

**Before:**

```text
MODEL ──► A1111 Prompt ──► MODEL ──► Sampler
CLIP  ──►                ──► COND  ──►
```

**After:**

```text
CLIP ──► A1111 Prompt ──► COND ──► Sampler
```

#### Automatic Step Detection

The node now automatically detects the step count from connected sampler/scheduler nodes by inspecting the workflow graph. No manual configuration needed!

### ✨ New Features

- **Auto-detect step count**: Steps are automatically extracted from downstream sampler/scheduler nodes
- **Simpler workflows**: No MODEL connection needed for scheduling/alternation
- **Smart validation**: Detects if you're using step-based syntax and validates accordingly
- **Works with custom nodes**: Uses generic detection (checks for "steps" input + "Scheduler"/"Sampler" in class name) instead of hardcoded node types

### 🔧 Technical Changes

- Replaced `StepConditioningHandler` with `A1111StepConditioningHook(TransformerOptionsHook)`
- Hook is now attached to conditioning output instead of model
- Added workflow graph traversal via hidden "PROMPT" and "UNIQUE_ID" inputs
- Removed `setup_step_conditioning_on_model()` function
- Updated `create_step_schedule_cond()` to use hook system
- Added `_get_downstream_steps()` for graph traversal
- Added `_uses_step_based_syntax()` for syntax detection

### 📝 Documentation Updates

- Updated README.md with new workflow examples and step detection behavior
- Created MIGRATION_GUIDE.md for users updating existing workflows
- Created MODEL_WRAPPER_ANALYSIS.md explaining the technical details
- Updated TODO.md with completed items

### ⚠️ Breaking Changes

**Node Signature Changed:**

- **Removed**: `model` input (optional)
- **Removed**: `model` output
- **Added**: Hidden `prompt` and `unique_id` inputs for graph traversal
- **Changed**: `steps` parameter now defaults to 0 (auto-detect) instead of 20

**Migration Required:**

- Remove MODEL connections from A1111 Style Prompt nodes
- Connect CONDITIONING directly to sampler
- Connect checkpoint MODEL directly to sampler (bypass the prompt node)
- `steps` parameter is now optional - set to 0 for auto-detection

See MIGRATION_GUIDE.md for detailed migration instructions.

### 🐛 Bug Fixes

- Fixed step detection to work with any sampler/scheduler combination
- Improved handling of sequence length mismatches
- Better error messages when step-based syntax is used without step information

### 📊 Performance

- No performance impact - same wrapper function, different registration method
- Slightly more efficient due to ComfyUI's hook caching system
- Graph traversal happens once at node execution, not during sampling

---

## Previous Releases

### Tag Autocomplete System

- Added A1111-style tag autocomplete
- Support for Danbooru/e621 databases
- Frequency tracking and sorting
- Theme-aware UI

### Token Counter

- Real-time token counting
- Visual boundary markers
- BREAK-aware counting
- Warning colors for long prompts

### Core Features

- A1111-style prompt parsing
- Scheduling syntax `[from:to:when]`
- Alternation syntax `[A|B]`
- Emphasis syntax `(text:1.2)`
- BREAK support for isolation
- Direct scaling (anti-burn)
- SDXL dual-CLIP support
