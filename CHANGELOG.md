# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-02-02

- **API Normalization**: The `autocomplete` API now robustly handles the `extra_files` parameter, coercing strings to lists and providing sensible defaults.
- **Memory Management Refinements**:
  - **Bounded Caching**: Implemented instance-level prompt encoding cache with strict enforcement during execution to prevent unbounded growth.
  - **Internal Cleanup**: Added explicit tensor deletion in `hooks.py` to prevent accumulation.
  - **Leak Prevention**: Refactored frontend extensions to properly disconnect `ResizeObserver` instances and remove global click listeners upon node removal.

- **Hotfixes & Stability**:
  - **Backend**: Restored missing `transformers_dict` initialization in `hooks.py` to fix `AttributeError` crash.
  - **Frontend**: Merged duplicate `nodeCreated` hooks in `autocomplete.js` to restore extension functionality.
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
