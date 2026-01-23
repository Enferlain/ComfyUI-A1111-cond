# Changelog

## [Unreleased] - Hook System Refactor + Auto Step Detection

### 🎉 Major Changes

#### Removed MODEL Input Requirement
The node now uses ComfyUI's `TransformerOptionsHook` system instead of requiring a MODEL input. This makes workflows simpler and more intuitive.

**Before:**
```
MODEL ──► A1111 Prompt ──► MODEL ──► Sampler
CLIP  ──►                ──► COND  ──►
```

**After:**
```
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
