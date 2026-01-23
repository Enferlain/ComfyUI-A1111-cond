# A1111 Prompt Node

A custom ComfyUI node that implements **A1111-style prompt handling** with proper isolation, emphasis math, scheduling, and alternation.

## Installation

Requires `lark` for full nested syntax support:

```bash
pip install lark
```

## Features

### Positive prompt node

<img width="790" height="527" alt="image" src="https://github.com/user-attachments/assets/67c12e99-cf35-4d2e-af84-0e258fa6915d" />

### Negative prompt node

<img width="363" height="241" alt="image" src="https://github.com/user-attachments/assets/ba1da788-dc47-4939-bcbe-d1699ea9f872" />

### Full setup with optional TIPO support

<img width="1755" height="813" alt="image" src="https://github.com/user-attachments/assets/d430189f-6f50-42e9-8503-afa8d6d7db8a" />

### Core Features

- **Hard Chunking (The Sandbox)**: Tokens split into 75-token chunks with padding, preventing concept bleeding
- **Direct Scaling (Anti-Burn)**: Uses `z * weight` instead of Comfy's interpolation, avoiding artifacts at high weights
- **BREAK Support**: Fully isolated context windows - each BREAK segment is tokenized separately
- **Emphasis**: `(text:1.2)`, `(text)`, `[text]`
- **TIPO support**: TIPO prompt output can connect directly into the node, and it will show the generated prompt when the node receives it. Should use [my fork](https://github.com/Enferlain/z-tipo-extension/tree/custom) to preserve weighting emphasis in the a1111 snytax.

### Token Counter

The node displays a **live token count** in the header, showing tokens per 77-token sequence:

```
45/75 | 32/75
```

- Each number shows tokens in that sequence (max 75 usable per sequence)
- BREAK creates new sequences: `dog, cat BREAK bird` → `6/75 | 1/75`
- Updates in real-time as you type
- Uses ComfyUI's native tokenizer for accurate counts
- Clickable for breakdown and input ids

**Warning Colors:**

- **Gray** (default): Normal prompt length
- **Yellow/Orange**: 300+ total tokens (4+ chunks) - getting long
- **Red**: 450+ total tokens (6+ chunks) - may impact quality/memory

**Boundary Markers:**

- Orange vertical bars appear in the text where 75-token chunk boundaries fall
- Blue vertical bars mark BREAK positions
- Markers align with word boundaries and update in real-time

### Tag Autocomplete

The node includes [**A1111-style tag autocomplete**](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete) functionality:

- **Trigger**: Start typing any tag (2+ characters)
- **Database**: Uses Danbooru/e621 tag databases (~140k tags)
- **Search**: Matches tag names and aliases
- **Navigation**: Use ↑/↓ arrows, Tab/Enter to select, Escape to close
- **Color coding**: Tags are colored by type (general, artist, character, etc.)
- **Post counts**: Shows tag popularity for better selection
- **Frequency sorting**: Your frequently used tags appear first with a ★ indicator
- **Theme support**: Automatically adapts to ComfyUI's theme (dark/light/custom)

**Features:**
- Alias support: Type `sole_female` → suggests `1girl`
- Smart insertion: Automatically adds commas and handles spacing
- Parenthesis escaping: `name_(artist)` → `name_\(artist\)`
- Real-time search with 100ms debouncing
- Usage tracking: Tags you use often are prioritized in results
- Quality tags: Automatically includes `extra-quality-tags.csv` for common quality/style tags
- Theme-aware: Respects your ComfyUI color scheme

**Available tag databases:**
- `danbooru.csv` - Main Danbooru database (~140k tags)
- `e621.csv` - E621 database (furry-focused)
- `extra-quality-tags.csv` - Quality and style tags (auto-loaded)
- Custom CSV files can be added to `data/tags/`

**Frequency Management:**
Open browser console and use:
- `window.A1111Autocomplete.getStats()` - View your most used tags
- `window.A1111Autocomplete.resetFrequency()` - Clear usage data
- `window.A1111Autocomplete.exportFrequency()` - Backup your data

### Scheduling

| Syntax          | Meaning                                  | Requires steps? |
| --------------- | ---------------------------------------- | --------------- |
| `[cat:dog:0.5]` | Switch from "cat" to "dog" at 50%        | No              |
| `[cat:dog:10]`  | Switch at step 10 (requires steps input) | **Yes**         |
| `[glasses:0.5]` | Add "glasses" at 50%                     | No              |
| `[glasses:10]`  | Add at step 10                           | **Yes**         |
| `[hat::0.7]`    | Remove "hat" at 70%                      | No              |
| `[hat::15]`     | Remove at step 15                        | **Yes**         |

**Important**: 
- **Percentage-based syntax** (with decimals like `0.5`) works automatically - no steps parameter needed
- **Step-based syntax** (integers like `10`) requires setting the `steps` parameter to match your sampler
- The node will detect which syntax you're using and show an error if steps are needed but not provided

**Nested syntax supported:**

```
[honovy, exsys:chen bin, [as109|fkey], [sweetonedollar:11]:0.4]
```

### Alternation

| Syntax           | Meaning                                           |
| ---------------- | ------------------------------------------------- |
| `[white\|black]` | Switches per-step (step 1=white, step 2=black...) |
| `[A\|B\|C]`      | Cycles through options each step                  |
| `[A\|]`          | Alternates between A and nothing                  |

### Scheduled Alternation (Extension)

Control when alternation starts or stops:

| Syntax               | Meaning                                   |
| -------------------- | ----------------------------------------- |
| `[as109\|fkey::0.6]` | Alternate until 60%, then nothing         |
| `[as109\|fkey:0.4]`  | Nothing until 40%, then start alternating |
| `[as109\|fkey::15]`  | Alternate until step 15, then nothing     |

**Combining with scheduling to lock to a value:**

```
[as109|fkey::0.6][:as109:0.6]
```

This alternates until 60%, then switches to just "as109".

### Model Support

- **SDXL**: Dual CLIP (clip_l + clip_g) with independent weight scaling
- **SD1.5**: Full support

---

## Usage

This pack provides **two nodes**:

| Node                              | Use Case                                          |
| --------------------------------- | ------------------------------------------------- |
| **A1111 Style Prompt**            | Positive prompt (supports alternation/scheduling) |
| **A1111 Style Prompt (Negative)** | Negative prompt (no alternation support)          |

### A1111 Style Prompt (Positive)

#### Inputs

| Input         | Type   | Required | Description                                                                                           |
| ------------- | ------ | -------- | ----------------------------------------------------------------------------------------------------- |
| clip          | CLIP   | Yes      | The CLIP model                                                                                        |
| text          | STRING | Yes      | Prompt with A1111 syntax                                                                              |
| steps         | INT    | No       | Required for step-based syntax like `[thing:10]`. Set to 0 (default) for percentage-only syntax.     |
| normalization | BOOL   | No       | Enable EmphasisOriginalNoNorm                                                                         |
| debug         | BOOL   | No       | Show detailed schedule information                                                                    |

#### Outputs

| Output       | Type         | Description              |
| ------------ | ------------ | ------------------------ |
| conditioning | CONDITIONING | The encoded conditioning |

**Important**: 
- **Percentage syntax** `[cat:dog:0.5]` works with `steps=0` (auto-detects from sampler)
- **Step-based syntax** `[cat:dog:10]` requires setting `steps` to match your sampler's step count
- If you use step-based syntax with `steps=0`, you'll get a clear error message

### A1111 Style Prompt (Negative)

#### Inputs

Same as positive, but **without MODEL input**.

#### Outputs

| Output       | Type         | Description              |
| ------------ | ------------ | ------------------------ |
| conditioning | CONDITIONING | The encoded conditioning |

> **Note:** If scheduling/alternation syntax is used in the negative node, it will use the **first step's prompt only** and log an informational message.

### Workflow

The node works with any standard ComfyUI workflow:

```
        ┌─────────────────────────────┐
CLIP  ──┤  [A1111 Style Prompt]       ├──► CONDITIONING ──► Sampler (positive)
        └─────────────────────────────┘

        ┌─────────────────────────────┐
CLIP  ──┤  [A1111 Style Prompt (Neg)] ├──► CONDITIONING ──► Sampler (negative)
        └─────────────────────────────┘
```

**Alternation and scheduling work automatically** - no MODEL connection needed! The node uses ComfyUI's hook system to access step information during sampling.

---

## Examples

```
# Basic emphasis
a (beautiful:1.3) landscape

# BREAK for isolation
artist name BREAK character name, wearing a hat

# Step-based scheduling (with steps=28)
[mountains:ocean:14] at sunset

# Alternation (requires MODEL connected)
1girl, [as109|fkey], detailed

# Add element at specific step
1girl, [sweetonedollar:11], high quality

# Nested scheduling with alternation
1girl, [honovy:chen bin, [as109|fkey]:0.4]

# Scheduled alternation - stop at 60%
1girl, [as109|fkey::0.6], detailed

# Combined
(epic:1.2) [forest:city:0.3] BREAK detailed background
```

---

## Debug Mode

Enable the `debug` input to see detailed information:

```
[A1111 Prompt] Unique prompts: 4 (will encode each once)
[A1111 Prompt] Step transitions: 27 across 28 steps
[A1111 Prompt] Alternation pattern sample:
  Step 0: 1girl, chen bin, yoneyama mai, as109, , agm...
  Step 1: 1girl, chen bin, yoneyama mai, fkey, , agm...
  Step 2: 1girl, chen bin, yoneyama mai, as109, , agm...
```

---

## Technical Notes

### How Scheduling Works

The node uses ComfyUI's `TransformerOptionsHook` system to swap conditioning per-step:

- All unique prompts are encoded once (efficient caching)
- Per-step embeddings are stored in a hook attached to the conditioning
- During sampling, the hook receives `sample_sigmas` from the sampler
- The hook calculates the current step and swaps to the appropriate embedding
- This works with any sampler/scheduler automatically

### Step Parameter Behavior

**Two modes of operation:**

1. **Percentage-only mode** (`steps=0`, default):
   - Use syntax like `[cat:dog:0.5]` (with decimal points)
   - Works with any sampler step count automatically
   - Recommended for most users

2. **Step-based mode** (`steps=20`, etc.):
   - Use syntax like `[cat:dog:10]` (integers)
   - Must match your sampler's step count for accurate timing
   - Useful when you need precise step control

The node automatically detects which syntax you're using and validates accordingly.

### Known Limitations

1. **Alternation is positive-only**: Only the main node supports alternation. The negative node will use the first step's prompt if scheduling syntax is present.

2. **Step-based syntax requires matching steps**: If you write `[thing:10]` with `steps=20` but your sampler uses 30 steps, the transition will happen at step 10 (not scaled). Use percentage syntax for automatic scaling.

2. **Visual parity**: While the **prompt schedule** matches A1111 exactly (the same prompt text at each step), the **visual effect** may differ due to architectural differences:
   - A1111 applies conditioning at the CFGDenoiser level (before model call)
   - This node applies conditioning via model wrapper (during model call)

Use the **scheduled alternation** syntax (`[a|b::0.6]`) if you need to control exactly when alternation stops.

---

## Performance

- **Efficient encoding**: Unique prompts are encoded only once and cached
- `[A|B]` with 28 steps only encodes 2 prompts, not 28
- All embeddings are padded to the same length for efficient swapping
- BREAK segments are batched together for efficient processing
