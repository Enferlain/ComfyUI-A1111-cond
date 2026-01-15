# A1111 Prompt Node

A custom ComfyUI node that implements **true A1111-style prompt handling** with proper isolation, emphasis math, and step-based scheduling.

## Installation

Requires `lark` for full nested syntax support:

```bash
pip install lark
```

## Features

### Core Features

- **Hard Chunking (The Sandbox)**: Tokens split into 75-token chunks with padding, preventing concept bleeding
- **Direct Scaling (Anti-Burn)**: Uses `z * weight` instead of Comfy's interpolation, avoiding artifacts at high weights
- **BREAK Support**: Fully isolated context windows - each BREAK segment is tokenized separately
- **Emphasis**: `(text:1.2)`, `(text)`, `[text]`

### Step-Based Scheduling ✨

**True A1111 parity!** When you provide a MODEL input, step-based scheduling works exactly like A1111:

| Syntax          | Meaning                                  |
| --------------- | ---------------------------------------- |
| `[cat:dog:0.5]` | Switch from "cat" to "dog" at 50%        |
| `[cat:dog:10]`  | Switch at step 10 (requires steps input) |
| `[glasses:0.5]` | Add "glasses" at 50%                     |
| `[glasses:10]`  | Add at step 10                           |
| `[hat::0.7]`    | Remove "hat" at 70%                      |
| `[hat::15]`     | Remove at step 15                        |

**Nested syntax supported:**

```
[honovy, exsys:chen bin, [as109|fkey], [sweetonedollar:11]:0.4]
```

### Alternation ✨

- `[white|black]` - Switches per-step (step 1 = white, step 2 = black, step 3 = white...)
- `[A|B|C]` - Cycles through options each step
- `[A|]` - Alternates between A and nothing

**True per-step switching!** When MODEL is connected, alternation works exactly like A1111 - the conditioning is swapped at each step during sampling.

### Model Support

- **SDXL**: Dual CLIP (clip_l + clip_g) with independent weight scaling
- **SD1.5**: Full support

---

## Usage

### Inputs

| Input         | Type   | Required | Description                                   |
| ------------- | ------ | -------- | --------------------------------------------- |
| clip          | CLIP   | Yes      | The CLIP model                                |
| text          | STRING | Yes      | Prompt with A1111 syntax                      |
| model         | MODEL  | No       | Connect for step-based scheduling/alternation |
| steps         | INT    | No       | Total sampling steps (default: 20)            |
| normalization | BOOL   | No       | Enable EmphasisOriginalNoNorm                 |
| debug         | BOOL   | No       | Show detailed schedule information            |

### Outputs

| Output       | Type         | Description                                     |
| ------------ | ------------ | ----------------------------------------------- |
| conditioning | CONDITIONING | The encoded conditioning                        |
| model        | MODEL        | Model with step-conditioning (when input given) |

### Basic Workflow

For **simple prompts** (no scheduling/alternation):

```
CLIP ──► [A1111 Style Prompt] ──► CONDITIONING ──► Sampler
```

For **step-based scheduling/alternation**:

```
CLIP ──────┐
           ├──► [A1111 Style Prompt] ──► CONDITIONING ──► Sampler
MODEL ─────┘                         └──► MODEL ─────────►
```

Connect the output MODEL to your sampler to enable true per-step conditioning switching.

---

## How It Works

### Problem I had with comfy prompting

ComfyUI uses **sigma-based percentage ranges** for conditioning:

- Converts step percentages to sigma values
- Non-linear sigma distribution means timing differs from A1111
- First alternation option often appears weaker

### What we do instead

When MODEL is connected with step-based scheduling:

1. **All unique prompts are pre-encoded** once (efficient caching)
2. **A model wrapper is registered** that intercepts each sampling step
3. **At each step**, the wrapper swaps the conditioning to the correct per-step embedding
4. **Exact step matching** based on the sigma schedule

This achieves **true A1111 parity** - the same prompt produces the same style balance in both A1111 and ComfyUI.

---

## Examples

```
# Basic emphasis
a (beautiful:1.3) landscape

# BREAK for isolation
artist name BREAK character name, wearing a hat

# Step-based scheduling (with steps=28)
[mountains:ocean:14] at sunset

# Alternation - requires MODEL input for true per-step switching
1girl, [as109|fkey], detailed

# Add element at specific step
1girl, [sweetonedollar:11], high quality

# Nested scheduling with alternation
1girl, [honovy:chen bin, [as109|fkey]:0.4]

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
  Step 14: 1girl, chen bin, yoneyama mai, as109, , agm...
  Step 27: 1girl, chen bin, yoneyama mai, fkey, , agm...
```

---

## Notes

- **Without MODEL input**: Scheduling/alternation still works but uses ComfyUI's standard sigma-based percentages (may differ slightly from A1111)
- **With MODEL input**: True per-step conditioning switching, matching A1111 behavior
- **Performance**: Unique prompts are encoded only once and cached, so `[A|B]` with 28 steps only encodes 2 prompts, not 28
