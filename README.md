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

### Scheduling

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

| Input         | Type   | Required | Description                                      |
| ------------- | ------ | -------- | ------------------------------------------------ |
| clip          | CLIP   | Yes      | The CLIP model                                   |
| text          | STRING | Yes      | Prompt with A1111 syntax                         |
| model         | MODEL  | No       | Required for alternation/step-based conditioning |
| steps         | INT    | No       | Total sampling steps (default: 20)               |
| normalization | BOOL   | No       | Enable EmphasisOriginalNoNorm                    |
| debug         | BOOL   | No       | Show detailed schedule information               |

#### Outputs

| Output       | Type         | Description              |
| ------------ | ------------ | ------------------------ |
| conditioning | CONDITIONING | The encoded conditioning |
| model        | MODEL        | Model with step wrapper  |

### A1111 Style Prompt (Negative)

#### Inputs

Same as positive, but **without MODEL input**.

#### Outputs

| Output       | Type         | Description              |
| ------------ | ------------ | ------------------------ |
| conditioning | CONDITIONING | The encoded conditioning |

> **Note:** If scheduling/alternation syntax is used in the negative node, it will use the **first step's prompt only** and log an informational message.

### Workflow

For **alternation and per-step scheduling** to work correctly, you must connect the MODEL input on the positive node:

```
        ┌─────────────────────────────┐
MODEL ──┤  [A1111 Style Prompt]       ├──► MODEL ──────────► Sampler
CLIP  ──┤                             ├──► CONDITIONING ──► (positive)
        └─────────────────────────────┘

        ┌─────────────────────────────┐
CLIP  ──┤  [A1111 Style Prompt (Neg)] ├──► CONDITIONING ──► (negative)
        └─────────────────────────────┘
```

Without MODEL connected, only static prompts work (no alternation).

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

The node generates the correct prompt text for each step, matching A1111's behavior:

- All unique prompts are encoded once (efficient caching)
- Per-step embeddings are stored and swapped via model wrapper
- Sigma-based step detection maps the sampling progress to steps

### Known Limitations

1. **Alternation is positive-only**: Only the main node (with MODEL input) supports alternation. The negative node will use the first step's prompt if scheduling syntax is present.

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
