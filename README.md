# A1111 Prompt Node (God Node)

A custom ComfyUI node that implements A1111-style prompt handling with proper isolation and emphasis math.

## Installation

Requires `lark` for full nested syntax support:

```bash
pip install lark
```

## ✅ Working Features

### Core Features

- **Hard Chunking (The Sandbox)**: Tokens split into 75-token chunks with padding, preventing concept bleeding
- **Direct Scaling (Anti-Burn)**: Uses `z * weight` instead of Comfy's interpolation, avoiding artifacts at high weights
- **BREAK Support**: Fully isolated context windows
- **Emphasis**: `(text:1.2)`, `(text)`, `[text]`

### Scheduling (Fully Recursive)

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

- `[white|black]` - Switches per-step (step 1 = white, step 2 = black, step 3 = white...)
- `[A|B|C]` - Cycles through options each step

> ⚠️ **Note**: First option may appear weaker than in A1111 due to sigma mapping differences. See Known Limitations.

### Model Support

- **SDXL**: Dual CLIP (clip_l + clip_g) with independent weight scaling
- **SD1.5**: Full support

### Inputs

| Input         | Type   | Description                                      |
| ------------- | ------ | ------------------------------------------------ |
| clip          | CLIP   | The CLIP model                                   |
| text          | STRING | Prompt with A1111 syntax                         |
| steps         | INT    | Total sampling steps (for step-based scheduling) |
| normalization | BOOL   | Enable EmphasisOriginalNoNorm                    |

---

## 🚧 Known Limitations

### Alternation Weight Difference

**Issue**: When using alternation `[A|B]`, the first option appears weaker in ComfyUI compared to A1111.

**Example**: With prompt `[as109|fkey]`:

- **A1111**: `as109` (first option) dominates the style
- **ComfyUI**: `fkey` (second option) appears stronger

**Root Cause**: Architectural difference in how scheduling is applied:

| Aspect        | A1111                                     | ComfyUI                                    |
| ------------- | ----------------------------------------- | ------------------------------------------ |
| **Selection** | Step-based: `current_step <= end_at_step` | Sigma-based: percentage → sigma conversion |
| **Mapping**   | Linear step numbers (1, 2, 3...)          | Non-linear sigma distribution              |
| **Boundary**  | Inclusive (`<=`)                          | Exclusive (`<` and `>`)                    |

A1111 reconstructs conditioning per-step using discrete step numbers. ComfyUI uses `start_percent`/`end_percent` which converts through a non-linear sigma schedule. Even with identical step-to-percentage mapping, the sigma distributions differ.

**Potential Solutions** (not yet implemented):

1. **Access sampler's actual sigma schedule**: At encoding time, query the model's sigma schedule and create percentage ranges that match the exact sigma values for each step.

2. **Custom conditioning callback**: Implement a sampler hook that intercepts conditioning selection and applies A1111-style step-based logic directly.

3. **Sigma-aware percentage mapping**: Pre-compute the sigma distribution for common samplers and create lookup tables for accurate step→percentage conversion.

### Other Limitations

- **Token Counter**: Display pending frontend integration

---

## Usage Examples

```
# Basic emphasis
a (beautiful:1.3) landscape

# BREAK for isolation
artist name BREAK character name, wearing a hat

# Step-based scheduling (with steps=28)
[mountains:ocean:14] at sunset

# Nested scheduling with alternation
1girl, [honovy:chen bin, [as109|fkey]:0.4]

# Combined
(epic:1.2) [forest:city:0.3] BREAK detailed background
```
