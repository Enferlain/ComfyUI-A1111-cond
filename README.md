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

- `[white|black]` - Blends concepts (50/50 average)
- `[A|B|C]` - Equal blend of 3 options (33% each)

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

- **Token Counter**: Display pending frontend integration
- **True Alternation**: Uses blending, not per-step switching (would need sampler hooks)
- **Without Lark**: Nested syntax won't parse; falls back to simple regex

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
