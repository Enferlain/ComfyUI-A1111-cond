# Conditioning implementations

Both prompt nodes use `get_encoder(clip)`. They do not branch on model names.

| Module | Responsibility |
| --- | --- |
| `base.py` | Encoding contract and capabilities for BREAK, scheduling, and normalization |
| `clip.py` | CLIP/SDXL token batches, BREAK isolation, direct emphasis scaling and pooling |
| `anima.py` | Anima mode selection and preservation of the metadata consumed by its T5-token adapter |
| `anima_tokenization.py` | Single-pass Qwen/T5 tokenization and character-span weight mapping |
| `anima_attention.py` | Conditioning-scoped log-weight attention masks after key normalization |
| `__init__.py` | Explicit selection from supported token-stream contracts |

An implementation returns `(conditioning_tensor, metadata_dict)`. Metadata must
include every field needed by the downstream model, including pooled output,
attention masks, IDs, and weights when present. Token streams can have independent
lengths. Implementations do not retain per-request state.

The positive node owns wildcard expansion, schedule parsing, encoding caches and
per-step orchestration. The negative node selects the first parsed prompt and
uses the same encoding contract. The existing schedule hook switches embedding
tensors, while pooled output stays fixed at the first prompt's value. In
particular, SDXL pooled conditioning is not switched with later prompts.
Additional metadata is rejected when scheduling until it can be switched
correctly. The metadata guard runs before caching each new scheduled encoding.

CLIP supports BREAK, direct weighting, normalization and the existing schedule
path. Anima supports static native, single-pass value, and attention weighting. Anima
BREAK is rejected because isolation through the downstream adapter is undefined;
encoding fragments separately before that adapter does not establish isolation.
Anima scheduling and normalization are also unsupported.

Nodes expose `anima_emphasis`, defaulting to `native`, and include it in encoding
cache keys. Non-native choices on other model families fail explicitly. Attention
mode neutralizes adapter value weights and attaches a `HookedOnly` transformer
hook to that conditioning. Comfy batches different hook groups separately. A
diffusion wrapper creates local attn1/attn2 markers and an attention override;
the override adds `log(weight)` as a mask through Comfy's existing attention
backend, composing existing masks and overrides. Model modules and caller option
dictionaries are not mutated. Multiple Anima attention hooks merged into the
same conditioning are explicitly unsupported. Native mode imports neither the
single-pass tokenizer helper nor the attention hook module.

To add another conditioning path, implement it in a separate module, declare its
capabilities, and add an explicit selector entry. Its tests should verify native
plain-text parity, weighting, metadata, and supported operations through the
consuming model. Unknown token streams must not fall back to CLIP.
