# Anima emphasis experiment — 2026-11-08

The run started on November 8; analysis continued after midnight on November 9
(Budapest time). Artifact paths retain the run start date.

The experiment separates syntax-induced token changes, adapter-output value
scaling, and attention-probability weighting. All candidates remain under
`research/`; production nodes still use native ComfyUI Anima encoding.

## Evidence and scope

- Actual local ComfyUI tokenizers, Qwen3-0.6B TE and learned AnimaYume v1.5 adapter.
- Four artist tags (`@as109`, `@fkey`, `@mignon`, `@nnn yryr`) and three concepts
  (`chibi`, `red hair`, `watercolor`), each at the start and middle of a prompt,
  with weights 1.0, 1.3 and 2.0: 42 weighted cases, 14 neutral-weight comparisons.
- Adapter diagnostics use FP32 weights/activations. The image experiment passes
  prepared FP32 adapter outputs to the BF16 diffusion backbone. “Native” labels
  mean native tokenization/weighting semantics through that same preparation;
  they do not claim bitwise identity to a default BF16 native workflow.
- Twelve diagnostic images: two artist prompts, six modes, one seed (12345),
  AnimaYume v1.5, 768×768, 24 Euler/normal steps, CFG 4, no LoRAs. Negative prompt:
  `low quality, blurry, jpeg artifacts`. One checkpoint and seed cannot establish
  general style fidelity or calibrate an SDXL-equivalent strength.

## 1. Neutral parentheses are not neutral in native tokenization

At weight 1.0, Qwen IDs changed in 7/14 comparisons, all middle-position cases.
T5 IDs changed in 14/14 comparisons, including extra tokens near punctuation.
All 14 native adapter results differed from their plain-prompt counterparts.
Earlier observations that T5 fragment tokenization was effectively lossless do
not generalize to these punctuation/artist-tag cases.

The single-pass prototype reuses ComfyUI's emphasis parser, reconstructs the
complete plain text and tokenizes both streams once. Fast T5 character offsets
map weights back to tokens. It preserves plain-text token IDs in every case and
produces bitwise-identical adapter conditioning at weight 1.0 in all 14 cases.
At non-neutral weights, vectors outside the weighted token span stay unchanged.
Tokens crossing emphasis boundaries receive a character-averaged weight and are
reported explicitly. Whitespace-only and special tokens remain neutral.

Qwen token weights deliberately remain 1.0, as in the native Anima tokenizer;
emphasis travels through the T5 weights into the adapter output. The runner
asserts this invariant and includes token weights in the Qwen cache key. It
verifies the prepared native result against `Anima.preprocess_text_embeds` in all
42 cases and records the verified indices. Single-pass mode intentionally changes
native tokenization semantics; it is not equivalent to native at every weight.

## 2. Key normalization changes what a strength multiplier does

Anima's Cosmos cross-attention applies RMS normalization after its bias-free key
projection. Positive per-token scaling therefore largely cancels in the keys,
while the values retain the scale. This gives an amplitude control without the
same direct attention-logit scaling present in an unnormalized-key path.

The actual learned first cross-attention block, using synthetic image queries
and the `@as109` token span at weight 2, measured:

| Measurement | Result |
| --- | --- |
| Relative key change after normalization | 0.0000936 |
| Error against exactly weighted values | 0 |
| Mean artist attention-mass ratio, native scaling | 1.000013× |
| Mean artist attention-mass ratio, additive log-weight bias | 1.9828× |

This plausibly explains a different strength response from SDXL. It is not a
direct SDXL image comparison or a universal weight conversion.

The experimental bias applies `log(weight)` to cross-attention logits after key
normalization, leaving values unscaled. CFG group tests cover both positive/
negative orders and multiple images per group. Negative groups retain zero bias;
neutral weights preserve the original attention backend. Overrides are temporary
and restored even on error. This is a different control, not a drop-in numerical
equivalent to native value scaling.

Mixed CFG batches run through the research SDPA backend on both branches, with
zero bias on the negative branch. Mathematical neutrality is tested; bitwise
parity with another optimized original backend is not asserted. Image comparisons
involving the bias mode can therefore include backend numerical differences.
Strictly positive weights are required. Weights below 1 are supported by the
formula and unit-tested at 0.5, but were not image-qualified here; zero and negative
weights are rejected.

## 3. Normalization should be an independent choice

Across the tested weight-2 cases, mean-preserving multipliers ranged from about
-1.20 to 6.13; one multiplier was negative and would invert the conditioning. A small signed mean is a poor amplitude
reference for these contextual vectors. Do not transplant the CLIP mean-rescale
rule into this path.

An RMS-preserving variant was finite and required multipliers about 0.57–0.79.
It restores global RMS but rescales unweighted tokens too. The two diagnostic
artist images tended to retain more of the baseline composition. That observation
does not establish better artist fidelity or universally better weighting.

## 4. Scheduling has a separate existing index error

The actual current hook selected `[0, 0]` for a two-step prepared schedule instead
of `[0, 1]`; a 30-step schedule duplicated index 14 and never selected index 29.
The probe retains the failed results. This must be corrected independently of
Anima's requirement to prepare complete adapter-dependent conditioning before
switching. Tracked in `A1111_Prompt_Node-9fl`; no hook behavior changed here.

## Practical direction

First qualify single-pass emphasis as an explicit compatibility option: its
neutral identity is a concrete improvement, and it leaves weighting strength
semantics separate from parsing. Keep native value scaling and experimental
attention-odds bias as distinct modes. Keep normalization off by default; RMS is
an experiment, and mean-preserving normalization is not justified by these data.
Broader checkpoint, prompt and seed comparisons are needed before promotion.

## Reproduction and artifacts

Run from the custom-node checkout using the installed ComfyUI Python environment:

```bash
python -X utf8 -u research/run_anima_emphasis.py --stage adapter --device cuda:1 \
  --concepts chibi 'red hair' watercolor --output research/results/anima-emphasis-2026-11-08/general-concepts
python -X utf8 -u research/run_anima_emphasis.py --stage adapter --save-prepared \
  --output research/results/anima-emphasis-2026-11-08/animayume-v15
python -X utf8 -u research/render_anima_emphasis.py \
  research/results/anima-emphasis-2026-11-08/animayume-v15/prepared.pt \
  --output research/results/anima-emphasis-2026-11-08/animayume-v15/images
python -X utf8 -u research/render_anima_emphasis.py \
  research/results/anima-emphasis-2026-11-08/animayume-v15/prepared.pt \
  --output research/results/anima-emphasis-2026-11-08/animayume-v15/images-attention-bias --attention-bias-only
python -X utf8 -u research/probe_anima_attention.py \
  research/results/anima-emphasis-2026-11-08/animayume-v15/prepared.pt \
  --output research/results/anima-emphasis-2026-11-08/animayume-v15/attention.json
python -X utf8 research/summarize_anima_emphasis.py research/results/anima-emphasis-2026-11-08/animayume-v15
```

Checkpoint/TE/VAE paths are CLI options; defaults identify the local test files.
Prepared bundles contain tensors and are loaded with `weights_only=True`. Image
plans bind bundle hash and sampler settings; sampled latents are retained for
resumable decoding. New bundles embed attention-bias weights. Legacy bundles
validate adjacent report weights against their stored weighted conditioning;
new bias-render plans also bind the token-report hash. The retained legacy bundle
was checked explicitly for both artist rows. The initial artist-only bundle must
be regenerated in a new directory to reproduce its byte-level hash with a changed
script version. Intermediate `.pt` tensors are gitignored; JSON and PNG findings
are retained deliberately.

The actual VAE's output mapping was verified as `[-1,0,1] -> [0,0.5,1]`; decoded
pixels already lie in `[0,1]`. Re-decoding the first retained latent reproduced its
saved PNG exactly, so no additional range conversion is required.

- [Combined image grid](results/anima-emphasis-2026-11-08/animayume-v15/comparison-all.png)
- [Full tokenization sweep](results/anima-emphasis-2026-11-08/general-concepts/tokenization.json)
- [Adapter metrics and retained schedule failures](results/anima-emphasis-2026-11-08/general-concepts/adapter.json)
- [Learned attention-block probe](results/anima-emphasis-2026-11-08/animayume-v15/attention.json)
- [Image differences](results/anima-emphasis-2026-11-08/animayume-v15/image_differences.json)
- [VAE and legacy weight-binding validation](results/anima-emphasis-2026-11-08/animayume-v15/render_validation.json)
- [Retained image-harness failures](results/anima-emphasis-2026-11-08/animayume-v15/images/failures.json)

Two initial integration attempts failed before image output: the harness first
used a four-dimensional latent, then removed only the batch axis from a decoded
video tensor. The corrected harness uses B,C,T,H,W latents and explicitly extracts
the single decoded frame. These failures do not provide model-quality evidence.
