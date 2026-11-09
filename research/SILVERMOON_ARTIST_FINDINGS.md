# SilvermoonMix Anima v2.3: supplied three-artist prompt

Run date: 2026-11-09, Europe/Budapest. Isolated research; no production behavior changed.

## Result

Each artist responds visibly when raised individually to 2 and 3 in this prompt.
Native value weighting and experimental single-pass value weighting both produce
clear changes. Log-weight attention bias is generally subtler here, particularly
for ssambatea. This experiment does not support promoting attention bias as a
stronger artist-style control.

Visual observations, without reference-image fidelity scoring:

| Increased artist | Visible changes at 2–3 |
| --- | --- |
| cutesexyrobutts | Softer, more painterly shading, warmer face and more modeled hair/fabric; generated signature-like text appears in several value-weighted outputs. |
| quasarcake | Face proportions/expression, stronger facial and fabric shadows, altered pose and framing. |
| ssambatea | Sharper, flatter and more graphic linework, darker contour/shadow treatment, changed expression and framing. |

These are observations of the rendered differences, not a claim of fidelity to
each artist. Style and composition move together. A generated signature is not
an independent style-quality measure. Responses from 2 to 3 are not uniformly
linear or monotonic in every visible attribute.

## Comparison sheets

- [cutesexyrobutts](results/silvermoon-v23-artists-2026-11-09/images/comparison_cutesexyrobutts.png)
- [quasarcake](results/silvermoon-v23-artists-2026-11-09/images/comparison_quasarcake.png)
- [ssambatea](results/silvermoon-v23-artists-2026-11-09/images/comparison_ssambatea.png)

Each sheet has three columns, weights 1, 2, 3. Top row: native weighting.
Middle row: single-pass value weighting. Bottom row: single-pass token spans
with attention bias instead of value scaling. Middle/bottom weight 1 is the
identical plain-prompt baseline. Native weight 1 includes explicit neutral
parentheses and can differ from plain.

## Controls and scope

- Exact supplied positive prompt, including all three artist tags and escaped
  character-name parentheses. Only one artist is wrapped and increased per
  row; the other two stay unweighted. Full prompts are retained in preparation.json.
- Exact checkpoint: `silvermoonmixAnima_v23.safetensors`, SHA256
  `fcf423c227cd2ade6df46636f609f3a961371fdb4b57d84825e210a613df5c0b`.
- Qwen TE: `qwen_3_06b_base.safetensors`; learned adapter from the specified
  Silvermoon checkpoint. FP32 Qwen/adapter preprocessing for all variants,
  BF16 diffusion backbone; default VAE precision. Native denotes native token
  weighting semantics under this shared preprocessing, not bitwise parity
  with a default BF16 end-to-end ComfyUI workflow.
- VAE: `qwen_image_vae.safetensors`. 768x768, seed 12345, 24 Euler/normal steps,
  CFG 4; negative `low quality, blurry, jpeg artifacts`. No LoRAs.
- 24 renders: 8 per artist, with 3 repeated baseline renders. All baseline
  PNG hashes identical. One seed, one prompt, one checkpoint; no SDXL comparison.
- No RMS or mean normalization was applied. Attention bias uses `log(weight)`
  after key normalization and applies only to positive CFG groups. Its mixed
  CFG SDPA path has mathematically zero bias on negative rows, without an
  assertion of bitwise equivalence to other optimized attention backends.

## Correctness checks

All 9 weighted prompt cases matched actual Anima preprocessing exactly. The
single-pass candidate preserved full-text Qwen/T5 IDs in all 9, and preserved
baseline conditioning exactly in all 3 weight-1 cases. All weighted artist spans
had zero mixed-weight token-boundary crossings. Native neutral syntax changed
conditioning in all 3 cases.

The existing 6 span/offset tests and 4 attention-bias tests pass. Both new scripts
compile. The image summary checks all 24 cases and repeated baseline hashes;
pixel RMSE is retained only as a change diagnostic, not a style score.

The key-normalization probe from the earlier research concerns attention
selection; these images demonstrate that native value scaling can still have a
large visual effect. Weak changes in key attention probabilities do not imply
weak overall image changes.

Single-pass remains useful for neutral syntax identity and separating syntax
changes from weights. This prompt provides no basis for claiming it is always
visually stronger than native weighting, or for choosing a universal strength
conversion. Lower increments around 1 and more seeds would be needed to
calibrate practical controls.

## Reproduce

Run in ComfyUI's Python environment, from the custom-node repository:

```bash
python -X utf8 -u research/prepare_silvermoon_artist_sweep.py --output research/results/silvermoon-v23-artists-2026-11-09
python -X utf8 -u research/render_anima_emphasis.py research/results/silvermoon-v23-artists-2026-11-09/prepared.pt --output research/results/silvermoon-v23-artists-2026-11-09/images --seed 12345 --steps 24 --size 768 --cfg 4
python -X utf8 research/summarize_silvermoon_artist_sweep.py research/results/silvermoon-v23-artists-2026-11-09/images
```

`prepared.pt` and sampled latents are local ignored intermediates. JSON and PNG
evidence is deliberately retained under the run directory. Preparation records
checkpoint hash, script hash, prompts, token spans, and parity checks; render
plan binds the prepared bundle hash and sampler settings. Production integration
and wider qualification remain tracked separately in Beads.

Review `/tmp/review-mcp-run-zq8ua18h` approved the preparation experiment with no
blocking findings. Follow-up corrections close the checkpoint file explicitly,
check unique artist occurrences, and reject accidental preparation overwrites;
they do not change tensor calculations. `preparation_source.py` in the run
directory preserves the exact executed source matching preparation.json's hash.
