# Five-seed signature and adherence check

2026-11-09, Europe/Budapest. Same supplied three-artist Umika prompt and
`silvermoonmixAnima_v23.safetensors`. No production behavior changed.

## Result

The original observation repeats across all five fixed seeds for cutesexyrobutts:
single-pass value weighting at 3 generates visible artist-name/signature-like
lettering, whereas attention bias at 3 does not show detectable signature text.
Attention bias still visibly changes shading and rendering compared with plain.

| Artist increased (other two unchanged) | Plain | Single-pass 3 | Attention bias 3 |
| --- | ---: | ---: | ---: |
| cutesexyrobutts | 0/5 | 5/5 | 0/5 |
| quasarcake | 0/5 | 0/5 | 0/5 |
| ssambatea | 0/5 | 0/5 | 0/5 |

Counts mean **images with detected signature-like text**, not individual marks.
Plain is the same five images across all rows, not 15 independent observations.
All five cutesexyrobutts single-pass images have clear recognizable lettering;
some contain two marks. No ambiguous text labels were needed in this sample.
"Not detected" is an inspection result, not a guarantee of absence.

This supports retaining attention bias as a candidate for controlling artist
influence with fewer generated signatures in this specific setting. It does
not establish signature suppression in general or superiority at equal style
strength. Weight 3 means different operations in the two methods; the subtler
response of attention bias could itself explain some of the difference. A useful
next qualification would match perceived style strength at different weights.

## Visual style and prompt observations

Assistant visually inspected the complete original seed sheets, all four new
seed sheets, and original-resolution top/bottom image strips for each artist.
No OCR or external artist-reference scoring was used.

- cutesexyrobutts: both weighting methods move shading toward softer modeled
  hair, face and fabric. Single-pass is usually more pronounced; attention bias
  does not simply leave the baseline unchanged.
- quasarcake: both methods change face proportions, shadows and framing, without
  a signature difference in this sample.
- ssambatea: both move toward stronger graphic contours and flatter shading;
  attention bias often retains more of the baseline framing.
- The broad subject, grey hair, blue maid/kimono outfit, halo, food and outdoor
  setting remain recognizable throughout. This is not a score for every prompt
  tag: precise side view, open mouth, exact eye color, and the requested hands-up
  pose are not uniformly satisfied. Weighting changes composition as well as
  rendering, so no overall adherence winner is claimed.
- Seed 12349 has a suspected extra hand near the mouth in both ssambatea-weighted
  methods. The cleaner signature result does not imply better anatomy.

Judging exact fidelity to each artist still benefits from reference images or
the user's familiarity. The watermark check itself does not require the user
to inspect every image.

## Evidence and controls

- Seeds chosen before sampling: **12345–12349**, retained without resampling.
  Original seed 12345 reused; four additional seeds rendered.
- Three variants per artist: plain, single-pass 3, attention bias 3. There are
  45 displayed cases, **35 unique image inputs** because plain repeats across
  artists; 36 new renders include repeated baseline controls.
- Unchanged prepared conditioning from the prior Silvermoon run, FP32 Qwen and
  learned adapter, BF16 diffusion backbone, qwen_image VAE. 768x768, 24 Euler /
  normal steps, CFG4, same negative prompt, no LoRAs. Native semantics are not
  included as a separate multi-seed arm.
- Attention bias acts after key normalization on positive groups, keeping value
  vectors unscaled. Its research SDPA implementation is not asserted bitwise
  equal to alternate optimized backends on zero-bias negative rows.
- Repeated baseline hashes match within every seed. Source image plan is checked
  against the original prepared-bundle hash; per-seed plans are checked against
  the subset hash; checkpoint/VAE/negative prompt and sampler settings agree.
- Review `/tmp/review-mcp-run-8aburtaq` approved with provenance hardening
  recommendations. The two medium items (source bundle binding and checkpoint /
  VAE comparison) were fixed and all completed evidence revalidated with
  `--summarize-only`. Exact case composition, atomic subset saving, and settings
  argument derivation were also tightened. Ten focused tests and compile/diff
  checks pass. No independent repeated render of original seed 12345 was added.

[cutesexyrobutts sheet](results/silvermoon-v23-multiseed-2026-11-09/comparison_cutesexyrobutts.png),
[quasarcake sheet](results/silvermoon-v23-multiseed-2026-11-09/comparison_quasarcake.png),
[ssambatea sheet](results/silvermoon-v23-multiseed-2026-11-09/comparison_ssambatea.png).
Rows are seeds 12345–12349; columns are plain, single-pass 3, attention bias 3.
`margins_*.png` shows original-resolution top and bottom strips.

`visual_assessment.json` retains per-image inspection labels keyed to image hashes,
including the shared-baseline caveat. `images.json` binds all images to their
seed/artist/variant; `experiment.json` records the fixed plan. Image/tensor inputs
are retained as PNG/JSON and ignored local `.pt` intermediates respectively.

Reproduce using ComfyUI's Python:

```bash
python -X utf8 -u research/run_silvermoon_multiseed.py
# Revalidate and rebuild sheets without sampling:
python -X utf8 research/run_silvermoon_multiseed.py --summarize-only
```
