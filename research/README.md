# Anima conditioning research

The runnable scripts and reports here explain the experiments behind Anima's
emphasis options. Runtime code lives in [`encoding/`](../encoding/README.md)
and does not import these scripts.

- [Initial emphasis investigation](ANIMA_EMPHASIS_FINDINGS.md): tokenization,
  adapter weights, normalization and cross-attention probes.
- [Silvermoon three-artist sweep](SILVERMOON_ARTIST_FINDINGS.md): the supplied
  prompt, one checkpoint and one seed, artist weights 1–3.
- [Five-seed follow-up](SILVERMOON_MULTISEED_FINDINGS.md): signature-like text,
  visible rendering differences, and prompt adherence checked separately.
- `verify_anima_modes.py`: production node/TE/adapter/sampler checks, including
  default parity, neutral identity, scoped hooks and unchanged Native sampling.

Use ComfyUI's Python environment. Scripts assume local Windows model paths by
default; inspect their arguments/constants before running elsewhere. Reports
contain the original settings and reproduction commands.

## Artifact policy

Small JSON reports, manifests, hashes, visual assessment labels, and the executed
source snapshot under `results/` are versioned. Generated PNG images and `.pt`
conditioning/latent bundles are ignored and retained locally. No existing outputs
were deleted by repository cleanup.

Image links in reports point to those local outputs; regenerate them with the
documented commands after a fresh checkout. Some render/resume commands require
the ignored prepared tensors, so run the corresponding preparation stage first.
Absolute model paths and timings in reports describe the original machine.

Five seeds on one prompt/checkpoint do not establish universal style fidelity,
signature suppression or a strength conversion between methods. Native stays
the default; the other modes are explicit choices.
