"""
Research harness: Anima (Qwen3-0.6B TE) vs the A1111 node's CLIP assumptions.

Run with the ComfyUI venv python, e.g. from the ComfyUI root:

    .venv/Scripts/python.exe -X utf8 custom_nodes/A1111_Prompt_Node/research/te_harness_anima.py

Rungs:
  R0  parity     - node output vs core CLIPTextEncode-equivalent, plain text
  R1  emphasis   - where do (tag:1.3) weights actually land for Anima
  R2  metadata   - native negative prompt parity
  R3  boundaries - reject unsupported scheduling and normalization

Purely tensor-level: no DiT, no sampling, no image generation.
Read-only w.r.t. the repo (prints findings, writes nothing).
"""

import sys
from pathlib import Path

COMFY_ROOT = r"D:/ComfyUI-current"
NODE_PKG_PARENT = COMFY_ROOT + r"/custom_nodes"
TE_PATH = r"D:/ComfyUI/models/text_encoders/qwen_3_06b_base.safetensors"

sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, NODE_PKG_PARENT)

import torch  # noqa: E402

# Simulate a headless process without registering API routes. The API helpers
# must now import cleanly even when the server dependency is unavailable.
sys.modules.setdefault("server", None)

import comfy.sd  # noqa: E402
import folder_paths  # noqa: E402
from A1111_Prompt_Node.nodes.prompt_node import A1111PromptNode  # noqa: E402
from A1111_Prompt_Node.nodes.negative_node import A1111PromptNegative  # noqa: E402

PLAIN = "a cat sitting on a windowsill"
WEIGHTED = "a (cat:1.3) sitting on a windowsill"
BREAK_TEXT = "a cat sitting BREAK on a windowsill"

results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def core_encode(clip, text):
    """What core CLIPTextEncode produces (full dict incl. extras)."""
    tokens = clip.tokenize(text)
    out = clip.encode_from_tokens(tokens, return_dict=True)
    return out  # {cond, pooled_output, **extras}


def node_encode(clip, text):
    """What the A1111 node returns for the same text (no schedule, no wildcards)."""
    node = A1111PromptNode()
    ret = node.encode(clip=clip, text=text, normalization=False, debug=False)
    cond_list = ret["result"][0]
    cond, meta = cond_list[0]
    return cond, meta


def main():
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    for k in ("text_encoders", "clip", "embeddings"):
        try:
            folder_paths.add_model_folder_path(k, str(Path(TE_PATH).parent))
        except Exception as e:  # noqa: BLE001
            print(f"(folder_paths note: {e})")

    print("loading qwen3_06b TE (CPU)...")
    clip = comfy.sd.load_clip(
        ckpt_paths=[TE_PATH],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    clip.load_model()

    csm = clip.cond_stage_model
    tok = clip.tokenizer
    print(f"clip_name           : {csm.clip_name}")
    print(f"tokenizer attrs     : {sorted(a for a in dir(tok) if not a.startswith('_'))[:0] or ''}")
    subtok = getattr(tok, csm.clip_name, None)
    t5tok = getattr(tok, "t5xxl", None)
    print(f"sub-tokenizers      : qwen3_06b={type(subtok).__name__}, t5xxl={type(t5tok).__name__}")
    if subtok is not None:
        print(f"qwen max_length     : {subtok.max_length} (pad_to_max_length={subtok.pad_to_max_length})")
    if t5tok is not None:
        print(f"t5   max_length     : {t5tok.max_length} (pad_to_max_length={t5tok.pad_to_max_length})")
    print(f"dict keys returned  : {sorted(clip.tokenize('x').keys())}")

    print("\n=== R0: parity, plain text ===")
    core = core_encode(clip, PLAIN)
    n_cond, n_meta = node_encode(clip, PLAIN)

    same_cond = core["cond"].shape == n_cond.shape and torch.equal(core["cond"], n_cond)
    check(
        "R0a node cond == core cond (bitwise)",
        bool(same_cond),
        f"core {tuple(core['cond'].shape)} vs node {tuple(n_cond.shape)}",
    )
    core_pooled = core.get("pooled_output")
    node_pooled = n_meta.get("pooled_output")
    same_pooled = (core_pooled is None and node_pooled is None) or (
        core_pooled is not None
        and node_pooled is not None
        and torch.equal(core_pooled, node_pooled)
    )
    check("R0b pooled matches", bool(same_pooled))

    core_extras = {k: v for k, v in core.items() if k not in ("cond", "pooled_output")}
    node_extras = {k: v for k, v in n_meta.items() if k != "pooled_output"}
    check(
        "R0c extras carried through",
        set(core_extras).issubset(set(node_extras)) and all(
            torch.equal(core_extras[k], node_extras[k]) for k in core_extras if k in node_extras
        ),
        f"core has {sorted(core_extras)}; node has {sorted(node_extras)}",
    )

    print("\n=== R1: emphasis, (cat:1.3) ===")
    # NOTE: emphasis ALSO changes tokenization on Qwen-family BPE vocabularies
    # (standalone fragment loses its word-start marker; see docs/archive/USAGE_ISSUES_FINDINGS.md
    # #8). T5 fragmentation can also change IDs around punctuation; the newer
    # ANIMA_EMPHASIS_FINDINGS.md supersedes the historical lossless-T5 claim.
    # So cond deltas are expected even with
    # neutralized weights; the rung-1 question is whether the WEIGHT CHANNEL
    # carries non-unity weights at exactly the tagged positions.
    core_plain = core_encode(clip, PLAIN)
    core_weighted = core_encode(clip, WEIGHTED)
    print(f"     cond seq: plain={core_plain['cond'].shape[1]}, weighted={core_weighted['cond'].shape[1]}"
          " (inherited Qwen fragment tokenization)")

    w = core_weighted.get("t5xxl_weights")
    ids = core_weighted.get("t5xxl_ids")
    t5tok = getattr(tok, "t5xxl", None)
    check("R1a core carries t5xxl_weights/t5xxl_ids extras", w is not None and ids is not None)
    if w is not None and ids is not None:
        toks = t5tok.tokenizer.convert_ids_to_tokens(ids.tolist())
        flagged = [(i, t, round(float(w[i]), 3)) for i, t in enumerate(toks) if abs(float(w[i]) - 1.0) > 1e-6]
        cat_positions = [i for i, t in enumerate(toks) if "cat" in t]
        print(f"     t5 tokens: {toks}")
        print(f"     non-unity weights: {flagged}")
        check(
            "R1b weights land exactly on 'cat' tokens",
            len(flagged) > 0 and all(i in cat_positions for i, _, _ in flagged) and len(flagged) == len(cat_positions),
            f"'cat' at {cat_positions}, weighted at {[i for i, _, _ in flagged]}, value={flagged[0][2] if flagged else None}",
        )

    n_cond_p, _ = node_encode(clip, PLAIN)
    n_cond_w, n_meta_w = node_encode(clip, WEIGHTED)
    # seq lengths differ (7 vs 8) due to segment-split tokenization - compare norms, not elementwise
    print(f"     node cond: plain {tuple(n_cond_p.shape)} norm={n_cond_p.norm():.4f}, "
          f"weighted {tuple(n_cond_w.shape)} norm={n_cond_w.norm():.4f}")
    has_extras = "t5xxl_weights" in n_meta_w
    check(
        "R1c node preserves native weight channel",
        bool(has_extras) and torch.equal(n_meta_w["t5xxl_weights"], w),
        f"extras in node output: {sorted(k for k in n_meta_w if k != 'pooled_output')}",
    )

    print("\n=== R2: negative prompt metadata ===")
    negative = A1111PromptNegative().encode(clip, WEIGHTED)["result"][0][0]
    check("R2 negative preserves weighted native encoding",
          torch.equal(negative[0], core_weighted["cond"]) and
          all(torch.equal(negative[1][key], core_weighted[key])
              for key in ("attention_mask", "t5xxl_ids", "t5xxl_weights")))

    print("\n=== R3: unsupported features fail explicitly ===")
    for text, normalization, expected_message in (
        (BREAK_TEXT, False, "BREAK"),
        ("[cat|dog]", False, "scheduling/alternation"),
        (PLAIN, True, "normalization"),
    ):
        try:
            A1111PromptNode().encode(
                clip, text, normalization=normalization,
                prompt={"2": {"class_type": "KSampler", "inputs": {
                    "positive": ["1", 0], "steps": 4,
                }}}, unique_id="1",
            )
        except ValueError as exc:
            check(f"R3 rejects {expected_message}", expected_message in str(exc))
        else:
            check(f"R3 rejects {expected_message}", False)

    print("\n=== summary ===")
    fails = [name for name, ok in results if not ok]
    print(f"{len(results) - len(fails)}/{len(results)} checks green; red: {fails or 'none'}")
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
