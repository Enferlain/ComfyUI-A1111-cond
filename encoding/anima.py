"""Anima's native Qwen embeddings and T5-token conditioning adapter contract."""

from .base import EncoderCapabilities, PromptEncoder

ANIMA_EMPHASIS_MODES = ("native", "single_pass", "attention")


class AnimaEncoder(PromptEncoder):
    name = "Anima"
    capabilities = EncoderCapabilities(breaks=False, scheduling=False, normalization=False)

    def __init__(self, emphasis="native"):
        if emphasis not in ANIMA_EMPHASIS_MODES:
            raise ValueError(f"Unknown Anima emphasis mode: {emphasis}")
        self.emphasis = emphasis

    def matches(self, clip, streams):
        return (
            streams == {"qwen3_06b", "t5xxl"}
            and getattr(getattr(clip, "cond_stage_model", None), "clip_name", None) == "qwen3_06b"
        )

    def encode(self, clip, text, normalization=False, debug=False):
        self.validate_request(text, normalization=normalization)
        hooks = None
        if self.emphasis == "native":
            tokens = clip.tokenize(text)
        else:
            from .anima_tokenization import single_pass_tokens

            tokens, _ = single_pass_tokens(clip.tokenizer, text)
            if self.emphasis == "attention":
                from .anima_attention import attention_hooks

                hooks = attention_hooks([token[1] for token in tokens["t5xxl"][0]])
                tokens["t5xxl"] = [[(token[0], 1.0) for token in tokens["t5xxl"][0]]]
        if not self.matches(clip, set(tokens)):
            raise ValueError("Anima encoder returned unexpected token streams.")
        # T5 IDs and weights are consumed by Anima's adapter, not by direct
        # scaling of Qwen embeddings. BREAK isolation through that adapter and
        # preprocessing-aware schedules have not been defined or qualified.
        output = clip.encode_from_tokens(tokens, return_dict=True).copy()
        if hooks is not None:
            existing = output.get("hooks")
            output["hooks"] = hooks if existing is None else existing.clone_and_combine(hooks)
        return output.pop("cond"), output
