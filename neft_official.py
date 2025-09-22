# neft_official.py
import torch

def NEFTune_safe(model, noise_alpha: float = 5.0):
    """
    'Official-style' NEFT: monkey-patch the model's embedding forward, adding
    Uniform(-mag, +mag) noise during TRAINING only.
    mag = alpha / sqrt(T * D)  (no pad-awareness, per the original snippet)
    This tries common embedding attribute paths and falls back to get_input_embeddings().
    """
    if noise_alpha <= 0:
        return model

    def _wrap(orig_forward):
        def _new_forward(x, *args, **kwargs):
            out = orig_forward(x, *args, **kwargs)  # [B, T, D]
            if model.training:
                B, T, D = out.shape
                # Compute magnitude as a Python float for device/dtype safety
                mag = float(noise_alpha) / float((T * D) ** 0.5)
                return out + torch.empty_like(out, dtype=out.dtype).uniform_(-mag, mag)
            return out
        return _new_forward

    # Try common embedding paths across LLaMA/Qwen-like models
    candidates = [
        "model.embed_tokens",          # Qwen/LLaMA (HF)
        "base_model.embed_tokens",     # some LLaMA wrappers
        "transformer.wte",             # GPT-2 like
        "gpt_neox.embed_in",           # GPT-NeoX like
    ]
    embed = None
    for path in candidates:
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            embed = obj
            break
        except AttributeError:
            continue

    if embed is None:
        # Fall back to HF API; this returns the module to patch
        embed = model.get_input_embeddings()

    # Monkey-patch forward
    embed.forward = _wrap(embed.forward)
    return model