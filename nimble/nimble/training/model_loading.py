"""Load the pinned CUDA Qwen checkpoint shared by schema training and evaluation."""

import torch
from transformers import Qwen3_5ForConditionalGeneration


def load_base(model_id, revision):
    if not torch.cuda.is_available():
        raise RuntimeError("This trainer requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This trainer requires BF16 support")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    model.config.use_cache = False
    return model
