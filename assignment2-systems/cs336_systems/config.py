import torch
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size : int = 10000
    num_layers: int = 12
    num_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072

    batch_size : int = 4
    context_length: int = 512

    steps: int = 8000
    warm_up_times:int = 5
    tests:int = 10
    device: torch.device = "cuda"
    dtype: torch.dtype =  torch.float32

    @classmethod
    def _from_size(cls, size: str, **overrides) -> "Config":
        presets = {
            "small": dict(d_model = 768, d_ff = 3072, num_layers = 12, num_heads = 12),
            "medium": dict(d_model = 1024, d_ff = 4096, num_layers = 24, num_heads = 16),
            "large": dict(d_model = 1280, d_ff = 5120, num_layers = 36, num_heads = 20),
            "xl": dict(d_model = 2560, d_ff = 10240, num_layers = 32, num_heads = 32),
            "10B": dict(d_model = 4608, d_ff = 12288, num_layers = 50, num_heads = 36)
        }

        return cls(**{**presets[size], **overrides})