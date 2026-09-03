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