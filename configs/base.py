from dataclasses import dataclass
import torch

@dataclass
class BaseConfig:
    lr: float = 1e-3
    steps: int = 16000
    batch_size : int = 64
    num_layers: int = 12
    num_heads = 12
    d_model: int = 768
    d_ff: int = 2048
    rope_theta = 10000
    device: torch.device = "cuda"
    seed: int = 42
    context_length: int = 256

base_config = BaseConfig()