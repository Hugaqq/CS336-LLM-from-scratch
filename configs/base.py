from dataclasses import dataclass, field
import torch
from pathlib import Path

@dataclass
class BaseConfig:
    lr: float = 1e-3
    steps: int = 16000
    batch_size : int = 64
    num_layers: int = 12
    num_heads: int = 12
    d_model: int = 768
    d_ff: int = 2048
    rope_theta:int = 10000
    seed: int = 42
    context_length: int = 256
    specialtokens_list: list[str] = field(default_factory = lambda: ["<|endoftext|>"])

    data_root: Path = Path("./data")
    dataset: str = "owt"
    checkpoint_root: Path = Path("./checkpoint")

    vocab_filepath: Path = field(init=False)
    merges_filepath: Path = field(init=False)
    train_file: Path = field(init=False)
    valid_file: Path = field(init=False)
    train_tokens: Path = field(init=False)
    valid_tokens: Path = field(init=False)

    def __post_init__(
            self
    ):
        self.vocab_filepath = self.data_root / self.dataset / (self.dataset + "_vocab.json")
        self.merges_filepath = self.data_root / self.dataset / (self.dataset + "_merges.txt")
        self.train_file = self.data_root / self.dataset / (self.dataset + "_train.txt")
        self.valid_file = self.data_root / self.dataset / (self.dataset + "_valid.txt")
        self.train_tokens = self.data_root / self.dataset / (self.dataset + "_train_tokens.npy")
        self.valid_tokens = self.data_root / self.dataset / (self.dataset + "_valid_tokens.npy")

base_config = BaseConfig()