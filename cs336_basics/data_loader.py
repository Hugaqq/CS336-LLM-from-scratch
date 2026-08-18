import torch
import numpy as np
import numpy.typing as npt
from torch import Tensor
from jaxtyping import Int, Float, Bool

def get_batch(
        x: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: str
) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    starts = np.random.randint(0, x.shape[0] - context_length, size = batch_size)
    inputs = np.stack(list(x[i: i + context_length] for i in starts))
    inputs = torch.tensor(inputs,dtype = torch.long, device = device)
    objects = np.stack(list(x[i + 1: i + context_length + 1] for i in starts))
    objects = torch.tensor(objects,dtype = torch.long, device = device)
    return inputs, objects