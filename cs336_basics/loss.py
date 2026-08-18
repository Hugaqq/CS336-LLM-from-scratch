import torch
from torch import Tensor
from math import exp
from jaxtyping import Float, Bool, Int
def cross_entropy(
        objects: Float[Tensor,"batch_size vocab_size"],
        targets: Int[Tensor, "batch_size"]
):
    B = objects.shape[-2]
    m = torch.amax(objects, -1)
    delta = objects - m.unsqueeze(-1)

    l = m + torch.log(torch.sum(torch.exp(delta), -1)) - objects[torch.arange(objects.shape[0]), targets]
    return torch.mean(l, 0)