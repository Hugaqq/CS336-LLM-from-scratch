import torch
from torch import nn
from einops import einsum, rearrange
from math import sqrt

class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int, 
            device : torch.device | None = None, 
            dtype : torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device=self.device, dtype=self.dtype))
        self.reset_parameter(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum( x, self.weight,"... d_in, d_out d_in -> ... d_out")

    def reset_parameter(
            self,
            x : torch.Tensor, 
            expectation: torch.Tensor | None = None
            ) -> torch.Tensor:
        if expectation == None:
            delta = sqrt(2 / (self.in_features + self.out_features))
            torch.nn.init.trunc_normal_(x, mean = 0.0, std = delta, a = -3.0 * delta, b = 3.0 * delta)
        else : 
            x.data.copy_(expectation.to(device=self.device, dtype= self.dtype))