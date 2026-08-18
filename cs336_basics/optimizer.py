import torch
from collections.abc import Callable, Iterable
from typing import Optional
from jaxtyping import Int, Float, Bool
import math


class sgd(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr = 1e-3
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr = 1e-3,
            weight_decay: float = 0.01,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8
    ):
        if weight_decay < 0:
            raise ValueError(f"Invalid learning rate: {weight_decay}")
        defaults = {"weight_decay": weight_decay}
        defaults.update({"lr": lr})
        defaults.update({"betas": betas})
        defaults.update({"eps": eps})
        super().__init__(params, defaults)

    def step(self ,closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                g = p.grad.data
                m = state.get("m", 0)
                v = state.get("v", 0)
                betas_pow = state.get("betas_pow", betas)
                next_m = betas[0] * m + (1 - betas[0]) * g
                next_v = betas[1] * v + (1 - betas[1]) * g * g

                next_m_hat = next_m / (1 - betas_pow[0])
                next_v_hat = next_v / (1 - betas_pow[1])

                p.data = (1 - lr * weight_decay) * p.data - lr * next_m_hat / (torch.sqrt(next_v_hat) + eps)

                state["t"] = t + 1
                state["m"] = next_m
                state["v"] = next_v
                state["betas_pow"] = (betas_pow[0] * betas[0], betas_pow[1] * betas[1])

        return loss


def scheduler(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w and T_w != 0:
        alpha_t = t / T_w * alpha_max
    elif (T_w == 0 or T_w <= t) and t <= T_c :
        alpha_t = alpha_min + (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (alpha_max - alpha_min) / 2
    elif t > T_c:
        alpha_t = alpha_min
    return alpha_t

def gradient_clip(
        parameters: Iterable[torch.nn.Parameter], 
        max_value: float,
        eps:float = 1e-6
        ):
    parameters = list(parameters)
    ltwo_norm_square = 0
    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is None:
                continue
            ltwo_norm_square += torch.sum(parameter.grad * parameter.grad)
        ltwo_norm = torch.sqrt(ltwo_norm_square)
    if ltwo_norm >= max_value:
        for parameter in parameters:
            if parameter.grad is None:
                continue
            parameter.grad.data *= max_value / (ltwo_norm + eps)
