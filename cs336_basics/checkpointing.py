import torch
import os
import typing

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer, 
        iteration: int, 
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    ) -> None:

    obj = {
        "model" : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "iteration": iteration
    }

    torch.save(obj, out)

def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer
    ) -> int | None:
    ckpt = torch.load(src)
    if ckpt.get("model", None) is not None:
        model.load_state_dict(ckpt.get("model"))

    if ckpt.get("optimizer", None) is not None:
        optimizer.load_state_dict(ckpt.get("optimizer"))

    return ckpt.get("iteration", None)