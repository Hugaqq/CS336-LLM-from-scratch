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
        optimizer: torch.optim.Optimizer,
        device : torch.device | None = None
    ) -> int | None:
    ckpt = torch.load(src, map_location = device)

    now_torch_compile_flag = False
    past_torch_compile_flag = False
    for key, value in ckpt["model"].items():
        if key.startwith("_orig_mod."):
            past_torch_compile_flag = True
        break
    for key, value in model.state_dict().items():
        if key.startswith("_orig_mod."):
            now_torch_compile_flag = True
        break

    l = len("_orig_mod.")
    if now_torch_compile_flag == False and past_torch_compile_flag == True:
        for key in list(ckpt["model"]):
            ckpt["model"][key[l:]] = ckpt["model"].pop(key)
    if now_torch_compile_flag == True and past_torch_compile_flag == False:
        for key in list(ckpt["model"]):
            ckpt["model"]["_orig_mod." + key] = ckpt["model"].pop(key)

    if ckpt.get("model", None) is not None:
        missing_keys, unexpected_keys = model.load_state_dict(ckpt.get("model")) 
        if missing_keys != [] or unexpected_keys != []:
            raise ValueError
        

    if ckpt.get("optimizer", None) is not None:
        optimizer.load_state_dict(ckpt.get("optimizer"))

    return ckpt.get("iteration", None)