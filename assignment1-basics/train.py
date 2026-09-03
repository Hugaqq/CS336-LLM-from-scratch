import numpy as np
import torch
import cs336_basics.checkpointing as ckpt
from cs336_basics.optimizer import AdamW, scheduler, gradient_clip
from cs336_basics.data_loader import get_batch
import cs336_basics.transformer_component as tc
from cs336_basics.loss import cross_entropy
from einops import rearrange
from pathlib import Path
import argparse
import wandb
import random
from configs.base import BaseConfig, base_config
from scripts.tokenize_dataset import tokenize

def train(
    running_config: BaseConfig = base_config,
    device: torch.device = "cuda",
    use_wandb: bool = False
):
    torch.manual_seed(running_config.seed)
    np.random.seed(running_config.seed)
    random.seed(running_config.seed)
    valid_test_times = 20

    train_array, valid_array, vocab_size = tokenize(running_config)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    start_step = 0
    files = list(Path(running_config.checkpoint_root).glob("ckpt_step_*.pt"))
    if files:
        latest = max(files, key = lambda p: int(p.stem.rsplit("_", 1)[-1]))

    run_id_file = Path("wandb_run_id.txt")
            
    Transformer = tc.transformer_lm(vocab_size, running_config.context_length, running_config.num_layers, running_config.d_model, running_config.d_ff, running_config.rope_theta, running_config.num_heads)
    Transformer.to(device)
    Transformer = torch.compile(Transformer)

    opt = AdamW(Transformer.parameters())
        
    if files:
        start_step = ckpt.load_checkpoint(latest, Transformer, opt, device)

    if use_wandb is True:
        wandb_config = {"d_model": running_config.d_model, "lr": running_config.lr, "batch_size": running_config.batch_size, "steps": running_config.steps}
        if run_id_file.exists():
            run_id = run_id_file.read_text().strip()
            wandb.init(
                project = "cs336-assignment",
                config = wandb_config,
                resume = "allow",
                id = run_id
            )
        else:
            wandb.init(
                project = "cs336-assignment",
                name = "baseline_0",
                config = wandb_config,
            )
            run_id_file.write_text(wandb.run.id)
    
    for t in range(start_step, running_config.steps):
        in_dices, targets = get_batch(train_array, running_config.batch_size, running_config.context_length, device = device)
        targets = rearrange(targets, "batch_size seq_len -> (batch_size seq_len)")

        opt.zero_grad()

        train_lr = scheduler(t, running_config.lr, 1e-4, 0.01 * running_config.steps, running_config.steps)
        for group in opt.param_groups:
            group["lr"] = train_lr

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = Transformer(in_dices)
            x = rearrange(x, "batch_size  max_seq_len vocab_size -> (batch_size  max_seq_len) vocab_size")
            loss = cross_entropy(x.to(torch.float32), targets)
        loss.backward()
        gradient_clip(list(Transformer.parameters()),1.0)
        opt.step()  
        if t % 2000 == 0:
            Transformer.eval()
            print(f"iteration {t} with train loss : {loss.item()}")
            with torch.no_grad():
                valid_loss_list = []
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    for i in range(valid_test_times):
                        valid_in_dices, valid_targets = get_batch(valid_array, running_config.batch_size, running_config.context_length, device)
                        valid_targets = rearrange(valid_targets, "batch_size seq_len -> (batch_size seq_len)")
                        valid_x = Transformer(valid_in_dices)
                        valid_x = rearrange(valid_x, "batch_size max_seq_len vocab_size -> (batch_size max_seq_len) vocab_size")
                        tmp_loss = cross_entropy(valid_x.to(torch.float32), valid_targets)
                        valid_loss_list.append(tmp_loss)
                    valid_loss = torch.stack(valid_loss_list, dim = 0).mean(dim = 0)
            print(f"iteration {t} with valid loss : {valid_loss}")

            if t != 0 and use_wandb is True:
                wandb.log({"train/loss": loss.item(), "step":t})
                wandb.log({"val/loss": valid_loss.item(), "step":t})
            if t != 0:
                ckpt.save_checkpoint(Transformer, opt, t, running_config.checkpoint_root / f"ckpt_step_{t}.pt")
            Transformer.train()
    ckpt.save_checkpoint(Transformer, opt, running_config.steps, running_config.checkpoint_root / "ckpt_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", action = "store_true")
    parser.add_argument("--device", type = torch.device, default="cuda")    
    args = parser.parse_args()
    running_config = base_config
    train(running_config, args.device, args.use_wandb)