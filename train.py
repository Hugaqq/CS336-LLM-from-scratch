import numpy as np
import torch
import cs336_basics.checkpointing as ckpt
from cs336_basics.optimizer import AdamW, scheduler, gradient_clip
from cs336_basics.data_loader import get_batch
import cs336_basics.transformer_component as tc
from cs336_basics.BPE.bpe import find_chunk_boundaries
from cs336_basics.Tokenizer.tokenizer import tokenizer
from cs336_basics.loss import cross_entropy
from einops import rearrange
from pathlib import Path
import os
import multiprocessing as mp
import time
import argparse
import wandb
import random

def train(
        lr: float | None = None, 
        steps: int = 1000,
        batch_size: int = 32,
        num_layers: int = 4,
        d_model: int = 512,
        device: torch.device = "mps",
        dtype: torch.dtype = torch.float32,
        seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs("./data/checkpoint", exist_ok=True)
    vocab_filepath = "./data/TinyStoriesV2-GPT4-vocab.json"
    merges_fliepath = "./data/TinyStoriesV2-GPT4-merges.txt"
    train_file = "./data/TinyStoriesV2-GPT4-train.txt"
    valid_file = "./data/TinyStoriesV2-GPT4-valid.txt"
    real_tokenizer = tokenizer.from_files(vocab_filepath, merges_fliepath, ["<|endoftext|>"])
    context_length = 256

    if Path(train_file + "encode_results.npy").exists() is False:
        parallel_encode(train_file, vocab_filepath, merges_fliepath)
    train_array = np.load(train_file + "encode_results.npy", mmap_mode = "r")
    
    if Path(valid_file + "encode_results.npy").exists() is False:
        parallel_encode(valid_file, vocab_filepath, merges_fliepath)
    valid_array = np.load(valid_file + "encode_results.npy", mmap_mode = "r")

    dir_ = "./data/checkpoint"
    start_step = 0
    files = list(Path(dir_).glob("ckpt_step_*.pt"))
    if files:
        latest = max(files, key = lambda p: int(p.stem.rsplit("_", 1)[-1]))

    run_id_file = Path("wandb_run_id.txt")

    d_ff = int((d_model * 8 / 3 // 64) * 64)
            
    Transformer = tc.transformer_lm(len(real_tokenizer.vocab), context_length, num_layers, d_model, d_ff, 10000, 16)
    Transformer.to(device)
    opt = AdamW(Transformer.parameters())
        
    config = {"d_model": d_model, "lr": lr, "batch_size": batch_size, "steps": steps}
    if files:
        start_step = ckpt.load_checkpoint(latest, Transformer, opt, device)
    # if run_id_file.exists():
    #     run_id = run_id_file.read_text().strip()
    #     wandb.init(
    #         project = "cs336-assignment",
    #         config = config,
    #         resume = "allow",
    #         id = run_id
    #     )
    # else:
    #     wandb.init(
    #         project = "cs336-assignment",
    #         name = "baseline_0",
    #         config = config,
    #     )
    #     run_id_file.write_text(wandb.run.id)
        
    valid_test_times = 20
    
    for t in range(start_step, steps):
        in_dices, targets = get_batch(train_array, batch_size, context_length, device = device)
        targets = rearrange(targets, "batch_size seq_len -> (batch_size seq_len)")

        opt.zero_grad()

        train_lr = scheduler(t, lr, 1e-4, 0.1 * steps, steps)
        for group in opt.param_groups:
            group["lr"] = train_lr

        x = Transformer.forward(in_dices)
        x = rearrange(x, "batch_size  max_seq_len vocab_size -> (batch_size  max_seq_len) vocab_size")

        loss = cross_entropy(x, targets)
        loss.backward()
        gradient_clip(list(Transformer.parameters()),1.0)
        opt.step()  
        if t==0:
            print(f"iteration {t} with train loss : {loss.item()}")
            # print(f"iteration {t} with valid loss : {valid_loss}")
        if t % 100 == 0 and t != 0:
            Transformer.eval()
            print(f"iteration {t} with train loss : {loss.item()}")
            with torch.no_grad():
                valid_loss_list = []
                for i in range(valid_test_times):
                    valid_in_dices, valid_targets = get_batch(valid_array, batch_size, context_length, device)
                    valid_targets = rearrange(valid_targets, "batch_size seq_len -> (batch_size seq_len)")
                    valid_x = Transformer.forward(valid_in_dices)
                    valid_x = rearrange(valid_x, "batch_size max_seq_len vocab_size -> (batch_size max_seq_len) vocab_size")
                    tmp_loss = cross_entropy(valid_x, valid_targets)
                    valid_loss_list.append(tmp_loss)
                valid_loss = torch.stack(valid_loss_list, dim = 0).mean(dim = 0)
            print(f"iteration {t} with valid loss : {valid_loss}")

            # wandb.log({"train/loss": loss.item(), "step":t})
            # wandb.log({"val/loss": valid_loss.item(), "step":t})

            checkpoint_path = f"./data/checkpoint/ckpt_step_{t}.pt"
            ckpt.save_checkpoint(Transformer, opt, t, checkpoint_path)
            Transformer.train()
    ckpt.save_checkpoint(Transformer, opt, steps, "./data/checkpoint/ckpt_final.pt")
def parallel_encode(input_path: str, vocab_filepath: str, merges_filepath: str) -> str:
    num_processes = os.cpu_count()
    num_chunks = num_processes * 16

    with open(input_path, "rb") as file:
        points_list = find_chunk_boundaries(file, num_chunks, "<|endoftext|>".encode("utf-8"))
    boundaries_list = zip(points_list[:-1], points_list[1:])
    tasks = []

    for boundary in boundaries_list:
        tasks.append((boundary[0], boundary[1], input_path, vocab_filepath, merges_filepath))

    with mp.Pool(num_processes) as pool:
        results = list(pool.imap(tokenizer_encode_worker, tasks))
        whole_array = np.concatenate(results, axis = 0)
        store_npy = np.lib.format.open_memmap(input_path + "encode_results.npy", mode = "w+", dtype = np.uint16, shape = (len(whole_array),))
        store_npy[:] = whole_array
        store_npy.flush()

def tokenizer_encode_worker(args):
    start, end, input_path, vocab_filepath, merges_filepath = args
    if start == end:
        return np.array([], dtype = np.uint16)
    else:
        real_tokenizer = tokenizer.from_files(vocab_filepath, merges_filepath, ["<|endoftext|>"])
        pre_array = []
        with open(input_path, "rb") as file:
            file.seek(start)
            text = file.read(end - start).decode("utf-8", errors = "ignore")
            return np.array(real_tokenizer.encode(text), dtype = np.uint16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--step", type = int, default = 8000)
    parser.add_argument("--seed",type = int, default = 42)
    args = parser.parse_args()
    train(lr = args.lr, steps = args.step, seed = args.seed, device = "cuda:5")