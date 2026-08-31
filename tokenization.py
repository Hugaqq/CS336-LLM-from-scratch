import numpy as np
from cs336_basics.BPE.bpe import find_chunk_boundaries
from cs336_basics.Tokenizer.tokenizer import tokenizer
import os
import multiprocessing as mp
from configs.base import BaseConfig


def tokenize(running_config: BaseConfig):
    real_tokenizer = tokenizer.from_files(running_config.vocab_filepath, running_config.merges_filepath, running_config.specialtokens_list)
    vocab_size = len(real_tokenizer.vocab)

    if running_config.train_tokens.exists() is False:
        parallel_encode(real_tokenizer, running_config.specialtokens_list, running_config.train_file)
    train_array = np.load(running_config.train_tokens, mmap_mode = "r")
    
    if running_config.valid_tokens.exists() is False:
        parallel_encode(real_tokenizer, running_config.specialtokens_list, running_config.valid_file)
    valid_array = np.load(running_config.valid_tokens, mmap_mode = "r")

    return train_array, valid_array, vocab_size

def parallel_encode(
        real_tokenizer: tokenizer,
        specialtokens_list: list[str],
        input_path: str, 
        ) -> str:
    num_processes = max(1, os.cpu_count() * 16)
    num_chunks = num_processes

    with open(input_path, "rb") as file:
        specialtokens_list = [token.encode("utf-8") for token in specialtokens_list]
        points_list = find_chunk_boundaries(file, num_chunks, specialtokens_list)
    boundaries_list = zip(points_list[:-1], points_list[1:])
    tasks = []

    for boundary in boundaries_list:
        tasks.append((real_tokenizer, boundary[0], boundary[1], input_path))

    with mp.Pool(num_processes) as pool:
        results = list(pool.imap(tokenizer_encode_worker, tasks))
        whole_array = np.concatenate(results, axis = 0)
        npy_start_with = str(input_path).removesuffix(".txt")
        store_npy = np.lib.format.open_memmap(npy_start_with + "_tokens.npy", mode = "w+", dtype = np.uint16, shape = (len(whole_array),))
        store_npy[:] = whole_array
        store_npy.flush()

def tokenizer_encode_worker(args):
    real_tokenizer, start, end, input_path = args
    res_array = np.array([], dtype = np.uint16)
    if start != end:
        with open(input_path, "rb") as file:
            file.seek(start)
            text = file.read(end - start).decode("utf-8", errors = "ignore")
            res_array = np.array(real_tokenizer.encode(text), dtype = np.uint16)
    return res_array