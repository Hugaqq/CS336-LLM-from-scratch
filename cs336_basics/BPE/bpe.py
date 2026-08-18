import regex as re
import os
import collections as col
import time
import multiprocessing as mp
from typing import BinaryIO
import heapq

def bpe_train(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    start = time.perf_counter()
    special_tokens = sort_special_tokens(special_tokens)
    regex_pattern = get_regex_pattern(special_tokens)

    t0 = time.perf_counter()

    bytes_special_tokens = []
    for special_token in special_tokens:
        bytes_special_tokens.append(special_token.encode("utf-8"))

    num_processes = os.cpu_count()
    num_chunks = os.cpu_count() * 16
    with open(input_path, "rb") as file:
        points_list = find_chunk_boundaries(file, num_chunks, bytes_special_tokens)
    boundaries_list = zip(points_list[:-1], points_list[1:])
    tasks = []
    for boundary in boundaries_list:
        tasks.append((boundary[0], boundary[1], input_path, special_tokens, regex_pattern))

    with mp.Pool(num_processes) as pool:
        results = pool.imap_unordered(pretokenize_worker, tasks)

        pretoken_to_f_dict = col.defaultdict(int)
        pretoken_to_tokenlist_dict = col.defaultdict(list)
    
        for part_pretoken_to_f_dict in results: 
            for pretoken, f in part_pretoken_to_f_dict.items():
                pretoken_to_f_dict[pretoken] += f

    for pretoken, f in pretoken_to_f_dict.items():
        pretoken_to_tokenlist_dict[pretoken] = [pretoken[i: i + 1] for i in range(len(pretoken))]

    vocab = {idx : bytes([idx]) for idx in range(0, 256)} 
    cur_top_idx = len(vocab) - 1
    merges = []
    pair_to_f_dict = col.defaultdict(int)
    pair_to_pretoken_set_dict = col.defaultdict(set)
# Intialization for pair_to_f
    t1 = time.perf_counter()
    print(f"time of pretokenization: {t1 - t0:.2f} s", flush = True)
    for pretoken, plus_f in pretoken_to_f_dict.items():
        token_list = pretoken_to_tokenlist_dict[pretoken]
        for i in range(0, len(token_list) - 1):
            cur_pair = (token_list[i], token_list[i + 1])
            pair_to_f_dict[cur_pair] += plus_f
            pair_to_pretoken_set_dict[cur_pair].add(pretoken)
    t2 = time.perf_counter()
    print(f"time of pair initialization: {t2 - t1:.2f} s", flush = True)
# Merge and renew vocab

    for i in range(vocab_size - len(vocab) - len(special_tokens)):
        if len(pair_to_f_dict) == 0:
            break

        max_pair = get_max(pair_to_f_dict)
        if max_pair == None:
            break
        cur_top_idx += 1
        vocab.update({cur_top_idx : max_pair[0] + max_pair[1]}) 
        merges.append(max_pair)
        
        
        for pretoken in pair_to_pretoken_set_dict[max_pair]:
            token_list = pretoken_to_tokenlist_dict[pretoken]
            tmp_old_pair_to_f_dict = col.defaultdict(int)
            tmp_new_pair_to_f_dict = col.defaultdict(int)
            old_token_list_slices = []
            new_token_list_slices = []
            new_token_list = [] 

            k = 0
            while k < len(token_list):
                if k == len(token_list) - 1:
                    new_token_list.append(token_list[k])
                    break
                tmp_pair = (token_list[k], token_list[k + 1])
                if tmp_pair == max_pair:
                    old_token_list_slices.append([k - 1, k + 2])
                    new_token_list.append(token_list[k] + token_list[k + 1])
                    k += 1
                else:
                    new_token_list.append(token_list[k])
                k += 1

                
            old_token_list_slices = slice_merge(old_token_list_slices, len(token_list) - 1)

            newtoken = max_pair[0] + max_pair[1]
            for index in range(len(new_token_list)):
                if new_token_list[index] == newtoken:
                    new_token_list_slices.append([index - 1, index + 1])
            new_token_list_slices = slice_merge(new_token_list_slices, len(new_token_list) - 1)

            for old_slice in old_token_list_slices:
                for i in range(old_slice[0], old_slice[1]):
                    tmp_pair = (token_list[i], token_list[i + 1])
                    tmp_old_pair_to_f_dict[tmp_pair] += 1
            tmp_pretoken_to_pair_flagset = set()
            for i in range(len(new_token_list) - 1):
                tmp_pair = (new_token_list[i], new_token_list[i + 1])
                tmp_pretoken_to_pair_flagset.add(tmp_pair)
            for pair, motiply_f in tmp_old_pair_to_f_dict.items():
                if pair == max_pair :
                    continue
                pair_to_f_dict[pair] -= motiply_f * pretoken_to_f_dict[pretoken]
                if pair not in tmp_pretoken_to_pair_flagset:
                    pair_to_pretoken_set_dict[pair].discard(pretoken)
                if pair_to_f_dict[pair] == 0:
                    pair_to_f_dict.pop(pair)

            for new_slice in new_token_list_slices:
                for i in range(new_slice[0], new_slice[1]):
                    tmp_pair = (new_token_list[i], new_token_list[i + 1])
                    tmp_new_pair_to_f_dict[tmp_pair] += 1

            for pair, motiply_f in tmp_new_pair_to_f_dict.items():
                pair_to_f_dict[pair] += motiply_f * pretoken_to_f_dict[pretoken]
                pair_to_pretoken_set_dict[pair].add(pretoken)

            pretoken_to_tokenlist_dict[pretoken] = new_token_list 
        pair_to_f_dict.pop(max_pair)

    t3 = time.perf_counter()
    print(f"time of merge loops: {t3 - t2:.2f} s",flush = True)
    cur_top = len(vocab)
    for special_token in special_tokens:
        vocab.update({cur_top: special_token.encode("utf-8")})
        cur_top += 1

    end = time.perf_counter()
    print(f"total time: {end - start:.2f} s",flush = True)
    return vocab, merges

def pretokenize_worker(args):
    start, end, input_path, special_tokens, regex_pattern = args
    with open(input_path, "rb") as file:
        file.seek(start)
        text = file.read(end - start).decode("utf-8", errors = "ignore")

    if regex_pattern == "":
        by_special_text_list = [text]
    else :
        regex_pattern = "(" + regex_pattern + ")"
        by_special_text_list = re.split(regex_pattern, text)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    part_pretoken_to_f_dict = col.defaultdict(int)

    for text_slice in by_special_text_list:
        if text_slice in special_tokens:
            continue
        for pretoken_match in re.finditer(PAT, text_slice):
            pretoken = pretoken_match.group().encode("utf-8")
            part_pretoken_to_f_dict[pretoken] += 1

    del text
    return part_pretoken_to_f_dict

def get_max(pair_to_f_dict: list[tuple[bytes, bytes], int]) -> int:    
    max_key = None
    max_value = -1
    for k, v in pair_to_f_dict.items():
        if v > max_value:
            max_value = v
            max_key = k
        elif v == max_value and max_key < k:
            max_key = k
    return max_key

def slice_merge(list_slices, max_len):
    if list_slices == []:
        return []
    new_list_slices = [list_slices[0]]
    for i in range(1, len(list_slices)):
        compare_slice = new_list_slices.pop()
        if compare_slice[1] >= list_slices[i][0]:
            new_list_slices.append([compare_slice[0], list_slices[i][1]])
        else :
            new_list_slices.append(compare_slice)
            new_list_slices.append(list_slices[i])
    if new_list_slices[0][0] < 0:
        new_list_slices[0][0] = 0
    if new_list_slices[-1][1] > max_len :
        new_list_slices[-1][1] = max_len
    return new_list_slices
    
def get_regex_pattern(special_tokens):
    return  "|".join(re.escape(special_token) for special_token in special_tokens)
    
def sort_special_tokens(special_tokens):
    special_tokens_dict = {}
    for special_token in special_tokens:
        if special_tokens_dict.get(special_token, 0) != 0:
            continue
        special_tokens_dict.update({special_token : len(special_token)})
    new_special_tokens = sorted(special_tokens_dict, 
                                key = special_tokens_dict.get, 
                                reverse = True)
    return new_special_tokens

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)


    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            final_found_at = -1
            for special_token in split_special_tokens:
                found_at = mini_chunk.find(special_token)
                final_found_at = max(found_at, final_found_at)
            if final_found_at != -1:
                chunk_boundaries[bi] = initial_position + final_found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
