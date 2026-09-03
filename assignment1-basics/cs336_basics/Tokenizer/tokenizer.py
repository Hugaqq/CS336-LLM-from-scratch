from abc import ABC
from collections.abc import Iterator, Iterable
import collections as col
import regex as re
import json
import math
import os
from typing import BinaryIO

def _get_real_token(
            password: dict[str, bytes],
            false_str_token : str
        ) -> bytes:
    real_token = b""
    for char in false_str_token:
        real_token += bytes([password[char]])
    return real_token 
    
def _gpt2_unicode_to_bytes():
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        characters = [chr(n) for n in cs]
        d1 = dict(zip(characters, bytes(bs)))
        d2 = dict(zip(bytes(bs), characters))
        return d1, d2

def write_disk(
        vocab_path : str,
        merges_path: str,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]]
    ) -> None:
        with open(merges_path, "w", encoding = "utf-8") as file:
            for l, r in merges:
                file.write(f"{_encode_token(l)} {_encode_token(r)}\n")
        with open(vocab_path, "w") as file:
            json.dump({_encode_token(v): i for i,v in vocab.items()}, 
                      file,
                      indent = 2,
                      ensure_ascii = False)

def _encode_token(token: bytes) -> str:
    _, byte_to_unicode = _gpt2_unicode_to_bytes()
    return "".join(byte_to_unicode[b] for b in token)

class tokenizer(ABC):
    def __init__(
        self,
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        
        self.special_tokens = special_tokens
        if self.special_tokens != None :
            self.special_tokens = self._sort_special_tokens(self.special_tokens)
        self.reverse_vocab = {}
        for k, v in self.vocab.items():
            self.reverse_vocab.update({v: k})
        cur_idx_top = len(self.vocab)
        if special_tokens != None:
            for special_token in self.special_tokens:
                if special_token.encode("utf-8") not in self.reverse_vocab:
                    self.vocab.update({cur_idx_top: special_token.encode("utf-8")})
                    self.reverse_vocab.update({special_token.encode("utf-8"): cur_idx_top})
                    cur_idx_top += 1
        self.merge_to_idx_dict = {}
        for i in range(len(self.merges)):
            self.merge_to_idx_dict[self.merges[i]] = i
        
    @classmethod
    def from_files(
            cls, 
            vocab_filepath: str, 
            merges_filepath: str, 
            special_tokens: list[str] | None = None
    ):
        password, _ = _gpt2_unicode_to_bytes()

        with open(vocab_filepath, "r", encoding = "utf-8") as file_vocab:
             vocab = json.load(file_vocab)

        vocab = {int(id): _get_real_token(password, token) for token, id in vocab.items()}

        merges = [] # merge : list[tuple(bytes)]
        with open(merges_filepath, "r", encoding = "utf-8") as file_merges:
            for line_merge in file_merges:
                merge_pair_list = line_merge.rstrip().split(" ")
                if merge_pair_list == []:
                    break
                token1 = _get_real_token(password, merge_pair_list[0])
                token2 = _get_real_token(password, merge_pair_list[1])
                merges.append((token1, token2))

        return cls(vocab, merges, special_tokens)

    def encode(
            self,
            text: str
        ) -> list[int] :
        
        if self.special_tokens == None :
            regex_pattern = ""
        else:
            # self.special_tokens = self._sort_special_tokens(self.special_tokens)
            regex_pattern = "|".join(re.escape(special_token) for special_token in self.special_tokens)

        if regex_pattern != "":
            regex_pattern = "(" + regex_pattern + ")"
            text_list = re.split(regex_pattern, text)
        else :
            text_list = [text]

        id_list = []
        pretoken_to_tokenlist_dict = {}

        for text_slice in text_list:
            if self.special_tokens != None and text_slice in self.special_tokens:
                id_list.append(self.reverse_vocab[text_slice.encode("utf-8")])
            else :
                PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                for pretoken in re.finditer(PAT, text_slice):
                    pretoken = pretoken.group().encode("utf-8")

                    if pretoken_to_tokenlist_dict.get(pretoken, None) == None:
                       
                        token_list = [pretoken[i: i + 1] for i in range(len(pretoken))]
                        
                        while True:
                            pair_list = list(zip(token_list[ : -1], token_list[1 : ]))

                            if pair_list == []:
                                break
                            
                            merge_pair = pair_list[0]
                            for pair in pair_list:
                                if self.merge_to_idx_dict.get(pair, math.inf) < self.merge_to_idx_dict.get(merge_pair, math.inf):
                                    merge_pair = pair

                            new_token_list = []

                            index = 0
                            while index < len(token_list):
                                if index == len(token_list) - 1:
                                    new_token_list.append(token_list[index])
                                    break
                                if (token_list[index], token_list[index + 1]) == merge_pair:
                                    new_token_list.append(token_list[index] + token_list[index + 1])
                                    index += 1
                                else:
                                    new_token_list.append(token_list[index])
                                index += 1

                            if self.merge_to_idx_dict.get(merge_pair, math.inf) == math.inf:
                                break

                            token_list = new_token_list

                        pretoken_to_tokenlist_dict.update({pretoken : token_list})


                    for token in pretoken_to_tokenlist_dict[pretoken]:
                        id_list.append(self.reverse_vocab[token])
        return id_list
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        string_buffer = ""
        chunk_size = 16 * 1024
        for line in iterable:
            string_buffer += line

            if len(string_buffer) >= chunk_size:
                found_out = -1
                if self.special_tokens != None:
                    for special_token in self.special_tokens:
                        if string_buffer.rfind(special_token) == -1:
                            continue
                        found_out = max(string_buffer.rfind(special_token) + len(special_token), found_out) 
                if found_out == -1:
                    last = None
                    for m in re.finditer(r"\s+", string_buffer):
                        last = m
                    if last != None:
                        found_out = max(last.start(), found_out)
                if found_out == -1:
                    continue
                encode_string = string_buffer[: found_out]
                string_buffer = string_buffer[found_out : len(string_buffer)]
            
                id_list = self.encode(encode_string)
                for id in id_list:
                    yield id
        if string_buffer != "":
            id_list = self.encode(string_buffer)
            for id in id_list:
                yield id 

    def decode(self, ids: list[int]) -> str :
        result_bytes = b"".join([self.vocab[id] for id in ids])
        result_str = result_bytes.decode("utf-8", errors = "replace")
        return result_str

    def _sort_special_tokens(self, special_tokens):
        special_tokens_to_len_dict = {}
        for special_token in special_tokens:
            if special_tokens_to_len_dict.get(special_token, 0) == 0:
                special_tokens_to_len_dict.update({special_token: len(special_token)})
        new_special_tokens = sorted(special_tokens,
                                key = special_tokens_to_len_dict.get,
                                reverse = True)
        return new_special_tokens