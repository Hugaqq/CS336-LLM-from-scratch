from cs336_basics.BPE.bpe import bpe_train
from cs336_basics.Tokenizer.tokenizer import write_disk

def main():
    vocab, merges = bpe_train(
           "./data/TinyStoriesV2-GPT4-train.txt",
           10000,
           ["<|endoftext|>"]
       )
    write_disk("./data/TinyStoriesV2-GPT4-vocab.json","./data/TinyStoriesV2-GPT4-merges.txt", vocab, merges)

if __name__ == "__main__":
   main()