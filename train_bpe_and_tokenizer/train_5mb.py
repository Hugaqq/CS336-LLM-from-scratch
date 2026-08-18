from cs336_basics.bpe import bpe_train
if __name__ == "__main__":
    vocab, merges = bpe_train(
        "./data/TinyStoriesV2-GPT4-train.txt",
        10000,
        ["<|endoftext|>"]
    )