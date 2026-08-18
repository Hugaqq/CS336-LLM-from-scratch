# train_owt.py（仓库根目录，临时脚本不入库也行）
from cs336_basics.bpe import bpe_train
from cs336_basics.tokenizer import write_disk

def main():
    vocab, merges = bpe_train(
        "./data/owt_train.txt",
        32000,
        ["<|endoftext|>"],
    )
    write_disk("./data/owt_vocab.json", "./data/owt_merges.txt", vocab, merges)

    # 交付物检查：最长 token
    longest = max(vocab.values(), key=len)
    print(len(longest), longest.decode("utf-8", errors="replace"))

if __name__ == "__main__":
    main()