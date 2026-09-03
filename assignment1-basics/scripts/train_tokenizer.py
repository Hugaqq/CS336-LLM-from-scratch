from cs336_basics.BPE.bpe import bpe_train
from cs336_basics.Tokenizer.tokenizer import write_disk
from configs.base import base_config, BaseConfig

def main(running_config : BaseConfig = base_config):
    input_path = running_config.train_file
    vocab, merges = bpe_train(
        input_path,
        32000,
        running_config.specialtokens_list
    )
    write_disk(running_config.vocab_filepath, running_config.merges_filepath, vocab, merges)

    longest = max(vocab.values(), key = len)
    print(len(longest), longest.decode("utf-8", errors = "replace"))

if __name__ == "__main__":
    main(base_config)