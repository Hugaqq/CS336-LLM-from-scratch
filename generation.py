import torch
from cs336_basics.Tokenizer.tokenizer import tokenizer
from cs336_basics.checkpointing import load_checkpoint
from cs336_basics.transformer_component import transformer_lm

def main():
        context_length = 256
        num_layers = 4
        d_model = 512
        d_ff = int((d_model * 8 / 3 // 64) * 64)
        device = "cuda:0"
        dtype = torch.float32
        rope_theta = 10000
        num_heads = 16
        max_new_tokens = 2048
        temperature = 0.8
        top_p = 0.3
        infer_tokenizer = tokenizer.from_files("./data/TinyStoriesV2-GPT4-vocab.json", "./data/TinyStoriesV2-GPT4-merges.txt", ["<|endoftext|>"])
        infer_transformer_lm = transformer_lm(
                                len(infer_tokenizer.vocab),
                                context_length,
                                num_layers,
                                d_model,
                                d_ff,
                                rope_theta,
                                num_heads
                                )
        opt = torch.optim.Optimizer
        load_checkpoint("./data/checkpoint/ckpt_final.pt",
                        infer_transformer_lm,
                        opt,
                        device)
        prompts = input()
        prompts_id = torch.tensor(infer_tokenizer.encode(prompts), dtype = torch.uint16)
        output_token_id_list = []
        for new_token_id in generate(infer_transformer_lm, prompts_id, max_new_tokens, temperature, top_p, infer_tokenizer.reverse_vocab["<|endoftext|>"].decode("utf-8"), context_length):
            output_token_id_list.append(new_token_id)
        output_str = infer_tokenizer.decode(output_token_id_list)
        print(output_str)


def generate(model, prompts_ids, max_new_tokens, temperature, top_p, eos_token_id, context_length):
    generate_num = max_new_tokens
    if top_p == 0:
        temperature = 0
    with torch.no_grad():
        for i in range(generate_num):
            if temperature != 0:
                v_prob = torch.softmax(model(prompts_ids)[-1] / temperature, dim = -1)
                values, indices = torch.sort(v_prob, dim = -1, descending = True)
                values_sum = torch.cumsum(values, dim = -1)
                values_cond = values_sum - values
                prob_target_ids = indices[values_cond < top_p]
                v_prob_selected = v_prob[prob_target_ids]
                
                target_idx = prob_target_ids[torch.multinomial(v_prob_selected, num_samples = 1)]
            else:
                target_idx = torch.reshape(torch.argmax(model(prompts_ids)[-1], dim = -1), (1,))
            
            if target_idx == eos_token_id:
                break
            yield target_idx.item()
            if context_length <= len(prompts_ids):
                prompts_ids = torch.cat((prompts_ids[1:], target_idx), dim = 0)
            else :
                prompts_ids = torch.cat((prompts_ids, target_idx), dim = 0)

if __name__ == "__main__":
    main()