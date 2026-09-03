import torch
from cs336_basics.Tokenizer.tokenizer import tokenizer
from cs336_basics.checkpointing import load_checkpoint
from cs336_basics.transformer_component import transformer_lm
from cs336_basics.optimizer import AdamW
from configs.base import base_config, BaseConfig

def main(
        running_config : BaseConfig = base_config,
        device: torch.device = "cuda",
        max_new_tokens: int = 2048,
        temperature: int = 1,
        top_p:float = 0.3
        ):
        infer_tokenizer = tokenizer.from_files(running_config.vocab_filepath, running_config.merges_filepath, running_config.specialtokens_list)
        infer_transformer_lm = transformer_lm(
                                len(infer_tokenizer.vocab),
                                running_config.context_length,
                                running_config.num_layers,
                                running_config.d_model,
                                running_config.d_ff,
                                running_config.rope_theta,
                                running_config.num_heads
                                ).to(device)
        opt = AdamW(infer_transformer_lm.parameters())
        load_checkpoint(running_config.checkpoint_root / "ckpt_final.pt",
                        infer_transformer_lm,
                        opt,
                        device)
        prompts = input()
        prompts_id = torch.tensor(infer_tokenizer.encode(prompts), dtype = torch.long).to(device)
        output_token_id_list = []
        for new_token_id in generate(infer_transformer_lm, prompts_id, max_new_tokens, temperature, top_p, infer_tokenizer.reverse_vocab["<|endoftext|>".encode("utf-8")], running_config.context_length):
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
    running_config = base_config
    main(base_config)