import torch
import argparse
import cs336_basics
import einx
import math
from einops import einsum, rearrange
from cs336_basics.nn_utils import softmax
import torch.cuda.nvtx as nvtx
from torch import Tensor
from timeit import default_timer
from jaxtyping import Float, Int, Bool
from cs336_systems.config import Config
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
   Q: Float[Tensor, "... queries d_k"],
   K: Float[Tensor, "... keys    d_k"],
   V: Float[Tensor, "... keys    d_v"],
   mask: Bool[Tensor, "... queries d_v"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
   d_k = K.shape[-1]
   with nvtx.range("computing attention scores"):
      attention_scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)

   if mask is not None:
      attention_scores = torch.where(mask, attention_scores, float("-inf"))

   with nvtx.range("computing softmax"):
      attention_weights = softmax(attention_scores, dim = -1)
   
   with nvtx.range("final matmul"):
      res = einsum(attention_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
   return res

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

def benchmark_test(
      running_config: Config,
      forward_only: bool = False,
      fandb: bool = False,
      full: bool = False
      ):
   benchmark_transformer_lm = BasicsTransformerLM(running_config.vocab_size, running_config.context_length, running_config.d_model, running_config.num_layers, running_config.num_heads, running_config.d_ff).to(running_config.device, running_config.dtype)

   opt = AdamW(benchmark_transformer_lm.parameters())
   x = torch.randint(0, running_config.vocab_size, size = (running_config.batch_size, running_config.context_length)).to(device = running_config.device)
   targets = torch.randint(0, running_config.vocab_size, size = (running_config.batch_size, running_config.context_length)).to(device = running_config.device)
   
   with nvtx.range("warm-up"):
      for _ in range(running_config.warm_up_times):
         if forward_only is False:
            opt.zero_grad()
         res = benchmark_transformer_lm(x)
         if forward_only is False:
            loss = cross_entropy(res, targets)
            loss.backward()
         if full is True:
            opt.step()

   torch.cuda.synchronize(device = running_config.device)

   t_whole = []

   for _ in range(running_config.tests):
      torch.cuda.synchronize(device = running_config.device)
      t0 = default_timer()
      if forward_only:
         res = benchmark_transformer_lm(x)
        
      if fandb:
         opt.zero_grad()
         res = benchmark_transformer_lm(x)
         loss = cross_entropy(res, targets)
         loss.backward()

      if full:
         opt.zero_grad()
         res = benchmark_transformer_lm(x)
         loss = cross_entropy(res, targets)
         loss.backward()
         opt.step()

      torch.cuda.synchronize(device = running_config.device)
      t1 = default_timer()
      t_whole.append(t1 - t0)
   
   t_whole = torch.Tensor(t_whole)
   mean_t = t_whole.mean(dim = -1)
   std_t = t_whole.std()
   print(f"test results: mean_t :{mean_t}")
   print(f"              std_t :{std_t}")

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument("--forward_only", action="store_true")
   parser.add_argument("--fandb", action = "store_true")
   parser.add_argument("--full", action = "store_true")
   args = parser.parse_args()
   if args.forward_only + args.fandb + args.full != 1:
      raise ValueError
   running_config = Config()
   benchmark_test(running_config, args.forward_only, args.fandb, args.full)

if __name__ == "__main__":
   main()