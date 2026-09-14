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

def benchmark_test_compute_performance(
      running_config: Config,
      forward_only: bool = False,
      fandb: bool = False,
      full: bool = False,
      autocast_flag: bool = False
      ):
   benchmark_transformer_lm = BasicsTransformerLM(running_config.vocab_size, running_config.context_length, running_config.d_model, running_config.num_layers, running_config.num_heads, running_config.d_ff).to(running_config.device, running_config.dtype)

   opt = AdamW(benchmark_transformer_lm.parameters())
   x = torch.randint(0, running_config.vocab_size, size = (running_config.batch_size, running_config.context_length)).to(device = running_config.device)
   targets = torch.randint(0, running_config.vocab_size, size = (running_config.batch_size, running_config.context_length)).to(device = running_config.device)
   
   with nvtx.range("warm-up"):
      for _ in range(running_config.warm_up_times):
         if forward_only is False:
            opt.zero_grad()
         with torch.autocast(device_type="cuda", dtype = torch.bfloat16, enabled = autocast_flag):
            res = benchmark_transformer_lm(x)
            if forward_only is False:
               loss = cross_entropy(res, targets)
         if forward_only is False:
            loss.backward()
         if full is True:
            opt.step()
         del res

      torch.cuda.synchronize(device = running_config.device)

   t_whole = []

   
   for _ in range(running_config.tests):
      torch.cuda.synchronize(device = running_config.device)
      t0 = default_timer()
      with nvtx.range("measurement"):
         if forward_only is False:
            opt.zero_grad()

         with torch.autocast(device_type = "cuda", dtype = torch.bfloat16, enabled = autocast_flag):
            res = benchmark_transformer_lm(x)
            if forward_only is False:
               loss = cross_entropy(res, targets)

         if forward_only:
            del res
         
         if fandb:
            loss.backward()

         if full:
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

def benchmark_test_memory_performance(
      running_config: Config,
      forward_only: bool = False,
      fandb: bool = False,
      full: bool = False,
      autocast_flag: bool = False,
      pickle_suffix: str = "0"
      ):
   benchmark_transformer_lm = BasicsTransformerLM(running_config.vocab_size, running_config.context_length, running_config.d_model, running_config.num_layers, running_config.num_heads, running_config.d_ff).to(running_config.device, running_config.dtype)

   opt = AdamW(benchmark_transformer_lm.parameters())
   x = torch.randint(0, running_config.vocab_size, size = (running_config.batch_size, running_config.context_length)).to(device = running_config.device)
   targets = torch.randint(0, running_config.vocab_size, size = (running_config.batch_size, running_config.context_length)).to(device = running_config.device)
   
   for _ in range(running_config.warm_up_times):
      if forward_only is False:
         opt.zero_grad()
      with torch.autocast(device_type="cuda", dtype = torch.bfloat16, enabled = autocast_flag):
         res = benchmark_transformer_lm(x)
         if forward_only is False:
            loss = cross_entropy(res, targets)
      if forward_only is False:
         loss.backward()
      if full is True:
         opt.step()
      del res
      torch.cuda.synchronize(device = running_config.device)

   torch.cuda.memory._record_memory_history(max_entries = 1000000)
   
   for _ in range(running_config.tests):
      if forward_only is False:
         opt.zero_grad()

      with torch.autocast(device_type = "cuda", dtype = torch.bfloat16, enabled = autocast_flag):
         res = benchmark_transformer_lm(x)
         if forward_only is False:
            loss = cross_entropy(res, targets)

      if forward_only:
         del res
      
      if fandb:
         loss.backward()

      if full:
         loss.backward()
         opt.step()
      torch.cuda.synchronize(device = running_config.device)

   torch.cuda.memory._dump_snapshot(f"memory_snapshot.pickle_{pickle_suffix}")
   torch.cuda.memory._record_memory_history(enabled=None)
      

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument("--forward_only", action="store_true")
   parser.add_argument("--fandb", action = "store_true")
   parser.add_argument("--full", action = "store_true")
   parser.add_argument("--device", type = torch.device, default = "cuda:0")
   parser.add_argument("--context_length", type = int, default = 256)
   parser.add_argument("--size", type = str, default = "small")
   parser.add_argument("--compute_test", action = "store_true")
   parser.add_argument("--memory_test", action = "store_true")
   parser.add_argument("--autocast", action = "store_true")
   parser.add_argument("--suffix", type = str, default = "0")

   args = parser.parse_args()

   assert args.forward_only + args.fandb + args.full == 1, "Please choose only one method to decide the test part: only-forward , forward and backward or full process?"
   assert args.compute_test + args.memory_test != 0, "Please choose test a benchmark method : memory or compute"
   running_config = Config._from_size(args.size, device = args.device, context_length = args.context_length)

   if args.compute_test:
      benchmark_test_compute_performance(running_config, args.forward_only, args.fandb, args.full, args.autocast)
   if args.memory_test:
      benchmark_test_memory_performance(running_config, args.forward_only, args.fandb, args.full, args.autocast, args.suffix)

if __name__ == "__main__":
   main()