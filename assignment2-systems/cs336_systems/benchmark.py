import torch
import argparse
from timeit import default_timer
from cs336_systems.config import Config
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

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
   
   
   for _ in range(running_config.warm_up_times):
      if forward_only is False:
         opt.zero_grad()
      res = benchmark_transformer_lm(x)
      if forward_only is False:
         loss = cross_entropy(res, targets)
         loss.backward()
      if full is True:
         opt.step()

   torch.cuda.synchronize()

   t_whole = []

   for step in range(running_config.tests):
      torch.cuda.synchronize()
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

      torch.cuda.synchronize()
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