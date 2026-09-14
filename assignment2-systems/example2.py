import torch.nn as nn
import random
import torch
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

class ToyModel(nn.Module):
 def __init__(self, in_features: int, out_features: int):
    super().__init__()
    self.fc1 = nn.Linear(in_features, 10, bias=False)
    self.ln = nn.LayerNorm(10)
    self.fc2 = nn.Linear(10, out_features, bias=False)
    self.relu = nn.ReLU()
 def forward(self, x):
    print(f"initial type : {x.dtype}")
    x = self.relu(self.fc1(x))

    print(f"relu_gate(x) type : {x.dtype}")
    if self.fc1.weight is not None:
        print(f"first Linear Layer weights type : {self.fc1.weight.dtype}")
    if self.fc1.bias is not None:
        print(f"first Linear Layer bias type : {self.fc1.bias.dtype}")

    x = self.ln(x)

    print(f"ln(x) type : {x.dtype}")
    if self.ln.bias is not None:
        print(f"LayerNorm bias type : {self.ln.bias.dtype}")
    if self.ln.weight is not None:
       print(f"LayerNorm weight type : {self.ln.weight.dtype}")

    x = self.fc2(x)

    print(f"Linear(x) type : {x.dtype}") 
    if self.fc2.weight is not None:
        print(f"second Linear Layer weights type : {self.fc2.weight.dtype}")
    if self.fc2.bias is not None:
        print(f"second Linear Layer bias type : {self.fc2.bias.dtype}")

    return x

def main():
  device = "cuda:0"
  torch.manual_seed(42)
  model = ToyModel(4, 4).to(device, torch.float32)
  x = torch.randn(4,).to(device, torch.float32)
  targets = torch.randint(0, 4, (), dtype = torch.int64).to(device)
  opt = AdamW(model.parameters())
  with torch.autocast(device_type="cuda",dtype = torch.float16):
    x = model(x)
    loss = cross_entropy(x, targets)
    print(f"logits type : {x.dtype}")
    print(f"loss type : {loss.data.dtype}")
  loss.backward()
  i = 0
  for group in opt.param_groups:
    for p in group["params"]:
        i += 1
        print(f"gradients {i} type : {p.grad.data.dtype}")


if __name__ == "__main__":
  main()