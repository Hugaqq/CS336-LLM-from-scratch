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

batch_size = 8
warm_up_times = 5
test_times = 100
device = "cuda"
is_compiled = True

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys   d_k"],
    V: Float[Tensor, "... keys   d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... queries d_model, ... keys d_model -> ... queries keys") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim = -1)

    with nvtx.range("final matmul"):
        res = einsum(attention_weights, V, "... queries keys, ... keys d_model -> ... queries d_model")

    return res

def main():
    compiled_annotated_scaled_dot_product_attention = torch.compile(annotated_scaled_dot_product_attention)
    if(is_compiled) :
        test_annotated_scaled_dot_product_attention = compiled_annotated_scaled_dot_product_attention
    else : test_annotated_scaled_dot_product_attention = annotated_scaled_dot_product_attention
    
    d_model_list = [16, 32, 64, 128]
    seq_len_list = [256, 1024, 4096, 8192, 16384]
    index = 0
    for d_model in d_model_list:
        for seq_len in seq_len_list:
            index += 1
            forward_timelist = []
            backward_timelist = []
            backward_before_memorylist = []

            upstream_res = torch.randn(batch_size, seq_len, d_model).to(device = device)

            test_Q = torch.randn([batch_size, seq_len, d_model], requires_grad=True, device = device)
            test_K = torch.randn([batch_size, seq_len, d_model], requires_grad=True, device = device)
            test_V = torch.randn([batch_size, seq_len, d_model], requires_grad=True, device = device)

            with nvtx.range("warm up"):
                for i in range(warm_up_times):
                    atten_res = test_annotated_scaled_dot_product_attention(test_Q, test_K, test_V)
                    atten_res.backward(gradient=upstream_res)
                    test_K.grad = None
                    test_Q.grad = None
                    test_V.grad = None

            with nvtx.range("real test"):
                for i in range(test_times):
                    torch.cuda.synchronize(device = device)
                    t0 = default_timer()
                    with nvtx.range("forwarding"):
                        atten_res = test_annotated_scaled_dot_product_attention(test_Q, test_K, test_V)
                    torch.cuda.synchronize(device = device)
                    t1 = default_timer()
                    
                    backward_before_memorylist.append(torch.cuda.memory_allocated(device = device) / (1024 ** 2))
                    
                    t2 = default_timer()
                    with nvtx.range("backwarding"):
                        atten_res.backward(gradient=upstream_res)
                    torch.cuda.synchronize(device = device)
                    t3 = default_timer()
                    
                    test_K.grad = None
                    test_Q.grad = None
                    test_V.grad = None
                    
                    forward_timelist.append(t1 - t0)
                    backward_timelist.append(t3 - t2)                    
                torch.cuda.synchronize(device = device)


            print(f"index: {index} with `d_model`: {d_model}, `seq_len`: {seq_len}")
            forward_timelist = torch.Tensor(forward_timelist)
            print("forward processing:")
            print(f"mean_t: {forward_timelist.mean(dim = -1)} s")
            print(f"std_t : {forward_timelist.std()} s")

            backward_timelist = torch.Tensor(backward_timelist)
            print("backward processing:")
            print(f"mean_t: {backward_timelist.mean(dim = -1)} s")
            print(f"std_t : {backward_timelist.std()} s")

            backward_before_memorylist = torch.Tensor(backward_before_memorylist)
            print("before backwarding memory occupation:")
            print(f"mean_mem: {backward_before_memorylist.mean(dim = -1)} MiB")
            print(f"std_mem : {backward_before_memorylist.std()} MiB")

                    
if __name__ == "__main__":
    main()
