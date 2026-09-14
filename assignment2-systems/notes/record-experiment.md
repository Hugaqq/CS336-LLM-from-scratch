
## Results of benchmark

### 1. Problem (benchmarking_script):  Benchmarking Script (4 points)
### (a) Write a script to perform basic end-to-end benchmarking of the forward pass, backward pass, and optimizer step in your model. 
assignment2-systems/cs336_systems/benchmark.py

### (b) Time the forward, backward, and optimizer step for the model sizes described in Section 2.1.2. Use 5 warmup steps and compute the average and standard deviation of timings over 10 measurement steps. How long does a forward pass take? How about a backward pass? Do you see high variability across measurements, or is the standard deviation small?
#### warm-ups = 5 tests = 10
```bash
(cs336-systems) (base) ➜  cs336_systems git:(main) ✗ python benchmark.py --forward_only
test results: mean_t :0.014275452122092247
              std_t :0.0015516368439421058
(cs336-systems) (base) ➜  cs336_systems git:(main) ✗ python benchmark.py --fandb       
test results: mean_t :0.04899284243583679
              std_t :0.0038728900253772736
(cs336-systems) (base) ➜  cs336_systems git:(main) ✗ python benchmark.py --full 
test results: mean_t :0.05858033895492554
              std_t :0.00013281998690217733
```

In fact, std_t is often no more than 1/10 of mean_t. Especially in the full benchmark, the ratio is decreasing to 1/50. To summarize, I think it's very small standard deviation and the calculation processing is stable.

### (c) One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without the warm-up steps. How does this affect your results? Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be different?

#### warm-ups = 0 tests = 10
```bash
(cs336-systems) (base) ➜  cs336_systems git:(main) ✗ python benchmark.py --forward_only
test results: mean_t :0.04637736827135086
              std_t :0.0975906029343605
(cs336-systems) (base) ➜  cs336_systems git:(main) ✗ python benchmark.py --fandb       
test results: mean_t :0.09854761511087418
              std_t :0.15663406252861023
(cs336-systems) (base) ➜  cs336_systems git:(main) ✗ python benchmark.py --full
test results: mean_t :0.10870836675167084
              std_t :0.1551879495382309
```

#### warm-ups = 1 tests = 10
```bash
(cs336-systems) (base) ➜  cs336_systems git:(main) ✗ python benchmark.py --forward_only
test results: mean_t :0.01813264563679695
              std_t :0.006169969215989113
(cs336-systems) (base) ➜  cs336_systems git:(main) ✗ python benchmark.py --fandb       
test results: mean_t :0.04899342358112335
              std_t :0.0033256839960813522
(cs336-systems) (base) ➜  cs336_systems git:(main) ✗ python benchmark.py --full 
test results: mean_t :0.059127479791641235
              std_t :0.001966441050171852
```

I think the cold start is slower than the stable processing, which makes the mean_t and std_t increase.

### 2. Problem (nsys_profile):  Nsight Systems Profiling (5 points)

### Profile your forward pass, backward pass, and optimizer step using nsys with two model sizes from Table 1 of your choice as well as three power-of-two context lengths larger than 128, where the largest available size should be the longest context length you can fit in memory. Pick thecombinations you think would be the most interesting to look at. For each profile answer the following questions:
#### (a) What is the total time spent on your forward pass? Does it match what we had measuredbefore with the Python standard library?
approximately 21ms. little longer than python library timer. I think it's nsys profiler slow the whole process.

#### (b) What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes? (Hint:look at the “CUDA GPU Kernel Summary” under “Stats System View”, and filter using NVTX ranges to identify which parts of the model are responsible for which kernels.)

*  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>(T1::Params)

* 37
* yes(
#### (c) Although the vast majority of FLOPs take place in matrix multiplications, you will notice that several other kernels still take a non-trivial amount of the overall runtime. What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?

* void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
 3.7%
*  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)3>>(int, T2, T3) 1.9%
* Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
1.5%	1.257 ms	720	1.746 μs	1.472 μs	1.376 μs	2.848 μs	420 ns	void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3) 1.5%
#### (d) Profile running one complete training step with your implementation of AdamW (i.e., the forward pass, computing the loss and running a backward pass, and finally an optimizer step, as you’d do during training). How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?

* In fact, the fraction of time spent on matrix multiplication decreases compared to doing inference.
* In other kernels, the most outstanding change is that the data movement operation like copy and reduce increase significantly.
#### (e) Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?
In the runtime, The time for two operations is approximately the same. But in FLOPs, matrix multiplication operations within the self-attention layer should multiply with a factor $$\frac{4d}{5}$$.

### 3. Problem(mixed_precision_accumulation)

```python
import torch

def main():
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print(s)

    s = torch.tensor(0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)

if __name__ == "__main__":
    main()
```

```bash
❯ python example.py
tensor(10.0001)
tensor(9.9531, dtype=torch.float16)
tensor(10.0021)
tensor(10.0021)
```

Let's explain it.

First of all, 0.01 couldn't actually be expressed in fp16 and fp32, actually be (0.00999999977648258209228515625) in fp32 and (0.01000213623046875) in fp16.

So, the first and second result is not 10.0000 because the small delta between the actual value and the ideal value in fp16/fp32 and the rounding in fp16/fp32 accumulation.

### 4. Problem (benchmarking_mixed_precision):  Benchmarking Mixed Precision (2 points)

#### (a)Consider the following model:

```python
class ToyModel(nn.Module):
 def __init__(self, in_features: int, out_features: int):
 super().__init__()
 self.fc1 = nn.Linear(in_features, 10, bias=False)
 self.ln = nn.LayerNorm(10)
 self.fc2 = nn.Linear(10, out_features, bias=False)
 self.relu = nn.ReLU()
 def forward(self, x):
 x = self.relu(self.fc1(x))
 x = self.ln(x)
 x = self.fc2(x)
 return x 
```

### Suppose we are training the model on a GPU and that the model parameters are originally in FP32. We’d like to use autocasting mixed precision with FP16. What are the data types of:
- **the model parameters within the autocast context?** 
  - torch.float32
- **the output of the first feed-forward layer (ToyModel.fc1)?** 
  - torch.float16 ( devide the result between weights and result)
- **the output of layer norm (ToyModel.ln)?** 
  - torch.float32
- **the model’s predicted logits?** 
  - torch.float16
- **the loss?** 
  - torch.float32
- **the model’s gradients?**
  -  torch.float32

### 5.Problem (memory_profiling):  Memory Profiling (4 points)
### Profile your complete training step of forward pass, backward pass, and optimizer step of the xl model from `Table 1` with context lengths of 128 and 2048.
#### (a) Add an option to your profiling script to run your model through the memory profiler. It may be helpful to reuse some of your previous infrastructure (e.g., to activate mixed-precision, load specific model sizes, etc). Then, run your script to get a memory profile of the xl model when either doing inference only (just forward pass) or a full training step. What do your memory timelines look like? Can you tell which stage is running based on the peaks you see?
Deliverable: Two images of the “Active memory timeline” of an xl model, from the memory_viz tool: one for the forward pass, and one for running a full training step (forward and backward
passes, then optimizer step), and a 2-3 sentence response.
#### (b) What is the peak memory usage of each context length when doing a forward pass? What about when doing a full training step?
Deliverable: A table with two numbers per context length.


#### (c) Find the peak memory usage of the xl model when using mixed-precision, for both a forward pass and a full training step. Does mixed-precision significantly affect memory usage?
Deliverable: A 2-3 sentence response.


#### (d) Consider the xl model. Given our reference hyperparameters, what is the size of a tensor of activations in the Transformer residual stream, in single-precision? Give this size in MiB (i.e., divide the number of bytes by $$1024^2).
Deliverable: A 1-2 sentence response with your derivation.


#### (e) Now look closely at the “Active Memory Timeline” from pytorch.org/memory_viz of a memory snapshot of the xl model doing a forward pass. When you reduce the “Detail” level, the tool hides the smallest allocations to the corresponding level (e.g., putting “Detail” at 10% only shows the 10% largest allocations). What is the size of the largest allocations shown? Looking through the stack trace, can you tell where those allocations come from?
Deliverable: A 1-2 sentence response.


#### (f) Nsight Systems also has flags for memory profiling. You can combine these with the Nsight flags from before to understand what allocations are happening at different steps in your model’s lifespan. Use the PyTorch-provided NVTX labels to determine how much memory is saved for backward (these tensors are often called residuals) by a single TransformerBlock in your model. Note the 5 largest contributing operations, and what percentage of the overall memory they contribute. During the backward pass, all these tensors will be freed, but new gradient tensors are emitted at the same time. Based on your profiles showing how much memory was allocated during the forward pass, and how much memory usage changes for every TransformerBlock in the backward pass, calculate how much memory the produced gradient tensors for a TransformerBlock take. Does the result match what you expect?
Deliverable: Screenshots from Nsight Systems and a 1-2 paragraph response.