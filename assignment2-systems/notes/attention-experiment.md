# Attention 与 Transformer 编译对照实验

实验使用 `cs336_systems/attention_profiling.py` 中的
`annotated_scaled_dot_product_attention`。执行入口是
`scripts/attention_experiment/run_experiment.py`，每个配置使用独立进程，
保留原实现的预热、CUDA 同步、计时、显存读取和梯度清除顺序。
实验对应 handout 的 `pytorch_attention` 和 `torch_compile` 两道题，
普通执行与 `torch.compile` 使用相同配置和随机种子 336。

## 配置与测量

- batch size：8；Q、K、V 不包含独立的 head 维度。
- dtype：FP32；性能实验不使用 causal mask。
- 向量维度：16、32、64、128。
- 序列长度：256、1024、4096、8192、16384。
- 每个配置预热 5 次，测量 100 次。
- forward 和 backward 分别计时，并在结束计时前完成 CUDA 同步。
- 在 forward 完成后、backward 开始前读取 `memory_allocated`，转换为 MiB。
- 显存查询和梯度清除位于耗时测量区间之外。
- 保留每次测量的原始样本，按配置计算平均值和样本标准差。
- 20 个形状各执行普通、compiled 两种模式，共 40 项测量。

正确性检查使用 PyTorch 官方 SDPA 的 `MATH` 后端作为对照，验证前向输出
以及 Q、K、V 的梯度。检查包含两个无 mask 配置和一个 causal mask 配置，
误差容限为 `atol=2e-5`、`rtol=2e-4`。实际检查结果写入
`correctness-eager.json` 和 `correctness-compiled.json`。

真实 CUDA OOM 作为该配置的实验结果保存，并继续其余配置。
其他运行错误会终止实验。性能数据不启用 Nsight、显存历史记录，
也不关闭 PyTorch 的显存缓存。

## 完整 Transformer 对照

`run_transformer_experiment.py` 在独立进程中调用原有 `benchmark.main()`，
使用 `--no-is_compiled` 选择普通执行，使用 `--is_compiled` 选择编译整个模型。
它使用 small、medium、large、xl、10B 五个模型配置，batch size 4、
context length 512、vocabulary size 10000、FP32；每项预热 5 次，测量 10 次。

每个模型分别测量 forward-only、forward + loss + backward，
以及包含 AdamW 更新的完整训练步骤。两个模式共 30 项测量。
编译和 optimizer 状态的首次创建发生在预热阶段。
原脚本输出平均值和标准差，实验保留其完整日志，不宣称记录了逐次耗时。

真实 OOM 记录为对应配置的结果，不使用降低精度或缩小模型的结果代替。
对照表写入有效运行目录的 `transformer/comparison.csv`。

## 服务器执行与自动下载

2026-10-03 的任务运行于 `zju-lab-rvpn`，服务器配置为 RTX 5090、
Python 3.13.12、PyTorch 2.11.0+cu130、CUDA 13.0。
SSH 别名使用本地 SOCKS5 ProxyCommand，上传、查询和下载均使用该别名。

远程等待服务为 `cs336-attention-20261003-0207.service`。
`wait_for_gpu.py` 每 60 秒检查显卡；连续三次满足无计算进程、
显存使用低于 128 MiB、GPU 利用率为零时才启动。
选择的物理显卡通过 `CUDA_VISIBLE_DEVICES` 映射到进程内的 CUDA 设备。
未知状态的显卡不参与选择。

运行期间每 5 秒检查计算进程。若出现其他人的任务，只停止本次实验的
进程组，将该次测量标为无效，然后重新等待。

本地下载任务为 `com.qvanium.cs336-attention-download-20261003-0207`。
`download_when_done.py` 查询真实远程状态，在实验完成或失败后下载结果与日志；
网络暂时不可用时继续等待。下载完成后写入 `download-complete.json`，
退出并禁用本次下载任务。

本地结果目录为 `results/attention/20261003-0207-compile-comparison/`。
该目录由已有 `.gitignore` 忽略，保存源码副本、服务配置、原始日志和实验数据。
`remote-status.json` 记录最近查询到的状态；最终有效运行目录由状态中的
`results_dir` 指定。只有终态为 `completed` 且对应
`attempt-status.json` 的 `valid_measurement` 为 `true` 时，才使用其数据。

每次有效运行包含环境信息、两个模式的正确性报告、
`summary.json`、`summary.csv`、`comparison.csv`，以及各配置的样本 JSON 和日志。
`comparison.csv` 并排列出 attention 前向、反向耗时和显存，并计算对应加速比；
`transformer/` 保存完整模型的汇总、对照表和各配置日志。
实际版本、显卡、源码 SHA256 和运行状态以这些文件为依据。

取消远程等待任务：

`ssh zju-lab-rvpn 'systemctl --user stop cs336-attention-20261003-0207.service'`

取消本地下载任务：

`launchctl bootout gui/$(id -u)/com.qvanium.cs336-attention-download-20261003-0207`
