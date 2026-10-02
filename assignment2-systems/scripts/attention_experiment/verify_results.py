import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--attempt", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
args = parser.parse_args()
attempt = args.attempt.resolve()
source = args.source.resolve()

attention_hash = hashlib.sha256((source / "cs336_systems/attention_profiling.py").read_bytes()).hexdigest()
benchmark_hash = hashlib.sha256((source / "cs336_systems/benchmark.py").read_bytes()).hexdigest()
environment = json.loads((attempt / "environment.json").read_text())
assert environment["attention_source_sha256"] == attention_hash
assert environment["device_name"] == "NVIDIA GeForce RTX 5090"
assert environment["pytorch_no_cuda_memory_caching"] != "1"
for mode in ["eager", "compiled"]:
    correctness = json.loads((attempt / f"correctness-{mode}.json").read_text())
    assert correctness["reference_backend"] == "SDPBackend.MATH"
    assert correctness["is_compiled"] == (mode == "compiled")
    assert len(correctness["cases"]) == 3 and all(case["passed"] for case in correctness["cases"])
    for case in correctness["cases"]:
        assert math.isfinite(case["output_max_abs_error"])
        assert all(math.isfinite(value) for value in case["gradient_max_abs_errors"].values())

attention = json.loads((attempt / "summary.json").read_text())
expected_attention = {(dimension, length, mode) for dimension in [16, 32, 64, 128]
                      for length in [256, 1024, 4096, 8192, 16384] for mode in ["eager", "compiled"]}
assert len(attention) == 40
assert {(row["d_model"], row["seq_len"], row["mode"]) for row in attention} == expected_attention
sample_count = 0
for row in attention:
    assert row["attention_source_sha256"] == attention_hash
    assert row["is_compiled"] == (row["mode"] == "compiled")
    name = f"{row['mode']}-d{row['d_model']}-s{row['seq_len']}"
    detail = json.loads((attempt / f"{name}.json").read_text())
    assert detail["status"] == row["status"]
    if row["status"] == "oom":
        assert "OutOfMemoryError" in (attempt / f"{name}.log").read_text()
        continue
    assert row["status"] == "success" and detail["batch_size"] == 8 and detail["dtype"] == "float32"
    assert all(math.isfinite(row[f"{field}_{statistic}"])
               for field in ["forward_s", "backward_s", "before_backward_mib"] for statistic in ["mean", "std"])
    assert detail["warmups"] == 5 and detail["iterations"] == 100 and len(detail["samples"]) == 100
    assert [sample["iteration"] for sample in detail["samples"]] == list(range(100))
    for sample in detail["samples"]:
        assert all(math.isfinite(sample[field]) and sample[field] > 0
                   for field in ["forward_s", "backward_s", "before_backward_mib"])
    sample_count += len(detail["samples"])
with (attempt / "comparison.csv").open(newline="") as handle:
    attention_comparison = list(csv.DictReader(handle))
assert len(attention_comparison) == 20
assert all(row[f"{mode}_status"] in ["success", "oom"] for row in attention_comparison for mode in ["eager", "compiled"])

transformer_dir = attempt / "transformer"
transformer = json.loads((transformer_dir / "summary.json").read_text())
expected_transformer = {(size, phase, mode) for size in ["small", "medium", "large", "xl", "10B"]
                        for phase in ["forward_only", "fandb", "full"] for mode in ["eager", "compiled"]}
assert len(transformer) == 30
assert {(row["size"], row["phase"], row["mode"]) for row in transformer} == expected_transformer
for row in transformer:
    assert row["benchmark_source_sha256"] == benchmark_hash
    assert row["batch_size"] == 4 and row["context_length"] == 512 and row["dtype"] == "float32"
    assert row["warmups"] == 5 and row["iterations"] == 10 and row["seed"] == 336
    name = f"{row['size']}-{row['phase']}-{row['mode']}"
    assert json.loads((transformer_dir / f"{name}.json").read_text()) == row
    log = (transformer_dir / f"{name}.log").read_text()
    if row["status"] == "oom":
        assert "OutOfMemoryError" in log
        continue
    assert row["status"] == "success"
    assert math.isfinite(row["mean_s"]) and row["mean_s"] > 0
    assert math.isfinite(row["std_s"]) and row["std_s"] >= 0
    assert float(re.search(r"mean_t\s*:\s*([0-9eE.+-]+)", log).group(1)) == row["mean_s"]
    assert float(re.search(r"std_t\s*:\s*([0-9eE.+-]+)", log).group(1)) == row["std_s"]
with (transformer_dir / "comparison.csv").open(newline="") as handle:
    transformer_comparison = list(csv.DictReader(handle))
assert len(transformer_comparison) == 15
assert all(row[f"{mode}_status"] in ["success", "oom"] for row in transformer_comparison for mode in ["eager", "compiled"])
report = {"validated": True, "attempt": attempt.name, "device_name": environment["device_name"],
          "gpu_index": int(environment["cuda_visible_devices"]), "attention_counts": dict(Counter(row["status"] for row in attention)),
          "attention_raw_samples": sample_count, "transformer_counts": dict(Counter(row["status"] for row in transformer)),
          "correctness_cases": 6, "attention_source_sha256": attention_hash,
          "benchmark_source_sha256": benchmark_hash}
(attempt / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
