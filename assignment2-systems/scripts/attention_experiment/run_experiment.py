import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from timeit import default_timer

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cs336_systems import attention_profiling as student

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def selected_attention(is_compiled):
    return torch.compile(student.annotated_scaled_dot_product_attention) if is_compiled else student.annotated_scaled_dot_product_attention


def check_correctness(output_dir, is_compiled):
    attention = selected_attention(is_compiled)
    mode = "compiled" if is_compiled else "eager"
    torch.manual_seed(336)
    report = []
    for length, dimension, causal in [(17, 16, False), (256, 64, False), (17, 16, True)]:
        inputs = [torch.randn(2, length, dimension, device="cuda", requires_grad=True) for _ in range(3)]
        reference_inputs = [value.detach().clone().requires_grad_() for value in inputs]
        upstream = torch.randn_like(inputs[0])
        mask = torch.ones(length, length, device="cuda", dtype=torch.bool).tril() if causal else None
        result = attention(*inputs, mask=mask)
        with sdpa_kernel(SDPBackend.MATH):
            reference = torch.nn.functional.scaled_dot_product_attention(*reference_inputs, attn_mask=mask)
        torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-4)
        result.backward(upstream)
        reference.backward(upstream)
        gradient_errors = {}
        for name, actual, expected in zip(["Q", "K", "V"], inputs, reference_inputs):
            assert actual.is_leaf and actual.grad is not None
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-5, rtol=2e-4)
            gradient_errors[name] = (actual.grad - expected.grad).abs().max().item()
        report.append({"length": length, "dimension": dimension, "causal": causal,
                       "output_max_abs_error": (result - reference).abs().max().item(),
                       "gradient_max_abs_errors": gradient_errors, "passed": True})
    write_json(output_dir / f"correctness-{mode}.json", {"reference_backend": "SDPBackend.MATH",
                                               "is_compiled": is_compiled,
                                               "atol": 2e-5, "rtol": 2e-4, "cases": report})
    write_json(output_dir / f"environment-{mode}.json", environment_report())
    print(f"{mode}: all 3 forward/gradient comparisons passed", flush=True)


def run_configuration(output_dir, dimension, length, is_compiled):
    attention = selected_attention(is_compiled)
    mode = "compiled" if is_compiled else "eager"
    torch.manual_seed(336)
    batch = student.batch_size
    device = student.device
    upstream = torch.randn(batch, length, dimension).to(device=device)
    q, k, v = [torch.randn(batch, length, dimension, requires_grad=True, device=device) for _ in range(3)]
    for _ in range(student.warm_up_times):
        result = attention(q, k, v)
        result.backward(gradient=upstream)
        q.grad = k.grad = v.grad = None
    samples = []
    with torch.cuda.nvtx.range("real test"):
        for iteration in range(student.test_times):
            torch.cuda.synchronize(device=device)
            t0 = default_timer()
            with torch.cuda.nvtx.range("forwarding"):
                result = attention(q, k, v)
            torch.cuda.synchronize(device=device)
            t1 = default_timer()
            memory_mib = torch.cuda.memory_allocated(device=device) / 1024**2
            t2 = default_timer()
            with torch.cuda.nvtx.range("backwarding"):
                result.backward(gradient=upstream)
            torch.cuda.synchronize(device=device)
            t3 = default_timer()
            q.grad = k.grad = v.grad = None
            samples.append({"iteration": iteration, "forward_s": t1 - t0,
                            "backward_s": t3 - t2, "before_backward_mib": memory_mib})
    record = {"status": "success", "d_model": dimension, "seq_len": length,
              "is_compiled": is_compiled, "mode": mode,
              "attention_source_sha256": hashlib.sha256(Path(student.__file__).read_bytes()).hexdigest(),
              "batch_size": batch, "dtype": "float32", "warmups": student.warm_up_times,
              "iterations": student.test_times, "samples": samples}
    for field in ["forward_s", "backward_s", "before_backward_mib"]:
        record[field + "_mean"] = mean(sample[field] for sample in samples)
        record[field + "_std"] = stdev(sample[field] for sample in samples)
    write_json(output_dir / f"{mode}-d{dimension}-s{length}.json", record)
    print(json.dumps({key: value for key, value in record.items() if key != "samples"}), flush=True)


def environment_report():
    return {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                   "hostname": platform.node(), "device_name": torch.cuda.get_device_name(),
                   "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                   "float32_matmul_precision": torch.get_float32_matmul_precision(),
                   "pytorch_no_cuda_memory_caching": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
                   "student_default_is_compiled": student.is_compiled,
                   "tested_modes": ["eager", "compiled"],
                   "attention_source_sha256": hashlib.sha256(Path(student.__file__).read_bytes()).hexdigest(),
                   "wrapper_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "method": "Each correctness mode and benchmark configuration in an independent process; same seed and student warmup/timing/memory/gradient-clear order. Compilation happens during warmup."}


def write_tables(output_dir, records):
    write_json(output_dir / "summary.json", records)
    columns = ["d_model", "seq_len", "status", "mode", "is_compiled", "attention_source_sha256",
               "forward_s_mean", "forward_s_std", "backward_s_mean", "backward_s_std",
               "before_backward_mib_mean", "before_backward_mib_std"]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    by_configuration = {(record["d_model"], record["seq_len"], record["mode"]): record for record in records}
    rows = []
    for dimension in [16, 32, 64, 128]:
        for length in [256, 1024, 4096, 8192, 16384]:
            row = {"d_model": dimension, "seq_len": length}
            for mode in ["eager", "compiled"]:
                record = by_configuration.get((dimension, length, mode))
                if record is None:
                    row[f"{mode}_status"] = "pending"
                    continue
                row[f"{mode}_status"] = record["status"]
                if record["status"] == "success":
                    for field in ["forward_s_mean", "backward_s_mean", "before_backward_mib_mean"]:
                        row[f"{mode}_{field}"] = record[field]
            if all(row.get(f"{mode}_status") == "success" for mode in ["eager", "compiled"]):
                for phase in ["forward", "backward"]:
                    row[f"{phase}_speedup"] = row[f"eager_{phase}_s_mean"] / row[f"compiled_{phase}_s_mean"]
            rows.append(row)
    comparison_columns = ["d_model", "seq_len", "eager_status", "compiled_status",
                          "eager_forward_s_mean", "compiled_forward_s_mean", "forward_speedup",
                          "eager_backward_s_mean", "compiled_backward_s_mean", "backward_speedup",
                          "eager_before_backward_mib_mean", "compiled_before_backward_mib_mean"]
    with (output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_columns)
        writer.writeheader()
        writer.writerows(rows)


def run_all(output_dir):
    for mode in ["eager", "compiled"]:
        with (output_dir / f"correctness-{mode}.log").open("w") as log:
            subprocess.run([sys.executable, __file__, "--output", str(output_dir), "--mode", mode,
                            "--correctness-only"], stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"{mode}: correctness passed", flush=True)
    environment = json.loads((output_dir / "environment-eager.json").read_text())
    write_json(output_dir / "environment.json", environment)
    records = []
    for dimension in [16, 32, 64, 128]:
        for length in [256, 1024, 4096, 8192, 16384]:
            for mode in ["eager", "compiled"]:
                name = f"{mode}-d{dimension}-s{length}"
                with (output_dir / f"{name}.log").open("w") as log:
                    process = subprocess.run([sys.executable, __file__, "--output", str(output_dir),
                                              "--mode", mode, "--dimension", str(dimension), "--length", str(length)],
                                             stdout=log, stderr=subprocess.STDOUT)
                if process.returncode == 42:
                    record = {"status": "oom", "d_model": dimension, "seq_len": length,
                              "mode": mode, "is_compiled": mode == "compiled",
                              "attention_source_sha256": environment["attention_source_sha256"]}
                    write_json(output_dir / f"{name}.json", record)
                elif process.returncode:
                    raise RuntimeError(f"{name} failed with exit {process.returncode}; inspect {name}.log")
                else:
                    record = json.loads((output_dir / f"{name}.json").read_text())
                    record.pop("samples")
                records.append(record)
                write_tables(output_dir, records)
                print(f"{name}: {record['status']}", flush=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("run_transformer_experiment.py")),
                    "--output", str(output_dir / "transformer")], check=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("verify_results.py")),
                    "--attempt", str(output_dir), "--source", str(Path(__file__).parent)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--length", type=int)
    parser.add_argument("--mode", choices=["eager", "compiled"])
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    if args.correctness_only:
        assert args.mode is not None
        check_correctness(args.output, args.mode == "compiled")
    elif args.dimension is None:
        run_all(args.output)
    else:
        assert args.mode is not None
        try:
            run_configuration(args.output, args.dimension, args.length, args.mode == "compiled")
        except torch.OutOfMemoryError:
            import traceback

            traceback.print_exc()
            sys.exit(42)
