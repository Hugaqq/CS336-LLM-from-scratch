import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import torch

from cs336_systems import benchmark
from cs336_systems.config import Config


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run_configuration(size, phase, mode):
    torch.manual_seed(336)
    sys.argv = [benchmark.__file__, "--compute_test", f"--{phase}", "--size", size,
                "--context_length", "512", "--device", "cuda:0",
                "--is_compiled" if mode == "compiled" else "--no-is_compiled"]
    benchmark.main()


def write_tables(output_dir, records):
    write_json(output_dir / "summary.json", records)
    columns = ["size", "phase", "mode", "status", "mean_s", "std_s", "batch_size",
               "context_length", "dtype", "warmups", "iterations", "seed", "benchmark_source_sha256"]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    by_configuration = {(record["size"], record["phase"], record["mode"]): record for record in records}
    rows = []
    for size in ["small", "medium", "large", "xl", "10B"]:
        for phase in ["forward_only", "fandb", "full"]:
            row = {"size": size, "phase": phase}
            for mode in ["eager", "compiled"]:
                record = by_configuration.get((size, phase, mode))
                row[f"{mode}_status"] = "pending" if record is None else record["status"]
                if record is not None and record["status"] == "success":
                    row[f"{mode}_mean_s"] = record["mean_s"]
                    row[f"{mode}_std_s"] = record["std_s"]
            if all(row[f"{mode}_status"] == "success" for mode in ["eager", "compiled"]):
                row["speedup"] = row["eager_mean_s"] / row["compiled_mean_s"]
            rows.append(row)
    columns = ["size", "phase", "eager_status", "compiled_status", "eager_mean_s", "compiled_mean_s",
               "eager_std_s", "compiled_std_s", "speedup"]
    with (output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_all(output_dir):
    output_dir.mkdir(exist_ok=True)
    source_hash = hashlib.sha256(Path(benchmark.__file__).read_bytes()).hexdigest()
    write_json(output_dir / "method.json", {
        "method": "Unmodified student benchmark.main CLI in an independent process per case; seed reset before model/input creation; original stdout/stderr retained. Original script reports mean/std, without individual timing samples.",
        "benchmark_source_sha256": source_hash,
        "wrapper_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cli_eager": "--no-is_compiled", "cli_compiled": "--is_compiled",
    })
    configurations = [("small", "full")]
    configurations.extend((size, phase) for size in ["small", "medium", "large", "xl", "10B"]
                          for phase in ["forward_only", "fandb", "full"] if (size, phase) != ("small", "full"))
    records = []
    for size, phase in configurations:
        config = Config._from_size(size, context_length=512)
        assert config.batch_size == 4 and config.dtype == torch.float32
        assert config.warm_up_times == 5 and config.tests == 10
        for mode in ["eager", "compiled"]:
            name = f"{size}-{phase}-{mode}"
            with (output_dir / f"{name}.log").open("w") as log:
                process = subprocess.run([sys.executable, __file__, "--output", str(output_dir),
                                          "--size", size, "--phase", phase, "--mode", mode],
                                         stdout=log, stderr=subprocess.STDOUT)
            record = {"size": size, "phase": phase, "mode": mode,
                      "batch_size": config.batch_size, "context_length": config.context_length,
                      "dtype": "float32", "warmups": config.warm_up_times, "iterations": config.tests,
                      "seed": 336, "benchmark_source_sha256": source_hash}
            if process.returncode == 42:
                record["status"] = "oom"
            elif process.returncode:
                raise RuntimeError(f"{name} failed with exit {process.returncode}; inspect {name}.log")
            else:
                stdout = (output_dir / f"{name}.log").read_text()
                mean_values = re.findall(r"mean_t\s*:\s*([0-9eE.+-]+)", stdout)
                std_values = re.findall(r"std_t\s*:\s*([0-9eE.+-]+)", stdout)
                assert len(mean_values) == len(std_values) == 1, f"Missing unique benchmark measurements for {name}"
                record.update(status="success", mean_s=float(mean_values[0]), std_s=float(std_values[0]))
                assert math.isfinite(record["mean_s"]) and math.isfinite(record["std_s"])
            write_json(output_dir / f"{name}.json", record)
            records.append(record)
            write_tables(output_dir, records)
            print(f"Transformer {name}: {record['status']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", choices=["small", "medium", "large", "xl", "10B"])
    parser.add_argument("--phase", choices=["forward_only", "fandb", "full"])
    parser.add_argument("--mode", choices=["eager", "compiled"])
    args = parser.parse_args()
    if args.size is None:
        run_all(args.output)
    else:
        assert args.phase is not None and args.mode is not None
        try:
            run_configuration(args.size, args.phase, args.mode)
        except torch.OutOfMemoryError:
            import traceback

            traceback.print_exc()
            sys.exit(42)
