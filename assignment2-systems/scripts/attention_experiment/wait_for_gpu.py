import argparse
import csv
import io
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--python", required=True)
args = parser.parse_args()
run_dir = args.run_dir.resolve()
source_dir = run_dir / "source"
python = args.python
status_path = run_dir / "status.json"


def write_status(state, **details):
    status = {"state": state, "updated_utc": datetime.now(timezone.utc).isoformat(), **details}
    pending = run_dir / "status.pending.json"
    pending.write_text(json.dumps(status, indent=2) + "\n")
    pending.replace(status_path)
    print(json.dumps(status), flush=True)


def query(arguments):
    completed = subprocess.run(["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
                               check=True, capture_output=True, text=True)
    return list(csv.reader(io.StringIO(completed.stdout), skipinitialspace=True))


def available_gpus():
    devices = query(["--query-gpu=index,uuid,memory.used,utilization.gpu"])
    processes = query(["--query-compute-apps=gpu_uuid,pid"])
    occupied = {row[0] for row in processes}
    return {int(index): uuid for index, uuid, memory, utilization in devices
            if memory.isdecimal() and utilization.isdecimal()
            and int(memory) < 128 and int(utilization) == 0 and uuid not in occupied}


def main():
    run_dir.mkdir(exist_ok=True)
    working = run_dir / "working"
    working.mkdir(exist_ok=True)
    counts = {}
    write_status("waiting", poll_seconds=60, consecutive_idle_checks=3)
    while True:
        devices = available_gpus()
        counts = {index: counts.get(index, 0) + 1 for index in devices}
        ready = [index for index, count in counts.items() if count >= 3]
        if ready:
            selected = min(ready)
            if selected not in available_gpus():
                counts = {}
                continue
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(selected)
            environment["PYTHONPATH"] = str(source_dir)
            environment["TMPDIR"] = str(working)
            environment["PYTHONUNBUFFERED"] = "1"
            assert environment.get("PYTORCH_NO_CUDA_MEMORY_CACHING") != "1"
            attempt_dir = run_dir / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            attempt_dir.mkdir(parents=True)
            write_status("running", gpu_index=selected, gpu_uuid=devices[selected], results_dir=str(attempt_dir))
            interference = False
            with (attempt_dir / "experiment.log").open("w") as log:
                process = subprocess.Popen([python, str(source_dir / "run_experiment.py"),
                                            "--output", str(attempt_dir)], cwd=source_dir,
                                           env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    while process.poll() is None:
                        time.sleep(5)
                        processes = query(["--query-compute-apps=gpu_uuid,pid"])
                        for uuid, pid in processes:
                            if uuid != devices[selected]:
                                continue
                            try:
                                process_group = os.getpgid(int(pid)) if pid.isdecimal() else None
                            except ProcessLookupError:
                                continue
                            if process_group != process.pid:
                                interference = True
                                print("Another compute process appeared on selected GPU; stopping own experiment", file=log, flush=True)
                                os.killpg(process.pid, signal.SIGTERM)
                                break
                        if interference:
                            break
                    process.wait()
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait()
            (attempt_dir / "attempt-status.json").write_text(json.dumps({
                "valid_measurement": not interference and process.returncode == 0,
                "returncode": process.returncode,
                "concurrent_process_detected": interference,
            }, indent=2) + "\n")
            if interference:
                write_status("waiting", rejected_attempt=str(attempt_dir), poll_seconds=60)
                counts = {}
                continue
            write_status("completed" if process.returncode == 0 else "failed",
                         gpu_index=selected, returncode=process.returncode,
                         concurrent_process_detected=interference, results_dir=str(attempt_dir))
            return process.returncode
        write_status("waiting", idle_candidate_checks=counts, poll_seconds=60)
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
