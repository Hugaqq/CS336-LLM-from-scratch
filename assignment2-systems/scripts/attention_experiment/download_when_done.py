import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--local-dir", type=Path, required=True)
parser.add_argument("--remote-dir", required=True)
parser.add_argument("--host", required=True)
parser.add_argument("--service", required=True)
parser.add_argument("--launch-label", required=True)
args = parser.parse_args()
local_dir = args.local_dir.resolve()
remote_dir = args.remote_dir
service = args.service


def main():
    if (local_dir / "download-complete.json").exists():
        return
    while True:
        completed = subprocess.run(["/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                    "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
                                    args.host, f"cat {remote_dir}/status.json"],
                                   capture_output=True, text=True)
        if completed.returncode:
            print(f"SSH temporarily unavailable: exit {completed.returncode}", flush=True)
            time.sleep(60)
            continue
        state = json.loads(completed.stdout)
        print(json.dumps(state), flush=True)
        (local_dir / "remote-status.json").write_text(json.dumps(state, indent=2) + "\n")
        if state["state"] in ["completed", "failed"]:
            break
        active = subprocess.run(["/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                 args.host, f"systemctl --user is-active {service}"],
                                capture_output=True, text=True)
        if active.returncode and active.stdout.strip() in ["failed", "inactive"]:
            state["state"] = "failed"
            state["reason"] = "Remote waiting service exited before completion; inspect waiting.log"
            break
        time.sleep(60)
    while True:
        transfer = subprocess.run(["/usr/bin/rsync", "-a", "--exclude=working", "--exclude=__pycache__",
                                   f"{args.host}:{remote_dir}/", str(local_dir) + "/"])
        if transfer.returncode == 0:
            break
        print(f"Download unavailable: exit {transfer.returncode}; retrying in 60 seconds", flush=True)
        time.sleep(60)
    (local_dir / "download-complete.json").write_text(json.dumps({"downloaded": True,
                                                                  "remote_status": state}, indent=2) + "\n")
    print("Remote artifacts downloaded", flush=True)
    subprocess.run(["/bin/launchctl", "disable", f"gui/{os.getuid()}/{args.launch_label}"], check=True)


if __name__ == "__main__":
    sys.exit(main())
