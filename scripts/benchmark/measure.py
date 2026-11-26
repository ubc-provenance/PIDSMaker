"""Runs one system on one dataset from scratch, and measures its time, memory and disk usage.

    python scripts/benchmark/measure.py --out runs/kairos_theia kairos THEIA_E5 [pidsmaker args...]

The run writes its outputs to an empty <out>/artifacts folder, so that nothing is reused from
previous runs, and its log to <out>/run.log. Once it ends, <out>/metrics.json holds:

- wall_s: wall time of the whole run, and task_times_s, the time of each task as logged
  by PIDSMaker (`Task times (s)`)
- peak_tree_pss_gb: peak RAM of the run, i.e. the proportional set size (PSS) summed over
  the main process and all its children, sampled every second. PSS splits the pages shared
  between processes, so forked workers are not counted twice. peak_proc_rss_gb is the peak
  resident memory of the largest single process
- gpu_max_reserved_gb: peak GPU memory reserved by PyTorch (torch.cuda.max_memory_reserved)
- disk_final_gb: size of all the outputs of the run once it ends, and disk_per_task_gb per task

The outputs are deleted after they are measured, unless --keep is given.
"""

import argparse
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import threading
import time

import psutil

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_pidsmaker():
    """Entry of the measured child process: pidsmaker/main.py, then its peak GPU memory."""
    gpu_file = os.environ.pop("PM_MEASURE_GPU_FILE")

    def dump_gpu():
        out = {}
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.is_initialized():
                out = {"max_reserved": torch.cuda.max_memory_reserved()}
        except Exception as e:
            out = {"error": repr(e)}
        with open(gpu_file, "w") as f:
            json.dump(out, f)

    import atexit

    atexit.register(dump_gpu)
    main_py = os.path.join(REPO, "pidsmaker", "main.py")
    sys.path.insert(0, os.path.dirname(main_py))
    sys.argv = [main_py] + sys.argv[2:]
    runpy.run_path(main_py, run_name="__main__")


def read_pss_kb(pid):
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def du_bytes(path):
    total = 0
    for dirpath, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", required=True, help="Folder of the run (deleted first if it exists)"
    )
    parser.add_argument("--keep", action="store_true", help="Keep the outputs of the run")
    parser.add_argument(
        "--threads", default="16", help="OMP_NUM_THREADS and MKL_NUM_THREADS of the run"
    )
    args, pidsmaker_args = parser.parse_known_args()

    out = os.path.abspath(args.out)
    art_dir = os.path.join(out, "artifacts")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(art_dir)

    env = dict(os.environ)
    env.update(
        PM_MEASURE_GPU_FILE=os.path.join(out, "gpu.json"),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS=args.threads,
        MKL_NUM_THREADS=args.threads,
    )
    cmd = [
        sys.executable,
        "-u",
        os.path.abspath(__file__),
        "--run-pidsmaker",
        *pidsmaker_args,
        f"--artifact_dir={art_dir}/",
    ]

    peak = {"pss": 0, "proc_rss": 0}
    start = time.time()
    with open(os.path.join(out, "run.log"), "w") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        root = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                procs = [root] + root.children(recursive=True)
            except psutil.Error:
                procs = []
            pss = 0
            for p in procs:
                try:
                    peak["proc_rss"] = max(peak["proc_rss"], p.memory_info().rss)
                except psutil.Error:
                    continue
                pss += read_pss_kb(p.pid) * 1024
            peak["pss"] = max(peak["pss"], pss)
            time.sleep(1)
    wall = time.time() - start

    text = open(os.path.join(out, "run.log"), errors="replace").read()
    task_times = {}
    found = re.findall(r"Task times \(s\): (.*)", text)
    if found:
        for kv in found[-1].split(","):
            k, v = kv.strip().split("=")
            task_times[k] = float(v)
    try:
        gpu = json.load(open(os.path.join(out, "gpu.json")))
    except (OSError, ValueError):
        gpu = {}

    gb = 1024**3
    metrics = {
        "args": pidsmaker_args,
        "returncode": proc.returncode,
        "wall_s": round(wall, 1),
        "task_times_s": task_times,
        "peak_tree_pss_gb": round(peak["pss"] / gb, 2),
        "peak_proc_rss_gb": round(peak["proc_rss"] / gb, 2),
        "gpu_max_reserved_gb": round(gpu.get("max_reserved", 0) / gb, 2),
        "disk_final_gb": round(du_bytes(art_dir) / gb, 3),
        "disk_per_task_gb": {
            d: round(du_bytes(os.path.join(art_dir, d)) / gb, 3)
            for d in sorted(os.listdir(art_dir))
            if os.path.isdir(os.path.join(art_dir, d))
        },
    }
    if not args.keep:
        shutil.rmtree(art_dir, ignore_errors=True)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(json.dumps(metrics, indent=1))
    sys.exit(proc.returncode)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run-pidsmaker":
        run_pidsmaker()
    else:
        main()
