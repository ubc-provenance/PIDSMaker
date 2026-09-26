import os
import sys

current_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.abspath(os.path.join(current_file_dir, os.pardir)))
sys.path.append(parent_dir)

import resource
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch

from pidsmaker.config import get_runtime_required_args, get_yml_cfg
from pidsmaker.factory import build_model
from pidsmaker.utils.data_utils import load_all_datasets
from pidsmaker.utils.utils import log_start, log_tqdm

# ###################################################### ######################################################
"""
NOTE: before running this script, kill all other processes to avoid noise in metrics
NOTE: the graphs and text embeddings for the used dataset should already exist on disk in the task path. If not, run the pipeline for this dataset.
Usage:
python plot_real_time_cpu_watt_memory.py velox CADETS_E3 --detection.graph_preprocessing.edge_batch_size_inference=1.0 --detection.graph_preprocessing.batch_mode=edges --tuned
"""
# ###################################################### ######################################################

PLOT_BY_CONSIDERING_TIME = False  # whether to space points with the original timestamp from the dataset for more realistic plot
MINUTES_TO_PLOT = 60


def main(cfg):
    log_start(__file__)
    # We use CPU here

    process = psutil.Process()
    device = torch.device("cuda")

    _, _, test_data, max_node_num = load_all_datasets(
        cfg, device, only_keep=MINUTES_TO_PLOT // 15
    )  # we take a sample or it takes ages
    test_data = test_data[0]

    times = []
    mems = []
    watts = []
    cpu_usages = []

    base_memory = process.memory_info().rss / 1024**2

    model = build_model(data_sample=test_data[0], device=device, cfg=cfg, max_node_num=max_node_num)

    for i, g in enumerate(log_tqdm(test_data, desc="Testing")):
        start_time = time.perf_counter()

        g = g.to(device)
        _ = model(g, inference=True, validation=False)
        g.to("cpu")

        end_time = time.perf_counter()
        # process_memory = process.memory_info().rss / 1024**2  # process memory used in MiB)

        # Calculate CPU usage
        elapsed_time = end_time - start_time
        elapsed_time *= 10 # to 15min
        times.append(elapsed_time)
        # mems.append(process_memory - base_memory)
        
        torch.cuda.empty_cache()
        peak_inference_gpu_memory = torch.cuda.max_memory_allocated(device=device) / (
            1024**3
        )
        torch.cuda.reset_peak_memory_stats(device=device)
        mems.append(peak_inference_gpu_memory)
            
        # if (i+1) % 100 == 0:
        #     break

    times = np.array(times)
    cpu_usages = np.array(cpu_usages)
    watts = np.array(watts)
    mems = np.array(mems)
    
    torch.save(times, "times.pkl")
    torch.save(mems, "mems.pkl")
    
    # mems = torch.load("mems.pkl")
    # times = torch.load("times.pkl")

    # Plot Memory usage over time
    x = np.arange(len(mems))
    plt.figure(figsize=(6, 4))
    plt.fill_between(x, mems, color="#8a2be2")
    plt.ylabel("Memory (GB)", fontsize=19)
    plt.xlabel("Time windows (15 min)", fontsize=19)
    plt.xticks([0, 500, 1000, 1500, 2000, 2500],  labels=[0, 50, 100, 150, 200, 250], fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xlim(left=0, right=len(mems))
    plt.ylim(bottom=0, top=mems.max() * 2)
    plt.tight_layout()
    plt.savefig(os.path.join(current_file_dir, f"mem.svg"))
    plt.savefig(os.path.join(current_file_dir, f"mem.pdf"))

    times = times * 10
    x = np.arange(len(times))
    plt.figure(figsize=(6, 4))
    plt.fill_between(x, times, color="#8a2be2")
    plt.ylabel("Inference Time (s)", fontsize=19)
    plt.xlabel("Time windows (15 min)", fontsize=19)
    plt.xticks([0, 500, 1000, 1500, 2000, 2500],  labels=[0, 50, 100, 150, 200, 250], fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xlim(left=0, right=len(times))
    plt.ylim(bottom=0, top=times.mean() * 3)
    plt.tight_layout()
    plt.savefig(os.path.join(current_file_dir, f"time.svg"))
    plt.savefig(os.path.join(current_file_dir, f"time.pdf"))


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
