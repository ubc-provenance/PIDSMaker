#!/usr/bin/env python3
"""
PIDSMaker Native GPU Visualizer Launcher
This script initializes the environment and launches the modularized native visualizer.
"""

import sys
import os
import argparse
from PyQt5.QtWidgets import QApplication

# Ensure module is discoverable if run standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pidsmaker.vizgen.native.loader import resolve_latest_viz_dir, load_data
from pidsmaker.vizgen.native.main_window import MainWindow

def main():
    parser = argparse.ArgumentParser(description="PIDSMaker Native GPU Visualizer")
    parser.add_argument(
        "model", type=str, nargs="?", default="orthrus", help="Model config name"
    )
    parser.add_argument(
        "dataset", type=str, nargs="?", default="CADETS_E3", help="Dataset name"
    )
    parser.add_argument("--force_restart", nargs="*", default=[])
    args = parser.parse_args()

    if not os.environ.get("DISPLAY"):
        print("\n[!] Error: DISPLAY environment variable is not set.")
        print("[!] Visualization requires an X11 server (GUI environment).")
        print("[!] If running in Docker, ensure X11 forwarding is configured in docker-compose.yml.")
        print("[!] You may need to run 'xhost +local:docker' on your host terminal before launching.")
        sys.exit(1)

    try:
        app = QApplication(sys.argv)
    except Exception as e:
        print(f"\n[!] Error initializing GUI: {e}")
        print("[!] Please ensure your X11 server accepts connections (e.g. run 'xhost +local:docker').")
        sys.exit(1)

    viz_dir = resolve_latest_viz_dir(args.dataset)
    if not viz_dir:
        fallback = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                f"embedding_viz_{args.dataset}_word2vec_points.json",
            )
        )
        if os.path.exists(fallback):
            viz_dir = os.path.dirname(fallback)
        else:
            print(f"Error: Could not find viz artifacts for {args.dataset}.")
            sys.exit(1)

    w2v_path = os.path.join(
        viz_dir, f"embedding_viz_{args.dataset}_word2vec_points.json"
    )
    enc_path = os.path.join(
        viz_dir, f"embedding_viz_{args.dataset}_encoder_points.json"
    )
    
    if not os.path.exists(enc_path):
        import glob
        matches = glob.glob(os.path.join(viz_dir, f"embedding_viz_{args.dataset}_encoder_*_points.json"))
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            enc_path = matches[0]

    load_path = enc_path if os.path.exists(enc_path) else w2v_path
    if not os.path.exists(load_path):
        print(f"Error: Could not find points.json in {viz_dir}")
        sys.exit(1)

    pos_hops, colors, sizes, metadata, stats, attack_edges = load_data(load_path)
    viz_cfg = {}
    viz_config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "config", "viz_config.yml")
    )
    if os.path.exists(viz_config_path):
        import yaml
        with open(viz_config_path) as f:
            viz_cfg = yaml.safe_load(f).get("embedding_viz", {})

    try:
        window = MainWindow(
            pos_hops,
            colors,
            sizes,
            metadata,
            stats,
            attack_edges,
            viz_cfg=viz_cfg,
            enc_path=enc_path,
            w2v_path=w2v_path,
            current_path=load_path,
        )
        window.showMaximized()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"\n[!] Error during GUI rendering: {e}")
        print("[!] This is usually caused by missing OpenGL libraries or X11 permission errors.")
        print("[!] Try running 'xhost +local:docker' on your host.")
        sys.exit(1)

if __name__ == "__main__":
    main()
