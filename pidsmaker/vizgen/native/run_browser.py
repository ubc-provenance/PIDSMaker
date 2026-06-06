import os
import sys
import glob
import yaml
from datetime import datetime

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextEdit, QPushButton, QLabel, QSplitter, QMessageBox, QAbstractItemView, QComboBox
)
from PyQt5.QtCore import Qt, QProcess

def dict_diff(d1, d2):
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        return d1 if d1 != d2 else None
    diff = {}
    for k, v in d1.items():
        if k not in d2:
            diff[k] = v
        else:
            sub_diff = dict_diff(v, d2[k])
            if sub_diff is not None:
                if isinstance(sub_diff, dict) and not sub_diff:
                    if not v and not d2[k]:
                        pass
                    else:
                        diff[k] = sub_diff
                else:
                    diff[k] = sub_diff
    return diff

def clean_dict(d):
    if not isinstance(d, dict):
        return d
    cleaned = {}
    
    KNOWN_METHODS = {
        "alacarte", "doc2vec", "fasttext", "flash", "temporal_rw", "word2vec",
        "custom_mlp", "gat", "gin", "graph_attention", "magic_gat", "sage", "tgn", "none", "rcaid_gat", "sum_aggregation", "glstm",
        "few_shot", "predict_edge_contrastive", "predict_edge_type", "predict_node_type", "reconstruct_edge_embeddings", "reconstruct_node_embeddings", "reconstruct_node_features", "reconstruct_masked_features", "predict_masked_struct", "detect_edge_few_shot",
        "global_batching", "inter_graph_batching", "intra_graph_batching",
        "edges", "tgn_last_neighbor",
        "depimpact", "synthetic_attack_naive", "rcaid_pseudo_graph",
        "kairos_idf_queue", "provnet_lof_queue"
    }
    
    active_method = None
    if "used_method" in d and isinstance(d["used_method"], str):
        active_method = d["used_method"]
    elif "used_methods" in d and isinstance(d["used_methods"], str):
        active_method = d["used_methods"]
        
    for k, v in d.items():
        if isinstance(k, str) and k.startswith('_'):
            continue
        if v is None or v == "" or v == [] or v == {}:
            continue
            
        if k in ["attack_to_time_window", "ground_truth_relative_path", "train_dates", "test_dates", "val_dates", "unused_dates", "database", "database_all_file", "host", "password", "port", "user", "node_label_features"]:
            continue
            
        if active_method and isinstance(v, dict) and k in KNOWN_METHODS and k != active_method:
            continue
            
        if isinstance(v, dict):
            v_clean = clean_dict(v)
            if v_clean:
                if len(v_clean) == 1 and list(v_clean.keys())[0] in ["used_method", "used_methods"] and v_clean[list(v_clean.keys())[0]] == "none":
                    continue
                cleaned[k] = v_clean
        else:
            cleaned[k] = v
    return cleaned

class RunBrowserDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.active_processes = []
        self.setWindowTitle("PIDSMaker - Local Run Browser")
        self.setMinimumSize(1300, 700)
        self.resize(1300, 700)
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e24;
                color: #e0e0e8;
            }
            QLabel {
                color: #e0e0e8;
            }
            QTableWidget {
                background-color: #2b2b36;
                color: #e0e0e8;
                gridline-color: #3f3f4e;
                border: 1px solid #3f3f4e;
                selection-background-color: #3b82f6;
                selection-color: white;
            }
            QHeaderView::section {
                background-color: #1e1e24;
                color: #9ca3af;
                padding: 4px;
                border: 1px solid #3f3f4e;
                font-weight: bold;
            }
            QTextEdit {
                background-color: #111115;
                color: #a3e635;
                border: 1px solid #3f3f4e;
                border-radius: 4px;
                font-family: monospace;
                padding: 10px;
            }
            QPushButton {
                background-color: #3f3f4e;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4f4f66;
            }
            QPushButton:disabled {
                background-color: #2b2b36;
                color: #6b7280;
            }
            QScrollBar:vertical { background: #1a1a24; width: 12px; margin: 0px; }
            QScrollBar::handle:vertical { background: #3a3a45; min-height: 20px; border-radius: 3px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
            
            QScrollBar:horizontal { background: #1a1a24; height: 12px; margin: 0px; }
            QScrollBar::handle:horizontal { background: #3a3a45; min-width: 20px; border-radius: 3px; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }
        """)
        self.selected_viz_dir = None
        self.selected_dataset = None
        self.init_ui()
        self.scan_runs()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Table of runs
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Date", "Dataset", "Model", "ADP", "Discrim", "Artifacts"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        splitter.addWidget(self.table)

        # Right panel: Config viewer & Settings
        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(0, 0, 0, 0)
        
        lbl_settings = QLabel("VISUALIZATION SETTINGS")
        lbl_settings.setStyleSheet("font-size: 14px; color: #9ca3af; font-weight: bold;")
        right_panel.addWidget(lbl_settings)
        
        self.cmb_epoch = QComboBox()
        self.cmb_epoch.setStyleSheet("""
            QComboBox {
                background-color: #2b2b36;
                color: white;
                padding: 5px;
                border: 1px solid #3f3f4e;
            }
            QComboBox QAbstractItemView {
                background-color: #2b2b36;
                color: white;
                selection-background-color: #3b82f6;
            }
        """)
        self.cmb_epoch.currentIndexChanged.connect(self.update_config_view)
        right_panel.addWidget(self.cmb_epoch)
        
        right_panel.addSpacing(10)
        
        lbl_cfg = QLabel("RUN CONFIGURATION")
        lbl_cfg.setStyleSheet("font-size: 14px; color: #9ca3af; font-weight: bold;")
        right_panel.addWidget(lbl_cfg)
        
        self.txt_config = QTextEdit()
        self.txt_config.setReadOnly(True)
        
        from PyQt5.QtWidgets import QWidget
        self.right_widget = QWidget()
        self.right_widget.setLayout(right_panel)
        self.right_widget.setEnabled(False) # Disabled by default
        right_panel.addWidget(self.txt_config)
        
        splitter.addWidget(self.right_widget)
        splitter.setSizes([750, 500])

        main_layout.addWidget(splitter)

        # Bottom buttons
        btn_layout = QHBoxLayout()
        self.btn_launch = QPushButton("Launch Visualizer")
        self.btn_launch.setEnabled(False)
        self.btn_launch.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 8px;")
        self.btn_launch.clicked.connect(self.launch_visualizer)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(self.btn_launch)
        main_layout.addLayout(btn_layout)

    def scan_runs(self):
        from PyQt5.QtWidgets import QApplication
        original_title = self.windowTitle()
        self.setWindowTitle("PIDSMaker - Local Run Browser (Scanning...)")
        QApplication.processEvents()
        
        pidsmaker_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        artifacts_root = "/home/artifacts" if os.path.exists("/home/artifacts") else os.path.join(pidsmaker_root, "artifacts")
        
        # Scan for ALL dataset directories under evaluation/evaluation/<HASH>/
        search_pattern = os.path.join(artifacts_root, "evaluation/evaluation/*/*")
        dataset_dirs = glob.glob(search_pattern)
        
        self.runs_data = []
        
        for d_path in dataset_dirs:
            if not os.path.isdir(d_path):
                continue
                
            try:
                parts = d_path.split(os.sep)
                dataset_idx = parts.index("evaluation") + 2
                run_hash = parts[dataset_idx]
                dataset = parts[dataset_idx + 1]
                
                viz_dir = os.path.join(d_path, "viz")
                has_viz = os.path.isdir(viz_dir)
                
                cfg_path = os.path.join(d_path, "run_config.yml")
                if not os.path.exists(cfg_path):
                    # main.py saves to parent of dataset dir (hash dir), so also check there
                    cfg_path = os.path.join(os.path.dirname(d_path), "run_config.yml")
                has_cfg = os.path.exists(cfg_path)
                
                # Get timestamp from the directory itself
                mtime = os.path.getmtime(d_path)
                dt = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                
                model_name = "Unknown"
                if has_cfg:
                    try:
                        with open(cfg_path, 'r') as f:
                            cfg_data = yaml.safe_load(f)
                            if isinstance(cfg_data, dict):
                                model_name = cfg_data.get("_model", "Unknown")
                    except Exception:
                        pass
                
                # If model is still Unknown and it has a viz directory, deduce it from filenames!
                if model_name == "Unknown" and has_viz:
                    try:
                        viz_files = os.listdir(viz_dir)
                        has_w2v = any("word2vec" in vf for vf in viz_files)
                        has_enc = any("encoder" in vf for vf in viz_files)
                        if has_w2v:
                            # Velox produces word2vec (and optionally encoder too)
                            model_name = "Velox (Word2Vec)"
                        elif has_enc:
                            model_name = "Orthrus (GNN)"
                    except Exception:
                        pass

                # Find default ADP and Discrim (from the newest stat file, if available)
                default_adp = "-"
                default_disc = "-"
                if has_viz and model_name != "Velox (Word2Vec)":
                    import torch
                    pr_dir = os.path.join(d_path, "precision_recall_dir")
                    stats_files = glob.glob(os.path.join(pr_dir, "stats_model_epoch_*.pth"))
                    if stats_files:
                        stats_files.sort(key=os.path.getmtime, reverse=True)
                        try:
                            d = torch.load(stats_files[0], map_location="cpu")
                            adp_score = d.get("adp_score", 0.0)
                            discrim_score = d.get("discrimination", 0.0)
                            default_adp = f"{adp_score:.4f}"
                            default_disc = f"{discrim_score:.4f}"
                        except Exception:
                            pass

                self.runs_data.append({
                    "date": dt,
                    "mtime": mtime,
                    "dataset": dataset,
                    "model": model_name,
                    "hash": run_hash[:8],
                    "full_hash": run_hash,
                    "has_viz": has_viz,
                    "viz_dir": viz_dir,
                    "config_path": cfg_path if has_cfg else None,
                    "default_adp": default_adp,
                    "default_disc": default_disc
                })
            except Exception as e:
                pass

        # Sort by date descending
        self.runs_data.sort(key=lambda x: x["mtime"], reverse=True)
        
        self.table.setRowCount(len(self.runs_data))
        for r, run in enumerate(self.runs_data):
            self.table.setItem(r, 0, QTableWidgetItem(run["date"]))
            self.table.setItem(r, 1, QTableWidgetItem(run["dataset"]))
            self.table.setItem(r, 2, QTableWidgetItem(run["model"]))
            self.table.setItem(r, 3, QTableWidgetItem(run["default_adp"]))
            self.table.setItem(r, 4, QTableWidgetItem(run["default_disc"]))
            
            viz_status = "Available" if run["has_viz"] else "Missing"
            item_viz = QTableWidgetItem(viz_status)
            item_viz.setForeground(Qt.green if run["has_viz"] else Qt.red)
            self.table.setItem(r, 5, item_viz)
            
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.setWindowTitle("PIDSMaker - Local Run Browser")

    def on_selection_changed(self):
        sel = self.table.selectedItems()
        if not sel:
            self.btn_launch.setEnabled(False)
            self.right_widget.setEnabled(False)
            self.txt_config.clear()
            return
        
        row = sel[0].row()
        run = self.runs_data[row]
        
        self.right_widget.setEnabled(True)
        self.cmb_epoch.blockSignals(True)
        self.cmb_epoch.clear()
        
        if run["has_viz"]:
            self.selected_viz_dir = run["viz_dir"]
            self.selected_dataset = run["dataset"]
            
            # Populate combobox with available json files
            viz_files = glob.glob(os.path.join(run["viz_dir"], "*_points.json"))
            for vf in viz_files:
                basename = os.path.basename(vf)
                label = basename
                if "word2vec" in basename:
                    label = "Word2Vec Embedding"
                elif "encoder_epoch_" in basename:
                    # extract epoch number
                    parts = basename.split("encoder_epoch_")
                    if len(parts) > 1:
                        epoch_num = parts[1].split("_")[0]
                        label = f"GNN Encoder (Epoch {epoch_num})"
                elif "encoder" in basename:
                    label = "GNN Encoder (Final)"
                    
                self.cmb_epoch.addItem(label, vf) # label is text, vf is underlying data
                
            self.btn_launch.setEnabled(self.cmb_epoch.count() > 0)
        else:
            self.selected_viz_dir = None
            self.selected_dataset = None
            self.btn_launch.setEnabled(False)
            
        self.cmb_epoch.blockSignals(False)
        self.update_config_view()

    def update_config_view(self):
        sel = self.table.selectedItems()
        if not sel:
            return
            
        row = sel[0].row()
        run = self.runs_data[row]
        
        selected_file = self.cmb_epoch.currentData()
        
        stats_text = ""
        if selected_file and run["has_viz"]:
            try:
                import torch
                pr_dir = os.path.join(os.path.dirname(run["viz_dir"]), "precision_recall_dir")
                basename = os.path.basename(selected_file)
                
                stats_path = None
                if "word2vec" in basename:
                    self.table.item(row, 3).setText("-")
                    self.table.item(row, 4).setText("-")
                else:
                    if "encoder_epoch_" in basename:
                        ep_str = basename.split("encoder_epoch_")[1].split("_")[0]
                        stats_path = os.path.join(pr_dir, f"stats_model_epoch_{ep_str}.pth")
                    elif "encoder" in basename:
                        # Just grab the newest stats file for legacy encoder names
                        stats_files = glob.glob(os.path.join(pr_dir, "stats_model_epoch_*.pth"))
                        if stats_files:
                            stats_files.sort(key=os.path.getmtime, reverse=True)
                            stats_path = stats_files[0]
                    
                    if stats_path and os.path.exists(stats_path):
                        d = torch.load(stats_path, map_location="cpu")
                        adp = d.get("adp_score", 0.0)
                        disc = d.get("discrimination", 0.0)
                        self.table.item(row, 3).setText(f"{adp:.4f}")
                        self.table.item(row, 4).setText(f"{disc:.4f}")
                    else:
                        self.table.item(row, 3).setText("-")
                        self.table.item(row, 4).setText("-")
            except Exception as e:
                self.table.item(row, 3).setText("-")
                self.table.item(row, 4).setText("-")
            
        if run["config_path"]:
            try:
                with open(run["config_path"], 'r') as f:
                    cfg_data = yaml.safe_load(f)
                    cleaned_cfg = clean_dict(cfg_data)
                    
                    # Try to diff against base config
                    diffed = False
                    include_yml = cfg_data.get("_include_yml", "")
                    if include_yml:
                        pidsmaker_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
                        base_cfg_path = os.path.join(pidsmaker_root, "config", f"{include_yml}.yml")
                        if os.path.exists(base_cfg_path):
                            try:
                                with open(base_cfg_path, 'r') as bf:
                                    base_data = yaml.safe_load(bf)
                                    base_cleaned = clean_dict(base_data)
                                    diff_data = dict_diff(cleaned_cfg, base_cleaned)
                                    if diff_data is None:
                                        diff_data = {}
                                    stats_text += f"# DIFF FROM BASE CONFIG ({include_yml}.yml)\n# Only showing modified hyperparameters.\n\n"
                                    stats_text += yaml.dump(diff_data, default_flow_style=False, sort_keys=True)
                                    diffed = True
                            except Exception:
                                pass
                                
                    if not diffed:
                        stats_text += yaml.dump(cleaned_cfg, default_flow_style=False, sort_keys=True)
            except Exception as e:
                stats_text += f"Error loading config: {e}"
        else:
            stats_text += "# No run_config.yml found for this legacy run.\n# Hyperparameters are unavailable."
            
        self.txt_config.setPlainText(stats_text)

    def launch_visualizer(self):
        selected_file = self.cmb_epoch.currentData()
        if not selected_file:
            return
            
        pidsmaker_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        viz_script = os.path.join(pidsmaker_root, "scripts", "native_viz.py")
        
        # Change button text to show loading briefly
        self.btn_launch.setText("Launching Visualizer...")
        self.btn_launch.setEnabled(False)
        self.btn_launch.setStyleSheet("QPushButton { background-color: #f59e0b; color: white; padding: 10px; font-weight: bold; border-radius: 4px; }")
        self.btn_launch.repaint()
        
        try:
            # Use QProcess to read stdout dynamically and restore button
            process = QProcess(self)
            process.setProcessChannelMode(QProcess.MergedChannels)
            
            def handle_stdout():
                output = process.readAllStandardOutput().data().decode()
                print(output, end="") # Forward to console
                if "[GUI_READY]" in output or "Error during GUI rendering" in output:
                    self._restore_launch_button()
                    
            def handle_finished():
                self._restore_launch_button()
                # Clean up finished processes
                if process in self.active_processes:
                    self.active_processes.remove(process)
                    
            process.readyReadStandardOutput.connect(handle_stdout)
            process.finished.connect(handle_finished)
            
            # Start process
            process.start(sys.executable, [viz_script, "--file", selected_file])
            self.active_processes.append(process)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch visualizer:\n{e}")
            self._restore_launch_button()
            
    def _restore_launch_button(self):
        try:
            if not self.btn_launch.isEnabled():
                self.btn_launch.setText("Launch Visualizer")
                self.btn_launch.setEnabled(True)
                self.btn_launch.setStyleSheet("QPushButton { background-color: #3b82f6; color: white; padding: 10px; font-weight: bold; border-radius: 4px; } QPushButton:hover { background-color: #2563eb; }")
                self.btn_launch.repaint()
        except RuntimeError:
            # Widget might be deleted if window is closing
            pass
            
    def closeEvent(self, event):
        # Kill all spawned child processes if the user closes the Run Browser
        for process in self.active_processes:
            if process.state() != QProcess.NotRunning:
                process.kill()
                process.waitForFinished(1000)
        event.accept()
