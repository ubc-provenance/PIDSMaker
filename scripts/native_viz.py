#!/usr/bin/env python3
"""
PIDSMaker Native GPU Visualizer (PyQt5 + VisPy)
Handles 10M+ points natively on the GPU.
"""

import colorsys
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import vispy.scene
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from vispy.scene import visuals

# Color Map Definition
C = {
    # Benign (Semantic Coloring) - Brightened
    "benign_process": [0.8, 0.5, 1.0, 0.35],  # Bright Purple
    "benign_file": [0.3, 1.0, 0.7, 0.30],  # Bright Teal
    "benign_netflow": [0.5, 0.8, 1.0, 0.40],  # Bright Blue
    # Detected Malicious
    "det_process": [1.0, 0.2, 0.2, 0.9],  # Red
    "det_file": [1.0, 0.8, 0.0, 0.9],  # Yellow
    "det_netflow": [0.2, 0.8, 1.0, 0.9],  # Cyan
    # Undetected Malicious
    "undet_process": [1.0, 0.6, 0.2, 0.9],  # Orange
    "undet_file": [1.0, 1.0, 0.4, 0.9],  # Light Yellow
    "undet_netflow": [0.2, 0.8, 1.0, 0.9],  # Cyan
    "default": [0.5, 0.5, 0.5, 0.1],
}


def get_color(p):
    ptype = (p.get("type") or "").lower()
    label = p.get("label", 0)
    det = p.get("detection_status", 0)

    is_process = "process" in ptype or "subject" in ptype
    is_file = "file" in ptype
    is_netflow = "netflow" in ptype

    if label == 0:
        if is_process:
            return C["benign_process"]
        if is_file:
            return C["benign_file"]
        if is_netflow:
            return C["benign_netflow"]
        return C["default"]

    if det in (0, 1):  # Detected (or ground truth only)
        if is_process:
            return C["det_process"]
        if is_file:
            return C["det_file"]
        if is_netflow:
            return C["det_netflow"]
        return C["det_process"]

    if det == 2:  # Undetected
        if is_process:
            return C["undet_process"]
        if is_file:
            return C["undet_file"]
        if is_netflow:
            return C["undet_netflow"]
        return C["undet_process"]

    return C["default"]


class MainWindow(QMainWindow):
    def __init__(
        self,
        pos_hops,
        colors,
        sizes,
        metadata,
        stats,
        attack_edges,
        viz_cfg=None,
        enc_path=None,
        w2v_path=None,
        current_path=None,
    ):
        super().__init__()
        self.setWindowTitle("PIDSMaker Native GPU Visualizer")
        self.resize(1600, 900)

        self.pos_hops = pos_hops
        self.current_hop = len(pos_hops) - 1
        self.pos = self.pos_hops[self.current_hop]
        self.colors = colors
        self.sizes = sizes
        self.metadata = metadata
        self.stats = stats
        self.viz_cfg = viz_cfg or {}
        self.enc_path = enc_path
        self.w2v_path = w2v_path
        self.current_path = current_path
        self.attack_edges = attack_edges
        self.visible_mask = np.ones(len(self.pos), dtype=bool)
        self.precompute_filters()

        # Calculate State-Persistence arrays
        self.tw_indices = np.array(
            [m.get("tw_idx", 0) for m in self.metadata], dtype=np.float32
        )
        self.tw_start = self.tw_indices.copy()
        self.tw_end = np.full(len(self.pos), np.inf, dtype=np.float32)
        node_tws = defaultdict(list)
        for i, m in enumerate(self.metadata):
            node_tws[m.get("node_id")].append((m.get("tw_idx", 0), i))

        for nid, occurrences in node_tws.items():
            occurrences.sort(key=lambda x: x[0])
            for k in range(len(occurrences) - 1):
                idx = occurrences[k][1]
                next_tw = occurrences[k + 1][0]
                self.tw_end[idx] = next_tw

        self.node_tws = node_tws

        # Parse available epochs
        self.available_epochs = []
        if self.enc_path:
            viz_dir = os.path.dirname(self.enc_path)
            ds_name = os.path.basename(self.enc_path).split("_")[
                2
            ]  # embedding_viz_{dataset}_...
            import glob

            epoch_files = glob.glob(
                os.path.join(
                    viz_dir, f"embedding_viz_{ds_name}_encoder_epoch_*_points.json"
                )
            )
            for ef in epoch_files:
                try:
                    ep_num = int(os.path.basename(ef).split("_epoch_")[1].split("_")[0])
                    self.available_epochs.append((ep_num, ef))
                except:
                    pass
            self.available_epochs.sort(key=lambda x: x[0])

        # Main Layout
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setCentralWidget(central_widget)

        # --- Left Panel ---
        left_panel = QFrame()
        left_panel.setObjectName("leftPanel")
        left_panel.setStyleSheet("""
            #leftPanel { background-color: #111116; border-right: 1px solid #2a2a35; }
            QWidget { color: #e0e0e0; }
            QGroupBox { font-weight: bold; border: 1px solid #333344; border-radius: 6px; margin-top: 14px; padding-top: 12px; color: #a0a0b0; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 12px; top: 6px; padding: 0 4px; background-color: #111116; }
            QLineEdit { background-color: #1a1a24; border: 1px solid #333; border-radius: 4px; padding: 4px; color: white; }
            QPushButton { background-color: #2a2a35; border: 1px solid #444; border-radius: 4px; padding: 5px; font-weight: bold; }
            QPushButton:hover { background-color: #3a3a45; }
            
            /* Custom Sleek Checkboxes */
            QCheckBox { color: #e0e0e0; outline: none; spacing: 8px; font-weight: normal; }
            QCheckBox::indicator {
                width: 14px; height: 14px;
                border-radius: 7px;
                border: 2px solid #444;
                background-color: transparent;
            }
            QCheckBox::indicator:hover { border: 2px solid #666; background-color: rgba(255, 255, 255, 0.05); }
            QCheckBox::indicator:checked { background-color: #3b82f6; border: 2px solid #3b82f6; }
            QCheckBox::indicator:checked:hover { background-color: #60a5fa; border: 2px solid #60a5fa; }
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 20, 15, 20)
        left_layout.setSpacing(15)
        left_layout.setAlignment(Qt.AlignTop)

        title = QLabel("<h2>Node Inspector</h2>")
        title.setStyleSheet("color: white; border: none;")
        left_layout.addWidget(title)

        # 1. Controls
        grp_controls = QGroupBox("CONTROLS")
        v_ctrl = QVBoxLayout(grp_controls)

        self.max_coord = max(
            np.max(np.abs(self.pos[:, 0])), np.max(np.abs(self.pos[:, 1]))
        )
        self.center_pos = tuple(
            np.median(self.pos, axis=0)
        )

        h_pan_x = QHBoxLayout()
        h_pan_x.addWidget(QLabel("Pan X:"))
        self.slider_pan_x = QSlider(Qt.Horizontal)
        self.slider_pan_x.setRange(-100, 100)
        self.slider_pan_x.setValue(0)
        self.slider_pan_x.valueChanged.connect(self.update_camera_center)
        h_pan_x.addWidget(self.slider_pan_x)
        v_ctrl.addLayout(h_pan_x)

        h_pan_y = QHBoxLayout()
        h_pan_y.addWidget(QLabel("Pan Y:"))
        self.slider_pan_y = QSlider(Qt.Horizontal)
        self.slider_pan_y.setRange(-100, 100)
        self.slider_pan_y.setValue(0)
        self.slider_pan_y.valueChanged.connect(self.update_camera_center)
        h_pan_y.addWidget(self.slider_pan_y)
        v_ctrl.addLayout(h_pan_y)

        h_pan_z = QHBoxLayout()
        h_pan_z.addWidget(QLabel("Pan Z:"))
        self.slider_pan_z = QSlider(Qt.Horizontal)
        self.slider_pan_z.setRange(-100, 100)
        self.slider_pan_z.setValue(0)
        self.slider_pan_z.valueChanged.connect(self.update_camera_center)
        h_pan_z.addWidget(self.slider_pan_z)
        v_ctrl.addLayout(h_pan_z)

        h_hops = QHBoxLayout()
        self.lbl_hops = QLabel(f"Hops ({self.current_hop}):")
        h_hops.addWidget(self.lbl_hops)
        self.slider_hops = QSlider(Qt.Horizontal)
        self.slider_hops.setMaximum(len(self.pos_hops) - 1)
        self.slider_hops.setValue(self.current_hop)
        self.slider_hops.valueChanged.connect(self.on_hop_scrub)
        h_hops.addWidget(self.slider_hops)
        v_ctrl.addLayout(h_hops)

        # Determine max Time Window for the slider
        self.tw_indices = np.array(
            [m.get("tw_idx", 0) for m in self.metadata], dtype=np.float32
        )
        self.max_tw = int(np.max(self.tw_indices)) if len(self.tw_indices) > 0 else 0

        h_buttons = QHBoxLayout()

        btn_reset_cam = QPushButton("Reset Camera")
        btn_reset_cam.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_reset_cam.clicked.connect(self.reset_camera)
        h_buttons.addWidget(btn_reset_cam)

        btn_reset_home = QPushButton("Reset Home")
        btn_reset_home.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_reset_home.clicked.connect(self.reset_camera)
        h_buttons.addWidget(btn_reset_home)

        v_ctrl.addLayout(h_buttons)

        if len(self.available_epochs) > 0:
            h_epoch = QHBoxLayout()
            self.lbl_epoch = QLabel(f"Epoch: {self.available_epochs[-1][0]}")
            h_epoch.addWidget(self.lbl_epoch)
            self.slider_epoch = QSlider(Qt.Horizontal)
            self.slider_epoch.setRange(0, len(self.available_epochs) - 1)
            self.slider_epoch.setValue(len(self.available_epochs) - 1)
            self.slider_epoch.valueChanged.connect(self.on_epoch_scrub)
            h_epoch.addWidget(self.slider_epoch)
            v_ctrl.addLayout(h_epoch)

        self.chk_temporal = QCheckBox("3D Temporal Mode")
        self.chk_temporal.setChecked(True)
        self.chk_temporal.stateChanged.connect(self.toggle_3d_mode)
        v_ctrl.addWidget(self.chk_temporal)
        left_layout.addWidget(grp_controls)

        # 2. Filter
        grp_filter = QGroupBox("FILTER")
        v_filter = QVBoxLayout(grp_filter)
        self.chk_benign = QCheckBox("Benign")
        self.chk_benign.setChecked(True)
        self.chk_det = QCheckBox("Detected")
        self.chk_det.setChecked(True)
        self.chk_undet = QCheckBox("Undetected")
        self.chk_undet.setChecked(True)

        for chk in [self.chk_benign, self.chk_det, self.chk_undet]:
            chk.stateChanged.connect(self.update_scatter)
            v_filter.addWidget(chk)
        left_layout.addWidget(grp_filter)

        # 3. Search
        grp_search = QGroupBox("SEARCH")
        v_search = QVBoxLayout(grp_search)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Node ID or path...")
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.setInterval(300) # 300ms debounce
        self.search_timer.timeout.connect(self.update_scatter)
        self.search_box.textChanged.connect(self.search_timer.start)
        v_search.addWidget(self.search_box)
        left_layout.addWidget(grp_search)

        # 4. Global Statistics
        grp_stats = QGroupBox("GLOBAL STATISTICS")
        v_stats = QVBoxLayout(grp_stats)
        desc = QLabel(
            "Overall unique nodes present across all<br>time windows in this dataset projection."
        )
        desc.setStyleSheet("color: #a0a0b0;")
        v_stats.addWidget(desc)

        flay = QFormLayout()
        self.lbl_tot = QLabel(str(stats["total"]))
        self.lbl_tot.setStyleSheet("font-weight: bold; font-size: 14px; color: white;")
        flay.addRow(
            QLabel(
                "<span style='font-weight:bold; font-size: 14px; color: white;'>Total Nodes:</span>"
            ),
            self.lbl_tot,
        )

        self.lbl_ben = QLabel(str(stats["benign"]))
        self.lbl_ben.setStyleSheet("font-weight: bold; color: #10B981;")
        flay.addRow(
            QLabel("<span style='color: #10B981;'>Benign Nodes:</span>"), self.lbl_ben
        )

        self.lbl_mal = QLabel(str(stats["malicious"]))
        self.lbl_mal.setStyleSheet("font-weight: bold; color: #EF4444;")
        flay.addRow(
            QLabel("<span style='color: #EF4444;'>Malicious Nodes:</span>"),
            self.lbl_mal,
        )

        # Sub-stats
        self.lbl_mal_proc = QLabel(
            f"<span style='color: #EF4444;'>{stats['mal_proc']}</span>"
        )
        self.lbl_mal_net = QLabel(
            f"<span style='color: #3B82F6;'>{stats['mal_net']}</span>"
        )
        self.lbl_mal_file = QLabel(
            f"<span style='color: #F59E0B;'>{stats['mal_file']}</span>"
        )
        flay.addRow(
            QLabel(
                "<span style='color: #EF4444; margin-left: 10px;'>Processes:</span>"
            ),
            self.lbl_mal_proc,
        )
        flay.addRow(
            QLabel("<span style='color: #3B82F6; margin-left: 10px;'>Netflows:</span>"),
            self.lbl_mal_net,
        )
        flay.addRow(
            QLabel("<span style='color: #F59E0B; margin-left: 10px;'>Files:</span>"),
            self.lbl_mal_file,
        )
        v_stats.addLayout(flay)
        left_layout.addWidget(grp_stats)

        # 5. Overlays
        grp_overlays = QGroupBox("OVERLAYS")
        v_overlays = QVBoxLayout(grp_overlays)
        self.chk_traj = QCheckBox("Show Temporal Trajectories")
        self.chk_traj.setChecked(self.viz_cfg.get("show_trajectories", True))
        self.chk_traj.stateChanged.connect(self.apply_visual_state)
        v_overlays.addWidget(self.chk_traj)

        self.chk_attack = QCheckBox("Show Attack Graph")
        self.chk_attack.setChecked(False)
        self.chk_attack.stateChanged.connect(self.apply_visual_state)
        v_overlays.addWidget(self.chk_attack)

        self.chk_heat = QCheckBox("Discrimination Heatmap")
        self.chk_heat.setChecked(False)
        self.chk_heat.stateChanged.connect(self.update_scatter)
        v_overlays.addWidget(self.chk_heat)

        left_layout.addWidget(grp_overlays)

        # 6. Selected Node Info
        grp_info = QGroupBox("SELECTED NODE")
        v_info = QVBoxLayout(grp_info)
        self.info_lbl = QLabel("Click a point to inspect...")
        self.info_lbl.setWordWrap(True)
        self.info_lbl.setStyleSheet(
            "font-family: monospace; font-size: 12px; color: white;"
        )
        v_info.addWidget(self.info_lbl)
        left_layout.addWidget(grp_info)

        left_layout.addStretch()

        # --- VisPy 3D Canvas ---
        self.canvas3d = vispy.scene.SceneCanvas(
            keys="interactive", show=False, bgcolor="#050508"
        )
        self.view3d = self.canvas3d.central_widget.add_view()

        self.camera = vispy.scene.cameras.TurntableCamera(
            center=self.center_pos, distance=80, fov=45
        )
        self.view3d.camera = self.camera

        # Fast path background markers
        self.scatter = visuals.Markers(antialias=0)
        self.scatter.set_data(self.pos, edge_width=0, face_color=self.colors, size=self.sizes)
        self.view3d.add(self.scatter)

        # Fast path highlight markers (drawn on top)
        self.scatter_hl = visuals.Markers(antialias=0)
        self.scatter_hl.set_data(np.zeros((1, 3), dtype=np.float32), size=0)
        self.view3d.add(self.scatter_hl)

        # Trajectory Line
        self.trajectory_line = visuals.Line(
            antialias=True, width=1.5, parent=self.view3d.scene
        )

        # Attack Graph Edges
        self.attack_lines = visuals.Line(
            connect="segments", antialias=True, width=1.5, parent=self.view3d.scene
        )

        from vispy.visuals.transforms import STTransform

        axis = visuals.XYZAxis(parent=self.view3d.scene)

        # Scale axis so it's large enough to see (relative to dataset size)
        scale_factor = (
            max(np.max(np.abs(self.pos[:, 0])), np.max(np.abs(self.pos[:, 1]))) / 3.0
        )
        if scale_factor < 1.0:
            scale_factor = 5.0

        axis.transform = STTransform(
            scale=(scale_factor, scale_factor, scale_factor), translate=self.center_pos
        )

        scroll_area = QScrollArea()
        scroll_area.setFixedWidth(380)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea { border: none; background-color: #111116; border-right: 1px solid #2a2a35; }
            QScrollBar:vertical { background: #1a1a24; width: 10px; margin: 0px; }
            QScrollBar::handle:vertical { background: #3a3a45; min-height: 20px; border-radius: 5px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)
        scroll_area.setWidget(left_panel)

        main_layout.addWidget(scroll_area)
        main_layout.addWidget(self.canvas3d.native)

        # --- OVERLAYS (Glassmorphism) ---
        glass_style = """
            QFrame { background-color: rgba(15, 15, 20, 200); border: 1px solid rgba(100, 180, 255, 0.2); border-radius: 8px; }
            QLabel { color: white; background: transparent; border: none; font-weight: bold; }
            QPushButton { background-color: rgba(59, 130, 246, 0.8); border: 1px solid rgba(100, 180, 255, 0.4); border-radius: 4px; padding: 6px; font-weight: bold; color: white; }
            QPushButton:hover { background-color: rgba(37, 99, 235, 1.0); }
        """

        # 1. Bottom Right (Stats & Swap)
        self.overlay_br = QFrame(self.canvas3d.native)
        self.overlay_br.setStyleSheet(glass_style)
        ov_layout = QVBoxLayout(self.overlay_br)

        if self.viz_cfg.get("show_stats_overlay", True):
            self.lbl_model = QLabel(
                "Model: "
                + (
                    "GNN (Encoder)"
                    if "encoder" in (self.current_path or "")
                    else "Word2Vec"
                )
            )
            ov_layout.addWidget(self.lbl_model)
            ov_layout.addWidget(QLabel("Total Nodes: " + str(self.stats["total"])))

        if (
            self.enc_path
            and self.w2v_path
            and os.path.exists(self.enc_path)
            and os.path.exists(self.w2v_path)
        ):
            self.btn_toggle_emb = QPushButton("Switch Embedding Space")
            self.btn_toggle_emb.clicked.connect(self.toggle_embeddings)
            ov_layout.addWidget(self.btn_toggle_emb)

        self.overlay_br.adjustSize()

        # 2. Bottom Center (Time Slider & Play)
        self.overlay_bc = QFrame(self.canvas3d.native)
        self.overlay_bc.setStyleSheet(glass_style)
        h_tw = QHBoxLayout(self.overlay_bc)

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self.toggle_play)
        h_tw.addWidget(self.btn_play)

        self.combo_speed = QComboBox()
        self.combo_speed.setView(QListView())
        self.combo_speed.addItems(["20x", "40x", "60x", "80x", "100x"])
        self.combo_speed.setStyleSheet("""
            QComboBox { 
                background-color: #2a2a3b; 
                border: 1px solid #4a4a5b; 
                border-radius: 4px; 
                padding: 4px 8px; 
                color: white; 
                font-weight: bold;
            }
            QComboBox:focus {
                outline: none;
                border: 1px solid #3b82f6;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a3b;
                color: white;
                selection-background-color: #3b82f6;
                border: 1px solid #4a4a5b;
                outline: none;
                padding: 0px;
                margin: 0px;
            }
            QComboBox QListView {
                background-color: #2a2a3b;
                border: 1px solid #4a4a5b;
                outline: none;
            }
        """)
        h_tw.addWidget(self.combo_speed)

        self.btn_reset_tw = QPushButton("↺ Reset")
        self.btn_reset_tw.clicked.connect(self.reset_time)
        h_tw.addWidget(self.btn_reset_tw)

        h_tw.addWidget(QLabel("Time:"))
        self.slider_tw = QSlider(Qt.Horizontal)
        self.slider_tw.setRange(-100, self.max_tw * 100)
        self.slider_tw.setValue(-100)
        self.slider_tw.setFixedWidth(400)
        self.slider_tw.valueChanged.connect(self.update_tw_label)
        self.slider_tw.sliderPressed.connect(self.pause_playback)
        self.lbl_tw_val = QLabel("All")
        self.lbl_tw_val.setFixedWidth(40)
        h_tw.addWidget(self.slider_tw)
        h_tw.addWidget(self.lbl_tw_val)
        self.overlay_bc.adjustSize()

        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.on_play_tick)

        # 3. Top Right (Legend)
        self.overlay_tr = QFrame(self.canvas3d.native)
        self.overlay_tr.setFixedWidth(280)
        self.overlay_tr.setStyleSheet(glass_style)
        v_leg_main = QVBoxLayout(self.overlay_tr)

        h_leg_head = QHBoxLayout()
        lbl_leg_title = QLabel("LEGEND")
        self.btn_collapse_leg = QPushButton("▲")
        self.btn_collapse_leg.setFixedSize(24, 24)
        self.btn_collapse_leg.setStyleSheet(
            "QPushButton { background: transparent; border: none; color: white; font-size: 16px; } QPushButton:hover { color: #3b82f6; }"
        )
        self.btn_collapse_leg.clicked.connect(self.toggle_legend)
        h_leg_head.addWidget(lbl_leg_title)
        h_leg_head.addStretch()
        h_leg_head.addWidget(self.btn_collapse_leg)
        v_leg_main.addLayout(h_leg_head)

        self.wgt_legend_items = QWidget()
        v_leg = QVBoxLayout(self.wgt_legend_items)
        v_leg.setContentsMargins(0, 5, 0, 0)

        legend_html = """
        <style>
            td { padding: 2px 8px 2px 0px; font-size: 11px; }
            th { text-align: left; padding-top: 8px; font-size: 12px; color: #ffffff; }
        </style>
        <table cellspacing="0" cellpadding="0">
            <tr><th colspan="2">Benign</th></tr>
            <tr><td><span style='color: rgb(204, 127, 255);'>■</span> Process</td>
                <td><span style='color: rgb(76, 255, 178);'>■</span> File</td>
                <td><span style='color: rgb(127, 204, 255);'>■</span> Netflow</td></tr>
                
            <tr><th colspan="2">Detected Malicious</th></tr>
            <tr><td><span style='color: rgb(255, 51, 51);'>■</span> Process</td>
                <td><span style='color: rgb(255, 204, 0);'>■</span> File</td>
                <td><span style='color: rgb(51, 204, 255);'>■</span> Netflow</td></tr>
                
            <tr><th colspan="2">Undetected Malicious</th></tr>
            <tr><td><span style='color: rgb(255, 153, 51);'>■</span> Process</td>
                <td><span style='color: rgb(255, 255, 102);'>■</span> File</td>
                <td><span style='color: rgb(51, 204, 255);'>■</span> Netflow</td></tr>
        </table>
        """
        self.lbl_leg_nodes = QLabel(legend_html)
        self.lbl_leg_nodes.setStyleSheet("color: #e0e0e8;")
        v_leg.addWidget(self.lbl_leg_nodes)

        self.lbl_leg_edges = QLabel(
            "<br><b>Temporal Trajectories:</b><br>"
            "Color: <span style='color:#60a5fa'>Blue (Start)</span> → <span style='color:#ef4444'>Red (End)</span><br><br>"
            "<b>Attack Graph Overlays:</b><br>"
            "<span style='color:#eab308'>■ Yellow Edge (30%)</span>: Unactivated<br>"
            "<span style='color:#ef4444'>■ Red Edge (85%->65%)</span>: Activated"
        )
        self.lbl_leg_edges.setWordWrap(True)
        self.lbl_leg_edges.setStyleSheet("color: #e0e0e8; font-size: 11px;")
        v_leg.addWidget(self.lbl_leg_edges)

        v_leg_main.addWidget(self.wgt_legend_items)
        self.overlay_tr.adjustSize()

        self.canvas3d.events.mouse_press.connect(self.on_mouse_press)
        self.canvas3d.events.mouse_release.connect(self.on_mouse_release)
        self.press_pos = None
        self.selected_node_id = None

    def update_overlay_pos(self):
        cw = self.canvas3d.native.width()
        ch = self.canvas3d.native.height()

        if hasattr(self, "overlay_br"):
            self.overlay_br.adjustSize()
            self.overlay_br.move(
                cw - self.overlay_br.width() - 20, ch - self.overlay_br.height() - 20
            )

        if hasattr(self, "overlay_tr"):
            self.overlay_tr.adjustSize()
            self.overlay_tr.move(cw - self.overlay_tr.width() - 20, 20)

        if hasattr(self, "overlay_bc"):
            self.overlay_bc.adjustSize()
            self.overlay_bc.move(
                (cw - self.overlay_bc.width()) // 2, ch - self.overlay_bc.height() - 20
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_overlay_pos()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(50, self.update_overlay_pos)

    def toggle_embeddings(self):
        if "word2vec" in self.current_path and self.enc_path:
            self.current_path = self.enc_path
        elif "encoder" in self.current_path and self.w2v_path:
            self.current_path = self.w2v_path
        else:
            return

        print(f"Hot-swapping to {os.path.basename(self.current_path)}...")
        self.btn_toggle_emb.setText("Loading Embeddings...")
        self.btn_toggle_emb.setEnabled(False)
        QApplication.processEvents()

        try:
            pos_hops, colors, sizes, metadata, stats, attack_edges = load_data(
                self.current_path
            )
            self.pos_hops = pos_hops
            self.current_hop = len(pos_hops) - 1
            self.pos = self.pos_hops[self.current_hop]
            self.colors = colors
            self.sizes = sizes
            self.metadata = metadata
            self.stats = stats
            self.attack_edges = attack_edges
            self.visible_mask = np.ones(len(self.pos), dtype=bool)
            self.precompute_filters()
        finally:
            self.btn_toggle_emb.setText("Switch Embedding Space")
            self.btn_toggle_emb.setEnabled(True)

        self.tw_indices = np.array(
            [m.get("tw_idx", 0) for m in self.metadata], dtype=np.float32
        )
        self.max_tw = int(np.max(self.tw_indices)) if len(self.tw_indices) > 0 else 0
        self.slider_tw.setRange(-100, self.max_tw * 100)

        # Calculate State-Persistence arrays
        self.tw_start = self.tw_indices.copy()
        self.tw_end = np.full(len(self.pos), np.inf, dtype=np.float32)
        node_tws = defaultdict(list)
        for i, m in enumerate(self.metadata):
            node_tws[m.get("node_id")].append((m.get("tw_idx", 0), i))

        for nid, occurrences in node_tws.items():
            occurrences.sort(key=lambda x: x[0])
            for k in range(len(occurrences) - 1):
                idx = occurrences[k][1]
                next_tw = occurrences[k + 1][0]
                self.tw_end[idx] = next_tw

        self.max_coord = max(
            np.max(np.abs(self.pos[:, 0])), np.max(np.abs(self.pos[:, 1]))
        )
        self.center_pos = tuple(
            np.median(self.pos, axis=0)
        )
        self.reset_camera()

        if hasattr(self, "lbl_model"):
            self.lbl_model.setText(
                "Model: "
                + ("GNN (Encoder)" if "encoder" in self.current_path else "Word2Vec")
            )
            self.overlay_br.adjustSize()

        if hasattr(self, "lbl_tot"):
            self.lbl_tot.setText(str(stats["total"]))
            self.lbl_ben.setText(str(stats["benign"]))
            self.lbl_mal.setText(str(stats["malicious"]))
            self.lbl_mal_proc.setText(
                f"<span style='color: #EF4444;'>{stats['mal_proc']}</span>"
            )
            self.lbl_mal_net.setText(
                f"<span style='color: #3B82F6;'>{stats['mal_net']}</span>"
            )
            self.lbl_mal_file.setText(
                f"<span style='color: #F59E0B;'>{stats['mal_file']}</span>"
            )

        self.update_scatter()

    def on_epoch_scrub(self, idx):
        ep_num, ef_path = self.available_epochs[idx]
        self.lbl_epoch.setText(f"Epoch: {ep_num}")
        self.current_path = ef_path

        print(f"Scrubbing to Epoch {ep_num}...")
        with open(self.current_path, "r") as f:
            pts = json.load(f)

        for i, p in enumerate(pts):
            self.pos[i] = [p["x"], p["y"], p["z"]]

        if hasattr(self, "lbl_model"):
            self.lbl_model.setText(f"Model: GNN (Epoch {ep_num})")
            self.overlay_br.adjustSize()

        self.apply_visual_state()

    def on_hop_scrub(self, val):
        self.current_hop = val
        self.lbl_hops.setText(f"Hops ({val}):")
        self.pos = self.pos_hops[val]
        self.apply_visual_state()

    def update_tw_label(self):
        v = self.slider_tw.value()
        if v < 0:
            self.lbl_tw_val.setText("All")
        else:
            self.lbl_tw_val.setText(f"{v/100.0:.1f}")
        self.apply_visual_state()

    def toggle_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.btn_play.setText("▶ Play")
        else:
            if self.slider_tw.value() >= self.slider_tw.maximum():
                self.slider_tw.setValue(-100)
            self.play_timer.start(100)  # 10 fps
            self.btn_play.setText("⏸ Pause")

    def pause_playback(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.btn_play.setText("▶ Play")

    def reset_time(self):
        if self.play_timer.isActive():
            self.toggle_play()
        self.slider_tw.setValue(-100)

    def on_play_tick(self):
        speed_txt = self.combo_speed.currentText().replace("x", "")
        multiplier = int(speed_txt) if speed_txt.isdigit() else 1

        # Base tick is 10 slider units (0.10 TWs per 100ms = 1 TW per second)
        v = self.slider_tw.value() + (10 * multiplier)
        if v >= self.slider_tw.maximum():
            v = self.slider_tw.maximum()
            self.slider_tw.setValue(v)
            self.toggle_play()
        else:
            self.slider_tw.setValue(v)

    def toggle_legend(self):
        is_vis = self.wgt_legend_items.isVisible()
        self.wgt_legend_items.setVisible(not is_vis)
        self.btn_collapse_leg.setText("▼" if is_vis else "▲")
        self.overlay_tr.setFixedSize(self.overlay_tr.layout().sizeHint())
        self.update_overlay_pos()

    def toggle_3d_mode(self, state=None):
        if self.chk_temporal.isChecked():
            self.camera = vispy.scene.cameras.TurntableCamera(
                center=self.center_pos, distance=80, fov=45, azimuth=0, elevation=30
            )
        else:
            self.camera = vispy.scene.cameras.PanZoomCamera(aspect=1)
            self.camera.rect = (
                self.center_pos[0] - self.max_coord,
                self.center_pos[1] - self.max_coord,
                self.max_coord * 2,
                self.max_coord * 2,
            )
        
        self.view3d.camera = self.camera
        self.apply_visual_state()

    def update_camera_center(self):
        # Scale the -100 to 100 slider range directly to the data coordinate bounds, offsetting from the true center
        cx = self.center_pos[0] + (self.slider_pan_x.value() / 100.0) * self.max_coord
        cy = self.center_pos[1] + (self.slider_pan_y.value() / 100.0) * self.max_coord
        cz = self.center_pos[2] + (self.slider_pan_z.value() / 100.0) * self.max_coord
        
        if isinstance(self.camera, vispy.scene.cameras.TurntableCamera):
            self.camera.center = (cx, cy, cz)
        elif hasattr(self.camera, 'center'):
            self.camera.center = (cx, cy)

    def reset_camera(self):
        self.slider_pan_x.setValue(0)
        self.slider_pan_y.setValue(0)
        self.slider_pan_z.setValue(0)
        if isinstance(self.camera, vispy.scene.cameras.TurntableCamera):
            self.camera.center = self.center_pos
            self.camera.distance = 80
            self.camera.azimuth = 0
            self.camera.elevation = 30
        else:
            self.camera.rect = (
                self.center_pos[0] - self.max_coord,
                self.center_pos[1] - self.max_coord,
                self.max_coord * 2,
                self.max_coord * 2,
            )

    def reset_hops(self):
        self.slider_hops.setValue(0)

    def precompute_filters(self):
        self.benign_mask = np.array([m.get("label", 0) == 0 for m in self.metadata], dtype=bool)
        self.det_mask = np.array([m.get("label", 0) == 1 and m.get("detection_status", 0) in (0, 1) for m in self.metadata], dtype=bool)
        self.undet_mask = np.array([m.get("label", 0) == 1 and m.get("detection_status", 0) == 2 for m in self.metadata], dtype=bool)
        self.search_corpus = [str(m.get("node_id", "")) + " " + m.get("path", "").lower() for m in self.metadata]

    def update_scatter(self):
        show_benign = self.chk_benign.isChecked()
        show_det = self.chk_det.isChecked()
        show_undet = self.chk_undet.isChecked()
        search_txt = self.search_box.text().strip().lower()

        mask = np.zeros(len(self.metadata), dtype=bool)
        if show_benign:
            mask |= self.benign_mask
        if show_det:
            mask |= self.det_mask
        if show_undet:
            mask |= self.undet_mask

        if search_txt:
            for i in np.where(mask)[0]:
                if search_txt not in self.search_corpus[i]:
                    mask[i] = False

        self.visible_mask = mask
        self.apply_visual_state()

    def apply_visual_state(self):
        if not self.visible_mask.any():
            self.scatter.set_data(np.zeros((1, 3), dtype=np.float32), size=0)
            self.scatter_hl.set_data(np.zeros((1, 3), dtype=np.float32), size=0)
            return

        display_colors = self.colors.copy()
        
        if hasattr(self, "chk_heat") and self.chk_heat.isChecked():
            from vispy.color import Colormap
            cm = Colormap(['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921'])
            scores = np.array([m.get("anomaly_score", 0.0) for m in self.metadata])
            max_score = np.max(scores) if len(scores) > 0 and np.max(scores) > 0 else 1.0
            norm_scores = np.clip(scores / max_score, 0, 1)
            heatmap_colors = cm.map(norm_scores)
            heatmap_colors[:, 3] = display_colors[:, 3]
            display_colors[self.benign_mask] = heatmap_colors[self.benign_mask]

        render_pos = self.pos.copy()
        if hasattr(self, "chk_temporal") and not self.chk_temporal.isChecked():
            render_pos[:, 2] = 0.0
            display_colors[self.benign_mask, 3] *= 0.15  # Drop benign opacity to 15% in 2D mode

        match_mask = np.zeros(len(self.metadata), dtype=bool)
        if hasattr(self, "selected_node_id") and self.selected_node_id is not None:
            # Dim all nodes but keep them somewhat visible
            display_colors[:, 3] *= 0.4
            # Find all instances of this exact node ID
            match_mask = np.array(
                [m.get("node_id") == self.selected_node_id for m in self.metadata]
            )

            # --- Trajectory Line ---
            if self.chk_traj.isChecked():
                indices = np.where(match_mask)[0]
                pts_full = [
                    (self.metadata[i].get("tw_idx", 0), render_pos[i]) for i in indices
                ]
                pts_full.sort(key=lambda x: x[0])

                # Filter points based on current time slider (if not "All")
                t_val = self.slider_tw.value() / 100.0
                fraction = 0.0
                if t_val >= 0:
                    pts = []
                    for p in pts_full:
                        if p[0] <= t_val:
                            pts.append(p)
                        else:
                            break

                    # Smoothly interpolate the growing edge towards the next point
                    if len(pts) > 0 and len(pts) < len(pts_full):
                        p_prev = pts[-1]
                        p_next = pts_full[len(pts)]
                        if p_next[0] > p_prev[0]:
                            fraction = (t_val - p_prev[0]) / (p_next[0] - p_prev[0])
                            interp_pos = p_prev[1] + fraction * (p_next[1] - p_prev[1])
                            pts.append((t_val, interp_pos))
                else:
                    pts = pts_full

                if len(pts) > 1:
                    line_pos = np.array([p[1] for p in pts])
                    line_colors = np.zeros((len(line_pos), 4))
                    for i in range(len(line_pos)):
                        # Maintain consistent color across time by using full trajectory length for mapping
                        if i == len(line_pos) - 1 and fraction > 0.0:
                            orig_idx = i - 1 + fraction
                        else:
                            orig_idx = i
                        ratio = orig_idx / max(1, len(pts_full) - 1)
                        # Rainbow from Blue (0.65) to Red (0.0)
                        h = 0.65 - (ratio * 0.65)
                        rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)

                        if len(line_pos) > 1:
                            age_ratio = i / (len(line_pos) - 1)
                        else:
                            age_ratio = 1.0

                        # Head glows bright (1.0), tail dims (0.2)
                        traj_opacity = 0.2 + (0.8 * (age_ratio**3))
                        line_colors[i] = [rgb[0], rgb[1], rgb[2], traj_opacity]
                    self.trajectory_line.set_data(pos=line_pos, color=line_colors)
                else:
                    self.trajectory_line.set_data(
                        pos=np.zeros((2, 3), dtype=np.float32), color=(0, 0, 0, 0)
                    )
            else:
                self.trajectory_line.set_data(
                    pos=np.zeros((2, 3), dtype=np.float32), color=(0, 0, 0, 0)
                )
        else:
            self.trajectory_line.set_data(
                pos=np.zeros((2, 3), dtype=np.float32), color=(0, 0, 0, 0)
            )

        # Apply Time Window State-Persistence Model
        t_val = self.slider_tw.value() / 100.0
        time_mask = np.ones(len(self.metadata), dtype=bool)

        if t_val >= 0:
            # Active if it's the MOST RECENT known state for a node up to time t_val
            time_mask = (self.tw_start <= t_val) & (t_val < self.tw_end)

            # Performance optimization: Only compute thermal heatmap on active points
            active_idx = np.where(time_mask)[0]
            if len(active_idx) > 0:
                # Calculate age (how long since this node was last active)
                age = t_val - self.tw_start[active_idx]

                # Thermal Heatmap: Actively firing nodes glow Hot Yellow/White and cool down
                blend_factor = np.clip(1.0 - (age / 1.5), 0.0, 1.0)
                hot_color = np.array([1.0, 1.0, 0.8], dtype=np.float32)
                for c_idx in range(3):
                    display_colors[active_idx, c_idx] = (
                        display_colors[active_idx, c_idx] * (1.0 - blend_factor)
                    ) + (hot_color[c_idx] * blend_factor)

                # Ghosting: fade to 0.50
                alphas = np.clip(1.0 - (age * 0.3), 0.50, 1.0)
                display_colors[active_idx, 3] *= alphas

        # --- Attack Graph ---
        if (
            hasattr(self, "chk_attack")
            and self.chk_attack.isChecked()
            and hasattr(self, "attack_edges")
            and self.attack_edges
        ):
            edge_pos = []
            edge_colors = []
            t_val_actual = (
                self.slider_tw.value() / 100.0
                if self.slider_tw.value() >= 0
                else float("inf")
            )

            for u, v in self.attack_edges:
                u_occ = self.node_tws.get(u, [])
                v_occ = self.node_tws.get(v, [])
                if not u_occ or not v_occ:
                    continue

                u_first = u_occ[0][0]
                v_first = v_occ[0][0]
                activation_time = max(u_first, v_first)

                u_idx = u_occ[0][1]
                if t_val_actual != float("inf"):
                    for tw, arr_idx in u_occ:
                        if tw <= t_val_actual:
                            u_idx = arr_idx
                        else:
                            break
                else:
                    u_idx = u_occ[-1][1]

                v_idx = v_occ[0][1]
                if t_val_actual != float("inf"):
                    for tw, arr_idx in v_occ:
                        if tw <= t_val_actual:
                            v_idx = arr_idx
                        else:
                            break
                else:
                    v_idx = v_occ[-1][1]

                # Dynamic Opacity & Color
                if t_val_actual < activation_time:
                    rgb = [1.0, 0.8, 0.0]  # Yellow
                    opacity = 0.30  # Future (Unactivated)
                elif (
                    t_val_actual >= activation_time
                    and t_val_actual < activation_time + 4.0
                ):
                    rgb = [0.93, 0.26, 0.26]  # Red
                    opacity = 0.85  # Just fired! Bright
                else:
                    rgb = [0.93, 0.26, 0.26]  # Red
                    opacity = 0.65  # Past (Lingering)

                if t_val_actual == float("inf"):
                    rgb = [0.93, 0.26, 0.26]  # Red
                    opacity = 0.65

                edge_pos.extend([render_pos[u_idx], render_pos[v_idx]])
                edge_colors.extend(
                    [
                        [rgb[0], rgb[1], rgb[2], opacity],
                        [rgb[0], rgb[1], rgb[2], opacity],
                    ]
                )

            if edge_pos:
                self.attack_lines.set_data(
                    pos=np.array(edge_pos, dtype=np.float32),
                    color=np.array(edge_colors, dtype=np.float32),
                )
            else:
                self.attack_lines.set_data(
                    pos=np.zeros((2, 3), dtype=np.float32), color=(0, 0, 0, 0)
                )
        else:
            self.attack_lines.set_data(
                pos=np.zeros((2, 3), dtype=np.float32), color=(0, 0, 0, 0)
            )

        # Performance: Hide inactive points via alpha instead of array resizing to avoid VBO reallocation
        display_colors[~time_mask, 3] = 0.0

        # Draw background nodes (fast path uses array sizes now)
        bg_mask = self.visible_mask & (~match_mask) & time_mask
        if bg_mask.any():
            self.scatter.set_data(
                render_pos[bg_mask],
                edge_width=0,
                face_color=display_colors[bg_mask],
                size=self.sizes[bg_mask],
            )
        else:
            self.scatter.set_data(np.zeros((1, 3), dtype=np.float32), size=0)

        # Draw highlighted nodes on top (fast path: scalar size=12)
        if (match_mask & time_mask).any():
            hl_mask = self.visible_mask & match_mask & time_mask
            self.scatter_hl.set_data(
                render_pos[hl_mask],
                edge_width=0,
                face_color=[1.0, 1.0, 1.0, 1.0],  # Bright white
                size=12,
            )
        else:
            self.scatter_hl.set_data(np.zeros((1, 3), dtype=np.float32), size=0)

        # Update dynamic legend
        if hasattr(self, "lbl_leg_benign"):
            show_ben = self.chk_benign.isChecked() and (self.stats["benign"] > 0)
            self.lbl_leg_benign.setVisible(show_ben)

            show_det = self.chk_det.isChecked() and (self.stats["malicious"] > 0)
            self.lbl_leg_det.setVisible(show_det)

            show_undet = self.chk_undet.isChecked() and (self.stats["malicious"] > 0)
            self.lbl_leg_undet.setVisible(show_undet)

            show_edges = self.selected_node_id is not None
            self.lbl_leg_edges.setVisible(show_edges)

            self.overlay_tr.setFixedSize(self.overlay_tr.layout().sizeHint())
            self.update_overlay_pos()

    def on_mouse_press(self, event):
        if event.button == 1:
            self.press_pos = event.pos

    def on_mouse_release(self, event):
        if event.button != 1 or self.press_pos is None:
            return

        # Differentiate between a "click" and a "camera drag"
        dx = event.pos[0] - self.press_pos[0]
        dy = event.pos[1] - self.press_pos[1]
        if (dx**2 + dy**2) ** 0.5 > 5:  # Dragged more than 5 pixels
            return

        click_x, click_y = event.pos

        # Get visible points
        visible_pos = self.pos[self.visible_mask]
        visible_indices = np.where(self.visible_mask)[0]

        if len(visible_pos) == 0:
            return

        # Use VisPy's built-in transform chain to go from Data -> Screen Pixels
        tr = self.scatter.get_transform("visual", "document")
        projected = tr.map(visible_pos)

        w = projected[:, 3].reshape(-1, 1)
        w[w == 0] = 1e-5

        pts_2d = projected[:, :2] / w

        # Find closest point in 2D screen space
        dist_sq = (pts_2d[:, 0] - click_x) ** 2 + (pts_2d[:, 1] - click_y) ** 2

        # Don't click points behind the camera
        dist_sq[w[:, 0] <= 0] = np.inf

        min_idx_local = np.argmin(dist_sq)
        if dist_sq[min_idx_local] < 100:  # 10 pixel radius
            actual_idx = visible_indices[min_idx_local]
            self.show_node(actual_idx)
        else:
            # Clicked empty space -> reset highlight
            self.selected_node_id = None
            self.info_lbl.setText("Click a point to inspect...")
            self.apply_visual_state()

    def show_node(self, idx):
        m = self.metadata[idx]
        self.selected_node_id = m.get("node_id")

        text = f"<b>ID:</b> <span style='color:#a0a0ff'>{m['node_id']}</span><br>"
        text += f"<b>Type:</b> {m.get('type', 'Unknown')}<br>"
        label_html = "<span style='color:#ef4444'>Malicious</span>" if m.get("label") == 1 else "<span style='color:#10b981'>Benign</span>"
        text += f"<b>Label:</b> {label_html}<br>"
        text += f"<b>Time Window:</b> {m.get('tw_label', 'Unknown')}<br>"
        if m.get("anomaly_score"):
            text += f"<b>Anomaly Score:</b> {m.get('anomaly_score', 0):.4f}<br>"
        if m.get("top_edge"):
            text += f"<b>Top Edge:</b> {m.get('top_edge', '')}<br>"
        text += "<br>"
        
        if "path" in m and m["path"]:
            text += f"<b>Path:</b> {m['path']}"

        self.info_lbl.setText(text)
        self.apply_visual_state()


def load_data(path):
    print(f"Loading {path}...")
    t0 = time.time()
    with open(path, "r") as f:
        pts = json.load(f)
    print(f"Loaded {len(pts)} points in {time.time()-t0:.2f}s")

    num_hops = len(pts[0].get("coords_hops", [[0,0,0]])) if "coords_hops" in pts[0] else 1
    pos_hops = [np.zeros((len(pts), 3), dtype=np.float32) for _ in range(num_hops)]
    colors = np.zeros((len(pts), 4), dtype=np.float32)
    sizes = np.zeros(len(pts), dtype=np.float32)

    stats = {
        "total": len(pts),
        "benign": 0,
        "malicious": 0,
        "mal_proc": 0,
        "mal_file": 0,
        "mal_net": 0,
    }

    for i, p in enumerate(pts):
        if "coords_hops" in p:
            for h in range(num_hops):
                pos_hops[h][i] = p["coords_hops"][h][:3]
        else:
            pos_hops[0][i] = [p.get("x", 0), p.get("y", 0), p.get("z", 0)]
            
        colors[i] = get_color(p)

        # Calc stats
        lbl = p.get("label", 0)
        ptype = (p.get("type") or "").lower()
        if lbl == 0:
            stats["benign"] += 1
            sizes[i] = 3.0
        else:
            stats["malicious"] += 1
            sizes[i] = 5.0  # Slightly larger for visibility (user request)
            if "process" in ptype or "subject" in ptype:
                stats["mal_proc"] += 1
            elif "file" in ptype:
                stats["mal_file"] += 1
            elif "netflow" in ptype:
                stats["mal_net"] += 1

    adj_path = path.replace("_points.json", "_adj.json")
    attack_edges = []
    if os.path.exists(adj_path):
        print(f"Loading adjacency list from {os.path.basename(adj_path)}...")
        with open(adj_path, "r") as f:
            adj = json.load(f)

        malicious_nodes = set()
        for p in pts:
            if p.get("label", 0) == 1:
                malicious_nodes.add(str(p["node_id"]))

        edge_set = set()
        for u, neighbors in adj.items():
            if u in malicious_nodes:
                for v in neighbors:
                    if str(v) in malicious_nodes:
                        pair = tuple(sorted([int(u), int(v)]))
                        edge_set.add(pair)
        attack_edges = list(edge_set)
        print(f"Extracted {len(attack_edges)} attack graph edges")

    return pos_hops, colors, sizes, pts, stats, attack_edges


def resolve_latest_viz_dir(dataset):
    """Find the viz directory for the given dataset.

    Checks viz_manifest.json first (exact paths), then falls back to globbing.
    """
    import glob

    pidsmaker_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.path.exists("/home/artifacts"):
        artifacts_root = "/home/artifacts"
    else:
        artifacts_root = os.environ.get(
            "PIDS_ARTIFACTS_DIR", os.path.join(pidsmaker_root, "artifacts")
        )

    # 1. Try manifest-first: look for viz_manifest.json
    manifest_patterns = [
        os.path.join(
            artifacts_root, "evaluation/evaluation/*", dataset, "viz_manifest.json"
        ),
        os.path.join(
            artifacts_root, "detection/evaluation/*", dataset, "viz_manifest.json"
        ),
    ]
    manifests = []
    for pattern in manifest_patterns:
        manifests.extend(glob.glob(pattern))
    if manifests:
        manifests.sort(key=os.path.getmtime, reverse=True)
        eval_dir = os.path.dirname(manifests[0])
        viz_dir = os.path.join(eval_dir, "viz")
        if os.path.isdir(viz_dir):
            return viz_dir

    # 2. Fallback: glob for viz dirs
    viz_dirs = []
    for base in ("evaluation/evaluation", "detection/evaluation"):
        viz_dirs.extend(
            glob.glob(os.path.join(artifacts_root, base, "*", dataset, "viz"))
        )
    if not viz_dirs:
        viz_dirs = glob.glob(os.path.join(artifacts_root, "viz"))
        if not viz_dirs:
            return None

    viz_dirs.sort(key=os.path.getmtime, reverse=True)
    return viz_dirs[0]


def main():
    import argparse

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
        print(
            "[!] If running in Docker, ensure X11 forwarding is configured in docker-compose.yml."
        )
        print(
            "[!] You may need to run 'xhost +local:docker' on your host terminal before launching."
        )
        sys.exit(1)

    try:
        app = QApplication(sys.argv)
    except Exception as e:
        print(f"\n[!] Error initializing GUI: {e}")
        print(
            "[!] Please ensure your X11 server accepts connections (e.g. run 'xhost +local:docker')."
        )
        sys.exit(1)

    viz_dir = resolve_latest_viz_dir(args.dataset)
    if not viz_dir:
        # One last fallback
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
    
    # If the default encoder file isn't there, we might have run with --all_epochs
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
        current_theme = "encoder" if "encoder" in load_path else "word2vec"
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
        print(
            "[!] This is usually caused by missing OpenGL libraries or X11 permission errors."
        )
        print("[!] Try running 'xhost +local:docker' on your host.")
        sys.exit(1)


if __name__ == "__main__":
    main()
