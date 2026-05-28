from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

def setup_left_panel(window):
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

    grp_controls = QGroupBox("CONTROLS")
    v_ctrl = QVBoxLayout(grp_controls)

    h_pan_x = QHBoxLayout()
    h_pan_x.addWidget(QLabel("Pan X:"))
    window.slider_pan_x = QSlider(Qt.Horizontal)
    window.slider_pan_x.setRange(-100, 100)
    window.slider_pan_x.setValue(0)
    window.slider_pan_x.valueChanged.connect(window.update_camera_center)
    h_pan_x.addWidget(window.slider_pan_x)
    v_ctrl.addLayout(h_pan_x)

    h_pan_y = QHBoxLayout()
    h_pan_y.addWidget(QLabel("Pan Y:"))
    window.slider_pan_y = QSlider(Qt.Horizontal)
    window.slider_pan_y.setRange(-100, 100)
    window.slider_pan_y.setValue(0)
    window.slider_pan_y.valueChanged.connect(window.update_camera_center)
    h_pan_y.addWidget(window.slider_pan_y)
    v_ctrl.addLayout(h_pan_y)

    h_pan_z = QHBoxLayout()
    h_pan_z.addWidget(QLabel("Pan Z:"))
    window.slider_pan_z = QSlider(Qt.Horizontal)
    window.slider_pan_z.setRange(-100, 100)
    window.slider_pan_z.setValue(0)
    window.slider_pan_z.valueChanged.connect(window.update_camera_center)
    h_pan_z.addWidget(window.slider_pan_z)
    v_ctrl.addLayout(h_pan_z)

    h_hops = QHBoxLayout()
    window.lbl_hops = QLabel(f"Hops ({window.current_hop}):")
    h_hops.addWidget(window.lbl_hops)
    window.slider_hops = QSlider(Qt.Horizontal)
    window.slider_hops.setMaximum(len(window.pos_hops) - 1)
    window.slider_hops.setValue(window.current_hop)
    window.slider_hops.valueChanged.connect(window.on_hop_scrub)
    h_hops.addWidget(window.slider_hops)
    v_ctrl.addLayout(h_hops)

    h_buttons = QHBoxLayout()

    btn_reset_cam = QPushButton("Reset Camera")
    btn_reset_cam.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    btn_reset_cam.clicked.connect(window.reset_camera)
    h_buttons.addWidget(btn_reset_cam)

    btn_reset_home = QPushButton("Reset Home")
    btn_reset_home.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    btn_reset_home.clicked.connect(window.reset_camera)
    h_buttons.addWidget(btn_reset_home)

    v_ctrl.addLayout(h_buttons)

    if len(window.available_epochs) > 0:
        current_idx = len(window.available_epochs) - 1
        for i, (ep_num, ef_path) in enumerate(window.available_epochs):
            if ef_path == window.current_path:
                current_idx = i
                break
                
        h_epoch = QHBoxLayout()
        window.lbl_epoch = QLabel(f"Epoch: {window.available_epochs[current_idx][0]}")
        h_epoch.addWidget(window.lbl_epoch)
        window.slider_epoch = QSlider(Qt.Horizontal)
        window.slider_epoch.setRange(0, len(window.available_epochs) - 1)
        window.slider_epoch.setValue(current_idx)
        window.slider_epoch.valueChanged.connect(window.on_epoch_slider_moved)
        window.slider_epoch.sliderReleased.connect(window.on_epoch_scrub)
        h_epoch.addWidget(window.slider_epoch)
        v_ctrl.addLayout(h_epoch)

    window.chk_temporal = QCheckBox("3D Temporal Mode")
    window.chk_temporal.setChecked(True)
    window.chk_temporal.stateChanged.connect(window.toggle_3d_mode)
    v_ctrl.addWidget(window.chk_temporal)
    left_layout.addWidget(grp_controls)

    grp_filter = QGroupBox("FILTER")
    v_filter = QVBoxLayout(grp_filter)
    window.chk_benign = QCheckBox("Benign")
    window.chk_benign.setChecked(True)
    window.chk_det = QCheckBox("Detected")
    window.chk_det.setChecked(True)
    window.chk_undet = QCheckBox("Undetected")
    window.chk_undet.setChecked(True)

    for chk in [window.chk_benign, window.chk_det, window.chk_undet]:
        chk.stateChanged.connect(window.update_scatter)
        v_filter.addWidget(chk)
    left_layout.addWidget(grp_filter)

    grp_search = QGroupBox("SEARCH")
    v_search = QVBoxLayout(grp_search)
    window.search_box = QLineEdit()
    window.search_box.setPlaceholderText("Node ID or path...")
    window.search_timer = QTimer()
    window.search_timer.setSingleShot(True)
    window.search_timer.setInterval(300)
    window.search_timer.timeout.connect(window.update_scatter)
    window.search_box.textChanged.connect(window.search_timer.start)
    v_search.addWidget(window.search_box)
    left_layout.addWidget(grp_search)

    grp_stats = QGroupBox("GLOBAL STATISTICS")
    v_stats = QVBoxLayout(grp_stats)
    desc = QLabel(
        "Overall unique nodes present across all<br>time windows in this dataset projection."
    )
    desc.setStyleSheet("color: #a0a0b0;")
    v_stats.addWidget(desc)

    flay = QFormLayout()
    
    window.lbl_ds_name = QLabel(getattr(window, 'dataset_name', 'Unknown'))
    window.lbl_ds_name.setStyleSheet("font-weight: bold; font-size: 13px; color: #60A5FA;")
    flay.addRow(
        QLabel("<span style='font-weight:bold; font-size: 13px; color: #60A5FA;'>Dataset:</span>"),
        window.lbl_ds_name,
    )
    
    window.lbl_metrics_global = QLabel("")
    window.lbl_metrics_global.setStyleSheet("font-weight: bold; font-size: 12px; color: #10B981;")
    if window.stats.get("adp") is not None:
        window.lbl_metrics_global.setText(f"ADP: {window.stats['adp']:.3f} | Disc: {window.stats['disc_score']:.3f}")
    else:
        window.lbl_metrics_global.setText("N/A")
    flay.addRow(
        QLabel("<span style='font-weight:bold; font-size: 12px; color: #10B981;'>Performance:</span>"),
        window.lbl_metrics_global,
    )

    window.lbl_tot = QLabel(str(window.stats["total"]))
    window.lbl_tot.setStyleSheet("font-weight: bold; font-size: 14px; color: white;")
    flay.addRow(
        QLabel(
            "<span style='font-weight:bold; font-size: 14px; color: white;'>Total Nodes:</span>"
        ),
        window.lbl_tot,
    )

    window.lbl_ben = QLabel(str(window.stats["benign"]))
    window.lbl_ben.setStyleSheet("font-weight: bold; color: #10B981;")
    flay.addRow(
        QLabel("<span style='color: #10B981;'>Benign Nodes:</span>"), window.lbl_ben
    )

    window.lbl_mal = QLabel(str(window.stats["malicious"]))
    window.lbl_mal.setStyleSheet("font-weight: bold; color: #EF4444;")
    flay.addRow(
        QLabel("<span style='color: #EF4444;'>Malicious Nodes:</span>"),
        window.lbl_mal,
    )

    window.lbl_mal_proc = QLabel(
        f"<span style='color: #EF4444;'>{window.stats['mal_proc']}</span>"
    )
    window.lbl_mal_net = QLabel(
        f"<span style='color: #3B82F6;'>{window.stats['mal_net']}</span>"
    )
    window.lbl_mal_file = QLabel(
        f"<span style='color: #F59E0B;'>{window.stats['mal_file']}</span>"
    )
    flay.addRow(
        QLabel(
            "<span style='color: #EF4444; margin-left: 10px;'>Processes:</span>"
        ),
        window.lbl_mal_proc,
    )
    flay.addRow(
        QLabel("<span style='color: #3B82F6; margin-left: 10px;'>Netflows:</span>"),
        window.lbl_mal_net,
    )
    flay.addRow(
        QLabel("<span style='color: #F59E0B; margin-left: 10px;'>Files:</span>"),
        window.lbl_mal_file,
    )
    v_stats.addLayout(flay)
    left_layout.addWidget(grp_stats)

    grp_overlays = QGroupBox("OVERLAYS")
    v_overlays = QVBoxLayout(grp_overlays)
    window.chk_traj = QCheckBox("Show Temporal Trajectories")
    window.chk_traj.setChecked(window.viz_cfg.get("show_trajectories", True))
    window.chk_traj.stateChanged.connect(window.apply_visual_state)
    v_overlays.addWidget(window.chk_traj)

    window.chk_attack = QCheckBox("Show Attack Graph")
    window.chk_attack.setChecked(False)
    window.chk_attack.stateChanged.connect(window.apply_visual_state)
    v_overlays.addWidget(window.chk_attack)

    window.chk_heat = QCheckBox("Discrimination Heatmap")
    window.chk_heat.setChecked(False)
    window.chk_heat.stateChanged.connect(window.update_scatter)
    v_overlays.addWidget(window.chk_heat)

    left_layout.addWidget(grp_overlays)

    grp_info = QGroupBox("SELECTED NODE")
    v_info = QVBoxLayout(grp_info)
    window.info_lbl = QLabel("Click a point to inspect...")
    window.info_lbl.setWordWrap(True)
    window.info_lbl.setStyleSheet(
        "font-family: monospace; font-size: 12px; color: white;"
    )
    v_info.addWidget(window.info_lbl)
    left_layout.addWidget(grp_info)

    left_layout.addStretch()
    
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

    return scroll_area


def setup_overlays(window):
    glass_style = """
        QFrame { background-color: rgba(15, 15, 20, 200); border: 1px solid rgba(100, 180, 255, 0.2); border-radius: 8px; }
        QLabel { color: white; background: transparent; border: none; font-weight: bold; }
        QPushButton { background-color: rgba(59, 130, 246, 0.8); border: 1px solid rgba(100, 180, 255, 0.4); border-radius: 4px; padding: 6px; font-weight: bold; color: white; }
        QPushButton:hover { background-color: rgba(37, 99, 235, 1.0); }
    """

    window.overlay_br = QFrame(window.canvas3d.native)
    window.overlay_br.setStyleSheet(glass_style)
    window.overlay_br.setMaximumWidth(300)
    ov_layout = QVBoxLayout(window.overlay_br)

    if window.viz_cfg.get("show_stats_overlay", True):
        window.lbl_model = QLabel(
            "Model: "
            + (
                "GNN (Encoder)"
                if "encoder" in (window.current_path or "")
                else "Word2Vec"
            )
        )
        ov_layout.addWidget(window.lbl_model)
        ov_layout.addWidget(QLabel("Total Nodes: " + str(window.stats["total"])))

        # Use the global reference since we moved it
        window.lbl_metrics = window.lbl_metrics_global

    import os
    if (
        window.enc_path
        and window.w2v_path
        and os.path.exists(window.enc_path)
        and os.path.exists(window.w2v_path)
    ):
        window.btn_toggle_emb = QPushButton("Switch Embedding Space")
        window.btn_toggle_emb.clicked.connect(window.toggle_embeddings)
        ov_layout.addWidget(window.btn_toggle_emb)

    window.lbl_status = QLabel("")
    window.lbl_status.setStyleSheet("color: #60a5fa; font-weight: normal; font-style: italic; font-size: 12px;")
    window.lbl_status.setWordWrap(True)
    ov_layout.addWidget(window.lbl_status)

    window.overlay_br.adjustSize()

    window.overlay_bc = QFrame(window.canvas3d.native)
    window.overlay_bc.setStyleSheet(glass_style)
    h_tw = QHBoxLayout(window.overlay_bc)

    window.btn_play = QPushButton("▶ Play")
    window.btn_play.clicked.connect(window.toggle_play)
    h_tw.addWidget(window.btn_play)

    window.combo_speed = QComboBox()
    window.combo_speed.setView(QListView())
    window.combo_speed.addItems(["20x", "40x", "60x", "80x", "100x"])
    window.combo_speed.setStyleSheet("""
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
    h_tw.addWidget(window.combo_speed)

    window.btn_reset_tw = QPushButton("↺ Reset")
    window.btn_reset_tw.clicked.connect(window.reset_time)
    h_tw.addWidget(window.btn_reset_tw)

    h_tw.addWidget(QLabel("Time:"))
    window.slider_tw = QSlider(Qt.Horizontal)
    window.slider_tw.setRange(-100, window.max_tw * 100)
    window.slider_tw.setValue(-100)
    window.slider_tw.setFixedWidth(250)
    window.slider_tw.valueChanged.connect(window.update_tw_label)
    window.slider_tw.sliderPressed.connect(window.pause_playback)
    window.lbl_tw_val = QLabel("All")
    window.lbl_tw_val.setFixedWidth(40)
    h_tw.addWidget(window.slider_tw)
    h_tw.addWidget(window.lbl_tw_val)
    window.overlay_bc.adjustSize()

    window.play_timer = QTimer()
    window.play_timer.timeout.connect(window.on_play_tick)

    window.overlay_tr = QFrame(window.canvas3d.native)
    window.overlay_tr.setFixedWidth(280)
    window.overlay_tr.setStyleSheet(glass_style)
    v_leg_main = QVBoxLayout(window.overlay_tr)

    h_leg_head = QHBoxLayout()
    lbl_leg_title = QLabel("LEGEND")
    window.btn_collapse_leg = QPushButton("▲")
    window.btn_collapse_leg.setFixedSize(24, 24)
    window.btn_collapse_leg.setStyleSheet(
        "QPushButton { background: transparent; border: none; color: white; font-size: 16px; } QPushButton:hover { color: #3b82f6; }"
    )
    window.btn_collapse_leg.clicked.connect(window.toggle_legend)
    h_leg_head.addWidget(lbl_leg_title)
    h_leg_head.addStretch()
    h_leg_head.addWidget(window.btn_collapse_leg)
    v_leg_main.addLayout(h_leg_head)

    window.wgt_legend_items = QWidget()
    v_leg = QVBoxLayout(window.wgt_legend_items)
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
    window.lbl_leg_nodes = QLabel(legend_html)
    window.lbl_leg_nodes.setStyleSheet("color: #e0e0e8;")
    v_leg.addWidget(window.lbl_leg_nodes)

    window.lbl_leg_edges = QLabel(
        "<br><b>Temporal Trajectories:</b><br>"
        "Color: <span style='color:#60a5fa'>Blue (Start)</span> → <span style='color:#ef4444'>Red (End)</span><br><br>"
        "<b>Attack Graph Overlays:</b><br>"
        "<span style='color:#eab308'>■ Yellow Edge (30%)</span>: Unactivated<br>"
        "<span style='color:#ef4444'>■ Red Edge (85%->65%)</span>: Activated"
    )
    window.lbl_leg_edges.setWordWrap(True)
    window.lbl_leg_edges.setStyleSheet("color: #e0e0e8; font-size: 11px;")
    v_leg.addWidget(window.lbl_leg_edges)

    v_leg_main.addWidget(window.wgt_legend_items)
    window.overlay_tr.adjustSize()
