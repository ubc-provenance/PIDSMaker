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
    QTextEdit,
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
                
        from PyQt5.QtWidgets import QComboBox
        v_epoch = QVBoxLayout()
        
        window.lbl_epoch = QLabel(f"Loaded: Epoch {window.available_epochs[current_idx][0]}")
        window.lbl_epoch.setStyleSheet("color: #9ca3af; font-size: 12px; font-weight: bold;")
        v_epoch.addWidget(window.lbl_epoch)
        
        window.cmb_epoch_select = QComboBox()
        window.cmb_epoch_select.setStyleSheet("""
            QComboBox {
                background-color: #2b2b36; color: white; padding: 5px 10px; border: 1px solid #3f3f4e; border-radius: 4px; font-weight: bold;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2b2b36; color: white; selection-background-color: #3b82f6; outline: none; border: 1px solid #3f3f4e;
            }
            QComboBox:disabled { background-color: #1e1e24; color: #6b7280; }
        """)
        for ep_num, _ in window.available_epochs:
            window.cmb_epoch_select.addItem(f"Epoch {ep_num}", ep_num)
        
        window.cmb_epoch_select.setCurrentIndex(current_idx)
        window.cmb_epoch_select.currentIndexChanged.connect(window.on_epoch_scrub)
        v_epoch.addWidget(window.cmb_epoch_select)
        
        v_ctrl.addLayout(v_epoch)

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
    
    h_csv = QHBoxLayout()
    window.btn_load_csv = QPushButton("Load CSV...")
    window.btn_load_csv.clicked.connect(window.load_node_csv)
    h_csv.addWidget(window.btn_load_csv)
    
    window.btn_clear_csv = QPushButton("Clear CSV")
    window.btn_clear_csv.clicked.connect(window.clear_node_csv)
    window.btn_clear_csv.hide()
    h_csv.addWidget(window.btn_clear_csv)
    v_search.addLayout(h_csv)
    
    window.lbl_csv_status = QLabel("")
    window.lbl_csv_status.setStyleSheet("color: #a0a0b0; font-size: 11px;")
    window.lbl_csv_status.hide()
    v_search.addWidget(window.lbl_csv_status)
    
    window.lbl_csv_terms = QLabel("")
    window.lbl_csv_terms.setWordWrap(True)
    window.lbl_csv_terms.setStyleSheet("color: #60a5fa; font-size: 11px; background-color: #1a1a24; padding: 4px; border-radius: 4px; border: 1px solid #333;")
    window.lbl_csv_terms.hide()
    v_search.addWidget(window.lbl_csv_terms)

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
        window.lbl_metrics_global.setText(f"ADP: {window.stats['adp']:.3f} | Discrim: {window.stats['disc_score']:.3f}")
    else:
        window.lbl_metrics_global.setText("N/A")
    flay.addRow(
        QLabel("<span style='font-weight:bold; font-size: 12px; color: #10B981;'>Performance:</span>"),
        window.lbl_metrics_global,
    )

    if window.stats.get("attack_start_tw", float('inf')) != float('inf'):
        tw = window.stats["attack_start_tw"]
        tm = window.stats.get("attack_start_time", "")
        txt = f"<span style='color:#fca636'>Window {tw}</span> <span style='font-size:10px;color:#a0a0b0'>({tm})</span>" if tm else f"<span style='color:#fca636'>Window {tw}</span>"
        lbl_astart = QLabel(txt)
        lbl_astart.setStyleSheet("font-weight: bold; font-size: 12px;")
        flay.addRow(QLabel("<span style='font-weight:bold; font-size: 12px; color: #fca636;'>Attack Start:</span>"), lbl_astart)

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

    grp_config = QGroupBox("RUN CONFIGURATION")
    v_config = QVBoxLayout(grp_config)
    txt_config = QTextEdit()
    txt_config.setReadOnly(True)
    txt_config.setStyleSheet("font-family: monospace; font-size: 11px; color: #a3e635; background-color: #111115; border: 1px solid #3f3f4e;")
    cfg_text = window.stats.get("run_config", "")
    if cfg_text:
        txt_config.setPlainText(cfg_text)
    else:
        txt_config.setPlainText("# No run_config.yml found for this run.\n# Hyperparameters are unavailable.")
    v_config.addWidget(txt_config)
    
    btn_expand_cfg = QPushButton("Expand Configuration")
    btn_expand_cfg.setStyleSheet("""
        QPushButton {
            background-color: #2a2a35; color: #a3e635; border: 1px solid #4a4a55; padding: 4px; border-radius: 3px; font-weight: bold;
        }
        QPushButton:hover { background-color: #3a3a45; }
    """)
    def show_full_config():
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
        dlg = QDialog(window)
        dlg.setWindowTitle("Full Run Configuration")
        dlg.resize(800, 600)
        dlg.setStyleSheet("QDialog { background-color: #111115; }")
        dl = QVBoxLayout(dlg)
        dl.setContentsMargins(10, 10, 10, 10)
        
        dt = QTextEdit()
        dt.setReadOnly(True)
        dt.setStyleSheet("""
            QTextEdit {
                font-family: monospace; font-size: 13px; color: #a3e635; 
                background-color: #0b0b0d; border: 1px solid #2a2a35; border-radius: 4px; padding: 10px;
            }
            QScrollBar:vertical { background: #1a1a24; width: 10px; margin: 0px; }
            QScrollBar::handle:vertical { background: #3a3a45; min-height: 20px; border-radius: 5px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)
        dt.setPlainText(cfg_text if cfg_text else "# No run_config.yml found.")
        dl.addWidget(dt)
        
        btn_close = QPushButton("Close")
        btn_close.setStyleSheet("""
            QPushButton {
                background-color: #2a2a35; color: white; border: none; padding: 8px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #e53e3e; }
        """)
        btn_close.clicked.connect(dlg.accept)
        dl.addWidget(btn_close)
        
        dlg.exec_()
    btn_expand_cfg.clicked.connect(show_full_config)
    v_config.addWidget(btn_expand_cfg)
    left_layout.addWidget(grp_config)

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
