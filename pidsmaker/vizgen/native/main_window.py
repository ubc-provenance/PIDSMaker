import os
import colorsys
import numpy as np
from collections import defaultdict
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QApplication
from PyQt5.QtCore import QTimer, QThread, pyqtSignal

import vispy.scene
from vispy.scene import visuals

from .shaders import TemporalMarkers
from .ui_components import setup_left_panel, setup_overlays
from .loader import load_data
import json

class DataLoaderThread(QThread):
    dataLoaded = pyqtSignal(object, int, int) # data_tuple, current_hop, ep_num

    def __init__(self, file_path, ep_num, current_hop):
        super().__init__()
        self.file_path = file_path
        self.ep_num = ep_num
        self.current_hop = current_hop

    def run(self):
        try:
            data = load_data(self.file_path)
            self.dataLoaded.emit(data, self.current_hop, self.ep_num)
        except Exception as e:
            print(f"Error loading data: {e}")
            self.dataLoaded.emit(None, self.current_hop, self.ep_num)


class MainWindow(QMainWindow):
    def __init__(
        self,
        pos_hops,
        colors,
        sizes,
        metadata,
        stats,
        attack_edges,
        full_adj=None,
        viz_cfg=None,
        enc_path=None,
        w2v_path=None,
        current_path=None,
        dataset_name="",
    ):
        super().__init__()
        self.setWindowTitle("PIDSMaker Native GPU Visualizer")
        
        self.dataset_name = dataset_name
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
        self.full_adj = full_adj or {}
        self.active_trace_nodes = None
        self.visible_mask = np.ones(len(self.pos), dtype=bool)
        self.precompute_filters()

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

        self.available_epochs = []
        if self.enc_path:
            viz_dir = os.path.dirname(self.enc_path)
            ds_name = os.path.basename(self.enc_path).split("_encoder_")[0].replace("embedding_viz_", "")
            import glob

            best_epoch_num = None
            manifest_path = os.path.join(os.path.dirname(viz_dir), "viz_manifest.json")
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, "r") as f:
                        import json
                        manifest = json.load(f)
                        if "epochs" in manifest and manifest["epochs"]:
                            sorted_epochs = sorted(manifest["epochs"], key=lambda x: x.get("adp", 0), reverse=True)
                            best_epoch_num = int(sorted_epochs[0].get("epoch", 0))
                except Exception:
                    pass

            best_enc_path = os.path.join(viz_dir, f"embedding_viz_{ds_name}_encoder_points.json")
            if os.path.exists(best_enc_path) and best_epoch_num is not None:
                self.available_epochs.append((best_epoch_num, best_enc_path))

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

        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setCentralWidget(central_widget)

        self.update_spatial_bounds()
        self.max_tw = int(np.max(self.tw_indices)) if len(self.tw_indices) > 0 else 0

        scroll_area = setup_left_panel(self)
        main_layout.addWidget(scroll_area)

        self.canvas3d = vispy.scene.SceneCanvas(
            keys="interactive", show=False, bgcolor="#050508"
        )
        self.view3d = self.canvas3d.central_widget.add_view()

        self.camera = vispy.scene.cameras.TurntableCamera(
            center=self.center_pos, distance=80, fov=45
        )
        self.view3d.camera = self.camera

        self.scatter = TemporalMarkers(antialias=0)
        self.scatter.set_data(self.pos, edge_width=0, face_color=self.colors, size=self.sizes)
        self.scatter.shared_program['a_tw_start'] = self.tw_start
        self.scatter.shared_program['a_tw_end'] = self.tw_end
        self.scatter.shared_program['u_time'] = -1.0
        self.view3d.add(self.scatter)

        self.scatter_hl = TemporalMarkers(antialias=0)
        self.scatter_hl.set_data(np.zeros((1, 3), dtype=np.float32), size=0)
        self.scatter_hl.shared_program['a_tw_start'] = np.array([0.0], dtype=np.float32)
        self.scatter_hl.shared_program['a_tw_end'] = np.array([np.inf], dtype=np.float32)
        self.scatter_hl.shared_program['u_time'] = -1.0
        self.view3d.add(self.scatter_hl)

        self.trajectory_line = visuals.Line(
            antialias=True, width=1.5, parent=self.view3d.scene
        )

        self.attack_lines = visuals.Line(
            connect="segments", antialias=True, width=1.5, parent=self.view3d.scene
        )

        from vispy.visuals.transforms import STTransform

        self.axis = visuals.XYZAxis(parent=self.view3d.scene)
        scale_factor = (
            max(np.max(np.abs(self.pos[:, 0])), np.max(np.abs(self.pos[:, 1]))) / 3.0
        )
        if scale_factor < 1.0:
            scale_factor = 5.0

        self.axis.transform = STTransform(
            scale=(scale_factor, scale_factor, scale_factor), translate=self.center_pos
        )

        main_layout.addWidget(self.canvas3d.native)

        setup_overlays(self)

        self.canvas3d.events.mouse_press.connect(self.on_mouse_press)
        self.canvas3d.events.mouse_release.connect(self.on_mouse_release)
        self.press_pos = None
        self.selected_node_id = None
        self.csv_filter_ids = set()

    def load_node_csv(self):
        from PyQt5.QtWidgets import QFileDialog
        import csv
        path, _ = QFileDialog.getOpenFileName(self, "Load Node CSV", "", "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)")
        if not path:
            return
        
        try:
            names = set()
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        val = row[0].strip().lower()
                        # Ignore common headers if they are accidentally included
                        if val and val not in ['name', 'path', 'id', 'node', 'category']:
                            names.add(val)

            self.csv_filter_names = names
            
            self.lbl_csv_status.setText(f"Loaded {len(names)} search terms:")
            self.lbl_csv_terms.setText(", ".join(sorted(list(names))))
            self.lbl_csv_terms.show()
            self.lbl_csv_status.show()
            self.btn_clear_csv.show()
            self.btn_load_csv.setText("Change...")
            
            self.update_scatter()
            self.focus_on_visible()
        except Exception as e:
            self.show_status(f"Error loading CSV: {e}", timeout=4000)

    def clear_node_csv(self):
        self.csv_filter_names = set()
        self.lbl_csv_status.hide()
        self.lbl_csv_terms.hide()
        self.lbl_csv_terms.setText("")
        self.btn_clear_csv.hide()
        self.btn_load_csv.setText("Load CSV...")
        self.update_scatter()
        self.update_spatial_bounds()

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
        if hasattr(self, "lbl_status") and not self.lbl_status.text():
            self.lbl_status.hide()

    def show_status(self, msg, timeout=0):
        if hasattr(self, "lbl_status"):
            self.lbl_status.setText(msg)
            self.lbl_status.setVisible(bool(msg))
            self.overlay_br.adjustSize()
            self.update_overlay_pos()
            QApplication.processEvents()
            
            if timeout > 0:
                QTimer.singleShot(timeout, lambda: self.show_status(""))

    def toggle_embeddings(self):
        if getattr(self, "_is_loading", False):
            print("A load operation is already in progress. Ignoring toggle request.")
            return

        if "word2vec" in self.current_path and self.enc_path:
            self.current_path = self.enc_path
        elif "encoder" in self.current_path and self.w2v_path:
            self.current_path = self.w2v_path
        else:
            return

        self._is_loading = True
        print(f"Hot-swapping to {os.path.basename(self.current_path)}...")
        self.btn_toggle_emb.setText("Loading Embeddings...")
        self.btn_toggle_emb.setEnabled(False)
        basename = os.path.basename(self.current_path)
        if len(basename) > 30:
            basename = basename[:12] + "..." + basename[-15:]
        self.show_status(f"Loading {basename}... Please wait.")

        try:
            pos_hops, colors, sizes, metadata, stats, attack_edges, full_adj = load_data(
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
            self.full_adj = full_adj
            self.visible_mask = np.ones(len(self.pos), dtype=bool)
            self.precompute_filters()
        finally:
            self._is_loading = False
            self.btn_toggle_emb.setText("Switch Embedding Space")
            self.btn_toggle_emb.setEnabled(True)
            self.show_status("Dataset loaded successfully.", timeout=3000)
            print(f"Finished switching embedding space to {os.path.basename(self.current_path)}.")

            if hasattr(self, "lbl_metrics"):
                if self.stats.get("adp") is not None:
                    self.lbl_metrics.setText(f"ADP: {self.stats['adp']:.3f}  |  Disc: {self.stats['disc_score']:.3f}")
                    self.lbl_metrics.show()
                else:
                    self.lbl_metrics.hide()

        self.tw_indices = np.array(
            [m.get("tw_idx", 0) for m in self.metadata], dtype=np.float32
        )
        self.max_tw = int(np.max(self.tw_indices)) if len(self.tw_indices) > 0 else 0
        self.slider_tw.setRange(-100, self.max_tw * 100)

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

        self.update_spatial_bounds()

        if hasattr(self, "slider_hops"):
            self.slider_hops.setMaximum(len(self.pos_hops) - 1)
            self.slider_hops.setValue(self.current_hop)

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

    def on_epoch_scrub(self):
        if getattr(self, "_is_loading", False):
            print("Already loading another epoch. Ignoring scrub request.")
            return

        self._is_loading = True
        idx = self.cmb_epoch_select.currentIndex()
        ep_num, ef_path = self.available_epochs[idx]
        self.lbl_epoch.setText(f"Loading Epoch {ep_num}...")
        basename = os.path.basename(ef_path)
        if len(basename) > 30:
            basename = basename[:12] + "..." + basename[-15:]
        self.show_status(f"Scrubbing to Epoch {ep_num} ({basename})...")
        if hasattr(self, "cmb_epoch_select"):
            self.cmb_epoch_select.setEnabled(False)

        self.current_path = ef_path
        print(f"Scrubbing to Epoch {ep_num}...")
        
        self.loader_thread = DataLoaderThread(ef_path, ep_num, self.current_hop)
        self.loader_thread.dataLoaded.connect(self.on_epoch_data_loaded)
        self.loader_thread.start()

    def on_epoch_data_loaded(self, data, current_hop, ep_num):
        self._is_loading = False
        if hasattr(self, "cmb_epoch_select"):
            self.cmb_epoch_select.setEnabled(True)
        if data:
            self.lbl_epoch.setText(f"Loaded: Epoch {ep_num}")
            self.lbl_epoch.setStyleSheet("color: #9ca3af; font-size: 12px; font-weight: bold;")
        if not data:
            self.lbl_epoch.setText(f"Error loading Epoch {ep_num}")
            self.lbl_epoch.setStyleSheet("color: #ef4444; font-size: 12px; font-weight: bold;")
            self.show_status(f"Failed to load Epoch {ep_num}.", timeout=3000)
            print(f"Error: Failed to load Epoch {ep_num}.")
            return
            
        pos_hops, colors, sizes, metadata, stats, attack_edges, full_adj = data
        self.pos_hops = pos_hops
        self.colors = colors
        self.sizes = sizes
        self.metadata = metadata
        self.stats = stats
        self.attack_edges = attack_edges
        self.full_adj = full_adj

        if hasattr(self, 'lbl_metrics_global'):
            if self.stats.get("adp") is not None:
                self.lbl_metrics_global.setText(f"ADP: {self.stats['adp']:.3f} | Discrim: {self.stats['disc_score']:.3f}")
            else:
                self.lbl_metrics_global.setText("N/A")
        if hasattr(self, 'lbl_tot'):
            self.lbl_tot.setText(str(self.stats.get("total", 0)))
        if hasattr(self, 'lbl_ben'):
            self.lbl_ben.setText(str(self.stats.get("benign", 0)))
        if hasattr(self, 'lbl_mal'):
            self.lbl_mal.setText(str(self.stats.get("malicious", 0)))

        self.precompute_filters()

        num_hops = len(pos_hops)
        
        self.show_status(f"Epoch {ep_num} loaded successfully.", timeout=3000)
        print(f"Finished loading Epoch {ep_num} ({num_hops} hops extracted).")
        
        self.current_hop = min(current_hop, num_hops - 1)
        
        if hasattr(self, "slider_hops"):
            self.slider_hops.setMaximum(num_hops - 1)
            self.slider_hops.setValue(self.current_hop)
            
        self.pos = self.pos_hops[self.current_hop]

        if hasattr(self, "lbl_model"):
            self.lbl_model.setText(f"Model: GNN (Epoch {ep_num})")
            self.overlay_br.adjustSize()

        if hasattr(self, "lbl_tot"):
            self.lbl_tot.setText(str(stats["total"]))
            self.lbl_ben.setText(str(stats["benign"]))
            self.lbl_mal.setText(str(stats["malicious"]))
            self.lbl_mal_proc.setText(f"<span style='color: #EF4444;'>{stats['mal_proc']}</span>")
            self.lbl_mal_net.setText(f"<span style='color: #3B82F6;'>{stats['mal_net']}</span>")
            self.lbl_mal_file.setText(f"<span style='color: #F59E0B;'>{stats['mal_file']}</span>")
            
        if hasattr(self, "lbl_metrics"):
            if stats.get("adp") is not None:
                self.lbl_metrics.setText(f"ADP: {stats['adp']:.3f} | Disc: {stats['disc_score']:.3f}")
            else:
                self.lbl_metrics.setText("N/A")

        self.tw_indices = np.array(
            [m.get("tw_idx", 0) for m in self.metadata], dtype=np.float32
        )
        self.max_tw = int(np.max(self.tw_indices)) if len(self.tw_indices) > 0 else 0
        if hasattr(self, "slider_tw"):
            self.slider_tw.setRange(-100, self.max_tw * 100)

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

        self.update_spatial_bounds()
        self.lbl_epoch.setText(f"Loaded: Epoch {ep_num}")
        self.lbl_epoch.setStyleSheet("color: #9ca3af; font-size: 12px; font-weight: bold;")
        self.update_scatter()

    def on_hop_scrub(self, val):
        self.current_hop = val
        self.lbl_hops.setText(f"Hops ({val}):")
        self.pos = self.pos_hops[val]
        self.update_spatial_bounds(reset_cam=False)
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
            self.play_timer.start(100)
            self.btn_play.setText("|| Pause")

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
        self.view3d.camera = self.camera
        self.apply_visual_state()

    def update_spatial_bounds(self, reset_cam=True):
        self.max_coord = max(
            np.max(np.abs(self.pos[:, 0])), np.max(np.abs(self.pos[:, 1]))
        ) if len(self.pos) > 0 else 1.0
        
        if len(self.pos) > 0:
            med = np.median(self.pos, axis=0)
            dist = np.sum((self.pos - med)**2, axis=1)
            threshold = np.percentile(dist, 99.5)
            core_pos = self.pos[dist <= threshold] if len(self.pos) > 0 else self.pos
            if len(core_pos) == 0: core_pos = self.pos
            self.center_pos = tuple((np.max(core_pos, axis=0) + np.min(core_pos, axis=0)) / 2.0)
        else:
            self.center_pos = (0, 0, 0)
            
        if hasattr(self, "axis"):
            scale_factor = (
                max(np.max(np.abs(self.pos[:, 0])), np.max(np.abs(self.pos[:, 1]))) / 3.0
            ) if len(self.pos) > 0 else 5.0
            
            if scale_factor < 1.0:
                scale_factor = 5.0
            from vispy.visuals.transforms import STTransform
            self.axis.transform = STTransform(
                scale=(scale_factor, scale_factor, scale_factor), translate=self.center_pos
            )
            
        if reset_cam and hasattr(self, "camera"):
            self.reset_camera()

    def update_camera_center(self):
        cx = self.center_pos[0] + (self.slider_pan_x.value() / 100.0) * self.max_coord
        cy = self.center_pos[1] + (self.slider_pan_y.value() / 100.0) * self.max_coord
        cz = self.center_pos[2] + (self.slider_pan_z.value() / 100.0) * self.max_coord
        
        if isinstance(self.camera, vispy.scene.cameras.TurntableCamera):
            self.camera.center = (cx, cy, cz)
        elif hasattr(self.camera, 'center'):
            self.camera.center = (cx, cy)

    def focus_on_visible(self):
        if not hasattr(self, 'visible_mask') or not self.visible_mask.any():
            return
            
        vis_pos = self.pos[self.visible_mask]
        
        self.max_coord = max(
            np.max(np.abs(vis_pos[:, 0])), np.max(np.abs(vis_pos[:, 1]))
        ) if len(vis_pos) > 0 else 1.0
        
        if len(vis_pos) > 0:
            self.center_pos = tuple((np.max(vis_pos, axis=0) + np.min(vis_pos, axis=0)) / 2.0)
        else:
            self.center_pos = (0, 0, 0)
            
        if hasattr(self, "camera"):
            self.reset_camera()

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

    def reset_selection(self):
        """Full reset: clear selection, stop playback, reset time slider and camera."""
        self.selected_node_id = None
        self.info_lbl.setText("Click a point to inspect...")

        if hasattr(self, "play_timer"):
            self.reset_time()
        elif hasattr(self, "slider_tw"):
            self.slider_tw.setValue(-100)

        if hasattr(self, "camera"):
            self.reset_camera()

        for attr in ("_last_bg_mask", "_last_display_colors", "_last_render_pos", "_last_display_sizes"):
            if hasattr(self, attr):
                delattr(self, attr)

        self.apply_visual_state()

    def reset_hops(self):
        self.slider_hops.setValue(0)

    def precompute_filters(self):
        self.benign_mask = np.array([m.get("label", 0) == 0 for m in self.metadata], dtype=bool)
        self.det_mask = np.array([m.get("label", 0) != 0 and m.get("detection_status", 0) in (0, 1) for m in self.metadata], dtype=bool)
        self.undet_mask = np.array([m.get("label", 0) != 0 and m.get("detection_status", 0) == 2 for m in self.metadata], dtype=bool)
        self.search_corpus = [str(m.get("node_id", "")) + " " + m.get("path", "").lower() for m in self.metadata]
        self.precompute_detection_cost()

    def precompute_detection_cost(self):
        """Sweep anomaly score thresholds to compute FP at full node recall and full campaign coverage."""
        scores = []
        unique_mal_ids = set()
        for i, m in enumerate(self.metadata):
            score = m.get("anomaly_score", 0.0) or 0.0
            label = m.get("label", 0)
            det = m.get("detection_status", 0)
            nid = m.get("node_id")
            campaign_ids = m.get("campaign_ids", [])
            scores.append((score, label, det, nid, campaign_ids))
            if label == 1:
                unique_mal_ids.add(nid)

        unique_det_ids = set()
        for m in self.metadata:
            if m.get("label") == 1 and m.get("detection_status") == 1:
                unique_det_ids.add(m.get("node_id"))

        scores.sort(key=lambda x: x[0], reverse=True)

        total_gt = len(unique_mal_ids)
        total_det = len(unique_det_ids)

        # Get total campaigns from stats (loaded from campaign_mapping.json)
        num_campaigns = self.stats.get("num_campaigns", 0)
        all_campaigns = set(range(num_campaigns)) if num_campaigns > 0 else set()

        # Detect current threshold from the data
        det_scores = [s[0] for s in scores if s[2] == 1]
        current_threshold = min(det_scores) if det_scores else 0.0
        current_fp = sum(1 for s in scores if s[1] == 0 and s[0] >= current_threshold)

        # Sweep for full node recall: detect every unique malicious node_id
        seen_nodes = set()
        fp_full_recall = 0
        thresh_full_recall = 0.0
        for score, label, det, nid, cids in scores:
            if label == 1:
                seen_nodes.add(nid)
            else:
                fp_full_recall += 1
            if len(seen_nodes) >= total_gt:
                thresh_full_recall = score
                break

        # Sweep for full campaign: at least one node from each campaign
        seen_campaigns = set()
        fp_full_campaign = 0
        thresh_full_campaign = 0.0
        if all_campaigns:
            for score, label, det, nid, cids in scores:
                if label == 1:
                    for cid in cids:
                        seen_campaigns.add(cid)
                else:
                    fp_full_campaign += 1
                if seen_campaigns >= all_campaigns:
                    thresh_full_campaign = score
                    break

        # Count currently detected campaigns
        det_campaigns = set()
        for m in self.metadata:
            if m.get("label") == 1 and m.get("detection_status") == 1:
                for cid in m.get("campaign_ids", []):
                    det_campaigns.add(cid)

        self.detection_cost = {
            "total_gt": total_gt,
            "total_det": total_det,
            "current_fp": current_fp,
            "current_threshold": current_threshold,
            "fp_full_recall": fp_full_recall,
            "thresh_full_recall": thresh_full_recall,
            "fp_full_campaign": fp_full_campaign,
            "thresh_full_campaign": thresh_full_campaign,
            "num_campaigns": num_campaigns,
            "det_campaigns": len(det_campaigns),
        }
        self.update_detection_cost_ui()

    def update_detection_cost_ui(self):
        if not hasattr(self, 'lbl_dc_gt'):
            return
        dc = self.detection_cost
        pct = dc['total_det'] / dc['total_gt'] * 100 if dc['total_gt'] > 0 else 0
        self.lbl_dc_gt.setText(f"{dc['total_gt']}")
        self.lbl_dc_det.setText(f"<span style='color:#60a5fa'>{dc['total_det']} / {dc['total_gt']}</span> <span style='color:#9ca3af'>({pct:.1f}%)</span>")
        self.lbl_dc_cur_fp.setText(f"{dc['current_fp']}")
        self.lbl_dc_full_recall.setText(f"<span style='color:#f87171'>{dc['fp_full_recall']:,}</span>")
        camp_pct = dc['det_campaigns'] / dc['num_campaigns'] * 100 if dc['num_campaigns'] > 0 else 0
        self.lbl_dc_full_campaign.setText(f"<span style='color:#fb923c'>{dc['fp_full_campaign']:,}</span>")
        self.lbl_dc_campaign_cov.setText(f"{dc['det_campaigns']} / {dc['num_campaigns']} ({camp_pct:.1f}%)")

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

        # Apply strict CSV filter based on names
        if hasattr(self, 'csv_filter_names') and len(self.csv_filter_names) > 0:
            for i in np.where(mask)[0]:
                path_str = str(self.metadata[i].get("path", "")).lower()
                cmd_str = str(self.metadata[i].get("cmd", "")).lower()
                
                match_found = False
                for name in self.csv_filter_names:
                    if name in path_str or name in cmd_str:
                        match_found = True
                        break
                        
                if not match_found:
                    mask[i] = False

        if search_txt:
            for i in np.where(mask)[0]:
                if search_txt not in self.search_corpus[i]:
                    mask[i] = False

        self.visible_mask = mask
        self.apply_visual_state()

    def apply_visual_state(self):
        if not self.visible_mask.any():
            self.scatter.set_data(np.zeros((1, 3), dtype=np.float32), size=0)
            self.scatter.shared_program['a_tw_start'] = np.zeros(1, dtype=np.float32)
            self.scatter.shared_program['a_tw_end'] = np.zeros(1, dtype=np.float32)
            self.scatter_hl.set_data(np.zeros((1, 3), dtype=np.float32), size=0)
            self.scatter_hl.shared_program['a_tw_start'] = np.zeros(1, dtype=np.float32)
            self.scatter_hl.shared_program['a_tw_end'] = np.zeros(1, dtype=np.float32)
            for attr in ("_last_bg_mask", "_last_display_colors", "_last_render_pos", "_last_display_sizes"):
                if hasattr(self, attr):
                    delattr(self, attr)
            return

        display_colors = self.colors.copy()
        display_sizes = self.sizes.copy()
        
        fp_campaign = hasattr(self, "chk_fp_campaign") and self.chk_fp_campaign.isChecked()
        fp_recall = hasattr(self, "chk_fp_recall") and self.chk_fp_recall.isChecked()

        if (fp_campaign or fp_recall) and hasattr(self, "detection_cost"):
            thresh = self.detection_cost.get("thresh_full_recall" if fp_recall else "thresh_full_campaign", 0.0)
            scores = np.array([m.get("anomaly_score", 0.0) for m in self.metadata])
            labels = np.array([m.get("label", 0) for m in self.metadata])
            
            tn_mask = (labels == 0) & (scores < thresh)
            fn_mask = (labels == 1) & (scores < thresh)
            tp_mask = (labels == 1) & (scores >= thresh)
            fp_mask = (labels == 0) & (scores >= thresh)

            fp_count = np.sum(fp_mask)
            
            # Dynamic opacity and size based on FP count to prevent blowing out the screen
            if fp_count > 1000:
                fp_alpha = 0.3
                fp_size = 4.0
            else:
                fp_alpha = 0.8
                fp_size = 7.0

            display_colors[tn_mask] = [0.2, 0.2, 0.2, 0.1]   # Greyscale, very low opacity
            display_colors[fn_mask] = [1.0, 0.2, 0.2, 0.1]   # Undetected: very less opacity
            display_colors[tp_mask] = [1.0, 0.2, 0.2, 0.4]   # Detected: less opacity
            display_colors[fp_mask] = [1.0, 0.7, 0.0, fp_alpha]   # FPs: bright orange
            display_sizes[fp_mask] = fp_size                 # FPs: Noticeable but scaled
            display_sizes[tp_mask] = 5.0                     # TPs: Standard malicious size
            display_sizes[fn_mask] = 2.0                     # FNs: Smaller
            display_sizes[tn_mask] = 2.0                     # TNs: Smaller
            
            # Print FP Node IDs to terminal if there's a manageable number
            fp_count = np.sum(fp_mask)
            if 0 < fp_count <= 20:
                fp_indices = np.where(fp_mask)[0]
                fp_nids = [self.metadata[i].get("node_id") for i in fp_indices]
                print(f"\n[Forensics] Highlighting {fp_count} False Positives. Node IDs: {fp_nids}")
            elif fp_count > 20:
                print(f"\n[Forensics] Highlighting {fp_count:,} False Positives (too many to list individually).")
                
        elif hasattr(self, "chk_heat") and self.chk_heat.isChecked():
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
            display_colors[self.benign_mask, 3] *= 0.55

        match_mask = np.zeros(len(self.metadata), dtype=bool)
        if hasattr(self, "selected_node_id") and self.selected_node_id is not None:
            display_colors[:, 3] *= 0.55
            display_colors[:, 3] = np.maximum(display_colors[:, 3], 0.12)
            match_mask = np.array(
                [m.get("node_id") == self.selected_node_id for m in self.metadata]
            )

            if self.chk_traj.isChecked():
                indices = np.where(match_mask)[0]
                pts_full = [
                    (self.metadata[i].get("tw_idx", 0), render_pos[i]) for i in indices
                ]
                pts_full.sort(key=lambda x: x[0])

                t_val = self.slider_tw.value() / 100.0
                fraction = 0.0
                if t_val >= 0:
                    pts = []
                    for p in pts_full:
                        if p[0] <= t_val:
                            pts.append(p)
                        else:
                            break

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
                        if i == len(line_pos) - 1 and fraction > 0.0:
                            orig_idx = i - 1 + fraction
                        else:
                            orig_idx = i
                        ratio = orig_idx / max(1, len(pts_full) - 1)
                        h = 0.65 - (ratio * 0.65)
                        rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)

                        if len(line_pos) > 1:
                            age_ratio = i / (len(line_pos) - 1)
                        else:
                            age_ratio = 1.0

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

        t_val = self.slider_tw.value() / 100.0 if (hasattr(self, "chk_temporal") and self.chk_temporal.isChecked()) else -1.0
        self.scatter.shared_program['u_time'] = t_val
        self.scatter_hl.shared_program['u_time'] = t_val

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

                if t_val_actual < activation_time:
                    rgb = [1.0, 0.8, 0.0]
                    opacity = 0.30
                elif (
                    t_val_actual >= activation_time
                    and t_val_actual < activation_time + 4.0
                ):
                    rgb = [0.93, 0.26, 0.26]
                    opacity = 0.85
                else:
                    rgb = [0.93, 0.26, 0.26]
                    opacity = 0.65

                if t_val_actual == float("inf"):
                    rgb = [0.93, 0.26, 0.26]
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

        bg_mask = self.visible_mask & (~match_mask)
        
        rebuild = True
        if hasattr(self, "_last_bg_mask") and hasattr(self, "_last_display_colors") and hasattr(self, "_last_render_pos") and hasattr(self, "_last_display_sizes"):
            if (np.array_equal(self._last_bg_mask, bg_mask) and 
                np.array_equal(self._last_display_colors, display_colors) and
                np.array_equal(self._last_display_sizes, display_sizes) and
                np.array_equal(self._last_render_pos, render_pos)):
                rebuild = False
                
        if rebuild:
            self._last_bg_mask = bg_mask.copy()
            self._last_display_colors = display_colors.copy()
            self._last_display_sizes = display_sizes.copy()
            self._last_render_pos = render_pos.copy()
            
            if bg_mask.any():
                self.scatter.set_data(
                    render_pos[bg_mask],
                    edge_width=0,
                    face_color=display_colors[bg_mask],
                    size=display_sizes[bg_mask],
                )
                self.scatter.shared_program['a_tw_start'] = self.tw_start[bg_mask]
                self.scatter.shared_program['a_tw_end'] = self.tw_end[bg_mask]
            else:
                self.scatter.set_data(np.zeros((1, 3), dtype=np.float32), size=0)
                self.scatter.shared_program['a_tw_start'] = np.zeros(1, dtype=np.float32)
                self.scatter.shared_program['a_tw_end'] = np.zeros(1, dtype=np.float32)

            if match_mask.any():
                hl_mask = self.visible_mask & match_mask
                self.scatter_hl.set_data(
                    render_pos[hl_mask],
                    edge_width=0,
                    face_color=[1.0, 1.0, 1.0, 0.82],
                    size=11,
                )
                self.scatter_hl.shared_program['a_tw_start'] = self.tw_start[hl_mask]
                self.scatter_hl.shared_program['a_tw_end'] = self.tw_end[hl_mask]
            else:
                self.scatter_hl.set_data(np.zeros((1, 3), dtype=np.float32), size=0)
                self.scatter_hl.shared_program['a_tw_start'] = np.zeros(1, dtype=np.float32)
                self.scatter_hl.shared_program['a_tw_end'] = np.zeros(1, dtype=np.float32)

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

        self.canvas3d.update()

    def on_mouse_press(self, event):
        if event.button == 1:
            self.press_pos = event.pos

    def on_mouse_release(self, event):
        if event.button != 1 or self.press_pos is None:
            return

        dx = event.pos[0] - self.press_pos[0]
        dy = event.pos[1] - self.press_pos[1]
        if (dx**2 + dy**2) ** 0.5 > 5:
            return

        click_x, click_y = event.pos

        visible_pos = self.pos[self.visible_mask]
        visible_indices = np.where(self.visible_mask)[0]

        if len(visible_pos) == 0:
            return

        tr = self.scatter.get_transform("visual", "document")
        projected = tr.map(visible_pos)

        w = projected[:, 3].reshape(-1, 1)
        w[w == 0] = 1e-5

        pts_2d = projected[:, :2] / w
        dist_sq = (pts_2d[:, 0] - click_x) ** 2 + (pts_2d[:, 1] - click_y) ** 2
        dist_sq[w[:, 0] <= 0] = np.inf

        min_idx_local = np.argmin(dist_sq)
        if dist_sq[min_idx_local] < 100:
            actual_idx = visible_indices[min_idx_local]
            self.show_node(actual_idx)
        else:
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
            text += f"<b>Path:</b> {m['path']}<br>"

        # Neighborhood inspection
        try:
            K = 20
            node_pos = self.pos[idx]
            dists = np.sum((self.pos - node_pos) ** 2, axis=1)
            dists[idx] = np.inf  # exclude self
            nn_indices = np.argpartition(dists, K)[:K]
            nn_indices = nn_indices[np.argsort(dists[nn_indices])]

            n_benign = n_det = n_undet = 0
            type_counts = defaultdict(int)
            path_counts = defaultdict(int)
            nn_scores = []
            for ni in nn_indices:
                nm = self.metadata[ni]
                lbl = nm.get("label", 0)
                det = nm.get("detection_status", 0)
                if lbl == 0:
                    n_benign += 1
                elif det == 1:
                    n_det += 1
                else:
                    n_undet += 1
                type_counts[nm.get("type", "?").lower()] += 1
                p = nm.get("path", "")
                if p:
                    path_counts[p[:50]] += 1
                nn_scores.append(nm.get("anomaly_score", 0) or 0)

            purity = max(n_benign, n_det, n_undet) / K
            my_score = m.get("anomaly_score", 0) or 0
            median_nn = sorted(nn_scores)[K // 2] if nn_scores else 0
            score_gap = my_score - median_nn

            text += "<br><span style='color:#818cf8; font-weight:bold; font-size:12px;'>NEIGHBORHOOD</span><br>"
            text += f"<span style='color:#10b981'>●</span> Benign: {n_benign}<br>"
            text += f"<span style='color:#60a5fa'>●</span> Detected: {n_det}<br>"
            text += f"<span style='color:#ef4444'>●</span> Undetected: {n_undet}<br>"

            types_str = ", ".join(f"{v} {k}" for k, v in sorted(type_counts.items(), key=lambda x: -x[1]))
            text += f"<b>Types:</b> {types_str}<br>"

            top_paths = sorted(path_counts.items(), key=lambda x: -x[1])[:3]
            paths_str = ", ".join(f"{p} ({c})" for p, c in top_paths)
            text += f"<b>Paths:</b> {paths_str}<br>"

            purity_color = "#10b981" if purity > 0.8 else "#fb923c" if purity > 0.5 else "#ef4444"
            text += f"<b>Purity:</b> <span style='color:{purity_color}'>{purity:.0%}</span> &nbsp; "
            gap_color = "#10b981" if abs(score_gap) > 3.0 else "#fb923c" if abs(score_gap) > 1.0 else "#ef4444"
            text += f"<b>Score Gap:</b> <span style='color:{gap_color}'>{score_gap:+.2f}</span>"
        except Exception:
            pass

        self.info_lbl.setText(text)
        self.apply_visual_state()
        
    def get_causal_trace(self, start_nid, max_nodes=10000):
        from collections import deque
        trace_nodes = {start_nid}
        
        # Forward BFS
        forward_q = deque()
        for edge in self.full_adj.get(str(start_nid), []):
            if isinstance(edge, dict) and edge.get("dir") == "out":
                forward_q.append((edge["nb"], edge["t"]))
        
        visited_fwd = set()
        while forward_q and len(trace_nodes) < max_nodes:
            curr_node, curr_time = forward_q.popleft()
            state_key = f"{curr_node}-{curr_time}"
            if state_key in visited_fwd: continue
            visited_fwd.add(state_key)
            trace_nodes.add(int(curr_node))
            
            for edge in self.full_adj.get(str(curr_node), []):
                if isinstance(edge, dict) and edge.get("dir") == "out" and edge.get("t", 0) >= curr_time:
                    forward_q.append((edge["nb"], edge["t"]))
                    
        # Backward BFS
        backward_q = deque()
        for edge in self.full_adj.get(str(start_nid), []):
            if isinstance(edge, dict) and edge.get("dir") == "in":
                backward_q.append((edge["nb"], edge["t"]))
                
        visited_bwd = set()
        while backward_q and len(trace_nodes) < max_nodes:
            curr_node, curr_time = backward_q.popleft()
            state_key = f"{curr_node}-{curr_time}"
            if state_key in visited_bwd: continue
            visited_bwd.add(state_key)
            trace_nodes.add(int(curr_node))
            
            for edge in self.full_adj.get(str(curr_node), []):
                if isinstance(edge, dict) and edge.get("dir") == "in" and edge.get("t", 0) <= curr_time:
                    backward_q.append((edge["nb"], edge["t"]))
                    
        return trace_nodes

    def show_causal_window(self):
        if not self.selected_node_id:
            self.show_status("No node selected for tracing.", timeout=3000)
            return
            
        self.btn_causal_trace.setText("Extracting...")
        self.btn_causal_trace.setEnabled(False)
        QApplication.processEvents()
        
        try:
            trace_ids = self.get_causal_trace(self.selected_node_id)
            
            # Gather node metadata for the traced nodes using fast hash lookup
            trace_metadata = []
            found_ids = set()
            for nid in trace_ids:
                if nid in self.node_tws:
                    for tw, arr_idx in self.node_tws[nid]:
                        trace_metadata.append(self.metadata[arr_idx])
                    found_ids.add(nid)
                    
            # Inject dummy metadata for nodes that exist in full_adj but were filtered from points
            missing_ids = trace_ids - found_ids
            for missing_id in missing_ids:
                trace_metadata.append({
                    "node_id": missing_id,
                    "type": "Filtered/Off-Graph",
                    "path": "Unknown (Excluded from 3D View)",
                    "anomaly_score": 0.0,
                    "label": 0,
                    "tw_idx": 0
                })
                    
            # Sort by time window
            trace_metadata.sort(key=lambda x: x.get("tw_idx", 0))
            
            # Deduplicate node appearances by taking the first appearance
            seen_nodes = set()
            unique_trace = []
            for m in trace_metadata:
                nid = m.get("node_id")
                if nid not in seen_nodes:
                    seen_nodes.add(nid)
                    unique_trace.append(m)
                    
            from .ui_components import CausalTraceWindow
            self.causal_window = CausalTraceWindow(
                self.selected_node_id, 
                unique_trace, 
                self, 
                full_adj=self.full_adj, 
                trace_node_ids=trace_ids
            )
            self.causal_window.show()
        finally:
            self.btn_causal_trace.setText("Extract Causal Subgraph")
            self.btn_causal_trace.setEnabled(True)

    def show_neighbors_window(self):
        if not self.selected_node_id:
            self.show_status("No node selected.", timeout=3000)
            return

        self.btn_neighbors.setText("Loading...")
        self.btn_neighbors.setEnabled(False)
        QApplication.processEvents()

        try:
            edges = self.full_adj.get(str(self.selected_node_id), [])
            
            # Build grouped data (one row per unique neighbor+dir)
            grouped = {}  # (nb, dir) -> { "indices": set, "dir": str }
            for edge in edges:
                if isinstance(edge, dict):
                    nb = int(edge.get("nb"))
                    dir_val = edge.get("dir", "unknown")
                else:
                    nb = int(edge)
                    dir_val = "unknown"
                
                if nb in self.node_tws:
                    key = (nb, dir_val)
                    if key not in grouped:
                        grouped[key] = {"indices": set(), "dir": dir_val}
                    for tw, arr_idx in self.node_tws[nb]:
                        grouped[key]["indices"].add(arr_idx)

            grouped_data = []
            raw_indices = []  # list of (arr_idx, dir_val) for ungrouped view
            
            for (nb, dir_val), info in grouped.items():
                sorted_idx = sorted(info["indices"], key=lambda i: self.metadata[i].get("tw_idx", 0))
                
                # Build grouped row
                base = self.metadata[sorted_idx[0]]
                tw_labels = []
                seen = set()
                for idx in sorted_idx:
                    label = self.metadata[idx].get("tw_label", str(self.metadata[idx].get("tw_idx", "")))
                    if label not in seen:
                        seen.add(label)
                        tw_labels.append(label)
                
                if len(tw_labels) > 3:
                    agg = f"{tw_labels[0]}, {tw_labels[1]} ... (+{len(tw_labels)-2} more)"
                else:
                    agg = ", ".join(tw_labels)

                grouped_data.append({
                    "edge_dir": dir_val,
                    "node_id": base.get("node_id"),
                    "type": base.get("type", ""),
                    "anomaly_score": base.get("anomaly_score", 0.0),
                    "label": base.get("label", 0),
                    "path": base.get("path", ""),
                    "cmd": base.get("cmd", ""),
                    "tw_idx": base.get("tw_idx", 0),
                    "aggregated_tw": agg,
                    "all_tw_labels": tw_labels,
                })
                
                for idx in sorted_idx:
                    raw_indices.append((idx, dir_val))

            grouped_data.sort(key=lambda x: x.get("anomaly_score", 0.0), reverse=True)
            
            from .ui_components import NodeNeighborsWindow
            self.neighbors_window = NodeNeighborsWindow(
                self.selected_node_id,
                grouped_data,
                raw_indices,
                self.metadata,
                self
            )
            self.neighbors_window.show()
        finally:
            self.btn_neighbors.setText("Show Anomalous Edges")
            self.btn_neighbors.setEnabled(True)

    def show_score_distribution(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
        except ImportError:
            self.show_status("Matplotlib is required for this feature.", timeout=3000)
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Anomaly Score Distribution")
        dlg.resize(900, 600)
        layout = QVBoxLayout(dlg)

        # Collect raw scores and group by node type / attack campaign
        raw_scores = []
        labels = []
        node_types = []
        detection_statuses = []
        for m in self.metadata:
            raw_scores.append(m.get("anomaly_score", 0.0) or 0.0)
            labels.append(m.get("label", 0))
            node_types.append((m.get("type") or "file").lower())
            detection_statuses.append(m.get("detection_status", 0))

        raw_scores = np.array(raw_scores, dtype=float)
        labels = np.array(labels, dtype=int)

        # Normalize to [0, 1] (matching the original evaluation plot)
        raw_min, raw_max = float(np.min(raw_scores)), float(np.max(raw_scores))
        span = (raw_max - raw_min) if raw_max > raw_min else 1.0
        norm_scores = (raw_scores - raw_min) / span

        # Colors — matching the original private repo exactly
        benign_type_colors = {
            "subject": "#1b5e20",
            "file":    "#bbbbbb",
            "netflow": "#a5d6a7",
        }
        alpha_val = 0.7

        attack_colors = {
            0: "black",
            1: "red",
            2: "#377eb8",
        }

        # Split benign by type, attack by campaign
        benign_by_type = defaultdict(list)
        attack_scores = {}
        for i in range(len(norm_scores)):
            if labels[i] == 0:
                ntype = node_types[i]
                if "subject" in ntype or "process" in ntype:
                    benign_by_type["subject"].append(norm_scores[i])
                elif "netflow" in ntype or "net" in ntype:
                    benign_by_type["netflow"].append(norm_scores[i])
                else:
                    benign_by_type["file"].append(norm_scores[i])
            else:
                cids = self.metadata[i].get("campaign_ids", [0])
                attack_type = cids[0] if cids else 0
                attack_scores.setdefault(attack_type, []).append(norm_scores[i])

        bins = np.linspace(0, 1, 75)

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

        legend_patches = []

        # Benign histograms (per-type, matching original)
        for ntype in ("subject", "file", "netflow"):
            vals = benign_by_type.get(ntype, [])
            if not vals:
                continue
            color = benign_type_colors[ntype]
            label = f"Benign ({ntype})"
            ax.hist(
                vals, bins=bins, alpha=alpha_val, label=label,
                color=color, edgecolor="black", linewidth=0.5, log=True
            )
            legend_patches.append(
                Patch(facecolor=color, edgecolor="black", alpha=alpha_val, label=label)
            )

        # Attack histograms (split by campaign)
        for attack_type, values in attack_scores.items():
            ax.hist(
                values, bins=bins, alpha=alpha_val, label=f"Attack #{attack_type+1}",
                color=attack_colors.get(attack_type, "black"),
                edgecolor="black", linewidth=0.5, log=True
            )

        for atype in sorted(attack_scores.keys()):
            if atype in attack_colors:
                legend_patches.append(
                    Patch(facecolor=attack_colors[atype], edgecolor="black",
                          alpha=alpha_val, label=f"Attack #{atype+1}")
                )

        # Compute discrimination zones: shade where precision >= 50% AND all attack TWs detected
        n_curve_points = 200
        precision_cut = 0.5
        thresholds_norm = np.linspace(0, 1, n_curve_points)
        thresholds_raw = raw_min + thresholds_norm * span

        # Compute precision and campaign coverage at each threshold (vectorized)
        precision_curve = np.zeros(n_curve_points)
        num_campaigns = self.stats.get("num_campaigns", 0)
        all_campaigns = set(range(num_campaigns)) if num_campaigns > 0 else set()

        # Build campaign_ids array for malicious nodes
        mal_campaign_ids = []
        for m in self.metadata:
            if m.get("label") == 1:
                mal_campaign_ids.append(m.get("campaign_ids", []))

        # Pre-sort for efficient threshold sweep
        mal_scores_sorted = np.sort(raw_scores[labels == 1])
        ben_scores_sorted = np.sort(raw_scores[labels == 0])
        mal_order = np.argsort(raw_scores[labels == 1])
        mal_campaigns_sorted = [mal_campaign_ids[i] for i in mal_order]

        det_curve = np.zeros(n_curve_points)
        for ti, thr in enumerate(thresholds_raw):
            tp = len(mal_scores_sorted) - np.searchsorted(mal_scores_sorted, thr, side='left')
            fp = len(ben_scores_sorted) - np.searchsorted(ben_scores_sorted, thr, side='left')
            precision_curve[ti] = tp / (tp + fp + 1e-12)
            # Count unique campaigns above threshold
            above_idx = np.searchsorted(mal_scores_sorted, thr, side='left')
            detected_camps = set()
            for cids in mal_campaigns_sorted[above_idx:]:
                detected_camps.update(cids)
            det_curve[ti] = len(detected_camps) / max(num_campaigns, 1)

        eps = 1e-12
        mask = (precision_curve >= precision_cut) & (det_curve >= 1.0 - eps)
        if np.any(mask):
            idx = np.where(mask)[0]
            splits = np.where(np.diff(idx) > 1)[0] + 1
            runs = np.split(idx, splits)
            shaded = False
            for run in runs:
                t_start = thresholds_norm[run[0]]
                t_end = thresholds_norm[run[-1]]
                lbl = "Good Zone (P≥50% & All Attacks)" if not shaded else None
                ax.axvspan(t_start, t_end, color="gray", alpha=0.2, label=lbl)
                shaded = True

        # Threshold line
        if hasattr(self, 'detection_cost'):
            cur = self.detection_cost.get('current_threshold', 0)
            norm_thresh = (cur - raw_min) / span
            if 0.0 <= norm_thresh <= 1.0:
                ax.axvline(
                    x=norm_thresh, color="black", linestyle="--", linewidth=1.5,
                    label=f"Threshold: {norm_thresh:.2f}"
                )

        ax.set_xlabel("Node anomaly scores", fontsize=12)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=12)
        ax.legend(handles=legend_patches, loc='upper right', fontsize=9, frameon=True, fancybox=True)
        fig.tight_layout()

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        dlg.exec_()

