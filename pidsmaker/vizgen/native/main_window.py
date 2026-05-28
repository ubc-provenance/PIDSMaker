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

    def on_epoch_slider_moved(self, idx):
        ep_num, _ = self.available_epochs[idx]
        self.lbl_epoch.setText(f"Epoch: {ep_num}")

    def on_epoch_scrub(self):
        if getattr(self, "_is_loading", False):
            print("Already loading another epoch. Ignoring scrub request.")
            return

        self._is_loading = True
        idx = self.slider_epoch.value()
        ep_num, ef_path = self.available_epochs[idx]
        self.lbl_epoch.setText(f"Epoch: {ep_num} (Loading...)")
        basename = os.path.basename(ef_path)
        if len(basename) > 30:
            basename = basename[:12] + "..." + basename[-15:]
        self.show_status(f"Scrubbing to Epoch {ep_num} ({basename})...")
        if hasattr(self, "slider_epoch"):
            self.slider_epoch.setEnabled(False)

        self.current_path = ef_path
        print(f"Scrubbing to Epoch {ep_num}...")
        
        self.loader_thread = DataLoaderThread(ef_path, ep_num, self.current_hop)
        self.loader_thread.dataLoaded.connect(self.on_epoch_data_loaded)
        self.loader_thread.start()

    def on_epoch_data_loaded(self, data, current_hop, ep_num):
        self._is_loading = False
        if hasattr(self, "slider_epoch"):
            self.slider_epoch.setEnabled(True)
            
        if not data:
            self.lbl_epoch.setText(f"Epoch: {ep_num} (Error)")
            self.show_status(f"Failed to load Epoch {ep_num}.", timeout=3000)
            print(f"Error: Failed to load Epoch {ep_num}.")
            return
            
        pos_hops, colors, sizes, metadata, stats, attack_edges = data
        self.pos_hops = pos_hops
        self.colors = colors
        self.sizes = sizes
        self.metadata = metadata
        self.stats = stats
        self.attack_edges = attack_edges

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
        self.lbl_epoch.setText(f"Epoch: {ep_num}")
        self.update_scatter()

    def on_hop_scrub(self, val):
        self.current_hop = val
        self.lbl_hops.setText(f"Hops ({val}):")
        self.pos = self.pos_hops[val]
        self.update_spatial_bounds()
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

    def update_spatial_bounds(self):
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
            
        if hasattr(self, "camera"):
            self.reset_camera()

    def update_camera_center(self):
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
        self.det_mask = np.array([m.get("label", 0) != 0 and m.get("detection_status", 0) in (0, 1) for m in self.metadata], dtype=bool)
        self.undet_mask = np.array([m.get("label", 0) != 0 and m.get("detection_status", 0) == 2 for m in self.metadata], dtype=bool)
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
            self.scatter.shared_program['a_tw_start'] = np.zeros(1, dtype=np.float32)
            self.scatter.shared_program['a_tw_end'] = np.zeros(1, dtype=np.float32)
            self.scatter_hl.set_data(np.zeros((1, 3), dtype=np.float32), size=0)
            self.scatter_hl.shared_program['a_tw_start'] = np.zeros(1, dtype=np.float32)
            self.scatter_hl.shared_program['a_tw_end'] = np.zeros(1, dtype=np.float32)
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
            display_colors[self.benign_mask, 3] *= 0.15

        match_mask = np.zeros(len(self.metadata), dtype=bool)
        if hasattr(self, "selected_node_id") and self.selected_node_id is not None:
            display_colors[:, 3] *= 0.4
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
        if hasattr(self, "_last_bg_mask") and hasattr(self, "_last_display_colors") and hasattr(self, "_last_render_pos"):
            if (np.array_equal(self._last_bg_mask, bg_mask) and 
                np.array_equal(self._last_display_colors, display_colors) and
                np.array_equal(self._last_render_pos, render_pos)):
                rebuild = False
                
        if rebuild:
            self._last_bg_mask = bg_mask.copy()
            self._last_display_colors = display_colors.copy()
            self._last_render_pos = render_pos.copy()
            
            if bg_mask.any():
                self.scatter.set_data(
                    render_pos[bg_mask],
                    edge_width=0,
                    face_color=display_colors[bg_mask],
                    size=self.sizes[bg_mask],
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
                    face_color=[1.0, 1.0, 1.0, 1.0],
                    size=12,
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
            text += f"<b>Path:</b> {m['path']}"

        self.info_lbl.setText(text)
        self.apply_visual_state()
