from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, QTimer, QAbstractTableModel
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QTableView,
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
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)

class NeighborsTableModel(QAbstractTableModel):
    """
    Fast model for the neighbors table.
    In grouped mode: _data is a list of pre-computed dicts.
    In raw mode: _data is a list of (arr_idx, dir_val) tuples referencing metadata_ref.
    """
    def __init__(self, data, is_grouped, metadata_ref=None):
        super().__init__()
        self._data = data
        self.is_grouped = is_grouped
        self._meta = metadata_ref  # only used in raw mode
        self.headers = ["Dir", "Time Window", "Node ID", "Type", "Score", "Path / Cmd"]

    def rowCount(self, parent=None):
        if parent is not None and parent.isValid():
            return 0  # flat table, no children
        return len(self._data)

    def columnCount(self, parent=None):
        if parent is not None and parent.isValid():
            return 0
        return len(self.headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        
        row = index.row()
        col = index.column()
        
        if row < 0 or row >= len(self._data):
            return None
        if col < 0 or col >= len(self.headers):
            return None

        try:
            if self.is_grouped:
                m = self._data[row]
                dir_val = m.get("edge_dir", "")
            else:
                arr_idx, dir_val = self._data[row]
                m = self._meta[arr_idx]
            
            if role == Qt.DisplayRole:
                if col == 0: return dir_val
                if col == 1: 
                    if self.is_grouped:
                        return m.get("aggregated_tw", "")
                    else:
                        return m.get("tw_label", str(m.get("tw_idx", "")))
                if col == 2: return str(m.get("node_id", ""))
                if col == 3: return m.get("type", "")
                if col == 4: return f"{m.get('anomaly_score', 0.0):.4f}"
                if col == 5: 
                    cmd_val = m.get("cmd", "")
                    return cmd_val if cmd_val and cmd_val != "None" else str(m.get("path", ""))
                    
            elif role == Qt.ForegroundRole:
                if col in (1, 2):
                    if m.get("label") == 1: return QColor(Qt.red)
                    else: return QColor(Qt.green)
                if col == 4:
                    score = m.get("anomaly_score", 0.0)
                    if score > 0.5: return QColor(Qt.red)
                    elif score > 0.1: return QColor(Qt.yellow)
                    
            elif role == Qt.ToolTipRole:
                if col == 1 and self.is_grouped:
                    all_tws = m.get("all_tw_labels", [])
                    if len(all_tws) > 3:
                        return "Active in time windows:\n" + "\n".join(all_tws)
        except Exception:
            pass
                    
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            if 0 <= section < len(self.headers):
                return self.headers[section]
        return None

class NodeNeighborsWindow(QDialog):
    def __init__(self, center_nid, grouped_data, raw_indices, metadata_ref, parent=None):
        super().__init__(parent)
        self.center_nid = center_nid
        self.setWindowTitle(f"Anomalous Edges for Node {center_nid}")
        self.resize(1000, 500)
        self.setStyleSheet("""
            QDialog { background-color: #111115; }
            QLabel { color: #e0e0e0; font-weight: bold; }
            QTableView {
                background-color: #1a1a24;
                color: #e0e0e0;
                gridline-color: #333344;
                border: 1px solid #3f3f4e;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #2a2a35;
                color: #a0a0b0;
                padding: 4px;
                border: 1px solid #333344;
                font-weight: bold;
            }
            QTableView::item:selected {
                background-color: #3b82f6;
                color: white;
            }
            QScrollBar:vertical { background: #1a1a24; width: 10px; margin: 0px; }
            QScrollBar::handle:vertical { background: #3a3a45; min-height: 20px; border-radius: 5px; }
            QPushButton {
                background-color: #2a2a35; color: white; border: none; padding: 8px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #3a3a45; }
        """)

        self._grouped_data = grouped_data
        self._raw_indices = raw_indices
        self._raw_sorted = None  # lazily sorted
        self._metadata_ref = metadata_ref

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: #60a5fa; font-size: 14px;")
        layout.addWidget(self.lbl_info)

        self.chk_group = QCheckBox("Group identical edges across time windows")
        self.chk_group.setChecked(True)
        self.chk_group.setStyleSheet("color: #a0a0b0; font-weight: bold; margin-bottom: 5px;")
        self.chk_group.stateChanged.connect(self._on_toggle)
        layout.addWidget(self.chk_group)

        self.table = QTableView()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)

        layout.addWidget(self.table)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        h_buttons = QHBoxLayout()
        h_buttons.addStretch()
        h_buttons.addWidget(btn_close)
        layout.addLayout(h_buttons)

        # Show grouped view immediately
        self._set_grouped_model()

    def _configure_headers(self):
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        if self.table.model() and self.table.model().columnCount() > 5:
            header.setSectionResizeMode(5, QHeaderView.Stretch)

    def _set_grouped_model(self):
        model = NeighborsTableModel(self._grouped_data, True)
        self.table.setModel(model)
        self._configure_headers()
        self.lbl_info.setText(f"Showing {len(self._grouped_data)} unique edges for Node {self.center_nid}, sorted by anomaly score.")

    def _set_raw_model(self):
        if self._raw_sorted is None:
            self._raw_sorted = sorted(
                self._raw_indices,
                key=lambda x: self._metadata_ref[x[0]].get("anomaly_score", 0.0),
                reverse=True
            )
        model = NeighborsTableModel(self._raw_sorted, False, self._metadata_ref)
        self.table.setModel(model)
        self._configure_headers()
        self.lbl_info.setText(f"Showing {len(self._raw_sorted)} edges (all time windows) for Node {self.center_nid}, sorted by anomaly score.")

    def _on_toggle(self):
        self.chk_group.setEnabled(False)
        self.lbl_info.setText("Switching view...")
        QApplication.processEvents()
        try:
            if self.chk_group.isChecked():
                self._set_grouped_model()
            else:
                self._set_raw_model()
        finally:
            self.chk_group.setEnabled(True)

class CausalTraceWindow(QDialog):
    def __init__(self, start_nid, trace_metadata, parent=None, full_adj=None, trace_node_ids=None):
        super().__init__(parent)
        self.start_nid = start_nid
        self.trace_metadata = trace_metadata
        self.setWindowTitle(f"Causal Subgraph for Node {start_nid}")
        self.resize(1100, 700)
        self.setStyleSheet("""
            QDialog { background-color: #111115; }
            QLabel { color: #e0e0e0; font-weight: bold; }
            QTabWidget::pane { border: 1px solid #3f3f4e; background-color: #111115; }
            QTabBar::tab { background-color: #2a2a35; color: #a0a0b0; padding: 8px 24px; min-width: 160px; min-height: 20px; border: 1px solid #3f3f4e; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; font-weight: bold; }
            QTabBar::tab:selected { background-color: #111115; color: #60a5fa; border-bottom: 2px solid #60a5fa; }
            QTabBar::tab:hover { background-color: #3a3a45; }
            QTableWidget {
                background-color: #1a1a24;
                color: #e0e0e0;
                gridline-color: #333344;
                border: 1px solid #3f3f4e;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #2a2a35;
                color: #a0a0b0;
                padding: 4px;
                border: 1px solid #333344;
                font-weight: bold;
            }
            QTableWidget::item:selected {
                background-color: #3b82f6;
                color: white;
            }
            QScrollBar:vertical { background: #1a1a24; width: 10px; margin: 0px; }
            QScrollBar::handle:vertical { background: #3a3a45; min-height: 20px; border-radius: 5px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QPushButton {
                background-color: #2a2a35; color: white; border: none; padding: 8px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #3a3a45; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        lbl_info = QLabel(f"Extracted {len(trace_metadata)} causally linked nodes for Node {start_nid}")
        lbl_info.setStyleSheet("color: #60a5fa; font-size: 14px;")
        layout.addWidget(lbl_info)

        lbl_desc = QLabel("Nodes are listed in chronological order. <br/>• The <b>Origin node</b> (the one you clicked) is highlighted.<br/>• <b>Forward Impact</b> traces events that happened <i>after</i> the origin.<br/>• <b>Backward Origin</b> traces events that happened <i>before</i> the origin.")
        lbl_desc.setStyleSheet("color: #a0a0b0; font-size: 13px; margin-bottom: 10px; line-height: 1.4;")
        layout.addWidget(lbl_desc)

        # Search bar
        h_search = QHBoxLayout()
        lbl_search = QLabel("Search:")
        lbl_search.setStyleSheet("color: #a0a0b0; font-weight: bold;")
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Enter node ID, type, or path to filter table and highlight graph...")
        self.txt_search.setStyleSheet("background-color: #2a2a35; color: white; border: 1px solid #3f3f4e; padding: 6px; border-radius: 4px;")
        h_search.addWidget(lbl_search)
        h_search.addWidget(self.txt_search)
        layout.addLayout(h_search)
        
        self.txt_search.textChanged.connect(self.on_search)

        # Tabbed interface: Table + Graph
        from PyQt5.QtWidgets import QTabWidget
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # === Tab 1: Table View ===
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 8, 0, 0)

        self.table = QTableWidget(len(trace_metadata), 5)
        self.table.setHorizontalHeaderLabels(["Time Window", "Node ID", "Type", "Score", "Path / Cmd"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        for i, m in enumerate(trace_metadata):
            tw_item = QTableWidgetItem(m.get("tw_label", str(m.get("tw_idx", ""))))
            id_item = QTableWidgetItem(str(m.get("node_id", "")))
            type_item = QTableWidgetItem(m.get("type", ""))

            score = m.get("anomaly_score", 0.0)
            score_item = QTableWidgetItem(f"{score:.4f}")
            if score > 0.5:
                score_item.setForeground(Qt.red)
            elif score > 0.1:
                score_item.setForeground(Qt.yellow)

            path_val = m.get("path", "")
            cmd_val = m.get("cmd", "")
            if cmd_val and cmd_val != "None":
                path_str = cmd_val
            else:
                path_str = path_val

            path_item = QTableWidgetItem(str(path_str))

            if m.get("label") == 1:
                tw_item.setForeground(Qt.red)
                id_item.setForeground(Qt.red)
            else:
                tw_item.setForeground(Qt.green)
                id_item.setForeground(Qt.green)

            is_origin = (str(m.get("node_id", "")) == str(start_nid))
            
            if is_origin:
                id_item.setText(id_item.text() + " (Origin)")
                # Highlight origin row background
                bg_color = QColor("#453b1b") # dark gold/yellow
                tw_item.setBackground(bg_color)
                id_item.setBackground(bg_color)
                type_item.setBackground(bg_color)
                score_item.setBackground(bg_color)
                path_item.setBackground(bg_color)
                
            self.table.setItem(i, 0, tw_item)
            self.table.setItem(i, 1, id_item)
            self.table.setItem(i, 2, type_item)
            self.table.setItem(i, 3, score_item)
            self.table.setItem(i, 4, path_item)

            if is_origin:
                self.table.scrollToItem(id_item, QAbstractItemView.PositionAtCenter)
                
        table_layout.addWidget(self.table)
        tabs.addTab(table_widget, "Chronological Table")

        # === Tab 2: Interactive Matplotlib Graph ===
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)
        graph_layout.setContentsMargins(0, 8, 0, 0)

        if len(trace_metadata) <= 1:
            lbl_err = QLabel("⚠️ Node is ISOLATED.\nNo causal edges (incoming or outgoing) exist in the raw dataset.")
            lbl_err.setStyleSheet("color: #ef4444; font-size: 16px; font-weight: bold; margin-top: 50px;")
            lbl_err.setAlignment(Qt.AlignCenter)
            graph_layout.addWidget(lbl_err)
            graph_layout.addStretch()
        else:
            try:
                import matplotlib
                matplotlib.use("Qt5Agg")
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
                from matplotlib.figure import Figure
                import networkx as nx

                fig = Figure(facecolor="#111115", dpi=100)
                canvas = FigureCanvas(fig)
                toolbar = NavigationToolbar(canvas, graph_widget)
                toolbar.setStyleSheet("QToolBar { background: #2a2a35; border: none; } QToolButton { color: white; }")
                toolbar.pan() # Enable panning by default
                graph_layout.addWidget(toolbar)
                graph_layout.addWidget(canvas)

                ax = fig.add_subplot(111)
                ax.set_facecolor("#111115")
                ax.set_title(f"Causal Subgraph — Node {start_nid}", color="white", fontsize=13, fontweight="bold")
                ax.tick_params(colors="#555")
                for spine in ax.spines.values():
                    spine.set_color("#333")

                self.G = nx.DiGraph()
                self.nid_to_meta = {}
                for m in trace_metadata:
                    nid = m.get("node_id")
                    self.nid_to_meta[nid] = m
                    self.G.add_node(nid)

                if full_adj and trace_node_ids:
                    for nid in trace_node_ids:
                        for edge in full_adj.get(str(nid), []):
                            if isinstance(edge, dict) and edge.get("dir") == "out":
                                nb = int(edge["nb"])
                                if nb in trace_node_ids:
                                    self.G.add_edge(nid, nb)

                # Custom BFS layout
                from collections import deque
                depths = {start_nid: 0}
                queue = deque([(start_nid, 0)])
                visited = {start_nid}
                while queue:
                    curr, d = queue.popleft()
                    for v in self.G.successors(curr):
                        if v not in visited:
                            visited.add(v)
                            depths[v] = d + 1
                            queue.append((v, d + 1))
                    for u in self.G.predecessors(curr):
                        if u not in visited:
                            visited.add(u)
                            depths[u] = d - 1
                            queue.append((u, d - 1))
                
                for n in self.G.nodes:
                    if n not in depths:
                        depths[n] = 0

                # Filter out benign nodes but preserve causal flow via BFS path extraction
                kept_nodes = {n for n in self.G.nodes if n == start_nid or self.nid_to_meta.get(n, {}).get("label", 0) == 1}
                new_G = nx.DiGraph()
                new_G.add_nodes_from(kept_nodes)
                
                for k_node in kept_nodes:
                    queue = deque([k_node])
                    visited = {k_node}
                    while queue:
                        curr = queue.popleft()
                        for neighbor in self.G.successors(curr):
                            if neighbor not in visited:
                                visited.add(neighbor)
                                if neighbor in kept_nodes:
                                    new_G.add_edge(k_node, neighbor)
                                    # Stop exploring this branch once we hit a kept node
                                else:
                                    queue.append(neighbor)
                                    
                self.G = new_G

                from collections import defaultdict
                by_depth = defaultdict(list)
                for n in self.G.nodes:
                    by_depth[depths[n]].append(n)

                pos = {}
                kept_depths = {n: depths[n] for n in self.G.nodes}
                min_d = min(kept_depths.values()) if kept_depths else 0
                max_d = max(kept_depths.values()) if kept_depths else 0
                d_range = max(1, max_d - min_d)

                for d, nodes in by_depth.items():
                    nodes.sort(key=lambda n: self.nid_to_meta.get(n, {}).get("anomaly_score", 0.0), reverse=True)
                    # Normalize X coordinate between -1 and 1 based on depth
                    x = -1.0 + 2.0 * (d - min_d) / d_range
                    for i, n in enumerate(nodes):
                        # Normalize Y coordinate between -1 and 1
                        y = 0 if len(nodes) == 1 else 1.0 - (2.0 * i / (len(nodes) - 1))
                        pos[n] = (x, y)

                # Relax overlapping dense nodes using spring physics, anchoring the origin
                num_nodes = len(self.G.nodes)
                if num_nodes > 1:
                    import math
                    k_val = 3.0 / math.sqrt(max(1, num_nodes))
                    iters = 10 if num_nodes > 500 else 30
                    self.pos = nx.spring_layout(self.G, pos=pos, fixed=[start_nid], iterations=iters, k=k_val, weight=None)
                else:
                    self.pos = pos

                if len(self.G.nodes) > 0:
                    node_colors = []
                    node_sizes = []
                    node_labels = {}
                    
                    show_all_labels = len(self.G.nodes) < 40

                    for n in self.G.nodes:
                        meta = self.nid_to_meta.get(n, {})
                        label_val = meta.get("label", 0)
                        score = meta.get("anomaly_score", 0.0)

                        is_origin = (n == start_nid)
                        is_malicious = (label_val == 1)

                        if is_origin: node_colors.append("#f59e0b")
                        elif is_malicious: node_colors.append("#ef4444")
                        else: node_colors.append("#10b981")

                        node_sizes.append(max(20, min(100, 20 + score * 80)))

                        if show_all_labels or is_origin or is_malicious:
                            path = str(meta.get("path", ""))
                            short = path.split("/")[-1] if "/" in path else str(n)
                            if len(short) > 12: short = short[:10] + ".."
                            node_labels[n] = short

                    nx.draw_networkx_edges(
                        self.G, self.pos, ax=ax, edge_color="#4a4a5b", arrows=True,
                        arrowsize=8, arrowstyle="-|>", connectionstyle="arc3,rad=0.1",
                        width=0.8, alpha=0.5, min_source_margin=6, min_target_margin=6
                    )

                    self.node_collection = nx.draw_networkx_nodes(
                        self.G, self.pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                        edgecolors="#222", linewidths=0.5, alpha=0.9
                    )

                    nx.draw_networkx_labels(
                        self.G, self.pos, ax=ax, labels=node_labels,
                        font_size=6, font_color="white", font_weight="bold"
                    )

                    # Legend
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f59e0b', markersize=10, label='Origin'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ef4444', markersize=10, label='Malicious'),
                    ]
                    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
                              facecolor='#1a1a24', edgecolor='#333', labelcolor='white')

                    # Interactive Tooltips via Motion Notify
                    self.annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                                             bbox=dict(boxstyle="round4,pad=0.5", fc="#2a2a35", ec="#60a5fa", alpha=0.9),
                                             color="white", fontsize=9, zorder=10)
                    self.annot.set_visible(False)
                    
                    def update_annot(ind):
                        node_idx = ind["ind"][0]
                        node_id = list(self.G.nodes)[node_idx]
                        meta = self.nid_to_meta.get(node_id, {})
                        pos_xy = self.pos[node_id]
                        self.annot.xy = pos_xy
                        
                        score = meta.get("anomaly_score", 0.0)
                        text = f"ID: {node_id}\nType: {meta.get('type', 'Unknown')}\nScore: {score:.4f}\nPath: {meta.get('path', 'None')}"
                        self.annot.set_text(text)
                        self.annot.get_bbox_patch().set_alpha(0.9)
                        
                    self.last_hovered_node = None
                    
                    def hover(event):
                        if event.button is not None:
                            return
                            
                        if not hasattr(self, 'node_collection') or not hasattr(self, 'annot'):
                            return
                        vis = self.annot.get_visible()
                        if event.inaxes == ax:
                            cont, ind = self.node_collection.contains(event)
                            if cont:
                                node_idx = ind["ind"][0]
                                node_id = list(self.G.nodes)[node_idx]
                                if self.last_hovered_node != node_id:
                                    self.last_hovered_node = node_id
                                    update_annot(ind)
                                    self.annot.set_visible(True)
                                    self.canvas.draw_idle()
                            else:
                                if vis:
                                    self.last_hovered_node = None
                                    self.annot.set_visible(False)
                                    self.canvas.draw_idle()
                        elif vis:
                            self.last_hovered_node = None
                            self.annot.set_visible(False)
                            self.canvas.draw_idle()

                    def zoom(event):
                        if event.inaxes == ax:
                            base_scale = 1.2
                            if event.button == 'up':
                                scale_factor = 1 / base_scale
                            elif event.button == 'down':
                                scale_factor = base_scale
                            else:
                                scale_factor = 1
                            
                            cur_xlim = ax.get_xlim()
                            cur_ylim = ax.get_ylim()
                            xdata = event.xdata
                            ydata = event.ydata
                            
                            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
                            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
                            
                            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
                            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
                            
                            ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
                            ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
                            self.canvas.draw_idle()

                    self.canvas = canvas
                    self.canvas.mpl_connect("motion_notify_event", hover)
                    self.canvas.mpl_connect("scroll_event", zoom)

                else:
                    # Solo node after filtering — show informational message
                    n_benign = len(trace_metadata) - len(kept_nodes)
                    ax.text(0.5, 0.5, f"Node {start_nid} has no malicious neighbors.\n"
                            f"All {n_benign} causal neighbors are benign.",
                            transform=ax.transAxes, ha='center', va='center',
                            color='#a0a0b0', fontsize=13, fontweight='bold')
                
                self.canvas = canvas
                ax.set_xticks([])
                ax.set_yticks([])
                fig.tight_layout()
                self.canvas.draw()

            except Exception as e:
                import traceback
                err_text = traceback.format_exc()
                print("Graph render failed:\n", err_text)
                lbl_err = QLabel(f"Graph view failed to render.\nError: {e}")
                lbl_err.setStyleSheet("color: #ef4444; font-size: 12px; padding: 20px;")
                graph_layout.addWidget(lbl_err)

        tabs.addTab(graph_widget, "Directed Graph View")

        # Bottom buttons
        h_buttons = QHBoxLayout()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        h_buttons.addStretch()
        h_buttons.addWidget(btn_close)
        layout.addLayout(h_buttons)

    def on_search(self, text):
        query = text.lower()
        
        # 1. Filter table rows
        for row in range(self.table.rowCount()):
            match = False
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item and query in item.text().lower():
                    match = True
                    break
            self.table.setRowHidden(row, not match)
            
        # 2. Highlight graph nodes dynamically
        if hasattr(self, 'node_collection') and hasattr(self, 'G'):
            colors = []
            sizes = []
            for n in self.G.nodes:
                meta = self.nid_to_meta.get(n, {})
                
                match = False
                if query:
                    if query in str(n).lower() or query in meta.get("type", "").lower() or query in str(meta.get("path", "")).lower():
                        match = True
                        
                is_origin = (n == self.start_nid)
                is_malicious = (meta.get("label", 0) == 1)
                
                base_c = "#f59e0b" if is_origin else ("#ef4444" if is_malicious else "#10b981")
                
                if query:
                    if match:
                        colors.append(base_c)
                        sizes.append(max(50, min(200, 50 + meta.get("anomaly_score", 0.0) * 100)))
                    else:
                        colors.append("#2a2a35") # Dimmed out
                        sizes.append(10)
                else:
                    colors.append(base_c)
                    sizes.append(max(20, min(100, 20 + meta.get("anomaly_score", 0.0) * 80)))
                    
            self.node_collection.set_facecolor(colors)
            self.node_collection.set_sizes(sizes)
            self.canvas.draw_idle()
            


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
            border-radius: 3px;
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
    btn_reset_home.clicked.connect(window.reset_selection)
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
    window.info_lbl.setMinimumHeight(120)
    window.info_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
    window.info_lbl.setStyleSheet(
        "font-family: monospace; font-size: 12px; color: white;"
    )
    v_info.addWidget(window.info_lbl)
    
    window.btn_causal_trace = QPushButton("Extract Causal Subgraph")
    window.btn_causal_trace.setStyleSheet("""
        QPushButton {
            background-color: rgba(100, 180, 255, 0.2); color: #60a5fa; border: 1px solid #3b82f6; border-radius: 4px; padding: 6px; font-weight: bold; margin-top: 10px;
        }
        QPushButton:hover { background-color: rgba(100, 180, 255, 0.4); }
    """)
    window.btn_causal_trace.clicked.connect(window.show_causal_window)
    v_info.addWidget(window.btn_causal_trace)
    
    window.btn_neighbors = QPushButton("Show Anomalous Edges")
    window.btn_neighbors.setStyleSheet("""
        QPushButton {
            background-color: rgba(255, 100, 100, 0.2); color: #ef4444; border: 1px solid #ef4444; border-radius: 4px; padding: 6px; font-weight: bold; margin-top: 5px;
        }
        QPushButton:hover { background-color: rgba(255, 100, 100, 0.4); }
    """)
    window.btn_neighbors.clicked.connect(window.show_neighbors_window)
    v_info.addWidget(window.btn_neighbors)
    
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
        "<b>Selection:</b><br>"
        "<span style='color:#ffffff'>● White (larger)</span>: Selected node<br><br>"
        "<b>FP Overlay:</b><br>"
        "<span style='color:#fb923c'>■ Orange</span>: False Positive at threshold<br><br>"
        "<b>Attack Graph:</b><br>"
        "<span style='color:#eab308'>■ Yellow</span>: Edge not yet activated<br>"
        "<span style='color:#ef4444'>■ Red</span>: Edge activated"
    )
    window.lbl_leg_edges.setWordWrap(True)
    window.lbl_leg_edges.setStyleSheet("color: #e0e0e8; font-size: 11px;")
    v_leg.addWidget(window.lbl_leg_edges)

    v_leg_main.addWidget(window.wgt_legend_items)
    window.overlay_tr.adjustSize()
