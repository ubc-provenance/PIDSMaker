"""
Temporal random walk sampler for provenance graphs.

Generates walks on PIDSMaker's NetworkX MultiDiGraphs,
returning node IDs and edge types suitable for tokenization.
"""

import bisect
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np


class ProvenanceWalkSampler:
    """Temporal random walk sampler on provenance graphs (NetworkX MultiDiGraph).

    Each walk produces:
        - walk_nodes: [node_id_0, node_id_1, ..., node_id_k]
        - walk_edge_types: [edge_type_01, edge_type_12, ..., edge_type_(k-1)k]
        - walk_times: [t_01, t_12, ..., t_(k-1)k]

    Walks are temporally constrained: timestamps must be monotonically increasing
    (forward) or decreasing (backward).
    """

    def __init__(
        self,
        graph,
        walk_length: int = 30,
        num_walks: int = 10,
        time_weight: str = "uniform",
        half_life: float = 1.0,
        random_walk_start: bool = False,
        diversity_weight: float = 0.0,
        node2vec_p: float = 1.0,
        node2vec_q: float = 1.0,
    ):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.time_weight = time_weight
        self.half_life = half_life
        self.random_walk_start = random_walk_start
        self.diversity_weight = diversity_weight
        self.node2vec_p = node2vec_p
        self.node2vec_q = node2vec_q

        self.forward_adj = self._build_forward_adj()
        self.backward_adj = self._build_backward_adj()

        # Pre-compute neighbor sets for Node2Vec bias (O(1) lookup)
        self._neighbor_sets: Optional[Dict] = None
        if node2vec_p != 1.0 or node2vec_q != 1.0:
            self._neighbor_sets = self._build_neighbor_sets()

        # Pre-sort adjacency lists by timestamp for O(log n) temporal filtering
        self.forward_times = self._sort_adj_and_extract_times(self.forward_adj)
        self.backward_times = self._sort_adj_and_extract_times(self.backward_adj)

        self.nodes = list(graph.nodes())

        # Node labels for diverse neighbor deduplication
        self.node_labels = {}
        for nid, attrs in graph.nodes(data=True):
            ntype = attrs.get("node_type", "unknown")
            label = attrs.get("label", "unknown")
            self.node_labels[nid] = (ntype, label)

    def save_state(self) -> dict:
        """Serialize sampler state for disk caching (excludes the NetworkX graph).

        Returns a dict suitable for torch.save / pickle.
        """
        return {
            "forward_adj": self.forward_adj,
            "backward_adj": self.backward_adj,
            "forward_times": self.forward_times,
            "backward_times": self.backward_times,
            "nodes": self.nodes,
            "node_labels": self.node_labels,
            "walk_length": self.walk_length,
            "num_walks": self.num_walks,
            "time_weight": self.time_weight,
            "half_life": self.half_life,
            "random_walk_start": self.random_walk_start,
            "diversity_weight": self.diversity_weight,
            "node2vec_p": self.node2vec_p,
            "node2vec_q": self.node2vec_q,
        }

    @classmethod
    def from_state(cls, state: dict) -> "ProvenanceWalkSampler":
        """Reconstruct a sampler from a saved state dict (no graph needed)."""
        sampler = object.__new__(cls)
        sampler.forward_adj = state["forward_adj"]
        sampler.backward_adj = state["backward_adj"]
        sampler.forward_times = state["forward_times"]
        sampler.backward_times = state["backward_times"]
        sampler.nodes = state["nodes"]
        sampler.node_labels = state["node_labels"]
        sampler.walk_length = state["walk_length"]
        sampler.num_walks = state["num_walks"]
        sampler.time_weight = state["time_weight"]
        sampler.half_life = state["half_life"]
        sampler.random_walk_start = state["random_walk_start"]
        sampler.diversity_weight = state["diversity_weight"]
        sampler.node2vec_p = state["node2vec_p"]
        sampler.node2vec_q = state["node2vec_q"]
        sampler.graph = None
        sampler._neighbor_sets = None
        if sampler.node2vec_p != 1.0 or sampler.node2vec_q != 1.0:
            neighbors = {}
            for src, edges in sampler.forward_adj.items():
                for dst, _, _ in edges:
                    neighbors.setdefault(src, set()).add(dst)
                    neighbors.setdefault(dst, set()).add(src)
            sampler._neighbor_sets = neighbors
        return sampler

    def _build_forward_adj(self) -> Dict:
        """Build forward adjacency: src -> [(dst, edge_type, time), ...]"""
        adj = {}
        for src, dst, _, attrs in self.graph.edges(data=True, keys=True):
            if src not in adj:
                adj[src] = []
            adj[src].append((dst, attrs.get("label", ""), attrs.get("time", 0)))
        return adj

    def _build_backward_adj(self) -> Dict:
        """Build backward adjacency: dst -> [(src, edge_type, time), ...]"""
        adj = {}
        for src, dst, _, attrs in self.graph.edges(data=True, keys=True):
            if dst not in adj:
                adj[dst] = []
            adj[dst].append((src, attrs.get("label", ""), attrs.get("time", 0)))
        return adj

    def _sort_adj_and_extract_times(self, adj: Dict) -> Dict:
        """Sort adjacency lists by timestamp and extract time arrays for binary search."""
        times = {}
        for node, edges in adj.items():
            edges.sort(key=lambda x: x[2])
            times[node] = [e[2] for e in edges]
        return times

    def _build_neighbor_sets(self) -> Dict:
        """Build undirected neighbor sets for Node2Vec bias computation."""
        neighbors = {}
        for src, dst, _, _ in self.graph.edges(data=True, keys=True):
            neighbors.setdefault(src, set()).add(dst)
            neighbors.setdefault(dst, set()).add(src)
        return neighbors

    def _pick_neighbor(
        self, options: list, times_sorted: list, last_time: float, forward: bool,
        walk_edge_types: Optional[List[str]] = None,
        prev_node: Optional[str] = None,
    ) -> Optional[tuple]:
        """Pick a neighbor respecting temporal constraints using binary search.

        Uses pre-sorted adjacency lists with bisect for O(log n) temporal
        filtering instead of O(n) list comprehension.

        When diversity_weight > 0 and walk_edge_types is provided, candidates
        whose edge type is underrepresented in the walk so far receive higher
        weight.  The diversity bonus for each candidate is:
            1 / (1 + count_of_this_edge_type_in_walk)
        scaled by diversity_weight before combining with the base (time) weight.

        When Node2Vec bias is active (p != 1 or q != 1) and prev_node is given,
        candidates are weighted by their structural relationship to prev_node:
            - candidate == prev_node (return): 1/p
            - candidate is a neighbor of prev_node (BFS): 1
            - otherwise (DFS): 1/q
        """
        # Binary search on pre-sorted time array
        if forward:
            start = bisect.bisect_right(times_sorted, last_time)
            end = len(options)
        else:
            start = 0
            end = bisect.bisect_left(times_sorted, last_time)

        if start >= end:
            return None

        # Fast path: uniform weights + no diversity bias + no Node2Vec bias
        has_node2vec = self._neighbor_sets is not None and prev_node is not None
        if self.time_weight != "exponential" and self.diversity_weight <= 0 and not has_node2vec:
            idx = random.randint(start, end - 1)
            return options[idx]

        # Slow path: need valid slice for weighted sampling
        valid = options[start:end]

        # Base weights from time weighting strategy
        if self.time_weight == "exponential":
            times = np.array([t for _, _, t in valid])
            base_w = np.exp(np.abs(last_time - times) / (-self.half_life))
            base_w = np.nan_to_num(base_w, nan=0.0)
            if base_w.sum() == 0:
                base_w = np.ones(len(valid))
        else:
            base_w = np.ones(len(valid))

        # Apply edge-type diversity bias
        if self.diversity_weight > 0 and walk_edge_types:
            edge_counts = Counter(walk_edge_types)
            diversity_w = np.array([
                1.0 / (1 + edge_counts.get(etype, 0))
                for _, etype, _ in valid
            ])
            # Blend: base * (1 + diversity_weight * diversity_bonus)
            weights = base_w * (1.0 + self.diversity_weight * diversity_w)
        else:
            weights = base_w

        # Apply Node2Vec 2nd-order bias (p/q)
        if has_node2vec:
            prev_neighbors = self._neighbor_sets.get(prev_node, set())
            inv_p = 1.0 / self.node2vec_p
            inv_q = 1.0 / self.node2vec_q
            n2v_w = np.array([
                inv_p if dst == prev_node
                else 1.0 if dst in prev_neighbors
                else inv_q
                for dst, _, _ in valid
            ])
            weights = weights * n2v_w

        total = weights.sum()
        if total == 0:
            idx = random.randint(0, len(valid) - 1)
            return valid[idx]
        weights /= total
        idx = np.random.choice(len(valid), p=weights)
        return valid[idx]

    def _extend_forward(self, walk_nodes, walk_edge_types, walk_times, start_node):
        """Extend walk forward in time from start_node."""
        if start_node not in self.forward_adj:
            return
        options = self.forward_adj[start_node]

        if self.random_walk_start and options:
            dst, etype, t = random.choice(options)
            walk_nodes.append(dst)
            walk_edge_types.append(etype)
            walk_times.append(t)
            last_time = t
        else:
            last_time = options[0][2] if options else -np.inf  # sorted → min is first

        while len(walk_nodes) < self.walk_length:
            current = walk_nodes[-1]
            if current not in self.forward_adj:
                break
            prev = walk_nodes[-2] if len(walk_nodes) >= 2 else None
            picked = self._pick_neighbor(
                self.forward_adj[current], self.forward_times[current],
                last_time, forward=True,
                walk_edge_types=walk_edge_types,
                prev_node=prev,
            )
            if picked is None:
                break
            dst, etype, t = picked
            walk_nodes.append(dst)
            walk_edge_types.append(etype)
            walk_times.append(t)
            last_time = t

    def _extend_backward(self, walk_nodes, walk_edge_types, walk_times, start_node):
        """Extend walk backward in time from start_node."""
        if start_node not in self.backward_adj:
            return
        options = self.backward_adj[start_node]

        # Build backward portion in reverse (append), then prepend once.
        rev_nodes = []
        rev_edges = []
        rev_times = []

        if self.random_walk_start and options:
            src, etype, t = random.choice(options)
            rev_nodes.append(src)
            rev_edges.append(etype)
            rev_times.append(t)
            last_time = t
        else:
            last_time = options[-1][2] if options else np.inf  # sorted → max is last

        while len(walk_nodes) + len(rev_nodes) < self.walk_length:
            current = rev_nodes[-1] if rev_nodes else start_node
            if current not in self.backward_adj:
                break
            prev = rev_nodes[-2] if len(rev_nodes) >= 2 else (start_node if rev_nodes else None)
            picked = self._pick_neighbor(
                self.backward_adj[current], self.backward_times[current],
                last_time, forward=False,
                walk_edge_types=rev_edges,
                prev_node=prev,
            )
            if picked is None:
                break
            src, etype, t = picked
            rev_nodes.append(src)
            rev_edges.append(etype)
            rev_times.append(t)
            last_time = t

        if rev_nodes:
            rev_nodes.reverse()
            rev_edges.reverse()
            rev_times.reverse()
            walk_nodes[:0] = rev_nodes
            walk_edge_types[:0] = rev_edges
            walk_times[:0] = rev_times

    def _single_walk(self, start_node: str) -> Tuple[List[str], List[str], List[float], int]:
        """Generate a single temporal random walk from start_node.

        Randomly chooses whether to start forward or backward (50/50),
        then extends in the other direction if the walk is too short.
        This ensures balanced coverage of both causal history (backward)
        and consequences (forward) during pretraining.

        When random_walk_start is True, the first hop is a randomly selected
        edge rather than always starting from the earliest/latest timestamp.

        Returns:
            walk_nodes: list of node IDs in the walk
            walk_edge_types: list of edge types between consecutive nodes
            walk_times: list of timestamps for each edge
            entity_pos: index of start_node in walk_nodes after extension
        """
        walk_nodes = [start_node]
        walk_edge_types = []
        walk_times = []

        # Randomly pick primary direction
        if random.random() < 0.5:
            self._extend_forward(walk_nodes, walk_edge_types, walk_times, start_node)
            n_before_bwd = len(walk_nodes)
            if len(walk_nodes) < self.walk_length:
                self._extend_backward(walk_nodes, walk_edge_types, walk_times, start_node)
            # Backward prepends, so entity moved by (new_len - n_before_bwd) positions
            entity_pos = len(walk_nodes) - n_before_bwd
        else:
            n_before_bwd = len(walk_nodes)  # always 1
            self._extend_backward(walk_nodes, walk_edge_types, walk_times, start_node)
            # Compute entity_pos AFTER backward (prepend) but BEFORE forward (append)
            entity_pos = len(walk_nodes) - n_before_bwd
            if len(walk_nodes) < self.walk_length:
                self._extend_forward(walk_nodes, walk_edge_types, walk_times, start_node)
        return walk_nodes, walk_edge_types, walk_times, entity_pos

    def sample_walks(
        self,
        batch_nodes: Optional[List[str]] = None,
    ) -> List[Tuple[List[str], List[str]]]:
        """Sample walks from a batch of start nodes.

        Args:
            batch_nodes: list of start node IDs. If None, sample from all nodes.

        Returns:
            List of (walk_nodes, walk_edge_types) tuples.
        """
        if batch_nodes is None:
            batch_nodes = self.nodes

        walks = []
        for node in batch_nodes:
            for _ in range(self.num_walks):
                walk_nodes, walk_edge_types, _, _ = self._single_walk(node)
                if len(walk_nodes) > 1:
                    walks.append((walk_nodes, walk_edge_types))

        return walks

    def sample_bidirectional_walk(
        self,
        target_node: str,
        max_ts: Optional[float] = None,
        walk_length: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """Sample a bidirectional walk centered on target_node.

        Splits the walk budget evenly between two types of PAST context:
          - Incoming-backward: follow incoming edges backward in time from
            target_node (t < max_ts), capturing who/what caused target_node
            to reach its current state (provenance).
          - Outgoing-past: follow outgoing edges backward in time from
            target_node (t < max_ts), capturing what target_node DID before
            the scored edge (its own generated activity).

        Both halves are strictly before max_ts — no future snooping.

        The returned sequence is:
            [...incoming_ancestors → target_node → ...past_outgoing_neighbors]

        This gives the model both provenance (who caused this?) and effect
        (what did this already trigger in the past?). For example, a malicious
        nginx that has already written to /tmp or connected to a C2 server
        before the scored edge will look different from a benign nginx that
        only forwarded to PHP-FPM and wrote access logs.

        Args:
            target_node: the node to centre the walk on (typically the src node
                of the scored edge).
            max_ts: timestamp of the scored edge. Both walk halves are
                constrained to edges with t < max_ts.
            walk_length: hop budget per direction (backward and outgoing-past each get walk_length hops).
        """
        wl = walk_length or self.walk_length

        # ── Backward portion ─────────────────────────────────────────────
        bwd_nodes_rev = [target_node]
        bwd_edge_types_rev = []
        last_time = max_ts if max_ts is not None else np.inf

        while len(bwd_nodes_rev) <= wl:
            current = bwd_nodes_rev[-1]
            if current not in self.backward_adj:
                break
            picked = self._pick_neighbor(
                self.backward_adj[current], self.backward_times[current],
                last_time, forward=False,
            )
            if picked is None:
                break
            src, etype, t = picked
            bwd_nodes_rev.append(src)
            bwd_edge_types_rev.append(etype)
            last_time = t

        bwd_nodes_rev.reverse()
        bwd_edge_types_rev.reverse()
        # bwd_nodes_rev[-1] == target_node

        # ── Outgoing-past portion ────────────────────────────────────────
        # Follow OUTGOING edges from target_node in descending time order
        # (t < max_ts), capturing what target_node DID before the scored edge.
        # This is purely within the past — no future snooping.
        fwd_nodes = []
        fwd_edge_types = []
        last_time = max_ts if max_ts is not None else np.inf
        current = target_node

        while len(fwd_nodes) < wl:
            if current not in self.forward_adj:
                break
            picked = self._pick_neighbor(
                self.forward_adj[current], self.forward_times[current],
                last_time, forward=False,
            )
            if picked is None:
                break
            dst, etype, t = picked
            fwd_nodes.append(dst)
            fwd_edge_types.append(etype)
            last_time = t
            current = dst

        # Combine: [...incoming_ancestors → target_node → ...past_outgoing]
        return bwd_nodes_rev + fwd_nodes, bwd_edge_types_rev + fwd_edge_types

    def sample_context_walk(
        self,
        target_node: str,
        max_ts: Optional[float] = None,
        walk_length: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """Sample a backward walk ending at target_node (for fine-tuning context).

        The walk goes backward in time from target_node, then is reversed
        to produce a chronologically ordered sequence ending at target_node.

        Args:
            target_node: the node to end the walk at
            max_ts: maximum timestamp (only use edges before this time)
            walk_length: override default walk length

        Returns:
            (walk_nodes, walk_edge_types) ending at target_node
        """
        wl = walk_length or self.walk_length
        # Build walk backward, then reverse once at the end (avoids O(n²) inserts)
        walk_nodes_rev = [target_node]
        walk_edge_types_rev = []
        last_time = max_ts if max_ts is not None else np.inf

        while len(walk_nodes_rev) < wl:
            current = walk_nodes_rev[-1]
            if current not in self.backward_adj:
                break

            picked = self._pick_neighbor(
                self.backward_adj[current], self.backward_times[current],
                last_time, forward=False,
            )
            if picked is None:
                break

            src, etype, t = picked
            walk_nodes_rev.append(src)
            walk_edge_types_rev.append(etype)
            last_time = t

        walk_nodes_rev.reverse()
        walk_edge_types_rev.reverse()
        return walk_nodes_rev, walk_edge_types_rev

    # ── T5 directional walk sampling ──────────────────────────────────

    def _get_node_times(self, node: str) -> List[float]:
        """Get sorted unique edge timestamps for a node across both directions."""
        times = set()
        if node in self.forward_adj:
            for _, _, t in self.forward_adj[node]:
                times.add(t)
        if node in self.backward_adj:
            for _, _, t in self.backward_adj[node]:
                times.add(t)
        return sorted(times)

    def _pick_neighbor_with_etype(
        self, options: list, times_sorted: list, last_time: float,
        forward: bool, preferred_etype: str,
    ) -> Optional[tuple]:
        """Pick a temporally valid neighbor with a preferred edge type.

        Returns None if no neighbor with the preferred type satisfies
        temporal constraints. The caller should fall back to _pick_neighbor.
        """
        if forward:
            start = bisect.bisect_right(times_sorted, last_time)
            end = len(options)
        else:
            start = 0
            end = bisect.bisect_left(times_sorted, last_time)
        if start >= end:
            return None
        candidates = [options[i] for i in range(start, end) if options[i][1] == preferred_etype]
        if candidates:
            return random.choice(candidates)
        return None

    def _directed_walk(
        self, start_node: str, entry_time: float, max_len: int,
        forward: bool, preferred_etype: Optional[str] = None,
    ) -> Optional[Tuple[List[str], List[str]]]:
        """Sample a single directional walk from start_node at entry_time.

        Args:
            start_node: entity node to walk from.
            entry_time: temporal entry point (first hop must be after/before).
            max_len: maximum number of hops.
            forward: True for forward walk, False for backward.
            preferred_etype: if set, try this edge type for the first hop.

        Returns:
            (context_edge_types, context_nodes) or None if no valid walk.
        """
        adj = self.forward_adj if forward else self.backward_adj
        adj_times = self.forward_times if forward else self.backward_times
        if start_node not in adj:
            return None

        edge_types: List[str] = []
        context_nodes: List[str] = []
        current = start_node
        last_time = entry_time

        while len(context_nodes) < max_len:
            if current not in adj:
                break

            picked = None
            # First hop: try preferred edge type for diversity
            if not context_nodes and preferred_etype is not None:
                picked = self._pick_neighbor_with_etype(
                    adj[current], adj_times[current], last_time, forward, preferred_etype
                )
            if picked is None:
                picked = self._pick_neighbor(
                    adj[current], adj_times[current],
                    last_time, forward=forward,
                    walk_edge_types=edge_types,
                )
            if picked is None:
                break

            neighbor, etype, t = picked
            context_nodes.append(neighbor)
            edge_types.append(etype)
            last_time = t
            current = neighbor

        if not context_nodes:
            return None
        return edge_types, context_nodes

    def sample_directional_walks(
        self, start_node: str, num_walks: int,
    ) -> List[Tuple[str, List[str], List[str]]]:
        """Sample diverse forward and backward walks for T5 pretraining.

        For each node, produces separate forward and backward walks with:
        - Random temporal entry points for different temporal views
        - Round-robin first-hop edge types for edge-type diversity
        - Random walk lengths between 1 and walk_length
        - Local deduplication

        Args:
            start_node: entity node to build walks from.
            num_walks: number of walk attempts per direction.

        Returns:
            List of (direction, context_edge_types, context_nodes).
            direction is 'forward' or 'backward'.
        """
        all_times = self._get_node_times(start_node)
        if not all_times:
            return []

        # Available first-hop edge types per direction (for diversity round-robin)
        fwd_etypes = sorted(set(e[1] for e in self.forward_adj.get(start_node, [])))
        bwd_etypes = sorted(set(e[1] for e in self.backward_adj.get(start_node, [])))

        results: List[Tuple[str, List[str], List[str]]] = []
        seen: set = set()

        for i in range(num_walks):
            max_len = random.randint(1, self.walk_length)
            entry_time = random.choice(all_times)

            # Forward walk
            if start_node in self.forward_adj:
                preferred = fwd_etypes[i % len(fwd_etypes)] if fwd_etypes else None
                result = self._directed_walk(
                    start_node, entry_time, max_len, forward=True,
                    preferred_etype=preferred,
                )
                if result is not None:
                    sig = ('f', tuple(result[0]), tuple(result[1]))
                    if sig not in seen:
                        seen.add(sig)
                        results.append(('forward', result[0], result[1]))

            # Backward walk
            if start_node in self.backward_adj:
                preferred = bwd_etypes[i % len(bwd_etypes)] if bwd_etypes else None
                result = self._directed_walk(
                    start_node, entry_time, max_len, forward=False,
                    preferred_etype=preferred,
                )
                if result is not None:
                    sig = ('b', tuple(result[0]), tuple(result[1]))
                    if sig not in seen:
                        seen.add(sig)
                        results.append(('backward', result[0], result[1]))

        return results

    # ── Temporal neighborhood sampling (for GNN-based / T5 pretraining) ─

    def _get_neighbors_at_time(
        self, node, ref_time: float, max_neighbors: int, forward: bool = False,
    ) -> List[Tuple]:
        """Get up to `max_neighbors` neighbors near `ref_time`.

        For backward (default): returns the last N edges *before* ref_time,
        sorted most-recent-first.
        For forward: returns the first N edges *after* ref_time, sorted
        earliest-first.
        """
        adj = self.forward_adj if forward else self.backward_adj
        adj_times = self.forward_times if forward else self.backward_times

        if node not in adj:
            return []

        edges = adj[node]
        times = adj_times[node]

        if forward:
            # First edge at or after ref_time
            idx = bisect.bisect_left(times, ref_time)
            if idx >= len(times):
                return []
            end = min(len(edges), idx + max_neighbors)
            neighbors = edges[idx:end]  # earliest first
        else:
            # Last edge strictly before ref_time
            idx = bisect.bisect_left(times, ref_time)
            if idx == 0:
                return []
            start = max(0, idx - max_neighbors)
            neighbors = edges[start:idx]
            neighbors.reverse()  # most recent first

        return neighbors

    def _get_backward_neighbors_at_time(
        self, node, ref_time: float, max_neighbors: int,
    ) -> List[Tuple]:
        """Get the last `max_neighbors` backward neighbors before `ref_time`.

        Returns list of (neighbor_node, edge_type, timestamp) sorted by time descending
        (most recent first).
        """
        return self._get_neighbors_at_time(node, ref_time, max_neighbors, forward=False)

    def sample_temporal_neighborhood(
        self, node, n_min: int = 5, n_max: int = 20,
        forward: bool = False,
    ) -> Optional[Tuple[str, float, List[Tuple]]]:
        """Sample a 1-hop temporal neighborhood for a node.

        Picks a random timestamp where the node had activity, then extracts
        up to N neighbors (N uniformly random in [n_min, n_max]).

        Args:
            node: target node ID
            n_min: minimum neighborhood size
            n_max: maximum neighborhood size
            forward: if True, sample forward neighbors (after ref_time);
                     if False (default), sample backward neighbors (before ref_time).

        Returns:
            (node, ref_time, neighbors) where neighbors is a list of
            (neighbor_node, edge_type, timestamp) tuples, or None if
            the node has insufficient neighbors in the requested direction.
        """
        all_times = self._get_node_times(node)
        if not all_times:
            return None

        ref_time = random.choice(all_times)
        n_neighbors = random.randint(n_min, n_max)
        neighbors = self._get_neighbors_at_time(node, ref_time, n_neighbors, forward=forward)

        if len(neighbors) < 2:
            return None

        return (node, ref_time, neighbors)

    # ── N-hop neighborhood extraction (for GNN distillation) ──────────

    def sample_nhop_neighborhood(
        self,
        node: str,
        n_neighbors_min: int = 5,
        n_neighbors_max: int = 20,
        ref_time: Optional[float] = None,
        diverse: bool = False,
    ) -> Optional[Tuple[str, List[str], List[Tuple[str, str, str]]]]:
        """Sample a 1-hop temporal neighborhood as a subgraph.

        Samples a random number of direct neighbors (both backward and
        forward) around a temporal reference point.  The neighbor count
        is drawn uniformly from [n_neighbors_min, n_neighbors_max] to
        prevent the GNN from overfitting to a fixed neighborhood size.

        Args:
            node: center node ID.
            n_neighbors_min: minimum neighbors to sample.
            n_neighbors_max: maximum neighbors to sample.
            ref_time: temporal reference point. If None, picked randomly
                from the node's activity timestamps.
            diverse: if True, dedup neighbors by (edge_type, neighbor_type,
                neighbor_label) and prioritize edge type diversity.

        Returns:
            (center_node, unique_nodes, edges) where edges is a list of
            (src, dst, edge_type) tuples, or None if the neighborhood is
            too sparse (fewer than 1 edge).
        """
        if ref_time is None:
            all_times = self._get_node_times(node)
            if not all_times:
                return None
            ref_time = random.choice(all_times)

        n_neighbors = random.randint(n_neighbors_min, n_neighbors_max)
        half_budget = max(1, n_neighbors // 2)

        if diverse:
            bwd_selected = self._diverse_neighbors(
                node, ref_time, half_budget, forward=False,
            )
            fwd_selected = self._diverse_neighbors(
                node, ref_time, half_budget, forward=True,
            )

            # Adaptive: if one direction empty, give full budget to the other
            if not bwd_selected and fwd_selected and len(fwd_selected) < n_neighbors:
                fwd_selected = self._diverse_neighbors(
                    node, ref_time, n_neighbors, forward=True,
                )
            elif not fwd_selected and bwd_selected and len(bwd_selected) < n_neighbors:
                bwd_selected = self._diverse_neighbors(
                    node, ref_time, n_neighbors, forward=False,
                )

            all_edges: List[Tuple[str, str, str]] = []
            visited = {node}
            for neighbor, edge_type, _t in bwd_selected:
                all_edges.append((neighbor, node, edge_type))
                visited.add(neighbor)
            for neighbor, edge_type, _t in fwd_selected:
                all_edges.append((node, neighbor, edge_type))
                visited.add(neighbor)
        else:
            all_edges = []
            visited = {node}

            bwd = self._get_neighbors_at_time(
                node, ref_time + 1e-6, half_budget, forward=False,
            )
            for neighbor, edge_type, _t in bwd:
                all_edges.append((neighbor, node, edge_type))
                visited.add(neighbor)

            fwd = self._get_neighbors_at_time(
                node, ref_time, half_budget, forward=True,
            )
            for neighbor, edge_type, _t in fwd:
                all_edges.append((node, neighbor, edge_type))
                visited.add(neighbor)

            if len(bwd) == 0 and len(fwd) > 0 and len(fwd) < n_neighbors:
                extra_fwd = self._get_neighbors_at_time(
                    node, ref_time, n_neighbors, forward=True,
                )
                for neighbor, edge_type, _t in extra_fwd:
                    if neighbor not in visited:
                        all_edges.append((node, neighbor, edge_type))
                        visited.add(neighbor)
            elif len(fwd) == 0 and len(bwd) > 0 and len(bwd) < n_neighbors:
                extra_bwd = self._get_neighbors_at_time(
                    node, ref_time + 1e-6, n_neighbors, forward=False,
                )
                for neighbor, edge_type, _t in extra_bwd:
                    if neighbor not in visited:
                        all_edges.append((neighbor, node, edge_type))
                        visited.add(neighbor)

        if len(all_edges) < 1:
            return None

        unique_nodes = [node] + [n for n in visited if n != node]
        return (node, unique_nodes, all_edges)

    def _diverse_neighbors(
        self,
        node: str,
        ref_time: float,
        budget: int,
        forward: bool = True,
    ) -> List[Tuple]:
        """Select diverse neighbors: dedup by (edge_type, neighbor_type, label),
        then prioritize edge type variety via round-robin.

        Fetches a large pool of temporal neighbors, deduplicates by
        (edge_type, neighbor_node_type, neighbor_label), then selects
        up to `budget` neighbors with round-robin across edge types
        to maximize action diversity.

        Returns:
            List of (neighbor, edge_type, time) tuples.
        """
        # Fetch large pool (5x budget) for dedup headroom
        pool_size = max(budget * 5, 100)
        ref = ref_time if forward else ref_time + 1e-6
        raw = self._get_neighbors_at_time(node, ref, pool_size, forward=forward)

        if not raw:
            return []

        # Dedup by (edge_type, neighbor_type, neighbor_label)
        seen_keys: set = set()
        deduped: List[Tuple] = []
        for neighbor, edge_type, t in raw:
            nlbl = self.node_labels.get(neighbor, ("unknown", "unknown"))
            key = (edge_type, nlbl[0], nlbl[1])
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append((neighbor, edge_type, t))

        if len(deduped) <= budget:
            return deduped

        # Group by edge type for round-robin selection
        by_etype: Dict[str, List[Tuple]] = {}
        for item in deduped:
            etype = item[1]
            if etype not in by_etype:
                by_etype[etype] = []
            by_etype[etype].append(item)

        # Shuffle within each group for randomness
        for items in by_etype.values():
            random.shuffle(items)

        # Round-robin: pick one per edge type, repeat until budget filled
        selected: List[Tuple] = []
        etypes = list(by_etype.keys())
        random.shuffle(etypes)

        while len(selected) < budget and etypes:
            next_etypes = []
            for etype in etypes:
                if len(selected) >= budget:
                    break
                if by_etype[etype]:
                    selected.append(by_etype[etype].pop())
                if by_etype[etype]:
                    next_etypes.append(etype)
            etypes = next_etypes

        return selected
