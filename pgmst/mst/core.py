"""
Multi-objective MST partitioning in embedding space (contiguity edges + OD modularity splits).
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from pgmst.physics_graph import _flow_cols
from pgmst.utils import norm_zone_key, resolve_hospital_column, zones_one_row_per_zone


@dataclass
class PartitionConfig:
    min_population: float = 10_000.0
    min_hospitals: float = 1.0
    min_local_index: float = 0.3
    max_population: float = 500_000.0
    population_column: str = "POPU"
    hospital_column: Optional[str] = None
    min_per_child_attributes: Dict[str, float] = field(default_factory=dict)
    max_cluster_attributes: Dict[str, float] = field(default_factory=dict)
    origin_col: Optional[str] = None
    destin_col: Optional[str] = None
    flow_col: Optional[str] = None
    embedding_x_col: str = "Emb_X"
    embedding_y_col: str = "Emb_Y"


def run_mst_partition(
    embedding_df: pd.DataFrame,
    df_flow_base: pd.DataFrame,
    df_adj: pd.DataFrame,
    coords_df: pd.DataFrame,
    config: Optional[PartitionConfig] = None,
) -> tuple[pd.DataFrame, dict, Any, Any, np.ndarray, float]:
    """
    Returns (partition_df, node_map, flow_matrix_directed, flow_matrix_sym, degrees, total_weight_m).
    """
    config = config or PartitionConfig()
    sys.setrecursionlimit(20000)

    df_emb = embedding_df.copy()
    df_attr = zones_one_row_per_zone(coords_df)
    df_flow = df_flow_base.copy()
    for col in ("O_idx", "D_idx"):
        if col in df_flow.columns:
            df_flow = df_flow.drop(columns=[col])

    oc, dc, fc = _flow_cols(df_flow, config.origin_col, config.destin_col, config.flow_col)

    pop_col = config.population_column
    if pop_col not in df_attr.columns:
        raise KeyError(f"population_column {pop_col!r} not found on coords_df.")

    nodes = df_attr["ZoneID"].unique()
    node_map = {id: i for i, id in enumerate(nodes)}
    reverse_map = {i: id for id, i in node_map.items()}
    num_nodes = len(nodes)

    hosp_col = config.hospital_column
    if hosp_col is not None:
        if hosp_col not in df_attr.columns:
            raise KeyError(f"hospital_column {hosp_col!r} not found on coords_df.")
    elif config.min_hospitals and config.min_hospitals > 0:
        hosp_col = resolve_hospital_column(df_attr)
    else:
        hosp_col = None

    if hosp_col is None:
        hosp_dict = {node_map[z]: 0.0 for z in node_map}
    else:
        hosp_dict = dict(zip(df_attr["ZoneID"].map(node_map), df_attr[hosp_col].astype(float)))

    pop_dict = dict(zip(df_attr["ZoneID"].map(node_map), df_attr[pop_col].astype(float)))

    extra_cols = set(config.min_per_child_attributes) | set(config.max_cluster_attributes)
    for c in extra_cols:
        if c not in df_attr.columns:
            raise KeyError(f"Constraint column {c!r} not found on coords_df.")
    idx_df = df_attr.set_index("ZoneID")
    extra_dicts: Dict[str, Dict[int, float]] = {}
    for c in extra_cols:
        extra_dicts[c] = {}
        for zid in node_map:
            try:
                v = idx_df.loc[zid, c]
                v = float(v) if pd.notna(v) else 0.0
            except KeyError:
                v = 0.0
            extra_dicts[c][node_map[zid]] = v

    ex, ey = config.embedding_x_col, config.embedding_y_col
    if ex not in df_emb.columns or ey not in df_emb.columns:
        raise KeyError(f"Embedding columns {ex!r}, {ey!r} not found on embedding_df.")
    emb_dict: Dict[int, np.ndarray] = {}
    for _, row in df_emb.iterrows():
        zk = norm_zone_key(row["ZoneID"])
        if zk in node_map:
            emb_dict[node_map[zk]] = np.array([float(row[ex]), float(row[ey])], dtype=float)

    for node_idx in node_map.values():
        if node_idx not in emb_dict:
            emb_dict[node_idx] = np.zeros(2)

    df_flow["O_idx"] = df_flow[oc].map(norm_zone_key).map(node_map)
    df_flow["D_idx"] = df_flow[dc].map(norm_zone_key).map(node_map)
    valid_flow = df_flow.dropna(subset=["O_idx", "D_idx"])

    O_indices = valid_flow["O_idx"].astype(int).values
    D_indices = valid_flow["D_idx"].astype(int).values
    Flow_values = valid_flow[fc].astype(float).values

    node_total_outflow = {i: 0.0 for i in range(num_nodes)}
    for idx, val in valid_flow.groupby("O_idx")[fc].sum().items():
        node_total_outflow[int(idx)] = val

    flow_matrix_directed = coo_matrix(
        (Flow_values, (O_indices, D_indices)), shape=(num_nodes, num_nodes)
    ).tocsr()
    flow_matrix_sym = flow_matrix_directed + flow_matrix_directed.T

    total_weight_m = flow_matrix_sym.sum() / 2.0
    if total_weight_m == 0:
        total_weight_m = 1.0

    degrees = np.array(flow_matrix_sym.sum(axis=1)).flatten()

    def calculate_li(node_list):
        if len(node_list) == 0:
            return 0.0
        indices = np.array(node_list, dtype=np.int32)
        total_out = sum(node_total_outflow.get(n, 0.0) for n in indices)
        if total_out == 0:
            return 0.0
        try:
            internal_flow = flow_matrix_directed[indices, :][:, indices].sum()
        except IndexError:
            return 0.0
        return internal_flow / total_out

    def calculate_modularity_term(node_indices):
        if len(node_indices) == 0:
            return -1e9
        idx_arr = np.array(node_indices, dtype=np.int32)
        internal_w = flow_matrix_sym[idx_arr, :][:, idx_arr].sum() / 2.0
        degree_sum = np.sum(degrees[idx_arr])
        term1 = internal_w / total_weight_m
        term2 = (degree_sum / (2 * total_weight_m)) ** 2
        return term1 - term2

    all_edges = []
    for _, row in df_adj.iterrows():
        u_orig = norm_zone_key(getattr(row, "focal"))
        v_orig = norm_zone_key(getattr(row, "neighbor"))
        if u_orig in node_map and v_orig in node_map:
            u, v = node_map[u_orig], node_map[v_orig]
            if u in emb_dict and v in emb_dict:
                dist = np.linalg.norm(emb_dict[u] - emb_dict[v])
                all_edges.append((u, v, dist))

    def cluster_exceeds_max(cluster: list[int]) -> bool:
        tp = sum(pop_dict.get(n, 0) for n in cluster)
        if tp > config.max_population:
            return True
        for col, mx in config.max_cluster_attributes.items():
            if sum(extra_dicts[col].get(n, 0.0) for n in cluster) > mx:
                return True
        return False

    def get_subtree_stats_dfs(mst, root, total_pop_parent, total_hosp_parent):
        edges_to_check = []
        visited = set()

        def _dfs(u):
            visited.add(u)
            current_pop = pop_dict.get(u, 0)
            current_hosp = hosp_dict.get(u, 0)
            current_nodes = [u]

            for v in mst.neighbors(u):
                if v not in visited:
                    weight = mst[u][v]["weight"]
                    child_pop, child_hosp, child_nodes = _dfs(v)

                    current_pop += child_pop
                    current_hosp += child_hosp
                    current_nodes.extend(child_nodes)

                    edges_to_check.append(
                        {
                            "u": u,
                            "v": v,
                            "weight": weight,
                            "pop_side": child_pop,
                            "hosp_side": child_hosp,
                            "nodes_side": child_nodes,
                        }
                    )

            return current_pop, current_hosp, current_nodes

        _dfs(root)
        return edges_to_check

    def split_cluster_optimized(current_nodes, is_force_split_attempt=False):
        if len(current_nodes) <= 1:
            return False, None, None

        node_set = set(current_nodes)
        sub_edges = [(u, v, w) for u, v, w in all_edges if u in node_set and v in node_set]

        G_sub = nx.Graph()
        G_sub.add_nodes_from(current_nodes)
        G_sub.add_weighted_edges_from(sub_edges)

        if not nx.is_connected(G_sub):
            comps = list(nx.connected_components(G_sub))
            if len(comps) > 1:
                comps.sort(key=len, reverse=True)
                return True, list(comps[0]), list(set(current_nodes) - set(comps[0]))
            return False, None, None

        mst = nx.minimum_spanning_tree(G_sub, weight="weight")

        total_pop_parent = sum(pop_dict.get(n, 0) for n in current_nodes)
        total_hosp_parent = sum(hosp_dict.get(n, 0) for n in current_nodes)

        q_parent = calculate_modularity_term(current_nodes)

        root_for_dfs = current_nodes[0]
        candidates = get_subtree_stats_dfs(mst, root_for_dfs, total_pop_parent, total_hosp_parent)

        best_cut_nodes_side = None
        best_score = -float("inf")

        for cand in candidates:
            pop_1 = cand["pop_side"]
            hosp_1 = cand["hosp_side"]
            pop_2 = total_pop_parent - pop_1
            hosp_2 = total_hosp_parent - hosp_1

            if (
                pop_1 < config.min_population
                or hosp_1 < config.min_hospitals
                or pop_2 < config.min_population
                or hosp_2 < config.min_hospitals
            ):
                continue

            nodes_1 = cand["nodes_side"]
            set_1 = set(nodes_1)
            nodes_2 = [n for n in current_nodes if n not in set_1]

            skip = False
            for col, minv in config.min_per_child_attributes.items():
                s1 = sum(extra_dicts[col].get(n, 0.0) for n in nodes_1)
                s2 = sum(extra_dicts[col].get(n, 0.0) for n in nodes_2)
                if s1 < minv or s2 < minv:
                    skip = True
                    break
            if skip:
                continue

            li_1 = calculate_li(nodes_1)
            li_2 = calculate_li(nodes_2)
            if li_1 < config.min_local_index or li_2 < config.min_local_index:
                continue

            q_1 = calculate_modularity_term(nodes_1)
            q_2 = calculate_modularity_term(nodes_2)
            score = (q_1 + q_2) - q_parent

            if score > best_score:
                best_score = score
                best_cut_nodes_side = nodes_1

        if best_cut_nodes_side:
            if best_score > 0 or is_force_split_attempt:
                nodes_1 = best_cut_nodes_side
                nodes_2 = list(set(current_nodes) - set(nodes_1))
                return True, nodes_1, nodes_2

        return False, None, None

    node_queue = deque([list(node_map.values())])
    final_clusters = []
    processed_clusters = set()

    while len(node_queue) > 0:
        current_cluster = node_queue.popleft()

        current_cluster_tuple = tuple(sorted(current_cluster))
        if current_cluster_tuple in processed_clusters:
            continue

        current_total_pop = sum(pop_dict.get(n, 0) for n in current_cluster)

        under_pop_cap = current_total_pop <= config.max_population
        under_extra_caps = all(
            sum(extra_dicts[col].get(n, 0.0) for n in current_cluster) <= mx
            for col, mx in config.max_cluster_attributes.items()
        )
        if under_pop_cap and under_extra_caps and current_total_pop < config.min_population * 2:
            final_clusters.append(current_cluster)
            processed_clusters.add(current_cluster_tuple)
            continue

        is_force_split_attempt = (current_total_pop > config.max_population) or cluster_exceeds_max(
            current_cluster
        )

        success, child_A, child_B = split_cluster_optimized(
            current_cluster, is_force_split_attempt=is_force_split_attempt
        )

        if success:
            pop_A = sum(pop_dict.get(n, 0) for n in child_A)

            if pop_A > config.max_population or pop_A >= config.min_population * 2 or cluster_exceeds_max(
                child_A
            ):
                node_queue.append(child_A)
            else:
                final_clusters.append(child_A)
                processed_clusters.add(tuple(sorted(child_A)))

            pop_B = sum(pop_dict.get(n, 0) for n in child_B)
            if pop_B > config.max_population or pop_B >= config.min_population * 2 or cluster_exceeds_max(
                child_B
            ):
                node_queue.append(child_B)
            else:
                final_clusters.append(child_B)
                processed_clusters.add(tuple(sorted(child_B)))
        else:
            final_clusters.append(current_cluster)
            processed_clusters.add(current_cluster_tuple)

    unique_final_clusters = []
    seen_clusters = set()
    for cluster in final_clusters:
        cluster_tuple = tuple(sorted(cluster))
        if cluster_tuple not in seen_clusters:
            unique_final_clusters.append(cluster)
            seen_clusters.add(cluster_tuple)
    final_clusters = unique_final_clusters

    results = []
    for comm_id, comp_nodes in enumerate(final_clusters):
        comp_array = np.array(comp_nodes, dtype=np.int32)
        total_hosp = sum(hosp_dict.get(n, 0) for n in comp_nodes)
        total_pop = sum(pop_dict.get(n, 0) for n in comp_nodes)
        final_li = calculate_li(comp_array)

        for node_idx in comp_nodes:
            results.append(
                {
                    "ZoneID": reverse_map[node_idx],
                    "Community_ID": comm_id,
                    "Comm_Pop": total_pop,
                    "Comm_Hosp": total_hosp,
                    "Comm_LI": final_li,
                }
            )

    return pd.DataFrame(results), node_map, flow_matrix_directed, flow_matrix_sym, degrees, total_weight_m


def total_modularity_partition(part_df, node_map, flow_matrix_sym, degrees, total_weight_m):
    comm = part_df.drop_duplicates(subset=["Community_ID"])
    tot = 0.0
    for cid in comm["Community_ID"].values:
        zids = part_df.loc[part_df["Community_ID"] == cid, "ZoneID"].tolist()
        idx = np.array([node_map[z] for z in zids if z in node_map], dtype=np.int32)
        if len(idx) == 0:
            continue
        internal_w = flow_matrix_sym[idx, :][:, idx].sum() / 2.0
        degree_sum = np.sum(degrees[idx])
        tot += internal_w / total_weight_m - (degree_sum / (2 * total_weight_m)) ** 2
    return float(tot)
