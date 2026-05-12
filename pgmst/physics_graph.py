"""
Physics-informed OD graph: half-life decay on flow weights + spatial glue (w_adj).

Matches the construction in SGCN_MST_IJGIS_2025-V01 §2-1 / notebook/_glue_w_adj_sensitivity_impl.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, coalesce

from pgmst.utils import coords_one_row_per_zone, norm_zone_key


def _flow_cols(df_flow: pd.DataFrame, origin_col: str | None, destin_col: str | None, flow_col: str | None):
    if origin_col and destin_col and flow_col:
        return origin_col, destin_col, flow_col
    if {"Origin", "Destin", "Flow"}.issubset(df_flow.columns):
        return "Origin", "Destin", "Flow"
    if len(df_flow.columns) < 3:
        raise ValueError("df_flow needs at least 3 columns or explicit origin/destin/flow column names.")
    c0, c1, c2 = df_flow.columns[0], df_flow.columns[1], df_flow.columns[2]
    return str(c0), str(c1), str(c2)


def compute_w_flow_nonzero(
    coords_df: pd.DataFrame,
    df_flow: pd.DataFrame,
    origin_col: str | None = None,
    destin_col: str | None = None,
    flow_col: str | None = None,
):
    """Non-zero flow edge weights w_flow (half-life vs distance, log1p(flow))."""
    oc, dc, fc = _flow_cols(df_flow, origin_col, destin_col, flow_col)
    coords_use = coords_one_row_per_zone(coords_df)
    nodes = coords_use["ZoneID"].unique()
    node_mapping = {zid: i for i, zid in enumerate(nodes)}
    raw_coords = coords_use.set_index("ZoneID")[["XCoord", "YCoord"]].to_dict("index")

    def get_raw_dist(id1, id2):
        if id1 not in raw_coords or id2 not in raw_coords:
            return 1.0
        c1, c2 = raw_coords[id1], raw_coords[id2]
        dist = np.sqrt((c1["XCoord"] - c2["XCoord"]) ** 2 + (c1["YCoord"] - c2["YCoord"]) ** 2)
        return max(dist, 0.001)

    raw_flow_dists = []
    raw_flow_vals = []
    temp_flow_edges = []

    for o, d, fl in zip(df_flow[oc], df_flow[dc], df_flow[fc]):
        origin = norm_zone_key(o)
        destin = norm_zone_key(d)
        if origin in node_mapping and destin in node_mapping and origin != destin:
            dist = get_raw_dist(origin, destin)
            fv = float(fl)
            if fv > 0:
                temp_flow_edges.append([node_mapping[origin], node_mapping[destin]])
                raw_flow_dists.append(dist)
                raw_flow_vals.append(fv)

    if len(temp_flow_edges) == 0:
        return None, node_mapping, nodes

    flow_dists_arr = np.array(raw_flow_dists)
    flow_vals_arr = np.array(raw_flow_vals)

    sorted_indices = np.argsort(flow_dists_arr)
    sorted_dists = flow_dists_arr[sorted_indices]
    sorted_weights = flow_vals_arr[sorted_indices]

    cumsum_weights = np.cumsum(sorted_weights)
    total_weight = cumsum_weights[-1]

    median_idx = np.searchsorted(cumsum_weights, 0.5 * total_weight)
    median_dist = sorted_dists[median_idx]

    if median_dist < 1e-6:
        median_dist = 1.0

    decay = np.power(0.5, flow_dists_arr / median_dist)
    w_flow = np.log1p(flow_vals_arr) * decay
    return w_flow, node_mapping, nodes


def _zone_hospital_destination_inflow(
    coords_use, df_flow, node_mapping, num_nodes, hosp_col: str, oc: str, dc: str, fc: str
):
    hosp_flag = coords_use.set_index("ZoneID")[hosp_col].astype(float)
    acc = np.zeros(num_nodes, dtype=np.float64)
    for _o, dest, fl in zip(df_flow[oc], df_flow[dc], df_flow[fc]):
        d = norm_zone_key(dest)
        if d not in node_mapping:
            continue
        if d not in hosp_flag.index or float(hosp_flag.loc[d]) < 0.5:
            continue
        acc[node_mapping[d]] += float(fl)
    return acc


def prepare_graph_tensors(
    coords_df: pd.DataFrame,
    df_flow: pd.DataFrame,
    df_adj: pd.DataFrame,
    spatial_glue: float,
    num_nodes: int,
    node_mapping: dict,
    origin_col: str | None = None,
    destin_col: str | None = None,
    flow_col: str | None = None,
):
    """Directed+undirected coalesced edges, self-loops, GCN normalization."""
    oc, dc, fc = _flow_cols(df_flow, origin_col, destin_col, flow_col)
    coords_use = coords_one_row_per_zone(coords_df)
    raw_coords = coords_use.set_index("ZoneID")[["XCoord", "YCoord"]].to_dict("index")

    def get_raw_dist(id1, id2):
        if id1 not in raw_coords or id2 not in raw_coords:
            return 1.0
        c1, c2 = raw_coords[id1], raw_coords[id2]
        dist = np.sqrt((c1["XCoord"] - c2["XCoord"]) ** 2 + (c1["YCoord"] - c2["YCoord"]) ** 2)
        return max(dist, 0.001)

    edges_list = []
    weights_list = []
    raw_flow_dists = []
    raw_flow_vals = []
    temp_flow_edges = []

    for o, d, fl in zip(df_flow[oc], df_flow[dc], df_flow[fc]):
        origin = norm_zone_key(o)
        destin = norm_zone_key(d)
        if origin in node_mapping and destin in node_mapping and origin != destin:
            dist = get_raw_dist(origin, destin)
            fv = float(fl)
            if fv > 0:
                temp_flow_edges.append([node_mapping[origin], node_mapping[destin]])
                raw_flow_dists.append(dist)
                raw_flow_vals.append(fv)

    if len(temp_flow_edges) > 0:
        flow_dists_arr = np.array(raw_flow_dists)
        flow_vals_arr = np.array(raw_flow_vals)
        sorted_indices = np.argsort(flow_dists_arr)
        sorted_dists = flow_dists_arr[sorted_indices]
        sorted_weights = flow_vals_arr[sorted_indices]
        cumsum_weights = np.cumsum(sorted_weights)
        total_weight = cumsum_weights[-1]
        median_idx = np.searchsorted(cumsum_weights, 0.5 * total_weight)
        median_dist = sorted_dists[median_idx]
        if median_dist < 1e-6:
            median_dist = 1.0
        decay = np.power(0.5, flow_dists_arr / median_dist)
        w_flow_e = np.log1p(flow_vals_arr) * decay
        edges_list.extend(temp_flow_edges)
        weights_list.extend(w_flow_e.tolist())

    for row in df_adj.itertuples():
        focal = norm_zone_key(getattr(row, "focal"))
        neighbor = norm_zone_key(getattr(row, "neighbor"))
        if focal in node_mapping and neighbor in node_mapping and focal != neighbor:
            edges_list.append([node_mapping[focal], node_mapping[neighbor]])
            weights_list.append(spatial_glue)

    if len(edges_list) == 0:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_weight = torch.tensor([1.0], dtype=torch.float)
    else:
        edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights_list, dtype=torch.float)
        row, col = edge_index
        undir_edge_index = torch.cat([edge_index, torch.stack([col, row], dim=0)], dim=1)
        undir_edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
        edge_index, edge_weight = coalesce(
            undir_edge_index, undir_edge_weight, num_nodes=num_nodes, reduce="add"
        )

    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, num_nodes=num_nodes, fill_value=spatial_glue
    )
    edge_index, edge_weight = gcn_norm(edge_index, edge_weight, num_nodes, add_self_loops=False)

    return edge_index, edge_weight, coords_use


def node_feature_matrix(
    coords_use: pd.DataFrame,
    df_flow: pd.DataFrame,
    node_mapping: dict,
    num_nodes: int,
    feature_mode: str,
    hosp_col: str,
    origin_col: str | None = None,
    destin_col: str | None = None,
    flow_col: str | None = None,
):
    """Column-wise z-scored node features for SGCN/GCN input."""
    oc, dc, fc = _flow_cols(df_flow, origin_col, destin_col, flow_col)
    coords_array = np.zeros((num_nodes, 2), dtype=np.float64)
    pop = np.zeros(num_nodes, dtype=np.float64)

    pop_series = coords_use.set_index("ZoneID")["POPU"] if "POPU" in coords_use.columns else None
    for zid, idx in node_mapping.items():
        row = coords_use.loc[coords_use["ZoneID"] == zid, ["XCoord", "YCoord"]]
        if len(row) > 0:
            coords_array[idx] = row.values[0].astype(np.float64)
        if pop_series is not None:
            try:
                pop[idx] = float(pop_series.loc[zid])
            except KeyError:
                pop[idx] = 0.0

    fm = feature_mode.strip().lower()
    hosp_inflow = np.zeros(num_nodes, dtype=np.float64)
    if fm in ("coords_hosp_inflow", "coords_pop_hosp_inflow"):
        hosp_inflow = _zone_hospital_destination_inflow(
            coords_use, df_flow, node_mapping, num_nodes, hosp_col, oc, dc, fc
        )

    if fm == "coords":
        X = coords_array
    elif fm == "coords_pop":
        X = np.column_stack([coords_array, np.log1p(np.maximum(pop, 0.0))])
    elif fm == "coords_hosp_inflow":
        X = np.column_stack([coords_array, np.log1p(np.maximum(hosp_inflow, 0.0))])
    elif fm == "coords_pop_hosp_inflow":
        X = np.column_stack(
            [
                coords_array,
                np.log1p(np.maximum(pop, 0.0)),
                np.log1p(np.maximum(hosp_inflow, 0.0)),
            ]
        )
    else:
        raise ValueError(
            f"Unknown feature_mode={feature_mode!r}. Use "
            "'coords', 'coords_pop', 'coords_hosp_inflow', or 'coords_pop_hosp_inflow'."
        )

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xn = (X - mean) / std
    return np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def init_sgconv_lin_weight(in_channels: int, out_channels: int = 2) -> torch.Tensor:
    """Identity on first two inputs (XY); equal small coupling for extra channels."""
    W = torch.zeros(out_channels, in_channels)
    if in_channels >= 2:
        W[0, 0] = 1.0
        W[1, 1] = 1.0
    elif in_channels == 1:
        W[0, 0] = 1.0
    if in_channels > 2:
        s = 1.0 / float(np.sqrt(in_channels - 2))
        W[:, 2:] = s
    return W
