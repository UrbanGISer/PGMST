"""GNN embedding backends (SGCN / GCN) on the physics-informed graph."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv

from pgmst.physics_graph import (
    compute_w_flow_nonzero,
    init_sgconv_lin_weight,
    node_feature_matrix,
    prepare_graph_tensors,
)
from pgmst.utils import coords_one_row_per_zone, norm_zone_key, resolve_hospital_column


class EmbeddingBackbone(str, Enum):
    SGCN = "sgcn"
    GCN = "gcn"


@dataclass
class EmbeddingConfig:
    backbone: EmbeddingBackbone | str = EmbeddingBackbone.SGCN
    k_hops: int = 5
    gcn_hidden: int = 16
    gcn_layers: int = 2
    spatial_glue: Optional[float] = None
    """Fixed w_adj on spatial edges; if None, use percentile of non-zero w_flow."""
    w_adj_percentile: float = 75.0
    feature_mode: Literal[
        "coords", "coords_pop", "coords_hosp_inflow", "coords_pop_hosp_inflow"
    ] = "coords"
    seed: int = 42


def _spatial_glue_value(
    coords_df: pd.DataFrame,
    df_flow: pd.DataFrame,
    cfg: EmbeddingConfig,
    origin_col: str | None = None,
    destin_col: str | None = None,
    flow_col: str | None = None,
) -> tuple[float, dict, np.ndarray]:
    w_flow, node_mapping, nodes_arr = compute_w_flow_nonzero(
        coords_df, df_flow, origin_col, destin_col, flow_col
    )
    if w_flow is None:
        raise RuntimeError(
            "No positive flow edges after matching OD to zones. "
            "Check Origin/Destin/Flow columns and ZoneID alignment."
        )
    num_nodes = len(nodes_arr)
    if cfg.spatial_glue is not None:
        w_adj = float(cfg.spatial_glue)
    else:
        w_adj = float(np.percentile(w_flow, cfg.w_adj_percentile))
    return w_adj, node_mapping, nodes_arr


def build_embedding(
    coords_df: pd.DataFrame,
    df_flow: pd.DataFrame,
    df_adj: pd.DataFrame,
    cfg: Optional[EmbeddingConfig] = None,
    origin_col: str | None = None,
    destin_col: str | None = None,
    flow_col: str | None = None,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """
    Returns (embedding_df with ZoneID, Emb_X, Emb_Y), node_mapping, nodes_arr.
    """
    cfg = cfg or EmbeddingConfig()
    w_adj, node_mapping, nodes_arr = _spatial_glue_value(
        coords_df, df_flow, cfg, origin_col, destin_col, flow_col
    )
    num_nodes = len(nodes_arr)
    coords_use = coords_one_row_per_zone(coords_df)
    try:
        hosp_col = resolve_hospital_column(coords_use)
    except KeyError:
        hosp_col = "hosp"
    edge_index, edge_weight, coords_use = prepare_graph_tensors(
        coords_df,
        df_flow,
        df_adj,
        w_adj,
        num_nodes,
        node_mapping,
        origin_col,
        destin_col,
        flow_col,
    )
    x_np = node_feature_matrix(
        coords_use,
        df_flow,
        node_mapping,
        num_nodes,
        cfg.feature_mode,
        hosp_col,
        origin_col,
        destin_col,
        flow_col,
    )
    x = torch.tensor(x_np, dtype=torch.float)
    torch.manual_seed(cfg.seed)

    backbone = cfg.backbone
    if isinstance(backbone, str):
        backbone = EmbeddingBackbone(backbone.lower())

    if backbone == EmbeddingBackbone.SGCN:
        z = _forward_sgcn(x, edge_index, edge_weight, cfg.k_hops)
    elif backbone == EmbeddingBackbone.GCN:
        z = _forward_gcn(x, edge_index, edge_weight, cfg.gcn_hidden, cfg.gcn_layers)
    else:
        raise ValueError(f"Unknown backbone: {cfg.backbone}")

    node_embeddings = z.numpy()
    reverse_mapping = {i: zid for zid, i in node_mapping.items()}
    zone_ids = [reverse_mapping[i] for i in range(num_nodes)]
    emb_df = pd.DataFrame(
        {"ZoneID": zone_ids, "Emb_X": node_embeddings[:, 0], "Emb_Y": node_embeddings[:, 1]}
    )
    return emb_df, node_mapping, nodes_arr


def _forward_sgcn(x, edge_index, edge_weight, k_hops: int) -> torch.Tensor:
    in_ch = x.shape[1]

    class PureDiffusionSGConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = SGConv(
                in_channels=in_ch, out_channels=2, K=k_hops, cached=False, bias=False
            )
            with torch.no_grad():
                self.conv.lin.weight.copy_(init_sgconv_lin_weight(in_ch, 2))
                self.conv.lin.weight.requires_grad = False

        def forward(self, x, edge_index, edge_weight):
            return self.conv(x, edge_index, edge_weight)

    model = PureDiffusionSGConv()
    model.eval()
    with torch.no_grad():
        return model(x, edge_index, edge_weight)


def _forward_gcn(x, edge_index, edge_weight, hidden: int, num_layers: int) -> torch.Tensor:
    in_ch = x.shape[1]
    layers_list = []
    ch = in_ch
    for i in range(num_layers):
        out = hidden if i < num_layers - 1 else 2
        layers_list.append(GCNConv(ch, out, bias=True))
        ch = out
    mod = torch.nn.ModuleList(layers_list)
    mod.eval()
    with torch.no_grad():
        h = x
        for i, conv in enumerate(mod):
            h = conv(h, edge_index, edge_weight)
            if i < len(mod) - 1:
                h = F.relu(h)
        if h.shape[1] > 2:
            h = h[:, :2]
        return h


def embedding_from_external_table(
    emb_table: pd.DataFrame,
    zone_col: str = "ZoneID",
    x_col: str = "Emb_X",
    y_col: str = "Emb_Y",
) -> pd.DataFrame:
    """Normalize external embeddings (e.g. Region2Vec, other GNN CSV) to PGMST column names."""
    out = emb_table[[zone_col, x_col, y_col]].copy()
    out.columns = ["ZoneID", "Emb_X", "Emb_Y"]
    out["ZoneID"] = out["ZoneID"].map(norm_zone_key)
    return out.dropna(subset=["ZoneID"])
