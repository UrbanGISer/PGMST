"""Physics-informed GNN embedding (default SGCN) as a small estimator class."""

from __future__ import annotations

from typing import Literal, Optional, Union

import pandas as pd

from pgmst.adjacency import contiguity_to_edges, validate_adjacency_ids
from pgmst.coords_prep import prepare_coords_table
from pgmst.embedding import EmbeddingBackbone, EmbeddingConfig, build_embedding
from pgmst.validate import validate_flow_endpoints
from pgmst.utils import norm_zone_key

ContiguitySpec = Union[Literal["queen", "rook"], pd.DataFrame]


class PGNN:
    """
    Default **SGCN** embedding on flow + contiguity graph (half-life OD weights, ``w_adj`` glue).

    Parameters
    ----------
    coords_df
        Zone table. Pass ``coordxy="x_col,y_col"`` for planar coordinates, or omit ``coordxy``
        to use ``geometry.centroid`` (requires ``geometry``).
    df_flow
        OD table: ``Origin, Destin, Flow`` or first three columns interpreted as O, D, flow.
    id_column
        Zone id column. If ``None``, uses ``coords_df.index``.
    w
        ``\"queen\"``, ``\"rook\"`` (``libpysal``), or a DataFrame with ``focal`` / ``neighbor``.
    k_hops, w_adj_percentile
        SGConv depth and spatial-glue percentile (defaults match the IJGIS notebook).
    flow_origin_col, flow_destin_col, flow_flow_col
        Optional explicit OD column names.
    coordxy
        ``"x_col,y_col"`` naming two numeric columns copied into ``XCoord``/``YCoord``.
        If ``None`` (default), centroids from ``geometry`` are used (requires ``geometry``).
    """

    def __init__(
        self,
        coords_df: pd.DataFrame,
        df_flow: pd.DataFrame,
        id_column: Optional[str] = None,
        w: ContiguitySpec = "queen",
        k_hops: int = 5,
        w_adj_percentile: float = 75.0,
        flow_origin_col: Optional[str] = None,
        flow_destin_col: Optional[str] = None,
        flow_flow_col: Optional[str] = None,
        coordxy: Optional[str] = None,
    ):
        self._coords_original = coords_df.copy()
        self._id_column = id_column
        self._internal = prepare_coords_table(coords_df, id_column, coordxy=coordxy)
        self._df_flow = df_flow.copy()
        self._w_spec = w
        self._oc = flow_origin_col
        self._dc = flow_destin_col
        self._fc = flow_flow_col

        valid_ids = set(self._internal["ZoneID"].dropna().map(norm_zone_key))
        valid_ids.discard(None)
        validate_flow_endpoints(self._df_flow, valid_ids, self._oc, self._dc, self._fc)

        if isinstance(w, pd.DataFrame):
            validate_adjacency_ids(w, valid_ids)
        self._df_adj = contiguity_to_edges(self._internal, "ZoneID", w)

        self._k_hops = int(k_hops)
        self._w_adj_percentile = float(w_adj_percentile)

    def embed(
        self,
        embed_x_col: str = "Emb_X",
        embed_y_col: str = "Emb_Y",
    ) -> pd.DataFrame:
        """
        Return a copy of the **original** ``coords_df`` with two appended embedding columns.

        Rows are aligned via ``id_column`` (or index) to the internal ``ZoneID`` keys.
        """
        cfg = EmbeddingConfig(
            backbone=EmbeddingBackbone.SGCN,
            k_hops=self._k_hops,
            w_adj_percentile=self._w_adj_percentile,
            spatial_glue=None,
        )
        emb_df, _, _ = build_embedding(
            self._internal,
            self._df_flow,
            self._df_adj,
            cfg,
            origin_col=self._oc,
            destin_col=self._dc,
            flow_col=self._fc,
        )

        merge_key = "_pgmst_norm_id"
        out = self._coords_original.copy()
        if self._id_column is not None:
            out[merge_key] = out[self._id_column].map(norm_zone_key)
        else:
            out[merge_key] = out.index.map(norm_zone_key)

        m = emb_df.rename(columns={"Emb_X": embed_x_col, "Emb_Y": embed_y_col})
        m = m.rename(columns={"ZoneID": "_pgmst_emb_zone"})
        out = out.merge(m, left_on=merge_key, right_on="_pgmst_emb_zone", how="left")
        out = out.drop(columns=["_pgmst_emb_zone", merge_key], errors="ignore")
        return out
