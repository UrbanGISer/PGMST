"""User-facing MST class and partition result bundle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from pgmst.adjacency import contiguity_to_edges, validate_adjacency_ids
from pgmst.coords_prep import prepare_mst_zone_table
from pgmst.mst.core import PartitionConfig, run_mst_partition, total_modularity_partition
from pgmst.utils import norm_zone_key
from pgmst.validate import validate_flow_endpoints

ConstraintTuple = Tuple[str, str, float]
ContiguitySpec = Union[Literal["queen", "rook"], pd.DataFrame]


def infer_population_column_from_constraints(constraints: Sequence[ConstraintTuple]) -> str:
    """
    If ``population_column`` is omitted on ``MST``, infer it from ``constraints``.

    Prefer a column that has both a ``min`` and a ``max`` entry (typical population + cap).
    Falls back to ``POPU`` when nothing matches.
    """
    if not constraints:
        return "POPU"
    min_cols = {c for c, m, _ in constraints if str(m).lower() == "min"}
    max_cols = {c for c, m, _ in constraints if str(m).lower() == "max"}
    both = sorted(min_cols & max_cols)
    if both:
        if "POPU" in both:
            return "POPU"
        return both[0]
    only_max = sorted(max_cols - min_cols)
    if only_max:
        return only_max[0]
    only_min = sorted(min_cols - max_cols)
    if only_min:
        return only_min[0]
    return "POPU"


def _constraints_to_partition_config(
    constraints: Sequence[ConstraintTuple],
    min_local_index: float,
    population_column: str,
    hospital_column: Optional[str],
    origin_col: str | None,
    destin_col: str | None,
    flow_col: str | None,
    embedding_x_col: str,
    embedding_y_col: str,
) -> PartitionConfig:
    pc = PartitionConfig(
        min_local_index=min_local_index,
        population_column=population_column,
        hospital_column=hospital_column,
        origin_col=origin_col,
        destin_col=destin_col,
        flow_col=flow_col,
        embedding_x_col=embedding_x_col,
        embedding_y_col=embedding_y_col,
        min_hospitals=0,
    )
    inferred_hosp_col: Optional[str] = hospital_column
    hosp_min_specified = False

    for item in constraints:
        if len(item) != 3:
            raise ValueError(f"Each constraint must be (column, 'min'|'max', value); got {item!r}")
        col, mode, val = item[0], str(item[1]).lower(), float(item[2])
        if mode not in ("min", "max"):
            raise ValueError(f"Constraint mode must be 'min' or 'max', got {item[1]!r}")

        if mode == "min":
            if col == population_column:
                pc.min_population = val
            elif hospital_column is not None and col == hospital_column:
                pc.min_hospitals = val
                hosp_min_specified = True
            elif inferred_hosp_col is not None and col == inferred_hosp_col:
                pc.min_hospitals = val
                hosp_min_specified = True
            elif inferred_hosp_col is None and col in ("hosp", "HOSP", "Hosp_ZoneI", "has_hosp"):
                inferred_hosp_col = col
                pc.hospital_column = col
                pc.min_hospitals = val
                hosp_min_specified = True
            else:
                pc.min_per_child_attributes[col] = val
        else:
            if col == population_column:
                pc.max_population = val
            else:
                pc.max_cluster_attributes[col] = val

    if inferred_hosp_col is not None:
        pc.hospital_column = inferred_hosp_col
    if not hosp_min_specified:
        pc.min_hospitals = 0.0
    return pc


def _reock(geometries: list) -> float:
    if not geometries:
        return 0.0
    try:
        import geopandas as gpd
        from shapely.ops import unary_union

        gs = gpd.GeoSeries(geometries)
        u = unary_union(gs.values)
        if u.is_empty:
            return 0.0
        ch = u.convex_hull
        if ch.area <= 0:
            return 0.0
        return float(u.area / ch.area)
    except Exception:
        return float("nan")


def _polsby_popper(geom) -> float:
    import math

    if geom is None or geom.is_empty:
        return 0.0
    p = float(geom.boundary.length)
    a = float(geom.area)
    if p < 1e-12 or a <= 0:
        return 0.0
    return float(4.0 * math.pi * a / (p * p))


@dataclass
class PartitionResult:
    """Output of ``MST.partition()`` — use ``.cluster``, ``.result``, ``.modularity``."""

    cluster: pd.DataFrame
    result: pd.DataFrame
    modularity: float
    _part_zone: pd.DataFrame = field(repr=False, default=None)
    _node_map: dict = field(repr=False, default=None)
    _flow_sym: Any = field(repr=False, default=None)
    _degrees: np.ndarray = field(repr=False, default=None)
    _twm: float = field(repr=False, default=None)


class MST:
    """
    Multi-objective MST partitioner in embedding space.

    Parameters
    ----------
    coords_df
        Zone attributes (``POPU``, ``hosp``, …). Must contain embedding columns ``emb``.
        Use ``id_column`` or the index for ``ZoneID``. Include ``geometry`` when ``w`` is
        ``\"queen\"`` or ``\"rook\"``; for an edge-list ``w``, geometry is only needed for
        dissolved map metrics in ``partition()`` output.
    id_column
        If None, zone ids are taken from ``coords_df.index`` (string-normalized).
    emb
        Names of two numeric columns (x, y) used as embedding coordinates for MST edge weights.
    w
        ``\"queen\"``, ``\"rook\"`` (``libpysal`` contiguity), or a DataFrame with ``focal`` / ``neighbor``.
    df_flow
        OD matrix; columns default to Origin, Destin, Flow, or the first three columns.
    constraints
        List of ``(column_name, \"min\"|\"max\", value)``. Population / hospital use the same
        semantics as the reference notebook (per-child minimum sums; cluster-level maximum for
        ``\"max\"`` on population or other attributes).
    min_local_index
        Minimum localization index (LI) required on **each** side of a modularity-improving cut.
    population_column
        Column used for population split / cap in the core solver. If ``None`` (default), it is
        **inferred** from ``constraints``: the first column that has both a ``min`` and a ``max``
        constraint (``POPU`` preferred when present); if none, falls back to ``\"POPU\"``.
    hospital_column
        Optional explicit hospital flag column; otherwise inferred from constraints or disabled
        when no ``(\"hosp\", \"min\", ...)`` style constraint is given and defaults leave ``min_hospitals`` 0.
    flow_origin_col, flow_destin_col, flow_flow_col
        Optional explicit OD column names.
    cluster_column_name
        Name of the community id column attached to ``cluster`` output (default ``cluster``).

    Notes
    -----
    The solver builds an MST on the **spatial adjacency** edges ``w``, with edge length =
    Euclidean distance in **embedding** space (``emb``). Cuts / DFS use **OD flow** and
    **tabular constraints** only; planar ``XCoord``/``YCoord`` are not used for the MST.
    """

    def __init__(
        self,
        coords_df: pd.DataFrame,
        df_flow: pd.DataFrame,
        id_column: Optional[str] = None,
        emb: Tuple[str, str] = ("Emb_X", "Emb_Y"),
        w: ContiguitySpec = "queen",
        constraints: Optional[Iterable[ConstraintTuple]] = None,
        min_local_index: float = 0.3,
        population_column: Optional[str] = None,
        hospital_column: Optional[str] = None,
        flow_origin_col: Optional[str] = None,
        flow_destin_col: Optional[str] = None,
        flow_flow_col: Optional[str] = None,
        cluster_column_name: str = "cluster",
    ):
        self._coords_original = coords_df.copy()
        self._coords_internal = prepare_mst_zone_table(coords_df, id_column)
        self._id_column = id_column
        self._emb = emb
        self._df_flow = df_flow.copy()
        self._w_spec = w
        self._constraints = list(constraints or [])
        self._min_li = float(min_local_index)
        self._population_column = population_column or infer_population_column_from_constraints(
            self._constraints
        )
        self._hospital_column = hospital_column
        self._oc, self._dc, self._fc = flow_origin_col, flow_destin_col, flow_flow_col
        self._cluster_col = cluster_column_name

        ex, ey = emb
        if ex not in self._coords_internal.columns or ey not in self._coords_internal.columns:
            raise KeyError(f"Embedding columns {emb!r} not found on coords_df.")

        valid_ids = set(self._coords_internal["ZoneID"].dropna().map(norm_zone_key))
        valid_ids.discard(None)
        validate_flow_endpoints(self._df_flow, valid_ids, self._oc, self._dc, self._fc)

        if isinstance(w, pd.DataFrame):
            validate_adjacency_ids(w, valid_ids)
            self._df_adj = contiguity_to_edges(self._coords_internal, "ZoneID", w)
        else:
            self._df_adj = contiguity_to_edges(self._coords_internal, "ZoneID", w)

        for col in self._constraints:
            if len(col) != 3:
                raise ValueError(f"Invalid constraint {col!r}")
            cname = col[0]
            if cname not in self._coords_internal.columns:
                raise KeyError(f"Constraint column {cname!r} not found on coords_df.")

        self._partition_config = _constraints_to_partition_config(
            self._constraints,
            min_local_index=self._min_li,
            population_column=self._population_column,
            hospital_column=self._hospital_column,
            origin_col=self._oc,
            destin_col=self._dc,
            flow_col=self._fc,
            embedding_x_col=emb[0],
            embedding_y_col=emb[1],
        )

    def partition(self) -> PartitionResult:
        emb_df = self._coords_internal[["ZoneID", self._emb[0], self._emb[1]]].copy()
        emb_df.columns = ["ZoneID", self._partition_config.embedding_x_col, self._partition_config.embedding_y_col]

        part_df, node_map, flow_dir, flow_sym, degrees, twm = run_mst_partition(
            emb_df,
            self._df_flow,
            self._df_adj,
            self._coords_internal,
            self._partition_config,
        )
        Q = total_modularity_partition(part_df, node_map, flow_sym, degrees, twm)

        merge_key = "_pgmst_norm_id"
        out = self._coords_original.copy()
        if self._id_column is not None:
            out[merge_key] = out[self._id_column].map(norm_zone_key)
        else:
            out[merge_key] = out.index.map(norm_zone_key)

        part_merge = part_df[["ZoneID", "Community_ID"]].drop_duplicates(subset=["ZoneID"])
        part_merge = part_merge.rename(columns={"ZoneID": "_pgmst_part_zone"})
        cluster = out.merge(part_merge, left_on=merge_key, right_on="_pgmst_part_zone", how="left")
        cluster = cluster.drop(columns=["_pgmst_part_zone", merge_key], errors="ignore").rename(
            columns={"Community_ID": self._cluster_col}
        )

        result = self._dissolved_metrics(part_df)

        return PartitionResult(
            cluster=cluster,
            result=result,
            modularity=Q,
            _part_zone=part_df,
            _node_map=node_map,
            _flow_sym=flow_sym,
            _degrees=degrees,
            _twm=twm,
        )

    def _dissolved_metrics(self, part_df: pd.DataFrame) -> pd.DataFrame:
        try:
            import geopandas as gpd
        except ImportError:
            gpd = None

        merged = self._coords_internal.merge(
            part_df[["ZoneID", "Community_ID"]].drop_duplicates("ZoneID"),
            on="ZoneID",
            how="inner",
        )

        comm_ids = sorted(part_df["Community_ID"].unique())
        rows = []
        constraint_cols = {c[0] for c in self._constraints}
        metric_cols = set(constraint_cols) | {self._population_column}
        hc = self._partition_config.hospital_column
        if hc:
            metric_cols.add(hc)

        for cid in comm_ids:
            zids = part_df.loc[part_df["Community_ID"] == cid, "ZoneID"].tolist()
            sub = self._coords_internal[self._coords_internal["ZoneID"].isin(zids)]
            row: dict[str, Any] = {"cluster": int(cid), "n_zones": len(zids)}

            comm_row = part_df[part_df["Community_ID"] == cid].drop_duplicates("Community_ID").iloc[0]
            row["Comm_LI"] = float(comm_row["Comm_LI"])
            row["Comm_Pop"] = float(comm_row["Comm_Pop"])
            row["Comm_Hosp"] = float(comm_row["Comm_Hosp"])

            dissolved_geom = None
            for col in sorted(metric_cols):
                if col not in sub.columns:
                    continue
                vals = sub[col].astype(float)
                row[f"{col}_sum"] = float(vals.sum())
                row[f"{col}_min"] = float(vals.min()) if len(vals) else float("nan")
                row[f"{col}_max"] = float(vals.max()) if len(vals) else float("nan")

            if gpd is not None and isinstance(merged, gpd.GeoDataFrame) and "geometry" in merged.columns:
                gci = merged.loc[merged["Community_ID"] == cid]
                geoms = gci.geometry.dropna().tolist()
                if geoms:
                    try:
                        dissolved_geom = gpd.GeoSeries(geoms, crs=merged.crs).unary_union
                        g2 = gpd.GeoSeries([dissolved_geom], crs=merged.crs)
                        g2p = g2.to_crs("EPSG:5070")
                        geom_p = g2p.iloc[0]
                        row["reock"] = _reock([geom_p])
                        row["polsby_popper"] = _polsby_popper(geom_p)
                    except Exception:
                        row["reock"] = float("nan")
                        row["polsby_popper"] = float("nan")
                        dissolved_geom = gpd.GeoSeries(geoms, crs=merged.crs).unary_union
                else:
                    row["reock"] = float("nan")
                    row["polsby_popper"] = float("nan")
            else:
                row["reock"] = float("nan")
                row["polsby_popper"] = float("nan")

            row["geometry"] = dissolved_geom
            rows.append(row)

        out_df = pd.DataFrame(rows)
        if gpd is not None and "geometry" in out_df.columns and out_df["geometry"].notna().any():
            try:
                return gpd.GeoDataFrame(out_df, geometry="geometry", crs=merged.crs)
            except Exception:
                pass
        return out_df.drop(columns=["geometry"], errors="ignore")
