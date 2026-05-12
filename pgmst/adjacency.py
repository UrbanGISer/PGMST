"""Spatial contiguity: libpysal Queen/Rook or user-supplied edge list."""

from __future__ import annotations

from typing import Literal, Union

import pandas as pd

from pgmst.utils import norm_zone_key

ContiguityMode = Literal["queen", "rook"]


def adjacency_edges_from_weights(
    coords_df: pd.DataFrame,
    id_col: str,
    mode: ContiguityMode,
) -> pd.DataFrame:
    """
    Build undirected focal–neighbor edge list from polygon geometry.

    Parameters
    ----------
    coords_df
        GeoDataFrame (or frame with ``geometry``) indexed or keyed by ``id_col``.
    id_col
        Column used as zone id (must exist unless ids are in index — then use
        ``prepare_coords_table`` first so ``ZoneID`` exists).
    mode
        ``"queen"`` or ``"rook"``.
    """
    try:
        from libpysal.weights import Queen, Rook
    except ImportError as e:
        raise ImportError(
            "Queen/Rook contiguity requires libpysal. Install with: pip install libpysal"
        ) from e

    if id_col not in coords_df.columns:
        raise KeyError(f"id_col {id_col!r} not found on coords_df.")

    if mode not in ("queen", "rook"):
        raise ValueError('mode must be "queen" or "rook"')

    W_cls = Queen if mode == "queen" else Rook
    # libpysal expects GeoDataFrame-like with geometry
    if not hasattr(coords_df, "geometry"):
        raise TypeError("Queen/Rook require a GeoDataFrame with a geometry column.")

    w = W_cls.from_dataframe(coords_df, idVariable=id_col)
    rows = []
    seen = set()
    for fid, nbrs in w.neighbors.items():
        fo = norm_zone_key(fid)
        if fo is None:
            continue
        for nb in nbrs:
            no = norm_zone_key(nb)
            if no is None or fo == no:
                continue
            a, b = (fo, no) if fo <= no else (no, fo)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            rows.append({"focal": a, "neighbor": b})
    return pd.DataFrame(rows)


def normalize_adjacency_df(df_adj: pd.DataFrame) -> pd.DataFrame:
    """Ensure focal/neighbor columns exist (case-insensitive rename)."""
    lower = {c.lower(): c for c in df_adj.columns}
    if "focal" in lower and "neighbor" in lower:
        out = df_adj[[lower["focal"], lower["neighbor"]]].copy()
        out.columns = ["focal", "neighbor"]
        return out
    raise ValueError("df_adj must contain columns focal and neighbor (any case).")


def validate_adjacency_ids(df_adj: pd.DataFrame, valid_ids: set[str]) -> None:
    """Raise if any focal/neighbor id is not in valid_ids."""
    out = normalize_adjacency_df(df_adj)
    for _, r in out.iterrows():
        for k in (r["focal"], r["neighbor"]):
            kk = norm_zone_key(k)
            if kk is not None and kk not in valid_ids:
                raise ValueError(f"Adjacency references unknown zone id: {k!r} (normalized {kk!r})")


def contiguity_to_edges(
    coords_df: pd.DataFrame,
    id_col: str,
    w: Union[ContiguityMode, pd.DataFrame],
) -> pd.DataFrame:
    """
    Unified contiguity → ``focal`` / ``neighbor`` edge list.

    ``w`` is ``"queen"``, ``"rook"``, or a DataFrame with focal/neighbor columns.
    """
    if isinstance(w, pd.DataFrame):
        return normalize_adjacency_df(w)
    if isinstance(w, str) and w.lower() in ("queen", "rook"):
        return adjacency_edges_from_weights(coords_df, id_col, w.lower())  # type: ignore[arg-type]
    raise ValueError('w must be "queen", "rook", or a pandas DataFrame (focal, neighbor).')
