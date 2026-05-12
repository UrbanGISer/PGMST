"""Attach ZoneID / XY for PGMST pipelines."""

from __future__ import annotations

import pandas as pd

from pgmst.utils import norm_zone_key


def _copy_frame_with_zone_id(coords_df: pd.DataFrame, id_col: str | None) -> pd.DataFrame:
    """Copy to GeoDataFrame when needed, attach normalized ``ZoneID``, one row per zone."""
    try:
        import geopandas as gpd

        if "geometry" in coords_df.columns and not isinstance(coords_df, gpd.GeoDataFrame):
            g = gpd.GeoDataFrame(coords_df.copy(), geometry="geometry", crs=getattr(coords_df, "crs", None))
        else:
            g = coords_df.copy()
    except ImportError:
        g = coords_df.copy()

    if id_col is None:
        g["ZoneID"] = g.index.map(lambda x: norm_zone_key(x))
    else:
        if id_col not in g.columns:
            raise KeyError(f"id_col {id_col!r} not in coords_df.")
        g["ZoneID"] = g[id_col].map(norm_zone_key)

    return g.dropna(subset=["ZoneID"]).drop_duplicates(subset=["ZoneID"], keep="first")


def prepare_mst_zone_table(coords_df: pd.DataFrame, id_col: str | None = None) -> pd.DataFrame:
    """
    Zone-attribute table for :class:`pgmst.mst.MST`: normalized ``ZoneID``, one row per zone.

    Does **not** add planar ``XCoord`` / ``YCoord``. Contiguity comes from ``w`` (Queen/Rook
    needs ``geometry``; edge list ``w`` does not). MST edge weights use only the embedding
    columns passed to :meth:`pgmst.mst.MST.partition`.
    """
    return _copy_frame_with_zone_id(coords_df, id_col)


def _parse_coordxy(coordxy: str | None) -> tuple[str, str] | None:
    """Return ``(x_col, y_col)`` or ``None`` if ``coordxy`` is empty / whitespace-only."""
    if coordxy is None:
        return None
    s = str(coordxy).strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            "coordxy must be two comma-separated column names, e.g. 'XCoord,YCoord' or 'lon,lat'; "
            f"got {coordxy!r}"
        )
    return parts[0], parts[1]


def prepare_coords_table(
    coords_df: pd.DataFrame,
    id_col: str | None = None,
    coordxy: str | None = None,
) -> pd.DataFrame:
    """
    Return a copy with ``ZoneID`` (normalized) and ``XCoord`` / ``YCoord``.

    If ``id_col`` is None, uses the frame's index as zone ids.

    Parameters
    ----------
    coordxy
        Two comma-separated **existing** column names for x and y (e.g. ``"XCoord,YCoord"``).
        Values are copied into ``XCoord`` / ``YCoord`` as floats.

        If ``None`` (default), ``XCoord`` / ``YCoord`` are taken from ``geometry.centroid``
        (requires a ``geometry`` column and geopandas).
    """
    g = _copy_frame_with_zone_id(coords_df, id_col)

    parsed = _parse_coordxy(coordxy)
    if parsed is not None:
        cx, cy = parsed
        if cx not in g.columns or cy not in g.columns:
            raise KeyError(f"coordxy columns not found on coords_df: {cx!r}, {cy!r}")
        g["XCoord"] = pd.to_numeric(g[cx], errors="coerce")
        g["YCoord"] = pd.to_numeric(g[cy], errors="coerce")
        return g

    try:
        import geopandas as gpd
    except ImportError as e:
        raise ValueError(
            "coordxy was not specified (centroid path) but geopandas is not installed; "
            "install geopandas or pass coordxy=\"your_x_col,your_y_col\"."
        ) from e

    if "geometry" not in g.columns:
        raise ValueError(
            "coordxy was not specified: need a geometry column to compute centroids, "
            'or pass coordxy="x_col,y_col" with your coordinate column names.'
        )

    if not isinstance(g, gpd.GeoDataFrame):
        g = gpd.GeoDataFrame(g, geometry="geometry", crs=getattr(g, "crs", None))
    if g.geometry.isna().all():
        raise ValueError("coordxy was not specified: geometry is all null; cannot compute centroids.")
    cent = g.geometry.centroid
    g["XCoord"] = cent.x
    g["YCoord"] = cent.y
    return g
