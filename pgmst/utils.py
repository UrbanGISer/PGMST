"""Shared zone keys and attribute-table helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def norm_zone_key(z):
    """
    Map shapefile ZoneID and OD Origin/Destin to one key type so int/str mismatches
    (e.g. 1 vs '1') do not drop all flow edges.
    """
    if z is None:
        return None
    try:
        if isinstance(z, float) and np.isnan(z):
            return None
    except TypeError:
        pass
    if isinstance(z, str) and z.strip() == "":
        return None
    try:
        return str(int(float(z)))
    except (ValueError, TypeError, OverflowError):
        return str(z).strip()


def coords_one_row_per_zone(coords_df: pd.DataFrame) -> pd.DataFrame:
    """Dropna XY, normalize ZoneID, one row per zone."""
    c = coords_df.dropna(subset=["XCoord", "YCoord"]).copy()
    c["ZoneID"] = c["ZoneID"].map(norm_zone_key)
    c = c.dropna(subset=["ZoneID"]).drop_duplicates("ZoneID", keep="first")
    return c


def zones_one_row_per_zone(coords_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ``ZoneID``, one row per zone. Used by MST partition (no planar XY required)."""
    if "ZoneID" not in coords_df.columns:
        raise KeyError("Expected column 'ZoneID' on coords_df.")
    c = coords_df.dropna(subset=["ZoneID"]).copy()
    c["ZoneID"] = c["ZoneID"].map(norm_zone_key)
    c = c.dropna(subset=["ZoneID"]).drop_duplicates("ZoneID", keep="first")
    return c


def resolve_hospital_column(coords_use: pd.DataFrame) -> str:
    for c in ("hosp", "Hosp_ZoneI", "HOSP", "has_hosp"):
        if c in coords_use.columns:
            return c
    raise KeyError(
        "No hospital column found on coords_df (expected one of: hosp, Hosp_ZoneI, HOSP, has_hosp)."
    )
