"""Cross-table ID checks for OD and adjacency."""

from __future__ import annotations

from typing import Optional

from pgmst.physics_graph import _flow_cols
from pgmst.utils import norm_zone_key


def validate_flow_endpoints(
    df_flow,
    valid_ids: set,
    origin_col: Optional[str] = None,
    destin_col: Optional[str] = None,
    flow_col: Optional[str] = None,
) -> None:
    oc, dc, _fc = _flow_cols(df_flow, origin_col, destin_col, flow_col)
    ends = set()
    for a, b in zip(df_flow[oc], df_flow[dc]):
        for x in (a, b):
            k = norm_zone_key(x)
            if k is not None:
                ends.add(k)
    missing = sorted(ends - valid_ids)
    if missing:
        raise ValueError(
            f"df_flow references {len(missing)} zone ids not present in coords id set "
            f"(showing up to 10): {missing[:10]}"
        )
