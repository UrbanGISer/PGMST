"""High-level PGMST run: embedding + MST partition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import pandas as pd

from pgmst.embedding import EmbeddingConfig, build_embedding
from pgmst.mst import PartitionConfig, run_mst_partition


@dataclass
class PGMSTConfig:
    embedding: EmbeddingConfig
    partition: PartitionConfig


def run_pgmst(
    coords_df: pd.DataFrame,
    df_flow: pd.DataFrame,
    df_adj: pd.DataFrame,
    config: Optional[PGMSTConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, Any, Any, float]:
    """
    Run full pipeline.

    Returns (embedding_df, partition_df, node_mapping, flow_matrix_sym, degrees, total_weight_m).
    """
    if config is None:
        config = PGMSTConfig(embedding=EmbeddingConfig(), partition=PartitionConfig())

    emb_df, node_mapping, _nodes = build_embedding(coords_df, df_flow, df_adj, config.embedding)
    part_df, node_map2, _fd, flow_sym, degrees, twm = run_mst_partition(
        emb_df, df_flow, df_adj, coords_df, config.partition
    )
    return emb_df, part_df, node_map2, flow_sym, degrees, twm
