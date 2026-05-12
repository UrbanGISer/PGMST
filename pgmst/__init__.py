"""
PGMST — Physics-informed Graph embedding + Multi-objective MST partitioning.

Public API (two-stage):
  * ``pgmst.pgnn.PGNN`` — SGCN embedding appended to your zone table.
  * ``pgmst.mst.MST`` — embedding-space MST partition + dissolved metrics / modularity.
"""

from pgmst.embedding import EmbeddingBackbone, EmbeddingConfig, build_embedding, embedding_from_external_table
from pgmst.mst import MST, PartitionConfig, PartitionResult, run_mst_partition, total_modularity_partition
from pgmst.pipeline import PGMSTConfig, run_pgmst
from pgmst.pgnn import PGNN

__all__ = [
    "PGNN",
    "MST",
    "PartitionResult",
    "PGMSTConfig",
    "run_pgmst",
    "EmbeddingBackbone",
    "EmbeddingConfig",
    "build_embedding",
    "embedding_from_external_table",
    "PartitionConfig",
    "run_mst_partition",
    "total_modularity_partition",
]

__version__ = "0.2.0"
