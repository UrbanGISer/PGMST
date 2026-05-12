# PGMST

**P**hysics-informed **G**raph embedding + **M**ulti-objective **Minimum Spanning Tree** (MST-based) regionalization.

**Citation: Lingbo Liu & Fahui Wang,(2026), Physics-informed graph learning for spatially contiguous and capacity-constrained hospital service area delineation, Computers, Environment and Urban Systems**

<img width="975" height="463" alt="image" src="https://github.com/user-attachments/assets/9f59dd8a-9ad5-4de8-9353-92a7fe5798d2" />

Figure 1. Workflow for constructing a physics-informed graph, learning SGCN embeddings, and trimming an embedding-weighted minimum spanning tree (MST) to obtain spatially contiguous hospital service areas (HSAs) under modularity, localization, and capacity controls. (1) Distance-Adjusted Flows, (2) Assign Glue Flows, (3) Combining Graphs & Normalization, (4) GNN Embedding, (5) Adjacent Graph with Embedding Feature, (6) MST Trimmed with Multi-Criteria. 


<img width="972" height="944" alt="image" src="https://github.com/user-attachments/assets/c0d07f9e-ee51-4af7-b2e3-50fc463dfb28" />

Figure 2. Visualization of GNN Embedding, MST and resulting HSAs. (a) Embedding displacement field, where green points denote original ZCTA centroids and red points denote the rescaled 2D embedded coordinates plotted in the same map bounding box. Curved lines indicate displacement from geographic position to embedded functional position. ZCTA boundaries are retained as a light geographic reference layer and are intentionally de-emphasized so that centroid displacements remain visible in dense metropolitan areas.  (b) The Minimum Spanning Tree (MST) backbone, where link thickness represents the strength of functional connectivity (inverse embedding distance). (c) The spatial distribution of the Localization Index (LI) for the final HSAs, indicating the degree of self-containment. (d) The total population of each delineated HSA. Darker blue regions represent major metropolitan areas where functional integrity was prioritized over maximum population thresholds.

Two-stage API:

1. **`pgmst.pgnn.PGNN`** — packaged **SGCN** pipeline (half-life OD weights + spatial glue on contiguity edges). `embed()` appends **`Emb_X` / `Emb_Y`** to your zone table.
2. **`pgmst.mst.MST`** — build an MST on the **spatial adjacency** graph with edge costs from **embedding distance**, then **modularity- and constraint-driven** partitioning (population / hospital / custom caps, minimum localization index).

Optional: GCN-style embedding via `pgmst.physics_graph` + PyTorch Geometric (see example notebook).

## Install

```bash
pip install pgmst
```

For a development editable install from a clone:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from pgmst import PGNN, MST

coords_emb = PGNN(
    coords_df,
    df_flow,
    id_column="ZoneID",
    w=df_adj,
    k_hops=5,
    w_adj_percentile=75.0,
    coordxy="XCoord,YCoord",
).embed()

pr = MST(
    coords_emb,
    df_flow,
    id_column="ZoneID",
    emb=("Emb_X", "Emb_Y"),
    w=df_adj,
    constraints=[("POPU", "min", 10_000), ("POPU", "max", 500_000), ("hosp", "min", 1)],
    min_local_index=0.3,
).partition()

print(pr.modularity)
print(pr.cluster.head())
```

## Example notebook

After installation, the Florida demo notebook is shipped inside the package:

- **`pgmst/examples/PGMST_Florida_demo.ipynb`**

You can copy it to your working directory or open it from `site-packages/pgmst/examples/`.  
The notebook expects OD/adjacency/shapefile paths: point `REPO` at a checkout of **GNNGeoCommunity** (or your own data) where `data/OD_All_Flows30K.csv`, `data/FLAdjUpdate.csv`, and `data/Florida_2011.zip` live.

## PyPI release

See **[docs/PYPI.md](docs/PYPI.md)** for local builds (`python -m build` / `twine`) and **GitHub Actions** publishing (Trusted Publisher on PyPI).

## Requirements

Python 3.10+, PyTorch, PyTorch Geometric, GeoPandas, NetworkX, SciPy, pandas, NumPy. Queen/Rook contiguity needs **libpysal**.

## License

MIT — see [LICENSE](LICENSE).
