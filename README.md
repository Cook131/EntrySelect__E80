<div align="center">

<!-- BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=24,20,17&height=200&section=header&text=Entry%20Point%20Resolver&fontSize=48&fontColor=cdd6f4&animation=fadeIn&fontAlignY=38&desc=Neural%20Pipeline%20for%20Warehouse%20Station%20Lines&descAlignY=58&descColor=cba6f7" alt="banner" width="100%"/>

<!-- BADGES -->
<p>
  <img src="https://img.shields.io/badge/status-WIP-f38ba8?style=for-the-badge&logo=construction&logoColor=1e1e2e" />
  <img src="https://img.shields.io/badge/Python-3.10+-cba6f7?style=for-the-badge&logo=python&logoColor=1e1e2e" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-f5c2e7?style=for-the-badge&logo=pytorch&logoColor=1e1e2e" />
  <img src="https://img.shields.io/badge/models-3%20MLPs-a6e3a1?style=for-the-badge&logo=tensorflow&logoColor=1e1e2e" />
  <img src="https://img.shields.io/badge/input-CAD%20CSV-89b4fa?style=for-the-badge&logo=files&logoColor=1e1e2e" />
</p>

<p>
  <a href="#-architecture">Architecture</a> •
  <a href="#-pipeline-flow">Pipeline</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-configuration">Config</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

</div>

---

## 🏭 Overview

**Entry Point Resolver** is a machine learning pipeline composed of **three lightweight MLPs** that, given a warehouse layout in CSV format (exported from CAD), automatically determines the optimal **entry point and entry angle** for each storage line (`Model_Station_Lines`).

The system combines supervised learning with geometric heuristics to produce robust results even on complex, non-orthogonal layouts.

> **Problem:** In warehouse design, every rack or shelf line needs an *entry point* — the end from which forklifts access it and the direction they approach. Determining this manually is tedious and error-prone. This system automates it using geometric context: distances to aisles, cluster orientation, back-to-back pairs, and more.

---

## 📁 Project Structure

```
entry-point-resolver/
│
├── entry_models.py       # Neural architectures + shared geometric utilities
├── training.py           # Preprocessing, pseudo-label generation, training loop
├── infer_only.py         # Loads checkpoints and runs inference on new CSVs
│
└── checkpoints/          # Auto-generated after training
    ├── entry_scorer.pt
    ├── angle_refiner.pt
    └── ambiguity_detector.pt
```

| File | Role |
|---|---|
| `entry_models.py` | Defines the three MLP networks and shared geometric utilities |
| `training.py` | Loads CSVs, preprocesses geometry, generates pseudo-labels, trains models |
| `infer_only.py` | Loads trained checkpoints and runs inference; outputs CSV + optional debug image |

---

## 🧠 Architecture

All three networks share the same [`SmallMLP`](entry_models.py) backbone:

```
Input → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
      → Linear(32) → BatchNorm → ReLU → Dropout(0.3)
      → Output
```

> 💡 Related reading: [Batch Normalization](https://arxiv.org/abs/1502.03167) · [Dropout Regularization](https://jmlr.org/papers/v15/srivastava14a.html) · [MLP fundamentals](https://en.wikipedia.org/wiki/Multilayer_perceptron)

---

### 🥇 `EntryScorer` — *The Core Model*

> **Binary classifier** · `BCEWithLogitsLoss` · 20 input features → 1 logit

Given a candidate endpoint of a line, predicts **how good it is as an entry point**. Its output decides which of the two endpoints is selected.

<details>
<summary><b>📊 Feature Vector (20 dims)</b></summary>

| # | Feature | Description |
|---|---|---|
| 0 | `dist_to_nearest_path` | Distance to nearest suggested path (norm) |
| 1 | `dist_to_boundary` | Distance to nearest boundary (norm) |
| 2 | `dist_to_nearest_btb` | Distance to nearest back-to-back pair (norm) |
| 3 | `is_external_btb_side` | Is this the outer side of a BTB pair? (0/1) |
| 4 | `segment_length` | Length of the segment (norm) |
| 5–6 | `local_mid_x/y` | Midpoint position relative to cluster centroid |
| 7–8 | `local_ep_x/y` | Endpoint position relative to cluster centroid |
| 9 | `cluster_size` | Number of lines in this cluster (norm) |
| 10 | `dist_to_bundle_plane` | Perpendicular distance to bundle central axis |
| 11 | `region_id_norm` | Normalized spatial region ID |
| 12–15 | `sin/cos_cluster_angle`, `sin/cos_entry_angle` | Angles without discontinuity |
| 16 | `angle_matches_cluster` | Does the entry angle match the cluster? (0/1) |
| 17 | `is_endpoint_A` | Is this endpoint A (start)? (0/1) |
| 18 | `path_exists` | Is there a nearby suggested path? (0/1) |
| 19 | `cone_mean_score` | Visibility cone obstruction score (norm) |

</details>

---

### 📐 `AngleRefiner` — *Oblique Angle Correction*

> **Regressor** · `MSELoss` · 8 input features → `delta_theta` (degrees)

Given the chosen entry point and the snapped base angle (multiple of 90°), predicts a **delta in degrees** to refine the final entry angle. Designed to handle non-orthogonal layouts from CAD.

> ⚠️ **Status:** Implementation started but incomplete. Training runs for only 1 epoch by default and inference is currently **disabled** (passes `None` instead of the model). See [Roadmap §2](#2--complete-anglerefiner-integration).

<details>
<summary><b>📊 Feature Vector (8 dims)</b></summary>

| # | Feature | Description |
|---|---|---|
| 0 | `dist_to_nearest_path_in_cone` | Distance to path inside visibility cone (norm) |
| 1–2 | `sin/cos_base_angle` | Entry base angle encoded as sin/cos |
| 3–4 | `sin/cos_path_angle` | Nearest path angle encoded as sin/cos |
| 5 | `angle_diff_entry_path` | Angular difference between entry and path (norm) |
| 6 | `segment_length` | Length of segment (norm) |
| 7 | `path_exists` | Is there a nearby path? (0/1) |

</details>

---

### 🔍 `AmbiguityDetector` — *Manual Review Flagging*

> **Binary classifier** · `BCEWithLogitsLoss` + balanced `pos_weight` · 8 input features → 1 logit

Detects lines where the entry decision is **ambiguous** and should be reviewed manually. The teacher label is generated by comparing the score difference between both endpoints.

Output: `NeedsReview = True` when probability exceeds `AMBIG_THRESHOLD` (default: `0.55`).

<details>
<summary><b>📊 Feature Vector (8 dims)</b></summary>

| # | Feature | Description |
|---|---|---|
| 0 | `score_diff` | \|score_A − score_B\| from EntryScorer (norm) |
| 1 | `max_score` | max(score_A, score_B) (norm) |
| 2 | `dist_to_nearest_path` | (norm) |
| 3 | `dist_to_boundary` | (norm) |
| 4 | `is_btb` | Is line part of a back-to-back pair? (0/1) |
| 5 | `cluster_size` | (norm) |
| 6 | `path_exists` | (0/1) |
| 7 | `angle_matches_cluster` | (0/1) |

</details>

---

## 🔄 Pipeline Flow

```
CSV Input (CAD export)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                  PREPROCESSING                        │
│  1. Layer separation (StationLines / Boundaries /     │
│     SuggestedPaths)                                   │
│  2. Angle normalization → 0/90/180/270°               │
│  3. Segment fusion  (collinear, gap < 30mm)           │
│  4. Clustering  (perp: 200mm, para: 500mm)            │
│  5. B2B pair detection  (gap < 100mm)                 │
│  6. Bundle detection  (parallel row groups)           │
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────┐
│               FEATURE EXTRACTION                      │
│  Per line: evaluate BOTH endpoints (A & B)            │
│  → 20 scoring features + 8 angle + 8 ambiguity feats  │
└───────────────────┬───────────────────────────────────┘
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
   EntryScorer  AngleRefiner  AmbiguityDetector
   (pick side)  (refine θ)    (flag review)
          └─────────┬─────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────┐
│              POST-PROCESSING (deterministic)          │
│  • Block Consistency: cluster leader homogenizes       │
│    direction across aligned neighbors                  │
│  • Bundle Alternation: B2B pairs repel each other,    │
│    pointing outward from the bundle central axis       │
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
         output_entries.csv  [+  debug.png]
```

> 💡 Related concepts: [Spatial Clustering](https://scikit-learn.org/stable/modules/clustering.html) · [Back-to-back racking](https://en.wikipedia.org/wiki/Pallet_racking) · [Geometric heuristics in warehouse design](https://www.sciencedirect.com/topics/engineering/warehouse-design)

---

## 🚀 Usage

### Installation

```bash
pip install torch numpy pandas matplotlib
```

### Training

```bash
# Single layout
python training.py lineas.csv

# Multiple layouts (data is concatenated)
python training.py layout_A.csv layout_B.csv layout_C.csv
```

Checkpoints are saved automatically to `checkpoints/`:
- `entry_scorer.pt`
- `angle_refiner.pt` *(only if > 10 angle samples)*
- `ambiguity_detector.pt` *(only if > 10 samples)*

### Inference

```bash
# Basic (produces output_entries.csv)
python infer_only.py lineas.csv

# Custom output path
python infer_only.py lineas.csv --output resultado.csv

# With debug visualization
python infer_only.py lineas.csv --output resultado.csv --plot

# Custom plot filename
python infer_only.py lineas.csv --plot --plot-file debug_layout_A.png
```

### CSV Format

**Input** — semicolon-delimited (`;`):

| Column | Description |
|---|---|
| `Start Point X / Y` | Start endpoint coordinates (mm) |
| `End Point X / Y` | End endpoint coordinates (mm) |
| `Angle` | Segment angle in degrees (normalized to 0/90/180/270) |
| `Layer` | CAD layer: `model_station_lines`, `model_boundaries`, or `model_suggested_paths` |

**Output**:

| Column | Description |
|---|---|
| `LineID` | Line index in the processed DataFrame |
| `EntryPointX / Y` | Chosen entry point coordinates |
| `EntryAngle` | Entry angle in degrees (0, 90, 180 or 270) |
| `ConfidenceScore` | Confidence score \[0, 1\] |
| `NeedsReview` | `True` if AmbiguityDetector flagged manual review |
| `RuleApplied` | Text describing which post-processing rules were applied |

---

## ⚙️ Configuration

### `training.py`

| Parameter | Default | Description |
|---|---|---|
| `SNAP_DIST_MM` | `30` | Max gap to merge collinear segment fragments |
| `BACK_TO_BACK_GAP_MM` | `100` | Max perpendicular separation for B2B detection |
| `CLUSTER_PERP_TOL` | `200` | Perpendicular tolerance for clustering lines |
| `CLUSTER_PARA_TOL` | `500` | Parallel tolerance for extending a cluster |
| `PATH_REWARD_MM` | `8000` | Max distance to path to receive scoring bonus |
| `AMBIGUITY_HIGH_THRESH` | `1.0` | `score_diff > X` → label 0 (unambiguous) |
| `AMBIGUITY_LOW_THRESH` | `0.3` | `score_diff < X` → label 1 (ambiguous) |
| `BATCH_SIZE` | `256` | Batch size for all models |
| `LR` | `1e-3` | Adam learning rate |
| `EPOCHS_SCORE` | `40` | Max epochs for EntryScorer (early stopping active) |
| `PATIENCE` | `6` | Epochs without val improvement before early stopping |
| `AUG_N_COPIES` | `3` | Data augmentation copies per training sample |

### `infer_only.py`

| Parameter | Default | Description |
|---|---|---|
| `AMBIG_THRESHOLD` | `0.55` | Min probability to flag `NeedsReview = True` |
| `ANGLE_DELTA_CLIP` | `45.0` | Max clip for predicted angle delta (degrees) |
| `CHECKPOINT_DIR` | `'checkpoints'` | Folder where `.pt` files are searched |

---

## 🗺️ Roadmap

> Items are ordered by priority. Contributions welcome!

### 1 · Boundary Line Preprocessing Performance

Currently `dist_point_to_df` iterates over **all** boundary segments per endpoint evaluation — O(n\_lines × n\_boundaries). This dominates compute on large layouts.

- [ ] **Ramer-Douglas-Peucker simplification** on boundary polylines (tolerance: 50–100mm) — [algorithm reference](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm)
- [ ] **Bounding-box pre-filter**: discard segments whose bbox is farther than `NORM_DIST` (10,000mm) before computing exact distance
- [ ] **Spatial index**: build a [`KDTree`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html) or [`STRtree`](https://shapely.readthedocs.io/en/stable/strtree.html) over boundary segment midpoints at startup — O(n) → O(log n)
- [ ] **Pre-compute `obs_pts` / `attr_pts`** densification once per layout (not per inference call)

### 2 · Complete AngleRefiner Integration

The `AngleRefiner` is designed for non-orthogonal CAD layouts but its integration is currently **incomplete**.

- [ ] **Activate in inference**: pass the real model in `run_inference()` instead of `None`; apply `delta_theta` with `clip(±ANGLE_DELTA_CLIP)`
- [ ] **Fix training labels**: `y_angle` is hardcoded to `0.0` when `path_exists=0`; generate real labels from the angular difference between the snapped angle and the geometric angle of the nearest path
- [ ] **Oblique angle detection**: pre-processing stage that flags segments where `angle_raw` differs > 15° from the nearest 90° multiple → `uses_oblique_angle=1` feature for the refiner
- [ ] **Exception handling**: wrap delta application in `try/except`; fallback to snapped base angle with a warning column in the output CSV
- [ ] **More training epochs**: raise `EPOCHS_ANGLE` from `1` to `30–50` once labels are correct

### 3 · Teacher Heuristic Improvements

Pseudo-labels from `training.py` are the direct source of supervision. Better heuristics → better models.

- [ ] **Human-corrected labels**: add support for an optional ground-truth CSV with manually validated entry points; when present, override the heuristic for covered lines
- [ ] **Auto-calibrate ambiguity thresholds**: compute `AMBIGUITY_HIGH_THRESH` and `AMBIGUITY_LOW_THRESH` from percentiles of `score_diff` per dataset instead of hardcoding
- [ ] **Ray-casting collision**: replace the current 1000mm projection with real ray-casting against boundary segments for more accurate direction penalties — [ray-segment intersection](https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection)

### 4 · Code Quality & Robustness

- [ ] **Remove debug block** in `infer_only.py` (marked `# DEBUG TEMPORAL` — perpendicular gap prints and bundle distribution)
- [ ] **Remove duplicate `plot_debug`** function in `infer_only.py` (defined twice, ~lines 379 and 424)
- [ ] **Expose inference params as CLI flags**: `--ambig-threshold`, `--angle-delta-clip`, `--checkpoint-dir`
- [ ] **Input CSV validation**: check for required columns at the top of `load_csv()` with clear error messages instead of cryptic `KeyError`s

### 5 · Training Scalability

- [ ] **Synthetic dataset generation**: programmatically generate regular rack grids with known entry points for pre-training the scorer — see [procedural layout generation](https://arxiv.org/search/?searchtype=all&query=procedural+warehouse+layout)
- [ ] **Fine-tuning / transfer learning**: load existing checkpoints and continue training on new CSVs instead of training from scratch each time
- [ ] **Domain-specific validation metrics**: replace generic accuracy with:
  - `% lines where entry point matches ground truth`
  - `ConfidenceScore distribution` (detect regressions between model versions)

---

## 🔧 Geometric Utilities

Defined in [`entry_models.py`](entry_models.py) and shared across training and inference:

| Function | Description |
|---|---|
| `angle_diff_deg(a, b)` | Minimum angular difference in \[0°, 180°\] |
| `dist_point_to_segment(px,py, ax,ay, bx,by)` | Minimum distance from point P to segment AB |
| `dist_point_to_df(px, py, df)` | Minimum distance from a point to any segment in a DataFrame |
| `cone_mean(px,py, angle, obstacles, attractors)` | Visibility cone obstruction score, weighted by 1/distance |
| `normalize_angle_90(angle)` | Snap to nearest multiple of 90° in {0, 90, 180, 270} |
| `opposite_angle(angle)` | Returns `(angle + 180) % 360` |

**Normalization constants:**

```python
NORM_COORD = 100_000.0   # typical layout coordinates (mm)
NORM_DIST  =  10_000.0   # path/boundary distances (mm)
NORM_ANGLE =    360.0    # angles → [0, 1]
NORM_CONE  =      1e-3   # cone scores (scale 1/mm)
```

---

## 📚 Further Reading

| Topic | Link |
|---|---|
| Binary Cross-Entropy with Logits | [PyTorch BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) |
| Batch Normalization | [Ioffe & Szegedy, 2015](https://arxiv.org/abs/1502.03167) |
| Spatial indexing with KDTree | [scipy.spatial.KDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html) |
| Shapely STRtree | [Shapely docs](https://shapely.readthedocs.io/en/stable/strtree.html) |
| Ramer-Douglas-Peucker | [Wikipedia](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm) |
| Warehouse slotting & layout | [MHIA overview](https://www.mhi.org/fundamentals/storage) |
| Transfer learning in PyTorch | [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) |
| Early stopping best practices | [deeplearning.ai](https://www.deeplearning.ai/ai-notes/regularization/) |

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=24,20,17&height=100&section=footer" width="100%"/>

*Entry Point Resolver — WIP · Handover Document*

</div>
