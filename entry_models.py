# ============================================================
#  entry_models.py
#  Define las tres redes MLP del pipeline de entry points:
#
#  EntryScorer      → dado un extremo, ¿cuán bueno es como entry?
#                     (BCEWithLogits, clasificación binaria)
#
#  AngleRefiner     → dado un entry + ángulo base, refina el ángulo
#                     (MSE, regresión de delta en grados)
#
#  AmbiguityDetector→ ¿la línea necesita revisión manual?
#                     (BCEWithLogits, clasificación binaria)
#
#  Utilidades geométricas compartidas con train / infer.
# ============================================================

import math
import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
#  PLANTILLA MLP
# ─────────────────────────────────────────────

class SmallMLP(nn.Module):
    """
    Arquitectura base: input → 64 → 32 → output  (ReLU + BatchNorm)
    Más estable que el original al agregar BN antes de activación.
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
#  MODELOS
# ─────────────────────────────────────────────

class EntryScorer(nn.Module):
    """
    Puntúa un extremo candidato como entry point.

    Features de entrada (20 dims):
      -- Distancias globales --
      0  dist_to_nearest_path       (norm)
      1  dist_to_boundary           (norm)
      2  dist_to_nearest_btb        (norm)
      3  is_external_btb_side       (0/1)
      -- Geometría del segmento --
      4  segment_length             (norm)
      -- Coordenadas LOCALES relativas al centroide del cluster --
      5  local_mid_x                (norm)   ← cx - cluster_cx
      6  local_mid_y                (norm)   ← cy - cluster_cy
      7  local_ep_x                 (norm)   ← ep_x - cluster_cx
      8  local_ep_y                 (norm)   ← ep_y - cluster_cy
      -- Bundle / cluster context --
      9  cluster_size               (norm)
      10 dist_to_bundle_plane       (norm)   ← distancia perp al eje central del bundle
      11 region_id_norm             (norm)   ← ID de región espacial normalizado
      -- Ángulos (sin/cos, sin discontinuidad) --
      12 sin_cluster_angle
      13 cos_cluster_angle
      14 sin_entry_angle
      15 cos_entry_angle
      -- Flags --
      16 angle_matches_cluster      (0/1)
      17 is_endpoint_A              (0/1)
      18 path_exists                (0/1)
      19 cone_mean_score     (norm)

    Salida: logit escalar (sigmoid → prob)
    """
    FEATURE_DIM = 20

    def __init__(self):
        super().__init__()
        self.mlp = SmallMLP(self.FEATURE_DIM, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # logit


class AngleRefiner(nn.Module):
    """
    Ajusta fino el ángulo de entrada dado el contexto del path más cercano.

    Features de entrada (8 dims — ver ANGLE_FEATURE_DIM):
      0  dist_to_nearest_path_in_cone  (norm)
      1  sin_base_angle                        ← sin del ángulo base de entrada
      2  cos_base_angle                        ← cos del ángulo base de entrada
      3  sin_path_angle                        ← sin del ángulo del path más cercano
      4  cos_path_angle                        ← cos del ángulo del path más cercano
      5  angle_diff_entry_path         (norm)  ← diferencia angular entry vs path
      6  segment_length                (norm)
      7  path_exists                   (0/1)

    Salida: delta_theta en grados (regresión; clip a ±45° en inferencia)
    Nota: la red aprende a NO mover el ángulo cuando path_exists=0.
    """
    FEATURE_DIM = 8

    def __init__(self):
        super().__init__()
        self.mlp = SmallMLP(self.FEATURE_DIM, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # delta_theta (escalar por muestra)


class AmbiguityDetector(nn.Module):
    """
    Detecta si una línea necesita revisión manual.

    Features de entrada (8 dims — ver AMBIGUITY_FEATURE_DIM):
      0  score_diff                (norm)   ← |score_A - score_B| del EntryScorer
      1  max_score                 (norm)   ← max(score_A, score_B)
      2  dist_to_nearest_path      (norm)
      3  dist_to_boundary          (norm)
      4  is_btb                    (0/1)
      5  cluster_size              (norm)
      6  path_exists               (0/1)
      7  angle_matches_cluster     (0/1)

    Salida: logit (sigmoid → prob de ambigüedad)
    Label teacher: 1 si score_diff < umbral bajo, 0 si score_diff > umbral alto.
    """
    FEATURE_DIM = 8

    def __init__(self):
        super().__init__()
        self.mlp = SmallMLP(self.FEATURE_DIM, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # logit


# ─────────────────────────────────────────────
#  UTILIDADES GEOMÉTRICAS
# ─────────────────────────────────────────────

def angle_diff_deg(a: float, b: float) -> float:
    """Diferencia angular mínima en [0, 180]."""
    return float(abs(((a - b + 180.0) % 360.0) - 180.0))


def dist_point_to_segment(px, py, ax, ay, bx, by) -> float:
    """Distancia mínima de punto P al segmento AB."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def dist_point_to_df(px, py, df) -> float:
    """Distancia mínima de un punto a cualquier segmento de un DataFrame."""
    if df is None or len(df) == 0:
        return float("inf")
    best = float("inf")
    for _, r in df.iterrows():
        d = dist_point_to_segment(px, py, r["sx"], r["sy"], r["ex"], r["ey"])
        if d < best:
            best = d
    return best


def cone_mean(px, py, angle_deg_val, obstacle_pts,attractor_pts,
                     half_angle: float = 45.0, max_dist: float = 2500.0) -> float:
    """
    Score de obstrucción en el cono de entrada (0 = libre).
    Atracción por puntos de interés (attractor_pts).
    Ponderado por 1/distancia para obstáculos más cercanos.
    """
    score = 0.0
    for ox, oy in obstacle_pts:
        dx, dy = ox - px, oy - py
        d = math.hypot(dx, dy)
        if d < 1e-6 or d > max_dist: continue
        obs_ang = math.degrees(math.atan2(dy, dx)) % 360.0
        diff = abs(((obs_ang - angle_deg_val + 180.0) % 360.0) - 180.0)
        if diff <= half_angle:
            score -= 1.0 / d
    for ox, oy in (attractor_pts or []):
        dx, dy = ox - px, oy - py
        d = math.hypot(dx, dy)
        if d < 1e-6 or d > max_dist: continue
        obs_ang = math.degrees(math.atan2(dy, dx)) % 360.0
        diff = abs(((obs_ang - angle_deg_val + 180.0) % 360.0) - 180.0)
        if diff <= half_angle:
            score += 1.0 / d
    return score


def normalize_angle_90(angle: float) -> int:
    """Snappea al múltiplo de 90 más cercano en {0, 90, 180, 270}."""
    a = float(angle) % 360.0
    return int(min([0, 90, 180, 270, 360], key=lambda c: abs(a - c)) % 360)


def opposite_angle(angle: int) -> int:
    return (angle + 180) % 360


# Constantes de normalización (mm / grados)
NORM_COORD = 100_000.0   # coordenadas típicas del layout
NORM_DIST  = 10_000.0    # distancias a paths/boundaries
NORM_ANGLE = 360.0       # ángulos → [0, 1]
NORM_CONE  = 1e-3        # score de cono (escala 1/mm)
