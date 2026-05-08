# ============================================================
#  train.py
#  1. Carga CSV y normaliza (capa Model_Station_Lines solamente)
#  2. Fusión de segmentos sueltos (geométrica)
#  3. Clustering y detección back-to-back
#  4. Teacher heurístico → genera pseudo-labels para las 3 redes
#  5. Entrena EntryScorer, AngleRefiner, AmbiguityDetector
#  6. Guarda checkpoints/
#
#  Uso:
#    python train.py lineas.csv
#    python train.py lineas.csv lineas2.csv   ← múltiples CSVs
# ============================================================

import os
import sys
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from entry_models import (
    EntryScorer, AngleRefiner, AmbiguityDetector,
    dist_point_to_segment, dist_point_to_df, cone_mean,
    normalize_angle_90, opposite_angle, angle_diff_deg,
    NORM_COORD, NORM_DIST, NORM_ANGLE, NORM_CONE,
)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
SNAP_DIST_MM         = 30       # fusión de segmentos sueltos
BACK_TO_BACK_GAP_MM  = 100      # separación perp máxima B2B
CLUSTER_PERP_TOL     = 200
CLUSTER_PARA_TOL     = 500
PATH_REWARD_MM       = 8000
BOUNDARY_PENALTY_MM  = 500
CONE_MAX_DIST_MM = 2500
 
AMBIGUITY_HIGH_THRESH = 1.0     # score_diff > X → label 0 (no ambiguo)
AMBIGUITY_LOW_THRESH  = 0.3     # score_diff < X → label 1 (ambiguo)

 
BATCH_SIZE   = 256
LR           = 1e-3
EPOCHS_SCORE = 40    # early stopping corta antes si no mejora
EPOCHS_ANGLE = 1
EPOCHS_AMBIG = 3
PATIENCE     = 6     # épocas sin mejora en val antes de parar
WEIGHT_DECAY = 1e-4   # subido de 1e-5: dataset pequeño con pseudo-labels ruidosos
VAL_SPLIT    = 0.15   # 15% validation, 15% test, 70% train
TEST_SPLIT   = 0.15
SEED         = 42

# Augmentation (solo sobre muestras de entrenamiento, nunca sobre val/test)
AUG_NOISE_MM  = 50.0   # ruido gaussiano en features de distancia — simula imprecision CAD
AUG_FLIP_PROB = 0.5    # prob de reflejar por eje (H y V independientes)
AUG_N_COPIES  = 3      # copias aumentadas por muestra original

LAYER_STATION = "model_station_lines"
LAYER_BOUNDARY = "model_boundaries"
LAYER_PATH = "model_suggested_paths"


# ─────────────────────────────────────────────
#  CARGA Y NORMALIZACIÓN
# ─────────────────────────────────────────────

def load_csv(path: str):
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip().lower() for c in df.columns]

    rename = {
        "start point x": "sx", "start point y": "sy",
        "end point x":   "ex", "end point y":   "ey",
        "angle": "angle_raw", "layer": "layer",
    }
    df.rename(columns=rename, inplace=True)
    df["layer_key"] = df["layer"].str.strip().str.lower()

    # Solo normalizar ángulo para Model_Station_Lines
    station_mask = df["layer_key"] == LAYER_STATION
    df.loc[station_mask, "angle_norm"] = (
        df.loc[station_mask, "angle_raw"].apply(normalize_angle_90)
    )

    wl  = df[station_mask].copy().reset_index(drop=True)
    bnd = df[df["layer_key"] == LAYER_BOUNDARY].copy().reset_index(drop=True)
    pth = df[df["layer_key"] == LAYER_PATH].copy().reset_index(drop=True)

    # Canonicalizar: start = extremo con menor coordenada en eje principal
    def _horiz(r): return abs(r["ex"] - r["sx"]) >= abs(r["ey"] - r["sy"])
    wl["horiz"] = wl.apply(_horiz, axis=1)

    def _canon(r):
        if r["horiz"] and r["sx"] > r["ex"]:
            r["sx"], r["ex"] = r["ex"], r["sx"]
            r["sy"], r["ey"] = r["ey"], r["sy"]
        elif not r["horiz"] and r["sy"] > r["ey"]:
            r["sx"], r["ex"] = r["ex"], r["sx"]
            r["sy"], r["ey"] = r["ey"], r["sy"]
        return r
    wl = wl.apply(_canon, axis=1)

    # Corregir angle_norm por geometría real (detecta copy-paste)
    def _fix_angle(r):
        geo_axis = 0 if r["horiz"] else 90
        decl_axis = int(r["angle_norm"]) % 180
        if decl_axis != geo_axis:
            return geo_axis
        return int(r["angle_norm"])
    wl["angle_norm"] = wl.apply(_fix_angle, axis=1)

    wl["length"] = wl.apply(
        lambda r: math.hypot(r["ex"] - r["sx"], r["ey"] - r["sy"]), axis=1
    )
    wl["cx"] = (wl["sx"] + wl["ex"]) / 2.0
    wl["cy"] = (wl["sy"] + wl["ey"]) / 2.0
    wl["line_id"] = wl.index

    return wl, bnd, pth


# ─────────────────────────────────────────────
#  FUSIÓN DE SEGMENTOS SUELTOS
# ─────────────────────────────────────────────

def fuse_loose_segments(wl: pd.DataFrame) -> pd.DataFrame:
    df = wl.copy()
    changed = True
    while changed:
        changed = False
        used = [False] * len(df)
        rows = df.to_dict("records")
        merged = []
        for i, ri in enumerate(rows):
            if used[i]:
                continue
            for j in range(i + 1, len(rows)):
                if used[j]:
                    continue
                rj = rows[j]
                if ri["horiz"] != rj["horiz"]:
                    continue
                if ri["horiz"]:
                    if abs(ri["sy"] - rj["sy"]) > SNAP_DIST_MM:
                        continue
                    gap = min(abs(ri["ex"] - rj["sx"]), abs(rj["ex"] - ri["sx"]))
                else:
                    if abs(ri["sx"] - rj["sx"]) > SNAP_DIST_MM:
                        continue
                    gap = min(abs(ri["ey"] - rj["sy"]), abs(rj["ey"] - ri["sy"]))
                if gap > SNAP_DIST_MM:
                    continue
                nr = ri.copy()
                if ri["horiz"]:
                    nr["sx"] = min(ri["sx"], rj["sx"])
                    nr["ex"] = max(ri["ex"], rj["ex"])
                    nr["sy"] = nr["ey"] = (ri["sy"] + rj["sy"]) / 2.0
                else:
                    nr["sy"] = min(ri["sy"], rj["sy"])
                    nr["ey"] = max(ri["ey"], rj["ey"])
                    nr["sx"] = nr["ex"] = (ri["sx"] + rj["sx"]) / 2.0
                nr["cx"] = (nr["sx"] + nr["ex"]) / 2.0
                nr["cy"] = (nr["sy"] + nr["ey"]) / 2.0
                nr["length"] = math.hypot(nr["ex"] - nr["sx"], nr["ey"] - nr["sy"])
                merged.append(nr)
                used[i] = used[j] = True
                changed = True
                break
            if not used[i]:
                merged.append(ri)
        df = pd.DataFrame(merged).reset_index(drop=True)
        df["line_id"] = df.index
    return df


# ─────────────────────────────────────────────
#  CLUSTERING Y BACK-TO-BACK
# ─────────────────────────────────────────────

def build_clusters(wl: pd.DataFrame) -> pd.DataFrame:
    wl = wl.copy()
    wl["cluster_id"]  = -1
    wl["btb_partner"] = -1
    cluster_id = 0

    for orient in [True, False]:
        sub = wl[wl["horiz"] == orient]
        if len(sub) == 0:
            continue
        perp = "sy" if orient else "sx"
        lo   = "sx" if orient else "sy"
        hi   = "ex" if orient else "ey"

        sub_s = sub.sort_values(perp)
        pvals = sub_s[perp].values
        idxs  = sub_s.index.tolist()

        groups, cur = [], [idxs[0]]
        for k in range(1, len(idxs)):
            if abs(pvals[k] - pvals[k - 1]) <= CLUSTER_PERP_TOL:
                cur.append(idxs[k])
            else:
                groups.append(cur); cur = [idxs[k]]
        groups.append(cur)

        for grp in groups:
            gdf  = sub.loc[grp].sort_values(lo)
            gidx = gdf.index.tolist()
            lo_v = gdf[lo].values
            hi_v = gdf[hi].values
            subs, cur_max = [[gidx[0]]], hi_v[0]
            for k in range(1, len(gidx)):
                if lo_v[k] <= cur_max + CLUSTER_PARA_TOL:
                    subs[-1].append(gidx[k])
                    cur_max = max(cur_max, hi_v[k])
                else:
                    subs.append([gidx[k]]); cur_max = hi_v[k]
            for sg in subs:
                wl.loc[sg, "cluster_id"] = cluster_id
                cluster_id += 1

    # Back-to-back
    for orient in [True, False]:
        sub  = wl[wl["horiz"] == orient]
        idxs = sub.index.tolist()
        perp = "sy" if orient else "sx"
        lo   = "sx" if orient else "sy"
        hi   = "ex" if orient else "ey"
        for ii in range(len(idxs)):
            i = idxs[ii]
            if wl.at[i, "btb_partner"] != -1:
                continue
            best_j, best_gap = -1, float("inf")
            for jj in range(ii + 1, len(idxs)):
                j   = idxs[jj]
                gap = abs(sub.at[i, perp] - sub.at[j, perp])
                if gap > BACK_TO_BACK_GAP_MM:
                    continue
                ovlp = min(sub.at[i, hi], sub.at[j, hi]) - max(sub.at[i, lo], sub.at[j, lo])
                if ovlp > 0 and gap < best_gap:
                    best_gap, best_j = gap, j
            if best_j != -1:
                wl.at[i, "btb_partner"]      = best_j
                wl.at[best_j, "btb_partner"] = i
    return wl


# ─────────────────────────────────────────────
#  SCENE PARSING — BUNDLE DETECTION
# ─────────────────────────────────────────────

# Tolerancias para bundle detection
BUNDLE_ROW_TOL   = 30     # max desviación perpendicular para considerar misma fila (mm)
BUNDLE_ROW_GAP   = 500   # max gap B2B entre filas para ser el mismo bundle (mm)
                          # debe ser menor que un pasillo real
BUNDLE_AISLE_MIN = 2000   # gap perpendicular mínimo que define un pasillo (no bundle)

def detect_bundles(wl: pd.DataFrame) -> pd.DataFrame:
    """
    Detección de bundles en dos niveles:
 
    Nivel 1 — Filas (rows):
        Líneas con la misma coordenada perpendicular (±BUNDLE_ROW_TOL) y que
        solapan en el eje paralelo → forman una fila.
        Cada fila tiene una coordenada perpendicular media (row_axis).
 
    Nivel 2 — Bundles:
        Filas adyacentes cuyo gap perpendicular ≤ BUNDLE_ROW_GAP forman un bundle.
        Un gap > BUNDLE_AISLE_MIN indica un pasillo y separa bundles.
 
    Columnas añadidas a wl:
        row_id       : ID de fila dentro del bundle (0 = primera fila, 1 = segunda, etc.)
        bundle_id    : ID del bundle
        bundle_row   : posición ordinal de la fila dentro del bundle (0, 1, 2, ...)
        bundle_size  : número de filas en el bundle
        bundle_cx/cy : centroide del bundle
        dist_to_plane: distancia perpendicular al eje central del bundle
        region_id    : cuadrante espacial (0–3) relativo al centroide global
    """

    wl = wl.copy()
    wl["row_id"]       = -1
    wl["bundle_id"]    = -1
    wl["bundle_row"]   = -1
    wl["bundle_size"]  = 1
    wl["bundle_cx"]    = wl["cx"]
    wl["bundle_cy"]    = wl["cy"]
    wl["dist_to_plane"] = 0.0
 
    row_id    = 0
    bundle_id = 0

    for orient in [True, False]:
        sub = wl[wl["horiz"] == orient]
        if len(sub) == 0:
            continue
        perp = "sy" if orient else "sx"
        lo   = "sx" if orient else "sy"
        hi   = "ex" if orient else "ey"
 
        # ── Nivel 1: agrupar en filas por coordenada perpendicular ──
        sub_s  = sub.sort_values(perp)
        pvals  = sub_s[perp].values
        idxs   = sub_s.index.tolist()
 
        # Agrupar líneas con perp similar → misma fila
        raw_rows, cur = [], [idxs[0]]
        for k in range(1, len(idxs)):
            if abs(pvals[k] - pvals[k - 1]) <= BUNDLE_ROW_TOL:
                cur.append(idxs[k])
            else:
                raw_rows.append(cur)
                cur = [idxs[k]]
        raw_rows.append(cur)
 
        # Dentro de cada fila, separar por solapamiento paralelo
        # (misma Y pero diferente segmento del pasillo → distintas filas)
        rows = []   # lista de (axis_val, [indices])
        for rgrp in raw_rows:
            gdf  = sub.loc[rgrp].sort_values(lo)
            gidx = gdf.index.tolist()
            lo_v = gdf[lo].values
            hi_v = gdf[hi].values
            axis = float(gdf[perp].mean())
 
            cur_row, cur_max = [gidx[0]], hi_v[0]
            for k in range(1, len(gidx)):
                if lo_v[k] <= cur_max + CLUSTER_PARA_TOL:
                    cur_row.append(gidx[k])
                    cur_max = max(cur_max, hi_v[k])
                else:
                    rows.append((axis, cur_row))
                    cur_row  = [gidx[k]]
                    cur_max  = hi_v[k]
                    axis     = float(sub.loc[gidx[k], perp])
            rows.append((axis, cur_row))
 
        # Asignar row_id a cada fila
        for axis, ridxs in rows:
            wl.loc[ridxs, "row_id"] = row_id
            row_id += 1
 
        # ── Nivel 2: agrupar filas adyacentes en bundles ──
        # Ordenar filas por su axis (coordenada perpendicular media)
        rows_sorted = sorted(rows, key=lambda r: r[0])
 
        cur_bundle = [rows_sorted[0]]
        for k in range(1, len(rows_sorted)):
            gap = abs(rows_sorted[k][0] - rows_sorted[k - 1][0])
            if gap <= BUNDLE_ROW_GAP:
                cur_bundle.append(rows_sorted[k])
            else:
                # Cerrar bundle actual y asignar
                _assign_bundle(wl, cur_bundle, bundle_id, perp)
                bundle_id += 1
                cur_bundle = [rows_sorted[k]]
        # Último bundle
        _assign_bundle(wl, cur_bundle, bundle_id, perp)
        bundle_id += 1
 
    # ── region_id: cuadrante espacial relativo al centroide global ──
    gcx = wl["cx"].mean()
    gcy = wl["cy"].mean()
    wl["region_id"] = (
        (wl["cx"] >= gcx).astype(int) * 2 +
        (wl["cy"] >= gcy).astype(int)
    )
 
    n_bundles = wl["bundle_id"].nunique()
    n_rows    = wl["row_id"].nunique()
    print(f"  Filas detectadas   : {n_rows}")
    print(f"  Bundles detectados : {n_bundles}")
    return wl

def _assign_bundle(wl: pd.DataFrame, bundle_rows: list, bundle_id: int, perp: str):
    """
    Asigna bundle_id, bundle_row, bundle_size, bundle_cx/cy y dist_to_plane
    a todas las líneas de un bundle dado su lista de filas.
    """
    all_idxs = [idx for _, ridxs in bundle_rows for idx in ridxs]
    n_rows   = len(bundle_rows)
 
    b_cx   = float(wl.loc[all_idxs, "cx"].mean())
    b_cy   = float(wl.loc[all_idxs, "cy"].mean())
    b_axis = float(wl.loc[all_idxs, perp].mean())
 
    wl.loc[all_idxs, "bundle_id"]   = bundle_id
    wl.loc[all_idxs, "bundle_size"] = n_rows
    wl.loc[all_idxs, "bundle_cx"]   = b_cx
    wl.loc[all_idxs, "bundle_cy"]   = b_cy
 
    for row_rank, (_, ridxs) in enumerate(
            sorted(bundle_rows, key=lambda r: float(wl.loc[r[1][0], perp]))):
        wl.loc[ridxs, "bundle_row"]    = row_rank
        wl.loc[ridxs, "dist_to_plane"] = abs(
            float(wl.loc[ridxs[0], perp]) - b_axis
        )

# ─────────────────────────────────────────────
#  TEACHER HEURÍSTICO → PSEUDO-LABELS
# ─────────────────────────────────────────────

def _endpoint_score(ex, ey, angle_cand, row, bnd, pth, wl, filtered_obs, attr_pts, cone_max_dist):
    """Heurística de puntuación — misma lógica que el resolver geométrico."""
    score = 0.0

    d_path = dist_point_to_df(ex, ey, pth)
    if d_path < PATH_REWARD_MM:
        # Aumentamos el peso (de 5.0 a 15.0) para que cuando HAYA un pasillo, 
        # la confianza sea altísima y ese sea el "Líder" del bloque.
        score += (1.0 - (d_path / PATH_REWARD_MM)) * 15.0

    # 1. Distancia base al boundary
    d_bnd = dist_point_to_df(ex, ey, bnd)
    
    # 2. VALIDACIÓN DE DIRECCIÓN (Vector de colisión)
    # Proyectamos un punto virtual 1 metro adelante en la dirección de la flecha
    proj_x = ex + 1000 * math.cos(math.radians(angle_cand))
    proj_y = ey + 1000 * math.sin(math.radians(angle_cand))
    
    # Si el punto proyectado está MÁS CERCA de la pared que el punto original,
    # significa que la flecha está APUNTANDO hacia el boundary.
    d_bnd_proj = dist_point_to_df(proj_x, proj_y, bnd)
    
    if d_bnd_proj < d_bnd:
        # Si apunto hacia el boundary, penalizo proporcionalmente a la cercanía
        # pero con un castigo base para que no sea la opción ganadora
        penalty_factor = (2000 - d_bnd) / 2000 if d_bnd < 2000 else 0.1
        score -= 15.0 * max(0.1, penalty_factor)

    pid = int(row["btb_partner"])
    if pid != -1 and pid < len(wl):
        partner = wl.loc[pid]
        d_p   = math.hypot(ex - partner["cx"], ey - partner["cy"])
        ox    = row["ex"] if ex == row["sx"] else row["sx"]
        oy    = row["ey"] if ey == row["sy"] else row["sy"]
        d_o   = math.hypot(ox - partner["cx"], oy - partner["cy"])
        score += -1.5 if d_p < d_o else 0.5

    cone_s = cone_mean(ex, ey, angle_cand, filtered_obs, attractor_pts=attr_pts, max_dist=cone_max_dist)
    score -= cone_s * NORM_CONE * 0.5

    return score


def _cluster_dominant_angle(wl, cluster_id):
    sub = wl[wl["cluster_id"] == cluster_id]
    if len(sub) == 0:
        return 0
    return int(sub["angle_norm"].mode()[0])

def build_datasets(wl: pd.DataFrame, bnd: pd.DataFrame, pth: pd.DataFrame):
    """
    Genera muestras para las 3 redes.
    Devuelve (X_score, y_score, X_angle, y_angle, X_ambig, y_ambig).
    """
    path_exists = float(len(pth) > 0)

    # Densificar obstáculos y puntos de atracción para que el cone_mean sea más efectivo.
    def _densify(sx, sy, ex, ey, step=300):
        pts = [(sx, sy), (ex, ey)]
        n = int(math.hypot(ex - sx, ey - sy) / step)
        for k in range(1, n):
            t = k / n
            pts.append((sx + t*(ex-sx), sy + t*(ey-sy)))
        return pts

    # obs_pts: lista de puntos de obstáculos (boundaries + walls) para el cone_mean
    obs_pts = []
    for _, r in wl.iterrows():
        obs_pts.extend(_densify(r["sx"], r["sy"], r["ex"], r["ey"]))
    if len(bnd) > 0:
        for _, r in bnd.iterrows():
            obs_pts.extend(_densify(r["sx"], r["sy"], r["ex"], r["ey"]))

    # attr_pts: lista de puntos de atracción (suggested paths) para el cone_mean
    attr_pts = []
    for _, r in pth.iterrows():
        attr_pts.extend(_densify(r["sx"], r["sy"], r["ex"], r["ey"]))


    X_score, y_score = [], []
    X_angle, y_angle = [], []
    X_ambig, y_ambig = [], []

    cluster_sizes = wl.groupby("cluster_id").size().to_dict()

    # Centroides de cada cluster para coordenadas locales
    cluster_centroids = wl.groupby("cluster_id")[["cx", "cy"]].mean().to_dict("index")

    for idx, row in wl.iterrows():
        a_norm   = int(row["angle_norm"])
        a_opp    = opposite_angle(a_norm)
        horiz    = row["horiz"]
        cid      = int(row["cluster_id"])
        c_size   = cluster_sizes.get(cid, 1)
        dom_ang  = _cluster_dominant_angle(wl, cid)
        is_btb   = int(row["btb_partner"] != -1)
        centroid = cluster_centroids.get(cid, {"cx": row["cx"], "cy": row["cy"]})
        cluster_cx = centroid["cx"]
        cluster_cy = centroid["cy"]

        # Extremos y ángulos candidatos
        end_A = (row["sx"], row["sy"])
        end_B = (row["ex"], row["ey"])
        ang_A = 180 if horiz else 270   # apunta hacia afuera desde A
        ang_B = 0   if horiz else 90    # apunta hacia afuera desde B
        # Restringir a {a_norm, a_opp}
        valid = {a_norm, a_opp}
        ang_A = min(valid, key=lambda v: angle_diff_deg(v, ang_A))
        ang_B = min(valid, key=lambda v: angle_diff_deg(v, ang_B))

        # Filtrar puntos de obstáculos para el cone_mean: excluir los que están demasiado cerca del extremo candidato
        self_pts = {(row["sx"], row["sy"]), (row["ex"], row["ey"])}
        filtered_obs_A = [(ox, oy) for ox, oy in obs_pts
                          if (ox, oy) not in self_pts
                          and math.hypot(ox - end_A[0], oy - end_A[1]) > 50.0]
        filtered_obs_B = [(ox, oy) for ox, oy in obs_pts
                          if (ox, oy) not in self_pts
                          and math.hypot(ox - end_B[0], oy - end_B[1]) > 50.0]

        # Las dos llamadas en el loop:
        sc_A = _endpoint_score(*end_A, ang_A, row, bnd, pth, wl, filtered_obs_A, attr_pts, CONE_MAX_DIST_MM)
        sc_B = _endpoint_score(*end_B, ang_B, row, bnd, pth, wl, filtered_obs_B, attr_pts, CONE_MAX_DIST_MM)

        d_path_A = dist_point_to_df(*end_A, pth)
        d_path_B = dist_point_to_df(*end_B, pth)
        d_bnd_A  = dist_point_to_df(*end_A, bnd)
        d_bnd_B  = dist_point_to_df(*end_B, bnd)

        pid = int(row["btb_partner"])
        if pid != -1 and pid < len(wl):
            partner = wl.loc[pid]
            d_btb_A = math.hypot(end_A[0] - partner["cx"], end_A[1] - partner["cy"])
            d_btb_B = math.hypot(end_B[0] - partner["cx"], end_B[1] - partner["cy"])
            ext_A   = float(d_btb_A > d_btb_B)
            ext_B   = float(d_btb_B > d_btb_A)
        else:
            d_btb_A = d_btb_B = float("inf")
            ext_A = ext_B = 0.0

        cone_A = cone_mean(*end_A, ang_A, filtered_obs_A, attractor_pts=attr_pts)
        cone_B = cone_mean(*end_B, ang_B, filtered_obs_B, attractor_pts=attr_pts)

        ang_matches = float(a_norm == dom_ang or a_opp == dom_ang)

        def _score_feat(ep_x, ep_y, ang_c, d_path, d_bnd, d_btb, ext, cone_s, is_A):
            # Coordenadas LOCALES relativas al centroide del cluster
            # El modelo aprende geometria local, no posicion absoluta
            local_mid_x = (row["cx"] - cluster_cx) / NORM_DIST
            local_mid_y = (row["cy"] - cluster_cy) / NORM_DIST
            local_ep_x  = (ep_x      - cluster_cx) / NORM_DIST
            local_ep_y  = (ep_y      - cluster_cy) / NORM_DIST
            d_plane     = float(row.get("dist_to_plane", 0.0)) / NORM_DIST
            region      = float(row.get("region_id", 0)) / 3.0
            return [
                min(d_path, NORM_DIST) / NORM_DIST,
                min(d_bnd,  NORM_DIST) / NORM_DIST,
                min(d_btb,  NORM_DIST) / NORM_DIST,
                ext,
                min(row["length"], NORM_COORD) / NORM_COORD,
                local_mid_x,
                local_mid_y,
                local_ep_x,
                local_ep_y,
                min(c_size, 50) / 50.0,
                min(d_plane, 1.0),
                region,
                math.sin(math.radians(dom_ang)),
                math.cos(math.radians(dom_ang)),
                math.sin(math.radians(ang_c)),
                math.cos(math.radians(ang_c)),
                ang_matches,
                float(is_A),
                path_exists,
                math.tanh(cone_s * 5000.0),
            ]

        feat_A = _score_feat(*end_A, ang_A, d_path_A, d_bnd_A, d_btb_A, ext_A, cone_A, True)
        feat_B = _score_feat(*end_B, ang_B, d_path_B, d_bnd_B, d_btb_B, ext_B, cone_B, False)

        # Label: 1 = el extremo correcto según heurística
        label_A = float(sc_A >= sc_B)
        label_B = float(sc_B >  sc_A)

        X_score.append(feat_A); y_score.append(label_A)
        X_score.append(feat_B); y_score.append(label_B)

        # ── AngleRefiner: Refuerzo Positivo "Interruptor" ──
        chosen_end = end_A if sc_A >= sc_B else end_B
        chosen_ang = ang_A if sc_A >= sc_B else ang_B
        cx, cy     = chosen_end

        d_path_ch = dist_point_to_df(cx, cy, pth)
        if len(pth) > 0 and d_path_ch < PATH_REWARD_MM:
            # Calcular ángulo del path más cercano
            best_pang = 0.0
            best_pd   = float("inf")
            for _, pr in pth.iterrows():
                d = dist_point_to_segment(cx, cy, pr["sx"], pr["sy"], pr["ex"], pr["ey"])
                if d < best_pd:
                    best_pd   = d
                    dx        = pr["ex"] - pr["sx"]
                    dy        = pr["ey"] - pr["sy"]
                    best_pang = math.degrees(math.atan2(dy, dx)) % 360.0

            # LÓGICA DE INTERRUPTOR: 
            # ¿El path está más cerca del ángulo elegido o de su opuesto?
            current_diff = angle_diff_deg(chosen_ang, best_pang)
            opp_ang = (chosen_ang + 180) % 360
            opp_diff = angle_diff_deg(opp_ang, best_pang)

            # Si el lado opuesto es claramente mejor para alcanzar el path
            # el target es 180. Si no, es 0.
            delta_target = 180.0 if opp_diff < (current_diff - 45) else 0.0

            feat_ang = [
                min(d_path_ch, NORM_DIST) / NORM_DIST,
                math.sin(math.radians(chosen_ang)),
                math.cos(math.radians(chosen_ang)),
                math.sin(math.radians(best_pang)),
                math.cos(math.radians(best_pang)),
                angle_diff_deg(chosen_ang, best_pang) / 180.0,
                min(row["length"], NORM_COORD) / NORM_COORD,
                path_exists,
            ]
            X_angle.append(feat_ang)
            y_angle.append(delta_target)

        # ── AmbiguityDetector ──
        score_diff = abs(sc_A - sc_B)
        max_score  = max(abs(sc_A), abs(sc_B))
        if score_diff < AMBIGUITY_LOW_THRESH:
            label_ambig = 1.0
        elif score_diff > AMBIGUITY_HIGH_THRESH:
            label_ambig = 0.0
        else:
            continue   # zona gris → no incluir en el dataset de ambigüedad

        feat_ambig = [
            min(score_diff, 3.0) / 3.0,
            min(max_score,  3.0) / 3.0,
            min(min(d_path_A, d_path_B), NORM_DIST) / NORM_DIST,
            min(min(d_bnd_A,  d_bnd_B),  NORM_DIST) / NORM_DIST,
            float(is_btb),
            min(c_size, 50) / 50.0,
            path_exists,
            ang_matches,
        ]
        X_ambig.append(feat_ambig)
        y_ambig.append(label_ambig)

    return (
        np.array(X_score, dtype=np.float32), np.array(y_score, dtype=np.float32),
        np.array(X_angle, dtype=np.float32), np.array(y_angle, dtype=np.float32),
        np.array(X_ambig, dtype=np.float32), np.array(y_ambig, dtype=np.float32),
    )


# ─────────────────────────────────────────────
#  AUGMENTATION
# ─────────────────────────────────────────────

def augment_samples(X: np.ndarray, y: np.ndarray,
                    n_copies: int = AUG_N_COPIES,
                    noise_mm: float = AUG_NOISE_MM,
                    flip_prob: float = AUG_FLIP_PROB,
                    seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera copias aumentadas de las muestras de entrenamiento.
 
    Estrategias aplicadas (seguras para este dominio):
 
    1. Ruido gaussiano en features de distancia (cols 0-2):
       Simula imprecision en coordenadas del dibujo CAD (±noise_mm).
       No toca features categoricos ni angulos.
 
    2. Reflejo de ejes — SOLO para EntryScorer (20 features):
       - Flip horizontal: invierte sin_entry_angle (col 14), toggle is_A (col 17).
       - Flip vertical:   invierte sin_cluster_angle (col 12), sin_entry_angle (col 14).
       Solo se aplica si X tiene >= 18 columnas para no romper
       AngleRefiner (8 features) ni AmbiguityDetector (8 features).
 
    NO se rotan coordenadas (las lineas son estrictamente ortogonales).
    NO se escala (las distancias absolutas son features criticos).
    """
    rng = np.random.default_rng(seed)
    Xaug, yaug = [X], [y]
    n_feat = X.shape[1]
    norm_dist_scale = noise_mm / NORM_DIST
 
    for _ in range(n_copies):
        Xc = X.copy()
 
        # 1. Ruido en features de distancia (cols 0, 1, 2) — aplica a todos
        noise = rng.normal(0.0, norm_dist_scale, size=(len(Xc), 3))
        Xc[:, :3] = np.clip(Xc[:, :3] + noise, 0.0, 1.0)
 
        # 2. Flips — solo para EntryScorer que tiene 20 features
        if n_feat >= 18:
            # col 14 = sin_entry_angle, col 17 = is_endpoint_A
            # col 12 = sin_cluster_angle
            flip_h = rng.random(len(Xc)) < flip_prob
            Xc[flip_h, 14] *= -1                      # sin_entry_angle
            Xc[flip_h, 17] = 1.0 - Xc[flip_h, 17]    # is_endpoint_A
 
            flip_v = rng.random(len(Xc)) < flip_prob
            Xc[flip_v, 12] *= -1   # sin_cluster_angle
            Xc[flip_v, 14] *= -1   # sin_entry_angle
 
        Xaug.append(Xc)
        yaug.append(y.copy())
 
    return np.concatenate(Xaug), np.concatenate(yaug)
 


# ─────────────────────────────────────────────
#  DATASET GENERICO
# ─────────────────────────────────────────────

class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def split_train_val_test(X: np.ndarray, y: np.ndarray,
                         val_frac: float = VAL_SPLIT,
                         test_frac: float = TEST_SPLIT,
                         seed: int = SEED):
    """
    Split 70/15/15. Augmentation se aplica SOLO sobre el train set.
    Devuelve (ds_train_aug, ds_val, ds_test).
    """
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(X))
    n_te = max(1, int(len(X) * test_frac))
    n_va = max(1, int(len(X) * val_frac))

    idx_te  = idx[:n_te]
    idx_va  = idx[n_te:n_te + n_va]
    idx_tr  = idx[n_te + n_va:]

    X_te, y_te = X[idx_te], y[idx_te]
    X_va, y_va = X[idx_va], y[idx_va]
    X_tr, y_tr = X[idx_tr], y[idx_tr]

    # Augmentar solo el train set
    X_tr_aug, y_tr_aug = augment_samples(X_tr, y_tr)

    return (
        ArrayDataset(X_tr_aug, y_tr_aug),
        ArrayDataset(X_va,     y_va),
        ArrayDataset(X_te,     y_te),
    )


# ─────────────────────────────────────────────
#  ENTRENAMIENTO GENERICO
# ─────────────────────────────────────────────

def _train_loop(model, dl_tr, dl_va, loss_fn, optimizer, epochs: int, name: str):
    device    = next(model.parameters()).device
    best_va   = float("inf")
    best_state = None
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in dl_tr:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb).squeeze(1)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(Xb)
        tr_loss /= len(dl_tr.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for Xb, yb in dl_va:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb).squeeze(1)
                va_loss += loss_fn(pred, yb).item() * len(Xb)
        va_loss /= len(dl_va.dataset)

        improved = va_loss < best_va - 1e-5
        marker   = " *" if improved else ""
        print(f"  [{name}] ep {ep:02d}/{epochs}  tr={tr_loss:.4f}  va={va_loss:.4f}{marker}")

        if improved:
            best_va    = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  [{name}] Early stop en ep {ep} (patience={PATIENCE})")
                break

    model.load_state_dict(best_state)
    return model


def _eval_test(model, ds_te, loss_fn, device, name: str, is_clf: bool = True):
    """Reporta loss y accuracy (clasificacion) o MAE (regresion) sobre el test set."""
    dl = DataLoader(ds_te, batch_size=BATCH_SIZE)
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for Xb, yb in dl:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb).squeeze(1)
            total_loss += loss_fn(pred, yb).item() * len(Xb)
            if is_clf:
                correct += ((torch.sigmoid(pred) > 0.5) == (yb > 0.5)).sum().item()
            else:
                correct += (pred - yb).abs().sum().item()   # MAE acumulado
            n += len(Xb)
    avg_loss = total_loss / n
    metric   = correct / n
    label    = "acc" if is_clf else "MAE"
    print(f"  [{name}] TEST  loss={avg_loss:.4f}  {label}={metric:.4f}")
    return avg_loss, metric


def train_entry_scorer(X, y, device, epochs=None):
    print(f"\n[EntryScorer] muestras base={len(X)}  positivas={int(y.sum())}")
    ds_tr, ds_va, ds_te = split_train_val_test(X, y)
    print(f"  split -> tr={len(ds_tr)}(+aug)  va={len(ds_va)}  te={len(ds_te)}")
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE)
    n_epochs = epochs if epochs is not None else EPOCHS_SCORE
    model = EntryScorer().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss  = nn.BCEWithLogitsLoss()
    model = _train_loop(model, dl_tr, dl_va, loss, opt, n_epochs, "EntryScorer")   
    _eval_test(model, ds_te, loss, device, "EntryScorer", is_clf=True)
    return model


def train_angle_refiner(X, y, device, epochs=None):
    print(f"\n[AngleRefiner] muestras base={len(X)}")
    ds_tr, ds_va, ds_te = split_train_val_test(X, y)
    print(f"  split -> tr={len(ds_tr)}(+aug)  va={len(ds_va)}  te={len(ds_te)}")
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE)
    n_epochs = epochs if epochs is not None else EPOCHS_ANGLE
    model = AngleRefiner().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss  = nn.MSELoss()
    model = _train_loop(model, dl_tr, dl_va, loss, opt, n_epochs, "AngleRefiner")
    _eval_test(model, ds_te, loss, device, "AngleRefiner", is_clf=False)
    return model


def train_ambiguity_detector(X, y, device, epochs=None):
    print(f"\n[AmbiguityDetector] muestras base={len(X)}  ambiguas={int(y.sum())}")
    ds_tr, ds_va, ds_te = split_train_val_test(X, y)
    print(f"  split -> tr={len(ds_tr)}(+aug)  va={len(ds_va)}  te={len(ds_te)}")
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE)
    n_epochs = epochs if epochs is not None else EPOCHS_AMBIG
    model = AmbiguityDetector().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pos_w = torch.tensor([(len(y) - y.sum()) / (y.sum() + 1e-6)], dtype=torch.float32).to(device)
    loss  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    model = _train_loop(model, dl_tr, dl_va, loss, opt, n_epochs, "AmbiguityDetector")    
    _eval_test(model, ds_te, loss, device, "AmbiguityDetector", is_clf=True)
    return model


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    csv_files = sys.argv[1:] if len(sys.argv) > 1 else ["lineas.csv"]

    # Acumular datos de todos los CSVs
    all_Xs, all_ys = [], []
    all_Xa, all_ya = [], []
    all_Xb, all_yb = [], []

    for csv_path in csv_files:
        print(f"\n=== Procesando: {csv_path} ===")
        wl, bnd, pth = load_csv(csv_path)
        print(f"  Station lines: {len(wl)}  |  Boundaries: {len(bnd)}  |  Paths: {len(pth)}")

        wl  = fuse_loose_segments(wl)
        wl  = build_clusters(wl)
        wl  = detect_bundles(wl)
        print(f"  Tras fusion: {len(wl)} segmentos  |  Clusters: {wl['cluster_id'].nunique()}")

        Xs, ys, Xa, ya, Xb, yb = build_datasets(wl, bnd, pth)
        all_Xs.append(Xs); all_ys.append(ys)
        all_Xa.append(Xa); all_ya.append(ya)
        all_Xb.append(Xb); all_yb.append(yb)

    X_score = np.concatenate(all_Xs); y_score = np.concatenate(all_ys)
    X_angle = np.concatenate(all_Xa); y_angle = np.concatenate(all_ya)
    X_ambig = np.concatenate(all_Xb); y_ambig = np.concatenate(all_yb)

    os.makedirs("checkpoints", exist_ok=True)

   # 1. Scorer: Entrenamiento normal (ej. 50-100 épocas)
    scorer  = train_entry_scorer(X_score, y_score, device, epochs=75) 
    torch.save(scorer.state_dict(), "checkpoints/entry_scorer.pt")
    print("\n✔ entry_scorer.pt guardado")


    # 2. Refiner: Ahora es opcional, pero si lo usas, dale épocas normales
    if len(X_angle) > 10:
        refiner = train_angle_refiner(X_angle, y_angle, device, epochs=50)
        torch.save(refiner.state_dict(), "checkpoints/angle_refiner.pt")
        print("✔ angle_refiner.pt guardado")
    else:
        print("\n⚠ Muy pocas muestras para AmbiguityDetector — saltado")

    # 3. AMBIGUITY DETECTOR: Aquí es donde bajamos la intensidad
    # Le damos pocas épocas para que no se obsesione con los errores del pasado
    if len(X_ambig) > 10:
        detector = train_ambiguity_detector(X_ambig, y_ambig, device, epochs=5) 
        torch.save(detector.state_dict(), "checkpoints/ambiguity_detector.pt")
        print("✔ ambiguity_detector.pt guardado")
    else:
        print("\n⚠ Muy pocas muestras para AmbiguityDetector — saltado")

    print("\n=== Entrenamiento completo ===\n")


if __name__ == "__main__":
    main()
