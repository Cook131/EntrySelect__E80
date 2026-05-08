# ============================================================
#  infer_only.py
#  Cargue los 3 modelos entrenados y produce el CSV de salida.
#
# -*- coding: utf-8 -*-
#
#  Uso:
#    python infer_only.py lineas.csv <-input> 
#                   esta forma solo genera el CSV de salida con las predicciones de la red (y reglas de consistencia).
#
#    python infer_only.py lineas.csv --output resultado.csv <-output> --plot <-plot opcional para generar visualización de debug>
#                   esta forma además de generar el CSV, crea una imagen "resultado.png" con la visualización de la solución
#   
#  NOTA: Asegúrese de tener los modelos entrenados (entry_scorer.pt, angle_refiner.pt, ambiguity_detector.pt) en el directorio "checkpoints/".  
# ============================================================

import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from entry_models import (
    EntryScorer, AngleRefiner, AmbiguityDetector,
    dist_point_to_segment, dist_point_to_df, cone_mean,
    normalize_angle_90, opposite_angle, angle_diff_deg,
    NORM_COORD, NORM_DIST, NORM_ANGLE, NORM_CONE,
)
from training import (
    load_csv, fuse_loose_segments, build_clusters, detect_bundles,
    _cluster_dominant_angle, _endpoint_score,
    BACK_TO_BACK_GAP_MM, PATH_REWARD_MM, LAYER_STATION,
    BUNDLE_ROW_TOL, BUNDLE_ROW_GAP, CONE_MAX_DIST_MM
)

CHECKPOINT_DIR   = "checkpoints"
AMBIG_THRESHOLD  = 0.55   # prob de ambigüedad para marcar revisión manual
# Constantes bundle (deben coincidir con train.py)
BUNDLE_ROW_TOL   = 30
BUNDLE_ROW_GAP   = 500
ANGLE_DELTA_CLIP = 45.0   # clip máximo del delta de ángulo (grados)


# ─────────────────────────────────────────────
#  CARGA DE MODELOS
# ─────────────────────────────────────────────

def load_models(device):
    def _load(cls, filename):
        path = os.path.join(CHECKPOINT_DIR, filename)
        if not os.path.exists(path):
            print(f"  ⚠ No encontrado: {path}  (se usará heurística)")
            return None
        m = cls().to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        print(f"  ✔ Cargado: {path}")
        return m

    scorer   = _load(EntryScorer,         "entry_scorer.pt")
    refiner  = _load(AngleRefiner,        "angle_refiner.pt")
    detector = _load(AmbiguityDetector,   "ambiguity_detector.pt")
    return scorer, refiner, detector


# ─────────────────────────────────────────────
#  EXTRACCIÓN DE FEATURES (igual que train.py)
# ─────────────────────────────────────────────

def _make_score_features(ep_x, ep_y, ang_c, row, d_path, d_bnd, d_btb,
                          ext, cone_s, is_A, c_size, dom_ang, a_norm, path_exists,
                          cluster_cx=0.0, cluster_cy=0.0):
    ang_matches = float(a_norm == dom_ang or opposite_angle(a_norm) == dom_ang)
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
 

def _make_angle_features(cx, cy, chosen_ang, row, pth, path_exists):
    d_path = dist_point_to_df(cx, cy, pth)
    best_pang, best_pd = 0.0, float("inf")
    for _, pr in pth.iterrows():
        d = dist_point_to_segment(cx, cy, pr["sx"], pr["sy"], pr["ex"], pr["ey"])
        if d < best_pd:
            best_pd   = d
            dx        = pr["ex"] - pr["sx"]
            dy        = pr["ey"] - pr["sy"]
            best_pang = math.degrees(math.atan2(dy, dx)) % 360.0
    ang_diff = angle_diff_deg(chosen_ang, best_pang)
    return [
        min(d_path, NORM_DIST) / NORM_DIST,
        math.sin(math.radians(chosen_ang)),
        math.cos(math.radians(chosen_ang)),
        math.sin(math.radians(best_pang)),
        math.cos(math.radians(best_pang)),
        ang_diff   / 180.0,
        min(row["length"], NORM_COORD) / NORM_COORD,
        path_exists,
    ]


def _make_ambig_features(sc_A, sc_B, d_path_A, d_path_B,
                          d_bnd_A, d_bnd_B, is_btb, c_size, path_exists, ang_matches):
    score_diff = abs(sc_A - sc_B)
    max_score  = max(abs(sc_A), abs(sc_B))
    return [
        min(score_diff, 3.0) / 3.0,
        min(max_score,  3.0) / 3.0,
        min(min(d_path_A, d_path_B), NORM_DIST) / NORM_DIST,
        min(min(d_bnd_A,  d_bnd_B),  NORM_DIST) / NORM_DIST,
        float(is_btb),
        min(c_size, 50) / 50.0,
        path_exists,
        ang_matches,
    ]


# ─────────────────────────────────────────────
#  INFERENCIA PRINCIPAL
# ─────────────────────────────────────────────

def run_inference(wl, bnd, pth, scorer, refiner, detector, device):
    path_exists = float(len(pth) > 0)
    cluster_sizes     = wl.groupby("cluster_id").size().to_dict()
    cluster_centroids = wl.groupby("cluster_id")[["cx", "cy"]].mean().to_dict("index")

    def _densify(sx, sy, ex, ey, step=300):
        pts = [(sx, sy), (ex, ey)]
        n = int(math.hypot(ex - sx, ey - sy) / step)
        for k in range(1, n):
            t = k / n
            pts.append((sx + t*(ex-sx), sy + t*(ey-sy)))
        return pts

    obs_pts = []
    for _, r in wl.iterrows():
        obs_pts.extend(_densify(r["sx"], r["sy"], r["ex"], r["ey"]))
    for _, r in bnd.iterrows():
        obs_pts.extend(_densify(r["sx"], r["sy"], r["ex"], r["ey"]))

    attr_pts = []
    for _, r in pth.iterrows():
        attr_pts.extend(_densify(r["sx"], r["sy"], r["ex"], r["ey"]))

    results = []

    for idx, row in wl.iterrows():
        a_norm  = int(row["angle_norm"])
        a_opp   = opposite_angle(a_norm)
        horiz   = row["horiz"]
        cid     = int(row["cluster_id"])
        c_size  = cluster_sizes.get(cid, 1)
        dom_ang = _cluster_dominant_angle(wl, cid)
        is_btb  = int(row["btb_partner"] != -1)
        centroid   = cluster_centroids.get(cid, {"cx": row["cx"], "cy": row["cy"]})
        cluster_cx = centroid["cx"]
        cluster_cy = centroid["cy"]

        end_A = (row["sx"], row["sy"])
        end_B = (row["ex"], row["ey"])
        
        # Lógica de ángulos base
        ang_A = 180 if horiz else 270
        ang_B = 0   if horiz else 90
        valid = {a_norm, a_opp}
        ang_A = min(valid, key=lambda v: angle_diff_deg(v, ang_A))
        ang_B = min(valid, key=lambda v: angle_diff_deg(v, ang_B))

        d_path_A = dist_point_to_df(*end_A, pth)
        d_path_B = dist_point_to_df(*end_B, pth)
        d_bnd_A  = dist_point_to_df(*end_A, bnd)
        d_bnd_B  = dist_point_to_df(*end_B, bnd)

        pid = int(row["btb_partner"])
        if pid != -1 and pid in wl.index:
            partner  = wl.loc[pid]
            d_btb_A  = math.hypot(end_A[0] - partner["cx"], end_A[1] - partner["cy"])
            d_btb_B  = math.hypot(end_B[0] - partner["cx"], end_B[1] - partner["cy"])
            ext_A    = float(d_btb_A > d_btb_B)
            ext_B    = float(d_btb_B > d_btb_A)
        else:
            d_btb_A = d_btb_B = 10000.0 # Valor alto si no hay partner
            ext_A = ext_B = 0.0

        self_pts = {(row["sx"], row["sy"]), (row["ex"], row["ey"])}
        filtered_obs_A = [(ox, oy) for ox, oy in obs_pts
                          if (ox, oy) not in self_pts
                          and math.hypot(ox - end_A[0], oy - end_A[1]) > 50.0]
        filtered_obs_B = [(ox, oy) for ox, oy in obs_pts
                          if (ox, oy) not in self_pts
                          and math.hypot(ox - end_B[0], oy - end_B[1]) > 50.0]

        cone_A = cone_mean(*end_A, ang_A, filtered_obs_A, attractor_pts=attr_pts)
        cone_B = cone_mean(*end_B, ang_B, filtered_obs_B, attractor_pts=attr_pts)
        ang_matches = float(a_norm == dom_ang or a_opp == dom_ang)

        # ── EntryScorer ──
        if scorer is not None:
            feat_A = _make_score_features(
                *end_A, ang_A, row, d_path_A, d_bnd_A, d_btb_A,
                ext_A, cone_A, True,  c_size, dom_ang, a_norm, path_exists,
                cluster_cx, cluster_cy)
            feat_B = _make_score_features(
                *end_B, ang_B, row, d_path_B, d_bnd_B, d_btb_B,
                ext_B, cone_B, False, c_size, dom_ang, a_norm, path_exists,
                cluster_cx, cluster_cy)
            X = torch.tensor([feat_A, feat_B], dtype=torch.float32).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(scorer(X)).squeeze(1).cpu().numpy()
            sc_A, sc_B = float(probs[0]), float(probs[1])
            score_source = "EntryScorer"
        else:
            sc_A = 0.5 # Heurística simplificada si no hay modelo
            sc_B = 0.5
            score_source = "heuristic"

        # Decisión del extremo
        if sc_A >= sc_B:
            entry_pt, entry_ang, raw_conf = end_A, ang_A, sc_A
        else:
            entry_pt, entry_ang, raw_conf = end_B, ang_B, sc_B

        # ── AmbiguityDetector ──
        needs_review = False
        ambig_prob   = 0.0
        if detector is not None:
            feat_ambig = _make_ambig_features(
                sc_A, sc_B, d_path_A, d_path_B, d_bnd_A, d_bnd_B,
                is_btb, c_size, path_exists, ang_matches)
            X_amb = torch.tensor([feat_ambig], dtype=torch.float32).to(device)
            with torch.no_grad():
                ambig_prob = float(torch.sigmoid(detector(X_amb)).squeeze().cpu())
            needs_review = ambig_prob > AMBIG_THRESHOLD

        # [ELIMINADO]: El bloque de AngleRefiner desaparece.
        # La red solo decide qué extremo (A o B) prefiere inicialmente.

        # Preparar string de regla
        angle_source = "BaseModel"
        ambig_str = "; AMBIGUOUS(review)" if needs_review else ""
        full_rule = f"{score_source}; {angle_source}{ambig_str}"

        results.append({
            "LineId": idx,
            "NormalizedAngle": int(row["angle_norm"]), # CORREGIDO: de 'angle' a 'angle_norm'
            "EntryPointX": entry_pt[0],
            "EntryPointY": entry_pt[1],
            "EntryAngle": int(entry_ang),
            "ConfidenceScore": float(max(sc_A, sc_B)),
            "AmbiguityProb": float(ambig_prob),
            "NeedsReview": needs_review,
            "RuleApplied": full_rule
        })

    return pd.DataFrame(results)

# ─────────────────────────────────────────────
#  CONSISTENCIA DE BLOQUE (post-proceso R6)
# ─────────────────────────────────────────────
def apply_block_consistency(out_df, wl_df):
    """
    Asegura que los segmentos back-to-back (BTB) no apunten al mismo lado.
    Si hay duda, usa la geometria del par para desempatar.
    """
    for idx,row in out_df.iterrows():
        pid = int(wl_df.loc[idx, "btb_partner"])
        if pid == -1 or pid not in out_df.index:
            continue

        # Datos de la estación A (estación actual) y B (estación enfrentada)
        res_a =  out_df.loc[idx]
        res_b =  out_df.loc[pid]
        
        # Si ambas apuntan al mismo lado (ángulos iguales), las corregimos
        if res_a["EntryAngle"] == res_b["EntryAngle"]:
            # Decidimos quien tiene quién tiene la razón según la distancia al pasillo real
            conf_a = res_a["ConfidenceScore"]
            conf_b = res_b["ConfidenceScore"]


            # 1. Intenamos desempatar con la confianza de la red (si hay una diferencia clara)
            if abs(conf_a - conf_b) > 0.05:
                    to_invert = pid if conf_a > conf_b else idx
            else:
                # 2. Si la confianza es similar, desempata con la geometría: 
                # calculamos el centro del par.
                mid_x = (wl_df.loc[idx, "cx"] + wl_df.loc[pid, "cx"]) / 2
                mid_y = (wl_df.loc[idx, "cy"] + wl_df.loc[pid, "cy"]) / 2

                # Para la estación A, ¿que arista esat más lejos del centro del par? Esa arista debería ser la que apunta hacia afuera.
                line_a = wl_df.loc[idx]
                dist_sx = math.hypot(line_a["sx"] - mid_x, line_a["sy"] - mid_y)
                dist_ex = math.hypot(line_a["ex"] - mid_x, line_a["ey"] - mid_y)

                # El extremo más alejado del centro del par debería ser el que apunta hacia afuera.
                # Si no coincide con la dirección actual, lo corregimos.
                curr_dist = math.hypot(res_a["EntryPointX"] - mid_x, res_a["EntryPointY"] - mid_y)
                if curr_dist < max(dist_sx, dist_ex):
                    to_invert = idx
                else:
                    to_invert = pid

            # Aplicar inversión
            new_angle = opposite_angle(out_df.at[to_invert, "EntryAngle"])
            out_df.at[to_invert, "EntryAngle"] = new_angle

            # Actualizar coordenadas del punto de entrada
            l_inv = wl_df.loc[to_invert]
            if new_angle == 0:   out_df.at[to_invert, "EntryPointX"] = max(l_inv["sx"], l_inv["ex"])
            if new_angle == 180: out_df.at[to_invert, "EntryPointX"] = min(l_inv["sx"], l_inv["ex"])
            if new_angle == 90:  out_df.at[to_invert, "EntryPointY"] = max(l_inv["sy"], l_inv["ey"])
            if new_angle == 270: out_df.at[to_invert, "EntryPointY"] = min(l_inv["sy"], l_inv["ey"])

            out_df.at[to_invert, "RuleApplied"] += "; BlockConsistency"

    return out_df



# ─────────────────────────────────────────────
#  ALTERNANCIA DE BUNDLE — HARD RULE
# ─────────────────────────────────────────────
 
def apply_bundle_alternation(out_df, wl_df):
    for idx, row in out_df.iterrows():
        pid = int(wl_df.loc[idx, "btb_partner"])
        if pid == -1 or pid not in out_df.index:
            continue
            
        line_a = wl_df.loc[idx]
        line_b = wl_df.loc[pid]
        
        # Eje central del par
        mid_x, mid_y = (line_a['cx'] + line_b['cx'])/2, (line_a['cy'] + line_b['cy'])/2
        
        # Vector de escape: del centro del par hacia afuera
        vx, vy = line_a['cx'] - mid_x, line_a['cy'] - mid_y
        
        # Si la flecha actual NO coincide con la dirección de escape, la corregimos
        escape_angle = normalize_angle_90(math.degrees(math.atan2(vy, vx)) % 360)
        
        if out_df.at[idx, "EntryAngle"] != escape_angle:
            out_df.at[idx, "EntryAngle"] = escape_angle
            # Mover punto al extremo correcto
            if escape_angle == 0:   out_df.at[idx, "EntryPointX"] = max(line_a["sx"], line_a["ex"])
            if escape_angle == 180: out_df.at[idx, "EntryPointX"] = min(line_a["sx"], line_a["ex"])
            if escape_angle == 90:  out_df.at[idx, "EntryPointY"] = max(line_a["sy"], line_a["ey"])
            if escape_angle == 270: out_df.at[idx, "EntryPointY"] = min(line_a["sy"], line_a["ey"])
            
    return out_df
 
 
def plot_debug(wl, bnd, pth, out, save_as="debug_infer.png"):
    fig, ax = plt.subplots(figsize=(20, 20), dpi=150)
    ax.set_aspect("equal")
    for _, r in wl.iterrows():
        ax.plot([r["sx"], r["ex"]], [r["sy"], r["ey"]],
                color="lightsteelblue", lw=0.7, zorder=1)
    for _, r in bnd.iterrows():
        ax.plot([r["sx"], r["ex"]], [r["sy"], r["ey"]],
                color="red", lw=1.2, zorder=3)
    for _, r in pth.iterrows():
        ax.plot([r["sx"], r["ex"]], [r["sy"], r["ey"]],
                color="orange", lw=1.0, ls="--", zorder=3)
 
    cmap = plt.cm.RdYlGn
    arrow_len = 600
    for _, r in out.iterrows():
        color = "magenta" if r["NeedsReview"] else cmap(float(r["ConfidenceScore"]))
        ax.scatter(r["EntryPointX"], r["EntryPointY"], color=color, s=8, zorder=5)
        dx = arrow_len * math.cos(math.radians(r["EntryAngle"]))
        dy = arrow_len * math.sin(math.radians(r["EntryAngle"]))
        ax.annotate("", xy=(r["EntryPointX"] + dx, r["EntryPointY"] + dy),
                    xytext=(r["EntryPointX"], r["EntryPointY"]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8), zorder=5)
 
    patches = [
        mpatches.Patch(color="lightsteelblue", label="Warehouse Lines"),
        mpatches.Patch(color="red",            label="Boundaries"),
        mpatches.Patch(color="orange",         label="Path Suggestions"),
        mpatches.Patch(color="green",          label="Entry (alta confianza)"),
        mpatches.Patch(color="red",            label="Entry (baja confianza)"),
        mpatches.Patch(color="magenta",        label="Ambiguo (revisar)"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    ax.set_title("Entry Point Resolver — Inferencia")
    ax.grid(True, lw=0.3)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    plt.close()
    print(f"  [plot] {save_as}")

# ─────────────────────────────────────────────
#  VISUALIZACIÓN DEBUG
# ─────────────────────────────────────────────

def plot_debug(wl, bnd, pth, out, save_as="debug_infer.png"):
    fig, ax = plt.subplots(figsize=(20, 20), dpi=150)
    ax.set_aspect("equal")
    for _, r in wl.iterrows():
        ax.plot([r["sx"], r["ex"]], [r["sy"], r["ey"]],
                color="lightsteelblue", lw=0.7, zorder=1)
    for _, r in bnd.iterrows():
        ax.plot([r["sx"], r["ex"]], [r["sy"], r["ey"]],
                color="red", lw=1.2, zorder=3)
    for _, r in pth.iterrows():
        ax.plot([r["sx"], r["ex"]], [r["sy"], r["ey"]],
                color="orange", lw=1.0, ls="--", zorder=3)

    cmap = plt.cm.RdYlGn
    arrow_len = 600
    for _, r in out.iterrows():
        color = "magenta" if r["NeedsReview"] else cmap(float(r["ConfidenceScore"]))
        ax.scatter(r["EntryPointX"], r["EntryPointY"], color=color, s=8, zorder=5)
        dx = arrow_len * math.cos(math.radians(r["EntryAngle"]))
        dy = arrow_len * math.sin(math.radians(r["EntryAngle"]))
        ax.annotate("", xy=(r["EntryPointX"] + dx, r["EntryPointY"] + dy),
                    xytext=(r["EntryPointX"], r["EntryPointY"]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8), zorder=5)

    patches = [
        mpatches.Patch(color="lightsteelblue", label="Warehouse Lines"),
        mpatches.Patch(color="red",            label="Boundaries"),
        mpatches.Patch(color="orange",         label="Path Suggestions"),
        mpatches.Patch(color="green",          label="Entry (alta confianza)"),
        mpatches.Patch(color="red",            label="Entry (baja confianza)"),
        mpatches.Patch(color="magenta",        label="Ambiguo (revisar)"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    ax.set_title("Entry Point Resolver — Inferencia")
    ax.grid(True, lw=0.3)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    plt.close()
    print(f"  [plot] {save_as}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(description="Entry Point Inference")
    parser.add_argument("input",  help="CSV de entrada (delimitado por ;)")
    parser.add_argument("--output",    default="output_entries.csv")
    parser.add_argument("--plot",      action="store_true")
    parser.add_argument("--plot-file", default="debug_infer.png")
    args = parser.parse_args()
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Entry Point Inference ===")
    print(f"  Device : {device}")
    print(f"  Input  : {args.input}")
 
    print("\n[1] Cargando modelos…")
    scorer, refiner, detector = load_models(device)
 
    print("\n[2] Cargando y preprocesando datos…")
    wl, bnd, pth = load_csv(args.input)
    print(f"  Station lines: {len(wl)}  |  Boundaries: {len(bnd)}  |  Paths: {len(pth)}")
 
    wl = fuse_loose_segments(wl)
    wl = build_clusters(wl)
    wl = detect_bundles(wl)

    # DEBUG TEMPORAL — quitar después
    for orient, label in [(True, "HORIZ"), (False, "VERT")]:
        sub = wl[wl["horiz"] == orient]
        if len(sub) < 2:
            continue
        perp_col = "sy" if orient else "sx"
        vals = sub[perp_col].sort_values().values
        gaps = pd.Series(vals).diff().dropna()
        small_gaps = gaps[(gaps > 0) & (gaps < 5000)].sort_values()
        print(f"\nGaps perpendiculares {label} (< 5000mm):")
        print(small_gaps.value_counts().head(15))

    print(f"\nbundle_row distribucion:")
    print(wl["bundle_row"].value_counts().sort_index())
    print(f"\nbundle_size distribucion:")
    print(wl["bundle_size"].value_counts().sort_index())
    #################################################3

    print(f"  Tras fusion: {len(wl)} segmentos  |  Clusters: {wl['cluster_id'].nunique()}")
    btb = (wl["btb_partner"] != -1).sum()
    print(f"  Pares back-to-back: {btb}")
 
    print("\n[3] Inferencia…")
    out = run_inference(wl, bnd, pth, scorer, None, detector, device) # 1. Red decide
    out = apply_block_consistency(out, wl)                          # 2. Vecinos alineados copian al líder
    out = apply_bundle_alternation(out, wl)                         # 3. Back-to-back se repelen
 
    n_review = out["NeedsReview"].sum()
    n_high   = (out["ConfidenceScore"] >= 0.6).sum()
    print(f"  Líneas procesadas     : {len(out)}")
    print(f"  Alta confianza (≥0.6) : {n_high}")
    print(f"  Requieren revisión    : {n_review}")
 
    out.to_csv(args.output, index=False, sep=";")
    print(f"\n✔ Exportado: {args.output}")
 
    if args.plot:
        plot_debug(wl, bnd, pth, out, save_as=args.plot_file)
 
    print("\n=== Listo ===\n")
 
 
if __name__ == "__main__":
    main()