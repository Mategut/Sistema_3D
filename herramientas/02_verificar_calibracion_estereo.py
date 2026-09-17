#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verificación ligera de la calibración estéreo activa.

No carga CREStereo ni ONNX Runtime. Comprueba:
- calibration_report.json = accepted;
- consistencia métrica T/P2/Q;
- claves y dimensiones de rectification_maps.npz;
- alineación epipolar natural del fondo vacío con RANSAC vertical.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CAL = ROOT / "sistema" / "calibracion_estereo"
BG = ROOT / "sistema" / "fondo_vacio"


def fail(msg: str) -> int:
    """Publica el motivo de rechazo y devuelve el código de salida 2."""
    print("[FAIL]", msg)
    return 2


def robust_fit(points_right, vertical_error, threshold=1.75, iterations=6000):
    """Ajusta el error vertical como función afín de las coordenadas derechas.

    RANSAC usa una semilla fija y refina el consenso por mínimos cuadrados.
    Devuelve coeficientes, máscara de inliers y residuos absolutos; si no hay
    soporte suficiente, devuelve None y residuos infinitos.
    """
    n = len(vertical_error)
    if n < 3:
        return None, np.zeros(n, bool), np.full(n, np.inf)
    X = np.column_stack([points_right[:, 0], points_right[:, 1], np.ones(n, float)])
    y = np.asarray(vertical_error, float)
    rng = np.random.default_rng(20260831)
    best = np.zeros(n, bool)
    best_count = 0
    best_med = float("inf")
    for _ in range(max(100, int(iterations))):
        ids = rng.choice(n, 3, replace=False)
        A = X[ids]
        if abs(float(np.linalg.det(A))) < 1e-7:
            continue
        try:
            coef = np.linalg.solve(A, y[ids])
        except np.linalg.LinAlgError:
            continue
        res = np.abs(X @ coef - y)
        mask = res <= threshold
        count = int(mask.sum())
        if count < 3:
            continue
        med = float(np.median(res[mask]))
        if count > best_count or (count == best_count and med < best_med):
            best_count = count
            best_med = med
            best = mask
    if best_count < 3:
        return None, best, np.full(n, np.inf)
    coef, *_ = np.linalg.lstsq(X[best], y[best], rcond=None)
    res = np.abs(X @ coef - y)
    mask = res <= threshold
    if int(mask.sum()) >= 3:
        coef, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        res = np.abs(X @ coef - y)
        mask = res <= threshold
    return coef, mask, res


def main() -> int:
    """Verifica los archivos activos y audita el fondo cuando está disponible.

    Comprueba dimensiones de mapas, escala métrica y evidencia epipolar. No
    modifica calibraciones ni aplica la corrección estimada. Devuelve 0 al
    superar las comprobaciones y 2 ante un rechazo explícito.
    """
    print(
        "[PROGRESO] Se está ejecutando: 02_verificar_calibracion_estereo.py",
        flush=True,
    )
    print("[PROGRESO] Verificando calibración estéreo activa", flush=True)
    report_path = CAL / "calibration_report.json"
    yaml_path = CAL / "stereo_initial.yaml"
    maps_path = CAL / "rectification_maps.npz"
    for p in (report_path, yaml_path, maps_path):
        if not p.is_file():
            return fail(f"Falta {p}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if str(report.get("quality", "")).lower() != "accepted":
        return fail(f"calibration_report quality={report.get('quality')}")
    if list(report.get("image_size") or []) != [1920, 1080]:
        return fail(f"Resolución de calibración incompatible: {report.get('image_size')}")

    maps = np.load(maps_path)
    needed = ("map1x", "map1y", "map2x", "map2y")
    for k in needed:
        if k not in maps.files:
            return fail(f"Falta clave {k} en NPZ")
        if maps[k].shape != (1080, 1920):
            return fail(f"{k} tiene forma {maps[k].shape}")
        if not np.isfinite(maps[k]).all():
            return fail(f"{k} contiene NaN/Inf")

    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return fail("No se pudo abrir stereo_initial.yaml")
    T = fs.getNode("T").mat()
    P1 = fs.getNode("P1").mat()
    P2 = fs.getNode("P2").mat()
    Q = fs.getNode("Q").mat()
    baseline = float(fs.getNode("baseline_mm").real())
    stereo_rms = float(fs.getNode("rms_stereo").real())
    fs.release()
    bT = float(np.linalg.norm(np.asarray(T, float).reshape(-1)[:3]))
    bP = float(-P2[0, 3] / P2[0, 0])
    bQ = float(1.0 / Q[3, 2])
    if max(abs(baseline - bT), abs(baseline - bP), abs(baseline - bQ)) > 1e-3:
        return fail(
            f"Baseline inconsistente: YAML={baseline:.6f}, T={bT:.6f}, P2={bP:.6f}, Q={bQ:.6f}"
        )

    left = cv2.imread(str(BG / "background_left.png"))
    right = cv2.imread(str(BG / "background_right.png"))
    if left is None or right is None:
        print("[WARN] No hay fondo vacío; se omite auditoría natural.")
        print(
            f"[PASS] Calibración estructural accepted | RMS={stereo_rms:.3f}px | B={baseline:.3f}mm"
        )
        return 0

    rl = cv2.remap(left, maps["map1x"], maps["map1y"], cv2.INTER_LINEAR)
    rr = cv2.remap(right, maps["map2x"], maps["map2y"], cv2.INTER_LINEAR)
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.005, edgeThreshold=15)
    kl, dl = sift.detectAndCompute(cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY), None)
    kr, dr = sift.detectAndCompute(cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY), None)
    if dl is None or dr is None:
        return fail("Fondo sin descriptores suficientes para auditoría epipolar")
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dl, dr, k=2)
    fx = float(P1[0, 0])
    dmin = max(8.0, 0.72 * fx * baseline / 1200.0)
    dmax = min(1920 * 0.92, 1.35 * fx * baseline / 150.0)
    cand = []
    for pair in knn:
        if len(pair) < 2:
            continue
        a, b = pair
        if a.distance >= 0.85 * b.distance:
            continue
        xl, yl = kl[a.queryIdx].pt
        xr, yr = kr[a.trainIdx].pt
        disp = xl - xr
        dy = yl - yr
        if dmin <= disp <= dmax and abs(dy) <= 60:
            cand.append((xr, yr, dy))
    if len(cand) < 45:
        return fail(f"Solo {len(cand)} correspondencias naturales; mínimo 45")
    pts = np.asarray([[x, y] for x, y, _ in cand], float)
    dy = np.asarray([d for _, _, d in cand], float)
    coef, inliers, res = robust_fit(pts, dy)
    if coef is None:
        return fail("RANSAC epipolar sin modelo estable")
    count = int(inliers.sum())
    ratio = count / len(cand)
    xspan = float(np.ptp(pts[inliers, 0]) / 1919.0)
    post95 = float(np.percentile(res[inliers], 95))
    corners = np.asarray([[0.0, 0.0], [1919.0, 0.0], [0.0, 1079.0], [1919.0, 1079.0]])
    pred = coef[0] * corners[:, 0] + coef[1] * corners[:, 1] + coef[2]
    maxcorr = float(np.max(np.abs(pred)))
    ok = (
        count >= 35
        and ratio >= 0.25
        and xspan >= 0.45
        and post95 <= 1.75
        and maxcorr <= 8.0
        and abs(float(coef[1])) <= 0.025
    )
    if not ok:
        return fail(
            f"Auditoría natural no aceptada: matches={len(cand)}, inliers={count}, ratio={ratio:.3f}, xspan={xspan:.3f}, postP95={post95:.3f}px, maxcorr={maxcorr:.3f}px"
        )

    print("=" * 70)
    print("CALIBRACIÓN ESTÉREO: PASS")
    print(f"RMS estéreo              : {stereo_rms:.6f} px")
    print(f"Baseline                 : {baseline:.6f} mm")
    print(
        f"Reporte checker epipolar : med={report['epipolar']['median_abs_dy_px']:.6f}px | P95={report['epipolar']['p95_abs_dy_px']:.6f}px"
    )
    print(f"Fondo natural            : matches={len(cand)} | inliers={count} | ratio={ratio:.3f}")
    print(f"Residual post-model P95  : {post95:.6f} px")
    print(f"Corrección máxima modelo : {maxcorr:.6f} px")
    print(
        "Estado                    : accepted_no_correction"
        if maxcorr <= 1.50
        else "Estado                    : accepted (corrección residual posible)"
    )
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
