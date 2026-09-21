#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PASO 13 V5.3 — RECONSTRUCCIÓN CON CONTINUIDAD Y RESPALDO OBSERVACIONAL

Reconstrucción general, sin suponer cubo, cilindro, pirámide ni otra forma.

La ruta normal ejecuta un único Screened Poisson sobre una representación
auxiliar ponderada por confianza y soporte. Después de Poisson se eliminan
únicamente componentes completas que sean pequeñas, estén aisladas y carezcan
de respaldo suficiente. Nunca se borran caras individuales del cuerpo válido.
Las regiones débiles usan Taubin restringido y tendencia cuadrática de la nube
a dos escalas. La proximidad aislada no inmoviliza una desviación local
verificada. Los cambios conservan orientación y no añaden pares de intersección.

La nube completa nunca se sustituye: se conserva para respaldo, color,
evaluación y trazabilidad. Ball Pivoting queda como rescate únicamente si
Poisson falla. El repliegue de fronteras verifica que no aparezcan nuevos pares de
intersección; el paso 16 conserva la validación semántica completa.
"""

from __future__ import annotations

# Monitor independiente de las utilidades instaladas en la aplicación.
# Solo se inicia al ejecutar este archivo, nunca al importarlo.
import sys as _sys
import time as _time
import threading as _threading
import atexit as _atexit

_estado_lock = _threading.Lock()
_estado_inicio = _time.monotonic()
_estado_actual = ("Cargando dependencias de geometría", _estado_inicio)
_estado_stop = _threading.Event()


def _mostrar_estado13():
    """Publica la actividad actual y los tiempos transcurridos con lectura protegida."""
    with _estado_lock:
        actividad, inicio = _estado_actual
    ahora = _time.monotonic()
    print(
        f"[PROGRESO] Paso 13 | {actividad} | Tiempo actividad: {ahora-inicio:.1f} s | "
        f"Tiempo total: {ahora-_estado_inicio:.1f} s",
        flush=True,
    )


def _estado13(actividad):
    """Actualiza la actividad bajo bloqueo y publica el nuevo estado."""
    global _estado_actual
    with _estado_lock:
        _estado_actual = (actividad, _time.monotonic())
    _mostrar_estado13()


def _latido13():
    """Publica periódicamente el estado hasta recibir la señal de detención."""
    while not _estado_stop.wait(10):
        _mostrar_estado13()


if __name__ == "__main__":
    for _stream in (_sys.stdout, _sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(line_buffering=True, write_through=True)
            except (ValueError, OSError):
                pass
    print("[EJECUTANDO] 13_reconstruir_superficie.py | Iniciando procesamiento", flush=True)
    _mostrar_estado13()
    _monitor13 = _threading.Thread(target=_latido13, name="estado_paso13", daemon=True)
    _monitor13.start()
    _atexit.register(_estado_stop.set)

from utilidades_progreso import operacion
from utilidades_rendimiento import query_threads
from utilidades_rendimiento import configurar as _configurar_recursos

# Los runtimes numéricos leen estos límites al importarse.
_configurar_recursos(__file__)


import os

_CPU_COUNT = max(1, os.cpu_count() or 1)
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, str(_CPU_COUNT))

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
except Exception as exc:
    raise SystemExit(f"Paso 13 V5.0 requiere SciPy: {exc}")

try:
    import open3d as o3d
except Exception as exc:
    raise SystemExit("Paso 13 V5.0 requiere Open3D 0.19.x. " f"Detalle: {exc}")


VERSION = "V5.8"
STEP = "13"


def make_parser():
    """Construye las opciones de línea de comandos de este paso."""
    p = argparse.ArgumentParser(
        description=(
            "Paso 13 V5.6 — selección Poisson/BPA por fidelidad observacional simétrica, "
            "respaldo observacional y regularización restringida."
        )
    )
    p.add_argument("--root", required=True)
    p.add_argument("--object", required=True)
    p.add_argument("--source", default="12_regularizacion_nube")
    p.add_argument("--output-name", default="13_reconstruccion_superficie")

    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="0 usa todos los procesadores lógicos disponibles.",
    )
    p.add_argument(
        "--poisson-depth",
        type=int,
        default=0,
        help="0 calcula depth automáticamente.",
    )
    p.add_argument("--poisson-min-depth", type=int, default=8)
    p.add_argument("--poisson-max-depth", type=int, default=8)
    p.add_argument("--poisson-scale", type=float, default=1.08)
    p.add_argument(
        "--poisson-target-cell-spacing-factor",
        type=float,
        default=0.90,
    )
    p.add_argument("--poisson-min-cell-mm", type=float, default=0.45)
    p.add_argument(
        "--poisson-density-trim-quantile",
        type=float,
        default=0.01,
        help=(
            "Recorte conservador de densidad Poisson. 0 lo desactiva; el valor "
            "por defecto elimina únicamente el 1%% de menor densidad antes de "
            "aplicar los gates de evidencia observacional."
        ),
    )
    p.add_argument("--bbox-margin-mm", type=float, default=4.0)
    p.add_argument(
        "--boundary-retraction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replegar fronteras extrapoladas con evidencia local y guardas geométricas.",
    )
    p.add_argument("--max-poisson-input-points", type=int, default=55000)
    p.add_argument(
        "--input-voxel-spacing-factor",
        type=float,
        default=1.10,
    )
    p.add_argument("--meshing-minimum-voxel-mm", type=float, default=1.0)
    p.add_argument("--meshing-confidence-exponent", type=float, default=1.5)
    p.add_argument("--meshing-support-exponent", type=float, default=0.65)
    # V5.0 — contrato de evidencia procedente de 11/12. El mallado no debe
    # volver a legitimar una capa que ya fue marcada como ambigua.
    p.add_argument(
        "--require-evidence-contract", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--minimum-evidence-class", type=int, default=2)
    p.add_argument("--minimum-independent-support", type=int, default=2)
    p.add_argument("--minimum-angular-span-poses", type=int, default=2)
    p.add_argument("--maximum-conflict-pose-ratio", type=float, default=0.35)
    p.add_argument("--minimum-heldout-pass-ratio", type=float, default=0.67)
    p.add_argument("--evidence-strength-exponent", type=float, default=1.0)
    p.add_argument("--component-min-triangles", type=int, default=60)
    p.add_argument(
        "--component-min-triangle-ratio",
        type=float,
        default=0.00025,
    )
    p.add_argument(
        "--support-trim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Conserva el nombre por compatibilidad. Solo permite eliminar "
            "componentes completas aisladas; nunca caras internas."
        ),
    )
    p.add_argument(
        "--support-strong-distance-mm",
        type=float,
        default=0.0,
        help="0 calcula automáticamente la distancia de respaldo fuerte.",
    )
    p.add_argument(
        "--support-reject-distance-mm",
        type=float,
        default=0.0,
        help="0 calcula la distancia donde la regularización llega al máximo.",
    )
    p.add_argument("--support-min-confidence", type=float, default=0.55)
    p.add_argument("--support-min-views", type=float, default=2.0)
    p.add_argument("--support-component-min-seed-faces", type=int, default=12)
    p.add_argument(
        "--support-component-min-seed-ratio",
        type=float,
        default=0.02,
    )
    p.add_argument(
        "--support-max-removed-triangle-ratio",
        type=float,
        default=0.45,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--weak-region-smoothing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--weak-region-iterations", type=int, default=5)
    p.add_argument("--weak-region-lambda", type=float, default=0.34)
    p.add_argument("--weak-region-mu", type=float, default=-0.35)
    p.add_argument("--weak-region-anchor-confidence", type=float, default=0.72)
    p.add_argument("--weak-region-anchor-support", type=float, default=3.0)
    p.add_argument(
        "--weak-region-reliability-weight",
        type=float,
        default=0.35,
    )
    p.add_argument(
        "--weak-region-max-displacement-mm",
        type=float,
        default=0.0,
        help="0 calcula un límite físico a partir del spacing de Poisson.",
    )
    p.add_argument(
        "--bpa-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="BPA se usa como rescate y también puede competir cuando Poisson muestra extrapolación no respaldada.",
    )
    p.add_argument("--bpa-radius-factors", default="1.5,2.5,4.0")
    p.add_argument(
        "--adaptive-bpa-competition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Evalúa BPA además de Poisson solo cuando la malla implícita muestra "
            "riesgo de extrapolación. La selección usa evidencia geométrica, no el "
            "nombre ni la forma conocida del objeto."
        ),
    )
    p.add_argument("--candidate-min-absolute-coverage", type=float, default=0.90)
    p.add_argument("--candidate-min-relative-coverage", type=float, default=0.94)
    p.add_argument("--poisson-risk-p90-spacing-factor", type=float, default=4.0)
    p.add_argument("--poisson-risk-p95-spacing-factor", type=float, default=5.0)
    p.add_argument("--poisson-risk-bbox-expansion-ratio", type=float, default=0.05)
    p.add_argument("--candidate-switch-min-score-gain", type=float, default=0.08)
    p.add_argument(
        "--candidate-min-largest-component-ratio",
        type=float,
        default=0.25,
        help=(
            "Guardia de seguridad para alternativas fragmentadas. No exige una sola "
            "componente dominante: la fragmentación se penaliza en la puntuación y "
            "solo se rechaza si es extrema."
        ),
    )
    p.add_argument("--candidate-max-components", type=int, default=20)
    p.add_argument("--candidate-max-boundary-edge-ratio", type=float, default=0.15)
    p.add_argument("--adaptive-observed-refinement", action=argparse.BooleanOptionalAction,
                   default=True, help="Si ningún candidato cumple, repetir BPA con menos reducción de muestras observadas.")
    p.add_argument("--evaluation-cloud-samples", type=int, default=30000)
    p.add_argument("--evaluation-mesh-samples", type=int, default=30000)
    p.add_argument("--coverage-gate-mm", type=float, default=3.0)
    p.add_argument("--warning-max-components", type=int, default=80)
    p.add_argument(
        "--warning-min-largest-component-ratio",
        type=float,
        default=0.94,
    )
    p.add_argument(
        "--warning-max-mesh-to-cloud-p90-mm",
        type=float,
        default=3.5,
    )
    p.add_argument(
        "--use-estimated-completion", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument("--completion-source", default="11_fusion_multivista")
    return p


def robust_stats(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(len(a)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def optional_scalar(data, key, default=None):
    """Lee un escalar finito del NPZ o devuelve el valor de respaldo."""
    if key not in data.files:
        return default
    value = np.asarray(data[key]).reshape(-1)
    if len(value) == 0:
        return default
    result = float(value[0])
    return result if np.isfinite(result) else default


def query_tree(tree, points, k=1):
    """Consulta vecinos con paralelismo y respaldo para versiones antiguas de SciPy."""
    try:
        return tree.query(points, k=k, workers=query_threads())
    except TypeError:
        return tree.query(points, k=k)


def nearest_spacing(points, max_samples=60000):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        raise RuntimeError("No hay suficientes puntos para medir spacing.")
    if len(points) > int(max_samples):
        indices = np.linspace(
            0,
            len(points) - 1,
            int(max_samples),
        ).astype(np.int64)
        query = points[indices]
    else:
        query = points
    distances, _ = query_tree(cKDTree(points), query, k=2)
    spacing = float(np.median(np.asarray(distances)[:, 1]))
    if not np.isfinite(spacing) or spacing <= 0:
        raise RuntimeError("El spacing estimado no es válido.")
    return spacing


def clean_mesh(mesh):
    """Elimina duplicados, triángulos degenerados y vértices sin uso en la malla."""
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


def make_point_cloud(points, colors, normals):
    """Construye una nube Open3D con colores normalizados y normales unitarias."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64) / 255.0)
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    pcd.normalize_normals()
    return pcd


def select_supported_surface_samples(points, normals, confidence, support, spacing):
    """Selecciona por continuidad local; conserva parches de baja densidad coherentes.

    No infiere nuevas poses: support es evidencia heredada. Densidad y dispersión
    se usan solo como diagnósticos geométricos, no como confianza observacional.
    """
    count = len(points)
    if count < 32:
        return np.ones(count, bool), np.ones(count), {"applied": False, "reason": "small_cloud"}
    h = max(float(spacing), 1e-6)
    unit = normals / np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    tree = cKDTree(points)
    agreement = np.zeros(count)
    neighbor_count = np.zeros(count, dtype=int)
    for start in range(0, count, 2048):
        ids = np.arange(start, min(count, start + 2048))
        dist, ix = query_tree(tree, points[ids], k=min(32, count))
        delta = points[ix] - points[ids, None, :]
        dot = np.einsum("bki,bi->bk", unit[ix], unit[ids])
        height = np.abs(np.einsum("bki,bi->bk", delta, unit[ids]))
        tangent2 = np.maximum(dist * dist - height * height, 0.0)
        local = (dist > 1e-9) & (dist <= 3 * h)
        consistent = (
            local & (dot >= np.cos(np.deg2rad(35))) & (height <= 0.30 * h + tangent2 / (8 * h))
        )
        neighbor_count[ids] = consistent.sum(axis=1)
        agreement[ids] = consistent.sum(axis=1) / np.maximum(local.sum(axis=1), 1)
    # Solo retirar muestras sin parche propio cuando existe una superficie
    # vecina claramente mejor respaldada; no conservar solo la componente mayor.
    strong = (neighbor_count >= 8) & (agreement >= 0.65) & (support >= 2) & (confidence >= 0.65)
    keep = np.ones(count, dtype=bool)
    if np.any(strong):
        distance, _ = query_tree(cKDTree(points[strong]), points, k=1)
        keep = ~((neighbor_count < 3) & (agreement < 0.25) & (distance <= 2 * h) & ~strong)
    score = np.clip(0.25 + 0.75 * agreement, 0.25, 1.0)
    report = {
        "applied": bool(np.any(~keep)),
        "input_points": count,
        "removed_points": int(np.count_nonzero(~keep)),
        "strong_local_samples": int(np.count_nonzero(strong)),
        "policy": "reject_only_locally_unsupported_samples_near_coherent_surface",
        "local_agreement": robust_stats(agreement),
        "density_is_not_pose_support": True,
    }
    if np.count_nonzero(keep) < 1000 or np.mean(keep) < 0.90:
        keep[:] = True
        report.update(applied=False, reason="coverage_safety_fallback", removed_points=0)
    nearest, _ = query_tree(cKDTree(points[keep]), points, k=1)
    report["selection_distance_mm"] = robust_stats(nearest)
    report["retained_fraction"] = float(np.mean(keep))
    return keep, score, report


def surface_voxel_groups(points, normals, voxel):
    """Un vóxel puede contener varias capas: no se promedian entre sí."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    cuts = np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1, len(order)]
    groups = np.empty(len(points), dtype=np.int64)
    normal = normals / np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    next_id = 0
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        representatives = []
        for idx in order[lo:hi]:
            assigned = False
            for anchor, group in representatives:
                delta = points[idx] - points[anchor]
                compatible = (
                    np.dot(normal[idx], normal[anchor]) >= np.cos(np.deg2rad(35))
                    and abs(np.dot(delta, normal[anchor])) <= 0.35 * voxel
                    and abs(np.dot(delta, normal[idx])) <= 0.35 * voxel
                )
                if compatible:
                    groups[idx] = group
                    assigned = True
                    break
            if not assigned:
                representatives.append((idx, next_id))
                groups[idx] = next_id
                next_id += 1
    return groups


def resolve_local_layers_and_normals(points, normals, confidence, support, spacing):
    """Resolver SOLO duplicación local débil; mantener separaciones resueltas.
    Las escalas son empíricas. La incertidumbre no es covarianza de cámara.
    Un grupo propio, coherente y multivista no se elimina por ser minoritario.
    """
    h = max(float(spacing), 1e-6)
    n = normals / np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    updated = n.copy()
    keep = np.ones(len(points), bool)
    records = dict(
        weak_competing_samples_removed=0,
        ambiguous_samples_preserved=0,
        normals_updated=0,
        transition_normals_preserved=0,
        uncertainty_source="local_residual_MAD_not_calibrated_covariance",
    )
    if len(points) < 32:
        return keep, updated, records
    tree = cKDTree(points)
    # Consultas por bloques: memoria acotada y KD-tree paralelo existente.
    for start in range(0, len(points), 1024):
        stop = min(len(points), start + 1024)
        distances, neighbors = query_tree(tree, points[start:stop], k=min(48, len(points)))
        for row, i in enumerate(range(start, stop)):
            ix = neighbors[row][distances[row] <= 3 * h]
            if len(ix) < 12:
                continue
            delta = points[ix] - points[i]
            dot = n[ix] @ n[i]
            family = dot >= np.cos(np.deg2rad(30.0))
            # En una transición con familias de normales distintas no promediar.
            transition = np.count_nonzero(~family) >= max(4, len(ix) // 5)
            ids = ix[family]
            delta = delta[family]
            dot = dot[family]
            if len(ids) < 12:
                continue
            axis = np.eye(3)[np.argmin(np.abs(n[i]))]
            a = np.cross(n[i], axis)
            a /= np.linalg.norm(a)
            b = np.cross(n[i], a)
            x, y, z = delta @ a, delta @ b, delta @ n[i]
            height = z + 0.5 * ((n[ids] @ a) * x + (n[ids] @ b) * y) / np.maximum(dot, 1e-9)
            order = np.argsort(height)
            cuts = np.r_[0, np.flatnonzero(np.diff(height[order]) > 0.25 * h) + 1, len(ids)]
            groups = [order[lo:hi] for lo, hi in zip(cuts[:-1], cuts[1:])]
            own = next(g for g in groups if np.any(ids[g] == i))

            # Comparar parches enteros por respaldo heredado y cobertura
            # tangencial. El número de puntos no se interpreta como poses nuevas.
            def patch_evidence(g):
                good = (support[ids[g]] >= 2) & (confidence[ids[g]] >= 0.65)
                sectors = np.unique(
                    np.floor((np.arctan2(y[g], x[g]) + np.pi) * 4 / np.pi).astype(int) % 8
                )
                extent_xy = np.ptp(np.column_stack((x[g], y[g])), axis=0)
                empirical = max(
                    0.06 * h, float(1.4826 * np.median(np.abs(height[g] - np.median(height[g]))))
                )
                return dict(
                    good=int(good.sum()),
                    sectors=len(sectors),
                    extent=float(np.min(extent_xy)),
                    sigma=empirical,
                    quality=float(np.median(np.minimum(support[ids[g]], 5) * confidence[ids[g]])),
                )

            evidence = [patch_evidence(g) for g in groups]
            own_index = next(k for k, g in enumerate(groups) if np.any(ids[g] == i))
            winner_index = max(
                range(len(groups)),
                key=lambda k: (
                    evidence[k]["sectors"] >= 6 and evidence[k]["good"] >= 8,
                    evidence[k]["sectors"],
                    evidence[k]["quality"],
                    evidence[k]["good"],
                ),
            )
            winner = groups[winner_index]
            if len(groups) > 1:
                own_e = evidence[own_index]
                win_e = evidence[winner_index]
                separation = abs(np.median(height[winner]) - np.median(height[own]))
                # No tocar transiciones ni una segunda superficie con cobertura
                # y evidencia propia. Resolver exclusivamente duplicación débil
                # dentro de la incertidumbre local y con solape tangencial.
                budget = min(0.8 * h, 2.0 * np.hypot(own_e["sigma"], win_e["sigma"]) + 0.25 * h)
                winner_xy = np.column_stack((x[winner], y[winner]))
                own_xy = np.column_stack((x[own], y[own]))
                tangent_gap = np.min(
                    np.linalg.norm(own_xy[:, None, :] - winner_xy[None, :, :], axis=2)
                )
                resolved = (
                    not transition
                    and winner_index != own_index
                    and win_e["good"] >= 8
                    and win_e["sectors"] >= 6
                    and win_e["extent"] >= 1.5 * h
                    and (own_e["good"] < 3 or own_e["sectors"] <= 3)
                    and win_e["quality"] >= own_e["quality"]
                    and 0.25 * h < separation <= budget
                    and tangent_gap < 0.6 * h
                )
                if resolved:
                    keep[i] = False
                    records["weak_competing_samples_removed"] += 1
                    continue
                records["ambiguous_samples_preserved"] += 1
            if transition:
                records["transition_normals_preserved"] += 1
                continue
            g = own
            if len(g) < 12:
                continue
            xx, yy = x[g] / h, y[g] / h
            angles = np.arctan2(yy, xx)
            sectors = np.unique(np.floor((angles + np.pi) * 4 / np.pi).astype(int) % 8)
            if len(sectors) < 6:
                continue
            A = np.column_stack((np.ones(len(g)), xx, yy, xx * xx, xx * yy, yy * yy))
            w = np.exp(-0.5 * (xx * xx + yy * yy) / 4.0)
            coef = None
            for _ in range(3):
                coef, _, rank, _ = np.linalg.lstsq(
                    A * np.sqrt(w)[:, None], z[g] * np.sqrt(w), rcond=1e-7
                )
                if rank < 6:
                    break
                residual = z[g] - A @ coef
                scale = max(0.06 * h, 1.4826 * np.median(np.abs(residual - np.median(residual))))
                w = np.exp(-0.5 * (xx * xx + yy * yy) / 4.0) * np.minimum(
                    1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-12)
                )
            if rank < 6 or scale > 0.2 * h:
                continue
            candidate = n[i] - coef[1] / h * a - coef[2] / h * b
            candidate /= max(np.linalg.norm(candidate), 1e-12)
            if candidate @ n[i] >= np.cos(np.deg2rad(15.0)):
                updated[i] = candidate
                records["normals_updated"] += 1
    records["weak_competing_samples_removed"] = int(np.count_nonzero(~keep))
    # No aceptar silenciosamente una pérdida extensa ni reintroducir las capas.
    if np.mean(keep) < 0.95:
        raise RuntimeError(
            "Paso 13: capas locales ambiguas afectan más del 5% de la nube; revisar fusión 11."
        )
    return keep, updated, records


def prepare_poisson_cloud(
    points,
    colors,
    normals,
    confidence,
    support,
    spacing,
    extent,
    args,
    *,
    evidence_strength=None,
    evidence_contract=False,
):
    """Crea el proxy de Poisson sin reabrir decisiones de capas de 11/12.

    Cuando existe contrato de evidencia, ``support`` representa soporte
    independiente y ``evidence_strength`` modula explícitamente cuánto puede
    influir cada observación. El paso 13 ya no vuelve a decidir cuál de dos
    capas es verdadera usando solo densidad/soporte local: esa decisión debe
    haberse validado contra poses independientes en 11 y preservado en 12.
    """
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    confidence = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0)
    support = np.maximum(np.asarray(support, dtype=np.float64), 0.0)
    if evidence_strength is None:
        evidence_strength = np.ones(len(points), dtype=np.float64)
    evidence_strength = np.clip(np.asarray(evidence_strength, dtype=np.float64), 0.02, 1.0)
    if len(evidence_strength) != len(points):
        raise ValueError("evidence_strength debe tener una entrada por punto.")

    maximum_points = max(1000, int(args.max_poisson_input_points))
    voxel = max(
        float(args.meshing_minimum_voxel_mm),
        float(spacing) * float(args.input_voxel_spacing_factor),
        float(extent) / 1024.0,
        1e-6,
    )
    effective_confidence = np.clip(confidence * evidence_strength, 0.0, 1.0)
    confidence_weight = np.power(
        np.clip(effective_confidence, 0.03, 1.0),
        max(0.0, float(args.meshing_confidence_exponent)),
    )
    support_weight = np.power(
        np.clip(np.maximum(support, 0.5) / 3.0, 0.25, 2.50),
        max(0.0, float(args.meshing_support_exponent)),
    )
    evidence_weight = np.power(
        evidence_strength,
        max(0.0, float(args.evidence_strength_exponent)),
    )

    original_count = len(points)
    keep, geometric_score, selection_report = select_supported_surface_samples(
        points, normals, effective_confidence, support, spacing
    )
    if bool(evidence_contract):
        layer_keep = np.ones(len(points), dtype=bool)
        layer_report = {
            "mode": "step11_12_evidence_contract_authoritative",
            "removed_points": 0,
            "reason": (
                "Las capas ambiguas ya fueron separadas/validadas en 11 y "
                "filtradas en 12; 13 no vuelve a adjudicar capas por geometría local."
            ),
        }
    else:
        layer_keep, normals, layer_report = resolve_local_layers_and_normals(
            points, normals, effective_confidence, support, spacing
        )
    keep &= layer_keep
    if np.count_nonzero(keep) < 500:
        raise RuntimeError(
            "Paso 13: menos de 500 muestras superan la selección de superficie; "
            "no se fuerza un mallado sobre evidencia insuficiente."
        )
    selection_report["layer_resolution"] = layer_report
    selection_report["removed_points"] = int(np.count_nonzero(~keep))
    selection_report["retained_fraction"] = float(np.mean(keep))
    selection_report["selection_distance_mm"] = robust_stats(
        query_tree(cKDTree(points[keep]), points, k=1)[0]
    )
    weights = np.clip(
        confidence_weight * support_weight * evidence_weight * geometric_score,
        1e-5,
        None,
    )[keep]
    points, colors, normals = points[keep], colors[keep], normals[keep]
    confidence, support = confidence[keep], support[keep]
    evidence_strength = evidence_strength[keep]
    print(
        f"[Paso 13] Superficie para Poisson: {len(points):,}/{original_count:,} "
        "muestras conservadas; evidencia multivista heredada respetada.",
        flush=True,
    )

    def aggregate(current_voxel):
        inverse = surface_voxel_groups(points, normals, float(current_voxel))
        cells = int(np.max(inverse)) + 1
        denominator = np.bincount(inverse, weights=weights, minlength=cells)

        def weighted_mean(values):
            values = np.asarray(values, dtype=np.float64)
            result = np.zeros((cells, values.shape[1]), dtype=np.float64)
            for column in range(values.shape[1]):
                result[:, column] = np.bincount(
                    inverse,
                    weights=weights * values[:, column],
                    minlength=cells,
                )
            result /= np.maximum(denominator[:, None], 1e-12)
            return result

        centers = weighted_mean(points)
        # Siempre escoger una observación real como posición del representante.
        distance = np.linalg.norm(points - centers[inverse], axis=1)
        ranking = np.lexsort((-weights, distance, inverse))
        _, first = np.unique(inverse[ranking], return_index=True)
        representatives = ranking[first]
        reduced_points = points[representatives].copy()
        reduced_colors = np.clip(weighted_mean(colors), 0.0, 255.0)
        reduced_normals = normals[representatives].copy()
        normal_length = np.linalg.norm(reduced_normals, axis=1)
        bad_normals = (~np.isfinite(normal_length)) | (normal_length < 1e-6)
        if np.any(bad_normals):
            representative = np.zeros(cells, dtype=np.int64)
            order = np.argsort(weights)[::-1]
            cell_order, first = np.unique(inverse[order], return_index=True)
            representative[cell_order] = order[first]
            reduced_normals[bad_normals] = normals[representative[bad_normals]]
            normal_length = np.linalg.norm(reduced_normals, axis=1)
        reduced_normals /= np.maximum(normal_length[:, None], 1e-12)
        reduced_confidence = np.bincount(
            inverse, weights=weights * confidence, minlength=cells
        ) / np.maximum(denominator, 1e-12)
        reduced_support = np.bincount(
            inverse, weights=weights * support, minlength=cells
        ) / np.maximum(denominator, 1e-12)
        reduced_evidence = np.bincount(
            inverse, weights=weights * evidence_strength, minlength=cells
        ) / np.maximum(denominator, 1e-12)
        return (
            reduced_points,
            reduced_colors,
            reduced_normals,
            reduced_confidence,
            reduced_support,
            reduced_evidence,
        )

    reduced = aggregate(voxel)
    for _ in range(16):
        if len(reduced[0]) <= maximum_points:
            break
        voxel *= 1.10
        reduced = aggregate(voxel)

    if len(reduced[0]) < 1000:
        cloud = make_point_cloud(points, colors, normals)
        return (
            cloud,
            0.0,
            {
                "mode": "full_cloud_safety_fallback",
                "points": int(len(points)),
                "voxel_mm": 0.0,
                "evidence_contract_used": bool(evidence_contract),
            },
        )

    cloud = make_point_cloud(reduced[0], reduced[1], reduced[2])
    report = {
        "mode": "evidence_preserving_surface_voxel_representatives",
        "surface_selection": selection_report,
        "input_points": int(original_count),
        "selected_points": int(len(points)),
        "incompatible_layers_averaged": False,
        "step13_layer_readjudication": False if evidence_contract else True,
        "evidence_contract_used": bool(evidence_contract),
        "points": int(len(reduced[0])),
        "voxel_mm": float(voxel),
        "reduction_ratio": float(len(reduced[0]) / max(len(points), 1)),
        "confidence": robust_stats(reduced[3]),
        "independent_support": robust_stats(reduced[4]),
        "evidence_strength": robust_stats(reduced[5]),
        "full_cloud_preserved_for_color_and_evaluation": True,
    }
    return cloud, float(voxel), report


def adaptive_depth(extent, spacing, args):
    if int(args.poisson_depth) > 0:
        depth = int(
            np.clip(
                int(args.poisson_depth),
                int(args.poisson_min_depth),
                int(args.poisson_max_depth),
            )
        )
        return depth, "cli_limited_by_safe_range"
    target_cell = max(
        float(args.poisson_min_cell_mm),
        float(args.poisson_target_cell_spacing_factor) * float(spacing),
    )
    raw = math.ceil(math.log2(max(float(extent) / target_cell, 2.0)))
    depth = int(
        np.clip(
            raw,
            int(args.poisson_min_depth),
            int(args.poisson_max_depth),
        )
    )
    return depth, "adaptive_spacing_limited"


@operacion("Reconstruir superficie mediante Poisson")
def create_poisson(pcd, depth, scale, threads):
    kwargs = {
        "depth": int(depth),
        "scale": float(scale),
        "linear_fit": True,
    }
    try:
        return o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            n_threads=int(threads),
            **kwargs,
        )
    except TypeError:
        return o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            **kwargs,
        )


def crop_to_cloud_bbox(mesh, points, margin_mm):
    minimum = np.min(points, axis=0) - float(margin_mm)
    maximum = np.max(points, axis=0) + float(margin_mm)
    try:
        return mesh.crop(o3d.geometry.AxisAlignedBoundingBox(minimum, maximum))
    except Exception:
        return mesh


def face_components_by_shared_edge(triangles):
    """Etiqueta componentes de caras usando aristas, no solo vértices."""
    triangles = np.asarray(triangles, dtype=np.int64)
    count = len(triangles)
    if count == 0:
        return 0, np.empty(0, dtype=np.int64)

    edges = np.vstack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )
    )
    edges.sort(axis=1)
    owners = np.tile(np.arange(count, dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    sorted_owners = owners[order]
    # Las apariciones contiguas de la misma arista enlazan sus caras. En una
    # arista no-manifold la cadena contigua también deja todo el grupo unido.
    shared = np.all(sorted_edges[1:] == sorted_edges[:-1], axis=1)
    first = sorted_owners[:-1][shared]
    second = sorted_owners[1:][shared]
    different = first != second
    first = first[different]
    second = second[different]
    if len(first):
        rows = np.concatenate((first, second))
        cols = np.concatenate((second, first))
        graph = coo_matrix(
            (
                np.ones(len(rows), dtype=np.uint8),
                (rows, cols),
            ),
            shape=(count, count),
        ).tocsr()
    else:
        graph = coo_matrix((count, count), dtype=np.uint8).tocsr()
    return connected_components(graph, directed=False)


def legacy_support_constrained_trim(
    mesh,
    cloud_points,
    confidence,
    support,
    proxy_spacing,
    confidence_available,
    support_available,
    args,
):
    """Implementación V4.1 conservada solo como referencia; no se ejecuta.

    Recorta la superficie implícita donde Poisson extrapoló sin mediciones.

    Se muestrean vértices, centroides y puntos medios de cada triángulo. Una
    cara solo puede sobrevivir si está dentro de la banda máxima y pertenece a
    una componente que contiene semillas cercanas y fiables. Así se conservan
    varias piezas legítimas, pero no láminas o puentes sin respaldo.
    """
    if not bool(args.support_trim):
        return mesh, {
            "enabled": False,
            "status": "disabled_by_cli",
            "triangles_before": int(len(mesh.triangles)),
            "triangles_after": int(len(mesh.triangles)),
        }

    original = copy.deepcopy(mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(vertices) == 0 or len(triangles) == 0:
        return mesh, {
            "enabled": True,
            "status": "empty_mesh",
            "triangles_before": int(len(triangles)),
            "triangles_after": int(len(triangles)),
        }

    automatic_strong = float(np.clip(2.25 * float(proxy_spacing), 1.25, 2.00))
    automatic_reject = float(np.clip(4.50 * float(proxy_spacing), 2.75, 3.50))
    strong_distance = (
        float(args.support_strong_distance_mm)
        if float(args.support_strong_distance_mm) > 0
        else automatic_strong
    )
    reject_distance = (
        float(args.support_reject_distance_mm)
        if float(args.support_reject_distance_mm) > 0
        else automatic_reject
    )
    reject_distance = max(reject_distance, strong_distance * 1.25)

    tree = cKDTree(np.asarray(cloud_points, dtype=np.float64))
    vertex_distance, _ = query_tree(tree, vertices, k=1)
    vertex_distance = np.asarray(vertex_distance, dtype=np.float64)

    tri_points = vertices[triangles]
    centroids = np.mean(tri_points, axis=1)
    midpoint_01 = 0.5 * (tri_points[:, 0] + tri_points[:, 1])
    midpoint_12 = 0.5 * (tri_points[:, 1] + tri_points[:, 2])
    midpoint_20 = 0.5 * (tri_points[:, 2] + tri_points[:, 0])
    face_samples = np.vstack((centroids, midpoint_01, midpoint_12, midpoint_20))
    sample_distance, sample_nearest = query_tree(tree, face_samples, k=1)
    sample_distance = np.asarray(sample_distance, dtype=np.float64).reshape(4, len(triangles))
    sample_nearest = np.asarray(sample_nearest, dtype=np.int64).reshape(4, len(triangles))
    centroid_nearest = sample_nearest[0]
    centroid_confidence = np.asarray(confidence, dtype=np.float64)[centroid_nearest]
    centroid_support = np.asarray(support, dtype=np.float64)[centroid_nearest]

    effective_min_confidence = float(args.support_min_confidence) if confidence_available else 0.0
    effective_min_support = float(args.support_min_views) if support_available else 1.0
    reliable_centroid = (centroid_confidence >= effective_min_confidence) & (
        centroid_support >= effective_min_support
    )

    def select_faces(current_reject):
        vertex_near_reject = np.sum(vertex_distance[triangles] <= float(current_reject), axis=1)
        sample_maximum = np.max(sample_distance, axis=0)
        candidate = (sample_maximum <= float(current_reject)) & (vertex_near_reject >= 2)
        vertex_near_strong = np.sum(vertex_distance[triangles] <= strong_distance, axis=1)
        seeds = (
            candidate
            & (sample_distance[0] <= strong_distance)
            & (vertex_near_strong >= 2)
            & reliable_centroid
        )
        reliability_fallback = False
        if not np.any(seeds):
            seeds = candidate & (sample_distance[0] <= strong_distance) & (vertex_near_strong >= 2)
            reliability_fallback = True

        candidate_ids = np.flatnonzero(candidate)
        keep = np.zeros(len(triangles), dtype=bool)
        records = []
        if len(candidate_ids):
            component_count, labels = face_components_by_shared_edge(triangles[candidate_ids])
            counts = np.bincount(labels, minlength=int(component_count))
            seed_counts = np.bincount(
                labels,
                weights=seeds[candidate_ids].astype(np.int64),
                minlength=int(component_count),
            ).astype(np.int64)
            seed_ratios = seed_counts / np.maximum(counts, 1)
            component_keep = (seed_counts >= int(args.support_component_min_seed_faces)) | (
                (seed_counts > 0) & (seed_ratios >= float(args.support_component_min_seed_ratio))
            )
            keep[candidate_ids] = component_keep[labels]
            for component in range(int(component_count)):
                records.append(
                    {
                        "component_id": int(component),
                        "candidate_faces": int(counts[component]),
                        "seed_faces": int(seed_counts[component]),
                        "seed_ratio": float(seed_ratios[component]),
                        "action": (
                            "keep_supported"
                            if component_keep[component]
                            else "remove_without_seed_support"
                        ),
                    }
                )
        return keep, seeds, reliability_fallback, records

    keep_faces, seed_faces, reliability_fallback, records = select_faces(reject_distance)
    removed_ratio = float(1.0 - np.count_nonzero(keep_faces) / max(len(triangles), 1))
    relaxed = False
    if removed_ratio > float(args.support_max_removed_triangle_ratio):
        relaxed_distance = min(
            max(reject_distance * 1.25, reject_distance + 0.50),
            4.0,
        )
        if relaxed_distance > reject_distance + 1e-9:
            reject_distance = float(relaxed_distance)
            keep_faces, seed_faces, reliability_fallback, records = select_faces(reject_distance)
            removed_ratio = float(1.0 - np.count_nonzero(keep_faces) / max(len(triangles), 1))
            relaxed = True

    kept_triangles = int(np.count_nonzero(keep_faces))
    if kept_triangles < 500:
        return original, {
            "enabled": True,
            "status": "safety_fallback_original_mesh_too_few_supported_faces",
            "strong_distance_mm": float(strong_distance),
            "reject_distance_mm": float(reject_distance),
            "triangles_before": int(len(triangles)),
            "triangles_after": int(len(triangles)),
            "candidate_triangles": kept_triangles,
            "removed_triangle_ratio": 0.0,
            "distance_band_relaxed": bool(relaxed),
        }

    mesh.remove_triangles_by_mask(~keep_faces)
    clean_mesh(mesh)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    finite_vertex_distance = vertex_distance[np.isfinite(vertex_distance)]
    return mesh, {
        "enabled": True,
        "status": "support_constrained_surface",
        "strong_distance_mm": float(strong_distance),
        "reject_distance_mm": float(reject_distance),
        "automatic_strong_distance_mm": float(automatic_strong),
        "automatic_reject_distance_mm": float(automatic_reject),
        "distance_band_relaxed": bool(relaxed),
        "confidence_evidence_available": bool(confidence_available),
        "support_evidence_available": bool(support_available),
        "effective_min_confidence": float(effective_min_confidence),
        "effective_min_support_views": float(effective_min_support),
        "reliability_seed_fallback": bool(reliability_fallback),
        "triangles_before": int(len(triangles)),
        "candidate_triangles": int(np.count_nonzero(keep_faces)),
        "seed_triangles": int(np.count_nonzero(seed_faces)),
        "triangles_after": int(len(mesh.triangles)),
        "removed_triangles": int(len(triangles) - len(mesh.triangles)),
        "removed_triangle_ratio": float(
            (len(triangles) - len(mesh.triangles)) / max(len(triangles), 1)
        ),
        "mesh_vertex_to_cloud_mm_before": robust_stats(finite_vertex_distance),
        "candidate_components": int(len(records)),
        "component_records_truncated": len(records) > 200,
        "component_records": records[:200],
        "sampling": "vertices_centroids_and_three_edge_midpoints",
    }


@operacion("Seleccionar componentes respaldadas")
def retain_supported_components(
    mesh,
    cloud_points,
    confidence,
    support,
    proxy_spacing,
    confidence_available,
    support_available,
    args,
):
    """Elimina componentes completas; nunca perfora una componente válida."""
    mesh = copy.deepcopy(mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(triangles) == 0:
        return mesh, {
            "enabled": bool(args.support_trim),
            "policy": "whole_components_only_no_individual_face_deletion",
            "components_before": 0,
            "components_removed": 0,
        }

    labels_raw, counts_raw, areas_raw = mesh.cluster_connected_triangles()
    labels = np.asarray(labels_raw, dtype=np.int64)
    counts = np.asarray(counts_raw, dtype=np.int64)
    areas = np.asarray(areas_raw, dtype=np.float64)
    total_triangles = max(int(np.sum(counts)), 1)
    total_area = max(float(np.sum(areas)), 1e-12)
    minimum_triangles = max(
        int(args.component_min_triangles),
        int(math.ceil(float(args.component_min_triangle_ratio) * total_triangles)),
    )

    automatic_anchor_distance = float(np.clip(1.50 * float(proxy_spacing), 0.90, 1.35))
    anchor_distance = (
        float(args.support_strong_distance_mm)
        if float(args.support_strong_distance_mm) > 0.0
        else automatic_anchor_distance
    )
    tree = cKDTree(np.asarray(cloud_points, dtype=np.float64))
    vertex_distance, nearest = query_tree(tree, vertices, k=1)
    nearest = np.asarray(nearest, dtype=np.int64)
    vertex_distance = np.asarray(vertex_distance, dtype=np.float64)
    vertex_confidence = np.asarray(confidence, dtype=np.float64)[nearest]
    vertex_support = np.asarray(support, dtype=np.float64)[nearest]
    minimum_confidence = float(args.support_min_confidence) if confidence_available else 0.0
    minimum_support = float(args.support_min_views) if support_available else 1.0
    strong_vertex = (
        (vertex_distance <= anchor_distance)
        & (vertex_confidence >= minimum_confidence)
        & (vertex_support >= minimum_support)
    )
    strong_face = np.sum(strong_vertex[triangles], axis=1) >= 2

    main = int(np.argmax(areas if np.any(areas > 0.0) else counts))
    keep_component = np.zeros(len(counts), dtype=bool)
    records = []
    order = np.argsort(labels, kind="stable")
    limits = np.searchsorted(labels[order], np.arange(len(counts) + 1))

    for component in range(len(counts)):
        face_ids = order[limits[component] : limits[component + 1]]
        vertex_ids = np.unique(triangles[face_ids].ravel())
        seed_faces = int(np.count_nonzero(strong_face[face_ids]))
        seed_ratio = float(seed_faces / max(len(face_ids), 1))
        area_ratio = float(areas[component] / total_area)
        extent = float(np.max(np.ptp(vertices[vertex_ids], axis=0))) if len(vertex_ids) else 0.0
        substantial = bool(
            counts[component] >= minimum_triangles
            or area_ratio >= 2.0 * float(args.component_min_triangle_ratio)
            or extent >= 6.0 * float(proxy_spacing)
        )
        evidenced = bool(
            seed_faces >= int(args.support_component_min_seed_faces)
            or (seed_faces >= 3 and seed_ratio >= float(args.support_component_min_seed_ratio))
        )
        # La componente dominante representa el cuerpo interpolado continuo y
        # nunca se recorta por ausencia local de muestras. Las secundarias se
        # conservan si combinan entidad geométrica y respaldo observacional.
        keep = bool(component == main or (evidenced and substantial))
        keep_component[component] = keep
        records.append(
            {
                "component_id": int(component),
                "triangles": int(counts[component]),
                "area_mm2": float(areas[component]),
                "area_ratio": area_ratio,
                "extent_mm": extent,
                "strong_seed_faces": seed_faces,
                "strong_seed_ratio": seed_ratio,
                "substantial_geometry": substantial,
                "observational_evidence": evidenced,
                "main_body_component": bool(component == main),
                "action": "keep_whole" if keep else "remove_whole_component",
            }
        )

    safety_fallback = False
    if not np.any(keep_component):
        keep_component[main] = True
        safety_fallback = True
    if not bool(args.support_trim):
        keep_component[:] = True

    remove_faces = ~keep_component[labels]
    if np.any(remove_faces):
        mesh.remove_triangles_by_mask(remove_faces)
        clean_mesh(mesh)

    return mesh, {
        "enabled": bool(args.support_trim),
        "policy": "whole_components_only_no_individual_face_deletion",
        "individual_faces_removed": 0,
        "interpolated_connected_regions_preserved": True,
        "anchor_distance_mm": float(anchor_distance),
        "automatic_anchor_distance_mm": float(automatic_anchor_distance),
        "components_before": int(len(counts)),
        "components_retained": int(np.count_nonzero(keep_component)),
        "components_removed": int(np.count_nonzero(~keep_component)),
        "triangles_before": int(len(triangles)),
        "triangles_after": int(len(mesh.triangles)),
        "minimum_triangles_used": int(minimum_triangles),
        "safety_fallback_main_component": bool(safety_fallback),
        "records_truncated": len(records) > 200,
        "records": records[:200],
    }


@operacion("Regularizar regiones débiles conectadas")
def local_observation_trend(vertices, triangles, cloud, confidence, support, spacing):
    """Predicción cuadrática a dos escalas; el soporte es agregado, no poses nuevas.

    La dispersión es un proxy empírico, no incertidumbre calibrada del sensor.
    No se extrapola fuera de vecindarios que rodean al vértice.
    """
    n = len(vertices)
    shift = np.zeros_like(vertices)
    eligible = np.zeros(n, dtype=bool)
    sigma_out = np.zeros(n)
    if len(cloud) < 32:
        return shift, eligible, sigma_out
    h = max(float(spacing), 1e-6)
    tree = cKDTree(cloud)
    cross = np.cross(
        vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
        vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
    )
    normal = np.zeros_like(vertices)
    for j in range(3):
        np.add.at(normal, triangles[:, j], cross)
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-12)
    # Caras de ambos lados de una arista fuerte no pueden compartir ajuste.
    unit = cross / np.maximum(np.linalg.norm(cross, axis=1)[:, None], 1e-12)
    sharp = np.zeros(n, dtype=bool)
    for j in range(3):
        bad = np.einsum("ij,ij->i", unit, normal[triangles[:, j]]) < np.cos(np.deg2rad(35))
        sharp[triangles[bad].ravel()] = True
    for start in range(0, n, 1024):
        stop = min(n, start + 1024)
        ids = np.arange(start, stop)
        distance, neighbors = query_tree(tree, vertices[ids], k=32)
        delta = cloud[neighbors] - vertices[ids, None, :]
        nz = normal[ids]
        axis = np.zeros_like(nz)
        axis[np.arange(len(ids)), np.argmin(np.abs(nz), axis=1)] = 1.0
        nx = np.cross(nz, axis)
        nx /= np.maximum(np.linalg.norm(nx, axis=1)[:, None], 1e-12)
        ny = np.cross(nz, nx)
        u = np.einsum("bki,bi->bk", delta, nx) / h
        v = np.einsum("bki,bi->bk", delta, ny) / h
        z = np.einsum("bki,bi->bk", delta, nz) / h
        predictions = []
        scatters = []
        valid_scales = []
        for k in (16, 32):
            uu, vv, zz = u[:, :k], v[:, :k], z[:, :k]
            design = np.stack((np.ones_like(uu), uu, vv, uu * uu, uu * vv, vv * vv), axis=2)
            base = np.clip(confidence[neighbors[:, :k]], 0.05, 1.0) * np.exp(
                -distance[:, :k] ** 2 / (8 * h * h)
            )
            weights = base.copy()
            coef = np.zeros((len(ids), 6))
            for _ in range(3):
                gram = np.einsum("bki,bk,bkj->bij", design, weights, design)
                rhs = np.einsum("bki,bk,bk->bi", design, weights, zz)
                ridge = 1e-7 * np.maximum(np.trace(gram, axis1=1, axis2=2), 1.0)
                coef = np.linalg.solve(gram + ridge[:, None, None] * np.eye(6), rhs[..., None])[
                    ..., 0
                ]
                residual = zz - np.einsum("bki,bi->bk", design, coef)
                scale = np.maximum(
                    0.08,
                    1.4826
                    * np.median(np.abs(residual - np.median(residual, axis=1)[:, None]), axis=1),
                )
                weights = base * np.minimum(
                    1.0, 1.5 * scale[:, None] / np.maximum(np.abs(residual), 1e-9)
                )
            angle = (np.arctan2(vv, uu) + np.pi) * (8.0 / (2 * np.pi))
            bins = np.minimum(angle.astype(int), 7)
            sectors = sum(np.any(bins == j, axis=1).astype(int) for j in range(8))
            eigen = np.linalg.eigvalsh(gram)
            observed = (confidence[neighbors[:, :k]] >= 0.65) & (support[neighbors[:, :k]] >= 2)
            ok = (
                (sectors >= 6)
                & (distance[:, k - 1] <= 4 * h)
                & (eigen[:, 0] > 1e-5 * np.maximum(eigen[:, -1], 1e-12))
                & (np.mean(observed, axis=1) >= 0.65)
                & (scale <= 0.35)
                & (np.hypot(coef[:, 1], coef[:, 2]) <= 0.5)
                & (np.mean(np.abs(residual) <= 2.5 * scale[:, None], axis=1) >= 0.85)
            )
            ordered = np.sort(residual, axis=1)
            gaps = np.diff(ordered, axis=1)[:, 3 : k - 4]
            separated_layers = np.max(gaps, axis=1) > np.maximum(0.20, 2.0 * scale)
            ok &= ~separated_layers
            predictions.append(coef[:, 0] * h)
            scatters.append(scale * h)
            valid_scales.append(ok)
        a, b = predictions
        sigma = np.maximum(*scatters)
        good = (
            valid_scales[0]
            & valid_scales[1]
            & ~sharp[ids]
            & (a * b > 0)
            & (np.abs(a - b) <= 0.5 * sigma)
            & (np.minimum(np.abs(a), np.abs(b)) > np.maximum(0.08 * h, sigma))
            & (np.maximum(np.abs(a), np.abs(b)) <= 2.5 * sigma)
        )
        budget = np.minimum(0.35 * h, 1.5 * sigma)
        amount = np.clip(0.5 * (0.5 * a + 0.5 * b), -budget, budget)
        shift[ids[good]] = amount[good, None] * nz[good]
        eligible[ids] = good
        sigma_out[ids] = sigma
    return shift, eligible, sigma_out


def guard_region_motion(mesh, original, proposed, triangles):
    """Rechazo local acumulativo; conserva conectividad y pares preexistentes."""
    baseline_cross = np.cross(
        original[triangles[:, 1]] - original[triangles[:, 0]],
        original[triangles[:, 2]] - original[triangles[:, 0]],
    )
    area = np.linalg.norm(baseline_cross, axis=1)
    q = proposed.copy()
    records = []
    if not np.any(np.linalg.norm(q - original, axis=1) > 1e-12):
        return original.copy(), {"applied": False, "reason": "no_motion"}

    def pairs(m):
        return {
            tuple(sorted(map(int, pair)))
            for pair in np.asarray(m.get_self_intersecting_triangles()).reshape(-1, 2)
        }

    try:
        base_pairs = pairs(mesh)
        for iteration in range(8):
            moved = np.linalg.norm(q - original, axis=1) > 1e-12
            affected = np.any(moved[triangles], axis=1)
            c = np.cross(
                q[triangles[:, 1]] - q[triangles[:, 0]], q[triangles[:, 2]] - q[triangles[:, 0]]
            )
            bad = np.flatnonzero(
                affected
                & (
                    (np.einsum("ij,ij->i", c, baseline_cross) <= 0)
                    | (np.linalg.norm(c, axis=1) < 0.2 * area)
                    | ~np.all(np.isfinite(c), axis=1)
                )
            )
            trial = copy.deepcopy(mesh)
            trial.vertices = o3d.utility.Vector3dVector(q)
            new_pairs = pairs(trial) - base_pairs
            if not len(bad) and not new_pairs:
                return q, {"applied": True, "local_rollbacks": records, "new_intersection_pairs": 0}
            faces = set(map(int, bad))
            for pair in new_pairs:
                faces.update(pair)
            reset = np.unique(triangles[list(faces)])
            q[reset] = original[reset]
            records.append(
                {
                    "iteration": iteration + 1,
                    "reset_vertices": int(len(reset)),
                    "new_pairs": len(new_pairs),
                }
            )
        return original.copy(), {
            "applied": False,
            "reason": "local_guard_exhausted",
            "local_rollbacks": records,
        }
    except Exception as exc:
        print(
            f"[Paso 13] Regularización conservada sin cambios: no se pudo verificar seguridad: {exc}",
            flush=True,
        )
        return original.copy(), {
            "applied": False,
            "reason": "verification_unavailable",
            "detail": str(exc),
        }


def restricted_weak_region_smoothing(
    mesh,
    cloud_points,
    confidence,
    support,
    proxy_spacing,
    confidence_available,
    support_available,
    args,
):
    """Taubin ponderado: suaviza evidencia débil y fija anclajes fuertes."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(vertices) == 0 or len(triangles) == 0:
        return mesh, {"enabled": False, "status": "empty_mesh"}

    automatic_anchor_distance = float(np.clip(1.50 * float(proxy_spacing), 0.90, 1.35))
    automatic_full_distance = float(np.clip(4.00 * float(proxy_spacing), 2.25, 3.25))
    anchor_distance = (
        float(args.support_strong_distance_mm)
        if float(args.support_strong_distance_mm) > 0.0
        else automatic_anchor_distance
    )
    full_distance = (
        float(args.support_reject_distance_mm)
        if float(args.support_reject_distance_mm) > 0.0
        else automatic_full_distance
    )
    full_distance = max(full_distance, 1.50 * anchor_distance)
    maximum_displacement = (
        float(args.weak_region_max_displacement_mm)
        if float(args.weak_region_max_displacement_mm) > 0.0
        else float(np.clip(2.25 * float(proxy_spacing), 1.00, 1.75))
    )

    tree = cKDTree(np.asarray(cloud_points, dtype=np.float64))
    vertex_distance, nearest = query_tree(tree, vertices, k=1)
    vertex_distance = np.asarray(vertex_distance, dtype=np.float64)
    nearest = np.asarray(nearest, dtype=np.int64)
    vertex_confidence = np.asarray(confidence, dtype=np.float64)[nearest]
    vertex_support = np.asarray(support, dtype=np.float64)[nearest]
    anchor_confidence = float(args.weak_region_anchor_confidence) if confidence_available else 0.0
    anchor_support = float(args.weak_region_anchor_support) if support_available else 1.0
    anchors = (
        (vertex_distance <= anchor_distance)
        & (vertex_confidence >= anchor_confidence)
        & (vertex_support >= anchor_support)
    )

    transition = np.clip(
        (vertex_distance - anchor_distance) / max(full_distance - anchor_distance, 1e-9),
        0.0,
        1.0,
    )
    distance_weight = transition * transition * (3.0 - 2.0 * transition)
    confidence_weakness = np.clip(
        (anchor_confidence - vertex_confidence) / max(anchor_confidence, 1e-9),
        0.0,
        1.0,
    )
    support_weakness = np.clip(
        (anchor_support - vertex_support) / max(anchor_support, 1e-9),
        0.0,
        1.0,
    )
    reliability_weakness = np.maximum(confidence_weakness, support_weakness)
    weights = np.maximum(
        distance_weight,
        float(args.weak_region_reliability_weight) * reliability_weakness,
    )
    trend = np.zeros_like(vertices)
    local_candidates = np.zeros(len(vertices), dtype=bool)
    local_sigma = np.zeros(len(vertices))
    if args.weak_region_smoothing and confidence_available and support_available:
        print(
            "[Paso 13] Evaluando respaldo de superficie a dos escalas, sin imponer una figura.",
            flush=True,
        )
        trend, local_candidates, local_sigma = local_observation_trend(
            vertices,
            triangles,
            np.asarray(cloud_points),
            np.asarray(confidence),
            np.asarray(support),
            proxy_spacing,
        )
    anchors[local_candidates] = False
    weights[anchors] = 0.0

    edges = np.vstack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )
    )
    edges.sort(axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_vertices = np.unique(unique_edges[counts == 1])
    if len(boundary_vertices):
        weights[boundary_vertices] = 0.0
        local_candidates[boundary_vertices] = False

    protected = np.unique(unique_edges[counts != 2])
    weights[protected] = 0.0
    local_candidates[protected] = False
    # Coincidentes no se separan aunque sus índices topológicos sean distintos.
    _, inverse, duplicate_counts = np.unique(
        vertices, axis=0, return_inverse=True, return_counts=True
    )
    duplicated = duplicate_counts[inverse] > 1
    weights[duplicated] = 0.0
    local_candidates[duplicated] = False
    rows = np.concatenate((unique_edges[:, 0], unique_edges[:, 1]))
    cols = np.concatenate((unique_edges[:, 1], unique_edges[:, 0]))
    adjacency = coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    original = vertices.copy()
    current = vertices.copy()
    iterations = max(0, int(args.weak_region_iterations))
    lambda_value = float(np.clip(args.weak_region_lambda, 0.0, 0.60))
    mu_value = float(np.clip(args.weak_region_mu, -0.65, 0.0))

    def apply_step(values, coefficient):
        neighbor_mean = adjacency.dot(values) / np.maximum(degree[:, None], 1.0)
        updated = values + coefficient * weights[:, None] * (neighbor_mean - values)
        updated[anchors] = original[anchors]
        delta = updated - original
        length = np.linalg.norm(delta, axis=1)
        scale = np.minimum(
            1.0,
            maximum_displacement / np.maximum(length, 1e-12),
        )
        updated = original + delta * scale[:, None]
        updated[anchors] = original[anchors]
        return updated

    applied = bool(
        args.weak_region_smoothing
        and iterations > 0
        and (np.any(weights > 0.0) or np.any(local_candidates))
    )
    motion_guard = {"applied": False, "reason": "disabled_or_no_candidates"}
    if applied:
        for _ in range(iterations):
            current = apply_step(current, lambda_value)
            current = apply_step(current, mu_value)
        current[local_candidates] = original[local_candidates] + trend[local_candidates]
        print(
            f"[Paso 13] Verificando regularización local: {np.count_nonzero(local_candidates):,} candidatos de superficie.",
            flush=True,
        )
        current, motion_guard = guard_region_motion(mesh, original, current, triangles)
        mesh.vertices = o3d.utility.Vector3dVector(current)
        applied = bool(np.any(np.linalg.norm(current - original, axis=1) > 1e-12))

    displacement = np.linalg.norm(current - original, axis=1)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh, {
        "enabled": bool(args.weak_region_smoothing),
        "applied": bool(applied),
        "method": "restricted_taubin_and_two_scale_observation_trend_with_local_guards",
        "local_surface_candidates": int(np.count_nonzero(local_candidates)),
        "local_surface_moved": int(np.count_nonzero(local_candidates & (displacement > 1e-12))),
        "uncertainty_source": "empirical_local_fit_residual_not_sensor_covariance",
        "local_sigma_mm": robust_stats(local_sigma[local_candidates]),
        "motion_guard": motion_guard,
        "topology_changed": False,
        "anchor_distance_mm": float(anchor_distance),
        "full_smoothing_distance_mm": float(full_distance),
        "automatic_anchor_distance_mm": float(automatic_anchor_distance),
        "automatic_full_smoothing_distance_mm": float(automatic_full_distance),
        "anchor_min_confidence": float(anchor_confidence),
        "anchor_min_support_views": float(anchor_support),
        "anchor_vertices": int(np.count_nonzero(anchors)),
        "weak_vertices": int(np.count_nonzero(weights > 0.0)),
        "boundary_vertices_protected": int(len(boundary_vertices)),
        "iterations": int(iterations),
        "lambda": float(lambda_value),
        "mu": float(mu_value),
        "maximum_displacement_mm": float(maximum_displacement),
        "actual_displacement_mm": robust_stats(displacement),
        "vertex_to_cloud_mm_before": robust_stats(vertex_distance),
        "normal_recalculation": True,
    }


def retain_nontrivial_components(mesh, min_triangles, min_ratio):
    """Retira solo islas inequívocamente pequeñas."""
    mesh = copy.deepcopy(mesh)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(triangles) == 0:
        return mesh, {"before": 0, "removed": 0, "after": 0}
    try:
        labels, counts, areas = mesh.cluster_connected_triangles()
    except Exception:
        return mesh, {"before": None, "removed": 0, "after": None}

    labels = np.asarray(labels, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    areas = np.asarray(areas, dtype=np.float64)
    if len(counts) <= 1:
        count = int(len(counts))
        return mesh, {"before": count, "removed": 0, "after": count}

    main = int(np.argmax(areas if np.any(areas > 0) else counts))
    total = max(int(np.sum(counts)), 1)
    threshold = max(
        int(min_triangles),
        int(math.ceil(float(min_ratio) * total)),
    )
    keep_components = counts >= threshold
    keep_components[main] = True
    remove_faces = ~keep_components[labels]
    removed = int(np.count_nonzero(~keep_components))
    if np.any(remove_faces):
        mesh.remove_triangles_by_mask(remove_faces)
        clean_mesh(mesh)

    return mesh, {
        "before": int(len(counts)),
        "removed": removed,
        "after": int(np.count_nonzero(keep_components)),
        "minimum_triangles_used": int(threshold),
    }


def colorize(mesh, cloud_points, cloud_colors):
    """Asigna a cada vértice el color del punto de referencia más cercano."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        return mesh
    _, indices = query_tree(cKDTree(cloud_points), vertices, k=1)
    values = cloud_colors[np.asarray(indices, dtype=np.int64)]
    mesh.vertex_colors = o3d.utility.Vector3dVector(values.astype(np.float64) / 255.0)
    return mesh


def edge_statistics(triangles, n_vertices):
    """Resume incidencia, bordes y componentes de frontera de la triangulación."""
    triangles = np.asarray(triangles, dtype=np.int64)
    if len(triangles) == 0:
        return {
            "unique_edges": 0,
            "boundary_edges": 0,
            "boundary_edge_ratio": 0.0,
            "boundary_components": 0,
            "nonmanifold_edges": 0,
            "nonmanifold_edge_ratio": 0.0,
        }
    edges = np.vstack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )
    )
    edges.sort(axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    nonmanifold = unique[counts > 2]

    boundary_count = 0
    if len(boundary):
        vertices = np.unique(boundary)
        remap = np.full(int(n_vertices), -1, dtype=np.int64)
        remap[vertices] = np.arange(len(vertices), dtype=np.int64)
        rows = np.concatenate((remap[boundary[:, 0]], remap[boundary[:, 1]]))
        cols = np.concatenate((remap[boundary[:, 1]], remap[boundary[:, 0]]))
        graph = coo_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
            shape=(len(vertices), len(vertices)),
        ).tocsr()
        boundary_count, _ = connected_components(graph, directed=False)

    return {
        "unique_edges": int(len(unique)),
        "boundary_edges": int(len(boundary)),
        "boundary_edge_ratio": float(len(boundary) / max(len(unique), 1)),
        "boundary_components": int(boundary_count),
        "nonmanifold_edges": int(len(nonmanifold)),
        "nonmanifold_edge_ratio": float(len(nonmanifold) / max(len(unique), 1)),
    }


def topology_fast(mesh):
    """Resume la topología y deja la prueba de intersecciones para el paso 16."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    report = {
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "self_intersecting": None,
        "self_intersection_check": "deferred_to_step_16",
    }
    report.update(edge_statistics(triangles, len(vertices)))
    try:
        _, counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(counts, dtype=np.int64)
        report["connected_components"] = int(len(counts))
        report["largest_component_triangle_ratio"] = (
            float(np.max(counts) / max(np.sum(counts), 1)) if len(counts) else None
        )
    except Exception:
        report["connected_components"] = None
        report["largest_component_triangle_ratio"] = None

    checks = (
        ("edge_manifold", lambda: mesh.is_edge_manifold(True)),
        ("vertex_manifold", mesh.is_vertex_manifold),
        ("orientable", mesh.is_orientable),
    )
    for key, function in checks:
        try:
            report[key] = bool(function())
        except Exception:
            report[key] = None
    return report


def mesh_distance(mesh, query_points):
    """Mide distancia a la superficie; si falla raycasting, usa los vértices como respaldo."""
    try:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        query = o3d.core.Tensor(np.asarray(query_points, dtype=np.float32))
        return scene.compute_distance(query, nthreads=0).numpy().astype(np.float64)
    except Exception:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if len(vertices) == 0:
            return np.full(len(query_points), np.inf, dtype=np.float64)
        distances, _ = query_tree(cKDTree(vertices), query_points, k=1)
        return np.asarray(distances, dtype=np.float64)


@operacion("Evaluar candidato de reconstrucción")
def evaluate(mesh, cloud_points, gate_mm, cloud_samples, mesh_samples, name="candidate"):
    """Mide cobertura y distancias entre la malla candidata y la nube de referencia."""
    rng = np.random.default_rng(1303)
    query = cloud_points
    if len(query) > int(cloud_samples):
        query = query[rng.choice(len(query), int(cloud_samples), replace=False)]
    cloud_to_mesh = mesh_distance(mesh, query)
    coverage = float(np.mean(cloud_to_mesh <= float(gate_mm)))

    sample_count = min(
        int(mesh_samples),
        max(5000, len(np.asarray(mesh.vertices))),
    )
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    faces = vertices[triangles]
    areas = np.linalg.norm(np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("Candidato sin área finita para evaluar.")
    sampled_faces = faces[rng.choice(len(faces), sample_count, p=areas / areas.sum())]
    uv = rng.random((sample_count, 2))
    root_u = np.sqrt(uv[:, 0])
    sampled_points = ((1-root_u)[:, None] * sampled_faces[:, 0]
                      + (root_u*(1-uv[:, 1]))[:, None] * sampled_faces[:, 1]
                      + (root_u*uv[:, 1])[:, None] * sampled_faces[:, 2])
    mesh_to_cloud, _ = query_tree(
        cKDTree(cloud_points),
        np.asarray(sampled_points, dtype=np.float64),
        k=1,
    )
    cloud_extent = np.ptp(cloud_points, axis=0)
    mesh_extent = np.ptp(np.asarray(mesh.vertices), axis=0)
    ratio = np.divide(
        mesh_extent,
        cloud_extent,
        out=np.full(3, np.nan),
        where=cloud_extent > 1e-9,
    )
    return {
        "name": str(name),
        "coverage_within_gate": coverage,
        "coverage_gate_mm": float(gate_mm),
        "cloud_to_mesh_mm": robust_stats(cloud_to_mesh),
        "mesh_to_cloud_mm": robust_stats(mesh_to_cloud),
        "bbox_extent_ratio_xyz": ratio.astype(float).tolist(),
        "topology": topology_fast(mesh),
        "expensive_self_intersection_check": "deferred_to_step_16",
    }


def _finite_metric(evaluation, section, key, default=float("inf")):
    try:
        value = evaluation.get(section, {}).get(key)
        value = float(value)
        return value if np.isfinite(value) else float(default)
    except Exception:
        return float(default)


def poisson_extrapolation_risk(evaluation, spacing, args):
    """Detecta extrapolación implícita sin asumir una forma geométrica concreta."""
    h = max(float(spacing), 1e-6)
    p90 = _finite_metric(evaluation, "mesh_to_cloud_mm", "p90")
    p95 = _finite_metric(evaluation, "mesh_to_cloud_mm", "p95")
    ratios = np.asarray(evaluation.get("bbox_extent_ratio_xyz", []), dtype=float)
    expansion = float(np.nanmax(np.maximum(ratios - 1.0, 0.0))) if ratios.size else float("inf")
    reasons = []
    if p90 > float(args.poisson_risk_p90_spacing_factor) * h:
        reasons.append(f"mesh_to_cloud_p90={p90:.3f}mm>{args.poisson_risk_p90_spacing_factor:.2f}*spacing")
    if p95 > float(args.poisson_risk_p95_spacing_factor) * h:
        reasons.append(f"mesh_to_cloud_p95={p95:.3f}mm>{args.poisson_risk_p95_spacing_factor:.2f}*spacing")
    if expansion > float(args.poisson_risk_bbox_expansion_ratio):
        reasons.append(f"bbox_expansion={expansion:.3%}>{args.poisson_risk_bbox_expansion_ratio:.3%}")
    topo = evaluation.get("topology", {})
    if int(topo.get("nonmanifold_edges") or 0) > 0:
        reasons.append(f"nonmanifold_edges={int(topo.get('nonmanifold_edges') or 0)}")
    return bool(reasons), {
        "risk": bool(reasons),
        "reasons": reasons,
        "mesh_to_cloud_p90_mm": p90,
        "mesh_to_cloud_p95_mm": p95,
        "maximum_bbox_expansion_ratio": expansion,
        "spacing_mm": h,
    }


def candidate_selection_score(evaluation, spacing):
    """Menor es mejor; prioriza evidencia observada en ambos sentidos.

    V5.6 evita que el número bruto de bucles de borde domine la decisión. Un
    método observacional como BPA puede dejar muchos huecos pequeños que los
    pasos topológicos posteriores pueden tratar; eso no debe pesar más que una
    malla implícita que se aleja varios milímetros de las observaciones.
    """
    h = max(float(spacing), 1e-6)
    coverage = float(evaluation.get("coverage_within_gate") or 0.0)
    m50 = _finite_metric(evaluation, "mesh_to_cloud_mm", "median")
    p90 = _finite_metric(evaluation, "mesh_to_cloud_mm", "p90")
    p95 = _finite_metric(evaluation, "mesh_to_cloud_mm", "p95")
    c_p90 = _finite_metric(evaluation, "cloud_to_mesh_mm", "p90")
    c_p95 = _finite_metric(evaluation, "cloud_to_mesh_mm", "p95")

    ratios = np.asarray(evaluation.get("bbox_extent_ratio_xyz", []), dtype=float)
    bbox_penalty = 0.0
    if ratios.size:
        bbox_penalty = float(
            np.nansum(np.maximum(np.abs(ratios - 1.0) - 0.01, 0.0))
        )

    topo = evaluation.get("topology", {})
    largest = topo.get("largest_component_triangle_ratio")
    largest = float(largest) if largest is not None and np.isfinite(largest) else 0.0
    nonmanifold = int(topo.get("nonmanifold_edges") or 0)
    components = topo.get("connected_components")
    components = int(components) if components is not None else 999
    boundary_ratio = float(topo.get("boundary_edge_ratio") or 0.0)
    boundary_components = topo.get("boundary_components")
    boundary_components = int(boundary_components) if boundary_components is not None else 0

    score = (
        10.0 * max(0.0, 1.0 - coverage)
        + 0.18 * (m50 / h)
        + 0.32 * (p90 / h)
        + 0.16 * (p95 / h)
        + 0.07 * (c_p90 / h)
        + 0.07 * (c_p95 / h)
        + 3.0 * bbox_penalty
        + 0.8 * max(0.0, 0.90 - largest)
        + 0.0015 * min(nonmanifold, 200)
        + 0.03 * max(0, components - 1)
        + 0.55 * min(max(boundary_ratio, 0.0), 1.0)
        # Penalización logarítmica: muchos bucles pequeños no equivalen a una
        # extrapolación geométrica grande.
        + 0.04 * math.log1p(max(0, boundary_components - 1))
    )
    return float(score)


def candidate_observational_fidelity_gate(evaluation, spacing, args):
    """Gate duro de fidelidad; no depende de la forma ni del método.

    Una topología reparable puede pasar a 14, pero una superficie situada varios
    espaciamientos lejos de todas las observaciones no se considera un candidato
    geométrico válido.
    """
    h = max(float(spacing), 1e-6)
    p90 = _finite_metric(evaluation, "mesh_to_cloud_mm", "p90")
    p95 = _finite_metric(evaluation, "mesh_to_cloud_mm", "p95")
    ratios = np.asarray(evaluation.get("bbox_extent_ratio_xyz", []), dtype=float)
    expansion = (
        float(np.nanmax(np.maximum(ratios - 1.0, 0.0)))
        if ratios.size else float("inf")
    )
    failures = []
    if p90 > float(args.poisson_risk_p90_spacing_factor) * h:
        failures.append("mesh_to_cloud_p90_exceeds_spacing_gate")
    if p95 > float(args.poisson_risk_p95_spacing_factor) * h:
        failures.append("mesh_to_cloud_p95_exceeds_spacing_gate")
    if expansion > float(args.poisson_risk_bbox_expansion_ratio):
        failures.append("bbox_expansion_exceeds_gate")
    return not failures, {
        "passed": not failures,
        "failures": failures,
        "mesh_to_cloud_p90_mm": float(p90),
        "mesh_to_cloud_p95_mm": float(p95),
        "spacing_mm": float(h),
        "p90_limit_mm": float(args.poisson_risk_p90_spacing_factor) * h,
        "p95_limit_mm": float(args.poisson_risk_p95_spacing_factor) * h,
        "maximum_bbox_expansion_ratio": float(expansion),
        "bbox_limit_ratio": float(args.poisson_risk_bbox_expansion_ratio),
    }


def choose_surface_candidate(candidates, spacing, args):
    """Selecciona únicamente entre candidatos geométricamente admisibles.

    V5.7 elimina el ganador Poisson por defecto. Si todos los métodos fallan
    fidelidad observacional o presentan fragmentación extrema, se rechaza la
    reconstrucción y se obliga a volver a la nube.
    """
    if not candidates:
        raise RuntimeError("Paso 13: no hay candidatos de superficie para seleccionar.")

    report = {
        "policy": "hard_observational_fidelity_then_quality_score",
        "minimum_absolute_coverage": float(args.candidate_min_absolute_coverage),
        "minimum_largest_component_ratio": float(args.candidate_min_largest_component_ratio),
        "maximum_components": int(args.candidate_max_components),
        "maximum_boundary_edge_ratio": float(args.candidate_max_boundary_edge_ratio),
        "candidates": {},
    }
    eligible_candidates = []
    for name, data in candidates.items():
        ev = data[1]
        score = candidate_selection_score(ev, spacing)
        cov = float(ev.get("coverage_within_gate") or 0.0)
        topo = ev.get("topology", {})
        largest = topo.get("largest_component_triangle_ratio")
        largest = float(largest) if largest is not None and np.isfinite(largest) else 0.0
        components = topo.get("connected_components")
        components = int(components) if components is not None else 999
        boundary_ratio = float(topo.get("boundary_edge_ratio") or 0.0)
        fidelity_ok, fidelity = candidate_observational_fidelity_gate(ev, spacing, args)
        reasons = []
        if not fidelity_ok:
            reasons.extend(fidelity["failures"])
        if cov < float(args.candidate_min_absolute_coverage):
            reasons.append("coverage_below_absolute_minimum")
        if largest < float(args.candidate_min_largest_component_ratio):
            reasons.append("extreme_fragmentation_largest_component")
        if components > int(args.candidate_max_components):
            reasons.append("too_many_connected_components")
        if boundary_ratio > float(args.candidate_max_boundary_edge_ratio):
            reasons.append("excessive_open_boundary_ratio")
        eligible = not reasons
        report["candidates"][name] = {
            "score": float(score),
            "coverage": float(cov),
            "largest_component_triangle_ratio": float(largest),
            "connected_components": int(components),
            "boundary_edge_ratio": float(boundary_ratio),
            "observational_fidelity": fidelity,
            "eligible": bool(eligible),
            "ineligibility_reasons": reasons,
        }
        if eligible:
            eligible_candidates.append((float(score), name, data))

    if not eligible_candidates:
        compact = {
            k: v["ineligibility_reasons"]
            for k, v in report["candidates"].items()
        }
        error = RuntimeError(
            "Paso 13 rechazado: ningún candidato de superficie conserva a la vez "
            "fidelidad observacional, cobertura y conectividad suficientes. "
            "No se seleccionará Poisson solo por ser continuo. Diagnóstico: "
            + json.dumps(compact, ensure_ascii=False)
        )
        error.selection_report = report
        raise error

    eligible_candidates.sort(key=lambda item: item[0])
    best_score, best_name, best_data = eligible_candidates[0]
    report["selected"] = best_name
    report["selected_score"] = float(best_score)
    report["selection_note"] = (
        "Ningún candidato puede ganar si extrapola más allá de los gates relativos "
        "al espaciado observado. La continuidad topológica nunca anula fidelidad."
    )
    return best_name, best_data, report


def create_bpa_fallback(pcd, spacing, voxel, factors_text):
    """Reconstruye con Ball Pivoting usando radios derivados del espaciado."""
    factors = [float(value) for value in factors_text.split(",") if value.strip()]
    radii = [max(float(spacing) * factor, float(voxel) * 0.75) for factor in factors]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii),
    )
    return clean_mesh(mesh), radii


@operacion("Replegar fronteras extrapoladas con evidencia local y guardas")
def retract_observation_boundaries(
    mesh,
    points,
    confidence,
    support,
    spacing,
    cloud_normals=None,
    position_uncertainty=None,
    search_margin_mm=0.0,
    enabled=True,
):
    """Repliega extremos abiertos con evidencia local, sin recortar caras.

    Se trabaja en el marco tangente de cada frontera, nunca en un eje global.
    La ausencia de puntos por sí sola no autoriza un movimiento: se exige una
    franja observada detrás del límite, continuidad entre vecinos de frontera
    y ausencia de observaciones compatibles por delante. Las zonas ambiguas
    permanecen inmóviles. La incertidumbre ausente se aproxima por muestreo y
    dispersión normal local; no se presenta como covarianza calibrada.
    """
    import heapq
    from collections import Counter

    started = time.perf_counter()
    v = np.asarray(mesh.vertices).copy()
    t = np.asarray(mesh.triangles).copy()
    report = {
        "enabled": bool(enabled),
        "applied": False,
        "method": "adaptive_local_boundary_evidence_with_geodesic_collar",
        "connectivity_modified": False,
        "shape_specific_assumptions": False,
        "normal_recalculation": False,
        "uncertainty_source": "sampling_and_local_normal_residual_proxy",
    }

    def finish(result, reason):
        report["reason"] = reason
        report["seconds"] = float(time.perf_counter() - started)
        print(
            f"[Paso 13 {VERSION}] Frontera: {reason} | "
            f"{report.get('candidate_vertices', 0)} candidatos | "
            f"{report.get('moved_vertices', 0)} vértices movidos.",
            flush=True,
        )
        return result, report

    if not enabled:
        return finish(mesh, "disabled")
    if len(v) == 0 or len(t) == 0 or len(points) < 12:
        return finish(mesh, "insufficient_geometry")
    h = max(float(spacing), 1e-6)
    points = np.asarray(points, dtype=float)
    confidence = np.clip(np.asarray(confidence, float).reshape(-1), 0, 1)
    support = np.asarray(support, float).reshape(-1)
    if cloud_normals is None:
        return finish(mesh, "cloud_normals_required_for_surface_disambiguation")
    cn = np.asarray(cloud_normals, float).copy()
    cn_len = np.linalg.norm(cn, axis=1)
    valid_cn = np.isfinite(cn_len) & (cn_len > 1e-12)
    cn /= np.maximum(cn_len[:, None], 1e-12)
    uncertainty = None
    if position_uncertainty is not None:
        uncertainty = np.asarray(position_uncertainty, float).reshape(-1)
        if len(uncertainty) != len(points):
            raise ValueError("Paso 13: position_uncertainty_mm no coincide con points.")
        report["uncertainty_source"] = "input_position_uncertainty_plus_local_sampling"

    edges = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    be = edges[counts == 1]
    boundary = np.zeros(len(v), bool)
    boundary[be.ravel()] = True
    ids = np.flatnonzero(boundary)
    report["boundary_vertices"] = int(len(ids))
    if not len(ids):
        return finish(mesh, "no_open_boundary")
    adjacency = [[] for _ in v]
    border_adjacency = [[] for _ in v]
    for a, b in edges:
        adjacency[a].append(int(b))
        adjacency[b].append(int(a))
    for a, b in be:
        border_adjacency[a].append(int(b))
        border_adjacency[b].append(int(a))
    ambiguous = np.zeros(len(v), bool)
    ambiguous[edges[counts > 2].ravel()] = True
    mesh.compute_vertex_normals()
    mn = np.asarray(mesh.vertex_normals).copy()
    tree = cKDTree(points)
    distance, nearest = query_tree(tree, v, k=1)
    # One parallel query reused for all local sampling tolerances.
    cloud_spacing = query_tree(tree, points, k=2)[0][:, 1]
    fixed = (
        (distance <= 0.8 * h) & (confidence[nearest] >= 0.72) & (support[nearest] >= 3)
    ) | ambiguous
    max_gap = max(10 * h, max(0.0, float(search_margin_mm)) + 2 * h)
    max_gap = min(max_gap, 18 * h)
    radii = np.minimum(max_gap + 5 * h, np.maximum(8 * h, distance[ids] + 5 * h))
    neighborhoods = tree.query_ball_point(v[ids], radii, workers=query_threads())
    proposed = np.zeros_like(v)
    directions = np.zeros_like(v)
    tolerance_by_vertex = np.zeros(len(v))
    status = {}
    report["maximum_evidence_gap_mm"] = float(max_gap)
    report["search_radius_mm"] = robust_stats(radii)
    print(
        f"[Paso 13 {VERSION}] Buscando límites observados en {len(ids)} "
        "vértices de frontera; marcos locales y normales compatibles...",
        flush=True,
    )

    for count, (i, local) in enumerate(zip(ids, neighborhoods), 1):
        if count % 250 == 0:
            print(f"[Paso 13 {VERSION}] Límites locales revisados: {count}/{len(ids)}.", flush=True)
        i = int(i)
        status[i] = "insufficient_local_evidence"
        if fixed[i]:
            status[i] = "strong_anchor_or_nonmanifold"
            continue
        if len(border_adjacency[i]) != 2:
            status[i] = "ambiguous_boundary_branch"
            continue
        if distance[i] > max_gap or len(local) < 12:
            continue
        # Multiple graph rings prevent one skinny triangle from defining the
        # direction. Crossing another boundary or component is never allowed.
        visited = {i}
        frontier = {i}
        interior = []
        for ring in range(4):
            next_ring = set()
            for j in frontier:
                for k in adjacency[j]:
                    if k in visited:
                        continue
                    visited.add(k)
                    if np.linalg.norm(v[k] - v[i]) <= 5 * h and not ambiguous[k]:
                        if not boundary[k]:
                            next_ring.add(k)
                            interior.append(k)
            frontier = next_ring
            if not frontier:
                break
        if len(interior) < 3:
            status[i] = "insufficient_interior_collar"
            continue
        a, b = border_adjacency[i]
        tangent = v[b] - v[a]
        tangent_length = np.linalg.norm(tangent)
        if tangent_length <= 1e-12:
            status[i] = "degenerate_boundary"
            continue
        tangent /= tangent_length
        normal = np.mean(mn[interior], axis=0)
        normal -= tangent * np.dot(normal, tangent)
        normal_length = np.linalg.norm(normal)
        if normal_length < 0.4:
            status[i] = "ambiguous_surface_normal"
            continue
        normal /= normal_length
        outward = np.cross(tangent, normal)
        outward /= max(np.linalg.norm(outward), 1e-12)
        interior_offset = v[interior] - v[i]
        if np.median(interior_offset @ outward) > 0:
            outward *= -1
        if np.mean(interior_offset @ outward < -0.05 * h) < 0.7:
            status[i] = "ambiguous_outward_direction"
            continue
        local = np.asarray(local, dtype=int)
        offset = points[local] - v[i]
        along = offset @ outward
        across = offset @ tangent
        normal_offset = offset @ normal
        # Only the same oriented sheet supplies endpoint evidence. Absolute
        # normal agreement tolerates inconsistent normal signs, not two layers.
        same_normal = (
            valid_cn[local]
            & (np.abs(cn[local] @ normal) >= 0.70)
            & (np.abs(cn[local] @ outward) <= 0.60)
        )
        corridor = (np.abs(across) <= 3 * h) & (np.abs(normal_offset) <= 2 * h)
        good = corridor & same_normal & (confidence[local] >= 0.55) & (support[local] >= 2)
        if np.count_nonzero(good) < 8:
            status[i] = "no_compatible_observed_strip"
            continue
        good_ids = local[good]
        local_spacing = cloud_spacing[good_ids]
        local_spacing = local_spacing[np.isfinite(local_spacing) & (local_spacing > 0)]
        gap = float(np.median(local_spacing)) if len(local_spacing) else h
        # This spread is perpendicular to the local sheet; along/across spread
        # is real surface extent and must not be treated as measurement noise.
        residual = normal_offset[good]
        sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        reliability = float(np.median(confidence[good_ids] * np.minimum(support[good_ids] / 4, 1)))
        tol = max(0.65 * h, 0.75 * gap) * (1 + 0.5 * (1 - reliability))
        tol = max(tol, 1.5 * sigma)
        if uncertainty is not None:
            u = uncertainty[good_ids]
            u = u[np.isfinite(u) & (u >= 0)]
            if len(u):
                tol = max(tol, 2 * float(np.median(u)))
        # Use two side strips plus the centre. Unsupported single-file tips
        # and mixed layers are not enough to establish a trustworthy endpoint.
        lanes = []
        for left, right in ((-3 * h, -0.75 * h), (-0.75 * h, 0.75 * h), (0.75 * h, 3 * h)):
            lane = good & (across >= left) & (across <= right)
            if np.count_nonzero(lane) >= 2:
                lanes.append(float(np.max(along[lane])))
        if len(lanes) < 3 or np.ptp(lanes) > max(2 * h, 2 * tol):
            status[i] = "endpoint_not_continuous_across_strip"
            continue
        # Any plausible lower-support observation can veto the retraction.
        plausible = corridor & same_normal & (confidence[local] >= 0.20) & (support[local] >= 1)
        limit = max(max(lanes), float(np.max(along[plausible])))
        excess = -limit - tol
        if excess <= 0.20 * h:
            status[i] = "already_within_observation_tolerance"
            continue
        if excess > max_gap or tol > 3 * h:
            status[i] = "uncertainty_or_extension_too_large"
            continue
        if np.count_nonzero(good & (along < limit - 2 * h)) < 4:
            status[i] = "no_supported_surface_behind_endpoint"
            continue
        # Probe farther ahead: data on the opposite side of an unobserved
        # opening indicates interpolation, not an extrapolated terminal rim.
        ahead = np.asarray(
            tree.query_ball_point(v[i] + 0.5 * max_gap * outward, max_gap + 3 * h), dtype=int
        )
        if len(ahead):
            off = points[ahead] - v[i]
            aa = off @ outward
            veto = (
                (aa > tol)
                & (aa <= 1.5 * max_gap)
                & (np.abs(off @ tangent) <= 3 * h)
                & (np.abs(off @ normal) <= 2 * h)
                & valid_cn[ahead]
                & (np.abs(cn[ahead] @ normal) >= 0.70)
                & (confidence[ahead] >= 0.20)
                & (support[ahead] >= 1)
            )
            if np.any(veto):
                status[i] = "observations_beyond_gap_preserve_interpolation"
                continue
        proposed[i] = -outward * (0.90 * excess)
        directions[i] = outward
        tolerance_by_vertex[i] = tol
        status[i] = "candidate"

    raw = np.linalg.norm(proposed, axis=1) > 0
    selected = raw.copy()
    # Require agreement along the boundary graph. Never average across a
    # spatially nearby but topologically separate rim or across a sharp corner.
    for i in np.flatnonzero(raw):
        neighborhood = {int(i)}
        frontier = {int(i)}
        for _ in range(3):
            frontier = {k for j in frontier for k in border_adjacency[j]} - neighborhood
            neighborhood.update(frontier)
        agreeing = [
            j for j in neighborhood if raw[j] and np.dot(directions[i], directions[j]) >= 0.8
        ]
        if len(agreeing) < 3:
            selected[i] = False
            status[int(i)] = "isolated_or_inconsistent_endpoint_evidence"
    delta = np.zeros_like(v)
    for i in np.flatnonzero(selected):
        neighbors = [
            j
            for j in border_adjacency[i]
            if selected[j] and np.dot(directions[i], directions[j]) >= 0.8
        ]
        magnitude = np.linalg.norm(proposed[i])
        if neighbors:
            magnitude = min(
                magnitude,
                0.5 * magnitude + 0.5 * np.median(np.linalg.norm(proposed[neighbors], axis=1)),
            )
        delta[i] = -directions[i] * magnitude
    report.update(
        raw_candidate_vertices=int(raw.sum()),
        candidate_vertices=int(selected.sum()),
        decision_counts=dict(Counter(status.values())),
        boundary_vertex_decisions=[{"vertex": int(i), "decision": status[int(i)]} for i in ids],
    )
    if not selected.any():
        return finish(mesh, "no_unambiguous_local_extrapolation")

    # Geodesic, physical-width collar. Strong anchors and all other boundaries
    # stay fixed. Harmonic relaxation spreads the displacement continuously.
    band = max(6 * h, 2 * float(np.linalg.norm(delta, axis=1).max()) + 3 * h)
    geodesic = np.full(len(v), np.inf)
    source = np.full(len(v), -1, dtype=int)
    queue = []
    blocked = fixed | (boundary & ~selected)
    for i in np.flatnonzero(selected):
        geodesic[i] = 0.0
        source[i] = i
        heapq.heappush(queue, (0.0, int(i)))
    while queue:
        dist, i = heapq.heappop(queue)
        if dist > geodesic[i]:
            continue
        for j in adjacency[i]:
            if blocked[j]:
                continue
            candidate_distance = dist + float(np.linalg.norm(v[j] - v[i]))
            if candidate_distance < min(band, geodesic[j]):
                geodesic[j] = candidate_distance
                source[j] = source[i]
                heapq.heappush(queue, (candidate_distance, j))
    active = np.isfinite(geodesic) & ~blocked
    collar = active & ~selected
    ratio = np.clip(geodesic[collar] / band, 0, 1)
    delta[collar] = delta[source[collar]] * (1 - 3 * ratio**2 + 2 * ratio**3)[:, None]
    lengths = np.maximum(np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1), 0.1 * h)
    rr = np.concatenate([edges[:, 0], edges[:, 1]])
    cc = np.concatenate([edges[:, 1], edges[:, 0]])
    ww = np.concatenate([1 / lengths, 1 / lengths])
    graph = coo_matrix((ww, (rr, cc)), shape=(len(v), len(v))).tocsr()
    degree = np.maximum(np.asarray(graph.sum(axis=1)).ravel(), 1e-12)
    for _ in range(32):
        average = (graph @ delta) / degree[:, None]
        delta[collar] = 0.35 * delta[collar] + 0.65 * average[collar]
    delta[~active] = 0
    report["collar_width_mm"] = float(band)
    report["strong_or_topological_anchors"] = int(fixed.sum())
    report["proposed_displacement_mm"] = robust_stats(np.linalg.norm(delta[active], axis=1))
    report["tolerance_mm"] = robust_stats(tolerance_by_vertex[selected])

    # Index-preserving line search. Existing invalid/zero-area faces may not be
    # moved; no new inversion, degeneration, or reported intersection is allowed.
    original_cross = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    area2 = np.linalg.norm(original_cross, axis=1)
    fragile = ~np.isfinite(area2) | (area2 <= 1e-12 * h * h)
    delta[np.unique(t[fragile])] = 0
    affected = np.any(np.linalg.norm(delta[t], axis=2) > 1e-12 * h, axis=1)
    # Thin triangles may need a smaller movement locally. Attenuate their
    # one-ring before the final global guard, rather than letting one fragile
    # face unnecessarily reduce every other independently supported endpoint.
    local_scale = np.ones(len(v))
    local_guard_history = []
    for _ in range(12):
        candidate = v + delta * local_scale[:, None]
        cross = np.cross(
            candidate[t[:, 1]] - candidate[t[:, 0]], candidate[t[:, 2]] - candidate[t[:, 0]]
        )
        safe = (
            np.all(np.isfinite(cross), axis=1)
            & (np.einsum("ij,ij->i", cross, original_cross) > 0)
            & (np.linalg.norm(cross, axis=1) >= 0.15 * area2)
        )
        unsafe = affected & ~safe
        if not np.any(unsafe):
            break
        local_guard_history.append(int(np.count_nonzero(unsafe)))
        core = np.unique(t[unsafe])
        ring = np.unique([j for i in core for j in adjacency[i]])
        local_scale[core] *= 0.5
        # A small decrease in the surrounding ring reduces sharp changes of
        # displacement. Core attenuation is intentionally stronger.
        outer = np.setdiff1d(ring, core, assume_unique=True)
        local_scale[outer] *= 0.85
    delta *= local_scale[:, None]
    report["local_orientation_guard_unsafe_faces"] = local_guard_history
    report["locally_attenuated_vertices"] = int(
        np.count_nonzero((local_scale < 1) & (np.linalg.norm(delta, axis=1) > 1e-12 * h))
    )
    if not np.any(np.linalg.norm(delta, axis=1) > 1e-12 * h):
        return finish(mesh, "all_proposed_vertices_pinned_by_geometry_guards")

    def intersection_pairs(m):
        return {
            tuple(sorted(map(int, pair)))
            for pair in np.asarray(m.get_self_intersecting_triangles()).reshape(-1, 2)
        }

    print(
        f"[Paso 13 {VERSION}] Comprobando repliegue: orientación, área y "
        "pares de intersección antes/después...",
        flush=True,
    )
    baseline = None
    trials = []
    accepted = None
    for attempt in range(9):
        alpha = 0.5**attempt
        candidate = v + alpha * delta
        cross = np.cross(
            candidate[t[:, 1]] - candidate[t[:, 0]], candidate[t[:, 2]] - candidate[t[:, 0]]
        )
        new_area = np.linalg.norm(cross, axis=1)
        safe = (
            np.all(np.isfinite(cross), axis=1)
            & (np.einsum("ij,ij->i", cross, original_cross) > 0)
            & (new_area >= 0.15 * area2)
        )
        if not np.all(safe[affected]):
            trials.append(
                {
                    "alpha": alpha,
                    "rejected": "orientation_or_area",
                    "unsafe_triangles": int(np.count_nonzero(~safe & affected)),
                }
            )
            continue
        trial = copy.deepcopy(mesh)
        trial.vertices = o3d.utility.Vector3dVector(candidate)
        if baseline is None:
            baseline = intersection_pairs(mesh)
        print(
            f"[Paso 13 {VERSION}] Guarda de intersecciones: "
            f"intento {attempt + 1}/9, factor de repliegue {alpha:.4f}.",
            flush=True,
        )
        new_pairs = intersection_pairs(trial) - baseline
        if new_pairs:
            trials.append(
                {"alpha": alpha, "rejected": "new_intersections", "pairs": len(new_pairs)}
            )
            continue
        accepted = trial
        shifts = np.linalg.norm(alpha * delta, axis=1)
        report.update(
            applied=bool(np.any(shifts > 1e-12 * h)),
            accepted_alpha=float(alpha),
            new_intersection_pairs=0,
            baseline_intersection_pairs=int(len(baseline)),
            maximum_shift_mm=float(shifts.max()),
            moved_vertices=int(np.count_nonzero(shifts > 1e-12 * h)),
            actual_displacement_mm=robust_stats(shifts[shifts > 1e-12 * h]),
            boundary_displacement_mm=robust_stats(shifts[selected]),
            residual_proposed_boundary_shift_mm=robust_stats(
                np.linalg.norm(proposed[selected] - alpha * delta[selected], axis=1)
            ),
        )
        break
    report["guard_trials"] = trials
    if accepted is None:
        return finish(mesh, "candidate_cancelled_by_geometry_guards")
    accepted.compute_triangle_normals()
    accepted.compute_vertex_normals()
    report["normal_recalculation"] = True
    report["vertices_before"] = report["vertices_after"] = int(len(v))
    report["triangles_before"] = report["triangles_after"] = int(len(t))
    print(
        f"[Paso 13 {VERSION}] Repliegue aceptado: máximo "
        f"{report['maximum_shift_mm']:.3f} mm, factor {report['accepted_alpha']:.4f}.",
        flush=True,
    )
    return finish(accepted, "accepted_with_geometry_guards")


def load_estimated_completion(root, source_cloud_path, source_name, enabled):
    """Guías auxiliares del 11; jamás sustituyen la nube de evaluación del 12."""
    import hashlib

    folder = root / "reconstruccion" / "multisesion" / source_name
    aux = folder / "completado_geometrico_estimado.npz"
    observed = folder / "nube_fusionada_multivista.npz"
    report = {
        "enabled": bool(enabled),
        "applied": False,
        "estimated_points": 0,
        "counts_as_observed_evidence": False,
        "source": str(aux),
    }
    empty = (np.empty((0, 3)), np.empty((0, 3), np.uint8), np.empty((0, 3)), np.empty(0))
    if not enabled or not aux.is_file():
        report["reason"] = "disabled_or_no_completion_file"
        return (*empty, report)
    if not observed.is_file():
        raise RuntimeError("Paso 13: falta la nube observada asociada al completado del paso 11.")
    if source_cloud_path.stat().st_mtime + 1 < observed.stat().st_mtime:
        raise RuntimeError(
            "Paso 13: el paso 12 es anterior al último paso 11. Ejecuta nuevamente el paso 12."
        )
    with np.load(aux, allow_pickle=False) as data:
        digest = str(np.asarray(data["observed_npz_sha256"]).item())
        if hashlib.sha256(observed.read_bytes()).hexdigest() != digest:
            raise RuntimeError(
                "Paso 13: el completado y la nube del paso 11 pertenecen a resultados distintos. Reejecuta 11 y 12."
            )
        schema = (
            int(np.asarray(data["schema_version"]).reshape(-1)[0])
            if "schema_version" in data.files
            else 0
        )
        if schema < 2:
            raise RuntimeError(
                "Paso 13: el completado del paso 11 es anterior al contrato multivista V2. "
                "Reejecuta 11; una guía geométrica no validada por silueta/visibilidad/espacio libre no puede usarse."
            )
        p = np.asarray(data["points"], float)
        c = np.asarray(data["colors"], np.uint8)
        n = np.asarray(data["normals"], float)
        q = np.asarray(data["confidence"], float).reshape(-1)
        support = np.asarray(data["support_views"]).reshape(-1)
        inferred = np.asarray(data["is_inferred"], bool).reshape(-1)
        kinds = np.asarray(data["kind"]).reshape(-1)
        validated = (
            np.asarray(data["multiview_validated"], bool).reshape(-1)
            if "multiview_validated" in data.files
            else np.zeros(len(p), bool)
        )
        independent_depth = (
            np.asarray(data["independent_depth_agreements"], np.uint8).reshape(-1)
            if "independent_depth_agreements" in data.files
            else np.zeros(len(p), np.uint8)
        )
        free_space = (
            np.asarray(data["free_space_contradictions"], np.uint8).reshape(-1)
            if "free_space_contradictions" in data.files
            else np.full(len(p), 255, np.uint8)
        )
        source_report = json.loads(str(np.asarray(data["report_json"]).item()))
    if p.ndim != 2 or p.shape[1:] != (3,) or c.shape != p.shape or n.shape != p.shape:
        raise ValueError("Paso 13: dimensiones inválidas en completado_geometrico_estimado.npz.")
    if any(
        len(a) != len(p)
        for a in (q, support, inferred, kinds, validated, independent_depth, free_space)
    ):
        raise ValueError("Paso 13: metadatos del completado no coinciden con sus puntos.")
    if np.any(support != 0) or not np.all(inferred):
        raise ValueError("Paso 13: una estimación geométrica no puede declarar respaldo observado.")
    if len(p) and (
        not np.all(validated) or np.any(independent_depth < 2) or np.any(free_space > 0)
    ):
        raise RuntimeError(
            "Paso 13: el archivo de completado contiene guías que no acreditan validación multivista estricta. "
            "Reejecuta el paso 11 V11.2 o posterior."
        )
    length = np.linalg.norm(n, axis=1)
    valid = (
        np.all(np.isfinite(p), axis=1)
        & np.all(np.isfinite(n), axis=1)
        & np.isfinite(q)
        & (length > 1e-10)
        & validated
        & (independent_depth >= 2)
        & (free_space == 0)
    )
    p, c, n, q = p[valid], c[valid], n[valid], np.clip(q[valid], 0.05, 0.25)
    n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-12)
    report.update(
        applied=bool(len(p)),
        estimated_points=int(len(p)),
        estimated_bottom_points=int(np.count_nonzero(kinds[valid] == 2)),
        source_report=source_report,
        reason="verified_multiview_auxiliary_guides",
        multiview_contract="silhouette_visibility_free_space_pose_diverse",
        lineage_check="step11_sha256_and_step12_modification_time",
    )
    print(
        f"[Paso 13 {VERSION}] Guías estimadas del paso 11: {len(p):,}; "
        "se usan en Poisson, no en las métricas de respaldo.",
        flush=True,
    )
    return p, c, n, q, report


def main():
    """Reconstruye y evalúa una superficie a partir de la nube regularizada."""
    args = make_parser().parse_args()
    started = time.perf_counter()
    root = Path(args.root).expanduser().resolve()
    source_dir = root / "reconstruccion" / "multisesion" / args.source
    cloud_path = source_dir / "nube_regularizada_general.npz"
    if not cloud_path.is_file():
        raise FileNotFoundError(cloud_path)

    print(f"\n[Paso 13 {VERSION}] Parte 1/6 | Cargando nube regularizada...")
    with np.load(cloud_path) as data:
        points = np.asarray(data["points"], dtype=np.float64)
        colors = np.asarray(data["colors"], dtype=np.uint8)
        normals = np.asarray(data["normals"], dtype=np.float64)
        confidence_available = "confidence" in data.files
        support_available = "support_views" in data.files
        confidence = (
            np.asarray(data["confidence"], dtype=np.float64).reshape(-1)
            if confidence_available
            else np.ones(len(points), dtype=np.float64)
        )
        support = (
            np.asarray(data["support_views"], dtype=np.float64).reshape(-1)
            if support_available
            else np.ones(len(points), dtype=np.float64)
        )
        evidence_fields_present = all(
            key in data.files
            for key in (
                "surface_evidence_class",
                "independent_support_poses",
                "support_angular_span_poses",
                "conflict_pose_ratio",
                "heldout_validation_available",
                "heldout_validation_pass_ratio",
                "evidence_strength",
            )
        )
        if "evidence_contract_valid" in data.files:
            evidence_contract_marker = bool(
                int(np.asarray(data["evidence_contract_valid"]).reshape(-1)[0])
            )
        else:
            # V4.0 de 12 podía publicar nombres de campos aun cuando provenían
            # de defaults legacy. Un span angular completamente nulo identifica
            # ese caso sin inventar evidencia.
            candidate_span = (
                np.asarray(data["support_angular_span_poses"], dtype=np.int16).reshape(-1)
                if "support_angular_span_poses" in data.files
                else np.zeros(len(points), dtype=np.int16)
            )
            evidence_contract_marker = bool(evidence_fields_present and np.any(candidate_span > 0))
        evidence_contract_available = bool(evidence_fields_present and evidence_contract_marker)
        surface_evidence = np.asarray(
            (
                data["surface_evidence_class"]
                if "surface_evidence_class" in data.files
                else np.full(len(points), 2)
            ),
            dtype=np.uint8,
        ).reshape(-1)
        independent_support = np.asarray(
            (
                data["independent_support_poses"]
                if "independent_support_poses" in data.files
                else support
            ),
            dtype=np.float64,
        ).reshape(-1)
        angular_span = np.asarray(
            (
                data["support_angular_span_poses"]
                if "support_angular_span_poses" in data.files
                else np.zeros(len(points))
            ),
            dtype=np.int16,
        ).reshape(-1)
        conflict_ratio = np.asarray(
            (
                data["conflict_pose_ratio"]
                if "conflict_pose_ratio" in data.files
                else np.zeros(len(points))
            ),
            dtype=np.float64,
        ).reshape(-1)
        heldout_available = np.asarray(
            (
                data["heldout_validation_available"]
                if "heldout_validation_available" in data.files
                else np.zeros(len(points))
            ),
            dtype=bool,
        ).reshape(-1)
        heldout_pass = np.asarray(
            (
                data["heldout_validation_pass_ratio"]
                if "heldout_validation_pass_ratio" in data.files
                else np.ones(len(points))
            ),
            dtype=np.float64,
        ).reshape(-1)
        heldout_tests = np.asarray(
            (
                data["heldout_validation_tests"]
                if "heldout_validation_tests" in data.files
                else np.zeros(len(points))
            ),
            dtype=np.int16,
        ).reshape(-1)
        heldout_pass_count = np.asarray(
            (
                data["heldout_validation_pass_count"]
                if "heldout_validation_pass_count" in data.files
                else np.rint(heldout_pass * np.maximum(heldout_tests, 0))
            ),
            dtype=np.int16,
        ).reshape(-1)
        evidence_contract_accept = np.asarray(
            (
                data["evidence_contract_accept"]
                if "evidence_contract_accept" in data.files
                else np.ones(len(points), dtype=np.uint8)
            ),
            dtype=bool,
        ).reshape(-1)
        evidence_contract_accept_published = "evidence_contract_accept" in data.files
        evidence_strength = np.asarray(
            (
                data["evidence_strength"]
                if "evidence_strength" in data.files
                else np.ones(len(points))
            ),
            dtype=np.float64,
        ).reshape(-1)
        # Optional per-point positional uncertainty. No guessed aliases: depth
        # spread and previous smoothing displacement are not this quantity.
        position_uncertainty = (
            np.asarray(data["position_uncertainty_mm"], dtype=np.float64).reshape(-1)
            if "position_uncertainty_mm" in data.files
            else None
        )
        fusion_voxel = optional_scalar(data, "fusion_voxel_mm", 0.0)
        meshing_reference = optional_scalar(
            data,
            "meshing_reference_spacing_mm",
            None,
        )
        if meshing_reference is None:
            meshing_reference = optional_scalar(
                data,
                "reference_spacing_mm",
                None,
            )

    evidence_arrays = (
        surface_evidence,
        independent_support,
        angular_span,
        conflict_ratio,
        heldout_available,
        heldout_tests,
        heldout_pass_count,
        heldout_pass,
        evidence_contract_accept,
        evidence_strength,
    )
    if not (
        len(colors) == len(points)
        and len(normals) == len(points)
        and len(confidence) == len(points)
        and len(support) == len(points)
        and all(len(a) == len(points) for a in evidence_arrays)
    ):
        raise ValueError(
            "Los metadatos geométricos y de evidencia del paso 12 deben tener "
            "la misma cantidad de muestras que points."
        )
    if bool(args.require_evidence_contract) and not evidence_contract_available:
        raise RuntimeError(
            "Paso 13 V5.3 detectó una salida 12 sin contrato de evidencia real. "
            "Esto ocurre cuando 12 fue generado desde un paso 11 anterior a V11.0: "
            "los campos pueden existir por compatibilidad, pero no contienen "
            "diversidad angular/held-out observacional válida. Reemplace 11 y 12 "
            "por las versiones entregadas y reanude el pipeline; el checkpoint "
            "debe recalcular 11 -> 12 -> 13. No se usa un fallback silencioso "
            "porque volvería a legitimar capas ambiguas."
        )

    if position_uncertainty is not None and len(position_uncertainty) != len(points):
        raise ValueError("position_uncertainty_mm debe tener una entrada por punto.")
    valid = (
        np.all(np.isfinite(points), axis=1)
        & np.all(np.isfinite(normals), axis=1)
        & np.isfinite(confidence)
        & np.isfinite(support)
        & np.isfinite(independent_support)
        & np.isfinite(conflict_ratio)
        & np.isfinite(heldout_pass)
        & np.isfinite(evidence_strength)
    )
    points = points[valid]
    colors = colors[valid]
    normals = normals[valid]
    confidence = confidence[valid]
    support = support[valid]
    surface_evidence = surface_evidence[valid]
    independent_support = independent_support[valid]
    angular_span = angular_span[valid]
    conflict_ratio = conflict_ratio[valid]
    heldout_available = heldout_available[valid]
    heldout_tests = heldout_tests[valid]
    heldout_pass_count = heldout_pass_count[valid]
    heldout_pass = heldout_pass[valid]
    evidence_contract_accept = evidence_contract_accept[valid]
    evidence_strength = np.clip(evidence_strength[valid], 0.0, 1.0)
    if evidence_contract_available:
        # El paso 12 es la única etapa que adjudica el contrato de evidencia.
        # 13 NO vuelve a reinterpretar held-out, conflicto o diversidad. Esto
        # evita que un cambio de representación (p.ej. 2/3 frente a 0.67) haga
        # desaparecer cobertura ya validada.
        if evidence_contract_accept_published:
            evidence_safe = evidence_contract_accept.copy()
            verification_mode = "authoritative_step12_accept_mask"
        else:
            # Compatibilidad con 12 V4.1: verificar con aritmética racional
            # exacta 2/3 en lugar de comparar floats contra 0.67.
            heldout_vote_ok = (~heldout_available) | (
                heldout_pass_count.astype(np.int64) * 3 >= 2 * heldout_tests.astype(np.int64)
            )
            evidence_safe = (
                (surface_evidence >= int(args.minimum_evidence_class))
                & (independent_support >= int(args.minimum_independent_support))
                & (angular_span >= int(args.minimum_angular_span_poses))
                & (conflict_ratio <= float(args.maximum_conflict_pose_ratio))
                & heldout_vote_ok
            )
            verification_mode = "legacy_exact_two_of_three_verification"

        rejected_by_contract = int(np.count_nonzero(~evidence_safe))
        retention = float(np.mean(evidence_safe)) if len(evidence_safe) else 0.0
        if rejected_by_contract:
            print(
                f"[Paso 13] Verificación del contrato ({verification_mode}): "
                f"{rejected_by_contract:,} muestras marcadas fuera del contrato. "
                "No se reinterpretan ni se rescatan por geometría local.",
                flush=True,
            )
        if retention < 0.98:
            raise RuntimeError(
                "Paso 13 detectó una salida 12 internamente inconsistente: "
                f"solo {retention:.1%} de sus puntos están marcados como aceptados "
                "por el contrato autoritativo. Reejecute 11 y 12 antes de mallar."
            )
        if position_uncertainty is not None:
            position_uncertainty = position_uncertainty[valid]
    else:
        rejected_by_contract = 0
        if position_uncertainty is not None:
            position_uncertainty = position_uncertainty[valid]
    if position_uncertainty is not None and len(position_uncertainty) != len(points):
        raise ValueError("position_uncertainty_mm dejó de coincidir tras filtrar evidencia.")
    if len(points) < 500:
        raise RuntimeError(
            "Muy pocos puntos finitos para reconstruir la superficie antes del mallado. "
            "Revise la salida real de 12; este error ya no se produce por recorte "
            "duplicado del contrato de evidencia."
        )

    measured_spacing = nearest_spacing(points)
    if meshing_reference is not None and np.isfinite(meshing_reference) and meshing_reference > 0:
        spacing = float(meshing_reference)
        spacing_source = "meshing_reference_spacing_mm"
    else:
        spacing = float(measured_spacing)
        spacing_source = "nearest_neighbor_spacing_current_cloud"

    extent = float(np.max(np.ptp(points, axis=0)))
    threads = _CPU_COUNT if int(args.threads) <= 0 else max(1, int(args.threads))
    estimated_p, estimated_c, estimated_n, estimated_confidence, completion_report = (
        load_estimated_completion(
            root, cloud_path, args.completion_source, args.use_estimated_completion
        )
    )
    # Only the Poisson proxy receives synthetic guides. The original arrays
    # below remain the sole source of fidelity metrics, anchors and trimming.
    proxy_points = np.concatenate([points, estimated_p])
    proxy_colors = np.concatenate([colors, estimated_c])
    proxy_normals = np.concatenate([normals, estimated_n])
    proxy_confidence = np.r_[confidence, estimated_confidence]
    # El soporte para mallado es el número de poses INDEPENDIENTES, no el bruto.
    proxy_support = np.r_[independent_support, np.zeros(len(estimated_p))]
    proxy_evidence_strength = np.r_[evidence_strength, np.full(len(estimated_p), 0.15)]
    poisson_pcd, downsample_voxel, proxy_report = prepare_poisson_cloud(
        proxy_points,
        proxy_colors,
        proxy_normals,
        proxy_confidence,
        proxy_support,
        spacing,
        extent,
        args,
        evidence_strength=proxy_evidence_strength,
        evidence_contract=bool(evidence_contract_available),
    )
    proxy_report["observed_input_points"] = int(len(points))
    proxy_report["estimated_input_points"] = int(len(estimated_p))
    proxy_report["guide_support_is_not_observational"] = True
    poisson_points = np.asarray(poisson_pcd.points, dtype=np.float64)
    poisson_spacing = nearest_spacing(poisson_points)
    depth_spacing = max(float(spacing), float(poisson_spacing))
    depth, depth_source = adaptive_depth(extent, depth_spacing, args)

    print(
        f"[Paso 13 {VERSION}] Parte 2/6 | Preparando Poisson: "
        f"{len(points):,} -> {len(poisson_pcd.points):,} puntos | "
        f"voxel={downsample_voxel:.3f} mm | "
        f"spacing={poisson_spacing:.3f} mm | "
        f"depth={depth} | hilos={threads}"
    )
    output = root / "reconstruccion" / "multisesion" / args.output_name
    output.mkdir(parents=True, exist_ok=True)

    poisson_started = time.perf_counter()
    print(f"[Paso 13 {VERSION}] Parte 3/6 | Screened Poisson + validación de extrapolación...")
    selected_method = "screened_poisson_adaptive"
    fallback_used = False
    bpa_radii = []
    evidence_confidence = np.clip(confidence * evidence_strength, 0.0, 1.0)
    evidence_support = independent_support.astype(np.float64)
    candidates = {}
    candidate_comparison = {"attempted": [], "selection": None, "poisson_risk": None}

    poisson_error = None
    try:
        poisson_mesh, densities = create_poisson(
            poisson_pcd, depth, args.poisson_scale, threads
        )
        densities = np.asarray(densities, dtype=np.float64)
        trim_quantile = float(args.poisson_density_trim_quantile)
        if len(densities) and trim_quantile > 0.0:
            threshold = float(np.quantile(densities, trim_quantile))
            poisson_mesh.remove_vertices_by_mask(densities < threshold)
        poisson_mesh = crop_to_cloud_bbox(poisson_mesh, points, args.bbox_margin_mm)
        poisson_mesh = clean_mesh(poisson_mesh)
        if len(poisson_mesh.triangles) < 500:
            raise RuntimeError("Poisson produjo menos de 500 triángulos.")
        poisson_mesh, poisson_cleanup = retain_supported_components(
            poisson_mesh, points, evidence_confidence, evidence_support,
            poisson_spacing, confidence_available, support_available, args,
        )
        poisson_eval = evaluate(
            poisson_mesh, points, args.coverage_gate_mm,
            args.evaluation_cloud_samples, args.evaluation_mesh_samples,
            name="screened_poisson_adaptive",
        )
        candidates["screened_poisson_adaptive"] = (
            poisson_mesh, poisson_eval, poisson_cleanup
        )
        candidate_comparison["attempted"].append("screened_poisson_adaptive")
        risk, risk_report = poisson_extrapolation_risk(poisson_eval, spacing, args)
        candidate_comparison["poisson_risk"] = risk_report
    except Exception as exc:
        poisson_error = str(exc)
        risk = True
        candidate_comparison["poisson_risk"] = {
            "risk": True, "reasons": [f"poisson_error:{exc}"]
        }

    observed_pcd = poisson_pcd
    observed_spacing = poisson_spacing
    if len(estimated_p):
        observed_pcd, _, _ = prepare_poisson_cloud(
            points, colors, normals, confidence, independent_support, spacing, extent, args,
            evidence_strength=evidence_strength, evidence_contract=bool(evidence_contract_available))
        observed_spacing = nearest_spacing(np.asarray(observed_pcd.points))
    should_try_bpa = bool(args.bpa_fallback) and (
        not candidates or (bool(args.adaptive_bpa_competition) and risk)
    )
    if should_try_bpa:
        try:
            print(
                f"[Paso 13 {VERSION}] Poisson con riesgo de extrapolación; "
                "evaluando BPA como candidato observacional..."
            )
            bpa_mesh, bpa_radii = create_bpa_fallback(
                observed_pcd, observed_spacing, float(fusion_voxel or 0.0),
                args.bpa_radius_factors,
            )
            if len(bpa_mesh.triangles) < 100:
                raise RuntimeError("BPA produjo muy pocos triángulos.")
            bpa_mesh, bpa_cleanup = retain_supported_components(
                bpa_mesh, points, evidence_confidence, evidence_support,
                poisson_spacing, confidence_available, support_available, args,
            )
            bpa_eval = evaluate(
                bpa_mesh, points, args.coverage_gate_mm,
                args.evaluation_cloud_samples, args.evaluation_mesh_samples,
                name="ball_pivoting_observational",
            )
            candidates["ball_pivoting_observational"] = (
                bpa_mesh, bpa_eval, bpa_cleanup
            )
            candidate_comparison["attempted"].append("ball_pivoting_observational")
        except Exception as bpa_error:
            candidate_comparison["bpa_error"] = str(bpa_error)

    if not candidates:
        raise RuntimeError(
            "Paso 13: Poisson falló y BPA no produjo una malla válida. "
            f"Poisson: {poisson_error}; BPA: {candidate_comparison.get('bpa_error')}"
        )

    try:
        selected_method, selected_data, selection_report = choose_surface_candidate(candidates, spacing, args)
    except RuntimeError as initial_failure:
        candidate_comparison["initial_selection"] = getattr(initial_failure, "selection_report", {})
        if args.adaptive_observed_refinement and args.bpa_fallback:
            print("[Paso 13] Ningún candidato admisible: conservando más muestras observadas para BPA.", flush=True)
            refined_args = argparse.Namespace(**vars(args))
            refined_args.meshing_minimum_voxel_mm = min(float(args.meshing_minimum_voxel_mm), 0.5*spacing)
            refined_args.input_voxel_spacing_factor = min(float(args.input_voxel_spacing_factor), 0.5)
            refined_pcd, _, refined_report = prepare_poisson_cloud(
                points, colors, normals, confidence, independent_support, spacing, extent, refined_args,
                evidence_strength=evidence_strength, evidence_contract=bool(evidence_contract_available))
            refined_spacing = nearest_spacing(np.asarray(refined_pcd.points))
            refined_mesh, refined_radii = create_bpa_fallback(
                refined_pcd, refined_spacing, float(fusion_voxel or 0.0), args.bpa_radius_factors)
            if len(refined_mesh.triangles) >= 100:
                refined_mesh, refined_cleanup = retain_supported_components(
                    refined_mesh, points, evidence_confidence, evidence_support,
                    refined_spacing, confidence_available, support_available, args)
                refined_eval = evaluate(refined_mesh, points, args.coverage_gate_mm,
                    args.evaluation_cloud_samples, args.evaluation_mesh_samples,
                    name="ball_pivoting_observed_refined")
                candidates["ball_pivoting_observed_refined"] = (refined_mesh, refined_eval, refined_cleanup)
                candidate_comparison["attempted"].append("ball_pivoting_observed_refined")
                candidate_comparison["observed_refinement"] = {
                    "proxy": refined_report, "radii_mm": refined_radii,
                    "estimated_guides_used": False, "thresholds_relaxed": False}
        try:
            selected_method, selected_data, selection_report = choose_surface_candidate(candidates, spacing, args)
        except RuntimeError as failure:
            candidate_comparison["selection"] = getattr(failure, "selection_report", {})
            (output / "resumen_13_reconstruccion_superficie.json").write_text(
                json.dumps({"quality": "rejected", "version": VERSION,
                    "candidate_comparison": candidate_comparison, "reason": str(failure)},
                    indent=2, ensure_ascii=False), encoding="utf-8")
            raise
    if selected_method == "ball_pivoting_observed_refined":
        bpa_radii = refined_radii
    mesh, pre_selection_evaluation, component_cleanup = selected_data
    candidate_comparison["selection"] = selection_report
    fallback_used = selected_method != "screened_poisson_adaptive"
    poisson_seconds = time.perf_counter() - poisson_started

    print(
        f"[Paso 13 {VERSION}] Selección: {selected_method} | "
        f"candidatos={list(candidates.keys())}"
    )
    print(
        f"[Paso 13 {VERSION}] Componentes seleccionadas: "
        f"{component_cleanup.get('components_before', 0)} -> "
        f"{component_cleanup.get('components_retained', 0)}"
    )

    print(
        f"[Paso 13 {VERSION}] Parte 5/6 | Suavizando regiones débiles " "con anclajes multivista..."
    )
    mesh, weak_regularization = restricted_weak_region_smoothing(
        mesh,
        points,
        evidence_confidence,
        evidence_support,
        poisson_spacing,
        confidence_available,
        support_available,
        args,
    )
    displacement_p90 = weak_regularization.get("actual_displacement_mm", {}).get("p90")
    print(
        f"[Paso 13 {VERSION}] Anclajes fijos: "
        f"{weak_regularization.get('anchor_vertices', 0):,} | "
        f"vértices débiles: "
        f"{weak_regularization.get('weak_vertices', 0):,} | "
        f"desplazamiento P90={displacement_p90} mm"
    )
    mesh, boundary_retraction = retract_observation_boundaries(
        mesh,
        points,
        evidence_confidence,
        evidence_support,
        spacing,
        cloud_normals=normals,
        position_uncertainty=position_uncertainty,
        search_margin_mm=args.bbox_margin_mm,
        enabled=args.boundary_retraction,
    )
    try:
        if mesh.is_orientable():
            mesh.orient_triangles()
    except Exception:
        pass
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh = colorize(mesh, points, colors)

    print(f"[Paso 13 {VERSION}] Parte 6/6 | Evaluación rápida y guardado...")
    evaluation = evaluate(
        mesh,
        points,
        args.coverage_gate_mm,
        args.evaluation_cloud_samples,
        args.evaluation_mesh_samples,
        name=selected_method,
    )
    topology = evaluation["topology"]
    warning_reasons = []
    components = topology.get("connected_components")
    largest = topology.get("largest_component_triangle_ratio")
    if components is None or components > int(args.warning_max_components):
        warning_reasons.append(f"connected_components={components}")
    if largest is None or largest < float(args.warning_min_largest_component_ratio):
        warning_reasons.append(f"largest_component_triangle_ratio={largest}")
    if topology.get("nonmanifold_edges", 0) > 0:
        warning_reasons.append(f"nonmanifold_edges={topology.get('nonmanifold_edges')}")
    mesh_to_cloud_p90 = evaluation.get("mesh_to_cloud_mm", {}).get("p90")
    if mesh_to_cloud_p90 is None or mesh_to_cloud_p90 > float(
        args.warning_max_mesh_to_cloud_p90_mm
    ):
        warning_reasons.append(f"mesh_to_cloud_p90_mm={mesh_to_cloud_p90}")

    if float(topology.get("boundary_edge_ratio") or 0.0) > 0.03:
        warning_reasons.append(f"open_boundary_edge_ratio={topology['boundary_edge_ratio']:.6f}")
    # Revalidate the actual exported geometry after smoothing/retraction.
    try:
        choose_surface_candidate({selected_method: (mesh, evaluation, component_cleanup)}, spacing, args)
    except RuntimeError as failure:
        (output / "resumen_13_reconstruccion_superficie.json").write_text(
            json.dumps({"quality": "rejected", "version": VERSION,
                "reason": "postprocessing_failed_surface_contract", "evaluation": evaluation,
                "candidate_comparison": candidate_comparison}, indent=2, ensure_ascii=False), encoding="utf-8")
        raise RuntimeError("Paso 13: la geometría final incumple el contrato después del posprocesamiento.") from failure
    quality = "accepted" if not warning_reasons else "warning"
    selected_path = output / "malla_final_seleccionada.ply"
    if not o3d.io.write_triangle_mesh(
        str(selected_path),
        mesh,
        write_ascii=False,
    ):
        raise RuntimeError("No se pudo guardar malla_final_seleccionada.ply")

    elapsed = time.perf_counter() - started
    report = {
        "schema_version": "4.8",
        "step": STEP,
        "version": VERSION,
        "quality": quality,
        "warning_reasons": warning_reasons,
        "method": (
            "evidence_guarded_adaptive_poisson_bpa_candidate_selection_with_"
            "whole_component_support_cleanup_and_restricted_weak_region_taubin"
        ),
        "selected_method": selected_method,
        "shape_specific_assumptions": False,
        "pose_reoptimization": False,
        "input_points": int(len(points)),
        "evidence_contract": {
            "available": bool(evidence_contract_available),
            "required": bool(args.require_evidence_contract),
            "rejected_before_meshing": int(rejected_by_contract),
            "surface_evidence_class": robust_stats(surface_evidence),
            "independent_support_poses": robust_stats(independent_support),
            "support_angular_span_poses": robust_stats(angular_span),
            "conflict_pose_ratio": robust_stats(conflict_ratio),
            "heldout_pass_ratio": (
                robust_stats(heldout_pass[heldout_available])
                if np.any(heldout_available)
                else robust_stats([])
            ),
            "evidence_strength": robust_stats(evidence_strength),
            "policy": "step11_12_evidence_is_authoritative_no_layer_readjudication",
        },
        "estimated_completion": completion_report,
        "poisson_input_points": int(len(poisson_pcd.points)),
        "input_downsample_voxel_mm": float(downsample_voxel),
        "poisson_proxy": proxy_report,
        "current_cloud_spacing_mm": float(measured_spacing),
        "meshing_spacing_mm": float(spacing),
        "meshing_spacing_source": spacing_source,
        "poisson_proxy_spacing_mm": float(poisson_spacing),
        "poisson_depth_spacing_mm": float(depth_spacing),
        "poisson_depth_used": int(depth),
        "poisson_depth_source": depth_source,
        "poisson_threads_requested": int(threads),
        "poisson_density_trim_quantile": float(args.poisson_density_trim_quantile),
        "component_cleanup": component_cleanup,
        "weak_region_regularization": weak_regularization,
        "observation_boundary_retraction": boundary_retraction,
        "fallback_used": bool(fallback_used),
        "bpa_radii_mm": [float(value) for value in bpa_radii],
        "candidate_comparison": candidate_comparison,
        "evaluation": evaluation,
        "evaluations": [data[1] for data in candidates.values()],
        "selected_mesh": str(selected_path),
        "outputs": {"selected": str(selected_path)},
        "timing_seconds": {
            "poisson_or_fallback": float(poisson_seconds),
            "total": float(elapsed),
        },
        "performance_note": (
            "Poisson trabaja sobre una representación auxiliar ponderada por "
            "evidencia independiente de 11/12. La nube completa validada fija los anclajes y "
            "pondera la regularización de las regiones débiles. "
            "El repliegue compara pares de auto-intersección antes/después; "
            "el paso 16 mantiene la validación semántica completa. "
            "BPA solo compite cuando Poisson muestra extrapolación no respaldada; "
            "la selección usa cobertura y distancia simétrica respecto a observaciones."
        ),
        "geometric_note": (
            "Nunca se eliminan caras individuales por falta de respaldo. Solo "
            "se retiran componentes completas pequeñas, aisladas y sin "
            "evidencia. Las zonas interpoladas conectadas se conservan. No "
            "utiliza primitivas, dimensiones ni clases de objeto."
        ),
    }

    _estado13("Guardando resumen y diagnósticos finales")
    summary_path = output / "resumen_13_reconstruccion_superficie.json"
    summary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    comparison_path = output / "comparacion_metodos_superficie_13.json"
    comparison_path.write_text(
        json.dumps(
            {
                "evaluations": [data[1] for data in candidates.values()],
                "selected_method": selected_method,
                "fallback_used": bool(fallback_used),
                "candidate_comparison": candidate_comparison,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\n========== PASO 13 {VERSION} COMPLETADO ==========")
    print(f"Método: {selected_method} | depth={depth} | " f"hilos solicitados={threads}")
    print(f"Puntos usados por Poisson: {len(poisson_pcd.points):,}/" f"{len(points):,}")
    print(f"Malla: {len(mesh.vertices):,} vértices | " f"{len(mesh.triangles):,} triángulos")
    print(
        f"Componentes: {topology.get('connected_components')} | "
        f"borde={topology.get('boundary_edge_ratio', 0.0):.3%}"
    )
    print(f"Calidad: {quality} | tiempo total: {elapsed:.1f} s")
    print("Salida:", output)
    print("=================================================\n")
    return 0


def _con_estado_visible(funcion, descripcion):
    """Envuelve una operación para publicar y restaurar su estado de progreso."""
    import functools

    @functools.wraps(funcion)
    def ejecutar(*args, **kwargs):
        global _estado_actual
        # Los procesos auxiliares no escriben estados que compitan con el padre.
        if __name__ != "__main__":
            return funcion(*args, **kwargs)
        with _estado_lock:
            anterior = _estado_actual
        _estado13(descripcion)
        try:
            return funcion(*args, **kwargs)
        except Exception as exc:
            print(f"[ERROR] Paso 13 | {descripcion} | {type(exc).__name__}: {exc}", flush=True)
            raise
        finally:
            with _estado_lock:
                _estado_actual = anterior

    return ejecutar


resolve_local_layers_and_normals = _con_estado_visible(
    resolve_local_layers_and_normals, "Resolviendo capas débiles y normales de transición"
)
prepare_poisson_cloud = _con_estado_visible(
    prepare_poisson_cloud, "Seleccionando superficie y preparando muestras para Poisson"
)
select_supported_surface_samples = _con_estado_visible(
    select_supported_surface_samples, "Evaluando continuidad local de la nube"
)
surface_voxel_groups = _con_estado_visible(
    surface_voxel_groups, "Separando capas dentro de los vóxeles"
)
create_poisson = _con_estado_visible(create_poisson, "Reconstruyendo la superficie con Poisson")
retain_supported_components = _con_estado_visible(
    retain_supported_components, "Evaluando respaldo de componentes"
)
restricted_weak_region_smoothing = _con_estado_visible(
    restricted_weak_region_smoothing, "Regularizando regiones débiles"
)
guard_region_motion = _con_estado_visible(
    guard_region_motion, "Verificando orientación e intersecciones"
)
retract_observation_boundaries = _con_estado_visible(
    retract_observation_boundaries, "Verificando y replegando fronteras"
)
colorize = _con_estado_visible(colorize, "Asignando colores a la malla")
evaluate = _con_estado_visible(evaluate, "Midiendo fidelidad y topología")


def _entrada_con_diagnostico():
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    from utilidades_progreso import informar_inicio

    informar_inicio(__file__, "13", "Reconstruir superficie")
    _estado13("Leyendo parámetros, malla y nube de entrada")
    try:
        codigo = main()
        _estado13("Completado" if codigo == 0 else f"Finalizado con observaciones: {codigo}")
        raise SystemExit(codigo)
    except Exception as exc:
        _estado13(f"ERROR: {type(exc).__name__}: {exc}")
        raise
    finally:
        _estado_stop.set()


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
