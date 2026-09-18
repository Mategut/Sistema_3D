#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paso 10 V3.3 — Registro calibrado con evaluación A/B del refinamiento transversal del eje.

Esta fase reemplaza el registro dependiente de geometría durante el runtime.

NO usa:
- cuboide;
- planos;
- Manhattan;
- cilindro;
- pirámide;
- número de caras;
- ICP 6DoF libre;
- correcciones angulares aprendidas del objeto.

Sí usa:
- nube multisesión de cada pose;
- eje físico congelado;
- 2055 pasos/vuelta;
- ángulos mecánicos;
- transformaciones rígidas de la plataforma.

De forma opcional EVALÚA únicamente la POSICIÓN de la línea del eje en sus
dos grados de libertad perpendiculares. La dirección del eje y todos los
ángulos permanecen congelados. En la ruta normal de reconstrucción la
calibración congelada sigue siendo autoritativa: el candidato refinado se
calcula y valida A/B, pero solo se aplica si además se autoriza de forma
explícita con --apply-axis-line-refinement. La evaluación:
- no supone ninguna forma concreta;
- se estima por separado con aristas pares e impares;
- exige mejoras independientes y de cierre;
- selecciona el desplazamiento con mejor equilibrio entre pares, impares y cierre;
- se descarta por completo si resulta inestable;
- se guarda como modelo temporal de poses, sin modificar la calibración.

La validación es geométricamente general:
- solape entre nubes vecinas;
- distancia euclídea;
- distancia punto-a-plano si existen normales;
- cierre P24 -> P00.

Una relación con poco solape se marca "uninformative"; no se interpreta
automáticamente como error de calibración porque objetos muy auto-oclusivos
pueden mostrar poca superficie común.

V3.0 elimina la hipótesis de escena estática: en objetos simétricos podía borrar
superficie real porque una pared giratoria puede parecer inmóvil en la cámara.
Cada voxel registrado se proyecta ahora sobre las 25 siluetas. Salir de una
silueta visible constituye una contradicción geométrica; caer dentro es
compatible, y quedar fuera del campo de visión no aporta evidencia. La
profundidad se conserva como comprobación secundaria de visibilidad/oclusión.
Solo se recuperan fronteras débiles conectadas. No se habilita ICP 6DoF ni se
supone ninguna forma, tamaño, número de piezas o color.
"""

from __future__ import annotations
from utilidades_progreso import operacion
from utilidades_rendimiento import query_threads
from utilidades_rendimiento import configurar as _configurar_recursos

# Los runtimes numéricos leen estos límites al importarse.
_configurar_recursos(__file__)


import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception as exc:
    raise SystemExit(f"Paso 10 requiere OpenCV: {exc}")

try:
    from scipy.spatial import cKDTree
    from scipy.optimize import minimize
    from scipy import ndimage
except Exception as exc:
    raise SystemExit(f"Paso 10 requiere SciPy: {exc}")

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def parser():
    """Construye las opciones de línea de comandos de este paso."""
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--object", required=True)
    p.add_argument(
        "--calibration",
        required=True,
        help="Calibración de plataforma vigente. Nunca se buscan resultados previos.",
    )
    p.add_argument(
        "--cloud-source",
        default="06_nubes_puntos",
    )
    p.add_argument(
        "--output-name",
        default="10_registro_calibrado",
    )
    p.add_argument("--expected-views", type=int, default=25)
    p.add_argument("--sample-points-per-view", type=int, default=12000)
    p.add_argument("--seed", type=int, default=500)

    # Métricas generales. No dependen de forma específica.
    p.add_argument("--overlap-distance-mm", type=float, default=4.5)
    p.add_argument("--metric-cap-mm", type=float, default=15.0)
    p.add_argument("--minimum-informative-overlap", type=float, default=0.12)
    p.add_argument("--surface-maximum-point-plane-rmse-mm", type=float, default=2.8)
    p.add_argument("--surface-maximum-point-plane-p90-mm", type=float, default=4.5)

    p.add_argument("--minimum-informative-primary-ratio", type=float, default=0.60)
    p.add_argument("--minimum-accepted-informative-primary-ratio", type=float, default=0.85)

    # Cierre: solo obligatorio si existe suficiente superficie común.
    p.add_argument("--closure-minimum-overlap", type=float, default=0.20)
    p.add_argument("--closure-maximum-point-plane-rmse-mm", type=float, default=2.8)
    p.add_argument("--closure-maximum-point-plane-p90-mm", type=float, default=4.5)

    p.add_argument("--union-voxel-mm", type=float, default=0.8)
    p.add_argument("--minimum-point-confidence", type=float, default=0.20)
    p.add_argument("--maximum-point-spread-mm", type=float, default=12.0)
    p.add_argument("--point-spread-reference-mm", type=float, default=6.0)
    p.add_argument("--minimum-reliable-points-per-view", type=int, default=2200)
    p.add_argument("--confidence-sampling-power", type=float, default=1.5)

    # V3.0 — tallado global por siluetas y profundidad como evidencia secundaria.
    p.add_argument("--reprojection-neighbor-pose-radius", type=int, default=3)
    p.add_argument("--reprojection-pixel-radius", type=float, default=2.5)
    p.add_argument("--reprojection-depth-z-score", type=float, default=3.5)
    p.add_argument("--reprojection-noise-floor-mm", type=float, default=1.0)
    p.add_argument("--reprojection-minimum-support-poses", type=int, default=2)
    p.add_argument("--reprojection-minimum-independent-agreements", type=int, default=2)
    p.add_argument("--reprojection-minimum-agreement-ratio", type=float, default=0.70)
    p.add_argument("--reprojection-minimum-angular-span-poses", type=int, default=2)
    p.add_argument("--reprojection-maximum-contradictions", type=int, default=1)
    p.add_argument("--reprojection-front-facing-cosine", type=float, default=0.05)
    p.add_argument("--weak-boundary-minimum-foreground-score", type=float, default=0.45)
    p.add_argument("--weak-boundary-recovery-hops", type=int, default=2)
    p.add_argument("--minimum-clean-union-voxels", type=int, default=500)
    p.add_argument("--silhouette-boundary-tolerance-px", type=float, default=4.0)
    p.add_argument("--silhouette-minimum-tested-poses", type=int, default=12)
    p.add_argument("--silhouette-strong-inside-ratio", type=float, default=0.88)
    p.add_argument("--silhouette-weak-inside-ratio", type=float, default=0.76)
    p.add_argument("--silhouette-maximum-strong-contradictions", type=int, default=3)
    p.add_argument("--silhouette-maximum-weak-contradictions", type=int, default=6)
    p.add_argument("--static-hypothesis-neighbor-radius", type=int, default=4)
    p.add_argument("--static-hypothesis-gate-mm", type=float, default=2.4)
    p.add_argument("--static-hypothesis-minimum-matches", type=int, default=3)
    p.add_argument("--static-hypothesis-dominance-margin", type=int, default=2)
    p.add_argument("--static-hypothesis-maximum-removal-ratio", type=float, default=0.30)

    # Refinamiento experimental. La ruta normal usa íntegramente la calibración
    # mecánica porque esta traslación es ambigua en objetos simétricos.
    p.add_argument(
        "--refine-axis-line",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnóstico opcional: refina la línea del eje con el objeto. "
            "Desactivado por defecto para mantener poses congeladas."
        ),
    )
    p.add_argument(
        "--apply-axis-line-refinement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Autoriza aplicar el candidato refinado si supera todas las guardas A/B. "
            "Por defecto solo se evalúa y se conserva la calibración congelada."
        ),
    )
    p.add_argument("--axis-line-search-limit-mm", type=float, default=12.0)
    p.add_argument("--axis-line-optimization-points", type=int, default=1800)
    p.add_argument("--axis-line-trim-axial-percent", type=float, default=12.0)
    p.add_argument("--axis-line-trim-transverse-percent", type=float, default=2.0)
    # V3.2 separa el gate barato de propuesta del gate final. Un candidato
    # pequeño y físicamente seguro puede llegar a validación completa aunque no
    # alcance la ganancia del mejor candidato (que puede exigir un desplazamiento
    # no permitido por incertidumbre).
    p.add_argument(
        "--axis-line-minimum-improvement-ratio",
        type=float,
        default=0.06,
        help="Parámetro legacy conservado por compatibilidad; V3.2 usa los gates proposal/reserved/full.",
    )
    p.add_argument("--axis-line-proposal-minimum-improvement-ratio", type=float, default=0.005)
    p.add_argument("--axis-line-reserved-minimum-improvement-ratio", type=float, default=0.005)
    p.add_argument("--axis-line-maximum-split-disagreement-mm", type=float, default=3.5)
    p.add_argument(
        "--axis-line-uncertainty-offset-factor",
        type=float,
        default=1.50,
        help=(
            "Límite inicial del desplazamiento como múltiplo de la incertidumbre "
            "mediana de profundidad; una ganancia independiente demostrada puede "
            "ampliarlo de forma acotada."
        ),
    )
    p.add_argument("--axis-line-maximum-uncertainty-offset-mm", type=float, default=4.0)
    p.add_argument(
        "--axis-line-conservative-gain-fraction",
        type=float,
        default=0.82,
        help=(
            "Fracción mínima de la mejor ganancia observada que debe conservar "
            "un candidato. Se mantiene por compatibilidad con ejecuciones previas."
        ),
    )
    p.add_argument(
        "--axis-line-minimum-closure-improvement-ratio",
        type=float,
        default=0.0,
        help="Mejora mínima exigida en el cierre para aceptar un candidato.",
    )
    p.add_argument(
        "--axis-line-cross-validation-balance-weight",
        type=float,
        default=0.10,
        help="Penalización por desequilibrio entre las validaciones par e impar.",
    )
    # V3.1 — aceptación final sobre TODOS los puntos fiables. Además de la
    # mejora global, ninguna pareja consecutiva puede degradarse de forma
    # material para ganar solamente el cierre P24->P00.
    p.add_argument("--axis-line-full-minimum-rmse-improvement-ratio", type=float, default=0.02)
    p.add_argument("--axis-line-full-minimum-p90-improvement-ratio", type=float, default=0.01)
    p.add_argument(
        "--axis-line-maximum-consecutive-rmse-degradation-ratio", type=float, default=0.015
    )
    p.add_argument(
        "--axis-line-maximum-consecutive-p90-degradation-ratio", type=float, default=0.015
    )
    p.add_argument(
        "--axis-line-maximum-consecutive-absolute-degradation-mm", type=float, default=0.03
    )
    p.add_argument("--axis-line-maximum-consecutive-overlap-drop", type=float, default=0.015)
    p.add_argument("--axis-line-maximum-degraded-consecutive-edges", type=int, default=0)

    # V3.2 — validación espacial por bandas axiales. No presupone forma del
    # objeto: usa únicamente el eje físico ya calibrado y compara las mismas
    # regiones antes/después.
    p.add_argument("--axis-line-regional-bands", type=int, default=3)
    p.add_argument("--axis-line-regional-minimum-points", type=int, default=250)
    p.add_argument("--axis-line-regional-maximum-rmse-degradation-ratio", type=float, default=0.02)
    p.add_argument("--axis-line-regional-maximum-p90-degradation-ratio", type=float, default=0.02)
    p.add_argument("--axis-line-regional-maximum-absolute-degradation-mm", type=float, default=0.05)
    p.add_argument("--axis-line-regional-maximum-overlap-drop", type=float, default=0.02)
    p.add_argument("--axis-line-maximum-degraded-regions", type=int, default=0)

    # Salud del registro: no sustituye los gates duros de rechazo. Sirve para
    # declarar WARNING cuando un registro es utilizable pero claramente más
    # débil que la incertidumbre observada y el cierre global.
    p.add_argument("--registration-warning-rmse-uncertainty-ratio", type=float, default=0.60)
    p.add_argument("--registration-warning-p90-uncertainty-ratio", type=float, default=0.95)
    p.add_argument("--registration-warning-closure-overlap", type=float, default=0.85)
    p.add_argument("--registration-warning-input-confidence-median", type=float, default=0.72)

    p.add_argument("--axis-line-maximum-evaluations", type=int, default=220)
    return p


def load_json(path: Path):
    """Lee un documento JSON codificado en UTF-8."""
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_rows(n):
    n = np.asarray(n, dtype=np.float64)
    if n.ndim != 2 or n.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float64)
    l = np.linalg.norm(n, axis=1)
    good = np.isfinite(l) & (l > 1e-12)
    out = np.zeros_like(n)
    out[good] = n[good] / l[good, None]
    return out


def transform_points(p, T):
    """Aplica a los puntos la rotación y traslación de una matriz homogénea."""
    p = np.asarray(p, dtype=np.float64)
    return p @ T[:3, :3].T + T[:3, 3]


def transform_normals(n, T):
    """Aplica solo la rotación a las normales y normaliza el resultado."""
    n = np.asarray(n, dtype=np.float64)
    if len(n) == 0:
        return n
    return normalize_rows(n @ T[:3, :3].T)


def stats(values):
    a = np.asarray(list(values), dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"count": 0, "median": None, "mean": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(len(a)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def weighted_quantile(values, weights, quantile):
    """Calcula un cuantil usando valores finitos y pesos positivos."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values, weights = values[valid], weights[valid]
    if len(values) == 0:
        return None
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    target = float(np.clip(quantile, 0.0, 1.0)) * float(np.sum(weights))
    index = int(np.searchsorted(np.cumsum(weights), target, side="left"))
    return float(values[min(index, len(values) - 1)])


def weighted_stats(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values, weights = values[valid], weights[valid]
    if len(values) == 0:
        return {
            "count": 0,
            "effective_weight": 0.0,
            "median": None,
            "mean": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    denominator = float(np.sum(weights))
    return {
        "count": int(len(values)),
        "effective_weight": denominator,
        "median": weighted_quantile(values, weights, 0.50),
        "mean": float(np.sum(weights * values) / max(denominator, 1e-12)),
        "p90": weighted_quantile(values, weights, 0.90),
        "p95": weighted_quantile(values, weights, 0.95),
        "max": float(np.max(values)),
    }


def resolve_calibration(root: Path, explicit: str) -> Path:
    """Exige una ruta explícita a la calibración de plataforma existente."""
    if not explicit:
        raise ValueError("--calibration es obligatorio en modo producto.")
    p = Path(explicit).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def validate_platform_calibration(cal: dict, expected_views: int = 25) -> None:
    """Rechaza calibraciones incompletas o incompatibles antes del registro."""
    allowed_status = {
        "candidate_ready_for_independent_validation",
        "accepted",
        "active",
    }
    status = str(cal.get("status", "")).strip().lower()
    if status not in allowed_status:
        raise RuntimeError(f"Estado de calibración no utilizable: {status or 'missing'}.")
    if (
        status == "candidate_ready_for_independent_validation"
        and cal.get("candidate_quality_passed") is not True
    ):
        raise RuntimeError("La calibración candidata no acredita los gates de calidad.")

    mechanical = cal.get("mechanical_model", {})
    if mechanical.get("steps_per_revolution") != 2055:
        raise RuntimeError("La calibración no corresponde a 2055 pasos.")
    if mechanical.get("poses_per_revolution") != expected_views:
        raise RuntimeError(f"La calibración no corresponde a {expected_views} poses.")
    sequence = mechanical.get("step_sequence")
    if not isinstance(sequence, list) or len(sequence) != expected_views:
        raise RuntimeError("La secuencia mecánica de pasos está ausente o incompleta.")
    try:
        sequence = [int(value) for value in sequence]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("La secuencia mecánica contiene pasos inválidos.") from exc
    if sum(sequence) != 2055 or sequence != [82, 82, 83, 82, 82] * 5:
        raise RuntimeError("La secuencia mecánica no coincide con 2055 pasos/vuelta.")

    poses = cal.get("poses")
    if not isinstance(poses, list) or len(poses) != expected_views:
        raise RuntimeError("La calibración no contiene exactamente 25 poses.")
    try:
        indices = [int(record["pose_index"]) for record in poses]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Los índices de pose de la calibración son inválidos.") from exc
    if sorted(indices) != list(range(expected_views)) or len(set(indices)) != expected_views:
        raise RuntimeError("Las poses de calibración no son P00..P24 únicas.")

    for record in poses:
        transform = np.asarray(
            record.get("transform_pose_to_P00_mechanical"),
            dtype=np.float64,
        )
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise RuntimeError(f"Transformación inválida en P{int(record['pose_index']):02d}.")
        if abs(float(record.get("runtime_correction_deg", 0.0))) > 1e-12:
            raise RuntimeError("La calibración runtime contiene corrección por objeto.")


def resolve_cloud_summary(root: Path, obj: str, source: str):
    """Localiza las nubes y su resumen multisesión del paso 06."""
    folder = root / "reconstruccion" / "multisesion" / source
    summary = folder / "resumen_06_nubes_puntos.json"
    if not summary.is_file():
        raise FileNotFoundError(f"Falta la salida 06 de ESTA ejecución: {summary}")
    return folder, summary


def load_views(cloud_dir: Path, summary: dict, expected: int, args, intrinsics):
    """Carga las vistas y sus atributos para el registro con la plataforma calibrada."""
    records = summary.get("views", [])
    if not records:
        raise ValueError("El resumen de 5.2 no contiene views.")

    views = []
    for i, rec in enumerate(records):
        pose = int(rec.get("pose_index", i))
        if pose < 0 or pose >= expected:
            continue
        if str(rec.get("quality", "")).lower() == "rejected":
            continue

        stem = str(rec.get("stem", f"P{pose:02d}"))
        candidates = [
            cloud_dir / f"{stem}_cloud_final.npz",
            cloud_dir / f"P{pose:02d}_cloud_final.npz",
        ]
        cloud_path = next((p for p in candidates if p.is_file()), None)
        if cloud_path is None:
            raise FileNotFoundError(f"No se encontró nube P{pose:02d}.")

        with np.load(cloud_path) as data:
            points = np.asarray(data["points"], dtype=np.float64)
            colors = (
                np.asarray(data["colors"], dtype=np.uint8)
                if "colors" in data.files
                else np.zeros((len(points), 3), dtype=np.uint8)
            )
            normals = (
                np.asarray(data["normals"], dtype=np.float64)
                if "normals" in data.files
                else np.empty((0, 3), dtype=np.float64)
            )
            confidence = (
                np.asarray(data["confidence"], dtype=np.float64)
                if "confidence" in data.files
                else np.ones(len(points), dtype=np.float64)
            )
            spread = (
                np.asarray(data["depth_spread_mm"], dtype=np.float64)
                if "depth_spread_mm" in data.files
                else np.full(len(points), float(args.point_spread_reference_mm))
            )
            depth_support = (
                np.asarray(data["depth_support"], dtype=np.float64)
                if "depth_support" in data.files
                else np.full(len(points), 2.0, dtype=np.float64)
            )
            foreground_score = (
                np.asarray(data["foreground_score"], dtype=np.float64)
                if "foreground_score" in data.files
                else confidence.copy()
            )
            background_residual_sigma = (
                np.asarray(data["background_residual_sigma"], dtype=np.float64)
                if "background_residual_sigma" in data.files
                else np.full(len(points), np.nan, dtype=np.float64)
            )
            depth_uncertainty = (
                np.asarray(data["depth_uncertainty_mm"], dtype=np.float64)
                if "depth_uncertainty_mm" in data.files
                else np.maximum(spread, float(args.reprojection_noise_floor_mm))
            )
            component_id = (
                np.asarray(data["component_id"], dtype=np.int32)
                if "component_id" in data.files
                else np.ones(len(points), dtype=np.int32)
            )
            stereo_score = (
                np.asarray(data["stereo_score"], dtype=np.float64)
                if "stereo_score" in data.files
                else foreground_score.copy()
            )
            disparity_uncertainty = (
                np.asarray(data["disparity_uncertainty_px"], dtype=np.float64)
                if "disparity_uncertainty_px" in data.files
                else np.full(len(points), 1.0, dtype=np.float64)
            )
            observed_sessions = (
                np.asarray(data["observed_sessions"], dtype=np.float64)
                if "observed_sessions" in data.files
                else depth_support.copy()
            )
            if "pixel_uv" in data.files:
                pixel_uv = np.asarray(data["pixel_uv"], dtype=np.float64)
            else:
                z = np.maximum(points[:, 2], 1e-8)
                pixel_uv = np.column_stack(
                    (
                        intrinsics["fx"] * points[:, 0] / z + intrinsics["cx"],
                        intrinsics["fy"] * points[:, 1] / z + intrinsics["cy"],
                    )
                )
            image_shape = (
                tuple(np.asarray(data["image_shape_hw"], dtype=np.int32).reshape(-1)[:2])
                if "image_shape_hw" in data.files
                else None
            )

        finite = np.all(np.isfinite(points), axis=1)
        original_count = len(points)
        colors = (
            colors if len(colors) == original_count else np.zeros((original_count, 3), np.uint8)
        )
        normals_valid = len(normals) == original_count
        confidence = confidence if len(confidence) == original_count else np.ones(original_count)
        spread = (
            spread
            if len(spread) == original_count
            else np.full(original_count, float(args.point_spread_reference_mm))
        )
        depth_support = (
            depth_support if len(depth_support) == original_count else np.full(original_count, 2.0)
        )
        foreground_score = (
            foreground_score if len(foreground_score) == original_count else confidence.copy()
        )
        background_residual_sigma = (
            background_residual_sigma
            if len(background_residual_sigma) == original_count
            else np.full(original_count, np.nan)
        )
        depth_uncertainty = (
            depth_uncertainty
            if len(depth_uncertainty) == original_count
            else np.maximum(spread, float(args.reprojection_noise_floor_mm))
        )
        component_id = (
            component_id
            if len(component_id) == original_count
            else np.ones(original_count, np.int32)
        )
        stereo_score = (
            stereo_score if len(stereo_score) == original_count else foreground_score.copy()
        )
        disparity_uncertainty = (
            disparity_uncertainty
            if len(disparity_uncertainty) == original_count
            else np.full(original_count, 1.0)
        )
        observed_sessions = (
            observed_sessions if len(observed_sessions) == original_count else depth_support.copy()
        )
        pixel_uv = (
            pixel_uv
            if pixel_uv.shape == (original_count, 2)
            else np.full((original_count, 2), np.nan)
        )
        if image_shape is None or len(image_shape) != 2 or min(image_shape) <= 0:
            finite_uv = np.all(np.isfinite(pixel_uv), axis=1)
            if not np.any(finite_uv):
                raise RuntimeError(f"P{pose:02d} no contiene pixel_uv ni dimensiones de imagen.")
            image_shape = (
                int(np.ceil(np.max(pixel_uv[finite_uv, 1]))) + 2,
                int(np.ceil(np.max(pixel_uv[finite_uv, 0]))) + 2,
            )

        confidence = np.where(np.isfinite(confidence), np.clip(confidence, 0.0, 1.0), 0.0)
        spread = np.where(np.isfinite(spread), np.maximum(spread, 0.0), np.inf)
        reliability = (
            confidence
            * np.exp(
                -0.5
                * np.square(
                    np.minimum(spread, 4.0 * float(args.point_spread_reference_mm))
                    / max(float(args.point_spread_reference_mm), 1e-6)
                )
            )
            * (0.75 + 0.25 * np.clip(depth_support / 3.0, 0.0, 1.0))
        )
        reliability *= 0.40 + 0.60 * np.clip(foreground_score, 0.0, 1.0)
        reliability *= 0.35 + 0.65 * np.clip(stereo_score, 0.0, 1.0)
        reliable = (
            finite
            & (confidence >= float(args.minimum_point_confidence))
            & (spread <= float(args.maximum_point_spread_mm))
        )
        minimum = min(
            original_count,
            max(100, int(args.minimum_reliable_points_per_view)),
        )
        fallback_used = False
        if int(np.count_nonzero(reliable)) < minimum:
            candidates = np.flatnonzero(finite)
            if len(candidates) < minimum:
                raise RuntimeError(f"P{pose:02d} contiene solo {len(candidates)} puntos finitos.")
            order = candidates[np.argsort(reliability[candidates])[::-1]]
            reliable = np.zeros(original_count, dtype=bool)
            reliable[order[:minimum]] = True
            fallback_used = True

        points = points[reliable]
        colors = colors[reliable]
        confidence = confidence[reliable]
        spread = spread[reliable]
        depth_support = depth_support[reliable]
        foreground_score = np.clip(foreground_score[reliable], 0.0, 1.0)
        background_residual_sigma = background_residual_sigma[reliable]
        depth_uncertainty = np.maximum(
            depth_uncertainty[reliable], float(args.reprojection_noise_floor_mm)
        )
        component_id = component_id[reliable]
        stereo_score = np.clip(stereo_score[reliable], 0.0, 1.0)
        disparity_uncertainty = np.maximum(disparity_uncertainty[reliable], 1e-3)
        observed_sessions = np.maximum(observed_sessions[reliable], 0.0)
        pixel_uv = pixel_uv[reliable]
        reliability = np.clip(reliability[reliable], 1e-4, 1.0)
        if normals_valid:
            normals = normalize_rows(normals[reliable])
        else:
            normals = np.empty((0, 3), dtype=np.float64)

        views.append(
            {
                "pose_index": pose,
                "stem": stem,
                "points": points,
                "colors": colors,
                "normals": normals,
                "confidence": confidence,
                "depth_spread_mm": spread,
                "depth_support": depth_support,
                "foreground_score": foreground_score,
                "background_residual_sigma": background_residual_sigma,
                "depth_uncertainty_mm": depth_uncertainty,
                "component_id": component_id,
                "stereo_score": stereo_score,
                "disparity_uncertainty_px": disparity_uncertainty,
                "observed_sessions": observed_sessions,
                "pixel_uv": pixel_uv,
                "image_shape_hw": tuple(int(v) for v in image_shape),
                "reliability": reliability,
                "reliability_filter": {
                    "points_before": int(original_count),
                    "points_after": int(len(points)),
                    "retained_ratio": float(len(points) / max(original_count, 1)),
                    "fallback_top_reliability_used": bool(fallback_used),
                    "confidence_median": float(np.median(confidence)),
                    "spread_p90_mm": float(np.percentile(spread, 90)),
                    "foreground_score_median": float(np.median(foreground_score)),
                    "stereo_score_median": float(np.median(stereo_score)),
                    "observed_sessions_median": float(np.median(observed_sessions)),
                },
                "cloud_path": str(cloud_path),
            }
        )

    views.sort(key=lambda r: r["pose_index"])
    if len(views) != expected:
        raise RuntimeError(f"Se esperaban {expected} nubes y se cargaron {len(views)}.")
    if [r["pose_index"] for r in views] != list(range(expected)):
        raise RuntimeError("Las poses no son P00..P24 continuas.")
    return views


def _filter_view_rows(view: dict, keep: np.ndarray) -> dict:
    """Filtra solamente arreglos cuya primera dimensión corresponde a puntos."""
    keep = np.asarray(keep, dtype=bool)
    result = {}
    for name, value in view.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == len(keep):
            result[name] = value[keep]
        else:
            result[name] = value
    return result


def reject_camera_static_hypothesis(registered_views, args):
    """Rechaza puntos más coherentes fijos en cámara que girando con el objeto.

    La comparación usa exactamente las mismas poses y el mismo radio para las
    dos hipótesis. Una figura simétrica puede ser coherente bajo ambas y no se
    elimina: solo se rechaza cuando la hipótesis estática domina por un margen
    explícito. El límite por pose evita cortes masivos ante evidencia ambigua.
    """
    expected = len(registered_views)
    radius = max(1, int(args.static_hypothesis_neighbor_radius))
    gate = max(float(args.static_hypothesis_gate_mm), 1e-6)
    minimum_matches = max(1, int(args.static_hypothesis_minimum_matches))
    dominance_margin = max(1, int(args.static_hypothesis_dominance_margin))
    maximum_removal_ratio = float(np.clip(args.static_hypothesis_maximum_removal_ratio, 0.0, 0.95))

    camera_trees = [cKDTree(v["camera_points"]) for v in registered_views]
    object_trees = [cKDTree(v["points"]) for v in registered_views]
    filtered = []
    diagnostics = []

    for source_pose, view in enumerate(registered_views):
        count = len(view["points"])
        static_matches = np.zeros(count, np.uint8)
        object_matches = np.zeros(count, np.uint8)
        tested_targets = set()
        for offset in range(1, radius + 1):
            tested_targets.add((source_pose - offset) % expected)
            tested_targets.add((source_pose + offset) % expected)
        tested_targets.discard(source_pose)
        for target_pose in sorted(tested_targets):
            static_distance, _ = camera_trees[target_pose].query(
                view["camera_points"],
                k=1,
                distance_upper_bound=gate,
                workers=query_threads(),
            )
            object_distance, _ = object_trees[target_pose].query(
                view["points"],
                k=1,
                distance_upper_bound=gate,
                workers=query_threads(),
            )
            static_matches += np.isfinite(static_distance).astype(np.uint8)
            object_matches += np.isfinite(object_distance).astype(np.uint8)

        dominance = static_matches.astype(np.int16) - object_matches.astype(np.int16)
        reject_candidate = (static_matches >= minimum_matches) & (dominance >= dominance_margin)
        candidate_indexes = np.flatnonzero(reject_candidate)
        maximum_remove = int(math.floor(maximum_removal_ratio * count))
        if len(candidate_indexes) > maximum_remove:
            # Orden conservador: domina más la escena estática, tiene más
            # coincidencias estáticas y menos coincidencias como objeto.
            order = np.lexsort(
                (
                    object_matches[candidate_indexes],
                    -static_matches[candidate_indexes].astype(np.int16),
                    -dominance[candidate_indexes],
                )
            )
            candidate_indexes = candidate_indexes[order[:maximum_remove]]
        reject = np.zeros(count, bool)
        reject[candidate_indexes] = True
        annotated = dict(view)
        annotated["static_hypothesis_matches"] = static_matches
        annotated["object_hypothesis_matches"] = object_matches
        annotated["static_hypothesis_dominance"] = dominance
        filtered_view = _filter_view_rows(annotated, ~reject)
        filtered.append(filtered_view)
        diagnostics.append(
            {
                "pose_index": int(view["pose_index"]),
                "points_before": int(count),
                "static_candidates": int(np.count_nonzero(reject_candidate)),
                "points_removed": int(np.count_nonzero(reject)),
                "points_after": int(np.count_nonzero(~reject)),
                "removal_ratio": float(np.mean(reject)) if count else 0.0,
                "static_matches": stats(static_matches),
                "object_matches": stats(object_matches),
                "static_dominance": stats(dominance),
                "removal_was_capped": bool(np.count_nonzero(reject_candidate) > maximum_remove),
            }
        )
    return filtered, diagnostics


def voxel_downsample_numpy(points, colors, voxel):
    if len(points) == 0 or voxel <= 0:
        return points, colors
    keys = np.floor(points / float(voxel)).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    return points[idx], colors[idx]


def aggregate_by_inverse(values, inverse, count, weights):
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=np.float64)
    denominator = np.bincount(inverse, weights=weights, minlength=count)
    if values.ndim == 1:
        numerator = np.bincount(
            inverse,
            weights=weights * values.astype(np.float64),
            minlength=count,
        )
        return numerator / np.maximum(denominator, 1e-12)
    result = np.zeros((count, values.shape[1]), dtype=np.float64)
    for axis_index in range(values.shape[1]):
        result[:, axis_index] = np.bincount(
            inverse,
            weights=weights * values[:, axis_index].astype(np.float64),
            minlength=count,
        )
    return result / np.maximum(denominator[:, None], 1e-12)


_POPCOUNT_BYTES = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def bit_count_u64(values):
    """Conteo exacto mediante tabla de bytes; evita un array N x 64."""
    values = np.ascontiguousarray(values, dtype=np.uint64)
    byte_values = values.view(np.uint8).reshape(-1, 8)
    return _POPCOUNT_BYTES[byte_values].sum(axis=1).astype(np.int16)


def pose_bit(pose):
    """Representa una pose mediante un único bit en un entero de 64 bits."""
    return np.left_shift(np.uint64(1), np.uint64(int(pose)))


def pose_mask_angular_span(values, expected_views):
    """Mayor separación circular entre dos poses presentes en cada máscara."""
    values = np.asarray(values, dtype=np.uint64)
    expected_views = max(1, int(expected_views))
    limit = np.uint64((1 << expected_views) - 1)
    span = np.zeros(len(values), np.uint8)
    for offset in range(1, expected_views // 2 + 1):
        rotated = (
            np.left_shift(values, np.uint64(offset))
            | np.right_shift(values, np.uint64(expected_views - offset))
        ) & limit
        span[(values & rotated) != 0] = np.uint8(offset)
    return span


def fuse_voxels_with_pose(registered_views, voxel_mm):
    """Fusiona por voxel garantizando como máximo un aporte por pose."""
    voxel_mm = max(float(voxel_mm), 1e-6)
    all_points = []
    all_colors = []
    all_normals = []
    all_reliability = []
    all_foreground = []
    all_stereo = []
    all_observed_sessions = []
    all_uncertainty = []
    all_poses = []
    for view in registered_views:
        count = len(view["points"])
        if count == 0:
            continue
        all_points.append(np.asarray(view["points"], dtype=np.float64))
        all_colors.append(np.asarray(view["colors"], dtype=np.float64))
        normals = np.asarray(view.get("normals", np.empty((0, 3))), dtype=np.float64)
        all_normals.append(normals if len(normals) == count else np.zeros((count, 3), np.float64))
        all_reliability.append(np.asarray(view["reliability"], dtype=np.float64))
        all_foreground.append(np.asarray(view["foreground_score"], dtype=np.float64))
        all_stereo.append(np.asarray(view["stereo_score"], dtype=np.float64))
        all_observed_sessions.append(np.asarray(view["observed_sessions"], dtype=np.float64))
        all_uncertainty.append(np.asarray(view["depth_uncertainty_mm"], dtype=np.float64))
        all_poses.append(np.full(count, int(view["pose_index"]), np.int16))

    if not all_points:
        raise RuntimeError("No hay puntos registrados para fusionar.")

    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    normals = np.vstack(all_normals)
    reliability = np.clip(np.concatenate(all_reliability), 1e-4, 1.0)
    foreground = np.clip(np.concatenate(all_foreground), 0.0, 1.0)
    stereo = np.clip(np.concatenate(all_stereo), 0.0, 1.0)
    observed_sessions = np.maximum(np.concatenate(all_observed_sessions), 0.0)
    uncertainty = np.maximum(np.concatenate(all_uncertainty), 1e-3)
    poses = np.concatenate(all_poses)
    weights = reliability * (0.30 + 0.70 * foreground) * (0.35 + 0.65 * stereo)

    voxel_keys = np.floor(points / voxel_mm).astype(np.int64)
    pose_voxel_keys = np.column_stack((voxel_keys, poses.astype(np.int64)))
    unique_pose_voxel, inverse_pose_voxel = np.unique(pose_voxel_keys, axis=0, return_inverse=True)
    pose_voxel_count = len(unique_pose_voxel)
    points_pose = aggregate_by_inverse(points, inverse_pose_voxel, pose_voxel_count, weights)
    colors_pose = aggregate_by_inverse(colors, inverse_pose_voxel, pose_voxel_count, weights)
    normals_pose = normalize_rows(
        aggregate_by_inverse(normals, inverse_pose_voxel, pose_voxel_count, weights)
    )
    foreground_pose = aggregate_by_inverse(
        foreground, inverse_pose_voxel, pose_voxel_count, weights
    )
    stereo_pose = aggregate_by_inverse(stereo, inverse_pose_voxel, pose_voxel_count, weights)
    observed_sessions_pose = aggregate_by_inverse(
        observed_sessions, inverse_pose_voxel, pose_voxel_count, weights
    )
    uncertainty_pose = aggregate_by_inverse(
        uncertainty, inverse_pose_voxel, pose_voxel_count, weights
    )
    pose_weight = np.clip(
        np.bincount(inverse_pose_voxel, weights=weights, minlength=pose_voxel_count),
        0.05,
        1.0,
    )
    poses_pose = unique_pose_voxel[:, 3].astype(np.int16)
    keys_pose = unique_pose_voxel[:, :3].astype(np.int64)

    unique_voxel, inverse_voxel = np.unique(keys_pose, axis=0, return_inverse=True)
    voxel_count = len(unique_voxel)
    fused_points = aggregate_by_inverse(points_pose, inverse_voxel, voxel_count, pose_weight)
    fused_colors = aggregate_by_inverse(colors_pose, inverse_voxel, voxel_count, pose_weight)
    fused_normal_raw = aggregate_by_inverse(normals_pose, inverse_voxel, voxel_count, pose_weight)
    fused_normals = normalize_rows(fused_normal_raw)
    normal_consistency = np.linalg.norm(fused_normal_raw, axis=1)
    fused_foreground = np.clip(
        aggregate_by_inverse(foreground_pose, inverse_voxel, voxel_count, pose_weight), 0.0, 1.0
    )
    fused_stereo = np.clip(
        aggregate_by_inverse(stereo_pose, inverse_voxel, voxel_count, pose_weight), 0.0, 1.0
    )
    fused_observed_sessions = np.maximum(
        aggregate_by_inverse(observed_sessions_pose, inverse_voxel, voxel_count, pose_weight), 0.0
    )
    fused_uncertainty = np.maximum(
        aggregate_by_inverse(uncertainty_pose, inverse_voxel, voxel_count, pose_weight), 1e-3
    )

    direct_pose_mask = np.zeros(voxel_count, dtype=np.uint64)
    np.bitwise_or.at(
        direct_pose_mask,
        inverse_voxel,
        np.asarray([pose_bit(p) for p in poses_pose], dtype=np.uint64),
    )
    direct_support = bit_count_u64(direct_pose_mask)
    residual2 = np.sum(np.square(points_pose - fused_points[inverse_voxel]), axis=1)
    spatial_spread = np.sqrt(
        np.maximum(
            np.bincount(
                inverse_voxel,
                weights=pose_weight * residual2,
                minlength=voxel_count,
            )
            / np.maximum(
                np.bincount(inverse_voxel, weights=pose_weight, minlength=voxel_count),
                1e-12,
            ),
            0.0,
        )
    )

    return {
        "points": fused_points,
        "colors": np.clip(np.rint(fused_colors), 0, 255).astype(np.uint8),
        "normals": fused_normals,
        "normal_consistency": np.clip(normal_consistency, 0.0, 1.0),
        "foreground_score": fused_foreground,
        "stereo_score": fused_stereo,
        "observed_sessions": fused_observed_sessions,
        "depth_uncertainty_mm": fused_uncertainty,
        "spatial_spread_mm": spatial_spread,
        "voxel_keys": unique_voxel,
        "direct_pose_mask": direct_pose_mask,
        "direct_support_poses": direct_support,
        "input_points": int(len(points)),
        "pose_voxel_contributions": int(pose_voxel_count),
    }


def build_projection_map(view):
    """Rasteriza una nube del paso 06 conservando su mejor muestra por píxel."""
    height, width = (int(v) for v in view["image_shape_hw"])
    depth = np.full((height, width), np.nan, np.float32)
    uncertainty = np.full((height, width), np.nan, np.float32)
    foreground = np.zeros((height, width), np.float32)
    uv = np.asarray(view["pixel_uv"], dtype=np.float64)
    local_points = np.asarray(view["camera_points"], dtype=np.float64)
    reliability = np.asarray(view["reliability"], dtype=np.float64)
    finite = (
        np.all(np.isfinite(uv), axis=1)
        & np.all(np.isfinite(local_points), axis=1)
        & (local_points[:, 2] > 1e-6)
    )
    u = np.rint(uv[finite, 0]).astype(np.int64)
    v = np.rint(uv[finite, 1]).astype(np.int64)
    source_index = np.flatnonzero(finite)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, source_index = u[inside], v[inside], source_index[inside]
    if len(source_index) == 0:
        raise RuntimeError(f"P{int(view['pose_index']):02d} no contiene muestras proyectables.")

    flat = v * width + u
    order = np.lexsort((reliability[source_index], flat))
    flat_sorted = flat[order]
    last = np.r_[flat_sorted[1:] != flat_sorted[:-1], True]
    chosen = source_index[order[last]]
    chosen_u = np.rint(uv[chosen, 0]).astype(np.int64)
    chosen_v = np.rint(uv[chosen, 1]).astype(np.int64)
    depth[chosen_v, chosen_u] = local_points[chosen, 2].astype(np.float32)
    uncertainty[chosen_v, chosen_u] = view["depth_uncertainty_mm"][chosen].astype(np.float32)
    foreground[chosen_v, chosen_u] = view["foreground_score"][chosen].astype(np.float32)

    valid = np.isfinite(depth)
    distance, nearest = ndimage.distance_transform_edt(~valid, return_indices=True)
    return {
        "depth": depth,
        "uncertainty": uncertainty,
        "foreground": foreground,
        "valid": valid,
        "nearest_distance_px": distance.astype(np.float32),
        "nearest_y": nearest[0].astype(np.int32),
        "nearest_x": nearest[1].astype(np.int32),
        "shape": (height, width),
    }


@operacion("Calcular respaldo multivista por reproyección")
def multiview_reprojection_support(fused, registered_views, transforms, intrinsics, args):
    """Comprueba cada voxel contra todas las vistas mediante espacio libre."""
    points = np.asarray(fused["points"], dtype=np.float64)
    normals = np.asarray(fused["normals"], dtype=np.float64)
    uncertainty = np.asarray(fused["depth_uncertainty_mm"], dtype=np.float64)
    direct = np.asarray(fused["direct_pose_mask"], dtype=np.uint64)
    expected = len(registered_views)
    agreement_mask = np.zeros(len(points), np.uint64)
    contradiction_mask = np.zeros(len(points), np.uint64)
    occluded_mask = np.zeros(len(points), np.uint64)
    tested_mask = np.zeros(len(points), np.uint64)
    per_target = []
    pixel_radius = max(float(args.reprojection_pixel_radius), 0.0)
    z_limit = max(float(args.reprojection_depth_z_score), 1e-6)
    noise_floor = max(float(args.reprojection_noise_floor_mm), 1e-6)
    front_cosine = float(args.reprojection_front_facing_cosine)

    for target_pose in range(expected):
        # Solo se mantiene un mapa denso a la vez. En imágenes Full HD esto
        # evita retener simultáneamente varios GB durante la reproyección.
        projection = build_projection_map(registered_views[target_pose])
        target_bit = pose_bit(target_pose)
        # Se prueban todas las muestras que no proceden de la propia vista.
        # Limitarse a poses vecinas permite que un error sistemático persista
        # durante varios ángulos consecutivos y sea contado como superficie.
        eligible = (direct & target_bit) == 0
        indexes = np.flatnonzero(eligible)
        if len(indexes) == 0:
            per_target.append(
                {
                    "target_pose": target_pose,
                    "eligible": 0,
                    "tested": 0,
                    "agreed": 0,
                    "occluded": 0,
                    "contradicted": 0,
                }
            )
            continue

        inverse_transform = np.linalg.inv(transforms[target_pose])
        camera_points = transform_points(points[indexes], inverse_transform)
        camera_normals = transform_normals(normals[indexes], inverse_transform)
        z = camera_points[:, 2]
        u_float = intrinsics["fx"] * camera_points[:, 0] / np.maximum(z, 1e-8) + intrinsics["cx"]
        v_float = intrinsics["fy"] * camera_points[:, 1] / np.maximum(z, 1e-8) + intrinsics["cy"]
        u = np.rint(u_float).astype(np.int64)
        v = np.rint(v_float).astype(np.int64)
        height, width = projection["shape"]
        inside = (
            np.isfinite(z)
            & (z > 1e-6)
            & np.isfinite(u_float)
            & np.isfinite(v_float)
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        local_indexes = np.flatnonzero(inside)
        if len(local_indexes) == 0:
            per_target.append(
                {
                    "target_pose": target_pose,
                    "eligible": int(len(indexes)),
                    "tested": 0,
                    "agreed": 0,
                    "occluded": 0,
                    "contradicted": 0,
                }
            )
            continue

        global_indexes = indexes[local_indexes]
        ui, vi = u[local_indexes], v[local_indexes]
        ray_to_camera = -camera_points[local_indexes]
        ray_length = np.linalg.norm(ray_to_camera, axis=1)
        good_normal = np.linalg.norm(camera_normals[local_indexes], axis=1) > 1e-8
        facing = np.zeros(len(local_indexes), bool)
        valid_facing = good_normal & (ray_length > 1e-8)
        facing[valid_facing] = (
            np.sum(
                camera_normals[local_indexes][valid_facing]
                * ray_to_camera[valid_facing]
                / ray_length[valid_facing, None],
                axis=1,
            )
            >= front_cosine
        )

        nearest_distance = projection["nearest_distance_px"][vi, ui]
        has_measurement = nearest_distance <= pixel_radius
        nearest_y = projection["nearest_y"][vi, ui]
        nearest_x = projection["nearest_x"][vi, ui]
        observed_depth = projection["depth"][nearest_y, nearest_x].astype(np.float64)
        observed_uncertainty = projection["uncertainty"][nearest_y, nearest_x].astype(np.float64)
        observed_foreground = projection["foreground"][nearest_y, nearest_x].astype(np.float64)
        has_measurement &= (
            np.isfinite(observed_depth)
            & np.isfinite(observed_uncertainty)
            & (observed_foreground >= float(args.weak_boundary_minimum_foreground_score))
        )

        combined_sigma = np.sqrt(
            np.square(np.maximum(uncertainty[global_indexes], noise_floor))
            + np.square(np.maximum(observed_uncertainty, noise_floor))
            + noise_floor * noise_floor
        )
        tolerance = z_limit * combined_sigma
        delta = z[local_indexes] - observed_depth
        agreed = has_measurement & (np.abs(delta) <= tolerance)
        occluded = has_measurement & (delta > tolerance)
        # La ausencia de profundidad no demuestra fondo: puede ser oclusión,
        # baja textura o un hueco del estéreo. Solo una medición válida situada
        # claramente detrás del voxel constituye contradicción de profundidad.
        # Si una medición válida está detrás del candidato, el segmento entre
        # cámara y superficie observada es espacio libre. Esto contradice el
        # voxel aunque su normal sea ruidosa o esté invertida. Las normales no
        # deben poder proteger una protuberancia falsa pegada a la superficie.
        contradicted = has_measurement & (delta < -tolerance)
        tested = agreed | occluded | contradicted

        agreement_mask[global_indexes[agreed]] |= target_bit
        occluded_mask[global_indexes[occluded]] |= target_bit
        contradiction_mask[global_indexes[contradicted]] |= target_bit
        tested_mask[global_indexes[tested]] |= target_bit
        per_target.append(
            {
                "target_pose": int(target_pose),
                "eligible": int(len(indexes)),
                "tested": int(np.count_nonzero(tested)),
                "agreed": int(np.count_nonzero(agreed)),
                "occluded": int(np.count_nonzero(occluded)),
                "contradicted": int(np.count_nonzero(contradicted)),
            }
        )

    support_mask = direct | agreement_mask
    decisive = bit_count_u64(agreement_mask) + bit_count_u64(contradiction_mask)
    agreement_count = bit_count_u64(agreement_mask)
    agreement_ratio = np.divide(
        agreement_count.astype(np.float32),
        np.maximum(decisive.astype(np.float32), 1.0),
    )
    return {
        "support_pose_mask": support_mask,
        "agreement_pose_mask": agreement_mask,
        "contradiction_pose_mask": contradiction_mask,
        "occluded_pose_mask": occluded_mask,
        "tested_pose_mask": tested_mask,
        "support_poses": bit_count_u64(support_mask),
        "reprojected_agreement_poses": agreement_count,
        "contradiction_poses": bit_count_u64(contradiction_mask),
        "occluded_poses": bit_count_u64(occluded_mask),
        "tested_poses": bit_count_u64(tested_mask),
        "agreement_ratio": agreement_ratio,
        # La diversidad angular se calcula solo entre confirmaciones ajenas;
        # la pose que originó el voxel no puede mejorar esta métrica.
        "angular_span_poses": pose_mask_angular_span(agreement_mask, expected),
        "per_target": per_target,
    }


def load_consensus_silhouettes(root, cloud_summary, registered_views):
    """Carga una silueta binaria por pose usando rutas reproducibles."""
    source_name = str(
        cloud_summary.get("consensus_source_name")
        or cloud_summary.get("parameters", {}).get("consensus_source", "05_consenso_multisesion")
    )
    folder = root / "reconstruccion" / "multisesion" / source_name
    masks = {}
    sources = {}
    for view in registered_views:
        pose = int(view["pose_index"])
        stem = str(view.get("stem", f"P{pose:02d}"))
        candidates = (
            folder / f"{stem}_silhouette_consensus.png",
            folder / f"{stem}_cloud_mask_consensus.png",
            folder / f"P{pose:02d}_silhouette_consensus.png",
            folder / f"P{pose:02d}_cloud_mask_consensus.png",
        )
        path = next((item for item in candidates if item.is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"P{pose:02d}: falta la silueta de consenso en {folder}. "
                "Ejecute nuevamente los pasos 05 y 06."
            )
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"No se pudo leer la silueta: {path}")
        expected_shape = tuple(
            int(value) for value in np.asarray(view["image_shape_hw"]).ravel()[:2]
        )
        if mask.shape != expected_shape:
            raise ValueError(
                f"P{pose:02d}: silueta {mask.shape} incompatible con " f"la nube {expected_shape}."
            )
        binary = mask > 0
        if not np.any(binary):
            raise RuntimeError(f"P{pose:02d}: la silueta está vacía: {path}")
        masks[pose] = binary
        sources[pose] = str(path)
    expected = list(range(len(registered_views)))
    if sorted(masks) != expected:
        raise RuntimeError("Las siluetas no cubren P00..P24: " f"{sorted(masks)}")
    return masks, sources


def global_silhouette_support(fused, registered_views, transforms, intrinsics, silhouettes, args):
    """Evalúa cada voxel contra todas las siluetas sin asumir una forma."""
    points = np.asarray(fused["points"], dtype=np.float64)
    inside_mask = np.zeros(len(points), dtype=np.uint64)
    outside_mask = np.zeros(len(points), dtype=np.uint64)
    tested_mask = np.zeros(len(points), dtype=np.uint64)
    per_pose = []
    tolerance_px = max(float(args.silhouette_boundary_tolerance_px), 0.0)

    for view in registered_views:
        pose = int(view["pose_index"])
        silhouette = np.asarray(silhouettes[pose], dtype=bool)
        height, width = silhouette.shape
        # Para píxeles exteriores, distancia euclídea hasta la silueta. Dentro
        # de ella el valor es cero. La tolerancia protege bordes subpíxel.
        outside_distance = cv2.distanceTransform((~silhouette).astype(np.uint8), cv2.DIST_L2, 5)
        camera_points = transform_points(points, np.linalg.inv(transforms[pose]))
        z = camera_points[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u_float = intrinsics["fx"] * camera_points[:, 0] / z + intrinsics["cx"]
            v_float = intrinsics["fy"] * camera_points[:, 1] / z + intrinsics["cy"]
        u = np.rint(u_float).astype(np.int64)
        v = np.rint(v_float).astype(np.int64)
        in_fov = (
            np.isfinite(z)
            & (z > 1e-6)
            & np.isfinite(u_float)
            & np.isfinite(v_float)
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        indexes = np.flatnonzero(in_fov)
        compatible = np.zeros(len(points), dtype=bool)
        if len(indexes):
            compatible[indexes] = outside_distance[v[indexes], u[indexes]] <= tolerance_px
        contradictory = in_fov & (~compatible)
        bit = pose_bit(pose)
        tested_mask[in_fov] |= bit
        inside_mask[compatible] |= bit
        outside_mask[contradictory] |= bit
        per_pose.append(
            {
                "pose_index": pose,
                "tested_voxels": int(np.count_nonzero(in_fov)),
                "compatible_voxels": int(np.count_nonzero(compatible)),
                "contradictory_voxels": int(np.count_nonzero(contradictory)),
            }
        )

    tested = bit_count_u64(tested_mask)
    inside = bit_count_u64(inside_mask)
    outside = bit_count_u64(outside_mask)
    ratio = np.divide(
        inside.astype(np.float32),
        np.maximum(tested.astype(np.float32), 1.0),
    )
    minimum_tested = min(
        max(1, int(args.silhouette_minimum_tested_poses)),
        len(registered_views),
    )
    strong = (
        (tested >= minimum_tested)
        & (ratio >= float(args.silhouette_strong_inside_ratio))
        & (outside <= int(args.silhouette_maximum_strong_contradictions))
    )
    weak = (
        (tested >= minimum_tested)
        & (ratio >= float(args.silhouette_weak_inside_ratio))
        & (outside <= int(args.silhouette_maximum_weak_contradictions))
    )
    return {
        "tested_pose_mask": tested_mask,
        "inside_pose_mask": inside_mask,
        "outside_pose_mask": outside_mask,
        "tested_poses": tested,
        "inside_poses": inside,
        "outside_poses": outside,
        "inside_ratio": ratio,
        "strong": strong,
        "weak": weak,
        "per_pose": per_pose,
    }


def _row_tokens(keys):
    """Codifica filas enteras contiguas para comparar claves de vóxel."""
    keys = np.ascontiguousarray(keys, dtype=np.int64)
    return keys.view(np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))).ravel()


def rows_in(query_keys, reference_keys):
    """Indica qué claves de la consulta aparecen entre las claves de referencia."""
    if len(query_keys) == 0 or len(reference_keys) == 0:
        return np.zeros(len(query_keys), bool)
    return np.isin(_row_tokens(query_keys), _row_tokens(reference_keys))


def recover_connected_weak_voxels(keys, strong, weak, maximum_hops):
    """Dilata el núcleo solo a través de voxels débiles ocupados y adyacentes."""
    keep = np.asarray(strong, bool).copy()
    weak_remaining = np.asarray(weak, bool) & (~keep)
    offsets = np.asarray(
        [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)
        ],
        dtype=np.int64,
    )
    recovered_each_hop = []
    for _ in range(max(0, int(maximum_hops))):
        candidate_indexes = np.flatnonzero(weak_remaining)
        if len(candidate_indexes) == 0 or not np.any(keep):
            recovered_each_hop.append(0)
            break
        candidate_keys = keys[candidate_indexes]
        kept_keys = keys[keep]
        adjacent = np.zeros(len(candidate_indexes), bool)
        for offset in offsets:
            adjacent |= rows_in(candidate_keys + offset, kept_keys)
            if np.all(adjacent):
                break
        newly = candidate_indexes[adjacent]
        keep[newly] = True
        weak_remaining[newly] = False
        recovered_each_hop.append(int(len(newly)))
        if len(newly) == 0:
            break
    return keep, recovered_each_hop


def filter_registered_views_by_voxels(registered_views, kept_keys, voxel_mm):
    """Filtra las vistas registradas usando los vóxeles conservados."""
    result = []
    per_pose = []
    for view in registered_views:
        keys = np.floor(view["points"] / float(voxel_mm)).astype(np.int64)
        keep = rows_in(keys, kept_keys)
        filtered = dict(view)
        for name in (
            "points",
            "colors",
            "normals",
            "confidence",
            "depth_spread_mm",
            "depth_support",
            "foreground_score",
            "background_residual_sigma",
            "depth_uncertainty_mm",
            "component_id",
            "stereo_score",
            "disparity_uncertainty_px",
            "observed_sessions",
            "pixel_uv",
            "reliability",
            "static_hypothesis_matches",
            "object_hypothesis_matches",
            "static_hypothesis_dominance",
        ):
            value = np.asarray(view.get(name, []))
            if len(value) == len(keep):
                filtered[name] = value[keep]
        # Se conserva la nube de cámara para rasterizar/reproyectar; sus filas
        # deben seguir exactamente el mismo filtro que los puntos registrados.
        camera_points = np.asarray(view.get("camera_points", []))
        if len(camera_points) == len(keep):
            filtered["camera_points"] = camera_points[keep]
        result.append(filtered)
        per_pose.append(
            {
                "pose_index": int(view["pose_index"]),
                "points_before": int(len(keep)),
                "points_after": int(np.count_nonzero(keep)),
                "removed": int(len(keep) - np.count_nonzero(keep)),
                "retained_ratio": float(np.mean(keep)) if len(keep) else 0.0,
            }
        )
    return result, per_pose


def sample_view(view, maximum, rng, power=1.5):
    """Selecciona una muestra de la vista según el presupuesto y los pesos de evidencia."""
    n = len(view["points"])
    if n <= maximum:
        idx = np.arange(n)
    else:
        probability = np.power(
            np.clip(view.get("reliability", np.ones(n)), 1e-6, 1.0),
            float(power),
        )
        probability /= np.sum(probability)
        idx = rng.choice(n, int(maximum), replace=False, p=probability)
    return {
        "points": view["points"][idx],
        "normals": (view["normals"][idx] if len(view["normals"]) == n else np.empty((0, 3))),
        "weights": view.get("reliability", np.ones(n))[idx],
    }


def _unit_vector(value, name):
    """Valida y normaliza un vector 3D asociado a la calibración."""
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        raise RuntimeError(f"{name} debe contener tres valores finitos.")
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise RuntimeError(f"{name} tiene longitud cero.")
    return vector / length


def axis_model_from_calibration(cal):
    """Extrae el eje unitario y el punto canónico de la línea de rotación."""
    model = cal.get("rotation_axis", {})
    axis = _unit_vector(model.get("unit_axis_xyz"), "rotation_axis.unit_axis_xyz")
    point = np.asarray(model.get("canonical_line_point_xyz_mm"), dtype=np.float64).reshape(-1)
    if point.size != 3 or not np.all(np.isfinite(point)):
        raise RuntimeError("rotation_axis.canonical_line_point_xyz_mm es inválido.")
    return axis, point


def perpendicular_basis(axis):
    """Base determinista del plano perpendicular al eje."""
    axis = _unit_vector(axis, "axis")
    candidates = np.eye(3, dtype=np.float64)
    reference = candidates[int(np.argmin(np.abs(candidates @ axis)))]
    first = reference - axis * float(np.dot(reference, axis))
    first = _unit_vector(first, "axis_perpendicular_basis_1")
    second = _unit_vector(np.cross(axis, first), "axis_perpendicular_basis_2")
    return first, second


def transforms_for_axis_line(pose_records, line_point):
    """Conserva cada rotación R y solo cambia el centro común: t=c-R*c."""
    line_point = np.asarray(line_point, dtype=np.float64)
    transforms = {}
    for pose, record in pose_records.items():
        original = np.asarray(record["transform_pose_to_P00_mechanical"], dtype=np.float64)
        T = original.copy()
        rotation = T[:3, :3]
        T[:3, 3] = line_point - rotation @ line_point
        transforms[int(pose)] = T
    return transforms


def transform_samples(samples, transforms):
    """Transforma las muestras de cada pose con su matriz correspondiente."""
    transformed = {}
    for pose, sample in samples.items():
        T = transforms[int(pose)]
        transformed[int(pose)] = {
            "points": transform_points(sample["points"], T),
            "normals": transform_normals(sample["normals"], T),
            "weights": np.asarray(
                sample.get("weights", np.ones(len(sample["points"]))),
                dtype=np.float64,
            ),
        }
    return transformed


def robust_axis_refinement_sample(view, axis, first, second, args, rng):
    """
    Recorta percentiles extremos solo para estimar el eje, de modo que restos
    dispersos de soporte/base no gobiernen el ajuste. La nube exportada nunca
    se recorta mediante esta función.
    """
    points = np.asarray(view["points"], dtype=np.float64)
    normals = np.asarray(view["normals"], dtype=np.float64)
    count = len(points)
    if count == 0:
        return {"points": points, "normals": np.empty((0, 3))}

    axial_percent = float(np.clip(args.axis_line_trim_axial_percent, 0.0, 35.0))
    transverse_percent = float(np.clip(args.axis_line_trim_transverse_percent, 0.0, 20.0))
    axial = points @ axis
    u = points @ first
    v = points @ second
    ah = np.percentile(axial, [axial_percent, 100.0 - axial_percent])
    uh = np.percentile(u, [transverse_percent, 100.0 - transverse_percent])
    vh = np.percentile(v, [transverse_percent, 100.0 - transverse_percent])
    keep = (
        (axial >= ah[0])
        & (axial <= ah[1])
        & (u >= uh[0])
        & (u <= uh[1])
        & (v >= vh[0])
        & (v <= vh[1])
    )

    minimum = min(count, max(300, int(round(0.30 * count))))
    if int(np.count_nonzero(keep)) < minimum:
        keep = (axial >= ah[0]) & (axial <= ah[1])
    if int(np.count_nonzero(keep)) < minimum:
        keep = np.ones(count, dtype=bool)

    indices = np.flatnonzero(keep)
    maximum = max(300, int(args.axis_line_optimization_points))
    if len(indices) > maximum:
        probability = np.power(
            np.clip(view.get("reliability", np.ones(count))[indices], 1e-6, 1.0),
            float(args.confidence_sampling_power),
        )
        probability /= np.sum(probability)
        indices = rng.choice(indices, maximum, replace=False, p=probability)
    return {
        "points": points[indices],
        "normals": (
            normals[indices] if len(normals) == count else np.empty((0, 3), dtype=np.float64)
        ),
        "weights": np.asarray(view.get("reliability", np.ones(count)), dtype=np.float64)[indices],
    }


def compact_alignment_score(
    offset,
    raw_samples,
    pose_records,
    base_point,
    first,
    second,
    edge_sources,
    args,
):
    """Métrica robusta y general para una traslación común de la línea."""
    offset = np.asarray(offset, dtype=np.float64)
    limit = float(args.axis_line_search_limit_mm)
    magnitude = float(np.linalg.norm(offset))
    if not np.all(np.isfinite(offset)) or magnitude > limit + 1e-9:
        excess = max(0.0, magnitude - limit)
        return {
            "score": float(1e9 + 1e7 * excess * excess),
            "overlap": 0.0,
            "clipped_rmse_mm": None,
            "edge_count": 0,
        }

    line = base_point + offset[0] * first + offset[1] * second
    placed = transform_samples(raw_samples, transforms_for_axis_line(pose_records, line))
    gate = float(args.overlap_distance_mm)
    cap = min(float(args.metric_cap_mm), max(gate + 1.0, 1.75 * gate))
    edge_scores = []
    overlaps = []
    for source in edge_sources:
        target = (int(source) + 1) % int(args.expected_views)
        sample_a = placed[int(source)]
        sample_b = placed[target]
        a = sample_a["points"]
        b = sample_b["points"]
        if len(a) == 0 or len(b) == 0:
            continue
        dab, iab = cKDTree(b).query(a, k=1, workers=query_threads())
        dba, iba = cKDTree(a).query(b, k=1, workers=query_threads())
        dab = np.asarray(dab, dtype=np.float64)
        dba = np.asarray(dba, dtype=np.float64)
        iab = np.asarray(iab, dtype=np.int64)
        iba = np.asarray(iba, dtype=np.int64)
        wa = np.clip(sample_a.get("weights", np.ones(len(a))), 1e-6, 1.0)
        wb = np.clip(sample_b.get("weights", np.ones(len(b))), 1e-6, 1.0)
        pair_ab = np.sqrt(wa * wb[iab])
        pair_ba = np.sqrt(wb * wa[iba])
        finite_ab = np.isfinite(dab) & np.isfinite(pair_ab)
        finite_ba = np.isfinite(dba) & np.isfinite(pair_ba)
        if not np.any(finite_ab) or not np.any(finite_ba):
            continue
        na = sample_a["normals"]
        nb = sample_b["normals"]
        mask_ab = reciprocal_surface_mask(a, na, b, nb, iab, dab, gate)
        mask_ba = reciprocal_surface_mask(b, nb, a, na, iba, dba, gate)
        if np.count_nonzero(mask_ab) < 30 or np.count_nonzero(mask_ba) < 30:
            edge_scores.append(cap * cap)
            overlaps.append(0.0)
            continue
        nb = nb / np.maximum(np.linalg.norm(nb, axis=1)[:, None], 1e-12)
        na = na / np.maximum(np.linalg.norm(na, axis=1)[:, None], 1e-12)
        rab = np.sum((a - b[iab]) * nb[iab], axis=1)
        rba = np.sum((b - a[iba]) * na[iba], axis=1)
        score_ab = np.average(
            np.where(mask_ab, np.minimum(rab * rab, cap * cap), cap * cap), weights=pair_ab
        )
        score_ba = np.average(
            np.where(mask_ba, np.minimum(rba * rba, cap * cap), cap * cap), weights=pair_ba
        )
        overlap_ab = np.average(mask_ab, weights=pair_ab)
        overlap_ba = np.average(mask_ba, weights=pair_ba)
        edge_scores.append(float(0.5 * (score_ab + score_ba)))
        overlaps.append(float(0.5 * (overlap_ab + overlap_ba)))

    if not edge_scores:
        return {
            "score": float("inf"),
            "overlap": 0.0,
            "clipped_rmse_mm": None,
            "edge_count": 0,
        }
    ordered = np.sort(np.asarray(edge_scores, dtype=np.float64))
    if len(ordered) >= 8:
        ordered = ordered[1:-1]
    score = float(np.mean(ordered))
    return {
        "score": score,
        "overlap": float(np.mean(overlaps)),
        "clipped_rmse_mm": float(math.sqrt(max(0.0, score))),
        "edge_count": int(len(edge_scores)),
    }


def _relative_improvement(before, after):
    """Calcula la reducción relativa de una métrica con comprobación de finitud."""
    before = float(before)
    after = float(after)
    if not np.isfinite(before) or before <= 1e-12 or not np.isfinite(after):
        return 0.0
    return float((before - after) / before)


def optimize_axis_line_split(
    edge_sources,
    raw_samples,
    pose_records,
    base_point,
    first,
    second,
    args,
):
    baseline = compact_alignment_score(
        np.zeros(2),
        raw_samples,
        pose_records,
        base_point,
        first,
        second,
        edge_sources,
        args,
    )
    limit = float(args.axis_line_search_limit_mm)

    def objective(value):
        value = np.asarray(value, dtype=np.float64)
        result = compact_alignment_score(
            value,
            raw_samples,
            pose_records,
            base_point,
            first,
            second,
            edge_sources,
            args,
        )
        prior = baseline["score"] * 0.03 * (float(np.linalg.norm(value)) / max(limit, 1e-9)) ** 2
        return float(result["score"] + prior)

    result = minimize(
        objective,
        np.zeros(2, dtype=np.float64),
        method="Powell",
        bounds=[(-limit, limit), (-limit, limit)],
        options={
            "maxfev": int(args.axis_line_maximum_evaluations),
            "xtol": 0.025,
            "ftol": 5e-4,
            "disp": False,
        },
    )
    offset = np.asarray(result.x, dtype=np.float64)
    norm = float(np.linalg.norm(offset))
    if norm > limit:
        offset *= limit / norm
    measured = compact_alignment_score(
        offset,
        raw_samples,
        pose_records,
        base_point,
        first,
        second,
        edge_sources,
        args,
    )
    return {
        "success": bool(bool(result.success) and bool(np.all(np.isfinite(offset)))),
        "message": str(result.message),
        "evaluations": int(result.nfev),
        "offset_basis_mm": offset.astype(float).tolist(),
        "offset_magnitude_mm": float(np.linalg.norm(offset)),
        "before": baseline,
        "after": measured,
        "improvement_ratio": _relative_improvement(baseline["score"], measured["score"]),
    }


def propose_axis_line_refinement(views, pose_records, cal, args, rng):
    """Propone una corrección transversal de la línea del eje para validarla."""
    axis, base_point = axis_model_from_calibration(cal)
    first, second = perpendicular_basis(axis)
    report = {
        "enabled": bool(args.refine_axis_line),
        "applied": False,
        "decision": "disabled_by_parameter",
        "axis_direction_fixed": True,
        "mechanical_angles_fixed": True,
        "runtime_icp_6dof_used": False,
        "search_limit_mm": float(args.axis_line_search_limit_mm),
        "original_line_point_xyz_mm": base_point.astype(float).tolist(),
        "basis_1_xyz": first.astype(float).tolist(),
        "basis_2_xyz": second.astype(float).tolist(),
    }
    original = transforms_for_axis_line(pose_records, base_point)
    if not args.refine_axis_line:
        return original, report

    robust_samples = {
        int(view["pose_index"]): robust_axis_refinement_sample(view, axis, first, second, args, rng)
        for view in views
    }
    # Reservar franjas axiales físicas antes de optimizar; no reutilizar sus
    # observaciones para elegir desplazamiento ni intensidad del candidato.
    validation_samples = {}
    for pid, sample in list(robust_samples.items()):
        band = np.floor((sample["points"] @ axis) / 4.0).astype(np.int64)
        hold = band % 3 == 0
        if np.count_nonzero(hold) < 60 or np.count_nonzero(~hold) < 120:
            report["decision"] = "fallback_insufficient_reserved_regions"
            return original, report
        validation_samples[pid] = {k: v[hold] for k, v in sample.items()}
        robust_samples[pid] = {k: v[~hold] for k, v in sample.items()}
    report["reserved_validation"] = "axial_4_mm_bands_modulo_3_never_used_for_candidate_selection"
    even_edges = list(range(0, int(args.expected_views) - 1, 2))
    odd_edges = list(range(1, int(args.expected_views) - 1, 2))
    closure_edges = [int(args.expected_views) - 1]
    even = optimize_axis_line_split(
        even_edges,
        robust_samples,
        pose_records,
        base_point,
        first,
        second,
        args,
    )
    odd = optimize_axis_line_split(
        odd_edges,
        robust_samples,
        pose_records,
        base_point,
        first,
        second,
        args,
    )
    even_offset = np.asarray(even["offset_basis_mm"], dtype=np.float64)
    odd_offset = np.asarray(odd["offset_basis_mm"], dtype=np.float64)
    disagreement = float(np.linalg.norm(even_offset - odd_offset))
    combined = 0.5 * (even_offset + odd_offset)
    combined_norm = float(np.linalg.norm(combined))
    limit = float(args.axis_line_search_limit_mm)
    if combined_norm > limit:
        combined *= limit / combined_norm

    uncertainties = np.concatenate(
        [np.asarray(view.get("depth_uncertainty_mm", ()), dtype=np.float64) for view in views]
    )
    uncertainties = uncertainties[np.isfinite(uncertainties) & (uncertainties > 0.0)]
    median_uncertainty = float(np.median(uncertainties)) if len(uncertainties) else 0.0
    base_uncertainty_cap = min(
        limit,
        float(args.axis_line_maximum_uncertainty_offset_mm),
        max(0.8, float(args.axis_line_uncertainty_offset_factor) * median_uncertainty),
    )

    baseline_even = compact_alignment_score(
        np.zeros(2),
        robust_samples,
        pose_records,
        base_point,
        first,
        second,
        even_edges,
        args,
    )
    baseline_odd = compact_alignment_score(
        np.zeros(2),
        robust_samples,
        pose_records,
        base_point,
        first,
        second,
        odd_edges,
        args,
    )
    baseline_closure = compact_alignment_score(
        np.zeros(2),
        robust_samples,
        pose_records,
        base_point,
        first,
        second,
        closure_edges,
        args,
    )
    full_even = compact_alignment_score(
        combined,
        robust_samples,
        pose_records,
        base_point,
        first,
        second,
        even_edges,
        args,
    )
    full_odd = compact_alignment_score(
        combined,
        robust_samples,
        pose_records,
        base_point,
        first,
        second,
        odd_edges,
        args,
    )
    full_closure = compact_alignment_score(
        combined,
        robust_samples,
        pose_records,
        base_point,
        first,
        second,
        closure_edges,
        args,
    )
    best_gain = 0.5 * (
        baseline_even["score"] + baseline_odd["score"] - full_even["score"] - full_odd["score"]
    )

    selected = None
    selected_validation = None
    gain_fraction = float(np.clip(args.axis_line_conservative_gain_fraction, 0.50, 1.0))
    # Gate barato: basta una mejora pequeña pero consistente para permitir la
    # validación cara con todos los puntos. La aceptación real sigue ocurriendo
    # después con RMSE/P90/cierre/relaciones consecutivas y regiones espaciales.
    proposal_minimum_improvement = max(
        0.0, float(args.axis_line_proposal_minimum_improvement_ratio)
    )
    reserved_minimum_improvement = max(
        0.0, float(args.axis_line_reserved_minimum_improvement_ratio)
    )
    minimum_closure_improvement = float(args.axis_line_minimum_closure_improvement_ratio)
    balance_weight = max(0.0, float(args.axis_line_cross_validation_balance_weight))
    candidate_evaluations = []
    for alpha in np.linspace(0.35, 1.0, 14):
        candidate = combined * float(alpha)
        candidate_even = compact_alignment_score(
            candidate,
            robust_samples,
            pose_records,
            base_point,
            first,
            second,
            even_edges,
            args,
        )
        candidate_odd = compact_alignment_score(
            candidate,
            robust_samples,
            pose_records,
            base_point,
            first,
            second,
            odd_edges,
            args,
        )
        candidate_closure = compact_alignment_score(
            candidate,
            robust_samples,
            pose_records,
            base_point,
            first,
            second,
            closure_edges,
            args,
        )
        actual_gain = 0.5 * (
            baseline_even["score"]
            + baseline_odd["score"]
            - candidate_even["score"]
            - candidate_odd["score"]
        )
        even_gain = _relative_improvement(baseline_even["score"], candidate_even["score"])
        odd_gain = _relative_improvement(baseline_odd["score"], candidate_odd["score"])
        closure_gain = _relative_improvement(baseline_closure["score"], candidate_closure["score"])
        even_ratio = float(candidate_even["score"] / max(abs(baseline_even["score"]), 1e-9))
        odd_ratio = float(candidate_odd["score"] / max(abs(baseline_odd["score"]), 1e-9))
        closure_ratio = float(
            candidate_closure["score"] / max(abs(baseline_closure["score"]), 1e-9)
        )
        # Una ganancia independiente permite ampliar el margen solo de forma
        # proporcional; una mejora débil nunca justifica mover el eje varios mm.
        independent_gain = min(even_gain, odd_gain)
        allowed_offset = min(
            limit,
            base_uncertainty_cap * (1.0 + 2.0 * max(0.0, independent_gain)),
        )
        selection_score = float(
            0.425 * even_ratio
            + 0.425 * odd_ratio
            + 0.150 * closure_ratio
            + balance_weight * abs(even_ratio - odd_ratio)
        )
        candidate_evaluations.append(
            {
                "alpha": float(alpha),
                "actual_gain": float(actual_gain),
                "even_after": candidate_even,
                "even_improvement_ratio": float(even_gain),
                "odd_after": candidate_odd,
                "odd_improvement_ratio": float(odd_gain),
                "closure_after": candidate_closure,
                "closure_improvement_ratio": float(closure_gain),
                "selection_score": selection_score,
                "allowed_offset_mm": float(allowed_offset),
                "offset_within_uncertainty_limit": bool(
                    np.linalg.norm(candidate) <= allowed_offset
                ),
            }
        )

    best_observed_gain = max(
        (item["actual_gain"] for item in candidate_evaluations),
        default=float("-inf"),
    )
    eligible_candidates = []
    for item in candidate_evaluations:
        eligible = bool(
            best_observed_gain > 0.0
            and item["actual_gain"] > 0.0
            and item["even_improvement_ratio"] >= proposal_minimum_improvement
            and item["odd_improvement_ratio"] >= proposal_minimum_improvement
            and item["closure_improvement_ratio"] >= minimum_closure_improvement
            and item["offset_within_uncertainty_limit"]
        )
        # La fracción de la mejor ganancia queda solo como diagnóstico/ranking.
        # Ya no bloquea un candidato seguro cuando el mejor desplazamiento
        # observado es físicamente inadmisible.
        item["meets_legacy_gain_fraction"] = bool(
            best_observed_gain > 0.0 and item["actual_gain"] >= gain_fraction * best_observed_gain
        )
        item["eligible"] = eligible
        if eligible:
            eligible_candidates.append(item)

    if eligible_candidates:
        chosen = min(
            eligible_candidates,
            key=lambda item: (item["selection_score"], -item["alpha"]),
        )
        selected = combined * float(chosen["alpha"])
        selected_validation = {
            "selection_policy": ("minimum_balanced_cross_validation_score_among_safe_candidates"),
            "selection_score": float(chosen["selection_score"]),
            "alpha": float(chosen["alpha"]),
            "even_before": baseline_even,
            "even_after": chosen["even_after"],
            "even_improvement_ratio": float(chosen["even_improvement_ratio"]),
            "odd_before": baseline_odd,
            "odd_after": chosen["odd_after"],
            "odd_improvement_ratio": float(chosen["odd_improvement_ratio"]),
            "closure_before": baseline_closure,
            "closure_after": chosen["closure_after"],
            "closure_improvement_ratio": float(chosen["closure_improvement_ratio"]),
        }

    stable = bool(
        even["success"]
        and odd["success"]
        and disagreement <= float(args.axis_line_maximum_split_disagreement_mm)
    )
    report.update(
        {
            "decision": "pending_full_resolution_validation",
            "split_even": even,
            "split_odd": odd,
            "split_disagreement_mm": disagreement,
            "maximum_split_disagreement_mm": float(args.axis_line_maximum_split_disagreement_mm),
            "combined_unscaled_offset_basis_mm": combined.astype(float).tolist(),
            "combined_unscaled_validation": {
                "even": full_even,
                "odd": full_odd,
                "closure": full_closure,
            },
            "best_unscaled_gain": float(best_gain),
            "best_observed_candidate_gain": float(best_observed_gain),
            "minimum_gain_fraction_legacy_diagnostic": float(gain_fraction),
            "proposal_minimum_improvement_ratio": float(proposal_minimum_improvement),
            "reserved_minimum_improvement_ratio": float(reserved_minimum_improvement),
            "median_depth_uncertainty_mm": median_uncertainty,
            "base_uncertainty_offset_cap_mm": float(base_uncertainty_cap),
            "uncertainty_offset_policy": "cap=base_cap*(1+2*minimum_even_odd_gain), bounded by search_limit",
            "minimum_closure_improvement_ratio": float(minimum_closure_improvement),
            "cross_validation_balance_weight": float(balance_weight),
            "alpha_candidates": candidate_evaluations,
            "stable_split_solution": stable,
        }
    )
    if not stable:
        report["decision"] = "fallback_split_solutions_disagree"
        return original, report
    if selected is None:
        if any(not item["offset_within_uncertainty_limit"] for item in candidate_evaluations):
            report["decision"] = "fallback_displacement_exceeds_uncertainty_evidence"
        else:
            report["decision"] = "fallback_independent_gains_insufficient"
        return original, report

    reserved_results = {}
    for name, edges in (
        ("adjacent", list(range(int(args.expected_views) - 1))),
        ("closure", closure_edges),
    ):
        before = compact_alignment_score(
            np.zeros(2), validation_samples, pose_records, base_point, first, second, edges, args
        )
        after = compact_alignment_score(
            selected, validation_samples, pose_records, base_point, first, second, edges, args
        )
        gain = _relative_improvement(before["score"], after["score"])
        passed = bool(
            np.isfinite(after["score"])
            and before["overlap"] > 0
            and gain >= reserved_minimum_improvement
            and after["overlap"] >= before["overlap"] - 0.01
            and after["edge_count"] == before["edge_count"]
        )
        reserved_results[name] = {
            "before": before,
            "after": after,
            "relative_gain": gain,
            "passed": passed,
        }
    report["reserved_region_validation"] = reserved_results
    if not all(item["passed"] for item in reserved_results.values()):
        report["decision"] = "fallback_reserved_regions_or_closure_failed"
        return original, report

    offset_xyz = selected[0] * first + selected[1] * second
    selected_line = base_point + offset_xyz
    report.update(
        {
            "decision": "proposed_pending_full_resolution_validation",
            "selected_offset_basis_mm": selected.astype(float).tolist(),
            "selected_offset_xyz_mm": offset_xyz.astype(float).tolist(),
            "selected_offset_magnitude_mm": float(np.linalg.norm(selected)),
            "selected_line_point_xyz_mm": selected_line.astype(float).tolist(),
            "compact_validation": selected_validation,
        }
    )
    return transforms_for_axis_line(pose_records, selected_line), report


def full_resolution_validation_snapshot(samples, args):
    """Métricas de validación usando la colección COMPLETA de cada pose.

    Guarda las 24 relaciones consecutivas por separado y el cierre P24->P00.
    De esta forma una mejora fuerte del cierre no puede ocultar deterioros
    locales entre vistas consecutivas.
    """
    edges = []
    expected = int(args.expected_views)
    for source in range(expected):
        target = (source + 1) % expected
        metric = symmetric_edge_metrics(samples[source], samples[target], args)
        if metric is not None:
            edges.append((source, target, metric))

    adjacent = [(s, t, m) for s, t, m in edges if s != expected - 1]
    closure_item = next(((s, t, m) for s, t, m in edges if s == expected - 1), None)

    def _edge_record(source, target, metric):
        pp = metric.get("point_plane")
        return {
            "source_pose": int(source),
            "target_pose": int(target),
            "overlap": float(metric["overlap"]),
            "point_plane_rmse_mm": None if pp is None else float(pp["rmse_mm"]),
            "point_plane_p90_mm": None if pp is None else float(pp["p90_abs_mm"]),
            "point_plane_count": 0 if pp is None else int(pp.get("count", 0)),
        }

    overlaps = [m["overlap"] for _, _, m in adjacent]
    rmses = [
        m["point_plane"]["rmse_mm"] for _, _, m in adjacent if m.get("point_plane") is not None
    ]
    p90s = [
        m["point_plane"]["p90_abs_mm"] for _, _, m in adjacent if m.get("point_plane") is not None
    ]
    closure = None if closure_item is None else closure_item[2]
    return {
        "edge_count": int(len(edges)),
        "adjacent_edge_count": int(len(adjacent)),
        "overlap_median": None if not overlaps else float(np.median(overlaps)),
        "overlap_mean": None if not overlaps else float(np.mean(overlaps)),
        "point_plane_rmse_median_mm": None if not rmses else float(np.median(rmses)),
        "point_plane_p90_median_mm": None if not p90s else float(np.median(p90s)),
        "adjacent_edges": [_edge_record(s, t, m) for s, t, m in adjacent],
        "closure": (
            None
            if closure is None
            else {
                "overlap": float(closure["overlap"]),
                "point_plane": closure["point_plane"],
            }
        ),
    }


def accept_full_resolution_refinement(before, after, args):
    """Acepta solo una mejora global real sin sacrificar vistas consecutivas."""
    reasons = []
    diagnostics = {
        "degraded_consecutive_edges": [],
        "full_resolution_uncertainty_cap_mm": None,
    }

    before_overlap = before.get("overlap_median")
    after_overlap = after.get("overlap_median")
    if before_overlap is None or after_overlap is None:
        reasons.append("solape_global_no_informativo")
    elif after_overlap < before_overlap - 0.005:
        reasons.append("disminuye_el_solape_mediano")

    before_rmse = before.get("point_plane_rmse_median_mm")
    after_rmse = after.get("point_plane_rmse_median_mm")
    minimum_rmse_gain = max(0.0, float(args.axis_line_full_minimum_rmse_improvement_ratio))
    if before_rmse is None or after_rmse is None:
        reasons.append("rmse_global_no_informativo")
        rmse_gain = 0.0
    else:
        rmse_gain = _relative_improvement(before_rmse, after_rmse)
        if rmse_gain < minimum_rmse_gain:
            reasons.append("rmse_global_no_mejora_suficiente")

    before_p90 = before.get("point_plane_p90_median_mm")
    after_p90 = after.get("point_plane_p90_median_mm")
    minimum_p90_gain = max(0.0, float(args.axis_line_full_minimum_p90_improvement_ratio))
    if before_p90 is None or after_p90 is None:
        reasons.append("p90_global_no_informativo")
        p90_gain = 0.0
    else:
        p90_gain = _relative_improvement(before_p90, after_p90)
        if p90_gain < minimum_p90_gain:
            reasons.append("p90_global_no_mejora_suficiente")

    before_closure = before.get("closure") or {}
    after_closure = after.get("closure") or {}
    if before_closure.get("overlap") is None or after_closure.get("overlap") is None:
        reasons.append("cierre_no_informativo")
        closure_rmse_gain = 0.0
        closure_p90_gain = 0.0
    else:
        if after_closure["overlap"] < before_closure["overlap"] - 0.005:
            reasons.append("empeora_el_solape_de_cierre")
        before_closure_pp = before_closure.get("point_plane")
        after_closure_pp = after_closure.get("point_plane")
        if before_closure_pp is None or after_closure_pp is None:
            reasons.append("metricas_de_cierre_no_informativas")
            closure_rmse_gain = 0.0
            closure_p90_gain = 0.0
        else:
            closure_rmse_gain = _relative_improvement(
                before_closure_pp["rmse_mm"], after_closure_pp["rmse_mm"]
            )
            closure_p90_gain = _relative_improvement(
                before_closure_pp["p90_abs_mm"], after_closure_pp["p90_abs_mm"]
            )
            if closure_rmse_gain < minimum_rmse_gain:
                reasons.append("rmse_de_cierre_no_mejora_suficiente")
            if closure_p90_gain < minimum_p90_gain:
                reasons.append("p90_de_cierre_no_mejora_suficiente")

    # Gate explícito de vistas consecutivas. Se toleran únicamente cambios
    # numéricos muy pequeños; por defecto no se admite ninguna arista degradada.
    b_edges = {(e["source_pose"], e["target_pose"]): e for e in before.get("adjacent_edges", [])}
    a_edges = {(e["source_pose"], e["target_pose"]): e for e in after.get("adjacent_edges", [])}
    rmse_ratio = max(0.0, float(args.axis_line_maximum_consecutive_rmse_degradation_ratio))
    p90_ratio = max(0.0, float(args.axis_line_maximum_consecutive_p90_degradation_ratio))
    abs_tol = max(0.0, float(args.axis_line_maximum_consecutive_absolute_degradation_mm))
    overlap_drop = max(0.0, float(args.axis_line_maximum_consecutive_overlap_drop))
    for key, b in b_edges.items():
        a = a_edges.get(key)
        if a is None:
            diagnostics["degraded_consecutive_edges"].append(
                {"edge": list(key), "reason": "missing_after"}
            )
            continue
        edge_reasons = []
        if a["overlap"] < b["overlap"] - overlap_drop:
            edge_reasons.append("overlap")
        br, ar = b.get("point_plane_rmse_mm"), a.get("point_plane_rmse_mm")
        bp, ap = b.get("point_plane_p90_mm"), a.get("point_plane_p90_mm")
        if br is None or ar is None or bp is None or ap is None:
            edge_reasons.append("point_plane_uninformative")
        else:
            if ar > br * (1.0 + rmse_ratio) + abs_tol:
                edge_reasons.append("rmse")
            if ap > bp * (1.0 + p90_ratio) + abs_tol:
                edge_reasons.append("p90")
        if edge_reasons:
            diagnostics["degraded_consecutive_edges"].append(
                {
                    "edge": list(key),
                    "reasons": edge_reasons,
                    "before": b,
                    "after": a,
                }
            )

    maximum_degraded = max(0, int(args.axis_line_maximum_degraded_consecutive_edges))
    if len(diagnostics["degraded_consecutive_edges"]) > maximum_degraded:
        reasons.append("degrada_vistas_consecutivas")

    diagnostics["improvement_ratios"] = {
        "adjacent_rmse": float(rmse_gain),
        "adjacent_p90": float(p90_gain),
        "closure_rmse": float(closure_rmse_gain),
        "closure_p90": float(closure_p90_gain),
    }
    return len(reasons) == 0, reasons, diagnostics


def _subset_metric_sample(sample, keep):
    keep = np.asarray(keep, dtype=bool)
    return {
        "points": np.asarray(sample["points"], dtype=np.float64)[keep],
        "normals": np.asarray(sample["normals"], dtype=np.float64)[keep],
        "weights": np.asarray(sample.get("weights", np.ones(len(keep))), dtype=np.float64)[keep],
    }


def axial_validation_bounds(samples, axis, bands):
    """Bandas cuantiles sobre el eje físico; no dependen de la forma del objeto."""
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    values = []
    for sample in samples.values():
        pts = np.asarray(sample.get("points", ()), dtype=np.float64)
        if pts.ndim == 2 and pts.shape[1] == 3 and len(pts):
            proj = pts @ axis
            proj = proj[np.isfinite(proj)]
            if len(proj):
                values.append(proj)
    if not values:
        return None
    values = np.concatenate(values)
    bands = max(2, int(bands))
    q = np.linspace(0.0, 1.0, bands + 1)
    bounds = np.quantile(values, q)
    # Evitar bandas degeneradas por cuantización o superficies planas extensas.
    if np.any(np.diff(bounds) <= 1e-6):
        lo, hi = float(np.min(values)), float(np.max(values))
        if hi <= lo + 1e-6:
            return None
        bounds = np.linspace(lo, hi, bands + 1)
    return np.asarray(bounds, dtype=np.float64)


def axial_region_validation_snapshot(samples, args, axis, bounds):
    """Evalúa cada relación consecutiva en regiones físicas independientes.

    La partición usa únicamente la coordenada a lo largo del eje de rotación
    calibrado. No presupone cilindro, plano, esfera ni ninguna primitiva.
    """
    if bounds is None:
        return {"available": False, "regions": [], "reason": "no_axial_bounds"}
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    expected = int(args.expected_views)
    minimum = max(50, int(args.axis_line_regional_minimum_points))
    regions = []
    for source in range(expected):
        target = (source + 1) % expected
        s = samples[source]
        q = samples[target]
        sp = np.asarray(s["points"], dtype=np.float64) @ axis
        qp = np.asarray(q["points"], dtype=np.float64) @ axis
        for band in range(len(bounds) - 1):
            lo, hi = float(bounds[band]), float(bounds[band + 1])
            if band == len(bounds) - 2:
                sm = (sp >= lo) & (sp <= hi)
                qm = (qp >= lo) & (qp <= hi)
            else:
                sm = (sp >= lo) & (sp < hi)
                qm = (qp >= lo) & (qp < hi)
            if np.count_nonzero(sm) < minimum or np.count_nonzero(qm) < minimum:
                continue
            metric = symmetric_edge_metrics(
                _subset_metric_sample(s, sm),
                _subset_metric_sample(q, qm),
                args,
            )
            if metric is None:
                continue
            pp = metric.get("point_plane")
            regions.append(
                {
                    "source_pose": int(source),
                    "target_pose": int(target),
                    "band": int(band),
                    "axial_min_mm": lo,
                    "axial_max_mm": hi,
                    "source_points": int(np.count_nonzero(sm)),
                    "target_points": int(np.count_nonzero(qm)),
                    "overlap": float(metric.get("overlap", 0.0)),
                    "point_plane_rmse_mm": None if pp is None else float(pp["rmse_mm"]),
                    "point_plane_p90_mm": None if pp is None else float(pp["p90_abs_mm"]),
                    "point_plane_count": 0 if pp is None else int(pp.get("count", 0)),
                }
            )
    return {
        "available": bool(regions),
        "bands": int(len(bounds) - 1),
        "bounds_mm": np.asarray(bounds, dtype=float).tolist(),
        "regions": regions,
    }


def compare_axial_region_validation(before, after, args):
    """Detecta regiones axiales degradadas al comparar el candidato y la base."""
    diagnostics = {"degraded_regions": []}
    if not before.get("available") or not after.get("available"):
        return False, ["validacion_regional_no_informativa"], diagnostics
    bmap = {(r["source_pose"], r["target_pose"], r["band"]): r for r in before.get("regions", [])}
    amap = {(r["source_pose"], r["target_pose"], r["band"]): r for r in after.get("regions", [])}
    rmse_ratio = max(0.0, float(args.axis_line_regional_maximum_rmse_degradation_ratio))
    p90_ratio = max(0.0, float(args.axis_line_regional_maximum_p90_degradation_ratio))
    abs_tol = max(0.0, float(args.axis_line_regional_maximum_absolute_degradation_mm))
    overlap_drop = max(0.0, float(args.axis_line_regional_maximum_overlap_drop))
    for key, b in bmap.items():
        a = amap.get(key)
        if a is None:
            diagnostics["degraded_regions"].append({"region": list(key), "reason": "missing_after"})
            continue
        reasons = []
        if a["overlap"] < b["overlap"] - overlap_drop:
            reasons.append("overlap")
        br, ar = b.get("point_plane_rmse_mm"), a.get("point_plane_rmse_mm")
        bp, ap = b.get("point_plane_p90_mm"), a.get("point_plane_p90_mm")
        if br is not None and ar is not None:
            if ar > br * (1.0 + rmse_ratio) + abs_tol:
                reasons.append("rmse")
        elif br is not None and ar is None:
            reasons.append("rmse_uninformative_after")
        if bp is not None and ap is not None:
            if ap > bp * (1.0 + p90_ratio) + abs_tol:
                reasons.append("p90")
        elif bp is not None and ap is None:
            reasons.append("p90_uninformative_after")
        if reasons:
            diagnostics["degraded_regions"].append(
                {
                    "region": list(key),
                    "reasons": reasons,
                    "before": b,
                    "after": a,
                }
            )
    maximum = max(0, int(args.axis_line_maximum_degraded_regions))
    passed = len(diagnostics["degraded_regions"]) <= maximum
    return passed, ([] if passed else ["degrada_regiones_axiales"]), diagnostics


def assess_registration_health(metrics, views, args):
    """Clasificación blanda del registro para trazabilidad experimental."""
    uncertainty = []
    input_conf = []
    for view in views:
        u = np.asarray(view.get("depth_uncertainty_mm", ()), dtype=np.float64)
        u = u[np.isfinite(u) & (u > 0.0)]
        if len(u):
            uncertainty.append(u)
        rf = view.get("reliability_filter", {})
        c = rf.get("confidence_median")
        if c is not None and np.isfinite(float(c)):
            input_conf.append(float(c))
    if uncertainty:
        median_uncertainty = float(np.median(np.concatenate(uncertainty)))
    else:
        median_uncertainty = float("nan")
    median_input_conf = float(np.median(input_conf)) if input_conf else float("nan")
    rmse = (metrics.get("primary_point_plane_rmse_mm") or {}).get("median")
    p90 = (metrics.get("primary_point_plane_p90_mm") or {}).get("median")
    closure = metrics.get("closure") or {}
    closure_overlap = closure.get("overlap")
    rmse_ratio = None
    p90_ratio = None
    reasons = []
    if np.isfinite(median_uncertainty) and median_uncertainty > 0:
        if rmse is not None:
            rmse_ratio = float(rmse) / median_uncertainty
            if rmse_ratio > float(args.registration_warning_rmse_uncertainty_ratio):
                reasons.append("rmse_alto_respecto_a_incertidumbre")
        if p90 is not None:
            p90_ratio = float(p90) / median_uncertainty
            if p90_ratio > float(args.registration_warning_p90_uncertainty_ratio):
                reasons.append("p90_alto_respecto_a_incertidumbre")
    if closure_overlap is not None and float(closure_overlap) < float(
        args.registration_warning_closure_overlap
    ):
        reasons.append("cierre_con_solape_bajo_para_registro_fuerte")
    if np.isfinite(median_input_conf) and median_input_conf < float(
        args.registration_warning_input_confidence_median
    ):
        reasons.append("confianza_mediana_de_entrada_baja")
    return {
        "status": "warning" if reasons else "accepted",
        "reasons": reasons,
        "median_depth_uncertainty_mm": (
            None if not np.isfinite(median_uncertainty) else median_uncertainty
        ),
        "median_input_confidence": (
            None if not np.isfinite(median_input_conf) else median_input_conf
        ),
        "primary_rmse_to_uncertainty_ratio": rmse_ratio,
        "primary_p90_to_uncertainty_ratio": p90_ratio,
        "closure_overlap": None if closure_overlap is None else float(closure_overlap),
        "policy": "soft_health_gate_only; hard rejection remains evaluate_registration",
    }


def reciprocal_surface_mask(src, src_n, tgt, tgt_n, indices, distances, gate):
    """Correspondencias observadas: recíprocas, cercanas y normales compatibles."""
    keep = np.isfinite(distances) & (distances <= gate)
    if len(src_n) != len(src) or len(tgt_n) != len(tgt):
        return np.zeros(len(src), bool)
    reverse = cKDTree(src).query(tgt, k=1, workers=query_threads())[1]
    keep &= reverse[indices] == np.arange(len(src))
    sn = np.asarray(src_n, float)
    tn = np.asarray(tgt_n, float)[indices]
    sl = np.linalg.norm(sn, axis=1)
    tl = np.linalg.norm(tn, axis=1)
    dots = np.sum(sn * tn, axis=1) / np.maximum(sl * tl, 1e-12)
    keep &= np.isfinite(dots) & (sl > 1e-9) & (tl > 1e-9) & (dots >= np.cos(np.deg2rad(25.0)))
    return keep


def directional_metrics(
    src,
    src_n,
    src_weights,
    tgt,
    tgt_n,
    tgt_weights,
    overlap_gate,
    cap,
    tree=None,
):
    """
    Métrica direccional GENERAL entre dos vistas ya colocadas por la
    calibración mecánica.

    Corrección V1.1:
    ----------------
    point-to-plane se calcula EXCLUSIVAMENTE sobre correspondencias cuyo
    vecino más cercano está dentro de ``overlap_gate``.

    En V1.0 se calculaba overlap con ese gate, pero luego point-to-plane usaba
    TODOS los vecinos finitos, incluso puntos sin superficie común visible.
    Eso penalizaba auto-oclusiones y cambios de cobertura como si fueran error
    de pose. En objetos arbitrarios esa métrica no es válida.

    Para una normal unitaria siempre se cumple:
        |point_to_plane| <= distancia_euclidea
    Por tanto, si una correspondencia supera el gate de solape, no debe entrar
    en la validación de alineación superficial.
    """
    if len(src) == 0 or len(tgt) == 0:
        return None

    if tree is None:
        tree = cKDTree(tgt)
    d, idx = tree.query(src, k=1, workers=query_threads())
    d = np.asarray(d, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)

    finite = np.isfinite(d)
    overlap_mask = finite & (d <= float(overlap_gate))

    src_weights = np.asarray(src_weights, dtype=np.float64)
    tgt_weights = np.asarray(tgt_weights, dtype=np.float64)
    if len(src_weights) != len(src):
        src_weights = np.ones(len(src), dtype=np.float64)
    if len(tgt_weights) != len(tgt):
        tgt_weights = np.ones(len(tgt), dtype=np.float64)
    pair_weights = np.sqrt(np.clip(src_weights, 1e-6, 1.0) * np.clip(tgt_weights[idx], 1e-6, 1.0))
    pair_weights = np.where(np.isfinite(pair_weights), pair_weights, 0.0)

    finite_count = int(np.count_nonzero(finite))
    overlap_count = int(np.count_nonzero(overlap_mask))
    finite_weight = float(np.sum(pair_weights[finite]))
    overlap = float(np.sum(pair_weights[overlap_mask]) / max(finite_weight, 1e-12))

    # Distancia euclídea global: solo diagnóstico de cobertura.
    clipped = np.minimum(d[finite], cap)
    euclidean = weighted_stats(clipped, pair_weights[finite])

    # Distancia euclídea únicamente dentro del solape: diagnóstico de registro.
    overlap_euclidean = weighted_stats(d[overlap_mask], pair_weights[overlap_mask])

    pp = None
    if len(tgt_n) == len(tgt) and len(src) == len(src_n) and overlap_count > 0:
        surface_mask = reciprocal_surface_mask(src, src_n, tgt, tgt_n, idx, d, overlap_gate)
        valid_idx = np.flatnonzero(surface_mask)
        q = tgt[idx[valid_idx]]
        n = tgt_n[idx[valid_idx]]
        p = src[valid_idx]
        pp_weights = pair_weights[valid_idx]

        nlen = np.linalg.norm(n, axis=1)
        good_n = np.isfinite(nlen) & (nlen > 1e-9)

        if np.any(good_n):
            q = q[good_n]
            p = p[good_n]
            n = n[good_n] / nlen[good_n, None]
            pp_weights = pp_weights[good_n]

            signed = np.sum((p - q) * n, axis=1)
            signed_valid = np.isfinite(signed) & np.isfinite(pp_weights)
            signed = signed[signed_valid]
            pp_weights = pp_weights[signed_valid]

            if len(signed):
                abs_pp = np.abs(signed)
                denominator = float(np.sum(pp_weights))
                pp = {
                    "rmse_mm": float(
                        np.sqrt(np.sum(pp_weights * signed * signed) / max(denominator, 1e-12))
                    ),
                    "p90_abs_mm": weighted_quantile(abs_pp, pp_weights, 0.90),
                    "median_abs_mm": weighted_quantile(abs_pp, pp_weights, 0.50),
                    "median_signed_mm": weighted_quantile(signed, pp_weights, 0.50),
                    "count": int(len(abs_pp)),
                    "effective_weight": denominator,
                    "maximum_possible_from_gate_mm": float(overlap_gate),
                }

    return {
        "overlap": overlap,
        "overlap_count": overlap_count,
        "finite_query_count": finite_count,
        "finite_query_weight": finite_weight,
        "euclidean_mm": euclidean,
        "overlap_euclidean_mm": overlap_euclidean,
        "surface_correspondence_policy": "reciprocal_distance_and_oriented_normals_25_deg",
        "point_plane": pp,
    }


def symmetric_edge_metrics(a, b, args):
    """Combina las métricas direccionales del par de vistas."""
    ab = directional_metrics(
        a["points"],
        a["normals"],
        a.get("weights", np.ones(len(a["points"]))),
        b["points"],
        b["normals"],
        b.get("weights", np.ones(len(b["points"]))),
        args.overlap_distance_mm,
        args.metric_cap_mm,
        tree=b.get("_metric_tree"),
    )
    ba = directional_metrics(
        b["points"],
        b["normals"],
        b.get("weights", np.ones(len(b["points"]))),
        a["points"],
        a["normals"],
        a.get("weights", np.ones(len(a["points"]))),
        args.overlap_distance_mm,
        args.metric_cap_mm,
        tree=a.get("_metric_tree"),
    )
    if ab is None or ba is None:
        return None

    overlap = 0.5 * (ab["overlap"] + ba["overlap"])

    pp_values = []
    for d in (ab, ba):
        if d["point_plane"] is not None:
            pp_values.append(d["point_plane"])

    pp = None
    if pp_values:
        pp_direction_weights = np.asarray(
            [r.get("effective_weight", r["count"]) for r in pp_values],
            dtype=np.float64,
        )
        pp_direction_weights /= max(float(np.sum(pp_direction_weights)), 1e-12)
        pp = {
            "rmse_mm": float(
                np.sum(pp_direction_weights * np.asarray([r["rmse_mm"] for r in pp_values]))
            ),
            "p90_abs_mm": float(
                np.sum(pp_direction_weights * np.asarray([r["p90_abs_mm"] for r in pp_values]))
            ),
            "median_abs_mm": float(
                np.sum(pp_direction_weights * np.asarray([r["median_abs_mm"] for r in pp_values]))
            ),
            "count": int(sum(r["count"] for r in pp_values)),
            "effective_weight": float(
                sum(r.get("effective_weight", r["count"]) for r in pp_values)
            ),
        }

    informative = overlap >= float(args.minimum_informative_overlap)
    if informative and pp is not None:
        surface_accepted = bool(
            pp["rmse_mm"] <= args.surface_maximum_point_plane_rmse_mm
            and pp["p90_abs_mm"] <= args.surface_maximum_point_plane_p90_mm
        )
    else:
        surface_accepted = False

    return {
        "overlap": overlap,
        "informative": informative,
        "surface_accepted": surface_accepted,
        "point_plane": pp,
        "forward": ab,
        "reverse": ba,
    }


@operacion("Evaluar solape y calidad del registro")
def evaluate_registration(samples, args, expected_views):
    """Calcula los gates con el mismo criterio antes o después de limpiar."""
    # Cache local a esta evaluación: nunca se reutiliza después de limpiar.
    samples = {
        i: {**samples[i], "_metric_tree": cKDTree(samples[i]["points"])}
        for i in range(expected_views)
    }
    edges = []
    for source in range(expected_views):
        target = (source + 1) % expected_views
        metric = symmetric_edge_metrics(samples[source], samples[target], args)
        if metric is not None:
            edges.append(
                {
                    "source_pose": source,
                    "target_pose": target,
                    "primary": True,
                    "closure": bool(source == expected_views - 1 and target == 0),
                    **metric,
                }
            )
    for source in range(expected_views):
        target = (source + 2) % expected_views
        metric = symmetric_edge_metrics(samples[source], samples[target], args)
        if metric is not None:
            edges.append(
                {
                    "source_pose": source,
                    "target_pose": target,
                    "primary": False,
                    "closure": False,
                    **metric,
                }
            )

    primary = [edge for edge in edges if edge["primary"]]
    informative = [edge for edge in primary if edge["informative"]]
    accepted = [edge for edge in informative if edge["surface_accepted"]]
    closure = next((edge for edge in primary if edge["closure"]), None)
    informative_ratio = len(informative) / max(float(expected_views), 1.0)
    accepted_ratio = len(accepted) / len(informative) if informative else 0.0
    pp_rmse = [
        edge["point_plane"]["rmse_mm"] for edge in informative if edge["point_plane"] is not None
    ]
    pp_p90 = [
        edge["point_plane"]["p90_abs_mm"] for edge in informative if edge["point_plane"] is not None
    ]

    warnings_out = []
    rejected = []
    if informative_ratio < args.minimum_informative_primary_ratio:
        warnings_out.append(
            f"Solo {len(informative)}/{expected_views} primarias tienen solape "
            "informativo; esto puede deberse a auto-oclusión del objeto."
        )
    if informative and accepted_ratio < args.minimum_accepted_informative_primary_ratio:
        rejected.append(
            f"Solo {len(accepted)}/{len(informative)} relaciones informativas "
            "son consistentes en superficie."
        )

    closure_status = "uninformative"
    if closure is not None:
        if closure["overlap"] >= args.closure_minimum_overlap:
            closure_status = "accepted"
            point_plane = closure["point_plane"]
            if point_plane is None:
                closure_status = "warning_no_normals"
                warnings_out.append("Cierre con solape pero sin normales para point-to-plane.")
            elif (
                point_plane["rmse_mm"] > args.closure_maximum_point_plane_rmse_mm
                or point_plane["p90_abs_mm"] > args.closure_maximum_point_plane_p90_mm
            ):
                closure_status = "rejected"
                rejected.append(
                    "El cierre P24->P00 tiene suficiente solape pero no es "
                    "consistente point-to-plane."
                )
        else:
            warnings_out.append(
                "Cierre P24->P00 con poco solape geométrico; se reporta "
                "uninformative y no se fuerza una correspondencia inexistente."
            )

    quality = "rejected" if rejected else ("warning" if warnings_out else "accepted")
    metrics = {
        "informative_primary_count": len(informative),
        "informative_primary_ratio": informative_ratio,
        "accepted_informative_primary_count": len(accepted),
        "accepted_informative_primary_ratio": accepted_ratio,
        "primary_point_plane_rmse_mm": stats(pp_rmse),
        "primary_point_plane_p90_mm": stats(pp_p90),
        "closure": (
            None
            if closure is None
            else {
                "overlap": closure["overlap"],
                "informative": closure["informative"],
                "surface_accepted": closure["surface_accepted"],
                "point_plane": closure["point_plane"],
                "status": closure_status,
            }
        ),
    }
    return {
        "edges": edges,
        "primary": primary,
        "informative": informative,
        "accepted": accepted,
        "closure": closure,
        "closure_status": closure_status,
        "reasons": warnings_out,
        "reject_reasons": rejected,
        "quality": quality,
        "metrics": metrics,
    }


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(points)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for p, c in zip(points, colors):
            fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} " f"{int(c[0])} {int(c[1])} {int(c[2])}\n")


def _projection_to_panel(points, dims, width, height, title):
    """
    Render 2D con OpenCV. Se usa como fallback cuando matplotlib no está
    instalado en el entorno.
    """
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(
            "No fue posible generar PNG: no están disponibles ni "
            f"matplotlib ni OpenCV. Detalle: {exc}"
        )

    panel = np.full((height, width, 3), 250, dtype=np.uint8)
    if len(points) == 0:
        cv2.putText(
            panel, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA
        )
        return panel

    x = points[:, dims[0]]
    y = points[:, dims[1]]
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) == 0:
        return panel

    # Percentiles robustos para evitar que unos pocos outliers destruyan escala.
    xmin, xmax = np.percentile(x, [0.5, 99.5])
    ymin, ymax = np.percentile(y, [0.5, 99.5])
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    margin = 32
    u = margin + (x - xmin) / (xmax - xmin) * (width - 2 * margin)
    v = height - margin - (y - ymin) / (ymax - ymin) * (height - 2 * margin)
    u = np.clip(u.astype(np.int32), 0, width - 1)
    v = np.clip(v.astype(np.int32), 0, height - 1)

    panel[v, u] = (70, 70, 70)
    cv2.putText(panel, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return panel


def save_preview(path: Path, registered_views: List[dict], quality: str, metrics: dict):
    """
    Genera siempre un PNG:
      - matplotlib si está disponible;
      - OpenCV como fallback.
    """
    all_p = []
    all_c = []
    for i, v in enumerate(registered_views):
        p = v["points"]
        if len(p) > 8000:
            idx = np.linspace(0, len(p) - 1, 8000).astype(int)
            p = p[idx]
        all_p.append(p)

        if plt is not None:
            color = np.tile(
                np.asarray(plt.cm.hsv(i / max(1, len(registered_views)))[:3]),
                (len(p), 1),
            )
            all_c.append(color)

    P = np.vstack(all_p)

    if plt is not None:
        C = np.vstack(all_c)

        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.scatter(P[:, 0], P[:, 1], c=C, s=0.25)
        ax1.set_title("X-Y")
        ax1.set_aspect("equal", adjustable="box")

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.scatter(P[:, 0], P[:, 2], c=C, s=0.25)
        ax2.set_title("X-Z")
        ax2.set_aspect("equal", adjustable="box")

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.scatter(P[:, 2], P[:, 1], c=C, s=0.25)
        ax3.set_title("Z-Y")
        ax3.set_aspect("equal", adjustable="box")

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis("off")
        lines = [
            f"Registro general calibrado V2.2 | {quality}",
            f"primarias informativas: {metrics['informative_primary_count']}/25",
            f"aceptadas/informativas: {metrics['accepted_informative_primary_ratio']:.1%}",
            f"PP RMSE mediano: {metrics['primary_point_plane_rmse_mm']['median']}",
            f"PP P90 mediano: {metrics['primary_point_plane_p90_mm']['median']}",
            f"cierre overlap: {metrics['closure'].get('overlap') if metrics['closure'] else None}",
            "PP calculado solo dentro del solape geométrico",
            "sin forma específica; sin ICP 6DoF; delta angular runtime=0°",
        ]
        ax4.text(0.03, 0.95, "\n\n".join(lines), va="top", fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return

    # Fallback OpenCV.
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(
            "No se pudo generar preview. Instala matplotlib o asegúrate de "
            f"tener OpenCV en el entorno. Detalle: {exc}"
        )

    # Submuestra global para render rápido.
    if len(P) > 120000:
        idx = np.linspace(0, len(P) - 1, 120000).astype(int)
        Pr = P[idx]
    else:
        Pr = P

    pw, ph = 720, 470
    xy = _projection_to_panel(Pr, (0, 1), pw, ph, "X-Y")
    xz = _projection_to_panel(Pr, (0, 2), pw, ph, "X-Z")
    zy = _projection_to_panel(Pr, (2, 1), pw, ph, "Z-Y")

    info = np.full((ph, pw, 3), 250, dtype=np.uint8)
    lines = [
        f"Registro general calibrado V2.2 | {quality}",
        f"Primarias informativas: {metrics['informative_primary_count']}/25",
        f"Aceptadas/informativas: {metrics['accepted_informative_primary_ratio']:.1%}",
        f"PP RMSE mediano: {metrics['primary_point_plane_rmse_mm']['median']}",
        f"PP P90 mediano: {metrics['primary_point_plane_p90_mm']['median']}",
        f"Cierre overlap: {metrics['closure'].get('overlap') if metrics['closure'] else None}",
        "PP: solo correspondencias dentro del gate de solape",
        "Sin forma especifica / ICP 6DoF / ajuste angular por objeto",
    ]
    y = 42
    for line in lines:
        cv2.putText(
            info, str(line), (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 1, cv2.LINE_AA
        )
        y += 48

    canvas = np.vstack(
        [
            np.hstack([xy, xz]),
            np.hstack([zy, info]),
        ]
    )
    ok = cv2.imwrite(str(path), canvas)
    if not ok:
        raise RuntimeError(f"OpenCV no pudo guardar preview en {path}")


def main():
    """Registra las vistas con la plataforma calibrada y valida el resultado completo."""
    args = parser().parse_args()
    root = Path(args.root).expanduser().resolve()
    obj = args.object.strip()

    print("[Paso 10 | 1/8] Cargando calibración y nubes de las 25 poses...")
    cal_path = resolve_calibration(root, args.calibration)
    cal = load_json(cal_path)
    validate_platform_calibration(cal, int(args.expected_views))

    cloud_dir, cloud_summary_path = resolve_cloud_summary(root, obj, args.cloud_source)
    cloud_summary = load_json(cloud_summary_path)
    if int(cloud_summary.get("schema_version", 0)) < 6:
        raise RuntimeError(
            "La salida 06 no publica el contrato de siluetas requerido por "
            "el paso 10 V3.0. Ejecute nuevamente el paso 06."
        )
    intrinsics_raw = cloud_summary.get("camera_intrinsics") or {}
    required_intrinsics = ("fx", "fy", "cx", "cy")
    if not all(name in intrinsics_raw for name in required_intrinsics):
        raise RuntimeError(
            "El paso 10 V3.0 requiere camera_intrinsics en el resumen 06. "
            "Ejecute primero el 06 entregado junto con este script."
        )
    intrinsics = {name: float(intrinsics_raw[name]) for name in required_intrinsics}
    views = load_views(
        cloud_dir,
        cloud_summary,
        int(args.expected_views),
        args,
        intrinsics,
    )

    pose_records = {int(r["pose_index"]): r for r in cal["poses"]}
    rng = np.random.default_rng(int(args.seed))

    # Misma muestra antes/después para que los gates sean comparables.
    raw_metric_samples = {
        int(view["pose_index"]): sample_view(
            view,
            int(args.sample_points_per_view),
            rng,
            float(args.confidence_sampling_power),
        )
        for view in views
    }
    # Esta colección no usa muestreo: decide si un refinamiento modifica la
    # salida de registro completa, con todos los puntos que ya superaron los
    # filtros de fiabilidad de cada pose.
    complete_metric_samples = {
        int(view["pose_index"]): {
            "points": np.asarray(view["points"], dtype=np.float64),
            "normals": np.asarray(view["normals"], dtype=np.float64),
            "weights": np.asarray(
                view.get("reliability", np.ones(len(view["points"]))), dtype=np.float64
            ),
        }
        for view in views
    }
    axis, original_line = axis_model_from_calibration(cal)
    original_transforms = transforms_for_axis_line(pose_records, original_line)
    original_metric_samples = transform_samples(raw_metric_samples, original_transforms)
    original_complete_metric_samples = transform_samples(
        complete_metric_samples, original_transforms
    )

    if args.refine_axis_line:
        print(
            "[Paso 10 | 2/8] Evaluando corrección transversal experimental "
            "(ángulos y dirección permanecen fijos)..."
        )
    else:
        print(
            "[Paso 10 | 2/8] Usando poses de calibración congeladas; "
            "no se optimiza el eje con la geometría del objeto."
        )
    proposed_transforms, axis_refinement = propose_axis_line_refinement(
        views, pose_records, cal, args, rng
    )
    selected_transforms = original_transforms
    if axis_refinement["decision"] == "proposed_pending_full_resolution_validation":
        print(
            "[Paso 10 | 3/8] Validando la propuesta con resolución completa "
            "y con el cierre P24->P00..."
        )
        proposed_metric_samples = transform_samples(raw_metric_samples, proposed_transforms)
        # La aceptación se calcula contra la salida completa sin refinamiento,
        # no contra las muestras compactas utilizadas para proponer el eje.
        proposed_complete_metric_samples = transform_samples(
            complete_metric_samples, proposed_transforms
        )
        before_validation = full_resolution_validation_snapshot(
            original_complete_metric_samples, args
        )
        after_validation = full_resolution_validation_snapshot(
            proposed_complete_metric_samples, args
        )
        full_accepted, full_reasons, full_diagnostics = accept_full_resolution_refinement(
            before_validation, after_validation, args
        )

        # V3.2: además de las métricas globales, comprobar que la mejora no
        # provenga de sacrificar una zona física del objeto. Las bandas son
        # cuantiles a lo largo del eje calibrado, por lo que son generales.
        regional_bounds = axial_validation_bounds(
            original_complete_metric_samples, axis, int(args.axis_line_regional_bands)
        )
        regional_before = axial_region_validation_snapshot(
            original_complete_metric_samples, args, axis, regional_bounds
        )
        regional_after = axial_region_validation_snapshot(
            proposed_complete_metric_samples, args, axis, regional_bounds
        )
        regional_ok, regional_reasons, regional_diagnostics = compare_axial_region_validation(
            regional_before, regional_after, args
        )
        if not regional_ok:
            full_accepted = False
            full_reasons = list(full_reasons) + list(regional_reasons)
        full_diagnostics["axial_region_validation"] = {
            "before": regional_before,
            "after": regional_after,
            "accepted": bool(regional_ok),
            "diagnostics": regional_diagnostics,
        }

        # Segunda guarda del desplazamiento basada en la mejora demostrada a
        # resolución completa, no solamente en la muestra usada para proponer.
        gains = full_diagnostics.get("improvement_ratios", {})
        demonstrated_gain = min(
            max(0.0, float(gains.get("adjacent_rmse", 0.0))),
            max(0.0, float(gains.get("adjacent_p90", 0.0))),
            max(0.0, float(gains.get("closure_rmse", 0.0))),
            max(0.0, float(gains.get("closure_p90", 0.0))),
        )
        base_cap = float(axis_refinement.get("base_uncertainty_offset_cap_mm", 0.0))
        full_cap = min(
            float(args.axis_line_search_limit_mm),
            float(args.axis_line_maximum_uncertainty_offset_mm),
            base_cap * (1.0 + 2.0 * demonstrated_gain),
        )
        full_diagnostics["full_resolution_uncertainty_cap_mm"] = float(full_cap)
        full_diagnostics["demonstrated_gain_for_offset_cap"] = float(demonstrated_gain)
        selected_offset = float(axis_refinement.get("selected_offset_magnitude_mm", 0.0))
        if selected_offset > full_cap + 1e-9:
            full_accepted = False
            full_reasons = list(full_reasons) + [
                "desplazamiento_excede_incertidumbre_con_mejora_completa"
            ]

        axis_refinement["full_resolution_validation"] = {
            "before": before_validation,
            "after": after_validation,
            "accepted": bool(full_accepted),
            "reject_reasons": full_reasons,
            "diagnostics": full_diagnostics,
        }
        axis_refinement["ab_policy"] = (
            "frozen_calibration_authoritative_candidate_only_applied_with_explicit_authorization"
        )
        axis_refinement["application_authorized"] = bool(args.apply_axis_line_refinement)
        axis_refinement["ab_validation_passed"] = bool(full_accepted)
        if full_accepted and args.apply_axis_line_refinement:
            axis_refinement["applied"] = True
            axis_refinement["decision"] = "accepted_and_applied_explicitly"
            selected_transforms = proposed_transforms
            samples = proposed_metric_samples
            print(
                "[Paso 10 | 3/8] Candidato A/B validado y APLICADO por autorización explícita: "
                f"{axis_refinement['selected_offset_magnitude_mm']:.3f} mm."
            )
        elif full_accepted:
            # Política V3.3: para una campaña reproducible la calibración congelada
            # es la referencia. El refinamiento se conserva como diagnóstico A/B,
            # pero no altera las poses de producción salvo petición explícita.
            axis_refinement["applied"] = False
            axis_refinement["decision"] = "validated_not_applied_frozen_calibration_policy"
            selected_transforms = original_transforms
            samples = original_metric_samples
            print(
                "[Paso 10 | 3/8] Candidato A/B validado, pero NO se aplica: "
                "se conserva la calibración congelada."
            )
        else:
            axis_refinement["applied"] = False
            axis_refinement["decision"] = "fallback_full_resolution_validation_failed"
            samples = original_metric_samples
            print(
                "[Paso 10 | 3/8] Corrección descartada; se conserva la calibración original."
            )
    else:
        samples = original_metric_samples
        print(
            "[Paso 10 | 3/8] No hay evidencia independiente suficiente; "
            "se conserva la calibración original."
        )

    print("[Paso 10 | 4/8] Registrando nubes con las poses mecánicas congeladas...")
    registered = []
    for view in views:
        pose = view["pose_index"]
        record = pose_records[pose]
        T = selected_transforms[pose]
        # Verificación crítica: runtime correction DEBE ser cero.
        if abs(float(record.get("runtime_correction_deg", 0.0))) > 1e-12:
            raise RuntimeError("La calibración runtime contiene corrección por objeto.")

        p = transform_points(view["points"], T)
        n = transform_normals(view["normals"], T)
        registered.append(
            {
                **view,
                "camera_points": np.asarray(view["points"], dtype=np.float64).copy(),
                "points": p,
                "normals": n,
                "T": T,
            }
        )

    preclean_evaluation = evaluate_registration(samples, args, int(args.expected_views))

    print("[Paso 10 | 5/8] Tallando la unión contra las 25 siluetas globales...")
    fused = fuse_voxels_with_pose(registered, float(args.union_voxel_mm))
    silhouettes, silhouette_sources = load_consensus_silhouettes(root, cloud_summary, registered)
    visual = global_silhouette_support(
        fused,
        registered,
        selected_transforms,
        intrinsics,
        silhouettes,
        args,
    )

    print("[Paso 10 | 6/8] Comprobando profundidad como evidencia secundaria...")
    reprojection = multiview_reprojection_support(
        fused,
        registered,
        selected_transforms,
        intrinsics,
        args,
    )
    support = reprojection["support_poses"]
    agreements = reprojection["reprojected_agreement_poses"]
    contradictions = reprojection["contradiction_poses"]
    agreement_ratio = reprojection["agreement_ratio"]
    angular_span = reprojection["angular_span_poses"]
    depth_supported = (
        (agreements >= int(args.reprojection_minimum_independent_agreements))
        | (fused["direct_support_poses"] >= 2)
        | ((fused["observed_sessions"] >= 2) & (fused["stereo_score"] >= 0.70))
    )
    depth_not_contradicted = contradictions <= int(args.reprojection_maximum_contradictions)
    strong = visual["strong"] & depth_supported & depth_not_contradicted
    weak = (
        visual["weak"]
        & depth_not_contradicted
        & (
            (agreements >= 1)
            | (fused["direct_support_poses"] >= 2)
            | (fused["observed_sessions"] >= 2)
        )
        & (fused["foreground_score"] >= float(args.weak_boundary_minimum_foreground_score))
        & (fused["stereo_score"] >= 0.34)
    )
    keep, recovered_each_hop = recover_connected_weak_voxels(
        fused["voxel_keys"],
        strong,
        weak,
        int(args.weak_boundary_recovery_hops),
    )
    if int(np.count_nonzero(keep)) < int(args.minimum_clean_union_voxels):
        raise RuntimeError(
            "La validación multivista dejó menos voxels que el mínimo seguro "
            f"({np.count_nonzero(keep)} < {args.minimum_clean_union_voxels}). "
            "No se recuperó la unión cruda de manera silenciosa."
        )

    kept_keys = fused["voxel_keys"][keep]
    cleaned_registered, cleaning_per_pose = filter_registered_views_by_voxels(
        registered,
        kept_keys,
        float(args.union_voxel_mm),
    )
    cleaned_metric_samples = {
        int(view["pose_index"]): sample_view(
            view,
            int(args.sample_points_per_view),
            rng,
            float(args.confidence_sampling_power),
        )
        for view in cleaned_registered
    }
    postclean_evaluation = evaluate_registration(
        cleaned_metric_samples, args, int(args.expected_views)
    )
    edges = postclean_evaluation["edges"]
    informative = postclean_evaluation["informative"]
    accepted = postclean_evaluation["accepted"]
    closure_status = postclean_evaluation["closure_status"]
    reasons = list(postclean_evaluation["reasons"])
    rejected = list(postclean_evaluation["reject_reasons"])
    quality = postclean_evaluation["quality"]
    metrics = postclean_evaluation["metrics"]

    registration_health = assess_registration_health(metrics, views, args)
    if quality != "rejected" and registration_health["status"] == "warning":
        quality = "warning"
        reasons.extend([f"health:{item}" for item in registration_health["reasons"]])

    output = root / "reconstruccion" / "multisesion" / args.output_name
    output.mkdir(parents=True, exist_ok=True)

    # Contrato temporal entre 10 y 11. No modifica la calibración global.
    # V3.3 conserva además dos modelos diagnósticos A/B muy pequeños: el
    # congelado y, cuando existe, el candidato refinado. El paso 11 consume
    # únicamente modelo_poses_runtime_validado.json.
    def _pose_model_payload(transforms, line_point, refinement_applied, role):
        return {
            "schema_version": 1,
            "method": "fixed_axis_direction_fixed_angles_common_line_position",
            "object": obj,
            "role": role,
            "source_calibration_path": str(cal_path),
            "axis_line_refinement_applied": bool(refinement_applied),
            "axis_direction_xyz": axis.astype(float).tolist(),
            "line_point_xyz_mm": np.asarray(line_point, dtype=np.float64).astype(float).tolist(),
            "mechanical_angles_fixed": True,
            "runtime_angle_corrections_used": False,
            "runtime_icp_6dof_used": False,
            "poses": [
                {
                    "pose_index": int(pose),
                    "physical_angle_deg": float(pose_records[pose].get("physical_angle_deg", 0.0)),
                    "transform_pose_to_P00_runtime": transforms[pose].astype(float).tolist(),
                }
                for pose in sorted(transforms)
            ],
        }

    frozen_pose_model_path = output / "modelo_poses_runtime_congelado.json"
    frozen_pose_model_path.write_text(
        json.dumps(
            _pose_model_payload(original_transforms, original_line, False, "frozen_calibration_A"),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    candidate_pose_model_path = None
    if (
        axis_refinement.get("full_resolution_validation") is not None
        and axis_refinement.get("selected_line_point_xyz_mm") is not None
    ):
        candidate_pose_model_path = output / "modelo_poses_runtime_candidato_refinado.json"
        candidate_pose_model_path.write_text(
            json.dumps(
                _pose_model_payload(
                    proposed_transforms,
                    axis_refinement["selected_line_point_xyz_mm"],
                    True,
                    "refined_candidate_B",
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    runtime_pose_model_path = output / "modelo_poses_runtime_validado.json"
    runtime_pose_model = {
        "schema_version": 1,
        "method": "fixed_axis_direction_fixed_angles_common_line_position",
        "object": obj,
        "source_calibration_path": str(cal_path),
        "axis_line_refinement_applied": bool(axis_refinement["applied"]),
        "axis_direction_xyz": axis.astype(float).tolist(),
        "line_point_xyz_mm": (
            axis_refinement.get("selected_line_point_xyz_mm")
            if axis_refinement["applied"]
            else original_line.astype(float).tolist()
        ),
        "mechanical_angles_fixed": True,
        "runtime_angle_corrections_used": False,
        "runtime_icp_6dof_used": False,
        "poses": [
            {
                "pose_index": int(pose),
                "physical_angle_deg": float(pose_records[pose].get("physical_angle_deg", 0.0)),
                "transform_pose_to_P00_runtime": (selected_transforms[pose].astype(float).tolist()),
            }
            for pose in sorted(selected_transforms)
        ],
    }
    runtime_pose_model_path.write_text(
        json.dumps(runtime_pose_model, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[Paso 10 | 7/8] Guardando unión limpia, procedencia y métricas...")

    all_points = fused["points"][keep]
    all_colors = fused["colors"][keep]
    all_normals = fused["normals"][keep]
    support_keep = reprojection["support_poses"][keep]
    direct_support_keep = fused["direct_support_poses"][keep]
    agreement_keep = reprojection["reprojected_agreement_poses"][keep]
    agreement_ratio_keep = reprojection["agreement_ratio"][keep]
    angular_span_keep = reprojection["angular_span_poses"][keep]
    contradiction_keep = reprojection["contradiction_poses"][keep]
    occluded_keep = reprojection["occluded_poses"][keep]
    silhouette_tested_keep = visual["tested_poses"][keep]
    silhouette_inside_keep = visual["inside_poses"][keep]
    silhouette_outside_keep = visual["outside_poses"][keep]
    silhouette_ratio_keep = visual["inside_ratio"][keep]
    confidence_keep = np.clip(
        0.35 * silhouette_ratio_keep
        + 0.20 * fused["foreground_score"][keep]
        + 0.15 * agreement_ratio_keep
        + 0.15 * fused["stereo_score"][keep]
        + 0.15 * np.exp(-0.5 * np.square(fused["spatial_spread_mm"][keep] / 2.0)),
        0.0,
        1.0,
    )
    save_ply(
        output / "union_general_calibrada.ply",
        all_points,
        all_colors,
    )
    np.savez_compressed(
        output / "union_general_calibrada.npz",
        points=all_points.astype(np.float32),
        colors=all_colors.astype(np.uint8),
        normals=all_normals.astype(np.float32),
        voxel_keys=fused["voxel_keys"][keep].astype(np.int64),
        direct_pose_mask=fused["direct_pose_mask"][keep].astype(np.uint64),
        support_pose_mask=reprojection["support_pose_mask"][keep].astype(np.uint64),
        agreement_pose_mask=reprojection["agreement_pose_mask"][keep].astype(np.uint64),
        contradiction_pose_mask=reprojection["contradiction_pose_mask"][keep].astype(np.uint64),
        silhouette_tested_pose_mask=visual["tested_pose_mask"][keep].astype(np.uint64),
        silhouette_inside_pose_mask=visual["inside_pose_mask"][keep].astype(np.uint64),
        silhouette_outside_pose_mask=visual["outside_pose_mask"][keep].astype(np.uint64),
        direct_support_poses=direct_support_keep.astype(np.uint16),
        support_poses=support_keep.astype(np.uint16),
        reprojected_agreement_poses=agreement_keep.astype(np.uint16),
        agreement_ratio=agreement_ratio_keep.astype(np.float32),
        angular_span_poses=angular_span_keep.astype(np.uint8),
        contradiction_poses=contradiction_keep.astype(np.uint16),
        occluded_poses=occluded_keep.astype(np.uint16),
        silhouette_tested_poses=silhouette_tested_keep.astype(np.uint16),
        silhouette_inside_poses=silhouette_inside_keep.astype(np.uint16),
        silhouette_outside_poses=silhouette_outside_keep.astype(np.uint16),
        silhouette_inside_ratio=silhouette_ratio_keep.astype(np.float32),
        foreground_score=fused["foreground_score"][keep].astype(np.float32),
        stereo_score=fused["stereo_score"][keep].astype(np.float32),
        observed_sessions=fused["observed_sessions"][keep].astype(np.float32),
        depth_uncertainty_mm=fused["depth_uncertainty_mm"][keep].astype(np.float32),
        spatial_spread_mm=fused["spatial_spread_mm"][keep].astype(np.float32),
        confidence=confidence_keep.astype(np.float32),
        union_voxel_mm=np.asarray([args.union_voxel_mm], dtype=np.float32),
    )

    registered_clean_outputs = []
    for view in cleaned_registered:
        pose = int(view["pose_index"])
        path = output / f"P{pose:02d}_registered_clean.npz"
        np.savez_compressed(
            path,
            points=np.asarray(view["points"], dtype=np.float32),
            colors=np.asarray(view["colors"], dtype=np.uint8),
            normals=np.asarray(view["normals"], dtype=np.float32),
            reliability=np.asarray(view["reliability"], dtype=np.float32),
            foreground_score=np.asarray(view["foreground_score"], dtype=np.float32),
            stereo_score=np.asarray(view["stereo_score"], dtype=np.float32),
            disparity_uncertainty_px=np.asarray(view["disparity_uncertainty_px"], dtype=np.float32),
            observed_sessions=np.asarray(view["observed_sessions"], dtype=np.float32),
            depth_uncertainty_mm=np.asarray(view["depth_uncertainty_mm"], dtype=np.float32),
            pixel_uv=np.asarray(view["pixel_uv"], dtype=np.float32),
            image_shape_hw=np.asarray(view["image_shape_hw"], dtype=np.int32),
            pose_index=np.asarray([pose], dtype=np.int16),
            transform_pose_to_P00=np.asarray(view["T"], dtype=np.float64),
        )
        registered_clean_outputs.append(
            {
                "pose_index": pose,
                "path": str(path),
            }
        )

    # CSV de aristas.
    csv_path = output / "calidad_registro_general_v1_2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "source_pose",
            "target_pose",
            "primary",
            "closure",
            "overlap",
            "informative",
            "surface_accepted",
            "point_plane_rmse_mm",
            "point_plane_p90_abs_mm",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for e in edges:
            pp = e["point_plane"]
            writer.writerow(
                {
                    "source_pose": e["source_pose"],
                    "target_pose": e["target_pose"],
                    "primary": e["primary"],
                    "closure": e["closure"],
                    "overlap": e["overlap"],
                    "informative": e["informative"],
                    "surface_accepted": e["surface_accepted"],
                    "point_plane_rmse_mm": None if pp is None else pp["rmse_mm"],
                    "point_plane_p90_abs_mm": None if pp is None else pp["p90_abs_mm"],
                }
            )

    summary = {
        "schema_version": 10,
        "method": (
            "frozen_platform_calibration_pose_provenance_voxel_fusion_"
            "global_multiview_silhouette_carving_depth_visibility_"
            "global_free_space_consistency_and_connected_weak_boundary_recovery"
        ),
        "object": obj,
        "quality": quality,
        "reasons": reasons,
        "reject_reasons": rejected,
        "calibration_path": str(cal_path),
        "calibration_status": cal.get("status"),
        "calibration_shape_assumptions_used": False,
        "runtime_angle_corrections_used": False,
        "runtime_icp_6dof_used": False,
        "runtime_axis_line_refinement_used": bool(axis_refinement["applied"]),
        "axis_line_ab_evaluated": bool(args.refine_axis_line),
        "axis_line_application_authorized": bool(args.apply_axis_line_refinement),
        "axis_line_production_policy": "frozen_calibration_by_default",
        "runtime_pose_model_frozen_path": str(frozen_pose_model_path),
        "runtime_pose_model_candidate_path": (
            None if candidate_pose_model_path is None else str(candidate_pose_model_path)
        ),
        "per_point_reliability_used": True,
        "input_reliability_filters": [
            {
                "pose_index": int(view["pose_index"]),
                **view["reliability_filter"],
            }
            for view in views
        ],
        "axis_line_refinement": axis_refinement,
        "runtime_pose_model_path": str(runtime_pose_model_path),
        "point_plane_correspondence_policy": "postclean_reciprocal_neighbors_distance_gate_and_oriented_normals_25deg",
        "views_processed": len(cleaned_registered),
        "preclean_metrics": preclean_evaluation["metrics"],
        "metrics": metrics,
        "registration_health": registration_health,
        "edges": edges,
        "static_hypothesis_filter": {
            "enabled": False,
            "reason": (
                "La simetría rotacional puede hacer que superficie real parezca "
                "estática en cámara; por ello esta hipótesis no elimina puntos."
            ),
        },
        "visual_hull_carving": {
            "policy": (
                "outside_silhouette_is_contradiction_inside_is_compatible_"
                "outside_fov_is_untested"
            ),
            "silhouette_sources": {
                f"P{pose:02d}": path for pose, path in sorted(silhouette_sources.items())
            },
            "boundary_tolerance_px": float(args.silhouette_boundary_tolerance_px),
            "minimum_tested_poses": int(args.silhouette_minimum_tested_poses),
            "strong_inside_ratio": float(args.silhouette_strong_inside_ratio),
            "weak_inside_ratio": float(args.silhouette_weak_inside_ratio),
            "maximum_strong_contradictions": int(args.silhouette_maximum_strong_contradictions),
            "maximum_weak_contradictions": int(args.silhouette_maximum_weak_contradictions),
            "tested_poses_kept": stats(silhouette_tested_keep),
            "inside_poses_kept": stats(silhouette_inside_keep),
            "outside_poses_kept": stats(silhouette_outside_keep),
            "inside_ratio_kept": stats(silhouette_ratio_keep),
            "per_pose": visual["per_pose"],
        },
        "multiview_cleaning": {
            "candidate_voxels": int(len(fused["points"])),
            "strong_voxels": int(np.count_nonzero(strong)),
            "weak_candidate_voxels": int(np.count_nonzero(weak)),
            "weak_recovered_each_hop": recovered_each_hop,
            "kept_voxels": int(np.count_nonzero(keep)),
            "removed_voxels": int(len(keep) - np.count_nonzero(keep)),
            "retained_ratio": float(np.mean(keep)),
            "input_points": int(fused["input_points"]),
            "pose_voxel_contributions": int(fused["pose_voxel_contributions"]),
            "direct_support_poses": stats(direct_support_keep),
            "support_poses_after_reprojection": stats(support_keep),
            "reprojected_agreement_poses": stats(agreement_keep),
            "agreement_ratio": stats(agreement_ratio_keep),
            "angular_span_poses": stats(angular_span_keep),
            "contradiction_poses": stats(contradiction_keep),
            "occluded_poses": stats(occluded_keep),
            "silhouette_tested_poses": stats(silhouette_tested_keep),
            "silhouette_inside_poses": stats(silhouette_inside_keep),
            "silhouette_outside_poses": stats(silhouette_outside_keep),
            "silhouette_inside_ratio": stats(silhouette_ratio_keep),
            "foreground_score": stats(fused["foreground_score"][keep]),
            "stereo_score": stats(fused["stereo_score"][keep]),
            "observed_sessions": stats(fused["observed_sessions"][keep]),
            "spatial_spread_mm": stats(fused["spatial_spread_mm"][keep]),
            "minimum_independent_agreements": int(args.reprojection_minimum_independent_agreements),
            "minimum_agreement_ratio": float(args.reprojection_minimum_agreement_ratio),
            "minimum_angular_span_poses": int(args.reprojection_minimum_angular_span_poses),
            "per_target_reprojection": reprojection["per_target"],
            "per_pose_after_cleaning": cleaning_per_pose,
        },
        "outputs": {
            "union_ply": str(output / "union_general_calibrada.ply"),
            "union_npz": str(output / "union_general_calibrada.npz"),
            "registered_clean_views": registered_clean_outputs,
        },
        "important_note": (
            "Los ángulos, el sentido y la dirección del eje provienen de la "
            "calibración congelada. Los voxels conservan qué poses aportaron; "
            "cada voxel debe ser compatible con las siluetas de múltiples poses. "
            "Una ausencia de profundidad se considera desconocida, no fondo. La "
            "profundidad solo añade acuerdo, oclusión o contradicción cuando hay "
            "una medición válida. "
            "Solo se recuperan fronteras débiles conectadas al núcleo respaldado. "
            "No se usa una forma específica ni se modifica la calibración."
        ),
    }
    summary_path = output / "resumen_10_registro_calibrado.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[Paso 10 | 8/8] Generando vista previa posterior a la limpieza...")
    save_preview(
        output / "preview_registro_general_calibrado_v1_2.png",
        cleaned_registered,
        quality,
        metrics,
    )

    print("\n========== PASO 10 — REGISTRO GENERAL CALIBRADO ==========")
    print("Objeto:", obj)
    print("Calidad:", quality)
    print(f"Primarias informativas: {len(informative)}/25")
    print(f"Aceptadas/informativas: " f"{len(accepted)}/{len(informative) if informative else 0}")
    print("Cierre:", closure_status)
    print(
        "Voxels multivista:",
        f"{len(fused['points']):,} -> {np.count_nonzero(keep):,}",
        f"| frontera recuperada={sum(recovered_each_hop):,}",
    )
    print("Calibración:", cal_path)
    print("Refinamiento de línea del eje:", axis_refinement["decision"])
    if axis_refinement["applied"]:
        print(
            "Desplazamiento transversal aplicado (mm):",
            axis_refinement["selected_offset_xyz_mm"],
        )
    print("Modelo de poses usado por el paso 11:", runtime_pose_model_path)
    print("Salida:", output)
    print("Sin forma específica, sin ICP 6DoF y sin corrección angular; calibración congelada por defecto.")
    print("=============================================================\n")
    return 0 if quality != "rejected" else 2


def _entrada_con_diagnostico():
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    from utilidades_progreso import informar_inicio

    informar_inicio(__file__, "10", "Registrar vistas calibradas")
    raise SystemExit(main())


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
