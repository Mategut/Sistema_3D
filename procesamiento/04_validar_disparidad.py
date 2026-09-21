#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Paso 04 V2.2 — Validación regional de disparidad orientada a observación.

Principios de diseño:
    - una falla del modelo regional NO debe borrar observaciones estéreo reales;
    - la confianza es un peso, no un veto binario;
    - una observación física solo se rechaza si contradice simultáneamente
      el modelo regional y su vecindario local;
    - los huecos sin disparidad pueden recuperarse únicamente con evidencia
      geométrica local o consenso entre modelos vecinos compatibles.

Cada superpíxel intenta ajustar de forma robusta:
    d(u,v) = a*u + b*v + c

La salida separa explícitamente:
    - observación directa validada;
    - observación fuerte preservada sin modelo;
    - recuperación local respaldada por un modelo regional;
    - observación ambigua preservada por coherencia local;
    - recuperación acotada de huecos por modelo propio o consenso vecino;
    - píxeles que siguen sin evidencia geométrica suficiente.

No se completa una forma conocida ni se usa geometría específica del objeto.
"""

from __future__ import annotations
from utilidades_progreso import operacion
from utilidades_rendimiento import configurar as _configurar_recursos

# Los runtimes numéricos leen estos límites al importarse.
_configurar_recursos(__file__)


import argparse
import csv
import math
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Cada worker es un proceso independiente. Si OpenCV/MKL/OpenBLAS abren a su
# vez varios hilos, 14 workers pueden convertirse en cientos de hilos y bajar
# el rendimiento. Por eso cada worker interno queda limitado a 1 hilo.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import cv2
import numpy as np

from utilidades_mascaras import (
    build_contact_sheet,
    color_labels,
    find_summary,
    finite_stats,
    imwrite_checked,
    load_json,
    load_stereo_geometry,
    overlay_mask,
    prepare_output_directory,
    save_json,
    scalar_visualization,
    validate_summary_context,
)


def build_parser() -> argparse.ArgumentParser:
    """Construye las opciones de línea de comandos de este paso."""
    parser = argparse.ArgumentParser(
        description="Validación regional de disparidad con SLIC y RANSAC."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--object", default="cubo")
    parser.add_argument("--session", default="S01")
    parser.add_argument(
        "--depth-source",
        default="02_estimacion_profundidad",
    )
    parser.add_argument(
        "--silhouette-source",
        default="03_mascara_objeto",
    )
    parser.add_argument(
        "--output-name",
        default="04_validacion_disparidad",
    )
    parser.add_argument("--only-angle", type=float, default=-1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--calibration-dir",
        default="",
        help="Ruta opcional a la calibración estéreo del montaje.",
    )
    parser.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Limpia la salida antes de una corrida completa.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Procesos paralelos por sesión. 0=automático: usa CPU lógica - 2 "
            "(por ejemplo, 14 workers en una CPU de 16 hilos). "
            "Usa 1 para ejecución secuencial."
        ),
    )

    parser.add_argument("--superpixel-region-size", type=int, default=18)
    parser.add_argument("--superpixel-ruler", type=float, default=12.0)
    parser.add_argument("--superpixel-iterations", type=int, default=12)
    parser.add_argument("--superpixel-min-element-size", type=int, default=20)

    parser.add_argument("--seed-minimum-confidence", type=float, default=0.62)
    parser.add_argument("--seed-minimum-count", type=int, default=24)
    parser.add_argument("--seed-minimum-ratio", type=float, default=0.035)
    parser.add_argument("--minimum-disparity-px", type=float, default=1.0)
    parser.add_argument("--maximum-disparity-px", type=float, default=1500.0)
    parser.add_argument(
        "--minimum-depth-mm",
        type=float,
        default=None,
        help="Límite físico cercano. Si se omite, se hereda de el paso CREStereo.",
    )
    parser.add_argument(
        "--maximum-depth-mm",
        type=float,
        default=None,
        help="Límite físico lejano. Si se omite, se hereda de el paso CREStereo.",
    )

    parser.add_argument("--ransac-iterations", type=int, default=900)
    parser.add_argument("--ransac-threshold-px", type=float, default=3.5)
    parser.add_argument("--minimum-inlier-ratio", type=float, default=0.62)
    parser.add_argument("--maximum-plane-rmse-px", type=float, default=3.2)
    parser.add_argument("--maximum-plane-mad-px", type=float, default=2.4)
    parser.add_argument("--minimum-region-area", type=int, default=35)

    parser.add_argument(
        "--adaptive-seed-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Si una región no alcanza suficientes semillas >= seed-minimum-confidence, "
            "intenta un segundo conjunto de semillas relativo a la confianza de esa región."
        ),
    )
    parser.add_argument(
        "--adaptive-seed-minimum-confidence",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--adaptive-seed-percentile",
        type=float,
        default=70.0,
    )
    parser.add_argument(
        "--preserve-strict-observations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Conserva observaciones de alta confianza y rango físico válido aunque "
            "el superpíxel no admita un buen modelo plano."
        ),
    )

    parser.add_argument(
        "--observed-consensus-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "En regiones sin modelo fiable conserva observaciones REALES de "
            "confianza media solo si forman un consenso espacial robusto. "
            "Nunca crea disparidad nueva."
        ),
    )
    parser.add_argument(
        "--observed-consensus-minimum-confidence",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--observed-consensus-minimum-count",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--observed-consensus-maximum-mad-px",
        type=float,
        default=3.50,
    )
    parser.add_argument(
        "--observed-consensus-maximum-rmse-px",
        type=float,
        default=5.50,
    )
    parser.add_argument(
        "--observed-consensus-sigma",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--observed-consensus-minimum-tolerance-px",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--observed-consensus-maximum-tolerance-px",
        type=float,
        default=8.0,
    )

    # V2.2 — validación observation-first.
    # Una observación dentro del rango físico ya no se elimina por un único
    # fallo regional. Se exige evidencia conjunta modelo + vecindario.
    parser.add_argument(
        "--observation-first-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--local-consistency-window-px",
        type=int,
        default=11,
    )
    parser.add_argument(
        "--local-consistency-minimum-neighbors",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--local-consistency-sigma",
        type=float,
        default=3.25,
    )
    parser.add_argument(
        "--local-consistency-minimum-tolerance-px",
        type=float,
        default=3.5,
    )
    parser.add_argument(
        "--local-consistency-maximum-tolerance-px",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--model-adaptive-tolerance-factor",
        type=float,
        default=2.75,
    )
    parser.add_argument(
        "--model-adaptive-maximum-tolerance-px",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--hard-outlier-model-factor",
        type=float,
        default=1.85,
    )
    parser.add_argument(
        "--hard-outlier-local-factor",
        type=float,
        default=1.65,
    )
    parser.add_argument(
        "--extreme-residual-model-factor",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--absolute-hard-outlier-px",
        type=float,
        default=24.0,
    )
    parser.add_argument(
        "--no-local-support-confidence-floor",
        type=float,
        default=0.18,
    )
    parser.add_argument(
        "--cross-region-gap-recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Recupera solo huecos SIN observación mediante consenso de al menos "
            "dos modelos regionales vecinos compatibles y sin cruzar un borde "
            "visual fuerte."
        ),
    )
    parser.add_argument(
        "--cross-region-gap-passes",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--cross-region-minimum-models",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--cross-region-maximum-model-spread-px",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--cross-region-maximum-boundary-gradient",
        type=float,
        default=28.0,
    )
    parser.add_argument(
        "--cross-region-minimum-boundary-pixels",
        type=int,
        default=6,
    )

    parser.add_argument("--validation-tolerance-px", type=float, default=6.0)
    parser.add_argument(
        "--fill-accepted-regions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Completa todo un superpíxel aceptado con el plano regional. "
            "Por defecto se conserva únicamente profundidad observada compatible, "
            "priorizando datos faltantes frente a geometría inventada."
        ),
    )
    parser.add_argument("--observed-blend-minimum", type=float, default=0.20)
    parser.add_argument("--observed-blend-maximum", type=float, default=0.90)

    parser.add_argument(
        "--model-gap-recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Interpola únicamente huecos sin disparidad dentro de una región con "
            "modelo robusto fuerte y cerca de inliers observados."
        ),
    )
    parser.add_argument(
        "--model-gap-max-distance-px",
        type=float,
        default=0.0,
        help=(
            "Distancia máxima a un inlier para completar un hueco. "
            "0 = 1.35 * superpixel-region-size."
        ),
    )
    parser.add_argument("--model-gap-minimum-inlier-ratio", type=float, default=0.72)
    parser.add_argument("--model-gap-maximum-rmse-px", type=float, default=2.60)
    parser.add_argument("--model-gap-maximum-mad-px", type=float, default=1.80)
    parser.add_argument("--model-gap-minimum-seeds", type=int, default=32)
    parser.add_argument(
        "--model-gap-minimum-seed-confidence-median",
        type=float,
        default=0.45,
    )

    parser.add_argument("--adjacency-jump-threshold-px", type=float, default=10.0)
    parser.add_argument("--adjacency-gradient-threshold", type=float, default=30.0)
    parser.add_argument("--adjacency-minimum-boundary-pixels", type=int, default=8)

    parser.add_argument("--minimum-valid-mask-ratio", type=float, default=0.20)
    parser.add_argument("--maximum-valid-mask-ratio", type=float, default=0.98)
    parser.add_argument("--cyclic-half-window", type=int, default=2)
    parser.add_argument("--cyclic-minimum-depth-tolerance-mm", type=float, default=25.0)
    parser.add_argument("--cyclic-depth-mad-factor", type=float, default=6.0)
    parser.add_argument("--closure-depth-tolerance-mm", type=float, default=15.0)
    parser.add_argument(
        "--nominal-360-is-closure",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Solo marca A360 como cierre si físicamente coincide con A000. "
            "La campaña V7.2 no genera A360; debe permanecer desactivado."
        ),
    )
    return parser


def create_superpixels(image: np.ndarray, args) -> Tuple[np.ndarray, np.ndarray, int]:
    """Segmenta la imagen en superpíxeles para el ajuste regional de disparidad."""
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "createSuperpixelSLIC"):
        raise RuntimeError(
            "La instalación de OpenCV no contiene ximgproc/SLIC. " "Instala opencv-contrib-python."
        )

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    algorithm = getattr(cv2.ximgproc, "SLICO", 101)
    slic = cv2.ximgproc.createSuperpixelSLIC(
        lab,
        algorithm=algorithm,
        region_size=int(args.superpixel_region_size),
        ruler=float(args.superpixel_ruler),
    )
    slic.iterate(int(args.superpixel_iterations))
    slic.enforceLabelConnectivity(int(args.superpixel_min_element_size))
    labels = slic.getLabels().astype(np.int32)
    contour = slic.getLabelContourMask(True)
    return labels, contour, int(slic.getNumberOfSuperpixels())


def normalized_coordinates(u: np.ndarray, v: np.ndarray, shape: Tuple[int, int]):
    """Construye las coordenadas normalizadas de la imagen."""
    h, w = shape
    un = (u.astype(np.float64) - 0.5 * (w - 1)) / max(w, 1)
    vn = (v.astype(np.float64) - 0.5 * (h - 1)) / max(h, 1)
    return un, vn


def design_matrix(u: np.ndarray, v: np.ndarray, shape: Tuple[int, int]):
    """Construye la matriz de diseño del modelo afín de disparidad."""
    un, vn = normalized_coordinates(u, v, shape)
    return np.column_stack((un, vn, np.ones(len(un), dtype=np.float64)))


def weighted_least_squares(
    matrix: np.ndarray,
    disparity: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Resuelve el ajuste lineal ponderado de los datos regionales."""
    weights = np.maximum(weights.astype(np.float64), 1e-8)
    root = np.sqrt(weights)
    weighted_matrix = matrix * root[:, None]
    weighted_disparity = disparity * root
    coefficients, _, _, _ = np.linalg.lstsq(
        weighted_matrix,
        weighted_disparity,
        rcond=None,
    )
    return coefficients


def robust_plane_fit(
    u: np.ndarray,
    v: np.ndarray,
    disparity: np.ndarray,
    confidence: np.ndarray,
    shape: Tuple[int, int],
    args,
    rng: np.random.Generator,
) -> Tuple[Optional[dict], dict]:
    """Ajuste robusto con diagnóstico explícito de la causa de fallo."""
    count = len(disparity)
    diagnostics = {
        "seed_count": int(count),
        "status": "failed",
        "failure_stage": None,
        "best_inlier_ratio": None,
        "best_median_error_px": None,
        "final_inlier_ratio": None,
        "final_rmse_px": None,
        "final_mad_px": None,
    }

    if count < args.seed_minimum_count:
        diagnostics["failure_stage"] = "insufficient_seed_count"
        return None, diagnostics

    matrix = design_matrix(u, v, shape)
    threshold = float(args.ransac_threshold_px)
    best_inliers = None
    best_count = 0
    best_error = float("inf")

    probabilities = np.maximum(confidence.astype(np.float64), 1e-4)
    probabilities = probabilities / np.sum(probabilities)

    for _ in range(int(args.ransac_iterations)):
        try:
            indexes = rng.choice(count, size=3, replace=False, p=probabilities)
        except ValueError:
            indexes = rng.choice(count, size=3, replace=False)

        sample_matrix = matrix[indexes]
        if np.linalg.matrix_rank(sample_matrix) < 3:
            continue

        coefficients, _, _, _ = np.linalg.lstsq(
            sample_matrix,
            disparity[indexes],
            rcond=None,
        )
        residual = np.abs(disparity - matrix @ coefficients)
        inliers = residual <= threshold
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 3:
            continue

        median_error = float(np.median(residual[inliers]))
        if inlier_count > best_count or (inlier_count == best_count and median_error < best_error):
            best_inliers = inliers
            best_count = inlier_count
            best_error = median_error

    if best_inliers is None:
        diagnostics["failure_stage"] = "ransac_no_consensus"
        return None, diagnostics

    diagnostics["best_inlier_ratio"] = float(best_count / count)
    diagnostics["best_median_error_px"] = float(best_error)

    if best_count / count < args.minimum_inlier_ratio:
        diagnostics["failure_stage"] = "ransac_low_inlier_ratio"
        return None, diagnostics

    coefficients = weighted_least_squares(
        matrix[best_inliers],
        disparity[best_inliers],
        confidence[best_inliers],
    )

    # IRLS con pérdida de Huber.
    for _ in range(12):
        residual = disparity - matrix @ coefficients
        absolute = np.abs(residual)
        median = float(np.median(absolute[best_inliers]))
        scale = max(0.5, 1.4826 * median)
        delta = 1.345 * scale
        huber = np.ones_like(absolute)
        large = absolute > delta
        huber[large] = delta / np.maximum(absolute[large], 1e-8)
        weights = confidence * huber

        new_coefficients = weighted_least_squares(
            matrix[best_inliers],
            disparity[best_inliers],
            weights[best_inliers],
        )
        if np.linalg.norm(new_coefficients - coefficients) < 1e-5:
            coefficients = new_coefficients
            break
        coefficients = new_coefficients

    residual = disparity - matrix @ coefficients
    absolute = np.abs(residual)
    inliers = absolute <= threshold
    inlier_count = int(np.count_nonzero(inliers))

    if inlier_count < 3:
        diagnostics["failure_stage"] = "irls_no_inliers"
        return None, diagnostics

    inlier_ratio = float(inlier_count / count)
    rmse = float(np.sqrt(np.mean(np.square(residual[inliers]))))
    mad = float(np.median(absolute[inliers]))

    diagnostics.update(
        {
            "final_inlier_ratio": inlier_ratio,
            "final_rmse_px": rmse,
            "final_mad_px": mad,
        }
    )

    if inlier_ratio < args.minimum_inlier_ratio:
        diagnostics["failure_stage"] = "final_low_inlier_ratio"
        return None, diagnostics
    if rmse > args.maximum_plane_rmse_px:
        diagnostics["failure_stage"] = "final_high_rmse"
        return None, diagnostics
    if mad > args.maximum_plane_mad_px:
        diagnostics["failure_stage"] = "final_high_mad"
        return None, diagnostics

    diagnostics["status"] = "accepted"
    diagnostics["failure_stage"] = None

    return {
        "coefficients": coefficients.astype(float).tolist(),
        "inlier_ratio": inlier_ratio,
        "rmse_px": rmse,
        "mad_px": mad,
        "seed_count": int(count),
        "inlier_count": inlier_count,
        "seed_disparity_median": float(np.median(disparity)),
        "seed_confidence_median": float(np.median(confidence)),
    }, diagnostics


def adaptive_seed_mask_for_region(
    region: np.ndarray,
    base_valid_mask: np.ndarray,
    confidence: np.ndarray,
    args,
) -> Tuple[np.ndarray, dict]:
    """Segundo nivel de semillas relativo a la propia región."""
    candidates = region & base_valid_mask
    values = confidence[candidates]
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.zeros(region.shape, dtype=bool), {
            "status": "unavailable",
            "threshold": None,
            "candidate_count": 0,
        }

    percentile_value = float(
        np.percentile(
            values,
            np.clip(args.adaptive_seed_percentile, 0.0, 100.0),
        )
    )
    threshold = min(
        float(args.seed_minimum_confidence),
        max(
            float(args.adaptive_seed_minimum_confidence),
            percentile_value,
        ),
    )

    mask = candidates & (confidence >= threshold)
    return mask, {
        "status": "available",
        "threshold": float(threshold),
        "candidate_count": int(values.size),
        "selected_count": int(np.count_nonzero(mask)),
        "confidence_median": float(np.median(values)),
        "confidence_p90": float(np.percentile(values, 90.0)),
    }


def robust_observed_consensus(
    region: np.ndarray,
    disparity: np.ndarray,
    confidence: np.ndarray,
    shape: Tuple[int, int],
    args,
) -> Tuple[np.ndarray, dict]:
    """Valida observaciones existentes con un modelo local NO generativo.

    A diferencia de `robust_plane_fit`, este ajuste no autoriza completar
    huecos. Su único propósito es responder:

        "¿Estas disparidades observadas de confianza media son coherentes
         entre sí dentro de este superpíxel?"

    Si la respuesta es sí, se conservan sus valores ORIGINALES.
    """
    candidates = (
        region
        & np.isfinite(disparity)
        & np.isfinite(confidence)
        & (confidence >= args.observed_consensus_minimum_confidence)
    )

    ys, xs = np.nonzero(candidates)
    count = int(len(xs))

    diagnostics = {
        "status": "failed",
        "candidate_count": count,
        "accepted_count": 0,
        "rmse_px": None,
        "mad_px": None,
        "tolerance_px": None,
        "reason": None,
    }

    if count < int(args.observed_consensus_minimum_count):
        diagnostics["reason"] = "insufficient_observations"
        return np.zeros(region.shape, dtype=bool), diagnostics

    values = disparity[ys, xs].astype(np.float64)
    weights = np.maximum(
        confidence[ys, xs].astype(np.float64),
        1e-4,
    )

    matrix = design_matrix(
        xs.astype(np.float64),
        ys.astype(np.float64),
        shape,
    )

    # Inicio por mínimos cuadrados ponderados.
    coefficients = weighted_least_squares(
        matrix,
        values,
        weights,
    )

    # IRLS Huber, deliberadamente más tolerante que el modelo generativo.
    for _ in range(15):
        residual = values - matrix @ coefficients
        absolute = np.abs(residual)
        median_abs = float(np.median(absolute))
        scale = max(0.75, 1.4826 * median_abs)
        delta = 1.345 * scale

        huber = np.ones_like(absolute)
        large = absolute > delta
        huber[large] = delta / np.maximum(absolute[large], 1e-8)

        new_coefficients = weighted_least_squares(
            matrix,
            values,
            weights * huber,
        )
        if np.linalg.norm(new_coefficients - coefficients) < 1e-5:
            coefficients = new_coefficients
            break
        coefficients = new_coefficients

    residual = values - matrix @ coefficients
    absolute = np.abs(residual)

    mad = float(np.median(absolute))
    robust_sigma = max(0.75, 1.4826 * mad)

    tolerance = float(
        np.clip(
            args.observed_consensus_sigma * robust_sigma,
            args.observed_consensus_minimum_tolerance_px,
            args.observed_consensus_maximum_tolerance_px,
        )
    )

    accepted_local = absolute <= tolerance
    accepted_count = int(np.count_nonzero(accepted_local))

    if accepted_count < int(args.observed_consensus_minimum_count):
        diagnostics.update(
            {
                "mad_px": mad,
                "tolerance_px": tolerance,
                "reason": "insufficient_consensus_inliers",
            }
        )
        return np.zeros(region.shape, dtype=bool), diagnostics

    rmse = float(np.sqrt(np.mean(np.square(residual[accepted_local]))))
    accepted_mad = float(np.median(absolute[accepted_local]))

    diagnostics.update(
        {
            "accepted_count": accepted_count,
            "rmse_px": rmse,
            "mad_px": accepted_mad,
            "tolerance_px": tolerance,
        }
    )

    # El fallback solo se habilita si el conjunto observado es compacto.
    if accepted_mad > args.observed_consensus_maximum_mad_px:
        diagnostics["reason"] = "consensus_mad_too_high"
        return np.zeros(region.shape, dtype=bool), diagnostics

    if rmse > args.observed_consensus_maximum_rmse_px:
        diagnostics["reason"] = "consensus_rmse_too_high"
        return np.zeros(region.shape, dtype=bool), diagnostics

    accepted = np.zeros(region.shape, dtype=bool)
    accepted[
        ys[accepted_local],
        xs[accepted_local],
    ] = True

    diagnostics["status"] = "accepted"
    diagnostics["reason"] = None
    return accepted, diagnostics


def compute_local_observation_statistics(
    disparity: np.ndarray,
    confidence: np.ndarray,
    physical_mask: np.ndarray,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Media/escala local ponderadas, excluyendo el propio píxel.

    Se calculan únicamente con observaciones físicas existentes. La confianza
    modula el peso, pero incluso una confianza moderada aporta evidencia.
    """
    valid = (
        np.asarray(physical_mask).astype(bool) & np.isfinite(disparity) & np.isfinite(confidence)
    )

    k = max(3, int(args.local_consistency_window_px))
    if k % 2 == 0:
        k += 1

    weights = np.zeros(disparity.shape, dtype=np.float32)
    weights[valid] = np.clip(confidence[valid], 0.08, 1.0).astype(np.float32)

    values = np.zeros(disparity.shape, dtype=np.float32)
    values[valid] = disparity[valid].astype(np.float32)

    ones = valid.astype(np.float32)
    kernel = (k, k)

    sum_w = cv2.boxFilter(weights, cv2.CV_32F, kernel, normalize=False)
    sum_d = cv2.boxFilter(weights * values, cv2.CV_32F, kernel, normalize=False)
    sum_d2 = cv2.boxFilter(weights * values * values, cv2.CV_32F, kernel, normalize=False)
    count = cv2.boxFilter(ones, cv2.CV_32F, kernel, normalize=False)

    # Excluir el propio píxel para evitar auto-confirmación.
    sum_w = sum_w - weights
    sum_d = sum_d - weights * values
    sum_d2 = sum_d2 - weights * values * values
    count = count - ones

    mean = np.full(disparity.shape, np.nan, dtype=np.float32)
    scale = np.full(disparity.shape, np.nan, dtype=np.float32)

    enough = (sum_w > 1e-5) & (count >= float(args.local_consistency_minimum_neighbors))
    mean[enough] = sum_d[enough] / sum_w[enough]

    variance = np.zeros(disparity.shape, dtype=np.float32)
    variance[enough] = np.maximum(
        0.0,
        sum_d2[enough] / sum_w[enough] - mean[enough] * mean[enough],
    )
    scale[enough] = np.sqrt(variance[enough])

    return mean, scale, count.astype(np.float32)


def adaptive_model_tolerance(fit: dict, args) -> float:
    """Tolerancia regional derivada de la calidad real del ajuste."""
    robust_scale = max(
        float(fit.get("mad_px", 0.0)) * 1.4826,
        float(fit.get("rmse_px", 0.0)),
        0.75,
    )
    return float(
        np.clip(
            max(
                float(args.validation_tolerance_px),
                float(args.model_adaptive_tolerance_factor) * robust_scale,
            ),
            float(args.validation_tolerance_px),
            float(args.model_adaptive_maximum_tolerance_px),
        )
    )


def classify_direct_observations(
    observed: np.ndarray,
    observed_physical: np.ndarray,
    confidence_values: np.ndarray,
    predicted: np.ndarray,
    plausible: np.ndarray,
    local_mean_values: np.ndarray,
    local_scale_values: np.ndarray,
    local_count_values: np.ndarray,
    fit: dict,
    args,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Clasificación observation-first para observaciones ya existentes.

    Un píxel físico solo se elimina como outlier fuerte cuando contradice el
    modelo regional Y, si hay soporte local suficiente, también contradice su
    vecindario. Sin soporte local, una confianza razonable evita el borrado.
    """
    residual_model = np.abs(observed - predicted)
    model_tol = adaptive_model_tolerance(fit, args)

    local_available = (
        np.isfinite(local_mean_values)
        & np.isfinite(local_scale_values)
        & (local_count_values >= float(args.local_consistency_minimum_neighbors))
    )
    local_residual = np.abs(observed - local_mean_values)
    local_tol = np.clip(
        float(args.local_consistency_sigma) * np.maximum(local_scale_values, 0.75),
        float(args.local_consistency_minimum_tolerance_px),
        float(args.local_consistency_maximum_tolerance_px),
    )

    model_good = plausible & (residual_model <= model_tol)
    local_good = local_available & (local_residual <= local_tol)

    model_hard_bad = (~plausible) | (
        residual_model > model_tol * float(args.hard_outlier_model_factor)
    )
    local_hard_bad = local_available & (
        local_residual > local_tol * float(args.hard_outlier_local_factor)
    )

    confidence_finite = np.where(
        np.isfinite(confidence_values),
        confidence_values,
        0.0,
    )

    # Con vecindario disponible se requieren DOS contradicciones. Sin
    # vecindario, solo se descarta si además la confianza es muy baja.
    extreme_model_bad = residual_model > np.maximum(
        model_tol * float(args.extreme_residual_model_factor),
        float(args.absolute_hard_outlier_px),
    )
    strong_local_counterevidence = local_good & (
        confidence_finite >= float(args.seed_minimum_confidence)
    )

    hard_outlier = observed_physical & (
        (
            model_hard_bad
            & (
                local_hard_bad
                | (
                    (~local_available)
                    & (confidence_finite < float(args.no_local_support_confidence_floor))
                )
            )
        )
        | (extreme_model_bad & (~strong_local_counterevidence))
    )

    if bool(args.observation_first_validation):
        keep = observed_physical & (~hard_outlier)
    else:
        keep = observed_physical & model_good

    # "Ambigua preservada" = real, no estrictamente compatible con el modelo,
    # pero conservada por coherencia local / ausencia de evidencia de outlier.
    ambiguous_kept = keep & (~model_good)

    diagnostics = {
        "model_tolerance_px": float(model_tol),
        "observed_physical": int(np.count_nonzero(observed_physical)),
        "model_good": int(np.count_nonzero(observed_physical & model_good)),
        "local_good": int(np.count_nonzero(observed_physical & local_good)),
        "hard_outliers": int(np.count_nonzero(hard_outlier)),
        "ambiguous_preserved": int(np.count_nonzero(ambiguous_kept)),
        "kept": int(np.count_nonzero(keep)),
    }
    return keep, ambiguous_kept, diagnostics


def recover_cross_region_missing_depth(
    labels: np.ndarray,
    silhouette: np.ndarray,
    image: np.ndarray,
    disparity: np.ndarray,
    base_valid_mask: np.ndarray,
    region_results: Dict[int, dict],
    regularized_disparity: np.ndarray,
    depth_valid_mask: np.ndarray,
    depth_source_map: np.ndarray,
    model_recovery_mask: np.ndarray,
    loss_reason_map: np.ndarray,
    effective_min_disparity: float,
    effective_max_disparity: float,
    args,
) -> dict:
    """Recuperación NO generativa de observaciones existentes y recuperación
    acotada de huecos sin observación mediante consenso entre modelos vecinos.

    La recuperación nunca reemplaza una disparidad física observada. Solo
    actúa donde `base_valid_mask` es falso y exige >=2 modelos vecinos que:
      - tengan ajuste robusto;
      - compartan una frontera visual suave con la región objetivo;
      - predigan disparidades mutuamente compatibles.
    """
    if not bool(args.cross_region_gap_recovery):
        return {"status": "disabled", "recovered_pixels": 0, "events": []}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    adjacency = build_adjacency(labels, silhouette)
    neighbor_edges: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for (a, b), pixels in adjacency.items():
        neighbor_edges.setdefault(a, []).append((b, pixels))
        neighbor_edges.setdefault(b, []).append((a, pixels))

    # Modelos propagados se permiten como apoyo en la segunda pasada, pero
    # conservan el número de modelos originales que los sustentó.
    propagated: Dict[int, dict] = {}
    events: List[dict] = []
    total_recovered = 0

    for pass_index in range(max(1, int(args.cross_region_gap_passes))):
        recovered_this_pass = 0

        for label_value in np.unique(labels[silhouette > 0]):
            label = int(label_value)
            region = (labels == label) & (silhouette > 0)

            # Solo huecos SIN observación física. No sustituir datos reales.
            missing = region & (depth_valid_mask == 0) & (~base_valid_mask)
            if np.count_nonzero(missing) == 0:
                continue

            candidates = []
            for neighbor, pixels in neighbor_edges.get(label, []):
                if len(pixels) < int(args.cross_region_minimum_boundary_pixels):
                    continue

                boundary_gradient = float(np.median(gradient[pixels[:, 1], pixels[:, 0]]))
                if boundary_gradient > float(args.cross_region_maximum_boundary_gradient):
                    continue

                fit = None
                origin_count = 1
                rr = region_results.get(int(neighbor))
                if rr is not None and rr.get("fit") is not None:
                    fit = rr["fit"]
                    if rr.get("model_recovery_blocked", False):
                        continue
                elif int(neighbor) in propagated:
                    fit = propagated[int(neighbor)]["fit"]
                    origin_count = int(propagated[int(neighbor)].get("origin_count", 2))

                if fit is None:
                    continue

                candidates.append((int(neighbor), fit, boundary_gradient, origin_count))

            if len(candidates) < int(args.cross_region_minimum_models):
                continue

            ys, xs = np.nonzero(missing)
            if len(xs) == 0:
                continue

            prediction_stack = []
            used_neighbors = []
            for neighbor, fit, boundary_gradient, origin_count in candidates:
                pred = predict_plane(
                    fit["coefficients"],
                    xs.astype(np.float64),
                    ys.astype(np.float64),
                    labels.shape,
                ).astype(np.float32)
                prediction_stack.append(pred)
                used_neighbors.append((neighbor, boundary_gradient, origin_count))

            stack = np.vstack(prediction_stack)
            valid_pred = (
                np.isfinite(stack)
                & (stack >= effective_min_disparity)
                & (stack <= effective_max_disparity)
            )
            valid_count = np.sum(valid_pred, axis=0)

            # Evita RuntimeWarning de columnas all-NaN. Solo se agregan
            # estadísticas donde ya existen suficientes modelos físicos válidos.
            eligible = valid_count >= int(args.cross_region_minimum_models)
            median_pred = np.full(valid_count.shape, np.nan, dtype=np.float32)
            spread = np.full(valid_count.shape, np.inf, dtype=np.float32)
            if np.any(eligible):
                work = np.where(valid_pred[:, eligible], stack[:, eligible], np.nan)
                median_pred[eligible] = np.nanmedian(work, axis=0).astype(np.float32)
                spread[eligible] = (np.nanmax(work, axis=0) - np.nanmin(work, axis=0)).astype(
                    np.float32
                )

            recover_local = (
                eligible
                & np.isfinite(median_pred)
                & (spread <= float(args.cross_region_maximum_model_spread_px))
            )

            if not np.any(recover_local):
                continue

            ry = ys[recover_local]
            rx = xs[recover_local]
            regularized_disparity[ry, rx] = median_pred[recover_local]
            depth_valid_mask[ry, rx] = 255
            depth_source_map[ry, rx] = 7
            model_recovery_mask[ry, rx] = 255
            loss_reason_map[ry, rx] = 0

            count = int(len(rx))
            recovered_this_pass += count
            total_recovered += count

            # Modelo virtual para una sola propagación adicional: promedio
            # robusto de coeficientes vecinos. No se marca como ajuste observado.
            coeffs = np.asarray([c[1]["coefficients"] for c in candidates], dtype=np.float64)
            median_coeffs = np.median(coeffs, axis=0)
            propagated[label] = {
                "fit": {"coefficients": median_coeffs.astype(float).tolist()},
                "origin_count": int(len(candidates)),
            }

            events.append(
                {
                    "pass": int(pass_index + 1),
                    "region": int(label),
                    "supporting_models": int(len(candidates)),
                    "recovered_pixels": count,
                    "median_prediction_spread_px": float(np.nanmedian(spread[recover_local])),
                    "neighbors": [int(v[0]) for v in used_neighbors],
                }
            )

        if recovered_this_pass == 0:
            break

    return {
        "status": "ok",
        "recovered_pixels": int(total_recovered),
        "events": events,
    }


def support_distance_map(
    region: np.ndarray,
    support_pixels: np.ndarray,
) -> np.ndarray:
    """Distancia euclídea al soporte observado, calculada solo en su bbox."""
    output = np.full(region.shape, np.inf, dtype=np.float32)
    ys, xs = np.nonzero(region)
    if xs.size == 0:
        return output

    support = support_pixels & region
    if np.count_nonzero(support) == 0:
        return output

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    patch_support = support[y0:y1, x0:x1]
    source = (~patch_support).astype(np.uint8)
    distance = cv2.distanceTransform(
        source,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    output[y0:y1, x0:x1] = distance
    return output


def model_is_strong_for_gap_recovery(
    fit: dict,
    args,
) -> bool:
    return bool(
        fit is not None
        and fit["inlier_ratio"] >= args.model_gap_minimum_inlier_ratio
        and fit["rmse_px"] <= args.model_gap_maximum_rmse_px
        and fit["mad_px"] <= args.model_gap_maximum_mad_px
        and fit["seed_count"] >= args.model_gap_minimum_seeds
        and fit["seed_confidence_median"] >= args.model_gap_minimum_seed_confidence_median
    )


def predict_plane(
    coefficients: Sequence[float],
    u: np.ndarray,
    v: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    """Evalúa el modelo regional afín en las coordenadas indicadas."""
    matrix = design_matrix(u, v, shape)
    return matrix @ np.asarray(coefficients, dtype=np.float64)


def build_adjacency(
    labels: np.ndarray,
    silhouette: np.ndarray,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Identifica las regiones que comparten frontera en el mapa de etiquetas."""
    pairs: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    active = silhouette > 0

    # Bordes horizontales.
    difference = (labels[:, :-1] != labels[:, 1:]) & active[:, :-1] & active[:, 1:]
    ys, xs = np.nonzero(difference)
    for y, x in zip(ys, xs):
        a = int(labels[y, x])
        b = int(labels[y, x + 1])
        key = (min(a, b), max(a, b))
        pairs.setdefault(key, []).append((x, y))

    # Bordes verticales.
    difference = (labels[:-1, :] != labels[1:, :]) & active[:-1, :] & active[1:, :]
    ys, xs = np.nonzero(difference)
    for y, x in zip(ys, xs):
        a = int(labels[y, x])
        b = int(labels[y + 1, x])
        key = (min(a, b), max(a, b))
        pairs.setdefault(key, []).append((x, y))

    return {key: np.asarray(value, dtype=np.int32) for key, value in pairs.items()}


def reject_inconsistent_adjacencies(
    region_results: Dict[int, dict],
    labels: np.ndarray,
    silhouette: np.ndarray,
    image: np.ndarray,
    args,
) -> List[dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    adjacency = build_adjacency(labels, silhouette)
    events = []

    for (a, b), pixels in adjacency.items():
        if len(pixels) < args.adjacency_minimum_boundary_pixels:
            continue
        if (
            a not in region_results
            or b not in region_results
            or region_results[a]["status"] != "accepted"
            or region_results[b]["status"] != "accepted"
        ):
            continue

        u = pixels[:, 0].astype(np.float64)
        v = pixels[:, 1].astype(np.float64)
        prediction_a = predict_plane(
            region_results[a]["fit"]["coefficients"],
            u,
            v,
            labels.shape,
        )
        prediction_b = predict_plane(
            region_results[b]["fit"]["coefficients"],
            u,
            v,
            labels.shape,
        )

        jump = float(np.median(np.abs(prediction_a - prediction_b)))
        boundary_gradient = float(np.median(gradient[pixels[:, 1], pixels[:, 0]]))

        event = {
            "region_a": a,
            "region_b": b,
            "boundary_pixels": int(len(pixels)),
            "median_disparity_jump_px": jump,
            "median_image_gradient": boundary_gradient,
            "action": "none",
        }

        if (
            jump > args.adjacency_jump_threshold_px
            and boundary_gradient < args.adjacency_gradient_threshold
        ):
            score_a = (
                region_results[a]["fit"]["inlier_ratio"]
                * region_results[a]["fit"]["seed_count"]
                / max(region_results[a]["fit"]["rmse_px"], 0.25)
            )
            score_b = (
                region_results[b]["fit"]["inlier_ratio"]
                * region_results[b]["fit"]["seed_count"]
                / max(region_results[b]["fit"]["rmse_px"], 0.25)
            )
            weaker = a if score_a < score_b else b

            # V2: una inconsistencia entre modelos regionales ya no elimina
            # observaciones estéreo fuertes. Solo impide completar huecos con
            # el modelo más débil.
            region_results[weaker]["model_recovery_blocked"] = True
            if region_results[weaker]["status"] == "accepted":
                region_results[weaker]["status"] = "warning"
            region_results[weaker]["reasons"].append(
                "Conflicto de adyacencia: se conserva observación directa, "
                "pero se bloquea recuperación por modelo regional."
            )
            event["action"] = f"block_model_recovery_{weaker}"

        events.append(event)

    return events


def resolve_worker_count(requested: int, job_count: int) -> int:
    """Número de procesos adecuado para la máquina y el lote actual."""
    logical = os.cpu_count() or 2

    if requested <= 0:
        # Reserva dos procesadores lógicos para Windows/UI/IO.
        workers = max(1, logical - 2)
    else:
        workers = max(1, requested)

    return min(workers, max(job_count, 1))


def _parallel_worker_initializer():
    """Inicialización de cada proceso hijo."""
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # SLIC/NumPy se ejecutan en CPU. Evitar que OpenCL cree otra capa de
    # paralelismo impredecible.
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


def _region_rng(view_index: int, label: int) -> np.random.Generator:
    """
    RNG determinista por vista+región.

    De esta forma el resultado RANSAC no depende del orden en que Windows
    programe los 14 procesos. Cada región siempre recibe la misma semilla.
    """
    seed = np.random.SeedSequence([500, int(view_index), int(label)])
    return np.random.default_rng(seed)


def process_view_job(job: dict) -> dict:
    """
    Procesa UNA vista completa.

    Esta es la unidad de paralelización. Cada proceso hace para su vista:
      SLIC -> superpíxeles -> RANSAC regional -> Huber/WLS ->
      adyacencias -> profundidad -> archivos diagnósticos.

    No existe memoria compartida mutable entre vistas.
    """
    args = argparse.Namespace(**job["args"])

    view_index = int(job["view_index"])
    total_views = int(job["total_views"])
    view = job["view"]

    depth_dir = Path(job["depth_dir"])
    silhouette_dir = Path(job["silhouette_dir"])
    output_dir = Path(job["output_dir"])

    effective_min_disparity = float(job["effective_min_disparity"])
    effective_max_disparity = float(job["effective_max_disparity"])
    fx = float(job["fx"])
    baseline_mm = float(job["baseline_mm"])
    source_view = job.get("source_view", {}) or {}

    stem = str(view["stem"])
    angle = float(view["angle_deg"])

    image = cv2.imread(
        str(depth_dir / f"{stem}_rect_L.png"),
        cv2.IMREAD_COLOR,
    )
    silhouette = cv2.imread(
        str(silhouette_dir / f"{stem}_silhouette_mask.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    rect_valid = cv2.imread(
        str(depth_dir / f"{stem}_rect_valid_mask.png"),
        cv2.IMREAD_GRAYSCALE,
    )

    disparity_path = depth_dir / f"{stem}_disparity_lr.npy"
    if not disparity_path.exists():
        disparity_path = depth_dir / f"{stem}_disparity.npy"

    confidence_path = depth_dir / f"{stem}_confidence.npy"
    lr_state_path = depth_dir / f"{stem}_lr_state.npy"

    if (
        image is None
        or silhouette is None
        or rect_valid is None
        or not disparity_path.exists()
        or not confidence_path.exists()
    ):
        return {
            "ok": False,
            "view_index": view_index,
            "stem": stem,
            "angle": angle,
            "error": f"Entradas incompletas para {stem}",
        }

    disparity = np.load(str(disparity_path)).astype(np.float32)
    confidence = np.load(str(confidence_path)).astype(np.float32)
    lr_state = (
        np.load(str(lr_state_path)).astype(np.uint8)
        if lr_state_path.is_file()
        else np.ones(disparity.shape, dtype=np.uint8)
    )

    if not (
        image.shape[:2]
        == silhouette.shape
        == rect_valid.shape
        == disparity.shape
        == confidence.shape
        == lr_state.shape
    ):
        raise ValueError(f"Dimensiones incompatibles en {stem}")

    labels, contours, superpixel_count = create_superpixels(
        image,
        args,
    )

    base_physical_mask = (
        (silhouette > 0)
        & (rect_valid > 0)
        & np.isfinite(disparity)
        & np.isfinite(confidence)
        & (disparity >= effective_min_disparity)
        & (disparity <= effective_max_disparity)
    )
    lr_hard_contradicted = lr_state == 4
    lr_weak_one_sided = lr_state == 3
    lr_strong = np.isin(lr_state, (1, 2))
    lr_eligible_observation = np.isin(lr_state, (1, 2, 3))
    # Estados 4 (contradicción fuerte) y 5 (evidencia directa insuficiente) no
    # se usan como observaciones ni como semillas regionales. Siguen contando
    # como disparidad físicamente existente para impedir que un modelo rellene
    # encima de una medición contradictoria/no demostrada.
    base_valid_mask = base_physical_mask & lr_eligible_observation
    strict_seed_mask = (
        base_valid_mask
        & lr_strong
        & (confidence >= args.seed_minimum_confidence)
    )

    region_results: Dict[int, dict] = {}
    active_labels = np.unique(labels[silhouette > 0])

    for label_value in active_labels:
        label = int(label_value)
        region = (labels == label) & (silhouette > 0)
        area = int(np.count_nonzero(region))

        strict_seeds = region & strict_seed_mask
        strict_seed_count = int(np.count_nonzero(strict_seeds))

        required = max(
            args.seed_minimum_count,
            int(math.ceil(area * args.seed_minimum_ratio)),
        )

        result = {
            "status": "rejected",
            "mode": "rejected",
            "reasons": [],
            "area_pixels": area,
            "seed_pixels": strict_seed_count,
            "strict_seed_pixels": strict_seed_count,
            "required_seed_pixels": int(required),
            "fit_seed_mode": None,
            "fit_seed_threshold": float(args.seed_minimum_confidence),
            "fit": None,
            "fit_diagnostics": None,
            "adaptive_seed_diagnostics": None,
            "observed_consensus_diagnostics": None,
            "model_recovery_blocked": False,
        }

        if area < args.minimum_region_area:
            if args.preserve_strict_observations and strict_seed_count > 0:
                result["status"] = "warning"
                result["mode"] = "observed_only"
                result["reasons"].append(
                    "Región pequeña: se conservan únicamente observaciones estrictas."
                )
            else:
                result["reasons"].append("Región demasiado pequeña.")
            region_results[label] = result
            continue

        fit = None
        fit_diag = None
        fit_seed_mask = None
        fit_seed_mode = None
        fit_seed_threshold = float(args.seed_minimum_confidence)

        if strict_seed_count >= required:
            sy, sx = np.nonzero(strict_seeds)
            fit, fit_diag = robust_plane_fit(
                sx.astype(np.float64),
                sy.astype(np.float64),
                disparity[strict_seeds].astype(np.float64),
                confidence[strict_seeds].astype(np.float64),
                disparity.shape,
                args,
                _region_rng(view_index, label),
            )
            fit_seed_mask = strict_seeds
            fit_seed_mode = "strict"

        # Si no hubo suficientes semillas estrictas o su modelo falló,
        # se permite una segunda tentativa adaptativa.
        if fit is None and args.adaptive_seed_fallback:
            adaptive_seeds, adaptive_diag = adaptive_seed_mask_for_region(
                region,
                base_valid_mask,
                confidence,
                args,
            )
            result["adaptive_seed_diagnostics"] = adaptive_diag
            adaptive_count = int(np.count_nonzero(adaptive_seeds))

            if adaptive_count >= required:
                sy, sx = np.nonzero(adaptive_seeds)
                fit2, fit_diag2 = robust_plane_fit(
                    sx.astype(np.float64),
                    sy.astype(np.float64),
                    disparity[adaptive_seeds].astype(np.float64),
                    confidence[adaptive_seeds].astype(np.float64),
                    disparity.shape,
                    args,
                    _region_rng(view_index, label + 100000),
                )
                if fit2 is not None:
                    fit = fit2
                    fit_diag = fit_diag2
                    fit_seed_mask = adaptive_seeds
                    fit_seed_mode = "adaptive"
                    fit_seed_threshold = float(adaptive_diag["threshold"])
                elif fit_diag is None:
                    fit_diag = fit_diag2

        result["fit_diagnostics"] = fit_diag

        if fit is not None:
            result["fit"] = fit
            result["fit_seed_mode"] = fit_seed_mode
            result["fit_seed_threshold"] = fit_seed_threshold
            result["fit_seed_pixels"] = int(np.count_nonzero(fit_seed_mask))
            result["mode"] = "model"
            result["status"] = "accepted" if fit_seed_mode == "strict" else "warning"
            if fit_seed_mode == "adaptive":
                result["reasons"].append("Modelo regional válido usando semillas adaptativas.")
        elif args.preserve_strict_observations and strict_seed_count > 0:
            result["mode"] = "observed_only"
            result["status"] = "warning"
            stage = fit_diag.get("failure_stage") if isinstance(fit_diag, dict) else None
            result["reasons"].append(
                "Sin modelo regional fiable; se preservan observaciones "
                "estrictas de alta confianza." + (f" Fallo: {stage}." if stage else "")
            )
        else:
            if strict_seed_count < required:
                result["reasons"].append(
                    f"Semillas estrictas insuficientes: " f"{strict_seed_count} < {required}."
                )
            else:
                stage = (
                    fit_diag.get("failure_stage") if isinstance(fit_diag, dict) else "desconocido"
                )
                result["reasons"].append(f"No se obtuvo modelo regional robusto: {stage}.")

        region_results[label] = result

    adjacency_events = reject_inconsistent_adjacencies(
        region_results,
        labels,
        silhouette,
        image,
        args,
    )

    regularized_disparity = np.full(
        disparity.shape,
        np.nan,
        dtype=np.float32,
    )
    depth_valid_mask = np.zeros(
        disparity.shape,
        dtype=np.uint8,
    )
    residual_map = np.full(
        disparity.shape,
        np.nan,
        dtype=np.float32,
    )

    # Proveniencia por píxel:
    # 0 = inválido/sin salida
    # 1 = observación estricta preservada sin modelo
    # 2 = observación compatible con modelo regional
    # 3 = hueco sin disparidad recuperado por modelo local fuerte
    # 4 = relleno regional completo heredado (--fill-accepted-regions)
    # 5 = observación REAL de confianza media preservada por consenso robusto
    # 6 = observación REAL ambigua preservada por coherencia local/observation-first
    # 7 = hueco SIN observación recuperado por consenso de modelos vecinos
    depth_source_map = np.zeros(disparity.shape, dtype=np.uint8)

    # Motivo principal cuando un píxel de la silueta no llega a profundidad:
    # 0 = válido/no aplica
    # 1 = no existe disparidad física observada
    # 2 = región sin modelo y sin consenso observado suficiente
    # 3 = disparidad observada incompatible con modelo
    # 4 = predicción fuera de rango físico
    # 5 = existe observación, pero confianza < mínimo del consenso
    # 6 = existe observación candidata, pero no pertenece al consenso robusto
    loss_reason_map = np.zeros(disparity.shape, dtype=np.uint8)

    model_recovery_mask = np.zeros(disparity.shape, dtype=np.uint8)
    observed_only_mask = np.zeros(disparity.shape, dtype=np.uint8)
    observed_consensus_mask = np.zeros(disparity.shape, dtype=np.uint8)
    ambiguous_observation_mask = np.zeros(disparity.shape, dtype=np.uint8)
    one_sided_validated_mask = np.zeros(disparity.shape, dtype=np.uint8)
    low_direct_validated_mask = np.zeros(disparity.shape, dtype=np.uint8)

    # V2.2: estadísticas locales calculadas una sola vez por vista.
    local_mean_map, local_scale_map, local_count_map = compute_local_observation_statistics(
        disparity,
        confidence,
        base_valid_mask,
        args,
    )

    model_gap_max_distance = (
        float(args.model_gap_max_distance_px)
        if args.model_gap_max_distance_px > 0
        else 1.35 * float(args.superpixel_region_size)
    )

    for label, result in region_results.items():
        region = (labels == label) & (silhouette > 0)
        ys, xs = np.nonzero(region)
        if len(xs) == 0:
            continue

        observed = disparity[ys, xs]
        observed_physical = (
            np.isfinite(observed)
            & (observed >= effective_min_disparity)
            & (observed <= effective_max_disparity)
        )
        state_here = lr_state[ys, xs]
        contradicted_here = state_here == 4
        weak_one_sided_here = state_here == 3
        strong_lr_here = np.isin(state_here, (1, 2))
        eligible_observed = observed_physical & np.isin(state_here, (1, 2, 3))

        strict_here = (
            eligible_observed
            & strong_lr_here
            & np.isfinite(confidence[ys, xs])
            & (confidence[ys, xs] >= args.seed_minimum_confidence)
        )

        # Regiones sin modelo: preservar observación física siempre que exista
        # coherencia local o evidencia estricta. La confianza NO es un veto.
        if result.get("mode") != "model" or result.get("fit") is None:
            accepted_direct = np.zeros(len(xs), dtype=bool)

            if args.preserve_strict_observations and np.any(strict_here):
                accepted_direct |= strict_here

            consensus_mask = np.zeros(disparity.shape, dtype=bool)
            consensus_diag = {
                "status": "disabled",
                "accepted_count": 0,
            }

            if args.observed_consensus_fallback:
                consensus_mask, consensus_diag = robust_observed_consensus(
                    region & base_valid_mask,
                    disparity,
                    confidence,
                    disparity.shape,
                    args,
                )
                accepted_direct |= consensus_mask[ys, xs]

            # Coherencia local para observaciones que no alcanzaron el consenso
            # regional. Esto rescata superficies lisas y de baja textura sin
            # crear una sola disparidad nueva.
            local_available = (
                np.isfinite(local_mean_map[ys, xs])
                & np.isfinite(local_scale_map[ys, xs])
                & (local_count_map[ys, xs] >= float(args.local_consistency_minimum_neighbors))
            )
            local_residual = np.abs(observed - local_mean_map[ys, xs])
            local_tolerance = np.clip(
                float(args.local_consistency_sigma) * np.maximum(local_scale_map[ys, xs], 0.75),
                float(args.local_consistency_minimum_tolerance_px),
                float(args.local_consistency_maximum_tolerance_px),
            )
            local_coherent = (
                eligible_observed & local_available & (local_residual <= local_tolerance)
            )
            accepted_direct |= local_coherent

            # Sin vecindario suficiente, una observación física de confianza
            # razonable se mantiene como ambigua en lugar de borrarse.
            conf_values = np.where(
                np.isfinite(confidence[ys, xs]),
                confidence[ys, xs],
                0.0,
            )
            no_local_ambiguous = (
                eligible_observed
                & (~local_available)
                & (conf_values >= float(args.no_local_support_confidence_floor))
            )
            if bool(args.observation_first_validation):
                accepted_direct |= no_local_ambiguous

            if np.any(accepted_direct):
                sy = ys[accepted_direct]
                sx = xs[accepted_direct]
                regularized_disparity[sy, sx] = observed[accepted_direct]
                depth_valid_mask[sy, sx] = 255

                # Prioridad de procedencia: estricto -> consenso -> local/ambiguo.
                strict_global = np.zeros(disparity.shape, dtype=bool)
                strict_global[ys[strict_here], xs[strict_here]] = True
                consensus_global = consensus_mask & (~strict_global)

                if np.any(strict_here):
                    depth_source_map[ys[strict_here], xs[strict_here]] = 1
                    observed_only_mask[ys[strict_here], xs[strict_here]] = 255

                if np.any(consensus_global):
                    depth_source_map[consensus_global] = 5
                    observed_consensus_mask[consensus_global] = 255

                weak_validated = (
                    region
                    & (depth_valid_mask > 0)
                    & (lr_state == 3)
                    & (depth_source_map == 0)
                )
                depth_source_map[weak_validated] = 8
                one_sided_validated_mask[weak_validated] = 255

                additional_ambiguous = region & (depth_valid_mask > 0) & (depth_source_map == 0)
                depth_source_map[additional_ambiguous] = 6
                ambiguous_observation_mask[additional_ambiguous] = 255

                if result["status"] == "rejected":
                    result["status"] = "warning"
                result["mode"] = "observed_preserved"
                result["reasons"].append(
                    "Sin plano regional fiable; se preserva evidencia estéreo "
                    "real mediante confianza estricta, consenso o coherencia local."
                )

            result["observed_consensus_diagnostics"] = consensus_diag
            result["local_observation_diagnostics"] = {
                "local_coherent": int(np.count_nonzero(local_coherent)),
                "no_local_ambiguous": int(np.count_nonzero(no_local_ambiguous)),
                "accepted_direct": int(np.count_nonzero(accepted_direct)),
            }

            remaining = region & (depth_valid_mask == 0)
            no_observed = remaining & (~base_valid_mask)
            rejected_observed = remaining & base_valid_mask

            loss_reason_map[no_observed] = 1
            loss_reason_map[rejected_observed] = 6

            strict_count = int(np.count_nonzero((depth_source_map == 1) & region))
            consensus_count = int(np.count_nonzero((depth_source_map == 5) & region))
            ambiguous_count = int(np.count_nonzero((depth_source_map == 6) & region))
            one_sided_count = int(np.count_nonzero((depth_source_map == 8) & region))

            result["validated_pixels"] = strict_count + consensus_count + ambiguous_count + one_sided_count
            result["validated_ratio"] = result["validated_pixels"] / max(result["area_pixels"], 1)
            result["model_recovered_pixels"] = 0
            result["observed_only_pixels"] = strict_count
            result["observed_consensus_pixels"] = consensus_count
            result["ambiguous_observed_pixels"] = ambiguous_count
            result["one_sided_validated_pixels"] = one_sided_count
            continue

        fit = result["fit"]
        predicted = predict_plane(
            fit["coefficients"],
            xs.astype(np.float64),
            ys.astype(np.float64),
            disparity.shape,
        ).astype(np.float32)

        plausible = (
            np.isfinite(predicted)
            & (predicted >= effective_min_disparity)
            & (predicted <= effective_max_disparity)
        )

        if not np.any(plausible):
            result["status"] = "warning"
            result["mode"] = "observed_preserved"
            result["reasons"].append(
                "Modelo fuera de rango físico; no se usa para generar profundidad. "
                "Se preservan observaciones físicas respaldadas por evidencia directa/local."
            )

            local_available = (
                np.isfinite(local_mean_map[ys, xs])
                & np.isfinite(local_scale_map[ys, xs])
                & (local_count_map[ys, xs] >= float(args.local_consistency_minimum_neighbors))
            )
            local_residual = np.abs(observed - local_mean_map[ys, xs])
            local_tolerance = np.clip(
                float(args.local_consistency_sigma) * np.maximum(local_scale_map[ys, xs], 0.75),
                float(args.local_consistency_minimum_tolerance_px),
                float(args.local_consistency_maximum_tolerance_px),
            )
            conf_values = np.where(
                np.isfinite(confidence[ys, xs]),
                confidence[ys, xs],
                0.0,
            )

            keep_direct = strict_here | (
                eligible_observed & local_available & (local_residual <= local_tolerance)
            )
            if bool(args.observation_first_validation):
                keep_direct |= (
                    eligible_observed
                    & (~local_available)
                    & (conf_values >= float(args.no_local_support_confidence_floor))
                )

            if np.any(keep_direct):
                sy = ys[keep_direct]
                sx = xs[keep_direct]
                regularized_disparity[sy, sx] = observed[keep_direct]
                depth_valid_mask[sy, sx] = 255

                strict_kept = keep_direct & strict_here
                if np.any(strict_kept):
                    depth_source_map[ys[strict_kept], xs[strict_kept]] = 1
                    observed_only_mask[ys[strict_kept], xs[strict_kept]] = 255

                weak_kept = keep_direct & weak_one_sided_here
                if np.any(weak_kept):
                    depth_source_map[ys[weak_kept], xs[weak_kept]] = 8
                    one_sided_validated_mask[ys[weak_kept], xs[weak_kept]] = 255

                ambiguous_kept = keep_direct & (~strict_here) & (~weak_one_sided_here)
                if np.any(ambiguous_kept):
                    depth_source_map[ys[ambiguous_kept], xs[ambiguous_kept]] = 6
                    ambiguous_observation_mask[ys[ambiguous_kept], xs[ambiguous_kept]] = 255

            invalid_here = region & (depth_valid_mask == 0)
            loss_reason_map[invalid_here & (~base_valid_mask)] = 1
            loss_reason_map[invalid_here & base_valid_mask] = 4

            strict_count = int(np.count_nonzero((depth_source_map == 1) & region))
            ambiguous_count = int(np.count_nonzero((depth_source_map == 6) & region))
            one_sided_count = int(np.count_nonzero((depth_source_map == 8) & region))
            result["validated_pixels"] = strict_count + ambiguous_count + one_sided_count
            result["validated_ratio"] = result["validated_pixels"] / max(result["area_pixels"], 1)
            result["model_recovered_pixels"] = 0
            result["observed_only_pixels"] = strict_count
            result["observed_consensus_pixels"] = 0
            result["ambiguous_observed_pixels"] = ambiguous_count
            result["one_sided_validated_pixels"] = one_sided_count
            continue

        residual = np.abs(observed - predicted)
        residual_map[ys, xs] = residual

        direct_keep, ambiguous_kept, direct_diag = classify_direct_observations(
            observed,
            eligible_observed,
            confidence[ys, xs],
            predicted,
            plausible,
            local_mean_map[ys, xs],
            local_scale_map[ys, xs],
            local_count_map[ys, xs],
            fit,
            args,
        )
        result["direct_observation_diagnostics"] = direct_diag

        model_tol = float(direct_diag["model_tolerance_px"])
        compatible_observed = eligible_observed & plausible & (residual <= model_tol)

        # Estado 5: nunca es semilla, nunca se acepta solo por confianza y no
        # participa del fallback sin modelo. Puede recuperarse únicamente cuando
        # DOS fuentes geométricas independientes coinciden: modelo regional y
        # vecindario local construido desde estados 1/2/3.
        low_direct_candidate = observed_physical & (state_here == 5)
        local_available_state5 = (
            np.isfinite(local_mean_map[ys, xs])
            & np.isfinite(local_scale_map[ys, xs])
            & (local_count_map[ys, xs] >= float(args.local_consistency_minimum_neighbors))
        )
        local_residual_state5 = np.abs(observed - local_mean_map[ys, xs])
        local_tolerance_state5 = np.clip(
            float(args.local_consistency_sigma) * np.maximum(local_scale_map[ys, xs], 0.75),
            float(args.local_consistency_minimum_tolerance_px),
            float(args.local_consistency_maximum_tolerance_px),
        )
        low_direct_joint_keep = (
            low_direct_candidate
            & plausible
            & (residual <= model_tol)
            & local_available_state5
            & (local_residual_state5 <= local_tolerance_state5)
            & np.isfinite(confidence[ys, xs])
            & (confidence[ys, xs] >= float(args.no_local_support_confidence_floor))
        )
        if np.any(low_direct_joint_keep):
            direct_keep |= low_direct_joint_keep
        result["low_direct_joint_validated_pixels"] = int(
            np.count_nonzero(low_direct_joint_keep)
        )

        output = observed.copy()
        valid_region = direct_keep.copy()

        # Modo heredado explícito: único caso en que se completa todo el
        # superpíxel. Permanece desactivado por defecto.
        if args.fill_accepted_regions:
            valid_region = plausible.copy()
            output = predicted.copy()
            blend = np.clip(
                confidence[ys, xs],
                args.observed_blend_minimum,
                args.observed_blend_maximum,
            )
            use_observed = direct_keep & plausible
            output[use_observed] = (
                blend[use_observed] * observed[use_observed]
                + (1.0 - blend[use_observed]) * predicted[use_observed]
            )
            recovered_local = np.zeros(len(xs), dtype=bool)
        else:
            # Recuperación local SOLO donde NO existe una observación física.
            # V2.2 permite cualquier modelo ya aceptado por RANSAC/IRLS; la
            # antigua segunda puerta de calidad era redundante y causaba huecos.
            can_recover = args.model_gap_recovery and not result.get(
                "model_recovery_blocked", False
            )

            recovered_local = np.zeros(len(xs), dtype=bool)

            if can_recover:
                direct_support = np.zeros(disparity.shape, dtype=bool)
                direct_support[ys[direct_keep], xs[direct_keep]] = True

                distance = support_distance_map(region, direct_support)
                region_distance = distance[ys, xs]

                recovered_local = (
                    (~observed_physical)
                    & plausible
                    & np.isfinite(region_distance)
                    & (region_distance <= model_gap_max_distance)
                )

                if np.any(recovered_local):
                    output[recovered_local] = predicted[recovered_local]
                    valid_region |= recovered_local

            result["model_recovered_pixels"] = int(np.count_nonzero(recovered_local))

        sy = ys[valid_region]
        sx = xs[valid_region]
        regularized_disparity[sy, sx] = output[valid_region]
        depth_valid_mask[sy, sx] = 255

        if args.fill_accepted_regions:
            depth_source_map[sy, sx] = 4
        else:
            low_direct_kept = direct_keep & (state_here == 5)
            if np.any(low_direct_kept):
                depth_source_map[
                    ys[low_direct_kept],
                    xs[low_direct_kept],
                ] = 9
                low_direct_validated_mask[
                    ys[low_direct_kept],
                    xs[low_direct_kept],
                ] = 255

            weak_direct_kept = direct_keep & weak_one_sided_here
            if np.any(weak_direct_kept):
                depth_source_map[
                    ys[weak_direct_kept],
                    xs[weak_direct_kept],
                ] = 8
                one_sided_validated_mask[
                    ys[weak_direct_kept],
                    xs[weak_direct_kept],
                ] = 255

            model_compatible_kept = (
                direct_keep & compatible_observed & (~weak_one_sided_here) & (state_here != 5)
            )
            if np.any(model_compatible_kept):
                depth_source_map[
                    ys[model_compatible_kept],
                    xs[model_compatible_kept],
                ] = 2

            ambiguous_nonweak = ambiguous_kept & (~weak_one_sided_here) & (state_here != 5)
            if np.any(ambiguous_nonweak):
                depth_source_map[
                    ys[ambiguous_nonweak],
                    xs[ambiguous_nonweak],
                ] = 6
                ambiguous_observation_mask[
                    ys[ambiguous_nonweak],
                    xs[ambiguous_nonweak],
                ] = 255

            if np.any(recovered_local):
                depth_source_map[
                    ys[recovered_local],
                    xs[recovered_local],
                ] = 3
                model_recovery_mask[
                    ys[recovered_local],
                    xs[recovered_local],
                ] = 255

        # Diagnóstico: solo se marca incompatible cuando una observación REAL
        # superó la prueba conjunta de outlier.
        missing_observed = (~observed_physical) & (~valid_region)
        incompatible = observed_physical & (~direct_keep) & (~valid_region)
        implausible = (~plausible) & (~valid_region)

        loss_reason_map[ys[missing_observed], xs[missing_observed]] = 1
        loss_reason_map[ys[incompatible], xs[incompatible]] = 3
        loss_reason_map[
            ys[implausible & (~observed_physical)], xs[implausible & (~observed_physical)]
        ] = 4

        result["validated_pixels"] = int(np.count_nonzero(valid_region))
        result["validated_ratio"] = result["validated_pixels"] / max(result["area_pixels"], 1)
        result["observed_only_pixels"] = 0
        result["observed_consensus_pixels"] = 0
        result["ambiguous_observed_pixels"] = int(np.count_nonzero(ambiguous_kept))
        result["model_gap_max_distance_px_effective"] = float(model_gap_max_distance)
        result["model_recovery_allowed"] = bool(
            args.model_gap_recovery and not result.get("model_recovery_blocked", False)
        )

    # V2.2: segunda oportunidad SOLO para huecos sin observación. Se exige
    # consenso entre al menos dos modelos vecinos compatibles.
    cross_region_recovery = recover_cross_region_missing_depth(
        labels,
        silhouette,
        image,
        disparity,
        base_physical_mask,
        region_results,
        regularized_disparity,
        depth_valid_mask,
        depth_source_map,
        model_recovery_mask,
        loss_reason_map,
        effective_min_disparity,
        effective_max_disparity,
        args,
    )

    final_cloud_mask = cv2.bitwise_and(
        silhouette,
        depth_valid_mask,
    )

    depth_regularized = np.full(
        disparity.shape,
        np.nan,
        dtype=np.float32,
    )

    valid_disparity = (
        np.isfinite(regularized_disparity)
        & (regularized_disparity >= effective_min_disparity)
        & (regularized_disparity <= effective_max_disparity)
        & (final_cloud_mask > 0)
    )

    depth_regularized[valid_disparity] = fx * baseline_mm / regularized_disparity[valid_disparity]

    silhouette_pixels = int(np.count_nonzero(silhouette))
    valid_pixels = int(np.count_nonzero(final_cloud_mask))

    valid_ratio = valid_pixels / max(
        silhouette_pixels,
        1,
    )

    depth_values = depth_regularized[np.isfinite(depth_regularized) & (final_cloud_mask > 0)]

    depth_median = float(np.median(depth_values)) if depth_values.size else None

    reasons = []
    quality = "accepted"

    if valid_ratio < args.minimum_valid_mask_ratio:
        quality = "warning"
        reasons.append(f"Cobertura de profundidad baja: " f"{valid_ratio:.2%}.")

    if valid_ratio > args.maximum_valid_mask_ratio:
        quality = "warning"
        reasons.append(f"Cobertura sospechosamente alta: " f"{valid_ratio:.2%}.")

    if depth_median is None:
        quality = "rejected"
        reasons.append("No quedó profundidad regional válida.")

    mask_path = output_dir / f"{stem}_depth_valid_mask.png"
    cloud_mask_path = output_dir / f"{stem}_cloud_mask.png"
    disparity_path_out = output_dir / f"{stem}_disparity_regularized.npy"
    depth_path_out = output_dir / f"{stem}_depth_regularized_mm.npy"
    residual_path = output_dir / f"{stem}_regional_residual.npy"
    overlay_path = output_dir / f"{stem}_cloud_mask_overlay.png"
    labels_path = output_dir / f"{stem}_superpixels.png"
    status_path = output_dir / f"{stem}_superpixel_status.png"
    residual_vis_path = output_dir / f"{stem}_regional_residual_vis.png"
    stats_path = output_dir / f"{stem}_regional_stats.json"

    source_map_path = output_dir / f"{stem}_depth_source_map.npy"
    source_vis_path = output_dir / f"{stem}_depth_source_map.png"
    loss_map_path = output_dir / f"{stem}_depth_loss_reason.npy"
    loss_vis_path = output_dir / f"{stem}_depth_loss_reason.png"
    recovery_mask_path = output_dir / f"{stem}_model_recovery_mask.png"
    observed_only_path = output_dir / f"{stem}_observed_only_mask.png"

    observed_consensus_path = output_dir / f"{stem}_observed_consensus_mask.png"
    one_sided_validated_path = output_dir / f"{stem}_one_sided_validated_mask.png"
    low_direct_validated_path = output_dir / f"{stem}_low_direct_validated_mask.png"
    ambiguous_observation_path = output_dir / f"{stem}_ambiguous_observation_preserved_mask.png"

    imwrite_checked(
        mask_path,
        depth_valid_mask,
    )
    imwrite_checked(
        cloud_mask_path,
        final_cloud_mask,
    )

    np.save(
        str(disparity_path_out),
        regularized_disparity,
    )
    np.save(
        str(depth_path_out),
        depth_regularized,
    )
    np.save(
        str(residual_path),
        residual_map,
    )

    np.save(str(source_map_path), depth_source_map)
    np.save(str(loss_map_path), loss_reason_map)

    source_vis = np.zeros((*depth_source_map.shape, 3), dtype=np.uint8)
    source_vis[depth_source_map == 1] = (255, 180, 0)  # cian/azul: observado estricto
    source_vis[depth_source_map == 2] = (40, 210, 40)  # verde: observado + modelo
    source_vis[depth_source_map == 3] = (0, 165, 255)  # naranja: recuperado por modelo
    source_vis[depth_source_map == 4] = (180, 80, 220)  # violeta: relleno heredado
    source_vis[depth_source_map == 5] = (0, 255, 255)  # amarillo: observación por consenso
    source_vis[depth_source_map == 6] = (255, 80, 220)  # rosa: observación ambigua preservada
    source_vis[depth_source_map == 7] = (220, 220, 60)  # turquesa: hueco por consenso vecino
    source_vis[depth_source_map == 8] = (80, 220, 255)  # amarillo claro: unilateral validada
    source_vis[depth_source_map == 9] = (120, 170, 255)  # naranja claro: directa débil validada doble
    imwrite_checked(source_vis_path, source_vis)

    loss_vis = np.zeros((*loss_reason_map.shape, 3), dtype=np.uint8)
    loss_vis[loss_reason_map == 1] = (80, 80, 220)  # rojo: sin disparidad observada
    loss_vis[loss_reason_map == 2] = (180, 80, 180)  # magenta: sin modelo fiable
    loss_vis[loss_reason_map == 3] = (0, 165, 255)  # naranja: residuo incompatible
    loss_vis[loss_reason_map == 4] = (255, 180, 0)  # cian: fuera rango predicho
    loss_vis[loss_reason_map == 5] = (120, 120, 120)  # gris: confianza demasiado baja
    loss_vis[loss_reason_map == 6] = (0, 255, 255)  # amarillo: sin consenso robusto
    imwrite_checked(loss_vis_path, loss_vis)

    imwrite_checked(recovery_mask_path, model_recovery_mask)
    imwrite_checked(observed_only_path, observed_only_mask)

    imwrite_checked(
        observed_consensus_path,
        observed_consensus_mask,
    )
    cv2.imwrite(str(one_sided_validated_path), one_sided_validated_mask)
    cv2.imwrite(str(low_direct_validated_path), low_direct_validated_mask)
    imwrite_checked(
        ambiguous_observation_path,
        ambiguous_observation_mask,
    )

    labels_visual = color_labels(labels)
    labels_visual[contours > 0] = (0, 0, 0)

    imwrite_checked(
        labels_path,
        labels_visual,
    )

    imwrite_checked(
        status_path,
        color_labels(
            labels,
            {label: result["status"] for label, result in region_results.items()},
        ),
    )

    imwrite_checked(
        residual_vis_path,
        scalar_visualization(
            residual_map,
            (np.isfinite(residual_map) & (silhouette > 0)),
            cv2.COLORMAP_MAGMA,
        ),
    )

    imwrite_checked(
        overlay_path,
        overlay_mask(
            image,
            final_cloud_mask,
            (f"{angle:05.1f}° | " "profundidad regional"),
        ),
    )

    record = {
        "stem": stem,
        "angle_deg": angle,
        "quality": quality,
        "reasons": reasons,
        "source_depth_quality": (
            source_view.get("session_quality")
            or source_view.get("quality")
            or source_view.get("local_quality")
        ),
        "source_depth_reasons": (
            source_view.get("session_reasons")
            or source_view.get("reasons")
            or source_view.get(
                "local_reasons",
                [],
            )
        ),
        "superpixel_count": (superpixel_count),
        "active_superpixels": int(len(active_labels)),
        "accepted_superpixels": int(
            sum(result["status"] == "accepted" for result in region_results.values())
        ),
        "rejected_superpixels": int(
            sum(result["status"] == "rejected" for result in region_results.values())
        ),
        "warning_superpixels": int(
            sum(result["status"] == "warning" for result in region_results.values())
        ),
        "model_superpixels": int(
            sum(result.get("mode") == "model" for result in region_results.values())
        ),
        "observed_only_superpixels": int(
            sum(result.get("mode") == "observed_only" for result in region_results.values())
        ),
        "silhouette_pixels": (silhouette_pixels),
        "valid_depth_pixels": (valid_pixels),
        "valid_ratio_of_silhouette": (valid_ratio),
        "strict_seed_pixels": int(np.count_nonzero(strict_seed_mask)),
        "physically_valid_observed_pixels": int(np.count_nonzero(base_valid_mask)),
        "observed_model_validated_pixels": int(np.count_nonzero(depth_source_map == 2)),
        "strict_observed_only_pixels": int(np.count_nonzero(depth_source_map == 1)),
        "observed_consensus_pixels": int(np.count_nonzero(depth_source_map == 5)),
        "ambiguous_observed_preserved_pixels": int(np.count_nonzero(depth_source_map == 6)),
        "cross_region_recovered_pixels": int(np.count_nonzero(depth_source_map == 7)),
        "one_sided_validated_pixels": int(np.count_nonzero(depth_source_map == 8)),
        "low_direct_joint_validated_pixels": int(np.count_nonzero(depth_source_map == 9)),
        "lr_hard_contradicted_pixels": int(np.count_nonzero(lr_hard_contradicted & (silhouette > 0))),
        "model_recovered_pixels": int(np.count_nonzero(depth_source_map == 3)),
        "legacy_model_fill_pixels": int(np.count_nonzero(depth_source_map == 4)),
        "remaining_missing_pixels": int(
            np.count_nonzero((silhouette > 0) & (depth_valid_mask == 0))
        ),
        "loss_budget": {
            "no_observed_disparity": int(np.count_nonzero(loss_reason_map == 1)),
            "no_reliable_regional_model": int(np.count_nonzero(loss_reason_map == 2)),
            "observed_model_incompatible": int(np.count_nonzero(loss_reason_map == 3)),
            "prediction_outside_physical_range": int(np.count_nonzero(loss_reason_map == 4)),
            "observed_below_consensus_confidence": int(np.count_nonzero(loss_reason_map == 5)),
            "observed_without_robust_consensus": int(np.count_nonzero(loss_reason_map == 6)),
        },
        "depth_median_mm": (depth_median),
        "depth": finite_stats(depth_values),
        "adjacency_events": (adjacency_events),
        "cross_region_gap_recovery": cross_region_recovery,
        "regions": {str(label): result for label, result in region_results.items()},
        "outputs": {
            "depth_valid_mask": (str(mask_path)),
            "depth_source_map": str(source_map_path),
            "depth_loss_reason_map": str(loss_map_path),
            "model_recovery_mask": str(recovery_mask_path),
            "observed_only_mask": str(observed_only_path),
            "observed_consensus_mask": str(observed_consensus_path),
            "ambiguous_observation_preserved_mask": str(ambiguous_observation_path),
            "cloud_mask": (str(cloud_mask_path)),
            "regularized_disparity": (str(disparity_path_out)),
            "regularized_depth": (str(depth_path_out)),
            "overlay": (str(overlay_path)),
        },
    }

    # El parent volverá a guardar este JSON después de cyclic_quality().
    save_json(
        stats_path,
        record,
    )

    return {
        "ok": True,
        "view_index": view_index,
        "total_views": total_views,
        "record": record,
        "overlay_path": str(overlay_path),
    }


@operacion("Validar disparidad por vistas")
def execute_view_jobs(
    jobs: List[dict],
    worker_count: int,
) -> Tuple[List[dict], List[Path]]:
    """
    Ejecuta las vistas serial o paralelamente y devuelve resultados ordenados.
    """
    results_by_index = {}

    if worker_count <= 1:
        _parallel_worker_initializer()

        for job in jobs:
            result = process_view_job(job)
            results_by_index[int(job["view_index"])] = result

            if result["ok"]:
                record = result["record"]
                print(
                    f"[{result['view_index']:02d}/"
                    f"{result['total_views']:02d}] "
                    f"{record['angle_deg']:05.1f}° | "
                    f"regiones="
                    f"{record['accepted_superpixels']}/"
                    f"{record['active_superpixels']} | "
                    f"cobertura="
                    f"{record['valid_ratio_of_silhouette']:.2%} | "
                    f"rec={record.get('model_recovered_pixels', 0)} | "
                    f"{record['quality']}"
                )
            else:
                print(f"[ERROR] " f"{result['error']}")

    else:
        # spawn es el comportamiento nativo de Windows y evita heredar estados
        # internos de OpenCV desde el proceso padre.
        context = mp.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_parallel_worker_initializer,
        ) as executor:

            future_map = {
                executor.submit(
                    process_view_job,
                    job,
                ): int(job["view_index"])
                for job in jobs
            }

            completed = 0

            for future in as_completed(future_map):
                view_index = future_map[future]

                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        "Falló el worker de la vista " f"{view_index}: {exc}"
                    ) from exc

                results_by_index[view_index] = result
                completed += 1

                if result["ok"]:
                    record = result["record"]

                    print(
                        f"[PAR {completed:02d}/"
                        f"{len(jobs):02d}] "
                        f"V{view_index:02d} | "
                        f"{record['angle_deg']:05.1f}° | "
                        f"regiones="
                        f"{record['accepted_superpixels']}/"
                        f"{record['active_superpixels']} | "
                        f"cobertura="
                        f"{record['valid_ratio_of_silhouette']:.2%} | "
                        f"{record['quality']}",
                        flush=True,
                    )
                else:
                    print(
                        f"[ERROR] " f"{result['error']}",
                        flush=True,
                    )

    records = []
    previews = []

    for view_index in sorted(results_by_index):
        result = results_by_index[view_index]

        if not result["ok"]:
            continue

        records.append(result["record"])
        previews.append(Path(result["overlay_path"]))

    # Seguridad adicional: independientemente del orden de finalización de
    # los workers, toda la validación temporal/cíclica usa orden angular.
    records.sort(key=lambda record: float(record["angle_deg"]))

    preview_map = {
        Path(result["overlay_path"]).name: Path(result["overlay_path"])
        for result in results_by_index.values()
        if result.get("ok")
    }

    previews = [preview_map[Path(record["outputs"]["overlay"]).name] for record in records]

    return records, previews


@operacion("Evaluar consistencia del ciclo")
def cyclic_quality(records: List[dict], args) -> dict:
    unique = [record for record in records if record["depth_median_mm"] is not None]
    unique.sort(key=lambda item: item["angle_deg"])

    if len(unique) >= 5:
        depths = np.asarray(
            [record["depth_median_mm"] for record in unique],
            dtype=np.float64,
        )
        for index, record in enumerate(unique):
            neighbors = []
            for offset in range(
                -args.cyclic_half_window,
                args.cyclic_half_window + 1,
            ):
                if offset == 0:
                    continue
                neighbors.append(depths[(index + offset) % len(depths)])
            neighbors = np.asarray(neighbors, dtype=np.float64)
            median = float(np.median(neighbors))
            mad = float(np.median(np.abs(neighbors - median)))
            limit = max(
                args.cyclic_minimum_depth_tolerance_mm,
                args.cyclic_depth_mad_factor * 1.4826 * mad,
            )
            deviation = abs(record["depth_median_mm"] - median)
            record["cyclic_neighbor_median_mm"] = median
            record["cyclic_deviation_mm"] = deviation
            record["cyclic_limit_mm"] = limit
            if deviation > limit:
                record["quality"] = "rejected"
                record["reasons"].append(
                    "Profundidad incompatible con las vistas vecinas: "
                    f"{deviation:.2f} mm > {limit:.2f} mm."
                )

    angle_zero = next(
        (record for record in records if record["angle_deg"] == 0),
        None,
    )
    angle_360 = next(
        (record for record in records if record["angle_deg"] == 360),
        None,
    )
    closure = {
        "available": bool(
            args.nominal_360_is_closure and angle_zero is not None and angle_360 is not None
        ),
        "depth_difference_mm": None,
        "tolerance_mm": args.closure_depth_tolerance_mm,
        "passed": None,
        "disabled_reason": (
            None
            if args.nominal_360_is_closure
            else "La campaña V7.2 no contiene A360; el cierre físico ya ocurrió en adquisición."
        ),
    }
    if closure["available"]:
        if angle_zero["depth_median_mm"] is not None and angle_360["depth_median_mm"] is not None:
            difference = abs(angle_zero["depth_median_mm"] - angle_360["depth_median_mm"])
            closure["depth_difference_mm"] = difference
            closure["passed"] = difference <= args.closure_depth_tolerance_mm
        angle_360["quality"] = "closure_only"
        angle_360["reasons"].append("La vista 360° se reserva para validar el cierre.")
    return closure


def main() -> int:
    """Valida la disparidad por regiones y exporta observaciones y calidad de sesión."""
    args = build_parser().parse_args()
    root = Path(args.root).expanduser().resolve()
    object_name = args.object.strip().lower()
    session = args.session.strip().upper()

    depth_dir = root / "reconstruccion" / session / args.depth_source
    silhouette_dir = root / "reconstruccion" / session / args.silhouette_source

    diagnostic_run = args.only_angle >= 0.0 or args.limit > 0
    output_name = args.output_name
    if diagnostic_run and output_name == "04_validacion_disparidad":
        suffix = (
            f"A{args.only_angle:05.1f}".replace(".", "p")
            if args.only_angle >= 0.0
            else f"limit{args.limit}"
        )
        output_name = f"{output_name}_diagnostico_{suffix}"
    output_dir = root / "reconstruccion" / session / output_name
    prepare_output_directory(
        output_dir,
        clean=bool(args.clean_output and not diagnostic_run),
    )

    silhouette_summary_path = find_summary(
        silhouette_dir,
        ("resumen_03_mascara_objeto.json",),
    )
    silhouette_summary = load_json(silhouette_summary_path)
    validate_summary_context(silhouette_summary, object_name, session, "Paso 03")

    depth_summary_path = find_summary(
        depth_dir,
        ("resumen_02_estimacion_profundidad.json",),
    )
    depth_summary = load_json(depth_summary_path)
    validate_summary_context(depth_summary, object_name, session, "Paso CREStereo")

    calibration = load_stereo_geometry(
        root,
        object_name=object_name,
        calibration_dir=(args.calibration_dir or None),
    )
    fx = float(calibration["fx"])
    baseline_mm = float(calibration["baseline_mm"])

    # La geometría usada para convertir disparidad a profundidad debe coincidir
    # con la que produjo el paso CREStereo. No se aceptan constantes silenciosas.
    source_fx = depth_summary.get("fx_rectified_px")
    source_baseline = depth_summary.get("baseline_mm")
    if source_fx is not None and not np.isclose(float(source_fx), fx, rtol=1e-6, atol=1e-4):
        raise ValueError(
            "Inconsistencia de calibración: fx del resumen CREStereo "
            f"({float(source_fx):.6f}) != fx actual ({fx:.6f})."
        )
    if source_baseline is not None and not np.isclose(
        float(source_baseline), baseline_mm, rtol=1e-6, atol=1e-4
    ):
        raise ValueError(
            "Inconsistencia de calibración: baseline del resumen CREStereo "
            f"({float(source_baseline):.6f}) != baseline actual ({baseline_mm:.6f})."
        )

    source_parameters = depth_summary.get("parameters", {}) or {}
    minimum_depth_mm = (
        float(args.minimum_depth_mm)
        if args.minimum_depth_mm is not None
        else float(source_parameters.get("minimum_depth_mm", 150.0))
    )
    maximum_depth_mm = (
        float(args.maximum_depth_mm)
        if args.maximum_depth_mm is not None
        else float(source_parameters.get("maximum_depth_mm", 1200.0))
    )
    if not (0.0 < minimum_depth_mm < maximum_depth_mm):
        raise ValueError(
            "Rango físico inválido: se requiere 0 < minimum_depth_mm " "< maximum_depth_mm."
        )

    disparity_from_far_depth = fx * baseline_mm / maximum_depth_mm
    disparity_from_near_depth = fx * baseline_mm / minimum_depth_mm
    effective_min_disparity = max(float(args.minimum_disparity_px), disparity_from_far_depth)
    effective_max_disparity = min(float(args.maximum_disparity_px), disparity_from_near_depth)
    if effective_max_disparity <= effective_min_disparity:
        raise ValueError(
            "Los límites de disparidad y profundidad no dejan un rango válido: "
            f"[{effective_min_disparity:.3f}, {effective_max_disparity:.3f}] px."
        )

    views = sorted(
        silhouette_summary["views"],
        key=lambda item: float(item["angle_deg"]),
    )
    if args.only_angle >= 0:
        views = [
            view
            for view in views
            if np.isclose(float(view["angle_deg"]), args.only_angle, atol=0.05)
        ]
    if args.limit > 0:
        views = views[: args.limit]

    source_views = {
        str(item.get("stem")): item
        for item in depth_summary.get("views", [])
        if item.get("stem") is not None
    }

    records = []
    preview_paths = []

    logical_cpus = os.cpu_count() or 2
    worker_count = resolve_worker_count(
        int(args.workers),
        len(views),
    )

    print("\n========== PASO 04 V2.2: DISPARIDAD REGIONAL ==========")
    print(
        "Rango físico: "
        f"{minimum_depth_mm:.1f}-{maximum_depth_mm:.1f} mm | "
        f"disparidad efectiva: {effective_min_disparity:.2f}-"
        f"{effective_max_disparity:.2f} px"
    )
    print(f"Relleno regional: " f"{'ACTIVO' if args.fill_accepted_regions else 'DESACTIVADO'}")
    print(f"CPU lógica detectada: {logical_cpus} | " f"workers 04: {worker_count}")
    print(
        "Paralelización: una vista completa por proceso. "
        "OpenCV/BLAS internos limitados a 1 hilo por worker."
    )

    jobs = []

    # vars(args) solo contiene tipos simples/serializables.
    worker_args = dict(vars(args))

    for view_index, view in enumerate(
        views,
        start=1,
    ):
        stem = str(view["stem"])

        jobs.append(
            {
                "view_index": view_index,
                "total_views": len(views),
                "view": view,
                "depth_dir": str(depth_dir),
                "silhouette_dir": str(silhouette_dir),
                "output_dir": str(output_dir),
                "effective_min_disparity": (effective_min_disparity),
                "effective_max_disparity": (effective_max_disparity),
                "fx": fx,
                "baseline_mm": (baseline_mm),
                "source_view": (
                    source_views.get(
                        stem,
                        {},
                    )
                ),
                "args": worker_args,
            }
        )

    records, preview_paths = execute_view_jobs(
        jobs,
        worker_count,
    )

    closure = cyclic_quality(records, args)
    for record in records:
        save_json(output_dir / f"{record['stem']}_regional_stats.json", record)

    build_contact_sheet(
        preview_paths,
        output_dir / "contact_sheet_mascaras_profundidad.png",
    )

    summary = {
        "schema_version": 4,
        "method": "slic_observation_first_dual_coherence_and_neighbor_consensus_gap_recovery_v2_2",
        "object": object_name,
        "session": session,
        "depth_source": str(depth_dir),
        "depth_source_summary": str(depth_summary_path),
        "silhouette_source": str(silhouette_dir),
        "calibration_dir": str(calibration["calibration_dir"]),
        "fx_rectified_px": fx,
        "baseline_mm": baseline_mm,
        "physical_depth_range_mm": [minimum_depth_mm, maximum_depth_mm],
        "effective_disparity_range_px": [effective_min_disparity, effective_max_disparity],
        "output_dir": str(output_dir),
        "parameters": vars(args),
        "closure": closure,
        "views_processed": len(records),
        "views_accepted": sum(r["quality"] == "accepted" for r in records),
        "views_warning": sum(r["quality"] == "warning" for r in records),
        "views_rejected": sum(r["quality"] == "rejected" for r in records),
        "views_closure_only": sum(r["quality"] == "closure_only" for r in records),
        "accepted_angles": [r["angle_deg"] for r in records if r["quality"] == "accepted"],
        "warning_angles": [r["angle_deg"] for r in records if r["quality"] == "warning"],
        "rejected_angles": [r["angle_deg"] for r in records if r["quality"] == "rejected"],
        "closure_only_angles": [r["angle_deg"] for r in records if r["quality"] == "closure_only"],
        "views": records,
        "important_note": (
            "cloud_mask = silhouette_mask ∩ depth_valid_mask. V2 nunca borra "
            "observaciones estrictas solo porque falle el plano regional. La "
            "recuperación de huecos usa predicción local únicamente cerca de "
            "inliers de un modelo fuerte; no rellena superpíxeles completos."
        ),
    }
    save_json(output_dir / "resumen_04_validacion_disparidad.json", summary)

    csv_path = output_dir / "calidad_validacion_regional.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = (
            "angle_deg",
            "stem",
            "quality",
            "source_depth_quality",
            "active_superpixels",
            "accepted_superpixels",
            "warning_superpixels",
            "rejected_superpixels",
            "model_superpixels",
            "observed_only_superpixels",
            "silhouette_pixels",
            "valid_depth_pixels",
            "valid_ratio_of_silhouette",
            "strict_seed_pixels",
            "physically_valid_observed_pixels",
            "strict_observed_only_pixels",
            "observed_consensus_pixels",
            "model_recovered_pixels",
            "remaining_missing_pixels",
            "depth_median_mm",
            "reasons",
        )
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "angle_deg": record["angle_deg"],
                    "stem": record["stem"],
                    "quality": record["quality"],
                    "source_depth_quality": record["source_depth_quality"],
                    "active_superpixels": record["active_superpixels"],
                    "accepted_superpixels": record["accepted_superpixels"],
                    "rejected_superpixels": record["rejected_superpixels"],
                    "warning_superpixels": record["warning_superpixels"],
                    "model_superpixels": record["model_superpixels"],
                    "observed_only_superpixels": record["observed_only_superpixels"],
                    "silhouette_pixels": record["silhouette_pixels"],
                    "valid_depth_pixels": record["valid_depth_pixels"],
                    "valid_ratio_of_silhouette": record["valid_ratio_of_silhouette"],
                    "strict_seed_pixels": record["strict_seed_pixels"],
                    "physically_valid_observed_pixels": record["physically_valid_observed_pixels"],
                    "strict_observed_only_pixels": record["strict_observed_only_pixels"],
                    "observed_consensus_pixels": record["observed_consensus_pixels"],
                    "model_recovered_pixels": record["model_recovered_pixels"],
                    "remaining_missing_pixels": record["remaining_missing_pixels"],
                    "depth_median_mm": record["depth_median_mm"],
                    "reasons": " | ".join(record["reasons"]),
                }
            )

    print("\n========== PASO 04 V2.2 COMPLETADO ==========")
    print(f"Salida: {output_dir}")
    print(f"Cierre: {closure}")
    print("===========================================")
    return 0


def _entrada_con_diagnostico():
    # Necesario/recomendado para ProcessPoolExecutor con spawn en Windows.
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    mp.freeze_support()
    from utilidades_progreso import informar_inicio

    informar_inicio(__file__, "04", "Validar disparidad regional")
    sys.exit(main())


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
