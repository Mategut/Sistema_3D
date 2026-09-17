#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Paso 06 V3.2 — Nubes con profundidad estéreo validada.

Consume Paso 05 y genera UNA nube robusta por pose física. Las nubes se
mantienen en coordenadas de cámara; todavía no hay registro multivista.

La profundidad métrica no se recentra ni se escala. La diferencia con el fondo
vacío solo genera candidatos de primer plano: la aceptación definitiva combina
procedencia observada, confianza, consistencia izquierda-derecha, residuo
regional, incertidumbre de disparidad propagada con el Jacobiano estéreo y
continuidad local. Solo las componentes que contienen semillas observadas y
fiables entran a la regularización y a la retroproyección. Esto evita convertir
errores de disparidad cercanos a la cámara en geometría, conserva objetos de
varias piezas y no presupone cubos, cilindros, planos ni tamaños.
"""

from __future__ import annotations
from utilidades_progreso import operacion
from utilidades_rendimiento import query_threads
from utilidades_rendimiento import ejecutar_items, contexto
from utilidades_rendimiento import configurar as _configurar_recursos

# Los runtimes numéricos leen estos límites al importarse.
_configurar_recursos(__file__)


import argparse
import csv
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception as exc:
    raise SystemExit(f"Paso 06 requiere SciPy: {exc}")

from utilidades_mascaras import (
    build_contact_sheet,
    finite_stats,
    load_stereo_geometry,
    make_cloud_preview,
    prepare_output_directory,
    save_json,
    save_ply_ascii,
)

try:
    import open3d as o3d
except Exception:
    o3d = None


def build_parser() -> argparse.ArgumentParser:
    """Construye las opciones de línea de comandos de este paso."""
    parser = argparse.ArgumentParser(
        description="Genera nubes robustas desde el consenso multisesión."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--object", default="cubo")
    parser.add_argument(
        "--consensus-source",
        default="05_consenso_multisesion",
    )
    parser.add_argument(
        "--output-name",
        default="06_nubes_puntos",
    )
    parser.add_argument("--calibration-dir", default="")
    parser.add_argument("--pixel-step", type=int, default=1)
    parser.add_argument("--voxel-mm", type=float, default=0.75)
    parser.add_argument("--sor-neighbors", type=int, default=50)
    parser.add_argument("--sor-std-ratio", type=float, default=1.55)
    parser.add_argument("--radius-nb-points", type=int, default=8)
    parser.add_argument("--radius-mm", type=float, default=2.8)
    parser.add_argument("--dbscan-eps-mm", type=float, default=4.8)
    parser.add_argument("--dbscan-min-points", type=int, default=16)
    parser.add_argument("--minimum-main-cluster-ratio", type=float, default=0.58)
    parser.add_argument("--normal-radius-mm", type=float, default=5.5)
    parser.add_argument("--normal-max-nn", type=int, default=60)
    parser.add_argument("--minimum-final-points", type=int, default=4500)
    parser.add_argument("--minimum-point-confidence", type=float, default=0.14)
    parser.add_argument("--maximum-depth-spread-mm", type=float, default=18.0)
    parser.add_argument("--depth-spread-reference-mm", type=float, default=6.0)
    parser.add_argument("--minimum-depth-support", type=int, default=2)
    parser.add_argument(
        "--background-depth-source",
        default="02_estimacion_profundidad",
        help=(
            "Subcarpeta de cada sesión que contiene background_depth_mm.npy. "
            "Se reutiliza el mismo fondo vacío y se aplican los desplazamientos "
            "registrados por el paso 05."
        ),
    )
    parser.add_argument("--foreground-weak-z-score", type=float, default=1.50)
    parser.add_argument("--foreground-strong-z-score", type=float, default=4.00)
    parser.add_argument("--background-noise-floor-mm", type=float, default=2.00)
    parser.add_argument("--background-relative-noise", type=float, default=0.003)
    parser.add_argument("--background-minimum-models", type=int, default=1)
    parser.add_argument("--foreground-component-minimum-strong-pixels", type=int, default=4)
    parser.add_argument("--foreground-boundary-recovery-px", type=float, default=2.0)
    parser.add_argument("--component-minimum-foreground-score", type=float, default=0.55)
    parser.add_argument("--disparity-uncertainty-floor-px", type=float, default=0.35)
    parser.add_argument("--stereo-weak-score", type=float, default=0.34)
    parser.add_argument("--stereo-strong-score", type=float, default=0.62)
    parser.add_argument("--stereo-minimum-observed-sessions", type=int, default=2)
    parser.add_argument("--stereo-lr-sigma-factor", type=float, default=3.0)
    parser.add_argument("--stereo-regional-sigma-factor", type=float, default=3.0)
    parser.add_argument("--stereo-local-neighbor-z-score", type=float, default=3.5)
    parser.add_argument("--stereo-minimum-consistent-neighbors", type=int, default=2)
    parser.add_argument("--stereo-gradient-edge-relaxation", type=float, default=5.0)
    parser.add_argument("--secondary-cluster-minimum-ratio", type=float, default=0.02)
    parser.add_argument(
        "--secondary-cluster-maximum-distance-mm",
        type=float,
        default=0.0,
        help=(
            "Límite opcional entre componentes. Cero lo desactiva para no "
            "imponer un tamaño máximo al objeto."
        ),
    )
    parser.add_argument("--secondary-cluster-minimum-confidence", type=float, default=0.38)
    parser.add_argument(
        "--inverse-depth-regularization", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--inverse-depth-iterations", type=int, default=8)
    parser.add_argument("--inverse-depth-smoothness", type=float, default=1.35)
    parser.add_argument("--inverse-depth-data-strength", type=float, default=1.0)
    parser.add_argument("--inverse-depth-edge-sigma-mm", type=float, default=2.5)
    parser.add_argument("--inverse-depth-color-sigma", type=float, default=0.12)
    parser.add_argument("--inverse-depth-maximum-displacement-mm", type=float, default=1.8)
    parser.add_argument(
        "--surface-smoothing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--surface-smoothing-iterations", type=int, default=1)
    parser.add_argument("--surface-smoothing-radius-mm", type=float, default=3.2)
    parser.add_argument("--surface-smoothing-max-neighbors", type=int, default=44)
    parser.add_argument("--surface-smoothing-min-neighbors", type=int, default=6)
    parser.add_argument("--surface-smoothing-normal-angle-deg", type=float, default=32.0)
    parser.add_argument("--surface-smoothing-plane-sigma-mm", type=float, default=0.85)
    parser.add_argument("--surface-smoothing-strength", type=float, default=0.30)
    parser.add_argument("--surface-smoothing-max-step-mm", type=float, default=0.20)
    parser.add_argument("--local-residual-reference-mm", type=float, default=0.90)
    return parser


def load_json(path: Path) -> dict:
    """Lee un documento JSON codificado en UTF-8."""
    return json.loads(path.read_text(encoding="utf-8"))


def _shift_image(array: np.ndarray, dy: int, dx: int, fill_value) -> np.ndarray:
    """Desplaza una matriz sin envolver sus bordes."""
    result = np.full_like(array, fill_value)
    h, w = array.shape[:2]
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    dy0, dy1 = max(0, dy), min(h, h + dy)
    dx0, dx1 = max(0, dx), min(w, w + dx)
    if sy1 > sy0 and sx1 > sx0:
        result[dy0:dy1, dx0:dx1] = array[sy0:sy1, sx0:sx1]
    return result


def _robust_sigma(values: np.ndarray, axis=0) -> np.ndarray:
    """Sigma robusta a partir de MAD, ignorando valores no finitos."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        center = np.nanmedian(values, axis=axis)
        mad = np.nanmedian(np.abs(values - np.expand_dims(center, axis)), axis=axis)
    return (1.4826 * mad).astype(np.float32)


def load_background_depths(
    root: Path,
    sessions: List[str],
    source: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """Carga el fondo métrico de cada sesión sin asumir una profundidad fija."""
    arrays: Dict[str, np.ndarray] = {}
    paths: Dict[str, str] = {}
    for session in sessions:
        path = root / "reconstruccion" / str(session) / str(source) / "background_depth_mm.npy"
        if not path.is_file():
            continue
        depth = np.load(str(path)).astype(np.float32)
        if depth.ndim != 2:
            raise ValueError(f"Fondo inválido en {path}: {depth.shape}")
        arrays[str(session)] = depth
        paths[str(session)] = str(path)
    if not arrays:
        expected = root / "reconstruccion" / "<sesion>" / str(source) / "background_depth_mm.npy"
        raise FileNotFoundError(
            "Paso 06 V3.2 necesita el fondo métrico producido por 02. "
            f"No se encontró ninguna entrada con el patrón: {expected}"
        )
    return arrays, paths


def aligned_background_model(
    view: dict,
    backgrounds: Dict[str, np.ndarray],
    shape: Tuple[int, int],
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Construye fondo y sigma en el sistema de píxel del consenso del paso 05."""
    sources = view.get("sources") or []
    aligned = []
    used = []
    if sources:
        for source in sources:
            session = str(source.get("session", ""))
            depth = backgrounds.get(session)
            if depth is None:
                continue
            if depth.shape != shape:
                raise ValueError(f"Fondo {session} con forma {depth.shape}; se esperaba {shape}.")
            alignment = source.get("alignment") or {}
            dx = int(alignment.get("dx", 0) or 0)
            dy = int(alignment.get("dy", 0) or 0)
            aligned.append(_shift_image(depth, dy, dx, np.nan))
            used.append({"session": session, "dx": dx, "dy": dy})
    else:
        for session, depth in backgrounds.items():
            if depth.shape == shape:
                aligned.append(depth)
                used.append({"session": session, "dx": 0, "dy": 0})

    if len(aligned) < max(1, int(args.background_minimum_models)):
        raise RuntimeError(
            "No hay suficientes fondos compatibles para la pose "
            f"P{int(view.get('pose_index', -1)):02d}: {len(aligned)}."
        )

    stack = np.stack(aligned).astype(np.float32)
    finite_count = np.sum(np.isfinite(stack), axis=0).astype(np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        background = np.nanmedian(stack, axis=0).astype(np.float32)
    observed_sigma = _robust_sigma(stack, axis=0)
    relative = float(args.background_relative_noise) * np.maximum(background, 0.0)
    model_floor = np.maximum(float(args.background_noise_floor_mm), relative)
    sigma = np.sqrt(
        np.square(np.where(np.isfinite(observed_sigma), observed_sigma, 0.0))
        + np.square(model_floor)
    ).astype(np.float32)
    valid = (
        np.isfinite(background)
        & (background > 1e-6)
        & (finite_count >= max(1, int(args.background_minimum_models)))
    )
    background[~valid] = np.nan
    sigma[~valid] = np.nan
    return (
        background,
        sigma,
        valid,
        {
            "models_used": used,
            "model_count": int(len(aligned)),
            "valid_pixels": int(np.count_nonzero(valid)),
            "background_depth_mm": finite_stats(background[valid]),
            "background_sigma_mm": finite_stats(sigma[valid]),
        },
    )


def _nanmedian_maps(arrays: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Combina mapas por mediana ignorando NaN; devuelve NaN si no hay entradas."""
    if not arrays:
        return np.full(shape, np.nan, np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.stack(arrays), axis=0).astype(np.float32)


def load_stereo_diagnostics_for_view(
    root: Path,
    view: dict,
    shape: Tuple[int, int],
    depth_source: str,
    regional_source: str,
) -> Tuple[dict, dict]:
    """Recupera evidencia estéreo de las sesiones que formaron el consenso."""
    confidence_maps: List[np.ndarray] = []
    lr_error_maps: List[np.ndarray] = []
    regional_residual_maps: List[np.ndarray] = []
    observed_maps: List[np.ndarray] = []
    available_maps: List[np.ndarray] = []
    sources_used = []

    for source in view.get("sources") or []:
        session = str(source.get("session", ""))
        stem = str(source.get("stem", ""))
        if not session or not stem:
            continue
        depth_dir = root / "reconstruccion" / session / depth_source
        regional_dir = root / "reconstruccion" / session / regional_source
        paths = {
            "confidence": depth_dir / f"{stem}_confidence.npy",
            "lr_error": depth_dir / f"{stem}_lr_error.npy",
            "regional_residual": regional_dir / f"{stem}_regional_residual.npy",
            "source_map": regional_dir / f"{stem}_depth_source_map.npy",
        }
        alignment = source.get("alignment") or {}
        dx = int(alignment.get("dx", 0) or 0)
        dy = int(alignment.get("dy", 0) or 0)
        loaded_any = False

        def load_shifted(path: Path, fill, dtype=np.float32):
            nonlocal loaded_any
            if not path.is_file():
                return None
            array = np.load(str(path)).astype(dtype)
            if array.shape != shape:
                raise ValueError(
                    f"Diagnóstico estéreo incompatible {path}: " f"{array.shape} != {shape}."
                )
            loaded_any = True
            return _shift_image(array, dy, dx, fill)

        confidence = load_shifted(paths["confidence"], np.nan)
        lr_error = load_shifted(paths["lr_error"], np.nan)
        regional_residual = load_shifted(paths["regional_residual"], np.nan)
        source_map = load_shifted(paths["source_map"], 0, np.uint8)
        if confidence is not None:
            confidence_maps.append(confidence)
        if lr_error is not None:
            lr_error_maps.append(np.abs(lr_error))
        if regional_residual is not None:
            regional_residual_maps.append(np.abs(regional_residual))
        if source_map is not None:
            # 1, 2, 5 y 6 conservan una medición física observada. Los códigos
            # 3, 4 y 7 son profundidad modelada/interpolada y no crean semillas.
            observed_maps.append(np.isin(source_map, (1, 2, 5, 6)))
            available_maps.append(source_map > 0)
        if loaded_any:
            sources_used.append({"session": session, "stem": stem, "dx": dx, "dy": dy})

    observed_count = (
        np.sum(np.stack(observed_maps), axis=0).astype(np.uint8)
        if observed_maps
        else np.zeros(shape, np.uint8)
    )
    diagnostic_count = (
        np.sum(np.stack(available_maps), axis=0).astype(np.uint8)
        if available_maps
        else np.zeros(shape, np.uint8)
    )
    maps = {
        "confidence": _nanmedian_maps(confidence_maps, shape),
        "lr_error_px": _nanmedian_maps(lr_error_maps, shape),
        "regional_residual_px": _nanmedian_maps(regional_residual_maps, shape),
        "observed_sessions": observed_count,
        "diagnostic_sessions": diagnostic_count,
    }
    return maps, {
        "sources_used": sources_used,
        "source_count": int(len(sources_used)),
        "confidence_available": bool(confidence_maps),
        "lr_error_available": bool(lr_error_maps),
        "regional_residual_available": bool(regional_residual_maps),
        "source_map_available": bool(available_maps),
    }


def classify_foreground(
    depth_mm: np.ndarray,
    spread_mm: np.ndarray,
    input_mask: np.ndarray,
    reliable_base: np.ndarray,
    point_confidence: np.ndarray,
    depth_support: np.ndarray,
    image_bgr: np.ndarray,
    background_mm: np.ndarray,
    background_sigma_mm: np.ndarray,
    background_valid: np.ndarray,
    stereo_diagnostics: dict,
    stereo_fb_px_mm: float,
    args,
) -> Tuple[np.ndarray, ...]:
    """Valida candidatos de fondo con evidencia estéreo independiente."""
    valid_depth = np.isfinite(depth_mm) & (depth_mm > 1e-6)
    comparable = valid_depth & background_valid & np.isfinite(background_sigma_mm)

    current_floor = np.maximum(
        float(args.background_noise_floor_mm),
        float(args.background_relative_noise) * np.maximum(depth_mm, 0.0),
    )
    # depth_spread es el rango intersesión; la mitad es una cota conservadora
    # para la desviación de una observación respecto al consenso.
    current_sigma = np.maximum(
        np.where(np.isfinite(spread_mm), 0.5 * np.maximum(spread_mm, 0.0), 0.0),
        current_floor,
    ).astype(np.float32)
    sigma_inverse = np.full(depth_mm.shape, np.nan, np.float32)
    residual_sigma = np.full(depth_mm.shape, np.nan, np.float32)
    if np.any(comparable):
        sigma_inverse[comparable] = np.sqrt(
            np.square(1000.0 * current_sigma[comparable] / np.square(depth_mm[comparable]))
            + np.square(
                1000.0 * background_sigma_mm[comparable] / np.square(background_mm[comparable])
            )
        )
        inverse_delta = 1000.0 / depth_mm[comparable] - 1000.0 / background_mm[comparable]
        residual_sigma[comparable] = (
            inverse_delta / np.maximum(sigma_inverse[comparable], 1e-8)
        ).astype(np.float32)

    weak_z = float(args.foreground_weak_z_score)
    strong_z = max(float(args.foreground_strong_z_score), weak_z + 1e-3)
    background_score = np.zeros(depth_mm.shape, np.float32)
    background_score[comparable] = np.clip(
        (residual_sigma[comparable] - weak_z) / (strong_z - weak_z),
        0.0,
        1.0,
    )

    fb = max(float(stereo_fb_px_mm), 1e-6)
    disparity = np.full(depth_mm.shape, np.nan, np.float32)
    disparity[valid_depth] = fb / depth_mm[valid_depth]
    disparity_floor = max(float(args.disparity_uncertainty_floor_px), 1e-3)
    disparity_sigma = np.full(depth_mm.shape, np.nan, np.float32)
    disparity_sigma[valid_depth] = np.sqrt(
        disparity_floor * disparity_floor
        + np.square(fb * current_sigma[valid_depth] / np.square(depth_mm[valid_depth]))
    ).astype(np.float32)

    diagnostic_confidence = np.asarray(stereo_diagnostics.get("confidence"), dtype=np.float32)
    confidence_score = np.where(
        np.isfinite(diagnostic_confidence),
        np.clip(diagnostic_confidence, 0.0, 1.0),
        np.clip(point_confidence, 0.0, 1.0),
    ).astype(np.float32)

    lr_error = np.asarray(stereo_diagnostics.get("lr_error_px"), dtype=np.float32)
    regional_residual = np.asarray(stereo_diagnostics.get("regional_residual_px"), dtype=np.float32)
    lr_scale = np.maximum(
        float(args.stereo_lr_sigma_factor) * disparity_sigma,
        1.0,
    )
    regional_scale = np.maximum(
        float(args.stereo_regional_sigma_factor) * disparity_sigma,
        0.75,
    )
    lr_score = np.where(
        np.isfinite(lr_error),
        np.exp(-0.5 * np.square(lr_error / lr_scale)),
        0.70,
    ).astype(np.float32)
    regional_score = np.where(
        np.isfinite(regional_residual),
        np.exp(-0.5 * np.square(regional_residual / regional_scale)),
        0.70,
    ).astype(np.float32)

    observed_sessions = np.asarray(stereo_diagnostics.get("observed_sessions"), dtype=np.uint8)
    diagnostic_sessions = np.asarray(stereo_diagnostics.get("diagnostic_sessions"), dtype=np.uint8)
    minimum_observed = max(1, int(args.stereo_minimum_observed_sessions))
    observed_score = np.where(
        diagnostic_sessions > 0,
        np.clip(observed_sessions.astype(np.float32) / minimum_observed, 0.0, 1.0),
        np.clip(depth_support.astype(np.float32) / minimum_observed, 0.0, 1.0),
    ).astype(np.float32)

    foreground_candidate = (
        reliable_base
        & (input_mask > 0)
        & (
            (comparable & (residual_sigma >= weak_z))
            | ((~background_valid) & (observed_score >= 1.0))
        )
    )

    consistent_neighbors = np.zeros(depth_mm.shape, np.uint8)
    available_neighbors = np.zeros(depth_mm.shape, np.uint8)
    local_z = max(float(args.stereo_local_neighbor_z_score), 1e-3)
    for dy, dx in (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ):
        neighbor_candidate = _shift_image(foreground_candidate, dy, dx, False)
        neighbor_disparity = _shift_image(disparity, dy, dx, np.nan)
        neighbor_sigma = _shift_image(disparity_sigma, dy, dx, np.nan)
        usable = foreground_candidate & neighbor_candidate & np.isfinite(neighbor_disparity)
        tolerance = local_z * np.sqrt(np.square(disparity_sigma) + np.square(neighbor_sigma))
        agreement = usable & (
            np.abs(disparity - neighbor_disparity) <= np.maximum(tolerance, disparity_floor)
        )
        available_neighbors += usable.astype(np.uint8)
        consistent_neighbors += agreement.astype(np.uint8)
    local_score = np.divide(
        consistent_neighbors.astype(np.float32),
        np.maximum(available_neighbors.astype(np.float32), 1.0),
    )

    # El gradiente de disparidad es una penalización suave. Los bordes visibles
    # en RGB relajan la penalización para conservar aristas y piezas delgadas.
    valid_for_gradient = foreground_candidate & np.isfinite(disparity)
    fill_value = (
        float(np.nanmedian(disparity[valid_for_gradient])) if np.any(valid_for_gradient) else 0.0
    )
    disparity_filled = np.where(valid_for_gradient, disparity, fill_value).astype(np.float32)
    disparity_gradient = (
        cv2.magnitude(
            cv2.Sobel(disparity_filled, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(disparity_filled, cv2.CV_32F, 0, 1, ksize=3),
        )
        / 8.0
    )
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    image_gradient = (
        cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        / 8.0
    )
    gradient_normalized = disparity_gradient / np.maximum(disparity_sigma, disparity_floor)
    allowed_gradient = 2.0 + float(args.stereo_gradient_edge_relaxation) * np.clip(
        image_gradient / 0.20, 0.0, 1.0
    )
    gradient_excess = np.maximum(gradient_normalized - allowed_gradient, 0.0)
    gradient_score = np.exp(-0.5 * np.square(gradient_excess / 3.0)).astype(np.float32)
    # No se penaliza el contorno únicamente por tener vecinos inválidos.
    gradient_interior = (
        cv2.erode(valid_for_gradient.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    )
    gradient_score[~gradient_interior] = 1.0

    stereo_score = (
        0.30 * confidence_score
        + 0.23 * lr_score
        + 0.17 * regional_score
        + 0.15 * observed_score
        + 0.15 * local_score
    ) * (0.75 + 0.25 * gradient_score)
    stereo_score = np.clip(stereo_score, 0.0, 1.0).astype(np.float32)

    observed_sufficient = ((diagnostic_sessions > 0) & (observed_sessions >= minimum_observed)) | (
        (diagnostic_sessions == 0) & (depth_support >= minimum_observed)
    )
    strong = (
        foreground_candidate
        & ((residual_sigma >= strong_z) | (~background_valid))
        & observed_sufficient
        & (stereo_score >= float(args.stereo_strong_score))
        & (consistent_neighbors >= max(1, int(args.stereo_minimum_consistent_neighbors)))
    )
    weak_allowed = (
        foreground_candidate
        & (stereo_score >= float(args.stereo_weak_score))
        & (consistent_neighbors >= 1)
    )

    # Cada componente se conserva por contener evidencia fuerte propia. No se
    # usa el tamaño relativo ni se fuerza una única componente principal.
    count, labels, component_stats, _ = cv2.connectedComponentsWithStats(
        weak_allowed.astype(np.uint8), connectivity=8
    )
    selected = np.zeros(depth_mm.shape, bool)
    component_id = np.zeros(depth_mm.shape, np.int32)
    decisions = []
    next_id = 1
    minimum_strong = max(1, int(args.foreground_component_minimum_strong_pixels))
    for label in range(1, count):
        component = labels == label
        strong_pixels = int(np.count_nonzero(component & strong))
        keep = strong_pixels >= minimum_strong
        if keep:
            selected[component] = True
            component_id[component] = next_id
            next_id += 1
        decisions.append(
            {
                "label": int(label),
                "pixels": int(component_stats[label, cv2.CC_STAT_AREA]),
                "strong_foreground_pixels": strong_pixels,
                "kept": bool(keep),
            }
        )

    # Recupera únicamente un borde estrecho de la máscara original alrededor
    # de una región ya validada; no deja que el fondo forme nuevas componentes.
    recovery_px = max(float(args.foreground_boundary_recovery_px), 0.0)
    recovered = np.zeros(depth_mm.shape, bool)
    if recovery_px > 0.0 and np.any(selected):
        distance = cv2.distanceTransform((~selected).astype(np.uint8), cv2.DIST_L2, 3)
        recovered = foreground_candidate & weak_allowed & (~selected) & (distance <= recovery_px)
        if np.any(recovered):
            selected |= recovered

    # Reetiquetar tras la recuperación evita depender de operaciones
    # morfológicas sobre enteros de 32 bits y deja una identidad coherente.
    final_count, final_labels = cv2.connectedComponents(selected.astype(np.uint8), connectivity=8)
    component_id = final_labels.astype(np.int32)

    foreground_score = np.clip(0.40 * background_score + 0.60 * stereo_score, 0.0, 1.0).astype(
        np.float32
    )
    foreground_score[~selected] = 0.0
    # Propagación Jacobiana inversa: sigma_Z = Z²/(fB) * sigma_d.
    depth_uncertainty = np.full(depth_mm.shape, np.nan, np.float32)
    depth_uncertainty[valid_depth] = np.sqrt(
        np.square(current_sigma[valid_depth])
        + np.square(np.square(depth_mm[valid_depth]) / fb * disparity_floor)
    ).astype(np.float32)

    return (
        selected,
        foreground_score,
        residual_sigma,
        depth_uncertainty,
        component_id,
        stereo_score,
        disparity_sigma,
        observed_sessions,
        {
            "method": (
                "background_candidate_then_stereo_uncertainty_" "and_observed_seed_validation"
            ),
            "weak_z_score": weak_z,
            "strong_z_score": strong_z,
            "stereo_weak_score": float(args.stereo_weak_score),
            "stereo_strong_score": float(args.stereo_strong_score),
            "comparable_pixels": int(np.count_nonzero(comparable & (input_mask > 0))),
            "foreground_candidate_pixels": int(np.count_nonzero(foreground_candidate)),
            "strong_pixels": int(np.count_nonzero(strong)),
            "selected_pixels": int(np.count_nonzero(selected)),
            "recovered_boundary_pixels": int(np.count_nonzero(recovered)),
            "component_count": int(count - 1),
            "kept_component_count": int(final_count - 1),
            "components": decisions,
            "residual_sigma": finite_stats(residual_sigma[comparable & (input_mask > 0)]),
            "current_depth_sigma_mm": finite_stats(current_sigma[valid_depth]),
            "disparity_sigma_px": finite_stats(disparity_sigma[selected]),
            "stereo_score": finite_stats(stereo_score[selected]),
            "observed_sessions": finite_stats(observed_sessions[selected]),
            "consistent_neighbors": finite_stats(consistent_neighbors[selected]),
            "gradient_score": finite_stats(gradient_score[selected]),
        },
    )


def regularize_inverse_depth(
    depth_mm: np.ndarray,
    image_bgr: np.ndarray,
    domain: np.ndarray,
    confidence: np.ndarray,
    args,
) -> Tuple[np.ndarray, dict]:
    """Difusión anisotrópica en 1/Z, anclada a la medición original."""
    valid = np.asarray(domain, bool) & np.isfinite(depth_mm) & (depth_mm > 1e-6)
    output = np.full(depth_mm.shape, np.nan, np.float32)
    output[valid] = depth_mm[valid].astype(np.float32)
    if not bool(args.inverse_depth_regularization) or not np.any(valid):
        return output, {
            "enabled": bool(args.inverse_depth_regularization),
            "iterations": 0,
            "valid_pixels": int(np.count_nonzero(valid)),
            "displacement_mm": finite_stats(np.zeros(np.count_nonzero(valid))),
        }

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    conf = np.where(valid & np.isfinite(confidence), np.clip(confidence, 0.0, 1.0), 0.0).astype(
        np.float32
    )
    inverse_original = np.zeros(depth_mm.shape, np.float32)
    inverse_original[valid] = 1000.0 / depth_mm[valid]
    inverse_current = inverse_original.copy()
    anchor = float(args.inverse_depth_data_strength) * (0.20 + 0.80 * conf)
    depth_sigma = max(float(args.inverse_depth_edge_sigma_mm), 1e-6)
    color_sigma = max(float(args.inverse_depth_color_sigma), 1e-6)
    smoothness = max(float(args.inverse_depth_smoothness), 0.0)
    maximum_move = max(float(args.inverse_depth_maximum_displacement_mm), 0.0)
    offsets = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 0.7071),
        (-1, 1, 0.7071),
        (1, -1, 0.7071),
        (1, 1, 0.7071),
    ]
    neighbor_models = []
    for dy, dx, distance_weight in offsets:
        pair_valid = valid & _shift_image(valid, dy, dx, False)
        neighbor_depth = _shift_image(depth_mm, dy, dx, np.nan)
        neighbor_gray = _shift_image(gray, dy, dx, np.nan)
        neighbor_conf = _shift_image(conf, dy, dx, 0.0)
        weight = (
            float(distance_weight)
            * np.exp(-0.5 * np.square(np.abs(depth_mm - neighbor_depth) / depth_sigma))
            * np.exp(-0.5 * np.square(np.abs(gray - neighbor_gray) / color_sigma))
            * np.sqrt(np.maximum(conf * neighbor_conf, 0.0))
        )
        weight = np.where(pair_valid & np.isfinite(weight), weight, 0.0).astype(np.float32)
        neighbor_models.append((dy, dx, weight))

    completed = 0
    for _ in range(max(0, int(args.inverse_depth_iterations))):
        numerator = anchor * inverse_original
        denominator = anchor.copy()
        for dy, dx, weight in neighbor_models:
            neighbor_inverse = _shift_image(inverse_current, dy, dx, 0.0)
            numerator += smoothness * weight * neighbor_inverse
            denominator += smoothness * weight
        update = valid & (denominator > 1e-8)
        candidate_inverse = inverse_current.copy()
        candidate_inverse[update] = numerator[update] / denominator[update]
        candidate_depth = np.full(depth_mm.shape, np.nan, np.float32)
        positive = update & (candidate_inverse > 1e-8)
        candidate_depth[positive] = 1000.0 / candidate_inverse[positive]
        if maximum_move > 0.0:
            candidate_depth[positive] = np.clip(
                candidate_depth[positive],
                depth_mm[positive] - maximum_move,
                depth_mm[positive] + maximum_move,
            )
        inverse_current[positive] = 1000.0 / candidate_depth[positive]
        completed += 1

    output[valid] = 1000.0 / np.maximum(inverse_current[valid], 1e-8)
    displacement = np.abs(output[valid] - depth_mm[valid])
    return output.astype(np.float32), {
        "enabled": True,
        "domain": "inverse_depth",
        "iterations": int(completed),
        "valid_pixels": int(np.count_nonzero(valid)),
        "edge_sigma_mm": float(depth_sigma),
        "color_sigma_normalized": float(color_sigma),
        "maximum_displacement_mm": float(maximum_move),
        "displacement_mm": finite_stats(displacement),
    }


def backproject(
    depth: np.ndarray,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    confidence: np.ndarray,
    spread: np.ndarray,
    depth_support: np.ndarray,
    agreement_support: np.ndarray,
    foreground_score: np.ndarray,
    background_residual_sigma: np.ndarray,
    background_valid: np.ndarray,
    depth_uncertainty_mm: np.ndarray,
    component_id: np.ndarray,
    stereo_score: np.ndarray,
    disparity_uncertainty_px: np.ndarray,
    observed_sessions: np.ndarray,
    intrinsics: Dict[str, float],
    pixel_step: int,
):
    """Retroproyecta los píxeles válidos y conserva los atributos de cada muestra."""
    h, w = depth.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    step = max(1, int(pixel_step))
    z = depth[::step, ::step]
    selected = (mask[::step, ::step] > 0) & np.isfinite(z)
    colors = image_bgr[::step, ::step]
    x_pixels = xx[::step, ::step]
    y_pixels = yy[::step, ::step]
    z_values = z[selected].astype(np.float32)
    x_values = (x_pixels[selected] - intrinsics["cx"]) * z_values / intrinsics["fx"]
    y_values = (y_pixels[selected] - intrinsics["cy"]) * z_values / intrinsics["fy"]
    points = np.column_stack((x_values, y_values, z_values)).astype(np.float32)
    colors_rgb = colors[selected][:, ::-1].astype(np.uint8)
    return {
        "points": points,
        "colors": colors_rgb,
        "pixel_uv": np.column_stack((x_pixels[selected], y_pixels[selected])).astype(np.float32),
        "confidence": confidence[::step, ::step][selected].astype(np.float32),
        "depth_spread_mm": spread[::step, ::step][selected].astype(np.float32),
        "depth_support": depth_support[::step, ::step][selected].astype(np.uint8),
        "agreement_support": agreement_support[::step, ::step][selected].astype(np.uint8),
        "foreground_score": foreground_score[::step, ::step][selected].astype(np.float32),
        "background_residual_sigma": background_residual_sigma[::step, ::step][selected].astype(
            np.float32
        ),
        "background_valid": background_valid[::step, ::step][selected].astype(np.uint8),
        "depth_uncertainty_mm": depth_uncertainty_mm[::step, ::step][selected].astype(np.float32),
        "component_id": component_id[::step, ::step][selected].astype(np.int32),
        "stereo_score": stereo_score[::step, ::step][selected].astype(np.float32),
        "disparity_uncertainty_px": disparity_uncertainty_px[::step, ::step][selected].astype(
            np.float32
        ),
        "observed_sessions": observed_sessions[::step, ::step][selected].astype(np.uint8),
    }


def _take_samples(samples: dict, indexes: np.ndarray) -> dict:
    """Aplica la misma selección de índices a todos los atributos de las muestras."""
    indexes = np.asarray(indexes, dtype=np.int64)
    return {key: np.asarray(value)[indexes] for key, value in samples.items()}


def _weighted_average(values, inverse, count, weights):
    """Agrega los valores por grupo mediante una media ponderada."""
    weights = np.asarray(weights, dtype=np.float64)
    denominator = np.bincount(inverse, weights=weights, minlength=count)
    if values.ndim == 1:
        numerator = np.bincount(
            inverse,
            weights=weights * values.astype(np.float64),
            minlength=count,
        )
        return numerator / np.maximum(denominator, 1e-12)
    result = np.zeros((count, values.shape[1]), np.float64)
    for axis in range(values.shape[1]):
        result[:, axis] = np.bincount(
            inverse,
            weights=weights * values[:, axis].astype(np.float64),
            minlength=count,
        )
    return result / np.maximum(denominator[:, None], 1e-12)


def voxel_downsample(samples: dict, voxel: float, spread_reference: float):
    """Reduce las muestras por vóxel conservando sus atributos de confianza."""
    points = samples["points"]
    if len(points) == 0 or voxel <= 0:
        return samples
    keys = np.floor(points / float(voxel)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = int(np.max(inverse)) + 1
    confidence = np.clip(samples["confidence"].astype(np.float64), 0.0, 1.0)
    spread = np.maximum(samples["depth_spread_mm"].astype(np.float64), 0.0)
    support = np.maximum(samples["depth_support"].astype(np.float64), 1.0)
    agreement = np.minimum(samples["agreement_support"].astype(np.float64), support)
    spread_score = np.exp(-0.5 * np.square(spread / max(float(spread_reference), 1e-6)))
    agreement_score = 0.55 + 0.45 * np.clip(agreement / support, 0.0, 1.0)
    weights = np.clip(confidence * spread_score * agreement_score, 0.02, 1.0)

    # El soporte se agrega de forma conservadora; usar el máximo convertía una
    # sola muestra favorable en propiedad de todo el voxel.
    mean_support = np.rint(
        _weighted_average(samples["depth_support"].astype(np.float64), inverse, count, weights)
    )
    mean_agreement = np.rint(
        _weighted_average(samples["agreement_support"].astype(np.float64), inverse, count, weights)
    )
    background_fraction = _weighted_average(
        samples["background_valid"].astype(np.float64), inverse, count, weights
    )

    # Identidad de componente por voto ponderado dentro del voxel.
    component_id = np.zeros(count, np.int32)
    component_vote = np.zeros(count, np.float64)
    source_component = samples["component_id"].astype(np.int32)
    for label in np.unique(source_component[source_component > 0]):
        votes = np.bincount(
            inverse,
            weights=weights * (source_component == label),
            minlength=count,
        )
        better = votes > component_vote
        component_id[better] = int(label)
        component_vote[better] = votes[better]
    return {
        "points": _weighted_average(points, inverse, count, weights).astype(np.float32),
        "colors": np.clip(
            np.rint(_weighted_average(samples["colors"], inverse, count, weights)),
            0,
            255,
        ).astype(np.uint8),
        "confidence": np.clip(
            _weighted_average(confidence, inverse, count, weights), 0.0, 1.0
        ).astype(np.float32),
        "depth_spread_mm": np.maximum(
            _weighted_average(spread, inverse, count, weights), 0.0
        ).astype(np.float32),
        "depth_support": np.clip(mean_support, 0, 255).astype(np.uint8),
        "agreement_support": np.clip(mean_agreement, 0, 255).astype(np.uint8),
        "pixel_uv": _weighted_average(
            samples["pixel_uv"].astype(np.float64), inverse, count, weights
        ).astype(np.float32),
        "foreground_score": np.clip(
            _weighted_average(
                samples["foreground_score"].astype(np.float64), inverse, count, weights
            ),
            0.0,
            1.0,
        ).astype(np.float32),
        "background_residual_sigma": _weighted_average(
            np.nan_to_num(
                samples["background_residual_sigma"].astype(np.float64),
                nan=0.0,
            ),
            inverse,
            count,
            weights * np.isfinite(samples["background_residual_sigma"]),
        ).astype(np.float32),
        "background_valid": (background_fraction >= 0.50).astype(np.uint8),
        "depth_uncertainty_mm": np.maximum(
            _weighted_average(
                samples["depth_uncertainty_mm"].astype(np.float64),
                inverse,
                count,
                weights,
            ),
            1e-3,
        ).astype(np.float32),
        "component_id": component_id,
        "stereo_score": np.clip(
            _weighted_average(samples["stereo_score"].astype(np.float64), inverse, count, weights),
            0.0,
            1.0,
        ).astype(np.float32),
        "disparity_uncertainty_px": np.maximum(
            _weighted_average(
                samples["disparity_uncertainty_px"].astype(np.float64),
                inverse,
                count,
                weights,
            ),
            1e-3,
        ).astype(np.float32),
        "observed_sessions": np.clip(
            np.rint(
                _weighted_average(
                    samples["observed_sessions"].astype(np.float64), inverse, count, weights
                )
            ),
            0,
            255,
        ).astype(np.uint8),
    }


def make_cloud(points: np.ndarray, colors: np.ndarray):
    """Construye una nube Open3D con colores RGB normalizados desde el rango 0..255."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    return cloud


def estimate_normals(points: np.ndarray, colors: np.ndarray, args) -> np.ndarray:
    """Estima normales para las muestras cuando existe un número suficiente de puntos."""
    if len(points) < 10:
        return np.empty((0, 3), dtype=np.float32)
    cloud = make_cloud(points, colors)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(args.normal_radius_mm),
            max_nn=int(args.normal_max_nn),
        )
    )
    cloud.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
    normals = np.asarray(cloud.normals, dtype=np.float64)
    length = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(length) & (length > 1e-12)
    normals[valid] /= length[valid, None]
    normals[~valid] = 0.0
    return normals.astype(np.float32)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Calcula la mediana de los valores ordenados según sus pesos no negativos."""
    order = np.argsort(values)
    values = values[order]
    weights = np.maximum(weights[order], 0.0)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return float(np.median(values))
    index = int(np.searchsorted(np.cumsum(weights), 0.5 * total, side="left"))
    return float(values[min(index, len(values) - 1)])


def local_surface_residual(
    points: np.ndarray,
    normals: np.ndarray,
    confidence: np.ndarray,
    args,
) -> np.ndarray:
    """Estima el residuo local respecto a la superficie descrita por las normales."""
    if len(points) < 3 or len(normals) != len(points):
        return np.full(len(points), np.nan, np.float32)
    count = min(max(3, int(args.surface_smoothing_max_neighbors)), len(points))
    tree = cKDTree(points)
    distance, index = tree.query(
        points,
        k=count,
        distance_upper_bound=float(args.surface_smoothing_radius_mm),
        workers=query_threads(),
    )
    if distance.ndim == 1:
        distance = distance[:, None]
        index = index[:, None]
    cos_limit = math.cos(math.radians(float(args.surface_smoothing_normal_angle_deg)))
    residual = np.full(len(points), np.nan, np.float32)
    for row in range(len(points)):
        usable = np.isfinite(distance[row]) & (index[row] < len(points))
        ids = index[row, usable].astype(np.int64)
        ids = ids[ids != row]
        if len(ids) < int(args.surface_smoothing_min_neighbors):
            continue
        compatible = np.sum(normals[ids] * normals[row], axis=1) >= cos_limit
        ids = ids[compatible]
        if len(ids) < int(args.surface_smoothing_min_neighbors):
            continue
        signed = (points[ids] - points[row]) @ normals[row]
        weights = np.clip(confidence[ids], 0.05, 1.0)
        residual[row] = weighted_median(np.abs(signed), weights)
    return residual


def smooth_surface(samples: dict, normals: np.ndarray, args) -> Tuple[dict, dict]:
    """Regulariza posiciones con confianza y límites locales de desplazamiento."""
    points = samples["points"].astype(np.float64).copy()
    confidence = np.clip(samples["confidence"].astype(np.float64), 0.0, 1.0)
    if not bool(args.surface_smoothing) or len(points) < 10 or len(normals) != len(points):
        residual = local_surface_residual(points, normals, confidence, args)
        return samples, {
            "enabled": False,
            "iterations": 0,
            "displacement_mm": finite_stats(np.zeros(len(points))),
            "local_residual_mm": finite_stats(residual),
        }

    original = points.copy()
    radius = max(float(args.surface_smoothing_radius_mm), 1e-6)
    sigma_plane = max(float(args.surface_smoothing_plane_sigma_mm), 1e-6)
    maximum_neighbors = max(3, int(args.surface_smoothing_max_neighbors))
    minimum_neighbors = max(3, int(args.surface_smoothing_min_neighbors))
    cos_limit = math.cos(math.radians(float(args.surface_smoothing_normal_angle_deg)))
    strength = float(np.clip(args.surface_smoothing_strength, 0.0, 1.0))
    max_step = max(float(args.surface_smoothing_max_step_mm), 0.0)

    for iteration in range(max(0, int(args.surface_smoothing_iterations))):
        tree = cKDTree(points)
        count = min(maximum_neighbors, len(points))
        distances, indexes = tree.query(
            points,
            k=count,
            distance_upper_bound=radius,
            workers=query_threads(),
        )
        if distances.ndim == 1:
            distances = distances[:, None]
            indexes = indexes[:, None]
        update = np.zeros(len(points), np.float64)
        for row in range(len(points)):
            usable = np.isfinite(distances[row]) & (indexes[row] < len(points))
            ids = indexes[row, usable].astype(np.int64)
            dist = distances[row, usable]
            nonself = ids != row
            ids, dist = ids[nonself], dist[nonself]
            if len(ids) < minimum_neighbors:
                continue
            normal_dot = np.sum(normals[ids] * normals[row], axis=1)
            compatible = normal_dot >= cos_limit
            ids, dist = ids[compatible], dist[compatible]
            if len(ids) < minimum_neighbors:
                continue
            signed = (points[ids] - points[row]) @ normals[row]
            spatial_weight = np.exp(-0.5 * np.square(dist / radius))
            plane_weight = np.exp(-0.5 * np.square(signed / sigma_plane))
            weights = spatial_weight * plane_weight * np.clip(confidence[ids], 0.05, 1.0)
            target = weighted_median(signed, weights)
            reliability_factor = 0.55 + 0.45 * (1.0 - confidence[row])
            update[row] = np.clip(
                strength * reliability_factor * target,
                -max_step,
                max_step,
            )
        points += update[:, None] * normals
        if iteration + 1 < int(args.surface_smoothing_iterations):
            refreshed = estimate_normals(points, samples["colors"], args)
            if len(refreshed) == len(points):
                normals = refreshed

    result = dict(samples)
    result["points"] = points.astype(np.float32)
    residual = local_surface_residual(points, normals, confidence, args)
    displacement = np.linalg.norm(points - original, axis=1)
    return result, {
        "enabled": True,
        "iterations": max(0, int(args.surface_smoothing_iterations)),
        "displacement_mm": finite_stats(displacement),
        "local_residual_mm": finite_stats(residual),
        "residual_array": residual,
    }


@operacion("Limpiar nube y estimar normales")
def clean_cloud(samples: dict, args):
    """Aplica los filtros de nube configurados y conserva estadísticas de limpieza."""
    points = samples["points"]
    colors = samples["colors"]
    stats = {
        "after_voxel": int(len(points)),
        "after_statistical": 0,
        "after_radius": 0,
        "after_dbscan": 0,
        "cluster_count": 0,
        "main_cluster_ratio": 0.0,
        "components_kept_by_foreground_evidence": 0,
        "components_rejected_without_foreground_evidence": 0,
        "component_decisions": [],
    }
    if len(points) == 0:
        samples["local_residual_mm"] = np.empty(0, np.float32)
        return samples, np.empty((0, 3), np.float32), stats

    cloud = make_cloud(points, colors)
    _, indexes = cloud.remove_statistical_outlier(
        nb_neighbors=args.sor_neighbors,
        std_ratio=args.sor_std_ratio,
    )
    indexes = np.asarray(indexes, dtype=np.int64)
    samples = _take_samples(samples, indexes)
    points, colors = samples["points"], samples["colors"]
    stats["after_statistical"] = int(len(points))
    if len(points) == 0:
        samples["local_residual_mm"] = np.empty(0, np.float32)
        return samples, np.empty((0, 3), np.float32), stats

    cloud = make_cloud(points, colors)
    _, indexes = cloud.remove_radius_outlier(
        nb_points=args.radius_nb_points,
        radius=args.radius_mm,
    )
    indexes = np.asarray(indexes, dtype=np.int64)
    samples = _take_samples(samples, indexes)
    points, colors = samples["points"], samples["colors"]
    stats["after_radius"] = int(len(points))
    if len(points) == 0:
        samples["local_residual_mm"] = np.empty(0, np.float32)
        return samples, np.empty((0, 3), np.float32), stats

    cloud = make_cloud(points, colors)
    labels = np.asarray(
        cloud.cluster_dbscan(
            eps=args.dbscan_eps_mm,
            min_points=args.dbscan_min_points,
            print_progress=False,
        ),
        dtype=np.int32,
    )
    valid = labels >= 0
    if np.any(valid):
        unique, counts = np.unique(labels[valid], return_counts=True)
        order = np.argsort(counts)[::-1]
        unique, counts = unique[order], counts[order]
        stats["cluster_count"] = int(len(unique))
        stats["main_cluster_ratio"] = float(counts[0] / max(len(points), 1))
        keep_labels = []
        for label, count in zip(unique, counts):
            member = labels == label
            foreground = np.clip(
                samples["foreground_score"][member].astype(np.float64),
                0.0,
                1.0,
            )
            stereo = np.clip(
                samples["stereo_score"][member].astype(np.float64),
                0.0,
                1.0,
            )
            observed = samples["observed_sessions"][member].astype(np.int32)
            validated_component = samples["component_id"][member] > 0
            p90_foreground = float(np.percentile(foreground, 90))
            median_foreground = float(np.median(foreground))
            # La componente ya fue validada en 2D por semillas observadas.
            # P95 evita que una pieza legítima con mucha frontera interpolada
            # sea descartada porque su mediana o P90 son modestos.
            p95_stereo = float(np.percentile(stereo, 95))
            observed_evidence_points = int(
                np.count_nonzero(observed >= max(1, int(args.stereo_minimum_observed_sessions)))
            )
            evidence = bool(
                np.any(validated_component)
                and observed_evidence_points > 0
                and p95_stereo >= 0.85 * float(args.stereo_strong_score)
            )
            if evidence:
                keep_labels.append(int(label))
                stats["components_kept_by_foreground_evidence"] += 1
            else:
                stats["components_rejected_without_foreground_evidence"] += 1
            stats["component_decisions"].append(
                {
                    "label": int(label),
                    "points": int(count),
                    "median_foreground_score": median_foreground,
                    "p90_foreground_score": p90_foreground,
                    "p95_stereo_score": p95_stereo,
                    "observed_evidence_points": observed_evidence_points,
                    "validated_source_component": bool(np.any(validated_component)),
                    "kept": evidence,
                }
            )
        keep = np.isin(labels, keep_labels)
        samples = _take_samples(samples, np.flatnonzero(keep))
        points, colors = samples["points"], samples["colors"]
    else:
        # Si DBSCAN no encontró ninguna componente, todos los puntos son ruido;
        # conservarlos sería exactamente el fallback que queremos evitar.
        stats["components_rejected_without_foreground_evidence"] = 1
        samples = _take_samples(samples, np.empty(0, dtype=np.int64))
        points, colors = samples["points"], samples["colors"]
    stats["after_dbscan"] = int(len(points))

    normals = estimate_normals(points, colors, args)
    samples, smoothing_stats = smooth_surface(samples, normals, args)
    points, colors = samples["points"], samples["colors"]
    residual = smoothing_stats.pop("residual_array", None)
    if residual is None:
        residual = local_surface_residual(points, normals, samples["confidence"], args)
    residual_score = np.where(
        np.isfinite(residual),
        np.exp(-0.5 * np.square(residual / max(float(args.local_residual_reference_mm), 1e-6))),
        0.70,
    )
    samples["confidence"] = np.clip(
        samples["confidence"] * (0.60 + 0.40 * residual_score),
        0.0,
        1.0,
    ).astype(np.float32)
    samples["local_residual_mm"] = residual.astype(np.float32)
    normals = estimate_normals(points, colors, args)
    stats["surface_smoothing"] = smoothing_stats
    return samples, normals, stats


def geometry_stats(points: np.ndarray) -> dict:
    """Resume cantidad, posición y extensión de la nube resultante."""
    if len(points) == 0:
        return {
            "count": 0,
            "extent_xyz_mm": None,
            "centroid_xyz_mm": None,
            "z": finite_stats(np.array([])),
        }
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    return {
        "count": int(len(points)),
        "minimum_xyz_mm": minimum.astype(float).tolist(),
        "maximum_xyz_mm": maximum.astype(float).tolist(),
        "extent_xyz_mm": (maximum - minimum).astype(float).tolist(),
        "centroid_xyz_mm": np.mean(points, axis=0).astype(float).tolist(),
        "z": finite_stats(points[:, 2]),
    }


def load_optional_array(
    path: Path,
    shape: Tuple[int, int],
    fallback,
    dtype,
) -> np.ndarray:
    """Carga un mapa opcional, verifica su forma y aplica el respaldo si falta."""
    if path.is_file():
        array = np.load(str(path)).astype(dtype)
        if array.shape != shape:
            raise ValueError(f"Dimensiones incompatibles en {path}: {array.shape} != {shape}")
        return array
    if np.isscalar(fallback):
        return np.full(shape, fallback, dtype=dtype)
    array = np.asarray(fallback, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"Fallback incompatible para {path}: {array.shape} != {shape}")
    return array


def _procesar_unidad_independiente(task):
    """Genera la nube de una pose con los mapas y el contexto del trabajador."""
    ctx = contexto()
    args = ctx["args"]
    background_depths = ctx["background_depths"]
    consensus_dir = ctx["consensus_dir"]
    depth_diagnostics_source = ctx["depth_diagnostics_source"]
    intrinsics = ctx["intrinsics"]
    output_dir = ctx["output_dir"]
    regional_diagnostics_source = ctx["regional_diagnostics_source"]
    root = ctx["root"]
    stereo_fb_px_mm = ctx["stereo_fb_px_mm"]
    views = ctx["views"]
    print("[PROGRESO] Validar profundidad y crear nube por pose: iniciando unidad", flush=True)
    records = []
    previews = []
    for index, view in [task]:
        stem = str(view["stem"])
        pose_index = int(view["pose_index"])
        physical_angle = float(view["physical_angle_deg"])
        image = cv2.imread(str(consensus_dir / f"{stem}_rect_L_consensus.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(
            str(consensus_dir / f"{stem}_cloud_mask_consensus.png"), cv2.IMREAD_GRAYSCALE
        )
        depth_path = consensus_dir / f"{stem}_depth_consensus_mm.npy"
        if image is None or mask is None or (not depth_path.is_file()):
            print(f"[ERROR] Entradas incompletas para P{pose_index:02d}")
            continue
        depth = np.load(str(depth_path)).astype(np.float32)
        if not image.shape[:2] == mask.shape == depth.shape:
            raise ValueError(f"Dimensiones incompatibles en {stem}")
        shape = depth.shape
        confidence = load_optional_array(
            consensus_dir / f"{stem}_confidence_consensus.npy", shape, 0.5, np.float32
        )
        spread = load_optional_array(
            consensus_dir / f"{stem}_depth_spread_mm.npy",
            shape,
            float(args.depth_spread_reference_mm),
            np.float32,
        )
        default_support = np.where(mask > 0, 2, 0).astype(np.uint8)
        depth_support = load_optional_array(
            consensus_dir / f"{stem}_depth_support.npy", shape, default_support, np.uint8
        )
        agreement_support = load_optional_array(
            consensus_dir / f"{stem}_agreement_support.npy", shape, depth_support, np.uint8
        )
        confidence = np.where(np.isfinite(confidence), np.clip(confidence, 0.0, 1.0), 0.0).astype(
            np.float32
        )
        spread = np.where(np.isfinite(spread), np.maximum(spread, 0.0), np.inf)
        support_denominator = np.maximum(depth_support.astype(np.float32), 1.0)
        agreement_ratio = np.clip(
            agreement_support.astype(np.float32) / support_denominator, 0.0, 1.0
        )
        spread_score = np.exp(
            -0.5
            * np.square(
                np.minimum(spread, 4.0 * float(args.depth_spread_reference_mm))
                / max(float(args.depth_spread_reference_mm), 1e-06)
            )
        )
        point_confidence = np.clip(
            confidence * (0.6 + 0.4 * agreement_ratio) * (0.55 + 0.45 * spread_score), 0.0, 1.0
        ).astype(np.float32)
        reliable = (
            (mask > 0)
            & np.isfinite(depth)
            & (depth_support >= int(args.minimum_depth_support))
            & (spread <= float(args.maximum_depth_spread_mm))
            & (point_confidence >= float(args.minimum_point_confidence))
        )
        background_depth, background_sigma, background_valid, background_stats = (
            aligned_background_model(view, background_depths, shape, args)
        )
        stereo_diagnostics, stereo_diagnostics_stats = load_stereo_diagnostics_for_view(
            root, view, shape, depth_diagnostics_source, regional_diagnostics_source
        )
        required_diagnostics = (
            stereo_diagnostics_stats["confidence_available"]
            and stereo_diagnostics_stats["lr_error_available"]
            and stereo_diagnostics_stats["source_map_available"]
        )
        if not required_diagnostics:
            raise RuntimeError(
                f"P{pose_index:02d}: faltan diagnósticos físicos de los pasos 02/04 (confidence, lr_error o depth_source_map). No se usará un fallback que vuelva a aceptar la máscara casi completa."
            )
        (
            validated_foreground,
            foreground_score,
            background_residual_sigma,
            depth_uncertainty_mm,
            foreground_component_id,
            stereo_score,
            disparity_uncertainty_px,
            observed_sessions,
            foreground_stats,
        ) = classify_foreground(
            depth,
            spread,
            mask,
            reliable,
            point_confidence,
            depth_support,
            image,
            background_depth,
            background_sigma,
            background_valid,
            stereo_diagnostics,
            stereo_fb_px_mm,
            args,
        )
        regularization_domain = validated_foreground.copy()
        regularized_depth, regularization_stats = regularize_inverse_depth(
            depth, image, regularization_domain, point_confidence, args
        )
        reliable = validated_foreground & np.isfinite(regularized_depth)
        reliable_mask = reliable.astype(np.uint8) * 255
        samples_raw = backproject(
            regularized_depth,
            image,
            reliable_mask,
            point_confidence,
            spread,
            depth_support,
            agreement_support,
            foreground_score,
            background_residual_sigma,
            background_valid,
            depth_uncertainty_mm,
            foreground_component_id,
            stereo_score,
            disparity_uncertainty_px,
            observed_sessions,
            intrinsics,
            args.pixel_step,
        )
        samples_voxel = voxel_downsample(samples_raw, args.voxel_mm, args.depth_spread_reference_mm)
        samples, normals, filter_stats = clean_cloud(samples_voxel, args)
        points = samples["points"]
        colors = samples["colors"]
        reasons = []
        quality = "accepted" if view["quality"] == "accepted" else "warning"
        if view["quality"] == "warning":
            reasons.append("El consenso 05 marcó esta pose como warning.")
        if len(points) == 0:
            quality = "rejected"
            reasons.append("La nube de consenso quedó vacía.")
        elif len(points) < args.minimum_final_points:
            quality = "warning"
            reasons.append(f"Pocos puntos finales: {len(points)}.")
        if (
            len(points) > 0
            and filter_stats["cluster_count"] > 0
            and (filter_stats["components_kept_by_foreground_evidence"] == 0)
        ):
            quality = "warning"
            reasons.append("Ninguna componente 3D conservó evidencia explícita de primer plano.")
        ply_path = output_dir / f"{stem}_cloud_final.ply"
        npz_path = output_dir / f"{stem}_cloud_final.npz"
        preview_path = output_dir / f"{stem}_cloud_final_preview.png"
        stats_path = output_dir / f"{stem}_cloud_final_stats.json"
        regularized_depth_path = output_dir / f"{stem}_depth_regularized_mm.npy"
        foreground_mask_path = output_dir / f"{stem}_foreground_validated.png"
        np.save(regularized_depth_path, regularized_depth.astype(np.float32))
        cv2.imwrite(str(foreground_mask_path), reliable_mask)
        save_ply_ascii(ply_path, points, colors, normals)
        np.savez_compressed(
            npz_path,
            points=points.astype(np.float32),
            colors=colors.astype(np.uint8),
            normals=normals.astype(np.float32),
            confidence=samples["confidence"].astype(np.float32),
            depth_spread_mm=samples["depth_spread_mm"].astype(np.float32),
            depth_support=samples["depth_support"].astype(np.uint8),
            agreement_support=samples["agreement_support"].astype(np.uint8),
            foreground_score=samples["foreground_score"].astype(np.float32),
            background_residual_sigma=samples["background_residual_sigma"].astype(np.float32),
            background_valid=samples["background_valid"].astype(np.uint8),
            depth_uncertainty_mm=samples["depth_uncertainty_mm"].astype(np.float32),
            component_id=samples["component_id"].astype(np.int32),
            stereo_score=samples["stereo_score"].astype(np.float32),
            disparity_uncertainty_px=samples["disparity_uncertainty_px"].astype(np.float32),
            observed_sessions=samples["observed_sessions"].astype(np.uint8),
            pixel_uv=samples["pixel_uv"].astype(np.float32),
            image_shape_hw=np.asarray(shape, dtype=np.int32),
            local_residual_mm=samples["local_residual_mm"].astype(np.float32),
            pose_index=np.asarray([pose_index], dtype=np.int32),
            physical_angle_deg=np.asarray([physical_angle], dtype=np.float64),
        )
        make_cloud_preview(
            points, colors, f"P{pose_index:02d} | {physical_angle:.1f}° | {quality}", preview_path
        )
        record = {
            "stem": stem,
            "pose_index": pose_index,
            "angle_deg": physical_angle,
            "physical_angle_deg": physical_angle,
            "quality": quality,
            "reasons": reasons,
            "source_quality": view["quality"],
            "mask_points_before_reliability": int(np.count_nonzero(mask)),
            "reliable_depth_pixels": int(np.count_nonzero(reliable)),
            "reliable_ratio_of_input_mask": float(
                np.count_nonzero(reliable) / max(np.count_nonzero(mask), 1)
            ),
            "background_model": background_stats,
            "stereo_diagnostics": stereo_diagnostics_stats,
            "foreground_classification": foreground_stats,
            "raw_points": int(len(samples_raw["points"])),
            "inverse_depth_regularization": regularization_stats,
            **filter_stats,
            "final_points": int(len(points)),
            "point_confidence": finite_stats(samples["confidence"]),
            "depth_spread_mm": finite_stats(samples["depth_spread_mm"]),
            "depth_support": finite_stats(samples["depth_support"]),
            "agreement_support": finite_stats(samples["agreement_support"]),
            "local_residual_mm": finite_stats(samples["local_residual_mm"]),
            "foreground_score": finite_stats(samples["foreground_score"]),
            "background_residual_sigma": finite_stats(samples["background_residual_sigma"]),
            "stereo_score": finite_stats(samples["stereo_score"]),
            "disparity_uncertainty_px": finite_stats(samples["disparity_uncertainty_px"]),
            "observed_sessions": finite_stats(samples["observed_sessions"]),
            "geometry": geometry_stats(points),
            "outputs": {
                "ply": str(ply_path),
                "npz": str(npz_path),
                "preview": str(preview_path),
                "regularized_depth_mm": str(regularized_depth_path),
                "validated_foreground_mask": str(foreground_mask_path),
                "source_silhouette_consensus": str(
                    consensus_dir / f"{stem}_silhouette_consensus.png"
                ),
                "source_cloud_mask_consensus": str(
                    consensus_dir / f"{stem}_cloud_mask_consensus.png"
                ),
            },
        }
        save_json(stats_path, record)
        records.append(record)
        previews.append(preview_path)
        print(
            f"[Paso 06 | 2/3 | {index:02d}/{len(views):02d}] P{pose_index:02d} {physical_angle:6.1f}° | raw={len(samples_raw['points']):7d} final={len(points):6d} | componentes_fg={foreground_stats['kept_component_count']} | {quality}"
        )
    return {"records": records, "previews": previews}


def main() -> int:
    """Genera y evalúa las nubes parciales a partir del consenso multisesión."""
    args = build_parser().parse_args()
    if o3d is None:
        raise SystemExit("Open3D no está instalado en el entorno activo.")

    root = Path(args.root).expanduser().resolve()
    object_name = args.object.strip().lower()
    consensus_dir = root / "reconstruccion" / "multisesion" / args.consensus_source
    summary_path = consensus_dir / "resumen_05_consenso_multisesion.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Falta consenso multisesión: {summary_path}")
    consensus_summary = load_json(summary_path)

    output_dir = root / "reconstruccion" / "multisesion" / args.output_name
    prepare_output_directory(output_dir, clean=True)

    calibration = load_stereo_geometry(
        root,
        object_name=object_name,
        calibration_dir=(args.calibration_dir or None),
    )
    intrinsics = {
        "fx": float(calibration["fx"]),
        "fy": float(calibration["fy"]),
        "cx": float(calibration["cx"]),
        "cy": float(calibration["cy"]),
    }
    stereo_fb_px_mm = float(calibration["fx"]) * float(calibration["baseline_mm"])

    sessions = [str(s) for s in consensus_summary.get("sessions", [])]
    background_source = str(args.background_depth_source)
    if background_source == "02_estimacion_profundidad" and consensus_summary.get(
        "parameters", {}
    ).get("depth_source"):
        background_source = str(consensus_summary["parameters"]["depth_source"])
    background_depths, background_paths = load_background_depths(
        root,
        sessions,
        background_source,
    )
    consensus_parameters = consensus_summary.get("parameters", {})
    depth_diagnostics_source = str(consensus_parameters.get("depth_source", background_source))
    regional_diagnostics_source = str(
        consensus_parameters.get("regional_source", "04_validacion_disparidad")
    )

    views = [
        view
        for view in consensus_summary.get("views", [])
        if str(view.get("quality", "")) in ("accepted", "warning")
    ]
    views.sort(key=lambda item: int(item["pose_index"]))
    records = []
    previews = []

    print("\n========== PASO 06: NUBES CONSENSO MULTISESIÓN ==========")
    print(
        "[Paso 06 | 1/3] Fondos métricos cargados y listos para "
        "clasificación normalizada por incertidumbre: "
        f"{len(background_depths)}."
    )
    _results = ejecutar_items(
        _procesar_unidad_independiente,
        {
            "args": args,
            "background_depths": background_depths,
            "consensus_dir": consensus_dir,
            "depth_diagnostics_source": depth_diagnostics_source,
            "intrinsics": intrinsics,
            "output_dir": output_dir,
            "regional_diagnostics_source": regional_diagnostics_source,
            "root": root,
            "stereo_fb_px_mm": stereo_fb_px_mm,
            "views": views,
        },
        enumerate(views, start=1),
        "Validar profundidad y crear nube por pose",
        reserve_mb=768,
    )
    for _result in _results:
        records.extend(_result["records"])
        previews.extend(_result["previews"])

    build_contact_sheet(
        previews,
        output_dir / "contact_sheet_nubes_consenso_multisesion.png",
    )
    summary = {
        "schema_version": 6,
        "method": (
            "background_candidates_stereo_uncertainty_validation_"
            "observed_seed_components_and_domain_limited_regularization"
        ),
        "object": object_name,
        "session": "multisesion",
        "sessions": consensus_summary.get("sessions", []),
        "source": str(consensus_dir),
        "consensus_source_name": str(args.consensus_source),
        "output_dir": str(output_dir),
        "physical_step_deg": consensus_summary.get("physical_step_deg"),
        "parameters": vars(args),
        "camera_intrinsics": intrinsics,
        "stereo_fb_px_mm": stereo_fb_px_mm,
        "background_depth_paths": background_paths,
        "background_depth_source_used": background_source,
        "depth_diagnostics_source_used": depth_diagnostics_source,
        "regional_diagnostics_source_used": regional_diagnostics_source,
        "views_processed": len(records),
        "views_accepted": sum(r["quality"] == "accepted" for r in records),
        "views_warning": sum(r["quality"] == "warning" for r in records),
        "views_rejected": sum(r["quality"] == "rejected" for r in records),
        "views": records,
        "important_note": (
            "Cada nube corresponde a una pose física corregida y fusiona "
            "observaciones multisesión antes del registro. La profundidad "
            "absoluta no se recentra ni se escala. Fondo y objeto se separan "
            "primero mediante un residuo unilateral en 1/Z. La aceptación exige "
            "además evidencia estéreo observada, consistencia izquierda-derecha, "
            "residuo regional, incertidumbre propagada y continuidad local. Cada "
            "componente necesita semillas observadas propias; no se conserva por "
            "tamaño. La regularización solo opera dentro del dominio validado."
        ),
    }
    save_json(
        output_dir / "resumen_06_nubes_puntos.json",
        summary,
    )

    csv_path = output_dir / "calidad_nubes_consenso_multisesion.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "pose_index",
            "physical_angle_deg",
            "quality",
            "raw_points",
            "after_voxel",
            "after_statistical",
            "after_radius",
            "after_dbscan",
            "final_points",
            "cluster_count",
            "main_cluster_ratio",
            "reliable_depth_pixels",
            "reliable_ratio_of_input_mask",
            "reasons",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: (" | ".join(record["reasons"]) if key == "reasons" else record.get(key))
                    for key in fieldnames
                }
            )

    print("[Paso 06 | 3/3] Guardando resumen y tabla de calidad...")
    print("\n========== PASO 06 COMPLETADO ==========")
    print(f"Salida: {output_dir}")
    print(
        f"Accepted={summary['views_accepted']} | Warning={summary['views_warning']} "
        f"| Rejected={summary['views_rejected']}"
    )
    print("===========================================\n")
    return 0


def _entrada_con_diagnostico():
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    from utilidades_progreso import informar_inicio

    informar_inicio(__file__, "06", "Crear nubes de puntos")
    raise SystemExit(main())


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
