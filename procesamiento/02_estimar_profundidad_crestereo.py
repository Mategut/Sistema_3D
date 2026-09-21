#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Paso 02 — Estimación de profundidad con CREStereo ONNX.

Entradas: pares estereoscópicos, calibración y fondo vacío del montaje.
La red recibe RGB float32 en el rango 0..255: el modelo ONNX incluido ya
contiene su normalización. Se calcula disparidad en ambos sentidos y se
evalúan consistencia izquierda-derecha, fotometría, gradiente y suavidad.

La profundidad científica (*_depth_mm.npy) conserva evidencia estéreo fuerte
y observaciones unilaterales débiles con procedencia explícita.
*_trusted_mask.png contiene solo evidencia fuerte; *_usable_mask.png añade
observaciones débiles que deberán ser confirmadas por 04/05/06. Ambas dependen
de rectificación, evidencia estéreo y rango físico. La máscara visual (*_object_mask.png) se conserva como diagnóstico
y no elimina por sí sola profundidad válida. Se guarda la disparidad del
fondo para la comparación local del paso 03.

La auditoría epipolar del fondo es diagnóstica y la rectificación calibrada
permanece congelada. Una tendencia SIFT no modifica automáticamente la imagen
derecha ni P1/P2/Q; si se confirma físicamente debe repetirse la calibración.
La coordenada
horizontal y la escala métrica fx*B/d permanecen definidas por la calibración.

La recuperación visual cerca del soporte exige profundidad fiable, conexión
con la semilla del objeto y separación del plano local de la plataforma.
La profundidad del fondo vacío no decide por sí sola la pertenencia al objeto.

CUDA es obligatorio por defecto con --provider auto/cuda. No se importa
PyTorch para evitar cargar otro runtime OpenMP en Windows; se utiliza el
modelo ONNX local. Los controles por vista y por sesión se conservan en los
resúmenes y las profundidades rechazadas se invalidan explícitamente.
"""

from __future__ import annotations
from utilidades_progreso import operacion
from utilidades_rendimiento import configurar as _configurar_recursos

# Los runtimes numéricos leen estos límites al importarse.
_configurar_recursos(__file__)


import argparse
import csv
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utilidades_mascaras import (
    adaptive_visual_anchor,
    background_depth_consistency_diagnostics,
    derive_depth_display_range,
    estimate_turntable_support_mask,
    fixed_depth_visualization,
    imwrite_checked,
    prepare_output_directory,
    resolve_calibration_dir,
    safe_rectification_domain,
)

try:
    import onnxruntime as ort
except Exception as exc:
    raise SystemExit(
        "No se pudo importar onnxruntime. Instala onnxruntime-gpu, "
        "onnxruntime-directml u onnxruntime.\n"
        f"Detalle: {exc}"
    )

# No se importa torch. En algunos entornos Conda de Windows, OpenCV/NumPy ya
# han inicializado OpenMP y la importación adicional de PyTorch intenta cargar
# otra libiomp5md.dll, lo que provoca OMP Error #15 y aborto del proceso.
# ONNX Runtime >= 1.21 puede localizar por sí mismo las DLL de CUDA/cuDNN y
# precargarlas sin inicializar el runtime completo de PyTorch.
ORT_PRELOAD_ERROR: Optional[str] = None
if os.name == "nt" and hasattr(ort, "preload_dlls"):
    try:
        ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None)
    except Exception as exc:
        # La disponibilidad real se valida después mediante los providers y el
        # detalle queda incluido en el diagnóstico accionable de CUDA.
        ORT_PRELOAD_ERROR = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Calibración, emparejamiento y rectificación
# ---------------------------------------------------------------------------


def read_yaml_matrix(yaml_path: Path, key: str) -> np.ndarray:
    """Lee una matriz de calibración y rechaza archivos o nodos ausentes."""
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"No se pudo abrir: {yaml_path}")

    node = fs.getNode(key)
    if node.empty():
        fs.release()
        raise KeyError(f"No se encontró la clave '{key}' en {yaml_path}")

    matrix = node.mat()
    fs.release()

    if matrix is None:
        raise RuntimeError(f"La clave '{key}' no contiene una matriz válida en {yaml_path}")
    return np.asarray(matrix)


def load_calibration(calib_dir: Path) -> Dict[str, np.ndarray | float]:
    """Carga las matrices estéreo y los mapas de rectificación del montaje."""
    stereo_yaml = calib_dir / "stereo_initial.yaml"
    maps_npz = calib_dir / "rectification_maps.npz"

    p1 = read_yaml_matrix(stereo_yaml, "P1")
    translation = read_yaml_matrix(stereo_yaml, "T")

    if not maps_npz.exists():
        raise FileNotFoundError(f"No existe: {maps_npz}")

    with np.load(str(maps_npz)) as maps:
        required = ("map1x", "map1y", "map2x", "map2y")
        missing = [key for key in required if key not in maps.files]
        if missing:
            raise KeyError("Faltan mapas de rectificación en el NPZ: " + ", ".join(missing))

        map1x = maps["map1x"]
        map1y = maps["map1y"]
        map2x = maps["map2x"]
        map2y = maps["map2y"]

    baseline_mm = float(np.linalg.norm(translation.reshape(-1)))
    if not np.isfinite(baseline_mm) or baseline_mm <= 0.0:
        raise ValueError(f"Línea base inválida en {stereo_yaml}: {baseline_mm}")

    map_shapes = [map1x.shape[:2], map1y.shape[:2], map2x.shape[:2], map2y.shape[:2]]
    if len(set(map_shapes)) != 1:
        raise ValueError(f"Mapas de rectificación con tamaños incompatibles: {map_shapes}")

    return {
        "P1": p1,
        "baseline_mm": baseline_mm,
        "map1x": map1x,
        "map1y": map1y,
        "map2x": map2x,
        "map2y": map2y,
    }


def pair_stems(
    left_dir: Path,
    right_dir: Path,
) -> List[Tuple[str, Path, Path]]:
    """Empareja las imágenes izquierda y derecha por su identificador de captura."""
    left_files = sorted(left_dir.glob("*.png"))
    right_map = {file.name.replace("_R.png", ""): file for file in right_dir.glob("*.png")}

    pairs: List[Tuple[str, Path, Path]] = []
    for left_file in left_files:
        stem = left_file.name.replace("_L.png", "")
        right_file = right_map.get(stem)
        if right_file is not None:
            pairs.append((stem, left_file, right_file))

    return pairs


def calibrated_image_shape(
    calib: Dict[str, np.ndarray | float],
) -> Tuple[int, int]:
    shape = np.asarray(calib["map1x"]).shape[:2]
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"Tamaño de mapas de rectificación inválido: {shape}")
    return int(shape[0]), int(shape[1])


def validate_calibrated_image_shape(
    image: np.ndarray,
    calib: Dict[str, np.ndarray | float],
    label: str,
) -> None:
    expected = calibrated_image_shape(calib)
    received = image.shape[:2]
    if received != expected:
        raise ValueError(
            f"{label}: resolución {received[1]}x{received[0]} incompatible con "
            f"la calibración {expected[1]}x{expected[0]}. No se redimensiona "
            "una captura antes de rectificar porque alteraría la geometría."
        )


def rectify_pair(
    img_l: np.ndarray,
    img_r: np.ndarray,
    calib: Dict[str, np.ndarray | float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Rectifica el par estéreo mediante los mapas de calibración."""
    validate_calibrated_image_shape(img_l, calib, "Imagen izquierda")
    validate_calibrated_image_shape(img_r, calib, "Imagen derecha")
    rect_l = cv2.remap(
        img_l,
        np.asarray(calib["map1x"]),
        np.asarray(calib["map1y"]),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rect_r = cv2.remap(
        img_r,
        np.asarray(calib["map2x"]),
        np.asarray(calib["map2y"]),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rect_l, rect_r


def build_valid_mask(
    calib: Dict[str, np.ndarray | float],
) -> np.ndarray:
    """
    Máscara de rectificación puramente geométrica.

    No depende del brillo de la escena. De esta forma un objeto negro o una
    región oscura siguen siendo válidos si los mapas de rectificación apuntan
    dentro de ambas imágenes originales.
    """
    h, w = calibrated_image_shape(calib)
    source_valid = np.full((h, w), 255, dtype=np.uint8)
    valid_l = cv2.remap(
        source_valid,
        np.asarray(calib["map1x"]),
        np.asarray(calib["map1y"]),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid_r = cv2.remap(
        source_valid,
        np.asarray(calib["map2x"]),
        np.asarray(calib["map2y"]),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return ((valid_l >= 254) & (valid_r >= 254)).astype(np.uint8) * 255


def build_rectified_valid_components(
    calib: Dict[str, np.ndarray | float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Dominios geométricos válidos izquierdo/derecho antes del ajuste epipolar."""
    h, w = calibrated_image_shape(calib)
    source_valid = np.full((h, w), 255, dtype=np.uint8)
    valid_l = cv2.remap(
        source_valid,
        np.asarray(calib["map1x"]),
        np.asarray(calib["map1y"]),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid_r = cv2.remap(
        source_valid,
        np.asarray(calib["map2x"]),
        np.asarray(calib["map2y"]),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return valid_l, valid_r


def _vertical_correction_maps(
    shape: Tuple[int, int],
    model: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Mapas inversos para y'=(1+b)y+a*x+c, manteniendo x exactamente."""
    h, w = int(shape[0]), int(shape[1])
    a = float(model["a"])
    b = float(model["b"])
    c = float(model["c"])
    denominator = 1.0 + b
    if not np.isfinite(denominator) or abs(denominator) < 0.90:
        raise ValueError(
            "Corrección epipolar vertical inválida: escala vertical no plausible "
            f"(1+b={denominator})."
        )
    xx, yy = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    map_x = xx
    map_y = ((yy - a * xx - c) / denominator).astype(np.float32)
    return map_x, map_y


def apply_vertical_epipolar_correction(
    image: np.ndarray,
    model: Optional[Dict[str, float]],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    if not model or not bool(model.get("applied", False)):
        return image
    map_x, map_y = _vertical_correction_maps(image.shape[:2], model)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _robust_vertical_epipolar_fit(
    points_right: np.ndarray,
    vertical_error: np.ndarray,
    *,
    threshold_px: float,
    iterations: int,
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """RANSAC para dy = a*xR + b*yR + c."""
    n = int(len(vertical_error))
    if n < 3:
        return None, np.zeros(n, dtype=bool), np.full(n, np.inf)

    X = np.column_stack(
        [
            points_right[:, 0],
            points_right[:, 1],
            np.ones(n, dtype=np.float64),
        ]
    ).astype(np.float64)
    y = vertical_error.astype(np.float64)
    rng = np.random.default_rng(20260831)

    best_mask = np.zeros(n, dtype=bool)
    best_residual = np.full(n, np.inf)
    best_count = 0
    best_median = float("inf")

    for _ in range(max(100, int(iterations))):
        ids = rng.choice(n, 3, replace=False)
        sample = X[ids]
        if abs(float(np.linalg.det(sample))) < 1e-7:
            continue
        try:
            coefficients = np.linalg.solve(sample, y[ids])
        except np.linalg.LinAlgError:
            continue
        residual = np.abs(X @ coefficients - y)
        mask = residual <= float(threshold_px)
        count = int(np.count_nonzero(mask))
        if count < 3:
            continue
        median = float(np.median(residual[mask]))
        if count > best_count or (count == best_count and median < best_median):
            best_count = count
            best_median = median
            best_mask = mask
            best_residual = residual

    if best_count < 3:
        return None, best_mask, best_residual

    # Reajuste LS sobre los inliers y una iteración de refinamiento.
    coefficients, *_ = np.linalg.lstsq(X[best_mask], y[best_mask], rcond=None)
    residual = np.abs(X @ coefficients - y)
    refined = residual <= float(threshold_px)
    if int(np.count_nonzero(refined)) >= 3:
        coefficients, *_ = np.linalg.lstsq(X[refined], y[refined], rcond=None)
        residual = np.abs(X @ coefficients - y)
        refined = residual <= float(threshold_px)
    return coefficients, refined, residual


def estimate_epipolar_vertical_correction(
    rect_left: np.ndarray,
    rect_right: np.ndarray,
    calibration: Dict[str, np.ndarray | float],
    args,
) -> Dict[str, object]:
    """Audita rectificación con correspondencias naturales del fondo.

    Solo modela error VERTICAL. Nunca modifica x ni estima una nueva escala
    métrica. Si la corrección requerida no está bien respaldada, devuelve un
    diagnóstico no utilizable para que el llamador detenga el paso.
    """
    if not bool(args.epipolar_audit):
        return {
            "enabled": False,
            "status": "disabled",
            "applied": False,
        }

    gray_l = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    try:
        detector = cv2.SIFT_create(
            nfeatures=int(args.epipolar_sift_features),
            contrastThreshold=float(args.epipolar_sift_contrast),
            edgeThreshold=15,
        )
    except Exception as exc:
        return {
            "enabled": True,
            "status": "failed",
            "applied": False,
            "reason": f"SIFT no disponible: {type(exc).__name__}: {exc}",
        }

    kp_l, des_l = detector.detectAndCompute(gray_l, None)
    kp_r, des_r = detector.detectAndCompute(gray_r, None)
    if des_l is None or des_r is None or len(kp_l) < 8 or len(kp_r) < 8:
        return {
            "enabled": True,
            "status": "failed",
            "applied": False,
            "reason": "No hay suficientes descriptores naturales en el fondo.",
            "keypoints_left": 0 if kp_l is None else len(kp_l),
            "keypoints_right": 0 if kp_r is None else len(kp_r),
        }

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(des_l, des_r, k=2)

    fx = float(np.asarray(calibration["P1"])[0, 0])
    baseline = float(calibration["baseline_mm"])
    minimum_depth = max(float(args.minimum_depth_mm), 1.0)
    maximum_depth = max(float(args.maximum_depth_mm), minimum_depth + 1.0)
    disparity_min = max(8.0, 0.72 * fx * baseline / maximum_depth)
    disparity_max = min(
        rect_left.shape[1] * 0.92,
        1.35 * fx * baseline / minimum_depth,
    )

    candidates = []
    for pair in knn:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance >= float(args.epipolar_match_ratio) * second.distance:
            continue
        x_l, y_l = kp_l[first.queryIdx].pt
        x_r, y_r = kp_r[first.trainIdx].pt
        disparity = float(x_l - x_r)
        dy = float(y_l - y_r)
        if not (
            disparity_min <= disparity <= disparity_max
            and abs(dy) <= float(args.epipolar_match_max_vertical_px)
        ):
            continue
        candidates.append((x_l, y_l, x_r, y_r, disparity, dy, first.distance))

    if len(candidates) < int(args.epipolar_minimum_matches):
        return {
            "enabled": True,
            "status": "failed",
            "applied": False,
            "reason": (
                "Correspondencias insuficientes para auditar rectificación: "
                f"{len(candidates)} < {int(args.epipolar_minimum_matches)}."
            ),
            "candidate_matches": len(candidates),
            "disparity_range_px": [float(disparity_min), float(disparity_max)],
        }

    right_points = np.asarray(
        [[item[2], item[3]] for item in candidates],
        dtype=np.float64,
    )
    vertical_error = np.asarray(
        [item[5] for item in candidates],
        dtype=np.float64,
    )

    coefficients, inliers, residual = _robust_vertical_epipolar_fit(
        right_points,
        vertical_error,
        threshold_px=float(args.epipolar_ransac_threshold_px),
        iterations=int(args.epipolar_ransac_iterations),
    )
    if coefficients is None:
        return {
            "enabled": True,
            "status": "failed",
            "applied": False,
            "reason": "RANSAC epipolar no encontró un modelo vertical estable.",
            "candidate_matches": len(candidates),
        }

    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = inlier_count / max(len(candidates), 1)
    if inlier_count:
        x_span = float(np.ptp(right_points[inliers, 0]) / max(rect_right.shape[1] - 1, 1))
        y_span = float(np.ptp(right_points[inliers, 1]) / max(rect_right.shape[0] - 1, 1))
        post_residual_p95 = float(np.percentile(residual[inliers], 95.0))
        post_residual_median = float(np.median(residual[inliers]))
        pre_abs_median = float(np.median(np.abs(vertical_error[inliers])))
        pre_abs_p95 = float(np.percentile(np.abs(vertical_error[inliers]), 95.0))
    else:
        x_span = y_span = 0.0
        post_residual_p95 = post_residual_median = float("inf")
        pre_abs_median = pre_abs_p95 = float("inf")

    a, b, c = (float(v) for v in coefficients)
    h, w = rect_right.shape[:2]
    corners = np.asarray(
        [[0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0]],
        dtype=np.float64,
    )
    predicted_corner_dy = a * corners[:, 0] + b * corners[:, 1] + c
    maximum_correction = float(np.max(np.abs(predicted_corner_dy)))
    systematic_range = float(np.max(predicted_corner_dy) - np.min(predicted_corner_dy))

    evidence_ok = (
        inlier_count >= int(args.epipolar_minimum_inliers)
        and inlier_ratio >= float(args.epipolar_minimum_inlier_ratio)
        and x_span >= float(args.epipolar_minimum_x_span_fraction)
        and post_residual_p95 <= float(args.epipolar_maximum_post_residual_p95_px)
        and maximum_correction <= float(args.epipolar_maximum_correction_px)
        and abs(b) <= float(args.epipolar_maximum_vertical_scale_delta)
    )

    needs_correction = maximum_correction > float(
        args.epipolar_no_correction_max_px
    ) or systematic_range > 2.0 * float(args.epipolar_no_correction_max_px)

    result: Dict[str, object] = {
        "enabled": True,
        "status": "accepted" if evidence_ok else "failed",
        "candidate_matches": len(candidates),
        "inliers": inlier_count,
        "inlier_ratio": float(inlier_ratio),
        "x_span_fraction": float(x_span),
        "y_span_fraction": float(y_span),
        "pre_abs_vertical_median_px": pre_abs_median,
        "pre_abs_vertical_p95_px": pre_abs_p95,
        "post_model_residual_median_px": post_residual_median,
        "post_model_residual_p95_px": post_residual_p95,
        "systematic_vertical_range_px": systematic_range,
        "maximum_vertical_correction_px": maximum_correction,
        "disparity_range_px": [float(disparity_min), float(disparity_max)],
        "model": {"a": a, "b": b, "c": c},
        "predicted_corner_dy_px": [float(v) for v in predicted_corner_dy],
        "needs_correction": bool(needs_correction),
        "applied": False,
    }

    if not evidence_ok:
        result["reason"] = (
            "La rectificación no pudo validarse con evidencia suficiente. "
            "Debe recalibrarse el par estéreo antes de confiar en la profundidad."
        )
        return result

    if not needs_correction:
        result["applied"] = False
        result["status"] = "accepted_no_correction"
        return result

    if not bool(args.epipolar_auto_correct):
        result["status"] = (
            "failed" if bool(getattr(args, "epipolar_audit_strict", False))
            else "warning_no_correction"
        )
        result["reason"] = (
            "El auditor SIFT detectó una tendencia vertical, pero NO se aplica una "
            "transformación adicional a imágenes ya rectificadas. La geometría oficial "
            "P1/P2/Q permanece congelada. Recalibre físicamente si la discrepancia se "
            "confirma con un patrón de calibración."
        )
        result["applied"] = False
        return result

    result["applied"] = True
    result["status"] = "accepted_corrected"
    # Duplicar coeficientes en la raíz simplifica el uso por remap.
    result["a"] = a
    result["b"] = b
    result["c"] = c
    return result


# ---------------------------------------------------------------------------
# Inferencia ONNX
# ---------------------------------------------------------------------------


def onnxruntime_environment_diagnostics() -> Dict[str, object]:
    """Devuelve evidencia suficiente para diagnosticar el provider real."""
    distributions: Dict[str, Optional[str]] = {}
    for package_name in (
        "onnxruntime-gpu",
        "onnxruntime",
        "onnxruntime-directml",
        "torch",
    ):
        try:
            distributions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            distributions[package_name] = None

    try:
        ort_device: Optional[str] = str(ort.get_device())
    except Exception:
        ort_device = None

    return {
        "python_executable": sys.executable,
        "onnxruntime_version": str(getattr(ort, "__version__", "unknown")),
        "onnxruntime_module": str(Path(ort.__file__).resolve()),
        "onnxruntime_device": ort_device,
        "available_providers": list(ort.get_available_providers()),
        "installed_distributions": distributions,
        "preload_dlls_available": bool(hasattr(ort, "preload_dlls")),
        "preload_dlls_error": ORT_PRELOAD_ERROR,
    }


def format_cuda_runtime_error(
    reason: str,
    diagnostics: Optional[Dict[str, object]] = None,
) -> str:
    """Mensaje que evita una ejecución CPU silenciosa y explica cómo repararla."""
    diag = diagnostics or onnxruntime_environment_diagnostics()
    executable = str(diag.get("python_executable") or sys.executable)
    quoted_python = f'"{executable}"'
    return (
        "CUDA no quedó activo para CREStereo y se canceló la ejecución para "
        "evitar procesar toda la sesión en CPU.\n"
        f"Motivo: {reason}\n"
        f"Python: {executable}\n"
        f"ONNX Runtime: {diag.get('onnxruntime_version')}\n"
        f"Módulo cargado: {diag.get('onnxruntime_module')}\n"
        f"Dispositivo ORT: {diag.get('onnxruntime_device')}\n"
        f"Providers detectados: {diag.get('available_providers')}\n"
        f"Paquetes: {diag.get('installed_distributions')}\n"
        f"Error preload_dlls: {diag.get('preload_dlls_error')}\n\n"
        "Reparación recomendada, dentro del MISMO entorno de Python:\n"
        f"  {quoted_python} -m pip uninstall -y onnxruntime "
        "onnxruntime-directml onnxruntime-gpu\n"
        f"  {quoted_python} -m pip install --upgrade pip\n"
        f'  {quoted_python} -m pip install --upgrade "onnxruntime-gpu[cuda,cudnn]"\n'
        "  Cierra y abre de nuevo la terminal/Anaconda Prompt.\n"
        "  Verifica con:\n"
        f'  {quoted_python} -c "import onnxruntime as o; '
        "o.preload_dlls() if hasattr(o,'preload_dlls') else None; "
        'print(o.__version__); print(o.get_available_providers())"\n'
        "El resultado debe incluir 'CUDAExecutionProvider'. Para una prueba "
        "CPU deliberada, usa --provider cpu o --no-require-cuda."
    )


class CREStereoONNX:
    def __init__(
        self,
        model_path: str,
        preferred_provider: str = "auto",
        require_cuda: bool = False,
    ):
        self.model_path = model_path
        self.available_providers = ort.get_available_providers()
        self.environment_diagnostics = onnxruntime_environment_diagnostics()
        self.require_cuda = bool(require_cuda)
        if self.require_cuda and "CUDAExecutionProvider" not in self.available_providers:
            raise RuntimeError(
                format_cuda_runtime_error(
                    "CUDAExecutionProvider no aparece entre los providers disponibles.",
                    self.environment_diagnostics,
                )
            )
        self.providers = self._select_providers(preferred_provider)

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = int(
            os.environ.get("SISTEMA3D_CPU_THREADS", os.cpu_count() or 1)
        )
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=self.providers,
            )
        except Exception as exc:
            if self.require_cuda:
                raise RuntimeError(
                    format_cuda_runtime_error(
                        f"falló la creación de la sesión ONNX: {type(exc).__name__}: {exc}",
                        self.environment_diagnostics,
                    )
                ) from exc
            raise

        active_providers = list(self.session.get_providers())
        self.environment_diagnostics["session_providers"] = active_providers
        if self.require_cuda and (
            not active_providers or active_providers[0] != "CUDAExecutionProvider"
        ):
            raise RuntimeError(
                format_cuda_runtime_error(
                    f"la sesión quedó activa con {active_providers}",
                    self.environment_diagnostics,
                )
            )

        self.input_names = [item.name for item in self.session.get_inputs()]
        if len(self.input_names) < 2:
            raise RuntimeError("El modelo ONNX no expone dos entradas de imagen.")

        self.output_names = [item.name for item in self.session.get_outputs()]
        self.input_shapes = [item.shape for item in self.session.get_inputs()]

        self.target_h, self.target_w = self._infer_target_size()

    def _select_providers(self, preferred: str) -> List[str]:
        preferred = preferred.lower().strip()

        explicit = {
            "cuda": "CUDAExecutionProvider",
            "directml": "DmlExecutionProvider",
            "cpu": "CPUExecutionProvider",
        }
        if preferred in explicit:
            requested = explicit[preferred]
            if requested not in self.available_providers:
                raise RuntimeError(
                    f"Se solicitó '{preferred}', pero {requested} no está disponible. "
                    f"Providers detectados: {self.available_providers}"
                )
            selected = [requested]
            if (
                requested != "CPUExecutionProvider"
                and "CPUExecutionProvider" in self.available_providers
            ):
                selected.append("CPUExecutionProvider")
            return selected

        candidates = [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
        selected = [provider for provider in candidates if provider in self.available_providers]
        if not selected:
            raise RuntimeError(
                "No hay providers utilizables. " f"Disponibles: {self.available_providers}"
            )
        return selected

    def _infer_target_size(self) -> Tuple[int, int]:
        for shape in self.input_shapes:
            if len(shape) != 4:
                continue

            h = shape[2]
            w = shape[3]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                return int(h), int(w)

        return -1, -1

    def _prepare_image(
        self,
        image_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, int | float]]:
        """Prepara la entrada sin deformar la relación de aspecto.

        Los modelos CREStereo de tamaño fijo suelen aceptar una rejilla concreta,
        pero deformar 16:9 a 4:3 altera pendientes, bordes y texturas. Se usa
        letterbox simétrico y luego la disparidad se deshace usando únicamente
        la escala horizontal real del contenido, no la del canvas completo.
        """
        original_h, original_w = image_bgr.shape[:2]

        if self.target_h > 0 and self.target_w > 0:
            processing_h = self.target_h
            processing_w = self.target_w
        else:
            processing_h = int(np.ceil(original_h / 8.0) * 8)
            processing_w = int(np.ceil(original_w / 8.0) * 8)

        scale = min(
            processing_w / float(original_w),
            processing_h / float(original_h),
        )
        content_w = max(1, min(processing_w, int(round(original_w * scale))))
        content_h = max(1, min(processing_h, int(round(original_h * scale))))
        pad_left = (processing_w - content_w) // 2
        pad_top = (processing_h - content_h) // 2
        pad_right = processing_w - content_w - pad_left
        pad_bottom = processing_h - content_h - pad_top

        interpolation = (
            cv2.INTER_AREA
            if content_w <= original_w and content_h <= original_h
            else cv2.INTER_LINEAR
        )
        resized = cv2.resize(
            image_bgr,
            (content_w, content_h),
            interpolation=interpolation,
        )
        # REFLECT101 evita crear dos bandas constantes que la red podría
        # interpretar como estructura. El mismo padding se aplica a ambas vistas.
        processed = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB).astype(np.float32)
        processed = np.transpose(processed, (2, 0, 1))[None, ...]

        meta: Dict[str, int | float] = {
            "processing_h": int(processing_h),
            "processing_w": int(processing_w),
            "content_h": int(content_h),
            "content_w": int(content_w),
            "pad_left": int(pad_left),
            "pad_top": int(pad_top),
            "original_h": int(original_h),
            "original_w": int(original_w),
            "scale_x": float(content_w / float(original_w)),
            "scale_y": float(content_h / float(original_h)),
        }
        return processed, meta

    def __call__(
        self,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
    ) -> np.ndarray:
        original_h, original_w = left_bgr.shape[:2]

        left_input, prep = self._prepare_image(left_bgr)
        right_input, prep_right = self._prepare_image(right_bgr)
        if (
            int(prep["content_h"]) != int(prep_right["content_h"])
            or int(prep["content_w"]) != int(prep_right["content_w"])
            or int(prep["pad_left"]) != int(prep_right["pad_left"])
            or int(prep["pad_top"]) != int(prep_right["pad_top"])
        ):
            raise RuntimeError("Las dos cámaras no comparten la misma geometría de letterbox.")

        outputs = self.session.run(
            self.output_names,
            {
                self.input_names[0]: left_input,
                self.input_names[1]: right_input,
            },
        )

        disparity = outputs[0]

        if disparity.ndim == 4:
            disparity = disparity[0, 0]
        elif disparity.ndim == 3:
            disparity = disparity[0]
        else:
            raise RuntimeError(f"Forma de salida no soportada: {disparity.shape}")

        disparity = disparity.astype(np.float32)

        # Retirar el letterbox antes de volver al tamaño original. La disparidad
        # está expresada en píxeles del contenido redimensionado; solo se escala
        # por la relación horizontal real original/contenido.
        y0 = int(prep["pad_top"])
        x0 = int(prep["pad_left"])
        ch = int(prep["content_h"])
        cw = int(prep["content_w"])
        disparity = disparity[y0:y0 + ch, x0:x0 + cw]
        if disparity.size == 0:
            raise RuntimeError("El recorte de letterbox produjo una disparidad vacía.")
        horizontal_scale = original_w / float(cw)
        disparity = (
            cv2.resize(
                disparity,
                (original_w, original_h),
                interpolation=cv2.INTER_LINEAR,
            )
            * horizontal_scale
        )

        return disparity

    def predict_bidirectional(
        self,
        rect_l: np.ndarray,
        rect_r: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtiene dos mapas con disparidad positiva:

        d_lr: referencia en la cámara izquierda.
        d_rl: referencia en la cámara derecha.

        Para calcular d_rl sin exigir que el modelo produzca disparidad
        negativa, se intercambian las cámaras y se invierten horizontalmente.
        Al terminar, el resultado se invierte de nuevo.
        """
        disparity_lr = self(rect_l, rect_r)

        flipped_right = cv2.flip(rect_r, 1)
        flipped_left = cv2.flip(rect_l, 1)

        disparity_flipped = self(
            flipped_right,
            flipped_left,
        )
        disparity_rl = cv2.flip(disparity_flipped, 1)

        return disparity_lr, disparity_rl


# ---------------------------------------------------------------------------
# Segmentación por fondo vacío
# ---------------------------------------------------------------------------


def robust_threshold_from_background(
    score: np.ndarray,
    rect_valid_mask: np.ndarray,
    roi_mask: np.ndarray,
    minimum_threshold: float,
    mad_factor: float,
) -> float:
    """Umbral robusto independiente de una ROI central histórica."""
    valid = (rect_valid_mask > 0) & (roi_mask > 0) & np.isfinite(score)
    values = score[valid]
    if values.size == 0:
        return float(minimum_threshold)

    cutoff = float(np.quantile(values, 0.62))
    stable_values = values[values <= cutoff]
    if stable_values.size < 2000:
        stable_values = values

    med = float(np.median(stable_values))
    mad = float(np.median(np.abs(stable_values - med)))
    sigma = max(0.75, 1.4826 * mad)
    return float(max(minimum_threshold, med + mad_factor * sigma))


def select_foreground_components(
    mask: np.ndarray,
    minimum_area: int,
) -> np.ndarray:
    """
    Conserva componentes próximas al centro y evita franjas largas del fondo.
    Puede retener más de una componente cuando ambas están cerca del objeto.
    """
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    h, w = mask.shape
    center = np.array([w / 2.0, h * 0.52], dtype=np.float64)
    image_diag = max(1.0, math.hypot(w, h))

    candidates = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        centroid = centroids[label]

        distance = float(np.linalg.norm(centroid - center))
        compactness = area / max(float(width * height), 1.0)
        center_score = math.exp(-5.0 * distance / image_diag)

        score = area * (0.40 + 0.60 * center_score) * (0.45 + 0.55 * compactness)
        candidates.append((score, label, area))

    if not candidates:
        return np.zeros_like(mask, dtype=np.uint8)

    candidates.sort(reverse=True)
    best_score = candidates[0][0]

    selected = np.zeros_like(mask, dtype=np.uint8)
    for score, label, area in candidates:
        # Permite piezas secundarias cercanas, pero no componentes débiles.
        if score < 0.18 * best_score:
            continue
        selected[labels == label] = 255

    return selected


def segment_foreground_from_background(
    rect_left: np.ndarray,
    background_left: np.ndarray,
    rect_valid_mask: np.ndarray,
    roi_mask: np.ndarray,
    minimum_threshold: float,
    mad_factor: float,
    minimum_area: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if rect_left.shape != background_left.shape:
        raise ValueError(
            "La imagen y el fondo rectificado tienen dimensiones distintas: "
            f"{rect_left.shape} vs {background_left.shape}"
        )

    lab_current = cv2.cvtColor(rect_left, cv2.COLOR_BGR2LAB)
    lab_background = cv2.cvtColor(background_left, cv2.COLOR_BGR2LAB)

    gray_current = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.cvtColor(background_left, cv2.COLOR_BGR2GRAY)

    lab_difference = cv2.absdiff(
        lab_current,
        lab_background,
    ).astype(np.float32)
    gray_difference = cv2.absdiff(
        gray_current,
        gray_background,
    ).astype(np.float32)

    chromatic_difference = np.max(
        lab_difference,
        axis=2,
    )
    score = 0.75 * chromatic_difference + 0.25 * gray_difference
    score = cv2.GaussianBlur(
        score,
        (5, 5),
        0,
    )

    threshold = robust_threshold_from_background(
        score,
        rect_valid_mask,
        roi_mask,
        minimum_threshold,
        mad_factor,
    )

    raw = ((score >= threshold) & (rect_valid_mask > 0) & (roi_mask > 0)).astype(np.uint8) * 255

    raw = cv2.morphologyEx(
        raw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        ),
        iterations=1,
    )
    raw = cv2.morphologyEx(
        raw,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (19, 19),
        ),
        iterations=2,
    )

    foreground = select_foreground_components(
        raw,
        minimum_area=minimum_area,
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (13, 13),
        ),
        iterations=1,
    )
    foreground = cv2.medianBlur(
        foreground,
        5,
    )

    overlay = rect_left.copy()
    tint = np.zeros_like(overlay)
    tint[:, :, 1] = foreground
    overlay = cv2.addWeighted(
        overlay,
        1.0,
        tint,
        0.35,
        0.0,
    )
    cv2.putText(
        overlay,
        f"fondo: umbral={threshold:.2f}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"fondo: umbral={threshold:.2f}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return foreground, overlay, threshold


def build_foreground_overlay(
    rect_left: np.ndarray,
    foreground: np.ndarray,
    line1: str,
    line2: str = "",
) -> np.ndarray:
    overlay = rect_left.copy()
    tint = np.zeros_like(overlay)
    tint[:, :, 1] = foreground
    overlay = cv2.addWeighted(
        overlay,
        1.0,
        tint,
        0.35,
        0.0,
    )

    lines = [line1]
    if line2:
        lines.append(line2)

    for index, line in enumerate(lines):
        y = 35 + 30 * index
        cv2.putText(
            overlay,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return overlay


def refine_foreground_with_background_depth(
    rgb_foreground: np.ndarray,
    current_depth_mm: np.ndarray,
    background_depth_mm: np.ndarray,
    rect_valid_mask: np.ndarray,
    roi_mask: np.ndarray,
    minimum_depth_separation_mm: float,
    minimum_area: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """V1.5: el fondo 3D es diagnóstico; jamás crea foreground."""
    domain = (rect_valid_mask > 0) & (roi_mask > 0)
    rgb = (rgb_foreground > 0) & domain
    refined = rgb.astype(np.uint8) * 255
    refined = cv2.morphologyEx(
        refined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )
    refined = cv2.morphologyEx(
        refined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1
    )
    refined = select_foreground_components(refined, minimum_area=minimum_area)
    comparable = domain & np.isfinite(current_depth_mm) & np.isfinite(background_depth_mm)
    delta = background_depth_mm - current_depth_mm
    depth_foreground = (comparable & (delta >= float(minimum_depth_separation_mm))).astype(
        np.uint8
    ) * 255
    return refined, depth_foreground


def _object_recovery_envelope(
    foreground: np.ndarray,
    support_mask: np.ndarray,
    domain_mask: np.ndarray,
    minimum_area: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Separa el cuerpo del objeto de la plataforma antes de crear la envolvente.

    La máscara RGB puede contener la plataforma completa. Usar su convex hull
    global hace que la envolvente cubra todo el soporte y deja cero muestras
    para ajustar el plano. Aquí la parte situada sobre el borde superior de la
    plataforma define el cuerpo; su caja se prolonga solo hasta el borde
    inferior del soporte para buscar la cara inferior ausente.
    """
    seed = ((foreground > 0) & (domain_mask > 0)).astype(np.uint8) * 255
    support = ((support_mask > 0) & (domain_mask > 0)).astype(np.uint8) * 255
    envelope = np.zeros_like(seed)
    recovery_zone = np.zeros_like(seed)
    diagnostics: Dict[str, object] = {
        "reason": None,
        "support_top_y": None,
        "support_bottom_y": None,
        "object_head_bbox_xywh": None,
        "horizontal_margin_px": None,
    }

    support_ys, _ = np.nonzero(support > 0)
    if support_ys.size == 0:
        diagnostics["reason"] = "support_mask_empty"
        return envelope, recovery_zone, diagnostics

    support_top = int(support_ys.min())
    support_bottom = int(support_ys.max()) + 1
    diagnostics["support_top_y"] = support_top
    diagnostics["support_bottom_y"] = support_bottom

    head_seed = seed.copy()
    head_seed[support_top:, :] = 0
    contours, _ = cv2.findContours(
        head_seed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    valid_contours = [
        contour for contour in contours if cv2.contourArea(contour) >= float(max(50, minimum_area))
    ]
    if not valid_contours:
        diagnostics["reason"] = "object_head_not_detected_above_support"
        return envelope, recovery_zone, diagnostics

    object_contour = max(valid_contours, key=cv2.contourArea)
    hull = cv2.convexHull(object_contour)
    cv2.fillConvexPoly(envelope, hull, 255)
    x, y, width, height = cv2.boundingRect(object_contour)
    margin = max(8, int(round(0.025 * width)))
    x0 = max(0, x - margin)
    x1 = min(seed.shape[1], x + width + margin)
    envelope[support_top:support_bottom, x0:x1] = 255
    recovery_zone[support_top:support_bottom, x0:x1] = 255

    domain_u8 = (domain_mask > 0).astype(np.uint8) * 255
    envelope = cv2.bitwise_and(envelope, domain_u8)
    recovery_zone = cv2.bitwise_and(recovery_zone, domain_u8)
    diagnostics.update(
        {
            "object_head_bbox_xywh": [int(x), int(y), int(width), int(height)],
            "horizontal_margin_px": int(margin),
            "envelope_pixels": int(np.count_nonzero(envelope)),
            "recovery_zone_pixels": int(np.count_nonzero(recovery_zone)),
        }
    )
    return envelope, recovery_zone, diagnostics


def _support_ring_sample_mask(
    support_mask: np.ndarray,
    seed_mask: np.ndarray,
    domain_mask: np.ndarray,
    physical_depth_mask: np.ndarray,
    foreground_exclusion_radius_px: int,
    minimum_samples: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Toma una corona visible del soporte, lejos de la máscara del objeto."""
    support = (support_mask > 0) & (domain_mask > 0)
    ys, xs = np.nonzero(support)
    empty = np.zeros_like(support_mask, dtype=np.uint8)
    diagnostics: Dict[str, object] = {
        "reason": None,
        "available_samples": 0,
        "ring_width_px": None,
        "exclusion_radius_px": None,
    }
    if xs.size == 0:
        diagnostics["reason"] = "support_mask_empty"
        return empty, diagnostics

    support_width = int(xs.max() - xs.min() + 1)
    support_height = int(ys.max() - ys.min() + 1)
    base_width = int(
        np.clip(
            round(0.04 * min(support_width, support_height)),
            8,
            24,
        )
    )
    ring_widths = list(
        dict.fromkeys(
            [
                base_width,
                min(32, max(base_width + 4, int(round(1.5 * base_width)))),
                min(40, max(base_width + 8, 2 * base_width)),
            ]
        )
    )
    radius = max(1, int(foreground_exclusion_radius_px))
    exclusion_radii = list(
        dict.fromkeys(
            [
                radius,
                max(10, radius - 5),
                max(6, radius // 2),
            ]
        )
    )

    best_mask = empty
    best_count = -1
    chosen_width = ring_widths[0]
    chosen_radius = exclusion_radii[0]
    support_u8 = support.astype(np.uint8) * 255
    seed_u8 = (seed_mask > 0).astype(np.uint8) * 255

    for exclusion_radius in exclusion_radii:
        exclusion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * exclusion_radius + 1, 2 * exclusion_radius + 1),
        )
        excluded = cv2.dilate(seed_u8, exclusion_kernel, iterations=1) > 0
        for ring_width in ring_widths:
            ring_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * ring_width + 1, 2 * ring_width + 1),
            )
            inner = cv2.erode(support_u8, ring_kernel, iterations=1) > 0
            ring = support & (~inner)
            samples = ring & (~excluded) & physical_depth_mask & (domain_mask > 0)
            count = int(np.count_nonzero(samples))
            if count > best_count:
                best_count = count
                best_mask = samples.astype(np.uint8) * 255
                chosen_width = ring_width
                chosen_radius = exclusion_radius
            if count >= int(minimum_samples):
                diagnostics.update(
                    {
                        "available_samples": count,
                        "ring_width_px": int(ring_width),
                        "exclusion_radius_px": int(exclusion_radius),
                        "support_bbox_xywh": [
                            int(xs.min()),
                            int(ys.min()),
                            support_width,
                            support_height,
                        ],
                    }
                )
                return samples.astype(np.uint8) * 255, diagnostics

    diagnostics.update(
        {
            "reason": "insufficient_support_ring_samples",
            "available_samples": max(best_count, 0),
            "ring_width_px": int(chosen_width),
            "exclusion_radius_px": int(chosen_radius),
            "support_bbox_xywh": [
                int(xs.min()),
                int(ys.min()),
                support_width,
                support_height,
            ],
        }
    )
    return best_mask, diagnostics


def _coverage_by_vertical_bands(
    mask: np.ndarray,
    envelope: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Cobertura relativa a la envolvente en bandas superior/media/inferior."""
    ys, xs = np.nonzero(envelope > 0)
    empty: Dict[str, Optional[float]] = {
        "upper": None,
        "middle": None,
        "lower": None,
    }
    if xs.size == 0:
        return empty

    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    edges = np.rint(np.linspace(y0, y1, 4)).astype(np.int32)
    result: Dict[str, Optional[float]] = {}
    for name, start, stop in zip(
        ("upper", "middle", "lower"),
        edges[:-1],
        edges[1:],
    ):
        band_domain = envelope[int(start) : int(stop)] > 0
        denominator = int(np.count_nonzero(band_domain))
        if denominator <= 0:
            result[name] = None
            continue
        band_mask = mask[int(start) : int(stop)] > 0
        result[name] = float(np.count_nonzero(band_mask & band_domain) / denominator)
    return result


def _fit_support_inverse_depth_plane(
    depth_mm: np.ndarray,
    sample_mask: np.ndarray,
    minimum_samples: int,
    maximum_samples: int = 120000,
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    """Ajusta robustamente 1/Z=a*x+b*y+c sobre la plataforma visible."""
    valid = (sample_mask > 0) & np.isfinite(depth_mm) & (depth_mm > 1.0)
    ys, xs = np.nonzero(valid)
    total_samples = int(xs.size)
    diagnostics: Dict[str, object] = {
        "available_samples": total_samples,
        "used_samples": 0,
        "inlier_samples": 0,
        "inlier_ratio": 0.0,
        "platform_delta_median_mm": None,
        "platform_delta_mad_mm": None,
    }
    if total_samples < int(minimum_samples):
        diagnostics["reason"] = "insufficient_support_samples"
        return None, diagnostics

    if total_samples > int(maximum_samples):
        indices = np.linspace(
            0,
            total_samples - 1,
            int(maximum_samples),
            dtype=np.int64,
        )
        xs = xs[indices]
        ys = ys[indices]

    h, w = depth_mm.shape
    xn = (xs.astype(np.float64) - 0.5 * (w - 1)) / max(float(w), 1.0)
    yn = (ys.astype(np.float64) - 0.5 * (h - 1)) / max(float(h), 1.0)
    design = np.column_stack([xn, yn, np.ones_like(xn)])
    inverse_depth = 1.0 / depth_mm[ys, xs].astype(np.float64)
    keep = np.ones(inverse_depth.shape, dtype=bool)
    coefficients = np.linalg.lstsq(design, inverse_depth, rcond=None)[0]

    for _ in range(6):
        predicted_inverse = design @ coefficients
        valid_prediction = predicted_inverse > 1e-9
        predicted_depth = np.full(predicted_inverse.shape, np.nan, np.float64)
        predicted_depth[valid_prediction] = 1.0 / predicted_inverse[valid_prediction]
        residual = depth_mm[ys, xs].astype(np.float64) - predicted_depth
        center = float(np.nanmedian(residual[keep]))
        mad = float(np.nanmedian(np.abs(residual[keep] - center)))
        sigma = max(1.0, 1.4826 * mad)
        new_keep = valid_prediction & (np.abs(residual - center) <= 3.0 * sigma)
        if np.count_nonzero(new_keep) < int(minimum_samples):
            break
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep
        coefficients = np.linalg.lstsq(
            design[keep],
            inverse_depth[keep],
            rcond=None,
        )[0]

    inlier_count = int(np.count_nonzero(keep))
    inlier_ratio = float(inlier_count / max(inverse_depth.size, 1))
    diagnostics["used_samples"] = int(inverse_depth.size)
    diagnostics["inlier_samples"] = inlier_count
    diagnostics["inlier_ratio"] = inlier_ratio
    if inlier_count < int(minimum_samples) or inlier_ratio < 0.45:
        diagnostics["reason"] = "support_plane_fit_not_robust"
        return None, diagnostics

    grid_y, grid_x = np.mgrid[0:h, 0:w]
    grid_xn = (grid_x.astype(np.float64) - 0.5 * (w - 1)) / max(float(w), 1.0)
    grid_yn = (grid_y.astype(np.float64) - 0.5 * (h - 1)) / max(float(h), 1.0)
    predicted_inverse_full = coefficients[0] * grid_xn + coefficients[1] * grid_yn + coefficients[2]
    predicted_depth_full = np.full(depth_mm.shape, np.nan, dtype=np.float32)
    good = predicted_inverse_full > 1e-9
    predicted_depth_full[good] = (1.0 / predicted_inverse_full[good]).astype(np.float32)

    platform_delta = predicted_depth_full[ys, xs].astype(np.float64) - depth_mm[ys, xs].astype(
        np.float64
    )
    platform_delta = platform_delta[keep & np.isfinite(platform_delta)]
    delta_median = float(np.median(platform_delta))
    delta_mad = float(np.median(np.abs(platform_delta - delta_median)))
    diagnostics.update(
        {
            "reason": None,
            "coefficients_inverse_depth_normalized": [float(value) for value in coefficients],
            "platform_delta_median_mm": delta_median,
            "platform_delta_mad_mm": delta_mad,
        }
    )
    return predicted_depth_full, diagnostics


def recover_foreground_from_stereo_geometry(
    rgb_foreground: np.ndarray,
    current_depth_mm: np.ndarray,
    stereo_trusted_mask: np.ndarray,
    physical_depth_mask: np.ndarray,
    support_mask: np.ndarray,
    support_model: Dict[str, object],
    roi_mask: np.ndarray,
    adaptive_anchor: np.ndarray,
    minimum_component_area: int,
    minimum_support_samples: int,
    foreground_exclusion_radius_px: int,
    minimum_separation_mm: float,
    maximum_added_ratio: float,
    enabled: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Recupera huecos RGB solo cuando la geometría estéreo los respalda.

    El método es deliberadamente conservador: nunca añade píxeles fuera de la
    envolvente visual y cancela por completo la recuperación si no puede ajustar
    de forma robusta el plano de la plataforma.
    """
    domain = (roi_mask > 0) & (adaptive_anchor > 0)
    seed = (rgb_foreground > 0) & domain
    envelope, recovery_zone, envelope_diag = _object_recovery_envelope(
        seed.astype(np.uint8) * 255,
        support_mask,
        domain.astype(np.uint8) * 255,
        minimum_area=minimum_component_area,
    )
    before_bands = _coverage_by_vertical_bands(
        seed.astype(np.uint8) * 255,
        envelope,
    )
    empty = np.zeros_like(rgb_foreground, dtype=np.uint8)
    empty_float = np.full(current_depth_mm.shape, np.nan, dtype=np.float32)
    diagnostics: Dict[str, object] = {
        "enabled": bool(enabled),
        "status": "not_applied",
        "reason": None,
        "seed_pixels": int(np.count_nonzero(seed)),
        "envelope_pixels": int(np.count_nonzero(envelope)),
        "added_pixels": 0,
        "added_ratio_of_seed": 0.0,
        "minimum_separation_mm": None,
        "coverage_before": before_bands,
        "coverage_after": before_bands,
        "object_envelope": envelope_diag,
    }

    if not enabled:
        diagnostics["reason"] = "disabled_by_argument"
        return seed.astype(np.uint8) * 255, empty, envelope, empty_float, diagnostics
    if support_model.get("status") != "detected" or np.count_nonzero(support_mask) == 0:
        diagnostics["reason"] = "support_not_detected"
        return seed.astype(np.uint8) * 255, empty, envelope, empty_float, diagnostics
    if np.count_nonzero(envelope) == 0:
        diagnostics["reason"] = str(envelope_diag.get("reason") or "visual_envelope_empty")
        return seed.astype(np.uint8) * 255, empty, envelope, empty_float, diagnostics

    support_samples, ring_diag = _support_ring_sample_mask(
        support_mask=support_mask,
        seed_mask=seed.astype(np.uint8) * 255,
        domain_mask=domain.astype(np.uint8) * 255,
        physical_depth_mask=(physical_depth_mask & np.isfinite(current_depth_mm)),
        foreground_exclusion_radius_px=foreground_exclusion_radius_px,
        minimum_samples=minimum_support_samples,
    )
    diagnostics["support_sampling"] = ring_diag
    predicted_support_depth, plane_diag = _fit_support_inverse_depth_plane(
        current_depth_mm,
        support_samples,
        minimum_samples=minimum_support_samples,
    )
    diagnostics["support_plane"] = plane_diag
    if predicted_support_depth is None:
        diagnostics["reason"] = str(plane_diag.get("reason"))
        return seed.astype(np.uint8) * 255, empty, envelope, empty_float, diagnostics

    support_delta_mm = predicted_support_depth.astype(np.float32) - current_depth_mm.astype(
        np.float32
    )
    platform_median = float(plane_diag["platform_delta_median_mm"])
    platform_mad = float(plane_diag["platform_delta_mad_mm"])
    adaptive_separation = max(
        float(minimum_separation_mm),
        platform_median + 1.5 * max(platform_mad, 1.0),
    )
    diagnostics["minimum_separation_mm"] = float(adaptive_separation)

    geometric_gate = (
        (envelope > 0)
        & (recovery_zone > 0)
        & (~seed)
        & (stereo_trusted_mask > 0)
        & physical_depth_mask
        & np.isfinite(support_delta_mm)
        & (support_delta_mm >= adaptive_separation)
    )
    candidate = geometric_gate.copy()
    candidate_u8 = candidate.astype(np.uint8) * 255
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    # El cierre solo une huecos pequeños. Después se reaplican TODAS las
    # condiciones geométricas para no aceptar píxeles no validados.
    candidate = (candidate_u8 > 0) & geometric_gate

    # Solo sobreviven regiones candidatas conectadas con la máscara RGB.
    combined = (seed | candidate).astype(np.uint8)
    count, labels = cv2.connectedComponents(combined, connectivity=8)
    recovered = seed.copy()
    accepted_candidate = np.zeros_like(seed)
    for label in range(1, count):
        component = labels == label
        if np.any(component & seed):
            accepted_candidate |= component & candidate
    recovered |= accepted_candidate

    added_pixels = int(np.count_nonzero(recovered & (~seed)))
    added_ratio = float(added_pixels / max(np.count_nonzero(seed), 1))
    if added_ratio > float(maximum_added_ratio):
        diagnostics.update(
            {
                "reason": "recovery_exceeded_safety_ratio",
                "candidate_pixels": int(np.count_nonzero(candidate)),
                "added_pixels_before_safety_cancel": added_pixels,
                "added_ratio_before_safety_cancel": added_ratio,
            }
        )
        return seed.astype(np.uint8) * 255, empty, envelope, support_delta_mm, diagnostics

    recovery_mask = (recovered & (~seed)).astype(np.uint8) * 255
    recovered_u8 = recovered.astype(np.uint8) * 255
    diagnostics.update(
        {
            "status": "applied" if added_pixels > 0 else "not_needed",
            "reason": None,
            "candidate_pixels": int(np.count_nonzero(candidate)),
            "added_pixels": added_pixels,
            "added_ratio_of_seed": added_ratio,
            "coverage_after": _coverage_by_vertical_bands(recovered_u8, envelope),
        }
    )
    return recovered_u8, recovery_mask, envelope, support_delta_mm, diagnostics


# ---------------------------------------------------------------------------
# Confianza estéreo
# ---------------------------------------------------------------------------


def disparity_to_depth_mm(
    disparity: np.ndarray,
    calib: Dict[str, np.ndarray | float],
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Convierte la disparidad válida a profundidad métrica en milímetros."""
    fx = float(np.asarray(calib["P1"])[0, 0])
    baseline_mm = float(calib["baseline_mm"])

    depth_mm = np.full(
        disparity.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = np.isfinite(disparity) & (disparity > 0.1) & (valid_mask > 0)

    depth_mm[valid] = (fx * baseline_mm) / disparity[valid]

    return depth_mm


def remap_right_to_left(
    right_array: np.ndarray,
    disparity_lr: np.ndarray,
    interpolation: int,
    border_value: float = np.nan,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remuestrea un mapa derecho sobre las coordenadas de la vista izquierda."""
    h, w = disparity_lr.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    sample_x = grid_x - disparity_lr.astype(np.float32)
    sample_y = grid_y

    inside = np.isfinite(sample_x) & (sample_x >= 0.0) & (sample_x <= w - 1.0)

    sampled = cv2.remap(
        right_array,
        sample_x,
        sample_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return sampled, inside


def clahe_gray(image_bgr: np.ndarray) -> np.ndarray:
    """Genera una imagen gris con contraste local para comparar el par rectificado."""
    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    return clahe.apply(gray).astype(np.float32)


def robust_median_filter_float(
    array: np.ndarray,
    valid_mask: np.ndarray,
    kernel_size: int = 9,
) -> np.ndarray:
    """
    Calcula una referencia local suave sobre float32 sin usar medianBlur.

    OpenCV 5 puede limitar cv2.medianBlur a imágenes CV_8U en algunas
    compilaciones optimizadas. Como la disparidad es float32, se utiliza
    una media gaussiana normalizada por la máscara de validez. Esto evita
    convertir la disparidad a 8 bits y conserva su escala en píxeles.
    """
    array = np.asarray(array, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(array)

    valid_values = array[valid]
    fallback = float(np.median(valid_values)) if valid_values.size else 0.0

    # GaussianBlur exige un tamaño impar positivo.
    kernel_size = int(kernel_size)
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1

    weights = valid.astype(np.float32)
    weighted_values = np.where(
        valid,
        array,
        0.0,
    ).astype(np.float32)

    numerator = cv2.GaussianBlur(
        weighted_values,
        (kernel_size, kernel_size),
        0,
        borderType=cv2.BORDER_REFLECT101,
    )
    denominator = cv2.GaussianBlur(
        weights,
        (kernel_size, kernel_size),
        0,
        borderType=cv2.BORDER_REFLECT101,
    )

    local_reference = np.full(
        array.shape,
        fallback,
        dtype=np.float32,
    )
    good = denominator > 1e-6
    local_reference[good] = numerator[good] / denominator[good]

    return local_reference


def local_standard_deviation(gray: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """Desviación estándar local, usada solo para observabilidad estéreo.

    No clasifica formas ni materiales. Una región casi uniforme aporta poca
    evidencia para declarar una contradicción LR como físicamente concluyente.
    """
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = np.asarray(gray, dtype=np.float32)
    mean = cv2.boxFilter(x, cv2.CV_32F, (kernel_size, kernel_size), normalize=True)
    mean_sq = cv2.boxFilter(x * x, cv2.CV_32F, (kernel_size, kernel_size), normalize=True)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(variance).astype(np.float32)


def compute_confidence(
    rect_l: np.ndarray,
    rect_r: np.ndarray,
    disparity_lr: np.ndarray,
    disparity_rl: np.ndarray,
    rect_valid_mask: np.ndarray,
    analysis_mask: np.ndarray,
    lr_abs_tolerance_px: float,
    lr_relative_tolerance: float,
    reverse_ratio_minimum: float,
    reverse_ratio_maximum: float,
    reverse_median_error_factor: float,
    photo_sigma: float,
    photo_hard_threshold: float,
    gradient_sigma: float,
    smoothness_sigma_px: float,
    minimum_confidence: float,
    minimum_disparity_px: float,
    local_observability_window_px: int = 9,
    local_observability_minimum_std: float = 4.0,
    one_sided_minimum_confidence: float = 0.50,
    one_sided_photo_threshold: float = 50.0,
    one_sided_minimum_smoothness_score: float = 0.55,
    lr_contradiction_factor: float = 2.5,
) -> Dict[str, object]:
    """Confianza estéreo con procedencia explícita y LR no binario.

    La confianza directa (fotometría + gradiente + suavidad) permanece separada
    de la consistencia izquierda-derecha. La validación inversa tiene tres
    resultados útiles:

    - fuerte bidireccional: LR/RL coinciden;
    - unilateral débil: la medición directa es plausible pero la inversa no es
      suficientemente observable o solo muestra una discrepancia moderada;
    - contradicha: existe textura local suficiente y LR/RL discrepan de forma
      fuerte, por lo que la observación no debe propagarse como evidencia real.

    Esto evita convertir automáticamente ``LR inconsistente`` en ``superficie
    inexistente`` en zonas oscuras/homogéneas, sin desactivar la protección LR.
    """
    sampled_rl, inside_lr = remap_right_to_left(
        disparity_rl.astype(np.float32),
        disparity_lr,
        interpolation=cv2.INTER_LINEAR,
        border_value=np.nan,
    )

    lr_error = np.abs(disparity_lr - sampled_rl).astype(np.float32)
    lr_tolerance = (
        lr_abs_tolerance_px + lr_relative_tolerance * np.maximum(disparity_lr, 0.0)
    ).astype(np.float32)

    lr_score = np.zeros_like(disparity_lr, dtype=np.float32)
    valid_lr_error = np.isfinite(lr_error) & np.isfinite(lr_tolerance) & (lr_tolerance > 0)
    lr_score[valid_lr_error] = np.exp(
        -0.5 * np.square(lr_error[valid_lr_error] / lr_tolerance[valid_lr_error])
    )

    ratio_mask = (
        (analysis_mask > 0)
        & (rect_valid_mask > 0)
        & np.isfinite(disparity_lr)
        & np.isfinite(disparity_rl)
        & (disparity_lr >= minimum_disparity_px)
        & (disparity_rl >= minimum_disparity_px)
    )
    if np.count_nonzero(ratio_mask) >= 500:
        median_lr = float(np.median(disparity_lr[ratio_mask]))
        median_rl = float(np.median(disparity_rl[ratio_mask]))
        reverse_ratio = median_rl / median_lr if median_lr > 1e-6 else float("nan")
    else:
        median_lr = float("nan")
        median_rl = float("nan")
        reverse_ratio = float("nan")

    reverse_eval_mask = (
        ratio_mask
        & inside_lr
        & np.isfinite(sampled_rl)
        & np.isfinite(lr_error)
        & np.isfinite(lr_tolerance)
    )
    if np.count_nonzero(reverse_eval_mask) >= 500:
        median_reverse_error = float(np.median(lr_error[reverse_eval_mask]))
        median_reverse_tolerance = float(np.median(lr_tolerance[reverse_eval_mask]))
    else:
        median_reverse_error = float("nan")
        median_reverse_tolerance = float("nan")

    reverse_reliable = bool(
        np.isfinite(reverse_ratio)
        and reverse_ratio_minimum <= reverse_ratio <= reverse_ratio_maximum
        and np.isfinite(median_reverse_error)
        and np.isfinite(median_reverse_tolerance)
        and median_reverse_error
        <= reverse_median_error_factor * max(median_reverse_tolerance, 1e-6)
    )

    left_gray = clahe_gray(rect_l)
    right_gray = clahe_gray(rect_r)

    warped_right, photo_inside = remap_right_to_left(
        right_gray,
        disparity_lr,
        interpolation=cv2.INTER_LINEAR,
        border_value=np.nan,
    )
    photo_error = np.abs(left_gray - warped_right).astype(np.float32)
    photo_score = np.zeros_like(disparity_lr, dtype=np.float32)
    valid_photo = np.isfinite(photo_error)
    safe_photo_sigma = max(photo_sigma, 1e-6)
    photo_score[valid_photo] = np.exp(-0.5 * np.square(photo_error[valid_photo] / safe_photo_sigma))

    left_gradient_x = cv2.Sobel(left_gray, cv2.CV_32F, 1, 0, ksize=3)
    right_gradient_x = cv2.Sobel(right_gray, cv2.CV_32F, 1, 0, ksize=3)
    warped_right_gradient, gradient_inside = remap_right_to_left(
        right_gradient_x,
        disparity_lr,
        interpolation=cv2.INTER_LINEAR,
        border_value=np.nan,
    )
    gradient_error = np.abs(left_gradient_x - warped_right_gradient).astype(np.float32)
    gradient_score = np.zeros_like(disparity_lr, dtype=np.float32)
    valid_gradient = np.isfinite(gradient_error)
    safe_gradient_sigma = max(gradient_sigma, 1e-6)
    gradient_score[valid_gradient] = np.exp(
        -0.5 * np.square(gradient_error[valid_gradient] / safe_gradient_sigma)
    )

    base_valid = (
        (rect_valid_mask > 0)
        & (analysis_mask > 0)
        & photo_inside
        & gradient_inside
        & np.isfinite(disparity_lr)
        & (disparity_lr >= minimum_disparity_px)
    )

    local_median = robust_median_filter_float(disparity_lr, base_valid, kernel_size=9)
    smoothness_error = np.abs(disparity_lr - local_median).astype(np.float32)
    smoothness_score = np.zeros_like(disparity_lr, dtype=np.float32)
    valid_smoothness = base_valid & np.isfinite(smoothness_error)
    safe_smoothness_sigma = max(smoothness_sigma_px, 1e-6)
    smoothness_score[valid_smoothness] = np.exp(
        -0.5 * np.square(smoothness_error[valid_smoothness] / safe_smoothness_sigma)
    )

    base_confidence = np.cbrt(
        np.clip(photo_score, 0.0, 1.0)
        * np.clip(gradient_score, 0.0, 1.0)
        * np.clip(smoothness_score, 0.0, 1.0)
    ).astype(np.float32)
    base_confidence[~base_valid] = 0.0

    # Observabilidad LR local. Para declarar una contradicción fuerte se exige
    # textura en AMBAS vistas alrededor de la correspondencia predicha.
    left_texture_std = local_standard_deviation(left_gray, local_observability_window_px)
    right_texture_std = local_standard_deviation(right_gray, local_observability_window_px)
    warped_right_texture_std, texture_inside = remap_right_to_left(
        right_texture_std,
        disparity_lr,
        interpolation=cv2.INTER_LINEAR,
        border_value=np.nan,
    )
    local_texture_std = np.minimum(left_texture_std, warped_right_texture_std).astype(np.float32)
    local_observability_score = np.zeros_like(disparity_lr, dtype=np.float32)
    observable_valid = base_valid & texture_inside & np.isfinite(local_texture_std)
    safe_obs_std = max(float(local_observability_minimum_std), 1e-6)
    local_observability_score[observable_valid] = np.clip(
        local_texture_std[observable_valid] / safe_obs_std, 0.0, 1.0
    )
    local_reverse_observable = observable_valid & (local_texture_std >= safe_obs_std)

    photometrically_plausible = (
        base_valid & np.isfinite(photo_error) & (photo_error <= photo_hard_threshold)
    )
    direct_strong = photometrically_plausible & (base_confidence >= minimum_confidence)

    reverse_available = (
        inside_lr
        & np.isfinite(sampled_rl)
        & (sampled_rl >= minimum_disparity_px)
        & np.isfinite(lr_error)
        & np.isfinite(lr_tolerance)
    )
    lr_consistent = reverse_available & (lr_error <= lr_tolerance)

    if reverse_reliable:
        strong_bidirectional = direct_strong & lr_consistent
        strong_direct_reverse_unavailable = np.zeros_like(base_valid, dtype=bool)
    else:
        strong_bidirectional = np.zeros_like(base_valid, dtype=bool)
        strong_direct_reverse_unavailable = direct_strong

    # Solo una región localmente observable puede convertir una discrepancia LR
    # en contradicción dura. En áreas homogéneas se conserva como incertidumbre.
    contradiction_threshold = np.maximum(
        float(lr_contradiction_factor) * lr_tolerance,
        lr_tolerance + float(lr_abs_tolerance_px),
    )
    contradicted = (
        direct_strong
        & reverse_reliable
        & reverse_available
        & local_reverse_observable
        & (lr_error > contradiction_threshold)
    )

    one_sided_direct = (
        base_valid
        & np.isfinite(photo_error)
        & (photo_error <= min(float(photo_hard_threshold), float(one_sided_photo_threshold)))
        & (base_confidence >= float(one_sided_minimum_confidence))
        & (smoothness_score >= float(one_sided_minimum_smoothness_score))
    )
    weak_one_sided = (
        one_sided_direct
        & (~strong_bidirectional)
        & (~strong_direct_reverse_unavailable)
        & (~contradicted)
    )

    # Estado explícito de procedencia LR.
    # 0 inválido/fuera de dominio
    # 1 fuerte bidireccional
    # 2 fuerte directo porque la inversa global no es fiable
    # 3 unilateral débil, requiere validación posterior
    # 4 contradicción LR local fuerte
    # 5 observación física con evidencia directa insuficiente
    lr_state = np.zeros(disparity_lr.shape, dtype=np.uint8)
    lr_state[base_valid] = 5
    lr_state[contradicted] = 4
    lr_state[weak_one_sided] = 3
    lr_state[strong_direct_reverse_unavailable] = 2
    lr_state[strong_bidirectional] = 1

    trusted = strong_bidirectional | strong_direct_reverse_unavailable
    usable = trusted | weak_one_sided
    consistent_mask = lr_consistent if reverse_reliable else base_valid

    # `confidence` pasa a significar confianza DIRECTA. LR se exporta por
    # separado para evitar penalizarlo una segunda vez en pasos posteriores.
    confidence = base_confidence.copy()
    legacy_combined_confidence = np.sqrt(
        np.clip(base_confidence, 0.0, 1.0) * np.clip(lr_score, 0.0, 1.0)
    ).astype(np.float32)
    legacy_combined_confidence[~base_valid] = 0.0

    return {
        "sampled_disparity_rl": sampled_rl,
        "lr_error": lr_error,
        "lr_tolerance": lr_tolerance,
        "photo_error": photo_error,
        "gradient_error": gradient_error,
        "smoothness_error": smoothness_error,
        "lr_score": lr_score,
        "photo_score": photo_score,
        "gradient_score": gradient_score,
        "smoothness_score": smoothness_score,
        "base_confidence": base_confidence,
        "confidence": confidence,
        "legacy_combined_confidence": legacy_combined_confidence,
        "local_texture_std": local_texture_std,
        "local_observability_score": local_observability_score,
        "local_reverse_observable": (local_reverse_observable.astype(np.uint8) * 255),
        "lr_state": lr_state,
        "geometry_valid": (base_valid.astype(np.uint8) * 255),
        "consistent_mask": (consistent_mask.astype(np.uint8) * 255),
        "trusted_mask": (trusted.astype(np.uint8) * 255),
        "weak_one_sided_mask": (weak_one_sided.astype(np.uint8) * 255),
        "contradicted_mask": (contradicted.astype(np.uint8) * 255),
        "usable_mask": (usable.astype(np.uint8) * 255),
        "reverse_reliable": reverse_reliable,
        "reverse_ratio": reverse_ratio,
        "median_reverse_error_px": median_reverse_error,
        "median_reverse_tolerance_px": median_reverse_tolerance,
        "median_disparity_lr_analysis": median_lr,
        "median_disparity_rl_analysis": median_rl,
    }


# ---------------------------------------------------------------------------
# Estadísticas y control de calidad
# ---------------------------------------------------------------------------



def secondary_roi_evidence(estimator, left, right, valid, object_mask,
                           base_disparity, base_depth, base_sigma, base, calibration, args):
    """Escala adicional diagnóstica. Nunca escribe en los arrays base.

    Ambas cámaras usan exactamente el mismo recorte. El estimador devuelve
    disparidad en píxeles del recorte original: no se vuelve a escalar fx*B/d.
    """
    shape = valid.shape
    result = {key: np.full(shape, np.nan, np.float32) for key in (
        'disparity_px', 'depth_mm', 'uncertainty_mm', 'direct_confidence',
        'photo_error', 'base_delta_mm', 'base_tolerance_mm')}
    for key in ('lr_state', 'candidate_mask', 'base_agrees', 'improves_evidence'):
        result[key] = np.zeros(shape, np.uint8)
    result['roi_xyxy'] = np.empty((0, 4), np.int32)
    result['effective_scale_gain'] = np.array(np.nan, np.float32)
    domain = (object_mask > 0) & (valid > 0)
    useful = (base['usable_mask'] > 0) & np.isfinite(base_sigma)
    # Seleccionar la celda con peor cobertura; una sola inferencia adicional
    # bidireccional por vista mantiene acotado el coste.
    h, w = shape
    choices = []
    for ys in np.array_split(np.arange(h), 3):
        for xs in np.array_split(np.arange(w), 3):
            tile = np.zeros(shape, bool)
            tile[np.ix_(ys, xs)] = True
            target = tile & domain
            count = int(target.sum())
            if count >= 64 and np.count_nonzero(target & useful) / count < 0.75:
                choices.append((int(np.count_nonzero(target & ~useful)), target))
    if not choices:
        return result
    target = max(choices, key=lambda item: item[0])[1]
    yy, xx = np.where(target)
    fb = float(calibration['P1'][0, 0]) * float(calibration['baseline_mm'])
    # Margen epipolar de todo el rango físico, no solo de la predicción base.
    margin = int(np.ceil(fb / max(args.minimum_depth_mm, 1.0))) + 16
    x0, x1 = max(0, int(xx.min()) - margin), min(w, int(xx.max()) + 33)
    y0, y1 = max(0, int(yy.min()) - 32), min(h, int(yy.max()) + 33)
    if (x1-x0) >= w and (y1-y0) >= h:
        return result
    sl = np.s_[y0:y1, x0:x1]
    # Registrar el aumento real, teniendo en cuenta el letterbox del modelo.
    if getattr(estimator, 'target_h', None) and getattr(estimator, 'target_w', None):
        scale_base = min(estimator.target_h/h, estimator.target_w/w)
        scale_roi = min(estimator.target_h/(y1-y0), estimator.target_w/(x1-x0))
        result['effective_scale_gain'] = np.array(scale_roi/scale_base, np.float32)
    dl, dr = estimator.predict_bidirectional(left[sl], right[sl])
    kwargs = {name: getattr(args, name) for name in (
        'lr_abs_tolerance_px', 'lr_relative_tolerance', 'reverse_ratio_minimum',
        'reverse_ratio_maximum', 'reverse_median_error_factor', 'photo_sigma',
        'photo_hard_threshold', 'gradient_sigma', 'smoothness_sigma_px',
        'minimum_confidence', 'minimum_disparity_px', 'local_observability_window_px',
        'local_observability_minimum_std', 'one_sided_minimum_confidence',
        'one_sided_photo_threshold', 'one_sided_minimum_smoothness_score',
        'lr_contradiction_factor')}
    c = compute_confidence(left[sl], right[sl], dl, dr, valid[sl],
                           object_mask[sl], **kwargs)
    depth = disparity_to_depth_mm(dl, calibration, valid[sl])
    sigma_d = (0.55 + 1.75*(1-np.clip(c['base_confidence'],0,1))
               + 0.85*(1-np.clip(c['smoothness_score'],0,1))
               + 0.85*(1-np.clip(c['local_observability_score'],0,1)))
    sigma_d[c['lr_state'] == 3] *= 1.60
    sigma_d[c['trusted_mask'] > 0] *= 0.85
    sigma = fb * sigma_d / np.maximum(dl, 1e-6)**2
    # Excluir el contexto de borde y correspondencias que abandonan el recorte.
    cy, cx = np.indices(dl.shape)
    interior = (cy >= 8) & (cy < dl.shape[0]-8) & (cx >= 8) & (cx < dl.shape[1]-8)
    interior &= (cx-dl >= 8) & (cx-dl < dl.shape[1]-8)
    candidate = (domain[sl] & interior & (c['usable_mask'] > 0)
                 & (c['lr_state'] != 4) & (base['lr_state'][sl] != 4)
                 & np.isfinite(depth) & (depth >= args.minimum_depth_mm)
                 & (depth <= args.maximum_depth_mm) & np.isfinite(sigma))
    delta = np.abs(depth-base_depth[sl])
    tolerance = np.minimum(6.0, 2.5*np.hypot(sigma, base_sigma[sl]))
    agrees = candidate & np.isfinite(tolerance) & (delta <= tolerance)
    better = (candidate & (c['base_confidence'] > base['base_confidence'][sl])
              & (c['photo_error'] < base['photo_error'][sl])
              & np.isfinite(base_sigma[sl]) & (sigma < base_sigma[sl]))
    for key, value in (('disparity_px',dl), ('depth_mm',depth), ('uncertainty_mm',sigma),
                       ('direct_confidence',c['base_confidence']), ('photo_error',c['photo_error']),
                       ('lr_state',c['lr_state']), ('base_delta_mm',delta),
                       ('base_tolerance_mm',tolerance), ('candidate_mask',candidate),
                       ('base_agrees',agrees), ('improves_evidence',better)):
        result[key][sl] = value
    result['roi_xyxy'] = np.array([[x0,y0,x1,y1]], np.int32)
    return result


def compare_roi_sessions(root, output_name):
    """Comparación diagnóstica por pose física, con registro residual de imagen.

    Actualiza todas las sesiones disponibles para no depender del orden de
    ejecución. Sin manifiesto o registro fiable no se declara coherencia.
    """
    manifest = root/'reconstruccion/multisesion/01_mapa_angular/mapa_angular_multisesion.json'
    if not manifest.exists():
        return
    groups = {}
    for rec in json.loads(manifest.read_text(encoding='utf-8'))['records']:
        directory = root/'reconstruccion'/rec['session']/output_name
        path = directory/(rec['stem']+'_roi_evidence.npz')
        image_path = directory/(rec['stem']+'_rect_L.png')
        if path.exists() and image_path.exists():
            groups.setdefault(rec['pose_index'], []).append((rec['session'],path,image_path))
    for members in groups.values():
        for session, path, image_path in members:
            with np.load(path, allow_pickle=False) as data:
                ref = {k:data[k] for k in data.files}
            support = np.zeros(ref['depth_mm'].shape, np.uint8)
            compared = support.copy()
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            used = set()
            for other_session, other_path, other_image_path in members:
                if other_session == session or other_session in used:
                    continue
                other_image = cv2.imread(str(other_image_path), cv2.IMREAD_GRAYSCALE)
                if image is None or other_image is None or image.shape != other_image.shape:
                    continue
                shift, response = cv2.phaseCorrelate(np.float32(image), np.float32(other_image))
                if not np.isfinite(response) or response < 0.5 or not np.all(np.isfinite(shift)) or max(map(abs,shift)) > 4:
                    continue
                with np.load(other_path, allow_pickle=False) as data:
                    if data['depth_mm'].shape != support.shape:
                        continue
                    matrix = np.float32([[1,0,-shift[0]],[0,1,-shift[1]]])
                    def warp(key, fill):
                        return cv2.warpAffine(data[key], matrix, (support.shape[1],support.shape[0]),
                            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
                    depth = warp('depth_mm', float('nan'))
                    sigma = warp('uncertainty_mm', float('nan'))
                    valid = (warp('candidate_mask',0)>0) & (ref['candidate_mask']>0)
                    tolerance = np.minimum(6.0,2.5*np.hypot(sigma,ref['uncertainty_mm']))
                    valid &= np.isfinite(depth) & np.isfinite(tolerance)
                    compared += valid.astype(np.uint8)
                    support += (valid & (np.abs(depth-ref['depth_mm'])<=tolerance)).astype(np.uint8)
                    used.add(other_session)
            # Archivo distinto: el paquete de evidencia de inferencia es inmutable.
            np.savez_compressed(path.with_name(path.stem+'_sessions.npz'),
                compared_sessions=compared, agreeing_sessions=support,
                diagnostic_only=np.array(True))


def centered_roi_mask(
    shape: Tuple[int, int],
    width_fraction: float,
    height_fraction: float,
    center_y_fraction: float,
) -> np.ndarray:
    h, w = shape

    roi_w = max(
        10,
        int(round(w * width_fraction)),
    )
    roi_h = max(
        10,
        int(round(h * height_fraction)),
    )

    center_x = w // 2
    center_y = int(round(h * center_y_fraction))

    x0 = max(0, center_x - roi_w // 2)
    y0 = max(0, center_y - roi_h // 2)
    x1 = min(w, x0 + roi_w)
    y1 = min(h, y0 + roi_h)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def finite_statistics(
    values: np.ndarray,
) -> Dict[str, Optional[float]]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "mean": None,
            "p90": None,
            "max": None,
            "mad": None,
        }

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": median,
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
        "mad": mad,
    }


def normalize_for_vis(
    array: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    invert: bool = False,
) -> np.ndarray:
    if valid_mask is None:
        valid = np.isfinite(array)
    else:
        valid = np.asarray(valid_mask).astype(bool) & np.isfinite(array)

    output = np.zeros(
        array.shape,
        dtype=np.uint8,
    )
    if not np.any(valid):
        return output

    values = array[valid]
    vmin = float(np.percentile(values, 2))
    vmax = float(np.percentile(values, 98))

    if vmax <= vmin:
        vmax = vmin + 1e-6

    scaled = (np.clip(array, vmin, vmax) - vmin) / (vmax - vmin)

    scaled = np.nan_to_num(
        scaled,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    if invert:
        scaled = 1.0 - scaled

    output[valid] = (scaled[valid] * 255.0).astype(np.uint8)

    return output


def classify_local_quality(
    metrics: Dict[str, object],
    minimum_trusted_ratio: float,
    minimum_center_depth_pixels: int,
    minimum_median_confidence: float,
    minimum_lower_band_coverage: float = 0.55,
    maximum_lower_band_drop: float = 0.30,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    usable_ratio = float(
        metrics.get("usable_ratio_of_object", metrics["trusted_ratio_of_object"])
    )
    center_depth_pixels = int(metrics["center_depth"]["count"])
    median_confidence = metrics.get("confidence_usable", metrics["confidence_trusted"])["median"]

    if usable_ratio < minimum_trusted_ratio:
        reasons.append(
            "Cobertura estéreo utilizable insuficiente: "
            f"{usable_ratio:.2%} < "
            f"{minimum_trusted_ratio:.2%}"
        )

    if center_depth_pixels < minimum_center_depth_pixels:
        reasons.append(
            "Muy pocos píxeles de profundidad confiable "
            f"en la ROI central: {center_depth_pixels} < "
            f"{minimum_center_depth_pixels}"
        )

    if median_confidence is None or float(median_confidence) < minimum_median_confidence:
        reasons.append("Confianza mediana insuficiente.")

    recovery = metrics.get("geometric_mask_recovery", {})
    coverage = recovery.get("coverage_after", {}) if isinstance(recovery, dict) else {}
    upper = coverage.get("upper") if isinstance(coverage, dict) else None
    middle = coverage.get("middle") if isinstance(coverage, dict) else None
    lower = coverage.get("lower") if isinstance(coverage, dict) else None
    if upper is not None and middle is not None and lower is not None:
        reference = 0.5 * (float(upper) + float(middle))
        lower_value = float(lower)
        if (
            reference >= 0.60
            and lower_value < float(minimum_lower_band_coverage)
            and (reference - lower_value) > float(maximum_lower_band_drop)
        ):
            reasons.append(
                "Colapso de cobertura en la banda inferior: "
                f"{lower_value:.2%}; referencia superior/media "
                f"{reference:.2%}."
            )

    quality = "accepted" if not reasons else "rejected_local"
    return quality, reasons


def robust_median_mad(
    values: List[float],
) -> Tuple[float, float, float]:
    """Resume los valores finitos mediante mediana y desviación absoluta mediana."""
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    robust_sigma = 1.4826 * mad
    return median, mad, robust_sigma


def apply_session_quality_control(
    records: List[Dict[str, object]],
    absolute_depth_tolerance_mm: float,
    depth_mad_factor: float,
    trusted_ratio_mad_factor: float,
    minimum_trusted_ratio: float,
) -> Dict[str, object]:
    local_candidates = [
        record
        for record in records
        if record["local_quality"] == "accepted"
        and record["metrics"]["center_depth"]["median"] is not None
    ]

    session_summary: Dict[str, object] = {
        "status": "not_enough_views",
        "local_candidates": len(local_candidates),
        "depth_median_mm": None,
        "depth_mad_mm": None,
        "depth_robust_sigma_mm": None,
        "effective_depth_tolerance_mm": None,
        "trusted_ratio_median": None,
        "trusted_ratio_mad": None,
        "effective_minimum_trusted_ratio": minimum_trusted_ratio,
    }

    if len(local_candidates) < 3:
        for record in records:
            if record["local_quality"] == "accepted":
                record["session_quality"] = "accepted_local_only"
                record["session_reasons"] = []
            else:
                record["session_quality"] = "rejected"
                record["session_reasons"] = list(record["local_reasons"])
        return session_summary

    center_depths = [
        float(record["metrics"]["center_depth"]["median"]) for record in local_candidates
    ]
    depth_median, depth_mad, depth_sigma = robust_median_mad(center_depths)
    effective_depth_tolerance = max(
        absolute_depth_tolerance_mm,
        depth_mad_factor * depth_sigma,
    )

    trusted_ratios = [
        float(
            record["metrics"].get(
                "usable_ratio_of_object", record["metrics"]["trusted_ratio_of_object"]
            )
        )
        for record in local_candidates
    ]
    ratio_median, ratio_mad, ratio_sigma = robust_median_mad(trusted_ratios)
    effective_minimum_ratio = max(
        minimum_trusted_ratio,
        ratio_median - trusted_ratio_mad_factor * ratio_sigma,
    )

    session_summary.update(
        {
            "status": "computed",
            "depth_median_mm": depth_median,
            "depth_mad_mm": depth_mad,
            "depth_robust_sigma_mm": depth_sigma,
            "effective_depth_tolerance_mm": (effective_depth_tolerance),
            "trusted_ratio_median": ratio_median,
            "trusted_ratio_mad": ratio_mad,
            "trusted_ratio_robust_sigma": ratio_sigma,
            "effective_minimum_trusted_ratio": (effective_minimum_ratio),
        }
    )

    for record in records:
        reasons = list(record["local_reasons"])

        if record["local_quality"] == "accepted":
            center_depth = float(record["metrics"]["center_depth"]["median"])
            depth_error = abs(center_depth - depth_median)
            record["metrics"]["session_depth_error_mm"] = depth_error

            trusted_ratio = float(
                record["metrics"].get(
                    "usable_ratio_of_object", record["metrics"]["trusted_ratio_of_object"]
                )
            )

            if depth_error > effective_depth_tolerance:
                reasons.append(
                    "Profundidad central incompatible con la sesión: "
                    f"error={depth_error:.3f} mm, "
                    f"límite={effective_depth_tolerance:.3f} mm"
                )

            if trusted_ratio < effective_minimum_ratio:
                reasons.append(
                    "Cobertura confiable atípicamente baja para la sesión: "
                    f"{trusted_ratio:.2%} < "
                    f"{effective_minimum_ratio:.2%}"
                )

        record["session_reasons"] = reasons
        record["session_quality"] = "accepted" if not reasons else "rejected"

    return session_summary


# ---------------------------------------------------------------------------
# Escritura de resultados
# ---------------------------------------------------------------------------


def save_initial_outputs(
    out_dir: Path,
    stem: str,
    rect_l: np.ndarray,
    rect_r: np.ndarray,
    rect_valid_mask: np.ndarray,
    object_mask: np.ndarray,
    object_overlay: np.ndarray,
    disparity_lr: np.ndarray,
    disparity_rl: np.ndarray,
    raw_depth_mm: np.ndarray,
    confident_depth_mm: np.ndarray,
    confidence_data: Dict[str, np.ndarray],
    metrics: Dict[str, object],
    provider_used: str,
    display_depth_min_mm: float,
    display_depth_max_mm: float,
) -> None:
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trusted_mask = confidence_data["trusted_mask"]
    trusted_object_diagnostic_mask = confidence_data.get(
        "trusted_object_diagnostic_mask",
        np.zeros_like(trusted_mask),
    )

    imwrite_checked(
        str(out_dir / f"{stem}_rect_L.png"),
        rect_l,
    )
    imwrite_checked(
        str(out_dir / f"{stem}_rect_R.png"),
        rect_r,
    )
    imwrite_checked(
        str(out_dir / f"{stem}_rect_valid_mask.png"),
        rect_valid_mask,
    )
    imwrite_checked(
        str(out_dir / f"{stem}_object_mask.png"),
        object_mask,
    )
    imwrite_checked(
        str(out_dir / f"{stem}_object_overlay.png"),
        object_overlay,
    )

    # Compatibilidad con las etapas posteriores: valid_mask representa el
    # dominio científico utilizable (fuerte + unilateral débil). La máscara
    # trusted_mask conserva exclusivamente evidencia fuerte.
    imwrite_checked(
        str(out_dir / f"{stem}_valid_mask.png"),
        confidence_data.get("usable_mask", trusted_mask),
    )
    imwrite_checked(
        str(out_dir / f"{stem}_trusted_mask.png"),
        trusted_mask,
    )
    imwrite_checked(
        str(out_dir / f"{stem}_trusted_object_diagnostic_mask.png"),
        trusted_object_diagnostic_mask,
    )
    imwrite_checked(
        str(out_dir / f"{stem}_consistent_mask.png"),
        confidence_data["consistent_mask"],
    )

    imwrite_checked(
        str(out_dir / f"{stem}_weak_one_sided_mask.png"),
        confidence_data["weak_one_sided_mask"],
    )
    imwrite_checked(
        str(out_dir / f"{stem}_contradicted_mask.png"),
        confidence_data["contradicted_mask"],
    )
    imwrite_checked(
        str(out_dir / f"{stem}_usable_mask.png"),
        confidence_data["usable_mask"],
    )
    imwrite_checked(
        str(out_dir / f"{stem}_local_reverse_observable.png"),
        confidence_data["local_reverse_observable"],
    )

    rect_pair = np.hstack([rect_l, rect_r])
    for y in range(
        50,
        rect_pair.shape[0],
        80,
    ):
        cv2.line(
            rect_pair,
            (0, y),
            (rect_pair.shape[1], y),
            (0, 255, 0),
            1,
        )
    imwrite_checked(
        str(out_dir / f"{stem}_rect_pair.png"),
        rect_pair,
    )

    lr_valid = np.isfinite(disparity_lr) & (disparity_lr > 0.1) & (rect_valid_mask > 0)
    rl_valid = np.isfinite(disparity_rl) & (disparity_rl > 0.1) & (rect_valid_mask > 0)

    disp_lr_vis = normalize_for_vis(
        disparity_lr,
        lr_valid,
    )
    disp_rl_vis = normalize_for_vis(
        disparity_rl,
        rl_valid,
    )

    imwrite_checked(
        str(out_dir / f"{stem}_disp_lr_vis.png"),
        cv2.applyColorMap(
            disp_lr_vis,
            cv2.COLORMAP_TURBO,
        ),
    )
    imwrite_checked(
        str(out_dir / f"{stem}_disp_rl_vis.png"),
        cv2.applyColorMap(
            disp_rl_vis,
            cv2.COLORMAP_TURBO,
        ),
    )

    # Nombre histórico conservado.
    imwrite_checked(
        str(out_dir / f"{stem}_disp_vis.png"),
        cv2.applyColorMap(
            disp_lr_vis,
            cv2.COLORMAP_TURBO,
        ),
    )

    lr_error_valid = np.isfinite(confidence_data["lr_error"]) & (rect_valid_mask > 0)
    lr_error_vis = normalize_for_vis(
        confidence_data["lr_error"],
        lr_error_valid,
    )
    imwrite_checked(
        str(out_dir / f"{stem}_lr_error_vis.png"),
        cv2.applyColorMap(
            lr_error_vis,
            cv2.COLORMAP_MAGMA,
        ),
    )

    photo_valid = np.isfinite(confidence_data["photo_error"]) & (rect_valid_mask > 0)
    photo_error_vis = normalize_for_vis(
        confidence_data["photo_error"],
        photo_valid,
    )
    imwrite_checked(
        str(out_dir / f"{stem}_photo_error_vis.png"),
        cv2.applyColorMap(
            photo_error_vis,
            cv2.COLORMAP_MAGMA,
        ),
    )

    gradient_valid = np.isfinite(confidence_data["gradient_error"]) & (object_mask > 0)
    gradient_error_vis = normalize_for_vis(
        confidence_data["gradient_error"],
        gradient_valid,
    )
    imwrite_checked(
        str(out_dir / f"{stem}_gradient_error_vis.png"),
        cv2.applyColorMap(
            gradient_error_vis,
            cv2.COLORMAP_MAGMA,
        ),
    )

    smoothness_valid = np.isfinite(confidence_data["smoothness_error"]) & (object_mask > 0)
    smoothness_error_vis = normalize_for_vis(
        confidence_data["smoothness_error"],
        smoothness_valid,
    )
    imwrite_checked(
        str(out_dir / f"{stem}_smoothness_error_vis.png"),
        cv2.applyColorMap(
            smoothness_error_vis,
            cv2.COLORMAP_MAGMA,
        ),
    )

    confidence_u8 = np.clip(
        confidence_data["confidence"] * 255.0,
        0,
        255,
    ).astype(np.uint8)
    imwrite_checked(
        str(out_dir / f"{stem}_confidence.png"),
        confidence_u8,
    )
    imwrite_checked(
        str(out_dir / f"{stem}_confidence_vis.png"),
        cv2.applyColorMap(
            confidence_u8,
            cv2.COLORMAP_VIRIDIS,
        ),
    )

    raw_depth_valid = np.isfinite(raw_depth_mm)
    raw_depth_vis = normalize_for_vis(
        raw_depth_mm,
        raw_depth_valid,
        invert=True,
    )
    imwrite_checked(
        str(out_dir / f"{stem}_depth_raw_vis.png"),
        cv2.applyColorMap(
            raw_depth_vis,
            cv2.COLORMAP_VIRIDIS,
        ),
    )

    confident_depth_valid = np.isfinite(confident_depth_mm)
    confident_depth_vis = normalize_for_vis(
        confident_depth_mm,
        confident_depth_valid,
        invert=True,
    )
    imwrite_checked(
        str(out_dir / f"{stem}_depth_vis.png"),
        cv2.applyColorMap(
            confident_depth_vis,
            cv2.COLORMAP_VIRIDIS,
        ),
    )

    imwrite_checked(
        str(out_dir / f"{stem}_depth_raw_absolute_mm_vis.png"),
        fixed_depth_visualization(
            raw_depth_mm,
            raw_depth_valid,
            display_depth_min_mm,
            display_depth_max_mm,
        ),
    )
    imwrite_checked(
        str(out_dir / f"{stem}_depth_absolute_mm_vis.png"),
        fixed_depth_visualization(
            confident_depth_mm,
            confident_depth_valid,
            display_depth_min_mm,
            display_depth_max_mm,
        ),
    )

    np.save(
        str(out_dir / f"{stem}_disparity.npy"),
        disparity_lr,
    )
    np.save(
        str(out_dir / f"{stem}_disparity_lr.npy"),
        disparity_lr,
    )
    np.save(
        str(out_dir / f"{stem}_disparity_rl.npy"),
        disparity_rl,
    )
    np.save(
        str(out_dir / f"{stem}_lr_error.npy"),
        confidence_data["lr_error"],
    )
    np.save(
        str(out_dir / f"{stem}_photo_error.npy"),
        confidence_data["photo_error"],
    )
    np.save(
        str(out_dir / f"{stem}_confidence.npy"),
        confidence_data["confidence"],
    )

    np.save(
        str(out_dir / f"{stem}_base_confidence.npy"),
        confidence_data["base_confidence"],
    )
    np.save(
        str(out_dir / f"{stem}_lr_score.npy"),
        confidence_data["lr_score"],
    )
    np.save(
        str(out_dir / f"{stem}_lr_state.npy"),
        confidence_data["lr_state"],
    )
    np.save(
        str(out_dir / f"{stem}_local_observability_score.npy"),
        confidence_data["local_observability_score"],
    )
    np.save(
        str(out_dir / f"{stem}_local_texture_std.npy"),
        confidence_data["local_texture_std"],
    )
    np.save(
        str(out_dir / f"{stem}_legacy_combined_confidence.npy"),
        confidence_data["legacy_combined_confidence"],
    )
    np.save(
        str(out_dir / f"{stem}_depth_raw_mm.npy"),
        raw_depth_mm,
    )
    np.save(
        str(out_dir / f"{stem}_depth_before_session_qc_mm.npy"),
        confident_depth_mm,
    )

    # Se escribe provisionalmente; el control de sesión puede anularlo.
    np.save(
        str(out_dir / f"{stem}_depth_mm.npy"),
        confident_depth_mm,
    )

    stats = {
        "provider_used": provider_used,
        "local_quality": metrics["local_quality"],
        "local_reasons": metrics["local_reasons"],
        "metrics": metrics,
    }
    (out_dir / f"{stem}_stats.json").write_text(
        json.dumps(
            stats,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def finalize_view_output(
    out_dir: Path,
    record: Dict[str, object],
    keep_rejected_depth: bool,
) -> None:
    stem = str(record["stem"])
    accepted = record["session_quality"] in ("accepted", "accepted_local_only")

    before_qc_path = out_dir / f"{stem}_depth_before_session_qc_mm.npy"
    trusted_mask_path = out_dir / f"{stem}_trusted_mask.png"
    usable_mask_path = out_dir / f"{stem}_usable_mask.png"

    depth_before_qc = np.load(str(before_qc_path)).astype(np.float32)
    trusted_mask = cv2.imread(str(trusted_mask_path), cv2.IMREAD_GRAYSCALE)
    usable_mask = cv2.imread(str(usable_mask_path), cv2.IMREAD_GRAYSCALE)

    if trusted_mask is None:
        trusted_mask = np.zeros(depth_before_qc.shape, dtype=np.uint8)
    if usable_mask is None:
        usable_mask = trusted_mask.copy()

    # V2.3.0: el QC de vista/sesión queda como diagnóstico.
    # Nunca se destruye una observación estéreo que ya pasó los filtros por
    # píxel. La aceptación final se resolverá en los pasos posteriores.
    final_depth = depth_before_qc
    final_mask = usable_mask

    np.save(
        str(out_dir / f"{stem}_depth_mm.npy"),
        final_depth,
    )
    imwrite_checked(
        str(out_dir / f"{stem}_valid_mask.png"),
        final_mask,
    )

    status_payload = {
        "stem": stem,
        "local_quality": record["local_quality"],
        "local_reasons": record["local_reasons"],
        "session_quality": record["session_quality"],
        "session_reasons": record["session_reasons"],
        "depth_enabled_for_downstream": True,
        "session_qc_is_advisory": True,
        "scientific_depth_depends_on_object_mask": False,
        "keep_rejected_depth": keep_rejected_depth,
    }
    (out_dir / f"{stem}_quality.json").write_text(
        json.dumps(
            status_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    stats_path = out_dir / f"{stem}_stats.json"
    stats_payload = json.loads(
        stats_path.read_text(
            encoding="utf-8",
        )
    )
    stats_payload.update(
        {
            "session_quality": record["session_quality"],
            "session_reasons": record["session_reasons"],
            "depth_enabled_for_downstream": bool(accepted or keep_rejected_depth),
        }
    )
    stats_path.write_text(
        json.dumps(
            stats_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_session_csv(
    path: Path,
    records: List[Dict[str, object]],
) -> None:
    fieldnames = [
        "stem",
        "local_quality",
        "session_quality",
        "trusted_ratio_object",
        "trusted_ratio_frame",
        "reverse_reliable",
        "reverse_ratio",
        "median_confidence",
        "center_depth_pixels",
        "center_depth_median_mm",
        "center_depth_mad_mm",
        "session_depth_error_mm",
        "local_reasons",
        "session_reasons",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for record in records:
            metrics = record["metrics"]
            writer.writerow(
                {
                    "stem": record["stem"],
                    "local_quality": record["local_quality"],
                    "session_quality": record["session_quality"],
                    "trusted_ratio_object": metrics["trusted_ratio_of_object"],
                    "trusted_ratio_frame": metrics["trusted_ratio_of_rect_valid"],
                    "reverse_reliable": metrics["reverse_reliable"],
                    "reverse_ratio": metrics["reverse_ratio"],
                    "median_confidence": metrics["confidence_trusted"]["median"],
                    "center_depth_pixels": metrics["center_depth"]["count"],
                    "center_depth_median_mm": metrics["center_depth"]["median"],
                    "center_depth_mad_mm": metrics["center_depth"]["mad"],
                    "session_depth_error_mm": metrics.get("session_depth_error_mm"),
                    "local_reasons": " | ".join(record["local_reasons"]),
                    "session_reasons": " | ".join(record["session_reasons"]),
                }
            )


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construye las opciones de línea de comandos de este paso."""
    parser = argparse.ArgumentParser(
        description=(
            "CREStereo ONNX local con fondo vacío, confianza híbrida "
            "y control de calidad de sesión."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Ruta al modelo CREStereo .onnx",
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Carpeta raíz Proyecto_3D",
    )
    parser.add_argument(
        "--object",
        default="cubo",
        help="Objeto: cubo/cilindro/piramide",
    )
    parser.add_argument(
        "--session",
        default="S01",
        help="Sesión, por ejemplo S01",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        choices=(
            "auto",
            "cuda",
            "directml",
            "cpu",
        ),
    )
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Con provider auto/cuda, cancela si CUDAExecutionProvider no queda "
            "activo. Evita una ejecución CPU silenciosa. --provider cpu o "
            "--no-require-cuda permiten una prueba deliberada sin CUDA."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Máximo de pares; 0 = todos",
    )
    parser.add_argument(
        "--output-name",
        default="02_estimacion_profundidad",
        help="Nombre de la carpeta de salida",
    )
    parser.add_argument(
        "--calibration-dir",
        default="",
        help="Ruta OBLIGATORIA a la calibración estéreo vigente del sistema.",
    )
    parser.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Limpia la salida antes de una corrida completa.",
    )

    # Fondo vacío y segmentación.
    parser.add_argument(
        "--background-dir",
        default="",
        help=(
            "Carpeta con background_left.png y background_right.png. "
            "Ruta OBLIGATORIA al fondo vacío vigente del sistema"
        ),
    )
    parser.add_argument(
        "--background-already-rectified",
        action="store_true",
    )
    parser.add_argument(
        "--background-minimum-threshold",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--background-mad-factor",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--background-minimum-area",
        type=int,
        default=900,
    )
    parser.add_argument(
        "--foreground-depth-separation-mm",
        type=float,
        default=18.0,
        help=("Separación mínima respecto al fondo vacío para considerar " "un píxel como objeto"),
    )
    parser.add_argument(
        "--geometric-mask-recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Recupera huecos de la máscara RGB únicamente cuando la "
            "geometría estéreo y el plano de la plataforma los respaldan."
        ),
    )
    parser.add_argument(
        "--geometry-recovery-min-separation-mm",
        type=float,
        default=6.0,
        help=(
            "Separación mínima por delante del plano local de la plataforma "
            "para recuperar un píxel visualmente ausente."
        ),
    )
    parser.add_argument(
        "--geometry-recovery-min-support-samples",
        type=int,
        default=3000,
        help=(
            "Muestras mínimas del anillo exterior de la plataforma para " "ajustar el plano local."
        ),
    )
    parser.add_argument(
        "--geometry-recovery-exclusion-radius-px",
        type=int,
        default=20,
        help="Margen excluido alrededor de la semilla RGB al ajustar la plataforma.",
    )
    parser.add_argument(
        "--geometry-recovery-max-added-ratio",
        type=float,
        default=0.75,
        help=(
            "Límite de seguridad: máximo de píxeles recuperados respecto a "
            "los píxeles de la semilla RGB."
        ),
    )

    # Consistencia estéreo.
    parser.add_argument(
        "--minimum-disparity-px",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--lr-abs-tolerance-px",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--lr-relative-tolerance",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--reverse-ratio-minimum",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--reverse-ratio-maximum",
        type=float,
        default=1.35,
    )
    parser.add_argument(
        "--reverse-median-error-factor",
        type=float,
        default=2.0,
        help=(
            "La predicción inversa solo se usa cuando su error mediano "
            "no supera este múltiplo de la tolerancia mediana"
        ),
    )
    parser.add_argument(
        "--photo-sigma",
        type=float,
        default=25.0,
    )
    parser.add_argument(
        "--photo-hard-threshold",
        type=float,
        default=70.0,
    )

    parser.add_argument(
        "--gradient-sigma",
        type=float,
        default=65.0,
    )
    parser.add_argument(
        "--smoothness-sigma-px",
        type=float,
        default=18.0,
    )
    parser.add_argument(
        "--minimum-confidence",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--local-observability-window-px",
        type=int,
        default=9,
        help="Ventana local para medir si LR/RL es observable por textura.",
    )
    parser.add_argument(
        "--local-observability-minimum-std",
        type=float,
        default=4.0,
        help=(
            "Desviación estándar local mínima (0..255) exigida en ambas vistas "
            "para tratar una discrepancia LR como contradicción fuerte."
        ),
    )
    parser.add_argument(
        "--one-sided-minimum-confidence",
        type=float,
        default=0.50,
        help="Confianza directa mínima para conservar una observación unilateral débil.",
    )
    parser.add_argument(
        "--one-sided-photo-threshold",
        type=float,
        default=50.0,
        help="Error fotométrico máximo para una observación unilateral débil.",
    )
    parser.add_argument(
        "--one-sided-minimum-smoothness-score",
        type=float,
        default=0.55,
        help="Suavidad local mínima para una observación unilateral débil.",
    )
    parser.add_argument(
        "--lr-contradiction-factor",
        type=float,
        default=2.5,
        help=(
            "Una discrepancia LR solo se considera contradicción fuerte cuando "
            "supera este múltiplo de la tolerancia y la región es observable."
        ),
    )

    # Rango físico.
    parser.add_argument(
        "--minimum-depth-mm",
        type=float,
        default=150.0,
    )
    parser.add_argument(
        "--maximum-depth-mm",
        type=float,
        default=1200.0,
    )

    # Dominio de producto: por defecto no existe recorte central.
    parser.add_argument(
        "--roi-mode",
        choices=("adaptive", "legacy-centered"),
        default="adaptive",
    )
    parser.add_argument("--rect-valid-erosion-px", type=int, default=2)
    parser.add_argument("--adaptive-roi-margin-fraction", type=float, default=0.18)
    parser.add_argument("--adaptive-roi-minimum-margin-px", type=int, default=24)
    parser.add_argument("--adaptive-roi-minimum-component-area", type=int, default=900)

    # Compatibilidad histórica; solo se usan con --roi-mode legacy-centered.
    parser.add_argument("--roi-width-fraction", type=float, default=0.58)
    parser.add_argument("--roi-height-fraction", type=float, default=0.72)
    parser.add_argument("--roi-center-y-fraction", type=float, default=0.52)

    # Calidad local.
    parser.add_argument(
        "--minimum-trusted-ratio",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--minimum-center-depth-pixels",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--minimum-median-confidence",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--minimum-lower-band-coverage",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--maximum-lower-band-drop",
        type=float,
        default=0.30,
    )

    # V2.2.4 — auditoría/corrección epipolar usando el fondo vacío.
    parser.add_argument(
        "--epipolar-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Verifica la alineación vertical después de rectificar. Si no puede "
            "validarse, el paso se detiene antes de ejecutar CREStereo."
        ),
    )
    parser.add_argument(
        "--epipolar-auto-correct",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compatibilidad experimental. Por defecto NO modifica imágenes ya "
            "rectificadas: una discrepancia del auditor se reporta y debe resolverse "
            "recalibrando, no deformando la imagen derecha sin actualizar P1/P2/Q."
        ),
    )
    parser.add_argument("--epipolar-sift-features", type=int, default=12000)
    parser.add_argument("--epipolar-sift-contrast", type=float, default=0.005)
    parser.add_argument("--epipolar-match-ratio", type=float, default=0.85)
    parser.add_argument("--epipolar-match-max-vertical-px", type=float, default=60.0)
    parser.add_argument("--epipolar-ransac-threshold-px", type=float, default=1.75)
    parser.add_argument("--epipolar-ransac-iterations", type=int, default=6000)
    parser.add_argument("--epipolar-minimum-matches", type=int, default=45)
    parser.add_argument("--epipolar-minimum-inliers", type=int, default=35)
    parser.add_argument("--epipolar-minimum-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--epipolar-minimum-x-span-fraction", type=float, default=0.45)
    parser.add_argument("--epipolar-maximum-post-residual-p95-px", type=float, default=1.75)
    parser.add_argument("--epipolar-maximum-correction-px", type=float, default=8.0)
    parser.add_argument("--epipolar-maximum-vertical-scale-delta", type=float, default=0.025)
    parser.add_argument("--epipolar-no-correction-max-px", type=float, default=1.50)
    parser.add_argument(
        "--epipolar-audit-strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Si está activo, una discrepancia sistemática detectada por SIFT detiene "
            "el paso. Por defecto el auditor es diagnóstico: nunca altera P1/P2/Q ni "
            "las imágenes rectificadas."
        ),
    )

    # Calidad de sesión.
    parser.add_argument(
        "--session-depth-tolerance-mm",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--session-depth-mad-factor",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--session-trusted-ratio-mad-factor",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--keep-rejected-depth",
        action="store_true",
        help=("Conserva profundidad en vistas rechazadas. " "Solo para diagnóstico."),
    )

    return parser


def main() -> None:
    """Estima profundidad por vista y guarda los productos y controles de calidad."""
    args = build_parser().parse_args()

    root = Path(args.root).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    object_name = args.object.strip().lower()
    session_name = args.session.strip().upper()

    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo ONNX: {model_path}")

    calibration_dir = resolve_calibration_dir(
        root,
        object_name=object_name,
        explicit=(args.calibration_dir or None),
        required_files=("stereo_initial.yaml", "rectification_maps.npz"),
    )
    calibration = load_calibration(calibration_dir)

    if not args.background_dir:
        raise ValueError(
            "Se requiere --background-dir explícito. "
            "No se reutilizan fondos de ejecuciones anteriores."
        )
    background_dir = Path(args.background_dir).expanduser().resolve()
    background_left_path = background_dir / "background_left.png"
    background_right_path = background_dir / "background_right.png"

    background_left = cv2.imread(
        str(background_left_path),
        cv2.IMREAD_COLOR,
    )
    background_right = cv2.imread(
        str(background_right_path),
        cv2.IMREAD_COLOR,
    )

    if background_left is None:
        raise FileNotFoundError(f"No se pudo leer: {background_left_path}")
    if background_right is None:
        raise FileNotFoundError(f"No se pudo leer: {background_right_path}")

    if args.background_already_rectified:
        validate_calibrated_image_shape(background_left, calibration, "Fondo izquierdo rectificado")
        validate_calibrated_image_shape(background_right, calibration, "Fondo derecho rectificado")
        rect_background_left = background_left
        rect_background_right = background_right
    else:
        (
            rect_background_left,
            rect_background_right,
        ) = rectify_pair(
            background_left,
            background_right,
            calibration,
        )

    # V2.2.4: la validez de CREStereo depende de que las correspondencias
    # estén epipolarmente alineadas. Auditar ANTES de cargar el modelo evita
    # horas de cómputo sobre una rectificación físicamente obsoleta.
    rect_background_right_pre_epipolar = rect_background_right.copy()
    epipolar_diagnostics = estimate_epipolar_vertical_correction(
        rect_background_left,
        rect_background_right,
        calibration,
        args,
    )
    if bool(args.epipolar_audit) and str(epipolar_diagnostics.get("status")) == "failed":
        raise RuntimeError(
            "AUDITORÍA EPIPOLAR RECHAZADA. "
            + str(epipolar_diagnostics.get("reason") or "")
            + "\nDiagnóstico: "
            + json.dumps(epipolar_diagnostics, ensure_ascii=False)
        )

    epipolar_model = epipolar_diagnostics if bool(epipolar_diagnostics.get("applied")) else None
    rect_background_right = apply_vertical_epipolar_correction(
        rect_background_right,
        epipolar_model,
        interpolation=cv2.INTER_LINEAR,
    )

    valid_left_rect, valid_right_rect = build_rectified_valid_components(calibration)
    valid_right_rect = apply_vertical_epipolar_correction(
        valid_right_rect,
        epipolar_model,
        interpolation=cv2.INTER_NEAREST,
    )
    rect_valid_mask_static = ((valid_left_rect >= 254) & (valid_right_rect >= 254)).astype(
        np.uint8
    ) * 255

    if bool(args.epipolar_audit):
        print("\n[EPIPOLAR] Auditoría del fondo rectificado:")
        print(
            f"  matches={epipolar_diagnostics.get('candidate_matches')} | "
            f"inliers={epipolar_diagnostics.get('inliers')} | "
            f"ratio={float(epipolar_diagnostics.get('inlier_ratio', 0.0)):.3f}"
        )
        print(
            "  error vertical sistemático: "
            f"rango={float(epipolar_diagnostics.get('systematic_vertical_range_px', 0.0)):.2f}px | "
            f"máx corrección={float(epipolar_diagnostics.get('maximum_vertical_correction_px', 0.0)):.2f}px"
        )
        print(
            "  residual robusto del modelo: "
            f"P95={float(epipolar_diagnostics.get('post_model_residual_p95_px', 0.0)):.2f}px"
        )
        if epipolar_model is not None:
            print(
                "  CORRECCIÓN ACTIVADA: x'=x; "
                f"y'=(1+{float(epipolar_model['b']):.6f})y + "
                f"{float(epipolar_model['a']):.6f}x + "
                f"{float(epipolar_model['c']):.3f}"
            )
        else:
            print("  Rectificación aceptada sin corrección adicional.")

    left_dir = root / "capturas" / session_name / "izquierda"
    right_dir = root / "capturas" / session_name / "derecha"

    if not left_dir.exists() or not right_dir.exists():
        raise FileNotFoundError("No existe la sesión esperada:\n" f"{left_dir}\n{right_dir}")

    pairs = pair_stems(
        left_dir,
        right_dir,
    )
    left_stems = {p.name.replace("_L.png", "") for p in left_dir.glob("*.png")}
    right_stems = {p.name.replace("_R.png", "") for p in right_dir.glob("*.png")}
    unmatched_left = sorted(left_stems - right_stems)
    unmatched_right = sorted(right_stems - left_stems)
    if unmatched_left or unmatched_right:
        print(
            "[WARN] Hay capturas sin pareja estéreo: "
            f"solo izquierda={len(unmatched_left)}, solo derecha={len(unmatched_right)}."
        )
    if not pairs:
        raise RuntimeError("No se encontraron pares izquierda-derecha.")

    if args.limit > 0:
        pairs = pairs[: args.limit]

    estimator = CREStereoONNX(
        str(model_path),
        preferred_provider=args.provider,
        require_cuda=bool(args.require_cuda and args.provider in ("auto", "cuda")),
    )
    provider_used = estimator.session.get_providers()[0]

    diagnostic_run = args.limit > 0
    output_name = args.output_name
    if diagnostic_run and output_name == "02_estimacion_profundidad":
        output_name = f"{output_name}_diagnostico_limit{args.limit}"
    output_dir = root / "reconstruccion" / session_name / output_name
    prepare_output_directory(
        output_dir,
        clean=bool(args.clean_output and not diagnostic_run),
    )
    imwrite_checked(
        str(output_dir / "background_rectified_left.png"),
        rect_background_left,
    )
    imwrite_checked(
        str(output_dir / "background_rectified_right_pre_epipolar.png"),
        rect_background_right_pre_epipolar,
    )
    imwrite_checked(
        str(output_dir / "background_rectified_right.png"),
        rect_background_right,
    )
    (output_dir / "epipolar_audit_v2_2_4.json").write_text(
        json.dumps(epipolar_diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Calculando profundidad geométrica del fondo vacío...")
    background_valid_mask = rect_valid_mask_static.copy()
    background_disparity_lr = estimator(
        rect_background_left,
        rect_background_right,
    )
    background_disparity_lr[background_valid_mask == 0] = np.nan

    # V2.3.0: producto científico reutilizable por 03.
    np.save(
        str(output_dir / "background_disparity_lr.npy"),
        background_disparity_lr.astype(np.float32),
    )

    background_depth_mm = disparity_to_depth_mm(
        background_disparity_lr,
        calibration,
        background_valid_mask,
    )
    np.save(
        str(output_dir / "background_depth_mm.npy"),
        background_depth_mm,
    )
    background_depth_valid = np.isfinite(background_depth_mm)
    imwrite_checked(
        str(output_dir / "background_depth_valid_mask.png"),
        background_depth_valid.astype(np.uint8) * 255,
    )
    display_depth_min_mm, display_depth_max_mm = derive_depth_display_range(
        background_depth_mm, args.minimum_depth_mm, args.maximum_depth_mm
    )
    imwrite_checked(
        str(output_dir / "background_depth_absolute_mm_vis.png"),
        fixed_depth_visualization(
            background_depth_mm, background_depth_valid, display_depth_min_mm, display_depth_max_mm
        ),
    )
    support_mask, support_model = estimate_turntable_support_mask(
        background_depth_mm, background_valid_mask, background_bgr=rect_background_left
    )
    imwrite_checked(str(output_dir / "background_support_mask.png"), support_mask)
    support_overlay = rect_background_left.copy()
    support_contours, _ = cv2.findContours(support_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(support_overlay, support_contours, -1, (255, 0, 255), 3)
    imwrite_checked(str(output_dir / "background_support_overlay.png"), support_overlay)
    (output_dir / "background_support_model.json").write_text(
        json.dumps(
            {
                **support_model,
                "display_depth_range_mm": [
                    float(display_depth_min_mm),
                    float(display_depth_max_mm),
                ],
                "role": "hardware_prior_visual_edge_validated_depth_diagnostic_fail_safe",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if support_model.get("status") == "detected":
        selected = support_model.get("image_candidate", {}).get("selected") or support_model.get(
            "depth_candidate", {}
        ).get("ellipse")
        print(
            "Plataforma detectada: "
            f"fuente={support_model.get('selected_source')} | "
            f"confianza={float(support_model.get('confidence_score', 0.0)):.3f} | "
            f"IoU imagen/profundidad={support_model.get('image_depth_iou')}"
        )
    else:
        print(
            "[WARN] No se obtuvo una plataforma suficientemente confiable. "
            "03 continuará SIN prior duro de soporte para evitar recortar objeto real."
        )

    records: List[Dict[str, object]] = []

    print(
        "\nIniciando confianza híbrida con fondo vacío. "
        "Cada vista requiere dos inferencias ONNX; la inversa se usa "
        "solo cuando es fiable."
    )

    for index, (stem, left_path, right_path) in enumerate(
        pairs,
        start=1,
    ):
        image_l = cv2.imread(
            str(left_path),
            cv2.IMREAD_COLOR,
        )
        image_r = cv2.imread(
            str(right_path),
            cv2.IMREAD_COLOR,
        )

        if image_l is None or image_r is None:
            print(f"[WARN] No se pudo leer: {stem}")
            continue

        rect_l, rect_r = rectify_pair(
            image_l,
            image_r,
            calibration,
        )
        rect_r = apply_vertical_epipolar_correction(
            rect_r,
            epipolar_model,
            interpolation=cv2.INTER_LINEAR,
        )
        rect_valid_mask = rect_valid_mask_static.copy()

        if args.roi_mode == "adaptive":
            roi_mask = safe_rectification_domain(
                rect_valid_mask,
                erosion_px=args.rect_valid_erosion_px,
            )
        else:
            roi_mask = centered_roi_mask(
                rect_l.shape[:2],
                args.roi_width_fraction,
                args.roi_height_fraction,
                args.roi_center_y_fraction,
            )
            roi_mask = cv2.bitwise_and(roi_mask, rect_valid_mask)

        rgb_object_mask, rgb_object_overlay, background_threshold = (
            segment_foreground_from_background(
                rect_left=rect_l,
                background_left=rect_background_left,
                rect_valid_mask=rect_valid_mask,
                roi_mask=roi_mask,
                minimum_threshold=(args.background_minimum_threshold),
                mad_factor=args.background_mad_factor,
                minimum_area=args.background_minimum_area,
            )
        )
        imwrite_checked(
            str(output_dir / f"{stem}_object_mask_rgb.png"),
            rgb_object_mask,
        )
        imwrite_checked(
            str(output_dir / f"{stem}_object_overlay_rgb.png"),
            rgb_object_overlay,
        )

        disparity_lr, disparity_rl = estimator.predict_bidirectional(
            rect_l,
            rect_r,
        )

        disparity_lr[rect_valid_mask == 0] = np.nan
        disparity_rl[rect_valid_mask == 0] = np.nan

        raw_depth_mm = disparity_to_depth_mm(
            disparity_lr,
            calibration,
            rect_valid_mask,
        )

        visual_object_mask, depth_foreground_mask = refine_foreground_with_background_depth(
            rgb_foreground=rgb_object_mask,
            current_depth_mm=raw_depth_mm,
            background_depth_mm=background_depth_mm,
            rect_valid_mask=rect_valid_mask,
            roi_mask=roi_mask,
            minimum_depth_separation_mm=(args.foreground_depth_separation_mm),
            minimum_area=args.background_minimum_area,
        )
        depth_background_qc = background_depth_consistency_diagnostics(
            raw_depth_mm, background_depth_mm, rect_valid_mask, rgb_object_mask
        )

        adaptive_anchor, adaptive_visual_candidate, adaptive_roi_diag = adaptive_visual_anchor(
            rect_l,
            rect_background_left,
            roi_mask,
            minimum_component_area=args.adaptive_roi_minimum_component_area,
            margin_fraction=args.adaptive_roi_margin_fraction,
            minimum_margin_px=args.adaptive_roi_minimum_margin_px,
        )

        imwrite_checked(
            str(output_dir / f"{stem}_adaptive_roi.png"),
            roi_mask,
        )
        imwrite_checked(
            str(output_dir / f"{stem}_adaptive_anchor.png"),
            adaptive_anchor,
        )
        imwrite_checked(
            str(output_dir / f"{stem}_adaptive_visual_candidate.png"),
            adaptive_visual_candidate,
        )

        adaptive_overlay = rect_l.copy()
        contours, _ = cv2.findContours(adaptive_anchor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(adaptive_overlay, contours, -1, (0, 255, 255), 3)
        imwrite_checked(
            str(output_dir / f"{stem}_adaptive_anchor_overlay.png"),
            adaptive_overlay,
        )
        imwrite_checked(
            str(output_dir / f"{stem}_depth_foreground_mask.png"),
            depth_foreground_mask,
        )

        confidence_data = compute_confidence(
            rect_l=rect_l,
            rect_r=rect_r,
            disparity_lr=disparity_lr,
            disparity_rl=disparity_rl,
            rect_valid_mask=rect_valid_mask,
            analysis_mask=roi_mask,
            lr_abs_tolerance_px=args.lr_abs_tolerance_px,
            lr_relative_tolerance=(args.lr_relative_tolerance),
            reverse_ratio_minimum=(args.reverse_ratio_minimum),
            reverse_ratio_maximum=(args.reverse_ratio_maximum),
            reverse_median_error_factor=(args.reverse_median_error_factor),
            photo_sigma=args.photo_sigma,
            photo_hard_threshold=(args.photo_hard_threshold),
            gradient_sigma=args.gradient_sigma,
            smoothness_sigma_px=(args.smoothness_sigma_px),
            minimum_confidence=(args.minimum_confidence),
            minimum_disparity_px=(args.minimum_disparity_px),
            local_observability_window_px=(args.local_observability_window_px),
            local_observability_minimum_std=(args.local_observability_minimum_std),
            one_sided_minimum_confidence=(args.one_sided_minimum_confidence),
            one_sided_photo_threshold=(args.one_sided_photo_threshold),
            one_sided_minimum_smoothness_score=(args.one_sided_minimum_smoothness_score),
            lr_contradiction_factor=(args.lr_contradiction_factor),
        )

        physical_depth_mask = (
            np.isfinite(raw_depth_mm)
            & (raw_depth_mm >= args.minimum_depth_mm)
            & (raw_depth_mm <= args.maximum_depth_mm)
        )

        stereo_trusted_before_object = np.asarray(
            confidence_data["trusted_mask"],
            dtype=np.uint8,
        ).copy()
        (
            object_mask,
            geometry_recovery_mask,
            geometry_recovery_envelope,
            support_plane_delta_mm,
            geometry_recovery_diag,
        ) = recover_foreground_from_stereo_geometry(
            rgb_foreground=visual_object_mask,
            current_depth_mm=raw_depth_mm,
            stereo_trusted_mask=stereo_trusted_before_object,
            physical_depth_mask=physical_depth_mask,
            support_mask=support_mask,
            support_model=support_model,
            roi_mask=roi_mask,
            adaptive_anchor=adaptive_anchor,
            minimum_component_area=args.background_minimum_area,
            minimum_support_samples=(args.geometry_recovery_min_support_samples),
            foreground_exclusion_radius_px=(args.geometry_recovery_exclusion_radius_px),
            minimum_separation_mm=(args.geometry_recovery_min_separation_mm),
            maximum_added_ratio=(args.geometry_recovery_max_added_ratio),
            enabled=args.geometric_mask_recovery,
        )

        object_overlay = build_foreground_overlay(
            rect_l,
            object_mask,
            (
                f"RGB umbral={background_threshold:.2f} | "
                f"recuperados={geometry_recovery_diag['added_pixels']} px"
            ),
            ("geometria conservadora | " f"estado={geometry_recovery_diag['status']}"),
        )
        imwrite_checked(
            str(output_dir / f"{stem}_object_mask_visual_seed.png"),
            visual_object_mask,
        )
        imwrite_checked(
            str(output_dir / f"{stem}_geometry_recovery_mask.png"),
            geometry_recovery_mask,
        )
        imwrite_checked(
            str(output_dir / f"{stem}_geometry_recovery_envelope.png"),
            geometry_recovery_envelope,
        )
        support_delta_valid = np.isfinite(support_plane_delta_mm) & (geometry_recovery_envelope > 0)
        support_delta_vis = normalize_for_vis(
            support_plane_delta_mm,
            support_delta_valid,
        )
        imwrite_checked(
            str(output_dir / f"{stem}_support_plane_separation_vis.png"),
            cv2.applyColorMap(support_delta_vis, cv2.COLORMAP_TURBO),
        )

        # --------------------------------------------------------------
        # V2.3.0 — PROFUNDIDAD DE ESCENA, NO MÁSCARA DE OBJETO
        # --------------------------------------------------------------
        trusted_scene_mask = (
            (stereo_trusted_before_object > 0) & physical_depth_mask & (rect_valid_mask > 0)
        )
        usable_scene_mask = (
            (np.asarray(confidence_data["usable_mask"]) > 0)
            & physical_depth_mask
            & (rect_valid_mask > 0)
        )
        weak_scene_mask = usable_scene_mask & (~trusted_scene_mask)

        # Solo diagnóstico: intersección con la segmentación visual. La
        # profundidad científica puede incluir observaciones unilaterales, pero
        # su procedencia queda explícita para 04/05/06.
        trusted_object_diagnostic_mask = trusted_scene_mask & (object_mask > 0)
        usable_object_diagnostic_mask = usable_scene_mask & (object_mask > 0)

        trusted_mask = trusted_scene_mask
        confidence_data["trusted_mask"] = trusted_scene_mask.astype(np.uint8) * 255
        confidence_data["usable_mask"] = usable_scene_mask.astype(np.uint8) * 255
        confidence_data["weak_one_sided_mask"] = weak_scene_mask.astype(np.uint8) * 255
        confidence_data["trusted_object_diagnostic_mask"] = (
            trusted_object_diagnostic_mask.astype(np.uint8) * 255
        )

        confident_depth_mm = np.full(raw_depth_mm.shape, np.nan, dtype=np.float32)
        confident_depth_mm[usable_scene_mask] = raw_depth_mm[usable_scene_mask]

        # Incertidumbre métrica para el consenso multisesión. No deriva del
        # error LR cuando la observación es unilateral: se calcula a partir de
        # confianza directa, suavidad y observabilidad, y se infla de forma
        # explícita para el estado débil.
        fx_rectified = float(calibration["P1"][0, 0])
        fb_px_mm = fx_rectified * float(calibration["baseline_mm"])
        disparity_uncertainty_px = (
            0.55
            + 1.75 * (1.0 - np.clip(confidence_data["base_confidence"], 0.0, 1.0))
            + 0.85 * (1.0 - np.clip(confidence_data["smoothness_score"], 0.0, 1.0))
            + 0.85 * (1.0 - np.clip(confidence_data["local_observability_score"], 0.0, 1.0))
        ).astype(np.float32)
        disparity_uncertainty_px[weak_scene_mask] *= np.float32(1.60)
        disparity_uncertainty_px[trusted_scene_mask] *= np.float32(0.85)
        depth_uncertainty_mm = np.full(raw_depth_mm.shape, np.nan, dtype=np.float32)
        valid_uncertainty = usable_scene_mask & np.isfinite(disparity_lr) & (disparity_lr > 1e-6)
        depth_uncertainty_mm[valid_uncertainty] = (
            fb_px_mm
            * disparity_uncertainty_px[valid_uncertainty]
            / np.square(disparity_lr[valid_uncertainty])
        ).astype(np.float32)
        np.save(str(output_dir / f"{stem}_disparity_uncertainty_px.npy"), disparity_uncertainty_px)
        np.save(str(output_dir / f"{stem}_depth_uncertainty_mm.npy"), depth_uncertainty_mm)

        roi_evidence = secondary_roi_evidence(
            estimator, rect_l, rect_r, rect_valid_mask, object_mask,
            disparity_lr, raw_depth_mm, depth_uncertainty_mm,
            confidence_data, calibration, args)
        np.savez_compressed(output_dir / f"{stem}_roi_evidence.npz",
                            **roi_evidence, diagnostic_only=np.array(True))

        rect_valid_pixels = int(np.count_nonzero(rect_valid_mask))
        trusted_scene_pixels = int(np.count_nonzero(trusted_scene_mask))
        weak_scene_pixels = int(np.count_nonzero(weak_scene_mask))
        usable_scene_pixels = int(np.count_nonzero(usable_scene_mask))
        trusted_object_pixels = int(np.count_nonzero(trusted_object_diagnostic_mask))
        usable_object_pixels = int(np.count_nonzero(usable_object_diagnostic_mask))
        object_pixels = int(np.count_nonzero(object_mask))

        center_depth_mask = usable_object_diagnostic_mask
        geometry_valid = confidence_data["geometry_valid"] > 0

        metrics: Dict[str, object] = {
            "rect_valid_pixels": rect_valid_pixels,
            "geometry_valid_pixels": int(np.count_nonzero(geometry_valid)),
            "object_pixels": object_pixels,
            "background_threshold": float(background_threshold),
            "foreground_depth_separation_mm": float(args.foreground_depth_separation_mm),
            "roi_strategy": (
                "full_rectification_domain_plus_adaptive_visual_anchor"
                if args.roi_mode == "adaptive"
                else "legacy_centered"
            ),
            "adaptive_roi": adaptive_roi_diag,
            "background_depth_consistency": depth_background_qc,
            "background_depth_used_for_foreground_membership": False,
            "current_support_plane_used_for_foreground_recovery": (
                geometry_recovery_diag.get("support_plane") is not None
                and geometry_recovery_diag.get("reason") is None
            ),
            "geometric_mask_recovery": geometry_recovery_diag,
            "depth_foreground_pixels": int(np.count_nonzero(depth_foreground_mask)),
            # trusted_ratio_of_object se conserva por compatibilidad con
            # el QC histórico, pero ya no decide la profundidad científica.
            "trusted_pixels": trusted_object_pixels,
            "trusted_scene_pixels": trusted_scene_pixels,
            "weak_one_sided_scene_pixels": weak_scene_pixels,
            "usable_scene_pixels": usable_scene_pixels,
            "trusted_object_diagnostic_pixels": trusted_object_pixels,
            "usable_object_diagnostic_pixels": usable_object_pixels,
            "trusted_ratio_of_object": (trusted_object_pixels / max(object_pixels, 1)),
            "usable_ratio_of_object": (usable_object_pixels / max(object_pixels, 1)),
            "trusted_ratio_of_rect_valid": (trusted_scene_pixels / max(rect_valid_pixels, 1)),
            "usable_ratio_of_rect_valid": (usable_scene_pixels / max(rect_valid_pixels, 1)),
            "scientific_depth_pixels": usable_scene_pixels,
            "scientific_depth_depends_on_object_mask": False,
            "reverse_reliable": bool(confidence_data["reverse_reliable"]),
            "reverse_ratio": (
                None
                if not np.isfinite(confidence_data["reverse_ratio"])
                else float(confidence_data["reverse_ratio"])
            ),
            "median_reverse_error_px": (
                None
                if not np.isfinite(confidence_data["median_reverse_error_px"])
                else float(confidence_data["median_reverse_error_px"])
            ),
            "median_reverse_tolerance_px": (
                None
                if not np.isfinite(confidence_data["median_reverse_tolerance_px"])
                else float(confidence_data["median_reverse_tolerance_px"])
            ),
            "consistent_ratio_of_geometry_valid": (
                np.count_nonzero(confidence_data["consistent_mask"])
                / max(
                    np.count_nonzero(geometry_valid),
                    1,
                )
            ),
            "disparity_lr": finite_statistics(
                disparity_lr[np.isfinite(disparity_lr) & (rect_valid_mask > 0)]
            ),
            "disparity_rl": finite_statistics(
                disparity_rl[np.isfinite(disparity_rl) & (rect_valid_mask > 0)]
            ),
            "lr_error_geometry_valid": finite_statistics(
                confidence_data["lr_error"][geometry_valid]
            ),
            "photo_error_geometry_valid": finite_statistics(
                confidence_data["photo_error"][geometry_valid]
            ),
            "gradient_error_geometry_valid": finite_statistics(
                confidence_data["gradient_error"][geometry_valid]
            ),
            "smoothness_error_geometry_valid": finite_statistics(
                confidence_data["smoothness_error"][geometry_valid]
            ),
            "confidence_geometry_valid": finite_statistics(
                confidence_data["confidence"][geometry_valid]
            ),
            "confidence_trusted": finite_statistics(confidence_data["confidence"][trusted_mask]),
            "confidence_usable": finite_statistics(
                confidence_data["confidence"][usable_scene_mask]
            ),
            "confidence_weak_one_sided": finite_statistics(
                confidence_data["confidence"][weak_scene_mask]
            ),
            "lr_state_counts": {
                str(code): int(np.count_nonzero(confidence_data["lr_state"] == code))
                for code in range(6)
            },
            "depth_uncertainty_mm": finite_statistics(
                depth_uncertainty_mm[usable_scene_mask & np.isfinite(depth_uncertainty_mm)]
            ),
            "raw_depth": finite_statistics(raw_depth_mm[np.isfinite(raw_depth_mm)]),
            "trusted_depth": finite_statistics(confident_depth_mm[np.isfinite(confident_depth_mm)]),
            "center_depth": finite_statistics(confident_depth_mm[center_depth_mask]),
        }

        local_quality, local_reasons = classify_local_quality(
            metrics,
            minimum_trusted_ratio=(args.minimum_trusted_ratio),
            minimum_center_depth_pixels=(args.minimum_center_depth_pixels),
            minimum_median_confidence=(args.minimum_median_confidence),
            minimum_lower_band_coverage=(args.minimum_lower_band_coverage),
            maximum_lower_band_drop=(args.maximum_lower_band_drop),
        )

        metrics["local_quality"] = local_quality
        metrics["local_reasons"] = local_reasons

        record: Dict[str, object] = {
            "stem": stem,
            "left_path": str(left_path),
            "right_path": str(right_path),
            "local_quality": local_quality,
            "local_reasons": local_reasons,
            "session_quality": "pending",
            "session_reasons": [],
            "metrics": metrics,
        }
        records.append(record)

        save_initial_outputs(
            out_dir=output_dir,
            stem=stem,
            rect_l=rect_l,
            rect_r=rect_r,
            rect_valid_mask=rect_valid_mask,
            object_mask=object_mask,
            object_overlay=object_overlay,
            disparity_lr=disparity_lr,
            disparity_rl=disparity_rl,
            raw_depth_mm=raw_depth_mm,
            confident_depth_mm=confident_depth_mm,
            confidence_data=confidence_data,
            metrics=metrics,
            provider_used=provider_used,
            display_depth_min_mm=display_depth_min_mm,
            display_depth_max_mm=display_depth_max_mm,
        )

        center_median = metrics["center_depth"]["median"]
        center_text = "sin datos" if center_median is None else f"{float(center_median):.2f} mm"

        print(
            f"[{index:02d}/{len(pairs):02d}] {stem}: "
            f"conf_obj={metrics['trusted_ratio_of_object']:.2%}, "
            f"z_obj={center_text}, "
            f"inversa={'ok' if metrics['reverse_reliable'] else 'omitida'}, "
            f"local={local_quality}"
        )

    if not records:
        raise RuntimeError("No se procesó ninguna vista.")

    compare_roi_sessions(root, output_name)

    session_qc = apply_session_quality_control(
        records=records,
        absolute_depth_tolerance_mm=(args.session_depth_tolerance_mm),
        depth_mad_factor=(args.session_depth_mad_factor),
        trusted_ratio_mad_factor=(args.session_trusted_ratio_mad_factor),
        minimum_trusted_ratio=(args.minimum_trusted_ratio),
    )

    for record in records:
        finalize_view_output(
            output_dir,
            record,
            keep_rejected_depth=(args.keep_rejected_depth),
        )

    accepted_records = [
        record
        for record in records
        if record["session_quality"] in ("accepted", "accepted_local_only")
    ]
    rejected_records = [record for record in records if record["session_quality"] == "rejected"]

    summary = {
        "schema_version": 10,
        "method": ("crestereo_onnx_v2_3_raw255_internal_normalization_scene_depth"),
        "object": object_name,
        "session": session_name,
        "pairs_found": len(pairs),
        "unmatched_left_stems": unmatched_left,
        "unmatched_right_stems": unmatched_right,
        "pairs_processed": len(records),
        "views_accepted": len(accepted_records),
        "views_rejected": len(rejected_records),
        "accepted_stems": [str(record["stem"]) for record in accepted_records],
        "rejected_stems": [str(record["stem"]) for record in rejected_records],
        "provider_used": provider_used,
        "available_providers": (estimator.available_providers),
        "onnxruntime_environment": estimator.environment_diagnostics,
        "model_path": str(model_path),
        "calibration_dir": str(calibration_dir),
        "model_input_shape": [str(shape) for shape in estimator.input_shapes],
        "baseline_mm": float(calibration["baseline_mm"]),
        "fx_rectified_px": float(np.asarray(calibration["P1"])[0, 0]),
        "outputs": str(output_dir),
        "parameters": {
            **vars(args),
            "crestereo_external_input_range": "RGB float32 0..255",
            "crestereo_model_internal_normalization": "input/255*2-1",
            "scientific_depth_depends_on_object_mask": False,
        },
        "session_quality_control": session_qc,
        "views": records,
        "confidence_scope_note": (
            "*_confidence.npy representa confianza DIRECTA (fotometría, gradiente y "
            "suavidad) y no incorpora LR. La consistencia LR se exporta por separado "
            "en *_lr_state.npy y *_lr_score.npy para evitar penalización doble."
        ),
        "compatibility_note": (
            "*_trusted_mask.png contiene evidencia fuerte. *_usable_mask.png y "
            "*_depth_mm.npy pueden incluir observaciones unilaterales débiles; estas "
            "no adquieren autoridad fuerte hasta ser validadas en 04/05/06. El QC "
            "de vista/sesión es diagnóstico y no destruye profundidad por píxel."
        ),
    }

    summary_path = output_dir / "resumen_02_estimacion_profundidad.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_session_csv(
        output_dir / "session_quality.csv",
        records,
    )

    print("\n========== PASO 02 COMPLETADO " "(CONFIANZA + CONTROL DE SESIÓN) ==========")
    print(f"Objeto: {object_name}")
    print(f"Sesión: {session_name}")
    print(f"Procesadas: {len(records)}")
    print(f"Aceptadas: {len(accepted_records)}")
    print(f"Rechazadas: {len(rejected_records)}")
    print(f"Provider: {provider_used}")
    print(f"Salida: {output_dir}")

    if rejected_records:
        print("\nVistas rechazadas:")
        for record in rejected_records:
            reasons = " | ".join(record["session_reasons"])
            print(f"  - {record['stem']}: {reasons}")

    print("==========================================================")


def _entrada_con_diagnostico():
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    from utilidades_progreso import informar_inicio

    informar_inicio(__file__, "02", "Estimar profundidad CREStereo")
    main()


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
