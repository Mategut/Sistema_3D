#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilidades compartidas para la cadena de máscaras finales."""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ANGLE_RE = re.compile(r"_A(\d+)", re.IGNORECASE)


def read_yaml_matrix(path: Path, key: str) -> np.ndarray:
    """Lee una matriz de OpenCV FileStorage y valida archivo, clave y contenido."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"No se pudo abrir: {path}")
    node = fs.getNode(key)
    if node.empty():
        fs.release()
        raise KeyError(f"No se encontró '{key}' en {path}")
    matrix = node.mat()
    fs.release()
    if matrix is None:
        raise RuntimeError(f"'{key}' no contiene una matriz válida en {path}")
    return np.asarray(matrix, dtype=np.float64)


CALIBRATION_DIR_NAME = "calibracion estereo"


def _has_calibration_files(path: Path, required_files: Sequence[str]) -> bool:
    """Comprueba los archivos mínimos de una carpeta de calibración estéreo."""
    return path.is_dir() and all((path / name).is_file() for name in required_files)


def resolve_calibration_dir(
    root: Path,
    object_name: str = "",
    explicit: Optional[Path | str] = None,
    required_files: Sequence[str] = ("stereo_initial.yaml",),
) -> Path:
    """Resuelve SOLO una calibración estéreo indicada explícitamente.

    Diseño de producto: nunca busca calibraciones dentro de objetos, campañas
    ni resultados históricos. La calibración es configuración persistente del
    sistema y debe ser proporcionada por el producto/orquestador.
    """
    if not explicit:
        raise ValueError(
            "Se requiere --calibration-dir explícito. "
            "No se buscan calibraciones en resultados o carpetas históricas."
        )
    path = Path(explicit).expanduser().resolve()
    if not _has_calibration_files(path, required_files):
        missing = [name for name in required_files if not (path / name).is_file()]
        raise FileNotFoundError(
            f"La calibración estéreo no está completa: {path}. " f"Faltan: {', '.join(missing)}"
        )
    return path


def load_stereo_geometry(
    root: Path,
    object_name: str = "cubo",
    calibration_dir: Optional[Path | str] = None,
) -> Dict[str, float | str]:
    calibration_dir = resolve_calibration_dir(
        root,
        object_name=object_name,
        explicit=calibration_dir,
        required_files=("stereo_initial.yaml",),
    )
    stereo_yaml = calibration_dir / "stereo_initial.yaml"
    p1 = read_yaml_matrix(stereo_yaml, "P1")
    translation = read_yaml_matrix(stereo_yaml, "T")
    if p1.shape != (3, 4):
        raise ValueError(f"P1 debe ser 3x4; se recibió {p1.shape}")
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    if translation.size < 3:
        raise ValueError(f"T debe contener al menos 3 valores; se recibió {translation.shape}")
    baseline_mm = float(np.linalg.norm(translation[:3]))
    if not np.isfinite(baseline_mm) or baseline_mm <= 0:
        raise ValueError(f"Línea base inválida en {stereo_yaml}: {baseline_mm}")
    return {
        "fx": float(p1[0, 0]),
        "fy": float(p1[1, 1]),
        "cx": float(p1[0, 2]),
        "cy": float(p1[1, 2]),
        "baseline_mm": baseline_mm,
        "calibration_dir": str(calibration_dir),
    }


def load_intrinsics(
    root: Path, object_name: str = "cubo", calibration_dir: Optional[Path | str] = None
) -> Dict[str, float]:
    """Obtiene los parámetros intrínsecos desde la calibración estéreo."""
    geometry = load_stereo_geometry(root, object_name=object_name, calibration_dir=calibration_dir)
    return {
        "fx": float(geometry["fx"]),
        "fy": float(geometry["fy"]),
        "cx": float(geometry["cx"]),
        "cy": float(geometry["cy"]),
    }


def validate_summary_context(
    summary: dict, object_name: str, session: str, source_label: str
) -> None:
    """Comprueba que un resumen corresponda al objeto y la sesión esperados."""
    expected_object = object_name.strip().lower()
    expected_session = session.strip().upper()
    summary_object = str(summary.get("object", "")).strip().lower()
    summary_session = str(summary.get("session", "")).strip().upper()
    if summary_object and summary_object != expected_object:
        raise ValueError(
            f"{source_label}: el resumen pertenece al objeto '{summary_object}', no a '{expected_object}'."
        )
    if summary_session and summary_session != expected_session:
        raise ValueError(
            f"{source_label}: el resumen pertenece a la sesión '{summary_session}', no a '{expected_session}'."
        )


def prepare_output_directory(path: Path, clean: bool = True) -> Path:
    """Prepara el directorio de salida con la política de limpieza indicada."""
    path = Path(path)
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def imwrite_checked(path: Path, image: np.ndarray) -> None:
    """Escribe una imagen y lanza un error si OpenCV no confirma la escritura."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"OpenCV no pudo guardar la imagen: {path}")


def parse_angle(stem: str) -> float:
    """Interpreta el token angular del nombre de captura.

    Formato V7.2:
        A0000 -> 0.0°
        A0144 -> 14.4°
        A3456 -> 345.6°

    Compatibilidad histórica:
        A000 -> 0°
        A015 -> 15°
        A345 -> 345°
    """
    match = ANGLE_RE.search(stem)
    if not match:
        raise ValueError(f"No se encontró el ángulo en: {stem}")
    token = match.group(1)
    value = int(token)
    if len(token) >= 4:
        return float(value) / 10.0
    return float(value)


VIEW_RE = re.compile(r"_V(\d+)", re.IGNORECASE)


def parse_view_index(stem: str) -> int:
    """Extrae del nombre de captura el índice de vista con base cero."""
    match = VIEW_RE.search(stem)
    if not match:
        raise ValueError(f"No se encontró la vista V### en: {stem}")
    return int(match.group(1)) - 1


def find_summary(directory: Path, preferred: Sequence[str]) -> Path:
    """Acepta únicamente nombres explícitos de la ejecución actual."""
    for name in preferred:
        path = directory / name
        if path.is_file():
            return path
    expected = ", ".join(str(directory / name) for name in preferred)
    raise FileNotFoundError(
        "No existe la salida exacta requerida de la etapa anterior. " f"Esperado: {expected}"
    )


def load_json(path: Path) -> dict:
    """Lee un documento JSON codificado en UTF-8."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def finite_stats(values: np.ndarray) -> dict:
    array = np.asarray(values)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
            "mad": None,
        }
    median = float(np.median(array))
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "median": median,
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mad": float(np.median(np.abs(array - median))),
    }


def robust_location_scale(values: np.ndarray, minimum_scale: float = 1.0):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0, float(minimum_scale)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    sigma = max(float(minimum_scale), 1.4826 * mad)
    return median, sigma


def odd(value: int, minimum: int = 3) -> int:
    """Devuelve un tamaño impar respetando el mínimo especificado."""
    result = max(minimum, int(value))
    return result if result % 2 else result + 1


def safe_rectification_domain(
    rect_valid_mask: np.ndarray,
    erosion_px: int = 2,
) -> np.ndarray:
    """Dominio amplio: no recorta el objeto, solo evita bordes inválidos."""
    valid = (np.asarray(rect_valid_mask) > 0).astype(np.uint8) * 255
    erosion_px = max(0, int(erosion_px))
    if erosion_px <= 0:
        return valid

    k = 2 * erosion_px + 1
    eroded = cv2.erode(
        valid,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        iterations=1,
    )
    # Seguridad: nunca perder una fracción grande por una máscara irregular.
    if np.count_nonzero(eroded) < 0.85 * max(np.count_nonzero(valid), 1):
        return valid
    return eroded


def _visual_change_score(
    image: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    if image.shape != background.shape:
        raise ValueError(f"Imagen/fondo incompatibles: {image.shape} vs {background.shape}")
    lab_i = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_b = cv2.cvtColor(background, cv2.COLOR_BGR2LAB)
    gray_i = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    lab_diff = cv2.absdiff(lab_i, lab_b).astype(np.float32)
    gray_diff = cv2.absdiff(gray_i, gray_b).astype(np.float32)
    score = (0.72 * np.max(lab_diff, axis=2) + 0.28 * gray_diff).astype(np.float32)
    return cv2.GaussianBlur(score, (5, 5), 0)


def robust_change_threshold(
    score: np.ndarray,
    valid_mask: np.ndarray,
    minimum_threshold: float = 7.0,
    mad_factor: float = 4.5,
    stable_quantile: float = 0.62,
) -> Tuple[float, np.ndarray]:
    """Aprende el ruido desde los píxeles de menor cambio de la escena."""
    valid = (np.asarray(valid_mask) > 0) & np.isfinite(score)
    values = np.asarray(score, dtype=np.float32)[valid]

    if values.size == 0:
        return float(minimum_threshold), np.zeros(score.shape, dtype=bool)

    q = float(np.clip(stable_quantile, 0.30, 0.85))
    cutoff = float(np.quantile(values, q))
    stable = valid & (score <= cutoff)
    stable_values = score[stable]
    if stable_values.size < 2000:
        stable = valid
        stable_values = score[stable]

    med = float(np.median(stable_values))
    mad = float(np.median(np.abs(stable_values - med)))
    sigma = max(0.75, 1.4826 * mad)
    return float(max(minimum_threshold, med + mad_factor * sigma)), stable


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Obtiene la caja delimitadora de los píxeles activos de la máscara."""
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expanded_bbox_mask(
    mask: np.ndarray,
    valid_domain: np.ndarray,
    margin_fraction: float = 0.18,
    minimum_margin_px: int = 24,
) -> Tuple[np.ndarray, dict]:
    """Anchor expandido. Nunca es un recorte duro del procesamiento."""
    h, w = mask.shape
    domain = (valid_domain > 0).astype(np.uint8) * 255
    bbox = mask_bbox(mask)

    if bbox is None:
        return domain.copy(), {
            "fallback_full_domain": True,
            "foreground_bbox_xyxy": None,
            "expanded_bbox_xyxy": [0, 0, w, h],
            "margin_px": None,
        }

    x0, y0, x1, y1 = bbox
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    mx = max(int(minimum_margin_px), int(round(bw * margin_fraction)))
    my = max(int(minimum_margin_px), int(round(bh * margin_fraction)))
    ex0, ey0 = max(0, x0 - mx), max(0, y0 - my)
    ex1, ey1 = min(w, x1 + mx), min(h, y1 + my)

    anchor = np.zeros((h, w), np.uint8)
    anchor[ey0:ey1, ex0:ex1] = 255
    anchor = cv2.bitwise_and(anchor, domain)

    return anchor, {
        "fallback_full_domain": False,
        "foreground_bbox_xyxy": [x0, y0, x1, y1],
        "expanded_bbox_xyxy": [ex0, ey0, ex1, ey1],
        "margin_px": [mx, my],
    }


def adaptive_visual_anchor(
    image: np.ndarray,
    background: np.ndarray,
    valid_domain: np.ndarray,
    minimum_threshold: float = 7.0,
    mad_factor: float = 4.5,
    stable_quantile: float = 0.62,
    minimum_component_area: int = 900,
    margin_fraction: float = 0.18,
    minimum_margin_px: int = 24,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Anchor visual automático; NO decide pertenencia final del objeto."""
    domain = (valid_domain > 0).astype(np.uint8) * 255
    score = _visual_change_score(image, background)
    threshold, stable = robust_change_threshold(
        score,
        domain,
        minimum_threshold=minimum_threshold,
        mad_factor=mad_factor,
        stable_quantile=stable_quantile,
    )

    raw = ((score >= threshold) & (domain > 0)).astype(np.uint8) * 255
    raw = cv2.morphologyEx(
        raw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    raw = cv2.morphologyEx(
        raw,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1,
    )

    selected = select_relevant_components(
        raw,
        minimum_area=max(100, int(minimum_component_area)),
        secondary_fraction=0.12,
        prior_mask=None,
    )

    anchor, bbox_diag = expanded_bbox_mask(
        selected,
        domain,
        margin_fraction=margin_fraction,
        minimum_margin_px=minimum_margin_px,
    )

    ratio = np.count_nonzero(selected) / max(np.count_nonzero(domain), 1)
    if ratio < 0.003:
        anchor = domain.copy()
        bbox_diag["fallback_full_domain"] = True
        bbox_diag["fallback_reason"] = "deteccion_visual_demasiado_pequena"

    return (
        anchor,
        selected,
        {
            "threshold": float(threshold),
            "stable_pixels": int(np.count_nonzero(stable)),
            "candidate_pixels": int(np.count_nonzero(raw)),
            "selected_pixels": int(np.count_nonzero(selected)),
            "selected_ratio_of_valid_domain": float(ratio),
            **bbox_diag,
        },
    )


def shadow_aware_hysteresis(
    weak_base: np.ndarray,
    strong_base: np.ndarray,
    shadow_candidate: np.ndarray,
) -> np.ndarray:
    """Sombra aislada se rechaza; zona tipo sombra unida al objeto puede quedar."""
    weak = np.asarray(weak_base).astype(bool)
    strong = np.asarray(strong_base).astype(bool)
    shadow = np.asarray(shadow_candidate).astype(bool)

    strong_seed = strong & (~shadow)
    return hysteresis_components(
        weak.astype(np.uint8) * 255,
        strong_seed.astype(np.uint8) * 255,
    )


def derive_depth_display_range(
    background_depth_mm: np.ndarray,
    minimum_depth_mm: float,
    maximum_depth_mm: float,
    near_margin_mm: float = 100.0,
    far_margin_mm: float = 100.0,
) -> Tuple[float, float]:
    values = np.asarray(background_depth_mm, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size < 1000:
        return float(minimum_depth_mm), float(maximum_depth_mm)
    p01, p99 = np.percentile(values, [1.0, 99.0])
    lo = max(float(minimum_depth_mm), float(p01) - float(near_margin_mm))
    hi = min(float(maximum_depth_mm), float(p99) + float(far_margin_mm))
    if hi - lo < 150.0:
        c = 0.5 * (lo + hi)
        lo = max(float(minimum_depth_mm), c - 75.0)
        hi = min(float(maximum_depth_mm), c + 75.0)
    return float(lo), float(hi)


def _ellipse_sample_points(
    ellipse,
    count: int = 720,
) -> np.ndarray:
    """Muestrea uniformemente el borde de una elipse OpenCV."""
    (cx, cy), (axis_a, axis_b), angle_deg = ellipse
    t = np.linspace(0.0, 2.0 * np.pi, max(90, int(count)), endpoint=False)
    angle = math.radians(float(angle_deg))
    ca, sa = math.cos(angle), math.sin(angle)
    x = 0.5 * float(axis_a) * np.cos(t)
    y = 0.5 * float(axis_b) * np.sin(t)
    xr = float(cx) + x * ca - y * sa
    yr = float(cy) + x * sa + y * ca
    return np.column_stack([xr, yr]).astype(np.float32)


def _ellipse_edge_support(
    ellipse,
    edge_distance: np.ndarray,
    sigma_px: float = 4.0,
) -> float:
    """Fracción suave del borde elíptico respaldada por bordes de imagen."""
    h, w = edge_distance.shape
    pts = _ellipse_sample_points(ellipse)
    x = np.rint(pts[:, 0]).astype(np.int32)
    y = np.rint(pts[:, 1]).astype(np.int32)
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if np.count_nonzero(valid) < 90:
        return 0.0
    d = edge_distance[y[valid], x[valid]].astype(np.float32)
    sigma = max(1.0, float(sigma_px))
    return float(np.mean(np.exp(-np.square(d / sigma))))


def _ellipse_geometry_score(ellipse, h: int, w: int) -> float:
    """Prior MUY suave del hardware: plataforma circular vista en perspectiva.

    V2.0.1: el *score* no contiene preferencia por un centro concreto. El centro
    se obtiene del borde observado. Solo quedan gates amplios de plausibilidad
    para evitar confundir paredes/suelo con una plataforma circular proyectada.
    """
    (cx, cy), (a, b), _ = ellipse
    major = max(float(a), float(b))
    minor = min(float(a), float(b))
    ratio = minor / max(major, 1e-6)

    if not (
        0.28 * w <= major <= 0.76 * w
        and 0.09 * h <= minor <= 0.46 * h
        and 0.10 * w <= float(cx) <= 0.92 * w
        and 0.40 * h <= float(cy) <= 0.97 * h
        and 0.14 <= ratio <= 0.62
    ):
        return 0.0

    # Solo forma/tamaño proyectado; NO centro. Ambos términos son deliberadamente
    # anchos y el borde de imagen sigue teniendo el peso dominante.
    aspect = math.exp(-(((ratio - 0.31) / 0.24) ** 2))
    size = math.exp(-(((major / max(float(w), 1.0) - 0.50) / 0.24) ** 2))
    return float(math.sqrt(max(aspect * size, 0.0)))


def _estimate_support_from_background_image(
    background_bgr: np.ndarray,
    rect_valid_mask: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """Detecta la plataforma directamente en el fondo vacío rectificado.

    La revisión del montaje mostró que ajustar la elipse directamente al mapa de
    profundidad puede absorber el suelo/pared cuando CREStereo suaviza superficies
    blancas poco texturizadas. Esta ruta usa múltiples umbrales de luminancia y
    selecciona la elipse cuyo borde coincide mejor con discontinuidades reales de
    la imagen. El centro NO se supone en el centro de la fotografía.
    """
    image = np.asarray(background_bgr)
    valid = np.asarray(rect_valid_mask) > 0
    h, w = valid.shape
    empty = np.zeros((h, w), np.uint8)

    diag = {
        "status": "not_detected",
        "method": "background_image_multithreshold_ellipse",
        "reason": None,
        "candidate_count": 0,
        "selected": None,
        "confidence_score": 0.0,
    }
    if image.ndim != 3 or image.shape[:2] != (h, w):
        diag["reason"] = "background_image_shape_mismatch"
        return empty, diag

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.uint8)

    # Dominio amplio. Se usa solo para generar candidatos, no para fijar centro.
    search = np.zeros((h, w), np.uint8)
    x0, x1 = int(round(0.08 * w)), int(round(0.96 * w))
    y0, y1 = int(round(0.38 * h)), int(round(0.98 * h))
    search[y0:y1, x0:x1] = 255
    search = cv2.bitwise_and(search, valid.astype(np.uint8) * 255)
    values = lightness[search > 0]
    if values.size < 5000:
        diag["reason"] = "insufficient_image_samples"
        return empty, diag

    # Distancia al borde de imagen. Los bordes se extraen con thresholds
    # adaptados al gradiente, no con una constante fotométrica del montaje.
    blur = cv2.GaussianBlur(lightness, (7, 7), 1.5)
    med = float(np.median(blur[search > 0]))
    low = int(np.clip(0.33 * med, 12, 90))
    high = int(np.clip(0.95 * med, low + 15, 180))
    edges = cv2.Canny(blur, low, high)
    edges = cv2.bitwise_and(edges, cv2.dilate(search, np.ones((5, 5), np.uint8)))
    edge_distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)

    candidates = []
    quantiles = np.linspace(0.58, 0.84, 14)
    for q in quantiles:
        threshold = float(np.quantile(values, q))
        binary = ((lightness >= threshold) & (search > 0)).astype(np.uint8) * 255
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 0.020 * h * w or len(contour) < 5:
                continue
            ellipse = cv2.fitEllipse(contour)
            geometry = _ellipse_geometry_score(ellipse, h, w)
            if geometry <= 0.0:
                continue
            edge_score = _ellipse_edge_support(ellipse, edge_distance, sigma_px=4.0)
            major = max(float(ellipse[1][0]), float(ellipse[1][1]))

            # El borde manda. Entre candidatos casi equivalentes se favorece
            # ligeramente la envolvente exterior para no recortar el plato.
            size_term = np.clip(major / max(0.60 * w, 1.0), 0.0, 1.25)
            score = 3.0 * edge_score + 0.40 * geometry + 0.35 * float(size_term)
            candidates.append(
                {
                    "score": float(score),
                    "quantile": float(q),
                    "threshold_l": float(threshold),
                    "area_pixels": area,
                    "edge_support": float(edge_score),
                    "geometry_score": float(geometry),
                    "ellipse": ellipse,
                }
            )

    diag["candidate_count"] = int(len(candidates))
    if not candidates:
        diag["reason"] = "no_plausible_image_ellipse"
        return empty, diag

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best_score = float(candidates[0]["score"])

    # Si varios candidatos tienen score prácticamente igual, se escoge el de
    # mayor eje mayor: corresponde mejor al borde exterior visible de la base.
    near_best = [c for c in candidates if c["score"] >= best_score - 0.035]
    best = max(
        near_best,
        key=lambda c: max(float(c["ellipse"][1][0]), float(c["ellipse"][1][1])),
    )

    ellipse = best["ellipse"]
    support = np.zeros((h, w), np.uint8)
    cv2.ellipse(support, ellipse, 255, -1)
    support[~valid] = 0

    edge_norm = float(np.clip(best["edge_support"] / 0.20, 0.0, 1.0))
    confidence = float(np.clip(0.68 * edge_norm + 0.32 * best["geometry_score"], 0.0, 1.0))
    (cx, cy), (a, b), angle = ellipse
    diag.update(
        {
            "status": "detected",
            "reason": None,
            "confidence_score": confidence,
            "selected": {
                "score": float(best["score"]),
                "quantile": float(best["quantile"]),
                "threshold_l": float(best["threshold_l"]),
                "edge_support": float(best["edge_support"]),
                "geometry_score": float(best["geometry_score"]),
                "ellipse": {
                    "center_xy": [float(cx), float(cy)],
                    "axes": [float(a), float(b)],
                    "angle_deg": float(angle),
                    "major_axis_fraction_of_width": float(max(a, b) / w),
                    "minor_axis_fraction_of_height": float(min(a, b) / h),
                },
            },
            "mask_ratio": float(np.count_nonzero(support) / max(np.count_nonzero(valid), 1)),
        }
    )
    return support, diag


def _estimate_support_from_depth(
    background_depth_mm: np.ndarray,
    rect_valid_mask: np.ndarray,
    search_x_fraction: Tuple[float, float] = (0.15, 0.85),
    search_y_fraction: Tuple[float, float] = (0.46, 0.92),
    near_quantile: float = 0.42,
) -> Tuple[np.ndarray, dict]:
    """Estimador de profundidad conservado como segunda evidencia/fallback."""
    depth = np.asarray(background_depth_mm, dtype=np.float32)
    valid = (np.asarray(rect_valid_mask) > 0) & np.isfinite(depth)
    h, w = depth.shape
    empty = np.zeros((h, w), np.uint8)
    x0 = max(0, min(w - 1, int(round(search_x_fraction[0] * w))))
    x1 = max(x0 + 1, min(w, int(round(search_x_fraction[1] * w))))
    y0 = max(0, min(h - 1, int(round(search_y_fraction[0] * h))))
    y1 = max(y0 + 1, min(h, int(round(search_y_fraction[1] * h))))
    zone = np.zeros((h, w), np.uint8)
    zone[y0:y1, x0:x1] = 255
    sm = valid & (zone > 0)
    values = depth[sm]
    diag = {
        "status": "not_detected",
        "method": "background_depth_quantile_ellipse",
        "search_zone_xyxy": [x0, y0, x1, y1],
        "near_quantile": float(near_quantile),
        "threshold_depth_mm": None,
        "ellipse": None,
        "mask_ratio": 0.0,
        "reason": None,
        "confidence_score": 0.0,
    }
    if values.size < 5000:
        diag["reason"] = "insufficient_background_depth_samples"
        return empty, diag
    threshold = float(np.quantile(values, np.clip(near_quantile, 0.15, 0.70)))
    diag["threshold_depth_mm"] = threshold
    near = (sm & (depth <= threshold)).astype(np.uint8) * 255
    near = cv2.morphologyEx(
        near,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=2,
    )
    near = cv2.morphologyEx(
        near,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (near > 0).astype(np.uint8), 8
    )
    cands = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(3000, int(0.005 * h * w)):
            continue
        bx = int(stats[label, cv2.CC_STAT_LEFT])
        by = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        comp = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            continue
        ellipse = cv2.fitEllipse(contour)
        geometry = _ellipse_geometry_score(ellipse, h, w)
        if geometry <= 0.0:
            continue
        af = area / float(h * w)
        # Diagnóstico de profundidad SIN centro preferido. Se valora que la
        # componente sea elíptica/plausible y suficientemente extensa.
        area_term = float(np.clip(af / 0.12, 0.0, 1.0))
        score = 0.82 * float(geometry) + 0.18 * area_term
        cands.append(
            {
                "label": int(label),
                "score": float(score),
                "area": area,
                "bbox": [bx, by, bw, bh],
                "centroid": np.asarray(centroids[label], np.float64).astype(float).tolist(),
                "geometry_score": float(geometry),
                "ellipse_raw": {
                    "center_xy": [float(ellipse[0][0]), float(ellipse[0][1])],
                    "axes": [float(ellipse[1][0]), float(ellipse[1][1])],
                    "angle_deg": float(ellipse[2]),
                },
                "_ellipse": ellipse,
            }
        )
    if not cands:
        diag["reason"] = "no_valid_support_component"
        return empty, diag
    cands.sort(key=lambda d: d["score"], reverse=True)
    best = cands[0]
    ellipse = best["_ellipse"]
    (cx, cy), (a, b), angle = ellipse
    geometry = float(best["geometry_score"])
    support = np.zeros((h, w), np.uint8)
    cv2.ellipse(support, ellipse, 255, -1)
    support[~valid] = 0
    diag.update(
        {
            "status": "detected",
            "reason": None,
            "confidence_score": float(0.45 * geometry),  # depth solo: confianza limitada
            "mask_ratio": float(np.count_nonzero(support) / max(np.count_nonzero(valid), 1)),
            "ellipse": {
                "center_xy": [float(cx), float(cy)],
                "axes": [float(a), float(b)],
                "angle_deg": float(angle),
                "major_axis_fraction_of_width": float(max(a, b) / w),
                "minor_axis_fraction_of_height": float(min(a, b) / h),
            },
            "selected_component": {k: v for k, v in best.items() if k != "_ellipse"},
            "candidate_components": [
                {k: v for k, v in c.items() if k != "_ellipse"} for c in cands
            ],
        }
    )
    return support, diag


def estimate_turntable_support_mask(
    background_depth_mm: np.ndarray,
    rect_valid_mask: np.ndarray,
    background_bgr: Optional[np.ndarray] = None,
    search_x_fraction: Tuple[float, float] = (0.15, 0.85),
    search_y_fraction: Tuple[float, float] = (0.46, 0.92),
    near_quantile: float = 0.42,
) -> Tuple[np.ndarray, dict]:
    """Máscara robusta de la plataforma mediante consenso imagen + profundidad.

    Política V2.0.1:
    - la imagen del fondo vacío es la fuente primaria para la elipse;
    - la profundidad se conserva únicamente como evidencia secundaria/diagnóstico;
    - una discrepancia entre ambas NO desplaza la elipse hacia la profundidad;
    - si la evidencia visual no es suficientemente fuerte, se devuelve máscara
      vacía: no se permite que CREStereo recorte objeto por un falso plato;
    - el score no contiene una preferencia por x=0.5 ni por ningún centro fijo.
    """
    depth_mask, depth_diag = _estimate_support_from_depth(
        background_depth_mm,
        rect_valid_mask,
        search_x_fraction=search_x_fraction,
        search_y_fraction=search_y_fraction,
        near_quantile=near_quantile,
    )

    image_mask = np.zeros_like(depth_mask)
    image_diag = {
        "status": "not_available",
        "method": "background_image_multithreshold_ellipse",
        "confidence_score": 0.0,
        "reason": "background_bgr_not_provided",
    }
    if background_bgr is not None:
        image_mask, image_diag = _estimate_support_from_background_image(
            background_bgr,
            rect_valid_mask,
        )

    selected_mask = np.zeros_like(depth_mask)
    selected_source = None
    selected_confidence = 0.0

    if (
        image_diag.get("status") == "detected"
        and float(image_diag.get("confidence_score", 0.0)) >= 0.46
    ):
        selected_mask = image_mask
        selected_source = "background_image_edge_validated"
        selected_confidence = float(image_diag.get("confidence_score", 0.0))

    # Política de seguridad: la profundidad del fondo se conserva como
    # DIAGNÓSTICO, pero nunca puede activar por sí sola un recorte duro. En
    # superficies blancas CREStereo puede suavizar plato + suelo y producir una
    # elipse grande/descentrada. Si la imagen no respalda la plataforma, es más
    # seguro continuar SIN prior de soporte que recortar objeto real.

    agreement = None
    if np.count_nonzero(image_mask) and np.count_nonzero(depth_mask):
        inter = np.count_nonzero((image_mask > 0) & (depth_mask > 0))
        union = np.count_nonzero((image_mask > 0) | (depth_mask > 0))
        agreement = float(inter / max(union, 1))
        if selected_source == "background_image_edge_validated" and agreement >= 0.60:
            selected_confidence = float(min(1.0, selected_confidence + 0.08))

    status = "detected" if selected_source is not None else "not_detected"
    diag = {
        "status": status,
        "method": "multicue_support_detector_v2_0_1_visual_hard_depth_diagnostic",
        "selected_source": selected_source,
        "confidence_score": float(selected_confidence),
        "image_depth_iou": agreement,
        "image_candidate": image_diag,
        "depth_candidate": depth_diag,
        "mask_ratio": float(
            np.count_nonzero(selected_mask)
            / max(np.count_nonzero(np.asarray(rect_valid_mask) > 0), 1)
        ),
        "reason": (
            None
            if selected_source is not None
            else (
                "No hubo una elipse con evidencia visual o de profundidad suficiente. "
                "Se evita aplicar un prior de plataforma incorrecto."
            )
        ),
    }
    return selected_mask, diag



def build_support_rim_guard_mask(
    support_mask: np.ndarray,
    valid_domain: np.ndarray,
    rim_px: int = 6,
) -> Tuple[np.ndarray, dict]:
    """Crea una banda fina exterior para cubrir el filo físico del plato.

    La elipse de ``support_mask`` se conserva sin cambios porque sigue siendo
    la superficie útil usada por la lógica de contacto. El guard es una
    segunda máscara, de pocos píxeles, formada únicamente FUERA del soporte.
    Su objetivo es absorber el pequeño borde gris que queda inmediatamente
    adyacente a la elipse estimada y que puede aparecer como falso foreground.

    Esta función no decide todavía qué parte del guard debe quedar protegida
    por la presencia de un objeto. Esa protección se resuelve en el paso 03 a
    partir de la semilla visual del objeto, antes de convertir el guard en un
    veto definitivo.
    """
    support_u8 = (np.asarray(support_mask) > 0).astype(np.uint8) * 255
    valid = np.asarray(valid_domain) > 0

    if support_u8.shape != valid.shape:
        raise ValueError(
            "support_mask y valid_domain deben tener la misma forma: "
            f"{support_u8.shape} vs {valid.shape}"
        )

    zero = np.zeros_like(support_u8)
    support_pixels = int(np.count_nonzero(support_u8))
    if support_pixels < 1000:
        return zero, {
            "status": "disabled",
            "reason": "support_not_available",
            "support_pixels": support_pixels,
            "rim_px": 0,
            "rim_pixels": 0,
        }

    radius = int(np.clip(int(rim_px), 0, 32))
    if radius <= 0:
        return zero, {
            "status": "disabled",
            "reason": "rim_px_is_zero",
            "support_pixels": support_pixels,
            "rim_px": 0,
            "rim_pixels": 0,
        }

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    expanded = cv2.dilate(support_u8, kernel, iterations=1)
    rim = (expanded > 0) & (support_u8 == 0) & valid
    result = rim.astype(np.uint8) * 255

    return result, {
        "status": "active",
        "principle": "thin_outer_rim_guard_without_changing_support_surface",
        "rim_px": int(radius),
        "support_pixels": support_pixels,
        "rim_pixels": int(np.count_nonzero(result)),
        "overlap_with_support_pixels": int(
            np.count_nonzero((result > 0) & (support_u8 > 0))
        ),
    }


def build_support_hardware_exclusion_mask(
    support_mask: np.ndarray,
    valid_domain: np.ndarray,
    lateral_fraction: float = 0.028,
    downward_fraction: float = 0.17,
    lower_start_fraction: float = 0.40,
    minimum_lateral_px: int = 6,
    minimum_downward_px: int = 12,
    maximum_lateral_px: int = 64,
    maximum_downward_px: int = 96,
) -> Tuple[np.ndarray, dict]:
    """Construye una falda de exclusión para el cuerpo visible de la plataforma.

    ``support_mask`` sigue representando exclusivamente la superficie superior
    útil del plato y NO se modifica. Esta función crea una segunda máscara,
    exterior y dirigida hacia abajo, para cubrir el aro/cuerpo gris del
    hardware que puede aparecer como falso foreground por pequeñas variaciones
    de iluminación.

    La expansión es deliberadamente asimétrica:
    - un margen lateral pequeño cubre el borde físico que sobresale;
    - la mayor expansión se hace hacia abajo, donde está el cuerpo del plato;
    - la parte alta de la elipse nunca se expande, para no recortar el objeto.

    El resultado es siempre disjunto de ``support_mask``.
    """
    support_u8 = (np.asarray(support_mask) > 0).astype(np.uint8) * 255
    valid = np.asarray(valid_domain) > 0

    if support_u8.shape != valid.shape:
        raise ValueError(
            "support_mask y valid_domain deben tener la misma forma: "
            f"{support_u8.shape} vs {valid.shape}"
        )

    zero = np.zeros_like(support_u8)
    ys, xs = np.nonzero(support_u8)
    if xs.size < 1000:
        return zero, {
            "status": "disabled",
            "reason": "support_not_available",
            "support_pixels": int(xs.size),
        }

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)

    lateral_px = int(
        np.clip(
            max(int(minimum_lateral_px), round(float(lateral_fraction) * width)),
            0,
            max(0, int(maximum_lateral_px)),
        )
    )
    downward_px = int(
        np.clip(
            max(int(minimum_downward_px), round(float(downward_fraction) * height)),
            0,
            max(0, int(maximum_downward_px)),
        )
    )

    # Anchor en la fila inferior: OpenCV dilata hacia abajo, no hacia arriba.
    # Así se conserva exactamente el borde superior que ya funciona bien.
    kernel = np.ones(
        (downward_px + 1, 2 * lateral_px + 1),
        dtype=np.uint8,
    )
    expanded = cv2.dilate(
        support_u8,
        kernel,
        anchor=(lateral_px, downward_px),
        iterations=1,
    )

    skirt = (expanded > 0) & (support_u8 == 0) & valid

    # El aro gris solo es visible en la mitad baja de la plataforma. Limitar la
    # falda aquí evita introducir un veto alrededor del arco superior, donde
    # puede encontrarse el objeto.
    start_fraction = float(np.clip(lower_start_fraction, 0.0, 1.0))
    start_y = int(round(y0 + start_fraction * height))
    skirt[: max(0, min(skirt.shape[0], start_y)), :] = False

    result = skirt.astype(np.uint8) * 255
    return result, {
        "status": "active",
        "principle": "directional_lower_hardware_skirt_without_changing_support_surface",
        "support_bbox_xyxy": [x0, y0, x1, y1],
        "support_width_px": int(width),
        "support_height_px": int(height),
        "lateral_fraction": float(lateral_fraction),
        "downward_fraction": float(downward_fraction),
        "lower_start_fraction": float(start_fraction),
        "lateral_px": int(lateral_px),
        "downward_px": int(downward_px),
        "start_y_px": int(start_y),
        "support_pixels": int(np.count_nonzero(support_u8)),
        "exclusion_pixels": int(np.count_nonzero(result)),
        "overlap_with_support_pixels": int(
            np.count_nonzero((result > 0) & (support_u8 > 0))
        ),
    }


def capture_volume_from_support(
    support_mask: np.ndarray,
    valid_domain: np.ndarray,
    lateral_margin_fraction: float = 0.10,
    bottom_margin_fraction: float = 0.03,
) -> Tuple[np.ndarray, dict]:
    """Volumen 2D derivado del plato detectado, no de la forma del objeto."""
    support = np.asarray(support_mask) > 0
    domain = (np.asarray(valid_domain) > 0).astype(np.uint8) * 255
    h, w = support.shape

    ys, xs = np.nonzero(support)
    if xs.size < 1000:
        return domain.copy(), {
            "status": "full_domain_fallback",
            "reason": "support_not_available",
            "capture_bbox_xyxy": [0, 0, w, h],
        }

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    support_width = max(1, x1 - x0)

    margin_x = max(
        20,
        int(round(float(lateral_margin_fraction) * support_width)),
    )
    margin_y = max(
        8,
        int(round(float(bottom_margin_fraction) * h)),
    )

    cx0 = max(0, x0 - margin_x)
    cx1 = min(w, x1 + margin_x)
    cy0 = 0
    cy1 = min(h, y1 + margin_y)

    volume = np.zeros((h, w), dtype=np.uint8)
    volume[cy0:cy1, cx0:cx1] = 255
    volume = cv2.bitwise_and(volume, domain)

    return volume, {
        "status": "support_derived",
        "support_bbox_xyxy": [x0, y0, x1, y1],
        "capture_bbox_xyxy": [cx0, cy0, cx1, cy1],
        "lateral_margin_px": int(margin_x),
        "bottom_margin_px": int(margin_y),
        "lateral_margin_fraction": float(lateral_margin_fraction),
        "bottom_margin_fraction": float(bottom_margin_fraction),
        "coverage_ratio_of_valid_domain": float(
            np.count_nonzero(volume) / max(np.count_nonzero(domain), 1)
        ),
    }


def graphcut_recover_foreground(
    image: np.ndarray,
    initial_foreground: np.ndarray,
    probable_foreground: np.ndarray,
    anchor_mask: np.ndarray,
    support_mask: np.ndarray,
    capture_volume: np.ndarray,
    iterations: int = 3,
    sure_fg_erosion_px: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Recuperación global conservadora mediante OpenCV GrabCut.

    Nunca borra foreground ya aceptado. GrabCut estudia únicamente
    (anchor ∪ support_mask) ∩ capture_volume. `probable_foreground` aporta
    semillas positivas, pero no funciona como recorte duro.
    """
    img = np.asarray(image)
    initial = np.asarray(initial_foreground) > 0
    probable = np.asarray(probable_foreground) > 0
    anchor = np.asarray(anchor_mask) > 0
    support = np.asarray(support_mask) > 0
    capture = np.asarray(capture_volume) > 0

    h, w = initial.shape
    zero = np.zeros((h, w), dtype=np.uint8)

    if img.shape[:2] != (h, w):
        raise ValueError("GraphCut: imagen y máscaras no coinciden.")

    # Zona de búsqueda global:
    # - anchor: región visual alrededor del objeto ya detectado;
    # - support: cualquier zona del plato que el objeto podría estar ocultando.
    #
    # Esto NO convierte el plato en foreground. Solo permite que GrabCut
    # decida globalmente si un píxel del soporte está siendo ocluido.
    search_zone = (anchor | support) & capture
    probable_seed = probable & search_zone

    initial_u8 = initial.astype(np.uint8) * 255
    er = max(1, int(sure_fg_erosion_px))
    sure_fg = (
        cv2.erode(
            initial_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * er + 1, 2 * er + 1),
            ),
            iterations=1,
        )
        > 0
    )

    if np.count_nonzero(sure_fg) < 500:
        return (
            initial_u8,
            zero,
            zero,
            {
                "status": "fallback",
                "reason": "insufficient_sure_foreground",
                "recovered_pixels": 0,
            },
        )

    gc = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # Fondo seguro fuera del volumen y fuera de la zona de búsqueda.
    gc[~capture] = cv2.GC_BGD
    gc[capture & (~search_zone)] = cv2.GC_BGD

    # Evidencia local positiva: foreground probable.
    gc[probable_seed] = cv2.GC_PR_FGD

    # La máscara aceptada permanece como foreground.
    gc[initial & capture] = cv2.GC_PR_FGD
    gc[sure_fg & capture] = cv2.GC_FGD

    if np.count_nonzero(gc == cv2.GC_BGD) < 500 or np.count_nonzero(gc == cv2.GC_FGD) < 500:
        return (
            initial_u8,
            zero,
            gc,
            {
                "status": "fallback",
                "reason": "insufficient_fg_or_bg_seeds",
                "recovered_pixels": 0,
            },
        )

    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            img,
            gc,
            None,
            bg_model,
            fg_model,
            max(1, int(iterations)),
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error as exc:
        return (
            initial_u8,
            zero,
            gc,
            {
                "status": "fallback",
                "reason": f"opencv_grabcut_error: {exc}",
                "recovered_pixels": 0,
            },
        )

    gc_fg = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)

    # La clasificación global puede recuperar píxeles que los filtros
    # locales no marcaron como probables, pero únicamente dentro de la zona
    # físicamente plausible (anchor ∪ soporte).
    recovered = gc_fg & search_zone & (~initial)
    refined = initial | recovered

    return (
        refined.astype(np.uint8) * 255,
        recovered.astype(np.uint8) * 255,
        gc,
        {
            "status": "ok",
            "iterations": int(iterations),
            "initial_pixels": int(np.count_nonzero(initial)),
            "probable_pixels": int(np.count_nonzero(probable)),
            "probable_seed_pixels": int(np.count_nonzero(probable_seed)),
            "search_zone_pixels": int(np.count_nonzero(search_zone)),
            "support_pixels": int(np.count_nonzero(support)),
            "sure_foreground_pixels": int(np.count_nonzero(sure_fg)),
            "recovered_pixels": int(np.count_nonzero(recovered)),
            "final_pixels": int(np.count_nonzero(refined)),
            "recovery_ratio_vs_initial": float(
                np.count_nonzero(recovered) / max(np.count_nonzero(initial), 1)
            ),
        },
    )


def graphcut_trimap_visualization(trimap: np.ndarray) -> np.ndarray:
    """Negro=fondo seguro, gris=fondo probable, verde=FG probable, blanco=FG seguro."""
    gc = np.asarray(trimap, dtype=np.uint8)
    vis = np.zeros((*gc.shape, 3), dtype=np.uint8)
    vis[gc == cv2.GC_BGD] = (0, 0, 0)
    vis[gc == cv2.GC_PR_BGD] = (80, 80, 80)
    vis[gc == cv2.GC_PR_FGD] = (0, 180, 0)
    vis[gc == cv2.GC_FGD] = (255, 255, 255)
    return vis


def background_depth_consistency_diagnostics(
    current_depth_mm: np.ndarray,
    background_depth_mm: np.ndarray,
    rect_valid_mask: np.ndarray,
    visual_foreground_mask: np.ndarray,
) -> dict:
    current = np.asarray(current_depth_mm, dtype=np.float32)
    background = np.asarray(background_depth_mm, dtype=np.float32)
    stable = (
        (np.asarray(rect_valid_mask) > 0)
        & (np.asarray(visual_foreground_mask) == 0)
        & np.isfinite(current)
        & np.isfinite(background)
    )
    delta = (background - current)[stable]
    if delta.size < 1000:
        return {
            "sample_count": int(delta.size),
            "reliable_for_foreground_segmentation": False,
            "reason": "insufficient_static_samples",
        }
    absolute = np.abs(delta)
    median_abs = float(np.median(absolute))
    p90 = float(np.percentile(absolute, 90))
    ratio20 = float(np.mean(absolute > 20.0))
    reliable = bool(median_abs <= 12.0 and p90 <= 35.0 and ratio20 <= 0.25)
    return {
        "sample_count": int(delta.size),
        "median_delta_mm": float(np.median(delta)),
        "median_abs_delta_mm": median_abs,
        "p90_abs_delta_mm": p90,
        "p95_abs_delta_mm": float(np.percentile(absolute, 95)),
        "ratio_abs_gt_20mm": ratio20,
        "ratio_abs_gt_50mm": float(np.mean(absolute > 50.0)),
        "reliable_for_foreground_segmentation": reliable,
        "reason": (
            None if reliable else "background_depth_not_stable_enough_on_visually_static_pixels"
        ),
    }


def fixed_depth_visualization(
    depth_mm: np.ndarray,
    valid_mask: np.ndarray,
    minimum_depth_mm: float,
    maximum_depth_mm: float,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Escala absoluta común. Cercano=rojo; lejano=azul/violeta."""
    depth = np.asarray(depth_mm, dtype=np.float32)
    valid = np.asarray(valid_mask).astype(bool) & np.isfinite(depth)

    lo, hi = float(minimum_depth_mm), float(maximum_depth_mm)
    if hi <= lo:
        raise ValueError("maximum_depth_mm debe ser mayor que minimum_depth_mm")

    norm = np.zeros(depth.shape, np.float32)
    norm[valid] = (depth[valid] - lo) / (hi - lo)
    norm = 1.0 - np.clip(norm, 0.0, 1.0)

    gray = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, colormap)
    color[~valid] = 0

    # Leyenda solo si cabe; nunca debe romper procesamiento por diagnóstico.
    h, w = depth.shape
    if h >= 160 and w >= 240:
        y0 = max(10, int(round(0.05 * h)))
        bar_h = min(max(100, int(round(0.32 * h))), h - y0 - 12)
        bar_w = max(16, min(28, int(round(0.018 * w))))
        x0 = max(85, w - bar_w - 18)

        if bar_h > 20 and x0 + bar_w <= w:
            gradient = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(-1, 1)
            gradient = np.repeat(gradient, bar_w, axis=1)
            grad_color = cv2.applyColorMap(gradient, colormap)
            color[y0 : y0 + bar_h, x0 : x0 + bar_w] = grad_color

            cv2.putText(
                color,
                f"{lo:.0f} mm",
                (max(2, x0 - 80), y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                color,
                f"{hi:.0f} mm",
                (max(2, x0 - 80), min(h - 8, y0 + bar_h)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return color


def ensure_same_shape(name: str, arrays: Sequence[np.ndarray]) -> None:
    """Valida que las matrices tengan dimensiones compatibles."""
    shapes = [array.shape[:2] for array in arrays]
    if len(set(shapes)) != 1:
        raise ValueError(f"Dimensiones incompatibles en {name}: {shapes}")


def central_roi(
    shape: Tuple[int, int],
    width_fraction: float,
    height_fraction: float,
    center_y_fraction: float = 0.50,
) -> np.ndarray:
    h, w = shape
    roi_w = max(10, int(round(w * width_fraction)))
    roi_h = max(10, int(round(h * height_fraction)))
    cx = w // 2
    cy = int(round(h * center_y_fraction))
    x0 = max(0, cx - roi_w // 2)
    x1 = min(w, cx + roi_w // 2)
    y0 = max(0, cy - roi_h // 2)
    y1 = min(h, cy + roi_h // 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def select_relevant_components(
    mask: np.ndarray,
    minimum_area: int,
    secondary_fraction: float,
    prior_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = mask.shape
    expected = np.array([w * 0.50, h * 0.50], dtype=np.float64)
    diagonal = max(math.hypot(w, h), 1.0)

    components = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        centroid = np.asarray(centroids[label], dtype=np.float64)
        distance = float(np.linalg.norm(centroid - expected))
        center_score = math.exp(-4.0 * distance / diagonal)
        compactness = area / max(float(width * height), 1.0)

        prior_overlap = 0.0
        if prior_mask is not None:
            component = labels == label
            prior_pixels = int(np.count_nonzero(component & (prior_mask > 0)))
            prior_overlap = prior_pixels / max(area, 1)

        score = (
            float(area)
            * (0.40 + 0.40 * center_score + 0.20 * prior_overlap)
            * (0.70 + 0.30 * compactness)
        )
        components.append((score, label, area))

    if not components:
        return np.zeros_like(mask, dtype=np.uint8)

    components.sort(reverse=True)
    best_score = components[0][0]
    output = np.zeros_like(mask, dtype=np.uint8)
    for score, label, _ in components:
        if score < secondary_fraction * best_score:
            continue
        output[labels == label] = 255
    return output


def fill_small_holes(mask: np.ndarray, maximum_hole_area: int) -> np.ndarray:
    foreground = mask > 0
    inverse = (~foreground).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    h, w = mask.shape
    output = mask.copy()
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_border = x == 0 or y == 0 or x + width >= w or y + height >= h
        if not touches_border and area <= maximum_hole_area:
            output[labels == label] = 255
    return output


def hysteresis_components(weak: np.ndarray, strong: np.ndarray) -> np.ndarray:
    weak_binary = (weak > 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(weak_binary, connectivity=8)
    strong_labels = np.unique(labels[strong > 0])
    strong_labels = strong_labels[strong_labels != 0]
    return np.isin(labels, strong_labels).astype(np.uint8) * 255


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> Optional[float]:
    """Calcula la intersección sobre unión de las máscaras binarias."""
    a = mask_a > 0
    b = mask_b > 0
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return None
    return float(np.count_nonzero(a & b) / union)


def overlay_mask(
    image: np.ndarray, mask: np.ndarray, label: str, probability: Optional[np.ndarray] = None
) -> np.ndarray:
    """Superpone una máscara coloreada sobre una imagen para diagnóstico."""
    output = image.copy()
    tint = np.zeros_like(output)
    tint[:, :, 1] = mask
    output = cv2.addWeighted(output, 1.0, tint, 0.35, 0.0)
    if probability is not None:
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, contours, -1, (0, 255, 255), 2)
    cv2.putText(
        output,
        label,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        label,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def scalar_visualization(
    array: np.ndarray, valid: np.ndarray, colormap: int = cv2.COLORMAP_TURBO, invert: bool = False
) -> np.ndarray:
    gray = np.zeros(array.shape, dtype=np.uint8)
    values = array[valid & np.isfinite(array)]
    if values.size:
        low, high = np.percentile(values, [2, 98])
        if high <= low:
            high = low + 1e-6
        normalized = (np.clip(array, low, high) - low) / (high - low)
        normalized = np.nan_to_num(normalized, nan=0.0)
        if invert:
            normalized = 1.0 - normalized
        gray[valid] = np.clip(normalized[valid] * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray, colormap)


def color_labels(labels: np.ndarray, accepted: Optional[Dict[int, str]] = None):
    """Genera una visualización coloreada del mapa de etiquetas."""
    output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    unique = np.unique(labels)
    rng = np.random.default_rng(500)
    palette = rng.integers(30, 230, size=(max(len(unique), 1), 3), dtype=np.uint8)
    for index, label in enumerate(unique):
        if label < 0:
            continue
        if accepted is None:
            output[labels == label] = palette[index % len(palette)]
        else:
            status = accepted.get(int(label), "ignored")
            color = {
                "accepted": (40, 210, 40),
                "rejected": (40, 40, 220),
                "warning": (30, 180, 230),
                "ignored": (80, 80, 80),
            }.get(status, (80, 80, 80))
            output[labels == label] = color
    return output


def save_ply_ascii(
    path: Path, points: np.ndarray, colors_rgb: np.ndarray, normals: Optional[np.ndarray] = None
) -> None:
    """Exporta posiciones y colores de una nube en formato PLY ASCII."""
    include_normals = normals is not None and len(normals) == len(points)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        if include_normals:
            file.write("property float nx\nproperty float ny\nproperty float nz\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("end_header\n")
        for index, point in enumerate(points):
            color = colors_rgb[index]
            if include_normals:
                normal = normals[index]
                file.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )
            else:
                file.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )


def projection_panel(
    points: np.ndarray,
    colors_rgb: np.ndarray,
    axis_x: int,
    axis_y: int,
    width: int,
    height: int,
    title: str,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    if len(points) == 0:
        cv2.putText(
            canvas, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA
        )
        return canvas

    indexes = np.arange(len(points))
    if len(indexes) > 250_000:
        rng = np.random.default_rng(500)
        indexes = rng.choice(indexes, 250_000, replace=False)
    p = points[indexes]
    c = colors_rgb[indexes][:, ::-1]
    c = np.minimum(c, np.array([140, 140, 140], dtype=np.uint8))

    x = p[:, axis_x]
    y = p[:, axis_y]
    x0, x1 = np.percentile(x, [1, 99])
    y0, y1 = np.percentile(y, [1, 99])
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= y0:
        y1 = y0 + 1.0
    px = (20 + (x - x0) / (x1 - x0) * (width - 40)).astype(np.int32)
    py = (20 + (1.0 - (y - y0) / (y1 - y0)) * (height - 40)).astype(np.int32)
    good = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    canvas[py[good], px[good]] = c[good]
    cv2.putText(
        canvas, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA
    )
    return canvas


def make_cloud_preview(points: np.ndarray, colors_rgb: np.ndarray, label: str, path: Path) -> None:
    width, height = 620, 450
    xy = projection_panel(points, colors_rgb, 0, 1, width, height, "X-Y")
    xz = projection_panel(points, colors_rgb, 0, 2, width, height, "X-Z")
    zy = projection_panel(points, colors_rgb, 2, 1, width, height, "Z-Y")
    info = np.full_like(zy, 248)
    lines = [label, f"Puntos: {len(points)}"]
    for index, line in enumerate(lines):
        cv2.putText(
            info,
            line,
            (30, 60 + index * 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
    preview = np.vstack((np.hstack((xy, xz)), np.hstack((zy, info))))
    cv2.imwrite(str(path), preview)


def build_contact_sheet(
    paths: Sequence[Path], output_path: Path, columns: int = 4, width: int = 480
) -> None:
    """Compone una hoja de contacto con las imágenes legibles de la lista."""
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = width / image.shape[1]
        resized = cv2.resize(
            image,
            (width, max(1, int(round(image.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        images.append(resized)
    if not images:
        return
    height = max(image.shape[0] for image in images)
    rows = int(math.ceil(len(images) / columns))
    sheet = np.full((rows * height, columns * width, 3), 248, dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        y0, x0 = row * height, column * width
        sheet[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1]] = image
    cv2.imwrite(str(output_path), sheet)
