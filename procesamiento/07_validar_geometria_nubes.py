#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Paso 07 — Validación geométrica métrica de nubes parciales.

Objetivo
--------
Evaluar cada nube producida por el paso 06 ANTES de cualquier registro.
Este paso NO transforma, alinea ni fusiona las nubes.

Para cada vista:
1. Carga la nube final (.npz) en coordenadas de cámara.
2. Extrae hasta N planos dominantes mediante RANSAC + refinamiento SVD.
3. Calcula métricas por plano:
   - proporción de puntos explicados;
   - RMSE, MAD y P95 punto-plano;
   - espesor robusto en la dirección normal;
   - estabilidad angular de normales;
   - extensión física y densidad aproximada.
4. Evalúa la relación angular entre planos.
5. Detecta posibles capas paralelas separadas.
6. Clasifica cada vista como accepted / warning / rejected.
7. Exporta JSON, CSV, NPZ, PLY coloreado y hojas de contacto.

Principio de diseño
-------------------
Una nube incompleta NO se rechaza por tener huecos. Se penaliza geometría
inconsistente (planos gruesos, residuos altos, capas dobles, etc.).

Compatibilidad
--------------
Está diseñado para consumir directamente:
    06_nubes_puntos/
        resumen_06_nubes_puntos.json
        *_cloud_final.npz

No usa ArUco, marcadores fiduciales ni información externa de pose.
"""

from __future__ import annotations
from utilidades_progreso import operacion
from utilidades_rendimiento import configurar as _configurar_recursos

# Los runtimes numéricos leen estos límites al importarse.
_configurar_recursos(__file__)


import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utilidades_mascaras import (
    build_contact_sheet,
    find_summary,
    load_json,
    prepare_output_directory,
    save_json,
    save_ply_ascii,
    validate_summary_context,
)

# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construye las opciones de línea de comandos de este paso."""
    parser = argparse.ArgumentParser(
        description=(
            "Paso 07: valida geométricamente las nubes parciales antes " "del registro multivista."
        )
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--object", default="cubo")
    parser.add_argument("--session", default="S01")
    parser.add_argument(
        "--cloud-source",
        default="06_nubes_puntos",
    )
    parser.add_argument(
        "--output-name",
        default="07_validacion_geometrica",
    )
    parser.add_argument("--only-angle", type=float, default=-1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=500)

    # Extracción de planos.
    parser.add_argument("--maximum-planes", type=int, default=3)
    parser.add_argument("--ransac-iterations", type=int, default=1200)
    parser.add_argument("--ransac-evaluation-points", type=int, default=9000)
    parser.add_argument("--plane-distance-threshold-mm", type=float, default=3.5)
    parser.add_argument("--minimum-plane-points", type=int, default=1200)
    parser.add_argument("--minimum-plane-ratio-total", type=float, default=0.045)
    parser.add_argument("--minimum-plane-ratio-remaining", type=float, default=0.10)
    parser.add_argument("--refinement-rounds", type=int, default=4)

    # Métricas y calidad.
    parser.add_argument("--minimum-cloud-points", type=int, default=4500)
    parser.add_argument("--minimum-best-plane-ratio", type=float, default=0.18)
    parser.add_argument("--minimum-explained-ratio", type=float, default=0.55)
    parser.add_argument("--warning-plane-rmse-mm", type=float, default=2.6)
    parser.add_argument("--warning-plane-p95-mm", type=float, default=4.5)
    parser.add_argument("--warning-plane-thickness95-mm", type=float, default=7.0)
    parser.add_argument("--warning-normal-p95-deg", type=float, default=32.0)
    parser.add_argument("--rejection-best-plane-rmse-mm", type=float, default=4.5)

    # Capas / relaciones entre planos.
    parser.add_argument("--parallel-angle-deg", type=float, default=12.0)
    parser.add_argument("--layer-minimum-separation-mm", type=float, default=4.0)
    parser.add_argument("--layer-minimum-plane-ratio", type=float, default=0.07)

    # Para el cuboide únicamente: caras visibles distintas deberían ser
    # aproximadamente ortogonales, salvo fragmentación coplanar/paralela.
    parser.add_argument(
        "--geometry-mode",
        choices=("auto", "generic", "cuboid"),
        default="auto",
    )
    parser.add_argument("--cuboid-minimum-orthogonal-angle-deg", type=float, default=72.0)
    parser.add_argument("--cuboid-major-plane-ratio", type=float, default=0.10)

    # Visualización.
    parser.add_argument("--preview-width", type=int, default=620)
    parser.add_argument("--preview-height", type=int, default=450)
    return parser


# ---------------------------------------------------------------------------
# Utilidades geométricas
# ---------------------------------------------------------------------------


def finite_points(points: np.ndarray) -> np.ndarray:
    """Devuelve una máscara de los puntos con tres coordenadas finitas."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"La nube debe tener forma Nx3; se recibió {points.shape}")
    return np.all(np.isfinite(points), axis=1)


def normalize_vector(vector: np.ndarray) -> Optional[np.ndarray]:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-12:
        return None
    return vector / norm


def plane_from_three(points3: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """Construye un plano a partir de tres puntos no colineales."""
    p0, p1, p2 = np.asarray(points3, dtype=np.float64)
    normal = np.cross(p1 - p0, p2 - p0)
    normal = normalize_vector(normal)
    if normal is None:
        return None
    d = -float(np.dot(normal, p0))
    return normal, d


def fit_plane_svd(points: np.ndarray) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    """Ajusta un plano mediante la descomposición en valores singulares."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return None
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    try:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if vt.shape != (3, 3):
        return None
    normal = normalize_vector(vt[-1])
    if normal is None:
        return None
    d = -float(np.dot(normal, centroid))
    return normal, d, singular_values


def signed_plane_distance(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """Evalúa la distancia firmada de los puntos al plano."""
    return np.asarray(points, dtype=np.float64) @ normal + float(d)


def acute_normal_angle_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    """Calcula en grados el ángulo agudo entre normales de planos."""
    dot = float(np.clip(abs(np.dot(n1, n2)), 0.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def plane_separation_mm(
    n1: np.ndarray,
    d1: float,
    n2: np.ndarray,
    d2: float,
) -> float:
    """Separación aproximada entre dos planos casi paralelos."""
    if float(np.dot(n1, n2)) < 0.0:
        n2 = -n2
        d2 = -d2
    return abs(float(d1) - float(d2))


def robust_span(values: np.ndarray, low: float = 2.5, high: float = 97.5) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    lo, hi = np.percentile(values, [low, high])
    return float(hi - lo)


def plane_basis(points: np.ndarray, normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    centered = points - np.mean(points, axis=0)
    # La SVD de los puntos del plano entrega dos ejes tangenciales estables.
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        u = normalize_vector(vt[0])
    except np.linalg.LinAlgError:
        u = None

    if u is None or abs(float(np.dot(u, normal))) > 0.25:
        # Construcción determinista de un vector no paralelo a la normal.
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(helper, normal))) > 0.85:
            helper = np.array([0.0, 1.0, 0.0])
        u = normalize_vector(np.cross(normal, helper))
        if u is None:
            u = np.array([1.0, 0.0, 0.0])

    # Elimina cualquier componente residual normal.
    u = u - float(np.dot(u, normal)) * normal
    u = normalize_vector(u)
    if u is None:
        u = np.array([1.0, 0.0, 0.0])
    v = normalize_vector(np.cross(normal, u))
    if v is None:
        v = np.array([0.0, 1.0, 0.0])
    return u, v


# ---------------------------------------------------------------------------
# RANSAC de planos
# ---------------------------------------------------------------------------


def ransac_plane(
    points: np.ndarray,
    args,
    rng: np.random.Generator,
) -> Optional[dict]:
    """Busca un plano dominante con muestreo RANSAC y refinamiento de sus inliers."""
    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    if count < max(3, int(args.minimum_plane_points)):
        return None

    evaluation_count = min(count, int(args.ransac_evaluation_points))
    if evaluation_count < count:
        evaluation_indices = rng.choice(count, evaluation_count, replace=False)
        evaluation_points = points[evaluation_indices]
    else:
        evaluation_points = points

    threshold = float(args.plane_distance_threshold_mm)
    best_model = None
    best_count = 0
    best_median = float("inf")

    for _ in range(int(args.ransac_iterations)):
        sample_idx = rng.choice(evaluation_count, 3, replace=False)
        model = plane_from_three(evaluation_points[sample_idx])
        if model is None:
            continue
        normal, d = model
        distances = np.abs(signed_plane_distance(evaluation_points, normal, d))
        inliers = distances <= threshold
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 3:
            continue
        median = float(np.median(distances[inliers]))
        if inlier_count > best_count or (inlier_count == best_count and median < best_median):
            best_model = (normal, d)
            best_count = inlier_count
            best_median = median

    if best_model is None:
        return None

    normal, d = best_model
    # Extiende el modelo candidato a todos los puntos y lo refina por SVD.
    inliers = np.abs(signed_plane_distance(points, normal, d)) <= threshold

    for _ in range(max(1, int(args.refinement_rounds))):
        if np.count_nonzero(inliers) < 3:
            return None
        refined = fit_plane_svd(points[inliers])
        if refined is None:
            return None
        normal, d, _ = refined
        new_inliers = np.abs(signed_plane_distance(points, normal, d)) <= threshold
        if np.array_equal(new_inliers, inliers):
            inliers = new_inliers
            break
        inliers = new_inliers

    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < int(args.minimum_plane_points):
        return None

    return {
        "normal": normal,
        "d": float(d),
        "inliers": inliers,
        "inlier_count": inlier_count,
    }


def extract_dominant_planes(
    points: np.ndarray,
    normals: np.ndarray,
    args,
    rng: np.random.Generator,
) -> Tuple[List[dict], np.ndarray]:
    """Extrae sucesivamente planos dominantes y registra su respaldo geométrico."""
    total = len(points)
    labels = np.full(total, -1, dtype=np.int32)
    remaining_indices = np.arange(total, dtype=np.int64)
    planes: List[dict] = []

    for plane_index in range(int(args.maximum_planes)):
        if len(remaining_indices) < int(args.minimum_plane_points):
            break

        model = ransac_plane(points[remaining_indices], args, rng)
        if model is None:
            break

        local_inliers = model["inliers"]
        global_indices = remaining_indices[local_inliers]
        count = len(global_indices)
        ratio_total = count / max(total, 1)
        ratio_remaining = count / max(len(remaining_indices), 1)

        if ratio_total < float(args.minimum_plane_ratio_total):
            break
        if ratio_remaining < float(args.minimum_plane_ratio_remaining):
            break

        normal = np.asarray(model["normal"], dtype=np.float64)
        d = float(model["d"])
        plane_points = points[global_indices]
        signed = signed_plane_distance(plane_points, normal, d)
        absolute = np.abs(signed)

        # Métricas punto-plano.
        rmse = float(np.sqrt(np.mean(np.square(signed))))
        signed_median = float(np.median(signed))
        mad = float(np.median(np.abs(signed - signed_median)))
        p95 = float(np.percentile(absolute, 95))
        thickness95 = robust_span(signed, 2.5, 97.5)

        # Extensión sobre el propio plano.
        u, v = plane_basis(plane_points, normal)
        centered = plane_points - np.mean(plane_points, axis=0)
        coord_u = centered @ u
        coord_v = centered @ v
        extent_u = robust_span(coord_u, 1.0, 99.0)
        extent_v = robust_span(coord_v, 1.0, 99.0)
        area_approx = max(extent_u * extent_v, 1e-9)
        density = float(count / area_approx)

        # Coherencia entre normales estimadas en 06 y normal del plano.
        normal_median = None
        normal_p95 = None
        valid_normal_count = 0
        if normals.ndim == 2 and normals.shape == points.shape:
            plane_normals = normals[global_indices].astype(np.float64)
            lengths = np.linalg.norm(plane_normals, axis=1)
            good = np.isfinite(lengths) & (lengths > 1e-6)
            good &= np.all(np.isfinite(plane_normals), axis=1)
            if np.any(good):
                unit = plane_normals[good] / lengths[good, None]
                dots = np.clip(np.abs(unit @ normal), 0.0, 1.0)
                angles = np.degrees(np.arccos(dots))
                normal_median = float(np.median(angles))
                normal_p95 = float(np.percentile(angles, 95))
                valid_normal_count = int(len(angles))

        labels[global_indices] = plane_index
        planes.append(
            {
                "plane_index": int(plane_index),
                "point_count": int(count),
                "ratio_total": float(ratio_total),
                "ratio_remaining_at_detection": float(ratio_remaining),
                "normal_xyz": normal.astype(float).tolist(),
                "d_mm": float(d),
                "centroid_xyz_mm": np.mean(plane_points, axis=0).astype(float).tolist(),
                "rmse_mm": rmse,
                "mad_mm": mad,
                "p95_abs_distance_mm": p95,
                "thickness95_mm": float(thickness95),
                "extent_u_mm": float(extent_u),
                "extent_v_mm": float(extent_v),
                "area_approx_mm2": float(area_approx),
                "density_points_per_mm2": density,
                "normal_angle_median_deg": normal_median,
                "normal_angle_p95_deg": normal_p95,
                "normal_valid_count": valid_normal_count,
            }
        )

        remaining_indices = remaining_indices[~local_inliers]

    return planes, labels


# ---------------------------------------------------------------------------
# Relaciones entre planos / diagnóstico de capas
# ---------------------------------------------------------------------------


def analyze_plane_relations(planes: Sequence[dict], args, geometry_mode: str) -> dict:
    """Evalúa relaciones angulares y separación entre los planos detectados."""
    pairs = []
    separated_parallel_pairs = []
    cuboid_angle_warnings = []

    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            a = planes[i]
            b = planes[j]
            n1 = np.asarray(a["normal_xyz"], dtype=np.float64)
            n2 = np.asarray(b["normal_xyz"], dtype=np.float64)
            angle = acute_normal_angle_deg(n1, n2)
            separation = None
            relation = "oblique"

            if angle <= float(args.parallel_angle_deg):
                relation = "parallel_or_coplanar"
                separation = plane_separation_mm(n1, float(a["d_mm"]), n2, float(b["d_mm"]))
                if (
                    separation >= float(args.layer_minimum_separation_mm)
                    and a["ratio_total"] >= float(args.layer_minimum_plane_ratio)
                    and b["ratio_total"] >= float(args.layer_minimum_plane_ratio)
                ):
                    relation = "parallel_separated_layers"
                    separated_parallel_pairs.append(
                        {
                            "plane_a": int(i),
                            "plane_b": int(j),
                            "angle_deg": float(angle),
                            "separation_mm": float(separation),
                        }
                    )
            elif angle >= float(args.cuboid_minimum_orthogonal_angle_deg):
                relation = "approximately_orthogonal"

            pair = {
                "plane_a": int(i),
                "plane_b": int(j),
                "angle_deg": float(angle),
                "relation": relation,
                "separation_mm": (None if separation is None else float(separation)),
            }
            pairs.append(pair)

            if geometry_mode == "cuboid":
                major = a["ratio_total"] >= float(args.cuboid_major_plane_ratio) and b[
                    "ratio_total"
                ] >= float(args.cuboid_major_plane_ratio)
                # Dos caras mayores distintas de un cuboide deberían ser
                # paralelas/coplanares o cercanas a 90°. Un ángulo intermedio
                # es una señal de posible geometría falsa.
                if (
                    major
                    and angle > float(args.parallel_angle_deg)
                    and angle < float(args.cuboid_minimum_orthogonal_angle_deg)
                ):
                    cuboid_angle_warnings.append(
                        {
                            "plane_a": int(i),
                            "plane_b": int(j),
                            "angle_deg": float(angle),
                        }
                    )

    return {
        "pairs": pairs,
        "separated_parallel_pairs": separated_parallel_pairs,
        "cuboid_angle_warnings": cuboid_angle_warnings,
    }


# ---------------------------------------------------------------------------
# Clasificación de calidad
# ---------------------------------------------------------------------------


def classify_view(
    point_count: int,
    planes: Sequence[dict],
    relations: dict,
    source_quality: str,
    source_reasons: Sequence[str],
    args,
) -> Tuple[str, List[str], dict]:
    """Asigna calidad a la vista a partir de sus evidencias geométricas."""
    reasons: List[str] = []
    severe: List[str] = []

    explained_ratio = float(sum(p["ratio_total"] for p in planes))
    best_plane_ratio = float(max((p["ratio_total"] for p in planes), default=0.0))
    best_plane_rmse = float(min((p["rmse_mm"] for p in planes), default=float("inf")))

    if point_count < int(args.minimum_cloud_points):
        severe.append(
            f"Nube demasiado pequeña: {point_count} < {args.minimum_cloud_points} puntos."
        )

    if not planes:
        severe.append("No se encontró ningún plano dominante robusto.")
    else:
        if best_plane_ratio < float(args.minimum_best_plane_ratio):
            severe.append(
                "Ningún plano explica una fracción suficiente de la nube: "
                f"{best_plane_ratio:.2%}."
            )
        if best_plane_rmse > float(args.rejection_best_plane_rmse_mm):
            severe.append(
                "Incluso el mejor plano presenta RMSE excesivo: " f"{best_plane_rmse:.2f} mm."
            )

    if planes and explained_ratio < float(args.minimum_explained_ratio):
        reasons.append(
            "Cobertura geométrica por planos relativamente baja: " f"{explained_ratio:.2%}."
        )

    for plane in planes:
        prefix = f"Plano {plane['plane_index'] + 1}"
        if plane["rmse_mm"] > float(args.warning_plane_rmse_mm):
            reasons.append(f"{prefix}: RMSE alto ({plane['rmse_mm']:.2f} mm).")
        if plane["p95_abs_distance_mm"] > float(args.warning_plane_p95_mm):
            reasons.append(
                f"{prefix}: P95 punto-plano alto " f"({plane['p95_abs_distance_mm']:.2f} mm)."
            )
        if plane["thickness95_mm"] > float(args.warning_plane_thickness95_mm):
            reasons.append(
                f"{prefix}: espesor robusto alto " f"({plane['thickness95_mm']:.2f} mm)."
            )
        p95_normal = plane.get("normal_angle_p95_deg")
        if p95_normal is not None and p95_normal > float(args.warning_normal_p95_deg):
            reasons.append(f"{prefix}: normales locales dispersas " f"(P95={p95_normal:.1f}°).")

    if relations["separated_parallel_pairs"]:
        reasons.append("Se detectaron planos paralelos separados compatibles con capas dobles.")
    if relations["cuboid_angle_warnings"]:
        reasons.append(
            "Hay planos dominantes cuyo ángulo no es compatible con caras "
            "paralelas ni aproximadamente ortogonales de un cuboide."
        )

    # Conserva trazabilidad de la calidad de el paso anterior.
    if source_quality == "warning":
        reasons.append("El paso 06 ya marcaba esta nube como warning.")
        reasons.extend(f"06: {reason}" for reason in source_reasons)
    elif source_quality == "rejected":
        severe.append("El paso 06 marcaba esta nube como rejected.")

    if severe:
        quality = "rejected"
        reasons = severe + reasons
    elif reasons:
        quality = "warning"
    else:
        quality = "accepted"

    metrics = {
        "explained_ratio": explained_ratio,
        "unexplained_ratio": max(0.0, 1.0 - explained_ratio),
        "best_plane_ratio": best_plane_ratio,
        "best_plane_rmse_mm": (None if not np.isfinite(best_plane_rmse) else best_plane_rmse),
        "plane_count": int(len(planes)),
        "parallel_separated_pair_count": int(len(relations["separated_parallel_pairs"])),
        "cuboid_angle_warning_count": int(len(relations["cuboid_angle_warnings"])),
    }
    return quality, reasons, metrics


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------


PLANE_COLORS_RGB = np.asarray(
    [
        [220, 70, 70],
        [70, 190, 85],
        [70, 120, 230],
    ],
    dtype=np.uint8,
)
UNASSIGNED_COLOR_RGB = np.asarray([150, 150, 150], dtype=np.uint8)


def label_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.tile(UNASSIGNED_COLOR_RGB, (len(labels), 1))
    for label in np.unique(labels):
        if label < 0:
            continue
        colors[labels == label] = PLANE_COLORS_RGB[int(label) % len(PLANE_COLORS_RGB)]
    return colors


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
        return canvas

    indexes = np.arange(len(points))
    if len(indexes) > 180_000:
        rng = np.random.default_rng(500)
        indexes = rng.choice(indexes, 180_000, replace=False)
    p = points[indexes]
    c = colors_rgb[indexes][:, ::-1]  # RGB -> BGR

    x = p[:, axis_x]
    y = p[:, axis_y]
    x0, x1 = np.percentile(x, [1, 99])
    y0, y1 = np.percentile(y, [1, 99])
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= y0:
        y1 = y0 + 1.0

    px = (24 + (x - x0) / (x1 - x0) * (width - 48)).astype(np.int32)
    py = (24 + (1.0 - (y - y0) / (y1 - y0)) * (height - 48)).astype(np.int32)
    good = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    canvas[py[good], px[good]] = c[good]

    cv2.putText(
        canvas,
        title,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return canvas


def put_text_lines(image: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    output = image.copy()
    y = 42
    for line in lines:
        cv2.putText(
            output,
            str(line),
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (35, 35, 35),
            2,
            cv2.LINE_AA,
        )
        y += 34
    return output


def make_validation_preview(
    points: np.ndarray,
    labels: np.ndarray,
    record: dict,
    path: Path,
    width: int,
    height: int,
) -> None:
    colors = label_colors(labels)
    xy = projection_panel(points, colors, 0, 1, width, height, "X-Y")
    xz = projection_panel(points, colors, 0, 2, width, height, "X-Z")
    zy = projection_panel(points, colors, 2, 1, width, height, "Z-Y")

    info = np.full((height, width, 3), 248, dtype=np.uint8)
    metrics = record["metrics"]
    lines = [
        f"{record['angle_deg']:.3f} deg | {record['quality']}",
        f"Puntos: {record['point_count']}",
        f"Planos: {metrics['plane_count']}",
        f"Explicado: {metrics['explained_ratio']:.1%}",
        f"Mejor plano: {metrics['best_plane_ratio']:.1%}",
    ]
    if metrics["best_plane_rmse_mm"] is not None:
        lines.append(f"RMSE mejor: {metrics['best_plane_rmse_mm']:.2f} mm")
    lines.append(f"Capas paralelas: {metrics['parallel_separated_pair_count']}")
    info = put_text_lines(info, lines)

    preview = np.vstack((np.hstack((xy, xz)), np.hstack((zy, info))))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), preview):
        raise OSError(f"No se pudo guardar: {path}")


# ---------------------------------------------------------------------------
# Estadísticas generales
# ---------------------------------------------------------------------------


def cloud_extent(points: np.ndarray) -> dict:
    if len(points) == 0:
        return {
            "minimum_xyz_mm": None,
            "maximum_xyz_mm": None,
            "extent_xyz_mm": None,
            "centroid_xyz_mm": None,
        }
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    return {
        "minimum_xyz_mm": minimum.astype(float).tolist(),
        "maximum_xyz_mm": maximum.astype(float).tolist(),
        "extent_xyz_mm": (maximum - minimum).astype(float).tolist(),
        "centroid_xyz_mm": np.mean(points, axis=0).astype(float).tolist(),
    }


def robust_session_stats(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "median": None, "mad": None, "min": None, "max": None}
    median = float(np.median(array))
    return {
        "count": int(array.size),
        "median": median,
        "mad": float(np.median(np.abs(array - median))),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------


def main() -> int:
    """Evalúa planos y consistencia geométrica de las nubes antes de registrar."""
    args = build_parser().parse_args()
    root = Path(args.root).expanduser().resolve()
    object_name = args.object.strip().lower()
    session = args.session.strip().upper()

    source_dir = root / "reconstruccion" / session / args.cloud_source
    summary_path = find_summary(
        source_dir,
        ("resumen_06_nubes_puntos.json",),
    )
    source_summary = load_json(summary_path)
    validate_summary_context(source_summary, object_name, session, "Paso 06")

    views = [
        view
        for view in source_summary.get("views", [])
        if str(view.get("quality", "")) in ("accepted", "warning")
    ]
    views.sort(key=lambda item: float(item["angle_deg"]))

    if args.only_angle >= 0:
        views = [
            view
            for view in views
            if np.isclose(float(view["angle_deg"]), float(args.only_angle), atol=0.05)
        ]
    if args.limit > 0:
        views = views[: int(args.limit)]

    diagnostic_run = args.only_angle >= 0 or args.limit > 0
    output_name = args.output_name
    if diagnostic_run and output_name == "07_validacion_geometrica":
        suffix = (
            f"A{args.only_angle:07.3f}".replace(".", "p")
            if args.only_angle >= 0
            else f"limit{args.limit}"
        )
        output_name = f"{output_name}_diagnostico_{suffix}"

    output_dir = root / "reconstruccion" / session / output_name
    prepare_output_directory(
        output_dir,
        clean=bool(args.clean_output and not diagnostic_run),
    )

    geometry_mode = args.geometry_mode
    if geometry_mode == "auto":
        geometry_mode = "cuboid" if object_name == "cubo" else "generic"

    rng = np.random.default_rng(int(args.seed))
    records: List[dict] = []
    previews: List[Path] = []

    print("\n========== PASO 07: VALIDACION GEOMETRICA ==========")
    print(f"Fuente: {source_dir}")
    print(f"Modo geométrico: {geometry_mode}")
    print(
        "Este paso NO registra ni transforma nubes; únicamente mide su " "consistencia geométrica."
    )

    for index, view in enumerate(views, start=1):
        stem = str(view["stem"])
        angle = float(view["angle_deg"])
        npz_path = source_dir / f"{stem}_cloud_final.npz"
        if not npz_path.is_file():
            print(f"[ERROR] Falta {npz_path}")
            continue

        with np.load(str(npz_path)) as data:
            if "points" not in data.files:
                print(f"[ERROR] {npz_path.name} no contiene 'points'.")
                continue
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

        valid = finite_points(points)
        points = points[valid]
        if len(colors) == len(valid):
            colors = colors[valid]
        else:
            colors = np.zeros((len(points), 3), dtype=np.uint8)
        if normals.ndim == 2 and len(normals) == len(valid):
            normals = normals[valid]
        else:
            normals = np.empty((0, 3), dtype=np.float64)

        planes, labels = extract_dominant_planes(points, normals, args, rng)
        relations = analyze_plane_relations(planes, args, geometry_mode)
        source_quality = str(view.get("quality", "accepted"))
        source_reasons = list(view.get("reasons", []))
        quality, reasons, metrics = classify_view(
            len(points),
            planes,
            relations,
            source_quality,
            source_reasons,
            args,
        )

        extent = cloud_extent(points)
        preview_path = output_dir / f"{stem}_geometric_validation_preview.png"
        labeled_ply_path = output_dir / f"{stem}_planes_labeled.ply"
        labels_npz_path = output_dir / f"{stem}_plane_labels.npz"
        stats_path = output_dir / f"{stem}_geometric_validation.json"

        labeled_colors = label_colors(labels)
        save_ply_ascii(
            labeled_ply_path,
            points.astype(np.float32),
            labeled_colors,
            normals.astype(np.float32) if normals.shape == points.shape else None,
        )
        np.savez_compressed(
            labels_npz_path,
            points=points.astype(np.float32),
            plane_labels=labels.astype(np.int32),
            plane_normals=np.asarray(
                [plane["normal_xyz"] for plane in planes], dtype=np.float32
            ).reshape((-1, 3)),
            plane_d_mm=np.asarray([plane["d_mm"] for plane in planes], dtype=np.float32),
            angle_deg=np.asarray([angle], dtype=np.float64),
        )

        record = {
            "stem": stem,
            "angle_deg": angle,
            "quality": quality,
            "reasons": reasons,
            "source_quality": source_quality,
            "source_reasons": source_reasons,
            "point_count": int(len(points)),
            "geometry_mode": geometry_mode,
            "cloud_geometry": extent,
            "metrics": metrics,
            "planes": planes,
            "plane_relations": relations,
            "outputs": {
                "preview": str(preview_path),
                "labeled_ply": str(labeled_ply_path),
                "labels_npz": str(labels_npz_path),
                "stats": str(stats_path),
            },
        }
        save_json(stats_path, record)
        make_validation_preview(
            points,
            labels,
            record,
            preview_path,
            width=int(args.preview_width),
            height=int(args.preview_height),
        )

        records.append(record)
        previews.append(preview_path)

        print(
            f"[{index:02d}/{len(views):02d}] {angle:8.3f}° | "
            f"puntos={len(points):6d} | "
            f"planos={metrics['plane_count']} | "
            f"explicado={metrics['explained_ratio']:.1%} | "
            f"mejor={metrics['best_plane_ratio']:.1%} | "
            f"{quality}"
        )
        if reasons:
            for reason in reasons[:3]:
                print(f"           - {reason}")
            if len(reasons) > 3:
                print(f"           - ... {len(reasons) - 3} observaciones adicionales")

    if previews:
        build_contact_sheet(
            previews,
            output_dir / "contact_sheet_validacion_geometrica.png",
            columns=4,
            width=480,
        )

    explained_values = [r["metrics"]["explained_ratio"] for r in records]
    best_plane_values = [r["metrics"]["best_plane_ratio"] for r in records]
    rmse_values = [plane["rmse_mm"] for record in records for plane in record["planes"]]
    thickness_values = [plane["thickness95_mm"] for record in records for plane in record["planes"]]

    registration_eligible_angles = [
        r["angle_deg"] for r in records if r["quality"] in ("accepted", "warning")
    ]
    rejected_angles = [r["angle_deg"] for r in records if r["quality"] == "rejected"]

    summary = {
        "schema_version": 1,
        "method": "metric_multiplane_ransac_svd_pre_registration_validation",
        "object": object_name,
        "session": session,
        "source_dir": str(source_dir),
        "source_summary": str(summary_path),
        "output_dir": str(output_dir),
        "geometry_mode": geometry_mode,
        "parameters": vars(args),
        "views_processed": len(records),
        "views_accepted": sum(r["quality"] == "accepted" for r in records),
        "views_warning": sum(r["quality"] == "warning" for r in records),
        "views_rejected": sum(r["quality"] == "rejected" for r in records),
        "registration_eligible_angles": registration_eligible_angles,
        "rejected_angles": rejected_angles,
        "session_statistics": {
            "explained_ratio": robust_session_stats(explained_values),
            "best_plane_ratio": robust_session_stats(best_plane_values),
            "plane_rmse_mm": robust_session_stats(rmse_values),
            "plane_thickness95_mm": robust_session_stats(thickness_values),
        },
        "views": records,
        "important_note": (
            "El paso 07 no realiza registro. Una vista con huecos puede ser "
            "válida si los puntos presentes son geométricamente coherentes. "
            "La clasificación se usa para decidir qué fragmentos pueden entrar "
            "a el paso 08."
        ),
    }
    save_json(
        output_dir / "resumen_07_validacion_geometrica.json",
        summary,
    )

    csv_path = output_dir / "calidad_geometrica_nubes.csv"
    fieldnames = (
        "angle_deg",
        "stem",
        "quality",
        "source_quality",
        "point_count",
        "plane_count",
        "explained_ratio",
        "best_plane_ratio",
        "best_plane_rmse_mm",
        "parallel_separated_pair_count",
        "cuboid_angle_warning_count",
        "reasons",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            metrics = record["metrics"]
            writer.writerow(
                {
                    "angle_deg": record["angle_deg"],
                    "stem": record["stem"],
                    "quality": record["quality"],
                    "source_quality": record["source_quality"],
                    "point_count": record["point_count"],
                    "plane_count": metrics["plane_count"],
                    "explained_ratio": metrics["explained_ratio"],
                    "best_plane_ratio": metrics["best_plane_ratio"],
                    "best_plane_rmse_mm": metrics["best_plane_rmse_mm"],
                    "parallel_separated_pair_count": metrics["parallel_separated_pair_count"],
                    "cuboid_angle_warning_count": metrics["cuboid_angle_warning_count"],
                    "reasons": " | ".join(record["reasons"]),
                }
            )

    print("\n========== PASO 07 COMPLETADO ==========")
    print(f"Salida: {output_dir}")
    print(
        f"Accepted={summary['views_accepted']} | "
        f"Warning={summary['views_warning']} | "
        f"Rejected={summary['views_rejected']}"
    )
    print(f"Ángulos elegibles para 08: {registration_eligible_angles}")
    if rejected_angles:
        print(f"Ángulos rechazados por 07: {rejected_angles}")
    print("===========================================")
    return 0


def _entrada_con_diagnostico():
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    from utilidades_progreso import informar_inicio

    informar_inicio(__file__, "07", "Validar geometría de las nubes")
    sys.exit(main())


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
