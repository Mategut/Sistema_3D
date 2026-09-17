#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PASO 12 V4.6 — LIMPIEZA 3D MULTIVISTA Y REGULARIZACIÓN SUPERFICIAL GENERAL

Objetivo
--------
Reducir rugosidad geométrica de alta frecuencia SIN asumir:
- cubo;
- caras planas;
- ángulos rectos;
- convexidad;
- dimensiones conocidas;
- una clase geométrica concreta.

El paso aplica primero una limpieza conservadora inicial:
1. Statistical Outlier Removal (SOR).
2. Radius Outlier Removal (ROR).
3. Preselección de puntos con soporte/confianza alto.

Después realiza una limpieza tridimensional independiente de la figura:
4. Mide respaldo espacial mediante número de vecinos locales.
5. Agrupa puntos por adyacencia geométrica con DBSCAN.
6. Evalúa cada componente combinando:
   - número y proporción de puntos;
   - densidad local;
   - soporte de vistas;
   - confianza;
   - distancia respecto a los componentes principales.
7. Conserva componentes separados cuando son grandes o tienen evidencia
   multivista suficiente. Nunca se limita a conservar solamente el mayor.
8. Elimina ruido disperso y componentes pequeños, alejados y con evidencia
   espacial/multivista insuficiente.

Después añade una regularización superficial general:
9. Recalcula desde cero las normales sobre la geometría ya limpia.
10. Suavizado bilateral punto-superficie:
   - el punto se mueve PRINCIPALMENTE a lo largo de su normal;
   - vecinos lejanos pesan menos;
   - vecinos con normal distinta pesan mucho menos;
   - soporte multivista y confianza ponderan la estimación;
   - discontinuidades geométricas reducen automáticamente el suavizado;
   - bordes abiertos se detectan por asimetría del vecindario y se protegen.
11. El desplazamiento por iteración y el desplazamiento total están acotados
   por el espaciamiento real de la nube.
12. Guarda de escala de muestreo:
   - compara el spacing antes y después del suavizado;
   - si el suavizado colapsa capas cercanas y reduce demasiado el spacing,
     mezcla adaptativamente la solución regularizada con la nube filtrada
     original hasta recuperar una escala de muestreo segura;
   - esto evita que 13 reduzca en exceso los radios de Ball Pivoting.
13. Reestimación final y orientación consistente de normales.

La regularización NO:
- agrega puntos;
- rellena huecos;
- cierra superficies;
- proyecta a planos globales;
- ajusta primitivas;
- modifica los registros ni la fusión de los pasos anteriores;
- fuerza watertight.

Compatibilidad
--------------
Conserva exactamente las salidas consumidas por 13:

    nube_regularizada_general.npz
    nube_regularizada_general.ply

y añade trazabilidad:

    nube_filtrada_sin_suavizado.npz
    nube_filtrada_sin_suavizado.ply
    desplazamiento_regularizacion_mm.npy
    diagnostico_regularizacion_superficial.npz
    preview_nube_regularizada_general.png
    preview_comparacion_regularizacion.png
    resumen_12_regularizacion_nube.json
"""

from __future__ import annotations
from utilidades_progreso import operacion
from utilidades_rendimiento import query_threads
from utilidades_rendimiento import configurar as _configurar_recursos

# Los runtimes numéricos leen estos límites al importarse.
_configurar_recursos(__file__)


import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception as exc:
    raise SystemExit(f"Paso 12 requiere SciPy: {exc}")

# Open3D se necesita para el pipeline real, pero se difiere el error hasta main
# para permitir pruebas unitarias del núcleo matemático.
try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parser():
    """Construye las opciones de línea de comandos de este paso."""
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--object", required=True)
    p.add_argument(
        "--source",
        default="11_fusion_multivista",
    )
    p.add_argument(
        "--output-name",
        default="12_regularizacion_nube",
    )

    # Limpieza histórica.
    p.add_argument("--sor-neighbors", type=int, default=24)
    p.add_argument("--sor-std-ratio", type=float, default=2.5)
    p.add_argument("--radius-neighbors", type=int, default=5)
    p.add_argument(
        "--radius-mm",
        type=float,
        default=0.0,
        help="0=auto: max(3.5, 2.5*voxel)",
    )
    p.add_argument("--always-preserve-support", type=int, default=3)
    p.add_argument("--always-preserve-confidence", type=float, default=0.72)

    # V4.0 — contrato de evidencia del paso 11. Soporte/confianza altos ya no
    # pueden rescatar automáticamente una capa local marcada como ambigua.
    p.add_argument("--minimum-input-evidence-class", type=int, default=2)
    p.add_argument("--minimum-input-independent-support", type=int, default=2)
    p.add_argument("--minimum-input-angular-span-poses", type=int, default=2)
    p.add_argument("--maximum-input-conflict-pose-ratio", type=float, default=0.35)
    p.add_argument(
        "--minimum-input-heldout-pass-ratio",
        type=float,
        default=(2.0 / 3.0),
        help=(
            "Compatibilidad/diagnóstico. La decisión de producción usa "
            "conteos enteros para evitar que 2/3 falle frente a 0.67."
        ),
    )
    p.add_argument("--minimum-input-heldout-pass-numerator", type=int, default=2)
    p.add_argument("--minimum-input-heldout-pass-denominator", type=int, default=3)
    p.add_argument(
        "--heldout-rescue-min-independent-support",
        type=int,
        default=3,
        help="Respaldo independiente mínimo para rescatar un voto held-out insuficiente.",
    )
    p.add_argument(
        "--heldout-rescue-min-angular-span-poses",
        type=int,
        default=3,
    )
    p.add_argument(
        "--heldout-rescue-max-conflict-pose-ratio",
        type=float,
        default=0.20,
    )
    p.add_argument(
        "--heldout-rescue-error-uncertainty-factor",
        type=float,
        default=1.0,
        help="Error mediano held-out máximo relativo a la incertidumbre local.",
    )
    p.add_argument(
        "--heldout-rescue-p90-uncertainty-factor",
        type=float,
        default=1.5,
        help="P90 held-out máximo relativo a la incertidumbre local.",
    )
    p.add_argument(
        "--heldout-uncertainty-floor-voxel-factor",
        type=float,
        default=0.12,
        help="Piso de incertidumbre para normalizar los errores held-out.",
    )
    p.add_argument("--pose-shared-support-strong", type=int, default=2)
    p.add_argument("--pose-shared-support-weak-weight", type=float, default=0.30)
    p.add_argument("--pose-no-shared-support-weight", type=float, default=0.05)

    # Limpieza tridimensional general previa al suavizado.
    p.add_argument(
        "--geometric-cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--adjacency-radius-mm",
        type=float,
        default=0.0,
        help="0=auto según voxel y spacing de la nube preseleccionada.",
    )
    p.add_argument("--adjacency-voxel-factor", type=float, default=2.25)
    p.add_argument("--adjacency-spacing-factor", type=float, default=5.50)
    p.add_argument("--component-dbscan-min-points", type=int, default=4)
    p.add_argument(
        "--density-radius-mm",
        type=float,
        default=0.0,
        help="0=auto; radio para medir respaldo espacial local.",
    )
    p.add_argument("--density-radius-voxel-factor", type=float, default=2.80)
    p.add_argument("--density-radius-spacing-factor", type=float, default=7.00)
    p.add_argument("--density-min-neighbors", type=int, default=6)
    p.add_argument("--density-high-quality-min-neighbors", type=int, default=3)
    p.add_argument(
        "--component-large-fraction",
        type=float,
        default=0.010,
        help="Fracción que hace significativo un componente por tamaño.",
    )
    p.add_argument("--component-min-points", type=int, default=48)
    p.add_argument("--component-score-threshold", type=float, default=0.62)
    p.add_argument("--component-far-score-penalty", type=float, default=0.14)
    p.add_argument("--component-high-evidence-score", type=float, default=0.82)
    p.add_argument(
        "--component-near-distance-mm",
        type=float,
        default=0.0,
        help="0=auto según radio de adyacencia y escala robusta.",
    )
    p.add_argument("--component-near-adjacency-factor", type=float, default=6.0)
    p.add_argument("--component-near-diagonal-ratio", type=float, default=0.06)
    p.add_argument("--maximum-component-diagnostics", type=int, default=250)

    # Regularización superficial.
    p.add_argument(
        "--surface-regularization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--smooth-iterations", type=int, default=2)
    p.add_argument("--smooth-knn", type=int, default=36)
    p.add_argument(
        "--smooth-radius-mm",
        type=float,
        default=0.0,
        help="0=auto según spacing y voxel.",
    )
    p.add_argument(
        "--smooth-radius-spacing-factor",
        type=float,
        default=3.20,
    )
    p.add_argument(
        "--smooth-spatial-sigma-factor",
        type=float,
        default=0.52,
    )
    p.add_argument(
        "--smooth-normal-sigma-deg",
        type=float,
        default=16.0,
    )
    p.add_argument(
        "--feature-protect-angle-deg",
        type=float,
        default=21.0,
        help=(
            "Dispersión local de normales a partir de la cual se reduce "
            "progresivamente el suavizado."
        ),
    )
    p.add_argument(
        "--feature-min-strength",
        type=float,
        default=0.10,
    )
    p.add_argument(
        "--boundary-asymmetry-low",
        type=float,
        default=0.12,
    )
    p.add_argument(
        "--boundary-asymmetry-high",
        type=float,
        default=0.34,
    )
    p.add_argument(
        "--boundary-min-strength",
        type=float,
        default=0.12,
    )
    p.add_argument(
        "--smooth-strength",
        type=float,
        default=0.64,
    )
    p.add_argument(
        "--max-shift-per-iteration-spacing",
        type=float,
        default=0.42,
    )
    p.add_argument(
        "--max-total-shift-spacing",
        type=float,
        default=1.05,
    )
    p.add_argument(
        "--minimum-smoothing-neighbors",
        type=int,
        default=10,
    )
    p.add_argument(
        "--height-robust-scale-spacing-floor",
        type=float,
        default=0.10,
    )
    p.add_argument(
        "--height-robust-sigma-factor",
        type=float,
        default=2.60,
    )

    # Guarda de escala de muestreo.
    p.add_argument(
        "--minimum-spacing-ratio-after-smoothing",
        type=float,
        default=0.82,
        help=(
            "El spacing NN mediano tras suavizado no puede caer por debajo "
            "de esta fracción del spacing filtrado previo. Si cae, se reduce "
            "adaptativamente la intensidad efectiva de regularización."
        ),
    )
    p.add_argument(
        "--spacing-guard-bisection-steps",
        type=int,
        default=10,
    )
    p.add_argument(
        "--spacing-guard-sample-points",
        type=int,
        default=30000,
    )

    # Calidad: soporte y confianza.
    p.add_argument(
        "--support-neighbor-weight-exponent",
        type=float,
        default=0.35,
    )
    p.add_argument(
        "--confidence-neighbor-weight-exponent",
        type=float,
        default=0.65,
    )
    p.add_argument(
        "--quality-anchor-strength",
        type=float,
        default=0.28,
        help=(
            "Puntos de soporte/confianza alta se mueven ligeramente menos; la procedencia por pose evita mezclar capas incompatibles; "
            "no quedan congelados."
        ),
    )

    # Normales.
    p.add_argument(
        "--normal-pca-knn",
        type=int,
        default=30,
    )
    p.add_argument(
        "--normal-pca-radius-factor",
        type=float,
        default=3.4,
    )
    p.add_argument(
        "--normal-radius-mm",
        type=float,
        default=0.0,
        help="0=auto para orientación final Open3D.",
    )
    p.add_argument("--normal-max-nn", type=int, default=50)
    p.add_argument("--normal-orientation-k", type=int, default=30)

    # Diagnóstico.
    p.add_argument(
        "--roughness-evaluation-knn",
        type=int,
        default=28,
    )
    p.add_argument(
        "--maximum-diagnostic-points",
        type=int,
        default=60000,
    )
    p.add_argument(
        "--warning-extent-change-ratio",
        type=float,
        default=0.025,
        help="Warning si un extent robusto cambia >2.5%%.",
    )
    return p


# ---------------------------------------------------------------------------
# Estadística y escala
# ---------------------------------------------------------------------------


def stats(values):
    a = np.asarray(values, dtype=float)
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


def nearest_spacing(points):
    if len(points) < 2:
        return None

    if len(points) > 60000:
        idx = np.linspace(
            0,
            len(points) - 1,
            60000,
        ).astype(int)
        q = points[idx]
    else:
        q = points

    tree = cKDTree(points)
    d, _ = tree.query(
        q,
        k=2,
        workers=query_threads(),
    )
    return stats(d[:, 1])


def robust_extent(points):
    p = np.asarray(points, dtype=np.float64)
    if len(p) == 0:
        return None

    lo = np.percentile(p, 1.0, axis=0)
    hi = np.percentile(p, 99.0, axis=0)
    extent = hi - lo
    return {
        "p01": lo.astype(float).tolist(),
        "p99": hi.astype(float).tolist(),
        "extent": extent.astype(float).tolist(),
    }


def safe_unit_rows(vectors):
    """Normaliza vectores por fila y usa el eje Z para las filas inválidas."""
    v = np.asarray(vectors, dtype=np.float64).copy()
    n = np.linalg.norm(v, axis=1)
    good = np.isfinite(n) & (n > 1e-12)
    v[good] /= n[good, None]
    v[~good] = np.asarray([0.0, 0.0, 1.0])
    return v


_POPCOUNT_BYTES = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def bit_count_u64(values):
    """Cuenta bits activos de cada entero uint64 conservando la forma del array."""
    values = np.ascontiguousarray(values, dtype=np.uint64)
    shape = values.shape
    flat = values.reshape(-1)
    if len(flat) == 0:
        return np.zeros(shape, dtype=np.int16)
    raw = flat.view(np.uint8).reshape(-1, 8)
    return _POPCOUNT_BYTES[raw].sum(axis=1).astype(np.int16).reshape(shape)


# ---------------------------------------------------------------------------
# Limpieza 3D por respaldo espacial, componentes y evidencia multivista
# ---------------------------------------------------------------------------


def radius_neighbor_counts(points: np.ndarray, radius: float) -> np.ndarray:
    """Cuenta vecinos (sin incluir el propio punto) usando todos los núcleos.

    ``return_length`` evita construir listas de vecinos y mantiene acotado el
    consumo de memoria. El fallback conserva compatibilidad con SciPy antiguo.
    """
    p = np.asarray(points, dtype=np.float64)
    if len(p) == 0:
        return np.zeros(0, dtype=np.int32)

    tree = cKDTree(p)
    try:
        count = tree.query_ball_point(
            p,
            r=float(radius),
            return_length=True,
            workers=query_threads(),
        )
    except TypeError:
        try:
            neighborhoods = tree.query_ball_point(
                p,
                r=float(radius),
                workers=query_threads(),
            )
        except TypeError:
            neighborhoods = tree.query_ball_point(
                p,
                r=float(radius),
            )
        count = np.fromiter(
            (len(v) for v in neighborhoods),
            dtype=np.int32,
            count=len(p),
        )

    return np.maximum(
        np.asarray(count, dtype=np.int32) - 1,
        0,
    )


def _sample_indices(indices: np.ndarray, maximum: int = 1024) -> np.ndarray:
    """Reduce determinísticamente una selección de índices al máximo indicado."""
    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) <= int(maximum):
        return idx
    positions = np.linspace(
        0,
        len(idx) - 1,
        int(maximum),
    ).astype(np.int64)
    return idx[positions]


@operacion("Evaluar componentes geométricas")
def clean_geometric_components(
    points: np.ndarray,
    colors: np.ndarray,
    support: np.ndarray,
    confidence: np.ndarray,
    selection_class: np.ndarray,
    *,
    voxel: float,
    spacing: float,
    args,
):
    """Limpia una nube sin imponer una forma ni conservar solo el mayor.

    La decisión se toma en dos escalas:

    1. Respaldo local: elimina puntos espacialmente aislados, con una excepción
       limitada para muestras de alta calidad que todavía tengan vecinos.
    2. Componente: DBSCAN agrupa por adyacencia y cada grupo se evalúa por
       tamaño, densidad, soporte, confianza y distancia a los componentes
       principales. Un componente separado puede sobrevivir por tamaño o por
       evidencia multivista, por lo que se admiten objetos de varias partes.
    """
    p = np.asarray(points, dtype=np.float64)
    c = np.asarray(colors, dtype=np.uint8)
    s = np.asarray(support)
    q = np.asarray(confidence, dtype=np.float64)
    cls = np.asarray(selection_class)

    n_input = len(p)
    if n_input == 0:
        raise RuntimeError("La limpieza 3D recibió una nube vacía.")

    if not bool(args.geometric_cleanup):
        mask = np.ones(n_input, dtype=bool)
        return {
            "points": p,
            "colors": c,
            "support": s,
            "confidence": q,
            "selection_class": cls,
            "keep_mask": mask,
            "local_neighbor_count": np.zeros(n_input, dtype=np.int32),
            "component_label": np.zeros(n_input, dtype=np.int32),
            "report": {
                "enabled": False,
                "input_points": int(n_input),
                "final_points": int(n_input),
                "removed_points": 0,
            },
        }

    adjacency_radius = (
        float(args.adjacency_radius_mm)
        if float(args.adjacency_radius_mm) > 0
        else max(
            float(args.adjacency_voxel_factor) * float(voxel),
            float(args.adjacency_spacing_factor) * float(spacing),
        )
    )
    density_radius = (
        float(args.density_radius_mm)
        if float(args.density_radius_mm) > 0
        else max(
            float(args.density_radius_voxel_factor) * float(voxel),
            float(args.density_radius_spacing_factor) * float(spacing),
            1.15 * adjacency_radius,
        )
    )

    print("[12/1] Midiendo respaldo espacial local " f"(radio={density_radius:.3f} mm)...")
    neighbor_count = radius_neighbor_counts(
        p,
        density_radius,
    )

    min_neighbors = max(1, int(args.density_min_neighbors))
    high_quality_min_neighbors = int(
        np.clip(
            int(args.density_high_quality_min_neighbors),
            1,
            min_neighbors,
        )
    )
    high_quality = (s >= int(args.always_preserve_support)) | (
        q >= float(args.always_preserve_confidence)
    )

    spatially_backed = (neighbor_count >= min_neighbors) | (
        high_quality & (neighbor_count >= high_quality_min_neighbors)
    )

    backed_indices = np.flatnonzero(spatially_backed)
    if len(backed_indices) < 500:
        raise RuntimeError(
            "La limpieza por respaldo espacial dejó menos de 500 puntos. "
            "Revise el spacing o reduzca --density-min-neighbors."
        )

    backed_points = p[backed_indices]
    backed_colors = c[backed_indices]
    backed_support = s[backed_indices]
    backed_confidence = q[backed_indices]
    backed_class = cls[backed_indices]
    backed_neighbors = neighbor_count[backed_indices]

    print("[12/2] Agrupando por adyacencia geométrica " f"(radio={adjacency_radius:.3f} mm)...")
    cluster_cloud = make_pcd(
        backed_points,
        backed_colors,
    )
    labels = np.asarray(
        cluster_cloud.cluster_dbscan(
            eps=float(adjacency_radius),
            min_points=max(2, int(args.component_dbscan_min_points)),
            print_progress=False,
        ),
        dtype=np.int32,
    )

    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        # No borra toda la nube si los parámetros automáticos resultan
        # demasiado estrictos; deja constancia explícita en el resumen.
        final_mask = spatially_backed.copy()
        return {
            "points": p[final_mask],
            "colors": c[final_mask],
            "support": s[final_mask],
            "confidence": q[final_mask],
            "selection_class": cls[final_mask],
            "keep_mask": final_mask,
            "local_neighbor_count": neighbor_count[final_mask],
            "component_label": np.full(
                np.count_nonzero(final_mask),
                -1,
                dtype=np.int32,
            ),
            "report": {
                "enabled": True,
                "fallback": "no_dbscan_components",
                "input_points": int(n_input),
                "after_spatial_backing": int(len(backed_points)),
                "final_points": int(np.count_nonzero(final_mask)),
                "removed_points": int(n_input - np.count_nonzero(final_mask)),
                "adjacency_radius_mm": float(adjacency_radius),
                "density_radius_mm": float(density_radius),
            },
        }

    component_count = int(np.max(valid_labels)) + 1
    component_sizes = np.bincount(
        valid_labels,
        minlength=component_count,
    ).astype(np.int64)
    largest_label = int(np.argmax(component_sizes))

    large_fraction = float(
        np.clip(
            args.component_large_fraction,
            1e-5,
            0.50,
        )
    )
    primary_size = max(
        int(math.ceil(large_fraction * len(backed_points))),
        max(2, int(args.component_min_points)),
    )
    primary_labels = set(np.flatnonzero(component_sizes >= primary_size).astype(int).tolist())
    primary_labels.add(largest_label)

    primary_mask = np.isin(
        labels,
        np.asarray(sorted(primary_labels), dtype=np.int32),
    )
    primary_points = backed_points[primary_mask]
    primary_tree = cKDTree(primary_points)

    primary_extent = robust_extent(primary_points)
    primary_diagonal = float(np.linalg.norm(np.asarray(primary_extent["extent"], dtype=np.float64)))
    near_distance = (
        float(args.component_near_distance_mm)
        if float(args.component_near_distance_mm) > 0
        else max(
            float(args.component_near_adjacency_factor) * adjacency_radius,
            float(args.component_near_diagonal_ratio) * primary_diagonal,
        )
    )

    base_threshold = float(
        np.clip(
            args.component_score_threshold,
            0.0,
            1.0,
        )
    )
    far_penalty = max(0.0, float(args.component_far_score_penalty))
    high_evidence_threshold = float(
        np.clip(
            args.component_high_evidence_score,
            0.0,
            1.0,
        )
    )
    min_component_points = max(2, int(args.component_min_points))
    strong_support_threshold = max(
        3,
        int(args.always_preserve_support),
    )

    retained_labels = set()
    component_records = []

    print(f"[12/3] Evaluando {component_count} componentes con evidencia multivista...")
    for label in range(component_count):
        idx = np.flatnonzero(labels == label)
        if len(idx) == 0:
            continue

        count = int(len(idx))
        fraction = float(count / len(backed_points))
        density_median = float(np.median(backed_neighbors[idx]))
        density_p10 = float(np.percentile(backed_neighbors[idx], 10))
        support_median = float(np.median(backed_support[idx]))
        support_mean = float(np.mean(backed_support[idx]))
        strong_fraction = float(np.mean(backed_support[idx] >= strong_support_threshold))
        confidence_median = float(np.median(backed_confidence[idx]))
        confidence_mean = float(np.mean(backed_confidence[idx]))

        if label in primary_labels:
            gap_p10 = 0.0
        else:
            sample_idx = _sample_indices(idx, maximum=1024)
            distances, _ = primary_tree.query(
                backed_points[sample_idx],
                k=1,
                workers=query_threads(),
            )
            finite_distance = np.asarray(distances, dtype=np.float64)
            finite_distance = finite_distance[np.isfinite(finite_distance)]
            gap_p10 = (
                float(np.percentile(finite_distance, 10)) if len(finite_distance) else float("inf")
            )

        size_score = float(
            np.clip(
                fraction / large_fraction,
                0.0,
                1.0,
            )
        )
        density_target = max(12.0, 2.0 * min_neighbors)
        density_score = float(
            np.clip(
                (density_median - high_quality_min_neighbors)
                / max(
                    density_target - high_quality_min_neighbors,
                    1.0,
                ),
                0.0,
                1.0,
            )
        )
        support_score = float(
            np.clip(
                (support_median - 2.0) / max(strong_support_threshold - 2.0, 1.0),
                0.0,
                1.0,
            )
        )
        confidence_score = float(
            np.clip(
                (confidence_median - 0.55) / 0.35,
                0.0,
                1.0,
            )
        )

        evidence_score = float(
            0.15 * size_score
            + 0.15 * density_score
            + 0.25 * support_score
            + 0.20 * strong_fraction
            + 0.25 * confidence_score
        )

        far_level = float(
            np.clip(
                (gap_p10 - near_distance) / max(near_distance, 1e-9),
                0.0,
                1.0,
            )
        )
        required_score = float(
            np.clip(
                base_threshold + far_penalty * far_level,
                0.0,
                0.95,
            )
        )

        is_primary = label in primary_labels
        is_large = fraction >= large_fraction and density_median >= high_quality_min_neighbors
        well_supported = (
            count >= min_component_points
            and density_median >= high_quality_min_neighbors
            and support_median >= strong_support_threshold
            and strong_fraction >= 0.60
            and confidence_median >= 0.64
        )
        credible = (
            count >= min_component_points
            and density_median >= high_quality_min_neighbors + 1
            and evidence_score >= required_score
        )
        near_credible = (
            gap_p10 <= near_distance
            and count >= min_component_points
            and density_median >= high_quality_min_neighbors + 1
            and evidence_score >= max(0.40, base_threshold - 0.14)
        )
        high_evidence = (
            count >= min_component_points
            and density_median >= high_quality_min_neighbors
            and evidence_score >= high_evidence_threshold
        )

        keep_component = bool(
            is_primary or is_large or well_supported or credible or near_credible or high_evidence
        )
        if keep_component:
            retained_labels.add(label)

        reasons = []
        if is_primary:
            reasons.append("primary")
        if is_large:
            reasons.append("large")
        if well_supported:
            reasons.append("multiview_supported")
        if credible:
            reasons.append("combined_evidence")
        if near_credible:
            reasons.append("near_supported")
        if high_evidence:
            reasons.append("high_evidence")
        if not reasons:
            reasons.append("insufficient_combined_evidence")

        component_records.append(
            {
                "label": int(label),
                "kept": bool(keep_component),
                "reasons": reasons,
                "points": count,
                "fraction": fraction,
                "density_neighbors_median": density_median,
                "density_neighbors_p10": density_p10,
                "support_median": support_median,
                "support_mean": support_mean,
                "strong_support_fraction": strong_fraction,
                "confidence_median": confidence_median,
                "confidence_mean": confidence_mean,
                "gap_to_primary_p10_mm": gap_p10,
                "far_level": far_level,
                "evidence_score": evidence_score,
                "required_score": required_score,
            }
        )

    keep_backed = np.isin(
        labels,
        np.asarray(sorted(retained_labels), dtype=np.int32),
    )
    final_mask = np.zeros(n_input, dtype=bool)
    final_mask[backed_indices[keep_backed]] = True

    final_count = int(np.count_nonzero(final_mask))
    if final_count < 500:
        raise RuntimeError(
            "La limpieza de componentes dejó menos de 500 puntos. "
            "Reduzca --component-score-threshold o revise la escala de la nube."
        )

    max_diagnostics = max(1, int(args.maximum_component_diagnostics))
    records_sorted = sorted(
        component_records,
        key=lambda item: item["points"],
        reverse=True,
    )
    records_saved = records_sorted[:max_diagnostics]

    report = {
        "enabled": True,
        "shape_specific_assumptions": False,
        "input_points": int(n_input),
        "after_spatial_backing": int(len(backed_points)),
        "spatially_unbacked_removed": int(n_input - len(backed_points)),
        "dbscan_noise_points": int(np.count_nonzero(labels < 0)),
        "components_total": int(component_count),
        "components_retained": int(len(retained_labels)),
        "components_rejected": int(component_count - len(retained_labels)),
        "final_points": final_count,
        "removed_points": int(n_input - final_count),
        "adjacency_radius_mm": float(adjacency_radius),
        "density_radius_mm": float(density_radius),
        "density_neighbor_count": stats(neighbor_count),
        "primary_component_labels": sorted(int(v) for v in primary_labels),
        "retained_component_labels": sorted(int(v) for v in retained_labels),
        "primary_robust_diagonal_mm": float(primary_diagonal),
        "near_distance_mm": float(near_distance),
        "component_records_saved": int(len(records_saved)),
        "component_records_omitted": int(max(0, len(records_sorted) - len(records_saved))),
        "components": records_saved,
        "decision_note": (
            "Los componentes no se filtran por una forma esperada ni se "
            "conserva únicamente el mayor. Un componente separado sobrevive "
            "si su tamaño, densidad o evidencia multivista lo justifican."
        ),
    }

    kept_backed_indices = backed_indices[keep_backed]
    return {
        "points": p[final_mask],
        "colors": c[final_mask],
        "support": s[final_mask],
        "confidence": q[final_mask],
        "selection_class": cls[final_mask],
        "keep_mask": final_mask,
        "local_neighbor_count": neighbor_count[final_mask],
        "component_label": labels[keep_backed],
        "kept_source_indices": kept_backed_indices,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Normales PCA vectorizadas
# ---------------------------------------------------------------------------


def estimate_pca_normals(
    points: np.ndarray,
    *,
    k: int,
    radius: float,
    reference_normals: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estima normales locales con PCA ponderado.

    Devuelve:
        normals
        curvature = lambda_min / sum(lambda)
        neighbor_counts
    """
    p = np.asarray(points, dtype=np.float64)
    n_points = len(p)

    if n_points < 4:
        normals = np.tile(
            np.asarray([[0.0, 0.0, 1.0]]),
            (n_points, 1),
        )
        return (
            normals,
            np.zeros(n_points, dtype=float),
            np.zeros(n_points, dtype=np.int32),
        )

    kk = int(np.clip(k, 4, n_points))
    tree = cKDTree(p)
    distances, indices = tree.query(
        p,
        k=kk,
        workers=query_threads(),
    )

    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    valid = (
        np.isfinite(distances)
        & (indices >= 0)
        & (indices < n_points)
        & (distances <= float(radius))
        & (distances > 1e-10)
    )

    safe_indices = np.clip(
        indices,
        0,
        n_points - 1,
    )
    neigh = p[safe_indices]

    sigma = max(
        0.45 * float(radius),
        1e-9,
    )
    weights = np.exp(-0.5 * (distances / sigma) ** 2)
    weights *= valid

    wsum = np.sum(weights, axis=1)
    good = wsum > 1e-10

    centroid = p.copy()
    centroid[good] = (
        np.einsum(
            "nk,nkj->nj",
            weights[good],
            neigh[good],
        )
        / wsum[good, None]
    )

    diff = neigh - centroid[:, None, :]
    covariance = np.einsum(
        "nk,nki,nkj->nij",
        weights,
        diff,
        diff,
    )
    covariance[good] /= wsum[good, None, None]

    # Evita matrices degeneradas.
    covariance[~good] = np.eye(3)[None, :, :] * 1e-9

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    normals = safe_unit_rows(normals)

    counts = np.sum(valid, axis=1).astype(np.int32)
    bad_count = counts < 3

    if reference_normals is not None:
        ref = safe_unit_rows(reference_normals)
        sign = np.sum(normals * ref, axis=1)
        normals[sign < 0] *= -1.0
        normals[bad_count] = ref[bad_count]

    trace = np.sum(
        np.maximum(eigenvalues, 0.0),
        axis=1,
    )
    curvature = np.divide(
        np.maximum(eigenvalues[:, 0], 0.0),
        trace,
        out=np.zeros(n_points, dtype=float),
        where=trace > 1e-15,
    )

    return normals, curvature, counts


# ---------------------------------------------------------------------------
# Regularización bilateral punto-superficie
# ---------------------------------------------------------------------------


def _neighbor_quality_weights(
    support_neighbors,
    confidence_neighbors,
    *,
    support_median,
    support_exponent,
    confidence_exponent,
):
    s = np.asarray(
        support_neighbors,
        dtype=np.float64,
    )
    c = np.asarray(
        confidence_neighbors,
        dtype=np.float64,
    )

    s_scale = max(float(support_median), 1.0)
    s_norm = np.clip(
        s / s_scale,
        0.35,
        2.50,
    )

    c_norm = np.clip(c, 0.0, 1.0)
    c_norm = 0.45 + 1.10 * c_norm

    return np.power(
        s_norm,
        float(support_exponent),
    ) * np.power(
        c_norm,
        float(confidence_exponent),
    )


def surface_bilateral_iteration(
    points: np.ndarray,
    normals: np.ndarray,
    support: np.ndarray,
    confidence: np.ndarray,
    *,
    spacing: float,
    radius: float,
    k: int,
    original_points: np.ndarray,
    support_pose_mask: Optional[np.ndarray] = None,
    disagreement_pose_mask: Optional[np.ndarray] = None,
    evidence_strength: Optional[np.ndarray] = None,
    args,
):
    p = np.asarray(points, dtype=np.float64)
    n = safe_unit_rows(normals)
    n_points = len(p)

    kk = int(
        np.clip(
            k,
            4,
            n_points,
        )
    )

    tree = cKDTree(p)
    distances, indices = tree.query(
        p,
        k=kk,
        workers=query_threads(),
    )
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    valid = (
        np.isfinite(distances)
        & (indices >= 0)
        & (indices < n_points)
        & (distances <= float(radius))
        & (distances > 1e-10)
    )

    safe_indices = np.clip(
        indices,
        0,
        n_points - 1,
    )

    q = p[safe_indices]
    qn = n[safe_indices]

    vector = q - p[:, None, :]
    signed_height = np.einsum(
        "nkj,nj->nk",
        vector,
        n,
    )

    # Distancia tangencial local.
    tangent = vector - signed_height[:, :, None] * n[:, None, :]
    tangent_distance = np.linalg.norm(
        tangent,
        axis=2,
    )

    # Peso espacial.
    spatial_sigma = max(
        float(args.smooth_spatial_sigma_factor) * float(radius),
        1e-8,
    )
    w_spatial = np.exp(-0.5 * (distances / spatial_sigma) ** 2)

    # Peso por coherencia de normal: preserva discontinuidades/aristas.
    dot_normals = np.abs(
        np.einsum(
            "nkj,nj->nk",
            qn,
            n,
        )
    )
    dot_normals = np.clip(
        dot_normals,
        0.0,
        1.0,
    )
    angle_deg = np.degrees(np.arccos(dot_normals))

    normal_sigma = max(
        float(args.smooth_normal_sigma_deg),
        1e-3,
    )
    w_normal = np.exp(-0.5 * (angle_deg / normal_sigma) ** 2)

    # Peso de calidad multivista.
    support_neighbors = support[safe_indices]
    confidence_neighbors = confidence[safe_indices]
    support_positive = support[np.isfinite(support) & (support > 0)]
    support_median = float(np.median(support_positive)) if len(support_positive) else 1.0

    w_quality = _neighbor_quality_weights(
        support_neighbors,
        confidence_neighbors,
        support_median=support_median,
        support_exponent=args.support_neighbor_weight_exponent,
        confidence_exponent=args.confidence_neighbor_weight_exponent,
    )

    # Compatibilidad de procedencia: una capa respaldada por un conjunto de
    # poses distinto no debe arrastrar otra durante el suavizado. Esto no impone
    # forma; únicamente usa la evidencia multivista publicada por el paso 11.
    if support_pose_mask is not None and len(support_pose_mask) == n_points:
        own_mask = np.asarray(support_pose_mask, dtype=np.uint64)[:, None]
        neighbor_mask = np.asarray(support_pose_mask, dtype=np.uint64)[safe_indices]
        shared = bit_count_u64(own_mask & neighbor_mask)
        strong_shared = max(1, int(args.pose_shared_support_strong))
        w_pose = np.where(
            shared >= strong_shared,
            1.0,
            np.where(
                shared >= 1,
                float(args.pose_shared_support_weak_weight),
                float(args.pose_no_shared_support_weight),
            ),
        ).astype(np.float64)
        if disagreement_pose_mask is not None and len(disagreement_pose_mask) == n_points:
            own_conflict = np.asarray(disagreement_pose_mask, dtype=np.uint64)[:, None]
            neigh_conflict = np.asarray(disagreement_pose_mask, dtype=np.uint64)[safe_indices]
            crossed = bit_count_u64((own_conflict & neighbor_mask) | (neigh_conflict & own_mask))
            w_pose *= np.exp(-0.80 * crossed.astype(np.float64))
    else:
        w_pose = np.ones_like(w_quality, dtype=np.float64)

    if evidence_strength is not None and len(evidence_strength) == n_points:
        evidence_strength = np.clip(np.asarray(evidence_strength, dtype=np.float64), 0.0, 1.0)
        neighbor_evidence = evidence_strength[safe_indices]
        w_evidence = 0.20 + 0.80 * neighbor_evidence
    else:
        evidence_strength = np.ones(n_points, dtype=np.float64)
        w_evidence = np.ones_like(w_quality, dtype=np.float64)

    # Primer conjunto coherente para escala robusta en altura.
    coherent = valid & (
        angle_deg
        <= max(
            1.6 * float(args.feature_protect_angle_deg),
            30.0,
        )
    )

    # V2.1: mediana enmascarada para no emitir All-NaN slice warnings
    # en puntos que temporalmente no tienen vecinos coherentes.
    h_masked = np.ma.array(
        signed_height,
        mask=~coherent,
    )
    h_median = (
        np.ma.median(
            h_masked,
            axis=1,
        )
        .filled(0.0)
        .astype(np.float64)
    )

    h_abs_dev = np.abs(signed_height - h_median[:, None])
    dev_masked = np.ma.array(
        h_abs_dev,
        mask=~coherent,
    )
    mad = (
        np.ma.median(
            dev_masked,
            axis=1,
        )
        .filled(0.0)
        .astype(np.float64)
    )

    h_sigma = np.maximum(
        1.4826 * mad,
        float(args.height_robust_scale_spacing_floor) * float(spacing),
    )

    h_denom = np.maximum(
        float(args.height_robust_sigma_factor) * h_sigma,
        1e-8,
    )
    w_height = np.exp(-0.5 * (h_abs_dev / h_denom[:, None]) ** 2)

    weights = w_spatial * w_normal * w_height * w_quality * w_pose * w_evidence * valid

    wsum = np.sum(
        weights,
        axis=1,
    )
    neighbor_count = np.sum(
        valid,
        axis=1,
    )

    raw_shift = np.divide(
        np.sum(
            weights * signed_height,
            axis=1,
        ),
        wsum,
        out=np.zeros(
            n_points,
            dtype=np.float64,
        ),
        where=wsum > 1e-12,
    )

    # --------------------------------------------------------------
    # Protección de características:
    # dispersión de normales local.
    # --------------------------------------------------------------
    base_for_feature = w_spatial * valid
    base_sum = np.sum(
        base_for_feature,
        axis=1,
    )
    normal_dispersion = np.divide(
        np.sum(
            base_for_feature * angle_deg,
            axis=1,
        ),
        base_sum,
        out=np.full(
            n_points,
            90.0,
            dtype=np.float64,
        ),
        where=base_sum > 1e-12,
    )

    feature_angle = max(
        float(args.feature_protect_angle_deg),
        1e-6,
    )
    feature_factor = 1.0 / (1.0 + (normal_dispersion / feature_angle) ** 4)
    feature_factor = np.clip(
        feature_factor,
        float(args.feature_min_strength),
        1.0,
    )

    # --------------------------------------------------------------
    # Protección de borde abierto:
    # un vecindario interior es aproximadamente balanceado en el plano
    # tangente; junto a un borde, su centro de masa tangencial se desplaza.
    # --------------------------------------------------------------
    boundary_base = w_spatial * w_normal * valid
    boundary_sum = np.sum(
        boundary_base,
        axis=1,
    )

    mean_tangent = np.divide(
        np.einsum(
            "nk,nkj->nj",
            boundary_base,
            tangent,
        ),
        boundary_sum[:, None],
        out=np.zeros(
            (n_points, 3),
            dtype=np.float64,
        ),
        where=boundary_sum[:, None] > 1e-12,
    )

    mean_tangent_radius = np.divide(
        np.sum(
            boundary_base * tangent_distance,
            axis=1,
        ),
        boundary_sum,
        out=np.ones(
            n_points,
            dtype=np.float64,
        ),
        where=boundary_sum > 1e-12,
    )

    boundary_asymmetry = np.linalg.norm(
        mean_tangent,
        axis=1,
    ) / np.maximum(
        mean_tangent_radius,
        1e-9,
    )

    b_low = float(args.boundary_asymmetry_low)
    b_high = max(
        float(args.boundary_asymmetry_high),
        b_low + 1e-6,
    )

    t = np.clip(
        (boundary_asymmetry - b_low) / (b_high - b_low),
        0.0,
        1.0,
    )
    boundary_factor = 1.0 - t * (1.0 - float(args.boundary_min_strength))

    # --------------------------------------------------------------
    # Puntos con alta evidencia se usan como anclas suaves, no rígidas.
    # --------------------------------------------------------------
    target_support = np.clip(
        support.astype(np.float64)
        / max(
            support_median,
            1.0,
        ),
        0.5,
        2.5,
    )
    target_conf = np.clip(
        confidence.astype(np.float64),
        0.0,
        1.0,
    )

    target_quality = (
        0.55
        * np.clip(
            target_support - 1.0,
            0.0,
            1.5,
        )
        / 1.5
        + 0.45 * target_conf
    )
    anchor_factor = 1.0 / (1.0 + float(args.quality_anchor_strength) * target_quality)

    strength = (
        float(args.smooth_strength)
        * feature_factor
        * boundary_factor
        * anchor_factor
        * (0.35 + 0.65 * evidence_strength)
    )

    enough = neighbor_count >= int(args.minimum_smoothing_neighbors)
    strength[~enough] = 0.0

    shift = raw_shift * strength

    max_step = float(args.max_shift_per_iteration_spacing) * float(spacing)
    step_clipped = np.abs(shift) > max_step
    shift = np.clip(
        shift,
        -max_step,
        max_step,
    )

    candidate = p + shift[:, None] * n

    # Cap total Euclídeo respecto al punto filtrado original.
    total_cap = float(args.max_total_shift_spacing) * float(spacing)
    total_vector = candidate - original_points
    total_norm = np.linalg.norm(
        total_vector,
        axis=1,
    )
    total_clipped = total_norm > total_cap

    if np.any(total_clipped):
        candidate[total_clipped] = original_points[total_clipped] + total_vector[total_clipped] * (
            total_cap
            / total_norm[
                total_clipped,
                None,
            ]
        )

    diagnostics = {
        "shift_abs_mm": np.abs(
            np.sum(
                (candidate - p) * n,
                axis=1,
            )
        ),
        "raw_shift_abs_mm": np.abs(raw_shift),
        "feature_factor": feature_factor,
        "boundary_factor": boundary_factor,
        "boundary_asymmetry": boundary_asymmetry,
        "normal_dispersion_deg": normal_dispersion,
        "neighbor_count": neighbor_count,
        "step_clipped": step_clipped,
        "total_clipped": total_clipped,
        "height_sigma_mm": h_sigma,
        "pose_compatibility_mean": np.divide(
            np.sum(w_spatial * w_pose * valid, axis=1),
            np.sum(w_spatial * valid, axis=1),
            out=np.zeros(n_points, dtype=np.float64),
            where=np.sum(w_spatial * valid, axis=1) > 1e-12,
        ),
    }

    return candidate, diagnostics


@operacion("Regularizar superficie de la nube")
def regularize_surface(
    points: np.ndarray,
    support: np.ndarray,
    confidence: np.ndarray,
    *,
    spacing: float,
    voxel: float,
    initial_normals: Optional[np.ndarray],
    support_pose_mask: Optional[np.ndarray] = None,
    disagreement_pose_mask: Optional[np.ndarray] = None,
    evidence_strength: Optional[np.ndarray] = None,
    args,
):
    p0 = np.asarray(
        points,
        dtype=np.float64,
    )
    p = p0.copy()

    smooth_radius = (
        float(args.smooth_radius_mm)
        if float(args.smooth_radius_mm) > 0
        else max(
            float(args.smooth_radius_spacing_factor) * float(spacing),
            2.0 * float(voxel),
            1.5 * float(spacing),
        )
    )

    normal_radius = max(
        float(args.normal_pca_radius_factor) * float(spacing),
        1.20 * smooth_radius,
    )

    # Primera normal local.
    normals, curvature, normal_counts = estimate_pca_normals(
        p,
        k=int(args.normal_pca_knn),
        radius=normal_radius,
        reference_normals=initial_normals,
    )

    aggregate = {
        "feature_factor_min": np.ones(
            len(p),
            dtype=np.float64,
        ),
        "boundary_factor_min": np.ones(
            len(p),
            dtype=np.float64,
        ),
        "boundary_asymmetry_max": np.zeros(
            len(p),
            dtype=np.float64,
        ),
        "normal_dispersion_max_deg": np.zeros(
            len(p),
            dtype=np.float64,
        ),
        "step_clipped_any": np.zeros(
            len(p),
            dtype=bool,
        ),
        "total_clipped_any": np.zeros(
            len(p),
            dtype=bool,
        ),
        "pose_compatibility_min": np.ones(
            len(p),
            dtype=np.float64,
        ),
    }

    per_iteration = []

    iterations = max(
        0,
        int(args.smooth_iterations),
    )

    for iteration in range(iterations):
        next_points, diag = surface_bilateral_iteration(
            p,
            normals,
            support,
            confidence,
            spacing=float(spacing),
            radius=float(smooth_radius),
            k=int(args.smooth_knn),
            original_points=p0,
            support_pose_mask=support_pose_mask,
            disagreement_pose_mask=disagreement_pose_mask,
            evidence_strength=evidence_strength,
            args=args,
        )

        per_iteration.append(
            {
                "iteration": iteration + 1,
                "shift_abs_mm": stats(diag["shift_abs_mm"]),
                "raw_shift_abs_mm": stats(diag["raw_shift_abs_mm"]),
                "normal_dispersion_deg": stats(diag["normal_dispersion_deg"]),
                "boundary_asymmetry": stats(diag["boundary_asymmetry"]),
                "step_clipped_points": int(np.count_nonzero(diag["step_clipped"])),
                "total_clipped_points": int(np.count_nonzero(diag["total_clipped"])),
            }
        )

        aggregate["feature_factor_min"] = np.minimum(
            aggregate["feature_factor_min"],
            diag["feature_factor"],
        )
        aggregate["boundary_factor_min"] = np.minimum(
            aggregate["boundary_factor_min"],
            diag["boundary_factor"],
        )
        aggregate["boundary_asymmetry_max"] = np.maximum(
            aggregate["boundary_asymmetry_max"],
            diag["boundary_asymmetry"],
        )
        aggregate["normal_dispersion_max_deg"] = np.maximum(
            aggregate["normal_dispersion_max_deg"],
            diag["normal_dispersion_deg"],
        )
        aggregate["step_clipped_any"] |= diag["step_clipped"]
        aggregate["total_clipped_any"] |= diag["total_clipped"]
        aggregate["pose_compatibility_min"] = np.minimum(
            aggregate["pose_compatibility_min"],
            diag["pose_compatibility_mean"],
        )

        p = next_points

        # Reestima normales en la geometría actual, alineadas a las anteriores.
        normals, curvature, normal_counts = estimate_pca_normals(
            p,
            k=int(args.normal_pca_knn),
            radius=normal_radius,
            reference_normals=normals,
        )

    total_displacement = np.linalg.norm(
        p - p0,
        axis=1,
    )

    return {
        "points": p,
        "normals": normals,
        "curvature": curvature,
        "normal_neighbor_count": normal_counts,
        "total_displacement_mm": total_displacement,
        "smooth_radius_mm": float(smooth_radius),
        "normal_pca_radius_mm": float(normal_radius),
        "aggregate": aggregate,
        "iterations": per_iteration,
    }


# ---------------------------------------------------------------------------
# Métrica de rugosidad local
# ---------------------------------------------------------------------------


def evaluate_local_roughness(
    points: np.ndarray,
    *,
    k: int,
    radius: float,
    normals: Optional[np.ndarray] = None,
    maximum_points: int = 60000,
):
    """Mide rugosidad local respecto a las normales de los vecindarios."""
    p = np.asarray(
        points,
        dtype=np.float64,
    )

    if len(p) < 5:
        return {
            "sample_points": int(len(p)),
            "roughness_mm": stats([]),
        }

    if len(p) > int(maximum_points):
        idx_sample = np.linspace(
            0,
            len(p) - 1,
            int(maximum_points),
        ).astype(int)
    else:
        idx_sample = np.arange(
            len(p),
            dtype=int,
        )

    sample = p[idx_sample]

    if normals is None:
        n_full, _, _ = estimate_pca_normals(
            p,
            k=max(8, int(k)),
            radius=float(radius),
            reference_normals=None,
        )
    else:
        n_full = safe_unit_rows(normals)

    n_sample = n_full[idx_sample]

    tree = cKDTree(p)
    kk = int(
        np.clip(
            k,
            4,
            len(p),
        )
    )

    d, ind = tree.query(
        sample,
        k=kk,
        workers=query_threads(),
    )
    if d.ndim == 1:
        d = d[:, None]
        ind = ind[:, None]

    valid = np.isfinite(d) & (ind >= 0) & (ind < len(p)) & (d > 1e-10) & (d <= float(radius))

    safe_ind = np.clip(
        ind,
        0,
        len(p) - 1,
    )
    q = p[safe_ind]
    vec = q - sample[:, None, :]
    h = np.einsum(
        "nkj,nj->nk",
        vec,
        n_sample,
    )

    h = np.where(
        valid,
        h,
        np.nan,
    )
    med = np.nanmedian(
        h,
        axis=1,
    )
    absdev = np.abs(h - med[:, None])
    mad = np.nanmedian(
        absdev,
        axis=1,
    )
    rough = 1.4826 * mad

    count = np.sum(
        valid,
        axis=1,
    )
    rough[count < 5] = np.nan

    return {
        "sample_points": int(len(sample)),
        "roughness_mm": stats(rough),
        "neighbor_count": stats(count),
    }


# ---------------------------------------------------------------------------
# V2.1 — Guarda de escala de muestreo
# ---------------------------------------------------------------------------


def spacing_median_deterministic(
    points: np.ndarray,
    maximum_sample_points: int = 30000,
) -> float:
    """Estima el espaciado mediano mediante un muestreo determinista."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return float("nan")

    if len(p) > int(maximum_sample_points):
        idx = np.linspace(
            0,
            len(p) - 1,
            int(maximum_sample_points),
        ).astype(int)
        query = p[idx]
    else:
        query = p

    tree = cKDTree(p)
    d, _ = tree.query(
        query,
        k=2,
        workers=query_threads(),
    )
    nn = np.asarray(d[:, 1], dtype=np.float64)
    nn = nn[np.isfinite(nn) & (nn > 1e-12)]
    if len(nn) == 0:
        return float("nan")
    return float(np.median(nn))


def apply_sampling_scale_guard(
    original_points: np.ndarray,
    smoothed_points: np.ndarray,
    *,
    reference_spacing_mm: float,
    minimum_ratio: float,
    bisection_steps: int,
    sample_points: int,
):
    """Reduce solo la INTENSIDAD efectiva del suavizado si colapsa el spacing.

    No añade/elimina puntos y no modifica topología. Busca el mayor alpha:

        P_final = P_original + alpha * (P_smooth - P_original)

    tal que spacing(P_final) >= minimum_ratio * spacing_original.
    """
    p0 = np.asarray(original_points, dtype=np.float64)
    ps = np.asarray(smoothed_points, dtype=np.float64)

    reference = float(reference_spacing_mm)
    min_ratio = float(np.clip(minimum_ratio, 0.55, 0.99))
    target = reference * min_ratio

    full_spacing = spacing_median_deterministic(
        ps,
        maximum_sample_points=int(sample_points),
    )

    if not np.isfinite(full_spacing) or full_spacing >= target:
        return ps, {
            "activated": False,
            "blend_alpha": 1.0,
            "reference_spacing_mm": reference,
            "target_minimum_spacing_mm": target,
            "full_smoothing_spacing_mm": (
                float(full_spacing) if np.isfinite(full_spacing) else None
            ),
            "final_spacing_mm": (float(full_spacing) if np.isfinite(full_spacing) else None),
            "minimum_ratio": min_ratio,
        }

    # alpha=0 es la nube filtrada original, que por construcción debe
    # satisfacer aproximadamente la referencia.
    low = 0.0
    high = 1.0
    best_alpha = 0.0
    best_points = p0.copy()
    best_spacing = spacing_median_deterministic(
        p0,
        maximum_sample_points=int(sample_points),
    )

    for _ in range(max(1, int(bisection_steps))):
        alpha = 0.5 * (low + high)
        candidate = p0 + alpha * (ps - p0)

        candidate_spacing = spacing_median_deterministic(
            candidate,
            maximum_sample_points=int(sample_points),
        )

        if np.isfinite(candidate_spacing) and candidate_spacing >= target:
            best_alpha = alpha
            best_points = candidate
            best_spacing = candidate_spacing
            low = alpha
        else:
            high = alpha

    return best_points, {
        "activated": True,
        "blend_alpha": float(best_alpha),
        "reference_spacing_mm": reference,
        "target_minimum_spacing_mm": float(target),
        "full_smoothing_spacing_mm": float(full_spacing),
        "final_spacing_mm": float(best_spacing),
        "minimum_ratio": min_ratio,
        "interpretation": (
            "Se redujo automáticamente la intensidad del suavizado para "
            "evitar colapso de capas y una caída excesiva del spacing que "
            "desestabilice Ball Pivoting."
        ),
    }


# ---------------------------------------------------------------------------
# Open3D / I/O
# ---------------------------------------------------------------------------


def require_open3d():
    """Detiene el paso con un diagnóstico si Open3D no está disponible."""
    if o3d is None:
        raise SystemExit(
            "Paso 12 requiere Open3D 0.19.x. "
            "En el entorno de la tesis ejecuta "
            "`python -m pip install open3d==0.19.0`."
        )


def make_pcd(
    points,
    colors,
    normals=None,
):
    """Construye una nube Open3D con posiciones, colores y normales opcionales."""
    require_open3d()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(
            points,
            dtype=np.float64,
        )
    )
    pcd.colors = o3d.utility.Vector3dVector(
        np.asarray(
            colors,
            dtype=np.float64,
        )
        / 255.0
    )

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(safe_unit_rows(normals))

    return pcd


def save_ply(path, pcd):
    """Guarda la nube en PLY y comprueba que Open3D confirme la escritura."""
    require_open3d()
    if not o3d.io.write_point_cloud(
        str(path),
        pcd,
        write_ascii=False,
        compressed=False,
    ):
        raise RuntimeError(f"No se pudo guardar {path}")


def orient_final_normals(
    points,
    colors,
    initial_normals,
    *,
    normal_radius,
    max_nn,
    orientation_k,
):
    """Estima y orienta las normales de la nube final."""
    pcd = make_pcd(
        points,
        colors,
        initial_normals,
    )

    # Reestima sobre la nube ya regularizada para 13.
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(normal_radius),
            max_nn=int(max_nn),
        )
    )
    pcd.normalize_normals()

    orientation_ok = True
    try:
        pcd.orient_normals_consistent_tangent_plane(int(orientation_k))
    except Exception as exc:
        orientation_ok = False
        print(
            "[WARNING] No se pudo propagar orientación de normales:",
            exc,
        )

    normals = np.asarray(
        pcd.normals,
        dtype=np.float64,
    )

    # Solo corrige SIGNO GLOBAL, no impone convexidad.
    if len(normals) == len(points) and len(points):
        centroid = np.median(
            points,
            axis=0,
        )
        votes = np.sum(
            normals * (points - centroid[None, :]),
            axis=1,
        )

        if np.nanmedian(votes) < 0:
            normals = -normals
            pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd, normals, orientation_ok


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


def preview(
    path,
    points,
    support,
    confidence,
    title,
):
    if plt is None:
        raise RuntimeError("No se puede generar preview porque matplotlib no está disponible.")

    idx = np.arange(len(points))
    if len(idx) > 140000:
        idx = np.linspace(
            0,
            len(points) - 1,
            140000,
        ).astype(int)

    P = points[idx]
    S = support[idx]
    C = confidence[idx]

    fig = plt.figure(figsize=(15, 10))

    for k, (
        dims,
        panel_title,
        col,
    ) in enumerate(
        [
            ((0, 1), "X-Y", S),
            ((0, 2), "X-Z", C),
            ((2, 1), "Z-Y", S),
        ],
        start=1,
    ):
        ax = fig.add_subplot(
            2,
            2,
            k,
        )
        ax.scatter(
            P[:, dims[0]],
            P[:, dims[1]],
            c=col,
            s=0.4,
            cmap="viridis",
        )
        ax.set_title(panel_title)
        ax.set_aspect(
            "equal",
            adjustable="box",
        )

    ax = fig.add_subplot(
        2,
        2,
        4,
    )
    ax.axis("off")
    ax.text(
        0.05,
        0.95,
        (
            f"{title}\n\n"
            f"Puntos: {len(points):,}\n"
            f"Soporte mediano: {np.median(support):.2f}\n"
            f"Confianza mediana: {np.median(confidence):.3f}"
        ),
        va="top",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
    )
    plt.close(fig)


def comparison_preview(
    path,
    before,
    after,
    displacement,
):
    if plt is None:
        raise RuntimeError(
            "No se puede generar preview de comparación porque matplotlib no está disponible."
        )

    n = len(before)
    idx = np.arange(n)
    if n > 90000:
        idx = np.linspace(
            0,
            n - 1,
            90000,
        ).astype(int)

    B = before[idx]
    A = after[idx]
    D = displacement[idx]

    fig = plt.figure(figsize=(16, 10))

    ax = fig.add_subplot(
        2,
        2,
        1,
    )
    ax.scatter(
        B[:, 0],
        B[:, 1],
        s=0.35,
    )
    ax.set_title("Antes — X-Y")
    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax = fig.add_subplot(
        2,
        2,
        2,
    )
    ax.scatter(
        A[:, 0],
        A[:, 1],
        s=0.35,
    )
    ax.set_title("Después — X-Y")
    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax = fig.add_subplot(
        2,
        2,
        3,
    )
    sc = ax.scatter(
        A[:, 0],
        A[:, 2],
        c=D,
        s=0.45,
        cmap="viridis",
    )
    ax.set_title("Desplazamiento — X-Z")
    ax.set_aspect(
        "equal",
        adjustable="box",
    )
    fig.colorbar(
        sc,
        ax=ax,
        label="mm",
    )

    ax = fig.add_subplot(
        2,
        2,
        4,
    )
    ax.axis("off")
    dstat = stats(displacement)
    ax.text(
        0.05,
        0.95,
        (
            "Limpieza 3D + regularización superficial V3.0\n\n"
            f"Desplazamiento mediano: {dstat['median']:.3f} mm\n"
            f"P90: {dstat['p90']:.3f} mm\n"
            f"P95: {dstat['p95']:.3f} mm\n"
            f"Máximo: {dstat['max']:.3f} mm"
        ),
        va="top",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Limpia y regulariza la nube fusionada con controles de escala y evidencia."""
    args = parser().parse_args()
    require_open3d()

    root = Path(args.root).expanduser().resolve()
    obj = args.object.strip()

    source = root / "reconstruccion" / "multisesion" / args.source
    npz = source / "nube_fusionada_multivista.npz"

    if not npz.is_file():
        raise FileNotFoundError(npz)

    with np.load(npz) as d:
        points_all = np.asarray(
            d["points"],
            dtype=np.float64,
        )
        colors_all = np.asarray(
            d["colors"],
            dtype=np.uint8,
        )
        support_all = np.asarray(
            d["support_views"],
            dtype=np.uint16,
        )
        confidence_all = np.asarray(
            d["confidence"],
            dtype=np.float64,
        )
        voxel = float(np.asarray(d["fusion_voxel_mm"]).ravel()[0])
        strong_core_support_threshold = int(
            np.asarray(
                d["strong_core_support_threshold"]
                if "strong_core_support_threshold" in d.files
                else np.asarray([4], dtype=np.uint16)
            ).ravel()[0]
        )
        selection_class_all = np.asarray(
            (
                d["consensus_selection_class"]
                if "consensus_selection_class" in d.files
                else np.where(support_all >= strong_core_support_threshold, 3, 2)
            ),
            dtype=np.uint8,
        )
        has_evidence_contract = "surface_evidence_class" in d.files
        surface_evidence_all = np.asarray(
            d["surface_evidence_class"] if has_evidence_contract else np.full(len(points_all), 2),
            dtype=np.uint8,
        )
        independent_support_all = np.asarray(
            (
                d["independent_support_poses"]
                if "independent_support_poses" in d.files
                else support_all
            ),
            dtype=np.int16,
        )
        angular_span_all = np.asarray(
            (
                d["support_angular_span_poses"]
                if "support_angular_span_poses" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=np.int16,
        )
        conflict_ratio_all = np.asarray(
            (
                d["conflict_pose_ratio"]
                if "conflict_pose_ratio" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=np.float64,
        )
        heldout_available_all = np.asarray(
            (
                d["heldout_validation_available"]
                if "heldout_validation_available" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=bool,
        )
        heldout_pass_all = np.asarray(
            (
                d["heldout_validation_pass_ratio"]
                if "heldout_validation_pass_ratio" in d.files
                else np.ones(len(points_all))
            ),
            dtype=np.float64,
        )
        heldout_tests_all = np.asarray(
            (
                d["heldout_validation_tests"]
                if "heldout_validation_tests" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=np.int16,
        )
        if "heldout_validation_pass_count" in d.files:
            heldout_pass_count_all = np.asarray(d["heldout_validation_pass_count"], dtype=np.int16)
        else:
            # Compatibilidad exacta con 11 V11.0: el ratio fue generado a
            # partir de enteros; se reconstruye el conteo por redondeo.
            heldout_pass_count_all = np.rint(
                heldout_pass_all * np.maximum(heldout_tests_all, 0)
            ).astype(np.int16)
        heldout_error_all = np.asarray(
            (
                d["heldout_validation_error_mm"]
                if "heldout_validation_error_mm" in d.files
                else np.full(len(points_all), np.nan)
            ),
            dtype=np.float64,
        )
        heldout_p90_all = np.asarray(
            (
                d["heldout_validation_p90_mm"]
                if "heldout_validation_p90_mm" in d.files
                else np.full(len(points_all), np.nan)
            ),
            dtype=np.float64,
        )
        position_uncertainty_all = np.asarray(
            (
                d["position_uncertainty_mm"]
                if "position_uncertainty_mm" in d.files
                else np.full(len(points_all), np.nan)
            ),
            dtype=np.float64,
        )
        support_pose_mask_all = np.asarray(
            (
                d["support_pose_mask"]
                if "support_pose_mask" in d.files
                else np.zeros(len(points_all), dtype=np.uint64)
            ),
            dtype=np.uint64,
        )
        disagreement_pose_mask_all = np.asarray(
            (
                d["disagreement_pose_mask"]
                if "disagreement_pose_mask" in d.files
                else np.zeros(len(points_all), dtype=np.uint64)
            ),
            dtype=np.uint64,
        )
        has_pose_layer_contract = (
            "validated_pose_layer_contract_valid" in d.files
            and bool(int(np.asarray(d["validated_pose_layer_contract_valid"]).reshape(-1)[0]))
            and "validated_layer_accept" in d.files
        )
        validated_layer_available_all = np.asarray(
            (
                d["validated_layer_available"]
                if "validated_layer_available" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=np.uint8,
        )
        validated_layer_accept_all = np.asarray(
            (
                d["validated_layer_accept"]
                if "validated_layer_accept" in d.files
                else np.ones(len(points_all))
            ),
            dtype=bool,
        )
        validated_layer_count_all = np.asarray(
            (
                d["validated_layer_count"]
                if "validated_layer_count" in d.files
                else np.ones(len(points_all))
            ),
            dtype=np.uint8,
        )
        validated_layer_ambiguous_all = np.asarray(
            (
                d["validated_layer_ambiguous"]
                if "validated_layer_ambiguous" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=bool,
        )
        validated_layer_resolved_all = np.asarray(
            (
                d["validated_layer_resolved"]
                if "validated_layer_resolved" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=bool,
        )
        validated_layer_separation_all = np.asarray(
            (
                d["validated_layer_separation_mm"]
                if "validated_layer_separation_mm" in d.files
                else np.full(len(points_all), np.nan)
            ),
            dtype=np.float64,
        )
        validated_layer_shift_all = np.asarray(
            (
                d["validated_layer_shift_mm"]
                if "validated_layer_shift_mm" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=np.float64,
        )
        validated_layer_margin_all = np.asarray(
            (
                d["validated_layer_score_margin"]
                if "validated_layer_score_margin" in d.files
                else np.full(len(points_all), np.nan)
            ),
            dtype=np.float64,
        )
        validated_layer_selected_mask_all = np.asarray(
            (
                d["validated_layer_selected_pose_mask"]
                if "validated_layer_selected_pose_mask" in d.files
                else support_pose_mask_all
            ),
            dtype=np.uint64,
        )
        validated_layer_secondary_mask_all = np.asarray(
            (
                d["validated_layer_secondary_pose_mask"]
                if "validated_layer_secondary_pose_mask" in d.files
                else np.zeros(len(points_all), dtype=np.uint64)
            ),
            dtype=np.uint64,
        )
        has_local_coherence_contract = (
            "local_coherence_contract_valid" in d.files
            and bool(int(np.asarray(d["local_coherence_contract_valid"]).reshape(-1)[0]))
            and "surface_local_coherence_score" in d.files
            and "surface_local_coherence_accept" in d.files
        )
        local_coherence_available_all = np.asarray(
            (
                d["surface_local_coherence_available"]
                if "surface_local_coherence_available" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=np.uint8,
        )
        local_coherence_accept_all = np.asarray(
            (
                d["surface_local_coherence_accept"]
                if "surface_local_coherence_accept" in d.files
                else np.ones(len(points_all))
            ),
            dtype=bool,
        )
        local_coherence_strict_all = np.asarray(
            (
                d["surface_local_coherence_strict"]
                if "surface_local_coherence_strict" in d.files
                else np.ones(len(points_all))
            ),
            dtype=bool,
        )
        local_feature_like_all = np.asarray(
            (
                d["surface_local_feature_like"]
                if "surface_local_feature_like" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=bool,
        )
        local_coherence_score_all = np.asarray(
            (
                d["surface_local_coherence_score"]
                if "surface_local_coherence_score" in d.files
                else np.ones(len(points_all))
            ),
            dtype=np.float64,
        )
        local_coherence_red_flags_all = np.asarray(
            (
                d["surface_local_coherence_red_flags"]
                if "surface_local_coherence_red_flags" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=np.uint8,
        )
        local_plane_p90_small_all = np.asarray(
            (
                d["surface_local_plane_p90_small_mm"]
                if "surface_local_plane_p90_small_mm" in d.files
                else np.full(len(points_all), np.nan)
            ),
            dtype=np.float64,
        )
        local_plane_p90_large_all = np.asarray(
            (
                d["surface_local_plane_p90_large_mm"]
                if "surface_local_plane_p90_large_mm" in d.files
                else np.full(len(points_all), np.nan)
            ),
            dtype=np.float64,
        )
        local_normal_p90_all = np.asarray(
            (
                d["surface_local_normal_p90_large_deg"]
                if "surface_local_normal_p90_large_deg" in d.files
                else np.full(len(points_all), np.nan)
            ),
            dtype=np.float64,
        )
        local_scale_normal_diff_all = np.asarray(
            (
                d["surface_local_scale_normal_difference_deg"]
                if "surface_local_scale_normal_difference_deg" in d.files
                else np.full(len(points_all), np.nan)
            ),
            dtype=np.float64,
        )
        local_variation_all = np.asarray(
            (
                d["surface_local_variation_large"]
                if "surface_local_variation_large" in d.files
                else np.full(len(points_all), np.nan)
            ),
            dtype=np.float64,
        )
        local_same_sheet_fraction_all = np.asarray(
            (
                d["surface_local_same_sheet_fraction"]
                if "surface_local_same_sheet_fraction" in d.files
                else np.full(len(points_all), np.nan)
            ),
            dtype=np.float64,
        )
        has_regional_pose_consensus_contract = (
            "regional_pose_consensus_contract_valid" in d.files
            and bool(int(np.asarray(d["regional_pose_consensus_contract_valid"]).reshape(-1)[0]))
            and "regional_pose_consensus_score" in d.files
        )
        regional_pose_available_all = np.asarray(
            (
                d["regional_pose_consensus_available"]
                if "regional_pose_consensus_available" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=np.uint8,
        )
        regional_pose_accept_all = np.asarray(
            (
                d["regional_pose_consensus_accept"]
                if "regional_pose_consensus_accept" in d.files
                else np.ones(len(points_all))
            ),
            dtype=bool,
        )
        regional_pose_ambiguous_all = np.asarray(
            (
                d["regional_pose_consensus_ambiguous"]
                if "regional_pose_consensus_ambiguous" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=bool,
        )
        regional_pose_feature_protected_all = np.asarray(
            (
                d["regional_pose_consensus_feature_protected"]
                if "regional_pose_consensus_feature_protected" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=bool,
        )
        regional_pose_shift_all = np.asarray(
            (
                d["regional_pose_consensus_shift_mm"]
                if "regional_pose_consensus_shift_mm" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=np.float64,
        )
        regional_pose_score_all = np.asarray(
            (
                d["regional_pose_consensus_score"]
                if "regional_pose_consensus_score" in d.files
                else np.ones(len(points_all))
            ),
            dtype=np.float64,
        )
        regional_pose_independent_all = np.asarray(
            (
                d["regional_pose_consensus_independent_predictions"]
                if "regional_pose_consensus_independent_predictions" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=np.int16,
        )
        regional_pose_dispersion_all = np.asarray(
            (
                d["regional_pose_consensus_pose_dispersion_mm"]
                if "regional_pose_consensus_pose_dispersion_mm" in d.files
                else np.full(len(points_all), np.nan)
            ),
            dtype=np.float64,
        )
        regional_pose_scale_difference_all = np.asarray(
            (
                d["regional_pose_consensus_scale_difference_mm"]
                if "regional_pose_consensus_scale_difference_mm" in d.files
                else np.full(len(points_all), np.nan)
            ),
            dtype=np.float64,
        )
        has_raw_pose_bias_contract = (
            "raw_pose_bias_contract_valid" in d.files
            and bool(int(np.asarray(d["raw_pose_bias_contract_valid"]).reshape(-1)[0]))
            and "raw_pose_bias_score" in d.files
        )
        raw_pose_bias_available_all = np.asarray(
            (
                d["raw_pose_bias_available"]
                if "raw_pose_bias_available" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=np.uint8,
        )
        raw_pose_bias_accept_all = np.asarray(
            (
                d["raw_pose_bias_accept"]
                if "raw_pose_bias_accept" in d.files
                else np.ones(len(points_all))
            ),
            dtype=bool,
        )
        raw_pose_bias_ambiguous_all = np.asarray(
            (
                d["raw_pose_bias_ambiguous"]
                if "raw_pose_bias_ambiguous" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=bool,
        )
        raw_pose_bias_feature_protected_all = np.asarray(
            (
                d["raw_pose_bias_feature_protected"]
                if "raw_pose_bias_feature_protected" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=bool,
        )
        raw_pose_bias_shift_all = np.asarray(
            (
                d["raw_pose_bias_shift_mm"]
                if "raw_pose_bias_shift_mm" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=np.float64,
        )
        raw_pose_bias_score_all = np.asarray(
            (
                d["raw_pose_bias_score"]
                if "raw_pose_bias_score" in d.files
                else np.ones(len(points_all))
            ),
            dtype=np.float64,
        )
        raw_pose_bias_independent_all = np.asarray(
            (
                d["raw_pose_bias_independent_poses"]
                if "raw_pose_bias_independent_poses" in d.files
                else np.zeros(len(points_all))
            ),
            dtype=np.int16,
        )
        raw_pose_bias_scale_difference_all = np.asarray(
            (
                d["raw_pose_bias_scale_difference_mm"]
                if "raw_pose_bias_scale_difference_mm" in d.files
                else np.full(len(points_all), np.nan)
            ),
            dtype=np.float64,
        )
        raw_pose_bias_model_p90_all = np.asarray(
            (
                d["raw_pose_bias_model_p90_mm"]
                if "raw_pose_bias_model_p90_mm" in d.files
                else np.full(len(points_all), np.nan)
            ),
            dtype=np.float64,
        )

    input_count = len(points_all)

    if not has_evidence_contract:
        print(
            "[WARNING] Paso 12 V4.6: la entrada del paso 11 no publica el contrato "
            "de evidencia pose-diversa/held-out. Se permite procesar para "
            "compatibilidad, pero la salida se marca evidence_contract_valid=0 y "
            "el paso 13 de producción exigirá reejecutar 11 y 12.",
            flush=True,
        )

    # --------------------------------------------------------------
    # Limpieza histórica conservadora.
    # --------------------------------------------------------------
    pcd_all = make_pcd(
        points_all,
        colors_all,
    )

    _, sor_idx = pcd_all.remove_statistical_outlier(
        nb_neighbors=int(args.sor_neighbors),
        std_ratio=float(args.sor_std_ratio),
    )
    _, ror_idx = pcd_all.remove_radius_outlier(
        nb_points=int(args.radius_neighbors),
        radius=float(
            args.radius_mm
            if args.radius_mm > 0
            else max(
                3.5,
                2.5 * voxel,
            )
        ),
    )

    sor = np.zeros(
        input_count,
        bool,
    )
    ror = np.zeros(
        input_count,
        bool,
    )

    sor[
        np.asarray(
            sor_idx,
            dtype=int,
        )
    ] = True
    ror[
        np.asarray(
            ror_idx,
            dtype=int,
        )
    ] = True

    if has_evidence_contract:
        heldout_num = max(0, int(args.minimum_input_heldout_pass_numerator))
        heldout_den = max(1, int(args.minimum_input_heldout_pass_denominator))
        heldout_vote_ok_all = (~heldout_available_all) | (
            heldout_pass_count_all.astype(np.int64) * heldout_den
            >= heldout_num * heldout_tests_all.astype(np.int64)
        )

        uncertainty_floor = max(
            1e-6,
            float(args.heldout_uncertainty_floor_voxel_factor) * float(voxel),
        )
        heldout_uncertainty_all = np.where(
            np.isfinite(position_uncertainty_all) & (position_uncertainty_all > 0),
            np.maximum(position_uncertainty_all, uncertainty_floor),
            uncertainty_floor,
        )
        heldout_error_ok_all = (
            heldout_available_all
            & np.isfinite(heldout_error_all)
            & np.isfinite(heldout_p90_all)
            & (
                heldout_error_all
                <= float(args.heldout_rescue_error_uncertainty_factor) * heldout_uncertainty_all
            )
            & (
                heldout_p90_all
                <= float(args.heldout_rescue_p90_uncertainty_factor) * heldout_uncertainty_all
            )
        )
        heldout_rescue_ok_all = (
            heldout_available_all
            & (~heldout_vote_ok_all)
            & heldout_error_ok_all
            & (independent_support_all >= int(args.heldout_rescue_min_independent_support))
            & (angular_span_all >= int(args.heldout_rescue_min_angular_span_poses))
            & (conflict_ratio_all <= float(args.heldout_rescue_max_conflict_pose_ratio))
        )
        heldout_compatible_all = (
            (~heldout_available_all) | heldout_vote_ok_all | heldout_rescue_ok_all
        )

        layer_contract_consistent_all = (
            (
                (~np.asarray(validated_layer_available_all, dtype=bool))
                | (surface_evidence_all < 3)
                | np.asarray(validated_layer_accept_all, dtype=bool)
            )
            if has_pose_layer_contract
            else np.ones(input_count, dtype=bool)
        )
        evidence_safe = (
            (surface_evidence_all >= int(args.minimum_input_evidence_class))
            & (independent_support_all >= int(args.minimum_input_independent_support))
            & (angular_span_all >= int(args.minimum_input_angular_span_poses))
            & (conflict_ratio_all <= float(args.maximum_input_conflict_pose_ratio))
            & heldout_compatible_all
            & layer_contract_consistent_all
        )
        # V4.6: además de la separación de hojas, se consume el contrato
        # regional multiescala por procedencia de poses de 11 V11.5.
        if has_local_coherence_contract:
            evidence_safe &= (surface_evidence_all >= 3) | local_coherence_accept_all
    else:
        # Compatibilidad con un paso 11 antiguo: no inventar evidencia ausente.
        heldout_vote_ok_all = np.ones(input_count, dtype=bool)
        heldout_rescue_ok_all = np.zeros(input_count, dtype=bool)
        heldout_compatible_all = np.ones(input_count, dtype=bool)
        heldout_uncertainty_all = np.full(input_count, np.nan, dtype=np.float64)
        layer_contract_consistent_all = np.ones(input_count, dtype=bool)
        evidence_safe = np.ones(input_count, dtype=bool)

    strong_input = evidence_safe & (
        (support_all >= int(args.always_preserve_support))
        | (confidence_all >= float(args.always_preserve_confidence))
    )

    # Un punto ambiguo ya no se salva únicamente por SOR/ROR, soporte o
    # confianza: primero debe superar el contrato de evidencia de 11.
    pre_keep = evidence_safe & (strong_input | (sor & ror))

    pre_points = points_all[pre_keep]
    pre_colors = colors_all[pre_keep]
    pre_support = support_all[pre_keep]
    pre_confidence = confidence_all[pre_keep]
    pre_selection_class = selection_class_all[pre_keep]
    pre_surface_evidence = surface_evidence_all[pre_keep]
    pre_independent_support = independent_support_all[pre_keep]
    pre_angular_span = angular_span_all[pre_keep]
    pre_conflict_ratio = conflict_ratio_all[pre_keep]
    pre_heldout_available = heldout_available_all[pre_keep]
    pre_heldout_tests = heldout_tests_all[pre_keep]
    pre_heldout_pass_count = heldout_pass_count_all[pre_keep]
    pre_heldout_pass = heldout_pass_all[pre_keep]
    pre_heldout_error = heldout_error_all[pre_keep]
    pre_heldout_p90 = heldout_p90_all[pre_keep]
    pre_position_uncertainty = position_uncertainty_all[pre_keep]
    pre_heldout_vote_ok = heldout_vote_ok_all[pre_keep]
    pre_heldout_rescue_ok = heldout_rescue_ok_all[pre_keep]
    pre_heldout_compatible = heldout_compatible_all[pre_keep]
    pre_support_pose_mask = support_pose_mask_all[pre_keep]
    pre_disagreement_pose_mask = disagreement_pose_mask_all[pre_keep]
    pre_validated_layer_available = validated_layer_available_all[pre_keep]
    pre_validated_layer_accept = validated_layer_accept_all[pre_keep]
    pre_validated_layer_count = validated_layer_count_all[pre_keep]
    pre_validated_layer_ambiguous = validated_layer_ambiguous_all[pre_keep]
    pre_validated_layer_resolved = validated_layer_resolved_all[pre_keep]
    pre_validated_layer_separation = validated_layer_separation_all[pre_keep]
    pre_validated_layer_shift = validated_layer_shift_all[pre_keep]
    pre_validated_layer_margin = validated_layer_margin_all[pre_keep]
    pre_validated_layer_selected_mask = validated_layer_selected_mask_all[pre_keep]
    pre_validated_layer_secondary_mask = validated_layer_secondary_mask_all[pre_keep]
    pre_local_coherence_available = local_coherence_available_all[pre_keep]
    pre_local_coherence_accept = local_coherence_accept_all[pre_keep]
    pre_local_coherence_strict = local_coherence_strict_all[pre_keep]
    pre_local_feature_like = local_feature_like_all[pre_keep]
    pre_local_coherence_score = local_coherence_score_all[pre_keep]
    pre_local_coherence_red_flags = local_coherence_red_flags_all[pre_keep]
    pre_local_plane_p90_small = local_plane_p90_small_all[pre_keep]
    pre_local_plane_p90_large = local_plane_p90_large_all[pre_keep]
    pre_local_normal_p90 = local_normal_p90_all[pre_keep]
    pre_local_scale_normal_diff = local_scale_normal_diff_all[pre_keep]
    pre_local_variation = local_variation_all[pre_keep]
    pre_local_same_sheet_fraction = local_same_sheet_fraction_all[pre_keep]
    pre_regional_pose_available = regional_pose_available_all[pre_keep]
    pre_regional_pose_accept = regional_pose_accept_all[pre_keep]
    pre_regional_pose_ambiguous = regional_pose_ambiguous_all[pre_keep]
    pre_regional_pose_feature_protected = regional_pose_feature_protected_all[pre_keep]
    pre_regional_pose_shift = regional_pose_shift_all[pre_keep]
    pre_regional_pose_score = regional_pose_score_all[pre_keep]
    pre_regional_pose_independent = regional_pose_independent_all[pre_keep]
    pre_regional_pose_dispersion = regional_pose_dispersion_all[pre_keep]
    pre_regional_pose_scale_difference = regional_pose_scale_difference_all[pre_keep]
    pre_raw_pose_bias_available = raw_pose_bias_available_all[pre_keep]
    pre_raw_pose_bias_accept = raw_pose_bias_accept_all[pre_keep]
    pre_raw_pose_bias_ambiguous = raw_pose_bias_ambiguous_all[pre_keep]
    pre_raw_pose_bias_feature_protected = raw_pose_bias_feature_protected_all[pre_keep]
    pre_raw_pose_bias_shift = raw_pose_bias_shift_all[pre_keep]
    pre_raw_pose_bias_score = raw_pose_bias_score_all[pre_keep]
    pre_raw_pose_bias_independent = raw_pose_bias_independent_all[pre_keep]
    pre_raw_pose_bias_scale_difference = raw_pose_bias_scale_difference_all[pre_keep]
    pre_raw_pose_bias_model_p90 = raw_pose_bias_model_p90_all[pre_keep]

    pre_filtered_count = len(pre_points)

    if pre_filtered_count < 500:
        raise RuntimeError("Muy pocos puntos después de SOR/ROR.")

    pre_spacing_info = nearest_spacing(pre_points)
    if pre_spacing_info is None or pre_spacing_info["median"] is None:
        raise RuntimeError("No se pudo estimar el spacing previo a la limpieza 3D.")

    pre_spacing = float(pre_spacing_info["median"])

    print(
        "[12] Iniciando limpieza tridimensional general: "
        f"{pre_filtered_count:,} puntos preseleccionados."
    )
    geometric = clean_geometric_components(
        pre_points,
        pre_colors,
        pre_support,
        pre_confidence,
        pre_selection_class,
        voxel=voxel,
        spacing=pre_spacing,
        args=args,
    )

    points = geometric["points"]
    colors = geometric["colors"]
    support = geometric["support"]
    confidence = geometric["confidence"]
    selection_class = geometric["selection_class"]
    local_neighbor_count = np.asarray(
        geometric["local_neighbor_count"],
        dtype=np.int32,
    )
    component_label = np.asarray(
        geometric["component_label"],
        dtype=np.int32,
    )
    geometric_keep_mask = np.asarray(geometric["keep_mask"], dtype=bool)
    surface_evidence = pre_surface_evidence[geometric_keep_mask]
    independent_support = pre_independent_support[geometric_keep_mask]
    angular_span = pre_angular_span[geometric_keep_mask]
    conflict_ratio = pre_conflict_ratio[geometric_keep_mask]
    heldout_available = pre_heldout_available[geometric_keep_mask]
    heldout_tests = pre_heldout_tests[geometric_keep_mask]
    heldout_pass_count = pre_heldout_pass_count[geometric_keep_mask]
    heldout_pass = pre_heldout_pass[geometric_keep_mask]
    heldout_error = pre_heldout_error[geometric_keep_mask]
    heldout_p90 = pre_heldout_p90[geometric_keep_mask]
    position_uncertainty = pre_position_uncertainty[geometric_keep_mask]
    heldout_vote_ok = pre_heldout_vote_ok[geometric_keep_mask]
    heldout_rescue_ok = pre_heldout_rescue_ok[geometric_keep_mask]
    heldout_compatible = pre_heldout_compatible[geometric_keep_mask]
    support_pose_mask = pre_support_pose_mask[geometric_keep_mask]
    disagreement_pose_mask = pre_disagreement_pose_mask[geometric_keep_mask]
    validated_layer_available = pre_validated_layer_available[geometric_keep_mask]
    validated_layer_accept = pre_validated_layer_accept[geometric_keep_mask]
    validated_layer_count = pre_validated_layer_count[geometric_keep_mask]
    validated_layer_ambiguous = pre_validated_layer_ambiguous[geometric_keep_mask]
    validated_layer_resolved = pre_validated_layer_resolved[geometric_keep_mask]
    validated_layer_separation = pre_validated_layer_separation[geometric_keep_mask]
    validated_layer_shift = pre_validated_layer_shift[geometric_keep_mask]
    validated_layer_margin = pre_validated_layer_margin[geometric_keep_mask]
    validated_layer_selected_mask = pre_validated_layer_selected_mask[geometric_keep_mask]
    validated_layer_secondary_mask = pre_validated_layer_secondary_mask[geometric_keep_mask]
    local_coherence_available = pre_local_coherence_available[geometric_keep_mask]
    local_coherence_accept = pre_local_coherence_accept[geometric_keep_mask]
    local_coherence_strict = pre_local_coherence_strict[geometric_keep_mask]
    local_feature_like = pre_local_feature_like[geometric_keep_mask]
    local_coherence_score = pre_local_coherence_score[geometric_keep_mask]
    local_coherence_red_flags = pre_local_coherence_red_flags[geometric_keep_mask]
    local_plane_p90_small = pre_local_plane_p90_small[geometric_keep_mask]
    local_plane_p90_large = pre_local_plane_p90_large[geometric_keep_mask]
    local_normal_p90 = pre_local_normal_p90[geometric_keep_mask]
    local_scale_normal_diff = pre_local_scale_normal_diff[geometric_keep_mask]
    local_variation = pre_local_variation[geometric_keep_mask]
    local_same_sheet_fraction = pre_local_same_sheet_fraction[geometric_keep_mask]
    regional_pose_available = pre_regional_pose_available[geometric_keep_mask]
    regional_pose_accept = pre_regional_pose_accept[geometric_keep_mask]
    regional_pose_ambiguous = pre_regional_pose_ambiguous[geometric_keep_mask]
    regional_pose_feature_protected = pre_regional_pose_feature_protected[geometric_keep_mask]
    regional_pose_shift = pre_regional_pose_shift[geometric_keep_mask]
    regional_pose_score = pre_regional_pose_score[geometric_keep_mask]
    regional_pose_independent = pre_regional_pose_independent[geometric_keep_mask]
    regional_pose_dispersion = pre_regional_pose_dispersion[geometric_keep_mask]
    regional_pose_scale_difference = pre_regional_pose_scale_difference[geometric_keep_mask]
    raw_pose_bias_available = pre_raw_pose_bias_available[geometric_keep_mask]
    raw_pose_bias_accept = pre_raw_pose_bias_accept[geometric_keep_mask]
    raw_pose_bias_ambiguous = pre_raw_pose_bias_ambiguous[geometric_keep_mask]
    raw_pose_bias_feature_protected = pre_raw_pose_bias_feature_protected[geometric_keep_mask]
    raw_pose_bias_shift = pre_raw_pose_bias_shift[geometric_keep_mask]
    raw_pose_bias_score = pre_raw_pose_bias_score[geometric_keep_mask]
    raw_pose_bias_independent = pre_raw_pose_bias_independent[geometric_keep_mask]
    raw_pose_bias_scale_difference = pre_raw_pose_bias_scale_difference[geometric_keep_mask]
    raw_pose_bias_model_p90 = pre_raw_pose_bias_model_p90[geometric_keep_mask]

    # Fuerza continua de evidencia: held-out no es un interruptor aislado.
    # El voto exacto, el error normalizado por incertidumbre, la diversidad
    # angular y el conflicto participan de forma conjunta.
    uncertainty_floor = max(1e-6, float(args.heldout_uncertainty_floor_voxel_factor) * float(voxel))
    uncertainty_ref = np.where(
        np.isfinite(position_uncertainty) & (position_uncertainty > 0),
        np.maximum(position_uncertainty, uncertainty_floor),
        uncertainty_floor,
    )
    pass_fraction = np.divide(
        heldout_pass_count.astype(np.float64),
        np.maximum(heldout_tests.astype(np.float64), 1.0),
        out=np.ones(len(points), dtype=np.float64),
        where=heldout_tests > 0,
    )
    median_error_score = np.where(
        heldout_available & np.isfinite(heldout_error),
        np.exp(-0.5 * (heldout_error / np.maximum(uncertainty_ref, 1e-9)) ** 2),
        0.85,
    )
    p90_error_score = np.where(
        heldout_available & np.isfinite(heldout_p90),
        np.exp(-0.5 * (heldout_p90 / np.maximum(1.5 * uncertainty_ref, 1e-9)) ** 2),
        0.85,
    )
    heldout_score = np.where(
        heldout_available,
        np.clip(
            0.55 * pass_fraction + 0.25 * median_error_score + 0.20 * p90_error_score,
            0.20,
            1.0,
        ),
        0.85,
    )
    diversity_score = np.minimum(
        np.clip(
            independent_support / max(float(args.minimum_input_independent_support), 1.0),
            0.50,
            1.0,
        ),
        np.clip(
            angular_span / max(float(args.minimum_input_angular_span_poses), 1.0),
            0.50,
            1.0,
        ),
    )
    conflict_score = np.clip(1.0 - conflict_ratio, 0.25, 1.0)
    class_score = np.where(
        surface_evidence >= 3,
        1.0,
        np.where(surface_evidence >= 2, 0.78, 0.20),
    )
    coherence_score_factor = np.where(
        surface_evidence >= 3,
        1.0,
        np.clip(local_coherence_score, 0.35, 1.0),
    )
    # Una característica local coherente (arista/esquina) no se penaliza por
    # tener varias familias de normales: V11.3 ya la validó explícitamente.
    coherence_score_factor = np.where(
        local_feature_like,
        np.maximum(coherence_score_factor, 0.85),
        coherence_score_factor,
    )
    layer_margin_clean = np.where(
        np.isfinite(validated_layer_margin), np.clip(validated_layer_margin, 0.0, 1.0), 1.0
    )
    layer_score_factor = np.ones(len(points), dtype=np.float64)
    layer_score_factor = np.where(
        validated_layer_resolved, 0.88 + 0.12 * layer_margin_clean, layer_score_factor
    )
    layer_score_factor = np.where(
        validated_layer_ambiguous, np.minimum(layer_score_factor, 0.72), layer_score_factor
    )
    regional_pose_factor = np.ones(len(points), dtype=np.float64)
    if has_regional_pose_consensus_contract:
        regional_pose_factor = np.where(
            regional_pose_available, np.clip(regional_pose_score, 0.45, 1.0), 1.0
        )
        regional_pose_factor = np.where(
            regional_pose_ambiguous, np.minimum(regional_pose_factor, 0.62), regional_pose_factor
        )
        # Una arista/esquina explícitamente protegida no debe perder autoridad
        # por no ser suavizable regionalmente.
        regional_pose_factor = np.where(
            regional_pose_feature_protected,
            np.maximum(regional_pose_factor, 0.90),
            regional_pose_factor,
        )
    raw_pose_bias_factor = np.ones(len(points), dtype=np.float64)
    if has_raw_pose_bias_contract:
        raw_pose_bias_factor = np.where(
            raw_pose_bias_available, np.clip(raw_pose_bias_score, 0.45, 1.0), 1.0
        )
        raw_pose_bias_factor = np.where(
            raw_pose_bias_ambiguous, np.minimum(raw_pose_bias_factor, 0.58), raw_pose_bias_factor
        )
        raw_pose_bias_factor = np.where(
            raw_pose_bias_feature_protected,
            np.maximum(raw_pose_bias_factor, 0.90),
            raw_pose_bias_factor,
        )
    evidence_strength = np.clip(
        class_score
        * heldout_score
        * diversity_score
        * conflict_score
        * coherence_score_factor
        * layer_score_factor
        * regional_pose_factor
        * raw_pose_bias_factor,
        0.10,
        1.0,
    )

    filtered_count = len(points)
    spacing_info = nearest_spacing(points)
    if spacing_info is None or spacing_info["median"] is None:
        raise RuntimeError("No se pudo estimar el spacing después de la limpieza 3D.")
    spacing = float(spacing_info["median"])

    print(
        "[12] Limpieza 3D completada: "
        f"{filtered_count:,}/{pre_filtered_count:,} puntos conservados; "
        f"{geometric['report'].get('components_retained', 0)} componentes retenidos."
    )

    # --------------------------------------------------------------
    # Normales recalculadas DESPUÉS de la limpieza geométrica.
    # La nube anterior no aporta normales para evitar propagar orientación
    # procedente de residuos eliminados.
    # --------------------------------------------------------------
    base_pcd = make_pcd(
        points,
        colors,
    )

    normal_radius_historical = float(
        args.normal_radius_mm
        if args.normal_radius_mm > 0
        else max(
            4.5,
            3.0 * voxel,
            3.2 * spacing,
        )
    )

    base_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius_historical,
            max_nn=int(args.normal_max_nn),
        )
    )
    base_pcd.normalize_normals()

    try:
        base_pcd.orient_normals_consistent_tangent_plane(int(args.normal_orientation_k))
    except Exception as exc:
        print(
            "[WARNING] Orientación inicial de normales no propagada:",
            exc,
        )

    initial_normals = np.asarray(
        base_pcd.normals,
        dtype=np.float64,
    )

    # --------------------------------------------------------------
    # Guardar la nube filtrada ANTES de suavizado.
    # --------------------------------------------------------------
    output = root / "reconstruccion" / "multisesion" / args.output_name
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    pre_npz = output / "nube_filtrada_sin_suavizado.npz"
    np.savez_compressed(
        pre_npz,
        points=points.astype(np.float32),
        colors=colors,
        normals=initial_normals.astype(np.float32),
        support_views=support,
        confidence=confidence.astype(np.float32),
        consensus_selection_class=selection_class,
        strong_core_support_threshold=np.asarray(
            [strong_core_support_threshold],
            dtype=np.uint16,
        ),
        fusion_voxel_mm=np.asarray(
            [voxel],
            dtype=np.float32,
        ),
        reference_spacing_mm=np.asarray(
            [spacing],
            dtype=np.float32,
        ),
        geometric_component_label=component_label,
        local_neighbor_count=local_neighbor_count,
        surface_evidence_class=surface_evidence,
        independent_support_poses=independent_support,
        support_angular_span_poses=angular_span.astype(np.int16),
        conflict_pose_ratio=conflict_ratio.astype(np.float32),
        heldout_validation_available=heldout_available.astype(np.uint8),
        heldout_validation_tests=heldout_tests.astype(np.int16),
        heldout_validation_pass_count=heldout_pass_count.astype(np.int16),
        heldout_validation_pass_ratio=heldout_pass.astype(np.float32),
        heldout_validation_error_mm=heldout_error.astype(np.float32),
        heldout_validation_p90_mm=heldout_p90.astype(np.float32),
        position_uncertainty_mm=position_uncertainty.astype(np.float32),
        heldout_vote_ok=heldout_vote_ok.astype(np.uint8),
        heldout_rescue_ok=heldout_rescue_ok.astype(np.uint8),
        heldout_compatible=heldout_compatible.astype(np.uint8),
        evidence_contract_accept=np.ones(len(points), dtype=np.uint8),
        support_pose_mask=support_pose_mask.astype(np.uint64),
        disagreement_pose_mask=disagreement_pose_mask.astype(np.uint64),
        validated_layer_available=validated_layer_available.astype(np.uint8),
        validated_layer_accept=validated_layer_accept.astype(np.uint8),
        validated_layer_count=validated_layer_count.astype(np.uint8),
        validated_layer_ambiguous=validated_layer_ambiguous.astype(np.uint8),
        validated_layer_resolved=validated_layer_resolved.astype(np.uint8),
        validated_layer_separation_mm=validated_layer_separation.astype(np.float32),
        validated_layer_shift_mm=validated_layer_shift.astype(np.float32),
        validated_layer_score_margin=validated_layer_margin.astype(np.float32),
        validated_layer_selected_pose_mask=validated_layer_selected_mask.astype(np.uint64),
        validated_layer_secondary_pose_mask=validated_layer_secondary_mask.astype(np.uint64),
        validated_pose_layer_contract_valid=np.asarray(
            [1 if has_pose_layer_contract else 0], dtype=np.uint8
        ),
        regional_pose_consensus_available=regional_pose_available.astype(np.uint8),
        regional_pose_consensus_accept=regional_pose_accept.astype(np.uint8),
        regional_pose_consensus_ambiguous=regional_pose_ambiguous.astype(np.uint8),
        regional_pose_consensus_feature_protected=regional_pose_feature_protected.astype(np.uint8),
        regional_pose_consensus_shift_mm=regional_pose_shift.astype(np.float32),
        regional_pose_consensus_score=regional_pose_score.astype(np.float32),
        regional_pose_consensus_independent_predictions=regional_pose_independent.astype(np.int16),
        regional_pose_consensus_pose_dispersion_mm=regional_pose_dispersion.astype(np.float32),
        regional_pose_consensus_scale_difference_mm=regional_pose_scale_difference.astype(
            np.float32
        ),
        regional_pose_consensus_contract_valid=np.asarray(
            [1 if has_regional_pose_consensus_contract else 0], dtype=np.uint8
        ),
        raw_pose_bias_available=raw_pose_bias_available.astype(np.uint8),
        raw_pose_bias_accept=raw_pose_bias_accept.astype(np.uint8),
        raw_pose_bias_ambiguous=raw_pose_bias_ambiguous.astype(np.uint8),
        raw_pose_bias_feature_protected=raw_pose_bias_feature_protected.astype(np.uint8),
        raw_pose_bias_shift_mm=raw_pose_bias_shift.astype(np.float32),
        raw_pose_bias_score=raw_pose_bias_score.astype(np.float32),
        raw_pose_bias_independent_poses=raw_pose_bias_independent.astype(np.int16),
        raw_pose_bias_scale_difference_mm=raw_pose_bias_scale_difference.astype(np.float32),
        raw_pose_bias_model_p90_mm=raw_pose_bias_model_p90.astype(np.float32),
        raw_pose_bias_contract_valid=np.asarray(
            [1 if has_raw_pose_bias_contract else 0], dtype=np.uint8
        ),
        evidence_strength=evidence_strength.astype(np.float32),
        surface_local_coherence_available=local_coherence_available.astype(np.uint8),
        surface_local_coherence_accept=local_coherence_accept.astype(np.uint8),
        surface_local_coherence_strict=local_coherence_strict.astype(np.uint8),
        surface_local_feature_like=local_feature_like.astype(np.uint8),
        surface_local_coherence_score=local_coherence_score.astype(np.float32),
        surface_local_coherence_red_flags=local_coherence_red_flags.astype(np.uint8),
        surface_local_plane_p90_small_mm=local_plane_p90_small.astype(np.float32),
        surface_local_plane_p90_large_mm=local_plane_p90_large.astype(np.float32),
        surface_local_normal_p90_large_deg=local_normal_p90.astype(np.float32),
        surface_local_scale_normal_difference_deg=local_scale_normal_diff.astype(np.float32),
        surface_local_variation_large=local_variation.astype(np.float32),
        surface_local_same_sheet_fraction=local_same_sheet_fraction.astype(np.float32),
        local_coherence_contract_valid=np.asarray(
            [1 if has_local_coherence_contract else 0], dtype=np.uint8
        ),
        evidence_contract_valid=np.asarray([1 if has_evidence_contract else 0], dtype=np.uint8),
    )

    pre_ply = output / "nube_filtrada_sin_suavizado.ply"
    save_ply(
        pre_ply,
        make_pcd(
            points,
            colors,
            initial_normals,
        ),
    )

    extent_before = robust_extent(points)

    roughness_radius = max(
        3.0 * spacing,
        2.0 * voxel,
    )
    roughness_before = evaluate_local_roughness(
        points,
        k=int(args.roughness_evaluation_knn),
        radius=roughness_radius,
        normals=initial_normals,
        maximum_points=int(args.maximum_diagnostic_points),
    )

    # --------------------------------------------------------------
    # Regularización superficial general.
    # --------------------------------------------------------------
    if bool(args.surface_regularization):
        result = regularize_surface(
            points,
            support.astype(np.float64),
            confidence,
            spacing=spacing,
            voxel=voxel,
            initial_normals=initial_normals,
            support_pose_mask=support_pose_mask,
            disagreement_pose_mask=disagreement_pose_mask,
            evidence_strength=evidence_strength,
            args=args,
        )
        regularized_points = result["points"]
        local_normals = result["normals"]
        displacement = result["total_displacement_mm"]
    else:
        result = {
            "points": points.copy(),
            "normals": initial_normals.copy(),
            "curvature": np.zeros(
                len(points),
                dtype=float,
            ),
            "normal_neighbor_count": np.zeros(
                len(points),
                dtype=int,
            ),
            "total_displacement_mm": np.zeros(
                len(points),
                dtype=float,
            ),
            "smooth_radius_mm": None,
            "normal_pca_radius_mm": None,
            "aggregate": {
                "feature_factor_min": np.ones(len(points)),
                "boundary_factor_min": np.ones(len(points)),
                "boundary_asymmetry_max": np.zeros(len(points)),
                "normal_dispersion_max_deg": np.zeros(len(points)),
                "step_clipped_any": np.zeros(
                    len(points),
                    dtype=bool,
                ),
                "total_clipped_any": np.zeros(
                    len(points),
                    dtype=bool,
                ),
                "pose_compatibility_min": np.ones(
                    len(points),
                    dtype=np.float64,
                ),
            },
            "iterations": [],
        }
        regularized_points = points.copy()
        local_normals = initial_normals.copy()
        displacement = np.zeros(
            len(points),
            dtype=float,
        )

    # --------------------------------------------------------------
    # V2.1 — Guarda de escala de muestreo.
    #
    # La V2.0 podía reducir mucho el spacing NN al colapsar ruido/capas
    # cercanas. 13 interpretaba entonces la nube como mucho más densa
    # y reducía en exceso sus radios de Ball Pivoting.
    # --------------------------------------------------------------
    (
        regularized_points,
        spacing_guard,
    ) = apply_sampling_scale_guard(
        points,
        regularized_points,
        reference_spacing_mm=spacing,
        minimum_ratio=float(args.minimum_spacing_ratio_after_smoothing),
        bisection_steps=int(args.spacing_guard_bisection_steps),
        sample_points=int(args.spacing_guard_sample_points),
    )

    # Desplazamiento REAL después de la guarda.
    displacement = np.linalg.norm(
        regularized_points - points,
        axis=1,
    )
    result["points"] = regularized_points
    result["total_displacement_mm"] = displacement

    # Reestima normales PCA tras la mezcla adaptativa.
    guarded_normal_radius = (
        float(result["normal_pca_radius_mm"])
        if result["normal_pca_radius_mm"] is not None
        else max(
            float(args.normal_pca_radius_factor) * spacing,
            1.20 * roughness_radius,
        )
    )
    local_normals, guarded_curvature, guarded_normal_counts = estimate_pca_normals(
        regularized_points,
        k=int(args.normal_pca_knn),
        radius=guarded_normal_radius,
        reference_normals=local_normals,
    )
    result["normals"] = local_normals
    result["curvature"] = guarded_curvature
    result["normal_neighbor_count"] = guarded_normal_counts

    # --------------------------------------------------------------
    # Normales finales para 13.
    # --------------------------------------------------------------
    final_pcd, final_normals, orientation_ok = orient_final_normals(
        regularized_points,
        colors,
        local_normals,
        normal_radius=normal_radius_historical,
        max_nn=int(args.normal_max_nn),
        orientation_k=int(args.normal_orientation_k),
    )

    roughness_after = evaluate_local_roughness(
        regularized_points,
        k=int(args.roughness_evaluation_knn),
        radius=roughness_radius,
        normals=final_normals,
        maximum_points=int(args.maximum_diagnostic_points),
    )

    extent_after = robust_extent(regularized_points)

    # --------------------------------------------------------------
    # Sanidad dimensional.
    # --------------------------------------------------------------
    before_extent = np.asarray(
        extent_before["extent"],
        dtype=float,
    )
    after_extent = np.asarray(
        extent_after["extent"],
        dtype=float,
    )

    extent_relative_change = np.divide(
        after_extent - before_extent,
        before_extent,
        out=np.zeros(
            3,
            dtype=float,
        ),
        where=before_extent > 1e-9,
    )

    warnings = []

    if geometric["report"].get("fallback"):
        warnings.append(
            "La agrupación geométrica no produjo componentes y se aplicó "
            "la salida de respaldo espacial como alternativa segura."
        )

    cleanup_retention_ratio = float(filtered_count / max(pre_filtered_count, 1))
    if cleanup_retention_ratio < 0.55:
        warnings.append(
            "La limpieza 3D conservó menos del 55% de los puntos "
            "preseleccionados; conviene revisar el resumen de componentes."
        )

    if np.max(np.abs(extent_relative_change)) > float(args.warning_extent_change_ratio):
        warnings.append(
            "La regularización cambió un extent robusto más de "
            f"{100.0*float(args.warning_extent_change_ratio):.1f}%."
        )

    total_cap_mm = float(args.max_total_shift_spacing) * spacing

    if (
        len(displacement)
        and np.percentile(
            displacement,
            95,
        )
        > 0.90 * total_cap_mm
    ):
        warnings.append("El P95 de desplazamiento está cerca del límite total permitido.")

    quality = "accepted" if not warnings else "warning"

    # --------------------------------------------------------------
    # Salidas compatibles con 13.
    # --------------------------------------------------------------
    out_npz = output / "nube_regularizada_general.npz"
    np.savez_compressed(
        out_npz,
        points=regularized_points.astype(np.float32),
        colors=colors,
        normals=final_normals.astype(np.float32),
        support_views=support,
        confidence=confidence.astype(np.float32),
        consensus_selection_class=selection_class,
        strong_core_support_threshold=np.asarray(
            [strong_core_support_threshold],
            dtype=np.uint16,
        ),
        fusion_voxel_mm=np.asarray(
            [voxel],
            dtype=np.float32,
        ),
        reference_spacing_mm=np.asarray(
            [spacing],
            dtype=np.float32,
        ),
        meshing_reference_spacing_mm=np.asarray(
            [spacing],
            dtype=np.float32,
        ),
        smoothing_displacement_mm=displacement.astype(np.float32),
        geometric_component_label=component_label,
        local_neighbor_count=local_neighbor_count,
        surface_evidence_class=surface_evidence,
        independent_support_poses=independent_support,
        support_angular_span_poses=angular_span.astype(np.int16),
        conflict_pose_ratio=conflict_ratio.astype(np.float32),
        heldout_validation_available=heldout_available.astype(np.uint8),
        heldout_validation_tests=heldout_tests.astype(np.int16),
        heldout_validation_pass_count=heldout_pass_count.astype(np.int16),
        heldout_validation_pass_ratio=heldout_pass.astype(np.float32),
        heldout_validation_error_mm=heldout_error.astype(np.float32),
        heldout_validation_p90_mm=heldout_p90.astype(np.float32),
        position_uncertainty_mm=position_uncertainty.astype(np.float32),
        heldout_vote_ok=heldout_vote_ok.astype(np.uint8),
        heldout_rescue_ok=heldout_rescue_ok.astype(np.uint8),
        heldout_compatible=heldout_compatible.astype(np.uint8),
        evidence_contract_accept=np.ones(len(points), dtype=np.uint8),
        support_pose_mask=support_pose_mask.astype(np.uint64),
        disagreement_pose_mask=disagreement_pose_mask.astype(np.uint64),
        regional_pose_consensus_available=regional_pose_available.astype(np.uint8),
        regional_pose_consensus_accept=regional_pose_accept.astype(np.uint8),
        regional_pose_consensus_ambiguous=regional_pose_ambiguous.astype(np.uint8),
        regional_pose_consensus_feature_protected=regional_pose_feature_protected.astype(np.uint8),
        regional_pose_consensus_shift_mm=regional_pose_shift.astype(np.float32),
        regional_pose_consensus_score=regional_pose_score.astype(np.float32),
        regional_pose_consensus_independent_predictions=regional_pose_independent.astype(np.int16),
        regional_pose_consensus_pose_dispersion_mm=regional_pose_dispersion.astype(np.float32),
        regional_pose_consensus_scale_difference_mm=regional_pose_scale_difference.astype(
            np.float32
        ),
        regional_pose_consensus_contract_valid=np.asarray(
            [1 if has_regional_pose_consensus_contract else 0], dtype=np.uint8
        ),
        raw_pose_bias_available=raw_pose_bias_available.astype(np.uint8),
        raw_pose_bias_accept=raw_pose_bias_accept.astype(np.uint8),
        raw_pose_bias_ambiguous=raw_pose_bias_ambiguous.astype(np.uint8),
        raw_pose_bias_feature_protected=raw_pose_bias_feature_protected.astype(np.uint8),
        raw_pose_bias_shift_mm=raw_pose_bias_shift.astype(np.float32),
        raw_pose_bias_score=raw_pose_bias_score.astype(np.float32),
        raw_pose_bias_independent_poses=raw_pose_bias_independent.astype(np.int16),
        raw_pose_bias_scale_difference_mm=raw_pose_bias_scale_difference.astype(np.float32),
        raw_pose_bias_model_p90_mm=raw_pose_bias_model_p90.astype(np.float32),
        raw_pose_bias_contract_valid=np.asarray(
            [1 if has_raw_pose_bias_contract else 0], dtype=np.uint8
        ),
        evidence_strength=evidence_strength.astype(np.float32),
        surface_local_coherence_available=local_coherence_available.astype(np.uint8),
        surface_local_coherence_accept=local_coherence_accept.astype(np.uint8),
        surface_local_coherence_strict=local_coherence_strict.astype(np.uint8),
        surface_local_feature_like=local_feature_like.astype(np.uint8),
        surface_local_coherence_score=local_coherence_score.astype(np.float32),
        surface_local_coherence_red_flags=local_coherence_red_flags.astype(np.uint8),
        surface_local_plane_p90_small_mm=local_plane_p90_small.astype(np.float32),
        surface_local_plane_p90_large_mm=local_plane_p90_large.astype(np.float32),
        surface_local_normal_p90_large_deg=local_normal_p90.astype(np.float32),
        surface_local_scale_normal_difference_deg=local_scale_normal_diff.astype(np.float32),
        surface_local_variation_large=local_variation.astype(np.float32),
        surface_local_same_sheet_fraction=local_same_sheet_fraction.astype(np.float32),
        local_coherence_contract_valid=np.asarray(
            [1 if has_local_coherence_contract else 0], dtype=np.uint8
        ),
        evidence_contract_valid=np.asarray([1 if has_evidence_contract else 0], dtype=np.uint8),
    )

    out_ply = output / "nube_regularizada_general.ply"
    save_ply(
        out_ply,
        final_pcd,
    )

    displacement_path = output / "desplazamiento_regularizacion_mm.npy"
    np.save(
        displacement_path,
        displacement.astype(np.float32),
    )

    aggregate = result["aggregate"]
    diagnostic_path = output / "diagnostico_regularizacion_superficial.npz"
    np.savez_compressed(
        diagnostic_path,
        displacement_mm=displacement.astype(np.float32),
        feature_factor_min=np.asarray(
            aggregate["feature_factor_min"],
            dtype=np.float32,
        ),
        boundary_factor_min=np.asarray(
            aggregate["boundary_factor_min"],
            dtype=np.float32,
        ),
        boundary_asymmetry_max=np.asarray(
            aggregate["boundary_asymmetry_max"],
            dtype=np.float32,
        ),
        normal_dispersion_max_deg=np.asarray(
            aggregate["normal_dispersion_max_deg"],
            dtype=np.float32,
        ),
        curvature=np.asarray(
            result["curvature"],
            dtype=np.float32,
        ),
        normal_neighbor_count=np.asarray(
            result["normal_neighbor_count"],
            dtype=np.int16,
        ),
        geometric_component_label=component_label,
        local_neighbor_count=local_neighbor_count,
        pose_compatibility_min=np.asarray(aggregate["pose_compatibility_min"], dtype=np.float32),
        surface_evidence_class=surface_evidence,
        independent_support_poses=independent_support,
        support_angular_span_poses=angular_span.astype(np.int16),
        conflict_pose_ratio=conflict_ratio.astype(np.float32),
        heldout_validation_available=heldout_available.astype(np.uint8),
        heldout_validation_tests=heldout_tests.astype(np.int16),
        heldout_validation_pass_count=heldout_pass_count.astype(np.int16),
        heldout_validation_pass_ratio=heldout_pass.astype(np.float32),
        heldout_validation_error_mm=heldout_error.astype(np.float32),
        heldout_validation_p90_mm=heldout_p90.astype(np.float32),
        position_uncertainty_mm=position_uncertainty.astype(np.float32),
        heldout_vote_ok=heldout_vote_ok.astype(np.uint8),
        heldout_rescue_ok=heldout_rescue_ok.astype(np.uint8),
        heldout_compatible=heldout_compatible.astype(np.uint8),
        evidence_contract_accept=np.ones(len(points), dtype=np.uint8),
        regional_pose_consensus_available=regional_pose_available.astype(np.uint8),
        regional_pose_consensus_accept=regional_pose_accept.astype(np.uint8),
        regional_pose_consensus_ambiguous=regional_pose_ambiguous.astype(np.uint8),
        regional_pose_consensus_feature_protected=regional_pose_feature_protected.astype(np.uint8),
        regional_pose_consensus_shift_mm=regional_pose_shift.astype(np.float32),
        regional_pose_consensus_score=regional_pose_score.astype(np.float32),
        regional_pose_consensus_independent_predictions=regional_pose_independent.astype(np.int16),
        regional_pose_consensus_pose_dispersion_mm=regional_pose_dispersion.astype(np.float32),
        regional_pose_consensus_scale_difference_mm=regional_pose_scale_difference.astype(
            np.float32
        ),
        regional_pose_consensus_contract_valid=np.asarray(
            [1 if has_regional_pose_consensus_contract else 0], dtype=np.uint8
        ),
        raw_pose_bias_available=raw_pose_bias_available.astype(np.uint8),
        raw_pose_bias_accept=raw_pose_bias_accept.astype(np.uint8),
        raw_pose_bias_ambiguous=raw_pose_bias_ambiguous.astype(np.uint8),
        raw_pose_bias_feature_protected=raw_pose_bias_feature_protected.astype(np.uint8),
        raw_pose_bias_shift_mm=raw_pose_bias_shift.astype(np.float32),
        raw_pose_bias_score=raw_pose_bias_score.astype(np.float32),
        raw_pose_bias_independent_poses=raw_pose_bias_independent.astype(np.int16),
        raw_pose_bias_scale_difference_mm=raw_pose_bias_scale_difference.astype(np.float32),
        raw_pose_bias_model_p90_mm=raw_pose_bias_model_p90.astype(np.float32),
        raw_pose_bias_contract_valid=np.asarray(
            [1 if has_raw_pose_bias_contract else 0], dtype=np.uint8
        ),
        evidence_strength=evidence_strength.astype(np.float32),
        evidence_contract_valid=np.asarray([1 if has_evidence_contract else 0], dtype=np.uint8),
    )

    prev = output / "preview_nube_regularizada_general.png"
    preview(
        prev,
        regularized_points,
        support,
        confidence,
        "12 V4.6 limpieza 3D + regularización superficial",
    )

    comparison_path = output / "preview_comparacion_regularizacion.png"
    comparison_preview(
        comparison_path,
        points,
        regularized_points,
        displacement,
    )

    final_spacing = nearest_spacing(regularized_points)

    rough_before_med = roughness_before["roughness_mm"]["median"]
    rough_after_med = roughness_after["roughness_mm"]["median"]

    if rough_before_med is not None and rough_after_med is not None and rough_before_med > 1e-12:
        roughness_reduction = 1.0 - rough_after_med / rough_before_med
    else:
        roughness_reduction = None

    summary = {
        "schema_version": 4,
        "quality": quality,
        "warning_reasons": warnings,
        "method": (
            "confidence_preserving_sor_ror_"
            "spatial_backing_multicomponent_multiview_cleanup_"
            "edge_aware_bilateral_point_to_surface_regularization"
        ),
        "shape_specific_assumptions": False,
        "evidence_contract_valid": bool(has_evidence_contract),
        "local_coherence_contract_valid": bool(has_local_coherence_contract),
        "regional_pose_consensus_contract_valid": bool(has_regional_pose_consensus_contract),
        "raw_pose_bias_contract_valid": bool(has_raw_pose_bias_contract),
        "evidence_contract_source": (
            "step_11_pose_diverse_heldout_local_coherence_contract"
            if has_evidence_contract
            else "legacy_or_missing_step_11_contract"
        ),
        "surface_regularization_enabled": bool(args.surface_regularization),
        "evidence_filter": {
            "policy": (
                "integer_heldout_vote_plus_uncertainty_normalized_rescue_"
                "with_pose_diversity_and_conflict_guard"
            ),
            "heldout_required_fraction_exact": (
                f"{int(args.minimum_input_heldout_pass_numerator)}/"
                f"{int(args.minimum_input_heldout_pass_denominator)}"
            ),
            "base_contract_candidates": (
                int(
                    np.count_nonzero(
                        (surface_evidence_all >= int(args.minimum_input_evidence_class))
                        & (independent_support_all >= int(args.minimum_input_independent_support))
                        & (angular_span_all >= int(args.minimum_input_angular_span_poses))
                        & (conflict_ratio_all <= float(args.maximum_input_conflict_pose_ratio))
                    )
                )
                if has_evidence_contract
                else int(input_count)
            ),
            "heldout_vote_pass": (
                int(np.count_nonzero(heldout_vote_ok_all))
                if has_evidence_contract
                else int(input_count)
            ),
            "heldout_rescued_by_error_diversity_conflict": (
                int(np.count_nonzero(heldout_rescue_ok_all)) if has_evidence_contract else 0
            ),
            "local_coherence_contract_available": bool(has_local_coherence_contract),
            "regional_pose_consensus_contract_available": bool(
                has_regional_pose_consensus_contract
            ),
            "regional_pose_consensus_available": (
                int(np.count_nonzero(regional_pose_available_all))
                if has_regional_pose_consensus_contract
                else 0
            ),
            "regional_pose_consensus_ambiguous": (
                int(np.count_nonzero(regional_pose_ambiguous_all))
                if has_regional_pose_consensus_contract
                else 0
            ),
            "regional_pose_consensus_corrected": (
                int(np.count_nonzero(np.abs(regional_pose_shift_all) > 1e-12))
                if has_regional_pose_consensus_contract
                else 0
            ),
            "raw_pose_bias_contract_available": bool(has_raw_pose_bias_contract),
            "raw_pose_bias_available": (
                int(np.count_nonzero(raw_pose_bias_available_all))
                if has_raw_pose_bias_contract
                else 0
            ),
            "raw_pose_bias_ambiguous": (
                int(np.count_nonzero(raw_pose_bias_ambiguous_all))
                if has_raw_pose_bias_contract
                else 0
            ),
            "raw_pose_bias_corrected": (
                int(np.count_nonzero(np.abs(raw_pose_bias_shift_all) > 1e-12))
                if has_raw_pose_bias_contract
                else 0
            ),
            "class2_local_coherence_available": int(
                np.count_nonzero((surface_evidence_all == 2) & (local_coherence_available_all > 0))
            ),
            "class2_local_coherence_rejected": (
                int(np.count_nonzero((surface_evidence_all == 2) & (~local_coherence_accept_all)))
                if has_local_coherence_contract
                else 0
            ),
            "class2_local_feature_like": (
                int(np.count_nonzero((surface_evidence_all == 2) & local_feature_like_all))
                if has_local_coherence_contract
                else 0
            ),
            "contract_accepted_before_sor_ror": int(np.count_nonzero(evidence_safe)),
            "contract_rejected_before_sor_ror": int(input_count - np.count_nonzero(evidence_safe)),
            "note": (
                "2/3 se evalúa con aritmética entera. Un voto held-out insuficiente "
                "solo puede rescatarse si el error es compatible con la incertidumbre local. "
                "Las observaciones clase 2 publicadas por 11 V11.3 además incorporan coherencia "
                "superficial multiescala, preservando aristas/esquinas coherentes sin imponer forma."
            ),
        },
        "parameters": vars(args),
        "input_points": int(input_count),
        "points_after_sor_ror_preselection": int(pre_filtered_count),
        "filtered_points_before_smoothing": int(filtered_count),
        "final_points": int(len(regularized_points)),
        "removed_points": int(input_count - len(regularized_points)),
        "strong_points_in_input": int(np.count_nonzero(strong_input)),
        "strong_points_preserved": int(
            np.count_nonzero(
                (support >= int(args.always_preserve_support))
                | (confidence >= float(args.always_preserve_confidence))
            )
        ),
        "strong_core_support_threshold": int(strong_core_support_threshold),
        "hierarchical_consensus_classes": {
            "strong_anchor": int(np.count_nonzero(selection_class == 3)),
            "local_extension": int(np.count_nonzero(selection_class == 2)),
            "legacy_singleton": int(np.count_nonzero(selection_class == 1)),
        },
        "sor_inlier_ratio": float(np.mean(sor)),
        "ror_inlier_ratio": float(np.mean(ror)),
        "fusion_voxel_mm": float(voxel),
        "nearest_neighbor_spacing_pre_cleanup_mm": pre_spacing_info,
        "nearest_neighbor_spacing_before_mm": spacing_info,
        "nearest_neighbor_spacing_after_mm": final_spacing,
        "geometric_cleanup": geometric["report"],
        "sampling_scale_guard": spacing_guard,
        "smoothing": {
            "radius_mm": result["smooth_radius_mm"],
            "normal_pca_radius_mm": result["normal_pca_radius_mm"],
            "iterations": result["iterations"],
            "total_displacement_mm": stats(displacement),
            "maximum_total_displacement_allowed_mm": float(total_cap_mm),
            "points_step_clipped_at_least_once": int(
                np.count_nonzero(aggregate["step_clipped_any"])
            ),
            "points_total_clipped_at_least_once": int(
                np.count_nonzero(aggregate["total_clipped_any"])
            ),
            "feature_factor_min": stats(aggregate["feature_factor_min"]),
            "boundary_factor_min": stats(aggregate["boundary_factor_min"]),
            "boundary_asymmetry_max": stats(aggregate["boundary_asymmetry_max"]),
            "normal_dispersion_max_deg": stats(aggregate["normal_dispersion_max_deg"]),
        },
        "roughness": {
            "before": roughness_before,
            "after": roughness_after,
            "median_reduction_ratio": (
                float(roughness_reduction) if roughness_reduction is not None else None
            ),
        },
        "dimension_preservation": {
            "robust_extent_before": extent_before,
            "robust_extent_after": extent_after,
            "relative_change_xyz": extent_relative_change.astype(float).tolist(),
            "maximum_absolute_relative_change": float(np.max(np.abs(extent_relative_change))),
        },
        "normal_orientation_propagation_ok": bool(orientation_ok),
        "support_views": stats(support),
        "confidence": stats(confidence),
        "outputs": {
            "filtered_before_smoothing_npz": str(pre_npz),
            "filtered_before_smoothing_ply": str(pre_ply),
            "npz": str(out_npz),
            "ply": str(out_ply),
            "displacement": str(displacement_path),
            "regularization_diagnostics": str(diagnostic_path),
            "preview": str(prev),
            "comparison_preview": str(comparison_path),
        },
        "important_note": (
            "La V3.0 elimina residuos mediante respaldo espacial, "
            "componentes y evidencia multivista, permitiendo conservar "
            "varias partes legítimas. Después recalcula las normales y mueve "
            "puntos solo localmente, principalmente sobre la normal estimada. "
            "No ajusta planos, primitivas ni geometría específica."
        ),
    }

    summary_path = output / "resumen_12_regularizacion_nube.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n========== PASO 12 V4.6 COMPLETADO ==========")
    print(f"Entrada 11: {input_count:,}")
    print(f"Tras SOR/ROR: {pre_filtered_count:,}")
    print(f"Tras limpieza 3D: {filtered_count:,}")
    print(
        "Componentes: "
        f"{geometric['report'].get('components_retained', 0)} retenidos / "
        f"{geometric['report'].get('components_total', 0)} detectados"
    )
    print(f"Final: {len(regularized_points):,}")
    print(f"Spacing: {spacing:.4f} mm")
    print(
        "Desplazamiento:",
        stats(displacement),
    )
    print(
        "Spacing guard:",
        spacing_guard,
    )
    print(
        "Rugosidad antes:",
        roughness_before["roughness_mm"],
    )
    print(
        "Rugosidad después:",
        roughness_after["roughness_mm"],
    )
    print(
        "Cambio extent XYZ:",
        extent_relative_change,
    )
    print(
        "Calidad 12:",
        quality,
    )
    print(
        "Salida:",
        output,
    )
    print("================================================\n")

    return 0


def _entrada_con_diagnostico():
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    from utilidades_progreso import informar_inicio

    informar_inicio(__file__, "12", "Regularizar nube de puntos")
    raise SystemExit(main())


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
