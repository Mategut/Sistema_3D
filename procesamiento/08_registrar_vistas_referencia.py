#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paso 08 — Calibración del registro con un objeto de referencia cuboide.

Este paso pertenece a la calibración de la plataforma. Utiliza planos del
patrón y coherencia Manhattan; la reconstrucción de objetos arbitrarios usa
después el registro calibrado del paso 10.

Modelo cinemático: un eje 3D, una línea de rotación, un sentido de giro y
25 ángulos físicos derivados de 2055 pasos/vuelta. Las sesiones se fusionan
antes del registro. No se estima un movimiento ICP libre de seis grados.

Las poses se estiman con planos dominantes ponderados por confianza. La
transformación resultante se aplica a la nube completa del paso 06, incluidos
los puntos que no participaron en la estimación de pose.

Procedimiento
-------------
1. Verificar las poses y sus ángulos contra el manifiesto angular.
2. Seleccionar planos útiles y estimar el eje por consenso entre normales.
3. Si falta consenso, evaluar semillas adicionales con las dos direcciones
   de giro y un prior físico suave de perpendicularidad eje/baseline.
4. Estimar centro y sentido mediante solape simétrico recortado.
5. Ajustar eje, centro y pequeñas correcciones angulares con bundle adjustment
   restringido y correspondencias entre vistas vecinas.
6. Evaluar cierre, grafo, error punto-plano, solape, montaje y compactación.
7. Exportar nubes registradas, poses y evidencia de calibración.

La observabilidad de cada pose determina sus límites y su prior angular:
una cara dominante ofrece menos libertad que varias caras informativas.
Las correcciones tienen regularización mecánica y continuidad circular.
La saturación de un límite se audita junto con las métricas globales.
Estas correcciones residuales son evidencia de calibración; el paso 09
conserva los ángulos mecánicos para reconstruir otros objetos.
"""

from __future__ import annotations
from utilidades_progreso import operacion
from utilidades_rendimiento import configurar as _configurar_recursos

# Los runtimes numéricos leen estos límites al importarse.
_configurar_recursos(__file__)


import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from scipy.optimize import differential_evolution, least_squares, minimize, minimize_scalar
    from scipy.spatial import cKDTree
except Exception as exc:
    raise SystemExit("Paso 08 V3.2 requiere SciPy (optimize + spatial).\n" f"Detalle: {exc}")

try:
    from utilidades_mascaras import save_ply_ascii
except Exception as exc:
    raise SystemExit(
        "No se pudo importar utilidades_mascaras.py. Colócalo junto al script.\n" f"Detalle: {exc}"
    )


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


@dataclass
class PlaneDecision:
    """Decisión y evidencia asociadas a un plano candidato para el registro."""

    index: int
    status: str
    confidence: float
    ratio: float
    rmse_mm: float
    normal_median_deg: Optional[float]
    normal_p95_deg: Optional[float]
    reasons: List[str] = field(default_factory=list)


@dataclass
class ViewData:
    """Datos geométricos y metadatos de una vista de calibración."""

    index: int
    pose_index: int
    stem: str
    angle_deg: float
    quality_step_07: str
    view_weight: float
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    registration_points: np.ndarray
    registration_normals: np.ndarray
    registration_weights: np.ndarray
    planes: List[dict]
    plane_decisions: List[PlaneDecision]
    reasons: List[str]


@dataclass
class AngularConstraint:
    """Límites y observabilidad de la corrección angular de una pose."""

    pose_index: int
    observability_score: float
    observability_class: str
    informative_plane_count: int
    distinct_lateral_family_count: int
    effective_lateral_support: float
    diversity_score: float
    normal_prior_raw_deg: float
    normal_prior_scatter_deg: Optional[float]
    normal_prior_reliability: float
    target_correction_deg: float
    sigma_deg: float
    limit_deg: float
    prior_weight: float
    plane_records: List[dict] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


@dataclass
class EdgeSpec:
    """Relación entre vistas utilizada como arista del grafo de registro."""

    source_index: int
    target_index: int
    order: int
    nominal_delta_deg: float
    primary: bool
    closure: bool
    base_weight: float
    history_weight: float = 1.0


@dataclass
class RegistrationSample:
    """Muestra de puntos y atributos para estimar el registro."""

    points: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    tree: cKDTree


@dataclass
class CorrespondenceBatch:
    """Lote de correspondencias geométricas para la optimización."""

    source_index: int
    target_index: int
    src_points: np.ndarray
    tgt_points: np.ndarray
    tgt_normals: np.ndarray
    weights: np.ndarray
    edge_weight: float
    direction: str


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construye las opciones de línea de comandos de este paso."""
    p = argparse.ArgumentParser(
        description=(
            "Paso 08 V3.2: registro multivista con eje/centro únicos, "
            "ángulos físicos y BA punto-a-plano restringido."
        )
    )
    p.add_argument("--root", required=True)
    p.add_argument("--object", default="cubo")
    p.add_argument("--session", default="multisesion")
    p.add_argument("--cloud-source", default="06_nubes_puntos")
    p.add_argument("--geometry-source", default="07_validacion_geometrica")
    p.add_argument("--output-name", default="08_registro_referencia")
    p.add_argument("--manifest", default="")
    p.add_argument("--clean-output", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=500)

    # Validación de campaña.
    p.add_argument("--expected-views", type=int, default=25)
    p.add_argument("--expected-steps-per-revolution", type=int, default=2055)
    p.add_argument("--angle-validation-tolerance-deg", type=float, default=0.02)

    # Selección de planos/puntos de pose.
    p.add_argument("--minimum-registration-points", type=int, default=3500)
    p.add_argument("--pose-plane-minimum-ratio", type=float, default=0.055)
    p.add_argument("--pose-plane-hard-rmse-mm", type=float, default=3.0)
    p.add_argument("--oblique-minor-ratio-to-major", type=float, default=0.58)
    p.add_argument("--oblique-absolute-minor-ratio", type=float, default=0.20)
    p.add_argument("--very-dispersed-normal-p95-deg", type=float, default=70.0)
    p.add_argument("--very-dispersed-max-ratio", type=float, default=0.14)

    # Eje.
    p.add_argument("--axis-plane-minimum-ratio", type=float, default=0.10)
    p.add_argument("--axis-pair-minimum-angle-deg", type=float, default=72.0)
    p.add_argument("--axis-pair-maximum-angle-deg", type=float, default=108.0)
    p.add_argument("--axis-consensus-tolerance-deg", type=float, default=9.0)
    p.add_argument("--axis-minimum-support-views", type=int, default=4)
    p.add_argument(
        "--axis-camera-vertical-max-deg",
        type=float,
        default=38.0,
        help=(
            "Escala angular del prior SUAVE hacia Y de cámara. "
            "No limita el eje durante la calibración salvo que se active "
            "--axis-use-camera-vertical-hard-gate."
        ),
    )
    p.add_argument(
        "--axis-use-camera-vertical-hard-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compatibilidad experimental. Por defecto False porque la orientación "
            "global del rig puede cambiar (por ejemplo, cámaras inclinadas hacia abajo)."
        ),
    )
    p.add_argument("--axis-ba-limit-deg", type=float, default=2.0)
    p.add_argument("--axis-prior-sigma-deg", type=float, default=0.9)

    # Refinamiento multivista del eje usando los ángulos físicos.
    p.add_argument("--axis-multiview-plane-minimum-ratio", type=float, default=0.07)
    p.add_argument("--axis-multiview-search-limit-deg", type=float, default=6.0)
    p.add_argument("--axis-multiview-prior-sigma-deg", type=float, default=3.0)
    p.add_argument("--axis-multiview-huber-deg", type=float, default=8.0)
    p.add_argument("--axis-multiview-maxiter", type=int, default=28)
    p.add_argument("--axis-multiview-popsize", type=int, default=8)

    # Fallback robusto de inicialización del eje (V3.2.1).
    # No relaja el consenso primario: solo se activa si ese consenso no alcanza
    # axis-minimum-support-views.
    p.add_argument(
        "--axis-fallback-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Si el consenso estricto de pares ortogonales falla, estima una "
            "semilla mediante coherencia Manhattan multivista y ángulos físicos."
        ),
    )
    p.add_argument("--axis-fallback-relaxed-min-angle-deg", type=float, default=45.0)
    p.add_argument("--axis-fallback-relaxed-max-angle-deg", type=float, default=135.0)
    p.add_argument("--axis-fallback-max-seeds", type=int, default=8)
    p.add_argument("--axis-fallback-seed-separation-deg", type=float, default=2.5)
    p.add_argument("--axis-fallback-search-limit-deg", type=float, default=10.0)
    p.add_argument("--axis-fallback-seed-prior-sigma-deg", type=float, default=6.0)
    p.add_argument("--axis-fallback-baseline-weight", type=float, default=0.18)

    # V3.2.2 — validación geométrica del fallback directamente con las nubes.
    # Se activa únicamente cuando el consenso estricto de eje no fue suficiente.
    # No usa resultados históricos ni geometría conocida del objeto.
    p.add_argument(
        "--axis-cloud-fallback-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cuando 08 entra por el fallback de eje, refina conjuntamente "
            "eje, centro y sentido usando las nubes actuales y los ángulos físicos."
        ),
    )
    p.add_argument("--axis-cloud-search-limit-deg", type=float, default=8.0)
    p.add_argument("--axis-cloud-center-radius-mm", type=float, default=95.0)
    p.add_argument("--axis-cloud-maxiter", type=int, default=11)
    p.add_argument("--axis-cloud-popsize", type=int, default=5)
    p.add_argument("--axis-cloud-tail-weight", type=float, default=0.35)
    p.add_argument("--axis-cloud-secondary-weight", type=float, default=0.35)
    p.add_argument("--axis-cloud-mount-multiplier", type=float, default=1.8)
    p.add_argument("--axis-cloud-overlap-target", type=float, default=0.55)
    p.add_argument("--axis-cloud-overlap-penalty-mm", type=float, default=7.0)

    # Durante un fallback débil de planos, el BA debe conservar mejor la
    # geometría tangencial y el prior físico del montaje para evitar que una
    # solución tipo arco/herradura gane solo por point-to-plane.
    p.add_argument("--fallback-ba-tangential-weight", type=float, default=0.22)
    p.add_argument(
        "--fallback-ba-mount-prior-equivalent-points",
        type=float,
        default=360.0,
    )
    p.add_argument(
        "--fallback-ba-center-prior-equivalent-points",
        type=float,
        default=90.0,
    )

    # V3.2.3 — la degeneración tangencial no depende únicamente de haber
    # entrado por fallback. Si la mayoría de poses tiene baja observabilidad
    # angular, un BA point-to-plane puede deslizar caras y mover el centro
    # aunque el eje inicial haya pasado el consenso estricto. En ese caso se
    # activa automáticamente una estabilización moderada.
    p.add_argument(
        "--auto-stabilize-low-observability",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--auto-stabilize-low-ratio",
        type=float,
        default=0.50,
        help="Activa BA estabilizado si esta fracción de poses es low.",
    )
    p.add_argument(
        "--auto-ba-tangential-weight",
        type=float,
        default=0.22,
    )
    p.add_argument(
        "--auto-ba-mount-prior-equivalent-points",
        type=float,
        default=300.0,
    )
    p.add_argument(
        "--auto-ba-center-prior-equivalent-points",
        type=float,
        default=75.0,
    )

    # Referencia física del montaje.
    # La incertidumbre por defecto es 30 mm porque el documento físico se toma
    # como referencia aproximada, no como metrología rígida.
    p.add_argument("--stereo-baseline-mm", type=float, default=81.0558)
    p.add_argument("--mount-midpoint-axis-distance-mm", type=float, default=400.0)
    p.add_argument("--mount-distance-sigma-mm", type=float, default=30.0)
    p.add_argument("--mount-symmetry-sigma-mm", type=float, default=25.0)
    p.add_argument("--mount-axis-baseline-sigma-deg", type=float, default=6.0)
    p.add_argument("--mount-coarse-prior-weight-mm", type=float, default=1.2)
    p.add_argument("--mount-prior-equivalent-points", type=float, default=180.0)

    # Peso efectivo de priors globales dentro de un BA con miles de
    # correspondencias. Se expresa como "correspondencias equivalentes".
    p.add_argument("--axis-prior-equivalent-points", type=float, default=90.0)
    p.add_argument("--center-prior-equivalent-points", type=float, default=45.0)
    p.add_argument("--angle-prior-equivalent-points", type=float, default=8.0)
    p.add_argument("--angle-smoothness-equivalent-points", type=float, default=9.0)
    p.add_argument("--angle-curvature-equivalent-points", type=float, default=7.0)

    # Grafo.
    p.add_argument("--maximum-edge-order", type=int, default=2)
    p.add_argument("--maximum-edge-gap-deg", type=float, default=30.5)
    p.add_argument("--secondary-edge-weight", type=float, default=0.72)

    # Muestras/solape grueso.
    p.add_argument("--optimization-voxel-mm", type=float, default=2.0)
    p.add_argument("--coarse-max-points-per-view", type=int, default=900)
    p.add_argument("--fine-max-points-per-view", type=int, default=3200)
    p.add_argument("--metric-trim-fraction", type=float, default=0.72)
    p.add_argument("--metric-overlap-distance-mm", type=float, default=5.0)
    p.add_argument("--metric-distance-cap-mm", type=float, default=15.0)
    p.add_argument("--metric-overlap-penalty-mm", type=float, default=5.0)

    # Centro/signo.
    p.add_argument("--center-global-search-radius-mm", type=float, default=75.0)
    p.add_argument("--center-global-maxiter", type=int, default=18)
    p.add_argument("--center-global-popsize", type=int, default=8)
    p.add_argument("--center-local-radius-mm", type=float, default=28.0)
    p.add_argument("--center-ba-limit-mm", type=float, default=30.0)
    p.add_argument("--center-prior-sigma-mm", type=float, default=24.0)

    # BA restringido.
    p.add_argument("--ba-rounds", type=int, default=4)
    p.add_argument(
        "--ba-geometry-only-rounds",
        type=int,
        default=2,
        help=(
            "Primeras rondas donde las correcciones angulares quedan "
            "prácticamente congeladas para resolver primero eje/centro."
        ),
    )
    p.add_argument("--ba-geometry-only-angle-limit-deg", type=float, default=0.020)
    p.add_argument("--ba-geometry-only-angle-sigma-deg", type=float, default=0.020)
    p.add_argument("--ba-correspondence-gates-mm", default="7.0,5.5,4.5,3.8")
    p.add_argument("--ba-max-correspondences-per-direction", type=int, default=320)
    p.add_argument("--ba-min-correspondences-per-direction", type=int, default=55)
    p.add_argument("--ba-point-plane-sigma-mm", type=float, default=1.8)
    p.add_argument("--ba-tangential-weight", type=float, default=0.10)
    p.add_argument("--ba-tangential-sigma-mm", type=float, default=5.0)
    p.add_argument(
        "--ba-robust-loss", choices=("linear", "soft_l1", "huber", "cauchy"), default="soft_l1"
    )
    p.add_argument("--ba-f-scale", type=float, default=1.0)
    p.add_argument("--ba-max-nfev", type=int, default=55)
    # Modelo angular V3.2.
    # angle-correction-limit-deg se conserva como TECHO GLOBAL de seguridad.
    p.add_argument("--angle-correction-limit-deg", type=float, default=0.60)
    p.add_argument("--angle-prior-sigma-deg", type=float, default=0.24)
    p.add_argument("--angle-smoothness-sigma-deg", type=float, default=0.32)
    p.add_argument("--angle-curvature-sigma-deg", type=float, default=0.22)

    # Observabilidad angular por pose.
    p.add_argument("--angular-plane-minimum-ratio", type=float, default=0.055)
    p.add_argument("--angular-minimum-tangent-sensitivity", type=float, default=0.45)
    p.add_argument("--angular-minimum-second-plane-weight", type=float, default=0.10)
    p.add_argument("--angular-full-diversity-second-weight", type=float, default=0.28)
    p.add_argument("--angular-support-reference", type=float, default=0.75)
    p.add_argument("--angular-observability-power", type=float, default=1.35)

    # Bounds adaptativos: el motor sigue siendo la fuente primaria.
    p.add_argument("--angular-low-limit-deg", type=float, default=0.18)
    p.add_argument("--angular-high-limit-deg", type=float, default=0.60)
    p.add_argument("--angular-low-sigma-deg", type=float, default=0.085)
    p.add_argument("--angular-high-sigma-deg", type=float, default=0.24)

    # Prior de normales: deliberadamente débil y solo con evidencia coherente.
    p.add_argument("--angular-normal-prior-max-target-deg", type=float, default=0.10)
    p.add_argument("--angular-normal-prior-scatter-sigma-deg", type=float, default=0.85)
    p.add_argument("--angular-normal-prior-mean-sigma-deg", type=float, default=2.5)
    p.add_argument("--angular-normal-prior-min-reliability", type=float, default=0.50)

    # Pesos del prior mecánico según observabilidad.
    p.add_argument("--angular-low-prior-weight", type=float, default=1.45)
    p.add_argument("--angular-high-prior-weight", type=float, default=0.80)

    # Historia/reponderación.
    p.add_argument("--history-overlap-reference", type=float, default=0.55)
    p.add_argument("--history-rmse-reference-mm", type=float, default=3.0)
    p.add_argument("--history-minimum-weight", type=float, default=0.20)

    # Validación final.
    p.add_argument("--final-overlap-distance-mm", type=float, default=4.5)
    p.add_argument("--final-primary-minimum-overlap", type=float, default=0.40)
    p.add_argument("--final-primary-maximum-rmse-mm", type=float, default=3.3)
    p.add_argument("--final-primary-maximum-p90-mm", type=float, default=7.5)
    p.add_argument("--warning-primary-acceptance", type=float, default=0.72)
    p.add_argument("--reject-primary-acceptance", type=float, default=0.55)
    p.add_argument("--warning-median-primary-overlap", type=float, default=0.55)
    p.add_argument("--reject-median-primary-overlap", type=float, default=0.35)
    p.add_argument("--warning-median-primary-rmse-mm", type=float, default=2.8)
    p.add_argument("--reject-median-primary-rmse-mm", type=float, default=4.0)
    p.add_argument("--warning-closure-rmse-mm", type=float, default=3.3)
    p.add_argument("--reject-closure-rmse-mm", type=float, default=5.0)
    p.add_argument("--final-primary-maximum-point-plane-rmse-mm", type=float, default=2.4)
    p.add_argument("--final-primary-maximum-point-plane-p90-mm", type=float, default=3.8)
    p.add_argument(
        "--tangential-warning-p90-mm",
        type=float,
        default=10.0,
        help="P90 euclídeo alto se reporta como warning tangencial, no como fallo de superficie.",
    )

    # Conformidad física: permite explícitamente diferencias de 2–3 cm.
    p.add_argument("--warning-mount-distance-error-mm", type=float, default=35.0)
    p.add_argument("--reject-mount-distance-error-mm", type=float, default=65.0)
    p.add_argument("--warning-mount-symmetry-error-mm", type=float, default=25.0)
    p.add_argument("--reject-mount-symmetry-error-mm", type=float, default=50.0)
    p.add_argument("--warning-axis-baseline-deviation-deg", type=float, default=7.0)
    p.add_argument("--reject-axis-baseline-deviation-deg", type=float, default=14.0)

    # Compactación global de caras.
    p.add_argument("--compactness-normal-gate-deg", type=float, default=25.0)
    p.add_argument("--compactness-minimum-points-per-face", type=int, default=500)
    p.add_argument(
        "--compactness-min-face-separation-mm",
        type=float,
        default=18.0,
        help=(
            "Separación mínima para interpretar dos clusters 1D como caras "
            "opuestas y no como una sola cara gruesa."
        ),
    )
    p.add_argument("--warning-global-face-p90-mm", type=float, default=5.5)
    p.add_argument("--reject-global-face-p90-mm", type=float, default=9.0)
    p.add_argument("--warning-manhattan-p90-deg", type=float, default=14.0)
    p.add_argument("--reject-manhattan-p90-deg", type=float, default=24.0)
    # Compatibilidad histórica; V3.2 ya no clasifica solo por max(|delta|).
    p.add_argument("--warning-max-angle-correction-deg", type=float, default=0.55)
    p.add_argument("--reject-max-angle-correction-deg", type=float, default=0.74)

    # Auditoría de saturación respecto al bound ADAPTATIVO de cada pose.
    p.add_argument("--angular-saturation-ratio", type=float, default=0.985)
    p.add_argument("--warning-total-angular-saturations", type=int, default=3)
    p.add_argument("--reject-total-angular-saturations", type=int, default=9)
    p.add_argument("--warning-high-observability-saturations", type=int, default=1)
    p.add_argument("--reject-high-observability-saturations", type=int, default=5)
    p.add_argument("--high-observability-threshold", type=float, default=0.65)
    p.add_argument("--warning-angular-prior-residual-deg", type=float, default=0.40)
    p.add_argument("--reject-angular-prior-residual-deg", type=float, default=0.80)
    p.add_argument("--warning-union-extent-ratio", type=float, default=1.75)
    p.add_argument("--reject-union-extent-ratio", type=float, default=2.40)

    # Exportación.
    p.add_argument("--union-voxel-mm", type=float, default=0.85)
    p.add_argument("--preview-width", type=int, default=760)
    p.add_argument("--preview-height", type=int, default=520)
    return p


# ---------------------------------------------------------------------------
# I/O / utilidades
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict:
    """Lee un documento JSON codificado en UTF-8."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def find_summary(directory: Path, preferred: Sequence[str]) -> Path:
    for name in preferred:
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No existe la salida exacta requerida de la ejecución actual: "
        + ", ".join(str(directory / name) for name in preferred)
    )


def prepare_output(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalize(v: np.ndarray) -> Optional[np.ndarray]:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        return None
    return v / n


def robust_stats(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "mad": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    med = float(np.median(arr))
    return {
        "count": int(arr.size),
        "median": med,
        "mean": float(np.mean(arr)),
        "mad": float(np.median(np.abs(arr - med))),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def weighted_robust_stats(
    values: Sequence[float],
    weights: Sequence[float],
) -> dict:
    """Estadística robusta ponderada, incluyendo cuantiles ponderados.

    Se usa para diagnósticos de normales donde una cara dominante y una región
    residual pequeña no deben tener el mismo peso estadístico.
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    good = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v = v[good]
    w = w[good]
    if len(v) == 0:
        return {
            "count": 0,
            "weight_sum": 0.0,
            "median": None,
            "mean": None,
            "mad": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }

    order = np.argsort(v)
    vs = v[order]
    ws = w[order]
    cumulative = np.cumsum(ws)
    total = float(cumulative[-1])

    def wq(q: float) -> float:
        target = float(np.clip(q, 0.0, 1.0)) * total
        j = int(np.searchsorted(cumulative, target, side="left"))
        j = min(max(j, 0), len(vs) - 1)
        return float(vs[j])

    med = wq(0.50)
    abs_dev = np.abs(v - med)
    order_d = np.argsort(abs_dev)
    ds = abs_dev[order_d]
    dws = w[order_d]
    dcum = np.cumsum(dws)
    jmad = int(np.searchsorted(dcum, 0.50 * total, side="left"))
    jmad = min(max(jmad, 0), len(ds) - 1)

    return {
        "count": int(len(v)),
        "weight_sum": total,
        "median": float(med),
        "mean": float(np.average(v, weights=w)),
        "mad": float(ds[jmad]),
        "p90": wq(0.90),
        "p95": wq(0.95),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    good = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[good], w[good]
    if len(v) == 0:
        return float("nan")
    order = np.argsort(v)
    v, w = v[order], w[order]
    cumulative = np.cumsum(w)
    idx = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
    return float(v[min(idx, len(v) - 1)])


def acute_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a, b = normalize(a), normalize(b)
    if a is None or b is None:
        return float("nan")
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def signed_angle_between_in_plane(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    """Ángulo firmado de a a b alrededor de axis, en radianes."""
    axis = normalize(axis)
    if axis is None:
        return 0.0
    ap = np.asarray(a, float) - np.dot(a, axis) * axis
    bp = np.asarray(b, float) - np.dot(b, axis) * axis
    ap, bp = normalize(ap), normalize(bp)
    if ap is None or bp is None:
        return 0.0
    s = float(np.dot(axis, np.cross(ap, bp)))
    c = float(np.clip(np.dot(ap, bp), -1.0, 1.0))
    return float(math.atan2(s, c))


def perpendicular_basis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    axis = normalize(axis)
    if axis is None:
        raise ValueError("Eje inválido")
    helpers = np.eye(3)
    helper = helpers[int(np.argmin(np.abs(helpers @ axis)))]
    e1 = normalize(np.cross(axis, helper))
    e2 = normalize(np.cross(axis, e1))
    if e1 is None or e2 is None:
        raise ValueError("No se pudo construir base perpendicular")
    return e1, e2


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalize(axis)
    if axis is None:
        raise ValueError("Eje inválido")
    x, y, z = axis
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def rotvec_matrix(rotvec: np.ndarray) -> np.ndarray:
    v = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(v))
    if theta < 1e-14:
        return np.eye(3, dtype=np.float64)
    return rotation_matrix(v / theta, theta)


def axis_transform(axis: np.ndarray, center: np.ndarray, angle_rad: float) -> np.ndarray:
    """Construye la transformación rígida de giro alrededor de una línea 3D."""
    R = rotation_matrix(axis, angle_rad)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = c - R @ c
    return T


def inverse_transform(T: np.ndarray) -> np.ndarray:
    """Invierte una transformación rígida a partir de su rotación y traslación."""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Aplica a los puntos la rotación y traslación de una matriz homogénea."""
    p = np.asarray(points, dtype=np.float64)
    return p @ T[:3, :3].T + T[:3, 3]


def transform_normals(normals: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Aplica solo la rotación a las normales y normaliza el resultado."""
    n = np.asarray(normals, dtype=np.float64) @ T[:3, :3].T
    lengths = np.linalg.norm(n, axis=1)
    good = np.isfinite(lengths) & (lengths > 1e-12)
    if np.any(good):
        n[good] /= lengths[good, None]
    return n


def robust_extent(points: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if len(p) == 0:
        return np.full(3, np.nan)
    lo = np.percentile(p, low, axis=0)
    hi = np.percentile(p, high, axis=0)
    return hi - lo


def numpy_voxel_indices(points: np.ndarray, voxel: float) -> np.ndarray:
    if len(points) == 0 or voxel <= 0:
        return np.arange(len(points), dtype=np.int64)
    keys = np.floor(np.asarray(points) / float(voxel)).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return np.sort(idx)


def voxel_downsample_with_colors(points: np.ndarray, colors: np.ndarray, voxel: float):
    if len(points) == 0 or voxel <= 0:
        return points, colors
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    n = int(inv.max()) + 1
    count = np.bincount(inv, minlength=n).astype(np.float64)
    p_sum = np.zeros((n, 3), np.float64)
    c_sum = np.zeros((n, 3), np.float64)
    np.add.at(p_sum, inv, points)
    np.add.at(c_sum, inv, colors.astype(np.float64))
    return p_sum / count[:, None], np.clip(c_sum / count[:, None], 0, 255).astype(np.uint8)


def hue_color_rgb(angle_deg: float) -> np.ndarray:
    hsv = np.uint8([[[int((float(angle_deg) % 360.0) / 2.0), 205, 235]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return bgr[::-1].astype(np.uint8)


# ---------------------------------------------------------------------------
# Campaña / planos
# ---------------------------------------------------------------------------


def default_manifest_path(root: Path, object_name: str) -> Path:
    return (
        root
        / "reconstruccion"
        / "multisesion"
        / "01_mapa_angular"
        / "mapa_angular_multisesion.json"
    )


def validate_campaign(
    cloud_summary: dict,
    manifest: Optional[dict],
    args,
) -> dict:
    records = [
        r
        for r in cloud_summary.get("views", [])
        if str(r.get("quality", "")) in ("accepted", "warning")
    ]
    records.sort(key=lambda r: int(r.get("pose_index", 9999)))
    if len(records) != int(args.expected_views):
        raise RuntimeError(
            f"08 V3 exige {args.expected_views} nubes elegibles; hay {len(records)}."
        )
    poses = [int(r.get("pose_index", -1)) for r in records]
    if poses != list(range(int(args.expected_views))):
        raise RuntimeError(f"Pose indices incompletos/desordenados: {poses}")

    diagnostics = {
        "view_count": len(records),
        "pose_indices": poses,
        "manifest_available": manifest is not None,
        "angle_differences_deg": [],
    }

    if manifest is not None:
        steps = int(manifest.get("steps_per_revolution", -1))
        if steps != int(args.expected_steps_per_revolution):
            raise RuntimeError(
                f"Manifest incompatible: pasos/vuelta={steps}, esperado={args.expected_steps_per_revolution}."
            )
        groups = {
            int(g["pose_index"]): float(g["physical_angle_deg"])
            for g in manifest.get("pose_groups", [])
        }
        if len(groups) != int(args.expected_views):
            raise RuntimeError("Manifest no contiene las 25 pose_groups esperadas.")
        for rec in records:
            pose = int(rec["pose_index"])
            cloud_angle = float(rec.get("physical_angle_deg", rec.get("angle_deg")))
            manifest_angle = float(groups[pose])
            diff = cloud_angle - manifest_angle
            diagnostics["angle_differences_deg"].append(diff)
            if abs(diff) > float(args.angle_validation_tolerance_deg):
                raise RuntimeError(
                    f"Ángulo incompatible P{pose:02d}: 06={cloud_angle:.9f}°, "
                    f"manifest={manifest_angle:.9f}°."
                )
    return diagnostics


def plane_fit_confidence(plane: dict) -> float:
    rmse = float(plane.get("rmse_mm", 3.0) or 3.0)
    med = plane.get("normal_angle_median_deg")
    p95 = plane.get("normal_angle_p95_deg")
    fit = math.exp(-0.5 * (rmse / 2.1) ** 2)
    if med is not None:
        fit *= math.exp(-0.5 * (max(0.0, float(med) - 7.0) / 18.0) ** 2)
    if p95 is not None:
        # P95 se usa solo como penalización suave; la auditoría 07 mostró
        # que puede ser alto por bordes aun cuando la mediana y RMSE son buenos.
        fit *= math.exp(-0.5 * (max(0.0, float(p95) - 38.0) / 55.0) ** 2)
    return float(np.clip(fit, 0.20, 1.0))


def plane_decisions(record: dict, args) -> List[PlaneDecision]:
    planes = list(record.get("planes", []))
    decisions: List[PlaneDecision] = []
    for i, p in enumerate(planes):
        ratio = float(p.get("ratio_total", 0.0) or 0.0)
        rmse = float(p.get("rmse_mm", 999.0) or 999.0)
        med = p.get("normal_angle_median_deg")
        p95 = p.get("normal_angle_p95_deg")
        status = "keep"
        reasons = []
        confidence = plane_fit_confidence(p)
        if ratio < float(args.pose_plane_minimum_ratio):
            status = "drop"
            reasons.append("plano demasiado pequeño para estimar pose")
        if rmse > float(args.pose_plane_hard_rmse_mm):
            status = "drop"
            reasons.append("RMSE planar demasiado alto")
        if (
            status != "drop"
            and p95 is not None
            and float(p95) > float(args.very_dispersed_normal_p95_deg)
            and ratio <= float(args.very_dispersed_max_ratio)
        ):
            status = "downweight"
            confidence *= 0.35
            reasons.append("plano pequeño con cola de normales muy dispersa")
        decisions.append(
            PlaneDecision(
                index=i,
                status=status,
                confidence=float(np.clip(confidence, 0.05, 1.0)),
                ratio=ratio,
                rmse_mm=rmse,
                normal_median_deg=(None if med is None else float(med)),
                normal_p95_deg=(None if p95 is None else float(p95)),
                reasons=reasons,
            )
        )

    # Capas paralelas separadas: la capa menor es evidencia secundaria, no pose.
    for pair in record.get("plane_relations", {}).get("separated_parallel_pairs", []):
        a, b = int(pair["plane_a"]), int(pair["plane_b"])
        if a >= len(decisions) or b >= len(decisions):
            continue
        da, db = decisions[a], decisions[b]
        weaker = da if (da.ratio, -da.rmse_mm) < (db.ratio, -db.rmse_mm) else db
        weaker.status = "drop"
        weaker.confidence = min(weaker.confidence, 0.08)
        weaker.reasons.append(
            f"capa paralela secundaria separada {float(pair.get('separation_mm', 0.0)):.2f} mm"
        )

    # Ángulo oblicuo incompatible en cuboide: si existe un plano claramente
    # minoritario, se excluye; si ambos son fuertes, se penaliza sin inventar
    # cuál es correcto.
    for pair in record.get("plane_relations", {}).get("cuboid_angle_warnings", []):
        a, b = int(pair["plane_a"]), int(pair["plane_b"])
        if a >= len(decisions) or b >= len(decisions):
            continue
        da, db = decisions[a], decisions[b]
        major, minor = (da, db) if da.ratio >= db.ratio else (db, da)
        if minor.ratio <= float(args.oblique_absolute_minor_ratio) or minor.ratio <= float(
            args.oblique_minor_ratio_to_major
        ) * max(major.ratio, 1e-9):
            minor.status = "drop"
            minor.confidence = min(minor.confidence, 0.10)
            minor.reasons.append(
                f"plano oblicuo minoritario ({float(pair.get('angle_deg', 0.0)):.1f}°)"
            )
        else:
            major.status = "downweight" if major.status == "keep" else major.status
            minor.status = "downweight" if minor.status == "keep" else minor.status
            major.confidence *= 0.82
            minor.confidence *= 0.55
            major.reasons.append("par oblicuo ambiguo: penalización suave")
            minor.reasons.append("par oblicuo ambiguo: penalización fuerte")

    return decisions


def view_weight_from_record(record: dict, decisions: Sequence[PlaneDecision]) -> float:
    quality = str(record.get("quality", "warning"))
    base = 1.0 if quality == "accepted" else 0.90
    metrics = record.get("metrics", {})
    explained = float(metrics.get("explained_ratio", 0.8) or 0.8)
    best = float(metrics.get("best_plane_ratio", 0.55) or 0.55)
    layers = int(metrics.get("parallel_separated_pair_count", 0) or 0)
    oblique = int(metrics.get("cuboid_angle_warning_count", 0) or 0)
    kept_ratio = sum(d.ratio for d in decisions if d.status != "drop")
    w = base
    w *= 0.72 + 0.28 * np.clip(explained, 0.0, 1.0)
    w *= 0.80 + 0.20 * np.clip(best, 0.0, 1.0)
    w *= 0.78 + 0.22 * np.clip(kept_ratio, 0.0, 1.0)
    if layers:
        w *= 0.88
    if oblique:
        w *= 0.84
    if layers and oblique:
        w *= 0.88
    return float(np.clip(w, 0.42, 1.0))


def build_registration_geometry(
    label_points: np.ndarray,
    labels: np.ndarray,
    record: dict,
    decisions: Sequence[PlaneDecision],
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    planes = list(record.get("planes", []))
    decision_map = {d.index: d for d in decisions}
    selected_points = []
    selected_normals = []
    selected_weights = []
    counts = {}
    for plane_idx, plane in enumerate(planes):
        decision = decision_map[plane_idx]
        mask = labels == plane_idx
        count = int(np.count_nonzero(mask))
        counts[str(plane_idx)] = {
            "points": count,
            "status": decision.status,
            "confidence": decision.confidence,
            "reasons": decision.reasons,
        }
        if decision.status == "drop" or count == 0:
            continue
        normal = normalize(np.asarray(plane["normal_xyz"], dtype=np.float64))
        if normal is None:
            continue
        pts = label_points[mask]
        selected_points.append(pts)
        selected_normals.append(np.tile(normal, (len(pts), 1)))
        weight = decision.confidence * (0.55 if decision.status == "downweight" else 1.0)
        selected_weights.append(np.full(len(pts), max(weight, 0.05), dtype=np.float64))

    if selected_points:
        p = np.vstack(selected_points)
        n = np.vstack(selected_normals)
        w = np.concatenate(selected_weights)
    else:
        p = np.empty((0, 3), np.float64)
        n = np.empty((0, 3), np.float64)
        w = np.empty((0,), np.float64)

    # Fallback: si la poda fue demasiado agresiva, usar el plano de mayor ratio
    # en vez de volver a toda la nube sin control.
    if len(p) < int(args.minimum_registration_points) and planes:
        order = np.argsort([float(pl.get("ratio_total", 0.0) or 0.0) for pl in planes])[::-1]
        fallback = []
        for idx in order:
            mask = labels == int(idx)
            if not np.any(mask):
                continue
            normal = normalize(np.asarray(planes[int(idx)]["normal_xyz"], float))
            if normal is None:
                continue
            fallback.append((label_points[mask], normal, int(idx)))
            if sum(len(x[0]) for x in fallback) >= int(args.minimum_registration_points):
                break
        if fallback:
            p = np.vstack([x[0] for x in fallback])
            n = np.vstack([np.tile(x[1], (len(x[0]), 1)) for x in fallback])
            w = np.concatenate([np.full(len(x[0]), 0.45, np.float64) for x in fallback])
            counts["fallback_planes"] = [x[2] for x in fallback]

    return p, n, w, counts


def load_views(
    cloud_dir: Path,
    geometry_dir: Path,
    cloud_summary: dict,
    geometry_summary: dict,
    args,
) -> List[ViewData]:
    geo_by_pose = {
        int(r.get("pose_index", i)): r for i, r in enumerate(geometry_summary.get("views", []))
    }
    # 07 actual no siempre guarda pose_index explícito: stem/orden es estable.
    if len(geo_by_pose) != len(geometry_summary.get("views", [])):
        geo_by_pose = {i: r for i, r in enumerate(geometry_summary.get("views", []))}

    clouds = [
        r
        for r in cloud_summary.get("views", [])
        if str(r.get("quality", "")) in ("accepted", "warning")
    ]
    clouds.sort(key=lambda r: int(r.get("pose_index", 9999)))
    geometry_records = list(geometry_summary.get("views", []))
    geometry_records.sort(key=lambda r: float(r.get("angle_deg", 9999.0)))

    views: List[ViewData] = []
    for order_index, cloud_record in enumerate(clouds):
        pose = int(cloud_record.get("pose_index", order_index))
        angle = float(cloud_record.get("physical_angle_deg", cloud_record.get("angle_deg")))

        # Buscar 07 por ángulo, no truncar a int.
        candidates = [
            r for r in geometry_records if abs(float(r.get("angle_deg", 9999.0)) - angle) <= 0.03
        ]
        if not candidates:
            raise RuntimeError(f"No se encontró registro 07 para P{pose:02d} / {angle:.6f}°")
        geo_record = candidates[0]
        if str(geo_record.get("quality", "")) == "rejected":
            continue

        stem = str(cloud_record["stem"])
        cloud_path = cloud_dir / f"{stem}_cloud_final.npz"
        labels_path = geometry_dir / f"{stem}_plane_labels.npz"
        if not cloud_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(f"Entradas incompletas para {stem}")

        with np.load(cloud_path) as data:
            points = np.asarray(data["points"], dtype=np.float64)
            colors = (
                np.asarray(data["colors"], dtype=np.uint8)
                if "colors" in data.files
                else np.zeros((len(points), 3), np.uint8)
            )
            normals = (
                np.asarray(data["normals"], dtype=np.float64)
                if "normals" in data.files
                else np.empty((0, 3), np.float64)
            )
        finite = np.all(np.isfinite(points), axis=1)
        points = points[finite]
        colors = (
            colors[finite] if len(colors) == len(finite) else np.zeros((len(points), 3), np.uint8)
        )
        normals = (
            normals[finite]
            if normals.ndim == 2 and len(normals) == len(finite)
            else np.empty((0, 3), np.float64)
        )

        with np.load(labels_path) as data:
            label_points = np.asarray(data["points"], dtype=np.float64)
            labels = np.asarray(data["plane_labels"], dtype=np.int32)
        good = np.all(np.isfinite(label_points), axis=1)
        label_points, labels = label_points[good], labels[good]

        decisions = plane_decisions(geo_record, args)
        reg_p, reg_n, reg_w, selection_diag = build_registration_geometry(
            label_points, labels, geo_record, decisions, args
        )
        if len(reg_p) < 500:
            raise RuntimeError(
                f"P{pose:02d}: geometría de pose insuficiente ({len(reg_p)} puntos)."
            )
        vw = view_weight_from_record(geo_record, decisions)
        views.append(
            ViewData(
                index=len(views),
                pose_index=pose,
                stem=stem,
                angle_deg=angle,
                quality_step_07=str(geo_record.get("quality", "warning")),
                view_weight=vw,
                points=points,
                colors=colors,
                normals=normals,
                registration_points=reg_p,
                registration_normals=reg_n,
                registration_weights=reg_w,
                planes=list(geo_record.get("planes", [])),
                plane_decisions=decisions,
                reasons=list(geo_record.get("reasons", [])) + [f"selección_pose={selection_diag}"],
            )
        )

    views.sort(key=lambda v: v.pose_index)
    for i, v in enumerate(views):
        v.index = i
    if len(views) != int(args.expected_views):
        raise RuntimeError(f"08 V3 esperaba {args.expected_views} vistas y cargó {len(views)}.")
    return views


# ---------------------------------------------------------------------------
# Eje inicial
# ---------------------------------------------------------------------------


def estimate_axis_from_planes(views: Sequence[ViewData], args) -> Tuple[np.ndarray, dict]:
    """
    Método primario: consenso estricto de productos cruz entre pares de planos
    aproximadamente ortogonales.

    V3.2.1:
    si el consenso estricto no alcanza axis_minimum_support_views, NO se
    modifican sus umbrales. Se deriva la inicialización a
    estimate_axis_multiview_fallback(), que usa todas las normales disponibles
    y los ángulos físicos conocidos.
    """
    vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    candidates = []
    for view in views:
        decisions = {d.index: d for d in view.plane_decisions}
        # Reconstruir relaciones directamente desde normales para no depender
        # del texto de clasificación de 07.
        planes = view.planes
        for i in range(len(planes)):
            for j in range(i + 1, len(planes)):
                di, dj = decisions.get(i), decisions.get(j)
                if di is None or dj is None or di.status == "drop" or dj.status == "drop":
                    continue
                if min(di.ratio, dj.ratio) < float(args.axis_plane_minimum_ratio):
                    continue
                n1 = normalize(np.asarray(planes[i]["normal_xyz"], float))
                n2 = normalize(np.asarray(planes[j]["normal_xyz"], float))
                if n1 is None or n2 is None:
                    continue
                angle = acute_angle_deg(n1, n2)
                if not (
                    float(args.axis_pair_minimum_angle_deg)
                    <= angle
                    <= float(args.axis_pair_maximum_angle_deg)
                ):
                    continue
                cross = normalize(np.cross(n1, n2))
                if cross is None:
                    continue
                vertical_angle = acute_angle_deg(cross, vertical)

                # La vertical de la plataforma pertenece al mundo físico, no al
                # sistema de coordenadas de la cámara. El gate duro permanece
                # desactivado por defecto.
                if bool(args.axis_use_camera_vertical_hard_gate) and vertical_angle > float(
                    args.axis_camera_vertical_max_deg
                ):
                    continue

                weight = (
                    view.view_weight
                    * math.sqrt(max(di.ratio * dj.ratio, 1e-8))
                    * math.sqrt(max(di.confidence * dj.confidence, 1e-8))
                )
                candidates.append((cross, weight, view.index, i, j, vertical_angle))

    tol = float(args.axis_consensus_tolerance_deg)
    best_seed = None
    for candidate in candidates:
        vec = candidate[0]
        by_view: Dict[int, float] = {}
        for other in candidates:
            residual = acute_angle_deg(vec, other[0])
            if np.isfinite(residual) and residual <= tol:
                by_view[other[2]] = max(by_view.get(other[2], 0.0), other[1])
        score = float(sum(by_view.values()))
        support = len(by_view)
        prior = max(
            0.0,
            1.0 - candidate[5] / max(float(args.axis_camera_vertical_max_deg), 1e-6),
        )
        key = (support, score + 0.10 * prior)
        if best_seed is None or key > best_seed[0]:
            best_seed = (key, vec)

    strict_support = 0 if best_seed is None else int(best_seed[0][0])
    strict_ok = best_seed is not None and strict_support >= int(args.axis_minimum_support_views)

    if not strict_ok:
        candidate_angles = [float(c[5]) for c in candidates if np.isfinite(float(c[5]))]
        angle_text = (
            "sin candidatos angulares"
            if not candidate_angles
            else (
                f"vertical cámara: mediana={np.median(candidate_angles):.2f}°, "
                f"min={np.min(candidate_angles):.2f}°, "
                f"max={np.max(candidate_angles):.2f}°"
            )
        )

        strict_failure = {
            "support_view_count": int(strict_support),
            "required_support_view_count": int(args.axis_minimum_support_views),
            "candidate_count": int(len(candidates)),
            "camera_vertical_candidate_angle_deg": robust_stats(candidate_angles),
            "message": (
                "Consenso estricto insuficiente; "
                f"mejor soporte={strict_support}/{int(args.axis_minimum_support_views)}, "
                f"candidatos={len(candidates)}, {angle_text}."
            ),
        }

        if bool(args.axis_fallback_enabled):
            fallback_axis, fallback_diag = estimate_axis_multiview_fallback(
                views,
                args,
                strict_candidates=candidates,
            )
            if fallback_axis is not None and fallback_diag.get("available", False):
                diag = {
                    "method": "multiview_physical_angle_fallback",
                    "fallback_used": True,
                    "axis_xyz": fallback_axis.tolist(),
                    "candidate_count": int(len(candidates)),
                    "used_candidate_count": int(fallback_diag.get("selected_seed_source_count", 0)),
                    # Se conserva el soporte estricto como auditoría; no se
                    # presenta como si el fallback hubiera "inventado" soporte.
                    "support_view_count": int(strict_support),
                    "support_pose_indices": [],
                    "residual_deg": fallback_diag.get(
                        "selected_normal_residual_deg",
                        {},
                    ),
                    "camera_vertical_angle_deg": acute_angle_deg(
                        fallback_axis,
                        vertical,
                    ),
                    "camera_vertical_prior": {
                        "hard_gate_enabled": bool(args.axis_use_camera_vertical_hard_gate),
                        "reference_scale_deg": float(args.axis_camera_vertical_max_deg),
                        "interpretation": (
                            "Y de cámara es solo referencia diagnóstica; "
                            "el fallback usa principalmente coherencia multivista "
                            "y ángulos físicos."
                        ),
                    },
                    "strict_consensus_failure": strict_failure,
                    "fallback": fallback_diag,
                }
                return fallback_axis, diag

        raise RuntimeError(
            strict_failure["message"] + " El fallback multivista no estuvo disponible o no produjo "
            "una solución finita. El fallo sigue siendo geométrico, no de rutas."
        )

    # Consenso estricto suficiente: comportamiento histórico V3.2.
    seed = best_seed[1]
    covariance = np.zeros((3, 3), dtype=np.float64)
    used = []
    for vec, weight, vi, pi, pj, vertical_angle in candidates:
        if acute_angle_deg(seed, vec) <= tol * 1.35:
            if np.dot(seed, vec) < 0:
                vec = -vec
            covariance += max(weight, 1e-8) * np.outer(vec, vec)
            used.append((vec, weight, vi, pi, pj, vertical_angle))
    values, vectors = np.linalg.eigh(covariance)
    axis = normalize(vectors[:, int(np.argmax(values))])
    if axis is None:
        raise RuntimeError("Eje degenerado.")
    if axis[1] < 0:
        axis = -axis

    diag = {
        "method": "strict_pair_consensus",
        "fallback_used": False,
        "axis_xyz": axis.tolist(),
        "candidate_count": len(candidates),
        "used_candidate_count": len(used),
        "support_view_count": len(set(x[2] for x in used)),
        "support_pose_indices": [views[i].pose_index for i in sorted(set(x[2] for x in used))],
        "residual_deg": robust_stats(acute_angle_deg(axis, x[0]) for x in used),
        "camera_vertical_angle_deg": acute_angle_deg(axis, vertical),
        "camera_vertical_prior": {
            "hard_gate_enabled": bool(args.axis_use_camera_vertical_hard_gate),
            "reference_scale_deg": float(args.axis_camera_vertical_max_deg),
            "interpretation": (
                "Y de cámara es solo prior suave; no representa la vertical del mundo "
                "cuando el rig está inclinado."
            ),
        },
    }
    return axis, diag


# ---------------------------------------------------------------------------
# V3.1: referencia física del montaje + eje multivista
# ---------------------------------------------------------------------------


def point_to_axis_line_distance(
    point: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
) -> float:
    p = np.asarray(point, dtype=np.float64).reshape(3)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    a = normalize(axis)
    if a is None:
        return float("nan")
    return float(np.linalg.norm(np.cross(p - c, a)))


def closest_point_on_axis_line(
    point: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    p = np.asarray(point, dtype=np.float64).reshape(3)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    a = normalize(axis)
    if a is None:
        return c.copy()
    return c + float(np.dot(p - c, a)) * a


def mounting_geometry_diagnostics(
    axis: np.ndarray,
    center: np.ndarray,
    args,
) -> dict:
    """
    Compara la línea estimada con el montaje físico sin asumir pitch de cámara.

    En el sistema rectificado se toma:
      cámara izquierda = [0,0,0]
      cámara derecha   = [baseline,0,0]
      punto medio      = [baseline/2,0,0]

    La distancia punto-medio -> línea de eje es invariante ante la inclinación
    global de las cámaras, por lo que es un prior más seguro que imponer Z=400.
    """
    baseline = float(args.stereo_baseline_mm)
    left = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    right = np.array([baseline, 0.0, 0.0], dtype=np.float64)
    midpoint = 0.5 * (left + right)

    d_left = point_to_axis_line_distance(left, center, axis)
    d_right = point_to_axis_line_distance(right, center, axis)
    d_mid = point_to_axis_line_distance(midpoint, center, axis)

    a = normalize(axis)
    baseline_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    # Desviación respecto a la perpendicularidad ideal eje ⟂ baseline.
    perpendicular_deviation_deg = (
        float(np.degrees(np.arcsin(np.clip(abs(float(np.dot(a, baseline_axis))), 0.0, 1.0))))
        if a is not None
        else float("nan")
    )

    target = float(args.mount_midpoint_axis_distance_mm)
    closest = closest_point_on_axis_line(midpoint, center, axis)

    return {
        "baseline_mm": baseline,
        "target_midpoint_axis_distance_mm": target,
        "midpoint_axis_distance_mm": float(d_mid),
        "midpoint_axis_distance_error_mm": float(d_mid - target),
        "left_axis_distance_mm": float(d_left),
        "right_axis_distance_mm": float(d_right),
        "left_right_symmetry_error_mm": float(d_left - d_right),
        "axis_baseline_perpendicular_deviation_deg": perpendicular_deviation_deg,
        "closest_point_to_stereo_midpoint_xyz_mm": closest.astype(float).tolist(),
        "reference_uncertainty_note": (
            "Los 400 mm son un prior aproximado. Los sigmas por defecto permiten "
            "diferencias físicas del orden de 2–3 cm."
        ),
    }


def pseudo_huber_scalar(z: float, delta: float = 1.0) -> float:
    z = float(z)
    d = max(float(delta), 1e-9)
    return float(d * d * (math.sqrt(1.0 + (z / d) ** 2) - 1.0))


def mounting_objective_penalty_mm(
    axis: np.ndarray,
    center: np.ndarray,
    args,
) -> float:
    diag = mounting_geometry_diagnostics(axis, center, args)
    z_distance = diag["midpoint_axis_distance_error_mm"] / max(
        float(args.mount_distance_sigma_mm), 1e-6
    )
    z_symmetry = diag["left_right_symmetry_error_mm"] / max(
        float(args.mount_symmetry_sigma_mm), 1e-6
    )
    z_perp = diag["axis_baseline_perpendicular_deviation_deg"] / max(
        float(args.mount_axis_baseline_sigma_deg), 1e-6
    )
    loss = (
        pseudo_huber_scalar(z_distance)
        + pseudo_huber_scalar(z_symmetry)
        + pseudo_huber_scalar(z_perp)
    )
    return float(args.mount_coarse_prior_weight_mm) * loss


def collect_multiview_plane_normals(
    views: Sequence[ViewData],
    args,
) -> List[Tuple[int, np.ndarray, float]]:
    records = []
    minimum_ratio = float(args.axis_multiview_plane_minimum_ratio)
    for view in views:
        decisions = {d.index: d for d in view.plane_decisions}
        for plane_index, plane in enumerate(view.planes):
            decision = decisions.get(plane_index)
            if decision is None or decision.status == "drop" or decision.ratio < minimum_ratio:
                continue
            normal = normalize(np.asarray(plane["normal_xyz"], dtype=np.float64))
            if normal is None:
                continue
            weight = view.view_weight * max(decision.ratio, 1e-5) * max(decision.confidence, 1e-5)
            records.append((view.index, normal, float(weight)))
    return records


def refine_axis_multiview_manhattan(
    views: Sequence[ViewData],
    axis_seed: np.ndarray,
    sign: int,
    args,
) -> Tuple[np.ndarray, dict]:
    """
    Refina el eje usando las 25 poses y los ángulos mecánicos conocidos.

    Para un eje candidato, cada normal observada se desrota a P00.
    Con el eje correcto, todas deben agruparse alrededor de un marco Manhattan
    global (dos familias horizontales + eje vertical del objeto).
    """
    records = collect_multiview_plane_normals(views, args)
    if len(records) < 8:
        return axis_seed.copy(), {
            "available": False,
            "reason": f"solo {len(records)} normales útiles",
        }

    axis_seed = normalize(axis_seed)
    if axis_seed is None:
        raise ValueError("axis_seed inválido")
    e1, e2 = perpendicular_basis(axis_seed)

    limit = math.radians(float(args.axis_multiview_search_limit_deg))
    huber_deg = max(float(args.axis_multiview_huber_deg), 1e-6)
    prior_sigma = max(float(args.axis_multiview_prior_sigma_deg), 1e-6)
    baseline_sigma = max(float(args.mount_axis_baseline_sigma_deg), 1e-6)

    def unpack(x):
        rotvec = float(x[0]) * e1 + float(x[1]) * e2
        axis = normalize(rotvec_matrix(rotvec) @ axis_seed)
        if axis is None:
            axis = axis_seed.copy()
        if axis[1] < 0:
            axis = -axis
        phi = float(x[2])
        return axis, phi

    def robust_angle_loss_deg(residual_deg):
        r = np.asarray(residual_deg, dtype=np.float64)
        a = np.abs(r)
        return np.where(
            a <= huber_deg,
            0.5 * a * a,
            huber_deg * (a - 0.5 * huber_deg),
        )

    def objective(x):
        axis, phi = unpack(x)
        h1, h2, av = frame_axes(axis, *perpendicular_basis(axis), phi)
        frame = (h1, h2, av)

        residuals = []
        weights = []
        for view_index, normal, weight in records:
            theta = math.radians(-float(sign) * float(views[view_index].angle_deg))
            n_ref = normalize(rotation_matrix(axis, theta) @ normal)
            if n_ref is None:
                continue
            residuals.append(nearest_frame_angle(n_ref, frame))
            weights.append(weight)

        if not residuals:
            return 1e9

        loss = robust_angle_loss_deg(residuals)
        base = float(np.average(loss, weights=np.maximum(weights, 1e-8)))

        seed_dev = acute_angle_deg(axis, axis_seed)
        baseline_dev = mounting_geometry_diagnostics(
            axis,
            np.zeros(3, dtype=np.float64),
            args,
        )["axis_baseline_perpendicular_deviation_deg"]

        # Priors suaves: no fuerzan el eje, solo evitan soluciones degeneradas.
        base += 0.5 * (seed_dev / prior_sigma) ** 2
        base += 0.25 * (baseline_dev / baseline_sigma) ** 2
        return base

    result = differential_evolution(
        objective,
        bounds=[(-limit, limit), (-limit, limit), (0.0, math.pi / 2.0)],
        seed=int(args.seed) + 3101,
        popsize=int(args.axis_multiview_popsize),
        maxiter=int(args.axis_multiview_maxiter),
        polish=True,
        tol=1e-4,
        workers=1,
        updating="immediate",
    )
    axis, phi = unpack(result.x)
    h1, h2, av = frame_axes(axis, *perpendicular_basis(axis), phi)

    residuals = []
    weighted_records = []
    for view_index, normal, weight in records:
        theta = math.radians(-float(sign) * float(views[view_index].angle_deg))
        n_ref = normalize(rotation_matrix(axis, theta) @ normal)
        if n_ref is None:
            continue
        residual = nearest_frame_angle(n_ref, (h1, h2, av))
        residuals.append(residual)
        weighted_records.append(
            {
                "pose_index": int(views[view_index].pose_index),
                "residual_deg": float(residual),
                "weight": float(weight),
            }
        )

    return axis, {
        "available": True,
        "success": bool(result.success),
        "message": str(result.message),
        "objective": float(result.fun),
        "normal_count": len(residuals),
        "axis_seed_xyz": axis_seed.tolist(),
        "axis_final_xyz": axis.tolist(),
        "axis_change_deg": acute_angle_deg(axis_seed, axis),
        "phi_deg": math.degrees(phi),
        "normal_residual_deg": robust_stats(residuals),
        "frame_axes_xyz": [h1.tolist(), h2.tolist(), av.tolist()],
        "records": weighted_records,
    }


def _axis_seed_pool_relaxed(
    views: Sequence[ViewData],
    args,
    strict_candidates: Optional[Sequence[tuple]] = None,
) -> List[dict]:
    """
    Construye semillas de eje SIN alterar el criterio estricto del método
    primario. Las semillas relajadas solo existen para arrancar el fallback.

    Fuentes:
      - candidatos estrictos que sí pudieron calcularse;
      - productos cruz de pares de planos útiles con ángulo 45..135°;
      - Y de cámara como semilla de último recurso, nunca como eje impuesto.
    """
    raw = []

    if strict_candidates:
        for vec, weight, vi, pi, pj, vertical_angle in strict_candidates:
            axis = normalize(vec)
            if axis is None:
                continue
            if axis[1] < 0:
                axis = -axis
            raw.append(
                {
                    "axis": axis,
                    "weight": float(weight) * 1.20,
                    "source": "strict_pair_candidate",
                    "view_index": int(vi),
                    "plane_pair": [int(pi), int(pj)],
                    "camera_vertical_angle_deg": float(vertical_angle),
                }
            )

    min_angle = float(args.axis_fallback_relaxed_min_angle_deg)
    max_angle = float(args.axis_fallback_relaxed_max_angle_deg)
    min_ratio = float(args.axis_multiview_plane_minimum_ratio)

    for view in views:
        decisions = {d.index: d for d in view.plane_decisions}
        for i in range(len(view.planes)):
            for j in range(i + 1, len(view.planes)):
                di, dj = decisions.get(i), decisions.get(j)
                if (
                    di is None
                    or dj is None
                    or di.status == "drop"
                    or dj.status == "drop"
                    or min(di.ratio, dj.ratio) < min_ratio
                ):
                    continue
                n1 = normalize(np.asarray(view.planes[i]["normal_xyz"], dtype=np.float64))
                n2 = normalize(np.asarray(view.planes[j]["normal_xyz"], dtype=np.float64))
                if n1 is None or n2 is None:
                    continue
                angle = acute_angle_deg(n1, n2)
                if not (min_angle <= angle <= max_angle):
                    continue
                axis = normalize(np.cross(n1, n2))
                if axis is None:
                    continue
                if axis[1] < 0:
                    axis = -axis
                vertical_angle = acute_angle_deg(
                    axis,
                    np.array([0.0, 1.0, 0.0], dtype=np.float64),
                )
                if bool(args.axis_use_camera_vertical_hard_gate) and vertical_angle > float(
                    args.axis_camera_vertical_max_deg
                ):
                    continue
                weight = (
                    view.view_weight
                    * math.sqrt(max(di.ratio * dj.ratio, 1e-8))
                    * math.sqrt(max(di.confidence * dj.confidence, 1e-8))
                    * max(
                        0.20,
                        math.sin(math.radians(min(angle, 180.0 - angle))),
                    )
                )
                raw.append(
                    {
                        "axis": axis,
                        "weight": float(weight),
                        "source": "relaxed_pair_candidate",
                        "view_index": int(view.index),
                        "plane_pair": [int(i), int(j)],
                        "camera_vertical_angle_deg": float(vertical_angle),
                        "pair_angle_deg": float(angle),
                    }
                )

    # Último recurso: permite que el optimizador se mueva alrededor de Y, pero
    # no fuerza que el eje final sea Y.
    raw.append(
        {
            "axis": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "weight": 0.02,
            "source": "camera_Y_last_resort_seed",
            "view_index": None,
            "plane_pair": None,
            "camera_vertical_angle_deg": 0.0,
        }
    )

    # Ordenar por evidencia y eliminar semillas casi idénticas (el signo del eje
    # ya está canonizado con Y positivo).
    raw.sort(key=lambda r: float(r["weight"]), reverse=True)
    selected = []
    separation = max(float(args.axis_fallback_seed_separation_deg), 0.25)
    for record in raw:
        axis = record["axis"]
        if any(acute_angle_deg(axis, prev["axis"]) < separation for prev in selected):
            continue
        selected.append(record)
        if len(selected) >= max(1, int(args.axis_fallback_max_seeds)):
            break

    return selected


def _multiview_axis_data_score(
    views: Sequence[ViewData],
    axis: np.ndarray,
    sign: int,
    args,
) -> Tuple[float, dict]:
    """
    Score común para comparar ejes provenientes de semillas distintas.

    No contiene prior hacia la semilla. Por ello puede comparar de forma justa
    resultados de múltiples inicializaciones.

    La función desrota todas las normales con los ángulos físicos y encuentra
    el mejor marco Manhattan global. Se añade únicamente un prior físico suave
    de perpendicularidad eje/baseline.
    """
    records = collect_multiview_plane_normals(views, args)
    axis = normalize(axis)
    if axis is None or len(records) < 8:
        return float("inf"), {
            "available": False,
            "normal_count": int(len(records)),
            "reason": "eje inválido o menos de 8 normales útiles",
        }

    e1, e2 = perpendicular_basis(axis)
    huber_deg = max(float(args.axis_multiview_huber_deg), 1e-6)

    def objective_phi(phi: float) -> float:
        frame = frame_axes(axis, e1, e2, float(phi))
        residuals = []
        weights = []
        for view_index, normal, weight in records:
            theta = math.radians(-float(sign) * float(views[view_index].angle_deg))
            n_ref = normalize(rotation_matrix(axis, theta) @ normal)
            if n_ref is None:
                continue
            residuals.append(nearest_frame_angle(n_ref, frame))
            weights.append(weight)

        if not residuals:
            return float("inf")

        r = np.asarray(residuals, dtype=np.float64)
        w = np.maximum(np.asarray(weights, dtype=np.float64), 1e-8)
        a = np.abs(r)
        loss = np.where(
            a <= huber_deg,
            0.5 * a * a,
            huber_deg * (a - 0.5 * huber_deg),
        )
        return float(np.average(loss, weights=w))

    result = minimize_scalar(
        objective_phi,
        bounds=(0.0, math.pi / 2.0),
        method="bounded",
        options={"xatol": 1e-5},
    )
    phi = float(result.x)
    frame = frame_axes(axis, e1, e2, phi)

    residuals = []
    weights = []
    for view_index, normal, weight in records:
        theta = math.radians(-float(sign) * float(views[view_index].angle_deg))
        n_ref = normalize(rotation_matrix(axis, theta) @ normal)
        if n_ref is None:
            continue
        residuals.append(nearest_frame_angle(n_ref, frame))
        weights.append(float(weight))

    baseline_dev = mounting_geometry_diagnostics(
        axis,
        np.zeros(3, dtype=np.float64),
        args,
    )["axis_baseline_perpendicular_deviation_deg"]
    baseline_sigma = max(
        float(args.mount_axis_baseline_sigma_deg),
        1e-6,
    )
    baseline_penalty = (
        float(args.axis_fallback_baseline_weight) * (float(baseline_dev) / baseline_sigma) ** 2
    )

    data_score = float(result.fun)
    total_score = float(data_score + baseline_penalty)
    return total_score, {
        "available": True,
        "sign": int(sign),
        "objective_data": data_score,
        "baseline_penalty": float(baseline_penalty),
        "objective_total": total_score,
        "phi_deg": float(math.degrees(phi)),
        "normal_count": int(len(residuals)),
        "normal_residual_deg": robust_stats(residuals),
        "baseline_perpendicular_deviation_deg": float(baseline_dev),
        "frame_axes_xyz": [a.tolist() for a in frame if a is not None],
    }


def estimate_axis_multiview_fallback(
    views: Sequence[ViewData],
    args,
    strict_candidates: Optional[Sequence[tuple]] = None,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Fallback V3.2.1.

    No disminuye axis-minimum-support-views ni aumenta la tolerancia del
    consenso estricto. En su lugar hace multi-start sobre semillas geométricas,
    prueba ambos sentidos de giro y usa toda la evidencia angular de las 25
    poses para escoger la inicialización más coherente.

    Esta rutina es deliberadamente más costosa; solo se ejecuta cuando el
    método primario no pudo producir una semilla fiable.
    """
    seeds = _axis_seed_pool_relaxed(
        views,
        args,
        strict_candidates=strict_candidates,
    )
    records = collect_multiview_plane_normals(views, args)
    if len(records) < 8 or not seeds:
        return None, {
            "available": False,
            "reason": (
                f"fallback sin evidencia suficiente: "
                f"normales={len(records)}, semillas={len(seeds)}"
            ),
            "normal_count": int(len(records)),
            "seed_count": int(len(seeds)),
        }

    # Para el fallback se permite una búsqueda algo más amplia alrededor de
    # cada semilla, manteniendo un prior suave hacia la propia semilla.
    fallback_args = argparse.Namespace(**vars(args))
    fallback_args.axis_multiview_search_limit_deg = max(
        float(args.axis_multiview_search_limit_deg),
        float(args.axis_fallback_search_limit_deg),
    )
    fallback_args.axis_multiview_prior_sigma_deg = max(
        float(args.axis_multiview_prior_sigma_deg),
        float(args.axis_fallback_seed_prior_sigma_deg),
    )

    trials = []
    best = None

    for seed_index, seed_record in enumerate(seeds):
        seed_axis = normalize(seed_record["axis"])
        if seed_axis is None:
            continue

        for sign in (+1, -1):
            try:
                refined_axis, refine_diag = refine_axis_multiview_manhattan(
                    views,
                    seed_axis,
                    sign,
                    fallback_args,
                )
                score, score_diag = _multiview_axis_data_score(
                    views,
                    refined_axis,
                    sign,
                    args,
                )
            except Exception as exc:
                trials.append(
                    {
                        "seed_index": int(seed_index),
                        "seed_source": seed_record["source"],
                        "seed_axis_xyz": seed_axis.tolist(),
                        "sign": int(sign),
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            trial = {
                "seed_index": int(seed_index),
                "seed_source": seed_record["source"],
                "seed_source_weight": float(seed_record["weight"]),
                "seed_axis_xyz": seed_axis.tolist(),
                "sign": int(sign),
                "success": bool(np.isfinite(score) and refine_diag.get("available", False)),
                "axis_xyz": refined_axis.tolist(),
                "axis_change_from_seed_deg": acute_angle_deg(
                    seed_axis,
                    refined_axis,
                ),
                "score": float(score),
                "refinement": refine_diag,
                "common_score": score_diag,
            }
            trials.append(trial)

            if not trial["success"]:
                continue

            # Desempate: primero coherencia global, después mayor evidencia de
            # la semilla. No se premia artificialmente el soporte estricto.
            key = (
                float(score),
                -float(seed_record["weight"]),
            )
            if best is None or key < best[0]:
                best = (
                    key,
                    refined_axis.copy(),
                    int(sign),
                    seed_record,
                    refine_diag,
                    score_diag,
                )

    if best is None:
        return None, {
            "available": False,
            "reason": "ninguna combinación semilla/signo produjo score finito",
            "normal_count": int(len(records)),
            "seed_count": int(len(seeds)),
            "trials": trials,
        }

    axis = normalize(best[1])
    if axis is None:
        return None, {
            "available": False,
            "reason": "la mejor solución degeneró al normalizar",
            "trials": trials,
        }
    if axis[1] < 0:
        axis = -axis

    chosen_seed = best[3]
    chosen_refine = best[4]
    chosen_score = best[5]

    return axis, {
        "available": True,
        "method": ("multi_start_multiview_manhattan_with_physical_angles"),
        "calibration_object_assumption": (
            "El fallback usa familias Manhattan del cuboide SOLO para "
            "calibrar el eje de la plataforma; no se aplica como supuesto "
            "de forma a objetos reconstruidos posteriormente."
        ),
        "normal_count": int(len(records)),
        "seed_count": int(len(seeds)),
        "selected_seed_source_count": int(
            sum(1 for r in seeds if r["source"] == chosen_seed["source"])
        ),
        "selected_seed_source": chosen_seed["source"],
        "selected_seed_weight": float(chosen_seed["weight"]),
        "selected_seed_axis_xyz": normalize(chosen_seed["axis"]).tolist(),
        "selected_sign_for_scoring": int(best[2]),
        "axis_xyz": axis.tolist(),
        "selected_score": float(best[0][0]),
        "selected_normal_residual_deg": chosen_score.get(
            "normal_residual_deg",
            {},
        ),
        "selected_common_score": chosen_score,
        "selected_refinement": chosen_refine,
        "seed_records": [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in record.items()}
            for record in seeds
        ],
        "trials": trials,
    }


# ---------------------------------------------------------------------------
# V3.2: observabilidad angular por pose
# ---------------------------------------------------------------------------


def wrap_periodic_deg(value: float, period_deg: float) -> float:
    """Envuelve un ángulo al intervalo [-period/2, +period/2)."""
    period = float(period_deg)
    return float((float(value) + 0.5 * period) % period - 0.5 * period)


def angular_constraint_to_dict(c: AngularConstraint) -> dict:
    return {
        "pose_index": int(c.pose_index),
        "observability_score": float(c.observability_score),
        "observability_class": str(c.observability_class),
        "informative_plane_count": int(c.informative_plane_count),
        "distinct_lateral_family_count": int(c.distinct_lateral_family_count),
        "effective_lateral_support": float(c.effective_lateral_support),
        "diversity_score": float(c.diversity_score),
        "normal_prior_raw_deg": float(c.normal_prior_raw_deg),
        "normal_prior_scatter_deg": (
            None if c.normal_prior_scatter_deg is None else float(c.normal_prior_scatter_deg)
        ),
        "normal_prior_reliability": float(c.normal_prior_reliability),
        "target_correction_deg": float(c.target_correction_deg),
        "sigma_deg": float(c.sigma_deg),
        "limit_deg": float(c.limit_deg),
        "prior_weight": float(c.prior_weight),
        "plane_records": c.plane_records,
        "reasons": c.reasons,
    }


def angular_observability_class(score: float) -> str:
    s = float(score)
    if s < 0.40:
        return "low"
    if s < 0.70:
        return "medium"
    return "high"


def build_angular_constraints(
    views: Sequence[ViewData],
    axis: np.ndarray,
    sign: int,
    frame_axes_xyz: Sequence[Sequence[float]],
    args,
) -> Tuple[List[AngularConstraint], dict]:
    """
    Construye un modelo angular distinto para cada pose.

    Principio:
    - una sola cara lateral aporta cierta orientación, pero es frágil;
    - dos caras laterales de familias distintas hacen mucho más observable
      la rotación alrededor del eje;
    - una cara superior/inferior (normal casi paralela al eje) prácticamente
      no informa el yaw;
    - el prior de normales nunca puede reemplazar al ángulo mecánico.

    El prior angular derivado de normales se obtiene únicamente cuando existen
    >=2 evidencias laterales y su dispersión es razonable. Incluso entonces,
    su objetivo final se limita a unos pocos décimos de grado.
    """
    axis = normalize(axis)
    if axis is None:
        raise ValueError("Eje inválido para observabilidad angular.")

    if len(frame_axes_xyz) != 3:
        e1, e2 = perpendicular_basis(axis)
        h1, h2 = e1, e2
    else:
        h1 = normalize(np.asarray(frame_axes_xyz[0], dtype=np.float64))
        h2 = normalize(np.asarray(frame_axes_xyz[1], dtype=np.float64))
        if h1 is None or h2 is None:
            e1, e2 = perpendicular_basis(axis)
            h1, h2 = e1, e2

    constraints: List[AngularConstraint] = []
    all_plane_records = []

    min_ratio = float(args.angular_plane_minimum_ratio)
    min_sensitivity = float(args.angular_minimum_tangent_sensitivity)
    min_second_weight = float(args.angular_minimum_second_plane_weight)
    full_second_weight = max(
        float(args.angular_full_diversity_second_weight),
        min_second_weight + 1e-6,
    )
    support_reference = max(float(args.angular_support_reference), 1e-6)
    power = max(float(args.angular_observability_power), 0.25)

    for view in views:
        plane_records = []
        candidate_values = []
        candidate_weights = []
        lateral_families = []

        for decision in view.plane_decisions:
            if decision.status == "drop":
                continue
            if decision.ratio < min_ratio:
                continue
            if decision.index >= len(view.planes):
                continue

            plane = view.planes[decision.index]
            normal = normalize(np.asarray(plane.get("normal_xyz"), dtype=np.float64))
            if normal is None:
                continue

            # Llevar la normal a P00 usando SOLO el ángulo mecánico nominal.
            nominal_unrotation = rotation_matrix(
                axis,
                math.radians(-float(sign) * float(view.angle_deg)),
            )
            normal_ref = normalize(nominal_unrotation @ normal)
            if normal_ref is None:
                continue

            axial = float(np.dot(normal_ref, axis))
            tangent = normal_ref - axial * axis
            sensitivity = float(np.linalg.norm(tangent))

            record = {
                "plane_index": int(decision.index),
                "status": str(decision.status),
                "ratio": float(decision.ratio),
                "confidence": float(decision.confidence),
                "rmse_mm": float(decision.rmse_mm),
                "tangent_sensitivity": sensitivity,
                "informative_for_yaw": False,
            }

            if sensitivity < min_sensitivity:
                record["reason"] = "normal demasiado paralela al eje: baja sensibilidad angular"
                plane_records.append(record)
                all_plane_records.append(
                    {
                        "pose_index": int(view.pose_index),
                        **record,
                    }
                )
                continue

            tangent /= max(sensitivity, 1e-12)
            x = float(np.dot(tangent, h1))
            y = float(np.dot(tangent, h2))
            azimuth_deg = float(np.degrees(np.arctan2(y, x)))

            # Para un cuboide, las caras laterales pertenecen a familias
            # separadas 90°. La familia se escoge cerca del ángulo nominal.
            nearest_family_index = int(round(azimuth_deg / 90.0))
            target_azimuth_deg = 90.0 * nearest_family_index
            signed_to_family_deg = wrap_periodic_deg(
                target_azimuth_deg - azimuth_deg,
                90.0,
            )

            # La pose_to_reference usa R(-sign*(angle+corr)).
            # Por tanto, el corr que produciría esa rotación de normal es:
            normal_candidate_deg = -float(sign) * signed_to_family_deg

            status_factor = 1.0 if decision.status == "keep" else 0.35
            weight = (
                float(decision.ratio)
                * float(decision.confidence)
                * sensitivity
                * sensitivity
                * status_factor
            )

            family_mod4 = nearest_family_index % 4
            record.update(
                {
                    "informative_for_yaw": True,
                    "azimuth_reference_deg": azimuth_deg,
                    "nearest_family_mod4": int(family_mod4),
                    "signed_to_family_deg": float(signed_to_family_deg),
                    "normal_candidate_correction_deg": float(normal_candidate_deg),
                    "angular_weight": float(weight),
                }
            )
            plane_records.append(record)
            all_plane_records.append(
                {
                    "pose_index": int(view.pose_index),
                    **record,
                }
            )

            if weight >= 0.025:
                candidate_values.append(float(normal_candidate_deg))
                candidate_weights.append(float(weight))
                lateral_families.append(int(family_mod4))

        weights = np.asarray(candidate_weights, dtype=np.float64)
        values = np.asarray(candidate_values, dtype=np.float64)

        informative_count = int(len(values))
        effective_support = float(np.sum(weights)) if informative_count else 0.0
        support_score = float(np.clip(effective_support / support_reference, 0.0, 1.0))

        distinct_families = len(set(lateral_families))
        sorted_weights = (
            np.sort(weights)[::-1] if informative_count else np.empty(0, dtype=np.float64)
        )
        second_weight = float(sorted_weights[1]) if len(sorted_weights) >= 2 else 0.0

        diversity_score = 0.0
        if informative_count >= 2 and distinct_families >= 2 and second_weight >= min_second_weight:
            diversity_score = float(
                np.clip(
                    (second_weight - min_second_weight) / (full_second_weight - min_second_weight),
                    0.0,
                    1.0,
                )
            )
            # Una segunda cara ya aporta información aunque todavía no alcance
            # el soporte de "diversidad plena".
            diversity_score = max(diversity_score, 0.35)

        # Con una sola cara la observabilidad no puede clasificarse como alta,
        # por mucho soporte que tenga esa cara.
        observability = (
            support_score
            * (0.32 + 0.68 * diversity_score)
            * float(np.clip(view.view_weight, 0.55, 1.0))
        )
        if distinct_families < 2:
            observability = min(observability, 0.38)
        observability = float(np.clip(observability, 0.0, 1.0))

        # Evidencia angular de las normales.
        normal_mean = 0.0
        scatter = None
        reliability = 0.0
        reasons = []

        if informative_count >= 2 and effective_support > 1e-9:
            norm_weights = weights / np.sum(weights)
            normal_mean = float(np.sum(norm_weights * values))
            scatter = float(np.sqrt(np.sum(norm_weights * (values - normal_mean) ** 2)))

            scatter_sigma = max(
                float(args.angular_normal_prior_scatter_sigma_deg),
                1e-6,
            )
            mean_sigma = max(
                float(args.angular_normal_prior_mean_sigma_deg),
                1e-6,
            )
            reliability = (
                diversity_score
                * math.exp(-0.5 * (scatter / scatter_sigma) ** 2)
                * math.exp(-0.5 * (normal_mean / mean_sigma) ** 2)
            )
            reliability = float(np.clip(reliability, 0.0, 1.0))

            if scatter > 2.0 * scatter_sigma:
                reasons.append("normales laterales discrepan: prior normal muy reducido")
        elif informative_count == 1:
            normal_mean = float(values[0])
            reasons.append("una sola cara lateral: el ángulo mecánico domina")
        else:
            reasons.append("sin caras laterales suficientes: el ángulo mecánico domina")

        if reliability < float(args.angular_normal_prior_min_reliability):
            reliability = 0.0

        normal_target_cap = float(args.angular_normal_prior_max_target_deg)
        # El prior normal solo desplaza ligeramente el centro del prior.
        target_correction = float(
            np.clip(
                reliability * normal_mean,
                -normal_target_cap,
                +normal_target_cap,
            )
        )

        # Bound y sigma crecen con la observabilidad.
        low_limit = float(args.angular_low_limit_deg)
        high_limit = min(
            float(args.angular_high_limit_deg),
            float(args.angle_correction_limit_deg),
        )
        alpha = observability**power
        limit_deg = float(low_limit + (high_limit - low_limit) * alpha)

        low_sigma = float(args.angular_low_sigma_deg)
        high_sigma = float(args.angular_high_sigma_deg)
        sigma_deg = float(low_sigma + (high_sigma - low_sigma) * (observability**1.20))

        prior_weight = float(
            float(args.angular_low_prior_weight)
            + (float(args.angular_high_prior_weight) - float(args.angular_low_prior_weight))
            * observability
        )

        obs_class = angular_observability_class(observability)

        constraints.append(
            AngularConstraint(
                pose_index=int(view.pose_index),
                observability_score=observability,
                observability_class=obs_class,
                informative_plane_count=informative_count,
                distinct_lateral_family_count=int(distinct_families),
                effective_lateral_support=effective_support,
                diversity_score=float(diversity_score),
                normal_prior_raw_deg=float(normal_mean),
                normal_prior_scatter_deg=(None if scatter is None else float(scatter)),
                normal_prior_reliability=float(reliability),
                target_correction_deg=target_correction,
                sigma_deg=max(sigma_deg, 1e-4),
                limit_deg=max(limit_deg, 1e-4),
                prior_weight=max(prior_weight, 0.05),
                plane_records=plane_records,
                reasons=reasons,
            )
        )

    if len(constraints) != len(views):
        raise RuntimeError("No se generó una restricción angular por cada vista.")

    # P00 define la referencia absoluta del objeto.
    constraints[0].target_correction_deg = 0.0
    constraints[0].limit_deg = 0.0
    constraints[0].sigma_deg = min(
        constraints[0].sigma_deg,
        float(args.ba_geometry_only_angle_sigma_deg),
    )
    constraints[0].reasons.append("P00 fija el origen angular y no se optimiza.")

    diagnostics = {
        "view_count": len(constraints),
        "low_count": sum(c.observability_class == "low" for c in constraints),
        "medium_count": sum(c.observability_class == "medium" for c in constraints),
        "high_count": sum(c.observability_class == "high" for c in constraints),
        "observability": robust_stats(c.observability_score for c in constraints),
        "adaptive_limit_deg": robust_stats(c.limit_deg for c in constraints[1:]),
        "adaptive_sigma_deg": robust_stats(c.sigma_deg for c in constraints[1:]),
        "normal_prior_reliability": robust_stats(c.normal_prior_reliability for c in constraints),
        "normal_prior_target_deg": robust_stats(abs(c.target_correction_deg) for c in constraints),
        "all_plane_records": all_plane_records,
    }
    return constraints, diagnostics


def angular_model_final_diagnostics(
    constraints: Sequence[AngularConstraint],
    corrections: Sequence[float],
    args,
) -> dict:
    corr = np.asarray(corrections, dtype=np.float64)
    records = []
    saturation_threshold = float(args.angular_saturation_ratio)
    high_threshold = float(args.high_observability_threshold)

    for i, constraint in enumerate(constraints):
        correction = float(corr[i])
        limit = float(constraint.limit_deg)
        ratio = 0.0 if limit <= 1e-12 else abs(correction) / limit
        saturated = bool(i > 0 and ratio >= saturation_threshold)
        prior_residual = float(correction - constraint.target_correction_deg)
        records.append(
            {
                **angular_constraint_to_dict(constraint),
                "final_correction_deg": correction,
                "corrected_angle_deg": None,
                "saturation_ratio": float(ratio),
                "saturated": saturated,
                "high_observability": bool(constraint.observability_score >= high_threshold),
                "prior_residual_deg": prior_residual,
                "equivalent_motor_steps": float(
                    correction / (360.0 / float(args.expected_steps_per_revolution))
                ),
            }
        )

    saturated_records = [r for r in records if r["saturated"]]
    high_saturated = [r for r in saturated_records if r["high_observability"]]
    reliable_prior = [
        r
        for r in records
        if r["normal_prior_reliability"] >= float(args.angular_normal_prior_min_reliability)
    ]

    return {
        "records": records,
        "saturated_count": len(saturated_records),
        "saturated_pose_indices": [int(r["pose_index"]) for r in saturated_records],
        "high_observability_saturated_count": len(high_saturated),
        "high_observability_saturated_pose_indices": [int(r["pose_index"]) for r in high_saturated],
        "max_saturation_ratio": (max((r["saturation_ratio"] for r in records[1:]), default=0.0)),
        "correction_abs_deg": robust_stats(abs(float(v)) for v in corr),
        "prior_residual_abs_deg_reliable": robust_stats(
            abs(float(r["prior_residual_deg"])) for r in reliable_prior
        ),
        "normal_prior_reliable_pose_count": len(reliable_prior),
    }


# ---------------------------------------------------------------------------
# Centro / muestras / grafo
# ---------------------------------------------------------------------------


def robust_center_proxy(points: np.ndarray) -> np.ndarray:
    return np.median(np.asarray(points, dtype=np.float64), axis=0)


def harmonic_center_seed(views: Sequence[ViewData], axis: np.ndarray) -> Tuple[np.ndarray, dict]:
    e1, e2 = perpendicular_basis(axis)
    proxies = np.asarray([robust_center_proxy(v.registration_points) for v in views])
    weights = np.asarray([v.view_weight for v in views], dtype=np.float64)
    theta = np.radians([v.angle_deg for v in views])
    X = np.column_stack([np.ones(len(views)), np.cos(theta), np.sin(theta)])
    rw = np.sqrt(np.maximum(weights, 1e-8))
    c1 = np.linalg.lstsq(X * rw[:, None], (proxies @ e1) * rw, rcond=None)[0]
    c2 = np.linalg.lstsq(X * rw[:, None], (proxies @ e2) * rw, rcond=None)[0]
    axis_coord = weighted_median(proxies @ axis, weights)
    center = c1[0] * e1 + c2[0] * e2 + axis_coord * axis
    return center, {
        "center_xyz_mm": center.tolist(),
        "proxy_extent_mm": robust_extent(proxies).tolist(),
        "note": "semilla armónica robusta; el componente a lo largo del eje es gauge",
    }


def positive_delta(source_deg: float, target_deg: float) -> float:
    delta = (float(target_deg) - float(source_deg)) % 360.0
    return 0.0 if delta < 1e-9 else float(delta)


def build_edges(views: Sequence[ViewData], args) -> List[EdgeSpec]:
    edges: List[EdgeSpec] = []
    n = len(views)
    for source in range(n):
        for order in range(1, max(1, int(args.maximum_edge_order)) + 1):
            target = (source + order) % n
            if target == source:
                continue
            delta = positive_delta(views[source].angle_deg, views[target].angle_deg)
            if delta <= 0.0 or delta > float(args.maximum_edge_gap_deg):
                continue
            primary = order == 1
            closure = primary and source == n - 1 and target == 0
            base_weight = math.sqrt(views[source].view_weight * views[target].view_weight)
            if not primary:
                base_weight *= float(args.secondary_edge_weight)
            if closure:
                base_weight *= 1.10
            edges.append(
                EdgeSpec(
                    source_index=source,
                    target_index=target,
                    order=order,
                    nominal_delta_deg=delta,
                    primary=primary,
                    closure=closure,
                    base_weight=float(base_weight),
                )
            )
    return edges


def build_samples(
    views: Sequence[ViewData],
    args,
    max_points: int,
) -> Dict[int, RegistrationSample]:
    samples = {}
    rng = np.random.default_rng(int(args.seed))
    for view in views:
        idx = numpy_voxel_indices(view.registration_points, float(args.optimization_voxel_mm))
        p = view.registration_points[idx]
        n = view.registration_normals[idx]
        w = view.registration_weights[idx]
        if len(p) > int(max_points):
            # Muestreo ponderado suave: evita que un plano minoritario ruidoso
            # domine solo por densidad, pero mantiene distribución espacial.
            prob = np.clip(w, 0.05, None)
            prob = prob / np.sum(prob)
            choose = rng.choice(len(p), int(max_points), replace=False, p=prob)
            p, n, w = p[choose], n[choose], w[choose]
        samples[view.index] = RegistrationSample(
            points=p,
            normals=n,
            weights=w,
            tree=cKDTree(p),
        )
    return samples


def trimmed_rmse(distances: np.ndarray, fraction: float, cap: float) -> float:
    d = np.asarray(distances, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float(cap)
    d = np.minimum(d, float(cap))
    keep = max(8, min(len(d), int(math.ceil(len(d) * float(fraction)))))
    if keep < len(d):
        d = np.partition(d, keep - 1)[:keep]
    return float(np.sqrt(np.mean(d * d)))


def pair_metrics_from_transform(
    T_source_to_target: np.ndarray,
    source: RegistrationSample,
    target: RegistrationSample,
    args,
    overlap_distance: Optional[float] = None,
) -> dict:
    threshold = float(overlap_distance or args.metric_overlap_distance_mm)
    source_to_target = transform_points(source.points, T_source_to_target)
    d1 = target.tree.query(source_to_target, k=1, workers=1)[0]
    T_inv = inverse_transform(T_source_to_target)
    target_to_source = transform_points(target.points, T_inv)
    d2 = source.tree.query(target_to_source, k=1, workers=1)[0]
    overlap = 0.5 * (np.mean(d1 <= threshold) + np.mean(d2 <= threshold))
    rmse = 0.5 * (
        trimmed_rmse(d1, args.metric_trim_fraction, args.metric_distance_cap_mm)
        + trimmed_rmse(d2, args.metric_trim_fraction, args.metric_distance_cap_mm)
    )
    both = np.concatenate(
        [
            np.minimum(d1, float(args.metric_distance_cap_mm)),
            np.minimum(d2, float(args.metric_distance_cap_mm)),
        ]
    )
    p90 = float(np.percentile(both, 90)) if len(both) else float(args.metric_distance_cap_mm)
    score = rmse + float(args.metric_overlap_penalty_mm) * (1.0 - overlap)
    return {
        "overlap": float(overlap),
        "trimmed_rmse_mm": float(rmse),
        "p90_mm": p90,
        "score": float(score),
    }


def robust_weighted_objective(
    values: Sequence[float], weights: Sequence[float], keep_fraction: float = 0.86
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[good], weights[good]
    if len(values) == 0:
        return 1e6
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    target = float(keep_fraction) * np.sum(weights)
    k = int(np.searchsorted(np.cumsum(weights), target, side="left")) + 1
    k = max(1, min(k, len(values)))
    return float(np.average(values[:k], weights=np.maximum(weights[:k], 1e-8)))


def relative_transform_model(
    axis: np.ndarray,
    center: np.ndarray,
    sign: int,
    source_angle: float,
    target_angle: float,
    source_correction: float = 0.0,
    target_correction: float = 0.0,
) -> np.ndarray:
    # Transformación de coordenadas de la vista source hacia la vista target.
    # Se usa el delta positivo mecánico; las correcciones son del ángulo físico
    # de plataforma antes del signo cámara/plataforma.
    delta = positive_delta(source_angle, target_angle)
    corrected_delta = delta + float(target_correction) - float(source_correction)
    return axis_transform(axis, center, math.radians(sign * corrected_delta))


def model_objective(
    axis: np.ndarray,
    center: np.ndarray,
    sign: int,
    edges: Sequence[EdgeSpec],
    views: Sequence[ViewData],
    samples: Dict[int, RegistrationSample],
    args,
) -> float:
    values, weights = [], []
    for edge in edges:
        if not edge.primary:
            continue
        T = relative_transform_model(
            axis,
            center,
            sign,
            views[edge.source_index].angle_deg,
            views[edge.target_index].angle_deg,
        )
        metrics = pair_metrics_from_transform(
            T, samples[edge.source_index], samples[edge.target_index], args
        )
        values.append(metrics["score"])
        weights.append(edge.base_weight)
    data_score = robust_weighted_objective(values, weights)
    physical_penalty = mounting_objective_penalty_mm(axis, center, args)
    return float(data_score + physical_penalty)


def optimize_center_for_sign(
    axis: np.ndarray,
    center_seed: np.ndarray,
    sign: int,
    edges: Sequence[EdgeSpec],
    views: Sequence[ViewData],
    samples: Dict[int, RegistrationSample],
    args,
) -> Tuple[np.ndarray, dict]:
    e1, e2 = perpendicular_basis(axis)
    radius = float(args.center_global_search_radius_mm)

    def unpack(x):
        return center_seed + float(x[0]) * e1 + float(x[1]) * e2

    def objective(x):
        return model_objective(axis, unpack(x), sign, edges, views, samples, args)

    result = differential_evolution(
        objective,
        bounds=[(-radius, radius), (-radius, radius)],
        seed=int(args.seed),
        popsize=int(args.center_global_popsize),
        maxiter=int(args.center_global_maxiter),
        tol=0.02,
        polish=True,
        workers=1,
        updating="immediate",
    )
    center = unpack(result.x)
    return center, {
        "sign": int(sign),
        "score": float(result.fun),
        "offset_e1_e2_mm": [float(result.x[0]), float(result.x[1])],
        "success": bool(result.success),
        "message": str(result.message),
    }


def choose_sign_and_center(
    axis: np.ndarray,
    center_seed: np.ndarray,
    edges: Sequence[EdgeSpec],
    views: Sequence[ViewData],
    samples: Dict[int, RegistrationSample],
    args,
) -> Tuple[int, np.ndarray, dict]:
    candidates = []
    for sign in (+1, -1):
        center, diag = optimize_center_for_sign(
            axis, center_seed, sign, edges, views, samples, args
        )
        candidates.append((diag["score"], sign, center, diag))
    candidates.sort(key=lambda item: item[0])
    best = candidates[0]
    diag = {
        "chosen_sign": int(best[1]),
        "chosen_score": float(best[0]),
        "score_margin": float(candidates[1][0] - best[0]),
        "candidates": {str(c[1]): c[3] for c in candidates},
    }
    return int(best[1]), best[2], diag


def cloud_axis_center_hypothesis_score(
    axis: np.ndarray,
    center: np.ndarray,
    sign: int,
    edges: Sequence[EdgeSpec],
    views: Sequence[ViewData],
    samples: Dict[int, RegistrationSample],
    args,
    include_secondary: bool = False,
) -> Tuple[float, dict]:
    """Score de hipótesis basado en geometría observada, no en planos Manhattan.

    La función combina:
      - ajuste euclídeo simétrico de aristas primarias;
      - cola P75 para que unas pocas aristas muy malas no queden ocultas por
        una media recortada;
      - penalización de solape bajo;
      - opcionalmente aristas de segundo orden;
      - prior físico SUAVE del montaje.

    El objetivo es evitar soluciones de arco/herradura que pueden verse bien
    en point-to-plane pero desplazan las vistas tangencialmente.
    """
    primary_scores = []
    primary_weights = []
    primary_overlaps = []
    secondary_scores = []
    secondary_weights = []

    for edge in edges:
        if (not edge.primary) and (not include_secondary):
            continue

        T = relative_transform_model(
            axis,
            center,
            sign,
            views[edge.source_index].angle_deg,
            views[edge.target_index].angle_deg,
        )
        metric = pair_metrics_from_transform(
            T,
            samples[edge.source_index],
            samples[edge.target_index],
            args,
        )

        if edge.primary:
            primary_scores.append(float(metric["score"]))
            primary_weights.append(float(edge.base_weight))
            primary_overlaps.append(float(metric["overlap"]))
        else:
            secondary_scores.append(float(metric["score"]))
            secondary_weights.append(
                float(edge.base_weight) * float(args.axis_cloud_secondary_weight)
            )

    if not primary_scores:
        return float("inf"), {
            "available": False,
            "reason": "sin aristas primarias evaluables",
        }

    p_scores = np.asarray(primary_scores, dtype=np.float64)
    p_weights = np.maximum(
        np.asarray(primary_weights, dtype=np.float64),
        1e-8,
    )
    overlaps = np.asarray(primary_overlaps, dtype=np.float64)

    core = robust_weighted_objective(
        p_scores,
        p_weights,
        keep_fraction=0.82,
    )
    p75 = float(np.quantile(p_scores, 0.75))
    median_overlap = float(np.median(overlaps))

    overlap_deficit = np.maximum(
        0.0,
        float(args.axis_cloud_overlap_target) - overlaps,
    )
    overlap_penalty = float(args.axis_cloud_overlap_penalty_mm) * float(
        np.average(overlap_deficit, weights=p_weights)
    )

    # Una solución con varias aristas casi sin intersección es peligrosa aunque
    # el núcleo recortado sea bueno.
    severe_low_fraction = float(np.mean(overlaps < 0.25))
    severe_penalty = 4.0 * severe_low_fraction

    secondary_term = 0.0
    if secondary_scores:
        secondary_term = robust_weighted_objective(
            secondary_scores,
            secondary_weights,
            keep_fraction=0.78,
        )

    mount_penalty = float(args.axis_cloud_mount_multiplier) * mounting_objective_penalty_mm(
        axis,
        center,
        args,
    )

    total = (
        float(core)
        + float(args.axis_cloud_tail_weight) * p75
        + overlap_penalty
        + severe_penalty
        + 0.18 * float(secondary_term)
        + mount_penalty
    )

    return float(total), {
        "available": True,
        "score_total": float(total),
        "primary_core_score": float(core),
        "primary_score_p75": float(p75),
        "primary_overlap_median": float(median_overlap),
        "primary_overlap_deficit_penalty": float(overlap_penalty),
        "severe_low_overlap_fraction": float(severe_low_fraction),
        "severe_low_overlap_penalty": float(severe_penalty),
        "secondary_term": float(secondary_term),
        "mount_penalty": float(mount_penalty),
        "mounting_geometry": mounting_geometry_diagnostics(
            axis,
            center,
            args,
        ),
        "primary_score_stats": robust_stats(primary_scores),
        "primary_overlap_stats": robust_stats(primary_overlaps),
    }


@operacion("Estimar eje y centro de la plataforma")
def joint_cloud_axis_center_initialization(
    views: Sequence[ViewData],
    edges: Sequence[EdgeSpec],
    samples: Dict[int, RegistrationSample],
    axis_seed: np.ndarray,
    args,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int], dict]:
    """V3.2.2: inicialización conjunta con las nubes de ESTA campaña.

    Se usa únicamente cuando la semilla de eje vino del fallback de planos.

    Parámetros optimizados:
      x[0:2] = perturbación angular del eje alrededor de axis_seed
      x[2:4] = desplazamiento del centro dentro del plano perpendicular al eje

    Para cada eje candidato se recalcula la semilla armónica del centro.
    Se prueban ambos sentidos de giro.

    No se usan archivos históricos, dimensiones del cubo ni primitivas de forma.
    """
    axis_seed = normalize(axis_seed)
    if axis_seed is None:
        return (
            None,
            None,
            None,
            {
                "available": False,
                "reason": "axis_seed inválido",
            },
        )

    e1_seed, e2_seed = perpendicular_basis(axis_seed)
    angle_limit = math.radians(
        max(
            0.5,
            float(args.axis_cloud_search_limit_deg),
        )
    )
    center_radius = max(
        10.0,
        float(args.axis_cloud_center_radius_mm),
    )

    def unpack(x):
        rotvec = float(x[0]) * e1_seed + float(x[1]) * e2_seed
        axis = normalize(rotvec_matrix(rotvec) @ axis_seed)
        if axis is None:
            axis = axis_seed.copy()
        if axis[1] < 0:
            axis = -axis

        center_seed, _ = harmonic_center_seed(
            views,
            axis,
        )
        c1, c2 = perpendicular_basis(axis)
        center = center_seed + float(x[2]) * c1 + float(x[3]) * c2
        return axis, center

    trials = []
    best = None

    bounds = [
        (-angle_limit, angle_limit),
        (-angle_limit, angle_limit),
        (-center_radius, center_radius),
        (-center_radius, center_radius),
    ]

    for sign in (+1, -1):
        # Vincular el signo en la definición evita que un evaluador diferido
        # observe el valor de la siguiente iteración del bucle.
        def objective(x, direction_sign=sign):
            axis, center = unpack(x)
            score, _ = cloud_axis_center_hypothesis_score(
                axis,
                center,
                direction_sign,
                edges,
                views,
                samples,
                args,
                include_secondary=False,
            )
            return float(score)

        try:
            result = differential_evolution(
                objective,
                bounds=bounds,
                seed=int(args.seed) + 7001 + (0 if sign > 0 else 97),
                popsize=max(
                    4,
                    int(args.axis_cloud_popsize),
                ),
                maxiter=max(
                    4,
                    int(args.axis_cloud_maxiter),
                ),
                polish=True,
                tol=0.015,
                workers=1,
                updating="immediate",
            )
            axis, center = unpack(result.x)

            # La comparación final sí incorpora aristas secundarias.
            final_score, final_diag = cloud_axis_center_hypothesis_score(
                axis,
                center,
                sign,
                edges,
                views,
                samples,
                args,
                include_secondary=True,
            )

            trial = {
                "sign": int(sign),
                "success": bool(np.isfinite(final_score)),
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "nfev": int(result.nfev),
                "axis_xyz": axis.tolist(),
                "center_xyz_mm": center.tolist(),
                "axis_change_from_seed_deg": acute_angle_deg(
                    axis_seed,
                    axis,
                ),
                "parameter_vector": np.asarray(
                    result.x,
                    dtype=float,
                ).tolist(),
                "optimization_score_primary_only": float(result.fun),
                "final_score_with_secondary": float(final_score),
                "score_diagnostics": final_diag,
            }
            trials.append(trial)

            if not trial["success"]:
                continue

            # Desempate físico únicamente después del score geométrico.
            mount_error = abs(
                float(
                    final_diag.get(
                        "mounting_geometry",
                        {},
                    ).get(
                        "midpoint_axis_distance_error_mm",
                        1e9,
                    )
                )
            )
            key = (
                float(final_score),
                float(mount_error),
            )
            if best is None or key < best[0]:
                best = (
                    key,
                    axis.copy(),
                    center.copy(),
                    int(sign),
                    trial,
                )

        except Exception as exc:
            trials.append(
                {
                    "sign": int(sign),
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if best is None:
        return (
            None,
            None,
            None,
            {
                "available": False,
                "reason": "ningún sentido produjo hipótesis finita",
                "trials": trials,
            },
        )

    selected = best[4]
    return (
        best[1],
        best[2],
        best[3],
        {
            "available": True,
            "method": ("joint_cloud_axis_center_sign_with_physical_angles"),
            "axis_seed_xyz": axis_seed.tolist(),
            "selected_sign": int(best[3]),
            "selected_axis_xyz": best[1].tolist(),
            "selected_center_xyz_mm": best[2].tolist(),
            "axis_change_from_seed_deg": acute_angle_deg(
                axis_seed,
                best[1],
            ),
            "selected_score": float(best[0][0]),
            "selected_trial": selected,
            "trials": trials,
            "important_note": (
                "La selección usa exclusivamente las nubes de la campaña actual, "
                "los ángulos físicos y priors suaves del montaje. No consulta "
                "calibraciones anteriores ni resultados de otros trabajos."
            ),
        },
    )


def refine_center_local(
    axis: np.ndarray,
    center0: np.ndarray,
    sign: int,
    edges: Sequence[EdgeSpec],
    views: Sequence[ViewData],
    samples: Dict[int, RegistrationSample],
    args,
) -> Tuple[np.ndarray, dict]:
    e1, e2 = perpendicular_basis(axis)
    radius = float(args.center_local_radius_mm)

    def unpack(x):
        return center0 + float(x[0]) * e1 + float(x[1]) * e2

    def objective(x):
        return model_objective(axis, unpack(x), sign, edges, views, samples, args)

    result = minimize(
        objective,
        x0=np.zeros(2),
        method="Powell",
        bounds=[(-radius, radius), (-radius, radius)],
        options={"maxiter": 90, "xtol": 1e-3, "ftol": 2e-4},
    )
    center = unpack(result.x)
    return center, {
        "score": float(result.fun),
        "offset_e1_e2_mm": result.x.astype(float).tolist(),
        "center_xyz_mm": center.astype(float).tolist(),
        "success": bool(result.success),
        "message": str(result.message),
    }


# ---------------------------------------------------------------------------
# Modelo de parámetros BA
# ---------------------------------------------------------------------------


def ba_gates(args) -> List[float]:
    values = [
        float(x.strip()) for x in str(args.ba_correspondence_gates_mm).split(",") if x.strip()
    ]
    if not values:
        values = [6.0, 4.5, 3.8]
    rounds = int(args.ba_rounds)
    if len(values) < rounds:
        values.extend([values[-1]] * (rounds - len(values)))
    return values[:rounds]


def unpack_ba_parameters(
    x: np.ndarray,
    axis0: np.ndarray,
    center0: np.ndarray,
    axis_e1: np.ndarray,
    axis_e2: np.ndarray,
    center_e1: np.ndarray,
    center_e2: np.ndarray,
    view_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # x[0:2] son radianes de inclinación alrededor de dos tangentes del eje.
    rotvec = float(x[0]) * axis_e1 + float(x[1]) * axis_e2
    axis = normalize(rotvec_matrix(rotvec) @ axis0)
    if axis is None:
        axis = axis0.copy()
    center = center0 + float(x[2]) * center_e1 + float(x[3]) * center_e2
    corrections = np.zeros(view_count, dtype=np.float64)
    if view_count > 1:
        corrections[1:] = np.asarray(x[4 : 4 + view_count - 1], dtype=np.float64)
    return axis, center, corrections


def pose_to_reference(
    axis: np.ndarray,
    center: np.ndarray,
    sign: int,
    angle_deg: float,
    correction_deg: float,
) -> np.ndarray:
    physical = float(angle_deg) + float(correction_deg)
    return axis_transform(axis, center, math.radians(-sign * physical))


def build_poses(
    views: Sequence[ViewData],
    axis: np.ndarray,
    center: np.ndarray,
    sign: int,
    corrections: np.ndarray,
) -> List[np.ndarray]:
    """Construye las transformaciones de las vistas en el marco de referencia."""
    return [
        pose_to_reference(axis, center, sign, v.angle_deg, corrections[i])
        for i, v in enumerate(views)
    ]


def relative_from_poses(source_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    return inverse_transform(target_pose) @ source_pose


# ---------------------------------------------------------------------------
# Correspondencias
# ---------------------------------------------------------------------------


def _select_correspondence_indices(
    distances: np.ndarray,
    point_weights: np.ndarray,
    gate: float,
    maximum: int,
    rng: np.random.Generator,
) -> np.ndarray:
    valid = np.flatnonzero(np.isfinite(distances) & (distances <= float(gate)))
    if len(valid) <= int(maximum):
        return valid
    # Mantener cobertura espacial/geométrica: sortear con probabilidad basada
    # en confianza y con una preferencia suave por menor distancia.
    d = distances[valid]
    w = np.clip(point_weights[valid], 0.03, None)
    w *= np.exp(-0.5 * (d / max(float(gate), 1e-6)) ** 2)
    w = w / np.sum(w)
    chosen = rng.choice(valid, int(maximum), replace=False, p=w)
    return np.asarray(chosen, dtype=np.int64)


def build_correspondence_batches(
    edges: Sequence[EdgeSpec],
    views: Sequence[ViewData],
    samples: Dict[int, RegistrationSample],
    poses: Sequence[np.ndarray],
    gate_mm: float,
    args,
    round_index: int,
) -> Tuple[List[CorrespondenceBatch], List[dict]]:
    batches: List[CorrespondenceBatch] = []
    diagnostics = []
    rng = np.random.default_rng(int(args.seed) + 1009 * (round_index + 1))
    for edge in edges:
        s = samples[edge.source_index]
        t = samples[edge.target_index]
        T_st = relative_from_poses(poses[edge.source_index], poses[edge.target_index])
        T_ts = inverse_transform(T_st)

        # source -> target
        p_st = transform_points(s.points, T_st)
        d_st, idx_st = t.tree.query(p_st, k=1, workers=1)
        idx_src = _select_correspondence_indices(
            d_st,
            s.weights,
            gate_mm,
            int(args.ba_max_correspondences_per_direction),
            rng,
        )
        if len(idx_src) >= int(args.ba_min_correspondences_per_direction):
            target_idx = idx_st[idx_src]
            point_w = np.sqrt(
                np.clip(s.weights[idx_src], 0.02, 1.0) * np.clip(t.weights[target_idx], 0.02, 1.0)
            )
            batches.append(
                CorrespondenceBatch(
                    source_index=edge.source_index,
                    target_index=edge.target_index,
                    src_points=s.points[idx_src],
                    tgt_points=t.points[target_idx],
                    tgt_normals=t.normals[target_idx],
                    weights=point_w,
                    edge_weight=float(edge.base_weight * edge.history_weight),
                    direction="source_to_target",
                )
            )

        # target -> source (como batch invertido)
        p_ts = transform_points(t.points, T_ts)
        d_ts, idx_ts = s.tree.query(p_ts, k=1, workers=1)
        idx_tgt = _select_correspondence_indices(
            d_ts,
            t.weights,
            gate_mm,
            int(args.ba_max_correspondences_per_direction),
            rng,
        )
        if len(idx_tgt) >= int(args.ba_min_correspondences_per_direction):
            source_idx = idx_ts[idx_tgt]
            point_w = np.sqrt(
                np.clip(t.weights[idx_tgt], 0.02, 1.0) * np.clip(s.weights[source_idx], 0.02, 1.0)
            )
            batches.append(
                CorrespondenceBatch(
                    source_index=edge.target_index,
                    target_index=edge.source_index,
                    src_points=t.points[idx_tgt],
                    tgt_points=s.points[source_idx],
                    tgt_normals=s.normals[source_idx],
                    weights=point_w,
                    edge_weight=float(edge.base_weight * edge.history_weight),
                    direction="target_to_source",
                )
            )

        diagnostics.append(
            {
                "source_pose": views[edge.source_index].pose_index,
                "target_pose": views[edge.target_index].pose_index,
                "order": edge.order,
                "closure": edge.closure,
                "gate_mm": float(gate_mm),
                "forward_candidates": int(np.count_nonzero(d_st <= gate_mm)),
                "reverse_candidates": int(np.count_nonzero(d_ts <= gate_mm)),
                "history_weight": float(edge.history_weight),
            }
        )
    return batches, diagnostics


# ---------------------------------------------------------------------------
# BA restringido
# ---------------------------------------------------------------------------


def make_ba_residual_function(
    batches: Sequence[CorrespondenceBatch],
    views: Sequence[ViewData],
    sign: int,
    axis0: np.ndarray,
    center0: np.ndarray,
    axis_e1: np.ndarray,
    axis_e2: np.ndarray,
    center_e1: np.ndarray,
    center_e2: np.ndarray,
    angular_constraints: Sequence[AngularConstraint],
    args,
):
    n = len(views)
    if len(angular_constraints) != n:
        raise ValueError("Falta una restricción angular por vista.")

    point_plane_sigma = max(float(args.ba_point_plane_sigma_mm), 1e-6)
    tangential_sigma = max(float(args.ba_tangential_sigma_mm), 1e-6)
    tangential_weight = max(float(args.ba_tangential_weight), 0.0)
    smooth_sigma = max(float(args.angle_smoothness_sigma_deg), 1e-6)
    curvature_sigma = max(float(args.angle_curvature_sigma_deg), 1e-6)
    axis_sigma = math.radians(max(float(args.axis_prior_sigma_deg), 1e-6))
    center_sigma = max(float(args.center_prior_sigma_mm), 1e-6)

    targets = np.asarray(
        [float(c.target_correction_deg) for c in angular_constraints], dtype=np.float64
    )
    sigmas = np.asarray(
        [max(float(c.sigma_deg), 1e-5) for c in angular_constraints], dtype=np.float64
    )
    prior_weights = np.asarray(
        [max(float(c.prior_weight), 0.05) for c in angular_constraints], dtype=np.float64
    )
    observability = np.asarray(
        [float(c.observability_score) for c in angular_constraints], dtype=np.float64
    )

    # En la etapa geometry_only el código marca esta bandera para forzar que
    # todos los targets sean cero: primero deben resolverse eje + centro.
    force_mechanical_zero = bool(getattr(args, "angular_force_mechanical_zero", False))
    if force_mechanical_zero:
        targets = np.zeros_like(targets)
        sigmas[:] = max(
            float(args.ba_geometry_only_angle_sigma_deg),
            1e-5,
        )
        prior_weights[:] = max(
            float(args.angular_low_prior_weight),
            1.0,
        )

    def residual(x: np.ndarray) -> np.ndarray:
        axis, center, corrections = unpack_ba_parameters(
            x, axis0, center0, axis_e1, axis_e2, center_e1, center_e2, n
        )
        poses = build_poses(views, axis, center, sign, corrections)
        parts = []

        for batch in batches:
            T = relative_from_poses(
                poses[batch.source_index],
                poses[batch.target_index],
            )
            transformed = transform_points(batch.src_points, T)
            diff = transformed - batch.tgt_points
            normals = batch.tgt_normals
            normal_residual = np.sum(diff * normals, axis=1)

            scale = np.sqrt(
                np.clip(
                    batch.weights * batch.edge_weight,
                    1e-6,
                    None,
                )
            )
            parts.append(scale * normal_residual / point_plane_sigma)

            if tangential_weight > 0:
                normal_component = normal_residual[:, None] * normals
                tangent = diff - normal_component
                tscale = math.sqrt(tangential_weight) * scale / tangential_sigma
                parts.append((tscale[:, None] * tangent).reshape(-1))

        # Priors globales.
        axis_prior_scale = math.sqrt(max(float(args.axis_prior_equivalent_points), 0.0))
        center_prior_scale = math.sqrt(max(float(args.center_prior_equivalent_points), 0.0))
        angle_prior_scale = math.sqrt(max(float(args.angle_prior_equivalent_points), 0.0))
        smooth_prior_scale = math.sqrt(max(float(args.angle_smoothness_equivalent_points), 0.0))
        curvature_prior_scale = math.sqrt(max(float(args.angle_curvature_equivalent_points), 0.0))
        mount_scale = math.sqrt(max(float(args.mount_prior_equivalent_points), 0.0))

        parts.append(axis_prior_scale * np.asarray(x[0:2], dtype=np.float64) / axis_sigma)
        parts.append(center_prior_scale * np.asarray(x[2:4], dtype=np.float64) / center_sigma)

        mount = mounting_geometry_diagnostics(axis, center, args)
        mount_residual = np.asarray(
            [
                mount["midpoint_axis_distance_error_mm"]
                / max(float(args.mount_distance_sigma_mm), 1e-6),
                mount["left_right_symmetry_error_mm"]
                / max(float(args.mount_symmetry_sigma_mm), 1e-6),
                mount["axis_baseline_perpendicular_deviation_deg"]
                / max(float(args.mount_axis_baseline_sigma_deg), 1e-6),
            ],
            dtype=np.float64,
        )
        parts.append(mount_scale * mount_residual)

        if n > 1:
            # Prior mecánico/normal POR POSE.
            corr = corrections[1:]
            target = targets[1:]
            sigma = sigmas[1:]
            pw = prior_weights[1:]
            parts.append(angle_prior_scale * np.sqrt(pw) * (corr - target) / sigma)

            # Continuidad circular.
            circular = np.r_[corrections, corrections[0]]
            diffs = np.diff(circular)

            # En baja observabilidad la suavidad pesa más, porque el dato local
            # tiene menos autoridad para inventar un pico angular.
            obs_next = np.r_[observability[1:], observability[0]]
            obs_pair = np.minimum(observability, obs_next)
            pair_weight = 1.0 + 1.35 * (1.0 - obs_pair)
            parts.append(smooth_prior_scale * np.sqrt(pair_weight) * diffs / smooth_sigma)

            # Segunda diferencia circular: penaliza picos aislados sin obligar
            # a que todas las correcciones sean idénticas.
            prev_corr = np.roll(corrections, 1)
            next_corr = np.roll(corrections, -1)
            second_diff = prev_corr - 2.0 * corrections + next_corr
            curvature_weight = 1.0 + 1.25 * (1.0 - observability)
            parts.append(
                curvature_prior_scale * np.sqrt(curvature_weight) * second_diff / curvature_sigma
            )

        return np.concatenate([np.ravel(p) for p in parts if np.size(p) > 0])

    return residual


def ba_bounds(
    view_count: int,
    args,
    angular_constraints: Optional[Sequence[AngularConstraint]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    axis_limit = math.radians(float(args.axis_ba_limit_deg))
    center_limit = float(args.center_ba_limit_mm)

    if angular_constraints is None:
        angle_limits = np.full(
            max(0, view_count - 1),
            float(args.angle_correction_limit_deg),
            dtype=np.float64,
        )
    else:
        if len(angular_constraints) != view_count:
            raise ValueError("angular_constraints debe tener una entrada por vista.")
        angle_limits = np.asarray(
            [
                min(
                    float(c.limit_deg),
                    float(args.angle_correction_limit_deg),
                )
                for c in angular_constraints[1:]
            ],
            dtype=np.float64,
        )

    lower = np.r_[
        [-axis_limit, -axis_limit, -center_limit, -center_limit],
        -angle_limits,
    ]
    upper = np.r_[
        [axis_limit, axis_limit, center_limit, center_limit],
        +angle_limits,
    ]
    return lower.astype(np.float64), upper.astype(np.float64)


def edge_metrics(
    edges: Sequence[EdgeSpec],
    views: Sequence[ViewData],
    samples: Dict[int, RegistrationSample],
    poses: Sequence[np.ndarray],
    args,
    overlap_distance: Optional[float] = None,
) -> List[dict]:
    records = []
    for edge in edges:
        T = relative_from_poses(poses[edge.source_index], poses[edge.target_index])
        metrics = pair_metrics_from_transform(
            T,
            samples[edge.source_index],
            samples[edge.target_index],
            args,
            overlap_distance=overlap_distance,
        )
        records.append(
            {
                "source_index": edge.source_index,
                "target_index": edge.target_index,
                "source_pose": views[edge.source_index].pose_index,
                "target_pose": views[edge.target_index].pose_index,
                "source_angle_deg": views[edge.source_index].angle_deg,
                "target_angle_deg": views[edge.target_index].angle_deg,
                "nominal_delta_deg": edge.nominal_delta_deg,
                "order": edge.order,
                "primary": edge.primary,
                "closure": edge.closure,
                "base_weight": edge.base_weight,
                "history_weight": edge.history_weight,
                **metrics,
            }
        )
    return records


def update_history_weights(edges: Sequence[EdgeSpec], metrics: Sequence[dict], args) -> None:
    reference_overlap = max(float(args.history_overlap_reference), 1e-6)
    reference_rmse = max(float(args.history_rmse_reference_mm), 1e-6)
    minimum = float(args.history_minimum_weight)
    for edge, metric in zip(edges, metrics):
        overlap_factor = np.clip(metric["overlap"] / reference_overlap, 0.0, 1.0)
        rmse_factor = math.exp(-0.5 * (metric["trimmed_rmse_mm"] / reference_rmse) ** 2)
        instantaneous = float(np.clip(overlap_factor * rmse_factor, minimum, 1.0))
        # Historia: no cambiar violentamente por una sola ronda.
        edge.history_weight = float(
            np.clip(
                0.55 * edge.history_weight + 0.45 * instantaneous,
                minimum,
                1.0,
            )
        )


@operacion("Optimizar registro restringido")
def run_constrained_ba(
    views: Sequence[ViewData],
    edges: Sequence[EdgeSpec],
    fine_samples: Dict[int, RegistrationSample],
    axis0: np.ndarray,
    center0: np.ndarray,
    sign: int,
    angular_constraints: Sequence[AngularConstraint],
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], dict]:
    """
    V3.2: BA en dos etapas con bounds y priors angulares por pose.

    Etapa A:
        primeras N rondas con correcciones angulares prácticamente congeladas.
        Se obliga al sistema a explicar el registro mediante eje + centro.

    Etapa B:
        se abre el rango angular residual según la observabilidad de cada pose
        (hasta ~0.60° por defecto).
    """
    axis_e1, axis_e2 = perpendicular_basis(axis0)
    center_e1, center_e2 = perpendicular_basis(axis0)
    n = len(views)
    x = np.zeros(4 + max(0, n - 1), dtype=np.float64)
    gates = ba_gates(args)
    rounds = []

    geometry_only_rounds = int(
        np.clip(
            int(args.ba_geometry_only_rounds),
            0,
            len(gates),
        )
    )

    for round_index, gate in enumerate(gates):
        geometry_only = round_index < geometry_only_rounds

        # Namespace temporal: solo cambia fuerza/límite angular por etapa.
        stage_args = argparse.Namespace(**vars(args))
        if geometry_only:
            stage_args.angle_correction_limit_deg = float(args.ba_geometry_only_angle_limit_deg)
            stage_args.angle_prior_sigma_deg = float(args.ba_geometry_only_angle_sigma_deg)
            stage_args.angular_force_mechanical_zero = True
        else:
            stage_args.angular_force_mechanical_zero = False

        lower, upper = ba_bounds(
            n,
            stage_args,
            angular_constraints=angular_constraints,
        )
        # Mantener x estrictamente dentro de los bounds de la nueva etapa.
        eps = 1e-10
        x = np.minimum(np.maximum(x, lower + eps), upper - eps)

        axis_current, center_current, corrections_current = unpack_ba_parameters(
            x, axis0, center0, axis_e1, axis_e2, center_e1, center_e2, n
        )
        poses_current = build_poses(views, axis_current, center_current, sign, corrections_current)
        batches, corr_diag = build_correspondence_batches(
            edges, views, fine_samples, poses_current, gate, stage_args, round_index
        )
        if not batches:
            raise RuntimeError(f"BA ronda {round_index + 1}: no quedaron correspondencias.")

        residual = make_ba_residual_function(
            batches,
            views,
            sign,
            axis0,
            center0,
            axis_e1,
            axis_e2,
            center_e1,
            center_e2,
            angular_constraints,
            stage_args,
        )
        before = residual(x)
        result = least_squares(
            residual,
            x0=x,
            bounds=(lower, upper),
            loss=str(stage_args.ba_robust_loss),
            f_scale=float(stage_args.ba_f_scale),
            x_scale="jac",
            max_nfev=int(stage_args.ba_max_nfev),
            method="trf",
            verbose=0,
        )
        x = result.x

        axis_after, center_after, corr_after = unpack_ba_parameters(
            x, axis0, center0, axis_e1, axis_e2, center_e1, center_e2, n
        )
        poses_after = build_poses(views, axis_after, center_after, sign, corr_after)
        metrics_after = edge_metrics(edges, views, fine_samples, poses_after, stage_args)
        update_history_weights(edges, metrics_after, stage_args)

        rounds.append(
            {
                "round": round_index + 1,
                "stage": "geometry_only" if geometry_only else "small_angle_refinement",
                "gate_mm": float(gate),
                "angle_limit_deg": float(stage_args.angle_correction_limit_deg),
                "batch_count": len(batches),
                "correspondence_count": int(sum(len(b.src_points) for b in batches)),
                "cost_before_half_sum_squares": float(0.5 * np.dot(before, before)),
                "cost_after_scipy": float(result.cost),
                "optimality": float(result.optimality),
                "nfev": int(result.nfev),
                "success": bool(result.success),
                "message": str(result.message),
                "axis_xyz": axis_after.tolist(),
                "center_xyz_mm": center_after.tolist(),
                "mounting_geometry": mounting_geometry_diagnostics(axis_after, center_after, args),
                "correction_deg": corr_after.tolist(),
                "angular_constraints": [angular_constraint_to_dict(c) for c in angular_constraints],
                "angular_final_diagnostics": angular_model_final_diagnostics(
                    angular_constraints,
                    corr_after,
                    args,
                ),
                "edge_metrics": metrics_after,
                "correspondence_diagnostics": corr_diag,
            }
        )

    axis, center, corrections = unpack_ba_parameters(
        x, axis0, center0, axis_e1, axis_e2, center_e1, center_e2, n
    )
    poses = build_poses(views, axis, center, sign, corrections)
    lower_final, upper_final = ba_bounds(
        n,
        args,
        angular_constraints=angular_constraints,
    )
    return (
        axis,
        center,
        corrections,
        poses,
        {
            "parameter_vector": x.astype(float).tolist(),
            "rounds": rounds,
            "bounds_lower_final": lower_final.tolist(),
            "bounds_upper_final": upper_final.tolist(),
            "strategy": (
                f"{geometry_only_rounds} rondas geometry_only + "
                f"{len(gates)-geometry_only_rounds} rondas "
                "small_angle_refinement_adaptativo"
            ),
            "angular_constraints": [angular_constraint_to_dict(c) for c in angular_constraints],
            "angular_final_diagnostics": angular_model_final_diagnostics(
                angular_constraints,
                corrections,
                args,
            ),
        },
    )


# ---------------------------------------------------------------------------
# Manhattan / diagnóstico
# ---------------------------------------------------------------------------


def significant_plane_normals(views: Sequence[ViewData]) -> List[Tuple[int, np.ndarray, float]]:
    records = []
    for view in views:
        decisions = {d.index: d for d in view.plane_decisions}
        for i, plane in enumerate(view.planes):
            decision = decisions.get(i)
            if decision is None or decision.status == "drop" or decision.ratio < 0.07:
                continue
            normal = normalize(np.asarray(plane["normal_xyz"], float))
            if normal is None:
                continue
            weight = view.view_weight * decision.ratio * decision.confidence
            records.append((view.index, normal, float(weight)))
    return records


def frame_axes(axis: np.ndarray, e1: np.ndarray, e2: np.ndarray, phi: float):
    h1 = normalize(math.cos(phi) * e1 + math.sin(phi) * e2)
    h2 = normalize(-math.sin(phi) * e1 + math.cos(phi) * e2)
    return h1, h2, normalize(axis)


def nearest_frame_angle(normal: np.ndarray, axes: Sequence[np.ndarray]) -> float:
    return min(acute_angle_deg(normal, axis) for axis in axes)


def manhattan_diagnostics(
    views: Sequence[ViewData],
    poses: Sequence[np.ndarray],
    axis: np.ndarray,
) -> dict:
    records = significant_plane_normals(views)
    if not records:
        return {"available": False, "reason": "sin planos significativos"}
    e1, e2 = perpendicular_basis(axis)
    transformed = []
    for vi, normal, weight in records:
        n = normalize(poses[vi][:3, :3] @ normal)
        if n is not None:
            transformed.append((n, weight, vi))

    def objective(phi):
        axes = frame_axes(axis, e1, e2, float(phi))
        residual = np.asarray([nearest_frame_angle(n, axes) for n, _, _ in transformed])
        weights = np.asarray([w for _, w, _ in transformed])
        huber = 8.0
        loss = np.where(
            residual <= huber,
            0.5 * residual * residual,
            huber * (residual - 0.5 * huber),
        )
        return float(np.average(loss, weights=np.maximum(weights, 1e-8)))

    result = minimize_scalar(
        objective,
        bounds=(0.0, math.pi / 2.0),
        method="bounded",
        options={"xatol": 1e-5},
    )
    phi = float(result.x)
    axes = frame_axes(axis, e1, e2, phi)
    residuals = [nearest_frame_angle(n, axes) for n, _, _ in transformed]
    residual_weights = [float(w) for _, w, _ in transformed]
    return {
        "available": True,
        "phi_deg": math.degrees(phi),
        "frame_axes_xyz": [a.tolist() for a in axes if a is not None],
        # Se conserva el estadístico crudo para auditoría.
        "normal_residual_deg": robust_stats(residuals),
        # V3.2.2: el criterio global usa también la evidencia/área de cada plano.
        "weighted_normal_residual_deg": weighted_robust_stats(
            residuals,
            residual_weights,
        ),
        "plane_count": len(transformed),
    }


def global_face_compactness(
    views: Sequence[ViewData],
    poses: Sequence[np.ndarray],
    manhattan: dict,
    args,
) -> dict:
    """
    Evalúa si las mismas caras observadas desde múltiples poses colapsan sobre
    planos compactos.

    IMPORTANTE:
    no usa el signo de las normales para distinguir caras opuestas porque la
    orientación de una normal estimada puede invertirse entre vistas.

    En cada familia Manhattan:
      1. se proyectan los puntos sobre el eje correspondiente;
      2. se intenta separar hasta dos clusters 1D (caras opuestas);
      3. se mide el espesor robusto de cada cluster.

    Una unión en arco/herradura genera clusters gruesos incluso si las parejas
    locales tienen buen point-to-plane.
    """
    if not manhattan.get("available"):
        return {
            "available": False,
            "reason": "Manhattan no disponible",
        }

    axes_raw = manhattan.get("frame_axes_xyz", [])
    if len(axes_raw) != 3:
        return {
            "available": False,
            "reason": "marco Manhattan incompleto",
        }

    axes = [normalize(np.asarray(a, dtype=np.float64)) for a in axes_raw]
    if any(a is None for a in axes):
        return {
            "available": False,
            "reason": "ejes Manhattan inválidos",
        }

    cos_gate = math.cos(math.radians(float(args.compactness_normal_gate_deg)))

    coords_by_family: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
    weights_by_family: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}

    for view, T in zip(views, poses):
        p = transform_points(view.registration_points, T)
        n = transform_normals(view.registration_normals, T)
        w = np.asarray(view.registration_weights, dtype=np.float64)

        if not (len(p) == len(n) == len(w)):
            continue

        dots = np.column_stack([n @ axis for axis in axes])
        family = np.argmax(np.abs(dots), axis=1)
        best_abs = np.max(np.abs(dots), axis=1)
        valid = best_abs >= cos_gate

        for family_index in range(3):
            selector = valid & (family == family_index)
            if not np.any(selector):
                continue
            coords_by_family[family_index].append(
                (p[selector] @ axes[family_index]).astype(np.float64)
            )
            weights_by_family[family_index].append(np.clip(w[selector], 0.02, 1.0))

    def weighted_center(values, weights):
        return weighted_median(values, weights)

    def split_one_or_two_faces(coord, weight):
        """
        Clustering 1D robusto, máximo dos caras.
        No depende del signo de la normal.
        """
        coord = np.asarray(coord, dtype=np.float64)
        weight = np.asarray(weight, dtype=np.float64)

        if len(coord) < 2 * int(args.compactness_minimum_points_per_face):
            return [(coord, weight)]

        q25, q75 = np.percentile(coord, [25.0, 75.0])
        c = np.array([q25, q75], dtype=np.float64)

        if abs(c[1] - c[0]) < float(args.compactness_min_face_separation_mm):
            return [(coord, weight)]

        assignment = np.zeros(len(coord), dtype=np.int32)
        for _ in range(20):
            d0 = np.abs(coord - c[0])
            d1 = np.abs(coord - c[1])
            new_assignment = (d1 < d0).astype(np.int32)

            if np.count_nonzero(new_assignment == 0) < int(
                args.compactness_minimum_points_per_face
            ) or np.count_nonzero(new_assignment == 1) < int(
                args.compactness_minimum_points_per_face
            ):
                return [(coord, weight)]

            new_c = np.array(
                [
                    weighted_center(
                        coord[new_assignment == k],
                        weight[new_assignment == k],
                    )
                    for k in (0, 1)
                ],
                dtype=np.float64,
            )

            if np.allclose(new_c, c, atol=1e-3):
                assignment = new_assignment
                c = new_c
                break

            assignment = new_assignment
            c = new_c

        separation = abs(float(c[1] - c[0]))
        if separation < float(args.compactness_min_face_separation_mm):
            return [(coord, weight)]

        result = []
        for k in (0, 1):
            selector = assignment == k
            if np.count_nonzero(selector) >= int(args.compactness_minimum_points_per_face):
                result.append((coord[selector], weight[selector]))
        return result if result else [(coord, weight)]

    records = []
    p90_values = []
    p95_values = []

    for family_index in range(3):
        if not coords_by_family[family_index]:
            continue

        coord = np.concatenate(coords_by_family[family_index])
        weight = np.concatenate(weights_by_family[family_index])

        if len(coord) < int(args.compactness_minimum_points_per_face):
            continue

        clusters = split_one_or_two_faces(coord, weight)
        cluster_centers = [weighted_center(c, w) for c, w in clusters]
        order = np.argsort(cluster_centers)

        for cluster_rank, cluster_index in enumerate(order):
            ccoord, cweight = clusters[int(cluster_index)]
            center_coord = weighted_center(ccoord, cweight)
            deviation = np.abs(ccoord - center_coord)

            record = {
                "axis_family": int(family_index),
                "face_cluster": int(cluster_rank),
                "count": int(len(ccoord)),
                "plane_coordinate_median_mm": float(center_coord),
                "deviation_mm": robust_stats(deviation),
            }
            records.append(record)

            if record["deviation_mm"]["p90"] is not None:
                p90_values.append(record["deviation_mm"]["p90"])
            if record["deviation_mm"]["p95"] is not None:
                p95_values.append(record["deviation_mm"]["p95"])

    if not records:
        return {
            "available": False,
            "reason": "ninguna familia alcanzó soporte suficiente",
        }

    return {
        "available": True,
        "face_count": len(records),
        "faces": records,
        "face_p90_mm": robust_stats(p90_values),
        "face_p95_mm": robust_stats(p95_values),
        "worst_face_p90_mm": float(max(p90_values)) if p90_values else None,
        "worst_face_p95_mm": float(max(p95_values)) if p95_values else None,
        "normal_gate_deg": float(args.compactness_normal_gate_deg),
        "minimum_face_separation_mm": float(args.compactness_min_face_separation_mm),
        "note": (
            "Las caras opuestas se separan por clustering 1D de coordenadas; "
            "no se confía en el signo de las normales."
        ),
    }


# ---------------------------------------------------------------------------
# Validación final
# ---------------------------------------------------------------------------


def graph_connected(view_count: int, edge_records: Sequence[dict]) -> bool:
    """Comprueba la conectividad del grafo de vistas con las aristas aceptadas."""
    adjacency = [set() for _ in range(view_count)]
    for r in edge_records:
        if not r.get("accepted", False):
            continue
        a, b = int(r["source_index"]), int(r["target_index"])
        adjacency[a].add(b)
        adjacency[b].add(a)
    visited = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nxt in adjacency[node]:
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    return len(visited) == view_count


def final_point_plane_metrics(
    edges: Sequence[EdgeSpec],
    views: Sequence[ViewData],
    samples: Dict[int, RegistrationSample],
    poses: Sequence[np.ndarray],
    args,
) -> Dict[Tuple[int, int], dict]:
    batches, _ = build_correspondence_batches(
        edges,
        views,
        samples,
        poses,
        float(args.final_overlap_distance_mm),
        args,
        round_index=999,
    )
    grouped: Dict[Tuple[int, int], List[np.ndarray]] = {}
    for batch in batches:
        T = relative_from_poses(poses[batch.source_index], poses[batch.target_index])
        transformed = transform_points(batch.src_points, T)
        diff = transformed - batch.tgt_points
        residual = np.sum(diff * batch.tgt_normals, axis=1)
        key = tuple(sorted((batch.source_index, batch.target_index)))
        grouped.setdefault(key, []).append(residual)
    result = {}
    for key, arrays in grouped.items():
        r = np.concatenate(arrays) if arrays else np.empty(0, np.float64)
        r = r[np.isfinite(r)]
        if len(r) == 0:
            result[key] = {"count": 0, "rmse_mm": None, "median_abs_mm": None, "p90_abs_mm": None}
        else:
            a = np.abs(r)
            result[key] = {
                "count": int(len(r)),
                "rmse_mm": float(np.sqrt(np.mean(r * r))),
                "median_abs_mm": float(np.median(a)),
                "p90_abs_mm": float(np.percentile(a, 90)),
            }
    return result


def final_edge_evaluation(
    edges: Sequence[EdgeSpec],
    views: Sequence[ViewData],
    samples: Dict[int, RegistrationSample],
    poses: Sequence[np.ndarray],
    args,
) -> List[dict]:
    records = edge_metrics(
        edges,
        views,
        samples,
        poses,
        args,
        overlap_distance=float(args.final_overlap_distance_mm),
    )
    p2plane = final_point_plane_metrics(edges, views, samples, poses, args)
    for record in records:
        key = tuple(sorted((int(record["source_index"]), int(record["target_index"]))))
        record["point_plane"] = p2plane.get(
            key, {"count": 0, "rmse_mm": None, "median_abs_mm": None, "p90_abs_mm": None}
        )
        pp_rmse = record["point_plane"].get("rmse_mm")
        pp_p90 = record["point_plane"].get("p90_abs_mm")

        pp_rmse_ok = pp_rmse is not None and pp_rmse <= float(
            args.final_primary_maximum_point_plane_rmse_mm
        )
        pp_p90_ok = pp_p90 is not None and pp_p90 <= float(
            args.final_primary_maximum_point_plane_p90_mm
        )

        # V3.1: criterio principal = consistencia de superficie.
        surface_accepted = bool(
            record["overlap"] >= float(args.final_primary_minimum_overlap)
            and pp_rmse_ok
            and pp_p90_ok
        )

        # El error euclídeo queda como diagnóstico tangencial.
        tangential_accepted = bool(
            record["trimmed_rmse_mm"] <= float(args.final_primary_maximum_rmse_mm)
            and record["p90_mm"] <= float(args.final_primary_maximum_p90_mm)
        )
        tangential_warning = bool(record["p90_mm"] > float(args.tangential_warning_p90_mm))

        record["surface_accepted"] = surface_accepted
        record["tangential_accepted"] = tangential_accepted
        record["tangential_warning"] = tangential_warning
        record["accepted"] = surface_accepted
    return records


def determine_quality(
    views: Sequence[ViewData],
    final_edges: Sequence[dict],
    manhattan: dict,
    compactness: dict,
    corrections: np.ndarray,
    angular_constraints: Sequence[AngularConstraint],
    axis: np.ndarray,
    center: np.ndarray,
    union_extent_ratio: float,
    args,
) -> Tuple[str, List[str], dict]:
    """Combina las métricas del registro para decidir calidad y motivos."""
    primary = [r for r in final_edges if r["primary"]]
    accepted_primary = [r for r in primary if r["accepted"]]
    closure = next((r for r in primary if r["closure"]), None)
    ratio = len(accepted_primary) / max(len(primary), 1)
    overlap_stats = robust_stats(r["overlap"] for r in primary)
    rmse_stats = robust_stats(r["trimmed_rmse_mm"] for r in primary)
    p90_stats = robust_stats(r["p90_mm"] for r in primary)
    correction_abs = np.abs(np.asarray(corrections, dtype=np.float64))
    correction_stats = robust_stats(correction_abs)
    camera_vertical_angle = acute_angle_deg(axis, np.array([0.0, 1.0, 0.0]))
    connected = graph_connected(len(views), final_edges)
    mounting = mounting_geometry_diagnostics(axis, center, args)
    angular_final = angular_model_final_diagnostics(
        angular_constraints,
        corrections,
        args,
    )

    pp_rmse_stats = robust_stats(
        r.get("point_plane", {}).get("rmse_mm")
        for r in primary
        if r.get("point_plane", {}).get("rmse_mm") is not None
    )
    pp_p90_stats = robust_stats(
        r.get("point_plane", {}).get("p90_abs_mm")
        for r in primary
        if r.get("point_plane", {}).get("p90_abs_mm") is not None
    )
    tangential_warning_count = sum(bool(r.get("tangential_warning", False)) for r in primary)

    reasons = []
    reject = []
    if not connected:
        reject.append("El grafo de aristas aceptadas no conecta las 25 poses.")
    if ratio < float(args.reject_primary_acceptance):
        reject.append(f"Aceptación primaria crítica: {ratio:.1%}.")
    elif ratio < float(args.warning_primary_acceptance):
        reasons.append(f"Aceptación primaria moderada: {ratio:.1%}.")

    if overlap_stats["median"] is not None:
        if overlap_stats["median"] < float(args.reject_median_primary_overlap):
            reject.append(f"Solape primario mediano crítico: {overlap_stats['median']:.3f}.")
        elif overlap_stats["median"] < float(args.warning_median_primary_overlap):
            reasons.append(f"Solape primario mediano bajo: {overlap_stats['median']:.3f}.")

    if rmse_stats["median"] is not None:
        if rmse_stats["median"] > float(args.reject_median_primary_rmse_mm):
            reject.append(f"RMSE primario mediano crítico: {rmse_stats['median']:.2f} mm.")
        elif rmse_stats["median"] > float(args.warning_median_primary_rmse_mm):
            reasons.append(f"RMSE primario mediano alto: {rmse_stats['median']:.2f} mm.")

    if closure is None:
        reject.append("No existe arista de cierre P24->P00.")
    else:
        closure_rmse = float(closure["trimmed_rmse_mm"])
        if closure_rmse > float(args.reject_closure_rmse_mm):
            reject.append(
                f"Cierre crítico: overlap={closure['overlap']:.3f}, RMSE={closure_rmse:.2f} mm."
            )
        elif (not closure["accepted"]) or closure_rmse > float(args.warning_closure_rmse_mm):
            reasons.append(
                f"Cierre débil: overlap={closure['overlap']:.3f}, RMSE={closure_rmse:.2f} mm."
            )

    # V3.2: no se rechaza por un máximo angular global. Se compara cada pose
    # contra SU bound adaptativo y contra su observabilidad.
    total_saturated = int(angular_final["saturated_count"])
    high_saturated = int(angular_final["high_observability_saturated_count"])

    if total_saturated >= int(args.reject_total_angular_saturations):
        reject.append(
            f"Demasiadas poses saturaron su bound angular adaptativo: "
            f"{total_saturated}/{len(views)}."
        )
    elif total_saturated >= int(args.warning_total_angular_saturations):
        reasons.append(
            f"{total_saturated}/{len(views)} poses tocaron su bound angular " "adaptativo."
        )

    if high_saturated >= int(args.reject_high_observability_saturations):
        reject.append(f"{high_saturated} poses de alta observabilidad saturaron su bound.")
    elif high_saturated >= int(args.warning_high_observability_saturations):
        reasons.append(f"{high_saturated} pose(s) de alta observabilidad tocaron su bound.")

    prior_stats = angular_final.get(
        "prior_residual_abs_deg_reliable",
        {},
    )
    max_prior_residual = prior_stats.get("max")
    if max_prior_residual is not None:
        if max_prior_residual > float(args.reject_angular_prior_residual_deg):
            reject.append(
                f"Corrección contradice prior angular fiable: " f"máx={max_prior_residual:.3f}°."
            )
        elif max_prior_residual > float(args.warning_angular_prior_residual_deg):
            reasons.append(
                f"Discrepancia moderada con prior angular de normales: "
                f"máx={max_prior_residual:.3f}°."
            )

    if manhattan.get("available"):
        p90_manhattan = manhattan.get(
            "weighted_normal_residual_deg",
            manhattan.get("normal_residual_deg", {}),
        ).get("p90")
        if p90_manhattan is not None:
            if p90_manhattan > float(args.reject_manhattan_p90_deg):
                reject.append(f"Consistencia Manhattan P90 muy alta: {p90_manhattan:.2f}°.")
            elif p90_manhattan > float(args.warning_manhattan_p90_deg):
                reasons.append(f"Consistencia Manhattan P90 alta: {p90_manhattan:.2f}°.")

    if union_extent_ratio > float(args.reject_union_extent_ratio):
        reject.append(f"La unión se expandió excesivamente: ratio={union_extent_ratio:.2f}.")
    elif union_extent_ratio > float(args.warning_union_extent_ratio):
        reasons.append(f"La unión presenta expansión geométrica: ratio={union_extent_ratio:.2f}.")

    if camera_vertical_angle > float(args.axis_camera_vertical_max_deg):
        reasons.append(
            f"Eje final muy inclinado respecto a Y de cámara: {camera_vertical_angle:.2f}°."
        )

    # Conformidad con montaje (prior aproximado, tolera 2-3 cm).
    mount_distance_error = abs(float(mounting["midpoint_axis_distance_error_mm"]))
    mount_symmetry_error = abs(float(mounting["left_right_symmetry_error_mm"]))
    mount_perp_error = abs(float(mounting["axis_baseline_perpendicular_deviation_deg"]))

    if mount_distance_error > float(args.reject_mount_distance_error_mm):
        reject.append(
            f"Eje incompatible con distancia física del montaje: error={mount_distance_error:.1f} mm."
        )
    elif mount_distance_error > float(args.warning_mount_distance_error_mm):
        reasons.append(
            f"Distancia cámara-eje fuera del prior suave: error={mount_distance_error:.1f} mm."
        )

    if mount_symmetry_error > float(args.reject_mount_symmetry_error_mm):
        reject.append(f"Simetría izquierda/derecha crítica: {mount_symmetry_error:.1f} mm.")
    elif mount_symmetry_error > float(args.warning_mount_symmetry_error_mm):
        reasons.append(f"Simetría izquierda/derecha moderada: {mount_symmetry_error:.1f} mm.")

    if mount_perp_error > float(args.reject_axis_baseline_deviation_deg):
        reject.append(f"Eje no perpendicular al baseline: desviación={mount_perp_error:.2f}°.")
    elif mount_perp_error > float(args.warning_axis_baseline_deviation_deg):
        reasons.append(f"Perpendicularidad eje/baseline moderada: {mount_perp_error:.2f}°.")

    # Compactación global de caras: defensa contra arco/herradura.
    worst_face_p90 = compactness.get("worst_face_p90_mm") if compactness.get("available") else None
    if worst_face_p90 is None:
        reasons.append("No se pudo evaluar compactación global de caras.")
    elif worst_face_p90 > float(args.reject_global_face_p90_mm):
        reject.append(f"Caras globales no compactan: peor P90={worst_face_p90:.2f} mm.")
    elif worst_face_p90 > float(args.warning_global_face_p90_mm):
        reasons.append(f"Compactación global moderada: peor P90={worst_face_p90:.2f} mm.")

    if tangential_warning_count > max(3, int(0.30 * len(primary))):
        reasons.append(
            f"{tangential_warning_count}/{len(primary)} aristas presentan P90 tangencial alto."
        )

    if reject:
        quality = "rejected"
        reasons = reject + reasons
    elif reasons:
        quality = "warning"
    else:
        quality = "accepted"

    metrics = {
        "primary_count": len(primary),
        "accepted_primary_count": len(accepted_primary),
        "primary_acceptance_ratio": float(ratio),
        "primary_overlap": overlap_stats,
        "primary_trimmed_rmse_mm": rmse_stats,
        "primary_p90_mm": p90_stats,
        "primary_point_plane_rmse_mm": pp_rmse_stats,
        "primary_point_plane_p90_abs_mm": pp_p90_stats,
        "tangential_warning_count": int(tangential_warning_count),
        "closure": closure,
        "angle_correction_abs_deg": correction_stats,
        "angular_model": angular_final,
        "camera_vertical_axis_angle_deg": float(camera_vertical_angle),
        "graph_connected": bool(connected),
        "union_extent_ratio": float(union_extent_ratio),
        "mounting_geometry": mounting,
        "global_face_compactness": compactness,
    }
    return quality, reasons, metrics


# ---------------------------------------------------------------------------
# Export / preview
# ---------------------------------------------------------------------------


def projection_panel(points, colors, axis_x, axis_y, width, height, title):
    canvas = np.full((height, width, 3), 255, np.uint8)
    if len(points) == 0:
        cv2.putText(canvas, title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        return canvas
    p = np.asarray(points, float)
    c = np.asarray(colors, np.uint8)[:, ::-1]  # RGB->BGR
    x, y = p[:, axis_x], p[:, axis_y]
    x0, x1 = np.percentile(x, [0.5, 99.5])
    y0, y1 = np.percentile(y, [0.5, 99.5])
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= y0:
        y1 = y0 + 1.0
    px = (20 + (x - x0) / (x1 - x0) * (width - 40)).astype(np.int32)
    py = (20 + (1.0 - (y - y0) / (y1 - y0)) * (height - 40)).astype(np.int32)
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    idx = np.flatnonzero(valid)
    if len(idx) > 180000:
        rng = np.random.default_rng(500)
        idx = rng.choice(idx, 180000, replace=False)
    canvas[py[idx], px[idx]] = c[idx]
    cv2.putText(
        canvas, title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 25, 25), 2, cv2.LINE_AA
    )
    return canvas


def make_preview(
    points: np.ndarray,
    view_colors: np.ndarray,
    quality: str,
    metrics: dict,
    manhattan: dict,
    path: Path,
    args,
) -> None:
    w, h = int(args.preview_width), int(args.preview_height)
    xy = projection_panel(points, view_colors, 0, 1, w, h, "X-Y")
    xz = projection_panel(points, view_colors, 0, 2, w, h, "X-Z")
    zy = projection_panel(points, view_colors, 2, 1, w, h, "Z-Y")
    info = np.full((h, w, 3), 248, np.uint8)
    lines = [
        f"08 V3.2.3 | calidad: {quality}",
        f"aristas primarias: {metrics['accepted_primary_count']}/{metrics['primary_count']} ({metrics['primary_acceptance_ratio']:.1%})",
        f"overlap mediano: {metrics['primary_overlap']['median']}",
        f"RMSE mediano: {metrics['primary_trimmed_rmse_mm']['median']} mm",
        f"cierre RMSE: {None if metrics['closure'] is None else metrics['closure']['trimmed_rmse_mm']} mm",
        f"max correccion angular: {metrics['angle_correction_abs_deg']['max']} deg",
        f"saturaciones adaptativas: {metrics.get('angular_model', {}).get('saturated_count')}",
        f"eje vs Y camara: {metrics['camera_vertical_axis_angle_deg']:.2f} deg",
        f"union extent ratio: {metrics['union_extent_ratio']:.2f}",
        (
            "Manhattan P90 ponderado: "
            f"{manhattan.get('weighted_normal_residual_deg', manhattan.get('normal_residual_deg', {})).get('p90') if manhattan.get('available') else None}"
        ),
        f"dist eje-midpoint: {metrics.get('mounting_geometry', {}).get('midpoint_axis_distance_mm')} mm",
        f"compactacion peor P90: {metrics.get('global_face_compactness', {}).get('worst_face_p90_mm')} mm",
        "poses: eje/centro unicos; sin ICP 6DoF libre",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            info,
            str(line),
            (25, 45 + 42 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.67,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
    canvas = np.vstack([np.hstack([xy, xz]), np.hstack([zy, info])])
    cv2.imwrite(str(path), canvas)


def save_axis_ply(path: Path, axis: np.ndarray, center: np.ndarray) -> None:
    t = np.linspace(-180.0, 180.0, 1200)
    points = center[None, :] + t[:, None] * axis[None, :]
    colors = np.tile(np.array([[25, 25, 25]], np.uint8), (len(points), 1))
    save_ply_ascii(path, points.astype(np.float32), colors)


@operacion("Exportar vistas registradas")
def export_registered(
    views: Sequence[ViewData],
    poses: Sequence[np.ndarray],
    output: Path,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict], float]:
    """Exporta las nubes transformadas y los productos del registro de referencia."""
    reg_dir = output / "nubes_registradas"
    reg_dir.mkdir(parents=True, exist_ok=True)
    all_points, all_rgb, all_view_colors = [], [], []
    pose_points, pose_colors = [], []
    records = []
    individual_extents = []

    for view, T in zip(views, poses):
        p = transform_points(view.points, T)
        n = (
            transform_normals(view.normals, T)
            if view.normals.shape == view.points.shape
            else np.empty((0, 3), np.float64)
        )
        reg_p = transform_points(view.registration_points, T)
        ply = reg_dir / f"{view.stem}_registered.ply"
        npz = reg_dir / f"{view.stem}_registered.npz"
        save_ply_ascii(
            ply,
            p.astype(np.float32),
            view.colors.astype(np.uint8),
            n.astype(np.float32) if n.shape == p.shape else None,
        )
        np.savez_compressed(
            npz,
            points=p.astype(np.float32),
            colors=view.colors.astype(np.uint8),
            normals=n.astype(np.float32) if n.shape == p.shape else np.empty((0, 3), np.float32),
            transform_to_reference=np.asarray(T, dtype=np.float64),
            pose_index=np.asarray([view.pose_index], dtype=np.int32),
            physical_angle_deg=np.asarray([view.angle_deg], dtype=np.float64),
        )
        view_color = np.tile(hue_color_rgb(view.angle_deg), (len(p), 1))
        all_points.append(p)
        all_rgb.append(view.colors)
        all_view_colors.append(view_color)
        pose_points.append(reg_p)
        pose_colors.append(np.tile(hue_color_rgb(view.angle_deg), (len(reg_p), 1)))
        individual_extents.append(robust_extent(p))
        records.append(
            {
                "pose_index": view.pose_index,
                "physical_angle_deg": view.angle_deg,
                "stem": view.stem,
                "view_weight": view.view_weight,
                "transform_to_reference": T.tolist(),
                "ply": str(ply),
                "npz": str(npz),
            }
        )

    full_points = np.vstack(all_points)
    full_rgb = np.vstack(all_rgb)
    full_view_colors = np.vstack(all_view_colors)
    reg_points = np.vstack(pose_points)
    reg_colors = np.vstack(pose_colors)

    p_rgb, c_rgb = voxel_downsample_with_colors(full_points, full_rgb, float(args.union_voxel_mm))
    save_ply_ascii(output / "union_registrada_rgb_voxel.ply", p_rgb.astype(np.float32), c_rgb)
    p_view, c_view = voxel_downsample_with_colors(
        full_points, full_view_colors, float(args.union_voxel_mm)
    )
    save_ply_ascii(
        output / "union_registrada_colores_por_vista.ply", p_view.astype(np.float32), c_view
    )
    p_pose, c_pose = voxel_downsample_with_colors(
        reg_points, reg_colors, float(args.union_voxel_mm)
    )
    save_ply_ascii(
        output / "union_geometria_usada_para_pose.ply", p_pose.astype(np.float32), c_pose
    )

    union_extent = robust_extent(full_points)
    median_individual = np.median(np.asarray(individual_extents), axis=0)
    ratios = union_extent / np.maximum(median_individual, 1e-6)
    union_extent_ratio = float(np.nanmax(ratios))
    return full_points, full_view_colors, reg_points, records, union_extent_ratio


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Estima el registro del patrón y exporta la evidencia de calibración."""
    args = build_parser().parse_args()
    root = Path(args.root).expanduser().resolve()
    object_name = args.object.strip().lower()
    session = args.session.strip().upper()
    recon = root / "reconstruccion" / session
    cloud_dir = recon / args.cloud_source
    geometry_dir = recon / args.geometry_source
    output = recon / args.output_name

    cloud_summary_path = find_summary(
        cloud_dir,
        ("resumen_06_nubes_puntos.json",),
    )
    geometry_summary_path = find_summary(
        geometry_dir,
        ("resumen_07_validacion_geometrica.json",),
    )
    cloud_summary = load_json(cloud_summary_path)
    geometry_summary = load_json(geometry_summary_path)

    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else default_manifest_path(root, object_name)
    )
    manifest = load_json(manifest_path) if manifest_path.is_file() else None
    campaign_diag = validate_campaign(cloud_summary, manifest, args)

    views = load_views(cloud_dir, geometry_dir, cloud_summary, geometry_summary, args)
    prepare_output(output, bool(args.clean_output))
    edges = build_edges(views, args)
    primary_count = sum(e.primary for e in edges)

    print("\n========== PASO 08 V3.2.3: REGISTRO MULTIVISTA RESTRINGIDO ==========")
    print(f"Objeto/sesión: {object_name} / {session}")
    print(f"Vistas: {len(views)} | aristas: {len(edges)} ({primary_count} primarias)")
    print("Modelo: eje/centro únicos + ángulos físicos + observabilidad angular por pose.")
    print("Estimación de pose: planos 07 ponderados; NO ICP 6DoF libre.\n")

    # Auditoría de selección de planos.
    plane_audit = []
    for view in views:
        plane_audit.append(
            {
                "pose_index": view.pose_index,
                "physical_angle_deg": view.angle_deg,
                "quality_step_07": view.quality_step_07,
                "view_weight": view.view_weight,
                "registration_points": int(len(view.registration_points)),
                "planes": [d.__dict__ for d in view.plane_decisions],
            }
        )
    save_json(output / "auditoria_geometria_pose.json", {"views": plane_audit})

    print("[1/10] Eje inicial por consenso robusto de planos...")
    axis_raw, axis_diag = estimate_axis_from_planes(views, args)
    print("      eje bruto =", np.array2string(axis_raw, precision=7))
    print(
        f"      método = {axis_diag.get('method', 'strict_pair_consensus')} | "
        f"soporte estricto = {axis_diag.get('support_view_count', 0)} vistas | "
        f"eje-vs-Y={axis_diag['camera_vertical_angle_deg']:.2f}°"
    )
    if axis_diag.get("fallback_used", False):
        fb = axis_diag.get("fallback", {})
        print(
            f"      fallback multivista ACTIVO | "
            f"semillas={fb.get('seed_count')} | "
            f"normales={fb.get('normal_count')} | "
            f"score={fb.get('selected_score')}"
        )

    # Las muestras gruesas se construyen una sola vez y alimentan tanto el
    # camino histórico como el rescate geométrico V3.2.2.
    coarse_samples = build_samples(
        views,
        args,
        int(args.coarse_max_points_per_view),
    )

    cloud_init_diag = {
        "available": False,
        "reason": "no requerido",
    }
    used_cloud_joint_initializer = False

    if bool(args.axis_cloud_fallback_enabled) and bool(axis_diag.get("fallback_used", False)):
        print(
            "\n[2/10] Validando fallback con inicialización conjunta "
            "eje + centro + sentido sobre nubes..."
        )

        axis_cloud, center_cloud, sign_cloud, cloud_init_diag = (
            joint_cloud_axis_center_initialization(
                views,
                edges,
                coarse_samples,
                axis_raw,
                args,
            )
        )

        if (
            axis_cloud is not None
            and center_cloud is not None
            and sign_cloud is not None
            and cloud_init_diag.get("available", False)
        ):
            used_cloud_joint_initializer = True
            axis0 = axis_cloud
            center0 = center_cloud
            sign = int(sign_cloud)

            center_seed_raw, center_seed_diag_raw = harmonic_center_seed(
                views,
                axis_raw,
            )
            center_sign_raw = center0.copy()
            center_seed, center_seed_diag = harmonic_center_seed(
                views,
                axis0,
            )
            center_coarse = center0.copy()

            sign_diag = {
                "method": "joint_cloud_axis_center_sign",
                "chosen_sign": int(sign),
                "chosen_score": float(
                    cloud_init_diag.get(
                        "selected_score",
                        float("nan"),
                    )
                ),
                "score_margin": None,
                "candidates": {
                    str(t.get("sign")): t
                    for t in cloud_init_diag.get("trials", [])
                    if t.get("success", False)
                },
            }
            sign_refined_diag = dict(sign_diag)
            center_coarse_diag = {
                "method": "joint_cloud_axis_center_sign",
                "score": float(
                    cloud_init_diag.get(
                        "selected_score",
                        float("nan"),
                    )
                ),
                "center_xyz_mm": center0.tolist(),
            }
            center_local_diag = {
                "method": ("joint_cloud_initializer_no_legacy_center_pullback"),
                "center_xyz_mm": center0.tolist(),
                "note": (
                    "No se ejecuta refine_center_local histórico después del "
                    "inicializador conjunto, porque ese objetivo primario "
                    "recortado podía volver a favorecer una solución en arco."
                ),
            }

            print(
                "      eje nube =",
                np.array2string(axis0, precision=7),
            )
            print(
                f"      cambio vs fallback de planos = "
                f"{cloud_init_diag['axis_change_from_seed_deg']:.3f}° | "
                f"signo={sign:+d}"
            )
            selected_trial = cloud_init_diag.get(
                "selected_trial",
                {},
            )
            score_diag = selected_trial.get(
                "score_diagnostics",
                {},
            )
            mount_cloud = score_diag.get(
                "mounting_geometry",
                mounting_geometry_diagnostics(
                    axis0,
                    center0,
                    args,
                ),
            )
            print(
                f"      overlap primario mediano previo = "
                f"{score_diag.get('primary_overlap_median')} | "
                f"distancia midpoint->eje="
                f"{mount_cloud.get('midpoint_axis_distance_mm'):.2f} mm"
            )

            print(
                "\n[3/10] Auditoría Manhattan del eje seleccionado " "(diagnóstico, no selector)..."
            )
            zero_corr = np.zeros(
                len(views),
                dtype=np.float64,
            )
            poses_pre = build_poses(
                views,
                axis0,
                center0,
                sign,
                zero_corr,
            )
            manhattan_pre = manhattan_diagnostics(
                views,
                poses_pre,
                axis0,
            )
            axis_multiview_diag = {
                "available": bool(manhattan_pre.get("available", False)),
                "method": ("cloud_guided_axis_no_manhattan_pullback"),
                "axis_seed_xyz": axis_raw.tolist(),
                "axis_final_xyz": axis0.tolist(),
                "axis_change_deg": acute_angle_deg(
                    axis_raw,
                    axis0,
                ),
                "normal_residual_deg": manhattan_pre.get(
                    "normal_residual_deg",
                    {},
                ),
                "weighted_normal_residual_deg": manhattan_pre.get(
                    "weighted_normal_residual_deg",
                    {},
                ),
                "frame_axes_xyz": manhattan_pre.get(
                    "frame_axes_xyz",
                    [],
                ),
                "note": (
                    "Como el consenso estricto falló, las normales de 07 "
                    "se usan para auditoría y observabilidad, pero no vuelven "
                    "a arrastrar el eje seleccionado por las nubes."
                ),
            }
            if manhattan_pre.get("available"):
                print(
                    f"      P90 Manhattan ponderado previo = "
                    f"{manhattan_pre.get('weighted_normal_residual_deg', {}).get('p90')}° "
                    f"| crudo="
                    f"{manhattan_pre.get('normal_residual_deg', {}).get('p90')}°"
                )

            print("\n[4/10] Centro inicial validado por nubes + montaje...")
            mount0 = mounting_geometry_diagnostics(
                axis0,
                center0,
                args,
            )
            print(
                "      centro inicial BA =",
                np.array2string(center0, precision=4),
            )
            print(
                f"      distancia midpoint->eje = "
                f"{mount0['midpoint_axis_distance_mm']:.2f} mm "
                f"(referencia {args.mount_midpoint_axis_distance_mm:.1f} ± "
                f"{args.mount_distance_sigma_mm:.1f} mm)"
            )

        else:
            print(
                "      El inicializador conjunto no produjo una solución "
                "finita; se conserva el camino V3.2.1."
            )

    if not used_cloud_joint_initializer:
        print("\n[2/10] Semilla inicial de centro + sentido de giro...")
        center_seed_raw, center_seed_diag_raw = harmonic_center_seed(
            views,
            axis_raw,
        )
        sign, center_sign_raw, sign_diag = choose_sign_and_center(
            axis_raw,
            center_seed_raw,
            edges,
            views,
            coarse_samples,
            args,
        )
        print(f"      signo = {sign:+d} | " f"margen={sign_diag['score_margin']:.4f}")

        print("\n[3/10] Refinando eje con normales desrotadas + " "ángulos físicos...")
        axis0, axis_multiview_diag = refine_axis_multiview_manhattan(
            views,
            axis_raw,
            sign,
            args,
        )
        print(
            "      eje multivista =",
            np.array2string(axis0, precision=7),
        )
        if axis_multiview_diag.get("available"):
            print(
                f"      cambio vs eje bruto = "
                f"{axis_multiview_diag['axis_change_deg']:.3f}° | "
                f"Manhattan previo P90="
                f"{axis_multiview_diag['normal_residual_deg']['p90']:.3f}°"
            )

        print("\n[4/10] Reestimando centro y verificando nuevamente " "el sentido...")
        center_seed, center_seed_diag = harmonic_center_seed(
            views,
            axis0,
        )
        sign_refined, center_coarse, sign_refined_diag = choose_sign_and_center(
            axis0,
            center_seed,
            edges,
            views,
            coarse_samples,
            args,
        )

        if int(sign_refined) != int(sign):
            print(
                f"      sentido cambió {sign:+d} -> {sign_refined:+d}; "
                "repitiendo refinamiento multivista del eje."
            )
            sign = int(sign_refined)
            axis0, axis_multiview_diag = refine_axis_multiview_manhattan(
                views,
                axis_raw,
                sign,
                args,
            )
            center_seed, center_seed_diag = harmonic_center_seed(
                views,
                axis0,
            )
            sign, center_coarse, sign_refined_diag = choose_sign_and_center(
                axis0,
                center_seed,
                edges,
                views,
                coarse_samples,
                args,
            )
        else:
            sign = int(sign_refined)

        center_coarse_diag = sign_refined_diag.get(
            "candidates",
            {},
        ).get(str(sign), {})
        center0, center_local_diag = refine_center_local(
            axis0,
            center_coarse,
            sign,
            edges,
            views,
            coarse_samples,
            args,
        )
        mount0 = mounting_geometry_diagnostics(
            axis0,
            center0,
            args,
        )
        print(
            "      centro inicial BA =",
            np.array2string(center0, precision=4),
        )
        print(
            f"      distancia midpoint->eje = "
            f"{mount0['midpoint_axis_distance_mm']:.2f} mm "
            f"(referencia {args.mount_midpoint_axis_distance_mm:.1f} ± "
            f"{args.mount_distance_sigma_mm:.1f} mm)"
        )

    print("\n[5/10] Construyendo observabilidad angular por pose...")
    frame_axes_for_angle = (
        axis_multiview_diag.get("frame_axes_xyz", [])
        if axis_multiview_diag.get("available")
        else []
    )
    angular_constraints, angular_diag = build_angular_constraints(
        views,
        axis0,
        sign,
        frame_axes_for_angle,
        args,
    )
    print(
        f"      observabilidad: low={angular_diag['low_count']} | "
        f"medium={angular_diag['medium_count']} | "
        f"high={angular_diag['high_count']}"
    )
    print(
        f"      límites adaptativos: "
        f"mediana={angular_diag['adaptive_limit_deg']['median']:.3f}° | "
        f"max={angular_diag['adaptive_limit_deg']['max']:.3f}°"
    )
    total_observability = max(
        1,
        int(angular_diag.get("low_count", 0))
        + int(angular_diag.get("medium_count", 0))
        + int(angular_diag.get("high_count", 0)),
    )
    low_observability_ratio = float(angular_diag.get("low_count", 0)) / float(total_observability)
    auto_stabilize_ba = bool(
        args.auto_stabilize_low_observability
        and low_observability_ratio >= float(args.auto_stabilize_low_ratio)
    )
    angular_diag["low_observability_ratio"] = float(low_observability_ratio)
    angular_diag["auto_stabilize_ba"] = bool(auto_stabilize_ba)

    save_json(
        output / "auditoria_observabilidad_angular_v3_2.json",
        {
            "summary": angular_diag,
            "views": [angular_constraint_to_dict(c) for c in angular_constraints],
        },
    )

    if auto_stabilize_ba:
        print(
            "      observabilidad baja dominante: "
            f"{100.0 * low_observability_ratio:.1f}% de poses low -> "
            "se activará BA tangencial estabilizado."
        )

    print("\n[6/10] BA restringido en dos etapas...")
    fine_samples = build_samples(views, args, int(args.fine_max_points_per_view))
    ba_args = args
    effective_ba_overrides = {}
    stabilize_reasons = []
    if used_cloud_joint_initializer:
        stabilize_reasons.append("fallback de eje validado por nubes")
    if auto_stabilize_ba:
        stabilize_reasons.append(
            "observabilidad angular baja dominante " f"({100.0 * low_observability_ratio:.1f}% low)"
        )

    if stabilize_reasons:
        ba_args = argparse.Namespace(**vars(args))

        tangential_target = float(args.ba_tangential_weight)
        mount_target = float(args.mount_prior_equivalent_points)
        center_target = float(args.center_prior_equivalent_points)

        if used_cloud_joint_initializer:
            tangential_target = max(
                tangential_target,
                float(args.fallback_ba_tangential_weight),
            )
            mount_target = max(
                mount_target,
                float(args.fallback_ba_mount_prior_equivalent_points),
            )
            center_target = max(
                center_target,
                float(args.fallback_ba_center_prior_equivalent_points),
            )

        if auto_stabilize_ba:
            tangential_target = max(
                tangential_target,
                float(args.auto_ba_tangential_weight),
            )
            mount_target = max(
                mount_target,
                float(args.auto_ba_mount_prior_equivalent_points),
            )
            center_target = max(
                center_target,
                float(args.auto_ba_center_prior_equivalent_points),
            )

        ba_args.ba_tangential_weight = tangential_target
        ba_args.mount_prior_equivalent_points = mount_target
        ba_args.center_prior_equivalent_points = center_target

        effective_ba_overrides = {
            "reason": "; ".join(stabilize_reasons),
            "low_observability_ratio": float(low_observability_ratio),
            "ba_tangential_weight": float(ba_args.ba_tangential_weight),
            "mount_prior_equivalent_points": float(ba_args.mount_prior_equivalent_points),
            "center_prior_equivalent_points": float(ba_args.center_prior_equivalent_points),
        }
        print(
            "      BA estabilizado: tangencial="
            f"{ba_args.ba_tangential_weight:.3f} | "
            "mount_eq="
            f"{ba_args.mount_prior_equivalent_points:.0f} | "
            "center_eq="
            f"{ba_args.center_prior_equivalent_points:.0f} | "
            f"razón={effective_ba_overrides['reason']}"
        )

    axis, center, corrections, poses, ba_diag = run_constrained_ba(
        views,
        edges,
        fine_samples,
        axis0,
        center0,
        sign,
        angular_constraints,
        ba_args,
    )
    mount_final = mounting_geometry_diagnostics(axis, center, args)
    print("      eje final =", np.array2string(axis, precision=7))
    print("      centro final =", np.array2string(center, precision=4))
    print(f"      max |corrección angular| = " f"{np.max(np.abs(corrections)):.4f}°")
    print(
        f"      distancia midpoint->eje = "
        f"{mount_final['midpoint_axis_distance_mm']:.2f} mm | "
        f"simetría L-R={mount_final['left_right_symmetry_error_mm']:.2f} mm"
    )

    print("\n[7/10] Evaluando aristas y cierre...")
    final_edges = final_edge_evaluation(edges, views, fine_samples, poses, args)
    primary = [r for r in final_edges if r["primary"]]
    closure = next((r for r in primary if r["closure"]), None)
    print(f"      primarias aceptadas = {sum(r['accepted'] for r in primary)}/{len(primary)}")
    if closure:
        print(
            f"      cierre P24->P00: overlap={closure['overlap']:.3f} | RMSE={closure['trimmed_rmse_mm']:.3f} mm"
        )

    print("\n[8/10] Diagnóstico Manhattan + compactación global...")
    manhattan = manhattan_diagnostics(views, poses, axis)
    if manhattan.get("available"):
        print(f"      P90 normales -> marco = {manhattan['normal_residual_deg']['p90']:.3f}°")

    compactness = global_face_compactness(views, poses, manhattan, args)
    if compactness.get("available"):
        print(
            f"      peor P90 de compactación de cara = "
            f"{compactness['worst_face_p90_mm']:.3f} mm"
        )

    print("\n[9/10] Exportando las 25 nubes registradas y uniones...")
    full_points, view_colors, pose_points, registered, union_extent_ratio = export_registered(
        views, poses, output, args
    )
    save_axis_ply(output / "eje_rotacion_diagnostico.ply", axis, center)

    print("\n[10/10] Clasificación global...")
    quality, reasons, metrics = determine_quality(
        views,
        final_edges,
        manhattan,
        compactness,
        corrections,
        angular_constraints,
        axis,
        center,
        union_extent_ratio,
        args,
    )
    metrics["view_count"] = len(views)
    metrics["pose_geometry_points"] = int(len(pose_points))
    print("      calidad =", quality)
    for reason in reasons:
        print("      -", reason)

    print("      guardando auditoría, CSV y preview...")
    make_preview(
        full_points,
        view_colors,
        quality,
        metrics,
        manhattan,
        output / "preview_union_registrada_v3_2.png",
        args,
    )

    with (output / "calidad_aristas_v3_2.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "source_pose",
            "target_pose",
            "source_angle_deg",
            "target_angle_deg",
            "nominal_delta_deg",
            "order",
            "primary",
            "closure",
            "base_weight",
            "history_weight",
            "overlap",
            "trimmed_rmse_mm",
            "p90_mm",
            "point_plane_rmse_mm",
            "point_plane_p90_abs_mm",
            "point_plane_count",
            "score",
            "surface_accepted",
            "tangential_accepted",
            "tangential_warning",
            "accepted",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in final_edges:
            row = {k: record.get(k) for k in fields}
            pp = record.get("point_plane", {})
            row["point_plane_rmse_mm"] = pp.get("rmse_mm")
            row["point_plane_p90_abs_mm"] = pp.get("p90_abs_mm")
            row["point_plane_count"] = pp.get("count")
            writer.writerow(row)

    with (output / "poses_registradas_v3_2.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "pose_index",
            "physical_angle_deg",
            "angle_correction_deg",
            "corrected_physical_angle_deg",
            "angular_observability_score",
            "angular_observability_class",
            "angular_limit_deg",
            "angular_sigma_deg",
            "angular_target_deg",
            "angular_normal_prior_reliability",
            "angular_saturation_ratio",
            "angular_saturated",
            "tx",
            "ty",
            "tz",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        angular_final_for_csv = angular_model_final_diagnostics(
            angular_constraints,
            corrections,
            args,
        )
        angular_by_pose = {int(r["pose_index"]): r for r in angular_final_for_csv["records"]}
        for view, correction, T in zip(views, corrections, poses):
            arow = angular_by_pose[int(view.pose_index)]
            writer.writerow(
                {
                    "pose_index": view.pose_index,
                    "physical_angle_deg": view.angle_deg,
                    "angle_correction_deg": float(correction),
                    "corrected_physical_angle_deg": float(view.angle_deg + correction),
                    "angular_observability_score": arow["observability_score"],
                    "angular_observability_class": arow["observability_class"],
                    "angular_limit_deg": arow["limit_deg"],
                    "angular_sigma_deg": arow["sigma_deg"],
                    "angular_target_deg": arow["target_correction_deg"],
                    "angular_normal_prior_reliability": arow["normal_prior_reliability"],
                    "angular_saturation_ratio": arow["saturation_ratio"],
                    "angular_saturated": arow["saturated"],
                    "tx": float(T[0, 3]),
                    "ty": float(T[1, 3]),
                    "tz": float(T[2, 3]),
                }
            )

    with (output / "observabilidad_angular_v3_2.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:
        fields = [
            "pose_index",
            "observability_score",
            "observability_class",
            "informative_plane_count",
            "distinct_lateral_family_count",
            "effective_lateral_support",
            "diversity_score",
            "normal_prior_raw_deg",
            "normal_prior_scatter_deg",
            "normal_prior_reliability",
            "target_correction_deg",
            "sigma_deg",
            "limit_deg",
            "prior_weight",
            "final_correction_deg",
            "saturation_ratio",
            "saturated",
            "equivalent_motor_steps",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        final_angle_diag = angular_model_final_diagnostics(
            angular_constraints,
            corrections,
            args,
        )
        for record in final_angle_diag["records"]:
            writer.writerow({k: record.get(k) for k in fields})

    summary = {
        "schema_version": 32,
        "implementation_patch": "3.2.3_adaptive_consensus_and_low_observability_ba_stabilization",
        "method": (
            "single_axis_single_center_physical_angles_mount_soft_prior_"
            "strict_axis_consensus_with_multiview_fallback_"
            "cloud_validated_joint_axis_center_sign_when_needed_"
            "weighted_manhattan_audit_observability_adaptive_angle_"
            "staged_point_to_plane_bundle_adjustment"
        ),
        "object": object_name,
        "session": session,
        "quality": quality,
        "reasons": reasons,
        "cloud_summary": str(cloud_summary_path),
        "geometry_summary": str(geometry_summary_path),
        "manifest": str(manifest_path) if manifest is not None else None,
        "campaign_validation": campaign_diag,
        "parameters": vars(args),
        "direction_sign": int(sign),
        "axis": {
            "raw_initial_xyz": axis_raw.tolist(),
            "multiview_initial_xyz": axis0.tolist(),
            "final_xyz": axis.tolist(),
            "raw_diagnostics": axis_diag,
            "multiview_diagnostics": axis_multiview_diag,
        },
        "center": {
            "raw_seed_xyz_mm": center_seed_raw.tolist(),
            "raw_sign_center_xyz_mm": center_sign_raw.tolist(),
            "seed_xyz_mm": center_seed.tolist(),
            "coarse_xyz_mm": center_coarse.tolist(),
            "final_xyz_mm": center.tolist(),
            "raw_seed_diagnostics": center_seed_diag_raw,
            "seed_diagnostics": center_seed_diag,
            "coarse_diagnostics": center_coarse_diag,
            "sign_diagnostics_initial": sign_diag,
            "sign_diagnostics_refined": sign_refined_diag,
            "local_refinement": center_local_diag,
            "mounting_geometry_initial_ba": mount0,
            "mounting_geometry_final": mount_final,
        },
        "angle_corrections_deg": {
            f"P{view.pose_index:02d}": float(corrections[i]) for i, view in enumerate(views)
        },
        "angular_observability": {
            "initial_diagnostics": angular_diag,
            "constraints": [angular_constraint_to_dict(c) for c in angular_constraints],
            "final": angular_model_final_diagnostics(
                angular_constraints,
                corrections,
                args,
            ),
        },
        "bundle_adjustment": ba_diag,
        "manhattan": manhattan,
        "global_face_compactness": compactness,
        "mounting_geometry": mount_final,
        "axis_center_cloud_initialization": cloud_init_diag,
        "effective_ba_overrides": effective_ba_overrides,
        "final_edge_evaluation": final_edges,
        "global_metrics": metrics,
        "registered_views": registered,
        "plane_pose_audit": plane_audit,
        "important_note": (
            "V3.2.3 mantiene un único eje y una única línea de rotación, sin "
            "poses 6DoF independientes. El montaje físico sigue siendo un prior "
            "suave con incertidumbre de 2–3 cm. La novedad principal es que la "
            "corrección angular ya no tiene el mismo bound en las 25 poses: "
            "cada vista recibe un límite, sigma y peso según la observabilidad "
            "real de sus planos 07. Las vistas con una sola cara dominante "
            "quedan fuertemente ancladas a los 2055 pasos; las vistas con dos "
            "caras laterales fuertes reciben más libertad. Las normales solo "
            "pueden desplazar débilmente el prior si dos o más planos dan una "
            "evidencia coherente. Se añade continuidad y curvatura angular para "
            "evitar picos aislados. La saturación de un bound adaptativo se "
            "audita según observabilidad y no produce rechazo automático por sí "
            "sola; siguen mandando cierre, point-to-plane, Manhattan, montaje y "
            "compactación global de caras."
        ),
    }
    save_json(output / "resumen_08_registro_referencia.json", summary)

    print("\n========== PASO 08 V3.2.3 COMPLETADO ==========")
    print("Salida:", output)
    print("Calidad global:", quality)
    print(
        f"Aristas primarias: {metrics['accepted_primary_count']}/{metrics['primary_count']} "
        f"({metrics['primary_acceptance_ratio']:.1%})"
    )
    print(f"Overlap primario mediano: {metrics['primary_overlap']['median']}")
    print(f"RMSE primario mediano: {metrics['primary_trimmed_rmse_mm']['median']} mm")
    print(f"Max corrección angular: {metrics['angle_correction_abs_deg']['max']}°")
    if manhattan.get("available"):
        print(
            "Manhattan P90 ponderado: "
            f"{manhattan.get('weighted_normal_residual_deg', manhattan['normal_residual_deg'])['p90']}° "
            f"| crudo={manhattan['normal_residual_deg']['p90']}°"
        )
    print("REVISAR preview_union_registrada_v3_2.png antes de continuar a la siguiente paso.")
    print("============================================================\n")
    return 0


def _entrada_con_diagnostico():
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    from utilidades_progreso import informar_inicio

    informar_inicio(__file__, "08", "Registrar vistas de referencia")
    raise SystemExit(main())


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
