#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Paso 03 V5.3 — Silueta visual conservadora + exclusión del hardware visible + recuperación local del contacto por evidencia estéreo.

Objetivo
--------
Responder únicamente:

    ¿Qué píxeles pertenecen VISUALMENTE al objeto?

Este paso NO utiliza:
- nube de puntos;
- forma conocida;
- caras;
- dimensiones del objeto;
- máscara preliminar de CREStereo como autoridad global.

La disparidad se usa SOLO dentro de la banda local de contacto objeto-soporte,
como verificación de oclusión respecto al mismo fondo vacío. Nunca se usa para
detectar foreground en paredes, mesa o resto de la escena.

Sí utiliza:
- imagen izquierda rectificada;
- fondo vacío rectificado único;
- diferencia Lab;
- cromaticidad;
- luminancia con signo;
- diferencia de gradientes;
- dominio completo válido de rectificación;
- anchor visual adaptativo respecto del fondo vacío;
- sombra conservadora por conectividad estructural;
- dominio físico del soporte sin tratarlo automáticamente como fondo;
- banda ambigua de contacto objeto-soporte derivada de la geometría de imagen;
- comparación robusta contra la captura de fondo vacío como evidencia, no como veto;
- conectividad geodésica desde el objeto ya detectado;
- recuperación geodésica local del contacto, sin GraphCut de soporte;
- bloqueo del plato únicamente fuera de la banda de contacto;
- histéresis;
- morfología pequeña;
- selección espacial de componentes;
- relleno limitado de huecos.

Compatible con nombres V7.2:
    A0000 -> 0.0°
    A0144 -> 14.4°
    ...
    A3456 -> 345.6°
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
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utilidades_mascaras import (
    adaptive_visual_anchor,
    build_contact_sheet,
    build_support_hardware_exclusion_mask,
    build_support_rim_guard_mask,
    capture_volume_from_support,
    expanded_bbox_mask,
    fill_small_holes,
    graphcut_recover_foreground,
    graphcut_trimap_visualization,
    find_summary,
    finite_stats,
    imwrite_checked,
    load_json,
    odd,
    overlay_mask,
    parse_angle,
    prepare_output_directory,
    robust_location_scale,
    save_json,
    safe_rectification_domain,
    scalar_visualization,
    select_relevant_components,
    shadow_aware_hysteresis,
    validate_summary_context,
)

# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construye las opciones de línea de comandos de este paso."""
    p = argparse.ArgumentParser(
        description=(
            "Máscara V5.2: cuerpo visual conservador + exclusión del hardware + recuperación estéreo "
            "local únicamente en contacto con plataforma."
        )
    )

    p.add_argument("--root", required=True)
    p.add_argument("--object", default="cubo")
    p.add_argument("--session", default="S01")
    p.add_argument(
        "--depth-source",
        default="02_estimacion_profundidad",
        help=(
            "Solo se usa esta carpeta para leer la imagen izquierda "
            "rectificada, el fondo rectificado y rect_valid_mask."
        ),
    )
    p.add_argument(
        "--output-name",
        default="03_mascara_objeto",
        help=("Se mantiene el nombre histórico para ser drop-in replacement " "de 03."),
    )

    p.add_argument("--only-angle", type=float, default=-1.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # ------------------------------------------------------------------
    # Dominio adaptativo de producto.
    # No existe rectángulo duro capaz de cortar el objeto.
    # ------------------------------------------------------------------
    p.add_argument("--rect-valid-erosion-px", type=int, default=2)
    p.add_argument("--adaptive-anchor-minimum-threshold", type=float, default=7.0)
    p.add_argument("--adaptive-anchor-mad-factor", type=float, default=4.5)
    p.add_argument("--adaptive-anchor-stable-quantile", type=float, default=0.62)
    p.add_argument("--adaptive-anchor-minimum-component-area", type=int, default=900)
    p.add_argument("--adaptive-anchor-margin-fraction", type=float, default=0.18)
    p.add_argument("--adaptive-anchor-minimum-margin-px", type=int, default=24)

    # Compatibilidad CLI histórica: se aceptan, pero V3.2 no los usa como recortes duros.
    p.add_argument("--roi-center-x-fraction", type=float, default=0.63)
    p.add_argument("--roi-center-y-fraction", type=float, default=0.43)
    p.add_argument("--roi-width-fraction", type=float, default=0.40)
    p.add_argument("--roi-height-fraction", type=float, default=0.76)
    p.add_argument("--support-cutoff-y-fraction", type=float, default=0.80)
    p.add_argument("--anchor-center-x-fraction", type=float, default=0.60)
    p.add_argument("--anchor-center-y-fraction", type=float, default=0.44)
    p.add_argument("--anchor-width-fraction", type=float, default=0.30)
    p.add_argument("--anchor-height-fraction", type=float, default=0.52)

    # ------------------------------------------------------------------
    # Evidencia visual.
    # ------------------------------------------------------------------
    p.add_argument("--weak-z-threshold", type=float, default=3.3)
    p.add_argument("--strong-z-threshold", type=float, default=5.8)
    p.add_argument("--minimum-raw-difference", type=float, default=8.0)

    # La cromaticidad tiene más peso que antes para no confundir sombra
    # acromática con objeto.
    p.add_argument("--chroma-weight", type=float, default=1.05)
    p.add_argument("--luminance-weight", type=float, default=0.42)
    p.add_argument("--gradient-weight", type=float, default=0.52)

    # ------------------------------------------------------------------
    # Rechazo de sombras.
    #
    # Una sombra típica:
    #   L_actual < L_fondo
    #   cambio de cromaticidad pequeño
    #   estructura de gradiente poco convincente
    # ------------------------------------------------------------------
    p.add_argument(
        "--shadow-min-darkening-l",
        type=float,
        default=10.0,
        help="Oscurecimiento mínimo ΔL (Lab) para considerar sombra.",
    )
    p.add_argument(
        "--shadow-max-chroma-z",
        type=float,
        default=3.0,
        help="Sombra: cambio cromático normalizado debe ser bajo.",
    )
    p.add_argument(
        "--shadow-max-gradient-z",
        type=float,
        default=4.0,
        help="Sombra: diferencia de gradiente normalizada moderada/baja.",
    )
    p.add_argument(
        "--shadow-strong-override-z",
        type=float,
        default=8.0,
        help=(
            "Si chroma/gradiente son extremadamente fuertes, no se descarta "
            "aunque exista oscurecimiento."
        ),
    )

    p.add_argument("--support-brightening-l", type=float, default=6.0)
    p.add_argument("--support-object-chroma-z", type=float, default=4.5)
    p.add_argument("--support-object-gradient-z", type=float, default=6.0)
    p.add_argument("--seed-anchor-margin-fraction", type=float, default=0.22)
    p.add_argument(
        "--capture-volume-lateral-margin-fraction",
        type=float,
        default=0.10,
    )
    p.add_argument(
        "--capture-volume-bottom-margin-fraction",
        type=float,
        default=0.03,
    )

    # ------------------------------------------------------------------
    # Exclusión del cuerpo físico de la plataforma.
    #
    # IMPORTANTE: NO agranda support_mask. La superficie útil sigue siendo
    # exactamente la misma. Se crea una segunda máscara exterior/inferior
    # para impedir que el aro gris del hardware entre en la silueta final.
    # ------------------------------------------------------------------
    p.add_argument(
        "--support-hardware-exclusion",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--support-hardware-lateral-fraction",
        type=float,
        default=0.028,
        help="Expansión lateral de la falda respecto al ancho del soporte.",
    )
    p.add_argument(
        "--support-hardware-downward-fraction",
        type=float,
        default=0.17,
        help="Expansión hacia abajo respecto a la altura del soporte.",
    )
    p.add_argument(
        "--support-hardware-lower-start-fraction",
        type=float,
        default=0.40,
        help=(
            "Fracción vertical del bbox del soporte desde la que puede existir "
            "la falda; evita expandir el arco superior."
        ),
    )
    p.add_argument("--support-hardware-min-lateral-px", type=int, default=6)
    p.add_argument("--support-hardware-min-downward-px", type=int, default=12)
    p.add_argument("--support-hardware-max-lateral-px", type=int, default=64)
    p.add_argument("--support-hardware-max-downward-px", type=int, default=96)

    # Banda fina para el pequeño filo gris inmediatamente exterior a la
    # elipse. No modifica support_mask ni la zona de contacto existente.
    p.add_argument(
        "--support-rim-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Excluye un filo fino alrededor del borde físico de la plataforma "
            "sin agrandar la superficie útil del soporte."
        ),
    )
    p.add_argument(
        "--support-rim-guard-px",
        type=int,
        default=6,
        help="Espesor máximo, en píxeles, del filo exterior de la plataforma.",
    )
    p.add_argument(
        "--support-rim-object-column-margin-px",
        type=int,
        default=10,
        help=(
            "Margen horizontal alrededor de columnas con cuerpo real del objeto; "
            "en esas columnas el filo no se usa como veto."
        ),
    )
    p.add_argument(
        "--support-rim-object-core-clearance-px",
        type=int,
        default=14,
        help=(
            "Separación mínima respecto del soporte para considerar una semilla "
            "como núcleo real del objeto y proteger sus columnas."
        ),
    )

    p.add_argument(
        "--graphcut-recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Recupera zonas ambiguas de baja textura sin borrar foreground ya aceptado."),
    )
    p.add_argument("--graphcut-iterations", type=int, default=3)
    p.add_argument("--graphcut-sure-fg-erosion-px", type=int, default=4)

    # ------------------------------------------------------------------
    # V5.0 — recuperación LOCAL del contacto objeto-plataforma.
    #
    # La disparidad NO participa en la detección global. Solo puede añadir
    # píxeles dentro del UNKNOWN ya derivado del cuerpo visual.
    # ------------------------------------------------------------------
    p.add_argument(
        "--local-contact-depth-verification",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Usa disparidad contra fondo vacío únicamente dentro de la " "banda local de contacto."
        ),
    )
    p.add_argument(
        "--local-contact-min-confidence",
        type=float,
        default=0.045,
    )
    p.add_argument(
        "--local-contact-max-lr-error-px",
        type=float,
        default=2.25,
    )
    p.add_argument(
        "--local-contact-reference-quantile",
        type=float,
        default=0.52,
        help=(
            "Cuantil de menor cambio RGB usado para aprender el sesgo de "
            "disparidad del plato visible."
        ),
    )
    p.add_argument(
        "--local-contact-min-reference-pixels",
        type=int,
        default=2500,
    )
    p.add_argument(
        "--local-contact-min-delta-px",
        type=float,
        default=0.85,
    )
    p.add_argument(
        "--local-contact-strong-min-delta-px",
        type=float,
        default=1.60,
    )
    p.add_argument(
        "--local-contact-mad-factor",
        type=float,
        default=4.0,
    )
    p.add_argument(
        "--local-contact-strong-mad-factor",
        type=float,
        default=6.0,
    )
    p.add_argument(
        "--local-contact-max-reference-sigma-px",
        type=float,
        default=1.20,
        help=(
            "Si la disparidad del plato visible es más inestable que esto, "
            "la cue estéreo se desactiva para esa vista."
        ),
    )
    p.add_argument(
        "--local-contact-seed-horizontal-radius-px",
        type=int,
        default=9,
    )
    p.add_argument(
        "--local-contact-seed-vertical-radius-px",
        type=int,
        default=34,
    )
    p.add_argument(
        "--local-contact-close-kernel",
        type=int,
        default=5,
    )
    p.add_argument(
        "--local-contact-visual-probability-min",
        type=float,
        default=0.16,
    )
    p.add_argument(
        "--local-contact-maximum-growth-iterations",
        type=int,
        default=400,
    )

    # ------------------------------------------------------------------
    # Banda ambigua objeto-soporte — V3.1
    # ------------------------------------------------------------------
    p.add_argument(
        "--support-occlusion-recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--support-occlusion-minimum-raw-difference",
        type=float,
        default=7.0,
    )
    p.add_argument(
        "--support-occlusion-strict-mad-factor",
        type=float,
        default=5.0,
    )
    p.add_argument(
        "--support-occlusion-loose-mad-factor",
        type=float,
        default=2.8,
    )
    p.add_argument(
        "--support-occlusion-stable-quantile",
        type=float,
        default=0.55,
    )
    p.add_argument(
        "--support-occlusion-contact-horizontal-radius-px",
        type=int,
        default=5,
        help="Tolerancia lateral para conectar objeto con soporte.",
    )
    p.add_argument(
        "--support-occlusion-contact-vertical-radius-px",
        type=int,
        default=20,
        help="Tolerancia vertical para conectar objeto con soporte.",
    )
    p.add_argument(
        "--support-occlusion-fallback-radius-px",
        type=int,
        default=42,
    )
    p.add_argument(
        "--support-occlusion-minimum-primary-marker-pixels",
        type=int,
        default=24,
    )
    p.add_argument(
        "--support-occlusion-minimum-strict-component-area",
        type=int,
        default=80,
    )
    p.add_argument(
        "--support-occlusion-minimum-loose-component-area",
        type=int,
        default=120,
    )
    p.add_argument(
        "--support-occlusion-open-kernel",
        type=int,
        default=3,
    )
    p.add_argument(
        "--support-occlusion-close-kernel",
        type=int,
        default=5,
    )
    p.add_argument(
        "--support-occlusion-shadow-dilation-px",
        type=int,
        default=2,
    )
    p.add_argument(
        "--support-occlusion-strict-chroma-z",
        type=float,
        default=3.4,
    )
    p.add_argument(
        "--support-occlusion-strict-gradient-z",
        type=float,
        default=5.2,
    )
    p.add_argument(
        "--support-occlusion-loose-chroma-z",
        type=float,
        default=1.45,
    )
    p.add_argument(
        "--support-occlusion-loose-gradient-z",
        type=float,
        default=2.0,
    )
    p.add_argument(
        "--support-occlusion-minimum-brightening-l",
        type=float,
        default=3.0,
    )

    # ------------------------------------------------------------------
    # V3.1 — banda ambigua de contacto.
    #
    # La región del soporte que cae inmediatamente bajo la proyección del
    # objeto se considera UNKNOWN aunque fotométricamente se parezca al plato.
    # ------------------------------------------------------------------
    p.add_argument(
        "--contact-band-horizontal-margin-px",
        type=int,
        default=10,
        help=(
            "Margen lateral de la banda UNKNOWN en píxeles rectificados. "
            "Amplía la zona evaluable sin aceptar automáticamente sus píxeles "
            "como objeto."
        ),
    )
    p.add_argument(
        "--contact-band-vertical-gap-px",
        type=int,
        default=32,
    )
    p.add_argument(
        "--contact-band-depth-fraction",
        type=float,
        default=0.90,
    )
    p.add_argument(
        "--contact-band-min-depth-px",
        type=int,
        default=55,
    )
    p.add_argument(
        "--contact-band-max-depth-px",
        type=int,
        default=320,
    )
    p.add_argument(
        "--contact-graphcut-iterations",
        type=int,
        default=5,
    )
    p.add_argument(
        "--contact-graphcut-sure-fg-erosion-px",
        type=int,
        default=3,
    )
    p.add_argument(
        "--contact-connect-horizontal-px",
        type=int,
        default=7,
    )
    p.add_argument(
        "--contact-connect-vertical-px",
        type=int,
        default=18,
    )
    p.add_argument(
        "--contact-min-component-area",
        type=int,
        default=120,
    )

    # ------------------------------------------------------------------
    # V3.2 — continuidad condicionada dentro de la banda UNKNOWN.
    #
    # No rellena por forma ni usa primitivas geométricas. Construye un prior
    # suave desde el cuerpo ya confirmado y lo atenúa al atravesar bordes
    # horizontales fuertes. Sirve para evitar cortes prematuros cuando
    # objeto y plataforma tienen apariencia muy similar.
    # ------------------------------------------------------------------
    p.add_argument(
        "--contact-continuity-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--contact-continuity-horizontal-margin-px",
        type=int,
        default=12,
    )
    p.add_argument(
        "--contact-continuity-decay-fraction",
        type=float,
        default=0.58,
    )
    p.add_argument(
        "--contact-continuity-min-decay-px",
        type=int,
        default=32,
    )
    p.add_argument(
        "--contact-continuity-edge-mad-factor",
        type=float,
        default=3.2,
    )
    p.add_argument(
        "--contact-continuity-edge-minimum",
        type=float,
        default=10.0,
    )
    p.add_argument(
        "--contact-continuity-edge-weight",
        type=float,
        default=0.75,
    )
    p.add_argument(
        "--contact-continuity-pr-fg-threshold",
        type=float,
        default=0.56,
    )
    p.add_argument(
        "--contact-continuity-rescue-threshold",
        type=float,
        default=0.68,
    )
    p.add_argument(
        "--contact-continuity-rescue-appearance-floor",
        type=float,
        default=0.18,
    )

    # ------------------------------------------------------------------
    # Morfología. El cierre baja de 9 a 5 para NO pegar cubo con sombras.
    # ------------------------------------------------------------------
    p.add_argument("--open-kernel", type=int, default=3)
    p.add_argument("--close-kernel", type=int, default=3)
    p.add_argument("--maximum-hole-area", type=int, default=6000)

    # Componentes.
    p.add_argument("--minimum-component-area", type=int, default=1200)
    p.add_argument(
        "--minimum-anchor-overlap-ratio",
        type=float,
        default=0.015,
        help=(
            "Fracción mínima del componente dentro del anchor. "
            "El mejor componente puede superar este filtro por score."
        ),
    )
    p.add_argument(
        "--component-center-sigma-fraction",
        type=float,
        default=0.22,
        help="Escala del término de proximidad al centro esperado.",
    )
    p.add_argument(
        "--secondary-component-fraction",
        type=float,
        default=0.10,
        help=(
            "Componente secundario solo se conserva si es ≥ esta fracción "
            "del principal y está espacialmente cerca."
        ),
    )
    p.add_argument(
        "--secondary-max-gap-px",
        type=int,
        default=22,
        help="Distancia máxima para unir componentes visuales del objeto.",
    )

    # Calidad.
    p.add_argument("--minimum-area-ratio", type=float, default=0.025)
    p.add_argument("--maximum-area-ratio", type=float, default=0.26)
    p.add_argument("--session-area-mad-factor", type=float, default=6.0)
    p.add_argument("--session-centroid-tolerance-px", type=float, default=120.0)

    return p


# ---------------------------------------------------------------------------
# Geometría 2D de la zona de captura
# ---------------------------------------------------------------------------


def normalized_rectangle_mask(
    shape: Tuple[int, int],
    center_x: float,
    center_y: float,
    width_fraction: float,
    height_fraction: float,
) -> np.ndarray:
    h, w = shape

    cx = float(np.clip(center_x, 0.0, 1.0)) * w
    cy = float(np.clip(center_y, 0.0, 1.0)) * h
    rw = float(np.clip(width_fraction, 0.01, 1.0)) * w
    rh = float(np.clip(height_fraction, 0.01, 1.0)) * h

    x0 = int(max(0, math.floor(cx - rw / 2.0)))
    x1 = int(min(w, math.ceil(cx + rw / 2.0)))
    y0 = int(max(0, math.floor(cy - rh / 2.0)))
    y1 = int(min(h, math.ceil(cy + rh / 2.0)))

    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def build_product_domain_and_anchor(
    image: np.ndarray,
    background: np.ndarray,
    rect_valid_mask: np.ndarray,
    support_mask: np.ndarray,
    args,
) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """Dominio derivado de rectificación + tamaño físico del plato."""
    rect_domain = safe_rectification_domain(
        rect_valid_mask,
        erosion_px=args.rect_valid_erosion_px,
    )

    roi, capture_diag = capture_volume_from_support(
        support_mask,
        rect_domain,
        lateral_margin_fraction=args.capture_volume_lateral_margin_fraction,
        bottom_margin_fraction=args.capture_volume_bottom_margin_fraction,
    )

    anchor, visual_candidate, anchor_diag = adaptive_visual_anchor(
        image,
        background,
        roi,
        minimum_threshold=args.adaptive_anchor_minimum_threshold,
        mad_factor=args.adaptive_anchor_mad_factor,
        stable_quantile=args.adaptive_anchor_stable_quantile,
        minimum_component_area=args.adaptive_anchor_minimum_component_area,
        margin_fraction=args.adaptive_anchor_margin_fraction,
        minimum_margin_px=args.adaptive_anchor_minimum_margin_px,
    )

    return (
        roi,
        anchor,
        {
            "roi_strategy": "support_derived_capture_volume",
            "roi_pixels": int(np.count_nonzero(roi)),
            "anchor_pixels": int(np.count_nonzero(anchor)),
            "capture_volume": capture_diag,
            "adaptive_anchor": anchor_diag,
            "legacy_roi_parameters_ignored": True,
            "support_cutoff_disabled": True,
        },
        visual_candidate,
    )


# ---------------------------------------------------------------------------
# Evidencia visual / sombra
# ---------------------------------------------------------------------------


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Calcula la magnitud del gradiente usada como evidencia visual."""
    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )
    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )
    return cv2.magnitude(gx, gy)


def estimate_noise_scales(
    image_lab: np.ndarray,
    background_lab: np.ndarray,
    gray_difference: np.ndarray,
    gradient_difference: np.ndarray,
    stable_mask: np.ndarray,
) -> dict:
    lab_difference = cv2.absdiff(
        image_lab,
        background_lab,
    ).astype(np.float32)

    scales = {}

    for channel, name, minimum in (
        (0, "l", 2.0),
        (1, "a", 1.5),
        (2, "b", 1.5),
    ):
        _, sigma = robust_location_scale(
            lab_difference[:, :, channel][stable_mask],
            minimum_scale=minimum,
        )
        scales[name] = float(sigma)

    _, sigma_gray = robust_location_scale(
        gray_difference[stable_mask],
        minimum_scale=2.0,
    )
    scales["gray"] = float(sigma_gray)

    _, sigma_grad = robust_location_scale(
        gradient_difference[stable_mask],
        minimum_scale=4.0,
    )
    scales["gradient"] = float(sigma_grad)

    return scales


def compute_visual_evidence(
    image: np.ndarray,
    background: np.ndarray,
    roi: np.ndarray,
    rect_valid_mask: np.ndarray,
    support_mask: np.ndarray,
    args,
) -> dict:
    image_lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )
    background_lab = cv2.cvtColor(
        background,
        cv2.COLOR_BGR2LAB,
    )

    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )
    gray_background = cv2.cvtColor(
        background,
        cv2.COLOR_BGR2GRAY,
    )

    lab_abs = cv2.absdiff(
        image_lab,
        background_lab,
    ).astype(np.float32)

    gray_difference = cv2.absdiff(
        gray_image,
        gray_background,
    ).astype(np.float32)

    gradient_image = gradient_magnitude(
        gray_image,
    )
    gradient_background = gradient_magnitude(
        gray_background,
    )

    gradient_difference = np.abs(gradient_image - gradient_background).astype(np.float32)

    # Signo de luminancia:
    # positivo = la imagen actual es más brillante;
    # negativo = se oscureció respecto al fondo.
    signed_delta_l = image_lab[:, :, 0].astype(np.float32) - background_lab[:, :, 0].astype(
        np.float32
    )

    rect_valid = rect_valid_mask > 0

    # Ruido aprendido de los píxeles de menor cambio de la escena válida.
    preliminary_change = (0.60 * np.max(lab_abs, axis=2) + 0.40 * gray_difference).astype(
        np.float32
    )

    valid_for_noise = rect_valid & (roi > 0) & np.isfinite(preliminary_change)
    values = preliminary_change[valid_for_noise]
    stable_mask = np.zeros_like(valid_for_noise, dtype=bool)
    if values.size:
        cutoff = float(np.quantile(values, 0.58))
        stable_mask = valid_for_noise & (preliminary_change <= cutoff)
    if np.count_nonzero(stable_mask) < 5000:
        stable_mask = valid_for_noise

    scales = estimate_noise_scales(
        image_lab,
        background_lab,
        gray_difference,
        gradient_difference,
        stable_mask,
    )

    z_l = lab_abs[:, :, 0] / scales["l"]
    z_a = lab_abs[:, :, 1] / scales["a"]
    z_b = lab_abs[:, :, 2] / scales["b"]
    z_gray = gray_difference / scales["gray"]
    z_gradient = gradient_difference / scales["gradient"]

    chroma_z = np.sqrt(0.5 * (z_a * z_a + z_b * z_b)).astype(np.float32)

    luminance_z = (0.5 * (z_l + z_gray)).astype(np.float32)

    score = (
        args.chroma_weight * chroma_z
        + args.luminance_weight * luminance_z
        + args.gradient_weight * z_gradient
    ).astype(np.float32)

    raw_difference = (
        0.52
        * np.max(
            lab_abs,
            axis=2,
        )
        + 0.48 * gray_difference
    ).astype(np.float32)

    # ---------------------------------------------------------------
    # Modelo explícito de sombra:
    #
    # - se oscureció;
    # - casi no cambió cromaticidad;
    # - el cambio estructural de bordes no es fuerte.
    #
    # Se conserva un override para bordes/cromas muy fuertes, evitando
    # eliminar una cara real del objeto solo por ser oscura.
    # ---------------------------------------------------------------
    shadow_candidate = (
        (signed_delta_l <= -args.shadow_min_darkening_l)
        & (chroma_z <= args.shadow_max_chroma_z)
        & (z_gradient <= args.shadow_max_gradient_z)
        & (roi > 0)
    )

    strong_object_override = (chroma_z >= args.shadow_strong_override_z) | (
        z_gradient >= args.shadow_strong_override_z
    )

    support = (support_mask > 0) & (roi > 0)
    shadow_mask = shadow_candidate & (~strong_object_override)

    support_object_evidence = (
        support
        & (raw_difference >= args.minimum_raw_difference)
        & (~shadow_mask)
        & (
            ((chroma_z >= args.support_object_chroma_z) & (z_gradient >= 1.5))
            | ((z_gradient >= args.support_object_gradient_z) & (chroma_z >= 1.0))
            | (
                (signed_delta_l >= args.support_brightening_l)
                & ((chroma_z >= 1.8) | (z_gradient >= 3.2))
            )
        )
    )

    support_shadow_reject = support & shadow_mask & (~support_object_evidence)

    weak = (
        (score >= args.weak_z_threshold)
        & (raw_difference >= args.minimum_raw_difference)
        & (roi > 0)
        & (~support_shadow_reject)
    )

    strong = (
        (score >= args.strong_z_threshold) & (roi > 0) & (~shadow_mask)
    ) | support_object_evidence

    probability = (
        1.0
        - np.exp(
            -np.maximum(
                score - args.weak_z_threshold,
                0.0,
            )
        )
    ).astype(np.float32)

    probability = np.clip(
        probability,
        0.0,
        1.0,
    )
    probability[roi == 0] = 0.0
    probability[shadow_mask] *= 0.55
    probability[support_shadow_reject] *= 0.08
    probability[support_object_evidence] = np.maximum(probability[support_object_evidence], 0.85)

    return {
        "weak": weak,
        "strong": strong,
        "shadow_mask": shadow_mask,
        "support_shadow_reject": support_shadow_reject,
        "support_object_evidence": support_object_evidence,
        "probability": probability,
        "score": score,
        "raw_difference": raw_difference,
        "signed_delta_l": signed_delta_l,
        "chroma_z": chroma_z,
        "luminance_z": luminance_z,
        "gradient_z": z_gradient,
        "noise_scales": scales,
        "stable_mask": stable_mask,
    }


# ---------------------------------------------------------------------------
# Oclusión explícita del soporte
# ---------------------------------------------------------------------------


def _robust_low_change_threshold(
    values: np.ndarray,
    minimum_threshold: float,
    mad_factor: float,
    stable_quantile: float,
) -> Tuple[float, dict]:
    x = np.asarray(values, dtype=np.float32)
    x = x[np.isfinite(x)]

    if x.size < 100:
        return float(minimum_threshold), {
            "status": "fallback",
            "sample_count": int(x.size),
            "threshold": float(minimum_threshold),
            "median": None,
            "mad_sigma": None,
        }

    q = float(np.clip(stable_quantile, 0.20, 0.80))
    cutoff = float(np.quantile(x, q))
    stable = x[x <= cutoff]
    if stable.size < 100:
        stable = x

    median = float(np.median(stable))
    mad = float(np.median(np.abs(stable - median)))
    sigma = max(0.75, 1.4826 * mad)
    threshold = max(
        float(minimum_threshold),
        median + float(mad_factor) * sigma,
    )

    return float(threshold), {
        "status": "ok",
        "sample_count": int(x.size),
        "stable_sample_count": int(stable.size),
        "stable_quantile": q,
        "stable_cutoff": cutoff,
        "median": median,
        "mad_sigma": float(sigma),
        "threshold": float(threshold),
    }


def _components_touching_marker(
    candidate: np.ndarray,
    marker: np.ndarray,
    minimum_area: int,
) -> Tuple[np.ndarray, dict]:
    """Conserva las componentes de la máscara que intersectan la semilla."""
    cand = np.asarray(candidate).astype(bool)
    mark = np.asarray(marker).astype(bool)
    output = np.zeros(cand.shape, dtype=np.uint8)

    if np.count_nonzero(cand) == 0:
        return output, {
            "component_count": 0,
            "selected_components": [],
            "selected_pixels": 0,
        }

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cand.astype(np.uint8),
        connectivity=8,
    )

    selected = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(minimum_area):
            continue
        comp = labels == label
        overlap = int(np.count_nonzero(comp & mark))
        if overlap <= 0:
            continue
        output[comp] = 255
        selected.append(
            {
                "label": int(label),
                "area": area,
                "marker_overlap_pixels": overlap,
            }
        )

    return output, {
        "component_count": int(count - 1),
        "selected_components": selected,
        "selected_pixels": int(np.count_nonzero(output)),
    }


def _column_bounds(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extremos por columna mediante reducciones NumPy; mismos centinelas."""
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    valid = np.any(m, axis=0)
    top = np.full(w, h, dtype=np.int32)
    bottom = np.full(w, -1, dtype=np.int32)
    if h:
        top[valid] = np.argmax(m, axis=0)[valid]
        bottom[valid] = h - 1 - np.argmax(m[::-1], axis=0)[valid]
    return top, bottom, valid


def visual_contact_extension(object_seed, support_mask, roi, evidence, exclusion):
    """Zona candidata conectada al cuerpo; nunca una máscara de objeto aceptada.

    La diferencia contra el fondo y los gradientes delimitan la región.
    Las sombras y el hardware son barreras; no se rellena una envolvente convexa.
    """
    seed = (np.asarray(object_seed) > 0) & (roi > 0)
    support = (support_mask > 0) & (roi > 0)
    ys, xs = np.nonzero(seed)
    empty = np.zeros(support.shape, np.uint8)
    if xs.size < 100:
        return empty, {"status": "insufficient_seed", "added_candidate_pixels": 0}
    width = int(xs.max() - xs.min() + 1)
    margin = int(np.clip(round(0.02 * width), 3, 20))
    shadow = np.asarray(evidence["shadow_mask"]).astype(bool)
    blocked = (np.asarray(exclusion) > 0) | shadow | (roi == 0)
    # Exigir diferencia visual; un borde aislado del plato no basta.
    visual = (np.asarray(evidence["weak"]).astype(bool)
              | np.asarray(evidence["strong"]).astype(bool))
    structural = ((np.asarray(evidence["chroma_z"]) >= 3.0)
                  | (np.asarray(evidence["gradient_z"]) >= 4.0))
    candidate = seed | (support & visual & structural & ~blocked)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    connected_domain = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    connected_domain &= ~blocked
    connected_domain |= seed
    _, labels = cv2.connectedComponents(connected_domain.astype(np.uint8), connectivity=8)
    touched = np.unique(labels[seed])
    touched = touched[touched != 0]
    connected = np.isin(labels, touched) & support & ~blocked
    # Margen proporcional alrededor del contorno real, no de su rectángulo.
    expanded = cv2.dilate(connected.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))) > 0
    expanded &= support & ~blocked
    return expanded.astype(np.uint8) * 255, {
        "status": "active", "seed_width_px": width, "margin_px": margin,
        "connected_visual_pixels": int(np.count_nonzero(connected)),
        "added_candidate_pixels": int(np.count_nonzero(expanded)),
        "policy": "visual_contour_candidates_only_no_depth_no_automatic_acceptance",
    }


def build_contact_ambiguity_band(
    support_mask: np.ndarray,
    roi: np.ndarray,
    object_seed: np.ndarray,
    args,
    support_evidence: Optional[np.ndarray] = None,
    visual_extension: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Construye una banda UNKNOWN donde objeto y soporte pueden solaparse.

    Esta función NO mira color ni diferencia contra el fondo para decidir si
    el píxel es fondo. El objetivo es evitar el error de V3.0:

        objeto claro ≈ plato claro -> "fondo seguro"

    La banda usa:
    - dominio físico del soporte;
    - columnas ocupadas por el objeto ya detectado;
    - proximidad vertical entre el cuerpo y la frontera superior del soporte;
    - evidencia visual estricta dentro del soporte, siempre que ya haya sido
      validada como conectada al cuerpo del objeto.

    La evidencia estricta permite abrir columnas laterales que pertenecen al
    objeto pero que nacen ya dentro de la proyección del plato (por ejemplo,
    caras inclinadas o bases anchas). No modifica la geometría del soporte ni
    depende del nombre o forma conocida del objeto.

    El soporte fuera de esta banda sí puede bloquearse como background.
    """
    support = (np.asarray(support_mask) > 0) & (np.asarray(roi) > 0)
    seed = (np.asarray(object_seed) > 0) & (np.asarray(roi) > 0)

    h, w = support.shape
    unknown = np.zeros((h, w), dtype=np.uint8)

    if np.count_nonzero(support) < 500 or np.count_nonzero(seed) < 100:
        locked = support.astype(np.uint8) * 255
        return (
            unknown,
            locked,
            {
                "status": "fallback",
                "reason": "insufficient_support_or_object_seed",
                "unknown_pixels": 0,
                "locked_pixels": int(np.count_nonzero(locked)),
            },
        )

    support_top, support_bottom, support_valid = _column_bounds(support)
    _, seed_bottom, seed_valid = _column_bounds(seed)

    # Columnas donde el objeto llega realmente cerca de la frontera superior
    # de la plataforma.
    gap = max(4, int(args.contact_band_vertical_gap_px))
    contact_columns = support_valid & seed_valid & (seed_bottom >= (support_top - gap))

    # Si el borde inferior del seed tiene pequeños huecos, se usa la
    # proyección horizontal del cuerpo, pero solo dentro del intervalo
    # principal de columnas que sí tienen contacto.
    margin = max(0, int(args.contact_band_horizontal_margin_px))
    cc = contact_columns.astype(np.uint8).reshape(1, -1) * 255

    if np.count_nonzero(contact_columns) > 0:
        kernel = np.ones((1, 2 * margin + 1), dtype=np.uint8)
        contact_columns = cv2.dilate(cc, kernel, iterations=1).reshape(-1) > 0
    else:
        # Fallback: objeto muy liso cuya semilla se detuvo justo antes del
        # soporte. Se usa su proyección, pero solo si la separación vertical
        # mínima es razonable.
        seed_cols = np.flatnonzero(seed_valid)
        if seed_cols.size > 0:
            min_gap = h
            for x in seed_cols:
                if support_valid[x]:
                    min_gap = min(
                        min_gap,
                        int(support_top[x] - seed_bottom[x]),
                    )
            if min_gap <= 2 * gap:
                proj = seed_valid.astype(np.uint8).reshape(1, -1) * 255
                kernel = np.ones((1, 2 * margin + 1), dtype=np.uint8)
                contact_columns = cv2.dilate(proj, kernel, iterations=1).reshape(-1) > 0

    # Limitar la banda a un prefijo de la plataforma visto desde su borde
    # superior. La fracción se refiere al espesor visible de la plataforma
    # en cada columna, no a una altura fija del objeto.
    depth_fraction = float(np.clip(args.contact_band_depth_fraction, 0.35, 0.95))
    min_depth = max(15, int(args.contact_band_min_depth_px))
    max_depth = max(min_depth, int(args.contact_band_max_depth_px))

    selected_columns = 0
    depth_values = []

    for x in np.flatnonzero(contact_columns & support_valid):
        y0 = int(support_top[x])
        y1 = int(support_bottom[x])
        thickness = max(1, y1 - y0 + 1)

        depth = int(round(depth_fraction * thickness))
        depth = int(np.clip(depth, min_depth, max_depth))
        ye = min(y1, y0 + depth)

        unknown[y0 : ye + 1, x] = support[y0 : ye + 1, x].astype(np.uint8) * 255
        selected_columns += 1
        depth_values.append(depth)

    # Pequeño cierre horizontal para evitar columnas alternantes.
    if selected_columns > 0:
        unknown = cv2.morphologyEx(
            unknown,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
            iterations=1,
        )
        unknown = cv2.bitwise_and(
            unknown,
            support.astype(np.uint8) * 255,
        )

    # ------------------------------------------------------------------
    # V5.3 — extensión lateral guiada por evidencia estricta del objeto.
    #
    # El método anterior abría UNKNOWN solo en columnas donde el cuerpo ya
    # existía FUERA del soporte. Eso recortaba objetos cuya parte inferior
    # aparece por primera vez dentro de la proyección del plato. Aquí se usa
    # únicamente evidencia estricta que ya fue validada como conectada al
    # cuerpo; nunca se desplaza ni se redimensiona support_mask.
    # ------------------------------------------------------------------
    evidence_added_pixels = 0
    evidence_columns_count = 0

    if support_evidence is not None:
        evidence_mask = (np.asarray(support_evidence) > 0) & support

        if np.count_nonzero(evidence_mask) > 0:
            # Un cierre muy pequeño une trazos de borde del mismo cuerpo sin
            # transformar una región grande del plato en candidato.
            evidence_mask = (
                cv2.morphologyEx(
                    evidence_mask.astype(np.uint8) * 255,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    iterations=1,
                )
                > 0
            ) & support

            _, evidence_bottom, evidence_valid = _column_bounds(evidence_mask)

            # Se admite un pequeño margen lateral para huecos entre aristas
            # visibles del mismo objeto. La autoridad sigue siendo la
            # evidencia estricta, no una proyección de forma.
            evidence_cols_u8 = evidence_valid.astype(np.uint8).reshape(1, -1) * 255
            ev_margin = max(2, int(args.contact_band_horizontal_margin_px))
            evidence_columns = (
                cv2.dilate(
                    evidence_cols_u8,
                    np.ones((1, 2 * ev_margin + 1), dtype=np.uint8),
                    iterations=1,
                ).reshape(-1)
                > 0
            )
            evidence_columns &= support_valid

            # Para columnas añadidas por el margen, interpolar el fondo de
            # evidencia desde la columna válida más cercana.
            valid_x = np.flatnonzero(evidence_valid)
            if valid_x.size > 0:
                for x in np.flatnonzero(evidence_columns):
                    if evidence_valid[x]:
                        y_ev = int(evidence_bottom[x])
                    else:
                        nearest = int(valid_x[np.argmin(np.abs(valid_x - x))])
                        y_ev = int(evidence_bottom[nearest])

                    if y_ev < 0:
                        continue

                    y0 = int(support_top[x])
                    y1 = int(support_bottom[x])
                    if y0 > y1:
                        continue

                    # Abrir solo hasta donde llega la evidencia más una
                    # tolerancia vertical corta para el contacto.
                    extra = max(4, int(args.contact_connect_vertical_px))
                    ye = min(y1, max(y0, y_ev + extra))
                    before = int(np.count_nonzero(unknown[y0 : ye + 1, x]))
                    unknown[y0 : ye + 1, x] = (
                        support[y0 : ye + 1, x].astype(np.uint8) * 255
                    )
                    after = int(np.count_nonzero(unknown[y0 : ye + 1, x]))
                    evidence_added_pixels += max(0, after - before)
                    evidence_columns_count += 1

            unknown = cv2.bitwise_and(
                unknown,
                support.astype(np.uint8) * 255,
            )

    adaptive_added = 0
    if visual_extension is not None:
        extension = (np.asarray(visual_extension) > 0) & support
        adaptive_added = int(np.count_nonzero(extension & (unknown == 0)))
        unknown[extension] = 255

    locked = (support & (unknown == 0)).astype(np.uint8) * 255

    return (
        unknown,
        locked,
        {
            "status": "ok",
            "contact_columns": int(selected_columns),
            "adaptive_visual_added_pixels": adaptive_added,
            "evidence_extension_columns": int(evidence_columns_count),
            "evidence_extension_added_pixels": int(evidence_added_pixels),
            "unknown_pixels": int(np.count_nonzero(unknown)),
            "locked_pixels": int(np.count_nonzero(locked)),
            "unknown_ratio_of_support": float(
                np.count_nonzero(unknown) / max(np.count_nonzero(support), 1)
            ),
            "depth_fraction": depth_fraction,
            "depth_px_median": (float(np.median(depth_values)) if depth_values else None),
            "depth_px_min": (int(np.min(depth_values)) if depth_values else None),
            "depth_px_max": (int(np.max(depth_values)) if depth_values else None),
        },
    )


def _normalize_feature(
    array: np.ndarray,
    mask: np.ndarray,
    lower_q: float = 0.02,
    upper_q: float = 0.98,
) -> np.ndarray:
    a = np.asarray(array, dtype=np.float32)
    m = np.asarray(mask).astype(bool) & np.isfinite(a)

    if np.count_nonzero(m) < 100:
        lo, hi = 0.0, 1.0
    else:
        vals = a[m]
        lo = float(np.quantile(vals, lower_q))
        hi = float(np.quantile(vals, upper_q))
        if hi <= lo + 1e-6:
            hi = lo + 1.0

    out = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return np.round(255.0 * out).astype(np.uint8)


def _lab_kmeans_centers(
    lab_image: np.ndarray,
    sample_mask: np.ndarray,
    maximum_clusters: int = 3,
    maximum_samples: int = 24000,
) -> Optional[np.ndarray]:
    """Centros multimodales Lab para FG/BG, deterministas."""
    samples = (
        np.asarray(lab_image)[np.asarray(sample_mask).astype(bool)]
        .reshape(-1, 3)
        .astype(np.float32)
    )

    if samples.shape[0] < 120:
        return None

    if samples.shape[0] > maximum_samples:
        # Muestreo determinista uniforme.
        indices = np.linspace(
            0,
            samples.shape[0] - 1,
            maximum_samples,
            dtype=np.int32,
        )
        samples = samples[indices]

    k = int(
        np.clip(
            samples.shape[0] // 900,
            1,
            maximum_clusters,
        )
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.15,
    )

    _compactness, _labels, centers = cv2.kmeans(
        samples,
        k,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    return centers.astype(np.float32)


def _minimum_lab_distance(
    lab_image: np.ndarray,
    centers: Optional[np.ndarray],
) -> np.ndarray:
    """Calcula la distancia mínima de color a los centros de referencia Lab."""
    h, w = lab_image.shape[:2]
    if centers is None or len(centers) == 0:
        return np.full((h, w), 1e3, dtype=np.float32)

    lab = lab_image.astype(np.float32)
    dist = np.full((h, w), np.inf, dtype=np.float32)

    for center in centers:
        delta = lab - center.reshape(1, 1, 3)
        d = np.sqrt(np.sum(delta * delta, axis=2))
        dist = np.minimum(dist, d)

    return dist.astype(np.float32)


def build_contact_feature_image(
    image: np.ndarray,
    background: np.ndarray,
    evidence: dict,
    domain: np.ndarray,
    object_foreground: np.ndarray,
    visible_support_background: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Características de contacto con modelos multimodales FG/BG.

    C0 = afinidad de apariencia FG frente a BG.
    C1 = luminancia Lab actual.
    C2 = estructura local / cambio cromático.

    La afinidad se aprende de:
      - FG: cuerpo ya aceptado, preferentemente cerca del contacto;
      - BG: plataforma visible bloqueada, preferentemente cerca del contacto.

    Así una cara blanca sobre un plato blanco no depende únicamente de
    |imagen - fondo|.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    dom = np.asarray(domain).astype(bool)
    fg = np.asarray(object_foreground).astype(bool)
    bg = np.asarray(visible_support_background).astype(bool)

    # Entrenamiento local alrededor del dominio ambiguo.
    near = (
        cv2.dilate(
            dom.astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (101, 141),
            ),
            iterations=1,
        )
        > 0
    )

    fg_local = fg & near
    bg_local = bg & near

    if np.count_nonzero(fg_local) < 600:
        fg_local = fg
    if np.count_nonzero(bg_local) < 600:
        bg_local = bg

    fg_centers = _lab_kmeans_centers(
        lab,
        fg_local,
        maximum_clusters=4,
    )
    bg_centers = _lab_kmeans_centers(
        lab,
        bg_local,
        maximum_clusters=4,
    )

    d_fg = _minimum_lab_distance(lab, fg_centers)
    d_bg = _minimum_lab_distance(lab, bg_centers)

    # Positivo -> visualmente más próximo al cuerpo que al plato.
    margin = d_bg - d_fg

    # Escala robusta para convertir el margen a [0,255].
    values = margin[dom & np.isfinite(margin)]
    if values.size >= 100:
        scale = max(
            2.0,
            1.4826 * float(np.median(np.abs(values - np.median(values)))),
        )
    else:
        scale = 5.0

    affinity = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                margin / scale,
                -12.0,
                12.0,
            )
        )
    )
    affinity_u8 = np.round(255.0 * affinity).astype(np.uint8)

    l_channel = lab[:, :, 0]

    grad = np.asarray(evidence["gradient_z"], dtype=np.float32)
    chroma = np.asarray(evidence["chroma_z"], dtype=np.float32)
    structural = np.maximum(
        np.clip(grad, 0.0, 10.0),
        0.80 * np.clip(chroma, 0.0, 10.0),
    )
    structural_u8 = _normalize_feature(structural, dom)

    feature = cv2.merge(
        [
            affinity_u8,
            l_channel,
            structural_u8,
        ]
    )

    diag = {
        "fg_training_pixels": int(np.count_nonzero(fg_local)),
        "bg_training_pixels": int(np.count_nonzero(bg_local)),
        "fg_cluster_count": (int(len(fg_centers)) if fg_centers is not None else 0),
        "bg_cluster_count": (int(len(bg_centers)) if bg_centers is not None else 0),
        "appearance_margin_median_unknown": (
            float(np.median(margin[dom])) if np.count_nonzero(dom) > 0 else None
        ),
        "appearance_scale": float(scale),
    }

    return feature, margin.astype(np.float32), diag


def _sigmoid01(x: np.ndarray) -> np.ndarray:
    """Transforma la evidencia con una sigmoide acotada entre cero y uno."""
    a = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(a, -12.0, 12.0)))).astype(np.float32)


def build_contact_continuity_prior(
    image: np.ndarray,
    object_mask_outside_support: np.ndarray,
    contact_unknown: np.ndarray,
    appearance_margin: np.ndarray,
    appearance_scale: float,
    evidence: dict,
    args,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Prior suave de continuidad objeto-soporte para V3.2.

    Principios:
    - parte únicamente del objeto ya confirmado fuera del soporte;
    - opera solo dentro de UNKNOWN;
    - no usa rectángulos, planos, caras ni dimensiones conocidas;
    - favorece continuidad hacia la zona de contacto;
    - se debilita gradualmente con la profundidad;
    - se debilita después de bordes horizontales fuertes;
    - no convierte por sí sola la banda completa en foreground.

    Retorna:
        continuity_prior: [0,1]
        combined_prior:   [0,1], apariencia + continuidad
    """
    unknown = np.asarray(contact_unknown).astype(bool)
    obj = np.asarray(object_mask_outside_support).astype(bool)

    h, w = unknown.shape
    zero = np.zeros((h, w), dtype=np.float32)

    if (
        not bool(args.contact_continuity_enabled)
        or np.count_nonzero(unknown) < 100
        or np.count_nonzero(obj) < 100
    ):
        return (
            zero,
            zero,
            {
                "status": "disabled",
                "reason": "disabled_or_insufficient_data",
            },
        )

    # Apariencia en probabilidad [0,1].
    scale = max(1e-3, float(appearance_scale))
    appearance_prob = _sigmoid01(np.asarray(appearance_margin, dtype=np.float32) / scale)

    # Borde horizontal: derivada vertical de L*.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)
    grad_y = np.abs(
        cv2.Sobel(
            l,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
        )
    )

    edge_values = grad_y[unknown & np.isfinite(grad_y)]
    if edge_values.size >= 100:
        med = float(np.median(edge_values))
        mad = float(np.median(np.abs(edge_values - med)))
        sigma = max(1.0, 1.4826 * mad)
        edge_threshold = max(
            float(args.contact_continuity_edge_minimum),
            med + float(args.contact_continuity_edge_mad_factor) * sigma,
        )
    else:
        med = 0.0
        sigma = 1.0
        edge_threshold = float(args.contact_continuity_edge_minimum)

    # Fondo visualmente oscuro/sombra no debe ganar continuidad solo por
    # proximidad. La sombra actúa como atenuador, no como veto absoluto.
    shadow = np.asarray(
        evidence.get(
            "shadow_mask",
            np.zeros_like(unknown),
        )
    ).astype(bool)

    # Para cada columna se busca el borde inferior del cuerpo confirmado.
    obj_valid = np.any(obj, axis=0)
    obj_bottom = np.full(w, -1, dtype=np.int32)
    for x in np.flatnonzero(obj_valid):
        ys = np.flatnonzero(obj[:, x])
        if ys.size > 0:
            obj_bottom[x] = int(ys[-1])

    # Se admiten pequeños huecos laterales buscando la columna de objeto más
    # cercana dentro de un margen corto.
    margin = max(
        1,
        int(args.contact_continuity_horizontal_margin_px),
    )

    effective_bottom = obj_bottom.copy()
    valid_cols = np.flatnonzero(obj_valid)

    if valid_cols.size > 0:
        invalid_unknown_cols = np.flatnonzero(np.any(unknown, axis=0) & (~obj_valid))

        for x in invalid_unknown_cols:
            lo = max(0, x - margin)
            hi = min(w, x + margin + 1)

            local = np.flatnonzero(obj_valid[lo:hi])
            if local.size == 0:
                continue

            candidates = local + lo
            nearest = candidates[np.argmin(np.abs(candidates - x))]
            effective_bottom[x] = obj_bottom[nearest]

    continuity = np.zeros((h, w), dtype=np.float32)

    decay_fraction = float(
        np.clip(
            args.contact_continuity_decay_fraction,
            0.20,
            1.20,
        )
    )
    min_decay = max(
        8,
        int(args.contact_continuity_min_decay_px),
    )
    edge_weight = max(
        0.0,
        float(args.contact_continuity_edge_weight),
    )

    processed_columns = 0
    barrier_columns = 0

    for x in np.flatnonzero(np.any(unknown, axis=0)):
        y_obj = int(effective_bottom[x])
        if y_obj < 0:
            continue

        ys = np.flatnonzero(unknown[:, x])
        if ys.size == 0:
            continue

        y0 = int(ys[0])
        y1 = int(ys[-1])
        depth = max(1, y1 - y0 + 1)

        decay = max(
            float(min_decay),
            decay_fraction * float(depth),
        )

        # Distancia vertical desde el cuerpo.
        yy = ys.astype(np.float32)
        dy = np.maximum(
            0.0,
            yy - float(y_obj),
        )
        base = np.exp(-dy / decay)

        # La barrera solo se acumula dentro de UNKNOWN.
        gy = grad_y[ys, x].astype(np.float32)
        excess = np.maximum(
            0.0,
            gy - edge_threshold,
        )
        denom = max(
            edge_threshold,
            sigma,
            1.0,
        )
        normalized_excess = excess / denom

        # Ignorar microbordes aislados muy próximos al inicio de UNKNOWN.
        # Esos bordes suelen ser producto del propio límite de la máscara.
        if normalized_excess.size > 4:
            normalized_excess[:3] *= 0.25

        cumulative_barrier = np.cumsum(normalized_excess)
        barrier_factor = np.exp(-edge_weight * cumulative_barrier)

        if np.max(cumulative_barrier) > 0.35:
            barrier_columns += 1

        local = base * barrier_factor

        # Una sombra reduce la confianza, pero no corta automáticamente una
        # pieza que pueda ser oscura.
        local_shadow = shadow[ys, x]
        local[local_shadow] *= 0.55

        continuity[ys, x] = np.clip(
            local,
            0.0,
            1.0,
        )
        processed_columns += 1

    # Apariencia + continuidad.
    #
    # La apariencia domina si ya es convincente. Si es ambigua, la
    # continuidad puede elevarla, pero nunca fuera de UNKNOWN.
    continuity_candidate = 0.72 * continuity + 0.28 * appearance_prob
    combined = np.maximum(
        appearance_prob,
        continuity_candidate,
    )
    combined[~unknown] = appearance_prob[~unknown]

    diag = {
        "status": "ok",
        "processed_columns": int(processed_columns),
        "barrier_columns": int(barrier_columns),
        "edge_threshold": float(edge_threshold),
        "edge_median": float(med),
        "edge_mad_sigma": float(sigma),
        "continuity_median_unknown": (
            float(np.median(continuity[unknown])) if np.count_nonzero(unknown) > 0 else None
        ),
        "continuity_p90_unknown": (
            float(
                np.quantile(
                    continuity[unknown],
                    0.90,
                )
            )
            if np.count_nonzero(unknown) > 0
            else None
        ),
        "combined_prior_median_unknown": (
            float(np.median(combined[unknown])) if np.count_nonzero(unknown) > 0 else None
        ),
    }

    return (
        continuity.astype(np.float32),
        combined.astype(np.float32),
        diag,
    )


def _keep_contact_components(
    recovered: np.ndarray,
    object_mask_outside_support: np.ndarray,
    minimum_area: int,
    connect_rx: int,
    connect_ry: int,
) -> Tuple[np.ndarray, dict]:
    """Conserva únicamente componentes recuperados conectados al objeto."""
    rec = np.asarray(recovered).astype(bool)
    obj = np.asarray(object_mask_outside_support).astype(bool)

    out = np.zeros(rec.shape, dtype=np.uint8)

    if np.count_nonzero(rec) == 0:
        return out, {
            "component_count": 0,
            "selected_count": 0,
            "selected_pixels": 0,
        }

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            2 * max(1, int(connect_rx)) + 1,
            2 * max(1, int(connect_ry)) + 1,
        ),
    )
    near_object = (
        cv2.dilate(
            obj.astype(np.uint8) * 255,
            kernel,
            iterations=1,
        )
        > 0
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        rec.astype(np.uint8),
        connectivity=8,
    )

    selected = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(minimum_area):
            continue

        comp = labels == label
        touch = int(np.count_nonzero(comp & near_object))
        if touch <= 0:
            continue

        out[comp] = 255
        selected.append(
            {
                "label": int(label),
                "area": area,
                "touch_pixels": touch,
            }
        )

    return out, {
        "component_count": int(n - 1),
        "selected_count": int(len(selected)),
        "selected_pixels": int(np.count_nonzero(out)),
        "selected_components": selected,
    }


def contact_aware_support_graphcut(
    image: np.ndarray,
    background: np.ndarray,
    object_mask_outside_support: np.ndarray,
    support_mask: np.ndarray,
    contact_unknown: np.ndarray,
    evidence: dict,
    roi: np.ndarray,
    args,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Resuelve UNKNOWN con apariencia + continuidad condicionada.

    V3.2 mantiene la banda de contacto de V3.1, pero evita que GraphCut
    corte prematuramente una pieza cuyo color se parece al soporte.

    La continuidad:
      - nace solo del cuerpo previamente confirmado;
      - se propaga únicamente dentro de UNKNOWN;
      - decae con la profundidad;
      - se atenúa después de bordes horizontales fuertes;
      - nunca puede expandirse fuera de UNKNOWN.

    No se usan primitivas geométricas ni dimensiones conocidas.
    """
    support = (np.asarray(support_mask) > 0) & (np.asarray(roi) > 0)
    unknown = (np.asarray(contact_unknown) > 0) & support
    object_out = (np.asarray(object_mask_outside_support) > 0) & (~support) & (np.asarray(roi) > 0)

    zero = np.zeros(
        support.shape,
        dtype=np.uint8,
    )

    if np.count_nonzero(unknown) < 200 or np.count_nonzero(object_out) < 500:
        return (
            zero,
            zero,
            {
                "status": "fallback",
                "reason": "insufficient_unknown_or_foreground",
                "recovered_pixels": 0,
            },
        )

    contact_dilate = (
        cv2.dilate(
            unknown.astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (21, 51),
            ),
            iterations=1,
        )
        > 0
    )

    training_domain = (contact_dilate | support | object_out) & (np.asarray(roi) > 0)

    visible_support_bg = support & (~unknown) & (~np.asarray(evidence["shadow_mask"]).astype(bool))

    (
        feature,
        appearance_margin,
        appearance_diag,
    ) = build_contact_feature_image(
        image,
        background,
        evidence,
        training_domain,
        object_out,
        visible_support_bg,
    )

    appearance_scale = max(
        1e-3,
        float(
            appearance_diag.get(
                "appearance_scale",
                5.0,
            )
        ),
    )

    (
        continuity_prior,
        combined_prior,
        continuity_diag,
    ) = build_contact_continuity_prior(
        image,
        object_out,
        unknown,
        appearance_margin,
        appearance_scale,
        evidence,
        args,
    )

    # El primer canal deja de ser solo afinidad cromática:
    # representa evidencia foreground combinada.
    feature = feature.copy()
    feature[:, :, 0] = np.round(
        255.0
        * np.clip(
            combined_prior,
            0.0,
            1.0,
        )
    ).astype(np.uint8)

    appearance_prob = _sigmoid01(appearance_margin / appearance_scale)

    gc = np.full(
        support.shape,
        cv2.GC_BGD,
        dtype=np.uint8,
    )

    # UNKNOWN parte como background probable.
    gc[unknown] = cv2.GC_PR_BGD

    # Evidencia foreground probable.
    positive = unknown & (
        (combined_prior >= float(args.contact_continuity_pr_fg_threshold))
        | evidence["support_object_evidence"]
        | evidence["strong"]
    )
    gc[positive] = cv2.GC_PR_FGD

    # FG seguro = interior erosionado del objeto confirmado.
    er = max(
        1,
        int(args.contact_graphcut_sure_fg_erosion_px),
    )
    sure_fg = (
        cv2.erode(
            object_out.astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * er + 1, 2 * er + 1),
            ),
            iterations=1,
        )
        > 0
    )

    gc[object_out] = cv2.GC_PR_FGD
    gc[sure_fg] = cv2.GC_FGD

    # Fuera de UNKNOWN el soporte sí es fondo seguro.
    gc[support & (~unknown)] = cv2.GC_BGD

    if np.count_nonzero(gc == cv2.GC_FGD) < 400 or np.count_nonzero(gc == cv2.GC_BGD) < 400:
        return (
            zero,
            gc,
            {
                "status": "fallback",
                "reason": "insufficient_fg_or_bg_training",
                "recovered_pixels": 0,
            },
        )

    bg_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )
    fg_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    try:
        cv2.grabCut(
            feature,
            gc,
            None,
            bg_model,
            fg_model,
            max(
                1,
                int(args.contact_graphcut_iterations),
            ),
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error as exc:
        return (
            zero,
            gc,
            {
                "status": "fallback",
                "reason": ("opencv_grabcut_error: " f"{exc}"),
                "recovered_pixels": 0,
            },
        )

    raw_fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)) & unknown

    # ------------------------------------------------------------------
    # Rescate conservador de continuidad.
    #
    # Si GraphCut rechaza un píxel que:
    #   - sigue dentro de UNKNOWN;
    #   - mantiene continuidad fuerte con el cuerpo;
    #   - no es una sombra fuerte;
    #   - no es absolutamente incompatible con la apariencia del objeto;
    # se permite volver a considerarlo candidato.
    #
    # Luego TODOS los candidatos deben seguir conectados al objeto mediante
    # _keep_contact_components.
    # ------------------------------------------------------------------
    shadow = np.asarray(evidence["shadow_mask"]).astype(bool)

    rescue = (
        unknown
        & (continuity_prior >= float(args.contact_continuity_rescue_threshold))
        & (appearance_prob >= float(args.contact_continuity_rescue_appearance_floor))
        & (~shadow)
    )

    fg = raw_fg | rescue

    fg_u8 = fg.astype(np.uint8) * 255

    # Cierre mínimo. Nunca sale de UNKNOWN.
    fg_u8 = cv2.morphologyEx(
        fg_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        ),
        iterations=1,
    )
    fg_u8 = cv2.bitwise_and(
        fg_u8,
        unknown.astype(np.uint8) * 255,
    )

    connected, comp_diag = _keep_contact_components(
        fg_u8,
        object_out,
        minimum_area=(args.contact_min_component_area),
        connect_rx=(args.contact_connect_horizontal_px),
        connect_ry=(args.contact_connect_vertical_px),
    )

    return (
        connected,
        gc,
        {
            "status": "ok",
            "unknown_pixels": int(np.count_nonzero(unknown)),
            "sure_fg_pixels": int(np.count_nonzero(sure_fg)),
            "probable_fg_pixels": int(np.count_nonzero(positive)),
            "raw_graphcut_fg_pixels": int(np.count_nonzero(raw_fg)),
            "continuity_rescue_pixels": int(np.count_nonzero(rescue)),
            "recovered_pixels": int(np.count_nonzero(connected)),
            "component_filter": comp_diag,
            "appearance_model": appearance_diag,
            "continuity_model": continuity_diag,
        },
    )


def derive_support_occlusion_search(
    support_mask: np.ndarray,
    roi: np.ndarray,
    object_seed: np.ndarray,
    evidence: dict,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Separa plataforma visible de plataforma plausiblemente ocluida.

    Regla fundamental:
        la plataforma completa nunca es espacio probable de objeto.

    Solo puede abrirse una región del soporte cuando:
        1) difiere de la captura de fondo;
        2) no es explicable como sombra;
        3) contiene evidencia visual suficiente;
        4) está conectada geodésicamente con evidencia del objeto.

    La conectividad primaria es anisotrópica: favorece continuidad vertical
    hacia la plataforma y penaliza saltos laterales.
    """
    support = (np.asarray(support_mask) > 0) & (np.asarray(roi) > 0)
    zero = np.zeros(support.shape, dtype=np.uint8)

    if np.count_nonzero(support) < 500:
        return (
            zero,
            zero,
            zero,
            {
                "status": "disabled",
                "reason": "support_not_available",
                "support_pixels": int(np.count_nonzero(support)),
            },
        )

    raw = np.asarray(evidence["raw_difference"], dtype=np.float32)
    chroma = np.asarray(evidence["chroma_z"], dtype=np.float32)
    gradient = np.asarray(evidence["gradient_z"], dtype=np.float32)
    signed_l = np.asarray(evidence["signed_delta_l"], dtype=np.float32)
    shadow = np.asarray(evidence["shadow_mask"]).astype(bool)

    shadow_dilation = max(0, int(args.support_occlusion_shadow_dilation_px))
    if shadow_dilation > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * shadow_dilation + 1, 2 * shadow_dilation + 1),
        )
        shadow_guard = (
            cv2.dilate(
                shadow.astype(np.uint8) * 255,
                k,
                iterations=1,
            )
            > 0
        )
    else:
        shadow_guard = shadow

    support_values = raw[support & np.isfinite(raw)]

    strict_threshold, strict_diag = _robust_low_change_threshold(
        support_values,
        args.support_occlusion_minimum_raw_difference,
        args.support_occlusion_strict_mad_factor,
        args.support_occlusion_stable_quantile,
    )
    loose_threshold, loose_diag = _robust_low_change_threshold(
        support_values,
        max(
            3.5,
            0.60 * args.support_occlusion_minimum_raw_difference,
        ),
        args.support_occlusion_loose_mad_factor,
        args.support_occlusion_stable_quantile,
    )

    strict_structural = (
        ((chroma >= args.support_occlusion_strict_chroma_z) & (gradient >= 1.25))
        | ((gradient >= args.support_occlusion_strict_gradient_z) & (chroma >= 0.85))
        | (
            (signed_l >= args.support_occlusion_minimum_brightening_l)
            & ((chroma >= 1.55) | (gradient >= 2.8))
        )
    )

    strict = (
        support & (~shadow_guard) & np.isfinite(raw) & (raw >= strict_threshold) & strict_structural
    )

    loose_structural = (
        (chroma >= args.support_occlusion_loose_chroma_z)
        | (gradient >= args.support_occlusion_loose_gradient_z)
        | ((signed_l >= args.support_occlusion_minimum_brightening_l) & (raw >= strict_threshold))
    )

    loose = (
        support & (~shadow_guard) & np.isfinite(raw) & (raw >= loose_threshold) & loose_structural
    )
    loose |= strict

    if args.support_occlusion_open_kernel > 0:
        loose = (
            cv2.morphologyEx(
                loose.astype(np.uint8) * 255,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        odd(args.support_occlusion_open_kernel),
                        odd(args.support_occlusion_open_kernel),
                    ),
                ),
                iterations=1,
            )
            > 0
        )

    if args.support_occlusion_close_kernel > 0:
        loose = (
            cv2.morphologyEx(
                loose.astype(np.uint8) * 255,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        odd(args.support_occlusion_close_kernel),
                        odd(args.support_occlusion_close_kernel),
                    ),
                ),
                iterations=1,
            )
            > 0
        )

    loose |= strict

    seed = np.asarray(object_seed).astype(bool)

    rx = max(
        1,
        int(args.support_occlusion_contact_horizontal_radius_px),
    )
    ry = max(
        rx,
        int(args.support_occlusion_contact_vertical_radius_px),
    )

    # Kernel anisotrópico: mucha menor tolerancia lateral.
    contact_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * rx + 1, 2 * ry + 1),
    )
    near_object = (
        cv2.dilate(
            seed.astype(np.uint8) * 255,
            contact_kernel,
            iterations=1,
        )
        > 0
    )

    # Proyección de columnas ocupadas por el cuerpo detectado.
    # Estar "cerca" lateralmente no basta: la oclusión debe caer debajo de
    # una columna que realmente contiene evidencia del objeto, admitiendo
    # solo un margen pequeño.
    seed_columns = np.any(seed, axis=0).astype(np.uint8) * 255
    column_kernel_width = 2 * max(2, rx) + 1
    seed_columns = (
        cv2.dilate(
            seed_columns.reshape(1, -1),
            np.ones((1, column_kernel_width), dtype=np.uint8),
            iterations=1,
        ).reshape(-1)
        > 0
    )

    column_projection = np.broadcast_to(
        seed_columns.reshape(1, -1),
        support.shape,
    )

    marker_primary = loose & near_object & column_projection

    minimum_primary_marker = max(
        int(args.support_occlusion_minimum_primary_marker_pixels),
        int(0.00004 * support.size),
    )

    marker_fallback = np.zeros_like(strict, dtype=bool)

    # Solo si falta contacto primario. El respaldo se restringe además a
    # la proyección horizontal del cuerpo; una reflexión lateral cercana no
    # puede iniciar una región por sí sola.
    if np.count_nonzero(marker_primary) < minimum_primary_marker:
        sy, sx = np.nonzero(seed)
        if sx.size > 0:
            fallback_ry = max(
                ry,
                int(args.support_occlusion_fallback_radius_px),
            )
            fallback_rx = max(
                rx,
                min(10, fallback_ry // 4),
            )
            fallback_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * fallback_rx + 1, 2 * fallback_ry + 1),
            )
            near_fallback = (
                cv2.dilate(
                    seed.astype(np.uint8) * 255,
                    fallback_kernel,
                    iterations=1,
                )
                > 0
            )

            projection_margin = max(6, rx)
            x0 = max(0, int(sx.min()) - projection_margin)
            x1 = min(strict.shape[1], int(sx.max()) + projection_margin + 1)
            projection = np.zeros_like(strict, dtype=bool)
            projection[:, x0:x1] = True

            marker_fallback = strict & near_fallback & projection

    marker = marker_primary | marker_fallback

    search, component_diag = _components_touching_marker(
        loose,
        marker,
        minimum_area=args.support_occlusion_minimum_loose_component_area,
    )

    # La evidencia estricta solo se conserva si pertenece a la zona conectada.
    strict_clean = np.zeros(strict.shape, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        strict.astype(np.uint8),
        connectivity=8,
    )
    selected_strict = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(args.support_occlusion_minimum_strict_component_area):
            continue
        comp = labels == label
        overlap = int(np.count_nonzero(comp & (search > 0)))
        if overlap <= 0:
            continue
        strict_clean[comp] = 255
        selected_strict.append(
            {
                "label": int(label),
                "area": area,
                "search_overlap_pixels": overlap,
            }
        )

    locked = (support & (search == 0)).astype(np.uint8) * 255

    return (
        strict_clean,
        search,
        locked,
        {
            "status": "ok",
            "support_pixels": int(np.count_nonzero(support)),
            "strict_pixels_raw": int(np.count_nonzero(strict)),
            "strict_pixels_selected": int(np.count_nonzero(strict_clean)),
            "loose_pixels": int(np.count_nonzero(loose)),
            "contact_radius_x_px": int(rx),
            "contact_radius_y_px": int(ry),
            "seed_projection_column_count": int(np.count_nonzero(seed_columns)),
            "marker_primary_pixels": int(np.count_nonzero(marker_primary)),
            "marker_primary_minimum_required": int(minimum_primary_marker),
            "marker_fallback_pixels": int(np.count_nonzero(marker_fallback)),
            "marker_fallback_used": bool(np.count_nonzero(marker_fallback) > 0),
            "search_pixels": int(np.count_nonzero(search)),
            "locked_support_background_pixels": int(np.count_nonzero(locked)),
            "search_ratio_of_support": float(
                np.count_nonzero(search) / max(np.count_nonzero(support), 1)
            ),
            "strict_threshold": strict_diag,
            "loose_threshold": loose_diag,
            "component_selection": component_diag,
            "strict_selected_components": selected_strict,
        },
    )


# ---------------------------------------------------------------------------
# Componentes con selección espacial
# ---------------------------------------------------------------------------


def bbox_gap(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    """Mide la separación entre dos cajas delimitadoras."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh

    dx = max(
        0,
        max(ax, bx) - min(ax1, bx1),
    )
    dy = max(
        0,
        max(ay, by) - min(ay1, by1),
    )

    return float(math.hypot(dx, dy))


def select_spatial_components(
    binary_mask: np.ndarray,
    anchor: np.ndarray,
    args,
) -> Tuple[np.ndarray, dict]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (binary_mask > 0).astype(np.uint8),
        connectivity=8,
    )

    h, w = binary_mask.shape
    anchor_y, anchor_x = np.nonzero(anchor > 0)
    if anchor_x.size > 0:
        expected_center = np.array(
            [float(np.mean(anchor_x)), float(np.mean(anchor_y))],
            dtype=np.float64,
        )
    else:
        expected_center = np.array(
            [0.5 * w, 0.5 * h],
            dtype=np.float64,
        )

    center_sigma = max(
        20.0,
        args.component_center_sigma_fraction * math.hypot(w, h),
    )

    components = []

    for label in range(1, count):
        x = int(
            stats[
                label,
                cv2.CC_STAT_LEFT,
            ]
        )
        y = int(
            stats[
                label,
                cv2.CC_STAT_TOP,
            ]
        )
        cw = int(
            stats[
                label,
                cv2.CC_STAT_WIDTH,
            ]
        )
        ch = int(
            stats[
                label,
                cv2.CC_STAT_HEIGHT,
            ]
        )
        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < args.minimum_component_area:
            continue

        component = labels == label
        anchor_overlap = int(np.count_nonzero(component & (anchor > 0)))
        anchor_ratio = anchor_overlap / max(area, 1)

        centroid_array = centroids[label].astype(np.float64)

        center_distance = float(np.linalg.norm(centroid_array - expected_center))

        # IMPORTANTE V2.1:
        # JSON no puede serializar np.ndarray. Se conserva el ndarray solo
        # durante los cálculos y se guarda una lista Python [x, y].
        centroid = centroid_array.astype(float).tolist()

        center_score = math.exp(-0.5 * (center_distance / center_sigma) ** 2)

        area_score = math.sqrt(
            area
            / max(
                binary_mask.size,
                1,
            )
        )

        # El anchor domina el score; área sola no puede ganar por ser una
        # sombra enorme.
        score = (
            3.5
            * min(
                1.0,
                anchor_ratio
                / max(
                    args.minimum_anchor_overlap_ratio,
                    1e-6,
                ),
            )
            + 2.0 * center_score
            + 0.8 * area_score
        )

        components.append(
            {
                "label": label,
                "area": area,
                "bbox": (
                    x,
                    y,
                    cw,
                    ch,
                ),
                "centroid": centroid,
                "anchor_overlap_pixels": anchor_overlap,
                "anchor_overlap_ratio": float(anchor_ratio),
                "center_distance_px": center_distance,
                "score": float(score),
            }
        )

    if not components:
        return (
            np.zeros_like(binary_mask),
            {
                "component_count": 0,
                "selected_labels": [],
                "components": [],
            },
        )

    components.sort(
        key=lambda c: c["score"],
        reverse=True,
    )

    primary = components[0]

    # El componente principal debe tocar el anchor, salvo que sea muy central.
    if primary["anchor_overlap_ratio"] < args.minimum_anchor_overlap_ratio and primary[
        "center_distance_px"
    ] > 0.18 * math.hypot(w, h):
        return (
            np.zeros_like(binary_mask),
            {
                "component_count": len(components),
                "selected_labels": [],
                "components": components,
                "rejected_primary": True,
            },
        )

    selected = [primary["label"]]

    # Componentes secundarios: solo piezas suficientemente grandes y muy
    # próximas al componente principal. Esto puede recuperar una arista
    # desconectada sin incorporar una sombra lateral.
    for component in components[1:]:
        if component["area"] < args.secondary_component_fraction * primary["area"]:
            continue

        gap = bbox_gap(
            primary["bbox"],
            component["bbox"],
        )

        if gap <= args.secondary_max_gap_px and (
            component["anchor_overlap_ratio"] >= 0.5 * args.minimum_anchor_overlap_ratio
            or component["center_distance_px"] < 0.20 * math.hypot(w, h)
        ):
            selected.append(component["label"])

    result = (
        np.isin(
            labels,
            selected,
        ).astype(np.uint8)
        * 255
    )

    return result, {
        "component_count": len(components),
        "selected_labels": selected,
        "primary_component": primary,
        "components": components,
    }


# ---------------------------------------------------------------------------
# Silueta completa
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# V5.0 — verificación estéreo LOCAL del contacto
# ---------------------------------------------------------------------------


def _robust_median_sigma(
    values: np.ndarray,
    minimum_sigma: float = 0.05,
) -> Tuple[float, float, int]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, float(minimum_sigma), 0
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    sigma = max(float(minimum_sigma), 1.4826 * mad)
    return median, sigma, int(x.size)


def local_contact_depth_evidence(
    disparity: Optional[np.ndarray],
    background_disparity: Optional[np.ndarray],
    confidence: Optional[np.ndarray],
    lr_error: Optional[np.ndarray],
    evidence: dict,
    support_mask: np.ndarray,
    contact_unknown: np.ndarray,
    roi: np.ndarray,
    rect_valid_mask: np.ndarray,
    args,
) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """Obtiene candidatos de oclusión SOLO dentro del contacto.

    Importante:
    -----------
    - Nunca produce foreground fuera de contact_unknown.
    - Aprende el sesgo d_actual-d_fondo sobre plato visible FUERA del contacto.
    - Si esa referencia no es estable, desactiva la cue estéreo para la vista.
    """
    shape = support_mask.shape
    empty = np.zeros(shape, dtype=np.uint8)
    delta_corrected = np.full(shape, np.nan, dtype=np.float32)

    if (
        not bool(args.local_contact_depth_verification)
        or disparity is None
        or background_disparity is None
        or confidence is None
    ):
        return (
            empty,
            empty,
            {
                "status": "disabled",
                "reason": "depth_inputs_missing_or_disabled",
                "candidate_pixels": 0,
                "strong_pixels": 0,
            },
            delta_corrected,
        )

    d = np.asarray(disparity, dtype=np.float32)
    db = np.asarray(background_disparity, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32)

    if d.shape != shape or db.shape != shape or conf.shape != shape:
        return (
            empty,
            empty,
            {
                "status": "disabled",
                "reason": "shape_mismatch",
                "candidate_pixels": 0,
                "strong_pixels": 0,
            },
            delta_corrected,
        )

    if lr_error is not None:
        lr = np.asarray(lr_error, dtype=np.float32)
        if lr.shape != shape:
            lr = None
    else:
        lr = None

    support = np.asarray(support_mask) > 0
    contact = np.asarray(contact_unknown) > 0
    domain = (
        (np.asarray(roi) > 0)
        & (np.asarray(rect_valid_mask) > 0)
        & np.isfinite(d)
        & np.isfinite(db)
        & np.isfinite(conf)
    )

    # Referencia: plataforma visible y de BAJO cambio visual,
    # explícitamente fuera de la banda donde el objeto podría ocluirla.
    reference_pool = support & (~contact) & domain

    if np.any(reference_pool):
        raw = np.asarray(evidence["raw_difference"], dtype=np.float32)
        raw_values = raw[reference_pool & np.isfinite(raw)]
        if raw_values.size:
            q = float(
                np.clip(
                    args.local_contact_reference_quantile,
                    0.20,
                    0.80,
                )
            )
            cutoff = float(np.quantile(raw_values, q))
            reference = (
                reference_pool
                & np.isfinite(raw)
                & (raw <= cutoff)
                & (conf >= float(args.local_contact_min_confidence))
            )
        else:
            reference = reference_pool & (conf >= float(args.local_contact_min_confidence))
    else:
        reference = reference_pool

    if lr is not None:
        reference &= np.isfinite(lr) & (lr <= float(args.local_contact_max_lr_error_px))

    delta_raw = d - db
    ref_values = delta_raw[reference]

    if ref_values.size < int(args.local_contact_min_reference_pixels):
        # Segundo intento: relajar SOLO la confianza, manteniendo plato visible,
        # low-change y fuera del contacto.
        reference = reference_pool
        if np.any(reference):
            raw = np.asarray(evidence["raw_difference"], dtype=np.float32)
            vals = raw[reference & np.isfinite(raw)]
            if vals.size:
                cutoff = float(
                    np.quantile(
                        vals,
                        float(np.clip(args.local_contact_reference_quantile, 0.20, 0.80)),
                    )
                )
                reference &= np.isfinite(raw) & (raw <= cutoff)
        if lr is not None:
            reference &= np.isfinite(lr) & (lr <= float(args.local_contact_max_lr_error_px))
        ref_values = delta_raw[reference]

    bias, sigma, sample_count = _robust_median_sigma(
        ref_values,
        minimum_sigma=0.08,
    )

    if sample_count < int(args.local_contact_min_reference_pixels):
        return (
            empty,
            empty,
            {
                "status": "disabled",
                "reason": "insufficient_local_platform_reference",
                "reference_pixels": int(sample_count),
                "bias_px": float(bias),
                "sigma_px": float(sigma),
                "candidate_pixels": 0,
                "strong_pixels": 0,
            },
            delta_corrected,
        )

    if sigma > float(args.local_contact_max_reference_sigma_px):
        return (
            empty,
            empty,
            {
                "status": "disabled",
                "reason": "local_platform_disparity_too_unstable",
                "reference_pixels": int(sample_count),
                "bias_px": float(bias),
                "sigma_px": float(sigma),
                "candidate_pixels": 0,
                "strong_pixels": 0,
            },
            delta_corrected,
        )

    delta_corrected = (delta_raw - float(bias)).astype(np.float32)

    weak_threshold = max(
        float(args.local_contact_min_delta_px),
        float(args.local_contact_mad_factor) * float(sigma),
    )
    strong_threshold = max(
        float(args.local_contact_strong_min_delta_px),
        float(args.local_contact_strong_mad_factor) * float(sigma),
        weak_threshold + 0.35,
    )

    stereo_valid = contact & domain & (conf >= float(args.local_contact_min_confidence))
    if lr is not None:
        stereo_valid &= np.isfinite(lr) & (lr <= float(args.local_contact_max_lr_error_px))

    weak = stereo_valid & (delta_corrected >= weak_threshold)
    strong = stereo_valid & (delta_corrected >= strong_threshold)

    return (
        weak.astype(np.uint8) * 255,
        strong.astype(np.uint8) * 255,
        {
            "status": "active",
            "reference_pixels": int(sample_count),
            "bias_px": float(bias),
            "sigma_px": float(sigma),
            "weak_threshold_px": float(weak_threshold),
            "strong_threshold_px": float(strong_threshold),
            "candidate_pixels": int(np.count_nonzero(weak)),
            "strong_pixels": int(np.count_nonzero(strong)),
            "note": (
                "Disparidad usada solo dentro de contact_unknown; "
                "sesgo aprendido sobre plataforma visible local."
            ),
        },
        delta_corrected,
    )


def geodesic_contact_recovery(
    object_mask_outside_support: np.ndarray,
    contact_unknown: np.ndarray,
    visual_weak: np.ndarray,
    visual_strong: np.ndarray,
    visual_probability: np.ndarray,
    shadow_mask: np.ndarray,
    depth_weak: np.ndarray,
    depth_strong: np.ndarray,
    args,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Recupera contacto por conectividad, no por clasificación global.

    El marcador nace EXCLUSIVAMENTE del cuerpo ya aceptado fuera del soporte.
    Ningún componente aislado de plataforma puede auto-sembrarse.
    """
    body = np.asarray(object_mask_outside_support) > 0
    contact = np.asarray(contact_unknown) > 0
    vweak = np.asarray(visual_weak).astype(bool)
    vstrong = np.asarray(visual_strong).astype(bool)
    prob = np.asarray(visual_probability, dtype=np.float32)
    shadow = np.asarray(shadow_mask).astype(bool)
    dweak = np.asarray(depth_weak) > 0
    dstrong = np.asarray(depth_strong) > 0

    # Cue visual puede entrar salvo sombra.
    visual_allowed = (
        contact
        & (~shadow)
        & (vweak | vstrong | (prob >= float(args.local_contact_visual_probability_min)))
    )

    # Cue estéreo puede rescatar incluso una zona visualmente parecida
    # al plato, pero SOLO si la geometría local dice que está delante.
    allowed = contact & (visual_allowed | dweak | dstrong)

    if int(args.local_contact_close_kernel) > 1:
        k = odd(int(args.local_contact_close_kernel))
        closed = (
            cv2.morphologyEx(
                allowed.astype(np.uint8) * 255,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
                iterations=1,
            )
            > 0
        )
        # El cierre no puede salir de la banda UNKNOWN.
        allowed = closed & contact

    # Semilla exclusivamente desde el cuerpo ya aceptado.
    rx = max(1, int(args.local_contact_seed_horizontal_radius_px))
    ry = max(1, int(args.local_contact_seed_vertical_radius_px))
    seed_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * rx + 1, 2 * ry + 1),
    )
    body_near = (
        cv2.dilate(
            body.astype(np.uint8) * 255,
            seed_kernel,
            iterations=1,
        )
        > 0
    )

    marker = allowed & body_near

    # Una cue fuerte NO crea una isla nueva; solo refuerza zonas ya alcanzables.
    strong_local = contact & (vstrong | dstrong)
    marker |= strong_local & body_near

    if not np.any(marker):
        return (
            np.zeros_like(object_mask_outside_support),
            allowed.astype(np.uint8) * 255,
            {
                "status": "empty_seed",
                "allowed_pixels": int(np.count_nonzero(allowed)),
                "recovered_pixels": 0,
                "iterations": 0,
            },
        )

    current = marker.astype(np.uint8) * 255
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    iterations = 0
    maximum = max(1, int(args.local_contact_maximum_growth_iterations))

    for iterations in range(1, maximum + 1):
        grown = cv2.dilate(current, small_kernel, iterations=1) > 0
        grown &= allowed
        nxt = grown.astype(np.uint8) * 255

        if np.array_equal(nxt, current):
            break
        current = nxt

    return (
        current,
        allowed.astype(np.uint8) * 255,
        {
            "status": "active",
            "allowed_pixels": int(np.count_nonzero(allowed)),
            "visual_allowed_pixels": int(np.count_nonzero(visual_allowed)),
            "depth_weak_pixels": int(np.count_nonzero(dweak & contact)),
            "depth_strong_pixels": int(np.count_nonzero(dstrong & contact)),
            "seed_pixels": int(np.count_nonzero(marker)),
            "recovered_pixels": int(np.count_nonzero(current)),
            "iterations": int(iterations),
            "principle": (
                "geodesic growth from accepted visual body; "
                "no isolated support component may seed itself"
            ),
        },
    )



def protect_support_rim_near_object(
    rim_guard: np.ndarray,
    support_mask: np.ndarray,
    object_seed: np.ndarray,
    valid_domain: np.ndarray,
    core_clearance_px: int = 14,
    column_margin_px: int = 10,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Evita que el guard fino recorte un objeto que cruza el borde del plato.

    El guard del filo es estático y proviene del hardware. Para no convertirlo
    en una suposición sobre la geometría del objeto, solo se desactiva en las
    columnas donde existe una semilla visual del objeto que se prolonga más
    allá de una banda cercana al soporte. Un falso borde del plato, al estar
    pegado a la elipse, no genera por sí solo ese núcleo profundo.
    """
    rim = np.asarray(rim_guard) > 0
    support = np.asarray(support_mask) > 0
    seed = np.asarray(object_seed) > 0
    valid = np.asarray(valid_domain) > 0

    if not (rim.shape == support.shape == seed.shape == valid.shape):
        raise ValueError("rim_guard, support_mask, object_seed y valid_domain deben coincidir")

    if not np.any(rim):
        zero = np.zeros(rim.shape, np.uint8)
        return zero, zero, {
            "status": "disabled",
            "reason": "rim_guard_empty",
            "protected_columns": 0,
            "raw_rim_pixels": 0,
            "effective_rim_pixels": 0,
        }

    clearance = int(np.clip(int(core_clearance_px), 1, 64))
    margin = int(np.clip(int(column_margin_px), 0, 64))

    support_u8 = support.astype(np.uint8) * 255
    near_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * clearance + 1, 2 * clearance + 1),
    )
    near_support = cv2.dilate(support_u8, near_kernel, iterations=1) > 0

    # Solo una semilla que se extiende claramente fuera del borde puede
    # proteger columnas. El filo gris, por definición, permanece en near_support.
    deep_object = seed & (~near_support) & valid
    protected_columns = np.any(deep_object, axis=0).astype(np.uint8) * 255

    if margin > 0 and np.any(protected_columns):
        protected_columns = cv2.dilate(
            protected_columns.reshape(1, -1),
            np.ones((1, 2 * margin + 1), np.uint8),
            iterations=1,
        ).reshape(-1)

    protected = np.broadcast_to(
        protected_columns.astype(bool)[None, :],
        rim.shape,
    ) & rim

    effective = rim & (~protected) & valid
    return (
        effective.astype(np.uint8) * 255,
        protected.astype(np.uint8) * 255,
        {
            "status": "active",
            "core_clearance_px": int(clearance),
            "column_margin_px": int(margin),
            "deep_object_pixels": int(np.count_nonzero(deep_object)),
            "protected_columns": int(np.count_nonzero(protected_columns)),
            "protected_rim_pixels": int(np.count_nonzero(protected)),
            "raw_rim_pixels": int(np.count_nonzero(rim)),
            "effective_rim_pixels": int(np.count_nonzero(effective)),
        },
    )


def refine_contact_by_visual_edges(image, background, mask, support_mask):
    """Refinamiento por apariencia/bordes, solo dentro del soporte.

    Conserva el cuerpo superior. No añade píxeles ni usa profundidad.
    La comparación con el fondo orienta etiquetas probables, nunca un veto.
    """
    support = support_mask > 0
    body = (mask > 0) & ~support
    sure = cv2.erode(body.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if np.count_nonzero(sure) < 100:
        return mask.copy(), {"status": "insufficient_body"}
    labels = np.full(mask.shape, cv2.GC_BGD, np.uint8)
    domain = cv2.dilate((mask > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    labels[domain] = cv2.GC_PR_BGD
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    back = cv2.cvtColor(background, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma_change = np.linalg.norm(lab[:, :, 1:] - back[:, :, 1:], axis=2)
    labels[(mask > 0) & (chroma_change > 6.0)] = cv2.GC_PR_FGD
    labels[body] = cv2.GC_PR_FGD
    labels[sure] = cv2.GC_FGD
    # GrabCut penaliza romper continuidad de apariencia y favorece bordes.
    smooth = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.setRNGSeed(0)
    cv2.grabCut(smooth, labels, None, np.zeros((1,65), np.float64),
                np.zeros((1,65), np.float64), 5, cv2.GC_INIT_WITH_MASK)
    accepted = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
    result = mask.copy()
    result[support & ~accepted] = 0
    return result, {"status": "active", "removed_pixels": int(np.count_nonzero((mask > 0) & (result == 0))),
                    "depth_used": False, "outside_support_unchanged": True}


@operacion("Crear silueta y resolver contacto con soporte")
def create_silhouette(
    image: np.ndarray,
    background: np.ndarray,
    rect_valid_mask: np.ndarray,
    support_mask: np.ndarray,
    args,
    disparity: Optional[np.ndarray] = None,
    background_disparity: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
    lr_error: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    h, w = image.shape[:2]

    if rect_valid_mask.shape != (h, w):
        raise ValueError(
            "rect_valid_mask no coincide con imagen: " f"{rect_valid_mask.shape} vs {(h, w)}"
        )

    roi, anchor, roi_diag, adaptive_visual_candidate = build_product_domain_and_anchor(
        image,
        background,
        rect_valid_mask,
        support_mask,
        args,
    )

    # La superficie superior detectada permanece intacta. Se manejan dos
    # exclusiones independientes del hardware:
    #   1) falda inferior: cuerpo/aro gris grande;
    #   2) rim guard: filo fino inmediatamente exterior a la elipse.
    # El rim guard se protege después en columnas ocupadas por un núcleo real
    # del objeto, para no recortar el objeto cuando cruza visualmente ese borde.
    if bool(args.support_hardware_exclusion):
        support_hardware_skirt, support_hardware_diag = (
            build_support_hardware_exclusion_mask(
                support_mask,
                rect_valid_mask,
                lateral_fraction=args.support_hardware_lateral_fraction,
                downward_fraction=args.support_hardware_downward_fraction,
                lower_start_fraction=args.support_hardware_lower_start_fraction,
                minimum_lateral_px=args.support_hardware_min_lateral_px,
                minimum_downward_px=args.support_hardware_min_downward_px,
                maximum_lateral_px=args.support_hardware_max_lateral_px,
                maximum_downward_px=args.support_hardware_max_downward_px,
            )
        )
    else:
        support_hardware_skirt = np.zeros_like(support_mask)
        support_hardware_diag = {
            "status": "disabled",
            "reason": "disabled_by_argument",
            "exclusion_pixels": 0,
        }

    if bool(args.support_rim_guard):
        support_rim_guard_raw, support_rim_raw_diag = build_support_rim_guard_mask(
            support_mask,
            rect_valid_mask,
            rim_px=args.support_rim_guard_px,
        )
    else:
        support_rim_guard_raw = np.zeros_like(support_mask)
        support_rim_raw_diag = {
            "status": "disabled",
            "reason": "disabled_by_argument",
            "rim_pixels": 0,
        }

    support_hardware_skirt_bool = (support_hardware_skirt > 0) & (roi > 0)

    evidence = compute_visual_evidence(image, background, roi, rect_valid_mask, support_mask, args)

    # La falda inferior sí es un veto seguro durante la búsqueda de semilla.
    # El filo fino aún NO se veta: primero necesitamos saber dónde está el
    # cuerpo real del objeto para no cortar sus columnas.
    strong_for_seed = evidence["strong"].copy()
    strong_for_seed[support_hardware_skirt_bool] = False
    strong_seed = strong_for_seed.astype(np.uint8) * 255
    selected_seed = select_relevant_components(
        strong_seed,
        minimum_area=max(300, int(args.minimum_component_area // 3)),
        secondary_fraction=0.18,
        prior_mask=None,
    )
    if np.count_nonzero(selected_seed) >= 300:
        anchor, seed_diag = expanded_bbox_mask(
            selected_seed,
            roi,
            margin_fraction=args.seed_anchor_margin_fraction,
            minimum_margin_px=args.adaptive_anchor_minimum_margin_px,
        )
        roi_diag["seed_anchor_source"] = "strong_object_evidence"
        roi_diag["seed_anchor"] = seed_diag
    else:
        roi_diag["seed_anchor_source"] = "adaptive_visual_fallback"

    # Activar el filo fino solo donde no hay evidencia de un cuerpo real que
    # se prolonga fuera de la vecindad inmediata del plato. Así el pequeño
    # borde gris queda bloqueado, pero el objeto no se recorta.
    (
        support_rim_guard,
        support_rim_protected,
        support_rim_guard_diag,
    ) = protect_support_rim_near_object(
        support_rim_guard_raw,
        support_mask,
        selected_seed,
        rect_valid_mask,
        core_clearance_px=args.support_rim_object_core_clearance_px,
        column_margin_px=args.support_rim_object_column_margin_px,
    )

    support_hardware_exclusion = cv2.bitwise_or(
        support_hardware_skirt,
        support_rim_guard,
    )
    support_hardware_bool = (support_hardware_exclusion > 0) & (roi > 0)

    support_bool = (support_mask > 0) & (roi > 0)

    # ------------------------------------------------------------------
    # V3.2: el soporte NO se clasifica aquí como fondo/objeto.
    # Primero se segmenta únicamente el cuerpo fuera del soporte.
    # Después una segunda etapa resuelve el contacto.
    # ------------------------------------------------------------------
    object_seed_for_support = (selected_seed > 0) & (~support_bool) & (anchor > 0)

    if np.count_nonzero(object_seed_for_support) < 120:
        fallback_candidate = (adaptive_visual_candidate > 0) & (~support_hardware_bool)
        fallback_visual_seed = select_relevant_components(
            (fallback_candidate.astype(np.uint8) * 255),
            minimum_area=max(
                300,
                int(args.minimum_component_area // 3),
            ),
            secondary_fraction=0.12,
            prior_mask=None,
        )
        object_seed_for_support = (fallback_visual_seed > 0) & (~support_bool) & (anchor > 0)

    # Primero se calcula evidencia estricta dentro del soporte. Esta salida
    # ya exige conexión espacial con el cuerpo y por tanto puede usarse para
    # ampliar de forma segura la banda UNKNOWN en objetos cuya base aparece
    # lateralmente dentro de la proyección del plato.
    (
        support_occlusion_strict,
        _legacy_support_search,
        _legacy_support_locked,
        support_occlusion_diag,
    ) = derive_support_occlusion_search(
        support_mask,
        roi,
        object_seed_for_support,
        evidence,
        args,
    )

    adaptive_contact_extension, adaptive_contact_diag = visual_contact_extension(
        object_seed_for_support, support_mask, roi, evidence, support_hardware_exclusion
    )

    (
        support_contact_unknown,
        support_background_locked,
        support_contact_band_diag,
    ) = build_contact_ambiguity_band(
        support_mask,
        roi,
        object_seed_for_support,
        args,
        support_evidence=support_occlusion_strict,
        visual_extension=adaptive_contact_extension,
    )
    support_contact_band_diag["adaptive_visual_extension"] = adaptive_contact_diag

    support_search_bool = support_contact_unknown > 0
    support_strict_bool = support_occlusion_strict > 0

    weak_effective = evidence["weak"].copy()
    strong_effective = evidence["strong"].copy()

    # Primera segmentación: exclusivamente fuera del soporte y fuera del
    # cuerpo físico conocido de la plataforma.
    weak_effective[support_bool] = False
    strong_effective[support_bool] = False
    weak_effective[support_hardware_bool] = False
    strong_effective[support_hardware_bool] = False

    effective_shadow_mask = (
        evidence["shadow_mask"] & (~evidence["support_object_evidence"])
    ) | support_bool | support_hardware_bool

    candidate = shadow_aware_hysteresis(
        weak_effective,
        strong_effective,
        effective_shadow_mask,
    )

    # Morfología deliberadamente pequeña para no volver a conectar
    # sombra/plataforma con el objeto.
    if args.open_kernel > 0:
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    odd(args.open_kernel),
                    odd(args.open_kernel),
                ),
            ),
            iterations=1,
        )

    if args.close_kernel > 0:
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    odd(args.close_kernel),
                    odd(args.close_kernel),
                ),
            ),
            iterations=1,
        )

    candidate = cv2.bitwise_and(
        candidate,
        roi,
    )
    candidate[support_hardware_bool] = 0

    mask, component_diag = select_spatial_components(
        candidate,
        anchor,
        args,
    )

    # ------------------------------------------------------------------
    # Segunda segmentación: resolver únicamente el contacto con el soporte.
    # Todo píxel del soporte que haya entrado en la máscara preliminar se
    # elimina y debe ser revalidado por este paso.
    # ------------------------------------------------------------------
    mask_outside_support = mask.copy()
    mask_outside_support[support_bool] = 0
    mask_outside_support[support_hardware_bool] = 0

    # ------------------------------------------------------------------
    # V5.0 — NO usar GraphCut para separar objeto/plataforma.
    #
    # 1) el cuerpo visual fuera del soporte queda intacto;
    # 2) la disparidad se compara con el fondo SOLO dentro del UNKNOWN;
    # 3) la recuperación crece geodésicamente desde el cuerpo aceptado.
    #
    # Resultado: una pared o una zona lejana con disparidad distinta jamás
    # puede entrar porque no pertenece al contacto y no está conectada al cuerpo.
    # ------------------------------------------------------------------
    (
        support_depth_weak,
        support_depth_strong,
        support_depth_diag,
        support_depth_delta,
    ) = local_contact_depth_evidence(
        disparity=disparity,
        background_disparity=background_disparity,
        confidence=confidence,
        lr_error=lr_error,
        evidence=evidence,
        support_mask=support_mask,
        contact_unknown=support_contact_unknown,
        roi=roi,
        rect_valid_mask=rect_valid_mask,
        args=args,
    )

    (
        support_contact_recovered,
        support_contact_allowed,
        support_contact_graphcut_diag,
    ) = geodesic_contact_recovery(
        object_mask_outside_support=mask_outside_support,
        contact_unknown=support_contact_unknown,
        visual_weak=evidence["weak"],
        visual_strong=evidence["strong"],
        visual_probability=evidence["probability"],
        shadow_mask=evidence["shadow_mask"],
        depth_weak=support_depth_weak,
        depth_strong=support_depth_strong,
        args=args,
    )

    # Mantener el nombre histórico "trimap" solo para compatibilidad
    # diagnóstica: 0=fondo bloqueado, 128=candidato local, 255=recuperado.
    support_contact_trimap = np.zeros_like(mask)
    support_contact_trimap[support_contact_allowed > 0] = 128
    support_contact_trimap[support_contact_recovered > 0] = 255

    mask = cv2.bitwise_or(
        mask_outside_support,
        support_contact_recovered,
    )

    graphcut_recovered = np.zeros_like(mask)
    graphcut_trimap = np.zeros_like(mask)
    graphcut_diag = {
        "status": "disabled",
        "recovered_pixels": 0,
    }

    if args.graphcut_recovery and np.count_nonzero(mask) > 0:
        # El GraphCut general solo puede trabajar FUERA del soporte.
        # La frontera objeto-plataforma ya fue resuelta por el paso V3.2.
        outside_support = (~support_bool) & (~support_hardware_bool) & (roi > 0)

        weak_near_object = weak_effective & (anchor > 0) & outside_support

        probable_graphcut = (weak_near_object).astype(np.uint8) * 255

        graphcut_anchor = ((anchor > 0) & outside_support).astype(np.uint8) * 255

        no_support_search = np.zeros_like(support_mask)

        (
            graphcut_mask,
            graphcut_recovered,
            graphcut_trimap,
            graphcut_diag,
        ) = graphcut_recover_foreground(
            image,
            mask,
            probable_graphcut,
            graphcut_anchor,
            no_support_search,
            roi,
            iterations=args.graphcut_iterations,
            sure_fg_erosion_px=args.graphcut_sure_fg_erosion_px,
        )

        # El GraphCut general tampoco puede modificar el soporte.
        graphcut_mask[support_bool] = mask[support_bool]
        graphcut_recovered[support_bool] = 0
        graphcut_mask[support_hardware_bool] = 0
        graphcut_recovered[support_hardware_bool] = 0

        # Recuperación: no sustitución.
        mask = cv2.bitwise_or(mask, graphcut_mask)

        mask, post_graphcut_components = select_spatial_components(
            mask,
            anchor,
            args,
        )
        graphcut_diag["post_component_selection"] = post_graphcut_components

    mask = fill_small_holes(
        mask,
        args.maximum_hole_area,
    )

    # Seguridad final: el relleno morfológico no puede volver a introducir el
    # aro físico de la plataforma. support_mask no se toca.
    mask[support_hardware_bool] = 0

    mask = cv2.bitwise_and(
        mask,
        roi,
    )

    mask, edge_cut_diag = refine_contact_by_visual_edges(image, background, mask, support_mask)

    area = int(np.count_nonzero(mask))

    ys, xs = np.nonzero(mask)

    centroid = (
        [
            float(np.mean(xs)),
            float(np.mean(ys)),
        ]
        if area > 0
        else None
    )

    diagnostics = {
        **roi_diag,
        "contact_edge_refinement": edge_cut_diag,
        "noise_scales": evidence["noise_scales"],
        "area_pixels": area,
        "area_ratio": area / float(h * w),
        "centroid_px": centroid,
        "weak_pixels": int(np.count_nonzero(evidence["weak"])),
        "strong_pixels": int(np.count_nonzero(evidence["strong"])),
        "shadow_pixels_rejected": int(np.count_nonzero(evidence["shadow_mask"])),
        "support_pixels": int(np.count_nonzero(support_mask)),
        "support_hardware_exclusion": {
            **support_hardware_diag,
            "combined_exclusion_pixels": int(np.count_nonzero(support_hardware_bool)),
            "rim_raw": support_rim_raw_diag,
            "rim_effective": support_rim_guard_diag,
        },
        "support_hardware_exclusion_pixels": int(
            np.count_nonzero(support_hardware_bool)
        ),
        "support_rim_guard_raw_pixels": int(np.count_nonzero(support_rim_guard_raw)),
        "support_rim_guard_pixels": int(np.count_nonzero(support_rim_guard)),
        "support_rim_protected_pixels": int(np.count_nonzero(support_rim_protected)),
        "support_shadow_pixels_hard_rejected": int(
            np.count_nonzero(evidence["support_shadow_reject"])
        ),
        "support_object_evidence_pixels": int(
            np.count_nonzero(evidence["support_object_evidence"])
        ),
        "support_occlusion": support_occlusion_diag,
        "support_occlusion_strict_pixels": int(np.count_nonzero(support_occlusion_strict)),
        "support_occlusion_search_pixels": int(np.count_nonzero(support_contact_unknown)),
        "support_background_locked_pixels": int(np.count_nonzero(support_background_locked)),
        "support_contact_band": support_contact_band_diag,
        "support_contact_method": "v5_3_local_geodesic_rgbd_with_strict_evidence_extension",
        "support_contact_graphcut": {
            "status": "replaced",
            "replacement": "local_geodesic_rgbd_contact_v5",
        },
        "support_contact_recovery": support_contact_graphcut_diag,
        "support_contact_depth": support_depth_diag,
        "support_contact_recovered_pixels": int(np.count_nonzero(support_contact_recovered)),
        "graphcut": graphcut_diag,
        "candidate_pixels_before_component_selection": int(np.count_nonzero(candidate)),
        "noise_sample_pixels": int(np.count_nonzero(evidence["stable_mask"])),
        "component_selection": component_diag,
        "probability": finite_stats(evidence["probability"][roi > 0]),
        "raw_difference": finite_stats(evidence["raw_difference"][roi > 0]),
    }

    debug = {
        "roi": roi,
        "anchor": anchor,
        "adaptive_visual_candidate": adaptive_visual_candidate,
        "shadow_mask": evidence["shadow_mask"].astype(np.uint8) * 255,
        "support_mask": (support_mask > 0).astype(np.uint8) * 255,
        "support_hardware_skirt": support_hardware_skirt,
        "support_rim_guard_raw": support_rim_guard_raw,
        "support_rim_guard": support_rim_guard,
        "support_rim_protected": support_rim_protected,
        "support_hardware_exclusion": support_hardware_exclusion,
        "support_shadow_reject": evidence["support_shadow_reject"].astype(np.uint8) * 255,
        "support_object_evidence": evidence["support_object_evidence"].astype(np.uint8) * 255,
        "support_occlusion_strict": support_occlusion_strict,
        "support_occlusion_search": support_contact_unknown,
        "support_contact_unknown": support_contact_unknown,
        "adaptive_contact_extension": adaptive_contact_extension,
        "support_contact_recovered": support_contact_recovered,
        "support_contact_trimap": support_contact_trimap,
        "support_contact_allowed": support_contact_allowed,
        "support_contact_depth_weak": support_depth_weak,
        "support_contact_depth_strong": support_depth_strong,
        "support_contact_depth_delta": support_depth_delta,
        "support_background_locked": support_background_locked,
        "object_seed_for_support": object_seed_for_support.astype(np.uint8) * 255,
        "effective_shadow_mask": effective_shadow_mask.astype(np.uint8) * 255,
        "graphcut_recovered": graphcut_recovered,
        "graphcut_trimap": graphcut_trimap,
        "candidate": candidate,
        "score": evidence["score"],
        "chroma_z": evidence["chroma_z"],
        "gradient_z": evidence["gradient_z"],
    }

    return (
        mask,
        evidence["probability"],
        diagnostics,
        debug,
    )


# ---------------------------------------------------------------------------
# Calidad de sesión
# ---------------------------------------------------------------------------


def session_quality(
    records: List[dict],
    args,
) -> dict:
    valid = [r for r in records if r["area_pixels"] > 0]

    areas = np.asarray(
        [r["area_pixels"] for r in valid],
        dtype=np.float64,
    )

    if len(areas) >= 3:
        area_median = float(np.median(areas))
        area_mad = float(np.median(np.abs(areas - area_median)))
        area_limit = max(
            0.12 * area_median,
            args.session_area_mad_factor * 1.4826 * area_mad,
        )
    else:
        area_median = None
        area_mad = None
        area_limit = None

    centroids = [r["centroid_px"] for r in valid if r["centroid_px"] is not None]

    centroid_median = (
        np.median(
            np.asarray(
                centroids,
                dtype=np.float64,
            ),
            axis=0,
        )
        if centroids
        else None
    )

    for record in records:
        reasons = list(record["local_reasons"])

        if area_median is not None and abs(record["area_pixels"] - area_median) > area_limit:
            reasons.append(
                "Área atípica en la sesión: "
                f"{record['area_pixels']} px vs "
                f"mediana {area_median:.1f} px."
            )

        centroid = record.get("centroid_px")

        if centroid_median is not None and centroid is not None:
            deviation = float(
                np.linalg.norm(
                    np.asarray(
                        centroid,
                        dtype=np.float64,
                    )
                    - centroid_median
                )
            )

            record["centroid_session_deviation_px"] = deviation

            if deviation > args.session_centroid_tolerance_px:
                reasons.append(
                    "Centroide atípico: "
                    f"{deviation:.1f} px > "
                    f"{args.session_centroid_tolerance_px:.1f} px."
                )

        record["session_reasons"] = reasons

        record["session_quality"] = "accepted" if not reasons else "warning"

    return {
        "area_median_pixels": area_median,
        "area_mad_pixels": area_mad,
        "area_tolerance_pixels": area_limit,
        "centroid_median_px": (
            None if centroid_median is None else centroid_median.astype(float).tolist()
        ),
        "centroid_tolerance_px": float(args.session_centroid_tolerance_px),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _process_mask_view(task):
    from utilidades_rendimiento import contexto
    ctx = contexto()
    args = ctx["args"]
    source_dir, output_dir = ctx["source_dir"], ctx["output_dir"]
    background, support_mask = ctx["background"], ctx["support_mask"]
    background_disparity = ctx["background_disparity"]
    total_views = ctx["total_views"]
    index, view = task
    stem = str(view["stem"])
    angle = float(parse_angle(stem))

    image_path = source_dir / f"{stem}_rect_L.png"

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        print(f"[ERROR] " f"No se pudo leer " f"{image_path}")
        return None

    if image.shape != background.shape:
        raise ValueError(
            f"{stem}: imagen " f"{image.shape} y fondo " f"{background.shape} " "no coinciden."
        )

    rect_valid_path = source_dir / f"{stem}_rect_valid_mask.png"

    rect_valid = cv2.imread(
        str(rect_valid_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if rect_valid is None:
        raise FileNotFoundError("Falta rect_valid_mask: " f"{rect_valid_path}")

    disparity = None
    confidence = None
    lr_error = None

    disparity_path = source_dir / f"{stem}_disparity_lr.npy"
    if not disparity_path.is_file():
        disparity_path = source_dir / f"{stem}_disparity.npy"

    confidence_path = source_dir / f"{stem}_confidence.npy"
    lr_error_path = source_dir / f"{stem}_lr_error.npy"

    try:
        if disparity_path.is_file():
            disparity = np.load(str(disparity_path)).astype(np.float32)
        if confidence_path.is_file():
            confidence = np.load(str(confidence_path)).astype(np.float32)
        if lr_error_path.is_file():
            lr_error = np.load(str(lr_error_path)).astype(np.float32)
    except Exception as exc:
        print(f"[WARN] {stem}: no se pudieron cargar auxiliares estéreo: {exc}")
        disparity = None
        confidence = None
        lr_error = None

    (
        mask,
        probability,
        diagnostics,
        debug,
    ) = create_silhouette(
        image,
        background,
        rect_valid,
        support_mask,
        args,
        disparity=disparity,
        background_disparity=background_disparity,
        confidence=confidence,
        lr_error=lr_error,
    )

    area_ratio = float(diagnostics["area_ratio"])

    reasons = []

    if area_ratio < args.minimum_area_ratio:
        reasons.append("Máscara demasiado pequeña: " f"{area_ratio:.2%}.")

    if area_ratio > args.maximum_area_ratio:
        reasons.append("Máscara demasiado grande: " f"{area_ratio:.2%}.")

    if diagnostics["centroid_px"] is None:
        reasons.append("Máscara vacía.")

    # Salidas principales, compatibles con 04.
    mask_path = output_dir / f"{stem}_silhouette_mask.png"
    prob_path = output_dir / f"{stem}_silhouette_probability.npy"
    prob_vis_path = output_dir / f"{stem}_silhouette_probability_vis.png"
    overlay_path = output_dir / f"{stem}_silhouette_overlay.png"
    stats_path = output_dir / f"{stem}_silhouette_stats.json"

    imwrite_checked(
        str(mask_path),
        mask,
    )
    np.save(
        str(prob_path),
        probability,
    )
    imwrite_checked(
        str(prob_vis_path),
        scalar_visualization(
            probability,
            np.isfinite(probability),
            cv2.COLORMAP_VIRIDIS,
        ),
    )
    imwrite_checked(
        str(overlay_path),
        overlay_mask(
            image,
            mask,
            (f"{angle:05.1f}° " "| silueta V5.3 local"),
            probability,
        ),
    )

    # Diagnósticos nuevos.
    roi_path = output_dir / f"{stem}_debug_roi.png"
    anchor_path = output_dir / f"{stem}_debug_anchor.png"
    shadow_path = output_dir / f"{stem}_debug_shadow_rejected.png"
    support_path = output_dir / f"{stem}_debug_support_hardware.png"
    support_hardware_exclusion_path = (
        output_dir / f"{stem}_debug_support_hardware_exclusion.png"
    )
    support_hardware_exclusion_overlay_path = (
        output_dir / f"{stem}_debug_support_hardware_exclusion_overlay.png"
    )
    support_rim_guard_path = output_dir / f"{stem}_debug_support_rim_guard.png"
    support_rim_guard_overlay_path = (
        output_dir / f"{stem}_debug_support_rim_guard_overlay.png"
    )
    capture_volume_path = output_dir / f"{stem}_debug_capture_volume.png"
    effective_shadow_path = output_dir / f"{stem}_debug_effective_shadow.png"
    graphcut_recovered_path = output_dir / f"{stem}_debug_graphcut_recovered.png"
    graphcut_trimap_path = output_dir / f"{stem}_debug_graphcut_trimap.png"
    support_shadow_path = output_dir / f"{stem}_debug_support_shadow_rejected.png"
    support_object_path = output_dir / f"{stem}_debug_support_object_evidence.png"
    support_occlusion_strict_path = output_dir / f"{stem}_debug_support_occlusion_strict.png"
    support_occlusion_search_path = output_dir / f"{stem}_debug_support_occlusion_search.png"
    support_background_locked_path = output_dir / f"{stem}_debug_support_background_locked.png"
    support_seed_path = output_dir / f"{stem}_debug_support_object_seed.png"
    support_occlusion_overlay_path = output_dir / f"{stem}_debug_support_occlusion_overlay.png"
    support_contact_unknown_path = output_dir / f"{stem}_debug_support_contact_unknown.png"
    support_contact_recovered_path = output_dir / f"{stem}_debug_support_contact_recovered.png"
    support_contact_trimap_path = output_dir / f"{stem}_debug_support_contact_trimap.png"
    support_contact_allowed_path = output_dir / f"{stem}_debug_support_contact_allowed.png"
    support_contact_depth_weak_path = (
        output_dir / f"{stem}_debug_support_contact_depth_weak.png"
    )
    support_contact_depth_strong_path = (
        output_dir / f"{stem}_debug_support_contact_depth_strong.png"
    )
    support_contact_depth_delta_path = (
        output_dir / f"{stem}_debug_support_contact_depth_delta.png"
    )
    candidate_path = output_dir / f"{stem}_debug_candidate_before_components.png"
    adaptive_candidate_path = output_dir / f"{stem}_debug_adaptive_visual_candidate.png"

    debug_roi = image.copy()
    contours, _ = cv2.findContours(
        debug["roi"],
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        debug_roi,
        contours,
        -1,
        (0, 255, 255),
        3,
    )
    imwrite_checked(
        str(roi_path),
        debug_roi,
    )

    debug_anchor = image.copy()
    contours, _ = cv2.findContours(
        debug["anchor"],
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        debug_anchor,
        contours,
        -1,
        (255, 0, 255),
        3,
    )
    imwrite_checked(
        str(anchor_path),
        debug_anchor,
    )

    shadow_overlay = image.copy()
    tint = np.zeros_like(image)
    tint[:, :, 2] = debug["shadow_mask"]
    shadow_overlay = cv2.addWeighted(
        shadow_overlay,
        1.0,
        tint,
        0.50,
        0.0,
    )
    imwrite_checked(
        str(shadow_path),
        shadow_overlay,
    )
    imwrite_checked(
        str(candidate_path),
        debug["candidate"],
    )
    imwrite_checked(
        str(adaptive_candidate_path),
        debug["adaptive_visual_candidate"],
    )
    imwrite_checked(str(support_path), debug["support_mask"])
    imwrite_checked(
        str(support_hardware_exclusion_path),
        debug["support_hardware_exclusion"],
    )

    hardware_overlay = image.copy()
    skirt_bool = debug["support_hardware_skirt"] > 0
    if np.any(skirt_bool):
        magenta = np.array([255, 0, 255], dtype=np.float32)
        base = hardware_overlay[skirt_bool].astype(np.float32)
        hardware_overlay[skirt_bool] = np.clip(
            0.55 * base + 0.45 * magenta, 0, 255
        ).astype(np.uint8)

    rim_bool = debug["support_rim_guard"] > 0
    if np.any(rim_bool):
        green = np.array([0, 255, 0], dtype=np.float32)
        base = hardware_overlay[rim_bool].astype(np.float32)
        hardware_overlay[rim_bool] = np.clip(
            0.55 * base + 0.45 * green, 0, 255
        ).astype(np.uint8)

    # Contorno verde exterior = superficie útil + filo fino. La máscara
    # support_mask original sigue intacta internamente.
    support_plus_rim = cv2.bitwise_or(
        debug["support_mask"],
        debug["support_rim_guard"],
    )
    support_contours, _ = cv2.findContours(
        support_plus_rim,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        hardware_overlay,
        support_contours,
        -1,
        (0, 255, 0),
        2,
    )
    imwrite_checked(
        str(support_hardware_exclusion_overlay_path),
        hardware_overlay,
    )

    imwrite_checked(str(support_rim_guard_path), debug["support_rim_guard"])
    rim_overlay = image.copy()
    rim_tint = np.zeros_like(image)
    rim_tint[:, :, 1] = debug["support_rim_guard"]
    rim_overlay = cv2.addWeighted(rim_overlay, 1.0, rim_tint, 0.45, 0.0)
    rim_contours, _ = cv2.findContours(
        support_plus_rim,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(rim_overlay, rim_contours, -1, (0, 255, 0), 2)
    imwrite_checked(str(support_rim_guard_overlay_path), rim_overlay)

    imwrite_checked(str(capture_volume_path), debug["roi"])
    imwrite_checked(str(effective_shadow_path), debug["effective_shadow_mask"])
    imwrite_checked(
        str(graphcut_recovered_path),
        debug["graphcut_recovered"],
    )
    imwrite_checked(
        str(graphcut_trimap_path),
        graphcut_trimap_visualization(debug["graphcut_trimap"]),
    )
    imwrite_checked(str(support_shadow_path), debug["support_shadow_reject"])
    imwrite_checked(str(support_object_path), debug["support_object_evidence"])
    imwrite_checked(str(support_occlusion_strict_path), debug["support_occlusion_strict"])
    imwrite_checked(str(support_occlusion_search_path), debug["support_occlusion_search"])
    imwrite_checked(str(support_background_locked_path), debug["support_background_locked"])
    imwrite_checked(
        str(output_dir / f"{stem}_debug_adaptive_contact_extension.png"),
        debug["adaptive_contact_extension"],
    )
    imwrite_checked(str(support_seed_path), debug["object_seed_for_support"])

    imwrite_checked(
        str(support_contact_unknown_path),
        debug["support_contact_unknown"],
    )
    imwrite_checked(
        str(support_contact_recovered_path),
        debug["support_contact_recovered"],
    )
    imwrite_checked(
        str(support_contact_trimap_path),
        graphcut_trimap_visualization(debug["support_contact_trimap"]),
    )
    imwrite_checked(
        str(support_contact_allowed_path),
        debug["support_contact_allowed"],
    )
    imwrite_checked(
        str(support_contact_depth_weak_path),
        debug["support_contact_depth_weak"],
    )
    imwrite_checked(
        str(support_contact_depth_strong_path),
        debug["support_contact_depth_strong"],
    )
    imwrite_checked(
        str(support_contact_depth_delta_path),
        scalar_visualization(
            debug["support_contact_depth_delta"],
            np.isfinite(debug["support_contact_depth_delta"]),
            cv2.COLORMAP_TURBO,
        ),
    )

    support_overlay = image.copy()
    tint = np.zeros_like(image)
    tint[:, :, 1] = debug["support_occlusion_search"]
    tint[:, :, 2] = debug["support_background_locked"]
    support_overlay = cv2.addWeighted(support_overlay, 0.82, tint, 0.35, 0.0)

    # Mostrar también en ESTE mismo diagnóstico la falda de hardware.
    # Antes la exclusión sí se aplicaba a la máscara final, pero este
    # overlay no la dibujaba, por lo que visualmente parecía que nada
    # había cambiado. Se usa magenta para distinguirla del rojo/verde
    # ya empleados por la lógica de oclusión del soporte.
    skirt_bool = debug["support_hardware_skirt"] > 0
    if np.any(skirt_bool):
        hw_color = np.array([255, 0, 255], dtype=np.float32)  # BGR: magenta
        base = support_overlay[skirt_bool].astype(np.float32)
        support_overlay[skirt_bool] = np.clip(
            0.55 * base + 0.45 * hw_color,
            0,
            255,
        ).astype(np.uint8)

    # El filo fino se muestra en verde porque forma parte del hardware
    # visible del plato que queremos cubrir. support_mask no se altera: la
    # unión solo existe para diagnóstico y veto exterior.
    rim_bool = debug["support_rim_guard"] > 0
    if np.any(rim_bool):
        rim_color = np.array([0, 255, 0], dtype=np.float32)
        base = support_overlay[rim_bool].astype(np.float32)
        support_overlay[rim_bool] = np.clip(
            0.55 * base + 0.45 * rim_color,
            0,
            255,
        ).astype(np.uint8)

    support_plus_rim = cv2.bitwise_or(
        debug["support_mask"],
        debug["support_rim_guard"],
    )
    support_contours, _ = cv2.findContours(
        support_plus_rim,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(support_overlay, support_contours, -1, (0, 255, 0), 2)

    contours, _ = cv2.findContours(
        debug["support_occlusion_strict"],
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(support_overlay, contours, -1, (0, 255, 255), 2)
    imwrite_checked(
        str(support_occlusion_overlay_path),
        support_overlay,
    )

    record = {
        "stem": stem,
        "angle_deg": angle,
        "source_local_quality": (view.get("local_quality")),
        "source_session_quality": (view.get("session_quality")),
        **diagnostics,
        "local_quality": ("accepted" if not reasons else "warning"),
        "local_reasons": reasons,
        "session_quality": ("pending"),
        "session_reasons": [],
        "outputs": {
            "mask": str(mask_path),
            "probability": str(prob_path),
            "overlay": str(overlay_path),
            "debug_roi": str(roi_path),
            "debug_anchor": str(anchor_path),
            "debug_shadow": str(shadow_path),
            "debug_candidate": str(candidate_path),
            "debug_support_hardware_exclusion": str(
                support_hardware_exclusion_path
            ),
            "debug_support_hardware_exclusion_overlay": str(
                support_hardware_exclusion_overlay_path
            ),
            "debug_support_rim_guard": str(support_rim_guard_path),
            "debug_support_rim_guard_overlay": str(support_rim_guard_overlay_path),
            "debug_support_occlusion_strict": str(support_occlusion_strict_path),
            "debug_support_occlusion_search": str(support_occlusion_search_path),
            "debug_support_background_locked": str(support_background_locked_path),
            "debug_support_occlusion_overlay": str(support_occlusion_overlay_path),
            "debug_support_contact_unknown": str(support_contact_unknown_path),
            "debug_support_contact_recovered": str(support_contact_recovered_path),
            "debug_support_contact_trimap": str(support_contact_trimap_path),
            "debug_support_contact_allowed": str(support_contact_allowed_path),
            "debug_support_contact_depth_weak": str(support_contact_depth_weak_path),
            "debug_support_contact_depth_strong": str(support_contact_depth_strong_path),
            "debug_support_contact_depth_delta": str(support_contact_depth_delta_path),
        },
    }

    save_json(
        stats_path,
        record,
    )




    print(
        f"[{index:02d}/"
        f"{total_views:02d}] "
        f"{angle:05.1f}° | "
        f"área={area_ratio:.2%} | "
        f"sombra_rechazada="
        f"{diagnostics['shadow_pixels_rejected']} px | "
        f"componentes="
        f"{diagnostics['component_selection']['component_count']} "
        f"-> "
        f"{len(diagnostics['component_selection']['selected_labels'])} | "
        f"{record['local_quality']}"
    )

    return record, overlay_path


def main() -> int:
    """Genera las siluetas del objeto y sus diagnósticos por vista y sesión."""
    args = build_parser().parse_args()

    root = Path(args.root).expanduser().resolve()

    object_name = args.object.strip().lower()
    session = args.session.strip().upper()

    source_dir = root / "reconstruccion" / session / args.depth_source

    is_partial = args.only_angle >= 0.0 or args.limit > 0

    output_name = args.output_name

    if is_partial and output_name == "03_mascara_objeto":
        suffix = (
            (
                f"A{args.only_angle:05.1f}".replace(
                    ".",
                    "p",
                )
            )
            if args.only_angle >= 0.0
            else f"limit{args.limit}"
        )
        output_name = f"{output_name}" f"_diagnostico_{suffix}"

    output_dir = root / "reconstruccion" / session / output_name

    prepare_output_directory(
        output_dir,
        clean=bool(args.clean_output),
    )

    summary_path = find_summary(
        source_dir,
        ("resumen_02_estimacion_profundidad.json",),
    )

    source_summary = load_json(summary_path)

    validate_summary_context(
        source_summary,
        object_name,
        session,
        "Paso de profundidad",
    )

    views = sorted(
        source_summary.get(
            "views",
            [],
        ),
        key=lambda item: (parse_angle(str(item["stem"]))),
    )

    if args.only_angle >= 0.0:
        views = [
            view
            for view in views
            if np.isclose(
                parse_angle(str(view["stem"])),
                args.only_angle,
                atol=0.05,
            )
        ]

    if args.limit > 0:
        views = views[: args.limit]

    background_path = source_dir / "background_rectified_left.png"

    background = cv2.imread(
        str(background_path),
        cv2.IMREAD_COLOR,
    )

    if background is None:
        raise FileNotFoundError("No se encontró fondo " f"rectificado: {background_path}")
    support_mask_path = source_dir / "background_support_mask.png"
    support_model_path = source_dir / "background_support_model.json"
    support_mask = cv2.imread(str(support_mask_path), cv2.IMREAD_GRAYSCALE)
    support_model = {}
    if support_model_path.is_file():
        try:
            support_model = load_json(support_model_path)
        except Exception as exc:
            print(f"[WARN] background_support_model.json inválido: {exc}")

    support_confidence = float(support_model.get("confidence_score", 0.0) or 0.0)
    support_status = str(support_model.get("status", "")).strip().lower()
    support_source = support_model.get("selected_source")

    # Fail-safe: una elipse dudosa es peor que no tener prior de plataforma.
    # La pertenencia del objeto sigue siendo visual y no se permite que un
    # soporte mal estimado recorte la parte inferior del objeto.
    if (
        support_mask is None
        or support_mask.shape != background.shape[:2]
        or support_status != "detected"
        or support_confidence < 0.46
    ):
        support_mask = np.zeros(background.shape[:2], dtype=np.uint8)
        print(
            "[WARN] Prior de plataforma no confiable; se desactiva para esta sesión. "
            f"status={support_status or 'desconocido'} | "
            f"confianza={support_confidence:.3f}"
        )
        support_prior_active = False
    else:
        # Sanidad adicional contra máscaras que accidentalmente ocupen gran parte
        # de la escena. Los límites son deliberadamente amplios.
        ratio = np.count_nonzero(support_mask) / max(support_mask.size, 1)
        if ratio < 0.025 or ratio > 0.30:
            print(
                "[WARN] Área de plataforma fuera del rango de sanidad; "
                f"ratio={ratio:.3f}. Se desactiva el prior."
            )
            support_mask[:] = 0
            support_prior_active = False
        else:
            support_prior_active = True
            print(
                "Prior de plataforma activo: "
                f"fuente={support_source} | confianza={support_confidence:.3f} | "
                f"área={ratio:.2%}"
            )

    # V5.0 — disparidad del fondo para verificar únicamente el contacto.
    background_disparity_path = source_dir / "background_disparity_lr.npy"
    background_disparity = None

    if background_disparity_path.is_file():
        try:
            background_disparity = np.load(str(background_disparity_path)).astype(np.float32)
            if background_disparity.shape != background.shape[:2]:
                print(
                    "[WARN] background_disparity_lr.npy tiene forma incompatible; "
                    "se desactiva verificación estéreo local."
                )
                background_disparity = None
        except Exception as exc:
            print(
                "[WARN] No se pudo cargar background_disparity_lr.npy: "
                f"{exc}. Se usará recuperación visual local."
            )
            background_disparity = None
    else:
        print(
            "[WARN] No existe background_disparity_lr.npy. "
            "El paso V5 seguirá funcionando visualmente, pero sin "
            "verificación estéreo del contacto."
        )

    records = []
    preview_paths = []

    print("\n========== " "PASO 03 V5.3: SILUETA " "ADAPTATIVA ==========")
    print(
        "Método: evidencia visual + plataforma multicue validada + "
        "falda inferior de exclusión de hardware + banda UNKNOWN profunda "
        "de contacto V3.3 + continuidad condicionada."
    )
    print("Disparidad: SOLO verificación local dentro del contacto; nunca detector global.")

    from utilidades_rendimiento import ejecutar_items
    results = ejecutar_items(
        _process_mask_view,
        {"args": args, "source_dir": source_dir, "output_dir": output_dir,
         "background": background, "support_mask": support_mask,
         "background_disparity": background_disparity, "total_views": len(views)},
        list(enumerate(views, start=1)), "Paso 03: mascaras por vista", reserve_mb=768,
    )
    records = [item[0] for item in results if item is not None]
    preview_paths = [item[1] for item in results if item is not None]

    stats = session_quality(
        records,
        args,
    )

    # V7.2 tiene 25 poses únicas y NO tiene A360.
    closure = {
        "available": False,
        "reason": (
            "La campaña V7.2 cierra físicamente la vuelta con CLOSE. " "No existe una captura A360."
        ),
    }

    for record in records:
        save_json(
            output_dir / (f"{record['stem']}" "_silhouette_stats.json"),
            record,
        )

    build_contact_sheet(
        preview_paths,
        output_dir / "contact_sheet_siluetas.png",
    )

    summary = {
        "schema_version": 5,
        "method": (
            "static_background_visual_core_v3_3_"
            "directional_support_hardware_exclusion_v1_"
            "local_support_contact_band_v5_3_"
            "local_disparity_background_verification_"
            "geodesic_contact_growth_v5"
        ),
        "object": object_name,
        "session": session,
        "source_dir": str(source_dir),
        "source_summary": str(summary_path),
        "background": str(background_path),
        "support_prior": {
            "active": bool(support_prior_active),
            "mask": str(support_mask_path),
            "model": str(support_model_path),
            "status": support_status,
            "confidence_score": float(support_confidence),
            "selected_source": support_source,
        },
        "output_dir": str(output_dir),
        "parameters": vars(args),
        "session_statistics": stats,
        "closure": closure,
        "views_processed": len(records),
        "views_accepted": sum(r["session_quality"] == "accepted" for r in records),
        "views_warning": sum(r["session_quality"] == "warning" for r in records),
        "views": records,
        "important_note": (
            "La pertenencia global sigue siendo visual. "
            "La superficie superior del soporte no se agranda; una segunda falda "
            "direccional excluye únicamente el cuerpo gris exterior/inferior del hardware. "
            "La disparidad solo puede recuperar píxeles dentro de la banda local "
            "de contacto objeto-plataforma y nunca fuera de ella. "
            "El contacto se resuelve por crecimiento geodésico desde el cuerpo "
            "ya aceptado, sin GraphCut y sin prior de forma."
        ),
    }

    save_json(
        output_dir / "resumen_03_mascara_objeto.json",
        summary,
    )

    csv_path = output_dir / "calidad_siluetas.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:
        fields = (
            "angle_deg",
            "stem",
            "area_pixels",
            "area_ratio",
            "centroid_x",
            "centroid_y",
            "shadow_pixels_rejected",
            "component_count",
            "selected_component_count",
            "local_quality",
            "session_quality",
            "reasons",
        )

        writer = csv.DictWriter(
            fh,
            fieldnames=fields,
        )
        writer.writeheader()

        for record in records:
            centroid = record["centroid_px"] or [
                None,
                None,
            ]

            selection = record["component_selection"]

            writer.writerow(
                {
                    "angle_deg": (record["angle_deg"]),
                    "stem": (record["stem"]),
                    "area_pixels": (record["area_pixels"]),
                    "area_ratio": (record["area_ratio"]),
                    "centroid_x": (centroid[0]),
                    "centroid_y": (centroid[1]),
                    "shadow_pixels_rejected": (record["shadow_pixels_rejected"]),
                    "component_count": (selection["component_count"]),
                    "selected_component_count": (len(selection["selected_labels"])),
                    "local_quality": (record["local_quality"]),
                    "session_quality": (record["session_quality"]),
                    "reasons": " | ".join(record["session_reasons"]),
                }
            )

    print("\n========== " "PASO 03 V5.3 COMPLETADO " "==========")
    print(f"Salida: {output_dir}")
    print("Revisar primero: " "contact_sheet_siluetas.png")
    print("Diagnóstico por vista: " "*_debug_shadow_rejected.png y *_debug_roi.png")
    print("==============================================")

    return 0


def _entrada_con_diagnostico():
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    from utilidades_progreso import informar_inicio

    informar_inicio(__file__, "03", "Crear máscara del objeto")
    sys.exit(main())


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
