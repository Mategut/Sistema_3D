#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Paso 05 V5.0 — consenso multisesión robusto en profundidad inversa.

Las sesiones representan la misma pose física por ``pose_index``. Solo se
permite una traslación 2D pequeña para compensar tolerancia mecánica; no se
permite escala, rotación ni deformación.

Principios del consenso:
1. La profundidad absoluta nunca se recentra, escala ni corrige entre sesiones.
   El tamaño y la posición métrica del objeto se conservan.
2. Un medoide ponderado identifica primero una única capa coherente. Solo las
   observaciones compatibles se fusionan mediante un estimador robusto en
   profundidad inversa (dominio equivalente a disparidad).
3. El desacuerdo entre sesiones no se oculta ampliando automáticamente el
   umbral. Se conserva como confianza, dispersión y soporte por píxel para que
   las etapas 06, 10 y 11 puedan ponderarlo.
4. Un píxel observado por al menos dos sesiones puede mantenerse con confianza
   baja. Esto evita convertir incertidumbre en huecos sin declarar el riesgo.
5. La alineación residual continúa limitada a traslación 2D entera, sin escala,
   rotación ni deformación.
6. El veto de plataforma por profundidad continúa DESACTIVADO por defecto.

No se usa geometría específica de cubo, no se fuerzan superficies planas y no
se rellenan huecos por interpolación espacial.
"""

from __future__ import annotations
from utilidades_progreso import operacion
from utilidades_rendimiento import ejecutar_items, contexto
from utilidades_rendimiento import configurar as _configurar_recursos

# Los runtimes numéricos leen estos límites al importarse.
_configurar_recursos(__file__)


import argparse
import csv
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utilidades_mascaras import build_contact_sheet
from utilidades_multisesion import (
    finite_stats,
    load_manifest,
    pairwise_iou,
)

DEPTH_SOURCE = "02_estimacion_profundidad"
SILHOUETTE_SOURCE = "03_mascara_objeto"
REGIONAL_SOURCE = "04_validacion_disparidad"


def build_parser():
    """Construye las opciones de línea de comandos de este paso."""
    p = argparse.ArgumentParser(
        description="Alinea residualmente S01/S02/S03 y genera consenso por pose."
    )
    p.add_argument("--root", required=True)
    p.add_argument("--object", default="cubo")
    p.add_argument("--manifest", default="")
    p.add_argument("--depth-source", default=DEPTH_SOURCE)
    p.add_argument("--silhouette-source", default=SILHOUETTE_SOURCE)
    p.add_argument("--regional-source", default=REGIONAL_SOURCE)
    p.add_argument("--output-name", default="05_consenso_multisesion")
    p.add_argument("--depth-agreement-mm", type=float, default=6.0)
    p.add_argument("--minimum-independent-depth-support", type=int, default=2)
    p.add_argument("--minimum-valid-ratio", type=float, default=0.18)

    # V4 — el umbral mide fiabilidad; no desplaza ni invalida automáticamente Z.
    p.add_argument(
        "--adaptive-depth-agreement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Deriva una escala diagnóstica de repetibilidad. No altera la "
            "profundidad ni decide por sí sola si un píxel se exporta."
        ),
    )
    p.add_argument("--depth-agreement-max-mm", type=float, default=12.0)
    p.add_argument("--depth-agreement-mad-factor", type=float, default=3.5)
    p.add_argument(
        "--depth-bias-correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opción histórica conservada por compatibilidad. En V5 los sesgos "
            "solo se diagnostican y nunca se aplican al consenso."
        ),
    )
    p.add_argument("--depth-bias-max-mm", type=float, default=8.0)
    p.add_argument("--depth-bias-minimum-pixels", type=int, default=1500)
    p.add_argument(
        "--stable-interior-margin-px",
        type=float,
        default=6.0,
        help=(
            "Margen mínimo respecto al borde de la silueta para estimar " "repetibilidad y sesgo Z."
        ),
    )
    p.add_argument(
        "--triplet-interior-rescue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opción histórica sin efecto geométrico en V5. Los píxeles con "
            "desacuerdo se conservan con confianza explícitamente reducida."
        ),
    )
    p.add_argument(
        "--triplet-rescue-max-span-mm",
        type=float,
        default=18.0,
        help="Span corregido máximo para el rescate triplete interior.",
    )
    p.add_argument(
        "--triplet-rescue-warning-ratio",
        type=float,
        default=0.08,
        help=(
            "Si más de esta fracción de la profundidad válida procede del "
            "rescate triplete, la vista queda como warning."
        ),
    )
    p.add_argument(
        "--low-agreement-warning-ratio",
        type=float,
        default=0.08,
        help=(
            "Fracción máxima de píxeles válidos sin dos observaciones dentro "
            "del umbral antes de marcar la pose como warning."
        ),
    )
    p.add_argument(
        "--regional-residual-reference-px",
        type=float,
        default=3.0,
        help="Escala de residual regional usada solo para calcular confianza.",
    )

    # V3 — veto de plataforma basado en fondo vacío. Es deliberadamente
    # conservador: profundidad solo puede QUITAR soporte confirmado, nunca
    # añadir objeto.
    p.add_argument(
        "--support-depth-veto",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnóstico opcional. Por defecto NO borra píxeles en la zona de "
            "contacto objeto-plataforma."
        ),
    )
    p.add_argument("--support-veto-minimum-sessions", type=int, default=2)
    p.add_argument("--support-noise-minimum-mm", type=float, default=4.0)
    p.add_argument("--support-separation-minimum-mm", type=float, default=8.0)
    p.add_argument("--support-separation-maximum-mm", type=float, default=24.0)
    p.add_argument("--support-separation-mad-factor", type=float, default=4.0)

    # Solo traslación residual. Nada de rotación/escala.
    p.add_argument("--alignment-max-x-px", type=int, default=35)
    p.add_argument("--alignment-max-y-px", type=int, default=15)
    p.add_argument("--alignment-downsample", type=int, default=4)
    p.add_argument("--alignment-local-radius", type=int, default=3)
    p.add_argument("--alignment-fullres-refine-radius", type=int, default=4)
    p.add_argument("--minimum-aligned-silhouette-iou", type=float, default=0.72)
    return p


def load_json(path: Path) -> dict:
    """Lee un documento JSON codificado en UTF-8."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def quality_rank(q: str) -> int:
    """Asigna una prioridad numérica a la calidad declarada de cada observación."""
    return {"accepted": 4, "warning": 3, "accepted_local_only": 2, "rejected": 0}.get(str(q), 1)


def load_regional_records(root, obj, sessions, folder):
    """Carga por sesión los registros de validación del paso 04."""
    result = {}
    for session in sessions:
        path = root / "reconstruccion" / session / folder / "resumen_04_validacion_disparidad.json"
        if not path.is_file():
            raise FileNotFoundError(f"Falta 04 para {session}: {path}")
        summary = load_json(path)
        for record in summary.get("views", []):
            result[(session, str(record["stem"]))] = record
    return result


def load_observation(root, obj, manifest_record, regional, depth_source, sil_source, reg_source):
    """Reúne profundidad, silueta y diagnósticos correspondientes a una captura."""
    session = manifest_record["session"]
    stem = manifest_record["stem"]
    base = root / "reconstruccion" / session

    silhouette = cv2.imread(
        str(base / sil_source / f"{stem}_silhouette_mask.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    cloud_mask = cv2.imread(
        str(base / reg_source / f"{stem}_cloud_mask.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    image = cv2.imread(
        str(base / depth_source / f"{stem}_rect_L.png"),
        cv2.IMREAD_COLOR,
    )
    depth_path = base / reg_source / f"{stem}_depth_regularized_mm.npy"
    conf_path = base / depth_source / f"{stem}_confidence.npy"
    raw_depth_path = base / depth_source / f"{stem}_depth_raw_mm.npy"
    background_depth_path = base / depth_source / "background_depth_mm.npy"
    support_mask_path = base / depth_source / "background_support_mask.png"
    regional_residual_path = base / reg_source / f"{stem}_regional_residual.npy"

    if silhouette is None or cloud_mask is None or image is None or not depth_path.is_file():
        return None

    depth = np.load(depth_path).astype(np.float32)
    confidence = (
        np.load(conf_path).astype(np.float32)
        if conf_path.is_file()
        else np.full(depth.shape, np.nan, np.float32)
    )
    raw_depth = (
        np.load(raw_depth_path).astype(np.float32)
        if raw_depth_path.is_file()
        else np.full(depth.shape, np.nan, np.float32)
    )
    background_depth = (
        np.load(background_depth_path).astype(np.float32)
        if background_depth_path.is_file()
        else np.full(depth.shape, np.nan, np.float32)
    )
    support_mask = cv2.imread(str(support_mask_path), cv2.IMREAD_GRAYSCALE)
    if support_mask is None:
        support_mask = np.zeros(depth.shape, np.uint8)
    regional_residual = (
        np.load(regional_residual_path).astype(np.float32)
        if regional_residual_path.is_file()
        else np.full(depth.shape, np.nan, np.float32)
    )

    shapes = {
        silhouette.shape,
        cloud_mask.shape,
        depth.shape,
        confidence.shape,
        raw_depth.shape,
        background_depth.shape,
        support_mask.shape,
        regional_residual.shape,
        image.shape[:2],
    }
    if len(shapes) != 1:
        raise ValueError(f"Dimensiones incompatibles {session}/{stem}: {shapes}")

    return {
        "session": session,
        "stem": stem,
        "manifest": manifest_record,
        "regional": regional,
        "silhouette": silhouette,
        "cloud_mask": cloud_mask,
        "image": image,
        "depth": depth,
        "confidence": confidence,
        "raw_depth": raw_depth,
        "background_depth": background_depth,
        "support_mask": support_mask,
        "regional_residual": regional_residual,
    }


def centroid(mask: np.ndarray):
    """Devuelve el centroide de la máscara o None cuando no tiene área."""
    m = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    if abs(m["m00"]) < 1e-9:
        return None
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], dtype=np.float64)


def shift_array(array: np.ndarray, dx: int, dy: int, fill):
    """Traslación entera sin interpolar profundidad."""
    h, w = array.shape[:2]
    out = np.empty_like(array)
    out[...] = fill

    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)

    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)

    if src_x1 > src_x0 and src_y1 > src_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return out


def mask_iou(a, b):
    """Calcula la intersección sobre unión de dos máscaras binarias."""
    aa = a > 0
    bb = b > 0
    union = int(np.count_nonzero(aa | bb))
    return None if union == 0 else float(np.count_nonzero(aa & bb) / union)


def estimate_translation(ref_mask, moving_mask, args):
    """Estima una traslación residual limitada a partir del solape de siluetas."""
    raw = mask_iou(ref_mask, moving_mask)
    c_ref = centroid(ref_mask)
    c_mov = centroid(moving_mask)

    if c_ref is None or c_mov is None:
        return {"dx": 0, "dy": 0, "raw_iou": raw, "aligned_iou": raw, "valid": False}

    factor = max(1, int(args.alignment_downsample))
    sw = max(1, ref_mask.shape[1] // factor)
    sh = max(1, ref_mask.shape[0] // factor)

    ref_small = cv2.resize(ref_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
    mov_small = cv2.resize(moving_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)

    dx0 = int(round((c_ref[0] - c_mov[0]) / factor))
    dy0 = int(round((c_ref[1] - c_mov[1]) / factor))
    max_x = max(1, int(math.ceil(args.alignment_max_x_px / factor)))
    max_y = max(1, int(math.ceil(args.alignment_max_y_px / factor)))

    dx0 = int(np.clip(dx0, -max_x, max_x))
    dy0 = int(np.clip(dy0, -max_y, max_y))

    best = (-1.0, dx0, dy0)
    r = max(0, int(args.alignment_local_radius))

    for dy in range(max(-max_y, dy0 - r), min(max_y, dy0 + r) + 1):
        for dx in range(max(-max_x, dx0 - r), min(max_x, dx0 + r) + 1):
            shifted = shift_array(mov_small, dx, dy, 0)
            score = mask_iou(ref_small, shifted)
            score = -1.0 if score is None else score
            if score > best[0]:
                best = (score, dx, dy)

    coarse_dx = int(best[1] * factor)
    coarse_dy = int(best[2] * factor)

    if abs(coarse_dx) > args.alignment_max_x_px or abs(coarse_dy) > args.alignment_max_y_px:
        return {"dx": 0, "dy": 0, "raw_iou": raw, "aligned_iou": raw, "valid": False}

    # Refinamiento entero a resolución completa. La búsqueda coarse trabaja
    # en múltiplos de ``factor``; sin este paso podía quedar un residual de
    # varios píxeles que perjudicara la comparación Z en bordes y esquinas.
    rr = max(0, int(args.alignment_fullres_refine_radius))
    best_full = (-1.0, coarse_dx, coarse_dy)
    for fy in range(
        max(-int(args.alignment_max_y_px), coarse_dy - rr),
        min(int(args.alignment_max_y_px), coarse_dy + rr) + 1,
    ):
        for fx in range(
            max(-int(args.alignment_max_x_px), coarse_dx - rr),
            min(int(args.alignment_max_x_px), coarse_dx + rr) + 1,
        ):
            shifted = shift_array(moving_mask, fx, fy, 0)
            score = mask_iou(ref_mask, shifted)
            score = -1.0 if score is None else score
            if score > best_full[0]:
                best_full = (score, fx, fy)

    aligned = None if best_full[0] < 0 else float(best_full[0])
    dx = int(best_full[1])
    dy = int(best_full[2])

    return {
        "dx": dx,
        "dy": dy,
        "raw_iou": raw,
        "aligned_iou": aligned,
        "valid": bool(aligned is not None and aligned >= args.minimum_aligned_silhouette_iou),
    }


def align_observation(obs, alignment):
    """Aplica a los mapas de una observación la traslación residual aceptada."""
    if alignment["dx"] == 0 and alignment["dy"] == 0:
        result = dict(obs)
    else:
        dx, dy = alignment["dx"], alignment["dy"]
        result = dict(obs)
        result["silhouette"] = shift_array(obs["silhouette"], dx, dy, 0)
        result["cloud_mask"] = shift_array(obs["cloud_mask"], dx, dy, 0)
        result["image"] = shift_array(obs["image"], dx, dy, 0)
        result["depth"] = shift_array(obs["depth"], dx, dy, np.nan)
        result["confidence"] = shift_array(obs["confidence"], dx, dy, np.nan)
        result["raw_depth"] = shift_array(obs["raw_depth"], dx, dy, np.nan)
        result["background_depth"] = shift_array(obs["background_depth"], dx, dy, np.nan)
        result["support_mask"] = shift_array(obs["support_mask"], dx, dy, 0)
        result["regional_residual"] = shift_array(obs["regional_residual"], dx, dy, np.nan)
    result["alignment"] = alignment
    return result


def _robust_center_sigma(values: np.ndarray):
    """Estima centro y dispersión robustos usando solo valores finitos."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None, 0
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    sigma = max(1e-6, 1.4826 * mad)
    return center, sigma, int(values.size)


def _majority_support_domain(observations):
    """Selecciona el soporte presente en al menos la mitad de las observaciones."""
    stack = np.stack([o["support_mask"] > 0 for o in observations])
    required = max(1, int(math.ceil(len(observations) * 0.5)))
    return np.sum(stack, axis=0) >= required


def build_stable_interior_domain(silhouette: np.ndarray, support_domain: np.ndarray, args):
    """Dominio interior usado solo para estimar repetibilidad entre sesiones."""
    binary = (silhouette > 0).astype(np.uint8)
    if not np.any(binary):
        return np.zeros(binary.shape, bool)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    margin = max(0.0, float(args.stable_interior_margin_px))
    stable = (dist >= margin) if margin > 0 else (binary > 0)
    # Nunca estimar estadística de repetibilidad sobre el soporte físico.
    stable &= ~support_domain
    return stable


def estimate_depth_biases(observations, stable_domain, args):
    """Mide desplazamientos Z entre repeticiones sin aplicarlos.

    El nombre se conserva para que lectores históricos del resumen continúen
    funcionando. Desde V4 el vector devuelto siempre es cero: una diferencia
    de profundidad puede representar posición o geometría real y no debe
    normalizarse. Las medianas se guardan únicamente como diagnóstico.
    """
    biases = [0.0] * len(observations)
    diagnostics = []
    if len(observations) < 2:
        return biases, diagnostics

    ref = observations[0]
    ref_valid = (ref["cloud_mask"] > 0) & np.isfinite(ref["depth"]) & stable_domain

    for index, obs in enumerate(observations):
        if index == 0:
            diagnostics.append(
                {
                    "session": obs["session"],
                    "bias_mm": 0.0,
                    "samples": int(np.count_nonzero(ref_valid)),
                    "applied": False,
                    "reason": "reference",
                }
            )
            continue
        valid = ref_valid & (obs["cloud_mask"] > 0) & np.isfinite(obs["depth"])
        # Evitar bordes/oclusiones extremadamente discordantes en la
        # estimación del sesgo escalar.
        diff = obs["depth"][valid].astype(np.float64) - ref["depth"][valid].astype(np.float64)
        diff = diff[np.isfinite(diff)]
        diff = diff[np.abs(diff) <= 35.0]
        center, sigma, count = _robust_center_sigma(diff)
        measurable = bool(center is not None and count >= int(args.depth_bias_minimum_pixels))
        diagnostics.append(
            {
                "session": obs["session"],
                "bias_mm": 0.0,
                "raw_median_mm": center,
                "robust_sigma_mm": sigma,
                "samples": count,
                "measurable": measurable,
                "applied": False,
                "used_for_fusion": False,
                "reason": (
                    "diagnostic_only_absolute_depth_preserved"
                    if measurable
                    else "insufficient_samples"
                ),
            }
        )
    return biases, diagnostics


def derive_depth_agreement(observations, biases, stable_domain, args):
    """Determina el umbral de acuerdo de profundidad según la evidencia disponible."""
    base = float(args.depth_agreement_mm)
    if not bool(args.adaptive_depth_agreement) or len(observations) < 2:
        return base, {"adaptive": False, "agreement_mm": base}

    samples = []
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            oi, oj = observations[i], observations[j]
            valid = (
                (oi["cloud_mask"] > 0)
                & (oj["cloud_mask"] > 0)
                & np.isfinite(oi["depth"])
                & np.isfinite(oj["depth"])
                & stable_domain
            )
            if np.count_nonzero(valid) < 500:
                continue
            # V4 compara las mediciones métricas originales. ``biases`` se
            # conserva en la firma únicamente por compatibilidad.
            di = oi["depth"][valid].astype(np.float64)
            dj = oj["depth"][valid].astype(np.float64)
            delta = np.abs(di - dj)
            delta = delta[np.isfinite(delta)]
            # La cola muy grande representa desacuerdo geométrico real y no
            # debe inflar la tolerancia de repetibilidad.
            delta = delta[delta <= 30.0]
            if delta.size:
                samples.append(delta)

    if not samples:
        return base, {"adaptive": True, "agreement_mm": base, "reason": "no_samples"}

    values = np.concatenate(samples)
    center, sigma, count = _robust_center_sigma(values)
    if center is None:
        return base, {"adaptive": True, "agreement_mm": base, "reason": "no_finite_samples"}
    candidate = center + float(args.depth_agreement_mad_factor) * sigma
    agreement = float(
        np.clip(
            max(base, candidate),
            base,
            max(base, float(args.depth_agreement_max_mm)),
        )
    )
    return agreement, {
        "adaptive": True,
        "agreement_mm": agreement,
        "sample_median_mm": center,
        "robust_sigma_mm": sigma,
        "samples": count,
        "maximum_mm": float(args.depth_agreement_max_mm),
    }


def support_depth_veto(observations, args):
    """Detecta soporte visible confirmado sin crear foreground.

    Para cada repetición se estima el ruido de ``background_depth-current`` en
    la parte visible del soporte (fuera de la silueta). Un píxel del soporte
    solo se veta cuando al menos N sesiones lo observan como compatible con
    el fondo y ninguna aporta separación positiva clara de objeto.
    """
    shape = observations[0]["depth"].shape
    if not bool(args.support_depth_veto):
        return (
            np.zeros(shape, bool),
            np.zeros(shape, np.uint8),
            {
                "enabled": False,
                "reason": "disabled",
            },
        )

    support_stack = np.stack([o["support_mask"] > 0 for o in observations])
    support_required = max(1, int(math.ceil(len(observations) * 0.5)))
    support_domain = np.sum(support_stack, axis=0) >= support_required

    platform_votes = np.zeros(shape, np.uint8)
    object_votes = np.zeros(shape, np.uint8)
    comparable_votes = np.zeros(shape, np.uint8)
    per_session = []

    for obs in observations:
        support = obs["support_mask"] > 0
        current = obs["raw_depth"]
        background = obs["background_depth"]
        comparable = support & np.isfinite(current) & np.isfinite(background)
        delta = background - current

        visible = comparable & (obs["silhouette"] == 0)
        values = delta[visible]
        center, sigma, count = _robust_center_sigma(values)
        if center is None or count < 1000:
            per_session.append(
                {
                    "session": obs["session"],
                    "usable": False,
                    "samples": count,
                }
            )
            continue

        noise = max(float(args.support_noise_minimum_mm), 2.5 * sigma)
        separation = float(
            np.clip(
                max(
                    float(args.support_separation_minimum_mm),
                    float(args.support_separation_mad_factor) * sigma,
                ),
                float(args.support_separation_minimum_mm),
                float(args.support_separation_maximum_mm),
            )
        )
        centered = delta - center
        platform_like = comparable & (np.abs(centered) <= noise)
        object_like = comparable & (centered >= separation)

        comparable_votes += comparable.astype(np.uint8)
        platform_votes += platform_like.astype(np.uint8)
        object_votes += object_like.astype(np.uint8)
        per_session.append(
            {
                "session": obs["session"],
                "usable": True,
                "samples": count,
                "background_delta_center_mm": center,
                "background_delta_sigma_mm": sigma,
                "platform_noise_band_mm": noise,
                "object_separation_mm": separation,
            }
        )

    minimum = max(2, int(args.support_veto_minimum_sessions))
    veto = (
        support_domain
        & (comparable_votes >= minimum)
        & (platform_votes >= minimum)
        & (object_votes == 0)
    )
    return (
        veto,
        object_votes,
        {
            "enabled": True,
            "minimum_sessions": minimum,
            "veto_pixels": int(np.count_nonzero(veto)),
            "object_evidence_pixels": int(np.count_nonzero(object_votes > 0)),
            "support_domain_pixels": int(np.count_nonzero(support_domain)),
            "sessions": per_session,
        },
    )


def consensus_depth(
    observations,
    agreement_mm,
    minimum_support,
    biases=None,
    stable_interior=None,
    triplet_rescue=False,
    triplet_rescue_max_span_mm=18.0,
    regional_residual_reference_px=3.0,
):
    """Consenso V5 robusto, métrico y sin correcciones globales de Z.

    El medoide identifica una sola capa por píxel. Las muestras compatibles con
    esa capa se combinan en 1/Z con pesos de confianza, residual y Huber. Nunca
    se mezclan capas incompatibles ni se aplica un desplazamiento global Z.

    ``agreement_mm`` se usa como escala de confianza. Un desacuerdo no borra
    automáticamente el píxel si existen al menos ``minimum_support`` medidas:
    se conserva con confianza baja para que la nube y la fusión decidan usando
    evidencia espacial y multivista.
    """
    depths = np.stack([o["depth"] for o in observations]).astype(np.float32)
    masks = np.stack([o["cloud_mask"] > 0 for o in observations])
    valid = masks & np.isfinite(depths)
    n = len(observations)

    source_confidence = np.stack(
        [np.asarray(o["confidence"], dtype=np.float32) for o in observations]
    )
    source_confidence = np.where(
        np.isfinite(source_confidence),
        np.clip(source_confidence, 0.0, 1.0),
        0.50,
    ).astype(np.float32)
    residuals = np.stack(
        [np.asarray(o["regional_residual"], dtype=np.float32) for o in observations]
    )

    h, w = depths.shape[1:]
    consensus = np.full((h, w), np.nan, np.float32)
    spread = np.full((h, w), np.nan, np.float32)
    support = np.zeros((h, w), np.uint8)
    agreement_support = np.zeros((h, w), np.uint8)
    confidence = np.full((h, w), np.nan, np.float32)
    selection_mode = np.zeros((h, w), np.uint8)
    low_agreement_map = np.zeros((h, w), np.uint8)

    if stable_interior is None:
        stable_interior = np.ones((h, w), bool)

    if n < 2:
        return (
            consensus,
            support,
            spread,
            np.zeros((h, w), np.uint8),
            low_agreement_map,
            agreement_support,
            confidence,
            selection_mode,
        )

    valid_count = np.sum(valid, axis=0)

    # Coste L1 ponderado de escoger cada observación como representante. El
    # término pequeño de confianza solo rompe empates (caso de dos sesiones).
    costs = np.full((n, h, w), np.inf, np.float32)
    for i in range(n):
        candidate_valid = valid[i]
        numerator = np.zeros((h, w), np.float32)
        denominator = np.zeros((h, w), np.float32)
        for j in range(n):
            comparable = candidate_valid & valid[j]
            weight = 0.25 + 0.75 * source_confidence[j]
            delta = np.abs(depths[i] - depths[j])
            numerator[comparable] += (weight[comparable] * delta[comparable]).astype(np.float32)
            denominator[comparable] += weight[comparable].astype(np.float32)
        usable = candidate_valid & (denominator > 0)
        costs[i, usable] = numerator[usable] / denominator[usable]
        costs[i, usable] += (
            0.02 * max(float(agreement_mm), 1e-6) * (1.0 - source_confidence[i, usable])
        ).astype(np.float32)

    selected_index = np.argmin(costs, axis=0).astype(np.int16)
    selected_depth = np.take_along_axis(depths, selected_index[None, :, :], axis=0)[0]
    selected_confidence = np.take_along_axis(source_confidence, selected_index[None, :, :], axis=0)[
        0
    ]
    selected_residual = np.take_along_axis(residuals, selected_index[None, :, :], axis=0)[0]

    distance_to_selected = np.abs(depths - selected_depth[None, :, :])
    agrees = valid & (distance_to_selected <= max(float(agreement_mm), 1e-6))
    agreement_count = np.sum(agrees, axis=0)

    # Distancia al respaldo independiente más cercano. A diferencia del span,
    # no castiga excesivamente una observación aislada cuando otras dos forman
    # una capa coherente.
    nearest_other = np.full((h, w), np.inf, np.float32)
    for i in range(n):
        chosen = selected_index == i
        if not np.any(chosen):
            continue
        for j in range(n):
            if i == j:
                continue
            comparable = chosen & valid[j]
            delta = np.abs(depths[i] - depths[j])
            better = comparable & (delta < nearest_other)
            nearest_other[better] = delta[better]

    # La dispersión exportada describe la capa respaldada alrededor del
    # medoide, no el rango total de todas las observaciones. Así, una tercera
    # sesión claramente discordante no invalida dos medidas coherentes. Si no
    # existe respaldo dentro del umbral, la distancia a la observación
    # independiente más cercana deja explícita la incertidumbre.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        agreeing_min = np.nanmin(np.where(agrees, depths, np.nan), axis=0)
        agreeing_max = np.nanmax(np.where(agrees, depths, np.nan), axis=0)
    agreeing_span = agreeing_max - agreeing_min
    robust_spread = np.where(
        agreement_count >= 2,
        agreeing_span,
        nearest_other,
    ).astype(np.float32)

    enough = valid_count >= int(minimum_support)
    # Reducir ruido dentro de la capa coherente antes de crear puntos 3D. En
    # estéreo la incertidumbre es más regular en disparidad (1/Z) que en Z.
    residual_reference = max(float(regional_residual_reference_px), 1e-6)
    residual_weight = np.where(
        np.isfinite(residuals),
        np.exp(-0.5 * np.square(residuals / residual_reference)),
        0.75,
    ).astype(np.float32)
    base_weight = ((0.15 + 0.85 * source_confidence) * (0.60 + 0.40 * residual_weight)).astype(
        np.float32
    )
    inverse_depth = np.zeros_like(depths, dtype=np.float32)
    np.divide(1.0, depths, out=inverse_depth, where=valid & (depths > 1e-6))
    fused_inverse = np.zeros((h, w), np.float32)
    np.divide(
        1.0,
        selected_depth,
        out=fused_inverse,
        where=np.isfinite(selected_depth) & (selected_depth > 1e-6),
    )
    huber_delta_mm = max(0.45 * float(agreement_mm), 0.75)
    for _ in range(3):
        current_depth = np.full((h, w), np.nan, np.float32)
        np.divide(
            1.0,
            fused_inverse,
            out=current_depth,
            where=fused_inverse > 1e-12,
        )
        error_mm = np.abs(depths - current_depth[None, :, :])
        huber_weight = np.minimum(
            1.0,
            huber_delta_mm / np.maximum(error_mm, 1e-6),
        ).astype(np.float32)
        weights = np.where(agrees, base_weight * huber_weight, 0.0)
        weight_sum = np.sum(weights, axis=0)
        numerator = np.sum(weights * inverse_depth, axis=0)
        update = weight_sum > 1e-8
        fused_inverse[update] = numerator[update] / weight_sum[update]

    fused_depth = selected_depth.astype(np.float32).copy()
    valid_inverse = fused_inverse > 1e-12
    fused_depth[valid_inverse] = 1.0 / fused_inverse[valid_inverse]
    consensus[enough] = selected_depth[enough].astype(np.float32)
    use_fused = enough & (agreement_count >= int(minimum_support)) & np.isfinite(fused_depth)
    consensus[use_fused] = fused_depth[use_fused].astype(np.float32)
    support[enough] = np.clip(valid_count[enough], 0, 255).astype(np.uint8)
    agreement_support[enough] = np.clip(agreement_count[enough], 0, 255).astype(np.uint8)
    spread[enough] = robust_spread[enough]

    scale = max(float(agreement_mm), 1e-6)
    nearest_score = np.exp(-0.5 * np.square(np.minimum(nearest_other, 4.0 * scale) / scale))
    span_score = np.exp(-0.5 * np.square(np.minimum(robust_spread, 8.0 * scale) / (2.0 * scale)))
    support_score = np.clip(
        (valid_count.astype(np.float32) - 1.0) / max(float(n - 1), 1.0),
        0.0,
        1.0,
    )
    residual_score = np.where(
        np.isfinite(selected_residual),
        np.exp(-0.5 * np.square(selected_residual / residual_reference)),
        0.75,
    )
    combined = (
        0.42 * selected_confidence + 0.28 * nearest_score + 0.15 * span_score + 0.15 * support_score
    ) * (0.65 + 0.35 * residual_score)
    confidence[enough] = np.clip(combined[enough], 0.0, 1.0).astype(np.float32)

    strong = enough & (agreement_count >= int(minimum_support))
    all_agree = strong & (agreement_count >= min(3, n))
    pair_agree = strong & ~all_agree
    low_agreement = enough & ~strong
    selection_mode[low_agreement] = 1
    selection_mode[pair_agree] = 2
    selection_mode[all_agree] = 3
    low_agreement_map[low_agreement] = 255

    consensus[~enough] = np.nan
    spread[~enough] = np.nan
    confidence[~enough] = np.nan

    return (
        consensus,
        support,
        spread,
        enough.astype(np.uint8) * 255,
        low_agreement_map,
        agreement_support,
        confidence,
        selection_mode,
    )


def finite_vis(array, valid, cmap):
    gray = np.zeros(array.shape, np.uint8)
    vals = array[valid & np.isfinite(array)]
    if vals.size:
        lo, hi = np.percentile(vals, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        norm = (np.clip(array, lo, hi) - lo) / (hi - lo)
        norm = np.nan_to_num(norm, nan=0.0)
        gray[valid] = np.clip(norm[valid] * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cmap)


def safe_angle_token(angle):
    """Codifica un ángulo para usarlo en nombres de archivo sin punto decimal."""
    return f"{angle:07.3f}".replace(".", "p")


def _procesar_unidad_independiente(task):
    """Fusiona y guarda una pose utilizando el contexto privado del trabajador."""
    ctx = contexto()
    args = ctx["args"]
    by_pose = ctx["by_pose"]
    obj = ctx["obj"]
    output = ctx["output"]
    regional_map = ctx["regional_map"]
    root = ctx["root"]
    sessions = ctx["sessions"]
    print(
        "[PROGRESO] Alinear sesiones y calcular consenso de cada pose: iniciando unidad", flush=True
    )
    records_out = []
    csv_rows = []
    previews = []
    for pose in [task]:
        members = by_pose[pose]
        loaded, ignored = ([], [])
        for member in members:
            regional = regional_map.get((member["session"], member["stem"]))
            if regional is None:
                ignored.append({"session": member["session"], "reason": "sin 04"})
                continue
            if str(regional.get("quality")) == "rejected":
                ignored.append({"session": member["session"], "reason": "04 rejected"})
                continue
            obs = load_observation(
                root,
                obj,
                member,
                regional,
                args.depth_source,
                args.silhouette_source,
                args.regional_source,
            )
            if obs is None:
                ignored.append({"session": member["session"], "reason": "archivos incompletos"})
                continue
            loaded.append(obs)
        if len(loaded) < args.minimum_independent_depth_support:
            records_out.append(
                {
                    "pose_index": pose,
                    "physical_angle_deg": float(members[0]["physical_angle_deg"]),
                    "quality": "rejected",
                    "reasons": ["Soporte multisesión insuficiente antes de alineación."],
                    "ignored": ignored,
                }
            )
            print(f"[P{pose:02d}] rejected | observaciones={len(loaded)}")
            continue
        loaded.sort(
            key=lambda o: (
                1 if o["session"] == sessions[0] else 0,
                quality_rank(o["regional"].get("quality")),
                float(o["regional"].get("valid_ratio_of_silhouette", 0.0) or 0.0),
            ),
            reverse=True,
        )
        reference = loaded[0]
        aligned = [
            align_observation(
                reference, {"dx": 0, "dy": 0, "raw_iou": 1.0, "aligned_iou": 1.0, "valid": True}
            )
        ]
        alignment_info = [
            {
                "session": reference["session"],
                "stem": reference["stem"],
                "reference": True,
                **aligned[0]["alignment"],
            }
        ]
        for obs in loaded[1:]:
            estimate = estimate_translation(reference["silhouette"], obs["silhouette"], args)
            alignment_info.append(
                {"session": obs["session"], "stem": obs["stem"], "reference": False, **estimate}
            )
            if not estimate["valid"]:
                ignored.append(
                    {
                        "session": obs["session"],
                        "stem": obs["stem"],
                        "reason": f"alineación residual no confiable (IoU={estimate['aligned_iou']})",
                        "alignment": estimate,
                    }
                )
                continue
            aligned.append(align_observation(obs, estimate))
        if len(aligned) < args.minimum_independent_depth_support:
            records_out.append(
                {
                    "pose_index": pose,
                    "physical_angle_deg": float(members[0]["physical_angle_deg"]),
                    "quality": "rejected",
                    "reasons": ["Soporte insuficiente después de alineación residual."],
                    "alignment": alignment_info,
                    "ignored": ignored,
                }
            )
            print(f"[P{pose:02d}] rejected tras alineación | obs={len(aligned)}")
            continue
        masks = [o["silhouette"] for o in aligned]
        stack = np.stack([m > 0 for m in masks])
        required = max(2, int(math.ceil(len(aligned) * 0.5)))
        silhouette_support = np.sum(stack, axis=0).astype(np.uint8)
        silhouette = (silhouette_support >= required).astype(np.uint8) * 255
        support_domain = _majority_support_domain(aligned)
        stable_interior = build_stable_interior_domain(silhouette, support_domain, args)
        depth_biases, depth_bias_diag = estimate_depth_biases(aligned, stable_interior, args)
        adaptive_agreement_mm, agreement_diag = derive_depth_agreement(
            aligned, depth_biases, stable_interior, args
        )
        (
            depth,
            depth_support,
            depth_spread,
            depth_valid,
            low_agreement_map,
            depth_agreement_support,
            confidence,
            selection_mode,
        ) = consensus_depth(
            aligned,
            adaptive_agreement_mm,
            args.minimum_independent_depth_support,
            biases=depth_biases,
            stable_interior=stable_interior,
            triplet_rescue=bool(args.triplet_interior_rescue),
            triplet_rescue_max_span_mm=float(args.triplet_rescue_max_span_mm),
            regional_residual_reference_px=float(args.regional_residual_reference_px),
        )
        cloud_mask = cv2.bitwise_and(silhouette, depth_valid)
        support_veto, support_object_votes, support_veto_diag = support_depth_veto(aligned, args)
        support_veto &= cloud_mask > 0
        cloud_mask[support_veto] = 0
        depth[cloud_mask == 0] = np.nan
        image_stack = np.stack([o["image"] for o in aligned])
        image = np.median(image_stack, axis=0).astype(np.uint8)
        confidence[cloud_mask == 0] = np.nan
        silhouette_pixels = int(np.count_nonzero(silhouette))
        valid_pixels = int(np.count_nonzero(cloud_mask))
        valid_ratio = valid_pixels / max(silhouette_pixels, 1)
        low_agreement_pixels = int(np.count_nonzero((low_agreement_map > 0) & (cloud_mask > 0)))
        low_agreement_ratio = low_agreement_pixels / max(valid_pixels, 1)
        physical = float(members[0]["physical_angle_deg"])
        nominal = float(members[0]["nominal_angle_deg"])
        steps = int(members[0]["cumulative_steps"])
        reasons = []
        quality = "accepted"
        if valid_pixels == 0:
            quality = "rejected"
            reasons.append("Consenso sin profundidad válida.")
        elif valid_ratio < args.minimum_valid_ratio:
            quality = "warning"
            reasons.append(f"Cobertura baja: {valid_ratio:.2%}.")
        if quality != "rejected" and low_agreement_ratio > float(args.low_agreement_warning_ratio):
            quality = "warning"
            reasons.append(
                f"Desacuerdo multisesión apreciable: {low_agreement_ratio:.2%} de la profundidad válida sin dos observaciones dentro del umbral."
            )
        agreement_warning_level = min(
            float(args.depth_agreement_max_mm), max(float(args.depth_agreement_mm) + 4.0, 10.0)
        )
        if (
            quality != "rejected"
            and float(adaptive_agreement_mm) >= agreement_warning_level
            and (float(adaptive_agreement_mm) > float(args.depth_agreement_mm) + 1e-06)
        ):
            quality = "warning"
            reasons.append(
                f"Repetibilidad de profundidad baja: tolerancia adaptativa={adaptive_agreement_mm:.2f} mm."
            )
        measured_offsets = [
            abs(float(item.get("raw_median_mm", 0.0) or 0.0))
            for item in depth_bias_diag
            if bool(item.get("measurable"))
        ]
        if (
            quality != "rejected"
            and measured_offsets
            and (max(measured_offsets) >= float(args.depth_bias_max_mm))
        ):
            quality = "warning"
            reasons.append(
                f"Desplazamiento métrico intersesión apreciable, conservado sin corrección: máximo={max(measured_offsets):.2f} mm."
            )
        stem = f"{obj.upper()}_CONS_P{pose:02d}_A{safe_angle_token(physical)}"
        paths = {
            "image": output / f"{stem}_rect_L_consensus.png",
            "silhouette": output / f"{stem}_silhouette_consensus.png",
            "silhouette_support": output / f"{stem}_silhouette_support.png",
            "depth": output / f"{stem}_depth_consensus_mm.npy",
            "depth_valid": output / f"{stem}_depth_valid_consensus.png",
            "cloud_mask": output / f"{stem}_cloud_mask_consensus.png",
            "depth_support": output / f"{stem}_depth_support.png",
            "depth_support_npy": output / f"{stem}_depth_support.npy",
            "agreement_support": output / f"{stem}_agreement_support.npy",
            "depth_spread": output / f"{stem}_depth_spread_mm.npy",
            "depth_spread_vis": output / f"{stem}_depth_spread_vis.png",
            "low_agreement": output / f"{stem}_low_agreement.png",
            "selection_mode": output / f"{stem}_selection_mode.npy",
            "stable_interior": output / f"{stem}_stable_interior.png",
            "confidence": output / f"{stem}_confidence_consensus.npy",
            "support_veto": output / f"{stem}_support_veto.png",
            "support_object_votes": output / f"{stem}_support_object_votes.png",
            "overlay": output / f"{stem}_consensus_overlay.png",
        }
        cv2.imwrite(str(paths["image"]), image)
        cv2.imwrite(str(paths["silhouette"]), silhouette)
        max_support = max(len(aligned), 1)
        cv2.imwrite(
            str(paths["silhouette_support"]),
            np.clip(silhouette_support.astype(np.float32) / max_support * 255, 0, 255).astype(
                np.uint8
            ),
        )
        np.save(paths["depth"], depth)
        cv2.imwrite(str(paths["depth_valid"]), depth_valid)
        cv2.imwrite(str(paths["cloud_mask"]), cloud_mask)
        cv2.imwrite(
            str(paths["depth_support"]),
            np.clip(depth_support.astype(np.float32) / max_support * 255, 0, 255).astype(np.uint8),
        )
        np.save(paths["depth_support_npy"], depth_support)
        np.save(paths["agreement_support"], depth_agreement_support)
        np.save(paths["depth_spread"], depth_spread)
        cv2.imwrite(
            str(paths["depth_spread_vis"]),
            finite_vis(depth_spread, np.isfinite(depth_spread), cv2.COLORMAP_MAGMA),
        )
        cv2.imwrite(str(paths["low_agreement"]), low_agreement_map)
        np.save(paths["selection_mode"], selection_mode)
        cv2.imwrite(str(paths["stable_interior"]), stable_interior.astype(np.uint8) * 255)
        np.save(paths["confidence"], confidence)
        cv2.imwrite(str(paths["support_veto"]), support_veto.astype(np.uint8) * 255)
        max_votes = max(len(aligned), 1)
        cv2.imwrite(
            str(paths["support_object_votes"]),
            np.clip(support_object_votes.astype(np.float32) / max_votes * 255.0, 0, 255).astype(
                np.uint8
            ),
        )
        overlay = image.copy()
        tint = np.zeros_like(overlay)
        tint[:, :, 1] = cloud_mask
        overlay = cv2.addWeighted(overlay, 1.0, tint, 0.35, 0)
        text = f"P{pose:02d} | {physical:.3f} deg | obs={len(aligned)} | valid={valid_ratio:.1%}"
        cv2.putText(
            overlay, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            overlay, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.imwrite(str(paths["overlay"]), overlay)
        previews.append(paths["overlay"])
        depth_values = depth[(cloud_mask > 0) & np.isfinite(depth)]
        spread_values = depth_spread[(cloud_mask > 0) & np.isfinite(depth_spread)]
        ious = pairwise_iou(masks)
        sources = []
        for o in aligned:
            sources.append(
                {
                    "session": o["session"],
                    "stem": o["stem"],
                    "regional_quality": o["regional"].get("quality"),
                    "alignment": o["alignment"],
                }
            )
        record = {
            "pose_index": pose,
            "nominal_angle_deg": nominal,
            "physical_angle_deg": physical,
            "cumulative_steps": steps,
            "stem": stem,
            "quality": quality,
            "reasons": reasons,
            "observations_loaded": len(loaded),
            "observations_used": len(aligned),
            "silhouette_required_support": required,
            "silhouette_pixels": silhouette_pixels,
            "valid_depth_pixels": valid_pixels,
            "valid_ratio_of_silhouette": valid_ratio,
            "pairwise_silhouette_iou_after_alignment": finite_stats(np.asarray(ious)),
            "depth": finite_stats(depth_values),
            "depth_spread_mm": finite_stats(spread_values),
            "adaptive_depth_agreement_mm": float(adaptive_agreement_mm),
            "depth_agreement_diagnostics": agreement_diag,
            "depth_bias_diagnostics": depth_bias_diag,
            "stable_interior_pixels": int(np.count_nonzero(stable_interior)),
            "low_agreement_pixels": low_agreement_pixels,
            "low_agreement_ratio_of_valid": low_agreement_ratio,
            "confidence": finite_stats(confidence[(cloud_mask > 0) & np.isfinite(confidence)]),
            "agreement_support": finite_stats(depth_agreement_support[cloud_mask > 0]),
            "support_depth_veto": support_veto_diag,
            "support_veto_pixels": int(np.count_nonzero(support_veto)),
            "alignment": alignment_info,
            "sources": sources,
            "ignored": ignored,
            "outputs": {k: str(v) for k, v in paths.items()},
        }
        save_json(output / f"{stem}_consensus_stats.json", record)
        records_out.append(record)
        csv_rows.append(
            {
                "pose_index": pose,
                "nominal_angle_deg": nominal,
                "physical_angle_deg": physical,
                "cumulative_steps": steps,
                "quality": quality,
                "observations_used": len(aligned),
                "valid_ratio_of_silhouette": valid_ratio,
                "depth_median_mm": record["depth"]["median"],
                "depth_spread_p95_mm": record["depth_spread_mm"]["p95"],
                "adaptive_depth_agreement_mm": float(adaptive_agreement_mm),
                "low_agreement_pixels": low_agreement_pixels,
                "low_agreement_ratio_of_valid": low_agreement_ratio,
                "confidence_median": record["confidence"]["median"],
                "support_veto_pixels": int(np.count_nonzero(support_veto)),
                "silhouette_iou_median_after_alignment": record[
                    "pairwise_silhouette_iou_after_alignment"
                ]["median"],
                "reasons": " | ".join(reasons),
            }
        )
        shifts = [f"{a['session']}({a['dx']:+d},{a['dy']:+d})" for a in alignment_info]
        print(
            f"[P{pose:02d}] {physical:9.3f}° | obs={len(aligned)} | valid={valid_ratio:6.2%} | tol={adaptive_agreement_mm:.2f}mm | acuerdo_bajo={low_agreement_pixels}px ({low_agreement_ratio:.1%}) | veto_plato={int(np.count_nonzero(support_veto))}px | shift={' '.join(shifts)} | {quality}"
        )
    return {"records_out": records_out, "csv_rows": csv_rows, "previews": previews}


def main():
    """Coordina el consenso por pose entre sesiones y guarda el resumen multisesión."""
    args = build_parser().parse_args()
    root = Path(args.root).expanduser().resolve()
    obj = args.object.strip().lower()

    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else root
        / "reconstruccion"
        / "multisesion"
        / "01_mapa_angular"
        / "mapa_angular_multisesion.json"
    )
    manifest = load_manifest(manifest_path)
    sessions = list(manifest["sessions"])
    regional_map = load_regional_records(root, obj, sessions, args.regional_source)

    output = root / "reconstruccion" / "multisesion" / args.output_name
    output.mkdir(parents=True, exist_ok=True)

    by_pose = defaultdict(list)
    for rec in manifest["records"]:
        by_pose[int(rec["pose_index"])].append(rec)

    records_out, csv_rows, previews = [], [], []

    print("\n========== PASO 05 — CONSENSO ALINEADO ==========")
    print(f"Sesiones: {sessions}")
    print(
        f"Alineación permitida: traslación |dx|≤{args.alignment_max_x_px}px, "
        f"|dy|≤{args.alignment_max_y_px}px. Sin rotación/escala."
    )

    _results = ejecutar_items(
        _procesar_unidad_independiente,
        {
            "args": args,
            "by_pose": by_pose,
            "obj": obj,
            "output": output,
            "regional_map": regional_map,
            "root": root,
            "sessions": sessions,
        },
        range(int(manifest["moves_per_revolution"])),
        "Alinear sesiones y calcular consenso de cada pose",
        reserve_mb=1024,
    )
    for _result in _results:
        records_out.extend(_result["records_out"])
        csv_rows.extend(_result["csv_rows"])
        previews.extend(_result["previews"])

    if previews:
        build_contact_sheet(
            previews,
            output / "contact_sheet_consenso_multisesion.png",
        )

    summary = {
        "schema_version": 6,
        "method": (
            "restricted_2d_alignment_single_layer_medoid_then_robust_"
            "inverse_depth_fusion_with_explicit_per_pixel_reliability"
        ),
        "object": obj,
        "session": "multisesion",
        "sessions": sessions,
        "manifest": str(manifest_path),
        "steps_per_revolution": manifest["steps_per_revolution"],
        "moves_per_revolution": manifest["moves_per_revolution"],
        "step_sequence": manifest["step_sequence"],
        "nominal_step_deg": manifest["nominal_step_deg"],
        "parameters": vars(args),
        "views_processed": len(records_out),
        "views_accepted": sum(r.get("quality") == "accepted" for r in records_out),
        "views_warning": sum(r.get("quality") == "warning" for r in records_out),
        "views_rejected": sum(r.get("quality") == "rejected" for r in records_out),
        "views": records_out,
        "important_note": (
            "La alineación multisesión solo permite traslación 2D pequeña. "
            "No se deforma la geometría con escala, rotación ni desplazamiento "
            "Z. El medoide selecciona una sola capa y únicamente sus medidas "
            "compatibles se fusionan robustamente en 1/Z. El desacuerdo se "
            "conserva como confianza, dispersión y soporte "
            "para las etapas geométricas posteriores. El veto de plataforma "
            "por profundidad permanece desactivado por defecto."
        ),
    }
    save_json(output / "resumen_05_consenso_multisesion.json", summary)

    fields = [
        "pose_index",
        "nominal_angle_deg",
        "physical_angle_deg",
        "cumulative_steps",
        "quality",
        "observations_used",
        "valid_ratio_of_silhouette",
        "depth_median_mm",
        "depth_spread_p95_mm",
        "adaptive_depth_agreement_mm",
        "low_agreement_pixels",
        "low_agreement_ratio_of_valid",
        "confidence_median",
        "support_veto_pixels",
        "silhouette_iou_median_after_alignment",
        "reasons",
    ]
    with (output / "calidad_consenso_multisesion.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n========== PASO 05 COMPLETADO ==========")
    print(
        f"Accepted={summary['views_accepted']} | "
        f"Warning={summary['views_warning']} | "
        f"Rejected={summary['views_rejected']}"
    )
    print(f"Salida: {output}")
    print("================================================\n")
    return 0


def _entrada_con_diagnostico():
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    from utilidades_progreso import informar_inicio

    informar_inicio(__file__, "05", "Crear consenso multisesión")
    raise SystemExit(main())


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
