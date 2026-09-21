#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PASO 11 V11.8 — FUSIÓN MULTIVISTA ROBUSTA GENERAL
==============================================

Entrada
-------
- 25 nubes ya registradas y depuradas de 10_registro_calibrado.
- calibracion_plataforma_candidata/definitiva.json.
- resultado ACCEPTED o WARNING trazable de 10_registro_calibrado; REJECTED se bloquea.
- modelo temporal de poses validado por 10, cuando exista y haya sido aceptado.

Salida
------
Una única nube 360° fusionada que conserva:
- posición robusta;
- color;
- normal;
- número de poses independientes que respaldan cada muestra;
- confianza multivista;
- dispersión espacial;
- coherencia de normales;
- incidencia media.

Diseño
------
NO usa forma específica:
- no cubo;
- no cilindro;
- no pirámide;
- no primitivas globales ni caras predeterminadas;
- no Manhattan;
- no ICP 6DoF;
- no correcciones angulares por objeto.

Fusión regional V11.8 (activada por defecto):
- La ruta normal NO selecciona primitivas geométricas: el refinamiento de plano/cilindro/cuadrática permanece desactivado por defecto;
- la conciliación usa superficies cuadráticas locales agnósticas a la forma y evidencia multivista;
- regiones de trabajo adaptativas, con máximo 512 semillas y halos compartidos;
- modelos cuadráticos solapados, conciliados simultáneamente sin deriva iterativa;
- una corrección conserva el soporte propio y se revalida contra cada pose;
- reparto de procesos y memoria mediante las utilidades optimizadas existentes;
- --no-regional-fusion conserva la ruta de ajuste local independiente V8;
- no corrige poses automáticamente ni garantiza circularidad perfecta.

Fusión local de base:

- conserva las observaciones individuales registradas de todas las poses;
- el voxel organiza semillas reales, con varias superficies por celda;
- compara vecinos de cada pose por posición y normal orientada;
- separa capas y ajusta parches cuadráticos locales mediante IRLS;
- limita el movimiento normal usando incertidumbre EMPÍRICA de consenso;
- conserva observaciones sin proyectar si el ajuste no está respaldado;
- registra soporte por pose, conflictos, incertidumbre y causas de fallback.
No presupone covarianzas calibradas que no estén presentes en las entradas.
La ruta anterior se puede seleccionar mediante --no-local-surface-fusion.

Selección V8.0: parches independientes con evidencia multivista propia,
continuidad tangencial, normales compatibles e incertidumbre acotada. El núcleo
de cinco poses es una etiqueta de evidencia, no una restricción de proximidad.
--no-independent-patches conserva la selección histórica para comparación.

V11.8 añade una segunda pasada de recuperación de COBERTURA OBSERVADA. No
interpola ni crea puntos: reconsidera únicamente surfels que ya existen en la
fusión local y que conservan evidencia multivista propia. Un candidato solo
puede volver si está conectado a superficie validada, mantiene normales y
residuo tangencial compatibles, posee respaldo de poses/confianza suficiente y
la recuperación mejora la cobertura sin degradar de forma material la calidad
global. Esto evita que un filtro local excesivamente conservador fragmente una
superficie real, sin introducir una forma geométrica esperada.

El completado exporta geometría INFERIDA en un archivo separado y es
conservador por diseño. Un contorno cerrado no basta: cada candidato debe ser
compatible con siluetas, visibilidad y espacio libre en múltiples poses
independientes, además de respetar un presupuesto local de tamaño y distancia
a observaciones. Las tapas terminales permanecen desactivadas por defecto.
No añade soporte multivista ficticio ni clasifica la forma del objeto. El paso
13 solo puede consumir guías que acrediten este contrato multivista.
"""

from __future__ import annotations
import sys as _sys
import time as _time
import threading as _threading
import atexit as _atexit

_estado_lock = _threading.Lock()
_estado_inicio = _time.monotonic()
_estado_actual = ("Cargando dependencias de geometría", _estado_inicio)
_estado_stop = _threading.Event()


def _mostrar_estado11():
    """Publica la actividad actual y los tiempos transcurridos con lectura protegida."""
    with _estado_lock:
        actividad, inicio = _estado_actual
    ahora = _time.monotonic()
    print(
        f"[PROGRESO] Paso 11 | {actividad} | Tiempo actividad: {ahora-inicio:.1f} s | "
        f"Tiempo total: {ahora-_estado_inicio:.1f} s",
        flush=True,
    )


def _estado11(actividad):
    """Actualiza la actividad bajo bloqueo y publica el nuevo estado."""
    global _estado_actual
    with _estado_lock:
        _estado_actual = (actividad, _time.monotonic())
    _mostrar_estado11()


def _latido11():
    """Publica periódicamente el estado hasta recibir la señal de detención."""
    while not _estado_stop.wait(10):
        _mostrar_estado11()


if __name__ == "__main__":
    for _stream in (_sys.stdout, _sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(line_buffering=True, write_through=True)
            except (ValueError, OSError):
                pass
    print("[EJECUTANDO] 11_fusionar_nubes.py | Iniciando procesamiento", flush=True)
    _mostrar_estado11()
    _monitor11 = _threading.Thread(target=_latido11, name="estado_paso11", daemon=True)
    _monitor11.start()
    _atexit.register(_estado_stop.set)

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
except Exception:
    cv2 = None

try:
    from scipy.spatial import cKDTree
except Exception as exc:
    raise SystemExit(f"Paso 11 requiere SciPy: {exc}")

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def build_parser():
    """Construye las opciones de línea de comandos de este paso."""
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--object", required=True)
    p.add_argument("--calibration", required=True, help="Calibración de plataforma vigente.")
    p.add_argument("--cloud-source", default="06_nubes_puntos")
    p.add_argument("--consensus-source", default="05_consenso_multisesion")
    p.add_argument(
        "--registration-source",
        default="10_registro_calibrado",
    )
    p.add_argument("--output-name", default="11_fusion_multivista")
    p.add_argument(
        "--align-platform-axis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Expresa la nube final en un marco canónico cuyo eje Y coincide "
            "con el eje físico de giro. Es una transformación rígida general."
        ),
    )

    # Resolución física: elegida en el orden del error residual observado en 10.
    p.add_argument("--prevoxel-mm", type=float, default=0.90)
    p.add_argument("--fusion-voxel-mm", type=float, default=1.50)

    # Soporte multivista histórico (se conserva para compatibilidad/fallback).
    p.add_argument("--minimum-core-support", type=int, default=2)
    p.add_argument(
        "--preserve-boundary-singletons", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--singleton-neighbor-radius-cells", type=int, default=1)
    p.add_argument("--singleton-minimum-core-neighbors", type=int, default=2)
    p.add_argument("--singleton-minimum-confidence", type=float, default=0.42)

    # Consenso jerárquico general. No presupone caras, cilindros ni planos.
    p.add_argument(
        "--hierarchical-consensus",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--preferred-anchor-support", type=int, default=5)
    p.add_argument("--fallback-anchor-support", type=int, default=3)
    p.add_argument("--minimum-anchor-points", type=int, default=1000)
    p.add_argument("--minimum-anchor-ratio", type=float, default=0.03)
    p.add_argument(
        "--target-anchor-ratio",
        type=float,
        default=0.10,
        help=(
            "Densidad mínima deseable del núcleo respecto a todos los voxels "
            "candidatos; permite bajar de soporte 5 a 4 sin asumir una forma."
        ),
    )
    p.add_argument("--minimum-anchor-extent-coverage", type=float, default=0.92)
    p.add_argument("--anchor-continuity-radius-voxels", type=float, default=1.80)
    p.add_argument("--anchor-minimum-local-neighbors", type=int, default=2)
    p.add_argument("--minimum-anchor-local-continuity-ratio", type=float, default=0.55)
    p.add_argument("--minimum-anchor-median-confidence", type=float, default=0.72)
    p.add_argument("--minimum-anchor-median-normal-consistency", type=float, default=0.82)
    p.add_argument("--maximum-anchor-spread-p90-mm", type=float, default=1.50)
    p.add_argument("--minimum-extension-support", type=int, default=2)
    p.add_argument("--extension-radius-one-level-voxels", type=float, default=1.15)
    p.add_argument("--extension-radius-two-levels-voxels", type=float, default=0.85)
    p.add_argument("--extension-radius-low-support-voxels", type=float, default=0.65)
    p.add_argument(
        "--extension-radius-voxels",
        type=float,
        default=0.0,
        help=(
            "Compatibilidad: si es >0 reemplaza los tres radios adaptativos "
            "por un único radio uniforme."
        ),
    )

    # Pesos.
    p.add_argument("--minimum-incidence-cosine", type=float, default=0.15)
    p.add_argument("--incidence-power", type=float, default=1.5)
    p.add_argument("--depth-spread-reference-mm", type=float, default=6.0)

    # Control final.
    p.add_argument("--minimum-fused-confidence", type=float, default=0.32)
    p.add_argument("--maximum-spread-for-full-score-mm", type=float, default=2.5)
    p.add_argument(
        "--local-surface-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fusión robusta de parches entre voxels vecinos.",
    )
    p.add_argument("--local-radius-voxels", type=float, default=2.5)
    p.add_argument("--local-neighbors-per-pose", type=int, default=12)
    p.add_argument("--local-normal-angle-deg", type=float, default=35.0)
    p.add_argument("--local-sigma-floor-voxels", type=float, default=0.10)
    p.add_argument("--local-max-shift-voxels", type=float, default=0.50)
    # V11.0 — soporte no significa solamente contar poses: se exige diversidad
    # angular y se valida el modelo local contra poses excluidas del ajuste.
    p.add_argument(
        "--minimum-independent-pose-separation",
        type=int,
        default=2,
        help="Separación circular mínima (en índices de pose) para contar evidencia independiente.",
    )
    p.add_argument("--minimum-independent-support-poses", type=int, default=2)
    p.add_argument("--minimum-support-angular-span-poses", type=int, default=2)
    p.add_argument("--minimum-unvalidated-support-poses", type=int, default=3)
    p.add_argument("--minimum-unvalidated-confidence", type=float, default=0.68)
    p.add_argument("--minimum-unvalidated-agreement", type=float, default=0.68)
    p.add_argument("--maximum-conflict-pose-ratio", type=float, default=0.35)
    p.add_argument(
        "--minimum-heldout-pass-ratio",
        type=float,
        default=(2.0 / 3.0),
        help="Compatibilidad/diagnóstico. La decisión usa conteos enteros, no comparación flotante.",
    )
    p.add_argument("--minimum-heldout-pass-numerator", type=int, default=2)
    p.add_argument("--minimum-heldout-pass-denominator", type=int, default=3)

    # V11.3 — coherencia local multiescala para observaciones clase 2.
    # No presupone primitivas ni una forma global: solo comprueba si una
    # observación no validada held-out pertenece a una variedad superficial
    # local estable o a una característica geométrica local coherente.
    p.add_argument("--class2-local-coherence", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--class2-coherence-small-neighbors", type=int, default=18)
    p.add_argument("--class2-coherence-large-neighbors", type=int, default=36)
    p.add_argument("--class2-coherence-min-neighbors", type=int, default=10)
    p.add_argument("--class2-coherence-small-plane-p90-voxel", type=float, default=0.55)
    p.add_argument("--class2-coherence-large-plane-p90-voxel", type=float, default=0.80)
    p.add_argument("--class2-coherence-small-plane-uncertainty-factor", type=float, default=1.75)
    p.add_argument("--class2-coherence-large-plane-uncertainty-factor", type=float, default=2.20)
    p.add_argument("--class2-coherence-normal-p90-deg", type=float, default=42.0)
    p.add_argument("--class2-coherence-scale-normal-deg", type=float, default=25.0)
    p.add_argument("--class2-coherence-max-surface-variation", type=float, default=0.16)
    p.add_argument("--class2-coherence-min-same-sheet-fraction", type=float, default=0.30)
    p.add_argument(
        "--class2-coherence-reject-flag-count",
        type=int,
        default=3,
        help="Se rechaza clase 2 solo si acumula este número de fallos independientes.",
    )
    p.add_argument(
        "--class2-coherence-strict-max-flags",
        type=int,
        default=1,
        help="Máximo de fallos para que una clase 2 pueda actuar como ancla fuerte.",
    )
    p.add_argument("--class2-feature-min-family-fraction", type=float, default=0.18)
    p.add_argument("--class2-feature-min-family-points", type=int, default=4)
    p.add_argument("--class2-feature-max-within-family-p80-deg", type=float, default=17.0)
    p.add_argument("--class2-feature-min-family-separation-deg", type=float, default=25.0)
    p.add_argument("--class2-feature-min-tangent-backing-fraction", type=float, default=0.60)

    # V11.8 — recuperación conservadora de cobertura OBSERVADA. No genera
    # muestras ni rellena huecos geométricamente: solo reincorpora candidatos
    # ya medidos que quedaron fuera por el filtro local, siempre que estén
    # respaldados por varias poses y conectados a superficie validada.
    p.add_argument("--observed-coverage-recovery", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--coverage-recovery-iterations", type=int, default=6)
    p.add_argument("--coverage-recovery-max-distance-voxels", type=float, default=3.00)
    p.add_argument("--coverage-recovery-candidate-radius-voxels", type=float, default=2.75)
    p.add_argument("--coverage-recovery-min-kept-neighbors", type=int, default=1)
    p.add_argument("--coverage-recovery-min-candidate-neighbors", type=int, default=2)
    p.add_argument("--coverage-recovery-normal-angle-deg", type=float, default=42.0)
    p.add_argument("--coverage-recovery-tangent-residual-voxel", type=float, default=0.42)
    p.add_argument("--coverage-recovery-tangent-uncertainty-factor", type=float, default=2.0)
    p.add_argument("--coverage-recovery-min-support", type=int, default=2)
    p.add_argument("--coverage-recovery-min-independent-support", type=int, default=2)
    p.add_argument("--coverage-recovery-min-angular-span", type=int, default=2)
    p.add_argument("--coverage-recovery-min-confidence", type=float, default=0.55)
    p.add_argument("--coverage-recovery-min-agreement", type=float, default=0.55)
    p.add_argument("--coverage-recovery-min-normal-consistency", type=float, default=0.82)
    p.add_argument("--coverage-recovery-max-conflict-ratio", type=float, default=0.32)
    p.add_argument("--coverage-recovery-max-uncertainty-voxel", type=float, default=0.70)
    p.add_argument("--coverage-recovery-rejected-class2-min-score", type=float, default=0.58)
    p.add_argument("--coverage-recovery-rejected-class2-max-red-flags", type=int, default=4)
    p.add_argument("--coverage-recovery-rejected-class2-min-support", type=int, default=3)
    p.add_argument("--coverage-recovery-rejected-class2-min-confidence", type=float, default=0.70)
    p.add_argument("--coverage-recovery-max-fraction-of-initial", type=float, default=1.25)
    p.add_argument("--coverage-recovery-min-coverage-improvement", type=float, default=0.005)
    p.add_argument("--coverage-recovery-max-confidence-drop", type=float, default=0.06)
    p.add_argument("--coverage-recovery-max-normal-consistency-drop", type=float, default=0.04)
    p.add_argument("--coverage-recovery-max-uncertainty-p90-factor", type=float, default=1.35)
    p.add_argument(
        "--minimum-final-occupied-cell-retention",
        type=float,
        default=0.35,
        help=(
            "V2: fracción mínima de celdas originales con TODOS sus candidatos a <=1 voxel "
            "de la selección final. El nombre de opción se conserva por compatibilidad; no es "
            "la razón entre cantidades de celdas finales y originales."
        ),
    )
    p.add_argument(
        "--minimum-final-two-voxel-coverage",
        type=float,
        default=0.80,
        help=(
            "Fracción mínima de candidatos observados que debe quedar a <=2 voxels de "
            "algún surfel retenido. Es cobertura de muestreo, no área física."
        ),
    )

    # V11.4 — separación post-fit de hojas respaldadas por grupos de poses.
    # Se aplica antes de declarar una observación como clase 3. No presupone
    # cilindros, planos ni otra primitiva: solo usa residuos firmados por pose
    # respecto al modelo local ya validado held-out.
    p.add_argument(
        "--validated-pose-layer-separation", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--validated-layer-min-supported-poses", type=int, default=4)
    p.add_argument("--validated-layer-min-independent-per-layer", type=int, default=2)
    p.add_argument("--validated-layer-min-separation-voxel", type=float, default=0.34)
    p.add_argument("--validated-layer-uncertainty-factor", type=float, default=2.25)
    p.add_argument("--validated-layer-within-sigma-factor", type=float, default=3.0)
    p.add_argument("--validated-layer-min-winner-ratio", type=float, default=1.35)
    p.add_argument("--validated-layer-min-independent-advantage", type=int, default=1)
    p.add_argument("--validated-layer-max-shift-voxel", type=float, default=0.30)
    p.add_argument("--validated-layer-max-shift-uncertainty-factor", type=float, default=1.50)

    # V11.6 — consenso regional multiescala por procedencia de poses.
    # Se ejecuta DESPUÉS de la selección propia del 11 y corrige únicamente
    # ondulaciones que pueden predecirse de manera consistente desde una corona
    # vecina en dos escalas y desde varias poses independientes.
    p.add_argument("--regional-pose-consensus", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--regional-consensus-small-radius-voxels", type=float, default=3.2)
    p.add_argument("--regional-consensus-large-radius-voxels", type=float, default=5.0)
    p.add_argument("--regional-consensus-inner-radius-voxels", type=float, default=1.0)
    p.add_argument("--regional-consensus-max-neighbors", type=int, default=112)
    p.add_argument("--regional-consensus-min-neighbors", type=int, default=10)
    p.add_argument("--regional-consensus-min-pose-neighbors", type=int, default=7)
    p.add_argument("--regional-consensus-normal-angle-deg", type=float, default=28.0)
    p.add_argument("--regional-consensus-max-normal-p90-deg", type=float, default=38.0)
    p.add_argument("--regional-consensus-min-tangent-bins", type=int, default=5)
    p.add_argument("--regional-consensus-fit-p90-voxel", type=float, default=0.60)
    p.add_argument("--regional-consensus-fit-uncertainty-factor", type=float, default=2.8)
    p.add_argument("--regional-consensus-scale-gate-voxel", type=float, default=0.20)
    p.add_argument("--regional-consensus-scale-gate-uncertainty-factor", type=float, default=1.25)
    p.add_argument("--regional-consensus-pose-dispersion-voxel", type=float, default=0.18)
    p.add_argument(
        "--regional-consensus-pose-dispersion-uncertainty-factor", type=float, default=1.20
    )
    p.add_argument("--regional-consensus-min-independent-predictions", type=int, default=2)
    p.add_argument("--regional-consensus-deadband-voxel", type=float, default=0.08)
    p.add_argument("--regional-consensus-deadband-uncertainty-factor", type=float, default=0.45)
    p.add_argument("--regional-consensus-max-shift-voxel", type=float, default=0.22)
    p.add_argument("--regional-consensus-max-shift-uncertainty-factor", type=float, default=1.25)
    p.add_argument("--regional-consensus-update-alpha", type=float, default=0.75)
    p.add_argument("--regional-consensus-min-spacing-ratio", type=float, default=0.92)
    p.add_argument("--regional-consensus-max-extent-change-fraction", type=float, default=0.0075)

    # V11.6 — corrección de sesgo relativo por composición de poses usando
    # residuos de las observaciones CRUDAS del ajuste local, no coordenadas ya fusionadas.
    # El modelo es aditivo y local: residual(parche,pose)=sesgo_pose-gauge_parche.
    # Solo corrige la variación del gauge entre parches; un desplazamiento común a todas
    # las poses queda intacto y por tanto no impone ninguna forma global.
    p.add_argument("--raw-pose-bias-consensus", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--raw-pose-bias-small-radius-voxels", type=float, default=4.0)
    p.add_argument("--raw-pose-bias-large-radius-voxels", type=float, default=7.0)
    p.add_argument("--raw-pose-bias-max-neighbors", type=int, default=180)
    p.add_argument("--raw-pose-bias-min-patches-per-pose", type=int, default=5)
    p.add_argument("--raw-pose-bias-min-independent-poses", type=int, default=3)
    p.add_argument("--raw-pose-bias-normal-angle-deg", type=float, default=24.0)
    p.add_argument("--raw-pose-bias-model-p90-voxel", type=float, default=0.22)
    p.add_argument("--raw-pose-bias-model-p90-uncertainty-factor", type=float, default=1.6)
    p.add_argument("--raw-pose-bias-scale-gate-voxel", type=float, default=0.14)
    p.add_argument("--raw-pose-bias-scale-gate-uncertainty-factor", type=float, default=1.0)
    p.add_argument("--raw-pose-bias-deadband-voxel", type=float, default=0.05)
    p.add_argument("--raw-pose-bias-deadband-uncertainty-factor", type=float, default=0.35)
    p.add_argument("--raw-pose-bias-max-shift-voxel", type=float, default=0.20)
    p.add_argument("--raw-pose-bias-max-shift-uncertainty-factor", type=float, default=1.2)
    p.add_argument("--raw-pose-bias-update-alpha", type=float, default=0.75)
    p.add_argument("--raw-pose-bias-min-spacing-ratio", type=float, default=0.94)
    p.add_argument("--raw-pose-bias-max-extent-change-fraction", type=float, default=0.005)

    p.add_argument(
        "--regional-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Conciliar parches solapados y verificar sus cambios contra cada pose.",
    )
    p.add_argument(
        "--geometry-refinement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnóstico/refinamiento opcional con modelos locales plano/cilindro/cuadrático. "
            "Desactivado por defecto: la salida normal permanece estrictamente agnóstica a primitivas."
        ),
    )
    p.add_argument("--independent-patches", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--complete-gaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Genera solo guías locales que superen validación multivista; no son mediciones.",
    )
    p.add_argument(
        "--complete-bottom",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Hipótesis opcional de cierre terminal. Desactivada por defecto por ser inferida.",
    )
    p.add_argument(
        "--completion-max-span-mm",
        type=float,
        default=0.0,
        help=(
            "0: límite automático estrictamente local = max(6 vox, min(12 vox, 18%% de la "
            "extensión robusta)). Se aplica también a cierres terminales explícitos."
        ),
    )
    p.add_argument("--completion-max-points", type=int, default=12000)
    p.add_argument("--completion-silhouette-tolerance-px", type=int, default=2)
    p.add_argument("--completion-min-silhouette-tested-poses", type=int, default=8)
    p.add_argument("--completion-min-silhouette-inside-ratio", type=float, default=0.90)
    p.add_argument("--completion-max-silhouette-contradictions", type=int, default=1)
    p.add_argument("--completion-depth-pixel-radius", type=float, default=3.0)
    p.add_argument("--completion-depth-neighbors", type=int, default=8)
    p.add_argument("--completion-depth-noise-floor-mm", type=float, default=1.0)
    p.add_argument("--completion-depth-uncertainty-factor", type=float, default=2.0)
    p.add_argument("--completion-min-depth-tested-poses", type=int, default=2)
    p.add_argument("--completion-min-independent-depth-agreements", type=int, default=2)
    p.add_argument("--completion-max-free-space-contradictions", type=int, default=0)
    p.add_argument("--completion-min-independent-pose-separation", type=int, default=2)
    p.add_argument("--completion-min-retained-fraction", type=float, default=0.70)
    p.add_argument("--completion-max-observation-distance-voxels", type=float, default=3.0)
    return p


def load_json(path: Path) -> dict:
    """Lee un documento JSON codificado en UTF-8."""
    return json.loads(path.read_text(encoding="utf-8"))


def finite_stats(values) -> dict:
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


def robust_extent(points: np.ndarray) -> Optional[np.ndarray]:
    """Extensión P01-P99, resistente a unos pocos puntos extremos."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 8:
        return None
    return np.percentile(p, 99.0, axis=0) - np.percentile(p, 1.0, axis=0)


def normalize_rows(n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    if n.ndim != 2 or n.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float64)
    lengths = np.linalg.norm(n, axis=1)
    out = np.zeros_like(n)
    good = np.isfinite(lengths) & (lengths > 1e-12)
    out[good] = n[good] / lengths[good, None]
    return out


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Aplica a los puntos la rotación y traslación de una matriz homogénea."""
    return points @ T[:3, :3].T + T[:3, 3]


def transform_normals(normals: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Aplica solo la rotación a las normales y normaliza el resultado."""
    if len(normals) == 0:
        return normals
    return normalize_rows(normals @ T[:3, :3].T)


def platform_canonical_frame(calibration: dict, registration: dict):
    """Marco rígido determinista: Y coincide con el eje físico de giro."""
    model = calibration.get("rotation_axis", {})
    axis = np.asarray(model.get("unit_axis_xyz"), dtype=np.float64).reshape(-1)
    if axis.size != 3 or not np.all(np.isfinite(axis)):
        raise RuntimeError("rotation_axis.unit_axis_xyz es inválido.")
    axis /= max(float(np.linalg.norm(axis)), 1e-12)

    refinement = registration.get("axis_line_refinement", {})
    if bool(registration.get("runtime_axis_line_refinement_used", False)):
        line_value = refinement.get("selected_line_point_xyz_mm")
    else:
        line_value = model.get("canonical_line_point_xyz_mm")
    origin = np.asarray(line_value, dtype=np.float64).reshape(-1)
    if origin.size != 3 or not np.all(np.isfinite(origin)):
        raise RuntimeError("No se pudo obtener el punto de la línea del eje de plataforma.")

    reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(reference @ axis)) > 0.92:
        reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = reference - axis * float(reference @ axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
    z_axis = np.cross(x_axis, axis)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-12)
    rotation = np.vstack([x_axis, axis, z_axis])
    if float(np.linalg.det(rotation)) < 0.0:
        z_axis *= -1.0
        rotation = np.vstack([x_axis, axis, z_axis])
    return (
        rotation,
        origin,
        {
            "enabled": True,
            "source_axis_xyz": axis.astype(float).tolist(),
            "source_line_point_xyz_mm": origin.astype(float).tolist(),
            "target_axis": "+Y",
            "rotation_source_to_platform": rotation.astype(float).tolist(),
            "rigid_only": True,
            "shape_assumptions_used": False,
        },
    )


def resolve_calibration(root: Path, explicit: str) -> Path:
    """Exige una ruta explícita a la calibración de plataforma existente."""
    if not explicit:
        raise ValueError("--calibration es obligatorio en modo producto.")
    p = Path(explicit).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def resolve_summary(folder: Path, preferred: Sequence[str]) -> Optional[Path]:
    """Localiza el primer resumen existente según el orden de preferencia."""
    for name in preferred:
        p = folder / name
        if p.is_file():
            return p
    return None


def validate_registration(root: Path, obj: str, source: str, calibration: Path):
    """Verifica el registro de entrada y su trazabilidad antes de fusionar."""
    folder = root / "reconstruccion" / "multisesion" / source
    summary = resolve_summary(folder, ("resumen_10_registro_calibrado.json",))
    if summary is None:
        raise FileNotFoundError(f"No existe resumen 10 en {folder}")
    data = load_json(summary)
    if int(data.get("schema_version", 0)) < 10:
        raise RuntimeError(
            "11 requiere el paso 10 con consistencia global de espacio libre. "
            "Ejecute nuevamente 06 y 10."
        )
    if "global_multiview_silhouette_carving" not in str(data.get("method", "")):
        raise RuntimeError("El resumen 10 no acredita tallado global por siluetas.")
    if "global_free_space_consistency" not in str(data.get("method", "")):
        raise RuntimeError("El resumen 10 no acredita consistencia global de espacio libre.")
    registration_quality = str(data.get("quality", "")).lower()
    if registration_quality not in {"accepted", "warning"}:
        raise RuntimeError(
            f"11 requiere 10 ACCEPTED/WARNING; REJECTED no se fusiona. Estado: {data.get('quality')}"
        )
    if data.get("runtime_angle_corrections_used") is not False:
        raise RuntimeError("10 usado no corresponde al runtime angular congelado.")
    if data.get("runtime_icp_6dof_used") is not False:
        raise RuntimeError("10 usado contiene ICP 6DoF.")
    registered_calibration = data.get("calibration_path")
    if registered_calibration:
        registered_path = Path(registered_calibration).expanduser().resolve()
        if registered_path != calibration.resolve():
            raise RuntimeError("El paso 10 fue ejecutado con otra calibración de plataforma.")
    return summary, data


def load_runtime_pose_transforms(registration: dict, calibration_path: Path, calibration: dict):
    """Consume el contrato de 10 sin aceptar rotaciones o ICP ocultos."""
    calibration_records = {int(record["pose_index"]): record for record in calibration["poses"]}
    original = {
        pose: np.asarray(record["transform_pose_to_P00_mechanical"], dtype=np.float64)
        for pose, record in calibration_records.items()
    }
    refinement_applied = bool(registration.get("runtime_axis_line_refinement_used", False))
    if not refinement_applied:
        return original, {
            "source": "frozen_platform_calibration",
            "axis_line_refinement_applied": False,
            "runtime_pose_model_path": None,
        }

    value = registration.get("runtime_pose_model_path")
    if not value:
        raise RuntimeError("10 declara refinamiento del eje pero no publicó el modelo de poses.")
    model_path = Path(value).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Falta el modelo temporal de poses validado por 10: {model_path}")
    model = load_json(model_path)
    if model.get("axis_line_refinement_applied") is not True:
        raise RuntimeError("El modelo temporal no acredita el refinamiento declarado.")
    if model.get("mechanical_angles_fixed") is not True:
        raise RuntimeError("El modelo temporal no garantiza ángulos mecánicos fijos.")
    if model.get("runtime_angle_corrections_used") is not False:
        raise RuntimeError("El modelo temporal contiene correcciones angulares.")
    if model.get("runtime_icp_6dof_used") is not False:
        raise RuntimeError("El modelo temporal contiene ICP 6DoF.")

    source_calibration = Path(model.get("source_calibration_path", "")).expanduser().resolve()
    if source_calibration != calibration_path.resolve():
        raise RuntimeError("El modelo temporal fue generado con otra calibración de plataforma.")

    poses = model.get("poses")
    if not isinstance(poses, list) or len(poses) != 25:
        raise RuntimeError("El modelo temporal no contiene exactamente 25 poses.")
    runtime = {}
    for record in poses:
        pose = int(record["pose_index"])
        if pose not in original or pose in runtime:
            raise RuntimeError("Índices inválidos o repetidos en el modelo temporal.")
        T = np.asarray(record["transform_pose_to_P00_runtime"], dtype=np.float64)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            raise RuntimeError(f"Transformación temporal inválida en P{pose:02d}.")
        if not np.allclose(T[:3, :3], original[pose][:3, :3], atol=1e-9, rtol=0.0):
            raise RuntimeError(f"El modelo temporal alteró la rotación mecánica de P{pose:02d}.")
        if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12, rtol=0.0):
            raise RuntimeError(f"Matriz homogénea inválida en P{pose:02d}.")
        runtime[pose] = T
    if sorted(runtime) != list(range(25)):
        raise RuntimeError("Las poses temporales no son P00..P24 continuas.")

    return runtime, {
        "source": "step_10_cross_validated_runtime_pose_model",
        "axis_line_refinement_applied": True,
        "runtime_pose_model_path": str(model_path),
        "line_point_xyz_mm": model.get("line_point_xyz_mm"),
        "axis_direction_xyz": model.get("axis_direction_xyz"),
    }


def load_consensus_pose_weights(
    root: Path, obj: str, source: str, spread_ref: float
) -> Dict[int, float]:
    """Peso de pose derivado del consenso multisesión, sin anular vistas.

    V11.7 incorpora explícitamente la fracción de desacuerdo multisesión cuando
    el resumen 05 la publica. Esto evita que una pose con muchas observaciones
    pero poca repetibilidad tenga la misma autoridad que una pose estable.
    """
    folder = root / "reconstruccion" / "multisesion" / source
    summary = resolve_summary(folder, ("resumen_05_consenso_multisesion.json",))
    if summary is None:
        return {i: 1.0 for i in range(25)}

    data = load_json(summary)
    weights = {}
    for i, rec in enumerate(data.get("views", [])):
        pose = int(rec.get("pose_index", i))
        quality = str(rec.get("quality", "accepted")).lower()
        q = 1.0 if quality == "accepted" else (0.82 if quality == "warning" else 0.50)
        spread = rec.get("depth_spread_mm", {}).get("p95")
        if spread is None:
            spread = rec.get("depth_spread_p95_mm")
        if spread is None or not np.isfinite(float(spread)):
            spread_factor = 0.85
        else:
            s = float(spread)
            spread_factor = 0.65 + 0.35 * math.exp(-0.5 * (s / max(spread_ref, 1e-6)) ** 2)

        disagreement = rec.get("low_agreement_ratio_of_valid")
        if disagreement is None:
            disagreement = rec.get("low_agreement_ratio")
        if disagreement is None or not np.isfinite(float(disagreement)):
            agreement_factor = 1.0
        else:
            d = float(np.clip(float(disagreement), 0.0, 1.0))
            agreement_factor = float(np.clip(math.exp(-2.0 * d), 0.55, 1.0))

        weights[pose] = float(np.clip(q * spread_factor * agreement_factor, 0.35, 1.0))
    for pose in range(25):
        weights.setdefault(pose, 1.0)
    return weights


def load_registration_pose_weights(registration: dict):
    """Autoridad relativa por pose a partir de las aristas consecutivas de 10.

    No modifica poses. Solo reduce la influencia de observaciones que participan
    en relaciones consecutivas claramente peores que la mediana de la misma
    captura. Es relativo al ensayo, por lo que no presupone tamaño ni forma.
    """
    edges = [
        e
        for e in registration.get("edges", [])
        if isinstance(e, dict) and e.get("primary") and e.get("point_plane") is not None
    ]
    if not edges:
        return {i: 1.0 for i in range(25)}, {"available": False, "reason": "no_primary_edges"}
    rmses = np.asarray([float(e["point_plane"]["rmse_mm"]) for e in edges], dtype=float)
    p90s = np.asarray([float(e["point_plane"]["p90_abs_mm"]) for e in edges], dtype=float)
    overlaps = np.asarray([float(e.get("overlap", 0.0)) for e in edges], dtype=float)
    med_rmse = max(float(np.median(rmses)), 1e-9)
    med_p90 = max(float(np.median(p90s)), 1e-9)
    med_overlap = max(float(np.median(overlaps)), 1e-9)
    edge_records = []
    incident = {i: [] for i in range(25)}
    for e in edges:
        rmse = float(e["point_plane"]["rmse_mm"])
        p90 = float(e["point_plane"]["p90_abs_mm"])
        overlap = float(e.get("overlap", 0.0))
        f_rmse = float(np.clip(med_rmse / max(rmse, 1e-9), 0.35, 1.0))
        f_p90 = float(np.clip(med_p90 / max(p90, 1e-9), 0.35, 1.0))
        f_overlap = float(np.clip(overlap / med_overlap, 0.50, 1.0))
        weight = float(np.clip(math.sqrt(f_rmse * f_p90 * f_overlap), 0.35, 1.0))
        s, q = int(e["source_pose"]), int(e["target_pose"])
        incident.setdefault(s, []).append(weight)
        incident.setdefault(q, []).append(weight)
        edge_records.append(
            {
                "source_pose": s,
                "target_pose": q,
                "weight": weight,
                "rmse_mm": rmse,
                "p90_mm": p90,
                "overlap": overlap,
            }
        )
    pose_weights = {}
    pose_records = []
    for pose in range(25):
        values = incident.get(pose, [])
        if not values:
            weight = 1.0
        else:
            vals = np.asarray(values, dtype=float)
            geometric = float(np.exp(np.mean(np.log(np.clip(vals, 1e-9, 1.0)))))
            weight = float(np.clip(math.sqrt(float(np.min(vals)) * geometric), 0.35, 1.0))
        pose_weights[pose] = weight
        pose_records.append({"pose_index": pose, "weight": weight})
    return pose_weights, {
        "available": True,
        "method": "relative_consecutive_edge_rmse_p90_overlap_authority",
        "registration_quality": registration.get("quality"),
        "median_rmse_mm": med_rmse,
        "median_p90_mm": med_p90,
        "median_overlap": med_overlap,
        "pose_weights": pose_records,
        "edge_weights": edge_records,
    }


def load_registered_clean_views(root: Path, source: str, registration: dict):
    """Carga P00..P24 ya transformadas y filtradas por el paso 10."""
    folder = root / "reconstruccion" / "multisesion" / source
    views = []
    declared = registration.get("outputs", {}).get("registered_clean_views", [])
    declared_poses = {
        int(item["pose_index"]): item
        for item in declared
        if isinstance(item, dict) and "pose_index" in item
    }
    for pose in range(25):
        cloud_path = folder / f"P{pose:02d}_registered_clean.npz"
        if not cloud_path.is_file():
            raise FileNotFoundError(
                f"Falta la vista depurada P{pose:02d} del paso 10: {cloud_path}"
            )

        with np.load(cloud_path) as data:
            points = np.asarray(data["points"], dtype=np.float64)
            colors = (
                np.asarray(data["colors"], dtype=np.uint8)
                if "colors" in data.files
                else np.full((len(points), 3), 180, dtype=np.uint8)
            )
            normals = (
                np.asarray(data["normals"], dtype=np.float64)
                if "normals" in data.files
                else np.empty((0, 3), dtype=np.float64)
            )
            reliability = (
                np.asarray(data["reliability"], dtype=np.float64)
                if "reliability" in data.files
                else np.ones(len(points), dtype=np.float64)
            )
            pixel_uv = (
                np.asarray(data["pixel_uv"], dtype=np.float64)
                if "pixel_uv" in data.files
                else np.empty((0, 2), dtype=np.float64)
            )
            image_shape_hw = (
                tuple(np.asarray(data["image_shape_hw"], dtype=np.int32).reshape(-1)[:2])
                if "image_shape_hw" in data.files
                else None
            )
            depth_uncertainty_mm = (
                np.asarray(data["depth_uncertainty_mm"], dtype=np.float64)
                if "depth_uncertainty_mm" in data.files
                else np.full(len(points), np.nan, dtype=np.float64)
            )

        good = np.all(np.isfinite(points), axis=1)
        points = points[good]
        colors = (
            colors[good] if len(colors) == len(good) else np.full((len(points), 3), 180, np.uint8)
        )
        normals = normals[good] if len(normals) == len(good) else np.empty((0, 3), np.float64)
        reliability = (
            reliability[good]
            if len(reliability) == len(good)
            else np.ones(len(points), dtype=np.float64)
        )
        pixel_uv = (
            pixel_uv[good] if len(pixel_uv) == len(good) else np.empty((0, 2), dtype=np.float64)
        )
        depth_uncertainty_mm = (
            depth_uncertainty_mm[good]
            if len(depth_uncertainty_mm) == len(good)
            else np.full(len(points), np.nan, dtype=np.float64)
        )
        normals = normalize_rows(normals)
        views.append(
            {
                "pose_index": pose,
                "quality": "accepted",
                "stem": f"P{pose:02d}",
                "points": points,
                "colors": colors,
                "normals": normals,
                "reliability": np.clip(reliability, 0.0, 1.0),
                "pixel_uv": pixel_uv,
                "image_shape_hw": image_shape_hw,
                "depth_uncertainty_mm": depth_uncertainty_mm,
                "already_registered": True,
                "path": str(cloud_path),
                "declared_output": declared_poses.get(pose),
            }
        )
    views.sort(key=lambda x: x["pose_index"])
    if len(views) != 25 or [v["pose_index"] for v in views] != list(range(25)):
        raise RuntimeError(f"11 requiere P00..P24. Cargadas: {[v['pose_index'] for v in views]}")
    return folder, views


def aggregate_by_inverse(values: np.ndarray, inverse: np.ndarray, count: int, weights=None):
    if weights is None:
        weights = np.ones(len(values), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    denom = np.bincount(inverse, weights=weights, minlength=count)
    if values.ndim == 1:
        num = np.bincount(inverse, weights=weights * values, minlength=count)
        return num / np.maximum(denom, 1e-12), denom
    out = np.zeros((count, values.shape[1]), dtype=np.float64)
    for j in range(values.shape[1]):
        out[:, j] = np.bincount(inverse, weights=weights * values[:, j], minlength=count)
    out /= np.maximum(denom[:, None], 1e-12)
    return out, denom


def per_pose_prevoxel(points, colors, normals, voxel):
    """Agrupa puntos de una pose por vóxel junto con colores y normales."""
    if len(points) == 0:
        return points, colors, normals
    keys = np.floor(points / float(voxel)).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    m = len(unique)
    p, _ = aggregate_by_inverse(points, inverse, m)
    c, _ = aggregate_by_inverse(colors.astype(np.float64), inverse, m)

    if len(normals) == len(points):
        n, _ = aggregate_by_inverse(normals, inverse, m)
        n = normalize_rows(n)
    else:
        n = np.zeros((m, 3), dtype=np.float64)

    return p, np.clip(np.rint(c), 0, 255).astype(np.uint8), n


def robust_neighbor_core_counts(keys: np.ndarray, core: np.ndarray, radius: int) -> np.ndarray:
    """Cuenta celdas de núcleo vecinas dentro del radio discreto indicado."""
    core_set = {tuple(k.tolist()) for k in keys[core]}
    out = np.zeros(len(keys), dtype=np.int16)
    if not core_set:
        return out
    offsets = [
        (dx, dy, dz)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    candidates = np.flatnonzero(~core)
    for idx in candidates:
        x, y, z = keys[idx]
        total = 0
        for dx, dy, dz in offsets:
            if (int(x + dx), int(y + dy), int(z + dz)) in core_set:
                total += 1
        out[idx] = total
    return out


def local_continuity_statistics(
    points: np.ndarray,
    radius_mm: float,
    minimum_neighbors: int,
) -> dict:
    """Mide continuidad local sin imponer conectividad ni forma global.

    La métrica admite concavidades, agujeros y objetos con varias regiones de
    superficie: únicamente comprueba si cada muestra tiene respaldo espacial
    cercano dentro de la resolución de fusión.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return {
            "points": 0,
            "radius_mm": float(radius_mm),
            "minimum_neighbors": int(minimum_neighbors),
            "continuous_point_ratio": 0.0,
            "neighbor_count": finite_stats([]),
        }

    radius_mm = max(float(radius_mm), 1e-9)
    minimum_neighbors = max(int(minimum_neighbors), 1)
    tree = cKDTree(points)
    # Contar directamente evita construir una lista Python por cada punto.
    counts = np.maximum(
        tree.query_ball_point(points, r=radius_mm, workers=query_threads(), return_length=True) - 1,
        0,
    ).astype(np.int32)
    return {
        "points": int(len(points)),
        "radius_mm": float(radius_mm),
        "minimum_neighbors": int(minimum_neighbors),
        "continuous_point_ratio": float(np.mean(counts >= minimum_neighbors)),
        "neighbor_count": finite_stats(counts),
    }


@operacion("Seleccionar consenso geométrico multivista")
def hierarchical_consensus_selection(
    points: np.ndarray,
    support: np.ndarray,
    confidence: np.ndarray,
    spread: np.ndarray,
    normal_consistency: np.ndarray,
    voxel_keys: np.ndarray,
    *,
    voxel_size_mm: float,
    minimum_confidence: float,
    enabled: bool,
    preferred_anchor_support: int,
    fallback_anchor_support: int,
    minimum_anchor_points: int,
    minimum_anchor_ratio: float,
    target_anchor_ratio: float,
    minimum_anchor_extent_coverage: float,
    anchor_continuity_radius_voxels: float,
    anchor_minimum_local_neighbors: int,
    minimum_anchor_local_continuity_ratio: float,
    minimum_anchor_median_confidence: float,
    minimum_anchor_median_normal_consistency: float,
    maximum_anchor_spread_p90_mm: float,
    minimum_extension_support: int,
    extension_radius_one_level_voxels: float,
    extension_radius_two_levels_voxels: float,
    extension_radius_low_support_voxels: float,
    uniform_extension_radius_voxels: float,
    legacy_core_support: int,
    preserve_legacy_singletons: bool,
    singleton_neighbor_radius_cells: int,
    singleton_minimum_core_neighbors: int,
    singleton_minimum_confidence: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Selecciona surfels por respaldo entre vistas, sin modelo de forma.

    `selection_class` usa 3=núcleo fuerte, 2=extensión local y
    1=singleton del modo de compatibilidad. Cero significa descartado.
    """
    points = np.asarray(points, dtype=np.float64)
    support = np.asarray(support, dtype=np.int16)
    confidence = np.asarray(confidence, dtype=np.float64)
    spread = np.asarray(spread, dtype=np.float64)
    normal_consistency = np.asarray(normal_consistency, dtype=np.float64)
    eligible = np.isfinite(confidence) & (confidence >= float(minimum_confidence))
    candidate_count = int(len(points))
    required_anchor_points = max(
        int(minimum_anchor_points),
        int(math.ceil(max(float(minimum_anchor_ratio), 0.0) * candidate_count)),
    )

    preferred = max(int(preferred_anchor_support), 2)
    fallback = max(min(int(fallback_anchor_support), preferred), 2)
    minimum_extension = max(int(minimum_extension_support), 2)
    legacy = max(int(legacy_core_support), 1)
    minimum_extent_coverage = float(np.clip(minimum_anchor_extent_coverage, 0.0, 1.0))
    target_anchor_ratio = float(np.clip(target_anchor_ratio, 0.0, 1.0))
    minimum_continuity_ratio = float(np.clip(minimum_anchor_local_continuity_ratio, 0.0, 1.0))
    continuity_radius_mm = max(float(anchor_continuity_radius_voxels), 0.0) * float(voxel_size_mm)
    minimum_median_confidence = float(np.clip(minimum_anchor_median_confidence, 0.0, 1.0))
    minimum_median_normal_consistency = float(
        np.clip(minimum_anchor_median_normal_consistency, 0.0, 1.0)
    )
    maximum_spread_p90 = max(float(maximum_anchor_spread_p90_mm), 0.0)

    # Se evalúan todos los umbrales. El soporte alto sigue teniendo prioridad,
    # pero no puede bloquear un nivel inmediatamente inferior si su núcleo es
    # demasiado ralo. La continuidad es exclusivamente local y no impone una
    # topología ni una clase de objeto.
    anchor = np.zeros(candidate_count, dtype=bool)
    anchor_support = legacy
    anchor_candidates = []
    safe_anchor_options = []
    dense_anchor_options = []
    mode = "legacy_compatibility"
    if bool(enabled):
        for threshold in range(preferred, fallback - 1, -1):
            candidate_mask = eligible & (support >= threshold)
            reference_threshold = max(threshold - 1, minimum_extension)
            reference_mask = eligible & (support >= reference_threshold)
            candidate_points = int(np.count_nonzero(candidate_mask))
            reference_points = int(np.count_nonzero(reference_mask))
            # Si el nivel inmediato no aporta observaciones, se continúa
            # descendiendo hasta encontrar una referencia realmente más amplia.
            while reference_threshold > minimum_extension and reference_points == candidate_points:
                reference_threshold -= 1
                reference_mask = eligible & (support >= reference_threshold)
                reference_points = int(np.count_nonzero(reference_mask))
            candidate_extent = robust_extent(points[candidate_mask])
            reference_extent = robust_extent(points[reference_mask])

            coverage = np.full(3, np.nan, dtype=np.float64)
            valid_axes = np.zeros(3, dtype=bool)
            if candidate_extent is not None and reference_extent is not None:
                # Un eje casi degenerado no debe invalidar superficies delgadas.
                valid_axes = reference_extent > max(2.0 * float(voxel_size_mm), 1e-9)
                coverage[valid_axes] = candidate_extent[valid_axes] / reference_extent[valid_axes]

            enough_points = candidate_points >= required_anchor_points
            covers_extent = bool(
                np.any(valid_axes) and np.all(coverage[valid_axes] >= minimum_extent_coverage)
            )
            candidate_ratio = float(candidate_points / candidate_count) if candidate_count else 0.0
            continuity = local_continuity_statistics(
                points[candidate_mask],
                continuity_radius_mm,
                anchor_minimum_local_neighbors,
            )
            locally_continuous = bool(
                continuity["continuous_point_ratio"] >= minimum_continuity_ratio
            )
            confidence_stats = finite_stats(confidence[candidate_mask])
            normal_stats = finite_stats(normal_consistency[candidate_mask])
            spread_stats = finite_stats(spread[candidate_mask])
            consistent = bool(
                confidence_stats["median"] is not None
                and normal_stats["median"] is not None
                and spread_stats["p90"] is not None
                and confidence_stats["median"] >= minimum_median_confidence
                and normal_stats["median"] >= minimum_median_normal_consistency
                and spread_stats["p90"] <= maximum_spread_p90
            )
            safe = bool(enough_points and covers_extent and locally_continuous and consistent)
            reaches_target_density = bool(candidate_ratio >= target_anchor_ratio)
            record = {
                "support_threshold": int(threshold),
                "reference_support_threshold": int(reference_threshold),
                "points": int(candidate_points),
                "reference_points": int(reference_points),
                "ratio_of_all_candidates": float(candidate_ratio),
                "target_anchor_ratio": float(target_anchor_ratio),
                "enough_points": bool(enough_points),
                "extent_coverage_xyz": [None if not np.isfinite(x) else float(x) for x in coverage],
                "minimum_valid_axis_coverage": (
                    float(np.min(coverage[valid_axes])) if np.any(valid_axes) else None
                ),
                "covers_reference_extent": bool(covers_extent),
                "local_continuity": continuity,
                "locally_continuous": bool(locally_continuous),
                "confidence": confidence_stats,
                "normal_consistency": normal_stats,
                "spread_mm": spread_stats,
                "multiview_consistent": bool(consistent),
                "safe_anchor": bool(safe),
                "reaches_target_density": bool(reaches_target_density),
                "selected": False,
            }
            anchor_candidates.append(record)
            if safe:
                option = (int(threshold), candidate_mask.copy(), record)
                safe_anchor_options.append(option)
                if reaches_target_density:
                    dense_anchor_options.append(option)

        # Se elige el mayor soporte que además alcanza una densidad útil. Así,
        # soporte 5 permanece preferido cuando basta, pero soporte 4 puede formar
        # el núcleo si 5 es correcto aunque excesivamente disperso.
        chosen_options = dense_anchor_options or safe_anchor_options
        if chosen_options:
            anchor_support, anchor, selected_record = chosen_options[0]
            selected_record["selected"] = True
            mode = (
                "adaptive_hierarchical_dense"
                if dense_anchor_options
                else "adaptive_hierarchical_safe_sparse_fallback"
            )

    if mode == "legacy_compatibility":
        anchor_support = legacy
        anchor = eligible & (support >= legacy)

    selection_class = np.zeros(candidate_count, dtype=np.uint8)
    distance_to_anchor = np.full(candidate_count, np.inf, dtype=np.float64)
    selection_class[anchor] = 3
    distance_to_anchor[anchor] = 0.0
    keep = anchor.copy()
    singleton_ok = np.zeros(candidate_count, dtype=bool)

    extension_radius_used = np.zeros(candidate_count, dtype=np.float64)
    retained_extension_by_support = {}
    if mode != "legacy_compatibility" and np.any(anchor):
        extension_candidates = (
            eligible & (support >= minimum_extension) & (support < anchor_support)
        )
        candidate_idx = np.flatnonzero(extension_candidates)
        if len(candidate_idx):
            tree = cKDTree(points[anchor])
            distances, _ = tree.query(points[candidate_idx], k=1, workers=query_threads())
            support_gap = anchor_support - support[candidate_idx].astype(np.int32)
            uniform_radius = max(float(uniform_extension_radius_voxels), 0.0)
            if uniform_radius > 0.0:
                radius_voxels = np.full(len(candidate_idx), uniform_radius, dtype=np.float64)
            else:
                radius_voxels = np.where(
                    support_gap <= 1,
                    max(float(extension_radius_one_level_voxels), 0.0),
                    np.where(
                        support_gap == 2,
                        max(float(extension_radius_two_levels_voxels), 0.0),
                        max(float(extension_radius_low_support_voxels), 0.0),
                    ),
                )
            radius_mm = radius_voxels * float(voxel_size_mm)
            accepted_local = distances <= radius_mm + 1e-9
            accepted_idx = candidate_idx[accepted_local]
            distance_to_anchor[candidate_idx] = distances
            extension_radius_used[candidate_idx] = radius_mm
            keep[accepted_idx] = True
            selection_class[accepted_idx] = 2
            for level in sorted(np.unique(support[candidate_idx]).astype(int).tolist()):
                level_mask = support[candidate_idx] == level
                retained_extension_by_support[str(level)] = {
                    "candidates": int(np.count_nonzero(level_mask)),
                    "retained": int(np.count_nonzero(level_mask & accepted_local)),
                    "radius_voxels": float(np.unique(radius_voxels[level_mask])[0]),
                    "radius_mm": float(np.unique(radius_mm[level_mask])[0]),
                }
    else:
        # Respaldo compatible para adquisiciones sin núcleo multivista fuerte.
        neighbor_core_count = robust_neighbor_core_counts(
            voxel_keys,
            anchor,
            int(singleton_neighbor_radius_cells),
        )
        singleton_ok = (
            (support == 1)
            & bool(preserve_legacy_singletons)
            & eligible
            & (neighbor_core_count >= int(singleton_minimum_core_neighbors))
            & (confidence >= float(singleton_minimum_confidence))
        )
        keep |= singleton_ok
        selection_class[singleton_ok] = 1

    anchor_extent = robust_extent(points[anchor])
    final_extent = robust_extent(points[keep])
    if anchor_extent is not None and final_extent is not None:
        extent_ratio = np.divide(
            final_extent,
            anchor_extent,
            out=np.full(3, np.nan, dtype=np.float64),
            where=anchor_extent > 1e-9,
        )
    else:
        extent_ratio = np.full(3, np.nan, dtype=np.float64)

    support_counts = {
        str(level): int(np.count_nonzero(support == level))
        for level in sorted(np.unique(support).astype(int).tolist())
    }
    retained_counts = {
        str(level): int(np.count_nonzero(keep & (support == level)))
        for level in sorted(np.unique(support).astype(int).tolist())
    }
    rejected_counts = {
        str(level): int(np.count_nonzero((~keep) & (support == level)))
        for level in sorted(np.unique(support).astype(int).tolist())
    }

    diagnostics = {
        "enabled": bool(enabled),
        "mode": mode,
        "anchor_support_threshold_used": int(anchor_support),
        "required_anchor_points": int(required_anchor_points),
        "target_anchor_ratio": float(target_anchor_ratio),
        "minimum_anchor_extent_coverage": float(minimum_extent_coverage),
        "minimum_anchor_local_continuity_ratio": float(minimum_continuity_ratio),
        "anchor_continuity_radius_voxels": float(anchor_continuity_radius_voxels),
        "anchor_continuity_radius_mm": float(continuity_radius_mm),
        "anchor_minimum_local_neighbors": int(anchor_minimum_local_neighbors),
        "minimum_anchor_median_confidence": float(minimum_median_confidence),
        "minimum_anchor_median_normal_consistency": float(minimum_median_normal_consistency),
        "maximum_anchor_spread_p90_mm": float(maximum_spread_p90),
        "anchor_candidates": anchor_candidates,
        "preferred_anchor_points": int(np.count_nonzero(eligible & (support >= preferred))),
        "fallback_anchor_points": int(np.count_nonzero(eligible & (support >= fallback))),
        "anchor_points": int(np.count_nonzero(anchor)),
        "anchor_ratio_of_candidates": (float(np.mean(anchor)) if candidate_count else 0.0),
        "local_extension_points": int(np.count_nonzero(selection_class == 2)),
        "legacy_singletons_preserved": int(np.count_nonzero(singleton_ok)),
        "final_points": int(np.count_nonzero(keep)),
        "candidate_support_counts": support_counts,
        "retained_support_counts": retained_counts,
        "rejected_support_counts": rejected_counts,
        "extension_radius_schedule_voxels": {
            "one_support_level_below_anchor": float(extension_radius_one_level_voxels),
            "two_support_levels_below_anchor": float(extension_radius_two_levels_voxels),
            "lower_support": float(extension_radius_low_support_voxels),
            "uniform_override": (
                float(uniform_extension_radius_voxels)
                if float(uniform_extension_radius_voxels) > 0.0
                else None
            ),
        },
        "retained_extension_by_support": retained_extension_by_support,
        # Campo escalar conservado por compatibilidad con lectores anteriores.
        "extension_radius_mm": float(voxel_size_mm)
        * (
            float(uniform_extension_radius_voxels)
            if float(uniform_extension_radius_voxels) > 0.0
            else float(extension_radius_one_level_voxels)
        ),
        "extension_radius_mm_stats": finite_stats(extension_radius_used[selection_class == 2]),
        "anchor_robust_extent_mm": (
            None if anchor_extent is None else anchor_extent.astype(float).tolist()
        ),
        "final_robust_extent_mm": (
            None if final_extent is None else final_extent.astype(float).tolist()
        ),
        "final_to_anchor_robust_extent_ratio_xyz": [
            None if not np.isfinite(x) else float(x) for x in extent_ratio
        ],
    }
    return keep, selection_class, distance_to_anchor, diagnostics


def save_ply(path, points, colors, normals, support, confidence):
    with path.open("w", encoding="utf-8") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(points)}\n")
        for s in ("x", "y", "z", "nx", "ny", "nz"):
            fh.write(f"property float {s}\n")
        for s in ("red", "green", "blue"):
            fh.write(f"property uchar {s}\n")
        fh.write("property ushort support_views\n")
        fh.write("property float confidence\n")
        fh.write("end_header\n")
        for p, c, n, s, cf in zip(points, colors, normals, support, confidence):
            fh.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{n[0]:.6f} {n[1]:.6f} {n[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} "
                f"{int(s)} {float(cf):.6f}\n"
            )


def make_preview(path, points, support, confidence):
    if plt is None:
        raise RuntimeError(
            "No se puede generar preview_fusion_multivista.png porque "
            "matplotlib no está disponible. Ejecuta primero "
            "herramientas/00_verificar_dependencias.py."
        )
    if len(points) > 140000:
        idx = np.linspace(0, len(points) - 1, 140000).astype(int)
    else:
        idx = np.arange(len(points))
    P = points[idx]
    S = support[idx]
    C = confidence[idx]

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(2, 2, 1)
    sc = ax.scatter(P[:, 0], P[:, 1], c=S, s=0.35, cmap="viridis")
    ax.set_title("X-Y | soporte de poses")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, fraction=0.04)

    ax = fig.add_subplot(2, 2, 2)
    sc = ax.scatter(P[:, 0], P[:, 2], c=C, s=0.35, cmap="plasma", vmin=0, vmax=1)
    ax.set_title("X-Z | confianza")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, fraction=0.04)

    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(P[:, 2], P[:, 1], c=np.clip(S, 1, 6), s=0.35, cmap="viridis")
    ax.set_title("Z-Y")
    ax.set_aspect("equal", adjustable="box")

    ax = fig.add_subplot(2, 2, 4)
    ax.axis("off")
    txt = [
        "11 | fusión multivista general",
        f"puntos fusionados: {len(points):,}",
        f"soporte mediano: {np.median(support):.1f} poses",
        f"soporte >=2: {np.mean(support>=2):.1%}",
        f"confianza mediana: {np.median(confidence):.3f}",
        "selección geométrica regional / sin reajuste de poses",
    ]
    ax.text(0.05, 0.95, "\n\n".join(txt), va="top", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# Fusión local V7.0. No ajusta primitivas globales ni modifica poses.
# Incertidumbre EMPIRICA de consenso: no sustituye una covarianza calibrada.
LOCAL_STATUS = {
    0: "insufficient_local_multiview_support",
    1: "robust_quadratic_projection_applied",
    2: "one_sided_or_sparse_neighborhood_kept_observation",
    3: "ill_conditioned_fit_kept_observation",
    4: "uncertain_fit_kept_observation",
    5: "incompatible_normal_change_kept_observation",
}


def local_surface_seeds(points, normals, weights, voxel, angle_deg):
    """Semillas REALES; permite varias superficies dentro de una celda.

    No promedia previamente caras de una arista ni paredes contrapuestas.
    Cada grupo conserva su observación de mayor peso como semilla.
    """
    keys = np.floor(points / voxel).astype(np.int64)
    cells, inv = np.unique(keys, axis=0, return_inverse=True)
    order = np.lexsort((np.arange(len(points)), -weights, inv))
    starts = np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1, len(order)]
    chosen = []
    cos_limit = math.cos(math.radians(angle_deg))
    for start, stop in zip(starts[:-1], starts[1:]):
        members = order[start:stop]
        while len(members):
            seed = int(members[0])
            chosen.append(seed)
            delta = points[members] - points[seed]
            same = normals[members] @ normals[seed] >= cos_limit
            # Distancia en AMBAS normales: no fusionar las dos caras de una arista.
            h1 = np.abs(delta @ normals[seed])
            h2 = np.abs(np.einsum("ij,ij->i", delta, normals[members]))
            same &= np.maximum(h1, h2) <= 0.40 * voxel
            same[0] = True
            members = members[~same]
    chosen = np.asarray(chosen, dtype=np.int64)
    return chosen, keys[chosen], len(cells)


def _pose_mask(pose_ids):
    """Codifica las poses independientes como bits de una máscara entera."""
    value = 0
    for pose in np.unique(pose_ids):
        value |= 1 << int(pose)
    return np.uint64(value)


def _pose_ids_from_mask(mask, expected=25):
    """Recupera los índices de pose presentes en una máscara de bits."""
    value = int(mask)
    return [pose for pose in range(int(expected)) if value & (1 << pose)]


def _circular_pose_distance(a, b, expected=25):
    """Calcula la separación mínima de dos poses en la vuelta completa."""
    delta = abs(int(a) - int(b))
    return min(delta, int(expected) - delta)


def _pose_diversity(pose_ids, expected=25, minimum_separation=2):
    """Diversidad circular: varias vistas consecutivas no cuentan como independientes.

    Se calcula el mayor subconjunto greedy bajo todos los orígenes circulares.
    Para 25 poses es determinista, barato y evita que P00,P01,P02 equivalgan a
    tres vistas ampliamente separadas.
    """
    unique = sorted(set(int(v) for v in np.asarray(pose_ids).reshape(-1)))
    if not unique:
        return 0, 0
    if len(unique) == 1:
        return 1, 0
    minimum_separation = max(1, int(minimum_separation))
    best = 1
    for start in range(int(expected)):
        ordered = sorted(unique, key=lambda value: (value - start) % int(expected))
        chosen = []
        for value in ordered:
            if all(
                _circular_pose_distance(value, other, expected) >= minimum_separation
                for other in chosen
            ):
                chosen.append(value)
        best = max(best, len(chosen))
    span = max(
        _circular_pose_distance(a, b, expected)
        for i, a in enumerate(unique)
        for b in unique[i + 1 :]
    )
    return int(best), int(span)


def _heldout_vote_ok(passed_tests, tests, numerator=2, denominator=3):
    """Decisión racional exacta para validación held-out.

    Evita el error numérico/semántico de comparar 2/3=0.666... contra 0.67.
    La condición se evalúa como passed*denominator >= numerator*tests.
    Sin pruebas disponibles no bloquea: la ausencia de held-out se trata en
    las demás guardas de diversidad, soporte y conflicto.
    """
    tests = int(tests)
    passed_tests = int(passed_tests)
    numerator = max(0, int(numerator))
    denominator = max(1, int(denominator))
    if tests <= 0:
        return True
    return passed_tests * denominator >= numerator * tests


def _mask_popcount(mask):
    """Cuenta las poses presentes en una máscara de bits."""
    return int(int(mask).bit_count())


def _fit_local_quadratic_for_validation(design, z, pose, base_w, train, voxel, floor):
    """Ajuste IRLS usado exclusivamente para validación independiente."""
    train = np.asarray(train, dtype=bool)
    if np.count_nonzero(train) < 12 or len(np.unique(pose[train])) < 3:
        return None
    A = design[train]
    target = z[train]
    weights = np.asarray(base_w[train], dtype=np.float64).copy()
    coeff = np.zeros(6, dtype=np.float64)
    sigma = float(floor)
    for _ in range(4):
        root = np.sqrt(np.maximum(weights, 1e-12))
        try:
            coeff, _, rank, singular = np.linalg.lstsq(A * root[:, None], target * root, rcond=1e-8)
        except np.linalg.LinAlgError:
            return None
        if rank < 6 or singular[-1] <= 0 or singular[0] / singular[-1] > 5000:
            return None
        residual = target - A @ coeff
        train_pose = pose[train]
        centers = np.zeros(25, dtype=np.float64)
        for pid in np.unique(train_pose):
            centers[int(pid)] = np.median(residual[train_pose == pid])
        within = residual - centers[train_pose]
        within_sigma = 1.4826 * np.median(np.abs(within))
        medians = np.asarray([centers[int(pid)] for pid in np.unique(train_pose)])
        between_sigma = 1.4826 * np.median(np.abs(medians - np.median(medians)))
        sigma = float(
            np.sqrt(floor * floor + within_sigma * within_sigma + between_sigma * between_sigma)
        )
        scale = min(sigma, 0.30 * float(voxel))
        weights = base_w[train] * np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-12))
    if not np.all(np.isfinite(coeff)):
        return None
    return coeff, sigma


def _cross_validate_local_quadratic(
    design,
    z,
    u,
    v,
    pose,
    base_w,
    observed_normals,
    normal,
    tangent,
    bitangent,
    radius,
    voxel,
    floor,
    cos_angle,
):
    """Valida la superficie con poses que no participaron en su ajuste.

    Se dejan fuera hasta tres poses distribuidas a lo largo del soporte. Cada
    prueba entrena con al menos tres poses completas y evalúa posición y normal
    únicamente sobre la pose reservada.
    """
    unique = np.unique(pose)
    result = {
        "available": False,
        "tests": 0,
        "passed_tests": 0,
        "pass_ratio": 0.0,
        "median_abs_error_mm": np.nan,
        "p90_abs_error_mm": np.nan,
    }
    if len(unique) < 4:
        return result
    # Hasta tres poses representativas, sin reutilizarlas durante su prueba.
    positions = np.linspace(0, len(unique) - 1, min(3, len(unique))).round().astype(int)
    heldout_poses = np.unique(unique[positions])
    records = []
    for heldout in heldout_poses:
        test = pose == heldout
        train = ~test
        fit = _fit_local_quadratic_for_validation(design, z, pose, base_w, train, voxel, floor)
        if fit is None or np.count_nonzero(test) < 2:
            continue
        coeff, sigma = fit
        residual = z[test] - design[test] @ coeff
        abs_residual = np.abs(residual)
        gx = (coeff[1] + 2 * coeff[3] * u[test] + coeff[4] * v[test]) / radius
        gy = (coeff[2] + coeff[4] * u[test] + 2 * coeff[5] * v[test]) / radius
        predicted = normal[None, :] - gx[:, None] * tangent - gy[:, None] * bitangent
        predicted = normalize_rows(predicted)
        normal_dot = np.sum(predicted * observed_normals[test], axis=1)
        median_error = float(np.median(abs_residual))
        p90_error = float(np.percentile(abs_residual, 90))
        median_dot = float(np.median(normal_dot))
        median_limit = max(0.18 * voxel, 2.5 * sigma)
        p90_limit = max(0.35 * voxel, 3.5 * sigma)
        passed = bool(
            median_error <= median_limit and p90_error <= p90_limit and median_dot >= cos_angle
        )
        records.append((median_error, p90_error, median_dot, passed))
    if not records:
        return result
    result["available"] = True
    result["tests"] = len(records)
    result["passed_tests"] = int(sum(1 for item in records if item[3]))
    result["pass_ratio"] = float(result["passed_tests"] / len(records))
    result["median_abs_error_mm"] = float(np.median([item[0] for item in records]))
    result["p90_abs_error_mm"] = float(np.median([item[1] for item in records]))
    result["median_normal_dot"] = float(np.median([item[2] for item in records]))
    return result


def _local_patch(seed_id, ids, context, proposed_shift=None):
    p, normals = context["points"], context["normals"]
    poses, weights = context["poses"], context["weights"]
    args = context["local_args"]
    voxel, radius = args["voxel"], args["radius"]
    seed, normal = p[seed_id], normals[seed_id]
    # Garantiza que la muestra que define la superficie esté en su vecindario.
    ids = np.unique(np.r_[ids, seed_id])
    all_ids = ids.copy()
    all_delta = p[all_ids] - seed
    all_central = np.linalg.norm(all_delta, axis=1) <= 0.95 * voxel
    original_central_poses = np.unique(poses[all_ids[all_central]])
    delta = p[ids] - seed
    dots = normals[ids] @ normal
    oriented = dots >= args["cos_angle"]
    incompatible = int(np.count_nonzero(~oriented))
    ids, delta, dots = ids[oriented], delta[oriented], dots[oriented]
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    tangent = np.cross(normal, axis)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    x, y, z = delta @ tangent, delta @ bitangent, delta @ normal
    tangent_r = np.hypot(x, y)
    # Compensa curvatura aproximada SOLO para separar capas. El ajuste final
    # estima su propia curvatura. Para z=0.5*k*x², z-0.5*grad(z)*x=0.
    gx = -(normals[ids] @ tangent) / np.maximum(dots, 1e-8)
    gy = -(normals[ids] @ bitangent) / np.maximum(dots, 1e-8)
    corrected_height = z - 0.5 * (gx * x + gy * y)
    order = np.argsort(corrected_height, kind="stable")
    sorted_height = corrected_height[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(sorted_height) > 0.35 * voxel) + 1, len(ids)]
    seed_at = int(np.flatnonzero(ids == seed_id)[0])
    group = next(order[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if seed_at in order[a:b])
    outside = np.ones(len(ids), dtype=bool)
    outside[group] = False
    # Un límite adicional impide enlazar cadenas de capas por puntos puente.
    mode = ~outside & (np.abs(corrected_height) <= 0.85 * voxel)
    mode[seed_at] = True
    outside = ~mode
    other_gap = (
        float(np.min(np.abs(corrected_height[outside]))) if np.any(outside) else float("inf")
    )
    incompatible += int(np.count_nonzero(outside))
    mode_count = len(cuts) - 1
    # Comparación entre poses ANTES del IRLS; una escala inflada entre vistas
    # nunca convierte dos capas en una sola. Escala calculada dentro de cada pose.
    candidate_poses = np.unique(poses[ids[mode]])
    centers, scales = {}, {}
    for pid in candidate_poses:
        values = corrected_height[mode & (poses[ids] == pid)]
        centers[int(pid)] = float(np.median(values))
        scales[int(pid)] = max(
            args["sigma_floor"],
            min(0.15 * voxel, float(1.4826 * np.median(np.abs(values - np.median(values))))),
        )
    # Referencia conjunta: una pose equivale a un voto, sin usar la semilla
    # como verdad. Solo dividir si hay dos grupos multivista separados por
    # un intervalo vacío mayor que la dispersión interna de ambas vistas.
    ordered_poses = sorted(candidate_poses, key=lambda pid: (centers[int(pid)], int(pid)))
    partitions = [[]]
    for pid in ordered_poses:
        if partitions[-1]:
            prev = partitions[-1][-1]
            gap = centers[int(pid)] - centers[int(prev)]
            gate = max(0.25 * voxel, 3.0 * np.hypot(scales[int(pid)], scales[int(prev)]))
            if gap > gate:
                partitions.append([])
        partitions[-1].append(int(pid))
    supported_partitions = [g for g in partitions if len(g) >= 2]
    accepted_poses = list(map(int, candidate_poses))
    if len(supported_partitions) >= 2:
        # Mantener superficies reales separadas; cada semilla localiza una capa,
        # pero ninguna pose decide cuáles de las otras son compatibles.
        chosen = min(
            supported_partitions,
            key=lambda g: (abs(float(np.median([centers[k] for k in g]))), -len(g), min(g)),
        )
        center = float(np.median([centers[k] for k in chosen]))
        # Solo exclusión inequívoca; sin reubicar una semilla hacia una capa lejana.
        if abs(center) <= 0.35 * voxel:
            accepted_poses = chosen
    elif len(supported_partitions) == 1 and len(partitions) > 1:
        chosen = supported_partitions[0]
        center = float(np.median([centers[k] for k in chosen]))
        # Una pose aislada se excluye únicamente frente a al menos tres vistas
        # coincidentes. Vecindarios ambiguos conservan el IRLS y sus guardas.
        if len(chosen) >= 3 and abs(center) <= 0.35 * voxel:
            accepted_poses = chosen
    reference = float(np.median([centers[int(k)] for k in accepted_poses]))
    pose_conflicts = np.setdiff1d(candidate_poses, accepted_poses)
    pose_rejected = mode & ~np.isin(poses[ids], accepted_poses)
    incompatible += int(np.count_nonzero(pose_rejected))
    mode &= ~pose_rejected
    mode_count = max(mode_count, len(supported_partitions))
    if np.any(pose_rejected):
        other_gap = min(other_gap, float(np.min(np.abs(corrected_height[pose_rejected]))))
    ids, x, y, z, tangent_r = (arr[mode] for arr in (ids, x, y, z, tangent_r))
    pose = poses[ids]
    u, v = x / radius, y / radius
    design = np.column_stack((np.ones(len(ids)), u, v, u * u, u * v, v * v))
    raw_w = weights[ids] * np.exp(-0.5 * (tangent_r / (0.60 * radius)) ** 2)
    sums = np.bincount(pose, weights=raw_w, minlength=25)
    peak = np.zeros(25)
    np.maximum.at(peak, pose, weights[ids])
    base_w = raw_w / np.maximum(sums[pose], 1e-12) * peak[pose]
    floor = args["sigma_floor"]
    coeff = np.zeros(6)
    heldout_validation = {
        "available": False,
        "tests": 0,
        "passed_tests": 0,
        "pass_ratio": 0.0,
        "median_abs_error_mm": np.nan,
        "p90_abs_error_mm": np.nan,
    }
    # Posición observada como fallback; no media entre capas ni relleno.
    status = 2
    can_fit = len(ids) >= 12 and len(np.unique(pose)) >= 3
    if can_fit:
        angles = np.sort(np.arctan2(y[tangent_r > 0.12 * voxel], x[tangent_r > 0.12 * voxel]))
        max_gap = (
            np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi])) if len(angles) >= 6 else 2 * np.pi
        )
        xy = np.column_stack((u, v))
        covariance = (xy.T * base_w) @ xy / max(base_w.sum(), 1e-12)
        eigen = np.linalg.eigvalsh(covariance)
        can_fit = max_gap < np.pi + 0.10 and eigen[0] > 0.04 * max(eigen[-1], 1e-12)
    sigma = floor
    if can_fit:
        robust_w = base_w.copy()
        status = 1
        for iteration in range(4):
            a = design * np.sqrt(robust_w)[:, None]
            target = z * np.sqrt(robust_w)
            try:
                coeff, _, rank, singular = np.linalg.lstsq(a, target, rcond=1e-8)
            except np.linalg.LinAlgError:
                status = 3
                break
            if rank < 6 or singular[-1] <= 0 or singular[0] / singular[-1] > 5000:
                status = 3
                break
            residual = z - design @ coeff
            unique_pose = np.unique(pose)
            by_pose = np.zeros(25)
            for pid in unique_pose:
                by_pose[pid] = np.median(residual[pose == pid])
            medians = by_pose[unique_pose]
            within = residual - by_pose[pose]
            within_sigma = 1.4826 * np.median(np.abs(within))
            between_sigma = 1.4826 * np.median(np.abs(medians - np.median(medians)))
            sigma = float(np.sqrt(floor**2 + within_sigma**2 + between_sigma**2))
            # Escala acotada: una nube de varias capas no legitima un salto grande.
            scale = min(sigma, 0.30 * voxel)
            robust_w = base_w * np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-12))
        if not np.all(np.isfinite(coeff)) or sigma > 0.45 * voxel:
            status = 4
    if status != 1:
        coeff = np.zeros(6)
        # Tendencia de curvatura por normales para evaluar evidencia de fallback.
        gx_mode = -(normals[ids] @ tangent) / np.maximum(normals[ids] @ normal, 1e-8)
        gy_mode = -(normals[ids] @ bitangent) / np.maximum(normals[ids] @ normal, 1e-8)
        residual = z - 0.5 * (gx_mode * x + gy_mode * y)
        sigma = max(floor, min(0.30 * voxel, float(1.4826 * np.median(np.abs(residual)))))
    else:
        residual = z - design @ coeff
    if status == 1:
        heldout_validation = _cross_validate_local_quadratic(
            design,
            z,
            u,
            v,
            pose,
            base_w,
            normals[ids],
            normal,
            tangent,
            bitangent,
            radius,
            voxel,
            floor,
            args["cos_angle"],
        )
        if heldout_validation["available"] and not _heldout_vote_ok(
            heldout_validation["passed_tests"],
            heldout_validation["tests"],
            args.get("minimum_heldout_pass_numerator", 2),
            args.get("minimum_heldout_pass_denominator", 3),
        ):
            status = 4

    if status == 1:
        predicted = (
            normal[None, :]
            - ((coeff[1] + 2 * coeff[3] * u + coeff[4] * v) / radius)[:, None] * tangent
            - ((coeff[2] + coeff[4] * u + 2 * coeff[5] * v) / radius)[:, None] * bitangent
        )
        predicted = normalize_rows(predicted)
        angular_ok = np.sum(predicted * normals[ids], axis=1) >= args["cos_angle"]
        # Vistas con posiciones compatibles pero normales incompatibles no votan.
        # La máscara se aplica al respaldo sin contaminar métricas con infinitos.
    else:
        angular_ok = np.ones(len(ids), dtype=bool)
    tolerance = min(0.65 * voxel, max(0.16 * voxel, 2.5 * sigma))
    inlier = (np.abs(residual) <= tolerance) & angular_ok
    central = tangent_r <= 0.95 * voxel
    # Soporte al punto de salida: poses próximas, no todas las del ajuste amplio.
    supported = central & inlier
    support_ids = np.unique(pose[supported])
    potential_ids = np.unique(pose[central])
    unsupported_ids = np.setdiff1d(potential_ids, support_ids)
    agreement = len(support_ids) / max(len(potential_ids), 1)
    if len(support_ids) < 2:
        status = 0
    position = seed.copy()
    result_normal = normal.copy()
    shift = 0.0
    # Las poses que contradicen el ajuste no se recuperan por conteo de vecinos.
    if status == 1 and (agreement < 0.60 or len(np.unique(pose[inlier])) < 3):
        status = 4
    budget = min(args["max_shift"], 2.0 * sigma, 0.45 * other_gap)
    if status == 1:
        result_normal = normal - (coeff[1] / radius) * tangent - (coeff[2] / radius) * bitangent
        result_normal /= max(np.linalg.norm(result_normal), 1e-12)
        if result_normal @ normal < args["cos_angle"]:
            status = 5
            result_normal = normal.copy()
        else:
            shift = float(
                np.clip(coeff[0] if proposed_shift is None else proposed_shift, -budget, budget)
            )
            position += shift * normal
            # La evidencia se vuelve a medir en la posición realmente limitada.
            residual = z - (design @ coeff - coeff[0] + shift)
            inlier = (np.abs(residual) <= tolerance) & angular_ok
            supported = central & inlier
            support_ids = np.unique(pose[supported])
            unsupported_ids = np.setdiff1d(potential_ids, support_ids)
            agreement = len(support_ids) / max(len(potential_ids), 1)
            if len(support_ids) < 2 or agreement < 0.60:
                # Revertir la proyección y recomputar métricas en la observación.
                status = 4
                position = seed.copy()
                result_normal = normal.copy()
                shift = 0.0
    if status != 1:
        gx_mode = -(normals[ids] @ tangent) / np.maximum(normals[ids] @ normal, 1e-8)
        gy_mode = -(normals[ids] @ bitangent) / np.maximum(normals[ids] @ normal, 1e-8)
        residual = z - 0.5 * (gx_mode * x + gy_mode * y)
        inlier = (np.abs(residual) <= tolerance) & angular_ok
        supported = central & inlier
        support_ids = np.unique(pose[supported])
        unsupported_ids = np.setdiff1d(potential_ids, support_ids)
        agreement = len(support_ids) / max(len(potential_ids), 1)
    if len(support_ids) < 2:
        status = 0
    elif status == 0:
        status = 4
    evidence_w = base_w * supported
    total = float(evidence_w.sum())
    if total > 0:
        color = np.sum(context["colors"][ids] * evidence_w[:, None], axis=0) / total
        normal_consistency = float(
            np.linalg.norm(np.sum(normals[ids] * evidence_w[:, None], axis=0) / total)
        )
        spread = float(np.sqrt(np.sum(evidence_w * residual**2) / total))
        incidence = float(np.mean([peak[pid] for pid in support_ids]))
    else:
        color = context["colors"][seed_id].copy()
        normal_consistency = 0.0
        spread = voxel
        incidence = 0.0
    confidence = (
        0.48 * min(len(support_ids) / 4.0, 1.0)
        + 0.20 * np.exp(-0.5 * (spread / args["spread_reference"]) ** 2)
        + 0.17 * normal_consistency
        + 0.15 * incidence
    ) * (0.60 + 0.40 * agreement)
    # Modelo válido en la posición REAL (incluido el recorte por incertidumbre).
    model_coeff = coeff.copy() if status == 1 else np.zeros(6)
    model_coeff[0] = shift
    pose_residual = np.full(25, np.nan)
    for pid in np.unique(pose):
        pose_residual[pid] = float(np.median(residual[pose == pid]))
    # Guardar también contradicciones excluidas ANTES del ajuste.
    # No cuentan como apoyo aunque procedan de observaciones de alta confianza.
    diagnostic_conflicts = np.union1d(
        unsupported_ids, np.setdiff1d(original_central_poses, support_ids)
    )
    for pid in candidate_poses:
        if not np.isfinite(pose_residual[int(pid)]):
            pose_residual[int(pid)] = centers[int(pid)] - reference
    independent_support, angular_span = _pose_diversity(
        support_ids,
        expected=args.get("expected_views", 25),
        minimum_separation=args.get("minimum_independent_pose_separation", 2),
    )
    conflict_pose_count = int(len(np.unique(diagnostic_conflicts)))
    conflict_ratio = float(conflict_pose_count / max(len(support_ids) + conflict_pose_count, 1))
    # Valores compatibles con el pipeline; diagnóstico por candidato aparte.
    return dict(
        points=position,
        colors=color,
        normals=result_normal,
        support=len(support_ids),
        confidence=np.clip(confidence, 0.0, 1.0),
        spread=spread,
        normal_consistency=normal_consistency,
        uncertainty=sigma,
        displacement=abs(shift),
        status=status,
        support_pose_mask=_pose_mask(support_ids),
        disagreement_pose_mask=_pose_mask(diagnostic_conflicts),
        agreement=agreement,
        incompatible=incompatible,
        observations=len(ids),
        modes=mode_count,
        displacement_budget=budget,
        independent_support_poses=independent_support,
        support_angular_span_poses=angular_span,
        conflict_pose_count=conflict_pose_count,
        conflict_pose_ratio=conflict_ratio,
        heldout_validation_available=int(bool(heldout_validation["available"])),
        heldout_validation_tests=int(heldout_validation["tests"]),
        heldout_validation_pass_count=int(heldout_validation["passed_tests"]),
        heldout_validation_pass_ratio=float(heldout_validation["pass_ratio"]),
        heldout_validation_error_mm=float(heldout_validation["median_abs_error_mm"]),
        heldout_validation_p90_mm=float(heldout_validation["p90_abs_error_mm"]),
        model_origin=seed.copy(),
        model_normal=normal.copy(),
        model_tangent=tangent,
        model_bitangent=bitangent,
        model_coeff=model_coeff,
        pose_residual=pose_residual,
    )


# V9: parches solapados, conciliación simultánea e informe por pose.
_LOCAL_SHAPES = {
    "points": 3,
    "colors": 3,
    "normals": 3,
    "model_origin": 3,
    "model_normal": 3,
    "model_tangent": 3,
    "model_bitangent": 3,
    "model_coeff": 6,
    "pose_residual": 25,
}


def _allocate_local(count, dtypes):
    """Reserva los arrays de salida para un bloque de fusión local."""
    return {
        name: np.zeros(
            (count, _LOCAL_SHAPES[name]) if name in _LOCAL_SHAPES else count, dtype=dtype
        )
        for name, dtype in dtypes.items()
    }


def _observations_for_queries(context, query):
    if "_pose_trees" not in context:
        context["_pose_trees"] = []
        for pose in range(25):
            idx = np.flatnonzero(context["poses"] == pose)
            context["_pose_trees"].append(
                (idx, cKDTree(context["points"][idx]) if len(idx) else None)
            )
    neighbors = [[] for _ in query]
    for idx, tree in context["_pose_trees"]:
        if tree is None:
            continue
        distances, local_ids = tree.query(
            query,
            k=context["local_args"]["neighbors_per_pose"],
            distance_upper_bound=context["local_args"]["radius"],
            workers=1,
        )
        for j in range(len(query)):
            valid = np.isfinite(distances[j]) & (local_ids[j] < len(idx))
            neighbors[j].extend(idx[local_ids[j][valid]].tolist())
    return neighbors


def _weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    c = np.cumsum(weights[order])
    return float(values[order[min(int(np.searchsorted(c, 0.5 * c[-1])), len(order) - 1)]])


def _regional_worker(task):
    """Procesa un bloque de conciliación regional con el contexto del trabajador."""
    from utilidades_rendimiento import contexto

    context = contexto()
    base = context["regional_baseline"]
    start, stop = task
    result = {k: v[start:stop].copy() for k, v in base.items()}
    if "_model_tree" not in context:
        context["_model_tree"] = cKDTree(base["model_origin"])
    radius = context["local_args"]["radius"]
    voxel = context["local_args"]["voxel"]
    query = base["model_origin"][start:stop]
    distances, near = context["_model_tree"].query(
        query, k=min(48, len(base["points"])), distance_upper_bound=0.85 * radius, workers=1
    )
    distances = np.asarray(distances).reshape(len(query), -1)
    near = np.asarray(near).reshape(len(query), -1)
    proposals = []
    for j, i in enumerate(range(start, stop)):
        if base["status"][i] != 1:
            result["regional_status"][j] = 1
            continue
        candidate = near[j][np.isfinite(distances[j])]
        candidate = candidate[(candidate != i) & (base["status"][candidate] == 1)]
        candidate = candidate[
            (base["normals"][candidate] @ base["normals"][i]) >= np.cos(np.deg2rad(25.0))
        ]
        # Soporte común REAL, nunca sumar las poses de parches vecinos.
        mask = int(base["support_pose_mask"][i])
        candidate = np.array(
            [
                int(k)
                for k in candidate
                if (int(base["support_pose_mask"][k]) & mask).bit_count() >= 2
            ],
            dtype=int,
        )
        if len(candidate) < 3:
            result["regional_status"][j] = 2
            continue
        delta = base["model_origin"][i] - base["model_origin"][candidate]
        u = np.einsum("ij,ij->i", delta, base["model_tangent"][candidate]) / radius
        v = np.einsum("ij,ij->i", delta, base["model_bitangent"][candidate]) / radius
        z = np.einsum("ij,ij->i", delta, base["model_normal"][candidate])
        coeff = base["model_coeff"][candidate]
        A = np.column_stack((np.ones(len(u)), u, v, u * u, u * v, v * v))
        dzdu = (coeff[:, 1] + 2 * coeff[:, 3] * u + coeff[:, 4] * v) / radius
        dzdv = (coeff[:, 2] + coeff[:, 4] * u + 2 * coeff[:, 5] * v) / radius
        gradient = (
            base["model_normal"][candidate]
            - dzdu[:, None] * base["model_tangent"][candidate]
            - dzdv[:, None] * base["model_bitangent"][candidate]
        )
        denom = gradient @ base["model_normal"][i]
        predicted = (np.einsum("ij,ij->i", A, coeff) - z) / np.maximum(denom, 0.1)
        own = float(np.dot(base["points"][i] - base["model_origin"][i], base["model_normal"][i]))
        sigma = np.sqrt(base["uncertainty"][candidate] ** 2 + base["uncertainty"][i] ** 2)
        # No mezclar láminas: diferencia normal y distancia al dominio del ajuste.
        valid = (
            (denom > 0.70)
            & (np.hypot(u, v) < 0.85)
            & (abs(predicted - own) <= np.minimum(0.40 * voxel, 1.5 * sigma))
        )
        candidate, u, v, predicted = candidate[valid], u[valid], v[valid], predicted[valid]
        if len(candidate) < 3:
            result["regional_status"][j] = 3
            continue
        rel = base["model_origin"][candidate] - base["model_origin"][i]
        xx = rel @ base["model_tangent"][i]
        yy = rel @ base["model_bitangent"][i]
        angles = np.sort(np.arctan2(yy, xx))
        gap = np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi]))
        if gap > np.pi + 0.10:
            result["regional_status"][j] = 2
            continue
        weights = (
            base["confidence"][candidate]
            * np.exp(-2 * (u * u + v * v))
            / np.maximum(base["uncertainty"][candidate] ** 2, (0.10 * voxel) ** 2)
        )
        target = _weighted_median(predicted, weights)
        deviation = _weighted_median(abs(predicted - target), weights) * 1.4826
        result["regional_disagreement_mm"][j] = deviation
        result["regional_neighbors"][j] = len(candidate)
        if deviation > max(0.12 * voxel, base["uncertainty"][i]):
            result["regional_status"][j] = 3
            continue
        # Una sola pasada de Jacobi; no propagar iterativamente ni acumular deriva.
        target = own + 0.65 * (target - own)
        target = float(
            np.clip(target, -base["displacement_budget"][i], base["displacement_budget"][i])
        )
        if abs(target - own) < 0.01 * voxel:
            result["regional_status"][j] = 4
            continue
        proposals.append((j, i, target))
    if proposals:
        obs = _observations_for_queries(
            context, np.asarray([base["model_origin"][i] for _, i, _ in proposals])
        )
        for (j, i, target), ids in zip(proposals, obs):
            row = _local_patch(
                int(context["seed_ids"][i]),
                np.asarray(ids, dtype=np.int64),
                context,
                proposed_shift=target,
            )
            # Reevaluación por pose sobre observaciones, no sobre interpolaciones.
            oldmask = int(base["support_pose_mask"][i])
            newmask = int(row["support_pose_mask"])
            accepted = (
                row["status"] == 1
                and (newmask & oldmask) == oldmask
                and row["agreement"] >= base["agreement"][i] - 0.02
                and row["spread"] <= base["spread"][i] + 0.05 * voxel
                and row["confidence"] >= base["confidence"][i] - 0.02
            )
            if not accepted:
                result["regional_status"][j] = 5
                continue
            for name in row:
                result[name][j] = row[name]
            result["regional_status"][j] = 6
            result["regional_shift_mm"][j] = np.linalg.norm(row["points"] - base["points"][i])
    return start, stop, result


def _regional_report(result, region_ids):
    states = {
        0: "not_evaluated",
        1: "original_observation_or_unreliable_fit",
        2: "insufficient_overlap",
        3: "incompatible_or_disagreeing_models",
        4: "already_consistent",
        5: "rejected_by_observations",
        6: "reconciled",
    }
    records = []
    for region in np.unique(region_ids):
        take = region_ids == region
        records.append(
            {
                "region": int(region),
                "candidates": int(take.sum()),
                "reconciled": int(np.count_nonzero(result["regional_status"][take] == 6)),
                "unresolved": int(
                    np.count_nonzero(np.isin(result["regional_status"][take], [1, 2, 3, 5]))
                ),
                "bounds_registered_P00_mm": [
                    result["model_origin"][take].min(axis=0).tolist(),
                    result["model_origin"][take].max(axis=0).tolist(),
                ],
                "disagreement_mm": finite_stats(result["regional_disagreement_mm"][take]),
                "support": finite_stats(result["support"][take]),
                "confidence": finite_stats(result["confidence"][take]),
            }
        )
    pose_report = []
    for pose in range(25):
        residual = result["pose_residual"][:, pose]
        valid = np.isfinite(residual)
        pose_report.append(
            {
                "pose": pose,
                "patches": int(valid.sum()),
                "regions": int(len(np.unique(region_ids[valid]))),
                "signed_normal_residual_mm": finite_stats(residual[valid]),
                "absolute_normal_residual_mm": finite_stats(abs(residual[valid])),
            }
        )
    return {
        "enabled": True,
        "method": "overlapping_quadratic_models_synchronous_reconciliation",
        "status_codes": states,
        "status_counts": {
            v: int(np.count_nonzero(result["regional_status"] == k)) for k, v in states.items()
        },
        "regions": records,
        "per_pose": pose_report,
        "pose_reoptimization": False,
        "observational_support_recomputed": True,
        "additional_points": 0,
        "note": "Residuals diagnose disagreement along local normals; they are not pose corrections or proof of registration error.",
    }


def _local_fusion_worker(task):
    """Procesa un bloque de ajuste superficial local con contexto compartido de entrada."""
    from utilidades_rendimiento import contexto

    context = contexto()
    start, stop = task
    seeds = context["seed_ids"][start:stop]
    neighbors = _observations_for_queries(context, context["points"][seeds])
    result = _allocate_local(len(seeds), context["output_dtypes"])
    for j, seed in enumerate(seeds):
        row = _local_patch(int(seed), np.asarray(neighbors[j], dtype=np.int64), context)
        for name in row:
            result[name][j] = row[name]
    return start, stop, result


def _spatial_region_order(seed_points, maximum=512):
    """Partición adaptativa solo para trabajo/diagnóstico; halos sin recorte."""
    pending = [np.arange(len(seed_points))]
    leaves = []
    while pending:
        ids = pending.pop()
        if len(ids) <= maximum:
            leaves.append(ids)
            continue
        axis = int(np.argmax(np.ptp(seed_points[ids], axis=0)))
        order = ids[np.argsort(seed_points[ids, axis], kind="stable")]
        half = len(order) // 2
        pending.extend([order[half:], order[:half]])
    order = np.concatenate(leaves) if leaves else np.empty(0, dtype=int)
    regions = (
        np.concatenate([np.full(len(ids), i, dtype=np.int32) for i, ids in enumerate(leaves)])
        if leaves
        else np.empty(0, dtype=np.int32)
    )
    return order, regions


def _geometry_predict(model, q):
    kind, origin, basis, coef = model
    if kind == 2:
        center, axis, radius = coef[:3], coef[3:6], coef[6]
        d = q - center
        radial = d - (d @ axis)[:, None] * axis
        length = np.linalg.norm(radial, axis=1)
        n = radial / np.maximum(length[:, None], 1e-12)
        return length - radius, n
    d = q - origin
    x, y, z = (d @ basis).T
    if kind == 1:
        return z, np.tile(basis[:, 2], (len(q), 1))
    h = coef[6]
    x = x / h
    y = y / h
    c = coef[:6]
    A = np.column_stack((np.ones(len(q)), x, y, x * x, x * y, y * y))
    gx = (c[1] + 2 * c[3] * x + c[4] * y) / h
    gy = (c[2] + c[4] * x + 2 * c[5] * y) / h
    n = basis[:, 2][None, :] - gx[:, None] * basis[:, 0] - gy[:, None] * basis[:, 1]
    length = np.linalg.norm(n, axis=1)
    return (z - A @ c) / length, n / length[:, None]


def _geometry_fit(kind, q, n, poses, weights, h):
    # Entrenamiento exclusivamente con las poses recibidas por el llamador.
    _, inv, count = np.unique(poses, return_inverse=True, return_counts=True)
    w = np.maximum(weights, 1e-6) / count[inv]
    w /= w.sum()
    origin = np.sum(q * w[:, None], axis=0)
    centered = q - origin
    eig, basis = np.linalg.eigh((centered * w[:, None]).T @ centered)
    if eig[1] < 0.02 * h * h:
        return None
    basis = basis[:, [2, 1, 0]]
    if np.sum((n @ basis[:, 2]) * w) < 0:
        basis[:, 2] *= -1
    if kind == 1:
        for _ in range(5):
            residual = (q - origin) @ basis[:, 2]
            sigma = max(0.08 * h, 1.4826 * np.median(np.abs(residual - np.median(residual))))
            rw = w * np.minimum(1.0, 1.5 * sigma / np.maximum(np.abs(residual), 1e-12))
            rw /= rw.sum()
            origin = np.sum(q * rw[:, None], axis=0)
            centered = q - origin
            _, axes = np.linalg.eigh((centered * rw[:, None]).T @ centered)
            basis = axes[:, [2, 1, 0]]
            if np.sum((n @ basis[:, 2]) * w) < 0:
                basis[:, 2] *= -1
        return kind, origin, basis, np.zeros(7)
    if kind == 3:
        x, y, z = ((q - origin) @ basis).T
        x /= h
        y /= h
        A = np.column_stack((np.ones(len(q)), x, y, x * x, x * y, y * y))
        rw = w.copy()
        for _ in range(5):
            c, _, rank, singular = np.linalg.lstsq(
                A * np.sqrt(rw)[:, None], z * np.sqrt(rw), rcond=1e-7
            )
            if rank < 6 or singular[0] / max(singular[-1], 1e-30) > 1e5:
                return None
            residual = z - A @ c
            sigma = max(0.08 * h, 1.4826 * np.median(np.abs(residual - np.median(residual))))
            rw = w * np.minimum(1.0, 1.5 * sigma / np.maximum(np.abs(residual), 1e-12))
        return kind, origin, basis, np.r_[c, h]
    from scipy.optimize import least_squares

    ne, axes = np.linalg.eigh((n * w[:, None]).T @ n)
    if ne[1] < 0.015 * max(ne[2], 1e-12):
        return None  # radio no identificable sobre un plano
    axis = axes[:, 0]
    u = axes[:, 2]
    v = np.cross(axis, u)
    xy = np.column_stack((centered @ u, centered @ v))
    A = np.column_stack((2 * xy, np.ones(len(q))))
    c, _, rank, _ = np.linalg.lstsq(
        A * np.sqrt(w)[:, None], np.sum(xy * xy, axis=1) * np.sqrt(w), rcond=1e-7
    )
    if rank < 3:
        return None
    radius = np.sqrt(max(c[2] + c[0] ** 2 + c[1] ** 2, 1e-12))
    center = c[0] * u + c[1] * v
    if not 0.5 * h < radius < 200 * h:
        return None
    theta = np.arctan2(axis[1], axis[0])
    phi = np.arcsin(np.clip(axis[2], -0.999999, 0.999999))

    def unpack(parameters):
        th, ph = parameters[3:5]
        ax = np.array([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)])
        return parameters[:3], ax, np.exp(parameters[5])

    def residual(parameters):
        ce, ax, r = unpack(parameters)
        d = centered - ce
        radial = d - (d @ ax)[:, None] * ax
        return np.r_[
            np.sqrt(w * len(q)) * (np.linalg.norm(radial, axis=1) - r) / h,
            0.3 * np.sqrt(w * len(q)) * (n @ ax),
            ce @ ax / h,
        ]

    fit = least_squares(
        residual,
        np.r_[center, theta, phi, np.log(radius)],
        bounds=(
            np.r_[[-np.inf] * 3, -np.inf, -np.pi / 2, np.log(0.5 * h)],
            np.r_[[np.inf] * 3, np.inf, np.pi / 2, np.log(200 * h)],
        ),
        loss="soft_l1",
        f_scale=0.3,
        max_nfev=70,
    )
    if not fit.success or not np.all(np.isfinite(fit.x)):
        return None
    center, axis, radius = unpack(fit.x)
    d = centered - center
    radial = d - (d @ axis)[:, None] * axis
    rn = radial / np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-12)
    # Se necesita variación angular y longitud axial para distinguir cilindro.
    if np.linalg.norm(np.sum(rn * w[:, None], axis=0)) > 0.995 or np.ptp(d @ axis) < 3 * h:
        return None
    return kind, origin, basis, np.r_[origin + center, axis, radius]


GEOMETRY_REASONS = {
    0: "accepted",
    1: "insufficient_same_surface_poses",
    2: "insufficient_same_surface_observations",
    3: "fit_not_identifiable",
    4: "held_out_pose_position_normal_or_coverage_disagreement",
    5: "unstable_cross_validation_projection",
    6: "unstable_cylinder_axis_or_radius",
    7: "ambiguous_model_choice",
    8: "no_valid_model",
    9: "numerical_fit_failure",
}


def _surface_patch_ids(base, h):
    """Parches acotados sobre grafo geométrico, separados de bloques de CPU."""
    q = base["points"]
    n = base["normals"]
    count = len(q)
    n = n / np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-12)
    d, neighbors = cKDTree(q).query(q, k=min(17, count))
    d = np.reshape(d, (count, -1))
    neighbors = np.reshape(neighbors, (count, -1))
    delta = q[neighbors] - q[:, None, :]
    dots = np.sum(n[:, None, :] * n[neighbors], axis=2)
    # En una pared curva, las alturas en normales extremas tienen signo opuesto.
    layer = np.abs(np.sum(delta * (n[:, None, :] + n[neighbors]), axis=2)) * 0.5
    sigma = np.minimum(np.nan_to_num(base["uncertainty"], nan=0.0, posinf=h), 0.15 * h)
    tolerance = np.minimum(0.30 * h, 0.08 * h + np.hypot(sigma[:, None], sigma[neighbors]))
    valid = (d <= 2.5 * h) & (dots >= np.cos(np.deg2rad(25))) & (layer <= tolerance)
    labels = np.full(count, -1, np.int32)
    patch = 0
    from collections import deque

    for seed in np.argsort(-base["support"], kind="stable"):
        if labels[seed] >= 0:
            continue
        labels[seed] = patch
        queue = deque([int(seed)])
        size = 1
        while queue:
            i = queue.popleft()
            for j in neighbors[i][valid[i]]:
                if labels[j] >= 0 or size >= 384:
                    continue
                if np.linalg.norm(q[j] - q[seed]) > 8 * h:
                    continue
                if n[j] @ n[seed] < np.cos(np.deg2rad(65)):
                    continue
                labels[j] = patch
                queue.append(int(j))
                size += 1
        patch += 1
    return labels


def _same_surface_observations(context, selected, ids, h):
    base = context["geometry_baseline"]
    q = base["points"][selected]
    n = base["normals"][selected]
    distance, nearest = cKDTree(q).query(context["points"][ids])
    delta = context["points"][ids] - q[nearest]
    raw_n = context["normals"][ids]
    ref_n = n[nearest]
    dot = np.sum(raw_n * ref_n, axis=1)
    height = np.abs(np.sum(delta * (raw_n + ref_n), axis=1)) * 0.5
    sigma = np.minimum(
        np.nan_to_num(base["uncertainty"][selected[nearest]], nan=0.0, posinf=h), 0.15 * h
    )
    keep = (
        (distance <= 2 * h)
        & (dot >= np.cos(np.deg2rad(25)))
        & (height <= np.minimum(0.30 * h, 0.10 * h + sigma))
    )
    return ids[keep]


def _adaptive_pose_folds(poses):
    """3–5 poses: dejar una fuera; >=6: tres grupos de poses completas."""
    unique = np.unique(poses)
    if len(unique) < 3:
        return np.zeros(len(poses), dtype=int), 0
    folds = len(unique) if len(unique) < 6 else 3
    return np.searchsorted(unique, poses) % folds, folds


def _expand_surface_patch(context, selected, h):
    """Halo de lectura conectado; no fusiona etiquetas ni inventa puntos."""
    if len(selected) >= 96:
        return selected
    base = context["geometry_baseline"]
    q = base["points"]
    n = base["normals"]
    if "_patch_halo_tree" not in context:
        context["_patch_halo_tree"] = cKDTree(q)
    tree = context["_patch_halo_tree"]
    origin = np.mean(q[selected], axis=0)
    chosen = set(map(int, selected))
    frontier = np.asarray(selected, int)
    for _ in range(3):
        d, j = tree.query(q[frontier], k=min(17, len(q)))
        d = np.reshape(d, (len(frontier), -1))
        j = np.reshape(j, (len(frontier), -1))
        added = []
        for row, i in enumerate(frontier):
            for distance, v in zip(d[row], j[row]):
                v = int(v)
                if v in chosen or distance > 2.5 * h or len(chosen) >= 256:
                    continue
                if np.linalg.norm(q[v] - origin) > 6 * h:
                    continue
                if n[i] @ n[v] < np.cos(np.deg2rad(25)):
                    continue
                # Misma tolerancia entre capas que al formar los parches.
                sigma = np.clip(
                    np.nan_to_num(base["uncertainty"][[i, v]], nan=0.0, posinf=h), 0, 0.15 * h
                )
                gate = min(0.30 * h, 0.08 * h + np.hypot(*sigma))
                if abs((q[v] - q[i]) @ (n[i] + n[v])) * 0.5 > gate:
                    continue
                chosen.add(v)
                added.append(v)
        if not added:
            break
        frontier = np.asarray(added, int)
    return np.asarray(sorted(chosen), int)


def _geometry_region(context, region):
    cache = context.setdefault("_geometry_cache", {})
    if region in cache:
        return cache[region]
    base = context["geometry_baseline"]
    h = context["local_args"]["voxel"]
    selected = np.flatnonzero(base["geometry_patch_id"] == region)
    original_count = len(selected)
    selected = _expand_surface_patch(context, selected, h)
    context.setdefault("_geometry_added", {})[region] = len(selected) - original_count
    center = np.mean(base["model_origin"][selected], axis=0)
    radius = np.max(np.linalg.norm(base["model_origin"][selected] - center, axis=1)) + 3 * h
    if "_geometry_raw_tree" not in context:
        context["_geometry_raw_tree"] = cKDTree(context["points"])
    ids = np.asarray(context["_geometry_raw_tree"].query_ball_point(center, radius), dtype=int)
    raw_count = len(ids)
    ids = _same_surface_observations(context, selected, ids, h)
    context.setdefault("_geometry_filtered", {})[region] = raw_count - len(ids)
    # Muestreo determinista equilibrado: ninguna pose domina por densidad.
    sampled = []
    for pose in np.unique(context["poses"][ids]):
        own = ids[context["poses"][ids] == pose]
        own = own[
            np.argsort(np.linalg.norm(context["points"][own] - center, axis=1), kind="stable")
        ]
        sampled.extend(own[np.linspace(0, len(own) - 1, min(48, len(own))).astype(int)])
    ids = np.asarray(sampled, int)
    poses = context["poses"][ids]
    unique = np.unique(poses)
    unknown = (None, 0.0, 0.0, 0, 0, 0)
    reasons = context.setdefault("_geometry_reasons", {}).setdefault(region, [0, 0, 0])
    labels, fold_count = _adaptive_pose_folds(poses)
    context.setdefault("_geometry_folds", {})[region] = fold_count
    context.setdefault("_geometry_pose_count", {})[region] = len(unique)
    if fold_count == 0 or len(ids) < 60:
        reasons[:] = [1 if fold_count == 0 else 2] * 3
        cache[region] = unknown
        return unknown
    q = context["points"][ids]
    n = context["normals"][ids]
    w = context["weights"][ids]
    contenders = []
    for kind in (1, 2, 3):
        costs = []
        errors = []
        fits = []
        valid = True
        for fold in range(fold_count):
            test = labels == fold
            train = ~test
            if np.count_nonzero(train) < 30 or np.count_nonzero(test) < 12:
                reasons[kind - 1] = 2
                valid = False
                break
            try:
                model = _geometry_fit(kind, q[train], n[train], poses[train], w[train], h)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                reasons[kind - 1] = 9
                valid = False
                break
            if model is None:
                reasons[kind - 1] = 3
                valid = False
                break
            train_residual, _ = _geometry_predict(model, q[train])
            train_sigma = max(
                0.08 * h,
                1.4826 * float(np.median(np.abs(train_residual - np.median(train_residual)))),
            )
            position_gate = min(0.8 * h, max(0.25 * h, 2.5 * train_sigma))
            residual, predicted = _geometry_predict(model, q[test])
            angles = np.degrees(
                np.arccos(np.clip(np.abs(np.sum(predicted * n[test], axis=1)), 0, 1))
            )
            # Cobertura real de las poses de validación dentro del soporte entrenado.
            train_xy = (q[train] - model[1]) @ model[2][:, :2]
            test_xy = (q[test] - model[1]) @ model[2][:, :2]
            lo = np.min(train_xy, axis=0) - h
            hi = np.max(train_xy, axis=0) + h
            covered = np.all((test_xy >= lo) & (test_xy <= hi), axis=1)
            nearest, _ = cKDTree(q[train]).query(q[test])
            covered &= nearest <= 3 * h
            perpose = []
            for pose in np.unique(poses[test]):
                own = poses[test] == pose
                agreement = (
                    (np.abs(residual[own]) <= position_gate) & (angles[own] <= 30) & covered[own]
                )
                if np.mean(agreement) < 0.65:
                    reasons[kind - 1] = 4
                    valid = False
                    break
                perpose.append(
                    float(np.median(np.abs(residual[own])) / h + 0.25 * np.median(angles[own]) / 30)
                )
            if not valid:
                break
            costs.append(float(np.mean(perpose) + 0.15 * np.percentile(np.abs(residual), 90) / h))
            errors.extend(np.abs(residual).tolist())
            fits.append(model)
        if not valid or len(fits) != fold_count:
            continue
        try:
            full = _geometry_fit(kind, q, n, poses, w, h)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            reasons[kind - 1] = 9
            continue
        if full is None:
            reasons[kind - 1] = 3
            continue
        probe = base["model_origin"][selected[:: max(1, len(selected) // 24)]]
        # Distancias firmadas pueden invertir signo en el cilindro: comparar
        # proyecciones 3D en vez de signos de normales.
        projections = []
        for m in fits:
            r, nn = _geometry_predict(m, probe)
            projections.append(probe - r[:, None] * nn)
        stability = float(np.max(np.linalg.norm(np.asarray(projections) - projections[0], axis=2)))
        if stability > 0.4 * h:
            reasons[kind - 1] = 5
            continue
        if kind == 2:
            radii = np.array([m[3][6] for m in fits])
            axes = np.array([m[3][3:6] for m in fits])
            if np.ptp(radii) > 0.20 * np.median(radii) or np.min(np.abs(axes @ axes[0])) < np.cos(
                np.deg2rad(12)
            ):
                reasons[kind - 1] = 6
                continue
        sigma = max(0.08 * h, float(np.median(errors)), stability * 0.5)
        score = float(np.mean(costs) + {1: 0.0, 2: 0.03, 3: 0.06}[kind])
        contenders.append((score, full, sigma))
    if not contenders:
        cache[region] = unknown
        return unknown
    contenders.sort(key=lambda item: item[0])
    score, model, sigma = contenders[0]
    if len(contenders) > 1 and contenders[1][0] - score < 0.025:
        # Modelos indistinguibles: solo preferir un plano si es competitivo.
        plane = next((item for item in contenders if item[1][0] == 1), None)
        if plane is None or plane[0] > score + 0.025:
            for item in contenders:
                reasons[item[1][0] - 1] = 7
            cache[region] = unknown
            return unknown
        score, model, sigma = plane
    answer = (
        model,
        score,
        sigma,
        int(_pose_mask(unique)),
        int(_pose_mask(unique[::fold_count])),
        len(contenders),
    )
    cache[region] = answer
    return answer


def _geometry_worker(task):
    """Evalúa un bloque del refinamiento geométrico opcional."""
    from utilidades_rendimiento import contexto

    context = contexto()
    base = context["geometry_baseline"]
    start, stop = task
    result = {k: v[start:stop].copy() for k, v in base.items()}
    proposals = []
    h = context["local_args"]["voxel"]
    for j, i in enumerate(range(start, stop)):
        patch = int(base["geometry_patch_id"][i])
        model, score, sigma, allmask, foldmask, nmodels = _geometry_region(context, patch)
        for column, name in enumerate(("plane", "cylinder", "curve")):
            result["geometry_reject_" + name][j] = context["_geometry_reasons"][patch][column]
        result["geometry_filtered_observations"][j] = context["_geometry_filtered"][patch]
        result["geometry_validation_folds"][j] = context["_geometry_folds"][patch]
        result["geometry_available_poses"][j] = context["_geometry_pose_count"][patch]
        result["geometry_added_neighbors"][j] = context["_geometry_added"][patch]
        result["geometry_contenders"][j] = nmodels
        if model is None:
            continue
        result["geometry_class"][j] = model[0]
        result["geometry_cv_score"][j] = score
        result["geometry_pose_mask"][j] = allmask
        result["geometry_fold0_test_mask"][j] = foldmask
        result["geometry_sigma"][j] = sigma
        if base["support"][i] < 2:
            continue
        residual, normal = _geometry_predict(model, base["points"][i : i + 1])
        dot = float(normal[0] @ base["model_normal"][i])
        if abs(dot) < np.cos(np.deg2rad(25)) or abs(residual[0]) > min(
            0.8 * h, 2 * sigma + 0.15 * h
        ):
            continue
        # Mover a lo largo de la normal local y revalidar las observaciones de
        # cada pose: la etiqueta geométrica no elimina capas ni puntos.
        step = float(np.clip(-residual[0] / dot, -min(0.4 * h, 2 * sigma), min(0.4 * h, 2 * sigma)))
        old = float((base["points"][i] - base["model_origin"][i]) @ base["model_normal"][i])
        target = float(
            np.clip(
                old + 0.65 * step, -base["displacement_budget"][i], base["displacement_budget"][i]
            )
        )
        if abs(target - old) < 0.005 * h:
            continue
        proposals.append((j, i, target))
    if proposals:
        neighborhoods = _observations_for_queries(
            context, np.asarray([base["model_origin"][i] for _, i, _ in proposals])
        )
        for (j, i, target), ids in zip(proposals, neighborhoods):
            row = _local_patch(
                int(context["seed_ids"][i]), np.asarray(ids, int), context, proposed_shift=target
            )
            oldmask = int(base["support_pose_mask"][i])
            newmask = int(row["support_pose_mask"])
            if (
                row["status"] != 1
                or (oldmask & newmask) != oldmask
                or row["agreement"] < base["agreement"][i] - 0.02
                or row["spread"] > base["spread"][i] + 0.05 * h
            ):
                result["geometry_rejected"][j] = 1
                continue
            for name, value in row.items():
                result[name][j] = value
            result["geometry_shift_mm"][j] = np.linalg.norm(row["points"] - base["points"][i])
    return start, stop, result


def neighboring_surface_fusion(points, colors, normals, poses, weights, args):
    """Coordina la fusión de superficies vecinas y reúne los resultados por bloques."""
    from utilidades_rendimiento import ejecutar_bloques

    voxel = float(args.fusion_voxel_mm)
    seeds, keys, occupied = local_surface_seeds(
        points, normals, weights, voxel, args.local_normal_angle_deg
    )
    order, region_ids = _spatial_region_order(points[seeds])
    seeds, keys = seeds[order], keys[order]
    dtypes = {
        name: np.float64
        for name in (
            "points",
            "colors",
            "normals",
            "confidence",
            "spread",
            "normal_consistency",
            "uncertainty",
            "displacement",
            "agreement",
            "displacement_budget",
            "model_origin",
            "model_normal",
            "model_tangent",
            "model_bitangent",
            "model_coeff",
            "pose_residual",
            "regional_shift_mm",
            "regional_disagreement_mm",
        )
    }
    dtypes.update(
        support=np.int16,
        status=np.uint8,
        support_pose_mask=np.uint64,
        disagreement_pose_mask=np.uint64,
        incompatible=np.int32,
        observations=np.int32,
        modes=np.int16,
        independent_support_poses=np.int16,
        support_angular_span_poses=np.int16,
        conflict_pose_count=np.int16,
        conflict_pose_ratio=np.float64,
        heldout_validation_available=np.uint8,
        heldout_validation_tests=np.int16,
        heldout_validation_pass_count=np.int16,
        heldout_validation_pass_ratio=np.float64,
        heldout_validation_error_mm=np.float64,
        heldout_validation_p90_mm=np.float64,
        regional_status=np.uint8,
        regional_neighbors=np.int16,
        region_id=np.int32,
    )
    dtypes.update(
        geometry_class=np.uint8,
        geometry_cv_score=np.float64,
        geometry_sigma=np.float64,
        geometry_pose_mask=np.uint64,
        geometry_fold0_test_mask=np.uint64,
        geometry_contenders=np.uint8,
        geometry_rejected=np.uint8,
        geometry_shift_mm=np.float64,
        geometry_patch_id=np.int32,
        geometry_reject_plane=np.uint8,
        geometry_reject_cylinder=np.uint8,
        geometry_reject_curve=np.uint8,
        geometry_filtered_observations=np.int32,
        geometry_validation_folds=np.uint8,
        geometry_available_poses=np.int16,
        geometry_added_neighbors=np.int32,
    )
    result = _allocate_local(len(seeds), dtypes)
    context = dict(
        points=points,
        colors=colors,
        normals=normals,
        poses=poses,
        weights=weights,
        seed_ids=seeds,
        output_dtypes=dtypes,
        local_args=dict(
            voxel=voxel,
            radius=args.local_radius_voxels * voxel,
            neighbors_per_pose=args.local_neighbors_per_pose,
            cos_angle=math.cos(math.radians(args.local_normal_angle_deg)),
            sigma_floor=args.local_sigma_floor_voxels * voxel,
            max_shift=args.local_max_shift_voxels * voxel,
            spread_reference=max(float(args.maximum_spread_for_full_score_mm), 1e-6),
            expected_views=25,
            minimum_independent_pose_separation=int(args.minimum_independent_pose_separation),
            minimum_heldout_pass_ratio=float(args.minimum_heldout_pass_ratio),
            minimum_heldout_pass_numerator=int(args.minimum_heldout_pass_numerator),
            minimum_heldout_pass_denominator=int(args.minimum_heldout_pass_denominator),
        ),
    )
    print(
        f"[Paso 11 | fusión local] {occupied:,} celdas, {len(seeds):,} parches; "
        "ajuste robusto por pose y conservación de capas separadas.",
        flush=True,
    )
    ejecutar_bloques(
        _local_fusion_worker,
        context,
        result,
        len(seeds),
        128,
        "Paso 11: estimar superficies locales entre vistas",
    )
    result["region_id"] = region_ids.copy()
    result["regional_disagreement_mm"].fill(np.nan)
    regional_summary = {"enabled": False}
    if args.regional_fusion and len(seeds):
        baseline = {k: v.copy() for k, v in result.items()}
        regional_context = {k: v for k, v in context.items() if not k.startswith("_")}
        regional_context["regional_baseline"] = baseline
        print(
            f"[Paso 11 | conciliación regional] {len(np.unique(region_ids))} regiones; "
            "solapamientos geométricos, soporte por pose y límites de incertidumbre.",
            flush=True,
        )
        ejecutar_bloques(
            _regional_worker,
            regional_context,
            result,
            len(seeds),
            128,
            "Paso 11: conciliar solapamientos y verificar observaciones originales",
        )
        regional_summary = _regional_report(result, region_ids)
        del baseline, regional_context

    geometry_summary = {"enabled": False, "reason": "primitive_refinement_disabled_by_default"}
    if bool(args.geometry_refinement) and len(seeds):
        _estado11("Comparando plano, cilindro y superficie curva con poses separadas")
        _estado11("Formando parches por continuidad de posición, normales y separación de capas")
        result["geometry_patch_id"] = _surface_patch_ids(result, voxel)
        geometry_context = {k: v for k, v in context.items() if not k.startswith("_")}
        result["geometry_cv_score"].fill(np.nan)
        geometry_context["geometry_baseline"] = {k: v.copy() for k, v in result.items()}
        ejecutar_bloques(
            _geometry_worker,
            geometry_context,
            result,
            len(seeds),
            512,
            "Paso 11: selección geométrica automática y validación cruzada por pose",
        )
        geometry_summary = {
            "enabled": True,
            "scope": "regional_surface_not_object_name",
            "models": {
                "0": "unidentified",
                "1": "plane",
                "2": "cylinder",
                "3": "general_quadratic_surface",
            },
            "validation": "leave_one_pose_out_for_3_to_5_otherwise_three_pose_groups",
            "minimum_distinct_poses": 3,
            "minimum_training_poses_per_fold": 2,
            "training_and_validation_poses_disjoint": True,
            "full_refit_after_validation": True,
            "object_name_used_for_selection": False,
            "per_seed_counts": {
                name: int(np.count_nonzero(result["geometry_class"] == code))
                for code, name in enumerate(("unidentified", "plane", "cylinder", "general_curve"))
            },
            "moved_candidates": int(np.count_nonzero(result["geometry_shift_mm"] > 0)),
            "rejected_by_original_observations": int(np.count_nonzero(result["geometry_rejected"])),
            "additional_shift_mm": finite_stats(result["geometry_shift_mm"]),
            "no_points_removed_by_model_selection": True,
        }
        _, representatives = np.unique(result["geometry_patch_id"], return_index=True)
        geometry_summary["expanded_patches"] = int(
            np.count_nonzero(result["geometry_added_neighbors"][representatives])
        )
        geometry_summary["available_poses_per_patch"] = finite_stats(
            result["geometry_available_poses"][representatives]
        )
        geometry_summary["validation_folds_per_patch"] = {
            str(k): int(np.count_nonzero(result["geometry_validation_folds"][representatives] == k))
            for k in (0, 3, 4, 5)
        }
        geometry_summary["surface_patches"] = int(len(representatives))
        geometry_summary["patches_independent_of_processing_blocks"] = True
        geometry_summary["model_rejection_counts_per_patch"] = {
            name: {
                reason: int(
                    np.count_nonzero(result["geometry_reject_" + name][representatives] == code)
                )
                for code, reason in GEOMETRY_REASONS.items()
            }
            for name in ("plane", "cylinder", "curve")
        }
        geometry_summary["excluded_observations_per_patch"] = finite_stats(
            result["geometry_filtered_observations"][representatives]
        )
        geometry_summary["reason_codes"] = GEOMETRY_REASONS
        del geometry_context

    # Dos semillas del MISMO voxel pueden converger a la misma superficie.
    # Elegir una observación resuelta; nunca promediar otra vez ni sumar soportes.
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    order = np.lexsort((seeds, -result["confidence"], -result["support"], inv))
    starts = np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1, len(order)]
    unique = np.ones(len(seeds), dtype=bool)
    for begin, end in zip(starts[:-1], starts[1:]):
        selected = []
        for idx in order[begin:end]:
            if any(
                np.linalg.norm(result["points"][idx] - result["points"][prev]) < 0.30 * voxel
                and result["normals"][idx] @ result["normals"][prev] > math.cos(math.radians(20.0))
                for prev in selected
            ):
                unique[idx] = False
            else:
                selected.append(int(idx))
    diagnostics = {
        name: values
        for name, values in result.items()
        if name not in ("colors", "normals") and not name.startswith("model_")
    }
    diagnostics["seed_points"] = points[seeds].copy()
    diagnostics["duplicate_candidate"] = ~unique
    retained = np.flatnonzero(unique)
    result = {name: values[unique] for name, values in result.items()}
    summary = dict(
        enabled=True,
        version="11.8",
        method="pose_diverse_heldout_pose_layer_local_coherence_observed_coverage_recovery_raw_pose_bias_multiscale_regional_consensus_fusion",
        regional_fusion=regional_summary,
        geometry_selection=geometry_summary,
        empirical_uncertainty=True,
        calibrated_covariance_available=False,
        prefit_pose_layer_gate=True,
        pose_reference="equal_pose_consensus_with_resolved_gaps",
        normal_derivative_consistency_required=True,
        conflict_masks_include_prefit_exclusions=True,
        registered_observations_preserved=int(len(points)),
        occupied_voxels=int(occupied),
        surface_seeds=int(len(seeds)),
        duplicate_candidates_removed=int(np.count_nonzero(~unique)),
        candidates_after_deduplication=int(len(retained)),
        status_codes=LOCAL_STATUS,
        status_counts={
            description: int(np.count_nonzero(result["status"] == code))
            for code, description in LOCAL_STATUS.items()
        },
        displacement_mm=finite_stats(result["displacement"]),
        uncertainty_mm=finite_stats(result["uncertainty"]),
        scope="Local quadratic fusion with pose diversity and held-out validation; optional primitive diagnostics disabled by default.",
        neighborhood_note="At most K nearest observations per pose within the metric radius; capped per-pose sampling.",
    )
    return result, keys[unique], diagnostics, retained, summary


def _feature_family_rescue(points, normals, uncertainty, seed_index, neighbor_indices, voxel, args):
    """Detecta aristas/esquinas locales por familias de normales, no por forma global."""
    ids = np.asarray(neighbor_indices, dtype=np.int64)
    if len(ids) < max(6, int(getattr(args, "class2_feature_min_family_points", 4)) * 2):
        return False
    ni = normalize_rows(np.asarray(normals[seed_index], dtype=np.float64).reshape(1, 3))[0]
    vv = normalize_rows(np.asarray(normals[ids], dtype=np.float64))
    # Las normales pueden llegar con signo opuesto; para el agrupamiento se
    # llevan al mismo hemisferio que la normal del punto semilla.
    flip = (vv @ ni) < 0.0
    vv[flip] *= -1.0
    delta = np.asarray(points[ids], dtype=np.float64) - np.asarray(
        points[seed_index], dtype=np.float64
    )
    u0 = float(uncertainty[seed_index]) if np.isfinite(uncertainty[seed_index]) else 0.25 * voxel
    uj = np.asarray(uncertainty[ids], dtype=np.float64)
    uj = np.where(np.isfinite(uj) & (uj > 0), uj, u0)
    tolerance = np.clip(1.70 * np.minimum(u0, uj), 0.20 * voxel, 0.55 * voxel)
    min_fraction = float(getattr(args, "class2_feature_min_family_fraction", 0.18))
    min_points = int(getattr(args, "class2_feature_min_family_points", 4))
    max_within = float(getattr(args, "class2_feature_max_within_family_p80_deg", 17.0))
    min_separation = float(getattr(args, "class2_feature_min_family_separation_deg", 25.0))
    min_backing = float(getattr(args, "class2_feature_min_tangent_backing_fraction", 0.60))

    # Se prueban dos y tres familias. Esto permite preservar aristas y esquinas
    # sin identificar cubos, cilindros ni ninguna primitiva concreta.
    for k in (2, 3):
        if len(ids) < k * min_points:
            continue
        centers = [ni.copy()]
        for _ in range(1, k):
            sims = np.column_stack([vv @ c for c in centers])
            closeness = np.max(sims, axis=1)
            centers.append(vv[int(np.argmin(closeness))].copy())
        centers = np.asarray(centers, dtype=np.float64)
        labels = np.zeros(len(vv), dtype=np.int16)
        valid_fit = True
        for _ in range(8):
            labels = np.argmax(vv @ centers.T, axis=1).astype(np.int16)
            updated = []
            for family in range(k):
                member = labels == family
                if not np.any(member):
                    valid_fit = False
                    break
                center = np.mean(vv[member], axis=0)
                norm = float(np.linalg.norm(center))
                if norm <= 1e-12:
                    valid_fit = False
                    break
                updated.append(center / norm)
            if not valid_fit:
                break
            centers = np.asarray(updated, dtype=np.float64)
        if not valid_fit:
            continue
        counts = np.asarray(
            [np.count_nonzero(labels == family) for family in range(k)], dtype=np.int32
        )
        active = (counts >= min_points) & (counts >= min_fraction * len(ids))
        active_ids = np.flatnonzero(active)
        if len(active_ids) < 2:
            continue
        families_ok = True
        for family in active_ids:
            member = labels == family
            angles = np.degrees(np.arccos(np.clip(vv[member] @ centers[family], -1.0, 1.0)))
            if float(np.percentile(angles, 80)) > max_within:
                families_ok = False
                break
            tangent_error = np.abs(delta[member] @ centers[family])
            if float(np.mean(tangent_error <= tolerance[member])) < min_backing:
                families_ok = False
                break
        if not families_ok:
            continue
        separations = []
        for a in range(len(active_ids)):
            for b in range(a + 1, len(active_ids)):
                dot = float(
                    np.clip(np.dot(centers[active_ids[a]], centers[active_ids[b]]), -1.0, 1.0)
                )
                separations.append(float(np.degrees(np.arccos(dot))))
        if separations and min(separations) >= min_separation:
            return True
    return False


def _resolve_validated_pose_layers(local_result, voxel, args):
    """Separa hojas/capas coherentes respaldadas por grupos independientes de poses.

    La validación held-out demuestra que un modelo predice poses excluidas, pero
    varias poses pueden compartir un sesgo espacial coherente. Este control usa
    únicamente los residuos firmados por pose respecto al parche local ya
    validado. Si detecta dos hojas multivista separadas:

    - con ganador claro, conserva solo ese grupo y desplaza el surfel como máximo
      dentro del presupuesto de incertidumbre;
    - sin ganador claro, marca el candidato como ambiguo para degradarlo de clase
      3 a clase 2, donde vuelve a pasar por coherencia local multiescala.

    No clasifica la forma del objeto ni ajusta primitivas globales.
    """
    n = 0 if local_result is None else len(local_result.get("points", []))
    arrays = {
        "validated_layer_available": np.zeros(n, dtype=np.uint8),
        "validated_layer_accept": np.ones(n, dtype=np.uint8),
        "validated_layer_count": np.ones(n, dtype=np.uint8),
        "validated_layer_ambiguous": np.zeros(n, dtype=np.uint8),
        "validated_layer_resolved": np.zeros(n, dtype=np.uint8),
        "validated_layer_selected_pose_mask": np.zeros(n, dtype=np.uint64),
        "validated_layer_secondary_pose_mask": np.zeros(n, dtype=np.uint64),
        "validated_layer_separation_mm": np.full(n, np.nan, dtype=np.float64),
        "validated_layer_shift_mm": np.zeros(n, dtype=np.float64),
        "validated_layer_winner_independent": np.zeros(n, dtype=np.int16),
        "validated_layer_runnerup_independent": np.zeros(n, dtype=np.int16),
        "validated_layer_score_margin": np.full(n, np.nan, dtype=np.float64),
    }
    report = {
        "enabled": bool(getattr(args, "validated_pose_layer_separation", True)),
        "evaluated": 0,
        "single_layer": 0,
        "multi_layer_detected": 0,
        "resolved_to_dominant_layer": 0,
        "ambiguous_downgraded": 0,
        "shift_mm": finite_stats([]),
        "separation_mm": finite_stats([]),
        "note": ("Post-fit signed pose residuals; no primitive or object-name assumptions."),
    }
    if n == 0 or not report["enabled"]:
        if local_result is not None:
            local_result.update(arrays)
        return report
    required = ("status", "support_pose_mask", "pose_residual", "uncertainty", "normals", "points")
    if any(name not in local_result for name in required):
        report["enabled"] = False
        report["reason"] = "missing_pose_layer_diagnostics"
        local_result.update(arrays)
        return report

    status = np.asarray(local_result["status"])
    support_mask = np.asarray(local_result["support_pose_mask"], dtype=np.uint64)
    pose_residual = np.asarray(local_result["pose_residual"], dtype=np.float64)
    uncertainty = np.asarray(local_result["uncertainty"], dtype=np.float64)
    normals = normalize_rows(np.asarray(local_result["normals"], dtype=np.float64))
    points = np.asarray(local_result["points"], dtype=np.float64)
    heldout_available = np.asarray(
        local_result.get("heldout_validation_available", np.zeros(n)), dtype=bool
    )
    heldout_tests = np.asarray(
        local_result.get("heldout_validation_tests", np.zeros(n)), dtype=np.int16
    )
    heldout_pass_count = np.asarray(
        local_result.get("heldout_validation_pass_count", np.zeros(n)), dtype=np.int16
    )
    min_support = max(4, int(getattr(args, "validated_layer_min_supported_poses", 4)))
    min_ind = max(2, int(getattr(args, "validated_layer_min_independent_per_layer", 2)))
    min_sep_pose = max(1, int(getattr(args, "minimum_independent_pose_separation", 2)))
    min_gap_voxel = max(0.05, float(getattr(args, "validated_layer_min_separation_voxel", 0.34)))
    uncertainty_factor = max(0.5, float(getattr(args, "validated_layer_uncertainty_factor", 2.25)))
    within_factor = max(1.0, float(getattr(args, "validated_layer_within_sigma_factor", 3.0)))
    winner_ratio = max(1.0, float(getattr(args, "validated_layer_min_winner_ratio", 1.35)))
    independent_advantage = max(
        0, int(getattr(args, "validated_layer_min_independent_advantage", 1))
    )
    max_shift_voxel = max(0.0, float(getattr(args, "validated_layer_max_shift_voxel", 0.30)))
    max_shift_unc = max(
        0.0, float(getattr(args, "validated_layer_max_shift_uncertainty_factor", 1.50))
    )
    heldout_num = max(0, int(getattr(args, "minimum_heldout_pass_numerator", 2)))
    heldout_den = max(1, int(getattr(args, "minimum_heldout_pass_denominator", 3)))

    shifts, separations = [], []
    candidates = np.flatnonzero(
        (status == 1)
        & heldout_available
        & (heldout_tests > 0)
        & (
            heldout_pass_count.astype(np.int64) * heldout_den
            >= heldout_num * heldout_tests.astype(np.int64)
        )
    )
    for i in candidates:
        pose_ids = np.asarray(_pose_ids_from_mask(support_mask[i], expected=25), dtype=np.int16)
        if len(pose_ids) < min_support:
            continue
        values = pose_residual[i, pose_ids].astype(np.float64)
        finite = np.isfinite(values)
        pose_ids, values = pose_ids[finite], values[finite]
        if len(values) < min_support:
            continue
        arrays["validated_layer_available"][i] = 1
        report["evaluated"] += 1
        order = np.argsort(values, kind="stable")
        p_sorted, v_sorted = pose_ids[order], values[order]
        u0 = (
            float(uncertainty[i])
            if np.isfinite(uncertainty[i]) and uncertainty[i] > 0
            else 0.18 * float(voxel)
        )
        floor_sigma = max(0.04 * float(voxel), 0.25 * u0)
        best = None
        # Se prueban todos los cortes 1D posibles. Ambos lados deben tener
        # diversidad angular propia; así una secuencia de poses vecinas no crea
        # artificialmente una segunda hoja.
        for cut in range(2, len(v_sorted) - 1):
            pa, pb = p_sorted[:cut], p_sorted[cut:]
            va, vb = v_sorted[:cut], v_sorted[cut:]
            inda, _ = _pose_diversity(pa, expected=25, minimum_separation=min_sep_pose)
            indb, _ = _pose_diversity(pb, expected=25, minimum_separation=min_sep_pose)
            if inda < min_ind or indb < min_ind:
                continue
            ma, mb = float(np.median(va)), float(np.median(vb))
            sa = max(floor_sigma, float(1.4826 * np.median(np.abs(va - ma))))
            sb = max(floor_sigma, float(1.4826 * np.median(np.abs(vb - mb))))
            separation = abs(mb - ma)
            gate = max(
                min_gap_voxel * float(voxel), uncertainty_factor * u0, within_factor * max(sa, sb)
            )
            empty_gap = float(v_sorted[cut] - v_sorted[cut - 1])
            # Una bimodalidad real requiere también un intervalo vacío local;
            # separar por medianas sin hueco podría partir una sola distribución.
            if separation < gate or empty_gap < max(0.50 * gate, 0.18 * float(voxel)):
                continue
            balance = min(len(va), len(vb)) / max(len(va), len(vb))
            score = (
                (separation / max(gate, 1e-9))
                * (empty_gap / max(gate, 1e-9))
                * (0.75 + 0.25 * balance)
            )
            record = (score, separation, gate, pa, pb, va, vb, inda, indb, ma, mb, sa, sb)
            if best is None or record[0] > best[0]:
                best = record
        if best is None:
            report["single_layer"] += 1
            arrays["validated_layer_count"][i] = 1
            arrays["validated_layer_selected_pose_mask"][i] = support_mask[i]
            continue

        score, separation, gate, pa, pb, va, vb, inda, indb, ma, mb, sa, sb = best
        report["multi_layer_detected"] += 1
        arrays["validated_layer_count"][i] = 2
        arrays["validated_layer_separation_mm"][i] = separation
        separations.append(separation)
        # Fiabilidad: primero diversidad independiente, después número de poses,
        # compactación intra-hoja y cercanía al surfel actual. No se favorece una
        # forma geométrica concreta.
        ra = (
            float(inda)
            + 0.30 * len(va)
            - 0.45 * sa / max(gate, 1e-9)
            - 0.10 * abs(ma) / max(gate, 1e-9)
        )
        rb = (
            float(indb)
            + 0.30 * len(vb)
            - 0.45 * sb / max(gate, 1e-9)
            - 0.10 * abs(mb) / max(gate, 1e-9)
        )
        if rb > ra:
            winner_p, loser_p = pb, pa
            winner_v, loser_v = vb, va
            winner_ind, loser_ind = indb, inda
            winner_med, loser_med = mb, ma
            winner_score, loser_score = rb, ra
        else:
            winner_p, loser_p = pa, pb
            winner_v, loser_v = va, vb
            winner_ind, loser_ind = inda, indb
            winner_med, loser_med = ma, mb
            winner_score, loser_score = ra, rb
        count_ratio = len(winner_v) / max(len(loser_v), 1)
        winner_fraction = len(winner_v) / max(len(winner_v) + len(loser_v), 1)
        independent_clear = (winner_ind >= loser_ind + independent_advantage) and (
            winner_fraction >= 0.60
        )
        count_clear = count_ratio >= winner_ratio
        margin = float((winner_score - loser_score) / max(abs(winner_score), 1e-9))
        arrays["validated_layer_winner_independent"][i] = int(winner_ind)
        arrays["validated_layer_runnerup_independent"][i] = int(loser_ind)
        arrays["validated_layer_score_margin"][i] = margin
        arrays["validated_layer_selected_pose_mask"][i] = _pose_mask(winner_p)
        arrays["validated_layer_secondary_pose_mask"][i] = _pose_mask(loser_p)
        if not (independent_clear or count_clear):
            arrays["validated_layer_accept"][i] = 0
            arrays["validated_layer_ambiguous"][i] = 1
            report["ambiguous_downgraded"] += 1
            continue

        # Ganador claro: mover solo a lo largo de la normal y dentro del
        # presupuesto físico. El corrimiento máximo nunca supera 0.30 voxel ni
        # 1.5 veces la incertidumbre por defecto.
        shift_cap = min(max_shift_voxel * float(voxel), max_shift_unc * u0)
        shift = float(np.clip(winner_med, -shift_cap, shift_cap))
        if not np.isfinite(shift):
            arrays["validated_layer_accept"][i] = 0
            arrays["validated_layer_ambiguous"][i] = 1
            report["ambiguous_downgraded"] += 1
            continue
        points[i] = points[i] + shift * normals[i]
        pose_residual[i, :] = pose_residual[i, :] - shift
        winner_mask = _pose_mask(winner_p)
        loser_mask = _pose_mask(loser_p)
        local_result["support_pose_mask"][i] = winner_mask
        local_result["disagreement_pose_mask"][i] = np.uint64(
            int(local_result["disagreement_pose_mask"][i]) | int(loser_mask)
        )
        local_result["support"][i] = int(len(winner_p))
        ind_new, span_new = _pose_diversity(winner_p, expected=25, minimum_separation=min_sep_pose)
        local_result["independent_support_poses"][i] = int(ind_new)
        local_result["support_angular_span_poses"][i] = int(span_new)
        conflict_ids = _pose_ids_from_mask(local_result["disagreement_pose_mask"][i], expected=25)
        local_result["conflict_pose_count"][i] = int(len(conflict_ids))
        local_result["conflict_pose_ratio"][i] = float(
            len(conflict_ids) / max(len(winner_p) + len(conflict_ids), 1)
        )
        local_result["agreement"][i] = float(len(winner_p) / max(len(winner_p) + len(loser_p), 1))
        winner_center = float(np.median(winner_v - shift))
        winner_sigma = max(
            floor_sigma, float(1.4826 * np.median(np.abs((winner_v - shift) - winner_center)))
        )
        local_result["spread"][i] = min(
            float(local_result["spread"][i]), max(winner_sigma, 0.05 * float(voxel))
        )
        local_result["confidence"][i] = float(
            np.clip(
                local_result["confidence"][i] * (0.90 + 0.10 * min(1.0, max(margin, 0.0))), 0.0, 1.0
            )
        )
        arrays["validated_layer_resolved"][i] = 1
        arrays["validated_layer_shift_mm"][i] = shift
        shifts.append(abs(shift))
        report["resolved_to_dominant_layer"] += 1

    local_result["points"][:] = points
    local_result["pose_residual"][:] = pose_residual
    local_result.update(arrays)
    report["shift_mm"] = finite_stats(shifts)
    report["separation_mm"] = finite_stats(separations)
    return report


def _evaluate_class2_local_coherence(points, normals, uncertainty, evidence_class, voxel, args):
    """Evalúa clase 2 con coherencia superficial local multiescala.

    El test es conservador: un punto solo se rechaza si acumula varios fallos
    independientes. Si el vecindario parece una arista/esquina coherente, se
    conserva mediante agrupamiento local de normales. No se ajusta ninguna
    primitiva global ni se usa el nombre del objeto.
    """
    n = len(points)
    result = {
        "available": np.zeros(n, dtype=np.uint8),
        "accept": np.ones(n, dtype=np.uint8),
        "strict": np.ones(n, dtype=np.uint8),
        "feature_like": np.zeros(n, dtype=np.uint8),
        "score": np.ones(n, dtype=np.float64),
        "red_flags": np.zeros(n, dtype=np.uint8),
        "plane_p90_small_mm": np.full(n, np.nan, dtype=np.float64),
        "plane_p90_large_mm": np.full(n, np.nan, dtype=np.float64),
        "normal_p90_large_deg": np.full(n, np.nan, dtype=np.float64),
        "scale_normal_difference_deg": np.full(n, np.nan, dtype=np.float64),
        "surface_variation_large": np.full(n, np.nan, dtype=np.float64),
        "same_sheet_fraction": np.full(n, np.nan, dtype=np.float64),
    }
    if not bool(getattr(args, "class2_local_coherence", True)):
        return result
    points = np.asarray(points, dtype=np.float64)
    normals = normalize_rows(np.asarray(normals, dtype=np.float64))
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    evidence_class = np.asarray(evidence_class, dtype=np.uint8)
    finite = np.all(np.isfinite(points), axis=1) & np.all(np.isfinite(normals), axis=1)
    base_idx = np.flatnonzero((evidence_class >= 2) & finite)
    target_idx = np.flatnonzero((evidence_class == 2) & finite)
    if len(base_idx) < 4 or len(target_idx) == 0:
        return result
    small_k = max(6, int(getattr(args, "class2_coherence_small_neighbors", 18)))
    large_k = max(small_k + 2, int(getattr(args, "class2_coherence_large_neighbors", 36)))
    minimum_neighbors = max(6, int(getattr(args, "class2_coherence_min_neighbors", 10)))
    query_k = min(len(base_idx), large_k + 1)
    tree = cKDTree(points[base_idx])
    distances, neighbors_local = tree.query(points[target_idx], k=query_k, workers=query_threads())
    if np.ndim(distances) == 1:
        distances = distances[:, None]
        neighbors_local = neighbors_local[:, None]

    def patch_metrics(seed, ids):
        q = points[ids]
        center = np.mean(q, axis=0)
        x = q - center
        covariance = (x.T @ x) / max(len(q), 1)
        values, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, 0]
        seed_normal = normals[seed]
        if np.dot(normal, seed_normal) < 0:
            normal = -normal
        residual = np.abs(x @ normal)
        variation = float(values[0] / max(float(np.sum(values)), 1e-12))
        return float(np.percentile(residual, 90)), normal, variation

    reject_count = max(1, int(getattr(args, "class2_coherence_reject_flag_count", 3)))
    strict_max = max(0, int(getattr(args, "class2_coherence_strict_max_flags", 1)))
    for row, seed in enumerate(target_idx):
        local_global = base_idx[np.asarray(neighbors_local[row], dtype=np.int64)]
        # Eliminar el propio punto y duplicados; conservar orden por distancia.
        ordered = []
        seen = set()
        for g in local_global:
            g = int(g)
            if g == int(seed) or g in seen:
                continue
            seen.add(g)
            ordered.append(g)
        if len(ordered) < minimum_neighbors:
            # Una pieza pequeña o aislada no se elimina por falta de vecinos.
            result["score"][seed] = 0.65
            result["strict"][seed] = 0
            continue
        small_ids = np.asarray(ordered[: min(small_k, len(ordered))], dtype=np.int64)
        large_ids = np.asarray(ordered[: min(large_k, len(ordered))], dtype=np.int64)
        p90_small, normal_small, _ = patch_metrics(seed, small_ids)
        p90_large, normal_large, variation_large = patch_metrics(seed, large_ids)
        seed_normal = normals[seed]
        normal_angles = np.degrees(
            np.arccos(np.clip(np.abs(normals[large_ids] @ seed_normal), 0.0, 1.0))
        )
        normal_p90 = float(np.percentile(normal_angles, 90))
        scale_dot = float(np.clip(abs(np.dot(normal_small, normal_large)), 0.0, 1.0))
        scale_angle = float(np.degrees(np.arccos(scale_dot)))
        delta = points[large_ids] - points[seed]
        average_normal = normalize_rows(normals[large_ids] + seed_normal[None, :])
        tangent_error = np.abs(np.einsum("ij,ij->i", delta, average_normal))
        pair_angle = np.degrees(
            np.arccos(np.clip(np.abs(normals[large_ids] @ seed_normal), 0.0, 1.0))
        )
        u0 = (
            float(uncertainty[seed])
            if np.isfinite(uncertainty[seed]) and uncertainty[seed] > 0
            else 0.25 * voxel
        )
        uj = uncertainty[large_ids]
        uj = np.where(np.isfinite(uj) & (uj > 0), uj, u0)
        same_tolerance = np.clip(1.50 * np.minimum(u0, uj), 0.18 * voxel, 0.50 * voxel)
        same_sheet = (pair_angle <= 32.0) & (tangent_error <= same_tolerance)
        same_fraction = float(np.mean(same_sheet)) if len(same_sheet) else 0.0

        small_limit = max(
            float(getattr(args, "class2_coherence_small_plane_p90_voxel", 0.55)) * voxel,
            float(getattr(args, "class2_coherence_small_plane_uncertainty_factor", 1.75)) * u0,
        )
        large_limit = max(
            float(getattr(args, "class2_coherence_large_plane_p90_voxel", 0.80)) * voxel,
            float(getattr(args, "class2_coherence_large_plane_uncertainty_factor", 2.20)) * u0,
        )
        normal_limit = float(getattr(args, "class2_coherence_normal_p90_deg", 42.0))
        scale_limit = float(getattr(args, "class2_coherence_scale_normal_deg", 25.0))
        variation_limit = float(getattr(args, "class2_coherence_max_surface_variation", 0.16))
        same_min = float(getattr(args, "class2_coherence_min_same_sheet_fraction", 0.30))
        flags = np.asarray(
            [
                p90_small > small_limit,
                p90_large > large_limit,
                normal_p90 > normal_limit,
                scale_angle > scale_limit,
                variation_large > variation_limit,
                same_fraction < same_min,
            ],
            dtype=bool,
        )
        red_flags = int(np.count_nonzero(flags))
        feature_like = False
        if red_flags >= reject_count or red_flags > strict_max:
            feature_like = _feature_family_rescue(
                points, normals, uncertainty, int(seed), large_ids, float(voxel), args
            )
        # Score continuo para que 12 reduzca la influencia de zonas dudosas sin
        # borrar por un único umbral.
        scores = np.asarray(
            [
                np.exp(-0.5 * (p90_small / max(small_limit, 1e-9)) ** 2),
                np.exp(-0.5 * (p90_large / max(large_limit, 1e-9)) ** 2),
                np.exp(-0.5 * (normal_p90 / max(normal_limit, 1e-9)) ** 2),
                np.exp(-0.5 * (scale_angle / max(scale_limit, 1e-9)) ** 2),
                np.exp(-0.5 * (variation_large / max(variation_limit, 1e-9)) ** 2),
                np.clip(same_fraction / max(same_min, 1e-9), 0.0, 1.0),
            ],
            dtype=np.float64,
        )
        score = float(np.clip(np.mean(scores), 0.0, 1.0))
        if feature_like:
            score = max(score, 0.75)
        accepted = (red_flags < reject_count) or feature_like
        strict = (red_flags <= strict_max) or feature_like
        result["available"][seed] = 1
        result["accept"][seed] = 1 if accepted else 0
        result["strict"][seed] = 1 if strict else 0
        result["feature_like"][seed] = 1 if feature_like else 0
        result["score"][seed] = score
        result["red_flags"][seed] = np.uint8(min(red_flags, 255))
        result["plane_p90_small_mm"][seed] = p90_small
        result["plane_p90_large_mm"][seed] = p90_large
        result["normal_p90_large_deg"][seed] = normal_p90
        result["scale_normal_difference_deg"][seed] = scale_angle
        result["surface_variation_large"][seed] = variation_large
        result["same_sheet_fraction"][seed] = same_fraction
    return result


def _ring_quadratic_prediction(u, v, z, weights, radius, voxel):
    """Predicción robusta z(0,0) desde una corona, sin primitiva de objeto."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(z) < 7 or not np.isfinite(radius) or radius <= 0:
        return None
    us = u / float(radius)
    vs = v / float(radius)
    design = np.column_stack((np.ones(len(z)), us, vs, us * us, us * vs, vs * vs))
    base = np.clip(np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), 1e-5, None)
    robust = base.copy()
    coeff = None
    condition = np.inf
    for _ in range(2):
        root = np.sqrt(np.maximum(robust, 1e-12))
        weighted = design * root[:, None]
        target = z * root
        try:
            coeff, _, rank, singular = np.linalg.lstsq(weighted, target, rcond=1e-8)
        except np.linalg.LinAlgError:
            return None
        if rank < 6 or len(singular) < 6 or singular[-1] <= 0:
            return None
        condition = float(singular[0] / singular[-1])
        if not np.isfinite(condition) or condition > 5000:
            return None
        residual = z - design @ coeff
        center = float(np.median(residual))
        sigma = max(0.04 * float(voxel), float(1.4826 * np.median(np.abs(residual - center))))
        robust = base * np.minimum(1.0, 1.5 * sigma / np.maximum(np.abs(residual - center), 1e-12))
    if coeff is None or not np.all(np.isfinite(coeff)):
        return None
    residual = z - design @ coeff
    return {
        "prediction_mm": float(coeff[0]),
        "p90_residual_mm": float(np.percentile(np.abs(residual), 90)),
        "condition": float(condition),
    }


def _tangent_coverage_bins(u, v, bins=8):
    """Cuenta sectores angulares ocupados en el plano tangente local."""
    if len(u) == 0:
        return 0
    angle = (np.arctan2(v, u) + 2 * np.pi) % (2 * np.pi)
    occupied = np.unique(np.floor(angle / (2 * np.pi / max(1, int(bins)))).astype(np.int16))
    return int(len(occupied))


def _nearest_spacing_median(points):
    """Estima el espaciado mediano entre vecinos de la nube."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return np.nan
    distances, _ = cKDTree(points).query(points, k=2, workers=query_threads())
    second = np.asarray(distances[:, 1], dtype=np.float64)
    second = second[np.isfinite(second) & (second > 1e-9)]
    return float(np.median(second)) if len(second) else np.nan


def _two_way_local_pose_bias(
    pose_residual, neighbor_ids, minimum_samples_per_pose, expected_views=25, iterations=4
):
    """Ajuste robusto r[j,p] = b[p] - g[j] + e sin asumir geometría.

    b[p] representa el sesgo relativo de una pose dentro del vecindario y g[j]
    el gauge inducido por la composición de poses del parche j. El sistema tiene
    una constante indeterminada; se fija con mediana(b)=0 porque un desplazamiento
    común no genera ondulación relativa y no debe corregirse.
    """
    R = np.asarray(pose_residual, dtype=np.float64)[np.asarray(neighbor_ids, dtype=np.int64)]
    if R.ndim != 2 or R.shape[1] < expected_views or len(R) < 6:
        return None
    R = R[:, :expected_views]
    finite = np.isfinite(R)
    counts = np.sum(finite, axis=0)
    active = np.flatnonzero(counts >= int(minimum_samples_per_pose))
    if len(active) < 3:
        return None
    b = np.full(expected_views, np.nan, dtype=np.float64)
    for p in active:
        b[p] = float(np.median(R[finite[:, p], p]))
    b[active] -= float(np.median(b[active]))
    g = np.full(len(R), np.nan, dtype=np.float64)
    for _ in range(max(2, int(iterations))):
        for j in range(len(R)):
            ps = active[finite[j, active]]
            if len(ps) >= 2:
                g[j] = float(np.median(b[ps] - R[j, ps]))
        for p in active:
            rows = np.flatnonzero(finite[:, p] & np.isfinite(g))
            if len(rows) >= int(minimum_samples_per_pose):
                b[p] = float(np.median(R[rows, p] + g[rows]))
        good_b = active[np.isfinite(b[active])]
        if len(good_b) < 3:
            return None
        b[good_b] -= float(np.median(b[good_b]))
    modeled = []
    for j in range(len(R)):
        if not np.isfinite(g[j]):
            continue
        ps = active[finite[j, active] & np.isfinite(b[active])]
        if len(ps):
            modeled.extend((R[j, ps] - (b[ps] - g[j])).tolist())
    modeled = np.asarray(modeled, dtype=np.float64)
    if len(modeled) < 12:
        return None
    return {
        "pose_bias": b,
        "active_poses": active.astype(np.int16),
        "model_p90_mm": float(np.percentile(np.abs(modeled), 90)),
        "model_median_mm": float(np.median(np.abs(modeled))),
    }


def _support_gauge_from_pose_bias(pose_bias, support_mask, expected=25, minimum_separation=2):
    poses = np.asarray(_pose_ids_from_mask(support_mask, expected=expected), dtype=np.int16)
    if len(poses) == 0:
        return None
    valid = poses[np.isfinite(np.asarray(pose_bias)[poses])]
    if len(valid) < 2:
        return None
    independent, _ = _pose_diversity(
        valid, expected=expected, minimum_separation=minimum_separation
    )
    return float(np.median(np.asarray(pose_bias)[valid])), int(independent), valid


def _postselection_raw_pose_bias_consensus(
    points,
    normals,
    support_pose_mask,
    pose_residual,
    confidence,
    uncertainty,
    evidence_class,
    coherence_strict,
    feature_like,
    voxel,
    args,
):
    """Corrige ondulación inducida por cambios de composición de poses.

    A diferencia del consenso regional V11.5, esta etapa usa los residuos por pose
    calculados desde observaciones originales antes de la fusión. En dos escalas
    resuelve un modelo aditivo pose/parche y corrige solo el gauge relativo que
    cambia con el conjunto de poses que soporta cada parche. No usa primitivas ni
    el nombre del objeto y nunca intenta recuperar un desplazamiento común a todas
    las poses.
    """
    points = np.asarray(points, dtype=np.float64)
    normals = normalize_rows(np.asarray(normals, dtype=np.float64))
    masks = np.asarray(support_pose_mask, dtype=np.uint64)
    residuals = np.asarray(pose_residual, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    evidence = np.asarray(evidence_class, dtype=np.uint8)
    coherence_strict = np.asarray(coherence_strict, dtype=bool)
    feature_like = np.asarray(feature_like, dtype=bool)
    n = len(points)
    result = {
        "available": np.zeros(n, dtype=np.uint8),
        "accept": np.ones(n, dtype=np.uint8),
        "ambiguous": np.zeros(n, dtype=np.uint8),
        "feature_protected": np.zeros(n, dtype=np.uint8),
        "shift_mm": np.zeros(n, dtype=np.float64),
        "score": np.ones(n, dtype=np.float64),
        "independent_poses": np.zeros(n, dtype=np.int16),
        "small_gauge_mm": np.full(n, np.nan, dtype=np.float64),
        "large_gauge_mm": np.full(n, np.nan, dtype=np.float64),
        "scale_difference_mm": np.full(n, np.nan, dtype=np.float64),
        "model_p90_mm": np.full(n, np.nan, dtype=np.float64),
    }
    report = {
        "enabled": bool(getattr(args, "raw_pose_bias_consensus", True)),
        "evaluated": 0,
        "available": 0,
        "corrected": 0,
        "ambiguous": 0,
        "feature_protected": 0,
        "accepted_global_alpha": 0.0,
        "shift_mm": finite_stats([]),
        "scale_difference_mm": finite_stats([]),
        "model_p90_mm": finite_stats([]),
        "method": "raw_observation_pose_residual_two_way_local_gauge_two_scale",
        "shape_assumptions_used": False,
        "common_mode_bias_preserved": True,
    }
    if n == 0 or not report["enabled"] or residuals.ndim != 2 or residuals.shape[0] != n:
        return points.copy(), result, report

    small_radius = max(2.0, float(getattr(args, "raw_pose_bias_small_radius_voxels", 4.0))) * float(
        voxel
    )
    large_radius = max(
        small_radius / float(voxel) + 1.0,
        float(getattr(args, "raw_pose_bias_large_radius_voxels", 7.0)),
    ) * float(voxel)
    max_neighbors = max(48, int(getattr(args, "raw_pose_bias_max_neighbors", 180)))
    min_samples = max(3, int(getattr(args, "raw_pose_bias_min_patches_per_pose", 5)))
    min_independent = max(2, int(getattr(args, "raw_pose_bias_min_independent_poses", 3)))
    min_pose_sep = max(1, int(getattr(args, "minimum_independent_pose_separation", 2)))
    cos_same = float(
        np.cos(
            np.deg2rad(
                np.clip(float(getattr(args, "raw_pose_bias_normal_angle_deg", 24.0)), 5.0, 55.0)
            )
        )
    )
    model_voxel = max(0.05, float(getattr(args, "raw_pose_bias_model_p90_voxel", 0.22)))
    model_unc = max(0.5, float(getattr(args, "raw_pose_bias_model_p90_uncertainty_factor", 1.6)))
    scale_voxel = max(0.03, float(getattr(args, "raw_pose_bias_scale_gate_voxel", 0.14)))
    scale_unc = max(0.4, float(getattr(args, "raw_pose_bias_scale_gate_uncertainty_factor", 1.0)))
    dead_voxel = max(0.0, float(getattr(args, "raw_pose_bias_deadband_voxel", 0.05)))
    dead_unc = max(0.0, float(getattr(args, "raw_pose_bias_deadband_uncertainty_factor", 0.35)))
    max_shift_voxel = max(0.0, float(getattr(args, "raw_pose_bias_max_shift_voxel", 0.20)))
    max_shift_unc = max(
        0.0, float(getattr(args, "raw_pose_bias_max_shift_uncertainty_factor", 1.2))
    )
    update_alpha = float(np.clip(getattr(args, "raw_pose_bias_update_alpha", 0.75), 0.0, 1.0))

    tree = cKDTree(points)
    k = min(max_neighbors, n)
    distances, neighbors = tree.query(points, k=k, workers=query_threads())
    raw_shift = np.zeros(n, dtype=np.float64)
    candidates = np.flatnonzero((evidence >= 3) | ((evidence == 2) & coherence_strict))
    for i in candidates:
        report["evaluated"] += 1
        if feature_like[i]:
            result["feature_protected"][i] = 1
            report["feature_protected"] += 1
            continue
        u0 = (
            float(uncertainty[i])
            if np.isfinite(uncertainty[i]) and uncertainty[i] > 0
            else 0.16 * float(voxel)
        )
        ids = np.asarray(neighbors[i, 1:], dtype=np.int64)
        dist = np.asarray(distances[i, 1:], dtype=np.float64)
        valid = (
            (dist <= large_radius)
            & (evidence[ids] >= 2)
            & (confidence[ids] >= 0.55)
            & ((normals[ids] @ normals[i]) >= cos_same)
        )
        ids, dist = ids[valid], dist[valid]
        if len(ids) < max(10, min_samples * 2):
            continue
        small_ids = ids[dist <= small_radius]
        if len(small_ids) < max(8, min_samples * 2):
            continue
        fit_small = _two_way_local_pose_bias(residuals, small_ids, min_samples, expected_views=25)
        fit_large = _two_way_local_pose_bias(residuals, ids, min_samples, expected_views=25)
        if fit_small is None or fit_large is None:
            continue
        gs = _support_gauge_from_pose_bias(fit_small["pose_bias"], masks[i], 25, min_pose_sep)
        gl = _support_gauge_from_pose_bias(fit_large["pose_bias"], masks[i], 25, min_pose_sep)
        if gs is None or gl is None:
            continue
        gauge_small, ind_small, poses_small = gs
        gauge_large, ind_large, poses_large = gl
        independent = min(ind_small, ind_large)
        if independent < min_independent:
            continue
        model_p90 = max(float(fit_small["model_p90_mm"]), float(fit_large["model_p90_mm"]))
        model_gate = max(model_voxel * float(voxel), model_unc * u0)
        scale_difference = abs(gauge_small - gauge_large)
        scale_gate = max(scale_voxel * float(voxel), scale_unc * u0)
        result["available"][i] = 1
        result["independent_poses"][i] = int(independent)
        result["small_gauge_mm"][i] = gauge_small
        result["large_gauge_mm"][i] = gauge_large
        result["scale_difference_mm"][i] = scale_difference
        result["model_p90_mm"][i] = model_p90
        result["score"][i] = float(
            np.clip(
                np.exp(-0.5 * np.square(model_p90 / max(model_gate, 1e-9)))
                * np.exp(-0.5 * np.square(scale_difference / max(scale_gate, 1e-9))),
                0.0,
                1.0,
            )
        )
        report["available"] += 1
        if model_p90 > model_gate or scale_difference > scale_gate:
            result["accept"][i] = 0
            result["ambiguous"][i] = 1
            report["ambiguous"] += 1
            continue
        gauge = 0.60 * gauge_small + 0.40 * gauge_large
        deadband = max(dead_voxel * float(voxel), dead_unc * u0)
        if abs(gauge) <= deadband:
            continue
        cap = min(max_shift_voxel * float(voxel), max_shift_unc * u0)
        # gauge positivo significa que la superficie fusionada está desplazada
        # hacia +normal respecto al gauge local de poses; se corrige en sentido opuesto.
        raw_shift[i] = float(np.clip(-update_alpha * gauge, -cap, cap))

    spacing_before = _nearest_spacing_median(points)
    q01_before, q99_before = np.percentile(points, [1, 99], axis=0)
    extent_before = np.maximum(q99_before - q01_before, 1e-9)
    minimum_spacing_ratio = float(
        np.clip(getattr(args, "raw_pose_bias_min_spacing_ratio", 0.94), 0.80, 1.0)
    )
    maximum_extent_change = max(
        0.001, float(getattr(args, "raw_pose_bias_max_extent_change_fraction", 0.005))
    )
    accepted_alpha = 0.0
    corrected = points.copy()
    for alpha in (1.0, 0.75, 0.50, 0.25, 0.0):
        candidate = points + (alpha * raw_shift)[:, None] * normals
        spacing_after = _nearest_spacing_median(candidate)
        q01_after, q99_after = np.percentile(candidate, [1, 99], axis=0)
        extent_after = np.maximum(q99_after - q01_after, 1e-9)
        relative_extent_change = float(np.max(np.abs(extent_after / extent_before - 1.0)))
        spacing_ok = (
            not np.isfinite(spacing_before)
            or not np.isfinite(spacing_after)
            or spacing_after >= minimum_spacing_ratio * spacing_before
        )
        if spacing_ok and relative_extent_change <= maximum_extent_change:
            corrected = candidate
            accepted_alpha = float(alpha)
            break
    actual_shift = accepted_alpha * raw_shift
    result["shift_mm"][:] = actual_shift
    report["accepted_global_alpha"] = accepted_alpha
    report["corrected"] = int(np.count_nonzero(np.abs(actual_shift) > 1e-12))
    report["shift_mm"] = finite_stats(np.abs(actual_shift[np.abs(actual_shift) > 1e-12]))
    report["scale_difference_mm"] = finite_stats(
        result["scale_difference_mm"][result["available"] > 0]
    )
    report["model_p90_mm"] = finite_stats(result["model_p90_mm"][result["available"] > 0])
    report["spacing_before_mm"] = spacing_before if np.isfinite(spacing_before) else None
    report["spacing_after_mm"] = _nearest_spacing_median(corrected)
    report["robust_extent_relative_change"] = (
        (np.percentile(corrected, 99, axis=0) - np.percentile(corrected, 1, axis=0)) / extent_before
        - 1.0
    ).tolist()
    return corrected, result, report


def _postselection_regional_pose_consensus(
    points,
    normals,
    support_pose_mask,
    confidence,
    uncertainty,
    evidence_class,
    coherence_strict,
    feature_like,
    voxel,
    args,
):
    """Reduce ondulación regional solo cuando varias procedencias de pose coinciden.

    El punto central NO participa en el ajuste. Se predice desde una corona vecina
    en dos escalas. Para cada pose que respalda el punto se ajusta una superficie
    cuadrática local usando exclusivamente vecinos cuya procedencia incluye esa
    pose. Después se exige acuerdo entre poses angularmente independientes y entre
    ambas escalas. Si el acuerdo falla no se inventa geometría: se marca ambigüedad
    y se reduce su autoridad aguas abajo.

    No se ajustan cilindros, planos globales, esferas ni dimensiones conocidas.
    """
    points = np.asarray(points, dtype=np.float64)
    normals = normalize_rows(np.asarray(normals, dtype=np.float64))
    masks = np.asarray(support_pose_mask, dtype=np.uint64)
    confidence = np.asarray(confidence, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    evidence = np.asarray(evidence_class, dtype=np.uint8)
    coherence_strict = np.asarray(coherence_strict, dtype=bool)
    feature_like = np.asarray(feature_like, dtype=bool)
    n = len(points)
    result = {
        "available": np.zeros(n, dtype=np.uint8),
        "accept": np.ones(n, dtype=np.uint8),
        "ambiguous": np.zeros(n, dtype=np.uint8),
        "feature_protected": np.zeros(n, dtype=np.uint8),
        "shift_mm": np.zeros(n, dtype=np.float64),
        "score": np.ones(n, dtype=np.float64),
        "independent_predictions": np.zeros(n, dtype=np.int16),
        "pose_dispersion_mm": np.full(n, np.nan, dtype=np.float64),
        "scale_difference_mm": np.full(n, np.nan, dtype=np.float64),
        "small_prediction_mm": np.full(n, np.nan, dtype=np.float64),
        "large_prediction_mm": np.full(n, np.nan, dtype=np.float64),
    }
    report = {
        "enabled": bool(getattr(args, "regional_pose_consensus", True)),
        "evaluated": 0,
        "available": 0,
        "corrected": 0,
        "ambiguous": 0,
        "feature_protected": 0,
        "accepted_global_alpha": 0.0,
        "shift_mm": finite_stats([]),
        "pose_dispersion_mm": finite_stats([]),
        "scale_difference_mm": finite_stats([]),
        "method": "two_scale_annular_quadratic_prediction_by_pose_provenance",
        "shape_assumptions_used": False,
    }
    if n == 0 or not report["enabled"]:
        return points.copy(), result, report

    small_radius = max(
        1.5, float(getattr(args, "regional_consensus_small_radius_voxels", 3.2))
    ) * float(voxel)
    large_radius = max(
        small_radius / float(voxel) + 0.5,
        float(getattr(args, "regional_consensus_large_radius_voxels", 5.0)),
    ) * float(voxel)
    inner_radius = max(
        0.25, float(getattr(args, "regional_consensus_inner_radius_voxels", 1.0))
    ) * float(voxel)
    max_neighbors = max(32, int(getattr(args, "regional_consensus_max_neighbors", 112)))
    min_neighbors = max(8, int(getattr(args, "regional_consensus_min_neighbors", 10)))
    min_pose_neighbors = max(6, int(getattr(args, "regional_consensus_min_pose_neighbors", 7)))
    cos_same = float(
        np.cos(
            np.deg2rad(
                np.clip(
                    float(getattr(args, "regional_consensus_normal_angle_deg", 28.0)), 5.0, 60.0
                )
            )
        )
    )
    max_normal_p90 = float(getattr(args, "regional_consensus_max_normal_p90_deg", 38.0))
    min_bins = max(3, int(getattr(args, "regional_consensus_min_tangent_bins", 5)))
    min_independent = max(
        2, int(getattr(args, "regional_consensus_min_independent_predictions", 2))
    )
    min_pose_sep = max(1, int(getattr(args, "minimum_independent_pose_separation", 2)))
    fit_voxel = max(0.1, float(getattr(args, "regional_consensus_fit_p90_voxel", 0.60)))
    fit_unc = max(0.5, float(getattr(args, "regional_consensus_fit_uncertainty_factor", 2.8)))
    scale_voxel = max(0.05, float(getattr(args, "regional_consensus_scale_gate_voxel", 0.20)))
    scale_unc = max(
        0.5, float(getattr(args, "regional_consensus_scale_gate_uncertainty_factor", 1.25))
    )
    disp_voxel = max(0.05, float(getattr(args, "regional_consensus_pose_dispersion_voxel", 0.18)))
    disp_unc = max(
        0.5, float(getattr(args, "regional_consensus_pose_dispersion_uncertainty_factor", 1.20))
    )
    dead_voxel = max(0.0, float(getattr(args, "regional_consensus_deadband_voxel", 0.08)))
    dead_unc = max(
        0.0, float(getattr(args, "regional_consensus_deadband_uncertainty_factor", 0.45))
    )
    max_shift_voxel = max(0.0, float(getattr(args, "regional_consensus_max_shift_voxel", 0.22)))
    max_shift_unc = max(
        0.0, float(getattr(args, "regional_consensus_max_shift_uncertainty_factor", 1.25))
    )
    update_alpha = float(np.clip(getattr(args, "regional_consensus_update_alpha", 0.75), 0.0, 1.0))

    tree = cKDTree(points)
    k = min(max_neighbors, n)
    distances, neighbors = tree.query(points, k=k, workers=query_threads())
    raw_shift = np.zeros(n, dtype=np.float64)
    candidates = np.flatnonzero((evidence >= 3) | ((evidence == 2) & coherence_strict))
    for i in candidates:
        report["evaluated"] += 1
        u0 = (
            float(uncertainty[i])
            if np.isfinite(uncertainty[i]) and uncertainty[i] > 0
            else 0.16 * float(voxel)
        )
        ids = np.asarray(neighbors[i, 1:], dtype=np.int64)
        dist = np.asarray(distances[i, 1:], dtype=np.float64)
        valid = (
            (dist <= large_radius)
            & (dist >= inner_radius)
            & (evidence[ids] >= 2)
            & (confidence[ids] >= 0.52)
            & ((normals[ids] @ normals[i]) >= cos_same)
        )
        ids, dist = ids[valid], dist[valid]
        if len(ids) < min_neighbors:
            continue
        normal_angles = np.degrees(np.arccos(np.clip(normals[ids] @ normals[i], -1.0, 1.0)))
        if float(np.percentile(normal_angles, 90)) > max_normal_p90:
            # Un cambio real de familias de normales tiene prioridad sobre el
            # deseo de suavizar. La detección existente decide si es rasgo.
            if _feature_family_rescue(
                points, normals, uncertainty, int(i), ids, float(voxel), args
            ):
                result["feature_protected"][i] = 1
                report["feature_protected"] += 1
            continue
        ni = normals[i]
        axis = np.array([1.0, 0.0, 0.0]) if abs(float(ni[0])) < 0.8 else np.array([0.0, 1.0, 0.0])
        tangent = np.cross(ni, axis)
        tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
        bitangent = np.cross(ni, tangent)
        delta = points[ids] - points[i]
        uu = delta @ tangent
        vv = delta @ bitangent
        zz = delta @ ni
        if _tangent_coverage_bins(uu, vv, 8) < min_bins:
            continue

        predictions, pose_ids, small_predictions, large_predictions = [], [], [], []
        for pose in _pose_ids_from_mask(masks[i], expected=25):
            supported = ((masks[ids] >> np.uint64(pose)) & np.uint64(1)) != 0
            if int(np.count_nonzero(supported)) < min_pose_neighbors:
                continue
            small = supported & (dist <= small_radius)
            if int(np.count_nonzero(small)) < min_pose_neighbors:
                continue
            w_small = confidence[ids[small]] * np.exp(-0.5 * np.square(dist[small] / small_radius))
            w_large = confidence[ids[supported]] * np.exp(
                -0.5 * np.square(dist[supported] / large_radius)
            )
            fit_small = _ring_quadratic_prediction(
                uu[small], vv[small], zz[small], w_small, small_radius, voxel
            )
            fit_large = _ring_quadratic_prediction(
                uu[supported], vv[supported], zz[supported], w_large, large_radius, voxel
            )
            if fit_small is None or fit_large is None:
                continue
            fit_gate = max(fit_voxel * float(voxel), fit_unc * u0)
            if (
                fit_small["p90_residual_mm"] > fit_gate
                or fit_large["p90_residual_mm"] > 1.15 * fit_gate
                or fit_small["condition"] > 150.0
                or fit_large["condition"] > 150.0
            ):
                continue
            scale_gate = max(scale_voxel * float(voxel), scale_unc * u0)
            ps = float(fit_small["prediction_mm"])
            pl = float(fit_large["prediction_mm"])
            if abs(ps - pl) > scale_gate:
                continue
            predictions.append(0.60 * ps + 0.40 * pl)
            small_predictions.append(ps)
            large_predictions.append(pl)
            pose_ids.append(int(pose))
        if len(predictions) < min_independent:
            continue
        independent, _ = _pose_diversity(pose_ids, expected=25, minimum_separation=min_pose_sep)
        if independent < min_independent:
            continue
        pred = np.asarray(predictions, dtype=np.float64)
        ps = np.asarray(small_predictions, dtype=np.float64)
        pl = np.asarray(large_predictions, dtype=np.float64)
        median_prediction = float(np.median(pred))
        pose_dispersion = float(np.percentile(np.abs(pred - median_prediction), 90))
        small_median = float(np.median(ps))
        large_median = float(np.median(pl))
        scale_difference = abs(small_median - large_median)
        pose_gate = max(disp_voxel * float(voxel), disp_unc * u0)
        scale_gate = max(scale_voxel * float(voxel), 1.10 * u0)
        result["available"][i] = 1
        result["independent_predictions"][i] = int(independent)
        result["pose_dispersion_mm"][i] = pose_dispersion
        result["scale_difference_mm"][i] = scale_difference
        result["small_prediction_mm"][i] = small_median
        result["large_prediction_mm"][i] = large_median
        result["score"][i] = float(
            np.clip(
                np.exp(-0.5 * np.square(pose_dispersion / max(pose_gate, 1e-9)))
                * np.exp(-0.5 * np.square(scale_difference / max(scale_gate, 1e-9))),
                0.0,
                1.0,
            )
        )
        report["available"] += 1
        if pose_dispersion > pose_gate or scale_difference > scale_gate:
            result["accept"][i] = 0
            result["ambiguous"][i] = 1
            report["ambiguous"] += 1
            continue
        deadband = max(dead_voxel * float(voxel), dead_unc * u0)
        if abs(median_prediction) <= deadband:
            continue
        cap = min(max_shift_voxel * float(voxel), max_shift_unc * u0)
        raw_shift[i] = float(np.clip(update_alpha * median_prediction, -cap, cap))

    # Todos los movimientos se aceptan/reducen conjuntamente. Esto evita que
    # miles de correcciones pequeñas colapsen el spacing o cambien la escala.
    spacing_before = _nearest_spacing_median(points)
    q01_before, q99_before = np.percentile(points, [1, 99], axis=0)
    extent_before = np.maximum(q99_before - q01_before, 1e-9)
    minimum_spacing_ratio = float(
        np.clip(getattr(args, "regional_consensus_min_spacing_ratio", 0.92), 0.75, 1.0)
    )
    maximum_extent_change = max(
        0.001, float(getattr(args, "regional_consensus_max_extent_change_fraction", 0.0075))
    )
    accepted_alpha = 0.0
    corrected = points.copy()
    for alpha in (1.0, 0.75, 0.50, 0.25, 0.0):
        candidate = points + (alpha * raw_shift)[:, None] * normals
        spacing_after = _nearest_spacing_median(candidate)
        q01_after, q99_after = np.percentile(candidate, [1, 99], axis=0)
        extent_after = np.maximum(q99_after - q01_after, 1e-9)
        relative_extent_change = float(np.max(np.abs(extent_after / extent_before - 1.0)))
        spacing_ok = (
            not np.isfinite(spacing_before)
            or not np.isfinite(spacing_after)
            or spacing_after >= minimum_spacing_ratio * spacing_before
        )
        if spacing_ok and relative_extent_change <= maximum_extent_change:
            corrected = candidate
            accepted_alpha = float(alpha)
            break
    actual_shift = accepted_alpha * raw_shift
    result["shift_mm"][:] = actual_shift
    report["accepted_global_alpha"] = accepted_alpha
    report["corrected"] = int(np.count_nonzero(np.abs(actual_shift) > 1e-12))
    report["shift_mm"] = finite_stats(np.abs(actual_shift[np.abs(actual_shift) > 1e-12]))
    report["pose_dispersion_mm"] = finite_stats(
        result["pose_dispersion_mm"][result["available"] > 0]
    )
    report["scale_difference_mm"] = finite_stats(
        result["scale_difference_mm"][result["available"] > 0]
    )
    report["spacing_before_mm"] = spacing_before if np.isfinite(spacing_before) else None
    report["spacing_after_mm"] = _nearest_spacing_median(corrected)
    report["robust_extent_relative_change"] = (
        (np.percentile(corrected, 99, axis=0) - np.percentile(corrected, 1, axis=0)) / extent_before
        - 1.0
    ).tolist()
    return corrected, result, report


def _selection_coverage_metrics(points, keep, reference, voxel):
    """Cobertura de muestreo respecto a candidatos observados.

    No estima área superficial verdadera. Solo mide qué tan lejos quedaron los
    candidatos observados de una muestra retenida y qué fracción de celdas
    ocupadas continúa representada.
    """
    points = np.asarray(points, dtype=np.float64)
    keep = np.asarray(keep, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    out = {
        "reference_points": int(np.count_nonzero(reference)),
        "selected_points": int(np.count_nonzero(keep)),
        "distance_to_retained_mm": finite_stats([]),
        "spatially_uncovered_ratio": {"1_voxels": None, "2_voxels": None, "4_voxels": None},
        "occupied_cell_retention_ratio": 0.0,
    }
    if not np.any(reference) or not np.any(keep):
        return out
    tree = cKDTree(points[keep])
    distance = tree.query(points[reference], workers=query_threads())[0]
    out["distance_to_retained_mm"] = finite_stats(distance)
    out["spatially_uncovered_ratio"] = {
        f"{f}_voxels": float(np.mean(distance > f * float(voxel))) for f in (1, 2, 4)
    }
    cells_ref = np.unique(
        np.floor(points[reference] / float(voxel)).astype(np.int64), axis=0
    )
    cells_keep = np.unique(
        np.floor(points[keep] / float(voxel)).astype(np.int64), axis=0
    )
    out["occupied_cell_retention_ratio"] = float(
        len(cells_keep) / max(len(cells_ref), 1)
    )
    return out


def _quality_snapshot(mask, support, confidence, normal_consistency, uncertainty, agreement):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return {
            "points": 0,
            "support": finite_stats([]),
            "confidence": finite_stats([]),
            "normal_consistency": finite_stats([]),
            "uncertainty_mm": finite_stats([]),
            "agreement": finite_stats([]),
        }
    return {
        "points": int(np.count_nonzero(mask)),
        "support": finite_stats(np.asarray(support)[mask]),
        "confidence": finite_stats(np.asarray(confidence)[mask]),
        "normal_consistency": finite_stats(np.asarray(normal_consistency)[mask]),
        "uncertainty_mm": finite_stats(np.asarray(uncertainty)[mask]),
        "agreement": finite_stats(np.asarray(agreement)[mask]),
    }


def _recover_observed_surface_coverage(
    points,
    normals,
    support,
    confidence,
    normal_consistency,
    uncertainty,
    agreement,
    independent,
    angular_span,
    conflict_ratio,
    evidence_class_pre,
    evidence_class,
    coherence,
    keep,
    voxel,
    minimum_confidence,
    args,
):
    """Recupera cobertura usando únicamente observaciones ya existentes.

    Esta pasada NO interpola puntos, NO desplaza observaciones y NO conoce la
    forma del objeto. Reconsidera candidatos descartados por el filtro local si
    mantienen evidencia multivista suficiente y pueden conectarse de forma
    geométricamente compatible con superficie ya validada.

    La expansión es iterativa pero acotada: cada candidato debe tener evidencia
    propia, respaldo local y compatibilidad de normal/residuo tangencial. Al
    final se compara cobertura y calidad global; si la recuperación no mejora
    cobertura o deteriora demasiado la calidad, se revierte por completo.
    """
    n = len(points)
    keep = np.asarray(keep, dtype=bool).copy()
    initial_keep = keep.copy()
    recovered = np.zeros(n, dtype=bool)
    recovered_iteration = np.zeros(n, dtype=np.uint8)
    recovery_score = np.zeros(n, dtype=np.float64)
    final_evidence = np.asarray(evidence_class, dtype=np.uint8).copy()

    enabled = bool(getattr(args, "observed_coverage_recovery", True))
    reference = (
        np.all(np.isfinite(points), axis=1)
        & (np.asarray(support) >= 2)
        & np.isfinite(confidence)
        & (np.asarray(confidence) >= float(minimum_confidence))
    )
    before_cov = _selection_coverage_metrics(points, initial_keep, reference, voxel)
    before_quality = _quality_snapshot(
        initial_keep,
        support,
        confidence,
        normal_consistency,
        uncertainty,
        agreement,
    )
    report = {
        "enabled": enabled,
        "policy": (
            "observed_only_multiview_connected_same_sheet_recovery_"
            "with_global_quality_guard"
        ),
        "interpolated_points": 0,
        "synthetic_points": 0,
        "initial_selected_points": int(np.count_nonzero(initial_keep)),
        "candidate_points": 0,
        "proposed_recovered_points": 0,
        "accepted_recovered_points": 0,
        "accepted": False,
        "reason": None,
        "iterations": [],
        "coverage_before": before_cov,
        "coverage_after": before_cov,
        "quality_before": before_quality,
        "quality_after": before_quality,
    }
    diagnostic = {
        "coverage_recovery_candidate": np.zeros(n, dtype=np.uint8),
        "coverage_recovered_observation": recovered.astype(np.uint8),
        "coverage_recovery_iteration": recovered_iteration,
        "coverage_recovery_score": recovery_score,
    }
    if not enabled:
        report["reason"] = "disabled"
        return keep, final_evidence, report, diagnostic
    if np.count_nonzero(initial_keep) < 3:
        report["reason"] = "insufficient_retained_surface"
        return keep, final_evidence, report, diagnostic

    points = np.asarray(points, dtype=np.float64)
    normals = normalize_rows(np.asarray(normals, dtype=np.float64))
    support = np.asarray(support, dtype=np.int16)
    confidence = np.asarray(confidence, dtype=np.float64)
    normal_consistency = np.asarray(normal_consistency, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    agreement = np.asarray(agreement, dtype=np.float64)
    independent = np.asarray(independent, dtype=np.int16)
    angular_span = np.asarray(angular_span, dtype=np.int16)
    conflict_ratio = np.asarray(conflict_ratio, dtype=np.float64)
    pre = np.asarray(evidence_class_pre, dtype=np.uint8)
    coh_accept = np.asarray(coherence.get("accept", np.ones(n)), dtype=bool)
    coh_score = np.asarray(coherence.get("score", np.ones(n)), dtype=np.float64)
    coh_flags = np.asarray(coherence.get("red_flags", np.zeros(n)), dtype=np.uint8)

    finite = (
        np.all(np.isfinite(points), axis=1)
        & np.all(np.isfinite(normals), axis=1)
        & np.isfinite(confidence)
        & np.isfinite(normal_consistency)
        & np.isfinite(uncertainty)
        & np.isfinite(agreement)
        & np.isfinite(conflict_ratio)
    )
    min_support = max(2, int(getattr(args, "coverage_recovery_min_support", 2)))
    min_independent = max(
        2, int(getattr(args, "coverage_recovery_min_independent_support", 2))
    )
    min_span = max(2, int(getattr(args, "coverage_recovery_min_angular_span", 2)))
    min_conf = max(
        float(minimum_confidence),
        float(getattr(args, "coverage_recovery_min_confidence", 0.60)),
    )
    min_agreement = float(getattr(args, "coverage_recovery_min_agreement", 0.60))
    min_normal_consistency = float(
        getattr(args, "coverage_recovery_min_normal_consistency", 0.86)
    )
    max_conflict = float(getattr(args, "coverage_recovery_max_conflict_ratio", 0.32))
    max_uncertainty = (
        float(getattr(args, "coverage_recovery_max_uncertainty_voxel", 0.55))
        * float(voxel)
    )

    rejected_class2 = (pre == 2) & (~coh_accept)
    rejected_rescue = (
        rejected_class2
        & (
            coh_score
            >= float(
                getattr(args, "coverage_recovery_rejected_class2_min_score", 0.58)
            )
        )
        & (
            coh_flags
            <= int(
                getattr(args, "coverage_recovery_rejected_class2_max_red_flags", 4)
            )
        )
        & (
            support
            >= int(
                getattr(args, "coverage_recovery_rejected_class2_min_support", 3)
            )
        )
        & (
            confidence
            >= float(
                getattr(args, "coverage_recovery_rejected_class2_min_confidence", 0.70)
            )
        )
        & (agreement >= max(min_agreement, 0.66))
        & (normal_consistency >= max(min_normal_consistency, 0.90))
    )
    evidence_ok = (pre >= 3) | ((pre == 2) & coh_accept) | rejected_rescue
    candidate = (
        (~initial_keep)
        & finite
        & evidence_ok
        & (support >= min_support)
        & (independent >= min_independent)
        & (angular_span >= min_span)
        & (confidence >= min_conf)
        & (agreement >= min_agreement)
        & (normal_consistency >= min_normal_consistency)
        & (conflict_ratio <= max_conflict)
        & (uncertainty <= max_uncertainty)
    )

    candidate_idx = np.flatnonzero(candidate)
    report["candidate_points"] = int(len(candidate_idx))
    diagnostic["coverage_recovery_candidate"][candidate_idx] = 1
    if len(candidate_idx) == 0:
        report["reason"] = "no_candidates_with_required_evidence"
        return keep, final_evidence, report, diagnostic

    # Un candidato aislado no puede iniciar una cadena. Se exige respaldo local
    # entre observaciones elegibles o, alternativamente, evidencia multivista
    # particularmente fuerte.
    candidate_radius = (
        max(
            0.5,
            float(getattr(args, "coverage_recovery_candidate_radius_voxels", 2.10)),
        )
        * float(voxel)
    )
    evidence_pool = initial_keep | candidate
    pool_idx = np.flatnonzero(evidence_pool)
    pool_tree = cKDTree(points[pool_idx])
    local_counts = (
        pool_tree.query_ball_point(
            points[candidate_idx],
            r=candidate_radius,
            workers=query_threads(),
            return_length=True,
        ).astype(np.int32)
        - 1
    )
    min_candidate_neighbors = max(
        1, int(getattr(args, "coverage_recovery_min_candidate_neighbors", 2))
    )
    locally_backed = local_counts >= min_candidate_neighbors
    locally_backed |= (support[candidate_idx] >= 4) & (
        confidence[candidate_idx] >= 0.78
    )
    candidate_idx = candidate_idx[locally_backed]
    if len(candidate_idx) == 0:
        report["reason"] = "candidates_not_locally_backed"
        return keep, final_evidence, report, diagnostic

    # Puntuación solo para limitar una recuperación excesiva; no define forma.
    denom_conflict = max(max_conflict, 1e-6)
    recovery_score_all = (
        0.20 * np.clip(confidence, 0.0, 1.0)
        + 0.16 * np.clip(normal_consistency, 0.0, 1.0)
        + 0.16 * np.clip(agreement, 0.0, 1.0)
        + 0.12 * np.clip(support.astype(np.float64) / 5.0, 0.0, 1.0)
        + 0.10 * np.clip(independent.astype(np.float64) / 4.0, 0.0, 1.0)
        + 0.10 * np.clip(coh_score, 0.0, 1.0)
        + 0.08 * np.clip(1.0 - conflict_ratio / denom_conflict, 0.0, 1.0)
        + 0.08
        * np.exp(
            -0.5
            * (
                uncertainty / np.maximum(0.40 * float(voxel), 1e-9)
            )
            ** 2
        )
    )
    recovery_score[:] = np.nan_to_num(
        recovery_score_all, nan=0.0, posinf=0.0, neginf=0.0
    )

    max_fraction = float(
        np.clip(
            getattr(args, "coverage_recovery_max_fraction_of_initial", 0.60),
            0.0,
            2.0,
        )
    )
    maximum_recovery = max(
        1,
        int(
            math.ceil(
                max_fraction * max(np.count_nonzero(initial_keep), 1)
            )
        ),
    )
    iterations = max(1, int(getattr(args, "coverage_recovery_iterations", 4)))
    radius = (
        max(
            0.5,
            float(getattr(args, "coverage_recovery_max_distance_voxels", 2.35)),
        )
        * float(voxel)
    )
    normal_cos = math.cos(
        math.radians(float(getattr(args, "coverage_recovery_normal_angle_deg", 42.0)))
    )
    min_kept_neighbors = max(
        1, int(getattr(args, "coverage_recovery_min_kept_neighbors", 1))
    )
    tangent_voxel = max(
        0.0, float(getattr(args, "coverage_recovery_tangent_residual_voxel", 0.42))
    )
    tangent_unc = max(
        0.0,
        float(getattr(args, "coverage_recovery_tangent_uncertainty_factor", 2.0)),
    )

    remaining_mask = np.zeros(n, dtype=bool)
    remaining_mask[candidate_idx] = True
    for iteration in range(1, iterations + 1):
        remaining = np.flatnonzero(remaining_mask & (~keep))
        if len(remaining) == 0 or np.count_nonzero(recovered) >= maximum_recovery:
            break
        current_idx = np.flatnonzero(keep)
        tree = cKDTree(points[current_idx])
        k = min(max(8, min_kept_neighbors * 4), len(current_idx))
        distances, neighbor_local = tree.query(
            points[remaining],
            k=k,
            distance_upper_bound=radius,
            workers=query_threads(),
        )
        if np.ndim(distances) == 1:
            distances = distances[:, None]
            neighbor_local = neighbor_local[:, None]
        valid_nb = np.isfinite(distances) & (neighbor_local < len(current_idx))
        safe_local = np.clip(neighbor_local, 0, max(len(current_idx) - 1, 0))
        neighbor_global = current_idx[safe_local]
        cand_n = normals[remaining][:, None, :]
        neigh_n = normals[neighbor_global]
        raw_dot = np.einsum("ijk,ijk->ij", cand_n, neigh_n)
        aligned_neigh_n = np.where(raw_dot[..., None] < 0.0, -neigh_n, neigh_n)
        middle = cand_n + aligned_neigh_n
        middle_len = np.linalg.norm(middle, axis=2)
        middle = middle / np.maximum(middle_len[..., None], 1e-12)
        delta = points[neighbor_global] - points[remaining][:, None, :]
        tangential_residual = np.abs(np.einsum("ijk,ijk->ij", delta, middle))
        neighbor_unc = uncertainty[neighbor_global]
        pair_unc = np.minimum(uncertainty[remaining][:, None], neighbor_unc)
        tolerance = np.maximum(
            tangent_voxel * float(voxel), tangent_unc * pair_unc
        )
        same_sheet = (
            valid_nb
            & (np.abs(raw_dot) >= normal_cos)
            & (tangential_residual <= tolerance)
        )
        same_count = np.sum(same_sheet, axis=1)
        required = np.full(len(remaining), min_kept_neighbors, dtype=np.int32)
        # Una clase 2 que el test multiescala había rechazado necesita dos
        # enlaces independientes a superficie ya retenida para regresar.
        rejected_rows = rejected_class2[remaining]
        required[rejected_rows] = np.maximum(required[rejected_rows], 2)
        chosen = remaining[same_count >= required]
        if len(chosen) == 0:
            report["iterations"].append(
                {
                    "iteration": int(iteration),
                    "candidates_tested": int(len(remaining)),
                    "accepted": 0,
                }
            )
            break
        remaining_budget = maximum_recovery - int(np.count_nonzero(recovered))
        if len(chosen) > remaining_budget:
            order = np.argsort(recovery_score[chosen])[::-1][:remaining_budget]
            chosen = chosen[order]
        keep[chosen] = True
        recovered[chosen] = True
        recovered_iteration[chosen] = np.uint8(min(iteration, 255))
        report["iterations"].append(
            {
                "iteration": int(iteration),
                "candidates_tested": int(len(remaining)),
                "accepted": int(len(chosen)),
                "recovered_total": int(np.count_nonzero(recovered)),
            }
        )

    proposed_count = int(np.count_nonzero(recovered))
    report["proposed_recovered_points"] = proposed_count
    if proposed_count == 0:
        report["reason"] = "no_candidate_connected_to_retained_surface"
        return initial_keep, final_evidence, report, diagnostic

    proposed_cov = _selection_coverage_metrics(points, keep, reference, voxel)
    proposed_quality = _quality_snapshot(
        keep,
        support,
        confidence,
        normal_consistency,
        uncertainty,
        agreement,
    )
    report["coverage_after"] = proposed_cov
    report["quality_after"] = proposed_quality

    def _metric(record, section, name):
        value = record.get(section, {}).get(name)
        return None if value is None or not np.isfinite(value) else float(value)

    before_u1 = before_cov["spatially_uncovered_ratio"].get("1_voxels")
    before_u2 = before_cov["spatially_uncovered_ratio"].get("2_voxels")
    after_u1 = proposed_cov["spatially_uncovered_ratio"].get("1_voxels")
    after_u2 = proposed_cov["spatially_uncovered_ratio"].get("2_voxels")
    improvement_1 = (
        float(before_u1) - float(after_u1)
        if before_u1 is not None and after_u1 is not None
        else 0.0
    )
    improvement_2 = (
        float(before_u2) - float(after_u2)
        if before_u2 is not None and after_u2 is not None
        else 0.0
    )
    cell_improvement = float(
        proposed_cov.get("occupied_cell_retention_ratio", 0.0)
        - before_cov.get("occupied_cell_retention_ratio", 0.0)
    )
    min_improvement = max(
        0.0,
        float(getattr(args, "coverage_recovery_min_coverage_improvement", 0.005)),
    )
    coverage_ok = (
        max(improvement_1, improvement_2, cell_improvement) >= min_improvement
    )

    before_conf = _metric(before_quality, "confidence", "median")
    after_conf = _metric(proposed_quality, "confidence", "median")
    before_norm = _metric(before_quality, "normal_consistency", "median")
    after_norm = _metric(proposed_quality, "normal_consistency", "median")
    before_unc = _metric(before_quality, "uncertainty_mm", "p90")
    after_unc = _metric(proposed_quality, "uncertainty_mm", "p90")
    max_conf_drop = max(
        0.0, float(getattr(args, "coverage_recovery_max_confidence_drop", 0.06))
    )
    max_norm_drop = max(
        0.0,
        float(
            getattr(args, "coverage_recovery_max_normal_consistency_drop", 0.04)
        ),
    )
    max_unc_factor = max(
        1.0,
        float(
            getattr(args, "coverage_recovery_max_uncertainty_p90_factor", 1.35)
        ),
    )
    confidence_ok = (
        before_conf is None
        or after_conf is None
        or after_conf >= before_conf - max_conf_drop
    )
    normal_ok = (
        before_norm is None
        or after_norm is None
        or after_norm >= before_norm - max_norm_drop
    )
    uncertainty_ok = (
        before_unc is None
        or after_unc is None
        or after_unc <= max(before_unc * max_unc_factor, 0.65 * float(voxel))
    )
    quality_ok = bool(confidence_ok and normal_ok and uncertainty_ok)
    report["validation"] = {
        "coverage_improvement_one_voxel": float(improvement_1),
        "coverage_improvement_two_voxels": float(improvement_2),
        "occupied_cell_retention_improvement": float(cell_improvement),
        "minimum_required_improvement": float(min_improvement),
        "coverage_ok": bool(coverage_ok),
        "confidence_ok": bool(confidence_ok),
        "normal_consistency_ok": bool(normal_ok),
        "uncertainty_p90_ok": bool(uncertainty_ok),
        "quality_ok": bool(quality_ok),
    }
    if not (coverage_ok and quality_ok):
        report["reason"] = "global_validation_failed"
        report["accepted"] = False
        report["accepted_recovered_points"] = 0
        diagnostic["coverage_recovery_iteration"] = recovered_iteration
        diagnostic["coverage_recovery_score"] = recovery_score
        return (
            initial_keep,
            np.asarray(evidence_class, dtype=np.uint8).copy(),
            report,
            diagnostic,
        )

    # Aceptada: una clase 2 recuperada sigue siendo clase 2; una clase 3
    # validada held-out conserva su clase. La recuperación nunca eleva autoridad.
    final_evidence[recovered & (pre == 2)] = 2
    final_evidence[recovered & (pre >= 3)] = np.maximum(
        final_evidence[recovered & (pre >= 3)], 3
    )
    report["accepted"] = True
    report["reason"] = "coverage_improved_without_material_quality_degradation"
    report["accepted_recovered_points"] = proposed_count
    report["recovered_rejected_class2"] = int(
        np.count_nonzero(recovered & rejected_class2)
    )
    report["recovered_validated_fit"] = int(
        np.count_nonzero(recovered & (pre >= 3))
    )
    report["recovered_accepted_class2"] = int(
        np.count_nonzero(recovered & (pre == 2) & coh_accept)
    )
    diagnostic["coverage_recovered_observation"] = recovered.astype(np.uint8)
    diagnostic["coverage_recovery_iteration"] = recovered_iteration
    diagnostic["coverage_recovery_score"] = recovery_score
    return keep, final_evidence, report, diagnostic


def select_independent_patches(
    points,
    normals,
    support,
    confidence,
    spread,
    normal_consistency,
    local_result,
    voxel,
    minimum_confidence,
    args=None,
):
    """Grafo de superficie con evidencia propia, diversidad angular y ambigüedad explícita."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(points)
    uncertainty = np.asarray(local_result["uncertainty"] if local_result is not None else spread)
    agreement = np.asarray(local_result["agreement"] if local_result is not None else np.ones(n))
    if local_result is not None:
        status = np.asarray(local_result["status"])
        independent = np.asarray(
            local_result.get("independent_support_poses", support), dtype=np.int16
        )
        angular_span = np.asarray(
            local_result.get("support_angular_span_poses", np.zeros(n)), dtype=np.int16
        )
        conflict_ratio = np.asarray(
            local_result.get("conflict_pose_ratio", np.zeros(n)), dtype=np.float64
        )
        heldout_available = np.asarray(
            local_result.get("heldout_validation_available", np.zeros(n)), dtype=bool
        )
        heldout_tests = np.asarray(
            local_result.get("heldout_validation_tests", np.zeros(n)), dtype=np.int16
        )
        heldout_pass_count = np.asarray(
            local_result.get(
                "heldout_validation_pass_count",
                np.rint(
                    np.asarray(local_result.get("heldout_validation_pass_ratio", np.zeros(n)))
                    * np.maximum(heldout_tests, 0)
                ),
            ),
            dtype=np.int16,
        )
        heldout_pass = np.asarray(
            local_result.get("heldout_validation_pass_ratio", np.zeros(n)), dtype=np.float64
        )
        layer_available = np.asarray(
            local_result.get("validated_layer_available", np.zeros(n)), dtype=bool
        )
        layer_accept = np.asarray(
            local_result.get("validated_layer_accept", np.ones(n)), dtype=bool
        )
        layer_ambiguous = np.asarray(
            local_result.get("validated_layer_ambiguous", np.zeros(n)), dtype=bool
        )
        layer_resolved = np.asarray(
            local_result.get("validated_layer_resolved", np.zeros(n)), dtype=bool
        )
    else:
        status = np.ones(n, dtype=np.uint8)
        independent = np.asarray(support, dtype=np.int16)
        angular_span = np.asarray(support, dtype=np.int16)
        conflict_ratio = np.zeros(n, dtype=np.float64)
        heldout_available = np.zeros(n, dtype=bool)
        heldout_tests = np.zeros(n, dtype=np.int16)
        heldout_pass_count = np.zeros(n, dtype=np.int16)
        heldout_pass = np.ones(n, dtype=np.float64)
        layer_available = np.zeros(n, dtype=bool)
        layer_accept = np.ones(n, dtype=bool)
        layer_ambiguous = np.zeros(n, dtype=bool)
        layer_resolved = np.zeros(n, dtype=bool)

    min_independent = int(getattr(args, "minimum_independent_support_poses", 2))
    min_span = int(getattr(args, "minimum_support_angular_span_poses", 2))
    min_unvalidated_support = int(getattr(args, "minimum_unvalidated_support_poses", 3))
    min_unvalidated_conf = float(getattr(args, "minimum_unvalidated_confidence", 0.70))
    min_unvalidated_agreement = float(getattr(args, "minimum_unvalidated_agreement", 0.70))
    max_conflict = float(getattr(args, "maximum_conflict_pose_ratio", 0.35))
    min_heldout = float(getattr(args, "minimum_heldout_pass_ratio", 2.0 / 3.0))
    heldout_num = max(0, int(getattr(args, "minimum_heldout_pass_numerator", 2)))
    heldout_den = max(1, int(getattr(args, "minimum_heldout_pass_denominator", 3)))

    diversity_ok = (independent >= min_independent) & (angular_span >= min_span)
    heldout_ok = (~heldout_available) | (
        heldout_pass_count.astype(np.int64) * heldout_den
        >= heldout_num * heldout_tests.astype(np.int64)
    )
    validated_fit = (
        (status == 1)
        & heldout_available
        & heldout_ok
        & diversity_ok
        & (conflict_ratio <= max_conflict)
        & layer_accept
    )
    strong_observed = (
        diversity_ok
        & (support >= min_unvalidated_support)
        & (confidence >= min_unvalidated_conf)
        & (agreement >= min_unvalidated_agreement)
        & (conflict_ratio <= max_conflict)
        & (((status != 1) | (~heldout_available)) | layer_ambiguous)
    )
    # Clase 3: ajuste validado contra poses excluidas. Clase 2: observación no
    # proyectada pero con evidencia independiente fuerte. Antes de permitir que
    # una clase 2 defina superficie, V11.3 comprueba coherencia local multiescala.
    evidence_class_pre_coherence = np.where(
        validated_fit, 3, np.where(strong_observed, 2, 0)
    ).astype(np.uint8)
    coherence = _evaluate_class2_local_coherence(
        points, normals, uncertainty, evidence_class_pre_coherence, voxel, args
    )
    evidence_class = evidence_class_pre_coherence.copy()
    rejected_class2 = (evidence_class == 2) & (np.asarray(coherence["accept"], dtype=bool) == 0)
    evidence_class[rejected_class2] = 0
    eligible = (
        np.all(np.isfinite(points), axis=1)
        & np.all(np.isfinite(normals), axis=1)
        & (support >= 2)
        & (confidence >= minimum_confidence)
        & (normal_consistency >= 0.80)
        & np.isfinite(uncertainty)
        & (uncertainty <= 0.45 * voxel)
        & (agreement >= 0.60)
        & (evidence_class >= 2)
    )
    idx = np.flatnonzero(eligible)
    keep = np.zeros(n, bool)
    labels_full = np.full(n, -1, np.int32)
    degree_full = np.zeros(n, np.int32)
    components = []
    if len(idx) >= 3:
        p = points[idx]
        normal = normalize_rows(normals[idx])
        tree = cKDTree(p)
        k = min(25, len(p))
        distance, neighbors = tree.query(p, k=k, workers=query_threads())
        spacing = np.maximum(distance[:, min(3, k - 1)] / np.sqrt(3), 0.2 * voxel)
        radius = np.clip(2.0 * spacing, 1.15 * voxel, 2.25 * voxel)
        # Include both directed kNN lists before unique; a one-way neighbor
        # must not be lost just because its array index happens to be smaller.
        all_edges = np.column_stack((np.repeat(np.arange(len(p)), k - 1), neighbors[:, 1:].ravel()))
        all_edges.sort(axis=1)
        all_edges = np.unique(all_edges, axis=0)
        rows, cols = all_edges.T
        delta = p[cols] - p[rows]
        dist = np.linalg.norm(delta, axis=1)
        dot = np.einsum("ij,ij->i", normal[rows], normal[cols])
        middle_normal = normalize_rows(normal[rows] + normal[cols])
        normal_residual = np.abs(np.einsum("ij,ij->i", delta, middle_normal))
        tolerance = np.clip(
            1.5 * np.minimum(uncertainty[idx[rows]], uncertainty[idx[cols]]),
            0.12 * voxel,
            0.40 * voxel,
        )
        edge_ok = (
            (dist > 1e-8)
            & (dist <= np.minimum(radius[rows], radius[cols]))
            & (dot >= np.cos(np.deg2rad(40)))
            & (normal_residual <= tolerance)
        )
        edge_delta = delta[edge_ok]
        rows, cols = rows[edge_ok], cols[edge_ok]
        degree = np.bincount(np.r_[rows, cols], minlength=len(p))
        covariance = np.zeros((len(p), 3, 3))
        outer = edge_delta[:, :, None] * edge_delta[:, None, :]
        np.add.at(covariance, rows, outer)
        np.add.at(covariance, cols, outer)
        eigen = np.linalg.eigvalsh(covariance / np.maximum(degree[:, None, None], 1))
        surface_neighborhood = eigen[:, 1] > 0.025 * np.maximum(eigen[:, 2], 1e-12)
        surface_neighborhood |= (support[idx] >= 4) & (confidence[idx] >= 0.72)
        # Remove one-edge bridges/tails from the connected core. Boundary
        # observations can be recovered below, but cannot chain far outwards.
        core_edge = (
            (degree[rows] >= 2)
            & (degree[cols] >= 2)
            & surface_neighborhood[rows]
            & surface_neighborhood[cols]
        )
        a, b = rows[core_edge], cols[core_edge]
        graph = coo_matrix(
            (np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(len(p), len(p))
        ).tocsr()
        count, labels = connected_components(graph, directed=False)
        core_keep = np.zeros(len(p), bool)
        ordered = np.argsort(labels, kind="stable")
        boundaries = np.r_[0, np.flatnonzero(np.diff(labels[ordered])) + 1, len(ordered)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            members = ordered[start:end]
            label = int(labels[members[0]])
            own = idx[members]
            connected_ratio = float(np.mean(degree[members] >= 2))
            strong = (support[own] >= 4) & (confidence[own] >= 0.72)
            # Size alone is never sufficient. Small independently reliable
            # parts remain valid; there is no largest-component-only rule.
            accepted = connected_ratio >= 0.80 and (
                (len(own) >= 12 and np.median(confidence[own]) >= 0.55)
                or (len(own) >= 5 and np.count_nonzero(strong) >= 4)
            )
            core_keep[members] = accepted
            components.append(
                {
                    "label": int(label),
                    "points": int(len(own)),
                    "median_support": float(np.median(support[own])),
                    "median_confidence": float(np.median(confidence[own])),
                    "connected_ratio": connected_ratio,
                    "retained": bool(accepted),
                }
            )
        backed_neighbors = np.zeros(len(p), int)
        np.add.at(backed_neighbors, rows, core_keep[cols].astype(int))
        np.add.at(backed_neighbors, cols, core_keep[rows].astype(int))
        chosen = core_keep | ((backed_neighbors >= 2) & (confidence[idx] >= 0.55))
        keep[idx] = chosen
        labels_full[idx] = labels
        degree_full[idx] = degree
    # V11.8 — segunda pasada de cobertura OBSERVADA. La recuperación no
    # crea ni desplaza puntos: reincorpora únicamente candidatos ya medidos y
    # con evidencia multivista propia que siguen conectados a la superficie.
    keep, evidence_class, coverage_recovery_report, coverage_recovery_diagnostic = (
        _recover_observed_surface_coverage(
            points=points,
            normals=normals,
            support=support,
            confidence=confidence,
            normal_consistency=normal_consistency,
            uncertainty=uncertainty,
            agreement=agreement,
            independent=independent,
            angular_span=angular_span,
            conflict_ratio=conflict_ratio,
            evidence_class_pre=evidence_class_pre_coherence,
            evidence_class=evidence_class,
            coherence=coherence,
            keep=keep,
            voxel=float(voxel),
            minimum_confidence=float(minimum_confidence),
            args=args,
        )
    )

    # Una clase 2 solo puede actuar como ancla fuerte si además superó el
    # nivel estricto de coherencia local. Las clases 3 siguen siendo anclas
    # por validación held-out.
    class2_strict = np.asarray(coherence["strict"], dtype=bool)
    strong = (
        keep & (support >= 5) & ((evidence_class >= 3) | ((evidence_class == 2) & class2_strict))
    )
    # Se conserva la semántica histórica de consensus_selection_class para no
    # romper lectores: 3=núcleo fuerte por soporte, 2=superficie retenida.
    classes = np.where(strong, 3, np.where(keep, 2, 0)).astype(np.uint8)
    anchor_distance = np.full(n, np.inf)
    if np.any(strong):
        anchor_distance = cKDTree(points[strong]).query(points, workers=query_threads())[0]
    distance_to_kept = np.full(n, np.inf)
    if np.any(keep):
        distance_to_kept = cKDTree(points[keep]).query(points, workers=query_threads())[0]
    reference = (support >= 2) & (confidence >= minimum_confidence)
    cells_all = np.unique(np.floor(points[reference] / voxel).astype(np.int64), axis=0)
    cells_kept = np.unique(np.floor(points[keep] / voxel).astype(np.int64), axis=0)
    report = {
        "enabled": True,
        "mode": "independent_multiview_surface_patches",
        "anchor_support_threshold_used": 5,
        "anchor_points": int(strong.sum()),
        "local_extension_points": int(np.count_nonzero(keep & ~strong)),
        "final_points": int(keep.sum()),
        "component_records": components,
        "observed_coverage_recovery": coverage_recovery_report,
        "candidate_support_counts": {},
        "retained_support_counts": {},
        "rejected_support_counts": {},
        "coverage": {
            "reference": "candidates_with_two_views_and_minimum_confidence",
            "reference_points": int(reference.sum()),
            "occupied_cells_before": int(len(cells_all)),
            "occupied_cells_after": int(len(cells_kept)),
            "occupied_cell_retention_ratio": float(len(cells_kept) / max(len(cells_all), 1)),
            "candidate_to_retained_distance_mm": finite_stats(distance_to_kept[reference]),
            "spatially_uncovered_ratio": {
                str(f)
                + "_voxels": (
                    float(np.mean(distance_to_kept[reference] > f * voxel))
                    if reference.any()
                    else None
                )
                for f in (1, 2, 4)
            },
            "note": "Sampling coverage, not measured true surface area.",
        },
    }
    for level in np.unique(support):
        key = str(int(level))
        report["candidate_support_counts"][key] = int(np.count_nonzero(support == level))
        report["retained_support_counts"][key] = int(np.count_nonzero(keep & (support == level)))
        report["rejected_support_counts"][key] = int(np.count_nonzero(~keep & (support == level)))
    report["evidence_policy"] = {
        "minimum_independent_support_poses": int(min_independent),
        "minimum_angular_span_poses": int(min_span),
        "minimum_heldout_pass_ratio_legacy": float(min_heldout),
        "minimum_heldout_pass_fraction_exact": f"{heldout_num}/{heldout_den}",
        "maximum_conflict_pose_ratio": float(max_conflict),
        "validated_fit_candidates": int(np.count_nonzero(validated_fit)),
        "validated_pose_layer_available": int(np.count_nonzero(layer_available)),
        "validated_pose_layer_resolved": int(np.count_nonzero(layer_resolved)),
        "validated_pose_layer_ambiguous_downgraded": int(np.count_nonzero(layer_ambiguous)),
        "strong_unvalidated_candidates": int(np.count_nonzero(strong_observed)),
        "class2_pre_coherence_candidates": int(np.count_nonzero(evidence_class_pre_coherence == 2)),
        "class2_coherence_available": int(
            np.count_nonzero(
                (evidence_class_pre_coherence == 2)
                & (np.asarray(coherence["available"], dtype=bool))
            )
        ),
        "class2_coherence_strict": int(
            np.count_nonzero(
                (evidence_class_pre_coherence == 2) & (np.asarray(coherence["strict"], dtype=bool))
            )
        ),
        "class2_coherence_warning": int(
            np.count_nonzero(
                (evidence_class_pre_coherence == 2)
                & (np.asarray(coherence["accept"], dtype=bool))
                & ~(np.asarray(coherence["strict"], dtype=bool))
            )
        ),
        "class2_coherence_rejected": int(np.count_nonzero(rejected_class2)),
        "class2_feature_rescued": int(
            np.count_nonzero(
                (evidence_class_pre_coherence == 2)
                & (np.asarray(coherence["feature_like"], dtype=bool))
            )
        ),
        "observed_coverage_recovery_accepted": bool(
            coverage_recovery_report.get("accepted", False)
        ),
        "observed_coverage_recovered_points": int(
            coverage_recovery_report.get("accepted_recovered_points", 0)
        ),
        "ambiguous_or_insufficient_candidates": int(np.count_nonzero(evidence_class == 0)),
        "local_coherence_note": (
            "Solo clase 2 se somete a coherencia multiescala. Se rechaza únicamente por múltiples "
            "fallos independientes; aristas/esquinas coherentes se preservan por familias de normales."
        ),
    }
    diagnostic = {
        "patch_label": labels_full,
        "compatible_neighbor_count": degree_full,
        "eligible_by_own_evidence": eligible,
        "distance_to_retained_mm": distance_to_kept,
        "surface_evidence_class": evidence_class,
        "independent_support_poses": independent,
        "support_angular_span_poses": angular_span,
        "conflict_pose_ratio": conflict_ratio,
        "heldout_validation_available": heldout_available,
        "heldout_validation_tests": heldout_tests,
        "heldout_validation_pass_count": heldout_pass_count,
        "heldout_validation_pass_ratio": heldout_pass,
        "validated_layer_available": layer_available.astype(np.uint8),
        "validated_layer_accept": layer_accept.astype(np.uint8),
        "validated_layer_ambiguous": layer_ambiguous.astype(np.uint8),
        "validated_layer_resolved": layer_resolved.astype(np.uint8),
        "surface_evidence_class_pre_coherence": evidence_class_pre_coherence,
        "surface_local_coherence_available": np.asarray(coherence["available"], dtype=np.uint8),
        "surface_local_coherence_accept": np.asarray(coherence["accept"], dtype=np.uint8),
        "surface_local_coherence_strict": np.asarray(coherence["strict"], dtype=np.uint8),
        "surface_local_feature_like": np.asarray(coherence["feature_like"], dtype=np.uint8),
        "surface_local_coherence_score": np.asarray(coherence["score"], dtype=np.float64),
        "surface_local_coherence_red_flags": np.asarray(coherence["red_flags"], dtype=np.uint8),
        "surface_local_plane_p90_small_mm": np.asarray(
            coherence["plane_p90_small_mm"], dtype=np.float64
        ),
        "surface_local_plane_p90_large_mm": np.asarray(
            coherence["plane_p90_large_mm"], dtype=np.float64
        ),
        "surface_local_normal_p90_large_deg": np.asarray(
            coherence["normal_p90_large_deg"], dtype=np.float64
        ),
        "surface_local_scale_normal_difference_deg": np.asarray(
            coherence["scale_normal_difference_deg"], dtype=np.float64
        ),
        "surface_local_variation_large": np.asarray(
            coherence["surface_variation_large"], dtype=np.float64
        ),
        "surface_local_same_sheet_fraction": np.asarray(
            coherence["same_sheet_fraction"], dtype=np.float64
        ),
    }
    diagnostic.update(coverage_recovery_diagnostic)
    return keep, classes, anchor_distance, report, diagnostic


def load_completion_validation_context(
    root, args, registration, views, pose_transforms, canonical_frame
):
    """Carga evidencia 2D/3D ya validada por 06/10 para verificar guías inferidas.

    No genera nuevas mediciones ni reoptimiza poses. Las siluetas provienen del
    mismo contrato usado por 10 y las profundidades se reconstruyen únicamente
    desde P00..P24_registered_clean.npz.
    """
    report = {
        "available": False,
        "silhouette_source": "step10_visual_hull_contract",
        "depth_source": "step10_registered_clean_views",
        "poses": 0,
        "reason": None,
    }
    if cv2 is None:
        report["reason"] = "opencv_unavailable"
        return None, report

    cloud_summary_path = (
        Path(root)
        / "reconstruccion"
        / "multisesion"
        / args.cloud_source
        / "resumen_06_nubes_puntos.json"
    )
    if not cloud_summary_path.is_file():
        report["reason"] = "missing_step06_summary"
        return None, report
    cloud_summary = load_json(cloud_summary_path)
    raw_intr = cloud_summary.get("camera_intrinsics") or {}
    required = ("fx", "fy", "cx", "cy")
    if not all(k in raw_intr for k in required):
        report["reason"] = "step06_intrinsics_unavailable"
        return None, report
    intrinsics = {k: float(raw_intr[k]) for k in required}

    sources = (registration.get("visual_hull_carving") or {}).get("silhouette_sources") or {}
    if len(sources) < 25:
        report["reason"] = "step10_silhouette_contract_incomplete"
        return None, report

    if canonical_frame.get("enabled"):
        rotation = np.asarray(canonical_frame["rotation_source_to_platform"], dtype=np.float64)
        origin = np.asarray(canonical_frame["source_line_point_xyz_mm"], dtype=np.float64)
    else:
        rotation = np.eye(3, dtype=np.float64)
        origin = np.zeros(3, dtype=np.float64)

    pose_data = []
    tol_px = max(0, int(args.completion_silhouette_tolerance_px))
    for view in views:
        pose = int(view["pose_index"])
        raw_path = sources.get(f"P{pose:02d}") or sources.get(str(pose))
        if not raw_path:
            report["reason"] = f"missing_silhouette_P{pose:02d}"
            return None, report
        silhouette_path = Path(raw_path)
        if not silhouette_path.is_file():
            report["reason"] = f"silhouette_file_missing_P{pose:02d}"
            return None, report
        mask = cv2.imread(str(silhouette_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            report["reason"] = f"silhouette_read_failed_P{pose:02d}"
            return None, report
        mask = mask > 0
        if tol_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1))
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0

        shape = view.get("image_shape_hw")
        if shape is None:
            shape = tuple(mask.shape[:2])
        shape = tuple(int(x) for x in shape[:2])
        if tuple(mask.shape[:2]) != shape:
            report["reason"] = f"silhouette_shape_mismatch_P{pose:02d}"
            return None, report

        T = np.asarray(pose_transforms[pose], dtype=np.float64)
        invT = np.linalg.inv(T)
        p00 = np.asarray(view["points"], dtype=np.float64)
        cam = transform_points(p00, invT)
        z = cam[:, 2]
        uv = np.asarray(view.get("pixel_uv", np.empty((0, 2))), dtype=np.float64)
        if len(uv) != len(p00):
            with np.errstate(divide="ignore", invalid="ignore"):
                uv = np.column_stack(
                    (
                        intrinsics["fx"] * cam[:, 0] / z + intrinsics["cx"],
                        intrinsics["fy"] * cam[:, 1] / z + intrinsics["cy"],
                    )
                )
        uncertainty = np.asarray(
            view.get("depth_uncertainty_mm", np.full(len(p00), np.nan)), dtype=np.float64
        )
        if len(uncertainty) != len(p00):
            uncertainty = np.full(len(p00), np.nan, dtype=np.float64)
        finite_unc = uncertainty[np.isfinite(uncertainty) & (uncertainty > 0)]
        fallback_unc = (
            float(np.median(finite_unc))
            if len(finite_unc)
            else float(args.completion_depth_noise_floor_mm)
        )
        uncertainty = np.where(
            np.isfinite(uncertainty) & (uncertainty > 0), uncertainty, fallback_unc
        )
        good = (
            np.all(np.isfinite(uv), axis=1)
            & np.isfinite(z)
            & (z > 1e-6)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < shape[1])
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < shape[0])
        )
        if np.count_nonzero(good) < 16:
            report["reason"] = f"insufficient_projection_samples_P{pose:02d}"
            return None, report
        uv_good = uv[good]
        pose_data.append(
            {
                "pose": pose,
                "inverse_transform": invT,
                "silhouette": mask,
                "shape": shape,
                "uv_tree": cKDTree(uv_good),
                "depth_mm": z[good],
                "uncertainty_mm": uncertainty[good],
            }
        )

    report.update(
        available=True,
        poses=len(pose_data),
        reason="validated_step06_step10_contract",
        camera_intrinsics=intrinsics,
        silhouette_tolerance_px=tol_px,
    )
    return {
        "intrinsics": intrinsics,
        "pose_data": pose_data,
        "canonical_rotation": rotation,
        "canonical_origin": origin,
    }, report


def validate_completion_candidates_multiview(candidate, candidate_normals, context, args, voxel):
    """Valida candidatos inferidos contra siluetas, visibilidad y espacio libre.

    Un candidato nunca se vuelve "observado". La validación solo decide si una
    hipótesis local es suficientemente compatible para usarse como guía.
    """
    p = np.asarray(candidate, dtype=np.float64)
    n = normalize_rows(np.asarray(candidate_normals, dtype=np.float64))
    count = len(p)
    zeros_u64 = lambda: np.zeros(count, dtype=np.uint64)
    silhouette_tested_mask = zeros_u64()
    silhouette_inside_mask = zeros_u64()
    silhouette_outside_mask = zeros_u64()
    depth_tested_mask = zeros_u64()
    depth_agreement_mask = zeros_u64()
    free_space_mask = zeros_u64()
    occluded_mask = zeros_u64()
    if context is None or count == 0:
        return {
            "passed": np.zeros(count, dtype=bool),
            "silhouette_tested_pose_mask": silhouette_tested_mask,
            "silhouette_inside_pose_mask": silhouette_inside_mask,
            "silhouette_outside_pose_mask": silhouette_outside_mask,
            "depth_tested_pose_mask": depth_tested_mask,
            "depth_agreement_pose_mask": depth_agreement_mask,
            "free_space_contradiction_pose_mask": free_space_mask,
            "occluded_pose_mask": occluded_mask,
            "silhouette_inside_ratio": np.zeros(count),
            "independent_depth_agreements": np.zeros(count, dtype=np.uint8),
        }

    R = np.asarray(context["canonical_rotation"], dtype=np.float64)
    origin = np.asarray(context["canonical_origin"], dtype=np.float64)
    # canonical = (P00-origin) @ R.T  =>  P00 = canonical @ R + origin
    p00 = p @ R + origin[None, :]
    intr = context["intrinsics"]
    k = max(1, int(args.completion_depth_neighbors))
    radius = max(float(args.completion_depth_pixel_radius), 0.5)
    noise_floor = max(float(args.completion_depth_noise_floor_mm), 1e-6)
    uncertainty_factor = max(float(args.completion_depth_uncertainty_factor), 0.0)

    for item in context["pose_data"]:
        pose = int(item["pose"])
        bit = np.uint64(1) << np.uint64(pose)
        cam = transform_points(p00, item["inverse_transform"])
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = intr["fx"] * cam[:, 0] / z + intr["cx"]
            v = intr["fy"] * cam[:, 1] / z + intr["cy"]
        h, w = item["shape"]
        ui = np.rint(u).astype(np.int64)
        vi = np.rint(v).astype(np.int64)
        in_fov = (
            np.isfinite(z)
            & (z > 1e-6)
            & np.isfinite(u)
            & np.isfinite(v)
            & (ui >= 0)
            & (ui < w)
            & (vi >= 0)
            & (vi < h)
        )
        silhouette_tested_mask[in_fov] |= bit
        inside = np.zeros(count, dtype=bool)
        ids = np.flatnonzero(in_fov)
        if len(ids):
            inside[ids] = item["silhouette"][vi[ids], ui[ids]]
        silhouette_inside_mask[inside] |= bit
        outside = in_fov & ~inside
        silhouette_outside_mask[outside] |= bit

        eligible = np.flatnonzero(inside)
        if not len(eligible):
            continue
        q = np.column_stack((u[eligible], v[eligible]))
        dd, ii = item["uv_tree"].query(q, k=k, distance_upper_bound=radius, workers=query_threads())
        dd = np.asarray(dd)
        ii = np.asarray(ii)
        if k == 1:
            dd = dd[:, None]
            ii = ii[:, None]
        for local_row, global_idx in enumerate(eligible):
            valid = np.isfinite(dd[local_row]) & (ii[local_row] < len(item["depth_mm"]))
            if not np.any(valid):
                continue
            neigh = ii[local_row, valid].astype(np.int64)
            depths = item["depth_mm"][neigh]
            uncertainties = item["uncertainty_mm"][neigh]
            observed_z = float(np.median(depths))
            local_unc = (
                float(np.median(uncertainties[np.isfinite(uncertainties)]))
                if np.any(np.isfinite(uncertainties))
                else noise_floor
            )
            tolerance = max(noise_floor, 0.75 * float(voxel), uncertainty_factor * local_unc)
            depth_tested_mask[global_idx] |= bit
            delta = float(z[global_idx] - observed_z)
            if delta < -tolerance:
                free_space_mask[global_idx] |= bit
            elif abs(delta) <= tolerance:
                depth_agreement_mask[global_idx] |= bit
            else:
                occluded_mask[global_idx] |= bit

    def pop(mask):
        return np.fromiter((int(int(x).bit_count()) for x in mask), dtype=np.int16, count=len(mask))

    silhouette_tested = pop(silhouette_tested_mask)
    silhouette_inside = pop(silhouette_inside_mask)
    silhouette_outside = pop(silhouette_outside_mask)
    depth_tested = pop(depth_tested_mask)
    depth_agreement = pop(depth_agreement_mask)
    free_space = pop(free_space_mask)
    occluded = pop(occluded_mask)
    inside_ratio = np.divide(silhouette_inside.astype(np.float64), np.maximum(silhouette_tested, 1))
    independent_depth = np.zeros(count, dtype=np.uint8)
    independent_inside = np.zeros(count, dtype=np.uint8)
    min_sep = max(1, int(args.completion_min_independent_pose_separation))
    for i in range(count):
        depth_ids = [pose for pose in range(25) if int(depth_agreement_mask[i]) & (1 << pose)]
        inside_ids = [pose for pose in range(25) if int(silhouette_inside_mask[i]) & (1 << pose)]
        independent_depth[i] = _pose_diversity(depth_ids, expected=25, minimum_separation=min_sep)[
            0
        ]
        independent_inside[i] = _pose_diversity(
            inside_ids, expected=25, minimum_separation=min_sep
        )[0]

    passed = (
        (silhouette_tested >= int(args.completion_min_silhouette_tested_poses))
        & (inside_ratio >= float(args.completion_min_silhouette_inside_ratio))
        & (silhouette_outside <= int(args.completion_max_silhouette_contradictions))
        & (independent_inside >= 3)
        & (depth_tested >= int(args.completion_min_depth_tested_poses))
        & (independent_depth >= int(args.completion_min_independent_depth_agreements))
        & (free_space <= int(args.completion_max_free_space_contradictions))
    )
    return {
        "passed": passed,
        "silhouette_tested_pose_mask": silhouette_tested_mask,
        "silhouette_inside_pose_mask": silhouette_inside_mask,
        "silhouette_outside_pose_mask": silhouette_outside_mask,
        "depth_tested_pose_mask": depth_tested_mask,
        "depth_agreement_pose_mask": depth_agreement_mask,
        "free_space_contradiction_pose_mask": free_space_mask,
        "occluded_pose_mask": occluded_mask,
        "silhouette_tested_poses": silhouette_tested,
        "silhouette_inside_poses": silhouette_inside,
        "silhouette_outside_poses": silhouette_outside,
        "silhouette_inside_ratio": inside_ratio,
        "depth_tested_poses": depth_tested,
        "depth_agreement_poses": depth_agreement,
        "free_space_contradictions": free_space,
        "occluded_poses": occluded,
        "independent_depth_agreements": independent_depth,
        "independent_silhouette_inside": independent_inside,
    }


def estimate_missing_patches(
    points, normals, colors, voxel, args, completion_context=None, completion_context_report=None
):
    """Guías inferidas, no observaciones, con validación multivista estricta.

    BPA solo propone contornos. La geometría local produce candidatos, pero un
    candidato no se exporta si contradice siluetas o espacio libre, o si no
    recibe acuerdo de profundidad desde poses independientes. No se usa radio,
    diámetro, plantilla de cilindro, convexidad ni clase de objeto.
    """
    import open3d as o3d
    from matplotlib.path import Path as PolygonPath
    import time

    started = time.perf_counter()
    report = {
        "enabled": bool(args.complete_gaps),
        "observed_support_assigned": 0,
        "base_closure_is_assumption": True,
        "patches": [],
        "generated_points": 0,
        "validation_contract": completion_context_report or {"available": False},
        "policy": "closed_boundary_proposal_plus_multiview_silhouette_visibility_free_space_validation",
    }
    empty = {
        "points": np.empty((0, 3)),
        "normals": np.empty((0, 3)),
        "colors": np.empty((0, 3), np.uint8),
        "patch_id": np.empty(0, np.int32),
        "uncertainty_mm": np.empty(0),
        "kind": np.empty(0, np.uint8),
        "multiview_validated": np.empty(0, bool),
        "silhouette_inside_ratio": np.empty(0, np.float32),
        "independent_depth_agreements": np.empty(0, np.uint8),
        "depth_tested_poses": np.empty(0, np.uint8),
        "free_space_contradictions": np.empty(0, np.uint8),
        "silhouette_outside_poses": np.empty(0, np.uint8),
        "depth_agreement_pose_mask": np.empty(0, np.uint64),
        "free_space_contradiction_pose_mask": np.empty(0, np.uint64),
    }
    if not args.complete_gaps:
        report["reason"] = "disabled"
        return empty, report
    if completion_context is None:
        report["reason"] = "multiview_validation_context_unavailable"
        return empty, report
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.normals = o3d.utility.Vector3dVector(normals)
    # Limit auxiliary work; original observations remain untouched.
    proxy = cloud.voxel_down_sample(0.65 * voxel)
    proxy.normalize_normals()
    print(
        "[Paso 11] Completado: construyendo conectividad auxiliar para localizar contornos...",
        flush=True,
    )
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        proxy, o3d.utility.DoubleVector([1.25 * voxel, 2.0 * voxel])
    )
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    if not len(t):
        report["reason"] = "no_auxiliary_surface"
        return empty, report
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    edges = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = edges[counts == 1]
    adjacency = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    visited = set()
    loops = []
    for seed in sorted(adjacency):
        if seed in visited:
            continue
        component, stack = set(), [seed]
        while stack:
            j = stack.pop()
            if j in component:
                continue
            component.add(j)
            stack.extend(k for k in adjacency[j] if k not in component)
        visited.update(component)
        if len(component) < 6 or any(len(adjacency[j]) != 2 for j in component):
            continue
        loop, previous, current = [seed], -1, seed
        while True:
            next_id = next(k for k in adjacency[current] if k != previous)
            if next_id == seed:
                break
            loop.append(next_id)
            previous, current = current, next_id
            if len(loop) > len(component):
                break
        if len(loop) == len(component):
            loops.append(loop)
    loops.sort(key=lambda loop: float(np.ptp(v[loop], axis=0).max()), reverse=True)
    tree = cKDTree(points)
    inferred = []
    inferred_normals = []
    inferred_colors = []
    patch_ids, uncertainties, kinds = [], [], []
    validation_inside_ratio = []
    validation_independent_depth = []
    validation_depth_tested = []
    validation_free_space = []
    validation_silhouette_outside = []
    validation_depth_masks = []
    validation_free_masks = []
    extent_vec = robust_extent(points)
    extent = (
        float(np.max(extent_vec)) if extent_vec is not None else float(np.ptp(points, axis=0).max())
    )
    automatic_span = max(6.0 * voxel, min(12.0 * voxel, 0.18 * extent))
    max_span = args.completion_max_span_mm if args.completion_max_span_mm > 0 else automatic_span
    report["maximum_local_span_mm"] = float(max_span)
    report["automatic_span_policy"] = "max(6*voxel,min(12*voxel,0.18*robust_extent))"
    budget = min(int(args.completion_max_points), max(1000, len(points) // 2))
    used = 0
    for patch_id, loop in enumerate(loops[:128]):
        record = {"patch_id": patch_id, "boundary_vertices": len(loop), "accepted": False}
        report["patches"].append(record)
        if len(loop) > 800 or used >= budget:
            record["reason"] = "complexity_budget"
            continue
        rim = v[loop]
        center = rim.mean(axis=0)
        _, _, basis = np.linalg.svd(rim - center, full_matrices=False)
        uv = (rim - center) @ basis[:2].T
        z = (rim - center) @ basis[2]
        span = float(np.ptp(uv, axis=0).max())
        is_base = (
            bool(args.align_platform_axis)
            and bool(args.complete_bottom)
            and abs(basis[2, 1]) >= 0.85
            and np.quantile(rim[:, 1], 0.9) <= np.min(points[:, 1]) + max(3 * voxel, 0.06 * extent)
        )
        record.update(span_mm=span, kind="estimated_bottom" if is_base else "interpolated_gap")
        if span < 2 * voxel or span > max_span:
            record["reason"] = "span_outside_local_completion_budget"
            continue
        if np.sqrt(np.mean(z * z)) > max(1.5 * voxel, 0.12 * span):
            record["reason"] = "boundary_not_a_single_local_chart"
            continue
        # Reject self-crossing projected boundaries; do not convexify them.
        crossing = False

        def cross2(a, b):
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

        for i in range(len(uv)):
            j = np.arange(i + 2, len(uv))
            if i == 0:
                j = j[j != len(uv) - 1]
            if not len(j):
                continue
            a, b = uv[i], uv[(i + 1) % len(uv)]
            c, d = uv[j], uv[(j + 1) % len(uv)]
            hit = (cross2(b - a, c - a) * cross2(b - a, d - a) < -1e-12) & (
                cross2(d - c, a - c) * cross2(d - c, b - c) < -1e-12
            )
            if hit.any():
                crossing = True
                break
        if crossing:
            record["reason"] = "self_crossing_projected_boundary"
            continue
        polygon = PolygonPath(np.vstack([uv, uv[0]]))
        scale = max(span, voxel)
        nearest = tree.query(rim, workers=query_threads())[1]
        nn = normals[nearest]
        coeff = np.zeros(6)

        def design(x, y):
            return np.column_stack([np.ones(len(x)), x, y, x * x, x * y, y * y])

        x, y = (uv / scale).T
        if is_base:
            if np.quantile(np.abs(z), 0.95) > 1.5 * voxel:
                record["reason"] = "bottom_rim_not_sufficiently_planar"
                continue
            coeff[0] = float(np.median(z))
        else:
            dot = nn @ basis[2]
            if np.median(dot) < 0:
                basis[2] *= -1
                z *= -1
                dot *= -1
            reliable = dot > 0.45
            if np.mean(reliable) < 0.8:
                record["reason"] = "mixed_normals_or_sharp_features"
                continue
            # Derivative observations disambiguate a curved continuation from
            # a flat disk when all boundary points have similar height.
            dx = np.column_stack([0 * x, 1 + 0 * x, 0 * x, 2 * x, y, 0 * x])
            dy = np.column_stack([0 * x, 0 * x, 1 + 0 * x, 0 * x, x, 2 * y])
            matrix = np.vstack([design(x, y), 0.3 * dx[reliable], 0.3 * dy[reliable]])
            target = np.r_[
                z,
                -0.3 * scale * (nn[reliable] @ basis[0]) / dot[reliable],
                -0.3 * scale * (nn[reliable] @ basis[1]) / dot[reliable],
            ]
            weights = np.ones(len(target))
            rank, condition = 0, np.inf
            for _ in range(3):
                coeff, _, rank, s = np.linalg.lstsq(
                    matrix * np.sqrt(weights[:, None]), target * np.sqrt(weights), rcond=1e-8
                )
                condition = s[0] / max(s[-1], 1e-12)
                error = matrix @ coeff - target
                weights = np.minimum(1, voxel / np.maximum(np.abs(error), 1e-12))
            if (
                rank < 6
                or condition > 1e4
                or np.quantile(np.abs(design(x, y) @ coeff - z), 0.9) > voxel
            ):
                record["reason"] = "boundary_fit_not_reliable"
                continue
        pitch = max(voxel, span / 160)
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        xx, yy = np.meshgrid(
            np.arange(lo[0], hi[0] + pitch, pitch), np.arange(lo[1], hi[1] + pitch, pitch)
        )
        query = np.column_stack([xx.ravel(), yy.ravel()])
        query = query[polygon.contains_points(query)]
        if not len(query):
            record["reason"] = "no_interior_samples"
            continue
        qx, qy = (query / scale).T
        height = design(qx, qy) @ coeff
        if not np.all(np.isfinite(height)) or np.max(np.abs(height)) > max(3 * voxel, 0.35 * span):
            record["reason"] = "unbounded_extrapolation"
            continue
        candidate = center + query @ basis[:2] + height[:, None] * basis[2]
        gx = (coeff[1] + 2 * coeff[3] * qx + coeff[4] * qy) / scale
        gy = (coeff[2] + coeff[4] * qx + 2 * coeff[5] * qy) / scale
        candidate_n = normalize_rows(
            basis[2][None, :] - gx[:, None] * basis[0] - gy[:, None] * basis[1]
        )
        if is_base and np.mean(candidate_n[:, 1]) > 0:
            candidate_n *= -1
        distances, source = tree.query(candidate, workers=query_threads())
        # Avoid replacing observed surfaces or generating a second sheet where
        # observations already occupy the interior. Missing points stay tagged.
        allowed = distances > 0.9 * pitch
        allowed &= distances <= float(args.completion_max_observation_distance_voxels) * voxel
        mesh_distance = scene.compute_distance(
            o3d.core.Tensor(candidate.astype(np.float32))
        ).numpy()
        allowed &= mesh_distance > 0.7 * pitch
        if inferred:
            previous_distance = cKDTree(np.concatenate(inferred)).query(
                candidate, workers=query_threads()
            )[0]
            allowed &= previous_distance > 0.9 * pitch
        take_geometric = np.flatnonzero(allowed)
        if len(take_geometric) < 4:
            record["reason"] = "interior_already_observed_or_extrapolation_too_far"
            continue

        validation = validate_completion_candidates_multiview(
            candidate[take_geometric], candidate_n[take_geometric], completion_context, args, voxel
        )
        passed_local = np.asarray(validation["passed"], dtype=bool)
        retained_fraction = float(np.mean(passed_local)) if len(passed_local) else 0.0
        record["multiview_validation"] = {
            "geometric_candidates": int(len(take_geometric)),
            "validated_candidates": int(np.count_nonzero(passed_local)),
            "retained_fraction": retained_fraction,
            "silhouette_inside_ratio": finite_stats(validation["silhouette_inside_ratio"]),
            "depth_tested_poses": finite_stats(validation["depth_tested_poses"]),
            "independent_depth_agreements": finite_stats(
                validation["independent_depth_agreements"]
            ),
            "free_space_contradictions": finite_stats(validation["free_space_contradictions"]),
            "silhouette_outside_poses": finite_stats(validation["silhouette_outside_poses"]),
        }
        if np.count_nonzero(passed_local) < 4 or retained_fraction < float(
            args.completion_min_retained_fraction
        ):
            record["reason"] = "multiview_validation_rejected_patch"
            continue
        take = take_geometric[passed_local]
        if len(take) > budget - used:
            record["reason"] = "point_budget_would_truncate_patch"
            continue
        inferred.append(candidate[take])
        inferred_normals.append(candidate_n[take])
        inferred_colors.append(colors[source[take]])
        patch_ids.append(np.full(len(take), patch_id, np.int32))
        uncertainties.append(np.maximum(voxel, 0.25 * distances[take]))
        kinds.append(np.full(len(take), 2 if is_base else 1, np.uint8))
        validation_inside_ratio.append(
            validation["silhouette_inside_ratio"][passed_local].astype(np.float32)
        )
        validation_independent_depth.append(
            validation["independent_depth_agreements"][passed_local].astype(np.uint8)
        )
        validation_depth_tested.append(
            validation["depth_tested_poses"][passed_local].astype(np.uint8)
        )
        validation_free_space.append(
            validation["free_space_contradictions"][passed_local].astype(np.uint8)
        )
        validation_silhouette_outside.append(
            validation["silhouette_outside_poses"][passed_local].astype(np.uint8)
        )
        validation_depth_masks.append(
            validation["depth_agreement_pose_mask"][passed_local].astype(np.uint64)
        )
        validation_free_masks.append(
            validation["free_space_contradiction_pose_mask"][passed_local].astype(np.uint64)
        )
        used += len(take)
        record.update(
            accepted=True,
            reason="estimated_from_closed_boundary_and_multiview_validated",
            points=int(len(take)),
            maximum_distance_to_observation_mm=float(distances[take].max()),
        )
        print(
            f"[Paso 11] Parche estimado {patch_id}: {len(take)} puntos validados multivista ({record['kind']}).",
            flush=True,
        )
    report.update(
        generated_points=used,
        closed_boundary_loops=len(loops),
        seconds=float(time.perf_counter() - started),
    )
    if not inferred:
        return empty, report
    return {
        "points": np.concatenate(inferred),
        "normals": np.concatenate(inferred_normals),
        "colors": np.concatenate(inferred_colors),
        "patch_id": np.concatenate(patch_ids),
        "uncertainty_mm": np.concatenate(uncertainties),
        "kind": np.concatenate(kinds),
        "multiview_validated": np.ones(used, dtype=bool),
        "silhouette_inside_ratio": np.concatenate(validation_inside_ratio),
        "independent_depth_agreements": np.concatenate(validation_independent_depth),
        "depth_tested_poses": np.concatenate(validation_depth_tested),
        "free_space_contradictions": np.concatenate(validation_free_space),
        "silhouette_outside_poses": np.concatenate(validation_silhouette_outside),
        "depth_agreement_pose_mask": np.concatenate(validation_depth_masks),
        "free_space_contradiction_pose_mask": np.concatenate(validation_free_masks),
    }, report


def measure_final_coverage(reference_points, final_points, selected, voxel):
    """Ocupación final y distancia contra TODOS los candidatos antes del ajuste.

    La razón de ocupación es un indicador de muestreo, no área superficial.
    El denominador y las consultas no se mueven con los puntos seleccionados.
    """
    reference = np.asarray(reference_points, dtype=np.float64)
    final = np.asarray(final_points, dtype=np.float64)
    keep = np.asarray(selected, dtype=bool)
    if (reference.shape != final.shape or reference.ndim != 2 or reference.shape[1] != 3
            or keep.shape != (len(reference),) or not np.isfinite(voxel) or voxel <= 0
            or not np.isfinite(reference).all() or not np.isfinite(final).all()):
        raise ValueError('Referencia de cobertura o geometría final incompatible.')
    reference_cells, cell_inverse = np.unique(
        np.floor(reference / voxel).astype(np.int64), axis=0, return_inverse=True)
    final_cells = np.unique(np.floor(final[keep] / voxel).astype(np.int64), axis=0)
    original_selected_cells = np.unique(np.floor(reference[keep] / voxel).astype(np.int64), axis=0)
    report = {
        'reference': 'immutable_all_candidates_before_support_selection',
        'candidate_points': int(len(reference)), 'selected_points': int(keep.sum()),
        'reference_occupied_cells': int(len(reference_cells)),
        'final_occupied_cells': int(len(final_cells)),
        'selected_original_occupied_cells': int(len(original_selected_cells)),
        'occupied_cell_retention': len(final_cells) / max(1,len(reference_cells)),
        'pre_adjustment_selected_cell_retention': len(original_selected_cells) / max(1,len(reference_cells)),
        'within_one_voxel_fraction': 0.0, 'within_two_voxels_fraction': 0.0,
        'reference_cell_coverage_one_voxel': 0.0,
        'reference_cell_coverage_two_voxels': 0.0,
        'cell_coverage_policy': 'Every original candidate in a cell must lie within the distance limit; all cells have equal weight.',
        'note': 'Final occupied cells / original occupied cells; not measured surface area.',
    }
    if keep.any():
        distances, _ = cKDTree(final[keep]).query(reference, workers=query_threads())
        cell_max_distance = np.zeros(len(reference_cells), dtype=np.float64)
        np.maximum.at(cell_max_distance, cell_inverse, distances)
        report['reference_cell_coverage_one_voxel'] = float(np.mean(cell_max_distance <= voxel))
        report['reference_cell_coverage_two_voxels'] = float(np.mean(cell_max_distance <= 2*voxel))
        report['distance_to_retained_mm'] = finite_stats(distances)
        report['within_one_voxel_fraction'] = float(np.mean(distances <= voxel))
        report['within_two_voxels_fraction'] = float(np.mean(distances <= 2*voxel))
    return report


def main():
    """Fusiona las observaciones registradas y exporta nube, atributos y diagnósticos."""
    args = build_parser().parse_args()
    if not np.isfinite(args.completion_max_span_mm) or args.completion_max_span_mm < 0:
        raise ValueError("completion-max-span-mm debe ser finito y no negativo.")
    if args.completion_max_points < 1000:
        raise ValueError(
            "completion-max-points debe ser al menos 1000; usa --no-complete-gaps para desactivar."
        )
    if not 0.50 <= float(args.completion_min_silhouette_inside_ratio) <= 1.0:
        raise ValueError("completion-min-silhouette-inside-ratio debe estar entre 0.50 y 1.0.")
    if not 0.0 < float(args.completion_min_retained_fraction) <= 1.0:
        raise ValueError("completion-min-retained-fraction debe estar en (0,1].")
    if int(args.completion_min_silhouette_tested_poses) < 1:
        raise ValueError("completion-min-silhouette-tested-poses debe ser >= 1.")
    if int(args.completion_min_depth_tested_poses) < 1:
        raise ValueError("completion-min-depth-tested-poses debe ser >= 1.")
    if int(args.completion_min_independent_depth_agreements) < 1:
        raise ValueError("completion-min-independent-depth-agreements debe ser >= 1.")
    if float(args.completion_max_observation_distance_voxels) <= 0:
        raise ValueError("completion-max-observation-distance-voxels debe ser > 0.")
    for name in (
        "fusion_voxel_mm",
        "prevoxel_mm",
        "local_radius_voxels",
        "local_sigma_floor_voxels",
        "maximum_spread_for_full_score_mm",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Paso 11: {name} debe ser finito y mayor que cero.")
    if not 2 <= args.local_neighbors_per_pose <= 64:
        raise ValueError("Paso 11: local-neighbors-per-pose debe estar entre 2 y 64.")
    if not np.isfinite(args.local_normal_angle_deg) or not 5 <= args.local_normal_angle_deg <= 60:
        raise ValueError("Paso 11: local-normal-angle-deg debe estar entre 5 y 60.")
    if not np.isfinite(args.local_max_shift_voxels) or not 0 <= args.local_max_shift_voxels <= 1:
        raise ValueError("Paso 11: local-max-shift-voxels debe estar entre 0 y 1.")
    if not 0.02 <= args.local_sigma_floor_voxels <= 0.30:
        raise ValueError("Paso 11: local-sigma-floor-voxels debe estar entre 0.02 y 0.30.")
    if not 1.25 <= args.local_radius_voxels <= 5:
        raise ValueError("Paso 11: local-radius-voxels debe estar entre 1.25 y 5.")
    root = Path(args.root).expanduser().resolve()
    obj = args.object.strip()

    print("[Paso 11 | 1/4] Validando calibración y contrato del paso 10...")
    calibration_path = resolve_calibration(root, args.calibration)
    calibration = load_json(calibration_path)
    if calibration.get("mechanical_model", {}).get("steps_per_revolution") != 2055:
        raise RuntimeError("La calibración no corresponde a 2055 pasos.")
    calibration_pose_records = {int(x["pose_index"]): x for x in calibration["poses"]}

    registration_summary_path, registration = validate_registration(
        root, obj, args.registration_source, calibration_path
    )
    pose_transforms, pose_transform_contract = load_runtime_pose_transforms(
        registration, calibration_path, calibration
    )
    print(
        "[Paso 11 | 1/4] Modelo de poses:",
        pose_transform_contract["source"],
    )
    print("[Paso 11 | 2/4] Cargando las vistas ya registradas y talladas por 10...")
    registered_clean_folder, views = load_registered_clean_views(
        root, args.registration_source, registration
    )
    pose_quality_weights = load_consensus_pose_weights(
        root, obj, args.consensus_source, args.depth_spread_reference_mm
    )
    registration_pose_weights, registration_pose_weight_report = load_registration_pose_weights(
        registration
    )

    output = root / "reconstruccion" / "multisesion" / args.output_name
    output.mkdir(parents=True, exist_ok=True)

    all_points = []
    all_colors = []
    all_normals = []
    all_pose_ids = []
    all_weights = []
    per_pose_records = []

    print("[Paso 11 | 3/4] Preparando observaciones y pesos de las 25 poses...")
    if args.local_surface_fusion:
        print(
            "[Paso 11] Ruta local activada: las muestras registradas se conservan "
            "sin promediado previo. prevoxel-mm solo se usa en la ruta legacy.",
            flush=True,
        )
    print("\n========== PASO 11 — FUSIÓN MULTIVISTA GENERAL ==========")
    for view in views:
        pose = int(view["pose_index"])
        record = calibration_pose_records[pose]
        if abs(float(record.get("runtime_correction_deg", 0.0))) > 1e-12:
            raise RuntimeError("La calibración runtime contiene corrección angular.")

        T = pose_transforms[pose]
        # Pxx_registered_clean.npz ya está expresado en el sistema P00. Aplicar
        # T otra vez duplicaría la rotación y reintroduciría capas desplazadas.
        if view.get("already_registered"):
            p = np.asarray(view["points"], dtype=np.float64)
            n = np.asarray(view["normals"], dtype=np.float64)
        else:
            p = transform_points(view["points"], T)
            n = transform_normals(view["normals"], T)
        c = view["colors"]
        camera = T[:3, 3].copy()

        if args.local_surface_fusion:
            if (
                len(n) != len(p)
                or not np.all(np.isfinite(n))
                or np.any(np.linalg.norm(n, axis=1) < 0.5)
            ):
                raise RuntimeError(
                    f"Paso 11: P{pose:02d} tiene normales ausentes o inválidas. "
                    "Regenere las normales en el paso 10; no se mezclarán capas sin ellas."
                )
            # Conservar TODAS las observaciones originales de 10 hasta el ajuste.
            n = normalize_rows(n)
        else:
            p, c, n = per_pose_prevoxel(p, c, n, args.prevoxel_mm)

        rays = camera[None, :] - p
        ray_len = np.linalg.norm(rays, axis=1)
        good_ray = ray_len > 1e-9
        unit_ray = np.zeros_like(rays)
        unit_ray[good_ray] = rays[good_ray] / ray_len[good_ray, None]

        if args.local_surface_fusion:
            # Orientación por visibilidad. No usar abs(dot) al comparar dos
            # superficies: las dos paredes de una lámina deben seguir separadas.
            n = n.copy()
            n[np.sum(n * unit_ray, axis=1) < 0] *= -1.0
        if len(n) == len(p):
            incidence = np.abs(np.sum(n * unit_ray, axis=1))
            incidence = np.clip(incidence, args.minimum_incidence_cosine, 1.0)
        else:
            incidence = np.full(len(p), 0.70, dtype=np.float64)

        valid_reliability = np.asarray(view["reliability"], dtype=np.float64)
        valid_reliability = valid_reliability[np.isfinite(valid_reliability)]
        reliability_weight = float(np.median(valid_reliability)) if len(valid_reliability) else 0.0
        consensus_weight = float(pose_quality_weights.get(pose, 1.0))
        registration_weight = float(registration_pose_weights.get(pose, 1.0))
        view_weight = (
            consensus_weight
            * registration_weight
            * float(np.clip(0.65 + 0.35 * reliability_weight, 0.65, 1.0))
        )
        weights = view_weight * np.power(incidence, args.incidence_power)
        if args.local_surface_fusion:
            local_reliability = np.asarray(view["reliability"], dtype=np.float64)
            local_reliability = np.nan_to_num(local_reliability, nan=0.0, posinf=0.0, neginf=0.0)
            weights *= np.clip(local_reliability, 0.05, 1.0)

        all_points.append(p)
        all_colors.append(c)
        all_normals.append(n if len(n) == len(p) else np.zeros((len(p), 3)))
        all_pose_ids.append(np.full(len(p), pose, dtype=np.int16))
        all_weights.append(weights)

        per_pose_records.append(
            {
                "pose_index": pose,
                "points_before": int(len(view["points"])),
                "points_after_prevoxel": int(len(p)),
                "view_quality_weight": view_weight,
                "consensus_pose_weight": consensus_weight,
                "registration_pose_weight": registration_weight,
                "median_step10_reliability": reliability_weight,
                "incidence": finite_stats(incidence),
            }
        )
        print(
            f"[Paso 11 | preparar P{pose:02d}] {len(view['points']):6d} -> {len(p):6d} observaciones | "
            f"w_pose={view_weight:.3f}"
        )

    if not all_points or sum(len(p) for p in all_points) == 0:
        raise RuntimeError("Paso 11: las vistas del paso 10 no contienen puntos válidos.")
    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    normals = np.vstack(all_normals)
    poses = np.concatenate(all_pose_ids)
    weights = np.concatenate(all_weights)

    local_summary = {"enabled": False, "method": "legacy_independent_voxel_average"}
    local_result = None
    local_diagnostic_path = None
    if args.local_surface_fusion:
        local_result, unique_voxel, local_diagnostics, local_retained, local_summary = (
            neighboring_surface_fusion(points, colors, normals, poses, weights, args)
        )
        layer_report = _resolve_validated_pose_layers(
            local_result, float(args.fusion_voxel_mm), args
        )
        local_summary["validated_pose_layer_separation"] = layer_report
        if layer_report.get("enabled"):
            print(
                f"[Paso 11 | capas por pose] evaluados={layer_report.get('evaluated',0):,}; "
                f"multicapa={layer_report.get('multi_layer_detected',0):,}; "
                f"resueltos={layer_report.get('resolved_to_dominant_layer',0):,}; "
                f"ambiguos={layer_report.get('ambiguous_downgraded',0):,}.",
                flush=True,
            )
        n_vox = len(unique_voxel)
        fused_p = local_result["points"]
        fused_c = local_result["colors"]
        fused_n = local_result["normals"]
        support = local_result["support"]
        confidence = local_result["confidence"]
        spread = local_result["spread"]
        normal_consistency = local_result["normal_consistency"]
    else:
        # Un único aporte por pose y voxel global.
        voxel_keys = np.floor(points / float(args.fusion_voxel_mm)).astype(np.int64)
        vp_keys = np.column_stack([voxel_keys, poses.astype(np.int64)])
        unique_vp, inv_vp = np.unique(vp_keys, axis=0, return_inverse=True)
        n_vp = len(unique_vp)

        p_vp, wsum_vp = aggregate_by_inverse(points, inv_vp, n_vp, weights)
        c_vp, _ = aggregate_by_inverse(colors.astype(np.float64), inv_vp, n_vp, weights)
        n_vp_arr, _ = aggregate_by_inverse(normals, inv_vp, n_vp, weights)
        n_vp_arr = normalize_rows(n_vp_arr)

        # Peso de cada pose/voxel queda acotado para evitar que una pose domine.
        vp_weight = np.bincount(inv_vp, weights=weights, minlength=n_vp)
        vp_weight = np.clip(vp_weight, 0.05, 1.0)
        vp_pose = unique_vp[:, 3].astype(np.int16)
        vp_voxel = unique_vp[:, :3].astype(np.int64)

        # Agrupar las contribuciones independientes de poses.
        unique_voxel, inv_voxel = np.unique(vp_voxel, axis=0, return_inverse=True)
        n_vox = len(unique_voxel)
        support = np.bincount(inv_voxel, minlength=n_vox).astype(np.int16)

        fused_p, denom = aggregate_by_inverse(p_vp, inv_voxel, n_vox, vp_weight)
        fused_c, _ = aggregate_by_inverse(c_vp, inv_voxel, n_vox, vp_weight)
        fused_n_raw, _ = aggregate_by_inverse(n_vp_arr, inv_voxel, n_vox, vp_weight)
        normal_consistency = np.linalg.norm(fused_n_raw, axis=1)
        fused_n = normalize_rows(fused_n_raw)

        # Dispersión ponderada de las observaciones de poses alrededor de la media.
        residual2 = np.sum((p_vp - fused_p[inv_voxel]) ** 2, axis=1)
        variance = np.bincount(
            inv_voxel, weights=vp_weight * residual2, minlength=n_vox
        ) / np.maximum(np.bincount(inv_voxel, weights=vp_weight, minlength=n_vox), 1e-12)
        spread = np.sqrt(np.maximum(variance, 0.0))

        mean_incidence_proxy = np.bincount(
            inv_voxel, weights=vp_weight, minlength=n_vox
        ) / np.maximum(support, 1)
        mean_incidence_proxy = np.clip(mean_incidence_proxy, 0.0, 1.0)

        support_score = np.clip(support.astype(np.float64) / 4.0, 0.0, 1.0)
        spread_score = np.exp(
            -0.5 * (spread / max(args.maximum_spread_for_full_score_mm, 1e-6)) ** 2
        )
        confidence = (
            0.48 * support_score
            + 0.20 * spread_score
            + 0.17 * np.clip(normal_consistency, 0.0, 1.0)
            + 0.15 * mean_incidence_proxy
        )
        confidence = np.clip(confidence, 0.0, 1.0)

    # El conteo de vecinos históricos se conserva en el NPZ para trazabilidad,
    # aunque la ruta normal usa distancia métrica al núcleo fuerte.
    core = support >= int(args.minimum_core_support)
    neighbor_core_count = robust_neighbor_core_counts(
        unique_voxel, core, int(args.singleton_neighbor_radius_cells)
    )
    keep, selection_class, distance_to_anchor, selection_info = hierarchical_consensus_selection(
        fused_p,
        support,
        confidence,
        spread,
        normal_consistency,
        unique_voxel,
        voxel_size_mm=float(args.fusion_voxel_mm),
        minimum_confidence=float(args.minimum_fused_confidence),
        enabled=bool(args.hierarchical_consensus),
        preferred_anchor_support=int(args.preferred_anchor_support),
        fallback_anchor_support=int(args.fallback_anchor_support),
        minimum_anchor_points=int(args.minimum_anchor_points),
        minimum_anchor_ratio=float(args.minimum_anchor_ratio),
        target_anchor_ratio=float(args.target_anchor_ratio),
        minimum_anchor_extent_coverage=float(args.minimum_anchor_extent_coverage),
        anchor_continuity_radius_voxels=float(args.anchor_continuity_radius_voxels),
        anchor_minimum_local_neighbors=int(args.anchor_minimum_local_neighbors),
        minimum_anchor_local_continuity_ratio=float(args.minimum_anchor_local_continuity_ratio),
        minimum_anchor_median_confidence=float(args.minimum_anchor_median_confidence),
        minimum_anchor_median_normal_consistency=float(
            args.minimum_anchor_median_normal_consistency
        ),
        maximum_anchor_spread_p90_mm=float(args.maximum_anchor_spread_p90_mm),
        minimum_extension_support=int(args.minimum_extension_support),
        extension_radius_one_level_voxels=float(args.extension_radius_one_level_voxels),
        extension_radius_two_levels_voxels=float(args.extension_radius_two_levels_voxels),
        extension_radius_low_support_voxels=float(args.extension_radius_low_support_voxels),
        uniform_extension_radius_voxels=float(args.extension_radius_voxels),
        legacy_core_support=int(args.minimum_core_support),
        preserve_legacy_singletons=bool(args.preserve_boundary_singletons),
        singleton_neighbor_radius_cells=int(args.singleton_neighbor_radius_cells),
        singleton_minimum_core_neighbors=int(args.singleton_minimum_core_neighbors),
        singleton_minimum_confidence=float(args.singleton_minimum_confidence),
    )

    # Referencia inmutable: las correcciones posteriores solo mueven retenidos.
    coverage_reference_points = np.asarray(fused_p, dtype=np.float64).copy()
    selection_diagnostic = {}
    if args.independent_patches:
        legacy_count = int(np.count_nonzero(keep))
        print(
            "[Paso 11] Seleccionando parches por evidencia propia y continuidad de superficie...",
            flush=True,
        )
        keep, selection_class, distance_to_anchor, selection_info, selection_diagnostic = (
            select_independent_patches(
                fused_p,
                fused_n,
                support,
                confidence,
                spread,
                normal_consistency,
                local_result,
                float(args.fusion_voxel_mm),
                float(args.minimum_fused_confidence),
                args=args,
            )
        )
        selection_info["legacy_selection_points_for_comparison"] = legacy_count
        print(
            f"[Paso 11] Selección: {np.count_nonzero(keep):,} puntos observados; "
            f"criterio anterior: {legacy_count:,}.",
            flush=True,
        )
        recovery_report = selection_info.get("observed_coverage_recovery", {})
        if recovery_report.get("enabled"):
            before_cov = recovery_report.get("coverage_before", {}).get(
                "spatially_uncovered_ratio", {}
            )
            after_cov = recovery_report.get("coverage_after", {}).get(
                "spatially_uncovered_ratio", {}
            )
            print(
                f"[Paso 11 | cobertura observada] candidatos="
                f"{recovery_report.get('candidate_points', 0):,}; recuperados="
                f"{recovery_report.get('accepted_recovered_points', 0):,}; "
                f"aceptada={bool(recovery_report.get('accepted', False))}; "
                f">2 vox: {100*float(before_cov.get('2_voxels') or 0):.1f}% -> "
                f"{100*float(after_cov.get('2_voxels') or 0):.1f}%.",
                flush=True,
            )

    # V11.6: antes del consenso geométrico regional, usar los residuos por pose
    # de las observaciones originales para estimar y retirar solo el gauge local
    # inducido por cambios del conjunto de poses que soporta cada parche.
    raw_pose_bias_diagnostic = {
        "available": np.zeros(int(np.count_nonzero(keep)), dtype=np.uint8),
        "accept": np.ones(int(np.count_nonzero(keep)), dtype=np.uint8),
        "ambiguous": np.zeros(int(np.count_nonzero(keep)), dtype=np.uint8),
        "feature_protected": np.zeros(int(np.count_nonzero(keep)), dtype=np.uint8),
        "shift_mm": np.zeros(int(np.count_nonzero(keep)), dtype=np.float64),
        "score": np.ones(int(np.count_nonzero(keep)), dtype=np.float64),
        "independent_poses": np.zeros(int(np.count_nonzero(keep)), dtype=np.int16),
        "small_gauge_mm": np.full(int(np.count_nonzero(keep)), np.nan),
        "large_gauge_mm": np.full(int(np.count_nonzero(keep)), np.nan),
        "scale_difference_mm": np.full(int(np.count_nonzero(keep)), np.nan),
        "model_p90_mm": np.full(int(np.count_nonzero(keep)), np.nan),
    }
    raw_pose_bias_report = {"enabled": False, "reason": "independent_patch_selection_disabled"}
    if (
        args.independent_patches
        and np.any(keep)
        and local_result is not None
        and "pose_residual" in local_result
    ):
        selected_evidence_bias = np.asarray(
            selection_diagnostic.get("surface_evidence_class", selection_class), dtype=np.uint8
        )[keep]
        selected_strict_bias = np.asarray(
            selection_diagnostic.get("surface_local_coherence_strict", np.ones(len(keep))),
            dtype=bool,
        )[keep]
        selected_feature_bias = np.asarray(
            selection_diagnostic.get("surface_local_feature_like", np.zeros(len(keep))), dtype=bool
        )[keep]
        selected_uncertainty_bias = np.asarray(
            local_result.get("uncertainty", np.full(len(fused_p), np.nan)), dtype=np.float64
        )[keep]
        selected_masks_bias = np.asarray(
            local_result.get("support_pose_mask", np.zeros(len(fused_p), dtype=np.uint64)),
            dtype=np.uint64,
        )[keep]
        selected_pose_residual = np.asarray(local_result["pose_residual"], dtype=np.float64)[keep]
        corrected_bias, raw_pose_bias_diagnostic, raw_pose_bias_report = (
            _postselection_raw_pose_bias_consensus(
                fused_p[keep],
                fused_n[keep],
                selected_masks_bias,
                selected_pose_residual,
                confidence[keep],
                selected_uncertainty_bias,
                selected_evidence_bias,
                selected_strict_bias,
                selected_feature_bias,
                float(args.fusion_voxel_mm),
                args,
            )
        )
        fused_p = np.asarray(fused_p, dtype=np.float64).copy()
        fused_p[keep] = corrected_bias
        # Mantener los residuos coherentes con la nueva posición: mover el surfel
        # +d sobre su normal resta d a todos los residuos firmados de esa semilla.
        actual_bias_shift = np.asarray(raw_pose_bias_diagnostic["shift_mm"], dtype=np.float64)
        local_result["pose_residual"][keep] = selected_pose_residual - actual_bias_shift[:, None]
        local_result["points"][keep] = corrected_bias
        local_summary["postselection_raw_pose_bias_consensus"] = raw_pose_bias_report
        print(
            f"[Paso 11 | sesgo crudo por pose] disponibles={raw_pose_bias_report.get('available',0):,}; "
            f"corregidos={raw_pose_bias_report.get('corrected',0):,}; "
            f"ambiguos={raw_pose_bias_report.get('ambiguous',0):,}; "
            f"alpha_global={raw_pose_bias_report.get('accepted_global_alpha',0):.2f}.",
            flush=True,
        )

    # V11.6: la selección ya decidió QUÉ muestras son observadas. Ahora se
    # comprueba si su posición regional puede predecirse de manera consistente
    # desde dos escalas y varias procedencias de pose. No se añaden puntos.
    regional_pose_diagnostic = {
        "available": np.zeros(int(np.count_nonzero(keep)), dtype=np.uint8),
        "accept": np.ones(int(np.count_nonzero(keep)), dtype=np.uint8),
        "ambiguous": np.zeros(int(np.count_nonzero(keep)), dtype=np.uint8),
        "feature_protected": np.zeros(int(np.count_nonzero(keep)), dtype=np.uint8),
        "shift_mm": np.zeros(int(np.count_nonzero(keep)), dtype=np.float64),
        "score": np.ones(int(np.count_nonzero(keep)), dtype=np.float64),
        "independent_predictions": np.zeros(int(np.count_nonzero(keep)), dtype=np.int16),
        "pose_dispersion_mm": np.full(int(np.count_nonzero(keep)), np.nan),
        "scale_difference_mm": np.full(int(np.count_nonzero(keep)), np.nan),
        "small_prediction_mm": np.full(int(np.count_nonzero(keep)), np.nan),
        "large_prediction_mm": np.full(int(np.count_nonzero(keep)), np.nan),
    }
    regional_pose_report = {"enabled": False, "reason": "independent_patch_selection_disabled"}
    if args.independent_patches and np.any(keep):
        selected_evidence = np.asarray(
            selection_diagnostic.get("surface_evidence_class", selection_class), dtype=np.uint8
        )[keep]
        selected_strict = np.asarray(
            selection_diagnostic.get("surface_local_coherence_strict", np.ones(len(keep))),
            dtype=bool,
        )[keep]
        selected_feature = np.asarray(
            selection_diagnostic.get("surface_local_feature_like", np.zeros(len(keep))), dtype=bool
        )[keep]
        selected_uncertainty = (
            np.asarray(
                local_result.get("uncertainty", np.full(len(fused_p), np.nan)), dtype=np.float64
            )[keep]
            if local_result is not None
            else np.full(np.count_nonzero(keep), np.nan)
        )
        selected_masks = (
            np.asarray(
                local_result.get("support_pose_mask", np.zeros(len(fused_p), dtype=np.uint64)),
                dtype=np.uint64,
            )[keep]
            if local_result is not None
            else np.zeros(np.count_nonzero(keep), dtype=np.uint64)
        )
        corrected_selected, regional_pose_diagnostic, regional_pose_report = (
            _postselection_regional_pose_consensus(
                fused_p[keep],
                fused_n[keep],
                selected_masks,
                confidence[keep],
                selected_uncertainty,
                selected_evidence,
                selected_strict,
                selected_feature,
                float(args.fusion_voxel_mm),
                args,
            )
        )
        fused_p = np.asarray(fused_p, dtype=np.float64).copy()
        fused_p[keep] = corrected_selected
        # Una clase 3 con desacuerdo regional explícito deja de ser ancla
        # autoritativa, pero no se elimina: 12 recibirá menor fuerza continua.
        full_evidence = np.asarray(
            selection_diagnostic.get("surface_evidence_class", selection_class), dtype=np.uint8
        ).copy()
        selected_evidence = full_evidence[keep].copy()
        downgrade = (regional_pose_diagnostic["ambiguous"] > 0) & (selected_evidence >= 3)
        selected_evidence[downgrade] = 2
        full_evidence[keep] = selected_evidence
        selection_diagnostic["surface_evidence_class"] = full_evidence
        local_summary["postselection_regional_pose_consensus"] = regional_pose_report
        print(
            f"[Paso 11 | consenso regional por pose] disponibles={regional_pose_report.get('available',0):,}; "
            f"corregidos={regional_pose_report.get('corrected',0):,}; "
            f"ambiguos={regional_pose_report.get('ambiguous',0):,}; "
            f"alpha_global={regional_pose_report.get('accepted_global_alpha',0):.2f}.",
            flush=True,
        )
    # Cobertura respecto a TODOS los candidatos, incluidos los que perdieron
    # respaldo durante el ajuste. No oculta pérdidas cambiando el denominador.
    coverage_report = measure_final_coverage(
        coverage_reference_points, fused_p, keep, float(args.fusion_voxel_mm))
    if np.any(keep):
        print(
            f"[Paso 11] Cobertura de candidatos originales a un vóxel: "
            f"{100*coverage_report['within_one_voxel_fraction']:.1f}%; "
            f"celdas finales/referencia: {100*coverage_report['occupied_cell_retention']:.1f}%.",
            flush=True,
        )
    coverage_report["gate_method"] = "spatial_reference_cell_coverage_v2"
    coverage_report["gate_note"] = "Cell-count retention is diagnostic only. Require original-cell proximity at one and two voxels, plus point proximity at two voxels."
    local_summary["all_candidate_coverage"] = coverage_report

    # V11.9 — contrato de cobertura mínimo. Una nube puede tener excelentes
    # métricas locales y aun así estar demasiado fragmentada para reconstruir una
    # superficie sin extrapolación. En ese caso se detiene aquí y no se delega a
    # Poisson la tarea de inventar las regiones faltantes.
    if np.any(keep):
        occupied_retention = float(coverage_report.get("reference_cell_coverage_one_voxel", 0.0))
        two_voxel_coverage = float(coverage_report.get("within_two_voxels_fraction", 0.0))
        coverage_report["minimum_required_reference_cell_coverage_one_voxel"] = float(
            args.minimum_final_occupied_cell_retention
        )
        coverage_report["minimum_required_two_voxel_coverage"] = float(
            args.minimum_final_two_voxel_coverage
        )
        coverage_report["passes_surface_sampling_gate"] = bool(
            occupied_retention >= float(args.minimum_final_occupied_cell_retention)
            and two_voxel_coverage >= float(args.minimum_final_two_voxel_coverage)
            and coverage_report["reference_cell_coverage_two_voxels"] >= float(args.minimum_final_two_voxel_coverage)
        )
    else:
        coverage_report["passes_surface_sampling_gate"] = False

    diagnostic_points = fused_p.copy()
    if args.align_platform_axis:
        diagnostic_rotation, diagnostic_origin, _ = platform_canonical_frame(
            calibration, registration
        )
        diagnostic_points = (diagnostic_points - diagnostic_origin[None, :]) @ diagnostic_rotation.T
    np.savez_compressed(
        output / "diagnostico_seleccion_parches.npz",
        points=diagnostic_points.astype(np.float32),
        coverage_reference_points=(
            (coverage_reference_points - diagnostic_origin[None, :]) @ diagnostic_rotation.T
            if args.align_platform_axis else coverage_reference_points
        ).astype(np.float64),
        selected=keep,
        support_views=support,
        confidence=confidence,
        **selection_diagnostic,
    )

    # Guardar también los candidatos no seleccionados para localizar desacuerdos.
    # El archivo permanece disponible incluso si el control mínimo detiene 11.
    if local_result is not None:
        local_diagnostics["selected_for_output"] = np.zeros(
            len(local_diagnostics["status"]), dtype=bool
        )
        local_diagnostics["selected_for_output"][local_retained] = keep
        if args.align_platform_axis:
            rot, origin, _ = platform_canonical_frame(calibration, registration)
            for name in ("points", "seed_points"):
                local_diagnostics[name] = (local_diagnostics[name] - origin[None, :]) @ rot.T
        local_diagnostic_path = output / "diagnostico_fusion_local.npz"
        np.savez_compressed(
            local_diagnostic_path,
            **local_diagnostics,
            coordinate_frame=np.asarray(
                "canonical_platform" if args.align_platform_axis else "registered_P00"
            ),
            status_codes_json=np.asarray(json.dumps(LOCAL_STATUS)),
        )
        local_summary["selected_points"] = int(np.count_nonzero(keep))
        region_coverage = []
        for rid in np.unique(local_result["region_id"]):
            group = local_result["region_id"] == rid
            reference = group & (local_result["support"] >= 2)
            cells_before = len(np.unique(unique_voxel[reference], axis=0))
            cells_after = len(np.unique(unique_voxel[reference & keep], axis=0))
            region_coverage.append(
                {
                    "region": int(rid),
                    "reference_candidates": int(reference.sum()),
                    "occupied_cells_before": cells_before,
                    "occupied_cells_after": cells_after,
                    "occupied_cell_retention_ratio": float(cells_after / max(1, cells_before)),
                    "retained": int(np.count_nonzero(reference & keep)),
                    "candidate_retention_ratio": float(
                        np.count_nonzero(reference & keep) / max(1, reference.sum())
                    ),
                }
            )
        local_summary["regional_selection_coverage"] = region_coverage

        local_summary["selected_fit_status_counts"] = {
            desc: int(np.count_nonzero(keep & (local_result["status"] == code)))
            for code, desc in LOCAL_STATUS.items()
        }
        print(
            f"[Paso 11 | fusión local] Proyecciones aceptadas: "
            f"{np.count_nonzero(local_result['status']==1):,}; "
            f"seleccionados: {np.count_nonzero(keep):,}. "
            f"Diagnóstico: {local_diagnostic_path.name}",
            flush=True,
        )

    # Persistir la auditoría también cuando el gate bloquea el mallado.
    audit = {
        "schema_version": 1,
        "quality": "accepted" if coverage_report.get("passes_surface_sampling_gate") else "rejected",
        "method": "immutable_candidate_reference_coverage_audit",
        "parameters": vars(args),
        "registration_summary": str(registration_summary_path),
        "coverage": coverage_report,
        "selection": selection_info,
        "local_surface_fusion": local_summary,
    }
    (output / "auditoria_11_cobertura.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=lambda x: x.tolist()
                   if isinstance(x, np.ndarray) else x.item()
                   if isinstance(x, np.generic) else str(x)), encoding="utf-8")
    if not bool(coverage_report.get("passes_surface_sampling_gate", False)):
        raise RuntimeError(
            "La nube fusionada conserva evidencia local pero no cobertura superficial "
            "suficiente para un mallado fiable: "
            f"celdas originales cubiertas a 1vox={100.0*float(coverage_report.get('reference_cell_coverage_one_voxel',0.0)):.1f}% "
            f"(mín {100.0*float(args.minimum_final_occupied_cell_retention):.1f}%), "
            f"celdas a 2vox={100.0*float(coverage_report.get('reference_cell_coverage_two_voxels',0.0)):.1f}%; "
            f"puntos a 2vox={100.0*float(coverage_report.get('within_two_voxels_fraction',0.0)):.1f}% "
            f"(mín {100.0*float(args.minimum_final_two_voxel_coverage):.1f}%). "
            "Revise pasos 02–10 o aumente cobertura observada; no se permite que Poisson "
            "complete automáticamente una nube insuficiente."
        )

    if int(np.count_nonzero(keep)) < 500:
        raise RuntimeError(
            "El consenso multivista dejó menos de 500 puntos. "
            "Revise el solapamiento entre vistas o ejecute explícitamente "
            "--no-hierarchical-consensus para diagnóstico."
        )

    singleton_ok = selection_class == 1

    fused_p = fused_p[keep]
    fused_c = np.clip(np.rint(fused_c[keep]), 0, 255).astype(np.uint8)
    fused_n = fused_n[keep]
    support_keep = support[keep]
    confidence_keep = confidence[keep]
    spread_keep = spread[keep]
    normal_consistency_keep = normal_consistency[keep]
    neighbor_core_keep = neighbor_core_count[keep]
    selection_class_keep = selection_class[keep]
    distance_to_anchor_keep = distance_to_anchor[keep]

    if bool(args.align_platform_axis):
        frame_rotation, frame_origin, canonical_frame = platform_canonical_frame(
            calibration, registration
        )
        fused_p = (fused_p - frame_origin[None, :]) @ frame_rotation.T
        fused_n = normalize_rows(fused_n @ frame_rotation.T)
    else:
        canonical_frame = {
            "enabled": False,
            "target_axis": "camera_frame_unchanged",
            "rigid_only": True,
            "shape_assumptions_used": False,
        }

    local_fields = {}
    if local_result is not None:
        field_map = {
            "local_fit_status": "status",
            "position_uncertainty_mm": "uncertainty",
            "fusion_displacement_mm": "displacement",
            "support_pose_mask": "support_pose_mask",
            "disagreement_pose_mask": "disagreement_pose_mask",
            "local_agreement": "agreement",
            "incompatible_observations": "incompatible",
            "local_observations": "observations",
            "independent_support_poses": "independent_support_poses",
            "support_angular_span_poses": "support_angular_span_poses",
            "conflict_pose_count": "conflict_pose_count",
            "conflict_pose_ratio": "conflict_pose_ratio",
            "heldout_validation_available": "heldout_validation_available",
            "heldout_validation_tests": "heldout_validation_tests",
            "heldout_validation_pass_count": "heldout_validation_pass_count",
            "heldout_validation_pass_ratio": "heldout_validation_pass_ratio",
            "heldout_validation_error_mm": "heldout_validation_error_mm",
            "heldout_validation_p90_mm": "heldout_validation_p90_mm",
            "validated_layer_available": "validated_layer_available",
            "validated_layer_accept": "validated_layer_accept",
            "validated_layer_count": "validated_layer_count",
            "validated_layer_ambiguous": "validated_layer_ambiguous",
            "validated_layer_resolved": "validated_layer_resolved",
            "validated_layer_selected_pose_mask": "validated_layer_selected_pose_mask",
            "validated_layer_secondary_pose_mask": "validated_layer_secondary_pose_mask",
            "validated_layer_separation_mm": "validated_layer_separation_mm",
            "validated_layer_shift_mm": "validated_layer_shift_mm",
            "validated_layer_winner_independent": "validated_layer_winner_independent",
            "validated_layer_runnerup_independent": "validated_layer_runnerup_independent",
            "validated_layer_score_margin": "validated_layer_score_margin",
            "region_id": "region_id",
            "regional_status": "regional_status",
            "regional_shift_mm": "regional_shift_mm",
            "regional_disagreement_mm": "regional_disagreement_mm",
            "geometry_class": "geometry_class",
            "geometry_cv_score": "geometry_cv_score",
            "geometry_uncertainty_mm": "geometry_sigma",
            "geometry_shift_mm": "geometry_shift_mm",
            "geometry_pose_mask": "geometry_pose_mask",
            "geometry_fold0_test_mask": "geometry_fold0_test_mask",
            "geometry_rejected": "geometry_rejected",
            "geometry_validation_folds": "geometry_validation_folds",
            "geometry_available_poses": "geometry_available_poses",
            "geometry_added_neighbors": "geometry_added_neighbors",
            "geometry_patch_id": "geometry_patch_id",
            "geometry_reject_plane": "geometry_reject_plane",
            "geometry_reject_cylinder": "geometry_reject_cylinder",
            "geometry_reject_curve": "geometry_reject_curve",
            "geometry_filtered_observations": "geometry_filtered_observations",
            "raw_pose_residual_mm": "pose_residual",
        }
        local_fields = {name: local_result[source][keep] for name, source in field_map.items()}
    surface_evidence_class_keep = np.asarray(
        selection_diagnostic.get("surface_evidence_class", selection_class), dtype=np.uint8
    )[keep]
    npz_path = output / "nube_fusionada_multivista.npz"
    np.savez_compressed(
        npz_path,
        points=fused_p.astype(np.float32),
        colors=fused_c,
        normals=fused_n.astype(np.float32),
        support_views=support_keep.astype(np.uint16),
        confidence=confidence_keep.astype(np.float32),
        spread_mm=spread_keep.astype(np.float32),
        normal_consistency=normal_consistency_keep.astype(np.float32),
        neighboring_core_voxels=neighbor_core_keep.astype(np.int16),
        consensus_selection_class=selection_class_keep.astype(np.uint8),
        surface_evidence_class=surface_evidence_class_keep.astype(np.uint8),
        distance_to_strong_core_mm=distance_to_anchor_keep.astype(np.float32),
        surface_evidence_class_pre_coherence=np.asarray(
            selection_diagnostic.get(
                "surface_evidence_class_pre_coherence", surface_evidence_class_keep
            ),
            dtype=np.uint8,
        )[keep],
        surface_local_coherence_available=np.asarray(
            selection_diagnostic.get("surface_local_coherence_available", np.ones(len(keep))),
            dtype=np.uint8,
        )[keep],
        surface_local_coherence_accept=np.asarray(
            selection_diagnostic.get("surface_local_coherence_accept", np.ones(len(keep))),
            dtype=np.uint8,
        )[keep],
        surface_local_coherence_strict=np.asarray(
            selection_diagnostic.get("surface_local_coherence_strict", np.ones(len(keep))),
            dtype=np.uint8,
        )[keep],
        surface_local_feature_like=np.asarray(
            selection_diagnostic.get("surface_local_feature_like", np.zeros(len(keep))),
            dtype=np.uint8,
        )[keep],
        surface_local_coherence_score=np.asarray(
            selection_diagnostic.get("surface_local_coherence_score", np.ones(len(keep))),
            dtype=np.float32,
        )[keep],
        surface_local_coherence_red_flags=np.asarray(
            selection_diagnostic.get("surface_local_coherence_red_flags", np.zeros(len(keep))),
            dtype=np.uint8,
        )[keep],
        surface_local_plane_p90_small_mm=np.asarray(
            selection_diagnostic.get(
                "surface_local_plane_p90_small_mm", np.full(len(keep), np.nan)
            ),
            dtype=np.float32,
        )[keep],
        surface_local_plane_p90_large_mm=np.asarray(
            selection_diagnostic.get(
                "surface_local_plane_p90_large_mm", np.full(len(keep), np.nan)
            ),
            dtype=np.float32,
        )[keep],
        surface_local_normal_p90_large_deg=np.asarray(
            selection_diagnostic.get(
                "surface_local_normal_p90_large_deg", np.full(len(keep), np.nan)
            ),
            dtype=np.float32,
        )[keep],
        surface_local_scale_normal_difference_deg=np.asarray(
            selection_diagnostic.get(
                "surface_local_scale_normal_difference_deg", np.full(len(keep), np.nan)
            ),
            dtype=np.float32,
        )[keep],
        surface_local_variation_large=np.asarray(
            selection_diagnostic.get("surface_local_variation_large", np.full(len(keep), np.nan)),
            dtype=np.float32,
        )[keep],
        surface_local_same_sheet_fraction=np.asarray(
            selection_diagnostic.get(
                "surface_local_same_sheet_fraction", np.full(len(keep), np.nan)
            ),
            dtype=np.float32,
        )[keep],
        coverage_recovered_observation=np.asarray(
            selection_diagnostic.get(
                "coverage_recovered_observation", np.zeros(len(keep))
            ),
            dtype=np.uint8,
        )[keep],
        coverage_recovery_iteration=np.asarray(
            selection_diagnostic.get(
                "coverage_recovery_iteration", np.zeros(len(keep))
            ),
            dtype=np.uint8,
        )[keep],
        coverage_recovery_score=np.asarray(
            selection_diagnostic.get(
                "coverage_recovery_score", np.zeros(len(keep))
            ),
            dtype=np.float32,
        )[keep],
        observed_coverage_recovery_contract_valid=np.asarray(
            [1 if bool(args.observed_coverage_recovery) else 0], dtype=np.uint8
        ),
        regional_pose_consensus_available=np.asarray(
            regional_pose_diagnostic["available"], dtype=np.uint8
        ),
        regional_pose_consensus_accept=np.asarray(
            regional_pose_diagnostic["accept"], dtype=np.uint8
        ),
        regional_pose_consensus_ambiguous=np.asarray(
            regional_pose_diagnostic["ambiguous"], dtype=np.uint8
        ),
        regional_pose_consensus_feature_protected=np.asarray(
            regional_pose_diagnostic["feature_protected"], dtype=np.uint8
        ),
        regional_pose_consensus_shift_mm=np.asarray(
            regional_pose_diagnostic["shift_mm"], dtype=np.float32
        ),
        regional_pose_consensus_score=np.asarray(
            regional_pose_diagnostic["score"], dtype=np.float32
        ),
        regional_pose_consensus_independent_predictions=np.asarray(
            regional_pose_diagnostic["independent_predictions"], dtype=np.int16
        ),
        regional_pose_consensus_pose_dispersion_mm=np.asarray(
            regional_pose_diagnostic["pose_dispersion_mm"], dtype=np.float32
        ),
        regional_pose_consensus_scale_difference_mm=np.asarray(
            regional_pose_diagnostic["scale_difference_mm"], dtype=np.float32
        ),
        regional_pose_consensus_small_prediction_mm=np.asarray(
            regional_pose_diagnostic["small_prediction_mm"], dtype=np.float32
        ),
        regional_pose_consensus_large_prediction_mm=np.asarray(
            regional_pose_diagnostic["large_prediction_mm"], dtype=np.float32
        ),
        regional_pose_consensus_contract_valid=np.asarray(
            [1 if bool(args.regional_pose_consensus) else 0], dtype=np.uint8
        ),
        raw_pose_bias_available=np.asarray(raw_pose_bias_diagnostic["available"], dtype=np.uint8),
        raw_pose_bias_accept=np.asarray(raw_pose_bias_diagnostic["accept"], dtype=np.uint8),
        raw_pose_bias_ambiguous=np.asarray(raw_pose_bias_diagnostic["ambiguous"], dtype=np.uint8),
        raw_pose_bias_feature_protected=np.asarray(
            raw_pose_bias_diagnostic["feature_protected"], dtype=np.uint8
        ),
        raw_pose_bias_shift_mm=np.asarray(raw_pose_bias_diagnostic["shift_mm"], dtype=np.float32),
        raw_pose_bias_score=np.asarray(raw_pose_bias_diagnostic["score"], dtype=np.float32),
        raw_pose_bias_independent_poses=np.asarray(
            raw_pose_bias_diagnostic["independent_poses"], dtype=np.int16
        ),
        raw_pose_bias_small_gauge_mm=np.asarray(
            raw_pose_bias_diagnostic["small_gauge_mm"], dtype=np.float32
        ),
        raw_pose_bias_large_gauge_mm=np.asarray(
            raw_pose_bias_diagnostic["large_gauge_mm"], dtype=np.float32
        ),
        raw_pose_bias_scale_difference_mm=np.asarray(
            raw_pose_bias_diagnostic["scale_difference_mm"], dtype=np.float32
        ),
        raw_pose_bias_model_p90_mm=np.asarray(
            raw_pose_bias_diagnostic["model_p90_mm"], dtype=np.float32
        ),
        raw_pose_bias_contract_valid=np.asarray(
            [1 if bool(args.raw_pose_bias_consensus) else 0], dtype=np.uint8
        ),
        local_coherence_contract_valid=np.asarray([1], dtype=np.uint8),
        validated_pose_layer_contract_valid=np.asarray(
            [1 if bool(args.validated_pose_layer_separation) else 0], dtype=np.uint8
        ),
        strong_core_support_threshold=np.asarray(
            [selection_info["anchor_support_threshold_used"]],
            dtype=np.uint16,
        ),
        fusion_voxel_mm=np.asarray([args.fusion_voxel_mm], dtype=np.float32),
        **local_fields,
    )
    # Synthesized geometry is exported separately and NEVER counted as poses.
    # Antes de generarla se reconstruye el mismo contrato de siluetas/visibilidad
    # usado por 10. Si no puede verificarse, el completado se omite de forma segura.
    completion_context, completion_context_report = load_completion_validation_context(
        root, args, registration, views, pose_transforms, canonical_frame
    )
    inferred, completion_report = estimate_missing_patches(
        fused_p,
        fused_n,
        fused_c,
        float(args.fusion_voxel_mm),
        args,
        completion_context=completion_context,
        completion_context_report=completion_context_report,
    )
    import hashlib

    source_digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    completion_path = output / "completado_geometrico_estimado.npz"
    np.savez_compressed(
        completion_path,
        **inferred,
        support_views=np.zeros(len(inferred["points"]), np.uint16),
        confidence=np.full(len(inferred["points"]), 0.20, np.float32),
        is_inferred=np.ones(len(inferred["points"]), bool),
        observed_npz_sha256=np.asarray(source_digest),
        voxel_mm=np.asarray([args.fusion_voxel_mm]),
        coordinate_frame=np.asarray(
            "canonical_platform" if args.align_platform_axis else "registered_P00"
        ),
        schema_version=np.asarray([2]),
        report_json=np.asarray(json.dumps(completion_report)),
    )
    ply_path = output / "nube_fusionada_multivista.ply"
    save_ply(ply_path, fused_p, fused_c, fused_n, support_keep, confidence_keep)

    preview_path = output / "preview_fusion_multivista.png"
    make_preview(preview_path, fused_p, support_keep, confidence_keep)
    completion_preview = None
    stale_completion_preview = output / "preview_completado_estimado.png"
    if plt is not None and len(inferred["points"]):
        completion_preview = stale_completion_preview
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        sample = np.linspace(0, len(fused_p) - 1, min(25000, len(fused_p)), dtype=int)
        for ax, (a, b), title in zip(axes, ((0, 1), (0, 2), (2, 1)), ("X-Y", "X-Z", "Z-Y")):
            ax.scatter(fused_p[sample, a], fused_p[sample, b], s=1, c="0.65", label="Observado")
            ax.scatter(
                inferred["points"][:, a],
                inferred["points"][:, b],
                s=2,
                c="tab:orange",
                label="Estimado",
            )
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="box")
        axes[0].legend(markerscale=3)
        fig.suptitle("Paso 11 | Guías estimadas para el mallado; no son mediciones")
        fig.tight_layout()
        fig.savefig(completion_preview, dpi=140)
        plt.close(fig)
    elif stale_completion_preview.exists():
        # Evita confundir una previsualización inferida de una ejecución vieja
        # con una ejecución nueva que generó cero puntos estimados.
        try:
            stale_completion_preview.unlink()
        except OSError:
            pass

    print("[Paso 11 | 4/4] Guardando nube fusionada y resumen...")
    summary = {
        "schema_version": 9,
        "method": (
            "step10_clean_views_pose_balanced_surface_patch_selection_"
            + (
                "neighboring_quadratic_surfel_fusion"
                if args.local_surface_fusion
                else "voxel_surfel_fusion"
            )
        ),
        "implementation": {
            "version": "11.8",
            "local_surface_fusion": bool(args.local_surface_fusion),
            "regional_fusion": bool(args.local_surface_fusion and args.regional_fusion),
        },
        "local_surface_fusion": local_summary,
        "geometric_completion": completion_report,
        "object": obj,
        "shape_specific_assumptions": False,
        "registration_reoptimized": False,
        "calibration": str(calibration_path),
        "registration_summary": str(registration_summary_path),
        "registration_quality": registration.get("quality"),
        "registration_health": registration.get("registration_health"),
        "registration_pose_weighting": registration_pose_weight_report,
        "pose_transform_contract": pose_transform_contract,
        "canonical_platform_frame": canonical_frame,
        "registered_clean_folder": str(registered_clean_folder),
        "parameters": vars(args),
        "input": {
            "poses": 25,
            "prevoxel_applied": not bool(args.local_surface_fusion),
            "prevoxel_points_total": int(len(points)),
            "per_pose": per_pose_records,
        },
        "fusion": {
            "candidate_voxels": int(n_vox),
            "candidate_count_semantics": (
                "surface_patches" if args.local_surface_fusion else "occupied_voxels"
            ),
            "spread_semantics": (
                "normal_surface_residual" if args.local_surface_fusion else "euclidean_voxel_spread"
            ),
            "core_voxels": int(np.count_nonzero(core)),
            "boundary_singletons_preserved": int(np.count_nonzero(singleton_ok)),
            "hierarchical_consensus": selection_info,
            "final_points": int(len(fused_p)),
            "support_views": finite_stats(support_keep),
            "confidence": finite_stats(confidence_keep),
            "spread_mm": finite_stats(spread_keep),
            "normal_consistency": finite_stats(normal_consistency_keep),
            "support_ge_2_ratio": float(np.mean(support_keep >= 2)) if len(support_keep) else 0.0,
            "support_ge_3_ratio": float(np.mean(support_keep >= 3)) if len(support_keep) else 0.0,
            "support_ge_4_ratio": float(np.mean(support_keep >= 4)) if len(support_keep) else 0.0,
        },
        "outputs": {
            "npz": str(npz_path),
            "ply": str(ply_path),
            "preview": str(preview_path),
            "estimated_completion": str(completion_path),
            "estimated_completion_preview": str(completion_preview) if completion_preview else None,
            "local_diagnostics": str(local_diagnostic_path) if local_diagnostic_path else None,
        },
        "important_note": (
            "La entrada es exclusivamente P00..P24 registrada y tallada por el "
            "paso 10; no se recargan ni se vuelven a transformar las nubes del "
            "paso 06. Si 10 publica WARNING, la fusión sigue siendo auditable "
            "pero la autoridad de cada pose se modula además por la calidad de "
            "sus aristas consecutivas; nunca se corrigen las poses en 11. "
            "La salida se expresa mediante una transformación rígida "
            "en el marco de la plataforma, con el eje físico como +Y. El voxel "
            "organiza observaciones. La ruta local conserva contribuciones por pose "
            "hasta resolver cada parche; ajusta superficies cuadráticas locales "
            "con desplazamiento acotado por incertidumbre empírica, preservando "
            "observaciones cuando no se acredita el ajuste. La ruta legacy usa "
            "medias ponderadas. El núcleo se elige mediante soporte, "
            "densidad, continuidad local y consistencia multivista; las extensiones "
            "usan proximidad local. No se clasifica la forma ni se modifica ninguna "
            "pose. El modo de parches independientes sustituye la restricción "
            "de distancia al núcleo. V11.8 puede recuperar únicamente observaciones "
            "reales descartadas por exceso de conservadurismo local cuando mantienen "
            "evidencia multivista, continuidad y compatibilidad de normales; la "
            "recuperación se revierte si no mejora cobertura o degrada materialmente "
            "la calidad global. El completado se guarda separado, no cuenta "
            "como evidencia observada y solo se exporta si cada guía supera "
            "validación multivista por silueta, visibilidad, profundidad y espacio libre."
        ),
    }
    summary_path = output / "resumen_11_fusion_multivista.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output / "estadisticas_por_pose_11.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "pose_index",
            "points_before",
            "points_after_prevoxel",
            "view_quality_weight",
            "consensus_pose_weight",
            "registration_pose_weight",
            "median_step10_reliability",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in per_pose_records:
            writer.writerow({k: r[k] for k in fields})

    print("\n========== PASO 11 COMPLETADO ==========")
    print(f"Observaciones usadas: {len(points):,}")
    print(f"Candidatos de superficie: {n_vox:,}")
    print(
        "Modo de consenso:",
        selection_info["mode"],
        f"| núcleo soporte>={selection_info['anchor_support_threshold_used']}",
    )
    print(f"Puntos del núcleo fuerte: {selection_info['anchor_points']:,}")
    print(f"Extensión local aceptada: {selection_info['local_extension_points']:,}")
    print(f"Singletons de compatibilidad: {np.count_nonzero(singleton_ok):,}")
    print(f"Nube fusionada final: {len(fused_p):,}")
    print(f"Soporte mediano: {np.median(support_keep):.2f}" if len(support_keep) else "Sin puntos")
    print("Salida:", output)
    print("===========================================\n")
    return 0


def _con_estado_visible(funcion, descripcion):
    """Envuelve una operación para publicar y restaurar su estado de progreso."""
    import functools

    @functools.wraps(funcion)
    def ejecutar(*args, **kwargs):
        global _estado_actual
        # Los procesos auxiliares no escriben estados que compitan con el padre.
        if __name__ != "__main__":
            return funcion(*args, **kwargs)
        with _estado_lock:
            anterior = _estado_actual
        _estado11(descripcion)
        try:
            return funcion(*args, **kwargs)
        except Exception as exc:
            print(f"[ERROR] Paso 11 | {descripcion} | {type(exc).__name__}: {exc}", flush=True)
            raise
        finally:
            with _estado_lock:
                _estado_actual = anterior

    return ejecutar


load_registered_clean_views = _con_estado_visible(
    load_registered_clean_views, "Leyendo observaciones registradas por pose"
)
neighboring_surface_fusion = _con_estado_visible(
    neighboring_surface_fusion, "Comparando posiciones, normales y capas entre poses"
)
select_independent_patches = _con_estado_visible(
    select_independent_patches, "Seleccionando parches con respaldo propio"
)
estimate_missing_patches = _con_estado_visible(
    estimate_missing_patches, "Evaluando completado separado de geometría observada"
)
save_ply = _con_estado_visible(save_ply, "Guardando nube fusionada")
make_preview = _con_estado_visible(make_preview, "Generando vista previa")


def _entrada_con_diagnostico():
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    from utilidades_progreso import informar_inicio

    informar_inicio(__file__, "11", "Fusionar nubes multivista")
    try:
        raise SystemExit(main())
    finally:
        _estado_stop.set()


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
