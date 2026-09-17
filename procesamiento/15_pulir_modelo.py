#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paso 15 — Pulido de malla con protección geométrica y topológica.

Consume la malla reparada por el paso 14 y la nube de referencia del paso 12.
Reduce ondulaciones y rugosidad sin asumir una clase geométrica del objeto.
La conectividad recibida permanece fija: no se añaden ni eliminan vértices
o triángulos y no se cierran bordes durante el pulido.

Método
------
- Filtrado bilateral de normales y tendencia cuadrática local multiescala.
- Corrección robusta de salientes en vecindarios conectados de la malla.
- Regularización Taubin con protección de aristas y bordes abiertos.
- Acabado residual adaptativo mediante ajustes cuadráticos robustos (IRLS).
- Corrección contextual limitada, con evidencia de la nube de referencia.

Los desplazamientos se limitan según el espaciado físico. El acabado adicional
usa por defecto un máximo de 0.28 veces el espaciado de referencia y puede
desactivarse con --no-residual-finish. Las superficies locales describen la
curvatura del vecindario; no imponen planos, cilindros ni dimensiones conocidas.

Protecciones
------------
Los bordes abiertos, las aristas persistentes y las costuras protegidas
restringen el movimiento. Se verifica la procedencia de las costuras por hash.
Los vértices geométricamente coincidentes reciben desplazamientos compatibles
para no abrir separaciones artificiales en la malla.

Las guardas comprueban orientación de caras, dimensiones, intersecciones y
fidelidad respecto a la nube. Ante conflictos se reduce el desplazamiento
local con una transición suave; el retroceso global queda como alternativa
de seguridad. El clasificador semántico comparte criterios con el paso 14.
Su tolerancia usa el espaciado de ese paso; el espaciado del paso 12 limita
la magnitud del pulido. Un candidato rechazado conserva la salida previa.

Salidas principales en 15_pulido_final/
-------------------------------------
malla_antes_regularizacion.ply, malla_regularizada_general.ply,
malla_final_topologica.ply, desplazamiento_vertices_mm.npy,
proteccion_vertices.npz, preview_regularizacion_malla.png y
resumen_15_pulido_final.json.

La malla resultante pasa a la validación de intersecciones del paso 16,
la validación final del paso 17 y la exportación del paso 18.
"""

from __future__ import annotations

# Monitor independiente de las utilidades instaladas en la aplicación.
# Solo se inicia al ejecutar este archivo, nunca al importarlo.
import sys as _sys
import time as _time
import threading as _threading
import atexit as _atexit

_estado_lock = _threading.Lock()
_estado_inicio = _time.monotonic()
_estado_actual = ("Cargando dependencias de geometría", _estado_inicio)
_estado_stop = _threading.Event()


def _mostrar_estado15():
    """Publica la actividad actual y los tiempos transcurridos con lectura protegida."""
    with _estado_lock:
        actividad, inicio = _estado_actual
    ahora = _time.monotonic()
    print(
        f"[PROGRESO] Paso 15 | {actividad} | Tiempo actividad: {ahora-inicio:.1f} s | "
        f"Tiempo total: {ahora-_estado_inicio:.1f} s",
        flush=True,
    )


def _estado15(actividad):
    """Actualiza la actividad bajo bloqueo y publica el nuevo estado."""
    global _estado_actual
    with _estado_lock:
        _estado_actual = (actividad, _time.monotonic())
    _mostrar_estado15()


def _latido15():
    """Publica periódicamente el estado hasta recibir la señal de detención."""
    while not _estado_stop.wait(10):
        _mostrar_estado15()


if __name__ == "__main__":
    for _stream in (_sys.stdout, _sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(line_buffering=True, write_through=True)
            except (ValueError, OSError):
                pass
    print("[EJECUTANDO] 15_pulir_modelo.py | Iniciando procesamiento", flush=True)
    _mostrar_estado15()
    _monitor15 = _threading.Thread(target=_latido15, name="estado_paso15", daemon=True)
    _monitor15.start()
    _atexit.register(_estado_stop.set)

from utilidades_progreso import operacion
from utilidades_rendimiento import query_threads
from utilidades_rendimiento import configurar as _configurar_recursos

# Los runtimes numéricos leen estos límites al importarse.
_configurar_recursos(__file__)

from utilidades_rendimiento import ejecutar_bloques, contexto

import argparse
import json
import hashlib
import math
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception as exc:
    raise SystemExit(f"paso 15 requiere SciPy: {exc}")

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import open3d as o3d
except Exception:
    o3d = None


IMPLEMENTATION_VERSION = "V1.25"
IMPLEMENTATION_NAME = "paso 15 pulido optimizado + acabado residual localizado"


def script_identity():
    """Identidad inequívoca del archivo realmente ejecutado."""
    script_path = Path(__file__).resolve()
    try:
        sha256 = hashlib.sha256(script_path.read_bytes()).hexdigest()
    except Exception:
        sha256 = None

    return {
        "version": IMPLEMENTATION_VERSION,
        "name": IMPLEMENTATION_NAME,
        "script_path": str(script_path),
        "script_sha256": sha256,
        "normal_multiscale_available": True,
        "adaptive_surface_fairing_available": True,
        "localized_residual_finish_available": True,
    }


def parser():
    """Construye las opciones de línea de comandos de este paso."""
    p = argparse.ArgumentParser(
        description="Paso 15 V1.25 — pulido optimizado y acabado residual localizado."
    )
    p.add_argument("--root", required=True)
    p.add_argument("--object", required=True)
    p.add_argument(
        "--mesh-source",
        default="14_limpieza_topologica",
    )
    p.add_argument(
        "--cloud-source",
        default="12_regularizacion_nube",
    )
    p.add_argument(
        "--topology-source",
        default="14_limpieza_topologica",
        help=(
            "Fuente del resumen Paso 14. Su "
            "estimated_point_spacing_mm se usa para coincident-lock y "
            "para la guarda semántica, con la misma escala física que usa Paso 14."
        ),
    )
    p.add_argument(
        "--output-name",
        default="15_pulido_final",
    )

    # V1.25 — el pulido y sus guardas usan únicamente evidencia multivista
    # validada por 11/12; soporte bruto de poses vecinas no cuenta como ancla.
    p.add_argument(
        "--require-evidence-contract", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--minimum-evidence-class", type=int, default=2)
    p.add_argument("--minimum-independent-support", type=int, default=2)
    p.add_argument("--minimum-angular-span-poses", type=int, default=2)
    p.add_argument("--maximum-conflict-pose-ratio", type=float, default=0.35)
    p.add_argument("--minimum-heldout-pass-ratio", type=float, default=0.67)

    # V1.4 — campo de normales + tendencia multiescala.
    p.add_argument(
        "--normal-trend-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--normal-filter-iterations",
        type=int,
        default=5,
    )
    p.add_argument(
        "--normal-filter-angle-sigma-deg",
        type=float,
        default=20.0,
    )
    p.add_argument(
        "--normal-filter-max-cross-angle-deg",
        type=float,
        default=60.0,
    )
    p.add_argument(
        "--normal-filter-self-weight",
        type=float,
        default=1.20,
    )
    p.add_argument(
        "--normal-filter-edge-weight-floor",
        type=float,
        default=0.32,
        help=(
            "Evita que una saliente de escala media quede totalmente "
            "congelada por el detector feature-aware. Las discontinuidades "
            "reales siguen bloqueadas por la diferencia angular bilateral."
        ),
    )
    p.add_argument(
        "--normal-trend-cycles",
        type=int,
        default=1,
        help=(
            "Un ciclo suele ser suficiente; las guardas posteriores limitan "
            "cualquier corrección excesiva."
        ),
    )
    p.add_argument(
        "--normal-trend-small-radius-spacing",
        type=float,
        default=4.0,
    )
    p.add_argument(
        "--normal-trend-large-radius-spacing",
        type=float,
        default=14.0,
        help=(
            "Escala grande usada como tendencia. Sigue siendo local y se "
            "expresa en múltiplos del spacing físico."
        ),
    )
    p.add_argument(
        "--normal-trend-max-neighbors",
        type=int,
        default=500,
    )
    p.add_argument(
        "--normal-trend-query-chunk",
        type=int,
        default=1200,
        help=(
            "Número de vértices consultados simultáneamente en KDTree. "
            "Mantiene bajo el uso de RAM aun con vecindarios grandes."
        ),
    )
    p.add_argument(
        "--normal-trend-min-small-neighbors",
        type=int,
        default=16,
    )
    p.add_argument(
        "--normal-trend-min-large-neighbors",
        type=int,
        default=32,
    )
    p.add_argument(
        "--normal-trend-neighbor-angle-deg",
        type=float,
        default=34.0,
        help=(
            "Vecinos con normales muy distintas se excluyen del ajuste, "
            "evitando cruzar esquinas/aristas reales."
        ),
    )
    p.add_argument(
        "--normal-trend-irls-iterations",
        type=int,
        default=3,
    )
    p.add_argument(
        "--normal-trend-huber-k",
        type=float,
        default=1.5,
    )
    p.add_argument(
        "--normal-trend-scale-difference-threshold-spacing",
        type=float,
        default=0.035,
        help=(
            "Diferencia mínima entre predicción pequeña/grande antes de "
            "considerar que existe detalle de escala media."
        ),
    )
    p.add_argument(
        "--normal-trend-scale-difference-transition-spacing",
        type=float,
        default=0.16,
    )
    p.add_argument(
        "--normal-trend-strength",
        type=float,
        default=0.88,
    )
    p.add_argument(
        "--normal-trend-max-shift-per-cycle-spacing",
        type=float,
        default=0.48,
    )
    p.add_argument(
        "--normal-trend-max-total-shift-spacing",
        type=float,
        default=0.75,
    )

    # V1.3 — despiking robusto de salientes locales.
    p.add_argument(
        "--despike-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--despike-rings",
        type=int,
        default=3,
        help="Número de anillos topológicos usados para la superficie local.",
    )
    p.add_argument(
        "--despike-min-neighbors",
        type=int,
        default=18,
    )
    p.add_argument(
        "--despike-zscore-threshold",
        type=float,
        default=3.0,
        help="Umbral robusto |residuo|/(1.4826*MAD).",
    )
    p.add_argument(
        "--despike-isolation-ratio",
        type=float,
        default=0.58,
        help=(
            "Fracción máxima de vecinos que también pueden ser outliers "
            "del mismo signo para considerar la saliente como aislada."
        ),
    )
    p.add_argument(
        "--despike-correction-strength",
        type=float,
        default=0.84,
    )
    p.add_argument(
        "--despike-max-shift-spacing",
        type=float,
        default=0.90,
    )
    p.add_argument(
        "--despike-feature-strength-min",
        type=float,
        default=0.38,
        help=(
            "Solo se corrigen vértices cuya protección feature-aware sea " "al menos este valor."
        ),
    )
    p.add_argument(
        "--despike-irls-iterations",
        type=int,
        default=3,
    )
    p.add_argument(
        "--despike-huber-k",
        type=float,
        default=1.5,
    )

    # Taubin moderado.
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--lambda-factor", type=float, default=0.44)
    p.add_argument("--mu-factor", type=float, default=-0.29)
    p.add_argument("--normal-component", type=float, default=1.0)
    p.add_argument("--tangential-component", type=float, default=0.10)

    # V1.11 — acabado residual para rugosidad espacialmente correlacionada.
    p.add_argument(
        "--surface-fairing-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Activa el pulido residual por parches cuadráticos locales. "
            "No supone ninguna forma geométrica."
        ),
    )
    p.add_argument(
        "--surface-fairing-cycles",
        type=int,
        default=4,
    )
    p.add_argument(
        "--surface-fairing-rings",
        type=int,
        default=5,
        help=(
            "Radio topológico del parche. Cinco anillos eliminan grumos "
            "de varias celdas sin convertir el ajuste en una forma global."
        ),
    )
    p.add_argument(
        "--surface-fairing-min-neighbors",
        type=int,
        default=24,
    )
    p.add_argument(
        "--surface-fairing-neighbor-angle-deg",
        type=float,
        default=32.0,
        help=("Impide que el ajuste atraviese aristas o pliegues reales."),
    )
    p.add_argument(
        "--surface-fairing-irls-iterations",
        type=int,
        default=3,
    )
    p.add_argument(
        "--surface-fairing-huber-k",
        type=float,
        default=1.35,
    )
    p.add_argument(
        "--surface-fairing-activation-threshold-spacing",
        type=float,
        default=0.018,
        help=("Residuo normal mínimo antes de intervenir, relativo al spacing."),
    )
    p.add_argument(
        "--surface-fairing-activation-transition-spacing",
        type=float,
        default=0.10,
    )
    p.add_argument(
        "--surface-fairing-noise-scale-factor",
        type=float,
        default=0.45,
        help=("Evita perseguir variaciones menores que el ruido robusto local."),
    )
    p.add_argument(
        "--surface-fairing-strength",
        type=float,
        default=0.72,
    )
    p.add_argument(
        "--surface-fairing-feature-strength-min",
        type=float,
        default=0.12,
        help=(
            "Por debajo de esta movilidad la arista se considera persistente " "y queda inmóvil."
        ),
    )
    p.add_argument(
        "--surface-fairing-max-shift-per-cycle-spacing",
        type=float,
        default=0.24,
    )
    p.add_argument(
        "--surface-fairing-max-total-shift-spacing",
        type=float,
        default=1.05,
    )
    p.add_argument(
        "--surface-fairing-convergence-spacing",
        type=float,
        default=0.004,
        help=("Detiene ciclos extra cuando el P90 del movimiento es menor."),
    )

    # Protección de geometría.
    p.add_argument("--feature-soft-angle-deg", type=float, default=28.0)
    p.add_argument("--feature-hard-angle-deg", type=float, default=65.0)
    p.add_argument("--feature-min-strength", type=float, default=0.05)
    p.add_argument("--boundary-ring-strength", type=float, default=0.30)
    p.add_argument("--edge-crossing-min-weight", type=float, default=0.04)
    p.add_argument(
        "--feature-guide-iterations",
        type=int,
        default=4,
        help=(
            "Suavizado SOLO de una copia guía usada para distinguir "
            "rugosidad de alta frecuencia de aristas persistentes."
        ),
    )
    p.add_argument(
        "--feature-guide-lambda",
        type=float,
        default=0.38,
    )

    # Límites métricos.
    p.add_argument(
        "--max-shift-per-pass-spacing",
        type=float,
        default=0.28,
    )
    p.add_argument(
        "--max-total-shift-spacing",
        type=float,
        default=0.85,
    )
    p.add_argument(
        "--max-robust-extent-change-ratio",
        type=float,
        default=0.012,
        help="Máximo cambio permitido en extent P01-P99 por eje (1.2%%).",
    )
    p.add_argument(
        "--max-orientation-flip-ratio",
        type=float,
        default=0.0,
        help="V1.1: no se permite ningún volteo de triángulo.",
    )
    p.add_argument(
        "--safety-bisection-steps",
        type=int,
        default=14,
    )
    p.add_argument(
        "--safety-local-rollback-rings",
        type=int,
        default=2,
    )
    p.add_argument(
        "--safety-local-max-expansions",
        type=int,
        default=8,
    )
    p.add_argument(
        "--safety-local-ring-floor",
        type=float,
        default=1.0,
        help=(
            "V1.8: a beta=0 toda la banda afectada vuelve exactamente "
            "a la geometría segura; evita fallback global alpha=0."
        ),
    )

    # V1.1 — guarda semántica idéntica a Paso 14.
    p.add_argument(
        "--semantic-intersection-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--semantic-guard-bisection-steps",
        type=int,
        default=16,
        help=("Bisección del alpha LOCAL. También se usa en el fallback " "global de seguridad."),
    )
    p.add_argument(
        "--semantic-local-rollback-rings",
        type=int,
        default=2,
        help=(
            "Anillos de transición alrededor de los triángulos que "
            "participan en una intersección."
        ),
    )
    p.add_argument(
        "--semantic-local-rollback-max-expansions",
        type=int,
        default=8,
        help=(
            "Máximo de expansiones de la región si el rollback local "
            "desplaza el conflicto hacia su frontera."
        ),
    )
    p.add_argument(
        "--semantic-local-rollback-ring-floor",
        type=float,
        default=1.0,
        help=(
            "Peso mínimo de rollback en el anillo exterior. Los vértices "
            "directamente conflictivos siempre tienen peso 1."
        ),
    )
    p.add_argument(
        "--semantic-geometric-epsilon-spacing-factor",
        type=float,
        default=1e-4,
    )
    p.add_argument(
        "--semantic-contact-locality-spacing-factor",
        type=float,
        default=1e-3,
    )

    # Acabado adicional tras las guardas: desactivable para recuperar V1.11.
    p.add_argument(
        "--residual-finish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Acabado residual local tras el pulido optimizado.",
    )
    p.add_argument("--residual-finish-cycles", type=int, choices=(1, 2), default=2)
    p.add_argument("--residual-finish-strength", type=float, default=0.60)
    p.add_argument(
        "--residual-finish-max-shift-spacing",
        type=float,
        default=0.28,
        help="Máximo desplazamiento adicional como fracción del spacing; hasta 0.35.",
    )

    # Diagnóstico.
    p.add_argument("--cloud-evaluation-samples", type=int, default=40000)
    p.add_argument("--preview-max-vertices", type=int, default=70000)
    return p


def require_open3d():
    """Detiene el paso con un diagnóstico si Open3D no está disponible."""
    if o3d is None:
        raise SystemExit("paso 15 requiere Open3D 0.19.x en el entorno de la tesis.")


def stats(a):
    x = np.asarray(a, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(len(x)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def safe_norm_rows(v):
    a = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(a, axis=1)
    out = np.zeros_like(a)
    good = np.isfinite(n) & (n > 1e-15)
    out[good] = a[good] / n[good, None]
    return out, n


def face_geometry(vertices, triangles):
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)]
    cross = np.cross(
        tri[:, 1] - tri[:, 0],
        tri[:, 2] - tri[:, 0],
    )
    normals, double_area = safe_norm_rows(cross)
    return normals, 0.5 * double_area


def vertex_normals(vertices, triangles):
    v = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)
    tri = v[t]
    cross = np.cross(
        tri[:, 1] - tri[:, 0],
        tri[:, 2] - tri[:, 0],
    )
    vn = np.zeros_like(v)
    for c in range(3):
        np.add.at(vn, t[:, c], cross)
    vn, _ = safe_norm_rows(vn)
    bad = np.linalg.norm(vn, axis=1) <= 1e-15
    vn[bad] = np.asarray([0.0, 0.0, 1.0])
    return vn


def smoothstep01(x):
    """Interpola suavemente en el intervalo unitario con valores extremos acotados."""
    z = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return z * z * (3.0 - 2.0 * z)


def build_coincident_vertex_groups(
    vertices,
    *,
    tolerance_mm,
):
    """Agrupa vértices cuya posición geométrica es coincidente.

    La tolerancia debe ser la misma `geometric_epsilon` usada por Paso 14.
    Se usa Union-Find para resolver componentes transitivas.
    """
    p = np.asarray(vertices, dtype=np.float64)
    n = len(p)

    if n == 0:
        return []

    tol = float(tolerance_mm)
    if not np.isfinite(tol) or tol <= 0:
        return []

    tree = cKDTree(p)
    try:
        pairs = tree.query_pairs(
            r=tol,
            output_type="ndarray",
        )
    except TypeError:
        pairs = np.asarray(
            list(tree.query_pairs(r=tol)),
            dtype=np.int64,
        )

    if pairs is None or np.asarray(pairs).size == 0:
        return []

    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)

    def find(x):
        x = int(x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, b in pairs:
        union(int(a), int(b))

    groups_dict = {}
    touched = np.unique(pairs.ravel())
    for idx in touched:
        root = find(int(idx))
        groups_dict.setdefault(root, []).append(int(idx))

    groups = [np.asarray(sorted(g), dtype=np.int64) for g in groups_dict.values() if len(g) > 1]
    groups.sort(key=lambda g: int(g[0]))
    return groups


def enforce_coincident_group_motion(
    candidate_vertices,
    original_vertices,
    groups,
    *,
    vertex_strength=None,
):
    """Impone el MISMO desplazamiento a cada grupo coincidente.

    Se conserva la pequeña diferencia geométrica original entre miembros:
        P_final_i = P_original_i + delta_grupo

    Si cualquier miembro está completamente protegido (p.ej. borde abierto),
    se congela todo el grupo para no mover indirectamente dicho borde.
    """
    if not groups:
        return np.asarray(candidate_vertices, dtype=np.float64).copy()

    candidate = np.asarray(candidate_vertices, dtype=np.float64).copy()
    original = np.asarray(original_vertices, dtype=np.float64)

    for group in groups:
        g = np.asarray(group, dtype=np.int64)

        if vertex_strength is not None and np.any(
            np.asarray(vertex_strength, dtype=np.float64)[g] <= 1e-12
        ):
            shared_delta = np.zeros(3, dtype=np.float64)
        else:
            delta = candidate[g] - original[g]
            shared_delta = np.mean(delta, axis=0)

        candidate[g] = original[g] + shared_delta[None, :]

    return candidate


def coincident_group_separation_stats(
    vertices,
    groups,
):
    p = np.asarray(vertices, dtype=np.float64)
    spans = []

    for group in groups:
        q = p[np.asarray(group, dtype=np.int64)]
        if len(q) < 2:
            continue
        # Máxima distancia al primer miembro: suficiente porque el grupo
        # se forma dentro de una tolerancia extremadamente pequeña.
        d = np.linalg.norm(q - q[0][None, :], axis=1)
        spans.append(float(np.max(d)))

    return stats(spans)


def build_face_pair_graph(triangles, structure):
    """Grafo de caras vecinas por arista interior."""
    t = np.asarray(triangles, dtype=np.int64)

    edge_to_faces = {}
    for fi, (a, b, c) in enumerate(t):
        for i, j in ((a, b), (b, c), (c, a)):
            e = (int(i), int(j))
            if e[0] > e[1]:
                e = (e[1], e[0])
            edge_to_faces.setdefault(e, []).append(int(fi))

    structure_weight = {
        tuple(map(int, e)): float(w)
        for e, w in zip(
            np.asarray(structure["edges"], dtype=np.int64),
            np.asarray(structure["edge_weight"], dtype=np.float64),
        )
    }

    fa, fb, ew = [], [], []
    protected_pairs = structure.get("terminal_seam_edges", set())
    for edge, faces in edge_to_faces.items():
        if edge in protected_pairs:
            continue
        if len(faces) < 2:
            continue
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                fa.append(faces[i])
                fb.append(faces[j])
                ew.append(float(structure_weight.get(edge, 1.0)))

    return (
        np.asarray(fa, dtype=np.int64),
        np.asarray(fb, dtype=np.int64),
        np.asarray(ew, dtype=np.float64),
    )


def pair_angles_deg(normals, face_a, face_b):
    n = np.asarray(normals, dtype=np.float64)
    if len(face_a) == 0:
        return np.empty(0, dtype=np.float64)
    dot = np.sum(n[face_a] * n[face_b], axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def bilateral_face_normal_filter(
    normals,
    face_areas,
    face_a,
    face_b,
    topology_weight,
    *,
    iterations,
    angle_sigma_deg,
    max_cross_angle_deg,
    self_weight,
    edge_weight_floor,
):
    """Media robusta bilateral del campo de normales."""
    n = np.asarray(normals, dtype=np.float64).copy()
    n, _ = safe_norm_rows(n)

    area = np.asarray(face_areas, dtype=np.float64)
    good_area = area[np.isfinite(area) & (area > 1e-12)]
    area_med = float(np.median(good_area)) if len(good_area) else 1.0
    area_norm = np.clip(
        area / max(area_med, 1e-12),
        0.35,
        2.5,
    )

    base_topology = np.maximum(
        np.asarray(topology_weight, dtype=np.float64),
        float(edge_weight_floor),
    )

    sigma = max(float(angle_sigma_deg), 1e-6)
    max_cross = max(float(max_cross_angle_deg), sigma)

    for _ in range(max(0, int(iterations))):
        accum = float(self_weight) * area_norm[:, None] * n
        wsum = float(self_weight) * area_norm

        if len(face_a):
            angle = pair_angles_deg(n, face_a, face_b)
            wang = np.exp(-0.5 * (angle / sigma) ** 2)
            wang[angle >= max_cross] = 0.0

            pair_area = np.sqrt(area_norm[face_a] * area_norm[face_b])
            w = base_topology * wang * pair_area

            np.add.at(
                accum,
                face_a,
                w[:, None] * n[face_b],
            )
            np.add.at(
                accum,
                face_b,
                w[:, None] * n[face_a],
            )
            np.add.at(wsum, face_a, w)
            np.add.at(wsum, face_b, w)

        n = np.divide(
            accum,
            wsum[:, None],
            out=n.copy(),
            where=wsum[:, None] > 1e-12,
        )
        n, _ = safe_norm_rows(n)

    return n


def filtered_vertex_normals(
    vertices,
    triangles,
    structure,
    *,
    args,
):
    """Filtra normales por cara y las proyecta a vértices."""
    p = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)

    fn, area = face_geometry(p, t)
    fa, fb, ew = build_face_pair_graph(t, structure)

    filtered = bilateral_face_normal_filter(
        fn,
        area,
        fa,
        fb,
        ew,
        iterations=int(args.normal_filter_iterations),
        angle_sigma_deg=float(args.normal_filter_angle_sigma_deg),
        max_cross_angle_deg=float(args.normal_filter_max_cross_angle_deg),
        self_weight=float(args.normal_filter_self_weight),
        edge_weight_floor=float(args.normal_filter_edge_weight_floor),
    )

    accum = np.zeros_like(p)
    wsum = np.zeros(len(p), dtype=np.float64)
    face_weight = np.maximum(area, 1e-12)

    for corner in range(3):
        vid = t[:, corner]
        np.add.at(
            accum,
            vid,
            face_weight[:, None] * filtered,
        )
        np.add.at(
            wsum,
            vid,
            face_weight,
        )

    vn = np.divide(
        accum,
        wsum[:, None],
        out=np.zeros_like(accum),
        where=wsum[:, None] > 1e-12,
    )
    vn, _ = safe_norm_rows(vn)

    bad = np.linalg.norm(vn, axis=1) <= 1e-12
    if np.any(bad):
        fallback = vertex_normals(p, t)
        vn[bad] = fallback[bad]

    return {
        "face_normals_before": fn,
        "face_normals_filtered": filtered,
        "vertex_normals_filtered": vn,
        "face_pair_a": fa,
        "face_pair_b": fb,
        "face_pair_weight": ew,
        "pair_angles_before_deg": pair_angles_deg(fn, fa, fb),
        "pair_angles_filtered_deg": pair_angles_deg(filtered, fa, fb),
    }


def tangent_basis_batch(normals):
    n = np.asarray(normals, dtype=np.float64).copy()
    n, _ = safe_norm_rows(n)

    helper = np.tile(
        np.asarray([0.0, 0.0, 1.0]),
        (len(n), 1),
    )
    mask = np.abs(n[:, 2]) >= 0.85
    helper[mask] = np.asarray([0.0, 1.0, 0.0])

    t1 = np.cross(helper, n)
    bad = np.linalg.norm(t1, axis=1) <= 1e-12
    helper[bad] = np.asarray([1.0, 0.0, 0.0])
    t1 = np.cross(helper, n)
    t1, _ = safe_norm_rows(t1)

    t2 = np.cross(n, t1)
    t2, _ = safe_norm_rows(t2)

    return t1, t2, n


def solve_weighted_quadratic_irls(
    u,
    v,
    w,
    distance,
    *,
    radius_mm,
    irls_iterations,
    huber_k,
    spacing_mm,
):
    """Ajuste cuadrático robusto rápido mediante ecuaciones normales 6x6."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)

    A = np.column_stack(
        [
            u * u,
            v * v,
            u * v,
            u,
            v,
            np.ones_like(u),
        ]
    )

    spatial = np.exp(-0.5 * (distance / max(0.50 * float(radius_mm), 1e-9)) ** 2)
    weights = spatial.copy()

    coef = None
    residual_scale = None

    for _ in range(max(1, int(irls_iterations))):
        M = A.T @ (weights[:, None] * A)
        rhs = A.T @ (weights * w)

        # Regularización numérica mínima; no impone planaridad.
        ridge = 1e-9 * max(float(np.trace(M)), 1.0)
        M = M + ridge * np.eye(6)

        try:
            coef = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return None, None

        residual = w - A @ coef
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        residual_scale = max(
            1.4826 * mad,
            0.02 * float(spacing_mm),
            1e-8,
        )

        z = np.abs(residual - med) / max(
            float(huber_k) * residual_scale,
            1e-12,
        )
        robust = np.ones_like(z)
        high = z > 1.0
        robust[high] = 1.0 / z[high]

        weights = spatial * robust

    return coef, residual_scale


def multiscale_normal_trend_pass(
    vertices,
    original_vertices,
    triangles,
    structure,
    coincident_groups,
    *,
    spacing_mm,
    args,
):
    """Proyección hacia una tendencia local de escala grande.

    La corrección se activa por la discrepancia entre dos escalas, no por
    asumir que la superficie sea plana.
    """
    p = np.asarray(vertices, dtype=np.float64)
    p0 = np.asarray(original_vertices, dtype=np.float64)

    normal_info = filtered_vertex_normals(
        p,
        triangles,
        structure,
        args=args,
    )
    vn = normal_info["vertex_normals_filtered"]
    t1, t2, vn = tangent_basis_batch(vn)

    small_radius = float(args.normal_trend_small_radius_spacing) * float(spacing_mm)
    large_radius = float(args.normal_trend_large_radius_spacing) * float(spacing_mm)
    if large_radius <= small_radius:
        large_radius = 1.5 * small_radius

    tree = cKDTree(p)
    k = int(
        np.clip(
            int(args.normal_trend_max_neighbors),
            8,
            len(p),
        )
    )
    chunk_size = max(
        64,
        int(args.normal_trend_query_chunk),
    )

    normal_cos = math.cos(math.radians(float(args.normal_trend_neighbor_angle_deg)))

    small_prediction = np.full(
        len(p),
        np.nan,
        dtype=np.float64,
    )
    large_prediction = np.full(
        len(p),
        np.nan,
        dtype=np.float64,
    )
    small_scale = np.full(
        len(p),
        np.nan,
        dtype=np.float64,
    )
    large_scale = np.full(
        len(p),
        np.nan,
        dtype=np.float64,
    )
    small_count = np.zeros(len(p), dtype=np.int32)
    large_count = np.zeros(len(p), dtype=np.int32)

    # Consulta por bloques: k=500 es deliberadamente amplio para que la
    # escala grande pueda "ver" alrededor de una saliente de varios mm,
    # pero no queremos reservar arrays N x 500 para toda la malla.
    progress_marks = {max(1, int(math.ceil(len(p) * f))) for f in (0.25, 0.50, 0.75, 1.0)}
    progress_printed = set()

    ejecutar_bloques(
        _worker_multiscale_normal_trend_pass,
        {
            "args": args,
            "chunk_size": chunk_size,
            "k": k,
            "large_count": large_count,
            "large_prediction": large_prediction,
            "large_radius": large_radius,
            "large_scale": large_scale,
            "normal_cos": normal_cos,
            "p": p,
            "small_count": small_count,
            "small_prediction": small_prediction,
            "small_radius": small_radius,
            "small_scale": small_scale,
            "spacing_mm": spacing_mm,
            "t1": t1,
            "t2": t2,
            "tree": tree,
            "vn": vn,
        },
        {
            "small_prediction": small_prediction,
            "large_prediction": large_prediction,
            "small_scale": small_scale,
            "large_scale": large_scale,
            "small_count": small_count,
            "large_count": large_count,
        },
        len(p),
        chunk_size,
        "Tendencia multiescala",
    )

    scale_difference = np.abs(large_prediction - small_prediction)

    threshold = float(args.normal_trend_scale_difference_threshold_spacing) * float(spacing_mm)
    transition = max(
        float(args.normal_trend_scale_difference_transition_spacing) * float(spacing_mm),
        1e-9,
    )

    gate = np.clip(
        (scale_difference - threshold) / transition,
        0.0,
        1.0,
    )
    # Transición suave para evitar umbrales visuales duros.
    gate = smoothstep01(gate)

    valid_prediction = np.isfinite(small_prediction) & np.isfinite(large_prediction)
    gate[~valid_prediction] = 0.0

    # Bordes y su corona no se corrigen con la tendencia de gran escala.
    gate[structure["boundary_vertex"]] = 0.0
    gate[structure["boundary_ring_vertex"]] = 0.0

    # La protección feature-aware sigue modulando, pero el criterio principal
    # para no cruzar una arista ya fue la similitud de normales del vecindario.
    strength = np.clip(
        structure["vertex_strength"],
        0.0,
        1.0,
    )

    shift_scalar = float(args.normal_trend_strength) * gate * strength * large_prediction
    shift_scalar = np.nan_to_num(
        shift_scalar,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    max_step = float(args.normal_trend_max_shift_per_cycle_spacing) * float(spacing_mm)
    shift_scalar = np.clip(
        shift_scalar,
        -max_step,
        max_step,
    )

    candidate = p + shift_scalar[:, None] * vn

    max_total = float(args.normal_trend_max_total_shift_spacing) * float(spacing_mm)
    total = candidate - p0
    total_norm = np.linalg.norm(total, axis=1)
    total_clip = total_norm > max_total
    if np.any(total_clip):
        candidate[total_clip] = p0[total_clip] + total[total_clip] * (
            max_total / total_norm[total_clip, None]
        )

    candidate = enforce_coincident_group_motion(
        candidate,
        p0,
        coincident_groups or [],
        vertex_strength=structure["vertex_strength"],
    )

    return candidate, {
        "normal_info": normal_info,
        "small_prediction_mm": small_prediction,
        "large_prediction_mm": large_prediction,
        "small_residual_scale_mm": small_scale,
        "large_residual_scale_mm": large_scale,
        "scale_difference_mm": scale_difference,
        "gate": gate,
        "shift_mm": np.linalg.norm(
            candidate - p,
            axis=1,
        ),
        "small_neighbor_count": small_count,
        "large_neighbor_count": large_count,
        "max_step_mm": float(max_step),
        "max_total_mm": float(max_total),
        "total_clipped": total_clip,
    }


def multiscale_normal_trend_regularization(
    vertices,
    original_vertices,
    triangles,
    structure,
    coincident_groups,
    *,
    spacing_mm,
    args,
):
    p0 = np.asarray(original_vertices, dtype=np.float64)
    p = np.asarray(vertices, dtype=np.float64).copy()

    cycles = []
    last = None

    for cycle in range(max(0, int(args.normal_trend_cycles))):
        next_p, diag = multiscale_normal_trend_pass(
            p,
            p0,
            triangles,
            structure,
            coincident_groups,
            spacing_mm=spacing_mm,
            args=args,
        )

        cycles.append(
            {
                "cycle": int(cycle + 1),
                "shift_mm": stats(diag["shift_mm"]),
                "scale_difference_mm": stats(diag["scale_difference_mm"]),
                "gate": stats(diag["gate"]),
                "corrected_vertices": int(np.count_nonzero(diag["gate"] > 0)),
                "strongly_corrected_vertices": int(np.count_nonzero(diag["gate"] >= 0.5)),
                "small_neighbor_count": stats(diag["small_neighbor_count"]),
                "large_neighbor_count": stats(diag["large_neighbor_count"]),
            }
        )

        p = next_p
        last = diag

    if last is None:
        normal_info = filtered_vertex_normals(
            p,
            triangles,
            structure,
            args=args,
        )
        last = {
            "normal_info": normal_info,
            "small_prediction_mm": np.full(len(p), np.nan),
            "large_prediction_mm": np.full(len(p), np.nan),
            "small_residual_scale_mm": np.full(len(p), np.nan),
            "large_residual_scale_mm": np.full(len(p), np.nan),
            "scale_difference_mm": np.full(len(p), np.nan),
            "gate": np.zeros(len(p)),
            "shift_mm": np.zeros(len(p)),
            "small_neighbor_count": np.zeros(len(p), dtype=np.int32),
            "large_neighbor_count": np.zeros(len(p), dtype=np.int32),
            "max_step_mm": 0.0,
            "max_total_mm": 0.0,
            "total_clipped": np.zeros(len(p), dtype=bool),
        }

    return p, {
        "cycles": cycles,
        "last": last,
        "total_shift_mm": np.linalg.norm(
            p - p0,
            axis=1,
        ),
    }


def build_vertex_adjacency(n_vertices, edges):
    """Construye los vecinos de cada vértice a partir de los triángulos."""
    adjacency = [set() for _ in range(int(n_vertices))]
    for a, b in np.asarray(edges, dtype=np.int64):
        a = int(a)
        b = int(b)
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency


def ring_neighbors(adjacency, center, rings):
    """Vecindario topológico hasta N anillos, sin incluir el centro."""
    center = int(center)
    visited = {center}
    frontier = {center}
    result = set()

    for _ in range(max(1, int(rings))):
        nxt = set()
        for u in frontier:
            nxt.update(adjacency[u])
        nxt.difference_update(visited)
        if not nxt:
            break
        result.update(nxt)
        visited.update(nxt)
        frontier = nxt

    return np.asarray(sorted(result), dtype=np.int64)


def tangent_basis_from_normal(normal):
    """Construye una base tangente ortogonal a la normal indicada."""
    n = np.asarray(normal, dtype=np.float64)
    nn = np.linalg.norm(n)
    if not np.isfinite(nn) or nn <= 1e-15:
        n = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n = n / nn

    # Escoger eje auxiliar evitando casi paralelismo.
    if abs(n[2]) < 0.85:
        helper = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

    t1 = np.cross(helper, n)
    t1n = np.linalg.norm(t1)
    if t1n <= 1e-15:
        helper = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        t1 = np.cross(helper, n)
        t1n = np.linalg.norm(t1)

    t1 = t1 / max(t1n, 1e-15)
    t2 = np.cross(n, t1)
    t2 = t2 / max(np.linalg.norm(t2), 1e-15)

    return t1, t2, n


def robust_scale_mad(values, floor=1e-9):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float(floor), 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = max(1.4826 * mad, float(floor))
    return scale, med


def fit_quadratic_surface_irls(
    u,
    v,
    w,
    *,
    irls_iterations=3,
    huber_k=1.5,
):
    """Ajuste robusto local w=f(u,v) con términos cuadráticos."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    A = np.column_stack(
        [
            u * u,
            v * v,
            u * v,
            u,
            v,
            np.ones_like(u),
        ]
    )

    if len(w) < 6:
        raise ValueError("Muy pocos puntos para ajuste cuadrático.")

    weights = np.ones(len(w), dtype=np.float64)
    coef = None

    for _ in range(max(1, int(irls_iterations))):
        sw = np.sqrt(np.clip(weights, 1e-8, None))
        Aw = A * sw[:, None]
        bw = w * sw

        coef, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        residual = w - A @ coef

        scale, med = robust_scale_mad(residual, floor=1e-8)
        centered = residual - med
        t = np.abs(centered) / max(float(huber_k) * scale, 1e-12)

        weights = np.ones_like(t)
        mask = t > 1.0
        weights[mask] = 1.0 / t[mask]

    return coef


def quadratic_predict(coef, u, v):
    """Evalúa la superficie cuadrática local en coordenadas tangentes."""
    a, b, c, d, e, f = np.asarray(coef, dtype=np.float64)
    return a * u * u + b * v * v + c * u * v + d * u + e * v + f


def compute_local_quadratic_residuals(
    vertices,
    triangles,
    structure,
    *,
    rings,
    min_neighbors,
    irls_iterations,
    huber_k,
):
    """Residuo normal robusto de cada vértice respecto a su superficie local."""
    p = np.asarray(vertices, dtype=np.float64)
    normals = vertex_normals(p, triangles)
    adjacency = build_vertex_adjacency(
        len(p),
        structure.get("fit_edges", structure["edges"]),
    )

    residuals = np.full(len(p), np.nan, dtype=np.float64)
    local_scale = np.full(len(p), np.nan, dtype=np.float64)
    neighbor_counts = np.zeros(len(p), dtype=np.int32)

    # Para análisis de aislamiento reutilizamos los vecinos.
    neighborhoods = [None] * len(p)

    ejecutar_bloques(
        _worker_compute_local_quadratic_residuals,
        {
            "adjacency": adjacency,
            "huber_k": huber_k,
            "irls_iterations": irls_iterations,
            "local_scale": local_scale,
            "min_neighbors": min_neighbors,
            "neighbor_counts": neighbor_counts,
            "neighborhoods": neighborhoods,
            "normals": normals,
            "p": p,
            "residuals": residuals,
            "rings": rings,
        },
        {
            "residuals": residuals,
            "local_scale": local_scale,
            "neighbor_counts": neighbor_counts,
            "neighborhoods": neighborhoods,
        },
        len(p),
        512,
        "Residuos locales",
    )

    return {
        "residual_mm": residuals,
        "local_scale_mm": local_scale,
        "neighbor_count": neighbor_counts,
        "neighborhoods": neighborhoods,
        "normals": normals,
    }


def despike_mesh_vertices(
    vertices,
    original_vertices,
    triangles,
    structure,
    coincident_groups,
    *,
    spacing_mm,
    args,
):
    """Corrige salientes locales robustas antes del Taubin."""
    p = np.asarray(vertices, dtype=np.float64)
    p0 = np.asarray(original_vertices, dtype=np.float64)

    diagnostics = compute_local_quadratic_residuals(
        p,
        triangles,
        structure,
        rings=int(args.despike_rings),
        min_neighbors=int(args.despike_min_neighbors),
        irls_iterations=int(args.despike_irls_iterations),
        huber_k=float(args.despike_huber_k),
    )

    residual = diagnostics["residual_mm"]
    scale = diagnostics["local_scale_mm"]
    normals = diagnostics["normals"]
    neighborhoods = diagnostics["neighborhoods"]

    zscore = np.divide(
        np.abs(residual),
        np.maximum(scale, 1e-12),
        out=np.zeros_like(residual),
        where=np.isfinite(residual) & np.isfinite(scale),
    )

    candidate_outlier = (
        np.isfinite(residual)
        & np.isfinite(scale)
        & (zscore >= float(args.despike_zscore_threshold))
    )

    # Restricciones fuertes.
    eligible = (
        candidate_outlier
        & (structure["vertex_strength"] >= float(args.despike_feature_strength_min))
        & (~structure["boundary_vertex"])
        & (~structure["boundary_ring_vertex"])
    )

    # Aislamiento: una saliente real espuria no debe formar una región extensa
    # con muchos vecinos outlier del mismo signo.
    isolated = np.zeros(len(p), dtype=bool)
    same_sign_ratio = np.ones(len(p), dtype=np.float64)

    for i in np.flatnonzero(eligible):
        neigh = neighborhoods[i]
        if neigh is None or len(neigh) == 0:
            continue

        valid = np.isfinite(residual[neigh])
        neigh = neigh[valid]
        if len(neigh) == 0:
            continue

        sign_i = np.sign(residual[i])
        if sign_i == 0:
            continue

        neigh_out = candidate_outlier[neigh]
        same = neigh_out & (np.sign(residual[neigh]) == sign_i)
        ratio = float(np.count_nonzero(same)) / float(len(neigh))
        same_sign_ratio[i] = ratio

        if ratio <= float(args.despike_isolation_ratio):
            isolated[i] = True

    selected = eligible & isolated

    # Corrección principalmente normal.
    shift_scalar = np.zeros(len(p), dtype=np.float64)
    shift_scalar[selected] = -float(args.despike_correction_strength) * residual[selected]

    max_shift = float(args.despike_max_shift_spacing) * float(spacing_mm)
    shift_scalar = np.clip(
        shift_scalar,
        -max_shift,
        max_shift,
    )

    candidate = p + shift_scalar[:, None] * normals

    # Coincident lock: el despiking tampoco puede separar costuras duplicadas.
    candidate = enforce_coincident_group_motion(
        candidate,
        p0,
        coincident_groups or [],
        vertex_strength=structure["vertex_strength"],
    )

    return candidate, {
        "residual_mm": residual,
        "local_scale_mm": scale,
        "robust_zscore": zscore,
        "candidate_outlier": candidate_outlier,
        "eligible": eligible,
        "isolated": isolated,
        "selected": selected,
        "same_sign_neighbor_ratio": same_sign_ratio,
        "shift_mm": np.linalg.norm(candidate - p, axis=1),
        "selected_count": int(np.count_nonzero(selected)),
        "max_shift_allowed_mm": float(max_shift),
    }


def surface_fairing_pass(
    vertices,
    original_vertices,
    triangles,
    structure,
    coincident_groups,
    neighborhoods,
    *,
    spacing_mm,
    args,
):
    """Una pasada de fairing robusto sin ninguna primitiva geométrica.

    El ajuste usa exclusivamente vecinos conectados por la malla. La
    predicción cuadrática representa la posición de la superficie local en la
    normal del vértice central; por tanto, no aplana sistemáticamente una zona
    curva como sí podría hacerlo una media cartesiana.
    """
    p = np.asarray(vertices, dtype=np.float64)
    p0 = np.asarray(original_vertices, dtype=np.float64)

    normal_info = filtered_vertex_normals(
        p,
        triangles,
        structure,
        args=args,
    )
    normals = normal_info["vertex_normals_filtered"]
    t1, t2, normals = tangent_basis_batch(normals)

    target_mm = np.full(len(p), np.nan, dtype=np.float64)
    local_scale_mm = np.full(len(p), np.nan, dtype=np.float64)
    compatible_count = np.zeros(len(p), dtype=np.int32)

    angle_cos = math.cos(math.radians(float(args.surface_fairing_neighbor_angle_deg)))
    min_neighbors = max(6, int(args.surface_fairing_min_neighbors))

    ejecutar_bloques(
        _worker_surface_fairing_pass,
        {
            "angle_cos": angle_cos,
            "args": args,
            "compatible_count": compatible_count,
            "local_scale_mm": local_scale_mm,
            "min_neighbors": min_neighbors,
            "neighborhoods": neighborhoods,
            "normals": normals,
            "p": p,
            "spacing_mm": spacing_mm,
            "structure": structure,
            "t1": t1,
            "t2": t2,
            "target_mm": target_mm,
        },
        {
            "target_mm": target_mm,
            "local_scale_mm": local_scale_mm,
            "compatible_count": compatible_count,
        },
        len(p),
        512,
        "Pulido adaptativo",
    )

    abs_target = np.abs(target_mm)
    absolute_floor = float(args.surface_fairing_activation_threshold_spacing) * float(spacing_mm)
    threshold = np.maximum(
        absolute_floor,
        float(args.surface_fairing_noise_scale_factor)
        * np.nan_to_num(
            local_scale_mm,
            nan=np.inf,
            posinf=np.inf,
            neginf=np.inf,
        ),
    )
    transition = max(
        float(args.surface_fairing_activation_transition_spacing) * float(spacing_mm),
        1e-9,
    )

    gate = smoothstep01(
        np.clip(
            (abs_target - threshold) / transition,
            0.0,
            1.0,
        )
    )
    valid = (
        np.isfinite(target_mm) & np.isfinite(local_scale_mm) & (compatible_count >= min_neighbors)
    )
    gate[~valid] = 0.0
    gate[structure["boundary_vertex"]] = 0.0
    gate[structure["boundary_ring_vertex"]] = 0.0

    # La movilidad multiescala baja solo en discontinuidades persistentes.
    mobility = np.clip(
        structure["vertex_strength"],
        0.0,
        1.0,
    )
    mobility[mobility < float(args.surface_fairing_feature_strength_min)] = 0.0

    shift_scalar = (
        float(args.surface_fairing_strength) * gate * mobility * np.nan_to_num(target_mm, nan=0.0)
    )

    max_step = float(args.surface_fairing_max_shift_per_cycle_spacing) * float(spacing_mm)
    shift_scalar = np.clip(shift_scalar, -max_step, max_step)

    candidate = p + shift_scalar[:, None] * normals

    max_total = float(args.surface_fairing_max_total_shift_spacing) * float(spacing_mm)
    total = candidate - p0
    total_norm = np.linalg.norm(total, axis=1)
    total_clipped = total_norm > max_total
    if np.any(total_clipped):
        candidate[total_clipped] = p0[total_clipped] + total[total_clipped] * (
            max_total / total_norm[total_clipped, None]
        )

    candidate = enforce_coincident_group_motion(
        candidate,
        p0,
        coincident_groups or [],
        vertex_strength=structure["vertex_strength"],
    )

    return candidate, {
        "normal_info": normal_info,
        "target_mm": target_mm,
        "local_scale_mm": local_scale_mm,
        "compatible_neighbor_count": compatible_count,
        "gate": gate,
        "active": gate > 0.0,
        "shift_mm": np.linalg.norm(candidate - p, axis=1),
        "max_step_mm": float(max_step),
        "max_total_mm": float(max_total),
        "total_clipped": total_clipped,
    }


def adaptive_surface_fairing_regularization(
    vertices,
    original_vertices,
    triangles,
    structure,
    coincident_groups,
    *,
    spacing_mm,
    args,
):
    """Acabado iterativo con vecindarios topológicos cacheados."""
    p = np.asarray(vertices, dtype=np.float64).copy()
    p0 = np.asarray(original_vertices, dtype=np.float64)

    adjacency = build_vertex_adjacency(
        len(p),
        structure.get("fit_edges", structure["edges"]),
    )
    neighborhoods = [
        ring_neighbors(
            adjacency,
            i,
            int(args.surface_fairing_rings),
        )
        for i in range(len(p))
    ]

    records = []
    last = None
    stop_threshold = float(args.surface_fairing_convergence_spacing) * float(spacing_mm)

    for cycle in range(max(0, int(args.surface_fairing_cycles))):
        candidate, diag = surface_fairing_pass(
            p,
            p0,
            triangles,
            structure,
            coincident_groups,
            neighborhoods,
            spacing_mm=spacing_mm,
            args=args,
        )

        active_count = int(np.count_nonzero(diag["active"]))
        # La convergencia se mide solo donde el fairing está activo; incluir
        # los ceros de las aristas protegidas podría detenerlo demasiado pronto.
        shift_stats = stats(diag["shift_mm"][diag["active"]])
        records.append(
            {
                "cycle": int(cycle + 1),
                "active_vertices": active_count,
                "target_mm": stats(diag["target_mm"]),
                "local_scale_mm": stats(diag["local_scale_mm"]),
                "gate": stats(diag["gate"]),
                "shift_mm": shift_stats,
                "total_clipped_vertices": int(np.count_nonzero(diag["total_clipped"])),
            }
        )

        p = candidate
        last = diag

        p90 = shift_stats["p90"]
        print(
            f"[Paso 15 V1.11] fairing residual ciclo {cycle + 1}: "
            f"activos={active_count:,}, P90_shift={float(p90 or 0.0):.6f} mm"
        )

        if active_count == 0 or (p90 is not None and float(p90) <= stop_threshold):
            break

    if last is None:
        normal_info = filtered_vertex_normals(
            p,
            triangles,
            structure,
            args=args,
        )
        last = {
            "normal_info": normal_info,
            "target_mm": np.full(len(p), np.nan),
            "local_scale_mm": np.full(len(p), np.nan),
            "compatible_neighbor_count": np.zeros(len(p), dtype=np.int32),
            "gate": np.zeros(len(p)),
            "active": np.zeros(len(p), dtype=bool),
            "shift_mm": np.zeros(len(p)),
            "max_step_mm": 0.0,
            "max_total_mm": 0.0,
            "total_clipped": np.zeros(len(p), dtype=bool),
        }

    return p, {
        "cycles": records,
        "last": last,
        "total_shift_mm": np.linalg.norm(p - np.asarray(vertices), axis=1),
        "total_shift_from_input_mm": np.linalg.norm(p - p0, axis=1),
        "convergence_threshold_mm": float(stop_threshold),
        "neighborhood_size": stats([len(x) for x in neighborhoods]),
    }


def feature_guide_vertices(
    vertices,
    edges,
    boundary_vertex,
    *,
    iterations,
    lambda_factor,
):
    """Crea una copia multiescala SOLO para detectar features.

    El objetivo es que una ondulación de 1-2 triángulos deje de parecer una
    arista real, mientras una discontinuidad geométrica persistente sigue
    apareciendo a una escala mayor.

    Esta copia nunca se exporta ni reemplaza la geometría científica.
    """
    p = np.asarray(vertices, dtype=np.float64).copy()
    e = np.asarray(edges, dtype=np.int64)
    boundary = np.asarray(boundary_vertex, dtype=bool)

    if len(e) == 0:
        return p

    a = e[:, 0]
    b = e[:, 1]

    for _ in range(max(0, int(iterations))):
        vec = p[b] - p[a]
        accum = np.zeros_like(p)
        count = np.zeros(len(p), dtype=np.float64)

        np.add.at(accum, a, vec)
        np.add.at(accum, b, -vec)
        np.add.at(count, a, 1.0)
        np.add.at(count, b, 1.0)

        lap = np.divide(
            accum,
            count[:, None],
            out=np.zeros_like(accum),
            where=count[:, None] > 0,
        )
        candidate = p + float(lambda_factor) * lap

        # Los bordes reales abiertos no se mueven ni siquiera en la guía.
        candidate[boundary] = p[boundary]
        p = candidate

    return p


def build_edge_structure(
    vertices,
    triangles,
    *,
    soft_angle_deg,
    hard_angle_deg,
    feature_min_strength,
    edge_crossing_min_weight,
    boundary_ring_strength,
    feature_guide_iterations,
    feature_guide_lambda,
):
    """Construye aristas únicas y pesos de protección FIJOS.

    Se calculan sobre la geometría ya reparada por Paso 14 y no se
    actualizan durante el suavizado, para no 'perder' una arista real a medida
    que se mueve la malla.
    """
    v = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)

    edge_faces: Dict[Tuple[int, int], list] = {}
    for fi, (a, b, c) in enumerate(t):
        for i, j in ((a, b), (b, c), (c, a)):
            e = (int(i), int(j))
            if e[0] > e[1]:
                e = (e[1], e[0])
            edge_faces.setdefault(e, []).append(fi)

    edges = np.asarray(list(edge_faces.keys()), dtype=np.int64)
    n_edges = len(edges)
    dihedral = np.zeros(n_edges, dtype=np.float64)
    boundary_edge = np.zeros(n_edges, dtype=bool)

    # Primero se identifica topológicamente el borde abierto.
    n_vertices = len(v)
    boundary_vertex = np.zeros(n_vertices, dtype=bool)
    for ei, edge in enumerate(map(tuple, edges.tolist())):
        faces = edge_faces[edge]
        if len(faces) == 1:
            boundary_edge[ei] = True
            boundary_vertex[edge[0]] = True
            boundary_vertex[edge[1]] = True

    # Detección multiescala: el diedro se mide sobre una copia guía
    # ligeramente suavizada, no sobre la rugosidad cruda que queremos quitar.
    guide_v = feature_guide_vertices(
        v,
        edges,
        boundary_vertex,
        iterations=feature_guide_iterations,
        lambda_factor=feature_guide_lambda,
    )
    fn, _ = face_geometry(guide_v, t)

    for ei, edge in enumerate(map(tuple, edges.tolist())):
        faces = edge_faces[edge]
        if len(faces) == 1:
            dihedral[ei] = 180.0
        elif len(faces) >= 2:
            n1 = fn[faces[0]]
            # Si por anomalía hay >2 caras, usamos la peor separación angular.
            worst = 0.0
            for fj in faces[1:]:
                n2 = fn[fj]
                dot = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
                ang = math.degrees(math.acos(dot))
                worst = max(worst, ang)
            dihedral[ei] = worst

    soft = float(soft_angle_deg)
    hard = max(float(hard_angle_deg), soft + 1e-6)
    transition = smoothstep01((dihedral - soft) / (hard - soft))

    edge_weight = 1.0 - transition * (1.0 - float(edge_crossing_min_weight))
    edge_weight[boundary_edge] = 0.0

    vertex_max_dihedral = np.zeros(n_vertices, dtype=np.float64)

    for ei, (a, b) in enumerate(edges):
        vertex_max_dihedral[a] = max(
            vertex_max_dihedral[a],
            dihedral[ei],
        )
        vertex_max_dihedral[b] = max(
            vertex_max_dihedral[b],
            dihedral[ei],
        )

    vt = smoothstep01((vertex_max_dihedral - soft) / (hard - soft))
    vertex_strength = 1.0 - vt * (1.0 - float(feature_min_strength))

    # Borde abierto: inmóvil para conservar exactamente el contorno del hueco.
    vertex_strength[boundary_vertex] = 0.0

    # Corona de un salto alrededor del borde, protegida pero no congelada.
    boundary_ring = np.zeros(n_vertices, dtype=bool)
    for a, b in edges:
        if boundary_vertex[a] and not boundary_vertex[b]:
            boundary_ring[b] = True
        if boundary_vertex[b] and not boundary_vertex[a]:
            boundary_ring[a] = True

    ring = boundary_ring & (~boundary_vertex)
    vertex_strength[ring] = np.minimum(
        vertex_strength[ring],
        float(boundary_ring_strength),
    )

    return {
        "edges": edges,
        "edge_weight": edge_weight,
        "dihedral_deg": dihedral,
        "boundary_edge": boundary_edge,
        "boundary_vertex": boundary_vertex,
        "boundary_ring_vertex": ring,
        "vertex_max_dihedral_deg": vertex_max_dihedral,
        "vertex_strength": np.clip(vertex_strength, 0.0, 1.0),
        "feature_guide_vertices": guide_v,
    }


def laplacian_vector(vertices, edges, edge_weight):
    p = np.asarray(vertices, dtype=np.float64)
    e = np.asarray(edges, dtype=np.int64)
    ew = np.asarray(edge_weight, dtype=np.float64)

    a = e[:, 0]
    b = e[:, 1]
    vec_ab = p[b] - p[a]
    lengths = np.linalg.norm(vec_ab, axis=1)

    # Peso inverso de longitud moderado: evita que una arista larga domine.
    w = np.divide(
        ew,
        np.maximum(lengths, 1e-9),
        out=np.zeros_like(ew),
        where=np.isfinite(lengths),
    )

    accum = np.zeros_like(p)
    wsum = np.zeros(len(p), dtype=np.float64)

    np.add.at(accum, a, w[:, None] * vec_ab)
    np.add.at(accum, b, -w[:, None] * vec_ab)
    np.add.at(wsum, a, w)
    np.add.at(wsum, b, w)

    return np.divide(
        accum,
        wsum[:, None],
        out=np.zeros_like(accum),
        where=wsum[:, None] > 1e-12,
    )


def limited_pass(
    vertices,
    original_vertices,
    triangles,
    structure,
    factor,
    *,
    coincident_groups=None,
    normal_component,
    tangential_component,
    max_step_mm,
    max_total_mm,
):
    p = np.asarray(vertices, dtype=np.float64)
    p0 = np.asarray(original_vertices, dtype=np.float64)

    lap = laplacian_vector(
        p,
        structure["edges"],
        structure["edge_weight"],
    )
    normals = vertex_normals(p, triangles)

    normal_scalar = np.sum(lap * normals, axis=1)
    normal_part = normal_scalar[:, None] * normals
    tangent_part = lap - normal_part

    direction = float(normal_component) * normal_part + float(tangential_component) * tangent_part

    displacement = float(factor) * structure["vertex_strength"][:, None] * direction

    step_norm = np.linalg.norm(displacement, axis=1)
    clip = step_norm > float(max_step_mm)
    if np.any(clip):
        displacement[clip] *= float(max_step_mm) / step_norm[clip, None]

    candidate = p + displacement

    total = candidate - p0
    total_norm = np.linalg.norm(total, axis=1)
    total_clip = total_norm > float(max_total_mm)
    if np.any(total_clip):
        candidate[total_clip] = p0[total_clip] + total[total_clip] * (
            float(max_total_mm) / total_norm[total_clip, None]
        )

    # V1.2: los vértices coincidentes no pueden separarse por el suavizado.
    candidate = enforce_coincident_group_motion(
        candidate,
        p0,
        coincident_groups or [],
        vertex_strength=structure["vertex_strength"],
    )

    candidate, alpha, trace, safe = vertexwise_orientation_guard(
        p, candidate, triangles, coincident_groups=coincident_groups, max_iterations=24
    )
    if not safe:
        candidate = p.copy()
    structure.setdefault("incremental_orientation", []).append(
        {
            "operation": "limited_pass",
            "accepted": bool(safe),
            "locally_restricted_vertices": int(np.count_nonzero(alpha < 1)),
            "iterations": len(trace),
        }
    )
    return candidate, {
        "step_clipped": clip,
        "total_clipped": total_clip,
        "step_mm": np.linalg.norm(candidate - p, axis=1),
    }


def robust_extent(vertices):
    p = np.asarray(vertices, dtype=np.float64)
    lo = np.percentile(p, 1.0, axis=0)
    hi = np.percentile(p, 99.0, axis=0)
    return {
        "p01": lo,
        "p99": hi,
        "extent": hi - lo,
    }


def safety_metrics(original_vertices, candidate_vertices, triangles):
    p0 = np.asarray(original_vertices, dtype=np.float64)
    p = np.asarray(candidate_vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)

    n0, a0 = face_geometry(p0, t)
    n1, a1 = face_geometry(p, t)
    dot = np.sum(n0 * n1, axis=1)

    positive = a0[np.isfinite(a0) & (a0 > 1e-15)]
    floor = max(1e-12, 1e-8 * (float(np.median(positive)) if len(positive) else 1.0))
    valid_original = np.isfinite(a0) & (a0 > floor)
    flip = valid_original & ((dot <= 0.0) | ~np.isfinite(dot))
    degenerate = valid_original & ((a1 <= np.maximum(floor, 0.05 * a0)) | ~np.isfinite(a1))
    t = np.asarray(triangles, dtype=np.int64)
    changed = np.any(np.any(p[t] != p0[t], axis=2), axis=1)
    degenerate |= (~valid_original) & changed

    e0 = robust_extent(p0)
    e1 = robust_extent(p)
    rel = np.divide(
        e1["extent"] - e0["extent"],
        e0["extent"],
        out=np.zeros(3, dtype=np.float64),
        where=e0["extent"] > 1e-9,
    )

    return {
        "worsened_fold_face_count": int(len(worsened_fold_faces(t, n0, n1))),
        "preexisting_degenerate_faces": int(np.count_nonzero(~valid_original)),
        "orientation_flip_ratio": float(np.mean(flip)) if len(flip) else 0.0,
        "orientation_flip_count": int(np.count_nonzero(flip)),
        "degenerate_face_ratio": float(np.mean(degenerate)) if len(degenerate) else 0.0,
        "degenerate_face_count": int(np.count_nonzero(degenerate)),
        "robust_extent_relative_change_xyz": rel.astype(float).tolist(),
        "maximum_absolute_robust_extent_change": float(np.max(np.abs(rel))),
    }


_QUALITY_TOPOLOGY_CACHE = None


def quality_face_pairs(triangles):
    """Pares de caras interiores; caché acotada a una conectividad inmutable."""
    global _QUALITY_TOPOLOGY_CACHE
    t = np.asarray(triangles, dtype=np.int64)
    if _QUALITY_TOPOLOGY_CACHE is not None and _QUALITY_TOPOLOGY_CACHE[0] is t:
        return _QUALITY_TOPOLOGY_CACHE[1]
    edges = np.sort(np.concatenate((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]])), axis=1)
    _, inv, count = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(count)[:-1]]
    interior = np.flatnonzero(count == 2)
    face_ids = np.tile(np.arange(len(t)), 3)
    pairs = np.column_stack(
        (face_ids[order[starts[interior]]], face_ids[order[starts[interior] + 1]])
    )
    _QUALITY_TOPOLOGY_CACHE = (t, pairs)
    return pairs


def worsened_fold_faces(triangles, original_normals, candidate_normals):
    """No crear/aumentar pliegues >28 grados en más de 5 grados por referencia.

    No obliga a aplanar aristas reales: conservarlas satisface la guarda.
    Se utiliza tanto entre etapas como frente a la entrada original.
    """
    pairs = quality_face_pairs(triangles)
    if not len(pairs):
        return np.empty(0, dtype=np.int64)
    before = np.degrees(
        np.arccos(
            np.clip(
                np.sum(original_normals[pairs[:, 0]] * original_normals[pairs[:, 1]], axis=1),
                -1.0,
                1.0,
            )
        )
    )
    after = np.degrees(
        np.arccos(
            np.clip(
                np.sum(candidate_normals[pairs[:, 0]] * candidate_normals[pairs[:, 1]], axis=1),
                -1.0,
                1.0,
            )
        )
    )
    bad = after > np.maximum(28.0, before + 5.0) + 1e-7
    return np.unique(pairs[bad])


def unsafe_orientation_face_indices(
    original_vertices,
    candidate_vertices,
    triangles,
):
    p0 = np.asarray(original_vertices, dtype=np.float64)
    p = np.asarray(candidate_vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)

    n0, a0 = face_geometry(p0, t)
    n1, a1 = face_geometry(p, t)

    dot = np.sum(n0 * n1, axis=1)
    positive = a0[np.isfinite(a0) & (a0 > 1e-15)]
    area_ref = float(np.median(positive)) if len(positive) else 1.0
    floor = max(1e-12, 1e-8 * area_ref)
    preexisting = (~np.isfinite(a0)) | (a0 <= floor)
    # Una cara casi degenerada de entrada no puede definir orientación fiable.
    # Se permite exclusivamente conservar sus tres posiciones EXACTAS.
    changed = np.any(np.any(p[t] != p0[t], axis=2), axis=1)
    bad_existing = preexisting & changed
    valid = ~preexisting
    flip = valid & ((dot <= 0.0) | ~np.isfinite(dot))
    degenerate = valid & ((a1 <= np.maximum(floor, 0.05 * a0)) | ~np.isfinite(a1))
    unsafe = flip | degenerate | bad_existing
    unsafe[worsened_fold_faces(t, n0, n1)] = True
    return np.flatnonzero(unsafe)


def rollback_weight_from_faces(
    n_vertices,
    triangles,
    face_indices,
    *,
    rings,
    ring_floor,
    coincident_groups=None,
):
    t = np.asarray(triangles, dtype=np.int64)
    weight = np.zeros(int(n_vertices), dtype=np.float64)

    faces = np.asarray(face_indices, dtype=np.int64)
    faces = faces[(faces >= 0) & (faces < len(t))]
    if len(faces) == 0:
        return weight

    seeds = set(
        map(
            int,
            np.unique(t[faces].ravel()),
        )
    )
    for i in seeds:
        weight[i] = 1.0

    adjacency = [set() for _ in range(int(n_vertices))]
    for a, b, c in t:
        a = int(a)
        b = int(b)
        c = int(c)
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))

    visited = set(seeds)
    frontier = set(seeds)
    rings = max(0, int(rings))
    floor = float(np.clip(ring_floor, 0.0, 1.0))

    for ring in range(1, rings + 1):
        nxt = set()
        for u in frontier:
            nxt.update(adjacency[u])
        nxt.difference_update(visited)
        if not nxt:
            break

        frac = float(ring) / float(rings + 1)
        rw = 1.0 - frac * (1.0 - floor)

        ids = np.fromiter(sorted(nxt), dtype=np.int64)
        weight[ids] = np.maximum(weight[ids], rw)

        visited.update(nxt)
        frontier = nxt

    if coincident_groups:
        for group in coincident_groups:
            g = np.asarray(group, dtype=np.int64)
            if len(g) == 0:
                continue
            gw = float(np.max(weight[g]))
            if gw > 0:
                weight[g] = gw

    return weight


def local_mix_from_original(
    original_vertices,
    candidate_vertices,
    rollback_weight,
    beta,
):
    p0 = np.asarray(original_vertices, dtype=np.float64)
    ps = np.asarray(candidate_vertices, dtype=np.float64)
    w = np.asarray(rollback_weight, dtype=np.float64)

    beta = float(np.clip(beta, 0.0, 1.0))
    alpha = 1.0 - w * (1.0 - beta)
    alpha = np.clip(alpha, 0.0, 1.0)

    return (
        p0 + alpha[:, None] * (ps - p0),
        alpha,
    )


def vertexwise_orientation_guard(
    original_vertices,
    candidate_vertices,
    triangles,
    *,
    coincident_groups=None,
    max_iterations=28,
):
    """Reduce solo el desplazamiento de los vertices que crean caras inseguras.

    La guarda anterior propagaba el rollback por anillos completos. Cuando los
    pocos triángulos problemáticos estaban distribuidos por toda la superficie,
    esos anillos acababan cubriendo casi toda la malla y anulaban el pulido.

    Aquí cada cara invertida o degenerada reduce a la mitad únicamente el paso
    de sus tres vértices. Las caras vecinas se verifican de nuevo en la siguiente
    iteración, por lo que la transición sigue siendo geométricamente segura sin
    convertir un problema local en un rollback global.
    """
    p0 = np.asarray(original_vertices, dtype=np.float64)
    ps = np.asarray(candidate_vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)
    displacement = ps - p0
    alpha = np.ones(len(p0), dtype=np.float64)
    records = []

    groups = [
        np.asarray(group, dtype=np.int64) for group in (coincident_groups or []) if len(group) > 1
    ]

    def synchronize_groups(values):
        # Todos los vértices coincidentes deben conservar exactamente el mismo
        # factor para no abrir costuras que el paso 14 cerró topológicamente.
        for group in groups:
            a = float(np.min(values[group]))
            values[group] = a

    # Fijar únicamente caras preexistentes sin orientación fiable.
    _, original_area = face_geometry(p0, t)
    positive = original_area[np.isfinite(original_area) & (original_area > 1e-15)]
    area_floor = max(1e-12, 1e-8 * (float(np.median(positive)) if len(positive) else 1.0))
    original_bad = (~np.isfinite(original_area)) | (original_area <= area_floor)
    if np.any(original_bad):
        alpha[np.unique(t[original_bad])] = 0.0
        synchronize_groups(alpha)
    candidate = p0 + alpha[:, None] * displacement
    for iteration in range(max(1, int(max_iterations))):
        unsafe = unsafe_orientation_face_indices(p0, candidate, t)
        records.append(
            {
                "iteration": int(iteration + 1),
                "unsafe_faces": int(len(unsafe)),
                "includes_local_fold_quality": True,
                "affected_vertices": int(np.count_nonzero(alpha < 1.0)),
                "minimum_alpha": float(np.min(alpha)) if len(alpha) else 1.0,
            }
        )
        if len(unsafe) == 0:
            return candidate, alpha, records, True

        affected = np.unique(t[np.asarray(unsafe, dtype=np.int64)].ravel())
        previous = alpha[affected].copy()
        alpha[affected] *= 0.5
        alpha[alpha < (2.0**-20)] = 0.0
        synchronize_groups(alpha)
        candidate = p0 + alpha[:, None] * displacement

        if np.array_equal(previous, alpha[affected]):
            break

    # Última salida conservadora: solo los vértices de las caras que aún no son
    # seguras regresan a la posición original. Se repite porque esa corrección
    # puede trasladar el conflicto a una cara adyacente.
    for iteration in range(64):
        unsafe = unsafe_orientation_face_indices(p0, candidate, t)
        if len(unsafe) == 0:
            return candidate, alpha, records, True
        affected = np.unique(t[np.asarray(unsafe, dtype=np.int64)].ravel())
        alpha[affected] = 0.0
        synchronize_groups(alpha)
        candidate = p0 + alpha[:, None] * displacement
        records.append(
            {
                "iteration": int(len(records) + 1),
                "unsafe_faces": int(len(unsafe)),
                "includes_local_fold_quality": True,
                "affected_vertices": int(np.count_nonzero(alpha < 1.0)),
                "minimum_alpha": float(np.min(alpha)) if len(alpha) else 1.0,
                "hard_local_rollback": True,
            }
        )

    safe = len(unsafe_orientation_face_indices(p0, candidate, t)) == 0
    return candidate, alpha, records, bool(safe)


def apply_global_safety_guard(
    original_vertices,
    smoothed_vertices,
    triangles,
    *,
    max_extent_change_ratio,
    max_orientation_flip_ratio,
    bisection_steps,
    coincident_groups=None,
    local_rollback_rings=2,
    local_max_expansions=4,
    local_ring_floor=0.25,
):
    """V1.8: extent global, flips/degenerados con rollback LOCAL."""
    p0 = np.asarray(original_vertices, dtype=np.float64)
    ps_full = np.asarray(smoothed_vertices, dtype=np.float64)

    # ----------------------------------------------------------
    # 1) Solo la dimensión robusta puede reducir globalmente.
    # ----------------------------------------------------------
    full_metrics = safety_metrics(
        p0,
        ps_full,
        triangles,
    )

    extent_alpha = 1.0
    ps = ps_full.copy()

    if full_metrics["maximum_absolute_robust_extent_change"] > float(max_extent_change_ratio):
        low = 0.0
        high = 1.0
        best = p0.copy()
        best_alpha = 0.0

        for _ in range(max(1, int(bisection_steps))):
            alpha = 0.5 * (low + high)
            candidate = p0 + alpha * (ps_full - p0)
            metrics = safety_metrics(p0, candidate, triangles)

            if metrics["maximum_absolute_robust_extent_change"] <= float(max_extent_change_ratio):
                best = candidate
                best_alpha = alpha
                low = alpha
            else:
                high = alpha

        ps = best
        extent_alpha = float(best_alpha)

    unsafe = unsafe_orientation_face_indices(
        p0,
        ps,
        triangles,
    )

    if len(unsafe) == 0:
        final_metrics = safety_metrics(p0, ps, triangles)
        return ps, {
            "activated": bool(extent_alpha < 1.0),
            "mode": ("extent_global_only" if extent_alpha < 1.0 else "none"),
            "extent_alpha": float(extent_alpha),
            "local_beta": 1.0,
            "affected_vertices": 0,
            "full_strength_vertices": int(len(p0)),
            "full_candidate_metrics": full_metrics,
            "final_metrics": final_metrics,
            "fallback_global_used": False,
        }

    # ----------------------------------------------------------
    # 2) Rollback adaptativo por vértice para flips/degenerados.
    # ----------------------------------------------------------
    vertexwise, vertex_alpha, vertex_records, vertexwise_safe = vertexwise_orientation_guard(
        p0,
        ps,
        triangles,
        coincident_groups=coincident_groups,
        max_iterations=max(20, 3 * int(bisection_steps)),
    )

    if vertexwise_safe:
        vertex_metrics = safety_metrics(p0, vertexwise, triangles)
        if (
            vertex_metrics["maximum_absolute_robust_extent_change"]
            <= float(max_extent_change_ratio)
            and vertex_metrics["orientation_flip_ratio"] <= float(max_orientation_flip_ratio)
            and vertex_metrics["degenerate_face_ratio"] <= 1e-4
        ):
            return vertexwise, {
                "activated": True,
                "mode": "vertexwise_orientation_rollback",
                "extent_alpha": float(extent_alpha),
                "local_beta": None,
                "rings_used": 0,
                "affected_vertices": int(np.count_nonzero(vertex_alpha < (1.0 - 1e-12))),
                "full_strength_vertices": int(np.count_nonzero(vertex_alpha >= (1.0 - 1e-12))),
                "vertex_alpha": {
                    "median": float(np.median(vertex_alpha)),
                    "mean": float(np.mean(vertex_alpha)),
                    "p05": float(np.percentile(vertex_alpha, 5)),
                    "min": float(np.min(vertex_alpha)),
                },
                "vertexwise_iterations": vertex_records,
                "full_candidate_metrics": full_metrics,
                "final_metrics": vertex_metrics,
                "fallback_global_used": False,
            }

    # Compatibilidad defensiva: la estrategia por anillos queda únicamente como
    # segundo recurso para mallas patológicas que no puedan resolverse de forma
    # local por vértice.
    best = None
    best_metrics = None
    best_alpha_field = None
    best_beta = 0.0
    best_rings = int(local_rollback_rings)
    expansion_records = []

    face_seed = np.asarray(unsafe, dtype=np.int64)

    for expansion in range(max(1, int(local_max_expansions))):
        rings = int(local_rollback_rings) + expansion

        weight = rollback_weight_from_faces(
            len(p0),
            triangles,
            face_seed,
            rings=rings,
            ring_floor=local_ring_floor,
            coincident_groups=coincident_groups,
        )

        zero_candidate, zero_alpha = local_mix_from_original(
            p0,
            ps,
            weight,
            0.0,
        )
        zero_unsafe = unsafe_orientation_face_indices(
            p0,
            zero_candidate,
            triangles,
        )
        zero_metrics = safety_metrics(
            p0,
            zero_candidate,
            triangles,
        )

        record = {
            "expansion": int(expansion + 1),
            "rings": int(rings),
            "seed_faces": int(len(face_seed)),
            "affected_vertices": int(np.count_nonzero(weight > 0)),
            "beta_zero_unsafe_faces": int(len(zero_unsafe)),
        }

        if len(zero_unsafe) > 0 or zero_metrics["maximum_absolute_robust_extent_change"] > float(
            max_extent_change_ratio
        ):
            face_seed = np.unique(np.concatenate([face_seed, zero_unsafe]))
            record["resolved_at_beta_zero"] = False
            expansion_records.append(record)
            continue

        record["resolved_at_beta_zero"] = True

        low = 0.0
        high = 1.0
        local_best = zero_candidate
        local_best_alpha = zero_alpha
        local_best_beta = 0.0
        local_best_metrics = zero_metrics

        for _ in range(max(1, int(bisection_steps))):
            beta = 0.5 * (low + high)
            candidate, alpha_field = local_mix_from_original(
                p0,
                ps,
                weight,
                beta,
            )
            unsafe_now = unsafe_orientation_face_indices(
                p0,
                candidate,
                triangles,
            )
            metrics = safety_metrics(
                p0,
                candidate,
                triangles,
            )

            ok = (
                len(unsafe_now) == 0
                and metrics["maximum_absolute_robust_extent_change"]
                <= float(max_extent_change_ratio)
                and metrics["orientation_flip_ratio"] <= float(max_orientation_flip_ratio)
                and metrics["degenerate_face_ratio"] <= 1e-4
            )

            if ok:
                local_best = candidate
                local_best_alpha = alpha_field
                local_best_beta = beta
                local_best_metrics = metrics
                low = beta
            else:
                high = beta

        final_unsafe = unsafe_orientation_face_indices(
            p0,
            local_best,
            triangles,
        )
        if len(final_unsafe) == 0:
            best = local_best
            best_alpha_field = local_best_alpha
            best_beta = float(local_best_beta)
            best_metrics = local_best_metrics
            best_rings = int(rings)
            record["local_beta"] = float(local_best_beta)
            expansion_records.append(record)
            break

        face_seed = np.unique(np.concatenate([face_seed, final_unsafe]))
        expansion_records.append(record)

    if best is not None:
        alpha = np.asarray(best_alpha_field, dtype=np.float64)
        return best, {
            "activated": True,
            "mode": "local_orientation_rollback",
            "extent_alpha": float(extent_alpha),
            "local_beta": float(best_beta),
            "rings_used": int(best_rings),
            "affected_vertices": int(np.count_nonzero(alpha < (1.0 - 1e-12))),
            "full_strength_vertices": int(np.count_nonzero(alpha >= (1.0 - 1e-12))),
            "vertex_alpha": {
                "median": float(np.median(alpha)),
                "mean": float(np.mean(alpha)),
                "p05": float(np.percentile(alpha, 5)),
                "min": float(np.min(alpha)),
            },
            "expansions": expansion_records,
            "full_candidate_metrics": full_metrics,
            "final_metrics": best_metrics,
            "fallback_global_used": False,
        }

    # ----------------------------------------------------------
    # 3) Fallback global: seguridad antes que apariencia.
    # ----------------------------------------------------------
    low = 0.0
    high = 1.0
    best_global = p0.copy()
    best_global_alpha = 0.0
    best_global_metrics = safety_metrics(
        p0,
        best_global,
        triangles,
    )

    for _ in range(max(1, int(bisection_steps))):
        alpha = 0.5 * (low + high)
        candidate = p0 + alpha * (ps_full - p0)
        metrics = safety_metrics(p0, candidate, triangles)
        unsafe_now = unsafe_orientation_face_indices(
            p0,
            candidate,
            triangles,
        )

        ok = (
            len(unsafe_now) == 0
            and metrics["maximum_absolute_robust_extent_change"] <= float(max_extent_change_ratio)
            and metrics["orientation_flip_ratio"] <= float(max_orientation_flip_ratio)
            and metrics["degenerate_face_ratio"] <= 1e-4
        )

        if ok:
            best_global = candidate
            best_global_alpha = alpha
            best_global_metrics = metrics
            low = alpha
        else:
            high = alpha

    return best_global, {
        "activated": True,
        "mode": "global_fallback",
        "extent_alpha": float(extent_alpha),
        "blend_alpha": float(best_global_alpha),
        "local_beta": None,
        "affected_vertices": int(len(p0)),
        "full_strength_vertices": 0,
        "expansions": expansion_records,
        "full_candidate_metrics": full_metrics,
        "final_metrics": best_global_metrics,
        "fallback_global_used": True,
    }


def load_terminal_seams(mesh_path, vertices, triangles, topology_info, spacing):
    """Procedencia explícita del paso 14; nunca inferir una costura por altura."""
    records = topology_info.get("local_closure", {}).get("records", [])
    records = [r for r in records if r.get("accepted") and r.get("protected_seam_xyz")]
    mask = np.zeros(len(vertices), dtype=bool)
    info = {"enabled": bool(records), "verified_source": False, "vertices": 0, "edges": 0}
    if not records:
        return mask, set(), info
    expected = topology_info.get("protected_seams_mesh_sha256")
    if not expected or hashlib.sha256(Path(mesh_path).read_bytes()).hexdigest() != expected:
        raise ValueError(
            "La malla del paso 14 no corresponde al resumen de aristas protegidas. Regenera el paso 14."
        )
    tol = max(1e-8, spacing * 1e-5)
    tree = cKDTree(vertices)
    mesh_edges = set()
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            mesh_edges.add(tuple(sorted((int(a), int(b)))))
    locked_edges = set()
    permitted = np.zeros_like(vertices)
    for rec in records:
        xyz = np.asarray(rec["protected_seam_xyz"], dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3 or not np.all(np.isfinite(xyz)):
            raise ValueError("Contorno protegido inválido en resumen del paso 14.")
        ids = tree.query_ball_point(xyz, r=tol)
        if any(not group for group in ids):
            raise ValueError("No se localiza toda la costura del paso 14 en su malla.")
        fixed = rec.get("seam_fixed_mask", [True] * len(xyz))
        if len(fixed) != len(xyz) or any(type(value) is not bool for value in fixed):
            raise ValueError("Máscara de anclajes de la costura inválida en paso 14.")
        budget = float(rec.get("seam_motion_budget_mm", 0.0))
        if not np.isfinite(budget) or budget < 0 or budget > 0.12 * spacing + 1e-9:
            raise ValueError("Presupuesto de redistribución de la costura inválido.")
        remaining = np.asarray(rec.get("seam_motion_remaining_mm", [0.0] * len(xyz)), dtype=float)
        if (
            remaining.shape != (len(xyz),)
            or not np.all(np.isfinite(remaining))
            or np.any(remaining < 0)
        ):
            raise ValueError("Presupuesto acumulado inválido en la costura del paso 14.")
        policy_ok = rec.get("seam_motion_policy") == "bounded_curve_tangential_redistribution_only"
        for j, group in enumerate(ids):
            mask[np.asarray(group, dtype=int)] = True
            # Solo redistribución sobre UN segmento existente, sin cruzar la
            # arista tapa-pared ni volver a estimar su altura en el pulido.
            # Las esquinas y las coincidencias topológicas permanecen inmóviles.
            if policy_ok and not fixed[j] and len(group) == 1 and budget > 0:
                back = xyz[j] - xyz[j - 1]
                ahead = xyz[(j + 1) % len(xyz)] - xyz[j]
                lb, la = np.linalg.norm(back), np.linalg.norm(ahead)
                if min(lb, la) > 1e-8 and np.dot(back, ahead) / (lb * la) >= np.cos(
                    np.deg2rad(20.0)
                ):
                    signed = 0.20 * (la - lb)
                    direction = ahead / la if signed >= 0 else -back / lb
                    travel = min(abs(signed), budget, float(remaining[j]), 0.15 * min(lb, la))
                    permitted[group[0]] = travel * direction
            pairs = {
                tuple(sorted((int(a), int(b))))
                for a in group
                for b in ids[(j + 1) % len(ids)]
                if a != b
            }
            found = pairs & mesh_edges
            if not found:
                raise ValueError(
                    "La costura protegida no es continua en la conectividad de la malla del paso 14."
                )
            locked_edges.update(found)
    info.update(
        verified_source=True,
        vertices=int(mask.sum()),
        edges=len(locked_edges),
        policy="anchored_crease_with_bounded_on_curve_redistribution",
        movable_vertices=int(np.count_nonzero(np.linalg.norm(permitted, axis=1) > 1e-12)),
        maximum_permitted_shift_mm=float(np.linalg.norm(permitted, axis=1).max()),
        _permitted_displacement=permitted,
    )
    return mask, locked_edges, info


def protect_terminal_seams(structure, mask, locked_edges):
    structure["terminal_seam_vertex"] = mask.copy()
    structure["terminal_seam_edges"] = set(locked_edges)
    structure["vertex_strength"][mask] = 0.0
    # Bloquear tránsito entre superficies a través de la costura. No etiquetar
    # esa costura cerrada como un hueco ni inflar el contador de bordes abiertos.
    e = structure["edges"]
    touching = mask[e[:, 0]] | mask[e[:, 1]]
    structure["fit_edges"] = e[~touching]
    seam = np.asarray([tuple(map(int, edge)) in locked_edges for edge in e], dtype=bool)
    structure["edge_weight"][seam] = 0.0
    structure["dihedral_deg"][seam] = 180.0
    structure["vertex_max_dihedral_deg"][mask] = 180.0


def _contextual_surface_fit(q, normals, confidence, mid, h):
    """Continuidad espacial del entorno a tres escalas, sin modelo de objeto.

    La zona central no determina la tendencia: se predice desde una corona.
    La confianza pondera observaciones; no sustituye la consistencia geométrica.
    """
    delta = q - mid
    distance = np.linalg.norm(delta, axis=1)
    outer = (distance >= 1.5 * h) & (distance <= 8 * h)
    if np.count_nonzero(outer) < 24:
        return None, "insufficient_surrounding_evidence"
    covariance = delta[outer].T @ delta[outer] / np.count_nonzero(outer)
    eig, axis = np.linalg.eigh(covariance)
    if eig[1] < h * h * 0.3:
        return None, "one_dimensional_surroundings"
    normal = axis[:, 0]
    if np.median(normals[outer] @ normal) < 0:
        normal = -normal
    a = axis[:, 2]
    b = np.cross(normal, a)
    x, y, z = delta @ a / h, delta @ b / h, delta @ normal
    radial = np.hypot(x, y)
    sectors = ((np.arctan2(y, x) + np.pi) * 4 / np.pi).astype(int) % 8
    A = np.column_stack((np.ones(len(q)), x, y, x * x, x * y, y * y))
    models = []
    for radius in (4.0, 6.0, 8.0):
        ring = (radial >= 1.5) & (radial <= radius) & (distance <= radius * h)
        if np.count_nonzero(ring) < 18:
            continue
        ids = np.flatnonzero(ring)
        occupancy = np.bincount(sectors[ids], minlength=8)
        # Compensar concentración de puntos: importa cubrir la superficie.
        weight0 = confidence[ids] / np.maximum(occupancy[sectors[ids]], 1)
        weight = weight0.copy()
        good = True
        for _ in range(4):
            coef, _, rank, singular = np.linalg.lstsq(
                A[ids] * np.sqrt(weight)[:, None], z[ids] * np.sqrt(weight), rcond=1e-7
            )
            if rank < 6 or singular[-1] <= 0 or singular[0] / singular[-1] > 5000:
                good = False
                break
            residual = z[ids] - A[ids] @ coef
            sigma = max(0.06 * h, 1.4826 * np.median(np.abs(residual - np.median(residual))))
            weight = weight0 * np.minimum(1.0, 1.5 * sigma / np.maximum(np.abs(residual), 1e-12))
        if not good or sigma > 0.30 * h:
            continue
        predicted = (
            normal[None, :]
            - ((coef[1] + 2 * coef[3] * x + coef[4] * y) / h)[:, None] * a
            - ((coef[2] + coef[4] * x + 2 * coef[5] * y) / h)[:, None] * b
        )
        predicted /= np.maximum(np.linalg.norm(predicted, axis=1)[:, None], 1e-12)
        residual = z - A @ coef
        consistent = (np.abs(residual) <= max(0.25 * h, 2.5 * sigma)) & (
            (predicted * normals).sum(axis=1) >= np.cos(np.deg2rad(30.0))
        )
        # El entorno debe ser suave en direcciones distintas, no solo contener
        # un grupo numeroso. Retener aristas, capas y esquinas con evidencia.
        covered = 0
        for sector in range(8):
            chosen = ring & (sectors == sector)
            if chosen.any() and np.average(consistent[chosen], weights=confidence[chosen]) >= 0.75:
                covered += 1
        if covered < 6 or np.average(consistent[ids], weights=weight0) < 0.80:
            continue
        models.append((coef, sigma))
    if len(models) < 2:
        return None, "surroundings_not_smooth_at_multiple_scales"
    coef, sigma = models[-1]
    # Coincidencia de las predicciones centrales y de la pendiente, no solo
    # del error de ajuste (una corona por sí sola puede sobreajustarse).
    probes = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, -1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 0, 1],
        ],
        float,
    )
    disagreement = max(float(np.max(np.abs(probes @ (coef - c)))) for c, _ in models)
    uncertainty = max(sigma, disagreement)
    if disagreement > max(0.18 * h, 2 * sigma):
        return None, "context_changes_with_scale"
    central = radial <= 1.5
    residual = z - A @ coef
    # El relieve central observado prevalece sobre la hipótesis de pared lisa.
    # Se evalúa fracción ponderada y extensión, no el tamaño de un grupo.
    if central.any():
        offset = np.abs(residual[central]) > max(0.20 * h, 3 * uncertainty)
        alternate_fraction = float(np.average(offset, weights=confidence[central]))
        if alternate_fraction > 0.35:
            return None, "central_relief_or_second_layer_supported"
        good_center = central & (np.abs(residual) <= max(0.25 * h, 2.5 * uncertainty))
        if not good_center.any():
            return None, "unobserved_center"
    else:
        return None, "unobserved_center"
    return (coef, normal, a, b, uncertainty, len(models)), "accepted"


def contextual_surface_model(q, normals, confidence, mid, h):
    """Referencia robusta por celdas: tolera ruido sin mezclar capas."""
    delta = q - mid
    outer = np.linalg.norm(delta, axis=1) >= 1.5 * h
    if outer.sum() < 24:
        return None, "insufficient_surrounding_evidence"
    _, basis = np.linalg.eigh(np.cov(delta[outer].T))
    uv = delta @ basis[:, 1:] / h
    keys = np.floor(uv).astype(np.int64)
    _, groups = np.unique(keys, axis=0, return_inverse=True)
    points = []
    directions = []
    confidences = []
    for group in range(int(groups.max()) + 1):
        ids = np.flatnonzero(groups == group)
        nn = normals[ids]
        direction = np.median(nn, axis=0)
        length = np.linalg.norm(direction)
        if length < 0.65:
            continue
        direction /= length
        # No promediar normales incompatibles ni dos estratos separados.
        if np.mean(nn @ direction < np.cos(np.deg2rad(40))) > 0.25:
            continue
        height = np.sort(q[ids] @ direction)
        if len(height) >= 4 and np.max(np.diff(height)) > 0.6 * h:
            continue
        points.append(np.median(q[ids], axis=0))
        directions.append(direction)
        confidences.append(float(np.median(confidence[ids])))
    if len(points) < 24:
        return None, "insufficient_spatial_cells"
    model, reason = _contextual_surface_fit(
        np.asarray(points), np.asarray(directions), np.asarray(confidences), mid, h
    )
    if model is None:
        return None, reason
    coef, n, a, b, sigma, _ = model
    x, y = delta @ a / h, delta @ b / h
    A = np.column_stack((np.ones(len(q)), x, y, x * x, x * y, y * y))
    central = np.hypot(x, y) <= 1.5
    if not central.any():
        return None, "unobserved_center"
    residual = delta @ n - A @ coef
    # Validar también contra datos originales, no solo sus representantes.
    if (
        np.average(
            np.abs(residual[central]) > max(0.25 * h, 3 * sigma), weights=confidence[central]
        )
        > 0.35
    ):
        return None, "central_relief_or_second_layer_supported"
    return model, "accepted"


@operacion("Evaluar continuidad superficial y grumos")
def classify_cloud_supported_features(vertices, triangles, structure, cloud_data, spacing, args):
    report = {
        "candidate_edges": 0,
        "released_fold_edges": 0,
        "supported_feature_edges": 0,
        "uncertain_edges_preserved": 0,
        "object_specific_model": False,
        "method": "perpendicular_deviation_cached_robust_surface_reference",
        "rejection_counts": {},
        "context_radii_spacing": [4.0, 6.0, 8.0],
        "central_exclusion_spacing": 1.5,
        "fixed_group_size_rejection": False,
    }
    structure["cloud_feature_evidence"] = report
    if cloud_data is None:
        report["reason"] = "cloud_evidence_unavailable"
        return
    cloud, cn, cf, cs = cloud_data
    if cn is None:
        report["reason"] = "cloud_normals_unavailable"
        return
    valid = (
        np.isfinite(cloud).all(axis=1)
        & np.isfinite(cn).all(axis=1)
        & np.isfinite(cf)
        & np.isfinite(cs)
    )
    valid &= (cs >= 2) & (cf >= 0.5) & (np.linalg.norm(cn, axis=1) > 1e-8)
    cloud = cloud[valid]
    cn = cn[valid]
    cf = cf[valid]
    cn = cn / np.maximum(np.linalg.norm(cn, axis=1)[:, None], 1e-12)
    report["eligible_observations"] = int(len(cloud))
    if len(cloud) < 24:
        report["reason"] = "insufficient_cloud"
        return
    h = max(float(spacing), 1e-6)
    edges = structure["edges"]
    actual_fn, _ = face_geometry(vertices, triangles)
    pair = {}
    for fi, face in enumerate(triangles):
        for u, v in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            pair.setdefault(tuple(sorted((int(u), int(v)))), []).append(fi)
    guide = structure["feature_guide_vertices"]
    guide_normals = vertex_normals(guide, triangles)
    guide_shift = np.abs(np.sum((guide - vertices) * guide_normals, axis=1))
    rough = np.max(guide_shift[edges], axis=1) > 0.15 * h
    candidates = np.flatnonzero(
        ((structure["dihedral_deg"] >= args.feature_soft_angle_deg) | rough)
        & ~structure["boundary_edge"]
    )
    report["candidate_edges"] = int(len(candidates))
    effective = structure["dihedral_deg"].copy()
    decisions = np.zeros(len(edges), np.uint8)
    geometry_class = np.zeros(len(edges), np.uint8)
    correction = np.zeros_like(vertices)
    weights = np.zeros(len(vertices))
    length_sum = np.zeros(len(vertices))
    tree = cKDTree(cloud)
    mids = vertices[edges[candidates]].mean(axis=1)
    directions = guide_normals[edges[candidates]].mean(axis=1)
    directions /= np.maximum(np.linalg.norm(directions, axis=1)[:, None], 1e-12)
    zone_keys = np.column_stack(
        (np.floor(mids / (1.5 * h)).astype(np.int64), np.round(directions * 3).astype(np.int64))
    )
    _, zone_ids = np.unique(zone_keys, axis=0, return_inverse=True)
    zone_cache = {}
    zone_count = int(zone_ids.max() + 1) if len(zone_ids) else 0
    zone_centers = np.zeros((zone_count, 3))
    np.add.at(zone_centers, zone_ids, mids)
    zone_centers /= np.maximum(np.bincount(zone_ids, minlength=zone_count)[:, None], 1)
    report["reference_zones"] = int(zone_ids.max() + 1) if len(zone_ids) else 0
    report["candidate_metric"] = "absolute_perpendicular_guide_deviation"
    report["reference_method"] = "robust_spatial_cells_multiscale_quadratic"

    def reject(reason):
        report["rejection_counts"][reason] = report["rejection_counts"].get(reason, 0) + 1

    for start in range(0, len(candidates), 128):
        block = candidates[start : start + 128]
        mid = vertices[edges[block]].mean(axis=1)
        for row, ei in enumerate(block):
            faces = pair[tuple(edges[ei])]
            if len(faces) != 2:
                reject("nonmanifold_edge")
                continue
            zone = int(zone_ids[start + row])
            if zone not in zone_cache:
                center = zone_centers[zone]
                distances, indices = tree.query(
                    center, k=min(512, len(cloud)), workers=query_threads()
                )
                ids = indices[distances <= 8 * h]
                model, reason = contextual_surface_model(cloud[ids], cn[ids], cf[ids], center, h)
                zone_cache[zone] = (center, ids, model, reason)
            center, ids, model, reason = zone_cache[zone]
            if model is None:
                reject(reason)
                continue
            coef, n, a, b, uncertainty, nscales = model
            endpoints = vertices[edges[ei]]
            delta = endpoints - center
            x, y = delta @ a / h, delta @ b / h
            A = np.column_stack((np.ones(2), x, y, x * x, x * y, y * y))
            gradient = (
                n[None, :]
                - ((coef[1] + 2 * coef[3] * x + coef[4] * y) / h)[:, None] * a
                - ((coef[2] + coef[4] * x + 2 * coef[5] * y) / h)[:, None] * b
            )
            gradient_length = np.maximum(np.linalg.norm(gradient, axis=1), 1e-12)
            offset = (delta @ n - A @ coef) / gradient_length
            perpendicular = gradient / gradient_length[:, None]
            if np.max(np.abs(offset)) > 1.5 * h:
                reject("deviation_exceeds_local_budget")
                continue
            centroid = vertices[triangles[faces]].mean(axis=1) - center
            cx, cy = centroid @ a / h, centroid @ b / h
            normals = (
                n[None, :]
                - ((coef[1] + 2 * coef[3] * cx + coef[4] * cy) / h)[:, None] * a
                - ((coef[2] + coef[4] * cx + 2 * coef[5] * cy) / h)[:, None] * b
            )
            normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
            angle = float(np.degrees(np.arccos(np.clip(normals[0] @ normals[1], -1, 1))))
            positional_bump = np.max(np.abs(offset)) > max(0.15 * h, 2 * uncertainty)
            angular_fold = effective[ei] >= angle + 8.0
            if angle >= args.feature_soft_angle_deg or not (angular_fold or positional_bump):
                reject("no_deviation_contradicted_by_context")
                continue
            # Cercanía al soporte comprobada para ambos extremos; la corona
            # no autoriza inventar una superficie dentro de un hueco grande.
            closest = np.min(
                np.linalg.norm(cloud[ids, None, :] - endpoints[None, :, :], axis=2), axis=0
            )
            if np.max(closest) > 2 * h:
                reject("endpoints_without_local_observations")
                continue
            decisions[ei] = 2
            effective[ei] = min(effective[ei], angle)
            geometry_class[ei] = 1 if np.linalg.norm(coef[3:]) < 0.10 * h else 2
            limit = min(0.48 * h, max(0.10 * h, 2 * uncertainty + 0.10 * h))
            selected = np.abs(offset) > max(0.10 * h, 2 * uncertainty)
            change = -perpendicular * np.clip(offset, -limit, limit)[:, None] * 0.8
            for j, vi in enumerate(edges[ei]):
                if not selected[j] or structure["boundary_vertex"][vi]:
                    continue
                weight = 1.0 / max(uncertainty, 0.06 * h)
                correction[vi] += weight * change[j]
                weights[vi] += weight
                length_sum[vi] += weight * np.linalg.norm(change[j])
        print(
            f"[PROGRESO] Paso 15 | Continuidad y grumos: {min(start+128,len(candidates))}/{len(candidates)} parches",
            flush=True,
        )
    # Si los entornos proponen direcciones opuestas, no resolver por promedio.
    coherent = np.linalg.norm(correction, axis=1) >= 0.8 * length_sum
    correction /= np.maximum(weights[:, None], 1e-12)
    correction[~coherent] = 0.0
    correction[structure["boundary_vertex"]] = 0.0
    structure["context_correction"] = correction
    structure["context_confirmed_vertex"] = np.linalg.norm(correction, axis=1) > 0
    report.update(
        released_fold_edges=int(np.count_nonzero(decisions == 2)),
        uncertain_edges_preserved=int(np.count_nonzero(decisions[candidates] == 0)),
        context_correction_candidates=int(np.count_nonzero(structure["context_confirmed_vertex"])),
        proposed_context_shift_mm=stats(np.linalg.norm(correction, axis=1)),
    )
    report["local_geometry_counts"] = {
        name: int(np.count_nonzero(geometry_class[candidates] == code))
        for code, name in enumerate(("uncertain", "planar", "curved", "edge_transition"))
    }
    structure["local_geometry_class"] = geometry_class
    structure["geometric_dihedral_deg"] = structure["dihedral_deg"].copy()
    structure["cloud_feature_class"] = decisions
    structure["dihedral_deg"] = effective
    soft = float(args.feature_soft_angle_deg)
    hard = max(float(args.feature_hard_angle_deg), soft + 1e-6)
    transition = smoothstep01((effective - soft) / (hard - soft))
    structure["edge_weight"] = 1 - transition * (1 - float(args.edge_crossing_min_weight))
    structure["edge_weight"][structure["boundary_edge"]] = 0.0
    maximum = np.zeros(len(vertices))
    np.maximum.at(maximum, edges[:, 0], effective)
    np.maximum.at(maximum, edges[:, 1], effective)
    structure["vertex_max_dihedral_deg"] = maximum
    strength = 1 - smoothstep01((maximum - soft) / (hard - soft)) * (
        1 - float(args.feature_min_strength)
    )
    strength[structure["boundary_vertex"]] = 0.0
    ring = structure["boundary_ring_vertex"]
    strength[ring] = np.minimum(strength[ring], float(args.boundary_ring_strength))
    structure["vertex_strength"] = np.clip(strength, 0, 1)
    print(
        f"[Paso 15] Continuidad contextual: {report['released_fold_edges']} pliegues confirmados; "
        f"{report['context_correction_candidates']} vértices candidatos a corrección.",
        flush=True,
    )


def regularize_mesh_core(
    vertices,
    triangles,
    *,
    spacing_mm,
    coincident_groups=None,
    args,
    terminal_seams=None,
    cloud_data=None,
):
    p0 = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)

    structure = build_edge_structure(
        p0,
        t,
        soft_angle_deg=args.feature_soft_angle_deg,
        hard_angle_deg=args.feature_hard_angle_deg,
        feature_min_strength=args.feature_min_strength,
        edge_crossing_min_weight=args.edge_crossing_min_weight,
        boundary_ring_strength=args.boundary_ring_strength,
        feature_guide_iterations=args.feature_guide_iterations,
        feature_guide_lambda=args.feature_guide_lambda,
    )

    classify_cloud_supported_features(p0, t, structure, cloud_data, spacing_mm, args)

    if terminal_seams is not None:
        seam_mask, seam_edges = terminal_seams[:2]
        protect_terminal_seams(structure, seam_mask, seam_edges)

    max_step = float(args.max_shift_per_pass_spacing) * float(spacing_mm)
    max_total = float(args.max_total_shift_spacing) * float(spacing_mm)

    p = p0.copy()
    iteration_records = []
    step_clipped_any = np.zeros(len(p), dtype=bool)
    total_clipped_any = np.zeros(len(p), dtype=bool)

    # La tendencia contextual también propone movimiento: no se limita a
    # quitar una etiqueta de protección. Bordes y costuras permanecen fijos.
    context_shift = structure.get("context_correction", np.zeros_like(p0)).copy()
    context_shift *= structure["vertex_strength"][:, None]
    if terminal_seams is not None:
        context_shift[terminal_seams[0]] = 0.0
    if np.any(context_shift):
        print("[PROGRESO] Paso 15 | Corrigiendo desviaciones respecto al entorno suave", flush=True)
        candidate = enforce_coincident_group_motion(
            p0 + context_shift,
            p0,
            coincident_groups or [],
            vertex_strength=structure["vertex_strength"],
        )
        p, alpha, trace, safe = vertexwise_orientation_guard(
            p0, candidate, t, coincident_groups=coincident_groups, max_iterations=28
        )
        if not safe:
            p = p0.copy()
        structure["cloud_feature_evidence"]["context_motion"] = {
            "accepted": bool(safe),
            "moved_vertices": int(np.count_nonzero(np.linalg.norm(p - p0, axis=1) > 1e-10)),
            "shift_mm": stats(np.linalg.norm(p - p0, axis=1)),
            "restricted_vertices": int(np.count_nonzero(alpha < 1)),
            "subject_to_final_intersection_and_extent_guards": True,
        }
        structure.setdefault("incremental_orientation", []).append(
            {
                "operation": "contextual_surface_correction",
                "accepted": bool(safe),
                "iterations": len(trace),
                "restricted_vertices": int(np.count_nonzero(alpha < 1)),
            }
        )

    # --------------------------------------------------------------
    # V1.4 — campo de normales + tendencia multiescala.
    # --------------------------------------------------------------
    if bool(args.normal_trend_enabled):
        p, normal_trend = multiscale_normal_trend_regularization(
            p,
            p0,
            t,
            structure,
            coincident_groups or [],
            spacing_mm=float(spacing_mm),
            args=args,
        )
    else:
        normal_info = filtered_vertex_normals(
            p,
            t,
            structure,
            args=args,
        )
        normal_trend = {
            "cycles": [],
            "last": {
                "normal_info": normal_info,
                "small_prediction_mm": np.full(len(p), np.nan),
                "large_prediction_mm": np.full(len(p), np.nan),
                "small_residual_scale_mm": np.full(len(p), np.nan),
                "large_residual_scale_mm": np.full(len(p), np.nan),
                "scale_difference_mm": np.full(len(p), np.nan),
                "gate": np.zeros(len(p)),
                "shift_mm": np.zeros(len(p)),
                "small_neighbor_count": np.zeros(len(p), dtype=np.int32),
                "large_neighbor_count": np.zeros(len(p), dtype=np.int32),
                "max_step_mm": 0.0,
                "max_total_mm": 0.0,
                "total_clipped": np.zeros(len(p), dtype=bool),
            },
            "total_shift_mm": np.zeros(len(p)),
        }

    # --------------------------------------------------------------
    # V1.3 — despiking robusto residual antes del Taubin.
    # --------------------------------------------------------------
    if bool(args.despike_enabled):
        p, despike = despike_mesh_vertices(
            p,
            p0,
            t,
            structure,
            coincident_groups or [],
            spacing_mm=float(spacing_mm),
            args=args,
        )
    else:
        despike = {
            "residual_mm": np.full(len(p), np.nan),
            "local_scale_mm": np.full(len(p), np.nan),
            "robust_zscore": np.zeros(len(p)),
            "candidate_outlier": np.zeros(len(p), dtype=bool),
            "eligible": np.zeros(len(p), dtype=bool),
            "isolated": np.zeros(len(p), dtype=bool),
            "selected": np.zeros(len(p), dtype=bool),
            "same_sign_neighbor_ratio": np.ones(len(p)),
            "shift_mm": np.zeros(len(p)),
            "selected_count": 0,
            "max_shift_allowed_mm": 0.0,
        }

    for i in range(max(0, int(args.iterations))):
        p, d1 = limited_pass(
            p,
            p0,
            t,
            structure,
            float(args.lambda_factor),
            coincident_groups=coincident_groups,
            normal_component=args.normal_component,
            tangential_component=args.tangential_component,
            max_step_mm=max_step,
            max_total_mm=max_total,
        )
        p, d2 = limited_pass(
            p,
            p0,
            t,
            structure,
            float(args.mu_factor),
            coincident_groups=coincident_groups,
            normal_component=args.normal_component,
            tangential_component=args.tangential_component,
            max_step_mm=max_step,
            max_total_mm=max_total,
        )

        step_clipped_any |= d1["step_clipped"] | d2["step_clipped"]
        total_clipped_any |= d1["total_clipped"] | d2["total_clipped"]

        iteration_records.append(
            {
                "iteration": i + 1,
                "lambda_step_mm": stats(d1["step_mm"]),
                "mu_step_mm": stats(d2["step_mm"]),
            }
        )

    # --------------------------------------------------------------
    # V1.11 — fairing residual adaptativo.
    # Corrige rugosidad formada por regiones, no únicamente picos aislados.
    # Se ejecuta al final para medir el residuo que realmente dejó Taubin.
    # --------------------------------------------------------------
    if bool(args.surface_fairing_enabled):
        p, surface_fairing = adaptive_surface_fairing_regularization(
            p,
            p0,
            t,
            structure,
            coincident_groups or [],
            spacing_mm=float(spacing_mm),
            args=args,
        )
    else:
        normal_info = filtered_vertex_normals(
            p,
            t,
            structure,
            args=args,
        )
        surface_fairing = {
            "cycles": [],
            "last": {
                "normal_info": normal_info,
                "target_mm": np.full(len(p), np.nan),
                "local_scale_mm": np.full(len(p), np.nan),
                "compatible_neighbor_count": np.zeros(len(p), dtype=np.int32),
                "gate": np.zeros(len(p)),
                "active": np.zeros(len(p), dtype=bool),
                "shift_mm": np.zeros(len(p)),
                "max_step_mm": 0.0,
                "max_total_mm": 0.0,
                "total_clipped": np.zeros(len(p), dtype=bool),
            },
            "total_shift_mm": np.zeros(len(p)),
            "total_shift_from_input_mm": np.linalg.norm(p - p0, axis=1),
            "convergence_threshold_mm": 0.0,
            "neighborhood_size": stats([]),
        }

    p = enforce_coincident_group_motion(
        p,
        p0,
        coincident_groups or [],
        vertex_strength=structure["vertex_strength"],
    )

    if terminal_seams is not None:
        # Desplazamiento acotado sobre el contorno corregido del 14. Se añade
        # ANTES de las guardas de orientación, extensión e intersecciones.
        permitted = terminal_seams[2] if len(terminal_seams) > 2 else np.zeros_like(p0)
        p[terminal_seams[0]] = p0[terminal_seams[0]] + permitted[terminal_seams[0]]

    guarded, guard = apply_global_safety_guard(
        p0,
        p,
        t,
        max_extent_change_ratio=args.max_robust_extent_change_ratio,
        max_orientation_flip_ratio=args.max_orientation_flip_ratio,
        bisection_steps=args.safety_bisection_steps,
        coincident_groups=coincident_groups,
        local_rollback_rings=args.safety_local_rollback_rings,
        local_max_expansions=args.safety_local_max_expansions,
        local_ring_floor=args.safety_local_ring_floor,
    )

    total_shift = np.linalg.norm(guarded - p0, axis=1)

    return {
        "motion_diagnostics": {
            "proposed_before_orientation_guard_mm": stats(np.linalg.norm(p - p0, axis=1)),
            "after_orientation_guard_mm": stats(np.linalg.norm(guarded - p0, axis=1)),
            "vertices_with_proposed_motion": int(
                np.count_nonzero(np.linalg.norm(p - p0, axis=1) > 1e-6)
            ),
            "vertices_retaining_motion": int(
                np.count_nonzero(np.linalg.norm(guarded - p0, axis=1) > 1e-6)
            ),
        },
        "vertices": guarded,
        "structure": structure,
        "guard": guard,
        "total_shift_mm": total_shift,
        "normal_trend": normal_trend,
        "despike": despike,
        "surface_fairing": surface_fairing,
        "iteration_records": iteration_records,
        "max_step_mm": float(max_step),
        "max_total_mm": float(
            max(
                max_total,
                float(surface_fairing["last"]["max_total_mm"]),
            )
        ),
        "step_clipped_any": step_clipped_any,
        "total_clipped_any": total_clipped_any,
    }


def semantic_intersection_analysis(
    vertices,
    triangles,
    *,
    spacing_mm,
    geometric_epsilon_spacing_factor=1e-4,
    contact_locality_spacing_factor=1e-3,
):
    """Clasifica intersecciones y conserva las parejas bloqueantes.

    La clasificación es exactamente la misma usada por Paso 14.
    """
    require_open3d()

    try:
        from clasificador_intersecciones import classify_triangle_pair
    except Exception as exc:
        raise RuntimeError(
            "Paso 15 requiere clasificador_intersecciones.py "
            "en la carpeta scripts, igual que Paso 14. "
            f"Detalle: {exc}"
        ) from exc

    v = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)

    mesh = mesh_from_arrays(v, t)
    pairs = np.asarray(
        mesh.get_self_intersecting_triangles(),
        dtype=np.int64,
    )
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    else:
        pairs = pairs.reshape(-1, 2)

    eps = max(
        1e-8,
        float(geometric_epsilon_spacing_factor) * float(spacing_mm),
    )
    contact_tol = max(
        10.0 * eps,
        float(contact_locality_spacing_factor) * float(spacing_mm),
    )

    categories = {}
    true_categories = {
        "transverse_intersection",
        "coplanar_overlap",
        "unexplained_coplanar_contact",
    }
    ambiguous_categories = {
        "numerically_ambiguous",
        "degenerate_ambiguous",
    }

    blocking_pairs = []
    true_count = 0
    ambiguous_count = 0
    contact_count = 0

    for fa, fb in pairs:
        fa = int(fa)
        fb = int(fb)

        ia = t[fa]
        ib = t[fb]

        result = classify_triangle_pair(
            v[ia],
            v[ib],
            triangle_indices_a=ia,
            triangle_indices_b=ib,
            geometric_epsilon=eps,
            contact_locality_tolerance=contact_tol,
        )

        category = str(result.get("category", "unknown"))
        categories[category] = categories.get(category, 0) + 1

        blocking_kind = None
        if category in true_categories:
            true_count += 1
            blocking_kind = "true"
        elif category in ambiguous_categories:
            ambiguous_count += 1
            blocking_kind = "ambiguous"
        else:
            contact_count += 1

        if blocking_kind is not None:
            blocking_pairs.append(
                {
                    "face_a": fa,
                    "face_b": fb,
                    "category": category,
                    "kind": blocking_kind,
                }
            )

    return {
        "raw_pair_count": int(len(pairs)),
        "true_self_intersection_pairs": int(true_count),
        "ambiguous_intersection_pairs": int(ambiguous_count),
        "nonblocking_contact_pairs": int(contact_count),
        "categories": {str(k): int(vv) for k, vv in sorted(categories.items())},
        "geometric_epsilon_mm": float(eps),
        "contact_locality_tolerance_mm": float(contact_tol),
        "blocking_pairs": blocking_pairs,
    }


def semantic_intersection_counts(
    vertices,
    triangles,
    *,
    spacing_mm,
    geometric_epsilon_spacing_factor=1e-4,
    contact_locality_spacing_factor=1e-3,
):
    """Versión compacta sin lista de parejas, para compatibilidad."""
    info = semantic_intersection_analysis(
        vertices,
        triangles,
        spacing_mm=spacing_mm,
        geometric_epsilon_spacing_factor=(geometric_epsilon_spacing_factor),
        contact_locality_spacing_factor=(contact_locality_spacing_factor),
    )
    return {k: v for k, v in info.items() if k != "blocking_pairs"}


def semantic_info_without_pairs(info):
    """Prepara el diagnóstico semántico sin incluir el listado de pares."""
    return {k: v for k, v in info.items() if k != "blocking_pairs"}


def semantic_acceptable(info, baseline):
    return int(info["true_self_intersection_pairs"]) <= int(
        baseline["true_self_intersection_pairs"]
    ) and int(info["ambiguous_intersection_pairs"]) <= int(baseline["ambiguous_intersection_pairs"])


def build_triangle_vertex_adjacency(n_vertices, triangles):
    """Relaciona cada vértice con los triángulos que lo contienen."""
    adjacency = [set() for _ in range(int(n_vertices))]
    t = np.asarray(triangles, dtype=np.int64)

    for a, b, c in t:
        a = int(a)
        b = int(b)
        c = int(c)
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))

    return adjacency


def conflict_seed_vertices(triangles, blocking_pairs):
    """Reúne los vértices de las caras implicadas en conflictos semánticos."""
    t = np.asarray(triangles, dtype=np.int64)
    seeds = set()

    for item in blocking_pairs:
        fa = int(item["face_a"])
        fb = int(item["face_b"])

        if 0 <= fa < len(t):
            seeds.update(map(int, t[fa]))
        if 0 <= fb < len(t):
            seeds.update(map(int, t[fb]))

    return seeds


def build_local_rollback_weight(
    n_vertices,
    triangles,
    blocking_pairs,
    *,
    rings,
    ring_floor,
    coincident_groups=None,
):
    """Campo [0,1] de cuánto participa cada vértice en el rollback.

    1.0 = vértice directamente implicado.
    0.0 = conservar 100 % del candidato.
    """
    n_vertices = int(n_vertices)
    weight = np.zeros(n_vertices, dtype=np.float64)

    seeds = conflict_seed_vertices(
        triangles,
        blocking_pairs,
    )

    if not seeds:
        return weight

    for idx in seeds:
        if 0 <= idx < n_vertices:
            weight[idx] = 1.0

    adjacency = build_triangle_vertex_adjacency(
        n_vertices,
        triangles,
    )

    visited = set(seeds)
    frontier = set(seeds)

    rings = max(0, int(rings))
    floor = float(np.clip(ring_floor, 0.0, 1.0))

    for ring in range(1, rings + 1):
        nxt = set()
        for u in frontier:
            nxt.update(adjacency[u])
        nxt.difference_update(visited)

        if not nxt:
            break

        # Transición lineal desde 1 en la zona conflictiva hasta ring_floor.
        if rings <= 0:
            ring_weight = floor
        else:
            fraction = float(ring) / float(rings + 1)
            ring_weight = 1.0 - fraction * (1.0 - floor)

        ids = np.fromiter(
            sorted(nxt),
            dtype=np.int64,
        )
        weight[ids] = np.maximum(
            weight[ids],
            ring_weight,
        )

        visited.update(nxt)
        frontier = nxt

    # Coincident Lock: si un miembro del grupo necesita rollback,
    # todos reciben exactamente el mismo peso máximo.
    if coincident_groups:
        for group in coincident_groups:
            g = np.asarray(group, dtype=np.int64)
            if len(g) == 0:
                continue
            gw = float(np.max(weight[g]))
            if gw > 0.0:
                weight[g] = gw

    return weight


def apply_local_alpha_field(
    original_vertices,
    candidate_vertices,
    rollback_weight,
    local_beta,
):
    """Mezcla local.

    beta=1 -> candidato completo.
    beta=0 -> los seeds vuelven completamente al original, mientras que
              los anillos hacen una transición suave.
    """
    p0 = np.asarray(original_vertices, dtype=np.float64)
    ps = np.asarray(candidate_vertices, dtype=np.float64)
    w = np.asarray(rollback_weight, dtype=np.float64)

    beta = float(np.clip(local_beta, 0.0, 1.0))

    vertex_alpha = 1.0 - w * (1.0 - beta)
    vertex_alpha = np.clip(
        vertex_alpha,
        0.0,
        1.0,
    )

    out = p0 + vertex_alpha[:, None] * (ps - p0)

    return out, vertex_alpha


def old_global_semantic_guard_fallback(
    original_vertices,
    candidate_vertices,
    triangles,
    *,
    baseline,
    spacing_mm,
    geometric_epsilon_spacing_factor,
    contact_locality_spacing_factor,
    bisection_steps,
):
    """Fallback V1.4: solo se usa si el rollback local no logra resolver."""
    p0 = np.asarray(original_vertices, dtype=np.float64)
    ps = np.asarray(candidate_vertices, dtype=np.float64)

    def classify(points):
        return semantic_intersection_analysis(
            points,
            triangles,
            spacing_mm=spacing_mm,
            geometric_epsilon_spacing_factor=(geometric_epsilon_spacing_factor),
            contact_locality_spacing_factor=(contact_locality_spacing_factor),
        )

    low = 0.0
    high = 1.0
    best_alpha = 0.0
    best = p0.copy()
    best_info = classify(p0)

    for _ in range(max(1, int(bisection_steps))):
        alpha = 0.5 * (low + high)
        current = p0 + alpha * (ps - p0)
        info = classify(current)

        if semantic_acceptable(info, baseline):
            best_alpha = alpha
            best = current
            best_info = info
            low = alpha
        else:
            high = alpha

    final_info = classify(best)
    if not semantic_acceptable(final_info, baseline):
        best_alpha = 0.0
        best = p0.copy()
        final_info = classify(best)

    return best, {
        "used": True,
        "blend_alpha": float(best_alpha),
        "final": semantic_info_without_pairs(final_info),
    }


@operacion("Verificar seguridad topológica del pulido")
def apply_semantic_intersection_guard(
    original_vertices,
    candidate_vertices,
    triangles,
    *,
    spacing_mm,
    geometric_epsilon_spacing_factor,
    contact_locality_spacing_factor,
    bisection_steps,
    local_rollback_rings=2,
    local_rollback_max_expansions=4,
    local_rollback_ring_floor=0.25,
    coincident_groups=None,
):
    """V1.5: máxima corrección segura mediante rollback LOCAL.

    Fuera de las regiones conflictivas:
        alpha_v = 1.0

    Solo los vértices cercanos a pares bloqueantes reciben:
        alpha_v < 1.0
    """
    p0 = np.asarray(original_vertices, dtype=np.float64)
    ps = np.asarray(candidate_vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)

    def classify(points):
        return semantic_intersection_analysis(
            points,
            t,
            spacing_mm=spacing_mm,
            geometric_epsilon_spacing_factor=(geometric_epsilon_spacing_factor),
            contact_locality_spacing_factor=(contact_locality_spacing_factor),
        )

    baseline = classify(p0)
    full = classify(ps)

    if semantic_acceptable(full, baseline):
        alpha = np.ones(len(p0), dtype=np.float64)
        return ps, {
            "activated": False,
            "mode": "none",
            "baseline": semantic_info_without_pairs(baseline),
            "full_candidate": semantic_info_without_pairs(full),
            "final": semantic_info_without_pairs(full),
            "local_beta": 1.0,
            "affected_vertices": 0,
            "full_strength_vertices": int(len(p0)),
            "vertex_alpha": {
                "count": int(len(alpha)),
                "median": 1.0,
                "mean": 1.0,
                "p05": 1.0,
                "p10": 1.0,
                "min": 1.0,
            },
            "fallback_global": {
                "used": False,
            },
        }

    # V1.21: antes de ampliar anillos o mezclar globalmente, inmovilizar
    # únicamente caras conflictivas. Cada transición se vuelve a verificar.
    localized = ps.copy()
    frozen = np.zeros(len(p0), dtype=bool)
    local_records = []
    current_info = full
    baseline_pairs = {
        tuple(sorted((int(x["face_a"]), int(x["face_b"]))))
        for x in baseline.get("blocking_pairs", [])
    }
    for attempt in range(8):
        pairs = [
            x
            for x in current_info.get("blocking_pairs", [])
            if tuple(sorted((int(x["face_a"]), int(x["face_b"])))) not in baseline_pairs
        ]
        seeds = conflict_seed_vertices(t, pairs)
        bad_faces = unsafe_orientation_face_indices(p0, localized, t)
        if len(bad_faces):
            seeds.update(map(int, t[bad_faces].ravel()))
        if not seeds and semantic_acceptable(current_info, baseline):
            metrics = safety_metrics(p0, localized, t)
            # No aceptar por este camino una expansión dimensional superior
            # a la del candidato que ya pasó la guarda dimensional.
            allowed_extent = safety_metrics(p0, ps, t)["maximum_absolute_robust_extent_change"]
            if metrics["maximum_absolute_robust_extent_change"] <= allowed_extent + 1e-12:
                print(
                    f"[Paso 15] Pulido recuperado con bloqueo local: {np.count_nonzero(frozen):,} "
                    "vértices inmovilizados; se conserva el movimiento del resto.",
                    flush=True,
                )
                return localized, {
                    "activated": True,
                    "mode": "conflict_faces_only_verified",
                    "baseline": semantic_info_without_pairs(baseline),
                    "full_candidate": semantic_info_without_pairs(full),
                    "final": semantic_info_without_pairs(current_info),
                    "local_beta": 0.0,
                    "affected_vertices": int(np.count_nonzero(frozen)),
                    "full_strength_vertices": int(np.count_nonzero(~frozen)),
                    "fallback_global": {"used": False},
                    "local_conflict_iterations": local_records,
                }
            break
        if not seeds:
            break
        previous_count = int(np.count_nonzero(frozen))
        frozen[list(seeds)] = True
        # Bloqueo de coincidencia: un grupo completo comparte la decisión.
        for group in (coincident_groups if coincident_groups is not None else []):
            group = np.asarray(group, dtype=int)
            if np.any(frozen[group]):
                frozen[group] = True
        localized[frozen] = p0[frozen]
        local_records.append(
            {
                "iteration": attempt + 1,
                "frozen_vertices": int(np.count_nonzero(frozen)),
                "blocking_pairs": len(pairs),
                "orientation_faces": int(len(bad_faces)),
            }
        )
        if np.count_nonzero(frozen) == previous_count:
            break
        current_info = classify(localized)

    # Acumulamos todos los conflictos encontrados durante las expansiones.
    accumulated_pairs = list(full["blocking_pairs"])
    best_candidate = None
    best_info = None
    best_alpha_field = None
    best_beta = 0.0
    best_rings = int(local_rollback_rings)
    expansion_records = []

    for expansion in range(max(1, int(local_rollback_max_expansions))):
        rings = max(0, int(local_rollback_rings)) + expansion

        rollback_weight = build_local_rollback_weight(
            len(p0),
            t,
            accumulated_pairs,
            rings=rings,
            ring_floor=float(local_rollback_ring_floor),
            coincident_groups=coincident_groups,
        )

        affected = rollback_weight > 0.0

        # Primero comprobamos el extremo beta=0.
        zero_candidate, zero_alpha = apply_local_alpha_field(
            p0,
            ps,
            rollback_weight,
            0.0,
        )
        zero_info = classify(zero_candidate)

        expansion_record = {
            "expansion": int(expansion + 1),
            "rings": int(rings),
            "seed_blocking_pairs": int(len(accumulated_pairs)),
            "affected_vertices": int(np.count_nonzero(affected)),
            "beta_zero_result": semantic_info_without_pairs(zero_info),
        }

        if not semantic_acceptable(zero_info, baseline):
            # Algún conflicto apareció en el borde de la región.
            # Se incorpora a la siguiente expansión.
            accumulated_pairs.extend(zero_info["blocking_pairs"])
            # deduplicar por cara/categoría
            unique = {}
            for item in accumulated_pairs:
                key = (
                    int(item["face_a"]),
                    int(item["face_b"]),
                    str(item["category"]),
                )
                unique[key] = item
            accumulated_pairs = list(unique.values())

            expansion_record["resolved_at_beta_zero"] = False
            expansion_records.append(expansion_record)
            continue

        expansion_record["resolved_at_beta_zero"] = True

        # Buscar el beta local máximo que siga siendo seguro.
        low = 0.0
        high = 1.0

        local_best_beta = 0.0
        local_best_candidate = zero_candidate
        local_best_info = zero_info
        local_best_alpha = zero_alpha

        for _ in range(max(1, int(bisection_steps))):
            beta = 0.5 * (low + high)

            current, alpha_field = apply_local_alpha_field(
                p0,
                ps,
                rollback_weight,
                beta,
            )
            info = classify(current)

            if semantic_acceptable(info, baseline):
                local_best_beta = beta
                local_best_candidate = current
                local_best_info = info
                local_best_alpha = alpha_field
                low = beta
            else:
                high = beta

        # Comprobación final explícita.
        final_check = classify(local_best_candidate)
        if semantic_acceptable(final_check, baseline):
            best_candidate = local_best_candidate
            best_info = final_check
            best_alpha_field = local_best_alpha
            best_beta = float(local_best_beta)
            best_rings = int(rings)

            expansion_record["local_beta"] = float(local_best_beta)
            expansion_record["final_result"] = semantic_info_without_pairs(final_check)
            expansion_records.append(expansion_record)
            break

        accumulated_pairs.extend(final_check["blocking_pairs"])
        expansion_record["local_beta"] = float(local_best_beta)
        expansion_record["final_safe"] = False
        expansion_records.append(expansion_record)

    if best_candidate is not None:
        alpha = np.asarray(best_alpha_field, dtype=np.float64)

        return best_candidate, {
            "activated": True,
            "mode": "local_rollback",
            "baseline": semantic_info_without_pairs(baseline),
            "full_candidate": semantic_info_without_pairs(full),
            "final": semantic_info_without_pairs(best_info),
            "local_beta": float(best_beta),
            "rings_used": int(best_rings),
            "affected_vertices": int(np.count_nonzero(alpha < (1.0 - 1e-12))),
            "full_strength_vertices": int(np.count_nonzero(alpha >= (1.0 - 1e-12))),
            "vertex_alpha": {
                "count": int(len(alpha)),
                "median": float(np.median(alpha)),
                "mean": float(np.mean(alpha)),
                "p05": float(np.percentile(alpha, 5)),
                "p10": float(np.percentile(alpha, 10)),
                "min": float(np.min(alpha)),
            },
            "expansions": expansion_records,
            "fallback_global": {
                "used": False,
            },
            "interpretation": (
                "Solo las regiones cercanas a las intersecciones fueron "
                "mezcladas hacia la malla original. Los vértices fuera de "
                "esas regiones conservaron el candidato V1.4 completo."
            ),
        }

    # Último recurso: nunca publicamos una malla inválida.
    fallback_candidate, fallback = old_global_semantic_guard_fallback(
        p0,
        ps,
        t,
        baseline=baseline,
        spacing_mm=spacing_mm,
        geometric_epsilon_spacing_factor=(geometric_epsilon_spacing_factor),
        contact_locality_spacing_factor=(contact_locality_spacing_factor),
        bisection_steps=bisection_steps,
    )

    final_info = classify(fallback_candidate)

    return fallback_candidate, {
        "activated": True,
        "mode": "global_fallback",
        "baseline": semantic_info_without_pairs(baseline),
        "full_candidate": semantic_info_without_pairs(full),
        "final": semantic_info_without_pairs(final_info),
        "local_beta": None,
        "rings_used": None,
        "affected_vertices": int(len(p0)),
        "full_strength_vertices": 0,
        "vertex_alpha": None,
        "expansions": expansion_records,
        "fallback_global": fallback,
        "interpretation": (
            "El rollback local no consiguió una solución segura dentro del "
            "número máximo de expansiones. Se usó el fallback global para "
            "garantizar cero nuevas intersecciones."
        ),
    }


def optional_scalar(npz_handle, key, default=None):
    """Obtiene un escalar opcional del archivo de atributos."""
    if key not in npz_handle.files:
        return default
    try:
        value = float(np.asarray(npz_handle[key]).ravel()[0])
    except Exception:
        return default
    if not np.isfinite(value) or value <= 0:
        return default
    return value


def median_mesh_edge_length(vertices, triangles):
    t = np.asarray(triangles, dtype=np.int64)
    e = np.vstack(
        [
            t[:, [0, 1]],
            t[:, [1, 2]],
            t[:, [2, 0]],
        ]
    )
    e.sort(axis=1)
    e = np.unique(e, axis=0)
    d = np.linalg.norm(
        np.asarray(vertices)[e[:, 1]] - np.asarray(vertices)[e[:, 0]],
        axis=1,
    )
    d = d[np.isfinite(d) & (d > 1e-12)]
    return float(np.median(d)) if len(d) else 1.0


def cloud_to_mesh_distances(mesh, query_points):
    try:
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tmesh)
        q = o3d.core.Tensor(np.asarray(query_points, dtype=np.float32))
        return scene.compute_distance(q, nthreads=0).numpy().astype(float)
    except Exception:
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        d, _ = cKDTree(verts).query(
            np.asarray(query_points, dtype=np.float64),
            k=1,
            workers=query_threads(),
        )
        return np.asarray(d, dtype=np.float64)


def mesh_from_arrays(vertices, triangles, colors=None):
    """Construye una malla Open3D a partir de sus arrays geométricos."""
    require_open3d()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    if colors is not None and len(colors) == len(vertices):
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    mesh.compute_vertex_normals()
    return mesh


def topology_snapshot(mesh):
    result = {
        "vertices": int(len(np.asarray(mesh.vertices))),
        "triangles": int(len(np.asarray(mesh.triangles))),
    }
    checks = [
        ("edge_manifold_allow_boundary", lambda: mesh.is_edge_manifold(True)),
        ("edge_manifold_closed", lambda: mesh.is_edge_manifold(False)),
        ("vertex_manifold", mesh.is_vertex_manifold),
        ("orientable", mesh.is_orientable),
        ("watertight", mesh.is_watertight),
        ("self_intersecting_raw_open3d", mesh.is_self_intersecting),
    ]
    for key, fn in checks:
        try:
            result[key] = bool(fn())
        except Exception:
            result[key] = None
    return result


def save_mesh(path, mesh):
    if not o3d.io.write_triangle_mesh(
        str(path),
        mesh,
        write_ascii=False,
    ):
        raise RuntimeError(f"No se pudo guardar {path}")


def make_preview(
    path,
    before,
    after,
    displacement,
    structure,
    max_vertices,
):
    if plt is None:
        return False

    n = len(before)
    if n > int(max_vertices):
        idx = np.linspace(
            0,
            n - 1,
            int(max_vertices),
        ).astype(int)
    else:
        idx = np.arange(n)

    B = before[idx]
    A = after[idx]
    D = displacement[idx]

    fig = plt.figure(figsize=(15, 10))

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.scatter(B[:, 0], B[:, 1], B[:, 2], s=0.25)
    ax.set_title("Antes paso 15")

    ax = fig.add_subplot(2, 2, 2, projection="3d")
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], s=0.25)
    ax.set_title("Después paso 15")

    ax = fig.add_subplot(2, 2, 3)
    sc = ax.scatter(
        A[:, 0],
        A[:, 2],
        c=D,
        s=0.4,
        cmap="viridis",
    )
    ax.set_title("Desplazamiento X-Z")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="mm")

    ax = fig.add_subplot(2, 2, 4)
    ax.axis("off")
    ds = stats(displacement)
    ax.text(
        0.02,
        0.98,
        (
            "Paso 15\n\n"
            f"Vértices: {len(before):,}\n"
            f"Boundary protegidos: "
            f"{np.count_nonzero(structure['boundary_vertex']):,}\n"
            f"Corona boundary: "
            f"{np.count_nonzero(structure['boundary_ring_vertex']):,}\n"
            f"Shift mediano: {ds['median']:.4f} mm\n"
            f"P90: {ds['p90']:.4f} mm\n"
            f"P95: {ds['p95']:.4f} mm\n"
            f"Máx: {ds['max']:.4f} mm"
        ),
        va="top",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _worker_multiscale_normal_trend_pass(task):
    begin, end = task
    ctx = contexto()
    args = ctx["args"]
    chunk_size = ctx["chunk_size"]
    k = ctx["k"]
    large_count = ctx["large_count"]
    large_prediction = ctx["large_prediction"]
    large_radius = ctx["large_radius"]
    large_scale = ctx["large_scale"]
    normal_cos = ctx["normal_cos"]
    p = ctx["p"]
    small_count = ctx["small_count"]
    small_prediction = ctx["small_prediction"]
    small_radius = ctx["small_radius"]
    small_scale = ctx["small_scale"]
    spacing_mm = ctx["spacing_mm"]
    t1 = ctx["t1"]
    t2 = ctx["t2"]
    tree = ctx["tree"]
    vn = ctx["vn"]
    chunk_start = begin
    chunk_end = min(len(p), chunk_start + chunk_size)
    query = p[chunk_start:chunk_end]
    distances, indices = tree.query(query, k=k, distance_upper_bound=large_radius, workers=1)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    safe_indices = np.clip(indices, 0, max(len(p) - 1, 0))
    neighbor_normals = vn[safe_indices]
    center_normals = vn[chunk_start:chunk_end]
    dot = np.sum(neighbor_normals * center_normals[:, None, :], axis=2)
    valid = (
        np.isfinite(distances)
        & (indices >= 0)
        & (indices < len(p))
        & (distances > 1e-10)
        & (dot >= normal_cos)
    )
    q_all = p[safe_indices]
    for local_i in range(chunk_end - chunk_start):
        i = chunk_start + local_i
        base = valid[local_i]
        large_mask = base & (distances[local_i] <= large_radius)
        small_mask = base & (distances[local_i] <= small_radius)
        small_count[i] = int(np.count_nonzero(small_mask))
        large_count[i] = int(np.count_nonzero(large_mask))
        if small_count[i] < int(args.normal_trend_min_small_neighbors) or large_count[i] < int(
            args.normal_trend_min_large_neighbors
        ):
            continue
        for mask, radius, pred_arr, scale_arr in (
            (small_mask, small_radius, small_prediction, small_scale),
            (large_mask, large_radius, large_prediction, large_scale),
        ):
            q = q_all[local_i, mask]
            d = q - p[i][None, :]
            u = d @ t1[i]
            v = d @ t2[i]
            w = d @ vn[i]
            dd = distances[local_i, mask]
            coef, residual_scale = solve_weighted_quadratic_irls(
                u,
                v,
                w,
                dd,
                radius_mm=radius,
                irls_iterations=int(args.normal_trend_irls_iterations),
                huber_k=float(args.normal_trend_huber_k),
                spacing_mm=float(spacing_mm),
            )
            if coef is None:
                continue
            pred_arr[i] = float(coef[5])
            scale_arr[i] = float(residual_scale)
    return (
        begin,
        end,
        {
            "small_prediction": small_prediction[begin:end],
            "large_prediction": large_prediction[begin:end],
            "small_scale": small_scale[begin:end],
            "large_scale": large_scale[begin:end],
            "small_count": small_count[begin:end],
            "large_count": large_count[begin:end],
        },
    )


def _worker_compute_local_quadratic_residuals(task):
    begin, end = task
    ctx = contexto()
    adjacency = ctx["adjacency"]
    huber_k = ctx["huber_k"]
    irls_iterations = ctx["irls_iterations"]
    local_scale = ctx["local_scale"]
    min_neighbors = ctx["min_neighbors"]
    neighbor_counts = ctx["neighbor_counts"]
    neighborhoods = ctx["neighborhoods"]
    normals = ctx["normals"]
    p = ctx["p"]
    residuals = ctx["residuals"]
    rings = ctx["rings"]
    for i in range(begin, end):
        neigh = ring_neighbors(adjacency, i, rings)
        neighborhoods[i] = neigh
        neighbor_counts[i] = int(len(neigh))
        if len(neigh) < int(min_neighbors):
            continue
        t1, t2, n = tangent_basis_from_normal(normals[i])
        q = p[neigh]
        origin = np.median(q, axis=0)
        dq = q - origin[None, :]
        u = dq @ t1
        v = dq @ t2
        w = dq @ n
        try:
            coef = fit_quadratic_surface_irls(
                u, v, w, irls_iterations=irls_iterations, huber_k=huber_k
            )
        except Exception:
            continue
        dc = p[i] - origin
        uc = float(np.dot(dc, t1))
        vc = float(np.dot(dc, t2))
        wc = float(np.dot(dc, n))
        predicted = float(quadratic_predict(coef, uc, vc))
        residuals[i] = wc - predicted
        neigh_pred = quadratic_predict(coef, u, v)
        neigh_res = w - neigh_pred
        scale, _ = robust_scale_mad(neigh_res, floor=1e-08)
        local_scale[i] = scale
    return (
        begin,
        end,
        {
            "residuals": residuals[begin:end],
            "local_scale": local_scale[begin:end],
            "neighbor_counts": neighbor_counts[begin:end],
            "neighborhoods": neighborhoods[begin:end],
        },
    )


def _worker_surface_fairing_pass(task):
    begin, end = task
    ctx = contexto()
    angle_cos = ctx["angle_cos"]
    args = ctx["args"]
    compatible_count = ctx["compatible_count"]
    local_scale_mm = ctx["local_scale_mm"]
    min_neighbors = ctx["min_neighbors"]
    neighborhoods = ctx["neighborhoods"]
    normals = ctx["normals"]
    p = ctx["p"]
    spacing_mm = ctx["spacing_mm"]
    structure = ctx["structure"]
    t1 = ctx["t1"]
    t2 = ctx["t2"]
    target_mm = ctx["target_mm"]
    for i in range(begin, end):
        neigh_all = neighborhoods[i]
        if (
            structure["boundary_vertex"][i]
            or structure["boundary_ring_vertex"][i]
            or structure["vertex_strength"][i] < float(args.surface_fairing_feature_strength_min)
        ):
            continue
        neigh = np.asarray(neigh_all, dtype=np.int64)
        if len(neigh) < min_neighbors:
            continue
        normal_dot = np.sum(normals[neigh] * normals[i][None, :], axis=1)
        compatible = normal_dot >= angle_cos
        neigh = neigh[compatible]
        compatible_count[i] = int(len(neigh))
        if len(neigh) < min_neighbors:
            continue
        delta = p[neigh] - p[i][None, :]
        u = delta @ t1[i]
        v = delta @ t2[i]
        w = delta @ normals[i]
        distance = np.linalg.norm(delta, axis=1)
        positive_distance = distance[np.isfinite(distance) & (distance > 1e-10)]
        if len(positive_distance) < min_neighbors:
            continue
        radius_mm = max(float(np.percentile(positive_distance, 90)), 1.5 * float(spacing_mm))
        coef, residual_scale = solve_weighted_quadratic_irls(
            u,
            v,
            w,
            distance,
            radius_mm=radius_mm,
            irls_iterations=int(args.surface_fairing_irls_iterations),
            huber_k=float(args.surface_fairing_huber_k),
            spacing_mm=float(spacing_mm),
        )
        if coef is None or residual_scale is None:
            continue
        target_mm[i] = float(coef[5])
        local_scale_mm[i] = float(residual_scale)
    return (
        begin,
        end,
        {
            "target_mm": target_mm[begin:end],
            "local_scale_mm": local_scale_mm[begin:end],
            "compatible_count": compatible_count[begin:end],
        },
    )


def _worker_residual_finish(task):
    """Evalúa solo candidatos; ajustes independientes con contexto por proceso."""
    begin, end = task
    c = contexto()
    p, vn, ids, h = c["p"], c["normals"], c["ids"], c["spacing"]
    target = np.zeros(end - begin)
    uncertainty = np.full(end - begin, np.nan)
    for row in range(begin, end):
        i = int(ids[row])
        t1, t2, n = tangent_basis_from_normal(vn[i])
        predictions, scales = [], []
        for rings, minimum in ((3, 12), (5, 24)):
            neigh = ring_neighbors(c["adjacency"], i, rings)
            if len(neigh) < minimum:
                break
            offset = p[neigh] - p[i]
            length = np.linalg.norm(offset, axis=1)
            compatible = vn[neigh] @ n >= math.cos(math.radians(35))
            keep = compatible & (length > 1e-10) & (length <= 5 * h)
            if np.count_nonzero(keep) < minimum or np.mean(compatible) < 0.75:
                break
            offset, length = offset[keep], length[keep]
            # Limitar el coste sin seleccionar vecinos de otra componente.
            if len(length) > 160:
                order = np.argsort(length, kind="stable")[:160]
                offset, length = offset[order], length[order]
            uv = np.column_stack((offset @ t1, offset @ t2)) / h
            angle = np.sort(np.arctan2(uv[:, 1], uv[:, 0]))
            gaps = np.diff(np.r_[angle, angle[0] + 2 * np.pi])
            if np.max(gaps) >= np.deg2rad(160):
                break  # No extrapolar donde falta entorno a un lado.
            eig = np.linalg.eigvalsh(uv.T @ uv / len(uv))
            if eig[0] <= 0 or eig[0] / max(eig[1], 1e-12) < 0.12:
                break
            coef, scale = solve_weighted_quadratic_irls(
                uv[:, 0],
                uv[:, 1],
                (offset @ n) / h,
                length / h,
                radius_mm=max(float(np.percentile(length, 90)) / h, 1.0),
                irls_iterations=3,
                huber_k=1.35,
                spacing_mm=1.0,
            )
            if coef is None or not np.all(np.isfinite(coef)) or not np.isfinite(scale):
                break
            predictions.append(float(coef[5]) * h)
            scales.append(float(scale) * h)
        if len(predictions) != 2:
            continue
        a, b = predictions
        noise = max(scales)
        amplitude = min(abs(a), abs(b))
        # Ambas escalas deben confirmar una desviación pequeña del mismo signo.
        # El umbral es de muestreo, no una incertidumbre estereoscópica calibrada.
        threshold = max(0.018 * h, 0.8 * noise)
        if a * b <= 0 or amplitude <= threshold or max(abs(a), abs(b)) > 0.85 * h:
            continue
        if abs(a - b) > max(0.045 * h, 0.5 * amplitude):
            continue  # Relieve dependiente de escala: conservarlo.
        gate = float(smoothstep01((amplitude - threshold) / max(0.08 * h, threshold)))
        target[row - begin] = np.sign(a) * amplitude * gate
        uncertainty[row - begin] = noise
    return begin, end, {"target": target, "uncertainty": uncertainty}


def _finish_neighbor_average(values, edges):
    """Promedio por adyacencia, sin mezclar capas por distancia euclídea."""
    a, b = edges[:, 0], edges[:, 1]
    out = np.zeros_like(values, dtype=np.float64)
    count = np.zeros(len(values), dtype=np.float64)
    np.add.at(out, a, values[b])
    np.add.at(out, b, values[a])
    np.add.at(count, a, 1)
    np.add.at(count, b, 1)
    if values.ndim == 2:
        count = count[:, None]
    return out / np.maximum(count, 1)


def _finish_surface_distances(vertices, triangles, points):
    """Distancia a triángulos; sin cambiar a distancia a vértices silenciosamente."""
    mesh = mesh_from_arrays(vertices, triangles)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    q = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
    return scene.compute_distance(q, nthreads=query_threads()).numpy().astype(float)


@operacion("Acabado residual localizado sobre el pulido ya validado")
def localized_residual_finish(
    vertices,
    original_vertices,
    triangles,
    structure,
    cloud_points,
    cloud_confidence,
    cloud_support,
    coincident_groups,
    spacing_mm,
    semantic_spacing,
    args,
):
    """Pasada adicional reversible; nunca rehace el suavizado ni cambia caras.

    Se mantiene como respaldo la salida del pulido anterior, ya protegida por sus
    guardas. Si esta pasada no es verificable o se rechaza, se devuelve ese respaldo.
    Los ajustes se calculan en paralelo solo para los candidatos. Se conservan
    fronteras, aristas, costuras coincidentes y puntos de observación confiables.
    """
    base = np.asarray(vertices, dtype=np.float64).copy()
    t = np.asarray(triangles, dtype=np.int64)
    h = float(spacing_mm)
    info = {
        "enabled": bool(args.residual_finish),
        "applied": False,
        "method": "sparse_two_scale_quadratic_residual_finish_local_fidelity_rollback",
        "connectivity_modified": False,
        "shape_specific_assumptions": False,
        "baseline": "optimized_polish_after_existing_guards",
        "cycles": [],
    }

    def skip(reason):
        info["reason"] = reason
        return base.copy(), info

    if not args.residual_finish:
        return skip("disabled")
    if not np.isfinite(h) or h <= 0:
        return skip("invalid_spacing")
    if len(base) < 40 or len(t) == 0:
        return skip("insufficient_geometry")
    limit = float(args.residual_finish_max_shift_spacing) * h
    info["maximum_additional_shift_mm"] = limit
    # El contorno recuperado en 13 y su corona quedan exactamente fijos.
    protected = structure["boundary_vertex"].copy() | structure["boundary_ring_vertex"]
    protected |= structure["vertex_strength"] < 0.75
    for g in coincident_groups:
        protected[np.asarray(g, dtype=int)] = True
    cloud = np.asarray(cloud_points, dtype=np.float64)
    valid = np.all(np.isfinite(cloud), axis=1)
    cloud = cloud[valid]
    if not len(cloud):
        return skip("no_valid_cloud_evidence")
    confidence = np.asarray(cloud_confidence, dtype=float)[valid]
    support = np.asarray(cloud_support, dtype=float)[valid]
    nearest_d, nearest_i = cKDTree(cloud).query(base, workers=query_threads())
    reliable = (
        np.isfinite(confidence) & np.isfinite(support) & (confidence >= 0.85) & (support >= 3)
    )
    protected |= (nearest_d <= 0.2 * h) & reliable[nearest_i]
    info["protected_vertices"] = int(protected.sum())
    edges = structure["edges"]
    traversable = (~structure["boundary_edge"]) & (
        structure["dihedral_deg"] < float(args.feature_soft_angle_deg)
    )
    traversable &= np.linalg.norm(base[edges[:, 1]] - base[edges[:, 0]], axis=1) <= 3 * h
    seam_mask = structure.get("terminal_seam_vertex", np.zeros(len(base), dtype=bool))
    traversable &= ~(seam_mask[edges[:, 0]] | seam_mask[edges[:, 1]])
    fit_edges = edges[traversable]
    if not len(fit_edges):
        return skip("no_smooth_neighborhoods")
    adjacency = build_vertex_adjacency(len(base), fit_edges)
    p = base.copy()
    for cycle in range(int(args.residual_finish_cycles)):
        vn = vertex_normals(p, t)
        local_mean = _finish_neighbor_average(p, fit_edges)
        curvature = np.einsum("ij,ij->i", local_mean - p, vn)
        high_frequency = np.abs(curvature - _finish_neighbor_average(curvature, fit_edges))
        eligible = (~protected) & (high_frequency > 0.012 * h)
        # Incluir la vecindad inmediata reduce saltos bruscos dentro del parche.
        nearby = np.zeros(len(p), bool)
        active_edges = eligible[fit_edges[:, 0]] | eligible[fit_edges[:, 1]]
        nearby[fit_edges[active_edges].ravel()] = True
        ids = np.flatnonzero(nearby & (~protected))
        if not len(ids):
            break
        target = np.zeros(len(ids))
        uncertainty = np.full(len(ids), np.nan)
        ejecutar_bloques(
            _worker_residual_finish,
            {"p": p, "normals": vn, "ids": ids, "spacing": h, "adjacency": adjacency},
            {"target": target, "uncertainty": uncertainty},
            len(ids),
            256,
            f"Acabado residual: ciclo {cycle+1}/{args.residual_finish_cycles}",
        )
        scalar = np.zeros(len(p))
        scalar[ids] = float(args.residual_finish_strength) * target
        # Solo amortiguar la transición entre candidatos ya confirmados.
        neighbor_scalar = _finish_neighbor_average(scalar, fit_edges)
        selected = scalar != 0
        scalar[selected] = 0.8 * scalar[selected] + 0.2 * neighbor_scalar[selected]
        scalar = np.clip(scalar, -0.6 * limit, 0.6 * limit)
        candidate = p + scalar[:, None] * vn
        delta = candidate - base
        length = np.linalg.norm(delta, axis=1)
        delta *= np.minimum(1.0, limit / np.maximum(length, 1e-15))[:, None]
        candidate = base + delta
        candidate[protected] = base[protected]
        info["cycles"].append(
            {
                "cycle": cycle + 1,
                "examined_vertices": int(len(ids)),
                "confirmed_vertices": int(selected.sum()),
                "candidate_shift_mm": stats(np.linalg.norm(candidate - p, axis=1)),
            }
        )
        p = candidate
        if not selected.any():
            break
    delta = p - base
    if not np.any(np.linalg.norm(delta, axis=1) > 1e-10):
        return skip("no_confirmed_residual")
    # Respetar también el presupuesto original de desplazamiento del paso 15.
    original_limit = (
        max(
            float(args.max_total_shift_spacing), float(args.surface_fairing_max_total_shift_spacing)
        )
        * h
    )
    from_original = base - np.asarray(original_vertices, dtype=float)
    previous_length = np.linalg.norm(from_original, axis=1)
    budget = np.maximum(previous_length, original_limit)
    aa = np.einsum("ij,ij->i", delta, delta)
    bb = np.einsum("ij,ij->i", from_original, delta)
    cc = np.einsum("ij,ij->i", from_original, from_original) - budget * budget
    bound = np.ones(len(base))
    moving = aa > 1e-20
    bound[moving] = np.clip(
        (-bb[moving] + np.sqrt(np.maximum(bb[moving] ** 2 - aa[moving] * cc[moving], 0)))
        / aa[moving],
        0,
        1,
    )
    delta *= bound[:, None]
    info["original_displacement_budget_mm"] = original_limit
    info["vertices_limited_by_original_budget"] = int(np.count_nonzero(bound < 1))
    if not np.any(np.linalg.norm(delta, axis=1) > 1e-10):
        return skip("original_displacement_budget_exhausted")

    # Guardas de esta pasada referidas a la malla ya pulida: un rechazo no vuelve
    # a la malla rugosa de 14. Las comprobaciones semánticas usan la misma escala.
    def analyze(q):
        return semantic_intersection_analysis(
            q,
            t,
            spacing_mm=semantic_spacing,
            geometric_epsilon_spacing_factor=args.semantic_geometric_epsilon_spacing_factor,
            contact_locality_spacing_factor=args.semantic_contact_locality_spacing_factor,
        )

    def blocking_keys(report):
        return {
            (min(x["face_a"], x["face_b"]), max(x["face_a"], x["face_b"]), x["category"])
            for x in report["blocking_pairs"]
        }

    try:
        baseline_semantic = analyze(base)
        existing = blocking_keys(baseline_semantic)
        initial_bad = set(unsafe_orientation_face_indices(base, base, t).tolist())
        initial_bad_source = set(
            unsafe_orientation_face_indices(original_vertices, base, t).tolist()
        )
        alpha = np.ones(len(base))
        protected_faces = np.asarray(sorted(initial_bad | initial_bad_source), dtype=int)
        if len(protected_faces):
            alpha[np.unique(t[protected_faces])] = 0
        info["local_guard"] = []
        accepted = None
        for attempt in range(4):
            for _ in range(18):
                trial = base + alpha[:, None] * delta
                bad = set(unsafe_orientation_face_indices(base, trial, t).tolist()) - initial_bad
                bad |= (
                    set(unsafe_orientation_face_indices(original_vertices, trial, t).tolist())
                    - initial_bad_source
                )
                if not bad:
                    break
                affected = np.unique(t[np.asarray(sorted(bad), dtype=int)])
                alpha[affected] *= 0.5
                alpha[alpha < 2**-16] = 0.0
            else:
                return skip("orientation_guard_kept_previous_polish")
            current = analyze(trial)
            new_keys = blocking_keys(current) - existing
            info["local_guard"].append(
                {
                    "attempt": attempt + 1,
                    "new_blocking_pairs": len(new_keys),
                    "reduced_vertices": int(np.count_nonzero(alpha < 1)),
                }
            )
            if not new_keys:
                accepted = trial
                info["final_intersections"] = semantic_info_without_pairs(current)
                break
            conflict_faces = np.unique([f for a, b, _ in new_keys for f in (a, b)])
            affected = np.zeros(len(base), bool)
            affected[np.unique(t[conflict_faces])] = True
            links = affected[edges[:, 0]] | affected[edges[:, 1]]
            affected[edges[links].ravel()] = True
            alpha[affected] = 0.0
        if accepted is None:
            return skip("intersection_guard_kept_previous_polish")
        geom = safety_metrics(base, accepted, t)
        source_geom = safety_metrics(original_vertices, accepted, t)
        if geom["maximum_absolute_robust_extent_change"] > 0.005 or source_geom[
            "maximum_absolute_robust_extent_change"
        ] > float(args.max_robust_extent_change_ratio):
            return skip("extent_guard_kept_previous_polish")
        # Mismo subconjunto determinista para comparar fidelidad: no usar el P90
        # mesh->cloud para eliminar interpolaciones o cambiar los gates del 17.
        qidx = np.arange(len(cloud), dtype=int)
        if len(qidx) > 12000:
            qidx = np.linspace(0, len(cloud) - 1, 12000, dtype=int)
        # Conserva los mismos umbrales: el rechazo se aplica a la región que
        # causó el deterioro, no automáticamente a TODO el acabado residual.
        before = _finish_surface_distances(base, t, cloud[qidx])
        if not np.all(np.isfinite(before)):
            return skip("nonfinite_fidelity_measurement")
        strong = reliable[qidx]
        info["local_fidelity_guard"] = []
        fidelity_delta = accepted - base
        fidelity_alpha = np.ones(len(base), dtype=float)
        baseline_scene = o3d.t.geometry.RaycastingScene()
        baseline_scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(mesh_from_arrays(base, t))
        )
        query_points = o3d.core.Tensor(np.asarray(cloud[qidx], dtype=np.float32))
        baseline_faces = (
            baseline_scene.compute_closest_points(query_points, nthreads=query_threads())[
                "primitive_ids"
            ]
            .numpy()
            .astype(np.int64)
        )
        fidelity_passed = False
        for fidelity_attempt in range(8):
            trial = base + fidelity_alpha[:, None] * fidelity_delta
            # Un retroceso parcial también puede cambiar la geometría de la
            # transición. Se verifican de nuevo orientación e intersecciones.
            bad = set(unsafe_orientation_face_indices(base, trial, t).tolist()) - initial_bad
            bad |= (
                set(unsafe_orientation_face_indices(original_vertices, trial, t).tolist())
                - initial_bad_source
            )
            if bad:
                affected = np.unique(t[np.asarray(sorted(bad), dtype=int)])
                fidelity_alpha[affected] = 0.0
                info["local_fidelity_guard"].append(
                    {
                        "attempt": fidelity_attempt + 1,
                        "reason": "local_orientation_rollback",
                        "faces": len(bad),
                    }
                )
                continue
            trial_semantic = analyze(trial)
            new_keys = blocking_keys(trial_semantic) - existing
            if new_keys:
                bad_faces = np.unique([f for a, b, _ in new_keys for f in (a, b)])
                affected = np.zeros(len(base), dtype=bool)
                affected[np.unique(t[bad_faces])] = True
                links = affected[edges[:, 0]] | affected[edges[:, 1]]
                affected[edges[links].ravel()] = True
                fidelity_alpha[affected] = 0.0
                info["local_fidelity_guard"].append(
                    {
                        "attempt": fidelity_attempt + 1,
                        "reason": "local_intersection_rollback",
                        "pairs": len(new_keys),
                    }
                )
                continue
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_from_arrays(trial, t)))
            closest = scene.compute_closest_points(query_points, nthreads=query_threads())
            nearest = closest["points"].numpy().astype(float)
            after = np.linalg.norm(nearest - cloud[qidx], axis=1)
            if not np.all(np.isfinite(after)):
                return skip("nonfinite_fidelity_measurement")
            increase = after - before
            p95_bad = np.percentile(increase, 95) > 0.08 * h
            p90_bad = np.percentile(after, 90) > np.percentile(before, 90) + 0.05 * h
            offending = strong & (increase > 0.15 * h)
            if p95_bad:
                offending |= increase > 0.08 * h
            if p90_bad:
                offending |= (increase > 0) & (after > np.percentile(before, 90) + 0.05 * h)
            info["cloud_fidelity"] = {
                "before_mm": stats(before),
                "after_mm": stats(after),
                "distance_increase_mm": stats(increase),
            }
            info["local_fidelity_guard"].append(
                {
                    "attempt": fidelity_attempt + 1,
                    "offending_observations": int(offending.sum()),
                    "rolled_back_vertices": int(np.count_nonzero(fidelity_alpha < 1)),
                    "p95_increase_exceeded": bool(p95_bad),
                    "p90_distance_exceeded": bool(p90_bad),
                }
            )
            if not offending.any() and not p95_bad and not p90_bad:
                accepted = trial
                info["final_intersections"] = semantic_info_without_pairs(trial_semantic)
                fidelity_passed = True
                break
            # Proteger las caras próximas tanto ANTES como DESPUÉS del movimiento;
            # así no se pierde la atribución si cambió el triángulo más cercano.
            candidate_faces = closest["primitive_ids"].numpy().astype(np.int64)
            face_ids = np.unique(np.r_[baseline_faces[offending], candidate_faces[offending]])
            face_ids = face_ids[(face_ids >= 0) & (face_ids < len(t))]
            affected = np.zeros(len(base), dtype=bool)
            affected[np.unique(t[face_ids])] = True
            links = affected[edges[:, 0]] | affected[edges[:, 1]]
            affected[edges[links].ravel()] = True
            if not np.any(
                affected & (fidelity_alpha > 0) & (np.linalg.norm(fidelity_delta, axis=1) > 1e-12)
            ):
                return skip("local_fidelity_guard_no_remaining_safe_adjustment")
            fidelity_alpha[affected] = 0.0
            print(
                f"[Paso 15] Fidelidad local: se protege la región de {int(offending.sum())} observaciones; "
                f"{int(np.count_nonzero(fidelity_alpha == 0))} vértices vuelven al pulido previo.",
                flush=True,
            )
        if not fidelity_passed:
            return skip("local_fidelity_iteration_limit_kept_previous_polish")
        # El retroceso local tampoco puede cambiar las dimensiones permitidas.
        geom = safety_metrics(base, accepted, t)
        source_geom = safety_metrics(original_vertices, accepted, t)
        if geom["maximum_absolute_robust_extent_change"] > 0.005 or source_geom[
            "maximum_absolute_robust_extent_change"
        ] > float(args.max_robust_extent_change_ratio):
            return skip("extent_guard_after_local_fidelity_rollback")
    except Exception as exc:
        print(
            f"[ADVERTENCIA] Acabado residual omitido: {type(exc).__name__}: {exc}. Se conserva el pulido previo.",
            flush=True,
        )
        info["verification_error"] = f"{type(exc).__name__}: {exc}"
        return skip("guard_verification_unavailable")
    moved = np.linalg.norm(accepted - base, axis=1)
    info.update(
        applied=bool(np.any(moved > 1e-10)),
        moved_vertices=int(np.count_nonzero(moved > 1e-10)),
        actual_additional_shift_mm=stats(moved),
        reason="accepted",
        boundary_maximum_shift_mm=(
            float(np.max(moved[structure["boundary_vertex"]]))
            if np.any(structure["boundary_vertex"])
            else 0.0
        ),
    )
    print(
        f"[Paso 15] Acabado residual: {info['moved_vertices']} vértices; máximo adicional {moved.max():.4f} mm.",
        flush=True,
    )
    return accepted, info


def main():
    """Pule la malla con guardas geométricas y exporta desplazamientos y diagnóstico."""
    cli = parser()
    args = cli.parse_args()
    if not np.isfinite(args.residual_finish_strength) or not 0 < args.residual_finish_strength <= 1:
        cli.error("--residual-finish-strength debe ser un número entre 0 (excluido) y 1.")
    if (
        not np.isfinite(args.residual_finish_max_shift_spacing)
        or not 0 < args.residual_finish_max_shift_spacing <= 0.35
    ):
        cli.error("--residual-finish-max-shift-spacing debe estar entre 0 (excluido) y 0.35.")

    identity = script_identity()

    print("\n[paso 15 IDENTIDAD]")
    print("Versión:", identity["version"])
    print("Nombre:", identity["name"])
    print("Archivo ejecutado:", identity["script_path"])
    print("SHA256:", identity["script_sha256"])
    print(
        "NORMAL TREND:",
        "ACTIVADO" if bool(args.normal_trend_enabled) else "DESACTIVADO",
    )
    print(
        "SURFACE FAIRING:",
        "ACTIVADO" if bool(args.surface_fairing_enabled) else "DESACTIVADO",
    )
    print("[/paso 15 IDENTIDAD]\n")

    require_open3d()

    root = Path(args.root).expanduser().resolve()
    obj = args.object.strip()

    multi = root / "reconstruccion" / "multisesion"

    mesh_dir = multi / args.mesh_source
    cloud_dir = multi / args.cloud_source
    topology_dir = multi / args.topology_source

    mesh_path = mesh_dir / "malla_final_topologica.ply"
    cloud_path = cloud_dir / "nube_regularizada_general.npz"
    topology_summary_path = topology_dir / "resumen_14_limpieza_topologica.json"

    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    if not cloud_path.is_file():
        raise FileNotFoundError(cloud_path)
    if not topology_summary_path.is_file():
        raise FileNotFoundError(topology_summary_path)

    source_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(source_mesh.triangles) == 0:
        raise RuntimeError("La malla topológica de Paso 14 está vacía.")

    vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
    triangles = np.asarray(source_mesh.triangles, dtype=np.int64)
    colors = (
        np.asarray(source_mesh.vertex_colors, dtype=np.float64)
        if source_mesh.has_vertex_colors()
        else None
    )

    if len(vertices) == 0 or len(triangles) == 0 or not np.all(np.isfinite(vertices)):
        raise RuntimeError("Malla de entrada inválida.")

    with np.load(cloud_path) as d:
        cloud_points_all = np.asarray(d["points"], dtype=np.float64)
        cloud_normals_all = np.asarray(d["normals"], dtype=np.float64) if "normals" in d else None
        if cloud_normals_all is not None and cloud_normals_all.shape != cloud_points_all.shape:
            raise ValueError("Normales de nube incompatibles con points.")
        cloud_confidence_raw = (
            np.asarray(d["confidence"], dtype=np.float64).reshape(-1)
            if "confidence" in d
            else np.zeros(len(cloud_points_all))
        )
        cloud_support_raw = (
            np.asarray(d["support_views"], dtype=np.float64).reshape(-1)
            if "support_views" in d
            else np.zeros(len(cloud_points_all))
        )
        evidence_contract_available = "surface_evidence_class" in d.files
        evidence_class = np.asarray(
            (
                d["surface_evidence_class"]
                if evidence_contract_available
                else np.full(len(cloud_points_all), 2)
            ),
            dtype=np.uint8,
        ).reshape(-1)
        independent_support = np.asarray(
            (
                d["independent_support_poses"]
                if "independent_support_poses" in d.files
                else cloud_support_raw
            ),
            dtype=np.float64,
        ).reshape(-1)
        angular_span = np.asarray(
            (
                d["support_angular_span_poses"]
                if "support_angular_span_poses" in d.files
                else np.zeros(len(cloud_points_all))
            ),
            dtype=np.int16,
        ).reshape(-1)
        conflict_ratio = np.asarray(
            (
                d["conflict_pose_ratio"]
                if "conflict_pose_ratio" in d.files
                else np.zeros(len(cloud_points_all))
            ),
            dtype=np.float64,
        ).reshape(-1)
        heldout_available = np.asarray(
            (
                d["heldout_validation_available"]
                if "heldout_validation_available" in d.files
                else np.zeros(len(cloud_points_all))
            ),
            dtype=bool,
        ).reshape(-1)
        heldout_pass = np.asarray(
            (
                d["heldout_validation_pass_ratio"]
                if "heldout_validation_pass_ratio" in d.files
                else np.ones(len(cloud_points_all))
            ),
            dtype=np.float64,
        ).reshape(-1)
        heldout_tests = np.asarray(
            (
                d["heldout_validation_tests"]
                if "heldout_validation_tests" in d.files
                else np.zeros(len(cloud_points_all))
            ),
            dtype=np.int16,
        ).reshape(-1)
        heldout_pass_count = np.asarray(
            (
                d["heldout_validation_pass_count"]
                if "heldout_validation_pass_count" in d.files
                else np.rint(heldout_pass * np.maximum(heldout_tests, 0))
            ),
            dtype=np.int16,
        ).reshape(-1)
        evidence_contract_accept = np.asarray(
            (
                d["evidence_contract_accept"]
                if "evidence_contract_accept" in d.files
                else np.ones(len(cloud_points_all), dtype=np.uint8)
            ),
            dtype=bool,
        ).reshape(-1)
        evidence_contract_accept_published = "evidence_contract_accept" in d.files
        evidence_strength = np.asarray(
            (
                d["evidence_strength"]
                if "evidence_strength" in d.files
                else np.ones(len(cloud_points_all))
            ),
            dtype=np.float64,
        ).reshape(-1)
        if cloud_points_all.ndim != 2 or cloud_points_all.shape[1] != 3:
            raise ValueError("La nube del paso 12 debe tener forma N x 3.")
        metadata = (
            cloud_confidence_raw,
            cloud_support_raw,
            evidence_class,
            independent_support,
            angular_span,
            conflict_ratio,
            heldout_available,
            heldout_tests,
            heldout_pass_count,
            heldout_pass,
            evidence_contract_accept,
            evidence_strength,
        )
        if any(len(a) != len(cloud_points_all) for a in metadata):
            raise ValueError(
                "Metadatos de evidencia del paso 12 no coinciden con el número de puntos."
            )
        if bool(args.require_evidence_contract) and not evidence_contract_available:
            raise RuntimeError("Paso 15 V1.25 requiere el contrato de evidencia de 11/12.")
        if evidence_contract_available:
            if evidence_contract_accept_published:
                # La selección de evidencia pertenece al paso 12. Paso 15 solo
                # consume su máscara y su fuerza continua; no vuelve a cambiar
                # el significado de held-out durante el pulido.
                evidence_safe = evidence_contract_accept.copy()
                evidence_gate_mode = "authoritative_step12_accept_mask"
            else:
                heldout_vote_ok = (~heldout_available) | (
                    heldout_pass_count.astype(np.int64) * 3 >= 2 * heldout_tests.astype(np.int64)
                )
                evidence_safe = (
                    (evidence_class >= int(args.minimum_evidence_class))
                    & (independent_support >= int(args.minimum_independent_support))
                    & (angular_span >= int(args.minimum_angular_span_poses))
                    & (conflict_ratio <= float(args.maximum_conflict_pose_ratio))
                    & heldout_vote_ok
                )
                evidence_gate_mode = "legacy_exact_two_of_three_verification"
        else:
            evidence_safe = np.ones(len(cloud_points_all), dtype=bool)
            evidence_gate_mode = "legacy_no_contract"
        if np.count_nonzero(evidence_safe) < 500:
            raise RuntimeError("Paso 15: evidencia segura insuficiente para controlar el pulido.")
        cloud_points = cloud_points_all[evidence_safe]
        cloud_normals = cloud_normals_all[evidence_safe] if cloud_normals_all is not None else None
        cloud_confidence = np.clip(
            cloud_confidence_raw[evidence_safe] * evidence_strength[evidence_safe], 0.0, 1.0
        )
        cloud_support = independent_support[evidence_safe]
        evidence_contract_report = {
            "available": bool(evidence_contract_available),
            "input_points": int(len(cloud_points_all)),
            "retained_points": int(len(cloud_points)),
            "rejected_points": int(np.count_nonzero(~evidence_safe)),
            "independent_support": stats(cloud_support),
            "effective_confidence": stats(cloud_confidence),
            "policy": "smoothing_and_fidelity_guards_use_only_pose_diverse_nonconflicting_evidence",
            "gate_mode": evidence_gate_mode,
        }

        # Escala física para límites de desplazamiento.
        smoothing_spacing = optional_scalar(
            d,
            "meshing_reference_spacing_mm",
            None,
        )
        smoothing_spacing_source = "meshing_reference_spacing_mm"

        if smoothing_spacing is None:
            smoothing_spacing = optional_scalar(
                d,
                "reference_spacing_mm",
                None,
            )
            smoothing_spacing_source = "reference_spacing_mm"

        if smoothing_spacing is None:
            smoothing_spacing = median_mesh_edge_length(
                vertices,
                triangles,
            )
            smoothing_spacing_source = "median_mesh_edge_length_fallback"

    smoothing_spacing = float(smoothing_spacing)

    # Escala SEMÁNTICA: exactamente la misma que usa Paso 14.
    topology_info = json.loads(topology_summary_path.read_text(encoding="utf-8"))
    if bool(args.require_evidence_contract):
        topo_evidence = topology_info.get("evidence_contract", {})
        if not bool(topo_evidence.get("available", False)):
            raise RuntimeError(
                "Paso 15 V1.25 requiere que paso 14 haya trabajado con el contrato de evidencia de 11/12."
            )
    semantic_spacing = float(topology_info["estimated_point_spacing_mm"])
    if not np.isfinite(semantic_spacing) or semantic_spacing <= 0:
        raise RuntimeError("estimated_point_spacing_mm inválido en el resumen Paso 14.")

    semantic_eps = max(
        1e-8,
        float(args.semantic_geometric_epsilon_spacing_factor) * semantic_spacing,
    )

    coincident_groups = build_coincident_vertex_groups(
        vertices,
        tolerance_mm=semantic_eps,
    )

    seam_mask, seam_edges, seam_info = load_terminal_seams(
        mesh_path, vertices, triangles, topology_info, semantic_spacing
    )
    seam_permitted = seam_info.pop("_permitted_displacement", np.zeros_like(vertices))
    print(
        f"[Paso 15] Arista tapa-pared protegida: {seam_info['vertices']} vértices; "
        f"{seam_info['edges']} segmentos; {seam_info.get('movable_vertices', 0)} vértices débiles "
        "admiten redistribución limitada sobre el contorno. Se conservan los anclajes y la arista.",
        flush=True,
    )

    result = regularize_mesh_core(
        vertices,
        triangles,
        spacing_mm=smoothing_spacing,
        coincident_groups=coincident_groups,
        args=args,
        terminal_seams=(seam_mask, seam_edges, seam_permitted),
        cloud_data=(cloud_points, cloud_normals, cloud_confidence, cloud_support),
    )
    final_vertices = result["vertices"]

    # --------------------------------------------------------------
    # V1.1 — Guarda semántica de auto-intersecciones.
    # La misma clasificación usada después por Paso 14 se ejecuta aquí
    # ANTES de publicar la geometría.
    # --------------------------------------------------------------
    if bool(args.semantic_intersection_guard):
        (
            final_vertices,
            semantic_guard,
        ) = apply_semantic_intersection_guard(
            vertices,
            final_vertices,
            triangles,
            spacing_mm=semantic_spacing,
            geometric_epsilon_spacing_factor=float(args.semantic_geometric_epsilon_spacing_factor),
            contact_locality_spacing_factor=float(args.semantic_contact_locality_spacing_factor),
            bisection_steps=int(args.semantic_guard_bisection_steps),
            local_rollback_rings=int(args.semantic_local_rollback_rings),
            local_rollback_max_expansions=int(args.semantic_local_rollback_max_expansions),
            local_rollback_ring_floor=float(args.semantic_local_rollback_ring_floor),
            coincident_groups=coincident_groups,
        )
    else:
        semantic_guard = {
            "activated": False,
            "disabled": True,
            "mode": "disabled",
            "baseline": None,
            "full_candidate": None,
            "final": None,
            "local_beta": 1.0,
            "affected_vertices": 0,
            "full_strength_vertices": int(len(vertices)),
            "fallback_global": {
                "used": False,
            },
        }

    result["motion_diagnostics"]["after_semantic_guard_mm"] = stats(
        np.linalg.norm(final_vertices - vertices, axis=1)
    )
    # El blend semántico también debe respetar el coincident-lock.
    final_vertices = enforce_coincident_group_motion(
        final_vertices,
        vertices,
        coincident_groups,
        vertex_strength=result["structure"]["vertex_strength"],
    )

    final_vertices, residual_finish = localized_residual_finish(
        final_vertices,
        vertices,
        triangles,
        result["structure"],
        cloud_points,
        cloud_confidence,
        cloud_support,
        coincident_groups,
        smoothing_spacing,
        semantic_spacing,
        args,
    )
    # La malla de respaldo ya estaba pulida. Las guardas del acabado adicional
    # nunca mezclan los vértices hacia la malla sin pulir del paso 14.

    seam_shift = np.linalg.norm(final_vertices[seam_mask] - vertices[seam_mask], axis=1)
    seam_info["maximum_shift_mm"] = float(seam_shift.max()) if len(seam_shift) else 0.0
    actual_seam = final_vertices[seam_mask] - vertices[seam_mask]
    permitted_seam = seam_permitted[seam_mask]
    length2 = np.einsum("ij,ij->i", permitted_seam, permitted_seam)
    alpha = np.einsum("ij,ij->i", actual_seam, permitted_seam) / np.maximum(length2, 1e-30)
    projection = np.clip(alpha, 0.0, 1.0)[:, None] * permitted_seam
    error = np.linalg.norm(actual_seam - projection, axis=1)
    seam_info["maximum_off_curve_motion_mm"] = float(error.max()) if len(error) else 0.0
    if np.any(error > 1e-8):
        raise RuntimeError(
            "El borde salió del segmento o presupuesto autorizado; no se publica la malla."
        )

    result["motion_diagnostics"]["after_residual_finish_mm"] = stats(
        np.linalg.norm(final_vertices - vertices, axis=1)
    )
    proposed_count = result["motion_diagnostics"]["vertices_with_proposed_motion"]
    retained_count = int(np.count_nonzero(np.linalg.norm(final_vertices - vertices, axis=1) > 1e-6))
    result["motion_diagnostics"]["final_vertices_with_motion"] = retained_count
    print(
        f"[Paso 15] Diagnóstico de pulido: {proposed_count:,} vértices con movimiento propuesto; "
        f"{retained_count:,} con movimiento final. Consulte motion_diagnostics para cada guarda.",
        flush=True,
    )
    if proposed_count and retained_count < 0.1 * proposed_count:
        print(
            "[Paso 15] Aviso: las guardas han limitado casi todo el pulido. "
            "Se conserva la seguridad; el resumen identifica la etapa que redujo el movimiento.",
            flush=True,
        )
    # El desplazamiento reportado debe ser el REAL después de todas las guardas.
    result["total_shift_mm"] = np.linalg.norm(
        final_vertices - vertices,
        axis=1,
    )
    result["vertices"] = final_vertices

    # Verificación fuerte: conectividad idéntica.
    final_triangles = triangles.copy()
    if not np.array_equal(final_triangles, triangles):
        raise RuntimeError("paso 15 alteró la conectividad: esto no está permitido.")

    before_mesh = mesh_from_arrays(
        vertices,
        triangles,
        colors,
    )
    after_mesh = mesh_from_arrays(
        final_vertices,
        final_triangles,
        colors,
    )

    after_mesh.compute_triangle_normals()
    after_mesh.compute_vertex_normals()

    # Fidelidad a la nube regularizada del paso 12.
    rng = np.random.default_rng(5430)
    cloud_query = cloud_points
    if len(cloud_query) > int(args.cloud_evaluation_samples):
        idx = rng.choice(
            len(cloud_query),
            int(args.cloud_evaluation_samples),
            replace=False,
        )
        cloud_query = cloud_query[idx]

    c2m_before = cloud_to_mesh_distances(
        before_mesh,
        cloud_query,
    )
    c2m_after = cloud_to_mesh_distances(
        after_mesh,
        cloud_query,
    )

    output = multi / args.output_name
    output.mkdir(parents=True, exist_ok=True)

    before_path = output / "malla_antes_regularizacion.ply"
    regularized_path = output / "malla_regularizada_general.ply"
    compatibility_path = output / "malla_final_topologica.ply"

    save_mesh(before_path, before_mesh)
    save_mesh(regularized_path, after_mesh)
    save_mesh(compatibility_path, after_mesh)

    displacement_path = output / "desplazamiento_vertices_mm.npy"
    np.save(
        displacement_path,
        result["total_shift_mm"].astype(np.float32),
    )

    structure = result["structure"]

    normal_trend = result["normal_trend"]
    nt_last = normal_trend["last"]
    nt_info = nt_last["normal_info"]
    normal_trend_path = output / "diagnostico_normales_post_relleno_v1_6.npz"
    np.savez_compressed(
        normal_trend_path,
        context_correction_mm=np.asarray(
            structure.get("context_correction", np.zeros_like(vertices)), dtype=np.float32
        ),
        classified_edges=structure["edges"],
        local_geometry_class=structure.get(
            "local_geometry_class", np.zeros(len(structure["edges"]), np.uint8)
        ),
        cloud_feature_class=structure.get(
            "cloud_feature_class", np.zeros(len(structure["edges"]), np.uint8)
        ),
        face_normals_before=np.asarray(
            nt_info["face_normals_before"],
            dtype=np.float32,
        ),
        face_normals_filtered=np.asarray(
            nt_info["face_normals_filtered"],
            dtype=np.float32,
        ),
        vertex_normals_filtered=np.asarray(
            nt_info["vertex_normals_filtered"],
            dtype=np.float32,
        ),
        pair_angles_before_deg=np.asarray(
            nt_info["pair_angles_before_deg"],
            dtype=np.float32,
        ),
        pair_angles_filtered_deg=np.asarray(
            nt_info["pair_angles_filtered_deg"],
            dtype=np.float32,
        ),
        small_prediction_mm=np.asarray(
            nt_last["small_prediction_mm"],
            dtype=np.float32,
        ),
        large_prediction_mm=np.asarray(
            nt_last["large_prediction_mm"],
            dtype=np.float32,
        ),
        scale_difference_mm=np.asarray(
            nt_last["scale_difference_mm"],
            dtype=np.float32,
        ),
        correction_gate=np.asarray(
            nt_last["gate"],
            dtype=np.float32,
        ),
        shift_mm=np.asarray(
            normal_trend["total_shift_mm"],
            dtype=np.float32,
        ),
    )

    despike = result["despike"]
    despike_path = output / "diagnostico_despiking_post_relleno_v1_6.npz"
    np.savez_compressed(
        despike_path,
        residual_mm=np.asarray(
            despike["residual_mm"],
            dtype=np.float32,
        ),
        local_scale_mm=np.asarray(
            despike["local_scale_mm"],
            dtype=np.float32,
        ),
        robust_zscore=np.asarray(
            despike["robust_zscore"],
            dtype=np.float32,
        ),
        candidate_outlier=np.asarray(
            despike["candidate_outlier"],
            dtype=bool,
        ),
        eligible=np.asarray(
            despike["eligible"],
            dtype=bool,
        ),
        isolated=np.asarray(
            despike["isolated"],
            dtype=bool,
        ),
        selected=np.asarray(
            despike["selected"],
            dtype=bool,
        ),
        same_sign_neighbor_ratio=np.asarray(
            despike["same_sign_neighbor_ratio"],
            dtype=np.float32,
        ),
        shift_mm=np.asarray(
            despike["shift_mm"],
            dtype=np.float32,
        ),
    )

    surface_fairing = result["surface_fairing"]
    sf_last = surface_fairing["last"]

    protection_path = output / "proteccion_vertices.npz"
    np.savez_compressed(
        protection_path,
        vertex_strength=structure["vertex_strength"].astype(np.float32),
        vertex_max_dihedral_deg=structure["vertex_max_dihedral_deg"].astype(np.float32),
        boundary_vertex=structure["boundary_vertex"],
        boundary_ring_vertex=structure["boundary_ring_vertex"],
        edge_dihedral_deg=structure["dihedral_deg"].astype(np.float32),
        boundary_edge=structure["boundary_edge"],
    )

    preview_path = output / "preview_regularizacion_malla.png"
    preview_written = make_preview(
        preview_path,
        vertices,
        final_vertices,
        result["total_shift_mm"],
        structure,
        args.preview_max_vertices,
    )

    before_topology = topology_snapshot(before_mesh)
    after_topology = topology_snapshot(after_mesh)

    feature_hard_vertices = structure["vertex_max_dihedral_deg"] >= float(
        args.feature_hard_angle_deg
    )

    warning_reasons = []
    guard = result["guard"]
    if guard["activated"]:
        warning_reasons.append(
            "La guarda de seguridad redujo el movimiento en regiones conflictivas; consulte su modo y los factores por vértice."
        )

    if semantic_guard.get("activated"):
        warning_reasons.append(
            "La guarda semántica local hizo rollback únicamente en las "
            "regiones necesarias para evitar nuevas auto-intersecciones."
        )

    if (
        stats(result["total_shift_mm"])["p95"] is not None
        and stats(result["total_shift_mm"])["p95"] > 0.85 * result["max_total_mm"]
    ):
        warning_reasons.append("El P95 de desplazamiento está próximo al límite total permitido.")

    if residual_finish.get("enabled") and residual_finish.get("reason") in {
        "orientation_guard_kept_previous_polish",
        "intersection_guard_kept_previous_polish",
        "extent_guard_kept_previous_polish",
        "fidelity_guard_kept_previous_polish",
        "guard_verification_unavailable",
        "nonfinite_fidelity_measurement",
    }:
        warning_reasons.append(
            "El acabado residual adicional se omitió por seguridad; se conserva el pulido previo. Motivo: "
            + residual_finish["reason"]
        )
    quality = "accepted" if not warning_reasons else "warning"

    report = {
        "schema_version": "1.2",
        "implementation": identity,
        "step": "15",
        "quality": quality,
        "warning_reasons": warning_reasons,
        "method": (
            "post_fill_bilateral_face_normal_filter_"
            "multiscale_robust_quadratic_surface_trend_"
            "robust_local_despiking_"
            "feature_aware_normal_dominant_taubin_"
            "adaptive_quadratic_residual_surface_fairing_"
            "vertexwise_orientation_and_local_semantic_rollback_sparse_residual_finish"
        ),
        "object": obj,
        "shape_specific_assumptions": False,
        "localized_residual_finish": residual_finish,
        "motion_diagnostics": result["motion_diagnostics"],
        "hole_filling": False,
        "watertight_forced": False,
        "vertices_added": 0,
        "vertices_removed": 0,
        "triangles_added": 0,
        "triangles_removed": 0,
        "connectivity_modified": False,
        "source_mesh": str(mesh_path),
        "terminal_seam_protection": seam_info,
        "source_cloud": str(cloud_path),
        "evidence_contract": evidence_contract_report,
        "smoothing_spacing_mm": smoothing_spacing,
        "smoothing_spacing_source": smoothing_spacing_source,
        "semantic_spacing_mm": semantic_spacing,
        "semantic_spacing_source": str(topology_summary_path),
        "semantic_geometric_epsilon_mm": semantic_eps,
        "coincident_lock": {
            "groups": int(len(coincident_groups)),
            "vertices_in_groups": int(sum(len(g) for g in coincident_groups)),
            "group_size": stats([len(g) for g in coincident_groups]),
            "separation_before_mm": coincident_group_separation_stats(
                vertices,
                coincident_groups,
            ),
            "separation_after_mm": coincident_group_separation_stats(
                final_vertices,
                coincident_groups,
            ),
        },
        "parameters": vars(args),
        "input": {
            "vertices": int(len(vertices)),
            "triangles": int(len(triangles)),
            "topology_raw_open3d": before_topology,
        },
        "protection": {
            "cloud_feature_evidence": structure.get("cloud_feature_evidence", {}),
            "incremental_orientation": structure.get("incremental_orientation", []),
            "boundary_vertices": int(np.count_nonzero(structure["boundary_vertex"])),
            "boundary_ring_vertices": int(np.count_nonzero(structure["boundary_ring_vertex"])),
            "hard_feature_vertices": int(np.count_nonzero(feature_hard_vertices)),
            "vertex_strength": stats(structure["vertex_strength"]),
            "max_dihedral_deg": stats(structure["vertex_max_dihedral_deg"]),
        },
        "normal_multiscale_trend": {
            "enabled": bool(args.normal_trend_enabled),
            "cycles": normal_trend["cycles"],
            "total_candidate_shift_mm": stats(normal_trend["total_shift_mm"]),
            "neighbor_face_angle_before_deg": stats(nt_info["pair_angles_before_deg"]),
            "neighbor_face_angle_filtered_deg": stats(nt_info["pair_angles_filtered_deg"]),
            "scale_difference_mm": stats(nt_last["scale_difference_mm"]),
            "gate": stats(nt_last["gate"]),
            "vertices_corrected": int(np.count_nonzero(nt_last["gate"] > 0)),
            "vertices_strongly_corrected": int(np.count_nonzero(nt_last["gate"] >= 0.5)),
            "diagnostic_file": str(normal_trend_path),
        },
        "despiking": {
            "enabled": bool(args.despike_enabled),
            "selected_vertices": int(result["despike"]["selected_count"]),
            "candidate_outliers": int(np.count_nonzero(result["despike"]["candidate_outlier"])),
            "eligible_outliers": int(np.count_nonzero(result["despike"]["eligible"])),
            "isolated_outliers": int(np.count_nonzero(result["despike"]["isolated"])),
            "residual_mm": stats(result["despike"]["residual_mm"]),
            "robust_zscore": stats(result["despike"]["robust_zscore"]),
            "despike_shift_mm": stats(result["despike"]["shift_mm"]),
            "maximum_despike_shift_allowed_mm": float(result["despike"]["max_shift_allowed_mm"]),
            "diagnostic_file": str(despike_path),
        },
        "adaptive_surface_fairing": {
            "enabled": bool(args.surface_fairing_enabled),
            "cycles": surface_fairing["cycles"],
            "cycles_executed": int(len(surface_fairing["cycles"])),
            "neighborhood_size": surface_fairing["neighborhood_size"],
            "convergence_threshold_mm": float(surface_fairing["convergence_threshold_mm"]),
            "active_vertices_last_cycle": int(np.count_nonzero(sf_last["active"])),
            "target_last_cycle_mm": stats(sf_last["target_mm"]),
            "local_scale_last_cycle_mm": stats(sf_last["local_scale_mm"]),
            "gate_last_cycle": stats(sf_last["gate"]),
            "total_fairing_shift_mm": stats(surface_fairing["total_shift_mm"]),
            "total_clipped_vertices_last_cycle": int(np.count_nonzero(sf_last["total_clipped"])),
        },
        "regularization": {
            "iterations": result["iteration_records"],
            "max_shift_per_pass_mm": result["max_step_mm"],
            "max_total_shift_mm": result["max_total_mm"],
            "total_displacement_mm": stats(result["total_shift_mm"]),
            "step_clipped_vertices": int(np.count_nonzero(result["step_clipped_any"])),
            "total_clipped_vertices": int(np.count_nonzero(result["total_clipped_any"])),
            "global_safety_guard": guard,
            "semantic_intersection_guard": semantic_guard,
        },
        "cloud_fidelity": {
            "cloud_to_mesh_before_mm": stats(c2m_before),
            "cloud_to_mesh_after_mm": stats(c2m_after),
        },
        "output": {
            "vertices": int(len(final_vertices)),
            "triangles": int(len(final_triangles)),
            "topology_raw_open3d": after_topology,
            "malla_antes": str(before_path),
            "malla_regularizada": str(regularized_path),
            "malla_final_topologica_compatibilidad": str(compatibility_path),
            "displacement": str(displacement_path),
            "protection": str(protection_path),
            "normal_multiscale_diagnostics": str(normal_trend_path),
            "despiking_diagnostics": str(despike_path),
            "preview": str(preview_path) if preview_written else None,
        },
        "important_note": (
            "Paso 15 V1.25 se ejecuta después de Paso 14 y conserva el contrato de evidencia de 11/12. "
            "Su función es únicamente el acabado geométrico: no añade, "
            "elimina ni conecta triángulos. La continuidad debe provenir "
            "principalmente de Paso 13 V5.0; flips e intersecciones nuevas se "
            "resuelven mediante rollback local antes de publicar la malla."
        ),
    }

    _estado15("Guardando resumen y diagnósticos finales")
    summary_path = output / "resumen_15_pulido_final.json"
    summary_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n========== PASO 15 V1.11 COMPLETADA ==========")
    print(f"Vértices/triángulos: {len(vertices):,}/{len(triangles):,}")
    print(
        f"Spacing suavizado/límites: {smoothing_spacing:.6f} mm | "
        f"fuente={smoothing_spacing_source}"
    )
    print(f"Spacing semántico paso 14: {semantic_spacing:.6f} mm | " f"eps={semantic_eps:.9f} mm")
    print(
        f"Coincident-lock: {len(coincident_groups):,} grupos | "
        f"{sum(len(g) for g in coincident_groups):,} vértices"
    )
    print(
        "Normales/tendencia final V1.11:",
        {
            "shift": stats(normal_trend["total_shift_mm"]),
            "angles_before": stats(nt_info["pair_angles_before_deg"]),
            "angles_filtered": stats(nt_info["pair_angles_filtered_deg"]),
            "scale_difference": stats(nt_last["scale_difference_mm"]),
            "vertices_corrected": int(np.count_nonzero(nt_last["gate"] > 0)),
        },
    )
    print(
        "Despiking residual:",
        {
            "candidate_outliers": int(np.count_nonzero(result["despike"]["candidate_outlier"])),
            "selected_vertices": int(result["despike"]["selected_count"]),
            "shift": stats(result["despike"]["shift_mm"]),
        },
    )
    print(
        "Fairing residual adaptativo:",
        {
            "cycles": int(len(surface_fairing["cycles"])),
            "active_last": int(np.count_nonzero(sf_last["active"])),
            "target_last": stats(sf_last["target_mm"]),
            "shift_total": stats(surface_fairing["total_shift_mm"]),
        },
    )
    print(
        "Desplazamiento total:",
        stats(result["total_shift_mm"]),
    )
    print(f"Boundary protegidos: " f"{np.count_nonzero(structure['boundary_vertex']):,}")
    print(f"Arista dura protegida (vértices): " f"{np.count_nonzero(feature_hard_vertices):,}")
    print(
        "Cloud->mesh antes:",
        stats(c2m_before),
    )
    print(
        "Cloud->mesh después:",
        stats(c2m_after),
    )
    print(
        "Safety guard:",
        guard,
    )
    print(
        "Semantic intersection guard:",
        semantic_guard,
    )
    print("Calidad:", quality)
    print("Salida V1.11:", output)
    print("==================================================\n")

    return 0


def guard_polishing_stage(function):
    # Cada etapa retorna (vértices, diagnóstico). Mantener firma para workers.
    import functools

    @functools.wraps(function)
    def guarded(vertices, original_vertices, triangles, structure, *args, **kwargs):
        before = np.asarray(vertices, dtype=float).copy()
        proposed, info = function(
            vertices, original_vertices, triangles, structure, *args, **kwargs
        )
        groups = kwargs.get("coincident_groups", args[0] if args else None)
        accepted, alpha, trace, safe = vertexwise_orientation_guard(
            before, proposed, triangles, coincident_groups=groups, max_iterations=28
        )
        if not safe:
            accepted = before
        record = {
            "operation": function.__name__,
            "accepted": bool(safe),
            "restricted_vertices": int(np.count_nonzero(alpha < 1)),
            "iterations": len(trace),
            "accepted_shift_mm": stats(np.linalg.norm(accepted - before, axis=1)),
        }
        structure.setdefault("incremental_orientation", []).append(record)
        info["incremental_orientation"] = record
        return accepted, info

    return guarded


multiscale_normal_trend_regularization = guard_polishing_stage(
    multiscale_normal_trend_regularization
)
despike_mesh_vertices = guard_polishing_stage(despike_mesh_vertices)
adaptive_surface_fairing_regularization = guard_polishing_stage(
    adaptive_surface_fairing_regularization
)


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
        _estado15(descripcion)
        try:
            return funcion(*args, **kwargs)
        except Exception as exc:
            print(f"[ERROR] Paso 15 | {descripcion} | {type(exc).__name__}: {exc}", flush=True)
            raise
        finally:
            with _estado_lock:
                _estado_actual = anterior

    return ejecutar


classify_cloud_supported_features = _con_estado_visible(
    classify_cloud_supported_features, "Contrastando aristas y pliegues con evidencia de nube"
)
regularize_mesh_core = _con_estado_visible(
    regularize_mesh_core, "Preparando y aplicando pulido de la malla"
)
multiscale_normal_trend_regularization = _con_estado_visible(
    multiscale_normal_trend_regularization, "Regularizando la tendencia de normales"
)
despike_mesh_vertices = _con_estado_visible(
    despike_mesh_vertices, "Evaluando irregularidades locales"
)
adaptive_surface_fairing_regularization = _con_estado_visible(
    adaptive_surface_fairing_regularization, "Aplicando pulido superficial adaptativo"
)
apply_global_safety_guard = _con_estado_visible(
    apply_global_safety_guard, "Verificando orientación y dimensiones"
)
vertexwise_orientation_guard = _con_estado_visible(
    vertexwise_orientation_guard, "Recuperando movimiento seguro por vértice"
)
apply_semantic_intersection_guard = _con_estado_visible(
    apply_semantic_intersection_guard, "Verificando intersecciones del pulido"
)
localized_residual_finish = _con_estado_visible(
    localized_residual_finish, "Aplicando acabado residual localizado"
)
cloud_to_mesh_distances = _con_estado_visible(
    cloud_to_mesh_distances, "Midiendo fidelidad a la nube"
)
save_mesh = _con_estado_visible(save_mesh, "Guardando malla")
make_preview = _con_estado_visible(make_preview, "Generando vista previa")
topology_snapshot = _con_estado_visible(topology_snapshot, "Verificando topología")


def _entrada_con_diagnostico():
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    from utilidades_progreso import informar_inicio

    informar_inicio(
        __file__,
        "15",
        "Pulir modelo con fairing adaptativo y guardas locales",
    )
    _estado15("Leyendo parámetros, malla y nube de entrada")
    try:
        codigo = main()
        _estado15("Completado" if codigo == 0 else f"Finalizado con observaciones: {codigo}")
        raise SystemExit(codigo)
    except Exception as exc:
        _estado15(f"ERROR: {type(exc).__name__}: {exc}")
        raise
    finally:
        _estado_stop.set()


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
