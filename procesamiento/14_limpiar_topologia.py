#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PASO 14 V2.0 — Reparación local sin añadir tapas terminales estimadas.
Repara pequeños contornos locales; no añade tapas terminales estimadas.
Publica actividad y tiempos periódicamente durante todas las etapas.
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


def _mostrar_estado14():
    """Publica la actividad actual y los tiempos transcurridos con lectura protegida."""
    with _estado_lock:
        actividad, inicio = _estado_actual
    ahora = _time.monotonic()
    print(
        f"[PROGRESO] Paso 14 | {actividad} | Tiempo actividad: {ahora-inicio:.1f} s | "
        f"Tiempo total: {ahora-_estado_inicio:.1f} s",
        flush=True,
    )


def _estado14(actividad):
    """Actualiza la actividad bajo bloqueo y publica el nuevo estado."""
    global _estado_actual
    with _estado_lock:
        _estado_actual = (actividad, _time.monotonic())
    _mostrar_estado14()


def _latido14():
    """Publica periódicamente el estado hasta recibir la señal de detención."""
    while not _estado_stop.wait(10):
        _mostrar_estado14()


if __name__ == "__main__":
    for _stream in (_sys.stdout, _sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(line_buffering=True, write_through=True)
            except (ValueError, OSError):
                pass
    print(
        "[EJECUTANDO] 14_limpiar_topologia.py | Reparación local sin tapa inferior estimada",
        flush=True,
    )
    _mostrar_estado14()
    _monitor14 = _threading.Thread(target=_latido14, name="estado_paso14", daemon=True)
    _monitor14.start()
    _atexit.register(_estado_stop.set)

from utilidades_progreso import operacion
from utilidades_rendimiento import query_threads
from utilidades_rendimiento import configurar as _configurar_recursos

# Los runtimes numéricos leen estos límites al importarse.
_configurar_recursos(__file__)


import argparse
import hashlib
import csv
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
except Exception as exc:
    raise SystemExit(f"Paso 14 V2.0 requiere SciPy: {exc}")

try:
    import open3d as o3d
except Exception as exc:
    raise SystemExit(
        "Paso 14 V2.0 requiere Open3D. En el entorno tesis ejecuta "
        "`python -m pip install open3d==0.19.0`. "
        f"Detalle: {exc}"
    )

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit("Paso 14 V2.0 requiere matplotlib para los previews. " f"Detalle: {exc}")


from clasificador_intersecciones import (
    ALLOWED_CONTACTS,
    BLOCKING_INTERSECTIONS,
    classify_triangle_pair,
)

# =============================================================================
# CLI
# =============================================================================


def save_inferred_face_provenance(source_path, final_mesh, output, all_estimated=False):
    import open3d as o3d
    import hashlib
    def keys(mesh):
        points = np.rint(np.asarray(mesh.vertices) / 1e-5).astype(np.int64)
        return [tuple(sorted(tuple(points[i]) for i in face)) for face in np.asarray(mesh.triangles)]
    original = set(keys(o3d.io.read_triangle_mesh(str(source_path))))
    inherited_keys = set()
    source_provenance = Path(source_path).parent / "procedencia_estimada.npz"
    if source_provenance.is_file():
        with np.load(source_provenance, allow_pickle=False) as data:
            if "inferred_faces" in data.files:
                expected = str(data["mesh_sha256"].item())
                if expected != hashlib.sha256(Path(source_path).read_bytes()).hexdigest():
                    raise RuntimeError("Procedencia de parches incompatible con la malla 13")
                source_keys = keys(o3d.io.read_triangle_mesh(str(source_path)))
                flags = np.asarray(data["inferred_faces"], dtype=bool)
                if flags.shape != (len(source_keys),):
                    raise RuntimeError("Procedencia de caras con longitud incompatible")
                inherited_keys = {key for key,flag in zip(source_keys,flags) if flag}
    inferred = np.array([all_estimated or key not in original or key in inherited_keys
                         for key in keys(final_mesh)], dtype=bool)
    np.savez_compressed(output / "procedencia_caras_relleno.npz",
        inferred_or_retriangulated=inferred,
        mesh_sha256=np.array(hashlib.sha256((output / "malla_final_topologica.ply").read_bytes()).hexdigest()),
        inferred_is_observed=np.array(False))

def make_parser():
    """Construye las opciones de línea de comandos de este paso."""
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--object", required=True)

    p.add_argument(
        "--mesh-source",
        default="13_reconstruccion_superficie",
    )
    p.add_argument(
        "--cloud-source",
        default="12_regularizacion_nube",
    )
    p.add_argument(
        "--output-name",
        default="14_limpieza_topologica",
    )

    # V2.0 — las reparaciones topológicas solo usan como anclaje observaciones
    # que conservaron evidencia independiente después de 11/12.
    p.add_argument(
        "--require-evidence-contract", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--minimum-evidence-class", type=int, default=2)
    p.add_argument("--minimum-independent-support", type=int, default=2)
    p.add_argument("--minimum-angular-span-poses", type=int, default=2)
    p.add_argument("--maximum-conflict-pose-ratio", type=float, default=0.35)
    p.add_argument("--minimum-heldout-pass-ratio", type=float, default=0.67)

    # Misma filosofía conservadora de componentes de C2 V1.
    p.add_argument("--tiny-component-max-triangles", type=int, default=20)
    p.add_argument("--tiny-component-max-area-ratio", type=float, default=0.0005)
    p.add_argument(
        "--tiny-component-max-extent-spacing-factor",
        type=float,
        default=8.0,
    )
    p.add_argument(
        "--tiny-component-min-median-support-to-keep",
        type=float,
        default=1.5,
    )
    p.add_argument(
        "--tiny-component-min-median-confidence-to-keep",
        type=float,
        default=0.68,
    )

    # V2.0 — segunda limpieza después de TODAS las reparaciones. Solo puede
    # retirar componentes completas diminutas que no existían como componente
    # independiente en la malla fuente del paso 13. Nunca reconecta por forma.
    p.add_argument(
        "--post-repair-component-cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--post-repair-source-ancestry-min-ratio",
        type=float,
        default=0.60,
    )
    p.add_argument(
        "--post-repair-dominant-area-ratio",
        type=float,
        default=20.0,
        help=(
            "La componente hermana dominante debe tener al menos esta relación "
            "de área para considerar la menor un fragmento desprendido."
        ),
    )

    # Protección durante reparación de orientabilidad.
    p.add_argument("--orientation-support-cap", type=float, default=5.0)
    p.add_argument("--orientation-support-weight", type=float, default=0.90)
    p.add_argument("--orientation-confidence-weight", type=float, default=1.35)
    p.add_argument("--orientation-area-weight", type=float, default=0.25)
    p.add_argument("--orientation-interior-weight", type=float, default=0.25)
    p.add_argument(
        "--max-orientation-face-removal-ratio",
        type=float,
        default=0.020,
        help="Máximo 2%% de caras de la malla tras limpieza de componentes.",
    )

    # Una superficie abierta sigue permitida cuando un cierre no supera las guardas.
    p.add_argument(
        "--reject-if-self-intersecting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Diagnóstico raw. V2.0 usa clasificación semántica para el gate."),
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
    p.add_argument(
        "--semantic-repair-max-iterations",
        type=int,
        default=4,
    )
    p.add_argument("--fill-existing-holes", action=argparse.BooleanOptionalAction,
                   default=False, help="Relleno inferido opcional de huecos preexistentes; no es evidencia observada.")
    p.add_argument("--existing-hole-max-diameter-mm", type=float, default=12.0)
    p.add_argument("--repair-holes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--estimated-terminal-caps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compatibilidad con lanzadores anteriores: las tapas estimadas permanecen retiradas en V2.0.",
    )
    p.add_argument("--repair-hole-max-diameter-spacing", type=float, default=8.0)
    p.add_argument("--closure-planarity-ratio", type=float, default=0.04)
    p.add_argument("--closure-max-loop-vertices", type=int, default=512)
    p.add_argument("--closure-max-new-triangles", type=int, default=30000)
    p.add_argument("--refine-terminal-caps", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rim-max-shift-spacing", type=float, default=1.0)
    return p


# =============================================================================
# UTILIDADES
# =============================================================================


def robust_stats(values):
    a = np.asarray(list(values), dtype=np.float64)
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


def triangle_areas(vertices, triangles):
    """Calcula el área de cada triángulo mediante el producto vectorial."""
    tri = vertices[triangles]
    return 0.5 * np.linalg.norm(
        np.cross(
            tri[:, 1] - tri[:, 0],
            tri[:, 2] - tri[:, 0],
        ),
        axis=1,
    )


def compact_mesh_arrays(vertices, triangles, colors=None):
    """Retira vértices sin uso y remapea triángulos y colores de forma consistente."""
    if len(triangles) == 0:
        raise RuntimeError("La malla quedó sin triángulos.")
    used = np.unique(triangles.ravel())
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    v = vertices[used]
    f = remap[triangles]
    c = colors[used] if colors is not None and len(colors) == len(vertices) else None
    return v, f, c


def make_mesh(vertices, triangles, colors=None):
    """Construye una malla Open3D con colores opcionales."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    if colors is not None and len(colors) == len(vertices):
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0)
        )

    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


# =============================================================================
# CONECTIVIDAD / COMPONENTES
# =============================================================================


def build_edge_incidence(triangles):
    """
    edge -> [(face_id, direction_relative_to_sorted_edge), ...]
    direction = +1 si la cara recorre min->max, -1 si max->min.
    """
    edge_faces = defaultdict(list)
    for fi, tri in enumerate(triangles):
        a, b, c = map(int, tri)
        for u, v in ((a, b), (b, c), (c, a)):
            if u < v:
                key, direction = (u, v), 1
            else:
                key, direction = (v, u), -1
            edge_faces[key].append((fi, direction))
    return edge_faces


def face_components(triangles):
    """Etiqueta componentes de caras conectadas por aristas."""
    n = len(triangles)
    if n == 0:
        return 0, np.empty(0, dtype=np.int32)

    edge_faces = build_edge_incidence(triangles)
    rows, cols = [], []

    for inc in edge_faces.values():
        if len(inc) < 2:
            continue
        root = inc[0][0]
        for other, _ in inc[1:]:
            rows.extend([root, other])
            cols.extend([other, root])

    if not rows:
        return n, np.arange(n, dtype=np.int32)

    graph = coo_matrix(
        (
            np.ones(len(rows), dtype=np.uint8),
            (rows, cols),
        ),
        shape=(n, n),
    ).tocsr()

    count, labels = connected_components(graph, directed=False)
    return int(count), labels.astype(np.int32)


def cloud_evidence_for_vertices(
    vertices,
    cloud_points,
    support,
    confidence,
    tree,
):
    """Asocia a los vértices la evidencia de sus vecinos en la nube."""
    d, idx = tree.query(vertices, k=1, workers=query_threads())
    idx = np.asarray(idx, dtype=np.int64)
    return (
        np.asarray(d, dtype=np.float64),
        np.asarray(support[idx], dtype=np.float64),
        np.asarray(confidence[idx], dtype=np.float64),
    )


@operacion("Evaluar componentes de la malla")
def component_cleanup(
    vertices,
    triangles,
    colors,
    cloud_points,
    support,
    confidence,
    spacing,
    args,
):
    count, labels = face_components(triangles)
    areas = triangle_areas(vertices, triangles)
    total_area = max(float(np.sum(areas)), 1e-12)
    tree = cKDTree(cloud_points)

    keep_faces = np.ones(len(triangles), dtype=bool)
    records = []

    max_extent_threshold = max(
        4.0,
        float(args.tiny_component_max_extent_spacing_factor) * spacing,
    )

    for cid in range(count):
        face_idx = np.flatnonzero(labels == cid)
        vertex_idx = np.unique(triangles[face_idx].ravel())
        pts = vertices[vertex_idx]

        comp_area = float(np.sum(areas[face_idx]))
        extent = np.ptp(pts, axis=0) if len(pts) else np.zeros(3)
        max_extent = float(np.max(extent)) if len(extent) else 0.0

        d, s, c = cloud_evidence_for_vertices(pts, cloud_points, support, confidence, tree)

        median_support = float(np.median(s)) if len(s) else None
        median_confidence = float(np.median(c)) if len(c) else None

        tiny_geometry = bool(
            len(face_idx) < int(args.tiny_component_max_triangles)
            and comp_area / total_area < float(args.tiny_component_max_area_ratio)
            and max_extent < max_extent_threshold
        )

        weak_evidence = bool(
            (
                median_support is None
                or median_support < float(args.tiny_component_min_median_support_to_keep)
            )
            and (
                median_confidence is None
                or median_confidence < float(args.tiny_component_min_median_confidence_to_keep)
            )
        )

        remove = tiny_geometry and weak_evidence
        if remove:
            keep_faces[face_idx] = False

        records.append(
            {
                "component_id": int(cid),
                "triangles": int(len(face_idx)),
                "vertices": int(len(vertex_idx)),
                "area_mm2": comp_area,
                "area_ratio": float(comp_area / total_area),
                "max_extent_mm": max_extent,
                "cloud_distance_median_mm": (float(np.median(d)) if len(d) else None),
                "cloud_distance_p90_mm": (float(np.percentile(d, 90)) if len(d) else None),
                "median_support": median_support,
                "median_confidence": median_confidence,
                "tiny_geometry": tiny_geometry,
                "weak_evidence": weak_evidence,
                "action": "remove_debris" if remove else "keep",
            }
        )

    triangles = triangles[keep_faces]
    vertices, triangles, colors = compact_mesh_arrays(vertices, triangles, colors)
    return vertices, triangles, colors, records


@operacion("Revisar componentes creadas por reparaciones")
def post_repair_component_cleanup(
    vertices,
    triangles,
    colors,
    *,
    source_vertices,
    source_triangles,
    cloud_points,
    support,
    confidence,
    spacing,
    args,
):
    """Retira únicamente fragmentos diminutos DESPRENDIDOS por las reparaciones.

    La decisión usa procedencia topológica, no una forma esperada:
    - se calculan las componentes de la malla fuente de paso 13;
    - cada componente final se asocia a su componente fuente por cercanía de
      vértices (la reparación no debe trasladar la geometría global);
    - una componente final pequeña solo puede eliminarse si comparte ancestro
      con otra componente final mucho mayor. Por tanto una pieza que YA era
      independiente en paso 13 nunca se elimina por este mecanismo.

    No se intentan puentes, soldaduras ni cierres: si una reparación separó un
    residuo diminuto, se retira la componente completa. Las componentes
    significativas quedan intactas y, si una separación grande aparece, se
    conserva para revisión en vez de inventar conectividad.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(triangles, dtype=np.int64)
    c = None if colors is None else np.asarray(colors, dtype=np.float64)
    sv = np.asarray(source_vertices, dtype=np.float64)
    sf = np.asarray(source_triangles, dtype=np.int64)

    if not bool(args.post_repair_component_cleanup) or len(f) == 0:
        return (
            v,
            f,
            c,
            {
                "enabled": bool(args.post_repair_component_cleanup),
                "applied": False,
                "reason": "disabled_or_empty",
                "components_before": int(face_components(f)[0]) if len(f) else 0,
                "components_after": int(face_components(f)[0]) if len(f) else 0,
                "components_removed": 0,
                "records": [],
            },
        )

    source_count, source_face_labels = face_components(sf)
    final_count, final_labels = face_components(f)
    if final_count <= 1 or source_count <= 0:
        return (
            v,
            f,
            c,
            {
                "enabled": True,
                "applied": False,
                "reason": "single_final_component_or_no_source_components",
                "source_components": int(source_count),
                "components_before": int(final_count),
                "components_after": int(final_count),
                "components_removed": 0,
                "records": [],
            },
        )

    # Etiqueta de procedencia por vértice fuente. Un vértice compartido por
    # varias componentes edge-connected se marca ambiguo y no decide ancestro.
    source_vertex_sets = [set() for _ in range(len(sv))]
    for fi, tri in enumerate(sf):
        lab = int(source_face_labels[fi])
        for vid in tri:
            source_vertex_sets[int(vid)].add(lab)
    source_vertex_label = np.full(len(sv), -1, dtype=np.int32)
    for vid, labels in enumerate(source_vertex_sets):
        if len(labels) == 1:
            source_vertex_label[vid] = int(next(iter(labels)))

    source_tree = cKDTree(sv)
    cloud_tree = cKDTree(np.asarray(cloud_points, dtype=np.float64))
    areas = triangle_areas(v, f)
    total_area = max(float(np.sum(areas)), 1e-12)
    ancestry_tolerance = max(2.0 * float(spacing), 1e-6)
    min_ancestry = float(np.clip(args.post_repair_source_ancestry_min_ratio, 0.0, 1.0))
    dominant_ratio_required = max(1.0, float(args.post_repair_dominant_area_ratio))
    max_extent_threshold = max(
        4.0,
        float(args.tiny_component_max_extent_spacing_factor) * float(spacing),
    )

    records = []
    for cid in range(int(final_count)):
        face_idx = np.flatnonzero(final_labels == cid)
        vertex_idx = np.unique(f[face_idx].ravel())
        pts = v[vertex_idx]
        comp_area = float(np.sum(areas[face_idx]))
        extent = np.ptp(pts, axis=0) if len(pts) else np.zeros(3)
        max_extent = float(np.max(extent)) if len(extent) else 0.0

        source_distance, source_index = source_tree.query(pts, k=1, workers=query_threads())
        source_distance = np.asarray(source_distance, dtype=np.float64)
        source_index = np.asarray(source_index, dtype=np.int64)
        mapped = (
            np.isfinite(source_distance)
            & (source_distance <= ancestry_tolerance)
            & (source_vertex_label[source_index] >= 0)
        )
        if np.any(mapped):
            labels_here = source_vertex_label[source_index[mapped]]
            counts = np.bincount(labels_here, minlength=max(source_count, 1))
            ancestor = int(np.argmax(counts))
            ancestor_votes = int(counts[ancestor])
            ancestry_ratio = float(ancestor_votes / max(len(vertex_idx), 1))
        else:
            ancestor = None
            ancestor_votes = 0
            ancestry_ratio = 0.0

        cloud_distance, cloud_index = cloud_tree.query(pts, k=1, workers=query_threads())
        cloud_index = np.asarray(cloud_index, dtype=np.int64)
        median_support = float(np.median(np.asarray(support)[cloud_index])) if len(pts) else None
        median_confidence = (
            float(np.median(np.asarray(confidence)[cloud_index])) if len(pts) else None
        )

        tiny_geometry = bool(
            len(face_idx) <= int(args.tiny_component_max_triangles)
            and comp_area / total_area < float(args.tiny_component_max_area_ratio)
            and max_extent < max_extent_threshold
        )
        records.append(
            {
                "component_id": int(cid),
                "triangles": int(len(face_idx)),
                "vertices": int(len(vertex_idx)),
                "area_mm2": comp_area,
                "area_ratio": float(comp_area / total_area),
                "max_extent_mm": max_extent,
                "tiny_geometry": tiny_geometry,
                "source_ancestor_component": ancestor,
                "source_ancestry_ratio": ancestry_ratio,
                "source_ancestry_votes": ancestor_votes,
                "source_distance_p90_mm": (
                    float(np.percentile(source_distance, 90)) if len(source_distance) else None
                ),
                "cloud_distance_median_mm": (
                    float(np.median(cloud_distance)) if len(cloud_distance) else None
                ),
                "cloud_distance_p90_mm": (
                    float(np.percentile(cloud_distance, 90)) if len(cloud_distance) else None
                ),
                "median_support": median_support,
                "median_confidence": median_confidence,
                "action": "keep_pending_ancestry_comparison",
            }
        )

    # Para cada ancestro fuente, identificar la componente final dominante.
    by_ancestor = defaultdict(list)
    for rec in records:
        anc = rec["source_ancestor_component"]
        if anc is not None and rec["source_ancestry_ratio"] >= min_ancestry:
            by_ancestor[int(anc)].append(rec)

    remove_components = set()
    for ancestor, group in by_ancestor.items():
        if len(group) <= 1:
            continue
        dominant = max(group, key=lambda r: (r["area_mm2"], r["triangles"]))
        dominant_area = max(float(dominant["area_mm2"]), 1e-12)
        for rec in group:
            if rec is dominant:
                rec["action"] = "keep_dominant_descendant"
                continue
            area_ratio_to_dominant = dominant_area / max(float(rec["area_mm2"]), 1e-12)
            rec["dominant_descendant_component"] = int(dominant["component_id"])
            rec["dominant_to_component_area_ratio"] = float(area_ratio_to_dominant)
            repair_created_tiny_fragment = bool(
                rec["tiny_geometry"]
                and rec["source_ancestry_ratio"] >= min_ancestry
                and area_ratio_to_dominant >= dominant_ratio_required
            )
            if repair_created_tiny_fragment:
                remove_components.add(int(rec["component_id"]))
                rec["action"] = "remove_post_repair_detached_fragment"
                rec["reason"] = (
                    "tiny_final_component_shares_source_ancestor_with_much_larger_descendant"
                )
            else:
                rec["action"] = "keep_nontrivial_or_ambiguous_descendant"

    # Componentes sin ancestro claro se conservan. Una geometría nueva de
    # cierre nunca se elimina solo por no encontrar procedencia directa.
    for rec in records:
        if rec["action"] == "keep_pending_ancestry_comparison":
            rec["action"] = "keep_no_shared_source_ancestry"

    if not remove_components:
        return (
            v,
            f,
            c,
            {
                "enabled": True,
                "applied": False,
                "reason": "no_repair_created_tiny_detached_fragments",
                "source_components": int(source_count),
                "components_before": int(final_count),
                "components_after": int(final_count),
                "components_removed": 0,
                "removed_component_ids": [],
                "ancestry_tolerance_mm": float(ancestry_tolerance),
                "records": records,
            },
        )

    keep_faces = ~np.isin(final_labels, np.asarray(sorted(remove_components), dtype=np.int32))
    f2 = f[keep_faces]
    v2, f2, c2 = compact_mesh_arrays(v, f2, c)
    after_count, _ = face_components(f2)
    return (
        v2,
        f2,
        c2,
        {
            "enabled": True,
            "applied": True,
            "reason": "removed_complete_fragments_created_by_topology_repair",
            "source_components": int(source_count),
            "components_before": int(final_count),
            "components_after": int(after_count),
            "components_removed": int(len(remove_components)),
            "removed_component_ids": sorted(int(x) for x in remove_components),
            "faces_removed": int(np.count_nonzero(~keep_faces)),
            "ancestry_tolerance_mm": float(ancestry_tolerance),
            "minimum_source_ancestry_ratio": float(min_ancestry),
            "minimum_dominant_area_ratio": float(dominant_ratio_required),
            "policy": (
                "remove_only_tiny_detached_descendants_of_same_source_component; "
                "never_remove_preexisting_independent_components_by_size_alone"
            ),
            "records": records,
        },
    )


# =============================================================================
# SPLIT SIMULTÁNEO DE FANS DE VÉRTICE
# =============================================================================


def split_vertex_fans(vertices, triangles, colors=None):
    """
    Convierte cada star de vértice en uno o más fans edge-connected.

    Si un vértice tiene varios fans desconectados:
    - crea una copia de la MISMA coordenada por fan;
    - reasigna índices;
    - no mueve geometría.

    A diferencia de una reparación secuencial, este método reconstruye todos
    los índices simultáneamente y evita invalidar la conectividad de vértices
    que todavía no se han procesado.
    """
    edge_faces = build_edge_incidence(triangles)

    incident_faces = [[] for _ in range(len(vertices))]
    for fi, tri in enumerate(triangles):
        for v in tri:
            incident_faces[int(v)].append(fi)

    new_vertices = []
    new_colors = [] if colors is not None else None
    corner_map = {}
    split_operations = []

    for v, faces in enumerate(incident_faces):
        if not faces:
            continue

        adjacency = {fi: set() for fi in faces}

        neighbors = set()
        for fi in faces:
            for u in triangles[fi]:
                u = int(u)
                if u != v:
                    neighbors.add(u)

        for u in neighbors:
            key = (min(u, v), max(u, v))
            inc = edge_faces.get(key, [])
            local_faces = [fi for fi, _ in inc if fi in adjacency]
            for i in range(len(local_faces)):
                for j in range(i + 1, len(local_faces)):
                    a = local_faces[i]
                    b = local_faces[j]
                    adjacency[a].add(b)
                    adjacency[b].add(a)

        unseen = set(faces)
        fans = []
        while unseen:
            seed = unseen.pop()
            stack = [seed]
            fan = [seed]
            while stack:
                cur = stack.pop()
                for nb in adjacency[cur]:
                    if nb in unseen:
                        unseen.remove(nb)
                        stack.append(nb)
                        fan.append(nb)
            fans.append(fan)

        for fan_id, fan in enumerate(fans):
            new_idx = len(new_vertices)
            new_vertices.append(vertices[v].copy())
            if new_colors is not None:
                new_colors.append(colors[v].copy())

            for fi in fan:
                corner_map[(v, fi)] = new_idx

            if fan_id > 0:
                split_operations.append(
                    {
                        "original_vertex": int(v),
                        "new_vertex": int(new_idx),
                        "fan_index": int(fan_id),
                        "faces_in_fan": int(len(fan)),
                    }
                )

    new_triangles = np.empty_like(triangles)
    for fi, tri in enumerate(triangles):
        for j, v in enumerate(tri):
            new_triangles[fi, j] = corner_map[(int(v), fi)]

    return (
        np.asarray(new_vertices, dtype=np.float64),
        new_triangles.astype(np.int64),
        (np.asarray(new_colors, dtype=np.float64) if new_colors is not None else None),
        split_operations,
    )


# =============================================================================
# ORIENTABILIDAD
# =============================================================================


def orientation_constraints(triangles):
    """
    Retorna:
      constraints: (face_a, face_b, xor_required, edge)
      boundary_edges
      nonmanifold_edges
    """
    edge_faces = build_edge_incidence(triangles)

    constraints = []
    boundary = []
    nonmanifold = []

    for edge, inc in edge_faces.items():
        if len(inc) == 1:
            boundary.append(edge)
        elif len(inc) == 2:
            (f1, d1), (f2, d2) = inc

            # Si recorren la arista en el mismo sentido, una de las dos caras
            # debe invertirse: XOR = 1.
            xor_required = 1 if d1 == d2 else 0
            constraints.append((int(f1), int(f2), int(xor_required), edge))
        else:
            nonmanifold.append(
                {
                    "edge": [int(edge[0]), int(edge[1])],
                    "incident_faces": [int(x[0]) for x in inc],
                }
            )

    return constraints, boundary, nonmanifold


def propagate_orientation(triangles):
    """
    Produce una asignación binaria y lista todas las restricciones
    contradictorias respecto a esa asignación.

    Una vez eliminada al menos una cara de CADA conflicto, la asignación
    restringida a las caras restantes satisface todas las aristas restantes.
    """
    constraints, boundary, nonmanifold = orientation_constraints(triangles)

    adjacency = [[] for _ in range(len(triangles))]
    for ci, (a, b, required, edge) in enumerate(constraints):
        adjacency[a].append((b, required, ci))
        adjacency[b].append((a, required, ci))

    orientation = np.full(len(triangles), -1, dtype=np.int8)
    conflict_indices = []

    for start in range(len(triangles)):
        if orientation[start] >= 0:
            continue

        orientation[start] = 0
        q = deque([start])

        while q:
            u = q.popleft()
            for v, required, ci in adjacency[u]:
                expected = int(orientation[u]) ^ int(required)

                if orientation[v] < 0:
                    orientation[v] = expected
                    q.append(v)
                elif int(orientation[v]) != expected:
                    # Cada restricción se registra una sola vez.
                    a, b, _, _ = constraints[ci]
                    if a < b:
                        conflict_indices.append(ci)

    # Eliminar posibles duplicados.
    conflict_indices = sorted(set(conflict_indices))

    return {
        "constraints": constraints,
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "orientation": orientation,
        "conflict_indices": conflict_indices,
    }


def face_evidence(
    vertices,
    triangles,
    cloud_points,
    support,
    confidence,
):
    tree = cKDTree(cloud_points)
    d, s, c = cloud_evidence_for_vertices(vertices, cloud_points, support, confidence, tree)

    face_support = np.median(s[triangles], axis=1)
    face_confidence = np.median(c[triangles], axis=1)
    face_cloud_distance = np.median(d[triangles], axis=1)
    areas = triangle_areas(vertices, triangles)

    # Número de aristas abiertas de cada cara.
    edge_faces = build_edge_incidence(triangles)
    boundary_per_face = np.zeros(len(triangles), dtype=np.int8)

    for edge, inc in edge_faces.items():
        if len(inc) == 1:
            boundary_per_face[inc[0][0]] += 1

    return {
        "support": face_support,
        "confidence": face_confidence,
        "cloud_distance": face_cloud_distance,
        "area": areas,
        "boundary_edges": boundary_per_face,
    }


def orientation_removal_cost(evidence, args):
    """Combina soporte, confianza y geometría para ponderar la retirada de caras."""
    support = evidence["support"]
    confidence = evidence["confidence"]
    area = evidence["area"]
    boundary_edges = evidence["boundary_edges"]

    support_norm = np.clip(
        support / max(float(args.orientation_support_cap), 1e-9),
        0.0,
        1.0,
    )
    confidence_norm = np.clip(confidence, 0.0, 1.0)

    med_area = max(float(np.median(area)), 1e-12)
    area_norm = np.clip(area / med_area, 0.0, 3.0) / 3.0

    # Una cara completamente interior cuesta más retirar que una ya adyacente
    # a borde. Esto reduce la creación innecesaria de nuevos agujeros.
    interior = (boundary_edges == 0).astype(np.float64)

    cost = (
        0.20
        + float(args.orientation_support_weight) * support_norm
        + float(args.orientation_confidence_weight) * confidence_norm
        + float(args.orientation_area_weight) * area_norm
        + float(args.orientation_interior_weight) * interior
    )

    return np.asarray(cost, dtype=np.float64)


def greedy_weighted_conflict_cover(
    constraints,
    conflict_indices,
    removal_cost,
):
    """
    Cobertura aproximada de conflictos.

    Cada conflicto es una arista entre dos caras. Debemos retirar al menos una
    cara de cada arista conflictiva.

    Greedy:
        prioridad = conflictos_no_cubiertos / coste_de_retirar_cara

    No busca un cubo ni una forma: solo topología + evidencia.
    """
    face_to_conflicts = defaultdict(set)

    for local_id, ci in enumerate(conflict_indices):
        a, b, _, edge = constraints[ci]
        face_to_conflicts[a].add(local_id)
        face_to_conflicts[b].add(local_id)

    uncovered = set(range(len(conflict_indices)))
    selected = []
    trace = []

    while uncovered:
        best_face = None
        best_score = -np.inf
        best_coverage = 0

        for face, associated in face_to_conflicts.items():
            if face in selected:
                continue

            coverage = len(associated & uncovered)
            if coverage == 0:
                continue

            score = coverage / max(float(removal_cost[face]), 1e-9)

            if score > best_score or (
                math.isclose(score, best_score, rel_tol=1e-12) and coverage > best_coverage
            ):
                best_face = int(face)
                best_score = float(score)
                best_coverage = int(coverage)

        if best_face is None:
            raise RuntimeError(
                "No fue posible cubrir todas las contradicciones de " "orientabilidad."
            )

        selected.append(best_face)
        covered = face_to_conflicts[best_face] & uncovered
        uncovered -= covered

        trace.append(
            {
                "face": best_face,
                "new_conflicts_covered": int(len(covered)),
                "removal_cost": float(removal_cost[best_face]),
                "priority_score": float(best_score),
                "remaining_conflicts": int(len(uncovered)),
            }
        )

    return np.asarray(selected, dtype=np.int64), trace


@operacion("Reparar orientación topológica")
def repair_orientation(
    vertices,
    triangles,
    colors,
    cloud_points,
    support,
    confidence,
    args,
):
    diagnostic = propagate_orientation(triangles)

    if diagnostic["nonmanifold_edges"]:
        raise RuntimeError(
            "La reparación de orientabilidad recibió aristas con >2 caras. "
            "Primero debe resolverse edge-manifold."
        )

    initial_conflicts = len(diagnostic["conflict_indices"])

    if initial_conflicts == 0:
        return (
            vertices,
            triangles,
            colors,
            {
                "initial_conflicts": 0,
                "faces_removed": 0,
                "face_removal_ratio": 0.0,
                "records": [],
                "cover_trace": [],
            },
        )

    evidence = face_evidence(
        vertices,
        triangles,
        cloud_points,
        support,
        confidence,
    )
    cost = orientation_removal_cost(evidence, args)

    selected, trace = greedy_weighted_conflict_cover(
        diagnostic["constraints"],
        diagnostic["conflict_indices"],
        cost,
    )

    max_remove = int(math.ceil(float(args.max_orientation_face_removal_ratio) * len(triangles)))

    if len(selected) > max_remove:
        raise RuntimeError(
            "La reparación de orientabilidad necesitaría retirar "
            f"{len(selected)} caras ({len(selected)/len(triangles):.2%}), "
            "por encima del presupuesto configurado "
            f"({args.max_orientation_face_removal_ratio:.2%}). "
            "No se modifica la malla."
        )

    records = []
    for fi in selected:
        records.append(
            {
                "face_before_removal": int(fi),
                "support_median": float(evidence["support"][fi]),
                "confidence_median": float(evidence["confidence"][fi]),
                "cloud_distance_median_mm": float(evidence["cloud_distance"][fi]),
                "area_mm2": float(evidence["area"][fi]),
                "boundary_edges_before_removal": int(evidence["boundary_edges"][fi]),
                "removal_cost": float(cost[fi]),
            }
        )

    keep = np.ones(len(triangles), dtype=bool)
    keep[selected] = False
    triangles2 = triangles[keep]

    vertices2, triangles2, colors2 = compact_mesh_arrays(vertices, triangles2, colors)

    # Comprobación matemática antes de seguir.
    after = propagate_orientation(triangles2)
    if after["conflict_indices"]:
        raise RuntimeError(
            "La cobertura de conflictos no produjo una superficie "
            "orientable. Esto no debería ocurrir; se aborta para no "
            "continuar con una malla ambigua."
        )

    return (
        vertices2,
        triangles2,
        colors2,
        {
            "initial_conflicts": int(initial_conflicts),
            "faces_removed": int(len(selected)),
            "face_removal_ratio": float(len(selected) / max(len(triangles), 1)),
            "records": records,
            "cover_trace": trace,
        },
    )


def orient_faces_consistently(triangles):
    """Reorienta las caras una vez resueltas las restricciones contradictorias."""
    diagnostic = propagate_orientation(triangles)

    if diagnostic["conflict_indices"]:
        raise RuntimeError("Se intentó orientar una malla que todavía contiene " "contradicciones.")

    out = triangles.copy()
    flip_idx = np.flatnonzero(diagnostic["orientation"] == 1)

    if len(flip_idx):
        out[flip_idx] = out[flip_idx][:, [0, 2, 1]]

    return out, int(len(flip_idx))


# =============================================================================
# ANÁLISIS DE BORDES / TOPOLOGÍA
# =============================================================================


def boundary_components(vertices, triangles):
    """Agrupa las aristas de frontera y registra aristas no manifold."""
    edge_faces = build_edge_incidence(triangles)

    boundary_edges = [edge for edge, inc in edge_faces.items() if len(inc) == 1]
    nonmanifold_edges = [edge for edge, inc in edge_faces.items() if len(inc) > 2]

    adjacency = defaultdict(list)
    for a, b in boundary_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    seen = set()
    records = []

    for seed in adjacency:
        if seed in seen:
            continue

        stack = [seed]
        seen.add(seed)
        verts = []

        while stack:
            u = stack.pop()
            verts.append(u)
            for nb in adjacency[u]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)

        degrees = [len(adjacency[v]) for v in verts]
        comp_set = set(verts)

        perimeter = 0.0
        edge_count = 0
        for a, b in boundary_edges:
            if a in comp_set and b in comp_set:
                perimeter += float(np.linalg.norm(vertices[a] - vertices[b]))
                edge_count += 1

        pts = vertices[np.asarray(verts, dtype=np.int64)]
        extent = np.ptp(pts, axis=0) if len(pts) else np.zeros(3)

        records.append(
            {
                "boundary_id": int(len(records)),
                "vertices": int(len(verts)),
                "edges": int(edge_count),
                "simple_closed_loop": bool(verts and all(d == 2 for d in degrees)),
                "perimeter_mm": float(perimeter),
                "diameter_bbox_mm": float(np.linalg.norm(extent)),
                "degree_min": int(min(degrees)) if degrees else None,
                "degree_max": int(max(degrees)) if degrees else None,
            }
        )

    return records, boundary_edges, nonmanifold_edges


def final_component_summary(vertices, triangles):
    """Resume cantidad de triángulos y área de cada componente final."""
    count, labels = face_components(triangles)
    areas = triangle_areas(vertices, triangles)

    if count == 0:
        return {
            "count": 0,
            "triangle_counts": [],
            "areas_mm2": [],
            "largest_triangle_ratio": None,
            "largest_area_ratio": None,
        }

    tri_counts = np.bincount(labels, minlength=count)
    comp_areas = np.bincount(
        labels,
        weights=areas,
        minlength=count,
    )

    return {
        "count": int(count),
        "triangle_counts": sorted(
            [int(x) for x in tri_counts],
            reverse=True,
        ),
        "areas_mm2": sorted(
            [float(x) for x in comp_areas],
            reverse=True,
        ),
        "largest_triangle_ratio": float(np.max(tri_counts) / max(np.sum(tri_counts), 1)),
        "largest_area_ratio": float(np.max(comp_areas) / max(np.sum(comp_areas), 1e-12)),
    }


def semantic_mesh(vertices, triangles):
    """Construye una malla auxiliar para comprobar pares de intersección."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    return mesh


def semantic_intersection_analysis(
    vertices,
    triangles,
    spacing,
    args,
):
    """Clasifica los pares raw reportados por Open3D."""
    mesh = semantic_mesh(vertices, triangles)

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
        float(args.semantic_geometric_epsilon_spacing_factor) * float(spacing),
    )
    contact_tol = max(
        10.0 * eps,
        float(args.semantic_contact_locality_spacing_factor) * float(spacing),
    )

    p = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)

    counts = Counter()
    blocking_pairs = []
    records = []

    for pair_index, (fa, fb) in enumerate(pairs):
        fa = int(fa)
        fb = int(fb)

        result = classify_triangle_pair(
            p[t[fa]],
            p[t[fb]],
            triangle_indices_a=t[fa],
            triangle_indices_b=t[fb],
            geometric_epsilon=eps,
            contact_locality_tolerance=contact_tol,
        )

        category = str(
            result.get(
                "category",
                "numerically_ambiguous",
            )
        )
        blocking = category in BLOCKING_INTERSECTIONS
        counts[category] += 1

        records.append(
            {
                "pair_index": int(pair_index),
                "face_a": fa,
                "face_b": fb,
                "category": category,
                "blocking": bool(blocking),
            }
        )

        if blocking:
            blocking_pairs.append((fa, fb, category))

    true_categories = {
        "transverse_intersection",
        "coplanar_overlap",
        "unexplained_coplanar_contact",
    }
    ambiguous_categories = {
        "numerically_ambiguous",
        "degenerate_ambiguous",
    }

    return {
        "raw_pair_count": int(len(pairs)),
        "true_self_intersection_pairs": int(sum(counts[c] for c in true_categories)),
        "ambiguous_intersection_pairs": int(sum(counts[c] for c in ambiguous_categories)),
        "nonblocking_contact_pairs": int(sum(counts[c] for c in ALLOWED_CONTACTS)),
        "categories": dict(counts),
        "blocking_pairs": blocking_pairs,
        "records": records,
        "geometric_epsilon_mm": float(eps),
        "contact_locality_tolerance_mm": float(contact_tol),
    }


def semantic_conflict_cover(
    blocking_pairs,
    removal_cost,
):
    """Retira un conjunto pequeño de caras que cubra todos los pares."""
    if not blocking_pairs:
        return np.empty(0, dtype=np.int64), []

    face_to_conflicts = defaultdict(set)

    for conflict_id, (fa, fb, category) in enumerate(blocking_pairs):
        face_to_conflicts[int(fa)].add(conflict_id)
        face_to_conflicts[int(fb)].add(conflict_id)

    uncovered = set(range(len(blocking_pairs)))
    selected = []
    trace = []

    while uncovered:
        best_face = None
        best_score = -np.inf
        best_coverage = 0

        for face, associated in face_to_conflicts.items():
            coverage = len(associated & uncovered)
            if coverage <= 0:
                continue

            score = coverage / max(
                float(removal_cost[int(face)]),
                1e-9,
            )

            if score > best_score or (
                math.isclose(
                    score,
                    best_score,
                    rel_tol=1e-12,
                )
                and coverage > best_coverage
            ):
                best_face = int(face)
                best_score = float(score)
                best_coverage = int(coverage)

        if best_face is None:
            raise RuntimeError("No fue posible cubrir los conflictos semánticos.")

        covered = face_to_conflicts[best_face] & uncovered
        uncovered -= covered
        selected.append(best_face)

        trace.append(
            {
                "face": int(best_face),
                "new_conflicts_covered": int(len(covered)),
                "removal_cost": float(removal_cost[best_face]),
                "priority_score": float(best_score),
                "remaining_conflicts": int(len(uncovered)),
            }
        )

    return (
        np.asarray(
            sorted(set(selected)),
            dtype=np.int64,
        ),
        trace,
    )


@operacion("Resolver intersecciones de la malla")
def repair_semantic_intersections(
    vertices,
    triangles,
    colors,
    cloud_points,
    support,
    confidence,
    spacing,
    args,
    *,
    already_removed_faces=0,
):
    """Resuelve cruces B/C usando el mismo presupuesto topológico del 2 %."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(triangles, dtype=np.int64)
    c = np.asarray(colors, dtype=np.float64) if colors is not None else None

    reference_face_count = int(len(f))
    budget = int(math.ceil(float(args.max_orientation_face_removal_ratio) * reference_face_count))

    removed_semantic = 0
    iterations = []

    for iteration in range(
        1,
        max(1, int(args.semantic_repair_max_iterations)) + 1,
    ):
        analysis = semantic_intersection_analysis(
            v,
            f,
            spacing,
            args,
        )

        blocking_pairs = analysis["blocking_pairs"]

        if not blocking_pairs:
            public = {
                k: value for k, value in analysis.items() if k not in ("records", "blocking_pairs")
            }
            return (
                v,
                f,
                c,
                {
                    "status": "clean",
                    "faces_removed": int(removed_semantic),
                    "total_removed_with_prior": int(already_removed_faces + removed_semantic),
                    "budget_faces": int(budget),
                    "iterations": iterations,
                    "final": public,
                },
            )

        evidence = face_evidence(
            v,
            f,
            cloud_points,
            support,
            confidence,
        )
        cost = orientation_removal_cost(
            evidence,
            args,
        )

        selected, trace = semantic_conflict_cover(
            blocking_pairs,
            cost,
        )

        projected = int(already_removed_faces) + int(removed_semantic) + int(len(selected))

        if projected > budget:
            public = {
                k: value for k, value in analysis.items() if k not in ("records", "blocking_pairs")
            }
            return (
                v,
                f,
                c,
                {
                    "status": "budget_exceeded",
                    "faces_removed": int(removed_semantic),
                    "total_removed_with_prior": int(already_removed_faces + removed_semantic),
                    "budget_faces": int(budget),
                    "required_additional_faces": int(len(selected)),
                    "iterations": iterations,
                    "final": public,
                },
            )

        keep = np.ones(len(f), dtype=bool)
        keep[selected] = False
        f = f[keep]

        v, f, c = compact_mesh_arrays(
            v,
            f,
            c,
        )

        # La retirada de una cara puede separar fans.
        v, f, c, split_ops = split_vertex_fans(
            v,
            f,
            c,
        )
        f, flipped = orient_faces_consistently(f)

        removed_semantic += int(len(selected))

        iterations.append(
            {
                "iteration": int(iteration),
                "blocking_pairs_before": int(len(blocking_pairs)),
                "true_before": int(analysis["true_self_intersection_pairs"]),
                "ambiguous_before": int(analysis["ambiguous_intersection_pairs"]),
                "faces_selected_count": int(len(selected)),
                "selected_faces": selected.astype(int).tolist(),
                "split_vertices_after_removal": int(len(split_ops)),
                "orientation_flips_after_removal": int(flipped),
                "cover_trace": trace,
            }
        )

    final_analysis = semantic_intersection_analysis(
        v,
        f,
        spacing,
        args,
    )
    public = {
        k: value for k, value in final_analysis.items() if k not in ("records", "blocking_pairs")
    }

    return (
        v,
        f,
        c,
        {
            "status": ("clean" if not final_analysis["blocking_pairs"] else "unresolved"),
            "faces_removed": int(removed_semantic),
            "total_removed_with_prior": int(already_removed_faces + removed_semantic),
            "budget_faces": int(budget),
            "iterations": iterations,
            "final": public,
        },
    )


def topology_snapshot(mesh):
    """Obtiene un resumen de las comprobaciones topológicas de Open3D."""
    out = {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
    }

    checks = (
        (
            "edge_manifold_allow_boundary",
            lambda: mesh.is_edge_manifold(True),
        ),
        (
            "edge_manifold_closed",
            lambda: mesh.is_edge_manifold(False),
        ),
        ("vertex_manifold", mesh.is_vertex_manifold),
        ("orientable", mesh.is_orientable),
        ("self_intersecting", mesh.is_self_intersecting),
        ("watertight", mesh.is_watertight),
    )

    for key, fn in checks:
        try:
            out[key] = bool(fn())
        except Exception:
            out[key] = None

    try:
        out["non_manifold_vertex_count"] = int(len(mesh.get_non_manifold_vertices()))
    except Exception:
        out["non_manifold_vertex_count"] = None

    try:
        out["non_manifold_edge_count_allow_boundary"] = int(
            len(mesh.get_non_manifold_edges(allow_boundary_edges=True))
        )
    except Exception:
        out["non_manifold_edge_count_allow_boundary"] = None

    return out


# =============================================================================
# EXPORTS
# =============================================================================


def write_csv(path, records, fields):
    """Escribe los registros con cabecera y el orden de columnas indicado."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k) for k in fields})


def preview(
    path,
    vertices,
    boundary_records,
    title,
    status,
    orientation_info,
    topology,
    closure_info=None,
    semantic_info=None,
    total_removed=0,
):
    points = np.asarray(vertices, dtype=np.float64)

    if len(points) > 120000:
        idx = np.linspace(0, len(points) - 1, 120000).astype(int)
        shown = points[idx]
    else:
        shown = points

    boundary_ids = set()
    # Reconstruimos los vértices de borde a partir de la geometría en la
    # función principal y se pasan en una variable global local.
    # Aquí se obtiene de boundary vertex list adjunta a cada registro si existe.
    for rec in boundary_records:
        for v in rec.get("_vertex_ids", []):
            boundary_ids.add(int(v))

    if boundary_ids:
        bp = points[np.asarray(sorted(boundary_ids), dtype=np.int64)]
    else:
        bp = np.empty((0, 3))

    fig = plt.figure(figsize=(15, 10))

    projections = (
        (1, (0, 1), "X-Y"),
        (2, (0, 2), "X-Z"),
        (3, (2, 1), "Z-Y"),
    )

    for subplot, dims, name in projections:
        ax = fig.add_subplot(2, 2, subplot)
        ax.scatter(
            shown[:, dims[0]],
            shown[:, dims[1]],
            s=0.25,
            c="0.72",
        )
        if len(bp):
            ax.scatter(
                bp[:, dims[0]],
                bp[:, dims[1]],
                s=1.2,
                c="red",
            )
        ax.set_title(f"{name} | rojo = bordes abiertos")
        ax.set_aspect("equal", adjustable="box")

    ax = fig.add_subplot(2, 2, 4)
    ax.axis("off")
    lines = [
        title,
        f"Estado: {status}",
        f"Vértices: {len(vertices):,}",
        f"Conflictos iniciales: " f"{orientation_info['initial_conflicts']}",
        f"Caras retiradas para orientabilidad: " f"{orientation_info['faces_removed']}",
        f"Orientable: {topology.get('orientable')}",
        f"Vertex-manifold: {topology.get('vertex_manifold')}",
        f"Caras retiradas en todas las reparaciones: {total_removed}",
        f"Intersecciones reales: {(semantic_info or {}).get('true_self_intersection_pairs', '?')}",
        f"Contactos no bloqueantes: {(semantic_info or {}).get('nonblocking_contact_pairs', '?')}",
        f"Huecos reparados: {(closure_info or {}).get('small_loops_closed', 0)}; "
        f"tapas estimadas: {(closure_info or {}).get('estimated_terminal_caps_closed', 0)}",
    ]
    ax.text(
        0.04,
        0.95,
        "\n\n".join(lines),
        va="top",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================


def _closure_edge_keys(vertices, triangles, quantum):
    """Procedencia geométrica: resiste la compactación y separación de fans."""

    def key(a, b):
        ends = [tuple(np.rint(vertices[i] / quantum).astype(np.int64)) for i in (a, b)]
        return tuple(sorted(ends))

    return {key(a, b) for (a, b), inc in build_edge_incidence(triangles).items() if len(inc) == 1}


def _closure_loops(triangles):
    """Recorre las aristas de frontera para identificar contornos cerrados."""
    incidence = build_edge_incidence(triangles)
    adjacency = defaultdict(list)
    for (a, b), inc in incidence.items():
        if len(inc) == 1:
            adjacency[a].append(b)
            adjacency[b].append(a)
    seen, loops, invalid = set(), [], 0
    for seed in sorted(adjacency):
        if seed in seen:
            continue
        stack, comp = [seed], set([seed])
        while stack:
            for nb in adjacency[stack.pop()]:
                if nb not in comp:
                    comp.add(nb)
                    stack.append(nb)
        seen.update(comp)
        if len(comp) < 3 or any(len(adjacency[i]) != 2 for i in comp):
            invalid += 1
            continue
        # Recorrido contrario a la cara existente: orientación de la tapa.
        a, b = seed, adjacency[seed][0]
        direction = incidence[tuple(sorted((a, b)))][0][1]
        if (1 if a < b else -1) == direction:
            b = adjacency[a][1]
        loop, previous, current = [a], a, b
        while current != a and len(loop) <= len(comp):
            loop.append(current)
            nxt = [x for x in adjacency[current] if x != previous][0]
            previous, current = current, nxt
        if current != a or len(loop) != len(comp):
            invalid += 1
            continue
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops, invalid


def _closure_cross(a, b):
    """Calcula el producto cruzado escalar de vectores bidimensionales."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _closure_chart(points):
    """Proyecta el contorno en un marco local y rechaza configuraciones colineales."""
    center = points.mean(axis=0)
    _, singular, basis = np.linalg.svd(points - center, full_matrices=False)
    if len(singular) < 3 or singular[1] < 1e-10:
        raise ValueError("contorno_colineal")
    xy = (points - center) @ basis[:2].T
    area = np.sum(_closure_cross(xy, np.roll(xy, -1, axis=0))) / 2
    if area < 0:
        basis[1] *= -1
        xy[:, 1] *= -1
    normal = np.cross(basis[0], basis[1])
    return center, basis[:2], normal, xy


def _closure_earclip(xy):
    """Triangulación restringida a un polígono simple, incluso cóncavo."""
    n = len(xy)
    eps = max(1e-14, float(np.ptp(xy, axis=0).max()) ** 2 * 1e-12)
    # Rechazar cruces y contactos no adyacentes en la proyección.
    for i in range(n):
        a, b = xy[i], xy[(i + 1) % n]
        for j in range(i + 1, n):
            if j == (i + 1) % n or (j + 1) % n == i:
                continue
            c, d = xy[j], xy[(j + 1) % n]
            if np.any(
                np.maximum(np.minimum(a, b), np.minimum(c, d))
                > np.minimum(np.maximum(a, b), np.maximum(c, d)) + np.sqrt(eps) * 1e-3
            ):
                continue
            ab = [_closure_cross(b - a, c - a), _closure_cross(b - a, d - a)]
            cd = [_closure_cross(d - c, a - c), _closure_cross(d - c, b - c)]
            if min(ab) <= eps and max(ab) >= -eps and min(cd) <= eps and max(cd) >= -eps:
                raise ValueError("contorno_proyectado_se_cruza")
    active, faces = list(range(n)), []
    while len(active) > 3:
        ears = []
        for pos, b in enumerate(active):
            a, c = active[pos - 1], active[(pos + 1) % len(active)]
            pa, pb, pc = xy[[a, b, c]]
            area = float(_closure_cross(pb - pa, pc - pa))
            if area <= eps:
                continue
            other = [x for x in active if x not in (a, b, c)]
            q = xy[other]
            inside = (
                (_closure_cross(pb - pa, q - pa) >= -eps)
                & (_closure_cross(pc - pb, q - pb) >= -eps)
                & (_closure_cross(pa - pc, q - pc) >= -eps)
            )
            if np.any(inside):
                continue
            quality = area / max(
                np.sum((pa - pb) ** 2) + np.sum((pb - pc) ** 2) + np.sum((pc - pa) ** 2), eps
            )
            ears.append((quality, pos, (a, b, c)))
        if not ears:
            raise ValueError("triangulacion_restringida_no_resuelta")
        _, pos, face = max(ears)
        faces.append(face)
        active.pop(pos)
    if _closure_cross(xy[active[1]] - xy[active[0]], xy[active[2]] - xy[active[0]]) <= eps:
        raise ValueError("triangulo_final_degenerado")
    faces.append(tuple(active))
    return np.asarray(faces, dtype=np.int64)


def _patch_quality(p, t):
    """Calcula una calidad adimensional de triángulos a partir de área y aristas."""
    q = p[t]
    area2 = np.linalg.norm(np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), axis=1)
    lengths2 = np.sum((q - np.roll(q, -1, axis=1)) ** 2, axis=2)
    return 2 * np.sqrt(3) * area2 / np.maximum(lengths2.sum(axis=1), 1e-30)


def _improve_local_diagonals(p, triangles, allowed, spacing, passes=4):
    """Cambios de diagonal, sin mover vértices ni cruzar aristas reales."""
    t = triangles.copy()
    changes = 0
    for _ in range(passes):
        incidence = build_edge_incidence(t)
        used, count = set(), 0
        for (u, v), inc in list(incidence.items()):
            if len(inc) != 2:
                continue
            f, g = inc[0][0], inc[1][0]
            if not (allowed[f] and allowed[g]) or f in used or g in used:
                continue
            a, b = (u, v) if inc[0][1] == 1 else (v, u)
            c = next(int(x) for x in t[f] if x not in (a, b))
            d = next(int(x) for x in t[g] if x not in (a, b))
            if c == d or tuple(sorted((c, d))) in incidence:
                continue
            old = t[[f, g]]
            new = np.asarray([(c, d, b), (d, c, a)], dtype=np.int64)
            oq, nq = _patch_quality(p, old), _patch_quality(p, new)
            if nq.min() <= max(oq.min() * 1.08, oq.min() + 0.015):
                continue
            xyz = p[old]
            on = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
            on /= np.maximum(np.linalg.norm(on, axis=1)[:, None], 1e-30)
            if np.dot(on[0], on[1]) < np.cos(np.deg2rad(25)):
                continue
            # Limita cuánto cambia la superficie no plana bajo la diagonal.
            if (
                max(abs(np.dot(p[d] - p[a], on[0])), abs(np.dot(p[c] - p[b], on[1])))
                > 0.15 * spacing
            ):
                continue
            xyz = p[new]
            nn = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
            nl = np.linalg.norm(nn, axis=1)
            if np.any(nl <= spacing**2 * 1e-10):
                continue
            nn /= nl[:, None]
            if np.min(nn @ on.T) < np.cos(np.deg2rad(35)):
                continue
            # Solo caras disjuntas en cada barrido: la incidencia no queda obsoleta.
            t[[f, g]] = new
            incidence[tuple(sorted((c, d)))] = [(f, 1), (g, -1)]
            used.update((f, g))
            count += 1
        changes += count
        if not count:
            break
    return t, changes


def _inside_polygon(points, rim):
    """Determina pertenencia al polígono mediante cruces de un rayo horizontal."""
    inside = np.zeros(len(points), dtype=bool)
    x, y = points[:, 0], points[:, 1]
    for a, b in zip(rim, np.roll(rim, -1, axis=0)):
        if abs(b[1] - a[1]) <= 1e-15:
            continue
        cross = ((a[1] > y) != (b[1] > y)) & (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0])
        inside ^= cross
    return inside


def _isotropic_terminal_patch(vertices, loop, spacing):
    """Dominio poligonal restringido; malla interior sin anillos radiales.

    Delaunay se acepta solo si conserva cada segmento del contorno y cubre
    exactamente el dominio. Alternativa: orejas restringidas y refinamiento.
    La altura interior es armónica, con posiciones exactas en la costura.
    """
    from scipy.spatial import Delaunay, QhullError
    from scipy.sparse.linalg import spsolve

    rim = vertices[loop]
    center, basis, normal, xy = _closure_chart(rim)
    ears = _closure_earclip(xy)
    n = len(loop)
    pitch = 1.5 * spacing
    extent = np.ptp(xy, axis=0)
    # Límite de trabajo ligado al área, no al número de vistas ni a una figura.
    pitch = max(pitch, float(np.sqrt(max(extent.prod(), 0) / 6000)))
    yy = np.arange(xy[:, 1].min() + pitch / 2, xy[:, 1].max(), pitch * np.sqrt(3) / 2)
    rows = []
    for j, y in enumerate(yy):
        xx = np.arange(xy[:, 0].min() + pitch / 2 + (j % 2) * pitch / 2, xy[:, 0].max(), pitch)
        rows.append(np.column_stack((xx, np.full(len(xx), y))))
    grid = np.vstack(rows) if rows else np.empty((0, 2))
    grid = grid[_inside_polygon(grid, xy)]
    # Separación respecto a segmentos completos, no solo sus vértices.
    dist2 = np.full(len(grid), np.inf)
    for a, b in zip(xy, np.roll(xy, -1, axis=0)):
        e = b - a
        lam = np.clip((grid - a) @ e / max(float(e @ e), 1e-30), 0, 1)
        dist2 = np.minimum(dist2, np.sum((grid - a - lam[:, None] * e) ** 2, axis=1))
    grid = grid[dist2 > (0.40 * pitch) ** 2]
    uv = np.vstack((xy, grid))
    wanted = {tuple(sorted((j, (j + 1) % n))) for j in range(n)}
    faces = None
    try:
        candidate = Delaunay(uv).simplices.copy()
        signed = _closure_cross(
            uv[candidate[:, 1]] - uv[candidate[:, 0]], uv[candidate[:, 2]] - uv[candidate[:, 0]]
        )
        candidate[signed < 0] = candidate[signed < 0][:, [0, 2, 1]]
        candidate = candidate[_inside_polygon(uv[candidate].mean(axis=1), xy)]
        inc = build_edge_incidence(candidate)
        boundary = {edge for edge, x in inc.items() if len(x) == 1}
        area = (
            _closure_cross(
                uv[candidate[:, 1]] - uv[candidate[:, 0]], uv[candidate[:, 2]] - uv[candidate[:, 0]]
            ).sum()
            / 2
        )
        expected = _closure_cross(xy, np.roll(xy, -1, axis=0)).sum() / 2
        if (
            boundary == wanted
            and all(len(x) <= 2 for x in inc.values())
            and abs(area - expected) <= max(1e-8, expected * 1e-8)
        ):
            faces = candidate
    except QhullError:
        pass
    method = "constrained_verified_delaunay"
    if faces is None:
        method = "earclip_refinement_with_local_diagonal_optimization"
        uv, faces = xy.copy(), ears.copy()
        for _ in range(7):
            area = (
                np.abs(
                    _closure_cross(
                        uv[faces[:, 1]] - uv[faces[:, 0]], uv[faces[:, 2]] - uv[faces[:, 0]]
                    )
                )
                / 2
            )
            ids = np.flatnonzero(area > 0.65 * pitch**2)
            if not len(ids) or len(faces) + 2 * len(ids) > 18000:
                break
            mids = np.arange(len(uv), len(uv) + len(ids))
            add = uv[faces[ids]].mean(axis=1)
            old = faces[ids]
            keep = np.ones(len(faces), bool)
            keep[ids] = False
            faces = np.vstack(
                (
                    faces[keep],
                    np.column_stack((old[:, 0], old[:, 1], mids)),
                    np.column_stack((old[:, 1], old[:, 2], mids)),
                    np.column_stack((old[:, 2], old[:, 0], mids)),
                )
            )
            uv = np.vstack((uv, add))
            faces, _ = _improve_local_diagonals(
                np.column_stack((uv, np.zeros(len(uv)))),
                faces,
                np.ones(len(faces), bool),
                spacing,
                passes=3,
            )
    # Suprimir únicamente vértices interiores sin uso; conservar TODOS los del rim.
    used = np.r_[np.arange(n), np.setdiff1d(np.unique(faces), np.arange(n))]
    remap = np.full(len(uv), -1, dtype=int)
    remap[used] = np.arange(len(used))
    uv, faces = uv[used], remap[faces]
    edges = np.asarray(list(build_edge_incidence(faces)), dtype=int)
    lengths = np.linalg.norm(uv[edges[:, 0]] - uv[edges[:, 1]], axis=1)
    weight = 1 / np.maximum(lengths, 0.15 * pitch)
    a, b = edges.T
    graph = coo_matrix(
        (np.r_[weight, weight], (np.r_[a, b], np.r_[b, a])), shape=(len(uv), len(uv))
    ).tocsr()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    from scipy.sparse import diags

    lap = diags(degree) - graph
    height = np.zeros(len(uv))
    height[:n] = (rim - center) @ normal
    if len(uv) > n:
        height[n:] = spsolve(lap[n:, n:].tocsc(), -lap[n:, :n] @ height[:n])
    if not np.all(np.isfinite(height)) or (
        len(height) > n
        and (
            height[n:].min() < height[:n].min() - 1e-6 or height[n:].max() > height[:n].max() + 1e-6
        )
    ):
        raise ValueError("interpolacion_armonica_no_acotada")
    xyz = center + uv @ basis + height[:, None] * normal
    xyz[:n] = rim
    mapping = np.r_[loop, np.arange(len(vertices), len(vertices) + len(uv) - n)]
    patch = mapping[faces]
    return (
        xyz[n:],
        patch,
        normal,
        {
            "method": method,
            "target_spacing_mm": pitch,
            "triangle_quality": robust_stats(_patch_quality(xyz, faces)),
            "rim_vertices_fixed_during_cap_meshing": n,
        },
    )


def _terminal_band(p, t, loop, width):
    """Vecindad geodésica; nunca unir capas solo por cercanía euclídea."""
    import heapq

    adjacency = defaultdict(list)
    for a, b in build_edge_incidence(t):
        length = float(np.linalg.norm(p[a] - p[b]))
        adjacency[a].append((b, length))
        adjacency[b].append((a, length))
    distance = np.full(len(p), np.inf)
    distance[loop] = 0.0
    queue = [(0.0, int(i)) for i in loop]
    heapq.heapify(queue)
    while queue:
        d, a = heapq.heappop(queue)
        if d != distance[a]:
            continue
        for b, length in adjacency[a]:
            nd = d + length
            if nd <= width and nd < distance[b]:
                distance[b] = nd
                heapq.heappush(queue, (nd, b))
    return distance, adjacency


def _regularize_weak_terminal_rim(p, t, loop, cloud, confidence, support, spacing, args):
    """Curva robusta conjunta + collar armónico anclado a observaciones.

    El presupuesto se mide SIEMPRE contra la entrada de esta operación.
    No se ajusta un círculo, cilindro ni un plano obligatorio. La altura local
    es una coordenada de trabajo; los radios se deducen del muestreo del borde.
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    center, basis, normal, uv = _closure_chart(p[loop])
    rim = p[loop]
    height = (rim - center) @ normal
    valid = np.all(np.isfinite(cloud), axis=1) & np.isfinite(confidence) & np.isfinite(support)
    cp, cf, cs = cloud[valid], confidence[valid], support[valid]
    info = {
        "method": "joint_robust_curve_with_anchored_harmonic_wall_collar",
        "uncertainty_kind": "local_cloud_dispersion_proxy",
        "applied": False,
        "moved_vertices": 0,
        "moved_rim_vertices": 0,
        "fixed_rim_mask": [True] * len(loop),
    }
    if len(cp) < 12:
        info["reason"] = "insufficient_cloud"
        return p.copy(), info
    tree = cKDTree(cp)
    dd, ii = tree.query(rim, k=min(16, len(cp)), workers=query_threads())
    sigma = np.clip(
        1.4826 * np.median(np.abs((cp[ii] - cp[ii[:, :1]]) @ normal), axis=1),
        0.10 * spacing,
        0.60 * spacing,
    )
    anchored = (cf[ii[:, 0]] >= 0.80) & (cs[ii[:, 0]] >= 4) & (dd[:, 0] <= spacing + sigma)
    duplicate_d, _ = cKDTree(p).query(rim, k=2, workers=query_threads())
    anchored |= duplicate_d[:, 1] < max(spacing * 1e-3, 1e-8)
    # Proteger esquinas del contorno EN SU PLANO; la ondulación vertical no
    # se declara arista real solo porque tenga una normal ruidosa.
    prev = uv - np.roll(uv, 1, axis=0)
    nex = np.roll(uv, -1, axis=0) - uv
    lengths = np.linalg.norm(prev, axis=1) * np.linalg.norm(nex, axis=1)
    turn = np.sum(prev * nex, axis=1) / np.maximum(lengths, 1e-20)
    corners = (turn < np.cos(np.deg2rad(40))) | (lengths < spacing**2 * 1e-6)
    anchored |= corners
    segment = np.linalg.norm(rim - np.roll(rim, -1, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(segment[:-1])]
    perimeter = float(segment.sum())
    if perimeter <= spacing or len(loop) < 8:
        info["reason"] = "rim_too_small_for_joint_fit"
        return p.copy(), info
    rim_spacing = float(np.median(segment[segment > spacing * 1e-3]))
    radii = (
        min(max(4 * spacing, 3 * rim_spacing), 0.20 * perimeter),
        min(max(7 * spacing, 5 * rim_spacing), 0.30 * perimeter),
    )
    target = np.zeros(len(loop))
    evidence = np.zeros(len(loop))
    for j in range(len(loop)):
        if anchored[j]:
            continue
        s = (arc - arc[j] + perimeter / 2) % perimeter - perimeter / 2
        predictions, scales = [], []
        for radius in radii:
            near = (abs(s) <= radius) & (np.arange(len(loop)) != j)
            if near.sum() < 5 or not (np.any(s[near] < 0) and np.any(s[near] > 0)):
                break
            x = s[near] / radius
            A = np.column_stack((np.ones(len(x)), x, x * x))
            weights = np.exp(-2 * x * x) * (0.4 + 0.6 * cf[ii[near, 0]])
            w = weights.copy()
            for _ in range(5):
                coef, _, rank, _ = np.linalg.lstsq(
                    A * np.sqrt(w[:, None]), height[near] * np.sqrt(w), rcond=None
                )
                if rank < 3:
                    break
                residual = height[near] - A @ coef
                scale = max(0.05 * spacing, 1.4826 * np.median(abs(residual - np.median(residual))))
                w = weights * np.minimum(1.0, 1.5 * scale / np.maximum(abs(residual), 1e-12))
            if rank < 3:
                break
            predictions.append(float(coef[0] - height[j]))
            scales.append(scale)
        if len(predictions) != 2:
            continue
        a, b = predictions
        # Exigir acuerdo de escala y evidencia local; incertidumbre NO implica
        # automáticamente que un vértice sea erróneo ni autoriza a aplanarlo.
        if a * b <= 0 or abs(a - b) > max(0.35 * spacing, 0.65 * min(abs(a), abs(b))):
            continue
        if min(abs(a), abs(b)) <= max(0.10 * spacing, 0.5 * max(scales)):
            continue
        local = (dd[j] <= max(radii[0], 3 * spacing)) & (cf[ii[j]] >= 0.65) & (cs[ii[j]] >= 2)
        if np.count_nonzero(local) < 6:
            continue
        target[j] = np.sign(a) * min(abs(a), abs(b))
        evidence[j] = 1.0 / (1.0 + (max(scales) / spacing) ** 2)
    active = evidence > 0
    info.update(
        fit_radii_mm=list(radii),
        strong_or_coincident_anchors=int(anchored.sum()),
        geometric_corner_anchors=int(corners.sum()),
        uncertainty_mm=robust_stats(sigma),
        confirmed_rim_vertices=int(active.sum()),
        fixed_rim_mask=(~active | anchored).tolist(),
    )
    if not np.any(active):
        info["reason"] = "no_weak_deviation_confirmed"
        return p.copy(), info
    # Resolver TODOS los desplazamientos confirmados conjuntamente. Dirichlet
    # cero fuera del tramo confirmado impide propagar una inferencia sin respaldo.
    n = len(loop)
    j = np.arange(n)
    k = (j + 1) % n
    weight = np.clip(rim_spacing / np.maximum(segment, 0.15 * rim_spacing), 0.2, 5.0)
    L = coo_matrix(
        (np.r_[weight, weight, -weight, -weight], (np.r_[j, k, j, k], np.r_[j, k, k, j])),
        shape=(n, n),
    ).tocsr()
    ids = np.flatnonzero(active & ~anchored)
    system = diags(evidence[ids] + 0.05) + 0.45 * L[ids][:, ids]
    solved = spsolve(system.tocsc(), evidence[ids] * target[ids])
    if not np.all(np.isfinite(solved)):
        info["reason"] = "nonfinite_joint_curve_fit"
        return p.copy(), info
    budgets = np.minimum(args.rim_max_shift_spacing * spacing, 2 * sigma)
    info["rim_total_budget_mm"] = budgets.tolist()
    curve_delta = np.zeros(n)
    # No invertir el sentido de una corrección confirmada por las dos escalas.
    curve_delta[ids] = np.sign(target[ids]) * np.minimum(
        np.maximum(solved * np.sign(target[ids]), 0), budgets[ids]
    )
    width = max(8 * spacing, min(4 * rim_spacing, 12 * spacing))
    distance, adjacency = _terminal_band(p, t, loop, width)
    band = np.flatnonzero(np.isfinite(distance))
    db, ib = tree.query(p[band], workers=query_threads())
    strong = (cf[ib] >= 0.80) & (cs[ib] >= 4) & (db <= spacing)
    fixed = np.ones(len(p), dtype=bool)
    fixed[band] = strong | (distance[band] >= 0.80 * width)
    # Toda costura topológica o abertura distinta de este contorno queda fija.
    for (a, b), inc in build_edge_incidence(t).items():
        if len(inc) != 2:
            fixed[[a, b]] = True
    near_duplicates, _ = cKDTree(p).query(p[band], k=2, workers=query_threads())
    fixed[band[near_duplicates[:, 1] < max(spacing * 1e-3, 1e-8)]] = True
    fixed[loop] = True
    scalar = np.zeros(len(p))
    scalar[loop] = curve_delta
    free = np.flatnonzero(~fixed)
    if len(free):
        lookup = np.full(len(p), -1, dtype=int)
        lookup[free] = np.arange(len(free))
        rows, cols, values, rhs = [], [], [], np.zeros(len(free))
        for row, a in enumerate(free):
            total = 0.10  # anclaje suave a geometría original; sistema definido positivo
            for b, length in adjacency[a]:
                w = spacing / max(length, 0.15 * spacing)
                total += w
                if lookup[b] >= 0:
                    rows.append(row)
                    cols.append(lookup[b])
                    values.append(-w)
                else:
                    rhs[row] += w * scalar[b]
            rows.append(row)
            cols.append(row)
            values.append(total)
        H = coo_matrix((values, (rows, cols)), shape=(len(free), len(free))).tocsc()
        scalar[free] = spsolve(H, rhs)
    if not np.all(np.isfinite(scalar)):
        info["reason"] = "nonfinite_collar_fit"
        return p.copy(), info
    delta = scalar[:, None] * normal
    moving = np.flatnonzero(np.linalg.norm(delta, axis=1) > 1e-12)
    info.update(
        collar_width_mm=float(width),
        collar_vertices=len(band),
        collar_anchors=int(np.count_nonzero(fixed[band])),
        total_budget_mm=float(budgets.max()),
    )
    if not len(moving):
        info["reason"] = "joint_solution_has_no_motion"
        return p.copy(), info
    # Evaluar todas las observaciones cercanas al collar, no solo las del borde.
    roi = np.flatnonzero((cf >= 0.65) & (cs >= 2))
    nd, _ = cKDTree(p[band]).query(cp[roi], workers=query_threads())
    roi = roi[nd <= width / 2 + 2 * spacing]
    if len(roi) < 6:
        info["reason"] = "insufficient_cloud_for_collar_guard"
        return p.copy(), info
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(semantic_mesh(p, t)))
    query = o3d.core.Tensor(cp[roi].astype(np.float32))
    distances = scene.compute_distance(query).numpy()
    affected = np.flatnonzero(np.any(np.isin(t, moving), axis=1))
    q = p[t[affected]]
    normals0 = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
    a0 = np.linalg.norm(normals0, axis=1)
    info["guard_trials"] = []
    for factor in (1.0, 0.5, 0.25, 0.125):
        trial = p + factor * delta
        q = trial[t[affected]]
        normals1 = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
        if np.any(np.sum(normals0 * normals1, axis=1) <= 0.25 * a0 * a0) or np.any(
            np.linalg.norm(normals1, axis=1) < 0.35 * a0
        ):
            info["guard_trials"].append({"factor": factor, "reason": "orientation_or_area"})
            continue
        report = semantic_intersection_analysis(trial, t, spacing, args)
        if report["blocking_pairs"]:
            info["guard_trials"].append({"factor": factor, "reason": "intersection"})
            continue
        check = o3d.t.geometry.RaycastingScene()
        check.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(semantic_mesh(trial, t)))
        increase = check.compute_distance(query).numpy() - distances
        if increase.max() > 0.25 * spacing or np.percentile(increase, 95) > 0.10 * spacing:
            info["guard_trials"].append({"factor": factor, "reason": "cloud_fidelity"})
            continue
        info.update(
            applied=True,
            moved_vertices=int(len(moving)),
            moved_rim_vertices=int(np.count_nonzero(abs(curve_delta) > 1e-12)),
            accepted_factor=factor,
            shift_mm=robust_stats(np.linalg.norm(factor * delta[moving], axis=1)),
            rim_shift_mm=robust_stats(abs(factor * curve_delta)),
            cloud_distance_increase_mm=robust_stats(increase),
            reason="accepted_joint_curve_and_collar_with_guards",
        )
        return trial, info
    info["reason"] = "joint_displacement_rejected_by_guards"
    return p.copy(), info


def _prepare_terminal_wall(p, t, loop, spacing):
    # Ensanchar solo cuando haya triángulos deficientes en el borde exterior.
    width = 3 * spacing
    quality = _patch_quality(p, t)
    for _ in range(3):
        distance, _ = _terminal_band(p, t, loop, width)
        allowed = np.all(distance[t] <= width, axis=1)
        fringe = allowed & (np.max(distance[t], axis=1) >= 0.6 * width)
        if width >= 9 * spacing or not np.any(fringe & (quality < 0.20)):
            break
        width += 3 * spacing
    refined, flips = _improve_local_diagonals(p, t, allowed, spacing, passes=8)
    return refined, {
        "band_width_mm": float(width),
        "eligible_faces": int(allowed.sum()),
        "diagonal_flips": flips,
        "poor_faces_before": int(np.count_nonzero(quality[allowed] < 0.20)),
        "poor_faces_after": int(np.count_nonzero(_patch_quality(p, refined)[allowed] < 0.20)),
        "policy": "adaptive_geodesic_band_preserving_crease_edges",
    }


def _closure_patch(vertices, loop, spacing, large):
    """Borde fijo; interior estimado. Sin radios, ejes ni figuras prefijadas."""
    rim = vertices[loop]
    center, basis, normal, xy = _closure_chart(rim)
    ears = _closure_earclip(xy)
    new, faces = [], []
    # Si el centro es visible desde TODO el contorno, anillos poligonales
    # interpolan la altura del borde hacia un plano interior. No son círculos.
    star = np.all(_closure_cross(np.roll(xy, -1, axis=0) - xy, -xy) > 1e-10)
    if star:
        radius = float(np.linalg.norm(xy, axis=1).max())
        layers = min(64, max(1, int(np.ceil(radius / max(1.5 * spacing, 1e-8))))) if large else 1
        previous = loop.copy()
        height = (rim - center) @ normal
        for layer in range(1, layers):
            fraction = 1.0 - layer / layers
            # Desvanecer solo la irregularidad perpendicular, sin mover el borde.
            ring = center + fraction * (xy @ basis) + (fraction**2 * height)[:, None] * normal
            ids = np.arange(len(vertices) + len(new), len(vertices) + len(new) + len(loop))
            new.extend(ring)
            for j in range(len(loop)):
                k = (j + 1) % len(loop)
                faces.extend([(previous[j], previous[k], ids[k]), (previous[j], ids[k], ids[j])])
            previous = ids
        mid = len(vertices) + len(new)
        new.append(center)
        faces.extend((previous[j], previous[(j + 1) % len(loop)], mid) for j in range(len(loop)))
    else:
        # Polígono cóncavo: cada oreja queda dentro del dominio, sin abanico
        # desde un centro que pueda caer fuera de la abertura.
        for a, b, c in ears:
            mid = len(vertices) + len(new)
            new.append(rim[[a, b, c]].mean(axis=0))
            faces.extend(
                [(loop[a], loop[b], mid), (loop[b], loop[c], mid), (loop[c], loop[a], mid)]
            )
    return np.asarray(new, dtype=float).reshape(-1, 3), np.asarray(faces, dtype=np.int64), normal


@operacion("Reparar pequeños contornos locales conservando abiertos los extremos no observados")
def close_repair_boundaries(
    vertices,
    triangles,
    colors,
    initial_boundary_keys,
    quantum,
    spacing,
    args,
    cloud_points,
    cloud_confidence,
    cloud_support,
):
    p, t = vertices.copy(), triangles.copy()
    c = None if colors is None else colors.copy()
    loops, complex_count = _closure_loops(t)
    info = {
        "enabled": bool(args.repair_holes or args.estimated_terminal_caps or args.fill_existing_holes),
        "existing_vertices_moved": 0,
        "existing_faces_modified": 0,
        "new_triangles_added": 0,
        "new_vertices_added": 0,
        "small_loops_closed": 0,
        "existing_holes_filled": 0,
        "estimated_terminal_caps_closed": 0,
        "complex_boundaries_skipped": complex_count,
        "records": [],
        "inferred_geometry_is_observed": False,
        "terminal_caps_policy": "disabled_no_unobserved_end_cap",
    }
    if not info["enabled"]:
        return p, t, c, info
    before = semantic_intersection_analysis(p, t, spacing, args)
    if before["blocking_pairs"]:
        info["reason"] = "intersecciones_previas_pendientes_no_se_anade_superficie"
        return p, t, c, info
    original_count = len(t)
    # Los pequeños se procesan primero; los controles incluyen parches ya aceptados.
    loops.sort(key=lambda ids: float(np.linalg.norm(np.ptp(p[ids], axis=0))))
    for number, loop in enumerate(loops):
        rim = p[loop]
        diameter = float(np.linalg.norm(np.ptp(rim, axis=0)))
        small = diameter <= args.repair_hole_max_diameter_spacing * spacing
        keys = set()
        for a, b in zip(loop, np.roll(loop, -1)):
            ends = [tuple(np.rint(p[i] / quantum).astype(np.int64)) for i in (a, b)]
            keys.add(tuple(sorted(ends)))
        inherited_ratio = len(keys & initial_boundary_keys) / max(len(keys), 1)
        fill_existing = bool(args.fill_existing_holes and inherited_ratio >= 1.0
                             and diameter <= args.existing_hole_max_diameter_mm)
        small = small or fill_existing
        rec = {
            "inferred_existing_hole": fill_existing,
            "boundary_id": number,
            "rim_vertices": len(loop),
            "diameter_mm": diameter,
            "initial_boundary_edge_ratio": inherited_ratio,
            "accepted": False,
            "kind": "repair_patch" if small else "estimated_terminal_cap",
        }
        info["records"].append(rec)
        try:
            if len(loop) > args.closure_max_loop_vertices:
                raise ValueError("limite_de_vertices_del_contorno")
            if small:
                if not fill_existing and (not args.repair_holes or inherited_ratio >= 1.0):
                    raise ValueError("contorno_pequeno_preexistente_o_reparacion_desactivada")
                # V2.0: un hueco creado durante la reparación solo se vuelve a
                # cerrar si su borde sigue rodeado por evidencia observacional
                # pose-diversa. Así no se rellena una región que 11/12 dejó
                # deliberadamente sin identificar por conflicto entre capas.
                rim_d, rim_i = cKDTree(cloud_points).query(rim, k=1, workers=query_threads())
                rim_ok = (
                    (rim_d <= 1.75 * spacing)
                    & (cloud_confidence[rim_i] >= 0.55)
                    & (cloud_support[rim_i] >= 2)
                )
                rec["observed_rim_support_ratio"] = float(np.mean(rim_ok))
                rec["observed_rim_distance_p90_mm"] = float(np.percentile(rim_d, 90))
                if np.mean(rim_ok) < 0.70:
                    raise ValueError("borde_del_hueco_sin_respaldo_multivista_suficiente")
            elif not args.estimated_terminal_caps or inherited_ratio < 0.5:
                raise ValueError("cierre_grande_no_preexistente_o_desactivado")
            center, basis, normal, xy = _closure_chart(rim)
            residual = np.abs((rim - center) @ normal)
            rec["plane_residual_p95_mm"] = float(np.percentile(residual, 95))
            if residual.max() > max(3 * spacing, args.closure_planarity_ratio * diameter):
                raise ValueError("contorno_no_aproximadamente_plano")
            if not small:
                # Extremo del cuerpo según el plano del contorno, no según Y ni
                # la etiqueta "cilindro". Permite arriba/abajo y cualquier pose.
                signed = (p - center) @ normal
                tol = max(3 * spacing, 1.5 * float(residual.max()))
                terminal = max(float(np.mean(signed <= tol)), float(np.mean(signed >= -tol)))
                rec["one_side_fraction"] = terminal
                if terminal < 0.98 or np.ptp(signed) < 6 * spacing:
                    raise ValueError("contorno_grande_no_terminal")
            candidate_options = []
            prep_info = {}
            if not small and args.refine_terminal_caps:
                try:
                    working_t, band_info = _prepare_terminal_wall(p, t, loop, spacing)
                    working_p, rim_info = _regularize_weak_terminal_rim(
                        p,
                        working_t,
                        loop,
                        cloud_points,
                        cloud_confidence,
                        cloud_support,
                        spacing,
                        args,
                    )
                    # El movimiento puede cambiar la calidad de las diagonales:
                    # preparar también la pared en su posición propuesta.
                    working_t, post_band = _prepare_terminal_wall(
                        working_p, working_t, loop, spacing
                    )
                    prep_info = {
                        "wall_band": band_info,
                        "wall_band_after_motion": post_band,
                        "rim_regularization": rim_info,
                    }
                    candidate_options.append((working_p, working_t, "local_remesh_and_weak_rim"))
                except (ValueError, np.linalg.LinAlgError) as exc:
                    prep_info = {"reason": "local_preparation_unavailable", "detail": str(exc)}
                    print(
                        f"[Paso 14] Preparación local omitida: {exc}. Se conserva la pared.",
                        flush=True,
                    )
                candidate_options.append((p, t, "cap_remesh_only"))
            # Respaldo del cierre anterior si las nuevas operaciones son rechazadas.
            candidate_options.append((p, t, "previous_guarded_closure"))
            accepted_candidate = None
            rec["candidate_attempts"] = []
            for wall_p, wall_t, mode in candidate_options:
                try:
                    if not small and mode != "previous_guarded_closure":
                        additions, patch, normal, mesh_info = _isotropic_terminal_patch(
                            wall_p, loop, spacing
                        )
                    else:
                        additions, patch, normal = _closure_patch(wall_p, loop, spacing, not small)
                        mesh_info = {"method": "previous_closure_fallback"}
                    if len(t) - original_count + len(patch) > args.closure_max_new_triangles:
                        raise ValueError("presupuesto_de_triangulos_agotado")
                    candidate_p = np.vstack((wall_p, additions))
                    candidate_t = np.vstack((wall_t, patch))
                    area_vectors = np.cross(
                        candidate_p[patch[:, 1]] - candidate_p[patch[:, 0]],
                        candidate_p[patch[:, 2]] - candidate_p[patch[:, 0]],
                    )
                    if np.any(area_vectors @ normal <= max(1e-14, spacing**2 * 1e-9)):
                        raise ValueError("parche_degenerado_o_invertido")
                    candidate_mesh = semantic_mesh(candidate_p, candidate_t)
                    if not (
                        candidate_mesh.is_edge_manifold(allow_boundary_edges=True)
                        and candidate_mesh.is_vertex_manifold()
                        and candidate_mesh.is_orientable()
                    ):
                        raise ValueError("parche_no_manifold_o_no_orientable")
                    incidence = build_edge_incidence(candidate_t)
                    if any(len(v) == 2 and v[0][1] == v[1][1] for v in incidence.values()):
                        raise ValueError("orientacion_inconsistente_en_la_costura")
                    report = semantic_intersection_analysis(candidate_p, candidate_t, spacing, args)
                    if report["blocking_pairs"]:
                        raise ValueError("parche_con_intersecciones_reales_o_ambiguas")
                    rec["candidate_attempts"].append({"mode": mode, "accepted": True})
                    accepted_candidate = (
                        candidate_p,
                        candidate_t,
                        additions,
                        patch,
                        wall_p,
                        wall_t,
                        mode,
                        mesh_info,
                    )
                    break
                except (ValueError, np.linalg.LinAlgError) as exc:
                    rec["candidate_attempts"].append(
                        {"mode": mode, "accepted": False, "reason": str(exc)}
                    )
                    print(f"[Paso 14] Contorno {number+1}, alternativa {mode}: {exc}.", flush=True)
            if accepted_candidate is None:
                raise ValueError("ninguna_alternativa_de_cierre_supero_las_guardas")
            candidate_p, candidate_t, additions, patch, wall_p, wall_t, mode, mesh_info = (
                accepted_candidate
            )
            moved = int(np.count_nonzero(np.linalg.norm(wall_p - p, axis=1) > 1e-12))
            modified = int(np.count_nonzero(np.any(wall_t != t, axis=1)))
            rec.update(
                accepted=True,
                reason="accepted_with_guards",
                mode=mode,
                face_start=len(t),
                face_stop=len(candidate_t),
                added_triangles=len(patch),
                added_vertices=len(additions),
                existing_vertices_moved=moved,
                wall_faces_retriangulated=modified,
                preparation=prep_info,
                preparation_retained=(mode == "local_remesh_and_weak_rim"),
                cap_meshing=mesh_info,
            )
            if not small:
                # Coordenadas finales de la costura: resistentes a la compactación
                # de índices. El 15 comprobará además el hash de la malla fuente.
                rec["protected_seam_xyz"] = wall_p[loop].tolist()
                rec["seam_is_observed"] = False
                regularization_info = prep_info.get("rim_regularization", {})
                rec["seam_fixed_mask"] = (
                    regularization_info.get("fixed_rim_mask", [True] * len(loop))
                    if mode == "local_remesh_and_weak_rim" and regularization_info.get("applied")
                    else [True] * len(loop)
                )
                rec["seam_motion_policy"] = "bounded_curve_tangential_redistribution_only"
                rec["seam_motion_budget_mm"] = 0.12 * spacing
                total_budget = np.asarray(
                    regularization_info.get("rim_total_budget_mm", [0.0] * len(loop)), dtype=float
                )
                used_budget = np.linalg.norm(wall_p[loop] - p[loop], axis=1)
                rec["seam_motion_remaining_mm"] = np.minimum(
                    0.12 * spacing, np.maximum(0.0, total_budget - used_budget)
                ).tolist()
                rec["seam_policy"] = "preserve_inferred_cap_wall_crease_after_bounded_refinement"
            if c is not None:
                c = np.vstack((c, np.tile(c[loop].mean(axis=0), (len(additions), 1))))
            p, t = candidate_p, candidate_t
            info["existing_vertices_moved"] += moved
            info["existing_faces_modified"] += modified
            info["existing_holes_filled"] += int(fill_existing)
            info["new_triangles_added"] += len(patch)
            info["new_vertices_added"] += len(additions)
            info["small_loops_closed" if small else "estimated_terminal_caps_closed"] += 1
            print(
                f"[Paso 14] Contorno {number+1}/{len(loops)}: {mode}, cerrado con {len(patch)} caras; "
                f"{moved} vértices regularizados y {modified} caras de pared retrianguladas.",
                flush=True,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            rec["reason"] = str(exc)
            print(
                f"[Paso 14] Contorno {number+1}/{len(loops)} conservado abierto: {exc}.", flush=True
            )
    info["reason"] = "completed_with_per_patch_guards"
    info["remaining_simple_loops"] = len(_closure_loops(t)[0])
    return p, t, c, info


def main():
    """Repara la topología y publica la malla con su diagnóstico de aceptación."""
    _estado14("Validando parámetros y rutas")
    args = make_parser().parse_args()
    if args.estimated_terminal_caps:
        print(
            "[Paso 14] Se ignora --estimated-terminal-caps: el cierre estimado de la cara no capturada está retirado.",
            flush=True,
        )
    args.estimated_terminal_caps = False
    args.refine_terminal_caps = False
    print(
        "[Paso 14] Reparación de agujeros pequeños creados por la limpieza; se conserva abierto el contorno grande sin crear una tapa estimada.",
        flush=True,
    )

    if (
        not np.isfinite(args.repair_hole_max_diameter_spacing)
        or args.repair_hole_max_diameter_spacing <= 0
        or not np.isfinite(args.closure_planarity_ratio)
        or not 0 < args.closure_planarity_ratio <= 0.15
        or args.closure_max_loop_vertices < 3
        or args.closure_max_new_triangles < 1
        or not np.isfinite(args.rim_max_shift_spacing)
        or not 0 < args.rim_max_shift_spacing <= 1
    ):
        raise ValueError(
            "Parámetros de cierre inválidos: escala positiva, planaridad (0,0.15], >=3 vértices y presupuesto positivo."
        )
    print(
        "[Paso 14 V2.0] Se está ejecutando limpieza topológica y reparación local sin tapa estimada.",
        flush=True,
    )
    root = Path(args.root).expanduser().resolve()
    object_name = args.object.strip()

    if not np.isfinite(args.existing_hole_max_diameter_mm) or args.existing_hole_max_diameter_mm <= 0:
        raise ValueError("El diámetro máximo de relleno debe ser positivo y finito")
    mesh_dir = root / "reconstruccion" / "multisesion" / args.mesh_source
    cloud_dir = root / "reconstruccion" / "multisesion" / args.cloud_source

    mesh_path = mesh_dir / "malla_final_seleccionada.ply"
    mesh_summary_path = mesh_dir / "resumen_13_reconstruccion_superficie.json"
    cloud_path = cloud_dir / "nube_regularizada_general.npz"

    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    if not cloud_path.is_file():
        raise FileNotFoundError(cloud_path)
    if bool(args.require_evidence_contract):
        if not mesh_summary_path.is_file():
            raise FileNotFoundError(mesh_summary_path)
        mesh_summary_info = json.loads(mesh_summary_path.read_text(encoding="utf-8"))
        if not bool(mesh_summary_info.get("evidence_contract", {}).get("available", False)):
            raise RuntimeError(
                "Paso 14 V2.0 requiere una salida del paso 13 que preserve el contrato de evidencia."
            )
    else:
        mesh_summary_info = (
            json.loads(mesh_summary_path.read_text(encoding="utf-8"))
            if mesh_summary_path.is_file()
            else {}
        )

    output = root / "reconstruccion" / "multisesion" / args.output_name
    output.mkdir(parents=True, exist_ok=True)

    _estado14("Leyendo la malla del paso 13")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise RuntimeError("La malla seleccionada de paso 13 está vacía.")

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    _estado14("Analizando topología inicial")
    initial_topology = topology_snapshot(mesh)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float64) if mesh.has_vertex_colors() else None
    # Procedencia topológica antes de cualquier reparación. Se usa únicamente
    # para detectar fragmentos que aparezcan DESPUÉS de retirar caras.
    source_vertices_for_ancestry = vertices.copy()
    source_triangles_for_ancestry = triangles.copy()

    closure_quantum = max(1e-9, float(np.linalg.norm(np.ptp(vertices, axis=0))) * 1e-9)
    original_boundary_keys = _closure_edge_keys(vertices, triangles, closure_quantum)

    _estado14("Leyendo nube y respaldo multivista")
    with np.load(cloud_path) as d:
        cloud_points = np.asarray(d["points"], dtype=np.float64)
        cloud_colors = np.asarray(d["colors"], dtype=np.uint8)
        support_raw = np.asarray(d["support_views"], dtype=np.float64).reshape(-1)
        confidence_raw = np.asarray(d["confidence"], dtype=np.float64).reshape(-1)
        evidence_contract_available = "surface_evidence_class" in d.files
        evidence_class = np.asarray(
            (
                d["surface_evidence_class"]
                if evidence_contract_available
                else np.full(len(cloud_points), 2)
            ),
            dtype=np.uint8,
        ).reshape(-1)
        independent_support = np.asarray(
            (
                d["independent_support_poses"]
                if "independent_support_poses" in d.files
                else support_raw
            ),
            dtype=np.float64,
        ).reshape(-1)
        angular_span = np.asarray(
            (
                d["support_angular_span_poses"]
                if "support_angular_span_poses" in d.files
                else np.zeros(len(cloud_points))
            ),
            dtype=np.int16,
        ).reshape(-1)
        conflict_ratio = np.asarray(
            (
                d["conflict_pose_ratio"]
                if "conflict_pose_ratio" in d.files
                else np.zeros(len(cloud_points))
            ),
            dtype=np.float64,
        ).reshape(-1)
        heldout_available = np.asarray(
            (
                d["heldout_validation_available"]
                if "heldout_validation_available" in d.files
                else np.zeros(len(cloud_points))
            ),
            dtype=bool,
        ).reshape(-1)
        heldout_pass = np.asarray(
            (
                d["heldout_validation_pass_ratio"]
                if "heldout_validation_pass_ratio" in d.files
                else np.ones(len(cloud_points))
            ),
            dtype=np.float64,
        ).reshape(-1)
        heldout_tests = np.asarray(
            (
                d["heldout_validation_tests"]
                if "heldout_validation_tests" in d.files
                else np.zeros(len(cloud_points))
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
                else np.ones(len(cloud_points), dtype=np.uint8)
            ),
            dtype=bool,
        ).reshape(-1)
        evidence_contract_accept_published = "evidence_contract_accept" in d.files
        evidence_strength = np.asarray(
            (
                d["evidence_strength"]
                if "evidence_strength" in d.files
                else np.ones(len(cloud_points))
            ),
            dtype=np.float64,
        ).reshape(-1)

    evidence_arrays = (
        support_raw,
        confidence_raw,
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
    if any(len(a) != len(cloud_points) for a in evidence_arrays):
        raise ValueError("Paso 14: metadatos de evidencia del paso 12 incompatibles con points.")
    if bool(args.require_evidence_contract) and not evidence_contract_available:
        raise RuntimeError(
            "Paso 14 V2.0 requiere el contrato de evidencia de 11/12. "
            "Reejecute 11-13 con los scripts actualizados."
        )
    if evidence_contract_available:
        if evidence_contract_accept_published:
            # 12 ya decidió el contrato combinando voto exacto, incertidumbre,
            # diversidad y conflicto. 14 no reinterpreta esa decisión.
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
        evidence_safe = np.ones(len(cloud_points), dtype=bool)
        evidence_gate_mode = "legacy_no_contract"
    if np.count_nonzero(evidence_safe) < 500:
        raise RuntimeError(
            "Paso 14: evidencia observacional segura insuficiente para reparar topología."
        )
    rejected_cloud_evidence = int(np.count_nonzero(~evidence_safe))
    cloud_points = cloud_points[evidence_safe]
    cloud_colors = cloud_colors[evidence_safe]
    # A partir de aquí 'support' significa diversidad independiente; la
    # confianza queda penalizada por conflicto/validación previa.
    support = independent_support[evidence_safe]
    confidence = np.clip(confidence_raw[evidence_safe] * evidence_strength[evidence_safe], 0.0, 1.0)
    evidence_contract_report = {
        "available": bool(evidence_contract_available),
        "rejected_points": rejected_cloud_evidence,
        "retained_points": int(len(cloud_points)),
        "independent_support": robust_stats(support),
        "effective_confidence": robust_stats(confidence),
        "policy": "topology_repairs_anchor_only_to_pose_diverse_nonconflicting_observations",
        "gate_mode": evidence_gate_mode,
    }

    _estado14("Calculando espaciado de la nube")
    # Escala real de muestreo.
    sample = cloud_points
    if len(sample) > 60000:
        sample = sample[np.linspace(0, len(sample) - 1, 60000).astype(int)]

    tree = cKDTree(cloud_points)
    nn, _ = tree.query(sample, k=2, workers=query_threads())
    spacing = float(np.median(nn[:, 1]))

    print("\n========== PASO 14 V2.0 — " "REPARACIÓN TOPOLÓGICA GENERAL ==========")
    print(f"Entrada: {len(vertices):,} vértices | " f"{len(triangles):,} triángulos")
    print(f"Spacing NN: {spacing:.4f} mm")
    print("Sin relleno de huecos / sin reajuste de poses / sin forma específica.")

    # -------------------------------------------------------------------------
    # 1. Limpieza conservadora de componentes.
    # -------------------------------------------------------------------------
    (
        vertices,
        triangles,
        colors,
        component_records,
    ) = component_cleanup(
        vertices,
        triangles,
        colors,
        cloud_points,
        support,
        confidence,
        spacing,
        args,
    )

    removed_components = sum(r["action"] == "remove_debris" for r in component_records)

    print(f"Componentes débiles eliminadas: " f"{removed_components}/{len(component_records)}")

    # -------------------------------------------------------------------------
    _estado14("Reparando conexiones de vértices")
    # 2. Bow-ties existentes.
    # -------------------------------------------------------------------------
    (
        vertices,
        triangles,
        colors,
        initial_splits,
    ) = split_vertex_fans(
        vertices,
        triangles,
        colors,
    )

    print(f"Splits bow-tie iniciales: {len(initial_splits)}")

    # -------------------------------------------------------------------------
    _estado14("Reparando aristas no manifold")
    # 3. Asegurar edge-manifold antes de orientar.
    # -------------------------------------------------------------------------
    intermediate = make_mesh(
        vertices,
        triangles,
        colors,
    )

    nm_edges = list(intermediate.get_non_manifold_edges(allow_boundary_edges=True))

    nonmanifold_edge_triangles_removed = 0

    if nm_edges:
        before = len(intermediate.triangles)
        intermediate.remove_non_manifold_edges()
        intermediate.remove_degenerate_triangles()
        intermediate.remove_unreferenced_vertices()
        nonmanifold_edge_triangles_removed = before - len(intermediate.triangles)

        vertices = np.asarray(
            intermediate.vertices,
            dtype=np.float64,
        )
        triangles = np.asarray(
            intermediate.triangles,
            dtype=np.int64,
        )
        colors = (
            np.asarray(
                intermediate.vertex_colors,
                dtype=np.float64,
            )
            if intermediate.has_vertex_colors()
            else None
        )

        (
            vertices,
            triangles,
            colors,
            edge_repair_splits,
        ) = split_vertex_fans(
            vertices,
            triangles,
            colors,
        )
    else:
        edge_repair_splits = []

    # -------------------------------------------------------------------------
    _estado14("Resolviendo conflictos de orientación")
    # 4. Reparación conservadora de ORIENTABILIDAD.
    # -------------------------------------------------------------------------
    (
        vertices,
        triangles,
        colors,
        orientation_info,
    ) = repair_orientation(
        vertices,
        triangles,
        colors,
        cloud_points,
        support,
        confidence,
        args,
    )

    print(f"Conflictos de orientación iniciales: " f"{orientation_info['initial_conflicts']}")
    print(
        f"Caras retiradas para cubrirlos: "
        f"{orientation_info['faces_removed']} "
        f"({orientation_info['face_removal_ratio']:.3%})"
    )

    # -------------------------------------------------------------------------
    _estado14("Verificando conexiones tras la reparación")
    # 5. La retirada de caras puede crear nuevos bow-ties.
    #    Se corrigen SIN mover puntos.
    # -------------------------------------------------------------------------
    (
        vertices,
        triangles,
        colors,
        final_splits,
    ) = split_vertex_fans(
        vertices,
        triangles,
        colors,
    )

    print(f"Splits bow-tie después de reparación: " f"{len(final_splits)}")

    # -------------------------------------------------------------------------
    _estado14("Orientando caras de forma consistente")
    # 6. Orientar consistentemente.
    # -------------------------------------------------------------------------
    triangles, flipped_faces = orient_faces_consistently(triangles)

    # -------------------------------------------------------------------------
    _estado14("Analizando y reparando intersecciones")
    # 6B. Reparación semántica mínima.
    # -------------------------------------------------------------------------
    prior_budget_faces = int(orientation_info["faces_removed"] + nonmanifold_edge_triangles_removed)

    (
        vertices,
        triangles,
        colors,
        semantic_repair_info,
    ) = repair_semantic_intersections(
        vertices,
        triangles,
        colors,
        cloud_points,
        support,
        confidence,
        spacing,
        args,
        already_removed_faces=prior_budget_faces,
    )

    print(
        "Reparación semántica: "
        f"estado={semantic_repair_info['status']} | "
        f"caras retiradas={semantic_repair_info['faces_removed']} | "
        f"total con reparaciones previas="
        f"{semantic_repair_info['total_removed_with_prior']}/"
        f"{semantic_repair_info['budget_faces']}"
    )

    _estado14("Retriangulando pequeños agujeros locales; sin tapa estimada")
    # Retriangular únicamente después de resolver los conflictos, sin volver a
    # insertar ciegamente las caras retiradas ni soldar capas por proximidad.
    vertices, triangles, colors, closure_info = close_repair_boundaries(
        vertices,
        triangles,
        colors,
        original_boundary_keys,
        closure_quantum,
        spacing,
        args,
        cloud_points,
        confidence,
        support,
    )
    semantic_repair_info["before_closure"] = semantic_repair_info.get("final", {})

    _estado14("Eliminando fragmentos diminutos creados por las reparaciones")
    vertices, triangles, colors, post_repair_cleanup = post_repair_component_cleanup(
        vertices,
        triangles,
        colors,
        source_vertices=source_vertices_for_ancestry,
        source_triangles=source_triangles_for_ancestry,
        cloud_points=cloud_points,
        support=support,
        confidence=confidence,
        spacing=spacing,
        args=args,
    )
    if post_repair_cleanup.get("applied"):
        print(
            "[Paso 14] Limpieza post-reparación: "
            f"{post_repair_cleanup.get('components_before')} -> "
            f"{post_repair_cleanup.get('components_after')} componentes; "
            f"{post_repair_cleanup.get('faces_removed', 0)} caras de fragmentos "
            "desprendidos retiradas.",
            flush=True,
        )

    semantic_repair_info["final"] = semantic_intersection_analysis(
        vertices, triangles, spacing, args
    )
    print(
        f"[Paso 14] Cierre: {closure_info['small_loops_closed']} huecos de reparación; "
        f"{closure_info['estimated_terminal_caps_closed']} tapas estimadas. Recalculando normales.",
        flush=True,
    )

    final_mesh = make_mesh(
        vertices,
        triangles,
        colors,
    )

    # Open3D tiene su propia propagación: la usamos como comprobación adicional.
    if final_mesh.is_orientable():
        try:
            final_mesh.orient_triangles()
        except Exception:
            pass

    final_mesh.compute_triangle_normals()
    final_mesh.compute_vertex_normals()

    _estado14("Verificando topología final y normales")
    topology = topology_snapshot(final_mesh)

    # -------------------------------------------------------------------------
    _estado14("Midiendo los bordes que permanecen abiertos")
    # 7. Cuantificar los bordes que continúan abiertos tras los cierres aceptados.
    # -------------------------------------------------------------------------
    vertices_final = np.asarray(
        final_mesh.vertices,
        dtype=np.float64,
    )
    triangles_final = np.asarray(
        final_mesh.triangles,
        dtype=np.int64,
    )

    boundary_records, boundary_edges, nm_numpy = boundary_components(
        vertices_final,
        triangles_final,
    )

    # Para preview solamente.
    boundary_vertex_set = defaultdict(set)
    for a, b in boundary_edges:
        boundary_vertex_set[0].add(int(a))
        boundary_vertex_set[0].add(int(b))

    boundary_vertices_all = sorted(boundary_vertex_set[0])

    # Añadir ids a los records solo temporalmente para preview:
    edge_adjacency = defaultdict(list)
    for a, b in boundary_edges:
        edge_adjacency[a].append(b)
        edge_adjacency[b].append(a)

    # Reconstrucción de ids por componente.
    seen = set()
    for record in boundary_records:
        # Localizamos una componente compatible por orden de extracción.
        record["_vertex_ids"] = []

    component_id = 0
    for seed in edge_adjacency:
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        ids = []
        while stack:
            u = stack.pop()
            ids.append(u)
            for nb in edge_adjacency[u]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if component_id < len(boundary_records):
            boundary_records[component_id]["_vertex_ids"] = ids
        component_id += 1

    components_final = final_component_summary(
        vertices_final,
        triangles_final,
    )

    boundary_summary = {
        "boundary_edge_count": int(len(boundary_edges)),
        "boundary_component_count": int(len(boundary_records)),
        "simple_closed_loop_count": int(sum(r["simple_closed_loop"] for r in boundary_records)),
        "total_boundary_perimeter_mm": float(sum(r["perimeter_mm"] for r in boundary_records)),
        "boundary_edge_ratio": float(
            len(boundary_edges)
            / max(
                len(build_edge_incidence(triangles_final)),
                1,
            )
        ),
        "perimeter_mm": robust_stats(r["perimeter_mm"] for r in boundary_records),
        "diameter_mm": robust_stats(r["diameter_bbox_mm"] for r in boundary_records),
        "nonmanifold_edge_count_numpy": int(len(nm_numpy)),
    }

    # -------------------------------------------------------------------------
    _estado14("Evaluando calidad topológica")
    # 8. Gate para pasar a paso 17.
    # -------------------------------------------------------------------------
    blockers = []

    if topology.get("edge_manifold_allow_boundary") is not True:
        blockers.append("edge_manifold_allow_boundary != true")
    if topology.get("vertex_manifold") is not True:
        blockers.append("vertex_manifold != true")
    if topology.get("orientable") is not True:
        blockers.append("orientable != true")
    semantic_final = semantic_repair_info.get(
        "final",
        {},
    )
    semantic_true = int(
        semantic_final.get(
            "true_self_intersection_pairs",
            0,
        )
        or 0
    )
    semantic_ambiguous = int(
        semantic_final.get(
            "ambiguous_intersection_pairs",
            0,
        )
        or 0
    )

    if semantic_true > 0:
        blockers.append(f"true_self_intersection_pairs={semantic_true}")
    if semantic_ambiguous > 0:
        blockers.append(f"ambiguous_intersection_pairs={semantic_ambiguous}")
    if semantic_repair_info.get("status") != "clean":
        blockers.append("semantic_intersection_repair != clean")
    if topology.get("non_manifold_vertex_count") not in (0, None):
        blockers.append("non_manifold_vertex_count > 0")
    if topology.get("non_manifold_edge_count_allow_boundary") not in (0, None):
        blockers.append("non_manifold_edge_count_allow_boundary > 0")

    status = "ready_for_step_15" if not blockers else "review_required"

    # -------------------------------------------------------------------------
    # 9. Guardados.
    # -------------------------------------------------------------------------
    _estado14("Guardando malla y diagnósticos")
    final_path = output / "malla_final_topologica.ply"
    if not o3d.io.write_triangle_mesh(
        str(final_path),
        final_mesh,
        write_ascii=False,
    ):
        raise RuntimeError("No se pudo guardar malla_final_topologica.ply")

    all_estimated = bool(mesh_summary_info.get("estimated_completion", {}).get("all_faces_are_estimated", False))
    if args.fill_existing_holes or all_estimated or mesh_summary_info.get("estimated_completion", {}).get("has_estimated_patches", False):
        save_inferred_face_provenance(mesh_path, final_mesh, output, all_estimated=all_estimated)

    # CSV componentes.
    component_csv = output / "analisis_componentes_14.csv"
    write_csv(
        component_csv,
        component_records,
        [
            "component_id",
            "triangles",
            "vertices",
            "area_mm2",
            "area_ratio",
            "max_extent_mm",
            "cloud_distance_median_mm",
            "cloud_distance_p90_mm",
            "median_support",
            "median_confidence",
            "tiny_geometry",
            "weak_evidence",
            "action",
        ],
    )

    # CSV de la segunda revisión de componentes, ya con todas las reparaciones.
    post_component_csv = output / "analisis_componentes_post_reparacion_14.csv"
    write_csv(
        post_component_csv,
        post_repair_cleanup.get("records", []),
        [
            "component_id",
            "triangles",
            "vertices",
            "area_mm2",
            "area_ratio",
            "max_extent_mm",
            "tiny_geometry",
            "source_ancestor_component",
            "source_ancestry_ratio",
            "source_ancestry_votes",
            "source_distance_p90_mm",
            "cloud_distance_median_mm",
            "cloud_distance_p90_mm",
            "median_support",
            "median_confidence",
            "dominant_descendant_component",
            "dominant_to_component_area_ratio",
            "action",
            "reason",
        ],
    )

    # CSV caras retiradas por orientabilidad.
    orientation_csv = output / "caras_eliminadas_orientabilidad_14.csv"
    write_csv(
        orientation_csv,
        orientation_info["records"],
        [
            "face_before_removal",
            "support_median",
            "confidence_median",
            "cloud_distance_median_mm",
            "area_mm2",
            "boundary_edges_before_removal",
            "removal_cost",
        ],
    )

    # CSV huecos/bordes finales.
    holes_csv = output / "analisis_huecos_14.csv"
    clean_boundary_records = []
    for rec in boundary_records:
        x = dict(rec)
        x.pop("_vertex_ids", None)
        clean_boundary_records.append(x)

    write_csv(
        holes_csv,
        clean_boundary_records,
        [
            "boundary_id",
            "vertices",
            "edges",
            "simple_closed_loop",
            "perimeter_mm",
            "diameter_bbox_mm",
            "degree_min",
            "degree_max",
        ],
    )

    _estado14("Generando vista previa")
    preview_path = output / "preview_topologia_14.png"
    preview(
        preview_path,
        vertices_final,
        boundary_records,
        "Paso 14 | reparación local sin tapa terminal estimada",
        status,
        orientation_info,
        topology,
        closure_info,
        semantic_repair_info["final"],
        semantic_repair_info["total_removed_with_prior"],
    )

    summary = {
        "estimated_completion": mesh_summary_info.get("estimated_completion", {}),
        "schema_version": "1.6",
        "method": (
            "pose_diverse_evidence_aware_component_cleanup_"
            "simultaneous_vertex_fan_split_"
            "weighted_orientation_conflict_face_cover_"
            "guarded_small_hole_retriangulation_post_repair_ancestry_cleanup_"
            "without_terminal_caps"
        ),
        "object": object_name,
        "shape_specific_assumptions": False,
        "pose_reoptimization": False,
        "vertex_displacement": bool(closure_info["existing_vertices_moved"]),
        "new_triangles_added": closure_info["new_triangles_added"],
        "hole_filling_enabled": bool(args.repair_holes or args.estimated_terminal_caps or args.fill_existing_holes),
        "local_closure": closure_info,
        "protected_seams_mesh_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        "source_mesh": str(mesh_path),
        "source_mesh_summary": str(mesh_summary_path) if mesh_summary_path.is_file() else None,
        "source_cloud": str(cloud_path),
        "evidence_contract": evidence_contract_report,
        "estimated_point_spacing_mm": spacing,
        "parameters": vars(args),
        "initial_topology": initial_topology,
        "component_cleanup": {
            "components_analyzed": int(len(component_records)),
            "components_removed": int(removed_components),
            "records": component_records,
        },
        "post_repair_component_cleanup": post_repair_cleanup,
        "initial_bowtie_split": {
            "operations": int(len(initial_splits)),
            "records": initial_splits,
        },
        "nonmanifold_edge_repair": {
            "edges_before": int(len(nm_edges)),
            "triangles_removed": int(nonmanifold_edge_triangles_removed),
            "bowtie_splits_after_edge_repair": int(len(edge_repair_splits)),
        },
        "orientation_repair": orientation_info,
        "semantic_intersection_repair": semantic_repair_info,
        "raw_self_intersecting_is_diagnostic_only": True,
        "post_orientation_bowtie_split": {
            "operations": int(len(final_splits)),
            "records": final_splits,
        },
        "consistent_face_flips": int(flipped_faces),
        "final_topology": topology,
        "final_components": components_final,
        "boundaries": boundary_summary,
        "status": status,
        "blockers_for_step_17": blockers,
        "outputs": {
            "mesh_final": str(final_path),
            "preview": str(preview_path),
            "components_csv": str(component_csv),
            "post_repair_components_csv": str(post_component_csv),
            "orientation_faces_csv": str(orientation_csv),
            "holes_csv": str(holes_csv),
        },
        "methodological_note": (
            "Solo se permite movimiento débil acotado del borde y retriangulación de una banda local. "
            "Se retriangulan huecos de reparación y, con fill_existing_holes, huecos preexistentes acotados como geometría inferida; las tapas terminales "
            "estimadas permanecen desactivadas. "
            "Los parches añadidos no equivalen a datos observados ni aumentan el soporte multivista. "
            "Los contornos no verificables se conservan abiertos y se documentan. "
            "Tras todas las reparaciones se eliminan solo componentes diminutas que "
            "se desprendieron de una componente fuente mayor; las piezas que ya eran "
            "independientes en paso 13 no se eliminan por tamaño."
        ),
    }

    _estado14("Guardando resumen de la reparación")
    summary_path = output / "resumen_14_limpieza_topologica.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n========== PASO 14 V2.0 COMPLETADA ==========")
    print("Estado:", status)
    print("Topología final:", topology)
    print(
        "Componente principal (área):",
        components_final["largest_area_ratio"],
    )
    print(
        "Aristas de borde:",
        boundary_summary["boundary_edge_count"],
    )
    print(
        "Componentes de borde:",
        boundary_summary["boundary_component_count"],
    )
    print("Cierre local:", "ACTIVADO" if closure_info["enabled"] else "DESACTIVADO")
    print("Caras nuevas estimadas:", closure_info["new_triangles_added"])
    print("Contornos conservados abiertos:", boundary_summary["boundary_component_count"])
    if blockers:
        print("Bloqueos para paso 17:")
        for item in blockers:
            print(" -", item)
    else:
        print("La malla cumple el gate topológico para continuar al paso 15.")
    print("Salida:", output)
    print("===================================================\n")

    return 0 if not blockers else 2


def _entrada_con_diagnostico():
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    from utilidades_progreso import informar_inicio

    informar_inicio(
        __file__, "14", "Reparar topología y pequeños huecos, sin tapa terminal estimada"
    )
    try:
        codigo = main()
        _estado14("Completado" if codigo == 0 else f"Finalizado con observaciones: código {codigo}")
        raise SystemExit(codigo)
    except Exception as exc:
        _estado14(f"ERROR: {type(exc).__name__}: {exc}")
        raise
    finally:
        _estado_stop.set()


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
