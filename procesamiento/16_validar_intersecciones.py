#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PASO 16 V1.5 — VALIDACIÓN GEOMÉTRICA DE AUTO-INTERSECCIONES

No modifica la malla.

Objetivo:
clasificar los pares que Open3D reporta mediante
TriangleMesh.get_self_intersecting_triangles() en:

A) contactos geométricos coincidentes:
   - contacto puntual;
   - contacto por arista;
   - adyacencia topológica.

B) auto-intersecciones reales:
   - cruce transversal;
   - solape coplanar.

C) ambiguos:
   - casos numéricamente no resolubles de forma segura.

Solo B o C bloquean paso 17.

Requisitos metodológicos:
- independiente de la forma del objeto;
- sin cuboides/planos/Manhattan;
- sin modificar poses;
- sin mover puntos;
- sin añadir/eliminar triángulos;
- tolerancias derivadas del spacing real de la nube.
"""

from __future__ import annotations
from utilidades_progreso import operacion
from utilidades_rendimiento import ejecutar_items, contexto
from utilidades_rendimiento import configurar as _configurar_recursos

# Los runtimes numéricos leen estos límites al importarse.
_configurar_recursos(__file__)


import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except Exception as exc:
    raise SystemExit("Paso 16 requiere Open3D 0.19.x. " f"Detalle: {exc}")

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit("Paso 16 requiere matplotlib. " f"Detalle: {exc}")

from clasificador_intersecciones import (
    ALLOWED_CONTACTS,
    BLOCKING_INTERSECTIONS,
    classify_triangle_pair,
)


def parser():
    """Construye las opciones de línea de comandos de este paso."""
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--object", required=True)
    p.add_argument(
        "--mesh-source",
        default="15_pulido_final",
    )
    p.add_argument(
        "--topology-source",
        default="14_limpieza_topologica",
        help=(
            "Fuente del resumen topológico V1.3. La geometría a validar "
            "puede venir del paso 15 sin perder la escala del paso 14."
        ),
    )
    p.add_argument(
        "--output-name",
        default="16_validacion_intersecciones",
    )
    p.add_argument(
        "--geometric-epsilon-spacing-factor",
        type=float,
        default=1e-4,
        help=("Tolerancia para posiciones coincidentes. " "Default = 0.0001 x spacing."),
    )
    p.add_argument(
        "--contact-locality-spacing-factor",
        type=float,
        default=1e-3,
        help=(
            "Tolerancia para confirmar que la intersección completa "
            "está limitada al contacto coincidente."
        ),
    )
    return p


def topology(mesh):
    """Recoge las comprobaciones topológicas de la malla sin modificarla."""
    out = {}
    checks = (
        (
            "edge_manifold_allow_boundary",
            lambda: mesh.is_edge_manifold(True),
        ),
        ("vertex_manifold", mesh.is_vertex_manifold),
        ("orientable", mesh.is_orientable),
        ("raw_self_intersecting", mesh.is_self_intersecting),
        ("raw_watertight", mesh.is_watertight),
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


def exact_duplicate_groups(vertices):
    """Agrupa vértices cuyas coordenadas coinciden exactamente."""
    groups = defaultdict(list)
    for i, p in enumerate(vertices):
        groups[tuple(map(float, p))].append(int(i))
    duplicates = [ids for ids in groups.values() if len(ids) > 1]
    return duplicates


def preview(path, vertices, triangles, records, status):
    fig = plt.figure(figsize=(15, 10))

    groups = defaultdict(list)
    for r in records:
        groups[r["category"]].extend([r["triangle_a"], r["triangle_b"]])

    projections = (
        (1, (0, 1), "X-Y"),
        (2, (0, 2), "X-Z"),
        (3, (2, 1), "Z-Y"),
    )

    sample = vertices
    if len(sample) > 100000:
        idx = np.linspace(
            0,
            len(sample) - 1,
            100000,
        ).astype(int)
        sample = sample[idx]

    for sp, dims, name in projections:
        ax = fig.add_subplot(2, 2, sp)
        ax.scatter(
            sample[:, dims[0]],
            sample[:, dims[1]],
            s=0.2,
            alpha=0.20,
        )

        for category, face_ids in sorted(groups.items()):
            face_ids = np.unique(np.asarray(face_ids, dtype=np.int64))
            if len(face_ids) == 0:
                continue
            centers = vertices[triangles[face_ids]].mean(axis=1)
            ax.scatter(
                centers[:, dims[0]],
                centers[:, dims[1]],
                s=8,
                label=category,
            )

        ax.set_title(name)
        ax.set_aspect("equal", adjustable="box")

    ax = fig.add_subplot(2, 2, 4)
    ax.axis("off")

    counts = Counter(r["category"] for r in records)
    lines = [
        "Paso 16 | clasificación de auto-intersecciones",
        f"Estado: {status}",
        f"Pares Open3D: {len(records)}",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"{key}: {value}")

    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        va="top",
        fontsize=11,
    )

    handles, labels = fig.axes[0].get_legend_handles_labels()
    if handles:
        fig.axes[0].legend(
            handles,
            labels,
            fontsize=7,
            loc="best",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _procesar_unidad_independiente(task):
    """Clasifica un bloque de pares con la geometría y tolerancias del contexto."""
    ctx = contexto()
    contact_tol = ctx["contact_tol"]
    eps = ctx["eps"]
    triangles = ctx["triangles"]
    vertices = ctx["vertices"]
    print("[PROGRESO] Clasificar intersecciones por bloques: iniciando unidad", flush=True)
    records = []
    for index, (fa, fb) in task:
        ia = triangles[int(fa)]
        ib = triangles[int(fb)]
        result = classify_triangle_pair(
            vertices[ia],
            vertices[ib],
            triangle_indices_a=ia,
            triangle_indices_b=ib,
            geometric_epsilon=eps,
            contact_locality_tolerance=contact_tol,
        )
        records.append(
            {
                "pair_index": int(index),
                "triangle_a": int(fa),
                "triangle_b": int(fb),
                "category": result["category"],
                "blocking": bool(result["blocking"]),
                "shared_geometric_positions": int(result.get("shared_geometric_positions", 0)),
                "intersection_point_count": int(len(result.get("intersection_points", []))),
                "coplanar_overlap_area_projected": result.get("coplanar_overlap_area_projected"),
            }
        )
    return {"records": records}


def main():
    """Clasifica intersecciones y determina los bloqueadores para la validación final."""
    args = parser().parse_args()

    root = Path(args.root).expanduser().resolve()
    obj = args.object.strip()

    source_dir = root / "reconstruccion" / "multisesion" / args.mesh_source

    mesh_path = source_dir / "malla_final_topologica.ply"
    topology_dir = root / "reconstruccion" / "multisesion" / args.topology_source
    source_summary = topology_dir / "resumen_14_limpieza_topologica.json"

    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    if not source_summary.is_file():
        raise FileNotFoundError(source_summary)

    source_info = json.loads(source_summary.read_text(encoding="utf-8"))

    spacing = float(source_info["estimated_point_spacing_mm"])

    eps = max(
        1e-8,
        float(args.geometric_epsilon_spacing_factor) * spacing,
    )
    contact_tol = max(
        10.0 * eps,
        float(args.contact_locality_spacing_factor) * spacing,
    )

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise RuntimeError("La malla geométrica a validar está vacía.")

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64,
    )
    triangles = np.asarray(
        mesh.triangles,
        dtype=np.int64,
    )

    raw_topology = topology(mesh)

    pairs = np.asarray(
        mesh.get_self_intersecting_triangles(),
        dtype=np.int64,
    )

    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    else:
        pairs = pairs.reshape(-1, 2)

    print("\n========== PASO 16 V1.5 — " "VALIDACIÓN DE INTERSECCIONES ==========")
    print("Malla:", mesh_path)
    print(f"Vértices={len(vertices):,} | " f"Triángulos={len(triangles):,}")
    print(f"Spacing={spacing:.6f} mm | " f"eps geométrico={eps:.9f} mm")
    print(f"Pares reportados por Open3D: {len(pairs):,}")

    records = []

    _results = ejecutar_items(
        _procesar_unidad_independiente,
        {"contact_tol": contact_tol, "eps": eps, "triangles": triangles, "vertices": vertices},
        [
            list(zip(range(a, min(a + 512, len(pairs))), pairs[a : a + 512]))
            for a in range(0, len(pairs), 512)
        ],
        "Clasificar intersecciones por bloques",
        reserve_mb=512,
    )
    for _result in _results:
        records.extend(_result["records"])

    counts = Counter(r["category"] for r in records)

    blocking_records = [r for r in records if r["blocking"]]
    contact_records = [r for r in records if not r["blocking"]]

    true_intersection_count = sum(
        counts.get(name, 0)
        for name in (
            "transverse_intersection",
            "coplanar_overlap",
            "unexplained_coplanar_contact",
        )
    )

    ambiguous_count = sum(
        counts.get(name, 0)
        for name in (
            "numerically_ambiguous",
            "degenerate_ambiguous",
        )
    )

    duplicate_groups = exact_duplicate_groups(vertices)
    duplicate_extra_vertices = int(sum(len(g) - 1 for g in duplicate_groups))

    blockers = []

    if raw_topology.get("edge_manifold_allow_boundary") is not True:
        blockers.append("edge_manifold_allow_boundary != true")

    if raw_topology.get("vertex_manifold") is not True:
        blockers.append("vertex_manifold != true")

    if raw_topology.get("orientable") is not True:
        blockers.append("orientable != true")

    if true_intersection_count > 0:
        blockers.append(f"true_self_intersection_pairs={true_intersection_count}")

    if ambiguous_count > 0:
        blockers.append(f"ambiguous_intersection_pairs={ambiguous_count}")

    status = "ready_for_step_17" if not blockers else "review_required"

    output = root / "reconstruccion" / "multisesion" / args.output_name
    output.mkdir(parents=True, exist_ok=True)

    csv_path = output / "intersecciones_clasificadas_16.csv"

    fields = [
        "pair_index",
        "triangle_a",
        "triangle_b",
        "category",
        "blocking",
        "shared_geometric_positions",
        "intersection_point_count",
        "coplanar_overlap_area_projected",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(records)

    preview_path = output / "preview_intersecciones_16.png"
    preview(
        preview_path,
        vertices,
        triangles,
        records,
        status,
    )

    report = {
        "estimated_completion": source_info.get("estimated_completion", {}),
        "schema_version": "1.2",
        "method": (
            "open3d_candidate_pairs_plus_" "explicit_triangle_triangle_contact_classification"
        ),
        "object": obj,
        "shape_specific_assumptions": False,
        "mesh_modified": False,
        "poses_modified": False,
        "vertices_modified": False,
        "triangles_modified": False,
        "source_mesh": str(mesh_path),
        "source_mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "source_topology_summary": str(source_summary),
        "source_topology_summary_sha256": hashlib.sha256(source_summary.read_bytes()).hexdigest(),
        "mesh_source_step": args.mesh_source,
        "topology_source_step": args.topology_source,
        "estimated_point_spacing_mm": spacing,
        "geometric_epsilon_mm": eps,
        "contact_locality_tolerance_mm": contact_tol,
        "raw_open3d_topology": raw_topology,
        "raw_open3d_intersection_pairs": int(len(pairs)),
        "classification_counts": dict(counts),
        "coincident_contact_pairs": int(len(contact_records)),
        "true_self_intersection_pairs": int(true_intersection_count),
        "ambiguous_intersection_pairs": int(ambiguous_count),
        "all_raw_pairs_explained": bool(
            len(records) == len(contact_records) + len(blocking_records)
        ),
        "duplicate_coordinate_groups": int(len(duplicate_groups)),
        "duplicate_extra_vertices": int(duplicate_extra_vertices),
        "semantic_self_intersection_status": (
            "no_true_self_intersections"
            if true_intersection_count == 0 and ambiguous_count == 0
            else "true_or_ambiguous_intersections_present"
        ),
        "status": status,
        "blockers_for_step_17": blockers,
        "policy": {
            "coincident_vertex_contact": ("documented_nonblocking_contact"),
            "coincident_edge_contact": ("documented_nonblocking_contact"),
            "topological_adjacency": ("nonblocking"),
            "transverse_intersection": "blocking",
            "coplanar_overlap": "blocking",
            "unexplained_coplanar_contact": "blocking",
            "numerically_ambiguous": "blocking",
            "degenerate_ambiguous": "blocking",
        },
        "outputs": {
            "csv": str(csv_path),
            "preview": str(preview_path),
        },
        "methodological_note": (
            "El booleano raw de Open3D se conserva como diagnóstico. "
            "Para la decisión se clasifican sus pares candidatos. "
            "Un contacto geométrico coincidente no se equipara a un "
            "cruce transversal ni a un solape superficial. "
            "Los casos ambiguos nunca se aceptan automáticamente."
        ),
    }

    report_path = output / "resumen_16_validacion_intersecciones.json"
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nClasificación:")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")

    print(
        "\nIntersecciones reales:",
        true_intersection_count,
    )
    print("Ambiguas:", ambiguous_count)
    print(
        "Contactos coincidentes/no bloqueantes:",
        len(contact_records),
    )
    print("Estado:", status)

    if blockers:
        print("Bloqueos:")
        for blocker in blockers:
            print(" -", blocker)
    else:
        print("Gate superado: puede ejecutarse " "paso 17 V1.8.")

    print("Salida:", output)
    print("====================================================\n")

    return 0 if not blockers else 2


def _entrada_con_diagnostico():
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    from utilidades_progreso import informar_inicio

    informar_inicio(__file__, "16", "Validar intersecciones")
    raise SystemExit(main())


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
