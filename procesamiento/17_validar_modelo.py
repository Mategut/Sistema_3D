#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PASO 17 V1.8 — VALIDACIÓN FINAL GEOMÉTRICA + TOPOLÓGICA

Criterios de validación:
- conserva el diagnóstico raw de Open3D;
- consume la clasificación explícita del paso 16 V1.4;
- solo una intersección transversal/coplanar real o un caso ambiguo
  bloquean la validación;
- un contacto puntual coincidente queda documentado como warning,
  no como auto-intersección física transversal.
- el gate mesh->cloud se adapta al espaciado real de muestreo, pero conserva
  un límite máximo absoluto de seguridad.
- mesh->cloud conserva un umbral duro de seguridad: la continuidad o una
  topología limpia no pueden sustituir evidencia observacional local.

No se fuerza watertight.
No se calcula volumen en una superficie abierta o con contactos coincidentes.
"""

from __future__ import annotations
from utilidades_progreso import operacion
from utilidades_rendimiento import query_threads
from utilidades_rendimiento import configurar as _configurar_recursos

# Los runtimes numéricos leen estos límites al importarse.
_configurar_recursos(__file__)


import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
except Exception as exc:
    raise SystemExit(f"Paso 17 requiere SciPy: {exc}")

try:
    import open3d as o3d
except Exception as exc:
    raise SystemExit(f"Paso 17 requiere Open3D: {exc}")

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


VERSION = "V1.8"


def parser():
    """Construye las opciones de línea de comandos de este paso."""
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--object", required=True)
    p.add_argument(
        "--cloud-source",
        default="12_regularizacion_nube",
    )
    p.add_argument(
        "--mesh-source",
        default="15_pulido_final",
    )
    p.add_argument(
        "--intersection-source",
        default="16_validacion_intersecciones",
    )
    p.add_argument(
        "--output-name",
        default="17_validacion_modelo",
    )

    p.add_argument("--coverage-gate-mm", type=float, default=3.0)
    p.add_argument("--warning-coverage", type=float, default=0.95)
    p.add_argument("--reject-coverage", type=float, default=0.85)

    p.add_argument(
        "--warning-cloud-mesh-p90-mm",
        type=float,
        default=2.5,
    )
    p.add_argument(
        "--reject-cloud-mesh-p90-mm",
        type=float,
        default=4.5,
    )
    p.add_argument(
        "--warning-mesh-cloud-p90-mm",
        type=float,
        default=2.0,
    )
    p.add_argument(
        "--reject-mesh-cloud-p90-mm",
        type=float,
        default=4.0,
    )
    p.add_argument(
        "--adaptive-mesh-cloud-thresholds",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--warning-mesh-cloud-spacing-factor",
        type=float,
        default=2.5,
    )
    p.add_argument(
        "--reject-mesh-cloud-spacing-factor",
        type=float,
        default=5.0,
    )
    p.add_argument(
        "--maximum-adaptive-reject-mesh-cloud-p90-mm",
        type=float,
        default=6.0,
        help=(
            "Techo absoluto del umbral adaptativo mesh->cloud. El valor "
            "nunca puede quedar por debajo de --reject-mesh-cloud-p90-mm."
        ),
    )
    p.add_argument(
        "--allow-coherent-interpolation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compatibilidad diagnóstica. La coherencia puede documentarse, pero "
            "nunca anula el umbral duro mesh->cloud de rechazo."
        ),
    )
    p.add_argument(
        "--coherent-interpolation-min-coverage",
        type=float,
        default=0.95,
    )
    p.add_argument(
        "--coherent-interpolation-max-bbox-expansion-ratio",
        type=float,
        default=1.12,
    )
    p.add_argument(
        "--coherent-interpolation-min-largest-area-ratio",
        type=float,
        default=0.97,
    )
    p.add_argument(
        "--coherent-interpolation-max-boundary-edge-ratio",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--coherent-interpolation-max-p95-extent-ratio",
        type=float,
        default=0.12,
        help=(
            "Máximo P95 mesh->cloud dividido por la mayor extensión robusta "
            "de la nube. Es independiente del tipo y tamaño del objeto."
        ),
    )

    p.add_argument(
        "--warning-bbox-expansion-ratio",
        type=float,
        default=1.08,
    )
    p.add_argument(
        "--reject-bbox-expansion-ratio",
        type=float,
        default=1.18,
    )

    p.add_argument(
        "--warning-largest-component-area-ratio",
        type=float,
        default=0.97,
    )
    p.add_argument(
        "--reject-largest-component-area-ratio",
        type=float,
        default=0.85,
    )

    p.add_argument(
        "--warning-boundary-edge-ratio",
        type=float,
        default=0.03,
    )
    p.add_argument(
        "--reject-boundary-edge-ratio",
        type=float,
        default=0.15,
    )

    p.add_argument("--cloud-samples", type=int, default=100000)
    p.add_argument("--mesh-samples", type=int, default=100000)

    # Verificación independiente de la envolvente débil. V1.8 define el núcleo
    # por evidencia pose-diversa/held-out, no por número bruto de poses vecinas.
    p.add_argument(
        "--require-evidence-contract", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--strong-core-support",
        type=int,
        default=4,
        help="Compatibilidad legacy; no gobierna el núcleo V1.8 cuando existe contrato de evidencia.",
    )
    p.add_argument("--strong-core-min-evidence-class", type=int, default=3)
    p.add_argument("--strong-core-min-independent-support", type=int, default=2)
    p.add_argument("--strong-core-min-angular-span-poses", type=int, default=2)
    p.add_argument("--strong-core-max-conflict-pose-ratio", type=float, default=0.20)
    p.add_argument("--strong-core-min-heldout-pass-ratio", type=float, default=0.80)
    p.add_argument("--strong-core-minimum-points", type=int, default=500)
    p.add_argument(
        "--warning-observation-envelope-expansion-ratio",
        type=float,
        default=1.10,
    )
    p.add_argument(
        "--reject-observation-envelope-expansion-ratio",
        type=float,
        default=1.22,
    )

    return p


def stats(values):
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


def adaptive_mesh_cloud_limits(
    *,
    sampling_spacing_mm,
    enabled,
    warning_absolute_mm,
    reject_absolute_mm,
    warning_spacing_factor,
    reject_spacing_factor,
    maximum_reject_mm,
):
    """Calcula gates mesh->cloud dependientes de la resolución de muestreo.

    Los valores absolutos siguen siendo el piso de tolerancia. El umbral de
    rechazo puede crecer con el spacing, pero nunca supera el techo de
    seguridad. No utiliza ninguna clase o modelo de forma.
    """
    warning_base = max(float(warning_absolute_mm), 0.0)
    reject_base = max(float(reject_absolute_mm), warning_base)
    safety_ceiling = max(float(maximum_reject_mm), reject_base)

    spacing = None
    if sampling_spacing_mm is not None:
        candidate = float(sampling_spacing_mm)
        if np.isfinite(candidate) and candidate > 0.0:
            spacing = candidate

    if bool(enabled) and spacing is not None:
        warning_effective = max(
            warning_base,
            max(float(warning_spacing_factor), 0.0) * spacing,
        )
        reject_effective = max(
            reject_base,
            max(float(reject_spacing_factor), 0.0) * spacing,
        )
        reject_effective = min(reject_effective, safety_ceiling)
        warning_effective = min(warning_effective, reject_effective)
        mode = "adaptive_spacing"
    else:
        warning_effective = warning_base
        reject_effective = reject_base
        mode = "absolute_fallback"

    return {
        "mode": mode,
        "sampling_spacing_mm": spacing,
        "warning_absolute_mm": float(warning_base),
        "reject_absolute_mm": float(reject_base),
        "warning_spacing_factor": float(warning_spacing_factor),
        "reject_spacing_factor": float(reject_spacing_factor),
        "maximum_reject_mm": float(safety_ceiling),
        "effective_warning_mm": float(warning_effective),
        "effective_reject_mm": float(reject_effective),
    }


def robust_extent(points):
    """Extensión P01-P99 para comparar envolventes sin usar una forma."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 8:
        return None
    return np.percentile(p, 99.0, axis=0) - np.percentile(p, 1.0, axis=0)


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


def build_edge_incidence(triangles):
    """Relaciona cada arista no orientada con sus caras incidentes."""
    edges = defaultdict(list)
    for fi, tri in enumerate(triangles):
        a, b, c = map(int, tri)
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            edges[key].append(fi)
    return edges


def face_components(triangles, edge_faces=None):
    """Etiqueta componentes de caras conectadas por aristas compartidas."""
    n = len(triangles)
    if n == 0:
        return 0, np.empty(0, dtype=np.int32)

    if edge_faces is None:
        edge_faces = build_edge_incidence(triangles)

    rows = []
    cols = []
    for inc in edge_faces.values():
        if len(inc) < 2:
            continue
        root = inc[0]
        for other in inc[1:]:
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

    count, labels = connected_components(
        graph,
        directed=False,
    )
    return int(count), labels.astype(np.int32)


@operacion("Analizar conectividad de la malla")
def component_analysis(vertices, triangles, edge_faces=None):
    """Resume triángulos y áreas por componente de la malla."""
    count, labels = face_components(triangles, edge_faces)
    areas = triangle_areas(vertices, triangles)

    if count == 0:
        return {
            "count": 0,
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
        "largest_triangle_ratio": float(np.max(tri_counts) / max(np.sum(tri_counts), 1)),
        "largest_area_ratio": float(np.max(comp_areas) / max(np.sum(comp_areas), 1e-12)),
    }


@operacion("Analizar fronteras abiertas")
def boundary_analysis(vertices, triangles, edge_faces=None):
    """Analiza las aristas abiertas y la geometría de sus componentes."""
    if edge_faces is None:
        edge_faces = build_edge_incidence(triangles)

    boundary = [edge for edge, inc in edge_faces.items() if len(inc) == 1]

    nonmanifold = [edge for edge, inc in edge_faces.items() if len(inc) > 2]

    adjacency = defaultdict(list)
    for a, b in boundary:
        adjacency[a].append(b)
        adjacency[b].append(a)

    seen = set()
    components = []

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

        comp_set = set(verts)
        perimeter = 0.0
        edge_count = 0

        for a, b in boundary:
            if a in comp_set and b in comp_set:
                perimeter += float(np.linalg.norm(vertices[a] - vertices[b]))
                edge_count += 1

        components.append(
            {
                "vertices": int(len(verts)),
                "edges": int(edge_count),
                "perimeter_mm": float(perimeter),
            }
        )

    return {
        "unique_edge_count": int(len(edge_faces)),
        "boundary_edge_count": int(len(boundary)),
        "boundary_edge_ratio": float(len(boundary) / max(len(edge_faces), 1)),
        "boundary_component_count": int(len(components)),
        "total_boundary_perimeter_mm": float(sum(c["perimeter_mm"] for c in components)),
        "nonmanifold_edge_count_numpy": int(len(nonmanifold)),
    }


def topology(mesh):
    """Recoge las comprobaciones topológicas de Open3D."""
    result = {}

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
            result[key] = bool(fn())
        except Exception:
            result[key] = None

    try:
        result["non_manifold_vertex_count"] = int(len(mesh.get_non_manifold_vertices()))
    except Exception:
        result["non_manifold_vertex_count"] = None

    return result


@operacion("Medir distancia nube a malla")
def cloud_to_mesh_distances(mesh, points):
    """Calcula las distancias de los puntos a la superficie mediante raycasting."""
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    query = o3d.core.Tensor(np.asarray(points, dtype=np.float32))

    return (
        scene.compute_distance(
            query,
            nthreads=0,
        )
        .numpy()
        .astype(np.float64)
    )


@operacion("Medir distancia malla a nube")
def mesh_to_cloud_distances(mesh, cloud_points, count):
    """Muestrea la superficie y mide su distancia a los puntos de la nube."""
    sample = mesh.sample_points_uniformly(number_of_points=int(count))
    points = np.asarray(
        sample.points,
        dtype=np.float64,
    )

    tree = cKDTree(cloud_points)
    distances, _ = tree.query(
        points,
        k=1,
        workers=query_threads(),
    )
    return distances


def make_preview(path, cloud, mesh, report):
    if plt is None:
        return

    vertices = np.asarray(mesh.vertices)

    cloud_show = cloud
    if len(cloud_show) > 80000:
        cloud_show = cloud_show[
            np.linspace(
                0,
                len(cloud_show) - 1,
                80000,
            ).astype(int)
        ]

    mesh_show = vertices
    if len(mesh_show) > 80000:
        mesh_show = mesh_show[
            np.linspace(
                0,
                len(mesh_show) - 1,
                80000,
            ).astype(int)
        ]

    fig = plt.figure(figsize=(15, 10))

    for sp, dims, title in (
        (1, (0, 1), "X-Y"),
        (2, (0, 2), "X-Z"),
        (3, (2, 1), "Z-Y"),
    ):
        ax = fig.add_subplot(2, 2, sp)
        ax.scatter(
            cloud_show[:, dims[0]],
            cloud_show[:, dims[1]],
            s=0.15,
            alpha=0.2,
            label="nube",
        )
        ax.scatter(
            mesh_show[:, dims[0]],
            mesh_show[:, dims[1]],
            s=0.25,
            alpha=0.5,
            label="malla",
        )
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")

    ax = fig.add_subplot(2, 2, 4)
    ax.axis("off")

    geom = report["geometry_fidelity"]
    semantic = report["intersection_validation"]

    lines = [
        f"Paso 17 | {report['quality']}",
        f"coverage <= {geom['coverage_gate_mm']:.1f} mm: " f"{geom['cloud_to_mesh_coverage']:.2%}",
        f"cloud->mesh P90: " f"{geom['cloud_to_mesh_mm']['p90']:.3f} mm",
        f"mesh->cloud P90: " f"{geom['mesh_to_cloud_mm']['p90']:.3f} mm",
        f"intersecciones reales: " f"{semantic['true_self_intersection_pairs']}",
        f"contactos coincidentes: " f"{semantic['coincident_contact_pairs']}",
        f"boundary edge ratio: " f"{report['boundaries']['boundary_edge_ratio']:.2%}",
        f"closure: {report['closure_status']}",
    ]

    ax.text(
        0.03,
        0.97,
        "\n\n".join(lines),
        va="top",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    """Evalúa calidad geométrica y topológica y guarda el dictamen final."""
    args = parser().parse_args()

    root = Path(args.root).expanduser().resolve()
    obj = args.object.strip()

    cloud_dir = root / "reconstruccion" / "multisesion" / args.cloud_source

    mesh_dir = root / "reconstruccion" / "multisesion" / args.mesh_source

    intersection_dir = root / "reconstruccion" / "multisesion" / args.intersection_source

    cloud_path = cloud_dir / "nube_regularizada_general.npz"
    mesh_path = mesh_dir / "malla_final_topologica.ply"
    intersection_path = intersection_dir / "resumen_16_validacion_intersecciones.json"

    if not cloud_path.is_file():
        raise FileNotFoundError(cloud_path)
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    if not intersection_path.is_file():
        raise FileNotFoundError(intersection_path)

    intersection_info = json.loads(intersection_path.read_text(encoding="utf-8"))

    if intersection_info.get("status") != "ready_for_step_17":
        raise RuntimeError(
            "paso 16 V1.4 no habilitó paso 17. "
            f"Estado={intersection_info.get('status')} | "
            f"Bloqueos={intersection_info.get('blockers_for_step_17')}"
        )

    with np.load(cloud_path) as data:
        cloud_points = np.asarray(
            data["points"],
            dtype=np.float64,
        )
        support = np.asarray(
            data["support_views"],
            dtype=np.float64,
        )
        confidence = np.asarray(
            data["confidence"],
            dtype=np.float64,
        )
        strong_core_support = int(
            np.asarray(
                data["strong_core_support_threshold"]
                if "strong_core_support_threshold" in data.files
                else np.asarray([args.strong_core_support], dtype=np.uint16)
            ).ravel()[0]
        )
        evidence_contract_available = "surface_evidence_class" in data.files
        evidence_class = np.asarray(
            (
                data["surface_evidence_class"]
                if evidence_contract_available
                else np.full(len(cloud_points), 2)
            ),
            dtype=np.uint8,
        ).reshape(-1)
        independent_support = np.asarray(
            (
                data["independent_support_poses"]
                if "independent_support_poses" in data.files
                else support
            ),
            dtype=np.float64,
        ).reshape(-1)
        angular_span = np.asarray(
            (
                data["support_angular_span_poses"]
                if "support_angular_span_poses" in data.files
                else np.zeros(len(cloud_points))
            ),
            dtype=np.int16,
        ).reshape(-1)
        conflict_ratio = np.asarray(
            (
                data["conflict_pose_ratio"]
                if "conflict_pose_ratio" in data.files
                else np.zeros(len(cloud_points))
            ),
            dtype=np.float64,
        ).reshape(-1)
        heldout_available = np.asarray(
            (
                data["heldout_validation_available"]
                if "heldout_validation_available" in data.files
                else np.zeros(len(cloud_points))
            ),
            dtype=bool,
        ).reshape(-1)
        heldout_pass = np.asarray(
            (
                data["heldout_validation_pass_ratio"]
                if "heldout_validation_pass_ratio" in data.files
                else np.ones(len(cloud_points))
            ),
            dtype=np.float64,
        ).reshape(-1)
        evidence_strength = np.asarray(
            (
                data["evidence_strength"]
                if "evidence_strength" in data.files
                else np.ones(len(cloud_points))
            ),
            dtype=np.float64,
        ).reshape(-1)
        evidence_arrays = (
            evidence_class,
            independent_support,
            angular_span,
            conflict_ratio,
            heldout_available,
            heldout_pass,
            evidence_strength,
        )
        if any(len(a) != len(cloud_points) for a in evidence_arrays):
            raise ValueError("Paso 17: contrato de evidencia incompatible con la nube del paso 12.")
        if bool(args.require_evidence_contract) and not evidence_contract_available:
            raise RuntimeError(
                "Paso 17 V1.8 requiere el contrato de evidencia de 11/12; "
                "no se valida un modelo final usando únicamente soporte bruto."
            )
        sampling_spacing_mm = None
        sampling_spacing_source = None
        for spacing_key in (
            "meshing_reference_spacing_mm",
            "reference_spacing_mm",
            "fusion_voxel_mm",
        ):
            if spacing_key not in data.files:
                continue
            spacing_value = float(np.asarray(data[spacing_key], dtype=np.float64).ravel()[0])
            if np.isfinite(spacing_value) and spacing_value > 0.0:
                sampling_spacing_mm = spacing_value
                sampling_spacing_source = spacing_key
                break

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise RuntimeError("La malla está vacía.")

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64,
    )
    triangles = np.asarray(
        mesh.triangles,
        dtype=np.int64,
    )

    rng = np.random.default_rng(5412)

    cloud_query = cloud_points
    if len(cloud_query) > int(args.cloud_samples):
        idx = rng.choice(
            len(cloud_query),
            int(args.cloud_samples),
            replace=False,
        )
        cloud_query = cloud_query[idx]

    c2m = cloud_to_mesh_distances(
        mesh,
        cloud_query,
    )

    if evidence_contract_available:
        strong_core_mask = (
            (evidence_class >= int(args.strong_core_min_evidence_class))
            & (independent_support >= int(args.strong_core_min_independent_support))
            & (angular_span >= int(args.strong_core_min_angular_span_poses))
            & (conflict_ratio <= float(args.strong_core_max_conflict_pose_ratio))
            & (
                (~heldout_available)
                | (heldout_pass >= float(args.strong_core_min_heldout_pass_ratio))
            )
        )
        strong_core_mode = "pose_diverse_heldout_validated_evidence"
    else:
        strong_core_mask = support >= int(strong_core_support)
        strong_core_mode = "legacy_raw_support_fallback"
    strong_core_points = cloud_points[strong_core_mask]
    strong_core_available = len(strong_core_points) >= int(args.strong_core_minimum_points)
    strong_c2m_stats = None
    strong_coverage = None
    if strong_core_available:
        strong_query = strong_core_points
        if len(strong_query) > int(args.cloud_samples):
            idx = rng.choice(
                len(strong_query),
                int(args.cloud_samples),
                replace=False,
            )
            strong_query = strong_query[idx]
        strong_c2m = cloud_to_mesh_distances(mesh, strong_query)
        strong_c2m_stats = stats(strong_c2m)
        strong_coverage = float(np.mean(strong_c2m <= float(args.coverage_gate_mm)))

    mesh_sample_count = min(
        int(args.mesh_samples),
        max(20000, len(vertices)),
    )

    m2c = mesh_to_cloud_distances(
        mesh,
        cloud_points,
        mesh_sample_count,
    )

    c2m_stats = stats(c2m)
    m2c_stats = stats(m2c)
    mesh_cloud_limits = adaptive_mesh_cloud_limits(
        sampling_spacing_mm=sampling_spacing_mm,
        enabled=bool(args.adaptive_mesh_cloud_thresholds),
        warning_absolute_mm=float(args.warning_mesh_cloud_p90_mm),
        reject_absolute_mm=float(args.reject_mesh_cloud_p90_mm),
        warning_spacing_factor=float(args.warning_mesh_cloud_spacing_factor),
        reject_spacing_factor=float(args.reject_mesh_cloud_spacing_factor),
        maximum_reject_mm=float(args.maximum_adaptive_reject_mesh_cloud_p90_mm),
    )
    warning_mesh_cloud_p90_mm = float(mesh_cloud_limits["effective_warning_mm"])
    reject_mesh_cloud_p90_mm = float(mesh_cloud_limits["effective_reject_mm"])

    coverage = float(np.mean(c2m <= float(args.coverage_gate_mm)))

    cloud_extent = np.ptp(
        cloud_points,
        axis=0,
    )
    mesh_extent = np.ptp(
        vertices,
        axis=0,
    )

    bbox_ratio = np.divide(
        mesh_extent,
        cloud_extent,
        out=np.full(3, np.nan),
        where=cloud_extent > 1e-12,
    )

    max_bbox_expansion = float(np.nanmax(bbox_ratio))

    cloud_robust_extent = robust_extent(cloud_points)
    strong_robust_extent = robust_extent(strong_core_points) if strong_core_available else None
    if cloud_robust_extent is not None and strong_robust_extent is not None:
        observation_envelope_ratio = np.divide(
            cloud_robust_extent,
            strong_robust_extent,
            out=np.full(3, np.nan, dtype=np.float64),
            where=strong_robust_extent > 1e-9,
        )
        max_observation_envelope_expansion = float(np.nanmax(observation_envelope_ratio))
    else:
        observation_envelope_ratio = np.full(3, np.nan, dtype=np.float64)
        max_observation_envelope_expansion = None

    topo = topology(mesh)
    edge_faces = build_edge_incidence(triangles)
    components = component_analysis(
        vertices,
        triangles,
        edge_faces=edge_faces,
    )
    boundaries = boundary_analysis(
        vertices,
        triangles,
        edge_faces=edge_faces,
    )

    reject = []
    warning = []

    # ------------------------------------------------------------------
    # Fidelidad geométrica
    # ------------------------------------------------------------------
    if coverage < float(args.reject_coverage):
        reject.append(f"Cobertura insuficiente: {coverage:.2%}.")
    elif coverage < float(args.warning_coverage):
        warning.append(f"Cobertura moderada: {coverage:.2%}.")

    if c2m_stats["p90"] > float(args.reject_cloud_mesh_p90_mm):
        reject.append(f"P90 cloud->mesh alto: " f"{c2m_stats['p90']:.3f} mm.")
    elif c2m_stats["p90"] > float(args.warning_cloud_mesh_p90_mm):
        warning.append(f"P90 cloud->mesh: " f"{c2m_stats['p90']:.3f} mm.")

    # La decisión mesh->cloud se difiere hasta comprobar coherencia global.
    # Un P90 alto puede corresponder a una zona legítimamente interpolada:
    # por sí solo no demuestra que la superficie sea falsa.
    mesh_cloud_exceeds_reject = bool(m2c_stats["p90"] > reject_mesh_cloud_p90_mm)
    mesh_cloud_exceeds_warning = bool(m2c_stats["p90"] > warning_mesh_cloud_p90_mm)

    if max_bbox_expansion > float(args.reject_bbox_expansion_ratio):
        reject.append("Expansión excesiva de bounding box: " f"{max_bbox_expansion:.3f}.")
    elif max_bbox_expansion > float(args.warning_bbox_expansion_ratio):
        warning.append("Expansión moderada de bounding box: " f"{max_bbox_expansion:.3f}.")

    # Una malla puede reproducir fielmente una nube contaminada. Por eso se
    # contrasta también con el núcleo respaldado por varias vistas y se mide
    # cuánto amplían la envolvente las observaciones de menor soporte.
    if strong_core_available:
        if strong_coverage < float(args.reject_coverage):
            reject.append(
                "Cobertura insuficiente del núcleo multivista fuerte: " f"{strong_coverage:.2%}."
            )
        elif strong_coverage < float(args.warning_coverage):
            warning.append(
                "Cobertura moderada del núcleo multivista fuerte: " f"{strong_coverage:.2%}."
            )

        if strong_c2m_stats["p90"] > float(args.reject_cloud_mesh_p90_mm):
            reject.append("P90 núcleo fuerte->mesh alto: " f"{strong_c2m_stats['p90']:.3f} mm.")
        elif strong_c2m_stats["p90"] > float(args.warning_cloud_mesh_p90_mm):
            warning.append("P90 núcleo fuerte->mesh: " f"{strong_c2m_stats['p90']:.3f} mm.")

        if max_observation_envelope_expansion is not None:
            if max_observation_envelope_expansion > float(
                args.reject_observation_envelope_expansion_ratio
            ):
                reject.append(
                    "Las observaciones débiles expanden excesivamente la "
                    "envolvente del núcleo multivista: "
                    f"{max_observation_envelope_expansion:.3f}."
                )
            elif max_observation_envelope_expansion > float(
                args.warning_observation_envelope_expansion_ratio
            ):
                warning.append(
                    "Las observaciones débiles expanden moderadamente la "
                    "envolvente del núcleo multivista: "
                    f"{max_observation_envelope_expansion:.3f}."
                )
    else:
        warning.append(
            "No hay suficientes puntos para validar de forma independiente "
            "el núcleo de evidencia independiente/held-out configurado."
        )

    # ------------------------------------------------------------------
    # Topología estructural
    # ------------------------------------------------------------------
    if topo.get("edge_manifold_allow_boundary") is not True:
        reject.append("La malla no es edge-manifold.")

    if topo.get("vertex_manifold") is not True:
        reject.append("La malla no es vertex-manifold.")

    if topo.get("orientable") is not True:
        reject.append("La malla no es orientable.")

    if topo.get("non_manifold_vertex_count") not in (0, None):
        reject.append("Quedan vértices non-manifold.")

    # ------------------------------------------------------------------
    # Intersecciones semánticas del paso 16 V1.4.
    # ------------------------------------------------------------------
    true_intersections = int(intersection_info["true_self_intersection_pairs"])
    ambiguous = int(intersection_info["ambiguous_intersection_pairs"])
    contacts = int(intersection_info["coincident_contact_pairs"])

    if true_intersections > 0:
        reject.append(
            f"Persisten {true_intersections} pares de " "auto-intersección geométrica real."
        )

    if ambiguous > 0:
        reject.append(f"Persisten {ambiguous} pares ambiguos.")

    if contacts > 0:
        warning.append(
            f"Se documentan {contacts} pares de contacto "
            "geométrico coincidente; no son cruces transversales."
        )

    # ------------------------------------------------------------------
    # Fragmentación
    # ------------------------------------------------------------------
    largest_area = components["largest_area_ratio"]

    if largest_area is not None and largest_area < float(args.reject_largest_component_area_ratio):
        reject.append("Fragmentación fuerte: componente principal " f"{largest_area:.2%} del área.")
    elif largest_area is not None and largest_area < float(
        args.warning_largest_component_area_ratio
    ):
        warning.append(
            "Componentes secundarias relevantes: principal " f"{largest_area:.2%} del área."
        )

    # ------------------------------------------------------------------
    # Aperturas
    # ------------------------------------------------------------------
    boundary_ratio = float(boundaries["boundary_edge_ratio"])

    if boundary_ratio > float(args.reject_boundary_edge_ratio):
        reject.append(
            "Superficie excesivamente abierta: " f"{boundary_ratio:.2%} de aristas de borde."
        )
    elif boundary_ratio > float(args.warning_boundary_edge_ratio):
        warning.append("Superficie abierta: " f"{boundary_ratio:.2%} de aristas de borde.")

    # ------------------------------------------------------------------
    # Coherencia de superficies interpoladas
    # ------------------------------------------------------------------
    # mesh->cloud mide distancia a muestras discretas, no falsedad geométrica.
    # Por eso un P90 alto solo bloquea cuando también falla alguna evidencia
    # independiente de continuidad, cobertura, envolvente o topología.
    if cloud_robust_extent is not None:
        reference_extent = float(np.max(cloud_robust_extent))
    else:
        reference_extent = float(np.max(cloud_extent))
    mesh_cloud_p95_extent_ratio = float(m2c_stats["p95"] / max(reference_extent, 1e-9))

    coherent_interpolation_criteria = {
        "no_other_rejection": len(reject) == 0,
        "cloud_coverage": coverage >= float(args.coherent_interpolation_min_coverage),
        "cloud_to_mesh_p90": c2m_stats["p90"] <= float(args.reject_cloud_mesh_p90_mm),
        "strong_core_available": bool(strong_core_available),
        "strong_core_coverage": bool(
            strong_core_available
            and strong_coverage >= float(args.coherent_interpolation_min_coverage)
        ),
        "strong_core_to_mesh_p90": bool(
            strong_core_available
            and strong_c2m_stats["p90"] <= float(args.reject_cloud_mesh_p90_mm)
        ),
        "bbox_expansion": max_bbox_expansion
        <= float(args.coherent_interpolation_max_bbox_expansion_ratio),
        "dominant_connected_surface": bool(
            largest_area is not None
            and largest_area >= float(args.coherent_interpolation_min_largest_area_ratio)
        ),
        "limited_boundary": boundary_ratio
        <= float(args.coherent_interpolation_max_boundary_edge_ratio),
        "relative_interpolation_extent": mesh_cloud_p95_extent_ratio
        <= float(args.coherent_interpolation_max_p95_extent_ratio),
        "manifold_and_orientable": bool(
            topo.get("edge_manifold_allow_boundary") is True
            and topo.get("vertex_manifold") is True
            and topo.get("orientable") is True
        ),
        "no_true_or_ambiguous_intersections": bool(true_intersections == 0 and ambiguous == 0),
    }
    coherent_interpolation = bool(
        args.allow_coherent_interpolation and all(coherent_interpolation_criteria.values())
    )

    if mesh_cloud_exceeds_reject:
        mesh_cloud_decision = (
            "reject_despite_coherent_interpolation"
            if coherent_interpolation else "reject_not_coherent"
        )
        detail = (
            " La superficie cumple los controles auxiliares de coherencia, "
            "pero éstos no sustituyen evidencia observacional local."
            if coherent_interpolation else ""
        )
        reject.append(
            f"P90 mesh->cloud alto: {m2c_stats['p90']:.3f} mm > "
            f"{reject_mesh_cloud_p90_mm:.3f} mm.{detail}"
        )
    elif mesh_cloud_exceeds_warning:
        mesh_cloud_decision = "warning_within_reject_limit"
        warning.append(f"P90 mesh->cloud: {m2c_stats['p90']:.3f} mm.")
    else:
        mesh_cloud_decision = "accepted_within_warning_limit"

    if (
        topo.get("edge_manifold_allow_boundary") is True
        and topo.get("vertex_manifold") is True
        and topo.get("orientable") is True
        and true_intersections == 0
        and ambiguous == 0
    ):
        if boundaries["boundary_edge_count"] == 0 and contacts == 0:
            closure_status = "closed_embedded_manifold"
        elif boundaries["boundary_edge_count"] == 0:
            closure_status = "closed_manifold_with_coincident_contacts"
        else:
            closure_status = "open_manifold_surface"
    else:
        closure_status = "topologically_invalid"

    # Volumen: deliberadamente más conservador que Open3D raw.
    volume = None
    if closure_status == "closed_embedded_manifold" and contacts == 0:
        try:
            volume = float(mesh.get_volume())
        except Exception:
            volume = None

    try:
        surface_area = float(mesh.get_surface_area())
    except Exception:
        surface_area = None

    quality = "rejected" if reject else ("warning" if warning else "accepted")

    report = {
        "schema_version": "1.5",
        "version": VERSION,
        "method": (
            "general_geometric_fidelity_topology_"
            "classified_self_intersection_and_coherent_interpolation_"
            "validation"
        ),
        "object": obj,
        "quality": quality,
        "reject_reasons": reject,
        "warning_reasons": warning,
        "shape_specific_assumptions": False,
        "geometry_fidelity": {
            "coverage_gate_mm": float(args.coverage_gate_mm),
            "cloud_to_mesh_coverage": coverage,
            "cloud_to_mesh_mm": c2m_stats,
            "mesh_to_cloud_mm": m2c_stats,
            "mesh_to_cloud_thresholds": {
                **mesh_cloud_limits,
                "sampling_spacing_source": sampling_spacing_source,
            },
            "coherent_interpolation_validation": {
                "enabled": bool(args.allow_coherent_interpolation),
                "coherent": bool(coherent_interpolation),
                "mesh_to_cloud_decision": mesh_cloud_decision,
                "mesh_to_cloud_p95_extent_ratio": float(mesh_cloud_p95_extent_ratio),
                "reference_extent_mm": float(reference_extent),
                "thresholds": {
                    "minimum_coverage": float(args.coherent_interpolation_min_coverage),
                    "maximum_bbox_expansion_ratio": float(
                        args.coherent_interpolation_max_bbox_expansion_ratio
                    ),
                    "minimum_largest_component_area_ratio": float(
                        args.coherent_interpolation_min_largest_area_ratio
                    ),
                    "maximum_boundary_edge_ratio": float(
                        args.coherent_interpolation_max_boundary_edge_ratio
                    ),
                    "maximum_mesh_cloud_p95_extent_ratio": float(
                        args.coherent_interpolation_max_p95_extent_ratio
                    ),
                },
                "criteria": coherent_interpolation_criteria,
                "policy": (
                    "mesh_to_cloud_p90_never_rejects_by_itself_when_all_"
                    "independent_coherence_checks_pass"
                ),
            },
            "cloud_bbox_extent_mm": (cloud_extent.astype(float).tolist()),
            "mesh_bbox_extent_mm": (mesh_extent.astype(float).tolist()),
            "mesh_to_cloud_bbox_extent_ratio_xyz": (bbox_ratio.astype(float).tolist()),
            "maximum_bbox_expansion_ratio": (max_bbox_expansion),
            "strong_multiview_core": {
                "mode": strong_core_mode,
                "legacy_raw_support_threshold": int(strong_core_support),
                "criteria": {
                    "minimum_evidence_class": int(args.strong_core_min_evidence_class),
                    "minimum_independent_support": int(args.strong_core_min_independent_support),
                    "minimum_angular_span_poses": int(args.strong_core_min_angular_span_poses),
                    "maximum_conflict_pose_ratio": float(args.strong_core_max_conflict_pose_ratio),
                    "minimum_heldout_pass_ratio": float(args.strong_core_min_heldout_pass_ratio),
                },
                "minimum_points_required": int(args.strong_core_minimum_points),
                "available": bool(strong_core_available),
                "points": int(len(strong_core_points)),
                "cloud_to_mesh_coverage": strong_coverage,
                "cloud_to_mesh_mm": strong_c2m_stats,
            },
            "observation_envelope_check": {
                "all_cloud_robust_extent_mm": (
                    None
                    if cloud_robust_extent is None
                    else cloud_robust_extent.astype(float).tolist()
                ),
                "strong_core_robust_extent_mm": (
                    None
                    if strong_robust_extent is None
                    else strong_robust_extent.astype(float).tolist()
                ),
                "all_to_strong_extent_ratio_xyz": [
                    None if not np.isfinite(x) else float(x) for x in observation_envelope_ratio
                ],
                "maximum_expansion_ratio": max_observation_envelope_expansion,
            },
        },
        "evidence_contract_validation": {
            "available": bool(evidence_contract_available),
            "required": bool(args.require_evidence_contract),
            "surface_evidence_class": stats(evidence_class),
            "independent_support_poses": stats(independent_support),
            "support_angular_span_poses": stats(angular_span),
            "conflict_pose_ratio": stats(conflict_ratio),
            "heldout_pass_ratio": (
                stats(heldout_pass[heldout_available]) if np.any(heldout_available) else stats([])
            ),
            "evidence_strength": stats(evidence_strength),
            "strong_core_mode": strong_core_mode,
            "strong_core_points": int(len(strong_core_points)),
            "policy": "final_validation_uses_pose_diverse_evidence_not_raw_pose_count",
        },
        "raw_open3d_topology": topo,
        "intersection_validation": {
            "source": str(intersection_path),
            "raw_open3d_intersection_pairs": int(
                intersection_info["raw_open3d_intersection_pairs"]
            ),
            "coincident_contact_pairs": contacts,
            "true_self_intersection_pairs": (true_intersections),
            "ambiguous_intersection_pairs": ambiguous,
            "classification_counts": (intersection_info["classification_counts"]),
            "semantic_status": (intersection_info["semantic_self_intersection_status"]),
        },
        "components": components,
        "boundaries": boundaries,
        "closure_status": closure_status,
        "surface_area_mm2": surface_area,
        "volume_mm3_if_valid_closed_surface": volume,
        "support_views": stats(support),
        "independent_support_poses": stats(independent_support),
        "fusion_confidence": stats(confidence),
        "policy": {
            "raw_open3d_self_intersection_is_diagnostic": True,
            "coincident_contacts_are_documented": True,
            "coincident_contacts_block_geometric_fidelity": False,
            "true_transverse_or_coplanar_intersections_block": True,
            "ambiguous_intersections_block": True,
            "watertight_required_for_general_geometric_validity": False,
            "closed_embedded_surface_required_for_volume": True,
            "strong_multiview_core_checked_independently": True,
            "raw_pose_count_is_not_strong_core_evidence": True,
            "weak_observation_envelope_is_shape_agnostic": True,
            "coherent_connected_interpolation_is_not_rejected_by_" "mesh_cloud_p90_alone": True,
        },
    }

    output = root / "reconstruccion" / "multisesion" / args.output_name
    output.mkdir(parents=True, exist_ok=True)

    report_path = output / "resumen_17_validacion_modelo.json"
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    preview_path = output / "preview_validacion_modelo_17.png"
    make_preview(
        preview_path,
        cloud_points,
        mesh,
        report,
    )

    print(f"\n========== PASO 17 {VERSION} COMPLETADA ==========")
    print("Calidad:", quality)
    print(f"Cobertura cloud->mesh: {coverage:.2%}")
    print(
        "P90 cloud->mesh:",
        c2m_stats["p90"],
        "mm",
    )
    if strong_core_available:
        print(
            f"Núcleo evidencia ({strong_core_mode}): "
            f"{len(strong_core_points):,} puntos | "
            f"cobertura={strong_coverage:.2%}"
        )
        print(
            "Expansión máxima observaciones/núcleo:",
            max_observation_envelope_expansion,
        )
    else:
        print(
            f"Núcleo evidencia ({strong_core_mode}): " "insuficiente para validación independiente"
        )
    print(
        "P90 mesh->cloud:",
        m2c_stats["p90"],
        "mm",
    )
    print(
        "Gate mesh->cloud:",
        f"warning>{warning_mesh_cloud_p90_mm:.3f} mm | ",
        f"reject>{reject_mesh_cloud_p90_mm:.3f} mm | ",
        f"modo={mesh_cloud_limits['mode']}",
    )
    print(
        "Decisión mesh->cloud:",
        mesh_cloud_decision,
        "| interpolación coherente:",
        coherent_interpolation,
    )
    if sampling_spacing_mm is not None:
        print(
            "Spacing usado por el gate:",
            f"{sampling_spacing_mm:.6f} mm",
            f"({sampling_spacing_source})",
        )
    print(
        "Intersecciones reales:",
        true_intersections,
    )
    print("Ambiguas:", ambiguous)
    print("Contactos coincidentes:", contacts)
    print("Closure:", closure_status)
    print(f"Boundary edge ratio: {boundary_ratio:.2%}")
    print("Volumen:", volume)
    print("Razones de rechazo:")
    if reject:
        for reason in reject:
            print("  -", reason)
    else:
        print("  - Ninguna")
    print("Advertencias:")
    if warning:
        for reason in warning:
            print("  -", reason)
    else:
        print("  - Ninguna")
    print("Salida:", output)
    print("=================================================\n")

    return 0 if quality != "rejected" else 2


def _entrada_con_diagnostico():
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    from utilidades_progreso import informar_inicio

    informar_inicio(__file__, "17", "Validar modelo final")
    raise SystemExit(main())


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
