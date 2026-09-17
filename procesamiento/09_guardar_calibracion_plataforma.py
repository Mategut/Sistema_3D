#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paso 09 — Congelar calibración candidata de la plataforma.

Objetivo
--------
Convertir el resultado geométrico V3.2 obtenido durante el desarrollo en una
CALIBRACIÓN DEL SISTEMA, almacenada fuera de cualquier objeto particular.

Principios:
- El eje y la línea de rotación son propiedades del montaje.
- 2055 pasos/vuelta es propiedad mecánica del montaje.
- Las correcciones angulares V3.2 se conservan SOLO como evidencia diagnóstica.
- Para objetos futuros se utilizan los ángulos MECÁNICOS, sin correcciones
  aprendidas del objeto de referencia.
- La línea del eje se representa mediante el punto de esa línea más próximo
  al punto medio estéreo. Esto elimina la ambigüedad de elegir cualquier punto
  a lo largo de la misma línea.
- La salida inicial es "candidate". No se promueve a definitiva hasta superar
  validación independiente con otros barridos.

No requiere que los objetos futuros sean cubos, cilindros, pirámides ni
poliedros.
"""

from __future__ import annotations
from utilidades_progreso import operacion
from utilidades_rendimiento import configurar as _configurar_recursos

# Los runtimes numéricos leen estos límites al importarse.
_configurar_recursos(__file__)


import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

STEPS_PER_REVOLUTION = 2055
POSES_PER_REVOLUTION = 25
DEFAULT_STEP_SEQUENCE = [
    82,
    82,
    83,
    82,
    82,
    82,
    82,
    83,
    82,
    82,
    82,
    82,
    83,
    82,
    82,
    82,
    82,
    83,
    82,
    82,
    82,
    82,
    83,
    82,
    82,
]


def parser():
    """Construye las opciones de línea de comandos de este paso."""
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--source-object", default="cubo")
    p.add_argument(
        "--registration-source",
        default="08_registro_referencia",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directorio persistente donde guardar la calibración de plataforma.",
    )
    p.add_argument(
        "--stereo-calibration-dir",
        required=True,
        help="Calibración estéreo vigente; no se buscan calibraciones históricas.",
    )
    p.add_argument("--baseline-mm", type=float, default=81.0558)
    p.add_argument("--mount-reference-distance-mm", type=float, default=400.0)
    p.add_argument("--mount-reference-sigma-mm", type=float, default=30.0)

    # Gates para aceptar que V3.2 es suficientemente estable como CANDIDATA.
    p.add_argument("--minimum-primary-surface-accepted-ratio", type=float, default=0.95)
    p.add_argument("--maximum-primary-point-plane-rmse-median-mm", type=float, default=2.5)
    p.add_argument("--maximum-primary-point-plane-p90-median-mm", type=float, default=4.0)
    p.add_argument("--minimum-closure-overlap", type=float, default=0.80)
    p.add_argument("--maximum-closure-rmse-mm", type=float, default=2.5)
    p.add_argument("--maximum-closure-point-plane-rmse-mm", type=float, default=2.0)
    p.add_argument("--maximum-angle-correction-deg", type=float, default=0.35)
    p.add_argument("--maximum-adaptive-saturations", type=int, default=1)
    p.add_argument("--maximum-manhattan-p90-deg", type=float, default=12.0)
    p.add_argument("--maximum-reference-compactness-p90-mm", type=float, default=6.0)
    p.add_argument("--maximum-mount-reference-error-mm", type=float, default=65.0)
    return p


def load_json(path: Path) -> dict:
    """Lee un documento JSON codificado en UTF-8."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    """Calcula la huella SHA-256 del archivo mediante lectura por bloques."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_existing(paths: Sequence[Path]) -> Dict[str, str]:
    """Obtiene las firmas SHA-256 de los archivos presentes."""
    out = {}
    for p in paths:
        if p.is_file():
            out[str(p.resolve())] = sha256_file(p)
    return out


def finite(values):
    """Convierte los valores a float64 y descarta los no finitos."""
    a = np.asarray(list(values), dtype=np.float64)
    return a[np.isfinite(a)]


def stats(values) -> dict:
    a = finite(values)
    if len(a) == 0:
        return {"count": 0, "median": None, "mean": None, "min": None, "max": None, "p90": None}
    return {
        "count": int(len(a)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "p90": float(np.percentile(a, 90.0)),
    }


def normalize(v) -> np.ndarray:
    """Normaliza un vector 3D y rechaza normas nulas o no finitas."""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("Vector inválido.")
    return v / n


def rotation_matrix(axis, angle_rad):
    """Calcula la matriz de rotación para un eje 3D y un ángulo en radianes."""
    a = normalize(axis)
    x, y, z = a
    K = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )
    I = np.eye(3)
    return I + math.sin(angle_rad) * K + (1.0 - math.cos(angle_rad)) * (K @ K)


def axis_transform(axis, line_point, angle_rad):
    """Construye el giro homogéneo alrededor de la línea de la plataforma."""
    R = rotation_matrix(axis, angle_rad)
    c = np.asarray(line_point, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = c - R @ c
    return T


def closest_point_on_line(point, line_point, axis):
    """Proyecta un punto sobre la línea definida por un origen y un eje."""
    p = np.asarray(point, dtype=np.float64).reshape(3)
    c = np.asarray(line_point, dtype=np.float64).reshape(3)
    a = normalize(axis)
    return c + float(np.dot(p - c, a)) * a


def point_to_line_distance(point, line_point, axis):
    """Calcula la distancia perpendicular de un punto a una línea 3D."""
    p = np.asarray(point, dtype=np.float64).reshape(3)
    c = np.asarray(line_point, dtype=np.float64).reshape(3)
    a = normalize(axis)
    return float(np.linalg.norm(np.cross(p - c, a)))


def explicit_stereo_calibration_files(folder: Path) -> List[Path]:
    """Exige los archivos de calibración estéreo en el directorio indicado."""
    folder = Path(folder).expanduser().resolve()
    required = [folder / "stereo_initial.yaml", folder / "rectification_maps.npz"]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Calibración estéreo explícita incompleta: " + ", ".join(map(str, missing))
        )
    optional = [folder / "left.yaml", folder / "right.yaml"]
    return required + [p for p in optional if p.is_file()]


def resolve_manifest(root: Path, obj: str) -> Path:
    """Localiza el manifiesto angular del trabajo de calibración."""
    path = (
        root
        / "reconstruccion"
        / "multisesion"
        / "01_mapa_angular"
        / "mapa_angular_multisesion.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Falta el manifiesto de ESTA ejecución: {path}")
    return path


def resolve_v32_files(root: Path, obj: str, source: str):
    """Localiza y verifica los productos necesarios del registro de referencia."""
    base = root / "reconstruccion" / "multisesion" / source
    summary = base / "resumen_08_registro_referencia.json"
    poses = base / "poses_registradas_v3_2.csv"
    edges = base / "calidad_aristas_v3_2.csv"
    obs = base / "observabilidad_angular_v3_2.csv"
    for p in (summary, poses, edges, obs):
        if not p.is_file():
            raise FileNotFoundError(f"Falta entrada V3.2: {p}")
    return base, summary, poses, edges, obs


def read_csv(path: Path) -> List[dict]:
    """Lee una tabla CSV con cabecera y devuelve sus filas como diccionarios."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def to_float(v, default=float("nan")):
    """Convierte a float y devuelve el valor de respaldo si falla la conversión."""
    try:
        return float(v)
    except Exception:
        return default


def to_bool(v):
    """Interpreta las representaciones textuales de verdadero admitidas por el flujo."""
    return str(v).strip().lower() in {"1", "true", "yes", "si", "sí"}


def extract_axis_center(summary: dict):
    """Obtiene eje y centro desde las variantes admitidas del resumen de registro."""
    axis_section = summary.get("axis", {})
    center_section = summary.get("center", {})

    axis = (
        axis_section.get("final_xyz")
        or axis_section.get("axis_final_xyz")
        or summary.get("final_axis_xyz")
    )
    center = (
        center_section.get("final_xyz_mm")
        or center_section.get("center_final_xyz_mm")
        or summary.get("final_center_xyz_mm")
    )
    sign = summary.get("direction_sign", summary.get("sign", +1))

    if axis is None or center is None:
        raise KeyError("El resumen V3.2 no contiene axis.final_xyz y/o center.final_xyz_mm.")
    return normalize(axis), np.asarray(center, dtype=np.float64), int(sign)


def extract_metrics(summary: dict, edges_rows: List[dict], obs_rows: List[dict]) -> dict:
    """Reúne métricas de calidad, cierre y observabilidad para la auditoría."""
    metrics = summary.get("quality_metrics", summary.get("metrics", {}))

    primary = [r for r in edges_rows if to_bool(r.get("primary"))]
    closure = [r for r in primary if to_bool(r.get("closure"))]
    nonclosure_primary = [r for r in primary if not to_bool(r.get("closure"))]

    accepted = [r for r in primary if to_bool(r.get("surface_accepted", r.get("accepted")))]

    pp_rmse = [to_float(r.get("point_plane_rmse_mm")) for r in primary]
    pp_p90 = [to_float(r.get("point_plane_p90_abs_mm")) for r in primary]

    closure_record = closure[0] if closure else None

    corrections = [abs(to_float(r.get("final_correction_deg"))) for r in obs_rows]
    saturations = sum(to_bool(r.get("saturated")) for r in obs_rows)

    manhattan_p90 = summary.get("manhattan", {}).get("normal_residual_deg", {}).get("p90")
    compactness = summary.get(
        "global_face_compactness",
        metrics.get("global_face_compactness", {}),
    )
    worst_compact = compactness.get("worst_face_p90_mm")

    mount = (
        summary.get("mounting_geometry")
        or summary.get("center", {}).get("mounting_geometry_final")
        or metrics.get("mounting_geometry", {})
    )

    return {
        "primary_edge_count": len(primary),
        "primary_surface_accepted_count": len(accepted),
        "primary_surface_accepted_ratio": (len(accepted) / len(primary) if primary else 0.0),
        "primary_point_plane_rmse_mm": stats(pp_rmse),
        "primary_point_plane_p90_abs_mm": stats(pp_p90),
        "closure": (
            None
            if closure_record is None
            else {
                "overlap": to_float(closure_record.get("overlap")),
                "trimmed_rmse_mm": to_float(closure_record.get("trimmed_rmse_mm")),
                "point_plane_rmse_mm": to_float(closure_record.get("point_plane_rmse_mm")),
                "point_plane_p90_abs_mm": to_float(closure_record.get("point_plane_p90_abs_mm")),
                "surface_accepted": to_bool(
                    closure_record.get("surface_accepted", closure_record.get("accepted"))
                ),
            }
        ),
        "angle_correction_abs_deg": stats(corrections),
        "adaptive_saturation_count": int(saturations),
        "manhattan_p90_deg_reference_only": (
            None if manhattan_p90 is None else float(manhattan_p90)
        ),
        "reference_compactness_worst_p90_mm": (
            None if worst_compact is None else float(worst_compact)
        ),
        "mounting_geometry": mount,
    }


def quality_gates(evidence: dict, args) -> Tuple[bool, List[dict]]:
    """Evalúa los criterios de aceptación y devuelve el detalle de cada comprobación."""
    checks = []

    def add(name, value, threshold, passed, meaning):
        checks.append(
            {
                "name": name,
                "value": value,
                "threshold": threshold,
                "passed": bool(passed),
                "meaning": meaning,
            }
        )

    ratio = evidence["primary_surface_accepted_ratio"]
    add(
        "primary_surface_accepted_ratio",
        ratio,
        f">={args.minimum_primary_surface_accepted_ratio}",
        ratio >= args.minimum_primary_surface_accepted_ratio,
        "Consistencia de superficie en el grafo de la referencia.",
    )

    pp_rmse = evidence["primary_point_plane_rmse_mm"]["median"]
    add(
        "primary_point_plane_rmse_median_mm",
        pp_rmse,
        f"<={args.maximum_primary_point_plane_rmse_median_mm}",
        pp_rmse is not None and pp_rmse <= args.maximum_primary_point_plane_rmse_median_mm,
        "Error normal a superficie de relaciones primarias.",
    )

    pp_p90 = evidence["primary_point_plane_p90_abs_mm"]["median"]
    add(
        "primary_point_plane_p90_median_mm",
        pp_p90,
        f"<={args.maximum_primary_point_plane_p90_median_mm}",
        pp_p90 is not None and pp_p90 <= args.maximum_primary_point_plane_p90_median_mm,
        "Cola robusta del error normal a superficie.",
    )

    closure = evidence["closure"] or {}
    overlap = closure.get("overlap")
    add(
        "closure_overlap",
        overlap,
        f">={args.minimum_closure_overlap}",
        overlap is not None and overlap >= args.minimum_closure_overlap,
        "El último sector debe cerrar contra P00.",
    )

    cr = closure.get("trimmed_rmse_mm")
    add(
        "closure_rmse_mm",
        cr,
        f"<={args.maximum_closure_rmse_mm}",
        cr is not None and cr <= args.maximum_closure_rmse_mm,
        "Cierre euclídeo robusto.",
    )

    cpr = closure.get("point_plane_rmse_mm")
    add(
        "closure_point_plane_rmse_mm",
        cpr,
        f"<={args.maximum_closure_point_plane_rmse_mm}",
        cpr is not None and cpr <= args.maximum_closure_point_plane_rmse_mm,
        "Cierre normal a superficie.",
    )

    maxcorr = evidence["angle_correction_abs_deg"]["max"]
    add(
        "maximum_reference_angle_correction_deg",
        maxcorr,
        f"<={args.maximum_angle_correction_deg}",
        maxcorr is not None and maxcorr <= args.maximum_angle_correction_deg,
        (
            "La calibración no debe depender de correcciones angulares grandes "
            "aprendidas de la referencia."
        ),
    )

    sat = evidence["adaptive_saturation_count"]
    add(
        "adaptive_saturation_count",
        sat,
        f"<={args.maximum_adaptive_saturations}",
        sat <= args.maximum_adaptive_saturations,
        "Ninguna o muy pocas poses deben intentar escapar del modelo mecánico.",
    )

    mp90 = evidence["manhattan_p90_deg_reference_only"]
    if mp90 is not None:
        add(
            "reference_manhattan_p90_deg",
            mp90,
            f"<={args.maximum_manhattan_p90_deg}",
            mp90 <= args.maximum_manhattan_p90_deg,
            (
                "Chequeo exclusivo de la referencia usada durante desarrollo; "
                "NO forma parte del runtime general."
            ),
        )

    cp90 = evidence["reference_compactness_worst_p90_mm"]
    if cp90 is not None:
        add(
            "reference_compactness_worst_p90_mm",
            cp90,
            f"<={args.maximum_reference_compactness_p90_mm}",
            cp90 <= args.maximum_reference_compactness_p90_mm,
            (
                "Chequeo de la referencia; no se exigirá planaridad ni "
                "compactación de caras a objetos futuros."
            ),
        )

    mount = evidence.get("mounting_geometry", {}) or {}
    merr = mount.get("midpoint_axis_distance_error_mm")
    if merr is not None:
        add(
            "mount_reference_distance_error_mm",
            abs(float(merr)),
            f"<={args.maximum_mount_reference_error_mm}",
            abs(float(merr)) <= args.maximum_mount_reference_error_mm,
            "Prior físico con tolerancia amplia de montaje.",
        )

    return all(c["passed"] for c in checks), checks


def build_pose_records(manifest: dict, poses_rows: List[dict], axis, line_point, sign):
    """Genera poses con ángulos mecánicos y mantiene la evidencia del registro."""
    by_pose_csv = {int(r["pose_index"]): r for r in poses_rows}

    groups = manifest.get("pose_groups", [])
    if not groups:
        # fallback desde CSV de poses
        groups = []
        cumulative = 0
        for pose in range(POSES_PER_REVOLUTION):
            if pose > 0:
                cumulative += DEFAULT_STEP_SEQUENCE[pose - 1]
            row = by_pose_csv[pose]
            groups.append(
                {
                    "pose_index": pose,
                    "cumulative_steps": cumulative,
                    "physical_angle_deg": to_float(row["physical_angle_deg"]),
                    "nominal_angle_deg": 360.0 * pose / POSES_PER_REVOLUTION,
                }
            )

    records = []
    for g in groups:
        pose = int(g["pose_index"])
        physical = float(g["physical_angle_deg"])
        cumulative = int(g["cumulative_steps"])
        row = by_pose_csv.get(pose, {})
        learned_corr = to_float(row.get("angle_correction_deg"), 0.0)

        # RUNTIME GENERAL: corrección = 0.0.
        T = axis_transform(
            axis,
            line_point,
            math.radians(-sign * physical),
        )

        records.append(
            {
                "pose_index": pose,
                "cumulative_steps": cumulative,
                "physical_angle_deg": physical,
                "runtime_correction_deg": 0.0,
                "reference_v3_2_correction_deg_diagnostic_only": learned_corr,
                "transform_pose_to_P00_mechanical": T.astype(float).tolist(),
            }
        )
    return records


def main():
    """Construye la calibración candidata de la plataforma y su auditoría."""
    args = parser().parse_args()
    root = Path(args.root).expanduser().resolve()
    obj = args.source_object.strip()

    base, summary_path, poses_path, edges_path, obs_path = resolve_v32_files(
        root, obj, args.registration_source
    )
    manifest_path = resolve_manifest(root, obj)

    summary = load_json(summary_path)
    manifest = load_json(manifest_path)
    poses_rows = read_csv(poses_path)
    edges_rows = read_csv(edges_path)
    obs_rows = read_csv(obs_path)

    axis, center_raw, sign = extract_axis_center(summary)

    baseline = float(
        summary.get("mounting_geometry", {}).get(
            "baseline_mm",
            args.baseline_mm,
        )
    )
    midpoint = np.array([baseline / 2.0, 0.0, 0.0], dtype=np.float64)

    # Representación canónica de la MISMA línea de eje.
    axis_point = closest_point_on_line(midpoint, center_raw, axis)
    midpoint_distance = point_to_line_distance(midpoint, axis_point, axis)

    evidence = extract_metrics(summary, edges_rows, obs_rows)
    passed, checks = quality_gates(evidence, args)

    if int(manifest.get("steps_per_revolution", STEPS_PER_REVOLUTION)) != STEPS_PER_REVOLUTION:
        raise RuntimeError("El manifiesto no corresponde a 2055 pasos/vuelta.")
    if int(manifest.get("moves_per_revolution", POSES_PER_REVOLUTION)) != POSES_PER_REVOLUTION:
        raise RuntimeError("El manifiesto no corresponde a 25 poses.")

    pose_records = build_pose_records(
        manifest,
        poses_rows,
        axis,
        axis_point,
        sign,
    )

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    stereo_files = explicit_stereo_calibration_files(Path(args.stereo_calibration_dir))
    source_files = [
        summary_path,
        poses_path,
        edges_path,
        obs_path,
        manifest_path,
        *stereo_files,
    ]
    fingerprints = hash_existing(source_files)

    calibration = {
        "schema_version": 1,
        "calibration_type": "turntable_axis_line_mechanical_pose_model",
        "status": (
            "candidate_ready_for_independent_validation"
            if passed
            else "candidate_rejected_by_quality_gates"
        ),
        "coordinate_frame": (
            "rectified_left_camera_metric_frame_mm; "
            "axis_line_point is the point on the axis closest to stereo midpoint"
        ),
        "source_reference_object": obj,
        "source_registration": str(base),
        "generalization_contract": {
            "runtime_uses_object_shape": False,
            "runtime_uses_manhattan": False,
            "runtime_uses_planes": False,
            "runtime_uses_reference_v3_2_angle_corrections": False,
            "future_object_pose_source": "2055_step_mechanical_model_only",
            "allowed_future_objects": (
                "Cualquier objeto rígido que esté dentro del volumen útil, "
                "sea visible por ambas cámaras y produzca profundidad estéreo suficiente."
            ),
            "explicitly_not_hardcoded": [
                "cubo",
                "cilindro",
                "pirámide",
                "número de caras",
                "planaridad",
                "ortogonalidad Manhattan",
            ],
        },
        "mechanical_model": {
            "steps_per_revolution": STEPS_PER_REVOLUTION,
            "poses_per_revolution": POSES_PER_REVOLUTION,
            "step_sequence": manifest.get("step_sequence", DEFAULT_STEP_SEQUENCE),
            "degrees_per_motor_step": 360.0 / STEPS_PER_REVOLUTION,
            "direction_sign": int(sign),
            "closure_steps": STEPS_PER_REVOLUTION,
            "closure_angle_deg": 360.0,
        },
        "stereo_geometry": {
            "baseline_mm": baseline,
            "stereo_midpoint_xyz_mm": midpoint.tolist(),
            "mount_reference_distance_mm": float(args.mount_reference_distance_mm),
            "mount_reference_sigma_mm": float(args.mount_reference_sigma_mm),
        },
        "rotation_axis": {
            "unit_axis_xyz": axis.astype(float).tolist(),
            "canonical_line_point_xyz_mm": axis_point.astype(float).tolist(),
            "source_center_point_xyz_mm_diagnostic_only": center_raw.astype(float).tolist(),
            "stereo_midpoint_to_axis_distance_mm": midpoint_distance,
        },
        "poses": pose_records,
        "reference_quality_evidence": evidence,
        "candidate_quality_checks": checks,
        "candidate_quality_passed": bool(passed),
        "fingerprints_sha256": fingerprints,
        "invalidation_rules": [
            "Mover una cámara o modificar su orientación.",
            "Cambiar la calibración estéreo o la resolución geométrica usada.",
            "Mover físicamente la plataforma respecto al par estéreo.",
            "Cambiar motor, reducción, acople o transmisión de la plataforma.",
            "Obtener una nueva medición de pasos/vuelta incompatible con 2055.",
            "Cambiar el sentido físico de giro sin actualizar direction_sign.",
        ],
        "promotion_requirements": {
            "minimum_independent_campaigns": 3,
            "minimum_distinct_unseen_geometries": 2,
            "same_frozen_calibration_required": True,
            "per_object_parameter_tuning_allowed": False,
            "note": (
                "Las tres sesiones S01/S02/S03 de una campaña prueban "
                "repetibilidad intra-campaña, no reemplazan campañas independientes."
            ),
        },
        "important_note": (
            "Los ángulos runtime son exclusivamente los derivados de pasos. "
            "Las pequeñas correcciones V3.2 pertenecen al ajuste de la referencia "
            "y NO se transfieren a objetos futuros."
        ),
    }

    candidate_path = output / "calibracion_plataforma_candidata.json"
    candidate_tmp = candidate_path.with_suffix(candidate_path.suffix + ".tmp")
    candidate_tmp.write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    candidate_tmp.replace(candidate_path)
    candidate_sha256 = sha256_file(candidate_path)

    csv_path = output / "transformaciones_mecanicas_25_poses.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "pose_index",
            "cumulative_steps",
            "physical_angle_deg",
            "runtime_correction_deg",
            "reference_v3_2_correction_deg_diagnostic_only",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in pose_records:
            writer.writerow({k: r[k] for k in fields})

    audit_path = output / "auditoria_congelacion_calibracion.json"
    audit_path.write_text(
        json.dumps(
            {
                "passed": bool(passed),
                "checks": checks,
                "axis_xyz": axis.tolist(),
                "axis_line_point_xyz_mm": axis_point.tolist(),
                "stereo_midpoint_to_axis_distance_mm": midpoint_distance,
                "calibration_path": str(candidate_path),
                "calibration_sha256": candidate_sha256,
                "calibration_size_bytes": candidate_path.stat().st_size,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n========== PASO 09 — CALIBRACIÓN CANDIDATA ==========")
    print("Estado:", calibration["status"])
    print("Eje:", np.array2string(axis, precision=8))
    print("Punto canónico línea:", np.array2string(axis_point, precision=4))
    print(f"Distancia midpoint->eje: {midpoint_distance:.3f} mm")
    print("Runtime futuro: ángulos mecánicos, corrección por pose = 0°")
    print("Calibración:", candidate_path)
    print("SHA-256:", candidate_sha256)
    print("Auditoría:", audit_path)
    print("==========================================================\n")

    if not passed:
        raise SystemExit(2)
    return 0


def _entrada_con_diagnostico():
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    from utilidades_progreso import informar_inicio

    informar_inicio(__file__, "09", "Guardar calibración de plataforma")
    raise SystemExit(main())


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
