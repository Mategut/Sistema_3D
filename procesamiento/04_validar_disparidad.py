#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paso 04: validación de profundidad observada.

La máscara 2D del paso 03 define el dominio del objeto. Este paso conserva los
píxeles que tienen una disparidad observada válida, pertenecen al dominio
rectificado y producen una profundidad dentro del rango físico configurado.

No se crean profundidades nuevas ni se interpolan huecos. Se mantienen los
archivos consumidos por los pasos 05 y 06 y se genera una hoja de contacto con
todas las vistas para inspección visual rápida.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utilidades_progreso import ejecutar_con_diagnostico, informar_inicio, operacion
from utilidades_mascaras import build_contact_sheet


METHOD = "physical_observed_depth_validation_v2"
SCHEMA_VERSION = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paso 04: validar profundidad observada y rango físico."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--object", default="cubo")
    parser.add_argument("--session", default="S01")
    parser.add_argument("--depth-source", default="02_estimacion_profundidad")
    parser.add_argument("--silhouette-source", default="03_mascara_objeto")
    parser.add_argument("--output-name", default="04_validacion_disparidad")
    parser.add_argument("--only-angle", type=float, default=-1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--calibration-dir", default="")
    parser.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Elimina salidas antiguas del paso 04 antes de una corrida completa.",
    )
    parser.add_argument(
        "--minimum-depth-mm",
        type=float,
        default=None,
        help="Sobrescribe el límite cercano heredado del paso 02.",
    )
    parser.add_argument(
        "--maximum-depth-mm",
        type=float,
        default=None,
        help="Sobrescribe el límite lejano heredado del paso 02.",
    )
    parser.add_argument(
        "--minimum-valid-mask-ratio",
        type=float,
        default=0.20,
        help="Solo genera warning si queda muy poca profundidad respecto a la silueta.",
    )
    parser.add_argument(
        "--debug-overlays",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compatibilidad. Los overlays estándar y la hoja de contacto se generan siempre.",
    )

    # Compatibilidad con comandos/checkpoints de versiones anteriores. Ya no
    # afectan el resultado actual del Paso 04.
    parser.add_argument("--workers", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--fill-accepted-regions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--nominal-360-is-closure",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def finite_stats(values: np.ndarray) -> dict[str, float | int | None]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "mean": None,
            "p90": None,
            "maximum": None,
        }
    return {
        "count": int(x.size),
        "minimum": float(np.min(x)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "p90": float(np.percentile(x, 90.0)),
        "maximum": float(np.max(x)),
    }


def _source_quality(record: dict[str, Any]) -> str | None:
    return (
        record.get("session_quality")
        or record.get("quality")
        or record.get("local_quality")
    )


def _source_reasons(record: dict[str, Any]) -> list[Any]:
    value = (
        record.get("session_reasons")
        or record.get("reasons")
        or record.get("local_reasons")
        or []
    )
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _pick_disparity(depth_dir: Path, stem: str) -> Path | None:
    for name in (f"{stem}_disparity_lr.npy", f"{stem}_disparity.npy"):
        candidate = depth_dir / name
        if candidate.is_file():
            return candidate
    return None


def _select_views(summary: dict[str, Any], only_angle: float, limit: int) -> list[dict[str, Any]]:
    views = sorted(
        [dict(v) for v in summary.get("views", []) if v.get("stem")],
        key=lambda v: (float(v.get("angle_deg", 0.0)), str(v.get("stem", ""))),
    )
    if only_angle >= 0.0:
        views = [v for v in views if abs(float(v.get("angle_deg", 0.0)) - only_angle) <= 1e-3]
    if limit > 0:
        views = views[:limit]
    return views


def _overlay(image: np.ndarray, mask: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    active = mask > 0
    if np.any(active):
        tint = np.zeros_like(out)
        tint[:, :, 1] = mask
        out = cv2.addWeighted(out, 1.0, tint, 0.35, 0.0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _clean_directory(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _depth_range(depth_summary: dict[str, Any], args: argparse.Namespace) -> tuple[float, float]:
    params = depth_summary.get("parameters") or {}
    inherited_min = float(params.get("minimum_depth_mm", 150.0) or 150.0)
    inherited_max = float(params.get("maximum_depth_mm", 1200.0) or 1200.0)
    minimum = inherited_min if args.minimum_depth_mm is None else float(args.minimum_depth_mm)
    maximum = inherited_max if args.maximum_depth_mm is None else float(args.maximum_depth_mm)
    minimum = max(minimum, 1.0)
    maximum = max(maximum, minimum + 1.0)
    return minimum, maximum


@operacion("Validar profundidad observada")
def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    session = str(args.session).upper()
    object_name = str(args.object)
    base = root / "reconstruccion" / session
    depth_dir = base / args.depth_source
    silhouette_dir = base / args.silhouette_source
    output_dir = base / args.output_name

    depth_summary_path = depth_dir / "resumen_02_estimacion_profundidad.json"
    silhouette_summary_path = silhouette_dir / "resumen_03_mascara_objeto.json"
    if not depth_summary_path.is_file():
        raise FileNotFoundError(f"Falta resumen del paso 02: {depth_summary_path}")
    if not silhouette_summary_path.is_file():
        raise FileNotFoundError(f"Falta resumen del paso 03: {silhouette_summary_path}")

    depth_summary = load_json(depth_summary_path)
    silhouette_summary = load_json(silhouette_summary_path)

    fx = float(depth_summary.get("fx_rectified_px") or 0.0)
    baseline_mm = float(depth_summary.get("baseline_mm") or 0.0)
    if not np.isfinite(fx) or fx <= 0.0 or not np.isfinite(baseline_mm) or baseline_mm <= 0.0:
        raise ValueError("El resumen del paso 02 no contiene fx/baseline válidos.")

    minimum_depth_mm, maximum_depth_mm = _depth_range(depth_summary, args)
    effective_min_disparity = fx * baseline_mm / maximum_depth_mm
    effective_max_disparity = fx * baseline_mm / minimum_depth_mm

    views = _select_views(silhouette_summary, args.only_angle, args.limit)
    if not views:
        raise RuntimeError("No hay vistas del paso 03 para procesar.")

    if args.clean_output and args.only_angle < 0.0:
        _clean_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_views = {
        str(v.get("stem")): v for v in depth_summary.get("views", []) if v.get("stem")
    }
    records: list[dict[str, Any]] = []

    for index, view in enumerate(views, 1):
        stem = str(view["stem"])
        angle = float(view.get("angle_deg", 0.0))
        silhouette_path = silhouette_dir / f"{stem}_silhouette_mask.png"
        rect_valid_path = depth_dir / f"{stem}_rect_valid_mask.png"
        disparity_path = _pick_disparity(depth_dir, stem)

        silhouette = cv2.imread(str(silhouette_path), cv2.IMREAD_GRAYSCALE)
        rect_valid = cv2.imread(str(rect_valid_path), cv2.IMREAD_GRAYSCALE)
        if silhouette is None:
            raise FileNotFoundError(f"Falta silueta: {silhouette_path}")
        if rect_valid is None:
            raise FileNotFoundError(f"Falta máscara rectificada: {rect_valid_path}")
        if disparity_path is None:
            raise FileNotFoundError(f"Falta disparidad para {stem} en {depth_dir}")

        disparity = np.load(str(disparity_path)).astype(np.float32)
        if disparity.shape != silhouette.shape or rect_valid.shape != silhouette.shape:
            raise ValueError(
                f"Dimensiones incompatibles en {stem}: silhouette={silhouette.shape}, "
                f"rect_valid={rect_valid.shape}, disparity={disparity.shape}."
            )

        observed = (
            (silhouette > 0)
            & (rect_valid > 0)
            & np.isfinite(disparity)
            & (disparity >= effective_min_disparity)
            & (disparity <= effective_max_disparity)
        )

        cloud_mask = observed.astype(np.uint8) * 255
        depth = np.full(disparity.shape, np.nan, np.float32)
        depth[observed] = (fx * baseline_mm / disparity[observed]).astype(np.float32)

        # Compatibilidad con 05/06. Código 1 = medición estéreo observada. No
        # no existe profundidad modelada ni interpolada en este paso.
        source_map = np.zeros(disparity.shape, np.uint8)
        source_map[observed] = 1

        # El residual regional deja de existir. Se conserva el archivo con NaN
        # para que 05/06 sepan explícitamente que esta evidencia no está disponible.
        regional_residual = np.full(disparity.shape, np.nan, np.float32)

        silhouette_pixels = int(np.count_nonzero(silhouette))
        valid_pixels = int(np.count_nonzero(observed))
        valid_ratio = valid_pixels / max(silhouette_pixels, 1)
        missing_pixels = max(silhouette_pixels - valid_pixels, 0)
        depth_values = depth[observed]

        reasons: list[str] = []
        if valid_pixels == 0:
            quality = "rejected"
            reasons.append("No quedó ninguna profundidad observada físicamente válida.")
        elif valid_ratio < float(args.minimum_valid_mask_ratio):
            quality = "warning"
            reasons.append(f"Cobertura de profundidad baja: {valid_ratio:.2%}.")
        else:
            quality = "accepted"

        cloud_mask_path = output_dir / f"{stem}_cloud_mask.png"
        depth_valid_path = output_dir / f"{stem}_depth_valid_mask.png"
        depth_path = output_dir / f"{stem}_depth_regularized_mm.npy"
        source_map_path = output_dir / f"{stem}_depth_source_map.npy"
        residual_path = output_dir / f"{stem}_regional_residual.npy"

        cv2.imwrite(str(cloud_mask_path), cloud_mask)
        cv2.imwrite(str(depth_valid_path), cloud_mask)
        np.save(str(depth_path), depth)
        np.save(str(source_map_path), source_map)
        np.save(str(residual_path), regional_residual)

        # Vista estándar de control visual. Se genera siempre porque es liviana y
        # permite construir la hoja de contacto de las 25 vistas.
        overlay_path = None
        image = cv2.imread(str(depth_dir / f"{stem}_rect_L.png"), cv2.IMREAD_COLOR)
        if image is not None and image.shape[:2] == cloud_mask.shape:
            overlay_path = output_dir / f"{stem}_cloud_mask_overlay.png"
            cv2.imwrite(
                str(overlay_path),
                _overlay(image, cloud_mask, f"{angle:05.1f} deg | profundidad valida"),
            )

        source_record = source_views.get(stem, {})
        record = {
            "stem": stem,
            "angle_deg": angle,
            "quality": quality,
            "reasons": reasons,
            "source_depth_quality": _source_quality(source_record),
            "source_depth_reasons": _source_reasons(source_record),
            "silhouette_pixels": silhouette_pixels,
            "valid_depth_pixels": valid_pixels,
            "valid_ratio_of_silhouette": float(valid_ratio),
            "physically_valid_observed_pixels": valid_pixels,
            "remaining_missing_pixels": missing_pixels,
            "model_recovered_pixels": 0,
            "interpolated_pixels": 0,
            "depth_median_mm": float(np.median(depth_values)) if depth_values.size else None,
            "depth": finite_stats(depth_values),
            "outputs": {
                "depth_valid_mask": str(depth_valid_path),
                "cloud_mask": str(cloud_mask_path),
                "regularized_depth": str(depth_path),
                "depth_source_map": str(source_map_path),
                "regional_residual": str(residual_path),
                "overlay": str(overlay_path) if overlay_path is not None else None,
            },
        }
        records.append(record)
        print(
            f"[{index:02d}/{len(views):02d}] {stem} | {angle:7.3f}° | "
            f"válido={valid_ratio:6.2%} | {quality}",
            flush=True,
        )

    # Resumen visual de todas las vistas, equivalente a la hoja que se
    # generaba anteriormente en este paso.
    preview_paths = [
        Path(r["outputs"]["overlay"])
        for r in records
        if r.get("outputs", {}).get("overlay")
    ]
    contact_sheet_path = output_dir / "contact_sheet_mascaras_profundidad.png"
    build_contact_sheet(preview_paths, contact_sheet_path)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "object": object_name,
        "session": session,
        "depth_source": str(depth_dir),
        "depth_source_summary": str(depth_summary_path),
        "silhouette_source": str(silhouette_dir),
        "silhouette_source_summary": str(silhouette_summary_path),
        "calibration_dir": str(args.calibration_dir) if args.calibration_dir else None,
        "fx_rectified_px": fx,
        "baseline_mm": baseline_mm,
        "physical_depth_range_mm": [minimum_depth_mm, maximum_depth_mm],
        "effective_disparity_range_px": [effective_min_disparity, effective_max_disparity],
        "output_dir": str(output_dir),
        "parameters": {
            "minimum_valid_mask_ratio": float(args.minimum_valid_mask_ratio),
            "standard_overlays": True,
            "contact_sheet": True,
            "regional_modeling": False,
            "gap_recovery": False,
            "depth_interpolation": False,
        },
        "views_processed": len(records),
        "views_accepted": sum(r["quality"] == "accepted" for r in records),
        "views_warning": sum(r["quality"] == "warning" for r in records),
        "views_rejected": sum(r["quality"] == "rejected" for r in records),
        "views_closure_only": 0,
        "accepted_angles": [r["angle_deg"] for r in records if r["quality"] == "accepted"],
        "warning_angles": [r["angle_deg"] for r in records if r["quality"] == "warning"],
        "rejected_angles": [r["angle_deg"] for r in records if r["quality"] == "rejected"],
        "closure_only_angles": [],
        "views": records,
        "contact_sheet": str(contact_sheet_path) if contact_sheet_path.is_file() else None,
        "important_note": (
            "La máscara de profundidad se obtiene de la intersección entre la silueta, el dominio "
            "rectificado válido y la disparidad observada dentro del rango físico. No se crean "
            "profundidades nuevas ni se interpolan huecos; el consenso multisesión y la validación "
            "geométrica permanecen en los pasos posteriores."
        ),
    }
    save_json(output_dir / "resumen_04_validacion_disparidad.json", summary)

    ratios = np.asarray([r["valid_ratio_of_silhouette"] for r in records], dtype=np.float64)
    print(
        f"Paso 04 terminado | vistas={len(records)} | "
        f"cobertura media={float(np.mean(ratios)):.2%} | "
        f"accepted={summary['views_accepted']} warning={summary['views_warning']} "
        f"rejected={summary['views_rejected']}",
        flush=True,
    )
    return 0


def main() -> int:
    args = build_parser().parse_args()
    informar_inicio(__file__, "04", "Validar profundidad")
    return run(args)


if __name__ == "__main__":
    ejecutar_con_diagnostico(main, __file__)
