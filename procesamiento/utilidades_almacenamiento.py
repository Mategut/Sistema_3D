#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Política de almacenamiento reducido para las campañas del Sistema 3D.

La reconstrucción completa genera numerosos artefactos intermedios necesarios
mientras el pipeline está trabajando (mapas por vista, nubes por pose,
diagnósticos, máscaras auxiliares, etc.). Una vez que una ejecución termina con
éxito, esos artefactos dejan de ser necesarios para el resultado científico
final y son la principal causa del crecimiento del tamaño de cada trabajo.

Este módulo NO cambia ningún algoritmo ni ninguna entrada de los pasos 01--18.
La compactación ocurre únicamente después de que el flujo haya terminado y
haya creado sus resultados finales. Se conservan:

- resúmenes JSON y tablas CSV de calidad;
- hojas de contacto y previsualizaciones globales;
- auditorías relevantes;
- estado del pipeline;
- ``resultado_final`` completo;
- capturas originales y ``documentacion`` completa.

Los archivos por vista/pose y las representaciones intermedias pesadas se
eliminan. Una campaña compactada puede volver a procesarse desde cero usando sus
capturas y referencias congeladas. No se pretende reutilizar checkpoints de una
campaña compactada porque sus dependencias intermedias ya no existen.
"""

from __future__ import annotations

import fnmatch
import json
import time
from pathlib import Path
from typing import Iterable

STORAGE_SCHEMA_VERSION = 1

# Patrones conservados por carpeta de paso. Solo se aplican a archivos que
# están directamente dentro de la carpeta del paso; las subcarpetas se recorren
# de forma independiente cuando corresponda.
KEEP_BY_STEP: dict[str, tuple[str, ...]] = {
    "01_mapa_angular": (
        "mapa_angular_multisesion.json",
        "mapa_angular_multisesion.csv",
    ),
    "02_estimacion_profundidad": (
        "resumen_02_estimacion_profundidad.json",
        "session_quality.csv",
        "epipolar_audit_v2_2_4.json",
        "background_support_model.json",
        "background_support_mask.png",
        "background_support_overlay.png",
        "background_depth_absolute_mm_vis.png",
    ),
    "03_mascara_objeto": (
        "resumen_03_mascara_objeto.json",
        "calidad_siluetas.csv",
        "contact_sheet_siluetas.png",
    ),
    "04_validacion_disparidad": (
        "resumen_04_validacion_disparidad.json",
        "contact_sheet_mascaras_profundidad.png",
    ),
    "05_consenso_multisesion": (
        "resumen_05_consenso_multisesion.json",
        "calidad_consenso_multisesion.csv",
        "contact_sheet_consenso_multisesion.png",
    ),
    "06_nubes_puntos": (
        "resumen_06_nubes_puntos.json",
        "calidad_nubes_consenso_multisesion.csv",
        "contact_sheet_nubes_consenso_multisesion.png",
    ),
    "07_validacion_geometrica": (
        "resumen_07_validacion_geometrica.json",
        "calidad_geometrica_nubes.csv",
        "contact_sheet_validacion_geometrica.png",
    ),
    "08_registro_referencia": (
        "resumen_08_registro_referencia.json",
        "auditoria_geometria_pose.json",
        "auditoria_observabilidad_angular_v3_2.json",
        "calidad_aristas_v3_2.csv",
        "observabilidad_angular_v3_2.csv",
        "poses_registradas_v3_2.csv",
        "preview_union_registrada_v3_2.png",
    ),
    "10_registro_calibrado": (
        "resumen_10_registro_calibrado.json",
        "calidad_registro_general_v1_2.csv",
        "modelo_poses_runtime_validado.json",
        "preview_registro_general_calibrado_v1_2.png",
    ),
    "11_fusion_multivista": (
        "resumen_11_fusion_multivista.json",
        "estadisticas_por_pose_11.csv",
        "preview_fusion_multivista.png",
        "preview_completado_estimado.png",
    ),
    "12_regularizacion_nube": (
        "resumen_12_regularizacion_nube.json",
        "preview_comparacion_regularizacion.png",
        "preview_nube_regularizada_general.png",
    ),
    "13_reconstruccion_superficie": (
        "resumen_13_reconstruccion_superficie.json",
        "comparacion_metodos_superficie_13.json",
    ),
    "14_limpieza_topologica": (
        "resumen_14_limpieza_topologica.json",
        "analisis_componentes_14.csv",
        "analisis_componentes_post_reparacion_14.csv",
        "analisis_huecos_14.csv",
        "caras_eliminadas_orientabilidad_14.csv",
        "preview_topologia_14.png",
    ),
    "15_pulido_final": (
        "resumen_15_pulido_final.json",
        "preview_regularizacion_malla.png",
    ),
    "16_validacion_intersecciones": (
        "resumen_16_validacion_intersecciones.json",
        "intersecciones_clasificadas_16.csv",
        "preview_intersecciones_16.png",
    ),
    "17_validacion_modelo": (
        "resumen_17_validacion_modelo.json",
        "preview_validacion_modelo_17.png",
    ),
    "18_exportacion_modelo": (
        "resumen_18_exportacion_modelo.json",
        "resumen_exportacion_18.json",
        "README_IMPORTAR_EN_BLENDER.txt",
    ),
}

# Carpetas internas que nunca se compactan mediante reglas de pasos.
ALWAYS_PRESERVE_DIR_NAMES = {
    "estado_pipeline",
}


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _tree_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += int(item.stat().st_size)
            except OSError:
                pass
    return total


def _format_mb(size_bytes: int) -> float:
    return round(float(size_bytes) / (1024.0 * 1024.0), 3)


def _compact_step_dir(step_dir: Path, patterns: tuple[str, ...]) -> dict:
    before = _tree_size(step_dir)
    removed_files = 0
    removed_bytes = 0
    kept_files = 0

    # Los pasos actuales escriben sus productos directamente en su carpeta.
    # Si en el futuro aparece una subcarpeta diagnóstica, se elimina completa
    # salvo que sea estado_pipeline (que no es una carpeta de paso).
    for child in list(step_dir.iterdir()):
        if child.is_file():
            if _matches(child.name, patterns):
                kept_files += 1
                continue
            try:
                size = int(child.stat().st_size)
                child.unlink()
                removed_files += 1
                removed_bytes += size
            except OSError:
                pass
        elif child.is_dir():
            # No hay subdirectorios de contrato en los pasos 01--18 actuales.
            # Se eliminan artefactos auxiliares recursivos.
            for f in child.rglob("*"):
                if f.is_file():
                    try:
                        removed_bytes += int(f.stat().st_size)
                        removed_files += 1
                    except OSError:
                        pass
            import shutil

            shutil.rmtree(child, ignore_errors=True)

    after = _tree_size(step_dir)
    return {
        "step_directory": str(step_dir),
        "before_bytes": before,
        "after_bytes": after,
        "removed_bytes": removed_bytes,
        "removed_files": removed_files,
        "kept_files": kept_files,
    }


def _annotate_compacted_summary(path: Path, report_path: Path) -> None:
    """Marca un resumen conservado para que no prometa artefactos ya retirados."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(data, dict):
        return
    data["storage_compacted"] = True
    data["storage_compaction_note"] = (
        "Los artefactos intermedios por vista/pose fueron retirados después de "
        "completar el pipeline. Las métricas de este resumen se conservan; algunas "
        "rutas históricas dentro de 'outputs' pueden apuntar a archivos ya eliminados."
    )
    data["storage_report"] = str(report_path)
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def compact_reconstruction(workspace: Path, *, reason: str = "pipeline_complete") -> dict:
    """Reduce ``reconstruccion`` conservando únicamente productos científicos finales.

    No toca ``capturas``, ``documentacion``, ``resultado_final`` ni las
    referencias congeladas de la campaña.
    """
    workspace = Path(workspace).expanduser().resolve()
    reconstruction = workspace / "reconstruccion"
    documentation = workspace / "documentacion"
    documentation.mkdir(parents=True, exist_ok=True)

    before = _tree_size(reconstruction)
    details: list[dict] = []

    if reconstruction.is_dir():
        for directory in sorted(
            (p for p in reconstruction.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            if directory.name in ALWAYS_PRESERVE_DIR_NAMES:
                continue
            patterns = KEEP_BY_STEP.get(directory.name)
            if patterns is None:
                continue
            details.append(_compact_step_dir(directory, patterns))

    after = _tree_size(reconstruction)
    report = {
        "schema_version": STORAGE_SCHEMA_VERSION,
        "mode": "reducido",
        "reason": str(reason),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "workspace": str(workspace),
        "reconstruction_before_bytes": int(before),
        "reconstruction_after_bytes": int(after),
        "reconstruction_removed_bytes": int(max(before - after, 0)),
        "reconstruction_before_mb": _format_mb(before),
        "reconstruction_after_mb": _format_mb(after),
        "reconstruction_removed_mb": _format_mb(max(before - after, 0)),
        "steps": sorted(details, key=lambda x: x["step_directory"]),
        "preserved_outside_reconstruction": [
            "capturas/",
            "documentacion/",
            "resultado_final/",
        ],
        "resume_policy": (
            "La campaña conserva capturas y referencias para reproducirse, pero los "
            "intermedios de checkpoints fueron retirados. Al reanudar después de una "
            "compactación, el pipeline recalcula los intermedios necesarios."
        ),
    }

    report_path = documentation / "resumen_almacenamiento.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for summary in reconstruction.rglob("resumen_*.json") if reconstruction.is_dir() else []:
        _annotate_compacted_summary(summary, report_path)

    # Marca el estado para impedir que --resume reutilice un resumen aislado
    # como si todavía existieran todas sus dependencias intermedias.
    state_path = reconstruction / "estado_pipeline" / "checkpoints.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["storage_compacted"] = True
            state["storage_compacted_at"] = report["created_at"]
            state["storage_report"] = str(report_path)
            state_path.write_text(
                json.dumps(state, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            # La compactación ya terminó; un fallo al anotar el checkpoint no
            # debe borrar resultados finales. Se deja constancia en el reporte.
            report["checkpoint_annotation_warning"] = True
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    return report


def print_compaction_report(report: dict) -> None:
    before = float(report.get("reconstruction_before_mb", 0.0) or 0.0)
    after = float(report.get("reconstruction_after_mb", 0.0) or 0.0)
    removed = float(report.get("reconstruction_removed_mb", 0.0) or 0.0)
    ratio = (100.0 * removed / before) if before > 0 else 0.0
    print("\n========== ALMACENAMIENTO REDUCIDO ==========", flush=True)
    print(f"Reconstrucción antes: {before:.1f} MB", flush=True)
    print(f"Reconstrucción después: {after:.1f} MB", flush=True)
    print(f"Liberado: {removed:.1f} MB ({ratio:.1f}%)", flush=True)
    print("Se conservaron resúmenes, CSV, hojas de contacto, previews y resultado_final.", flush=True)
    print("=============================================", flush=True)
