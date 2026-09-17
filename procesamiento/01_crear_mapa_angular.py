#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Paso 01 — manifiesto de poses V7.2 / 2055 pasos."""

from __future__ import annotations
from utilidades_progreso import operacion
from utilidades_rendimiento import configurar as _configurar_recursos

# Los runtimes numéricos leen estos límites al importarse.
_configurar_recursos(__file__)


import argparse
import json
from pathlib import Path

from utilidades_multisesion import (
    STEPS_PER_REVOLUTION,
    MOVES_PER_REVOLUTION,
    STEP_SEQUENCE,
    build_angular_manifest,
    central_image_similarity,
    discover_sessions,
    save_manifest,
)


def parser():
    """Construye las opciones de línea de comandos de este paso."""
    p = argparse.ArgumentParser(
        description="Valida S01/S02/S03 y genera el mapa de 25 poses físicas."
    )
    p.add_argument("--root", required=True)
    p.add_argument("--object", default="cubo")
    p.add_argument("--expected-sessions", type=int, default=3)
    return p


def main():
    """Crea el manifiesto angular y los diagnósticos de correspondencia entre sesiones."""
    args = parser().parse_args()
    root = Path(args.root).expanduser().resolve()
    obj = args.object.strip().lower()

    sessions = discover_sessions(root, obj)
    manifest = build_angular_manifest(
        root,
        obj,
        sessions,
        expected_sessions=int(args.expected_sessions),
    )

    # Diagnóstico visual por pose: no modifica ángulos ni alineación.
    records = manifest["records"]
    diagnostics = {"same_pose_similarity": []}
    for pose in range(MOVES_PER_REVOLUTION):
        members = [r for r in records if r["pose_index"] == pose]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                diagnostics["same_pose_similarity"].append(
                    {
                        "pose_index": pose,
                        "a": {"session": members[i]["session"], "stem": members[i]["stem"]},
                        "b": {"session": members[j]["session"], "stem": members[j]["stem"]},
                        "similarity": central_image_similarity(
                            Path(members[i]["left_path"]),
                            Path(members[j]["left_path"]),
                        ),
                    }
                )
    manifest["diagnostics"] = diagnostics

    output = root / "reconstruccion" / "multisesion" / "01_mapa_angular"
    json_path, csv_path = save_manifest(output, manifest)
    json_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== PASO 01 — CAPTURA 2055 ==========")
    print(f"Objeto: {obj}")
    print(f"Sesiones: {sessions}")
    print(f"Pasos/vuelta: {STEPS_PER_REVOLUTION}")
    print(f"Transiciones: {MOVES_PER_REVOLUTION}")
    print(f"Secuencia: {list(STEP_SEQUENCE)}")
    print("Cada sesión: pose 0 -> pose 24 y CLOSE -> pose 0")
    print("\nPoses:")
    for group in manifest["pose_groups"]:
        print(
            f"  P{group['pose_index']:02d} | "
            f"nominal={group['nominal_angle_deg']:6.1f}° | "
            f"pasos={group['cumulative_steps']:4d} | "
            f"físico={group['physical_angle_deg']:9.6f}° | "
            f"obs={group['independent_observations']}"
        )
    print(
        f"\nMáx. cuantización angular: " f"{manifest['maximum_angle_quantization_error_deg']:.6f}°"
    )
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print("================================================\n")
    return 0


def _entrada_con_diagnostico():
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    from utilidades_progreso import informar_inicio

    informar_inicio(__file__, "01", "Crear mapa angular multisesión")
    raise SystemExit(main())


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
