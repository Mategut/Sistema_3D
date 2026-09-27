#!/usr/bin/env python
"""Instala explícitamente una candidata para evaluación o una calibración revisada."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from procesamiento.utilidades_calibracion import promote_platform_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["evaluar", "activar"])
    parser.add_argument("--candidate-dir", type=Path, required=True,
                        help="resultado_calibracion_plataforma de la campaña de calibración")
    parser.add_argument("--evidence", type=Path,
                        help="Informe de revisión independiente; obligatorio para activar")
    args = parser.parse_args()
    if args.mode == "activar" and args.evidence is None:
        parser.error("activar requiere --evidence")
    if args.mode == "evaluar" and args.evidence is not None:
        parser.error("evaluar no utiliza --evidence; para validar usa activar")
    try:
        path = promote_platform_calibration(
            args.candidate_dir, args.evidence, ROOT / "sistema", ROOT / "registros",
            evaluation_only=args.mode == "evaluar",
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"No se instaló la calibración: {exc}", file=sys.stderr)
        return 2
    print(f"Calibración instalada: {path}")
    print("Solo evaluación; validación independiente pendiente." if args.mode == "evaluar"
          else "Activada con informe independiente revisado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
