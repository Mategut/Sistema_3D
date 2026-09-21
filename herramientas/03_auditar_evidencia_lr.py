#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audita la procedencia LR y su propagación 02 -> 04 para una vista.

Uso ejemplo:
    python herramientas/03_auditar_evidencia_lr.py \
        --depth-dir reconstruccion/S01/02_estimacion_profundidad \
        --regional-dir reconstruccion/S01/04_validacion_disparidad \
        --stem OBJ01_PIRAMIDE1_S01_V001_A0000

No modifica ningún archivo.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def pct(n: int, d: int) -> str:
    return f"{100.0*n/max(d,1):.2f}%"


def main() -> int:
    p = argparse.ArgumentParser(description="Audita estados LR y procedencia del paso 04.")
    p.add_argument("--depth-dir", required=True)
    p.add_argument("--regional-dir", default="")
    p.add_argument("--stem", required=True)
    p.add_argument("--object-mask", default="")
    args = p.parse_args()

    depth_dir = Path(args.depth_dir)
    regional_dir = Path(args.regional_dir) if args.regional_dir else None
    stem = args.stem

    state_path = depth_dir / f"{stem}_lr_state.npy"
    if not state_path.is_file():
        raise FileNotFoundError(f"No existe {state_path}")
    state = np.load(state_path).astype(np.uint8)

    if args.object_mask:
        mask = cv2.imread(str(Path(args.object_mask)), cv2.IMREAD_GRAYSCALE)
    else:
        mask = cv2.imread(str(depth_dir / f"{stem}_object_mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.ones(state.shape, np.uint8) * 255
    if mask.shape != state.shape:
        raise ValueError(f"Máscara {mask.shape} != lr_state {state.shape}")
    domain = mask > 0
    total = int(np.count_nonzero(domain))

    names = {
        0: "fuera/invalido",
        1: "fuerte_bidireccional",
        2: "fuerte_directo_sin_inversa_global",
        3: "unilateral_debil",
        4: "contradiccion_lr_fuerte",
        5: "evidencia_directa_insuficiente",
    }
    print(f"Dominio auditado: {total} px")
    for code in range(6):
        n = int(np.count_nonzero(domain & (state == code)))
        print(f"LR {code} {names[code]:34s}: {n:8d} | {pct(n,total)}")

    trusted = cv2.imread(str(depth_dir / f"{stem}_trusted_mask.png"), cv2.IMREAD_GRAYSCALE)
    usable = cv2.imread(str(depth_dir / f"{stem}_usable_mask.png"), cv2.IMREAD_GRAYSCALE)
    if trusted is not None:
        n = int(np.count_nonzero(domain & (trusted > 0)))
        print(f"trusted                              : {n:8d} | {pct(n,total)}")
    if usable is not None:
        n = int(np.count_nonzero(domain & (usable > 0)))
        print(f"usable fuerte+debil                  : {n:8d} | {pct(n,total)}")

    if regional_dir:
        source_path = regional_dir / f"{stem}_depth_source_map.npy"
        if source_path.is_file():
            source = np.load(source_path).astype(np.uint8)
            if source.shape != state.shape:
                raise ValueError(f"source_map {source.shape} != lr_state {state.shape}")
            print("\nProcedencia Paso 04:")
            for code in sorted(np.unique(source[domain])):
                n = int(np.count_nonzero(domain & (source == code)))
                print(f"source {int(code):2d}: {n:8d} | {pct(n,total)}")
            contradicted_accepted = int(np.count_nonzero(domain & (state == 4) & (source > 0)))
            print(f"Contradicciones LR aceptadas por 04: {contradicted_accepted}")
            if contradicted_accepted:
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
