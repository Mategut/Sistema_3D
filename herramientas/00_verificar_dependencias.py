#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verificación del entorno, dependencias y recursos de calibración del Sistema 3D.

Comprueba importación de paquetes y capacidades requeridas por el pipeline.
Puede instalar dependencias faltantes con el intérprete activo; --no-install
desactiva esa acción. Después de instalar se inicia una verificación nueva.
El informe distingue errores de advertencias, incluida la disponibilidad de
CUDA en ONNX Runtime. La instalación de paquetes no configura CUDA/cuDNN.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

# import_name: (nombre visible, paquete pip)
REQUIRED = {
    "numpy": ("NumPy", "numpy"),
    "cv2": ("OpenCV contrib", "opencv-contrib-python"),
    "scipy": ("SciPy", "scipy"),
    "open3d": ("Open3D", "open3d>=0.19.0"),
    "matplotlib": ("Matplotlib", "matplotlib"),
    "onnxruntime": ("ONNX Runtime GPU", "onnxruntime-gpu"),
    "serial": ("pyserial", "pyserial"),
    "PIL": ("Pillow", "Pillow"),
    "openpyxl": ("openpyxl", "openpyxl"),
    "psutil": ("psutil", "psutil"),
}

OPTIONAL = {
    "torch": "PyTorch",
}


def version_of(module) -> str:
    """Obtiene la versión declarada por un módulo o indica que se desconoce."""
    return str(getattr(module, "__version__", "desconocida"))


def numeric_version(version: str) -> tuple[int, ...]:
    """Extrae hasta tres grupos numéricos de la versión y completa con ceros."""
    values = [int(value) for value in re.findall(r"\d+", str(version))[:3]]
    return tuple(values + [0] * (3 - len(values)))


def unique(items: list[str]) -> list[str]:
    """Elimina elementos repetidos conservando su orden de aparición."""
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def install_packages(specs: list[str]) -> tuple[bool, list[str]]:
    """Instala paquetes con el mismo intérprete que ejecuta el verificador."""
    messages: list[str] = []
    ok = True
    for spec in unique(specs):
        print(f"[INSTALANDO] {spec}", flush=True)
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--upgrade",
            spec,
        ]
        try:
            completed = subprocess.run(cmd, check=False)
        except Exception as exc:
            ok = False
            messages.append(f"No se pudo ejecutar pip para {spec}: {type(exc).__name__}: {exc}")
            continue
        if completed.returncode != 0:
            ok = False
            messages.append(f"pip terminó con código {completed.returncode} al instalar {spec}.")
    return ok, messages


def run_fresh_verification(args: argparse.Namespace) -> int:
    """Vuelve a verificar en un proceso limpio después de instalar paquetes."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--no-install"]
    if args.quiet:
        cmd.append("--quiet")
    if args.json:
        cmd.extend(["--json", args.json])
    return subprocess.call(cmd)


def main() -> int:
    """Comprueba dependencias, capacidades y calibración y devuelve 0 o 2.

    Por defecto intenta instalar paquetes faltantes y repite la comprobación en
    un proceso nuevo. --no-install limita la acción a verificar. --json guarda
    el informe; --quiet reduce la salida cuando no se detectan errores.
    """
    parser = argparse.ArgumentParser(
        description="Verifica e instala las dependencias requeridas por el Sistema 3D."
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce la salida en consola.")
    parser.add_argument("--json", default="", help="Guarda el reporte de verificación en JSON.")
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Solo verifica; no instala paquetes faltantes.",
    )
    args = parser.parse_args()

    if not args.quiet:
        print("[PROGRESO] Verificando dependencias del Sistema 3D...", flush=True)

    report = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "required": {},
        "optional": {},
        "capabilities": {},
        "installed_during_run": [],
        "errors": [],
        "warnings": [],
    }

    if sys.version_info < (3, 10):
        report["errors"].append(
            "Se requiere Python 3.10 o superior. Python 3.11 es la versión recomendada."
        )
    elif sys.version_info >= (3, 13):
        report["warnings"].append(
            "Python 3.13 no es el entorno validado. Se recomienda Python 3.11."
        )

    loaded: dict[str, object] = {}
    install_specs: list[str] = []

    for import_name, (label, pip_spec) in REQUIRED.items():
        try:
            module = importlib.import_module(import_name)
            loaded[import_name] = module
            report["required"][import_name] = {
                "label": label,
                "pip": pip_spec,
                "ok": True,
                "version": version_of(module),
            }
        except Exception as exc:
            report["required"][import_name] = {
                "label": label,
                "pip": pip_spec,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            install_specs.append(pip_spec)

    # OpenCV debe incluir ximgproc/SLIC; opencv-python por sí solo no es suficiente.
    cv2 = loaded.get("cv2")
    if cv2 is not None:
        ximgproc_ok = bool(
            hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createSuperpixelSLIC")
        )
        report["capabilities"]["opencv_ximgproc_slic"] = ximgproc_ok
        if not ximgproc_ok:
            install_specs.append("opencv-contrib-python")
    else:
        report["capabilities"]["opencv_ximgproc_slic"] = False

    # Open3D 0.19 o posterior es requerido por las etapas de malla.
    o3d = loaded.get("open3d")
    if o3d is not None:
        o3d_version = version_of(o3d)
        version_ok = numeric_version(o3d_version) >= (0, 19, 0)
        report["capabilities"]["open3d_minimum_0_19"] = version_ok
        if not version_ok:
            install_specs.append("open3d>=0.19.0")
        mesh_ok = bool(
            hasattr(o3d, "geometry")
            and hasattr(o3d.geometry, "TriangleMesh")
            and hasattr(o3d.geometry, "PointCloud")
        )
        report["capabilities"]["open3d_geometry"] = mesh_ok
    else:
        report["capabilities"]["open3d_minimum_0_19"] = False
        report["capabilities"]["open3d_geometry"] = False

    # Si falta algo instalable, se instala y se repite la verificación desde cero.
    install_specs = unique(install_specs)
    if install_specs and not args.no_install:
        print("\nSe detectaron dependencias faltantes o incompatibles.", flush=True)
        ok, messages = install_packages(install_specs)
        report["installed_during_run"] = install_specs
        if not ok:
            report["errors"].extend(messages)
        else:
            print("\nInstalación terminada. Verificando nuevamente...\n", flush=True)
            return run_fresh_verification(args)

    # En modo solo-verificación, cualquier dependencia faltante pasa a error.
    if install_specs:
        for spec in install_specs:
            report["errors"].append(f"Dependencia pendiente: {spec}")

    # PyTorch no es obligatorio para ejecutar el pipeline.
    for import_name, label in OPTIONAL.items():
        try:
            module = importlib.import_module(import_name)
            report["optional"][import_name] = {
                "label": label,
                "ok": True,
                "version": version_of(module),
            }
        except Exception as exc:
            report["optional"][import_name] = {
                "label": label,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    # ONNX Runtime puede estar instalado y aun no tener CUDA disponible.
    ort = loaded.get("onnxruntime")
    if ort is not None:
        try:
            providers = list(ort.get_available_providers())
        except Exception as exc:
            providers = []
            report["warnings"].append(
                f"No se pudieron consultar los providers de ONNX Runtime: {exc}"
            )
        report["capabilities"]["onnxruntime_providers"] = providers
        if "CUDAExecutionProvider" not in providers:
            report["warnings"].append(
                "ONNX Runtime está instalado, pero CUDAExecutionProvider no está disponible. "
                "La instalación de CUDA/cuDNN debe corregirse manualmente si se desea ejecutar CREStereo en GPU."
            )

    # Verificación mínima de la calibración estéreo activa.
    root = Path(__file__).resolve().parent.parent
    calib_dir = root / "sistema" / "calibracion_estereo"
    stereo_yaml = calib_dir / "stereo_initial.yaml"
    maps_npz = calib_dir / "rectification_maps.npz"
    calib_report = calib_dir / "calibration_report.json"
    calibration_check = {"directory": str(calib_dir), "ok": True}

    for required_file in (stereo_yaml, maps_npz, calib_report):
        if not required_file.is_file():
            calibration_check["ok"] = False
            report["errors"].append(f"Falta calibración estéreo: {required_file.name}")

    np_module = loaded.get("numpy")
    if calibration_check["ok"] and np_module is not None:
        try:
            maps = np_module.load(maps_npz)
            needed = {"map1x", "map1y", "map2x", "map2y"}
            missing_maps = sorted(needed.difference(maps.files))
            if missing_maps:
                raise ValueError(f"faltan claves NPZ: {missing_maps}")
            for key in sorted(needed):
                arr = maps[key]
                if tuple(arr.shape) != (1080, 1920):
                    raise ValueError(f"{key} tiene forma {arr.shape}, esperada (1080, 1920)")
                if not np_module.isfinite(arr).all():
                    raise ValueError(f"{key} contiene valores no finitos")
            calibration_check["maps"] = "ok"
        except Exception as exc:
            calibration_check["ok"] = False
            report["errors"].append(f"rectification_maps.npz inválido: {exc}")

    if calib_report.is_file():
        try:
            cr = json.loads(calib_report.read_text(encoding="utf-8"))
            calibration_check["quality"] = cr.get("quality")
            calibration_check["baseline_mm"] = cr.get("baseline_mm")
            calibration_check["stereo_rms_px"] = cr.get("stereo_rms_px")
            calibration_check["epipolar_median_px"] = (cr.get("epipolar") or {}).get(
                "median_abs_dy_px"
            )
            if str(cr.get("quality", "")).lower() != "accepted":
                calibration_check["ok"] = False
                report["errors"].append(f"Calibración activa quality={cr.get('quality')}")
            if list(cr.get("image_size") or []) != [1920, 1080]:
                calibration_check["ok"] = False
                report["errors"].append("La calibración activa no corresponde a 1920x1080.")
        except Exception as exc:
            calibration_check["ok"] = False
            report["errors"].append(f"calibration_report.json inválido: {exc}")

    report["capabilities"]["stereo_calibration"] = calibration_check
    report["ok"] = not report["errors"]

    if args.json:
        out = Path(args.json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.quiet or not report["ok"]:
        print("=" * 68)
        print("SISTEMA 3D — VERIFICACIÓN DEL ENTORNO")
        print("Python:", report["python"])
        print("Versión:", report["python_version"])
        print("=" * 68)
        for rec in report["required"].values():
            mark = "OK" if rec["ok"] else "FALTA"
            detail = rec.get("version", rec.get("error", ""))
            print(f"[{mark:5}] {rec['label']}: {detail}")
        slic = report["capabilities"].get("opencv_ximgproc_slic")
        print("OpenCV ximgproc/SLIC:", "OK" if slic else "FALTA")
        providers = report["capabilities"].get("onnxruntime_providers")
        if providers is not None:
            print("ONNX providers:", providers)
        for warning in report["warnings"]:
            print("[WARN]", warning)
        for error in report["errors"]:
            print("[ERROR]", error)
        print("RESULTADO:", "PASS" if report["ok"] else "FAIL")
        print("=" * 68)

    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
