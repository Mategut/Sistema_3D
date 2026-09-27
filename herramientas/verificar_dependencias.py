#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Comprueba y prepara las dependencias del entorno de ejecución."""

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
    "skimage": ("scikit-image", "scikit-image"),
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
            "--constraint",
            str(Path(__file__).resolve().parent.parent / "requirements.txt"),
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
    if args.startup:
        cmd.append("--startup")
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
    parser.add_argument("--startup", action="store_true", help="Permite abrir la interfaz para reparar una calibración ausente o inválida.")
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

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    calibration_check = {"directory": str(root / "sistema/calibracion_estereo"), "ok": False}
    try:
        from procesamiento.utilidades_estereo import validate_stereo_calibration
        calibration = validate_stereo_calibration(root / "sistema/calibracion_estereo")
        calibration_check.update(ok=True, quality=calibration["quality"], baseline_mm=calibration["baseline_mm"])
    except (ImportError, ValueError) as exc:
        report["warnings" if args.startup else "errors"].append(str(exc))
    report["capabilities"]["stereo_calibration"] = calibration_check
    report["ready_for_reconstruction"] = calibration_check["ok"] and not report["errors"]
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
