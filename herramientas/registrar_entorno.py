"""Registra el entorno actual sin instalar paquetes ni ejecutar reconstrucciones."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from procesamiento.version_sistema import PRODUCT_VERSION


def snapshot():
    model = ROOT / "modelos" / "crestereo_init_iter10_480x640.onnx"
    model_identity = None
    if model.is_file():
        with model.open("rb") as stream:
            model_identity = {"filename": model.name, "size_bytes": model.stat().st_size,
                              "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    packages = sorted(
        ({"name": d.metadata["Name"], "version": d.version}
         for d in metadata.distributions() if d.metadata["Name"]),
        key=lambda item: (item["name"].lower(), item["version"]),
    )
    gpu = {"status": "not_available", "devices": []}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        gpu = {"status": "queried", "devices": [
            {"name": row.rsplit(",", 1)[0].strip(),
             "driver_version": row.rsplit(",", 1)[1].strip()}
            for row in result.stdout.splitlines() if "," in row
        ]}
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "product_version": PRODUCT_VERSION,
        "scope": "Current environment snapshot; not a historical campaign environment or a dependency lockfile",
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation(),
                   "architecture": platform.machine()},
        "os": {"system": platform.system(), "version": platform.version(),
               "release": platform.release(),
               "note": "Python platform fields identify the kernel; Windows 11 may report release 10"},
        "nvidia": gpu,
        "cuda_runtime_version": None,
        "cudnn_version": None,
        "gpu_inference_verified": False,
        "gpu_note": "Driver query does not verify CUDA/cuDNN libraries or inference availability",
        "model": model_identity,
        "python_distributions": packages,
        "limitations": "Does not capture Conda-only packages, system libraries, build artifacts or upstream model provenance",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="JSON de destino; sin esta opción imprime el registro.")
    args = parser.parse_args()
    content = json.dumps(snapshot(), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"Entorno registrado: {args.output}")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
