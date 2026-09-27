"""Corrige la autorreferencia histórica conservando respaldo de los informes.

No recalcula métricas ni sustituye las huellas de archivos que hayan cambiado.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=ROOT / "trabajos")
    args = parser.parse_args()
    (ROOT / "registros").mkdir(exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix="reparacion_manifiestos_", dir=ROOT / "registros"))
    log = []
    for report in sorted(args.jobs.glob("*/resultado_final/resumen_exportacion_18.json")):
        data = json.loads(report.read_text(encoding="utf-8"))
        if report.name not in data.get("files", {}):
            continue
        mismatches = []
        for name, info in data["files"].items():
            if name == report.name:
                continue
            path = report.parent / name
            if not path.is_file() or digest(path) != info.get("sha256") or path.stat().st_size != info.get("size_bytes"):
                mismatches.append(name)
        if mismatches:
            log.append({"campaign": report.parent.parent.name, "status": "not_modified_payload_mismatch", "files": mismatches})
            continue
        campaign = report.parent.parent
        target_backup = backup / campaign.name
        target_backup.mkdir()
        shutil.copy2(report, target_backup / report.name)
        data["files"].pop(report.name)
        data["manifest_policy"] = "payload_only_excludes_export_report"
        temp = report.with_suffix(".json.tmp")
        temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(report)
        phase = campaign / "reconstruccion/multisesion/18_exportacion_modelo/resumen_18_exportacion_modelo.json"
        if phase.is_file():
            original = json.loads(phase.read_text(encoding="utf-8"))
            shutil.copy2(phase, target_backup / phase.name)
            original["files"] = data["files"]
            original["manifest_policy"] = data["manifest_policy"]
            original["export_report_sha256"] = digest(report)
            temp = phase.with_suffix(".json.tmp")
            temp.write_text(json.dumps(original, indent=2, ensure_ascii=False), encoding="utf-8")
            temp.replace(phase)
        log.append({"campaign": campaign.name, "status": "repaired", "export_report_sha256": digest(report), "checkpoint": "No se alteró; al reanudar se verificará su vigencia."})
    (backup / "reparacion.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"backup": str(backup), "campaigns": log}, indent=2, ensure_ascii=False))
    return int(any(item["status"] != "repaired" for item in log))


if __name__ == "__main__":
    raise SystemExit(main())
