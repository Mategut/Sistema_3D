"""Promoción explícita de calibraciones respaldadas por revisión independiente.

Los JSON geométricos permanecen inmutables: su estado de activación y la
evidencia revisada se guardan aparte para conservar las huellas de campañas.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

from .utilidades_referencias import sha256_file, validate_campaign_references


def _read(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Se esperaba un objeto JSON: {path}")
    return data


def validate_promotion_evidence(candidate: Path, evidence_path: Path) -> dict:
    """Comprueba archivos y declaraciones del responsable de la revisión.

    Independencia física, geometrías no vistas y ausencia de ajuste por objeto
    requieren revisión humana; las métricas internas no prueban esas premisas.
    """
    candidate = Path(candidate).resolve()
    evidence_path = Path(evidence_path).resolve()
    calibration = _read(candidate)
    if calibration.get("candidate_quality_passed") is not True:
        raise ValueError("La candidata no supera los controles de calibración.")
    evidence = _read(evidence_path)
    if evidence.get("schema_version") != 1:
        raise ValueError("Versión de informe de validación independiente no admitida.")
    digest = sha256_file(candidate)
    if evidence.get("calibration_sha256") != digest:
        raise ValueError("El informe no corresponde a los bytes de esta candidata.")
    if not str(evidence.get("reviewer", "")).strip():
        raise ValueError("Falta el responsable de la revisión independiente.")
    if evidence.get("fixed_parameters_reviewed") is not True:
        raise ValueError("Debe revisarse y declarar la ausencia de ajuste por objeto.")
    protocol = evidence_path.parent / evidence.get("protocol_file", "")
    if not protocol.is_file() or sha256_file(protocol) != evidence.get("protocol_sha256"):
        raise ValueError("Falta el protocolo experimental o su huella no coincide.")
    requirements = calibration.get("promotion_requirements", {})
    minimum = max(3, int(requirements.get("minimum_independent_campaigns", 3)))
    minimum_geometries = max(2, int(requirements.get("minimum_distinct_unseen_geometries", 2)))
    campaigns = evidence.get("campaigns", [])
    if not isinstance(campaigns, list) or len(campaigns) < minimum:
        raise ValueError(f"Se necesitan al menos {minimum} campañas independientes.")
    roots, captures, geometries = set(), set(), set()
    verified = []
    for record in campaigns:
        workspace = (evidence_path.parent / record["workspace"]).resolve()
        if workspace in roots:
            raise ValueError("Una campaña no puede contarse más de una vez.")
        roots.add(workspace)
        if record.get("independent_acquisition") is not True:
            raise ValueError(f"No se acredita adquisición independiente: {workspace.name}")
        references = validate_campaign_references(workspace, "reconstruir")
        if sha256_file(references["platform"]) != digest:
            raise ValueError(f"La campaña usa otra calibración: {workspace.name}")
        capture_dir = workspace / "capturas"
        # Una copia de una campaña no se convierte en un experimento independiente.
        stereo_images = []
        for session in ("S01", "S02", "S03"):
            left = sorted((capture_dir / session / "izquierda").glob("*_L.png"))
            right = sorted((capture_dir / session / "derecha").glob("*_R.png"))
            if (len(left) != 25 or len(right) != 25
                    or {p.name[:-6] for p in left} != {p.name[:-6] for p in right}):
                raise ValueError(f"Faltan 25 pares verificables en {workspace.name}/{session}")
            stereo_images.extend(left + right)
        capture_digest = hashlib.sha256()
        # Ignorar nombres: renombrar/copiar la misma adquisición no aporta evidencia.
        hashes = sorted(sha256_file(p) for p in stereo_images)
        capture_digest.update("\n".join(hashes).encode("ascii"))
        capture_hash = capture_digest.hexdigest()
        if capture_hash in captures:
            raise ValueError("Se han presentado capturas duplicadas como campañas distintas.")
        captures.add(capture_hash)
        summary = workspace / "reconstruccion/multisesion/17_validacion_modelo/resumen_17_validacion_modelo.json"
        if not summary.is_file() or sha256_file(summary) != record.get("validation_sha256"):
            raise ValueError(f"Validación ausente o modificada: {workspace.name}")
        validation = _read(summary)
        if validation.get("quality") not in {"accepted", "warning"} or validation.get("reject_reasons"):
            raise ValueError(f"La campaña no supera la validación: {workspace.name}")
        if validation.get("quality") == "warning" and record.get("warnings_reviewed") is not True:
            raise ValueError(f"Falta revisar las advertencias: {workspace.name}")
        geometry = str(record.get("geometry", "")).strip().casefold()
        if not geometry:
            raise ValueError("Cada campaña debe identificar su geometría.")
        if record.get("unseen_geometry") is True:
            geometries.add(geometry)
        verified.append({
            "workspace": str(workspace), "geometry": geometry,
            "capture_sha256": capture_hash, "validation_sha256": sha256_file(summary),
            "quality": validation["quality"],
        })
    if len(geometries) < minimum_geometries:
        raise ValueError(f"Se requieren {minimum_geometries} geometrías no vistas distintas.")
    return {
        "schema_version": 1, "status": "active_independently_validated",
        "calibration_sha256": digest, "evidence_sha256": sha256_file(evidence_path),
        "reviewer": evidence["reviewer"], "campaigns": verified,
        "protocol_sha256": sha256_file(protocol),
        "protocol_source": str(protocol.resolve()),
        "scope": "Validación de plataforma; no certifica exactitud dimensional absoluta.",
    }


def promote_platform_calibration(local_output: Path, evidence_path: Path | None,
                                 system_dir: Path, records_dir: Path, *,
                                 evaluation_only: bool = False) -> Path:
    """Publica una candidata revisada, con copia de evidencia y recuperación ante fallo."""
    local_output, system_dir, records_dir = map(
        lambda p: Path(p).resolve(), (local_output, system_dir, records_dir)
    )
    candidate = local_output / "calibracion_plataforma_candidata.json"
    calibration = _read(candidate)
    if calibration.get("candidate_quality_passed") is not True:
        raise ValueError("La candidata no supera los controles de calibración.")
    if evaluation_only:
        activation = {
            "schema_version": 1, "status": "evaluation_only",
            "calibration_sha256": sha256_file(candidate),
            "scope": "Candidata operativa para adquirir evidencia; validación independiente pendiente.",
        }
    else:
        if evidence_path is None:
            raise ValueError("La activación validada requiere un informe independiente.")
        activation = validate_promotion_evidence(candidate, evidence_path)
    audit = _read(local_output / "auditoria_congelacion_calibracion.json")
    if audit.get("passed") is not True or audit.get("calibration_sha256") != activation["calibration_sha256"]:
        raise ValueError("La auditoría de congelación no acredita esta candidata.")
    stereo_contract = calibration["stereo_calibration_contract"]
    for key, name in (("stereo_initial_sha256", "stereo_initial.yaml"),
                      ("rectification_maps_sha256", "rectification_maps.npz")):
        if sha256_file(system_dir / "calibracion_estereo" / name) != stereo_contract.get(key):
            raise ValueError("La candidata no corresponde a la calibración estéreo activa.")
    stamp = str(time.time_ns())
    target = system_dir / "calibracion_plataforma"
    staged = system_dir / f"calibracion_plataforma.__tmp__{stamp}"
    backup = records_dir / f"calibracion_plataforma_reemplazada_{stamp}"
    records_dir.mkdir(parents=True, exist_ok=True)
    moved = False
    try:
        shutil.copytree(local_output, staged)
        shutil.copy2(candidate, staged / "calibracion_plataforma.json")
        if not evaluation_only:
            shutil.copy2(evidence_path, staged / "validacion_independiente.json")
            shutil.copy2(activation["protocol_source"], staged / "protocolo_validacion_independiente.txt")
            if sha256_file(staged / "validacion_independiente.json") != activation["evidence_sha256"]:
                raise ValueError("El informe cambió durante la promoción.")
            if sha256_file(staged / "protocolo_validacion_independiente.txt") != activation["protocol_sha256"]:
                raise ValueError("El protocolo cambió durante la promoción.")
        if sha256_file(staged / "calibracion_plataforma.json") != activation["calibration_sha256"]:
            raise ValueError("La candidata cambió durante la promoción.")
        activation["activated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (staged / "estado_activacion_plataforma.json").write_text(
            json.dumps(activation, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if target.exists():
            target.replace(backup)
            moved = True
        staged.replace(target)
    except Exception:
        if moved and not target.exists():
            backup.replace(target)
        raise
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return target / "calibracion_plataforma.json"
