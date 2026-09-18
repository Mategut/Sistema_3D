#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Referencias inmutables por campaña.

Cada trabajo conserva dentro de ``documentacion/`` una copia de los recursos
científicos con los que fue capturado y procesado. La estructura canónica es::

    trabajo/
    └── documentacion/
        ├── referencias/
        │   ├── fondo_vacio/
        │   ├── calibracion_estereo/
        │   ├── calibracion_plataforma/   # solo reconstrucción
        │   ├── modelo/
        │   └── firmware/
        ├── configuracion_captura.json
        └── referencias_campana.json

Así, ``capturas/`` contiene solo datos adquiridos y ``reconstruccion/`` solo
resultados calculados. Actualizar los recursos globales del sistema no cambia
una campaña histórica ni invalida su reproducibilidad.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

DOCUMENTATION_DIRNAME = "documentacion"
REFERENCES_DIRNAME = "referencias"
REFERENCES_METADATA_FILENAME = "referencias_campana.json"
REFERENCES_SCHEMA_VERSION = 2
LEGACY_REFERENCES_SCHEMA_VERSION = 1

MODEL_DIRNAME = "modelo"
STEREO_DIRNAME = "calibracion_estereo"
BACKGROUND_DIRNAME = "fondo_vacio"
PLATFORM_DIRNAME = "calibracion_plataforma"
FIRMWARE_DIRNAME = "firmware"
CAPTURE_CONFIG_FILENAME = "configuracion_captura.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcula SHA-256 leyendo el archivo por bloques."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _json_digest(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_inventory(root: Path) -> list[dict]:
    """Inventario por contenido, independiente de fechas de modificación."""
    root = Path(root)
    if root.is_file():
        return [
            {
                "path": root.name,
                "size": int(root.stat().st_size),
                "sha256": sha256_file(root),
            }
        ]
    if not root.is_dir():
        raise FileNotFoundError(root)

    records: list[dict] = []
    for file in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: str(p).lower()):
        records.append(
            {
                "path": str(file.relative_to(root)).replace("\\", "/"),
                "size": int(file.stat().st_size),
                "sha256": sha256_file(file),
            }
        )
    return records


def content_digest(path: Path) -> dict:
    """Huella de contenido de un archivo o directorio completo."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return {"exists": False, "digest": None, "files": 0}
    inventory = _file_inventory(path)
    return {
        "exists": True,
        "digest": _json_digest(inventory),
        "files": len(inventory),
    }


def _require_file(path: Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def _require_dir(path: Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def _copy_directory(source: Path, target: Path) -> None:
    source = _require_dir(source, f"Directorio fuente {source}")
    shutil.copytree(source, target, copy_function=shutil.copy2)


def _copy_file(source: Path, target: Path) -> None:
    source = _require_file(source, f"Archivo fuente {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".__tmp__{os.getpid()}_{time.time_ns()}")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def documentation_root(workspace: Path) -> Path:
    """Carpeta de documentación persistente de un trabajo."""
    return Path(workspace).expanduser().resolve() / DOCUMENTATION_DIRNAME


def reference_root(workspace: Path) -> Path:
    """Directorio canónico que contiene únicamente los recursos congelados."""
    return documentation_root(workspace) / REFERENCES_DIRNAME


def reference_metadata_path(workspace: Path) -> Path:
    """Metadatos del snapshot; viven junto a la documentación de la campaña."""
    return documentation_root(workspace) / REFERENCES_METADATA_FILENAME


def capture_config_path(workspace: Path) -> Path:
    """Configuración de captura congelada para la campaña."""
    return documentation_root(workspace) / CAPTURE_CONFIG_FILENAME


def _legacy_reference_root(workspace: Path) -> Path:
    """Ubicación usada por la primera implementación del congelamiento."""
    return Path(workspace).expanduser().resolve() / REFERENCES_DIRNAME


def _legacy_reference_metadata_path(workspace: Path) -> Path:
    return _legacy_reference_root(workspace) / REFERENCES_METADATA_FILENAME


def _legacy_capture_config_path(workspace: Path) -> Path:
    return _legacy_reference_root(workspace) / CAPTURE_CONFIG_FILENAME


def _snapshot_exists_in_new_layout(workspace: Path) -> bool:
    return reference_metadata_path(workspace).is_file()


def _snapshot_exists_in_legacy_layout(workspace: Path) -> bool:
    return _legacy_reference_metadata_path(workspace).is_file()


def has_frozen_references(workspace: Path) -> bool:
    """Indica si existe un snapshot nuevo o uno de la disposición anterior."""
    return _snapshot_exists_in_new_layout(workspace) or _snapshot_exists_in_legacy_layout(workspace)


def _resource_entry(source: Path, frozen_relative: str) -> dict:
    source = Path(source).expanduser().resolve()
    fp = content_digest(source)
    if not fp["exists"]:
        raise FileNotFoundError(source)
    return {
        "source_path_at_freeze": str(source),
        "source_content_digest": fp["digest"],
        "source_file_count": fp["files"],
        "frozen_path": frozen_relative.replace("\\", "/"),
    }


def _capture_config_record(path: Path) -> dict:
    path = _require_file(path, "Configuración de captura congelada")
    return {
        "path": f"{DOCUMENTATION_DIRNAME}/{CAPTURE_CONFIG_FILENAME}",
        "size": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _publish_snapshot_atomically(
    *,
    staged_references: Path,
    staged_config: Path,
    staged_metadata: Path,
    target_references: Path,
    target_config: Path,
    target_metadata: Path,
) -> None:
    """Publica las tres piezas del snapshot con rollback si algo falla."""
    token = f".__backup__{os.getpid()}_{time.time_ns()}"
    items = [
        (Path(target_references), Path(staged_references)),
        (Path(target_config), Path(staged_config)),
        (Path(target_metadata), Path(staged_metadata)),
    ]
    backups: dict[Path, Path] = {}
    published: list[Path] = []

    try:
        for target, _staged in items:
            if target.exists():
                backup = target.with_name(target.name + token)
                if backup.exists():
                    if backup.is_dir():
                        shutil.rmtree(backup)
                    else:
                        backup.unlink()
                target.replace(backup)
                backups[target] = backup

        for target, staged in items:
            target.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(target)
            published.append(target)

        for backup in backups.values():
            if backup.is_dir():
                shutil.rmtree(backup, ignore_errors=True)
            else:
                backup.unlink(missing_ok=True)
    except Exception:
        for target in reversed(published):
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
        for target, backup in backups.items():
            if backup.exists() and not target.exists():
                backup.replace(target)
        raise


def _migrate_legacy_layout_if_needed(workspace: Path) -> bool:
    """Migra ``trabajo/referencias`` a ``trabajo/documentacion`` una sola vez.

    La migración no cambia recursos científicos ni sus hashes; únicamente
    reorganiza el snapshot. Se ejecuta al abrir/validar una campaña creada por
    la primera versión de esta funcionalidad.
    """
    workspace = Path(workspace).expanduser().resolve()
    if _snapshot_exists_in_new_layout(workspace):
        return False
    old_meta_path = _legacy_reference_metadata_path(workspace)
    if not old_meta_path.is_file():
        return False

    old_root = _legacy_reference_root(workspace)
    old_config = _legacy_capture_config_path(workspace)
    data = json.loads(old_meta_path.read_text(encoding="utf-8"))
    schema = int(data.get("schema_version", LEGACY_REFERENCES_SCHEMA_VERSION))
    if schema not in {LEGACY_REFERENCES_SCHEMA_VERSION, REFERENCES_SCHEMA_VERSION}:
        raise ValueError(f"Versión heredada de referencias no compatible: {schema}")

    docs = documentation_root(workspace)
    docs.mkdir(parents=True, exist_ok=True)
    target = reference_root(workspace)
    target_config = capture_config_path(workspace)
    target_meta = reference_metadata_path(workspace)

    token = f".__migrate__{os.getpid()}_{time.time_ns()}"
    staged = docs / f"{REFERENCES_DIRNAME}{token}"
    staged_config = docs / f"{CAPTURE_CONFIG_FILENAME}{token}"
    staged_meta = docs / f"{REFERENCES_METADATA_FILENAME}{token}"

    try:
        staged.mkdir(parents=True, exist_ok=False)
        # Copiar todo el snapshot científico excepto los dos documentos que en
        # la nueva estructura viven directamente dentro de documentacion/.
        for child in old_root.iterdir():
            if child.name in {REFERENCES_METADATA_FILENAME, CAPTURE_CONFIG_FILENAME}:
                continue
            destination = staged / child.name
            if child.is_dir():
                shutil.copytree(child, destination, copy_function=shutil.copy2)
            elif child.is_file():
                shutil.copy2(child, destination)

        if old_config.is_file():
            shutil.copy2(old_config, staged_config)
        else:
            # Snapshot muy temprano: preservar la campaña con configuración vacía
            # explícita en lugar de inventar parámetros.
            staged_config.write_text("{}\n", encoding="utf-8")

        inventory = _file_inventory(staged)
        migrated = dict(data)
        migrated.update(
            {
                "schema_version": REFERENCES_SCHEMA_VERSION,
                "layout": "documentacion_v2",
                "references_directory": f"{DOCUMENTATION_DIRNAME}/{REFERENCES_DIRNAME}",
                "capture_config": f"{DOCUMENTATION_DIRNAME}/{CAPTURE_CONFIG_FILENAME}",
                "capture_config_record": _capture_config_record(staged_config),
                "frozen_files": inventory,
                "frozen_content_digest": _json_digest(inventory),
                "migrated_from_layout": "workspace_root_referencias_v1",
                "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        staged_meta.write_text(
            json.dumps(migrated, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        _publish_snapshot_atomically(
            staged_references=staged,
            staged_config=staged_config,
            staged_metadata=staged_meta,
            target_references=target,
            target_config=target_config,
            target_metadata=target_meta,
        )

        # Solo después de publicar correctamente se elimina la disposición vieja.
        shutil.rmtree(old_root, ignore_errors=True)
        return True
    except Exception:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
        staged_config.unlink(missing_ok=True)
        staged_meta.unlink(missing_ok=True)
        raise


def freeze_campaign_references(
    workspace: Path,
    mode: str,
    *,
    model_path: Path,
    stereo_dir: Path,
    background_dir: Path,
    platform_dir: Optional[Path] = None,
    firmware_path: Optional[Path] = None,
    capture_config: Optional[dict] = None,
    product_version: Optional[str] = None,
    overwrite: bool = False,
) -> dict:
    """Copia los recursos activos a ``documentacion/`` y guarda sus metadatos.

    La publicación se hace al final y con rollback. Para reconstrucción se
    congela también la calibración de plataforma.
    """
    workspace = Path(workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    mode = str(mode).strip().lower()
    if mode not in {"reconstruir", "calibrar-plataforma"}:
        raise ValueError(f"Modo de campaña inválido: {mode}")

    # Si esta campaña fue creada con la disposición anterior, primero se
    # normaliza; así overwrite conserva una sola ruta canónica.
    _migrate_legacy_layout_if_needed(workspace)

    model_path = _require_file(model_path, "Modelo ONNX")
    stereo_dir = _require_dir(stereo_dir, "Calibración estéreo")
    background_dir = _require_dir(background_dir, "Fondo vacío")

    _require_file(stereo_dir / "stereo_initial.yaml", "stereo_initial.yaml")
    _require_file(stereo_dir / "rectification_maps.npz", "rectification_maps.npz")
    _require_file(background_dir / "background_left.png", "background_left.png")
    _require_file(background_dir / "background_right.png", "background_right.png")

    if mode == "reconstruir":
        if platform_dir is None:
            raise ValueError("La reconstrucción requiere calibración de plataforma.")
        platform_dir = _require_dir(platform_dir, "Calibración de plataforma")
        _require_file(
            platform_dir / "calibracion_plataforma.json",
            "calibracion_plataforma.json",
        )

    docs = documentation_root(workspace)
    docs.mkdir(parents=True, exist_ok=True)
    target = reference_root(workspace)
    target_config = capture_config_path(workspace)
    target_metadata = reference_metadata_path(workspace)

    if (target.exists() or target_metadata.exists()) and not overwrite:
        raise FileExistsError(
            f"El trabajo ya tiene referencias congeladas: {target}. "
            "Usa overwrite=True solo antes de reiniciar completamente la captura."
        )

    token = f".__tmp__{os.getpid()}_{time.time_ns()}"
    staged = docs / f"{REFERENCES_DIRNAME}{token}"
    staged_config = docs / f"{CAPTURE_CONFIG_FILENAME}{token}"
    staged_metadata = docs / f"{REFERENCES_METADATA_FILENAME}{token}"
    staged.mkdir(parents=True, exist_ok=False)

    try:
        _copy_file(model_path, staged / MODEL_DIRNAME / model_path.name)
        _copy_directory(stereo_dir, staged / STEREO_DIRNAME)
        _copy_directory(background_dir, staged / BACKGROUND_DIRNAME)

        if mode == "reconstruir" and platform_dir is not None:
            _copy_directory(platform_dir, staged / PLATFORM_DIRNAME)

        firmware_rel = None
        if firmware_path is not None and Path(firmware_path).is_file():
            firmware_path = Path(firmware_path).expanduser().resolve()
            firmware_rel = f"{FIRMWARE_DIRNAME}/{firmware_path.name}"
            _copy_file(firmware_path, staged / firmware_rel)

        config = dict(capture_config or {})
        config.setdefault("frozen_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
        config.setdefault("mode", mode)
        if product_version is not None:
            config.setdefault("product_version", str(product_version))
        staged_config.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        resources: Dict[str, dict] = {
            "model": _resource_entry(model_path, f"{MODEL_DIRNAME}/{model_path.name}"),
            "stereo_calibration": _resource_entry(stereo_dir, STEREO_DIRNAME),
            "background": _resource_entry(background_dir, BACKGROUND_DIRNAME),
        }
        if mode == "reconstruir" and platform_dir is not None:
            resources["platform_calibration"] = _resource_entry(
                platform_dir,
                PLATFORM_DIRNAME,
            )
        if firmware_path is not None and Path(firmware_path).is_file() and firmware_rel:
            resources["firmware"] = _resource_entry(firmware_path, firmware_rel)

        frozen_inventory = _file_inventory(staged)
        metadata = {
            "schema_version": REFERENCES_SCHEMA_VERSION,
            "layout": "documentacion_v2",
            "policy": "immutable_campaign_snapshot",
            "mode": mode,
            "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "product_version": product_version,
            "references_directory": f"{DOCUMENTATION_DIRNAME}/{REFERENCES_DIRNAME}",
            "resources": resources,
            "capture_config": f"{DOCUMENTATION_DIRNAME}/{CAPTURE_CONFIG_FILENAME}",
            "capture_config_record": _capture_config_record(staged_config),
            "frozen_files": frozen_inventory,
            "frozen_content_digest": _json_digest(frozen_inventory),
            "note": (
                "Estas referencias pertenecen a esta campaña. Actualizar fondo, "
                "calibraciones o modelo globales no debe modificar este trabajo."
            ),
        }
        staged_metadata.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        _publish_snapshot_atomically(
            staged_references=staged,
            staged_config=staged_config,
            staged_metadata=staged_metadata,
            target_references=target,
            target_config=target_config,
            target_metadata=target_metadata,
        )

        legacy_root = _legacy_reference_root(workspace)
        if legacy_root.exists() and legacy_root != target:
            shutil.rmtree(legacy_root, ignore_errors=True)
        return metadata
    except Exception:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
        staged_config.unlink(missing_ok=True)
        staged_metadata.unlink(missing_ok=True)
        raise


def load_reference_metadata(workspace: Path) -> dict:
    _migrate_legacy_layout_if_needed(workspace)
    path = reference_metadata_path(workspace)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", -1)) != REFERENCES_SCHEMA_VERSION:
        raise ValueError(
            f"Versión de referencias no compatible: {data.get('schema_version')}"
        )
    return data


def frozen_reference_paths(workspace: Path, mode: str) -> dict:
    """Rutas canónicas de las referencias de una campaña."""
    root = reference_root(workspace)
    metadata = load_reference_metadata(workspace)
    resources = metadata.get("resources", {})

    model_rel = ((resources.get("model") or {}).get("frozen_path") or "").strip()
    if not model_rel:
        raise ValueError("La referencia congelada no declara el modelo ONNX.")

    result = {
        "frozen": True,
        "reference_root": root,
        "documentation_root": documentation_root(workspace),
        "metadata_path": reference_metadata_path(workspace),
        "capture_config": capture_config_path(workspace),
        "metadata": metadata,
        "model": root / model_rel,
        "stereo": root / STEREO_DIRNAME,
        "background": root / BACKGROUND_DIRNAME,
        "platform": None,
    }
    if str(mode).strip().lower() == "reconstruir":
        result["platform"] = root / PLATFORM_DIRNAME / "calibracion_plataforma.json"
    return result


def validate_campaign_references(
    workspace: Path,
    mode: str,
    *,
    verify_hashes: bool = True,
) -> dict:
    """Valida existencia e integridad del snapshot ubicado en documentacion/."""
    paths = frozen_reference_paths(workspace, mode)
    _require_file(paths["model"], "Modelo congelado")
    stereo = _require_dir(paths["stereo"], "Calibración estéreo congelada")
    background = _require_dir(paths["background"], "Fondo congelado")
    config = _require_file(paths["capture_config"], "Configuración de captura congelada")
    _require_file(stereo / "stereo_initial.yaml", "stereo_initial.yaml congelado")
    _require_file(stereo / "rectification_maps.npz", "rectification_maps.npz congelado")
    _require_file(background / "background_left.png", "background_left.png congelado")
    _require_file(background / "background_right.png", "background_right.png congelado")
    if paths["platform"] is not None:
        _require_file(paths["platform"], "Calibración de plataforma congelada")

    if verify_hashes:
        metadata = paths["metadata"]
        root = paths["reference_root"]
        mismatches = []
        for record in metadata.get("frozen_files", []):
            rel = str(record.get("path", ""))
            path = root / rel
            if not path.is_file():
                mismatches.append(f"falta {rel}")
                continue
            if int(path.stat().st_size) != int(record.get("size", -1)):
                mismatches.append(f"tamaño cambió: {rel}")
                continue
            if sha256_file(path) != str(record.get("sha256", "")):
                mismatches.append(f"contenido cambió: {rel}")

        config_record = metadata.get("capture_config_record") or {}
        if config_record:
            if int(config.stat().st_size) != int(config_record.get("size", -1)):
                mismatches.append(f"tamaño cambió: {CAPTURE_CONFIG_FILENAME}")
            elif sha256_file(config) != str(config_record.get("sha256", "")):
                mismatches.append(f"contenido cambió: {CAPTURE_CONFIG_FILENAME}")

        if mismatches:
            raise RuntimeError(
                "Las referencias congeladas fueron modificadas: " + "; ".join(mismatches[:8])
            )

    return paths


def compare_current_sources_to_snapshot(
    workspace: Path,
    mode: str,
    *,
    model_path: Path,
    stereo_dir: Path,
    background_dir: Path,
    platform_dir: Optional[Path] = None,
    firmware_path: Optional[Path] = None,
) -> dict:
    """Compara recursos globales actuales con los congelados en la campaña."""
    metadata = load_reference_metadata(workspace)
    resources = metadata.get("resources", {})

    current = {
        "model": Path(model_path),
        "stereo_calibration": Path(stereo_dir),
        "background": Path(background_dir),
    }
    if str(mode).strip().lower() == "reconstruir" and platform_dir is not None:
        current["platform_calibration"] = Path(platform_dir)
    if firmware_path is not None:
        current["firmware"] = Path(firmware_path)

    changed = []
    details = {}
    for key, path in current.items():
        expected = (resources.get(key) or {}).get("source_content_digest")
        now = content_digest(path)
        is_changed = (not now.get("exists")) or (expected is None) or (now.get("digest") != expected)
        details[key] = {
            "changed": bool(is_changed),
            "expected_digest": expected,
            "current_digest": now.get("digest"),
            "current_exists": bool(now.get("exists")),
        }
        if is_changed:
            changed.append(key)

    return {
        "changed": changed,
        "details": details,
        "frozen_at": metadata.get("frozen_at"),
    }


def resolve_campaign_references(
    workspace: Path,
    mode: str,
    *,
    fallback_model: Path,
    fallback_stereo: Path,
    fallback_background: Path,
    fallback_platform: Optional[Path] = None,
    verify_hashes: bool = True,
) -> dict:
    """Usa referencias congeladas si existen; conserva compatibilidad heredada."""
    workspace = Path(workspace).expanduser().resolve()
    if has_frozen_references(workspace):
        return validate_campaign_references(workspace, mode, verify_hashes=verify_hashes)

    # Campañas anteriores al congelamiento: no se inventa retrospectivamente
    # qué fondo o calibración les correspondía.
    return {
        "frozen": False,
        "legacy": True,
        "reference_root": None,
        "documentation_root": documentation_root(workspace),
        "metadata_path": None,
        "capture_config": None,
        "metadata": None,
        "model": Path(fallback_model).expanduser().resolve(),
        "stereo": Path(fallback_stereo).expanduser().resolve(),
        "background": Path(fallback_background).expanduser().resolve(),
        "platform": (
            Path(fallback_platform).expanduser().resolve()
            if fallback_platform is not None and str(fallback_platform)
            else None
        ),
    }
