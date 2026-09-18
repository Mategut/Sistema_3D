#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sistema 3D V3.2.1 — ejecución aislada, reanudación verificable y detención controlada.

Reglas
------
1. Un trabajo nunca reutiliza resultados de OTRO trabajo.
2. ``--resume`` solo reutiliza salidas del MISMO trabajo.
3. Cada paso se considera reutilizable únicamente si su contrato final existe,
   puede leerse y, cuando aplica, no declara ``quality=rejected``.
4. Los checkpoints guardan firma del script, dependencias Python locales, comando, entradas y archivos de
   contrato. Si cambia un paso o una entrada, ese paso y TODOS los siguientes
   se recalculan.
5. Sin ``--resume`` se mantiene el comportamiento histórico: se elimina
   ``reconstruccion/`` y se ejecuta desde el paso 01.

Estado persistente por trabajo
------------------------------
``reconstruccion/estado_pipeline/checkpoints.json``

El estado pertenece al trabajo; no se comparte entre campañas.
"""

from __future__ import annotations
from utilidades_rendimiento import configurar as _configurar_recursos

# Los runtimes numéricos leen estos límites al importarse.
_configurar_recursos(__file__)


import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from utilidades_multisesion import discover_sessions
from utilidades_referencias import resolve_campaign_references
from utilidades_almacenamiento import compact_reconstruction, print_compaction_report

DEPTH = "02_estimacion_profundidad"
SIL = "03_mascara_objeto"
REG = "04_validacion_disparidad"
CONS = "05_consenso_multisesion"
CLOUD = "06_nubes_puntos"
GEOM = "07_validacion_geometrica"
STATE_SCHEMA = 4


def parser():
    """Construye las opciones de línea de comandos de este paso."""
    p = argparse.ArgumentParser(
        description="Sistema 3D V3.2.1: aislado, reanudable y con detenciones controladas."
    )
    p.add_argument("mode", choices=["calibrar-plataforma", "reconstruir"])
    p.add_argument("--workspace", required=True, help="Raíz de ESTA campaña/capturas.")
    p.add_argument("--object", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--stereo-calibration-dir", required=True)
    p.add_argument("--background-dir", required=True)
    p.add_argument("--platform-calibration", default="", help="Obligatoria para reconstruir.")
    p.add_argument(
        "--platform-calibration-output-dir",
        default="",
        help="Obligatoria para calibrar-plataforma.",
    )
    p.add_argument(
        "--provider",
        choices=("auto", "cuda", "directml", "cpu"),
        default="auto",
        help=(
            "auto usa CUDA si está disponible y cae a DirectML/CPU de forma "
            "segura. Los valores explícitos exigen ese provider."
        ),
    )
    p.add_argument("--expected-sessions", type=int, default=3)
    p.add_argument("--regional-workers", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reutiliza únicamente checkpoints válidos del MISMO trabajo. "
            "Si encuentra un paso ausente, rechazado u obsoleto, continúa desde él."
        ),
    )
    p.add_argument(
        "--storage-mode",
        choices=("reducido", "completo"),
        default="reducido",
        help=(
            "reducido conserva al finalizar solo resúmenes, análisis y previews globales; "
            "completo mantiene todos los artefactos intermedios de diagnóstico."
        ),
    )
    return p


def require_file(path, label):
    """Resuelve la ruta y exige que corresponda a un archivo existente."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"{label}: {p}")
    return p


def require_dir(path, label):
    """Resuelve la ruta y exige que corresponda a un directorio existente."""
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"{label}: {p}")
    return p


def self_check_v72_angle_parser():
    """Bloquea el producto si A#### deja de interpretarse en décimas de grado."""
    from utilidades_mascaras import parse_angle

    checks = {
        "X_A0000": 0.0,
        "X_A0144": 14.4,
        "X_A0288": 28.8,
        "X_A3456": 345.6,
    }
    for stem, expected in checks.items():
        value = float(parse_angle(stem))
        if abs(value - expected) > 1e-9:
            raise RuntimeError(f"SELF-CHECK V7.2 FALLÓ: {stem} -> {value}, esperado {expected}")
    print("[SELF-CHECK V7.2] parser angular: OK")


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    """Calcula la huella SHA-256 del archivo mediante lectura por bloques."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _json_digest(payload) -> str:
    """Calcula una firma de JSON con claves ordenadas y separadores estables."""
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def fingerprint_path(path: Path) -> dict:
    """Firma estable de una entrada.

    Archivos: SHA-256 real. Directorios: manifiesto recursivo de nombres,
    tamaños y mtime_ns, más SHA-256 real de archivos pequeños de metadatos.
    Esto evita volver a hashear centenares de imágenes grandes en cada resume,
    pero detecta cambios normales de captura/salidas intermedias.
    """
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_file():
        return {
            "path": str(path.resolve()),
            "exists": True,
            "type": "file",
            "size": int(path.stat().st_size),
            "mtime_ns": int(path.stat().st_mtime_ns),
            "sha256": sha256_file(path),
        }

    records = []
    metadata_exts = {".json", ".csv", ".yaml", ".yml", ".xml", ".txt"}
    for f in sorted((x for x in path.rglob("*") if x.is_file()), key=lambda x: str(x).lower()):
        st = f.stat()
        rec = {
            "rel": str(f.relative_to(path)).replace("\\", "/"),
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }
        # Hash real para archivos de control pequeños; para nubes/imágenes grandes
        # se usa el manifiesto nombre+tamaño+mtime.
        if f.suffix.lower() in metadata_exts and st.st_size <= 16 * 1024 * 1024:
            rec["sha256"] = sha256_file(f)
        records.append(rec)
    return {
        "path": str(path.resolve()),
        "exists": True,
        "type": "dir",
        "file_count": len(records),
        "manifest_sha256": _json_digest(records),
    }


def local_python_dependencies(script: Path) -> list[Path]:
    """Resuelve imports locales transitivos para la firma de checkpoint.

    Problema que evita:
    un paso puede no cambiar de archivo principal, pero sí cambiar una utilidad
    compartida (por ejemplo ``utilidades_mascaras.py``). Un checkpoint basado solo
    en el script principal reutilizaría entonces una salida generada con código
    diferente. V3.2.0 incorpora las dependencias locales reales a la firma.
    """
    script = Path(script).resolve()
    base = script.parent
    visited: set[Path] = set()
    result: list[Path] = []

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in visited or not path.is_file():
            return
        visited.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            # La compilación general del paquete detectará el problema; aquí no
            # ocultamos el paso principal, simplemente no inventamos imports.
            return

        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])

        for name in sorted(names):
            candidate = base / f"{name}.py"
            if candidate.is_file() and candidate.resolve() != script:
                if candidate.resolve() not in result:
                    result.append(candidate.resolve())
                visit(candidate)

    visit(script)
    return sorted(result, key=lambda x: x.name.lower())


def latest_mtime(paths: Iterable[Path]) -> float:
    """Obtiene la fecha de modificación más reciente entre las rutas existentes."""
    latest = 0.0
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        if p.is_file():
            latest = max(latest, p.stat().st_mtime)
        else:
            for f in p.rglob("*"):
                if f.is_file():
                    try:
                        latest = max(latest, f.stat().st_mtime)
                    except OSError:
                        pass
    return latest


def atomic_write_json(path: Path, payload: dict) -> None:
    """Escribe el JSON en un temporal y reemplaza el destino al finalizar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> dict:
    """Lee un documento JSON codificado en UTF-8."""
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class ContractResult:
    """Resultado de comprobar si la salida de una etapa cumple su contrato."""

    ok: bool
    reason: str = "ok"


@dataclass
class Stage:
    """Comando, entradas y contratos necesarios para ejecutar o reutilizar una etapa."""

    stage_id: str
    label: str
    script: Path
    cmd: list[str]
    sentinels: list[Path]
    inputs: list[Path] = field(default_factory=list)
    quality_json: Optional[Path] = None
    quality_keys: Sequence[str] = ("quality",)
    reject_values: Sequence[str] = ("rejected", "reject", "failed", "error")
    extra_validator: Optional[Callable[[], ContractResult]] = None
    allow_code2: bool = False
    allow_bootstrap: bool = True
    # Cuando un paso termina correctamente pero declara que su salida no debe
    # ser consumida (por ejemplo, registro rechazado por calidad), se guarda
    # un checkpoint "blocked" y el pipeline termina limpiamente, sin traceback.
    controlled_stop_on_reject: bool = False

    def signature_payload(self) -> dict:
        dependencies = local_python_dependencies(self.script)
        return {
            "stage_id": self.stage_id,
            "script": fingerprint_path(self.script),
            "local_python_dependencies": [fingerprint_path(p) for p in dependencies],
            "command": [str(x) for x in self.cmd],
            "inputs": [fingerprint_path(p) for p in self.inputs],
        }

    def signature(self) -> str:
        return _json_digest(self.signature_payload())

    def contract_fingerprint(self) -> dict:
        return {str(p): fingerprint_path(p) for p in self.sentinels}

    def validate_artifacts(self) -> ContractResult:
        """Valida que el paso realmente terminó y dejó artefactos legibles.

        A diferencia de :meth:`validate_contract`, NO exige que la calidad sea
        aceptada. Esto permite distinguir un rechazo metodológico válido de un
        fallo de ejecución.
        """
        for p in self.sentinels:
            if not p.is_file():
                return ContractResult(False, f"falta {p.name}")
            if p.suffix.lower() == ".json":
                try:
                    load_json(p)
                except Exception as exc:
                    return ContractResult(False, f"JSON inválido {p.name}: {exc}")
        return ContractResult(True, "ok")

    def quality_rejection_reason(self) -> Optional[str]:
        if self.quality_json is None:
            return None
        if not self.quality_json.is_file():
            return None
        try:
            data = load_json(self.quality_json)
        except Exception:
            return None
        rejected = {str(x).strip().lower() for x in self.reject_values}
        for key in self.quality_keys:
            if key not in data:
                continue
            value = data.get(key)
            if str(value).strip().lower() in rejected:
                return f"{key}={value}"
        return None

    def validate_contract(self) -> ContractResult:
        artifacts = self.validate_artifacts()
        if not artifacts.ok:
            return artifacts

        if self.quality_json is not None:
            if not self.quality_json.is_file():
                return ContractResult(False, f"falta resumen de calidad {self.quality_json.name}")
            try:
                load_json(self.quality_json)
            except Exception as exc:
                return ContractResult(False, f"resumen de calidad inválido: {exc}")
            reject_reason = self.quality_rejection_reason()
            if reject_reason is not None:
                return ContractResult(False, reject_reason)

        if self.extra_validator is not None:
            result = self.extra_validator()
            if not result.ok:
                return result

        return ContractResult(True, "ok")


class CheckpointManager:
    """Gestiona checkpoints del trabajo y su validez frente a cambios de entradas."""

    def __init__(self, workspace: Path, obj: str, mode: str, resume: bool):
        self.workspace = workspace.resolve()
        self.obj = obj
        self.mode = mode
        self.resume = bool(resume)
        self.reconstruction = self.workspace / "reconstruccion"
        self.state_dir = self.reconstruction / "estado_pipeline"
        self.state_path = self.state_dir / "checkpoints.json"
        self._visited_stage_ids = set()
        self.force_from_here = not self.resume
        self.state = {
            "schema_version": STATE_SCHEMA,
            "workspace": str(self.workspace),
            "object": obj,
            "mode": mode,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "updated_at": None,
            "stages": {},
        }
        if self.resume and self.state_path.is_file():
            try:
                old = load_json(self.state_path)
                if (
                    int(old.get("schema_version", -1)) == STATE_SCHEMA
                    and Path(old.get("workspace", "")).resolve() == self.workspace
                    and old.get("object") == obj
                    and old.get("mode") == mode
                ):
                    self.state = old
                else:
                    print(
                        "[RESUME] Estado previo incompatible; se auditarán las salidas existentes."
                    )
            except Exception as exc:
                print(f"[RESUME] No se pudo leer el estado previo ({exc}); se auditarán salidas.")

    def reset(self):
        self._visited_stage_ids.clear()
        self.state = {
            "schema_version": STATE_SCHEMA,
            "workspace": str(self.workspace),
            "object": self.obj,
            "mode": self.mode,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "updated_at": None,
            "stages": {},
        }
        self.force_from_here = True

    def save(self):
        self.state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_write_json(self.state_path, self.state)

    def _checkpoint_valid(self, stage: Stage) -> ContractResult:
        if bool(self.state.get("storage_compacted", False)):
            return ContractResult(
                False,
                "campaña compactada: los intermedios fueron retirados y deben recalcularse",
            )
        contract = stage.validate_contract()
        if not contract.ok:
            return contract

        rec = self.state.get("stages", {}).get(stage.stage_id)
        if rec is None:
            if not stage.allow_bootstrap:
                return ContractResult(False, "paso sin checkpoint previo; bootstrap deshabilitado")
            # Bootstrap de una campaña creada antes de existir checkpoints.
            # Se exige que las salidas no sean más antiguas que script/entradas.
            if not stage.sentinels:
                return ContractResult(False, "sin sentinels")
            oldest_output = min(p.stat().st_mtime for p in stage.sentinels if p.is_file())
            newest_input = max(
                stage.script.stat().st_mtime if stage.script.is_file() else 0.0,
                latest_mtime(local_python_dependencies(stage.script)),
                latest_mtime(stage.inputs),
            )
            if newest_input > oldest_output + 2.0:
                return ContractResult(
                    False,
                    "salida existente anterior a una entrada/script; no es segura para bootstrap",
                )
            signature = stage.signature()
            self.state.setdefault("stages", {})[stage.stage_id] = {
                "status": "complete",
                "origin": "bootstrap_existing_output",
                "signature": signature,
                "contract": stage.contract_fingerprint(),
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self.save()
            return ContractResult(True, "bootstrap")

        if rec.get("status") != "complete":
            return ContractResult(False, f"checkpoint status={rec.get('status')}")
        current_signature = stage.signature()
        if rec.get("signature") != current_signature:
            return ContractResult(False, "firma de script/comando/entradas cambió")
        if rec.get("contract") != stage.contract_fingerprint():
            return ContractResult(False, "archivos de contrato cambiaron")
        return ContractResult(True, "checkpoint")

    def _record_blocked_stage(self, stage: Stage, return_code: int, reason: str) -> None:
        self.state.setdefault("stages", {})[stage.stage_id] = {
            "status": "blocked",
            "origin": "executed_controlled_stop",
            "signature": stage.signature(),
            "contract": stage.contract_fingerprint(),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "return_code": int(return_code),
            "block_reason": str(reason),
        }
        self.save()

    def _matching_blocked_checkpoint(self, stage: Stage) -> Optional[str]:
        rec = self.state.get("stages", {}).get(stage.stage_id)
        if not rec or rec.get("status") != "blocked":
            return None
        try:
            if rec.get("signature") != stage.signature():
                return None
            if rec.get("contract") != stage.contract_fingerprint():
                return None
        except Exception:
            return None
        return str(rec.get("block_reason") or "salida rechazada")

    def run_stage(self, stage: Stage) -> int:
        self._visited_stage_ids.add(stage.stage_id)
        if self.resume and not self.force_from_here:
            if stage.controlled_stop_on_reject:
                blocked_reason = self._matching_blocked_checkpoint(stage)
                if blocked_reason is not None:
                    print(
                        f"[RESUME] ⏸ {stage.stage_id} | {stage.label} | "
                        f"sigue detenido ({blocked_reason})",
                        flush=True,
                    )
                    return 2

            valid = self._checkpoint_valid(stage)
            if valid.ok:
                origin = (
                    "checkpoint" if valid.reason == "checkpoint" else "salida existente validada"
                )
                print(f"[RESUME] ✓ {stage.stage_id} | {stage.label} | reutilizado ({origin})")
                return 0
            self.force_from_here = True
            print(
                f"[RESUME] ✗ {stage.stage_id} | {stage.label} | {valid.reason}\n"
                f"[RESUME] Se reanuda desde {stage.stage_id}; todos los pasos posteriores "
                "se recalcularán.",
                flush=True,
            )

        # Antes de escribir salidas, deja constancia durable del paso en curso.
        # Los checkpoints posteriores quedan invalidados incluso si se cierra
        # la aplicación antes de que este paso termine.
        records = self.state.setdefault("stages", {})
        for stage_id, record in records.items():
            if stage_id not in self._visited_stage_ids:
                record["status"] = "invalidated"
        records[stage.stage_id] = {
            "status": "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "script": str(stage.script),
        }
        self.save()
        print(f"[ETAPA] {stage.stage_id} | {stage.label}", flush=True)
        print(f"\n========== {stage.label} ==========", flush=True)
        print(f"[PROGRESO] Se está ejecutando: {stage.script.name}", flush=True)
        print(f"[PROGRESO] {stage.label}", flush=True)
        print(">>>", " ".join(map(str, stage.cmd)), flush=True)
        child_env = os.environ.copy()
        child_env["PYTHONUTF8"] = "1"
        child_env["PYTHONUNBUFFERED"] = "1"
        child_env["PYTHONIOENCODING"] = "utf-8"
        import time as _perf_time

        _started = _perf_time.perf_counter()
        r = subprocess.run(stage.cmd, env=child_env, check=False)
        print(f"[TIEMPO] {stage.label}: {_perf_time.perf_counter()-_started:.1f} s", flush=True)

        # Un código 2 puede significar "resultado calculado, pero no apto para
        # continuar". Solo se trata como detención controlada en pasos que lo
        # declaran explícitamente y únicamente si el resumen de calidad existe
        # y confirma el rechazo. De lo contrario sigue siendo un error real.
        if r.returncode == 2 and stage.controlled_stop_on_reject:
            artifacts = stage.validate_artifacts()
            rejection = stage.quality_rejection_reason()
            if not artifacts.ok:
                raise subprocess.CalledProcessError(r.returncode, stage.cmd)
            if rejection is None:
                raise RuntimeError(
                    f"{stage.stage_id} devolvió código 2, pero no declaró un rechazo de calidad. "
                    "Se trata como fallo real para no ocultar errores."
                )
            self._record_blocked_stage(stage, r.returncode, rejection)
            print(
                f"[DETENIDO] {stage.stage_id} terminó correctamente, pero {rejection}.", flush=True
            )
            print(
                "[DETENIDO] Los diagnósticos fueron guardados y NO se consumirán aguas abajo.",
                flush=True,
            )
            return 2

        if r.returncode and not (stage.allow_code2 and r.returncode == 2):
            raise subprocess.CalledProcessError(r.returncode, stage.cmd)

        contract = stage.validate_contract()
        if not contract.ok:
            # Compatibilidad con scripts que históricamente devolvían 0 aunque
            # su resumen declarara rejected: en pasos de detención controlada
            # eso también debe parar limpiamente, nunca crear un checkpoint completo.
            if stage.controlled_stop_on_reject and stage.quality_rejection_reason() is not None:
                self._record_blocked_stage(stage, int(r.returncode), contract.reason)
                print(
                    f"[DETENIDO] {stage.stage_id} terminó correctamente, pero {contract.reason}.",
                    flush=True,
                )
                print(
                    "[DETENIDO] Los diagnósticos fueron guardados y NO se consumirán aguas abajo.",
                    flush=True,
                )
                return 2
            raise RuntimeError(
                f"{stage.stage_id} terminó, pero su salida NO es reutilizable: {contract.reason}. "
                "El pipeline se detiene antes de consumirla."
            )

        self.state.setdefault("stages", {})[stage.stage_id] = {
            "status": "complete",
            "origin": "executed",
            "signature": stage.signature(),
            "contract": stage.contract_fingerprint(),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "return_code": int(r.returncode),
        }
        self.save()
        print(f"[CHECKPOINT] ✓ {stage.stage_id} guardado", flush=True)
        print(f"[PROGRESO] Completado: {stage.label}", flush=True)
        return int(r.returncode)


def clean_current_results(workspace: Path, obj: str):
    """Elimina la reconstrucción previa y reinicia las salidas del trabajo actual."""
    object_dir = workspace
    reconstruction = object_dir / "reconstruccion"
    if reconstruction.exists():
        print("[CLEAN] Se elimina SOLO la reconstrucción previa de esta carpeta:", reconstruction)
        shutil.rmtree(reconstruction)
    reconstruction.mkdir(parents=True, exist_ok=True)

    # Al iniciar desde cero se evita que un OBJ anterior parezca pertenecer a
    # una reconstrucción nueva que todavía no ha llegado a la exportación.
    final_result = object_dir / "resultado_final"
    if final_result.exists():
        print("[CLEAN] Se elimina resultado_final previo:", final_result)
        shutil.rmtree(final_result)


def session_base(workspace: Path, obj: str, session: str) -> Path:
    """Devuelve el directorio de reconstrucción de una sesión del trabajo."""
    return workspace / "reconstruccion" / session


def multi_base(workspace: Path, obj: str) -> Path:
    """Devuelve el directorio de resultados multisesión del trabajo."""
    return workspace / "reconstruccion" / "multisesion"


def json_not_rejected(path: Path, keys=("quality",)) -> Callable[[], ContractResult]:
    """Crea un validador que comprueba lectura del JSON y ausencia de rechazo."""

    def validator():
        try:
            data = load_json(path)
        except Exception as exc:
            return ContractResult(False, f"no se puede leer {path.name}: {exc}")
        for key in keys:
            if key in data:
                value = str(data.get(key)).strip().lower()
                if value in {"rejected", "reject", "failed", "error"}:
                    return ContractResult(False, f"{key}={data.get(key)}")
        return ContractResult(True, "ok")

    return validator


def audit_platform_calibration(
    candidate: Path,
    canonical: Path,
    audit: Path,
) -> ContractResult:
    """Verifica la calibración y su correspondencia con la auditoría guardada."""
    cal = candidate if candidate.is_file() else canonical
    if not cal.is_file():
        return ContractResult(False, "no existe calibración candidata/canónica")
    try:
        data = load_json(cal)
    except Exception as exc:
        return ContractResult(False, f"calibración JSON inválida: {exc}")
    status = str(data.get("status", "")).strip().lower()
    allowed_status = {
        "candidate_ready_for_independent_validation",
        "accepted",
        "active",
    }
    if status not in allowed_status:
        return ContractResult(False, f"calibration status={status or 'missing'}")
    if not audit.is_file():
        return ContractResult(False, "falta auditoria_congelacion_calibracion.json")
    try:
        ad = load_json(audit)
    except Exception as exc:
        return ContractResult(False, f"auditoría inválida: {exc}")
    if ad.get("passed") is not True:
        return ContractResult(False, "auditoria_congelacion passed!=true")
    expected_sha = str(ad.get("calibration_sha256", "")).strip().lower()
    if not expected_sha:
        return ContractResult(False, "auditoría sin calibration_sha256")
    actual_sha = sha256_file(cal)
    if actual_sha.lower() != expected_sha:
        return ContractResult(False, "SHA-256 de calibración no coincide con auditoría")
    return ContractResult(True, "ok")


def validate_topology_summary(summary: Path) -> ContractResult:
    """Comprueba que el resumen de topología no bloquee la siguiente validación."""
    try:
        data = load_json(summary)
    except Exception as exc:
        return ContractResult(False, f"resumen de topología inválido: {exc}")
    blockers = data.get("blockers_for_step_17", [])
    if blockers:
        return ContractResult(False, f"bloqueadores topológicos: {blockers}")
    status = str(data.get("status", "")).strip().lower()
    if status not in {"ready_for_step_15", "accepted", "warning"}:
        return ContractResult(False, f"status={status or 'missing'}")
    return ContractResult(True, "ok")


def make_stage(
    stage_id: str,
    label: str,
    script: Path,
    cmd: list[str],
    sentinels: list[Path],
    inputs: list[Path],
    **kwargs,
) -> Stage:
    """Construye una etapa con sus contratos y dependencias explícitos."""
    if not script.is_file():
        raise FileNotFoundError(f"Script de {stage_id}: {script}")
    return Stage(stage_id, label, script, cmd, sentinels, inputs, **kwargs)


def build_preprocess_stages(a, base, workspace, obj, py, stereo, background, model):
    """Prepara las etapas de profundidad, máscaras y nubes para cada sesión."""
    sessions = discover_sessions(workspace, obj)
    if len(sessions) != a.expected_sessions:
        raise RuntimeError(f"Se esperaban {a.expected_sessions} sesiones; encontradas: {sessions}")

    captures_root = workspace / "capturas"
    multi = multi_base(workspace, obj)
    manifest = multi / "01_mapa_angular" / "mapa_angular_multisesion.json"

    script = base / "01_crear_mapa_angular.py"
    yield make_stage(
        "01",
        "Paso 01 | Crear mapa angular multisesión",
        script,
        [
            py,
            str(script),
            "--root",
            str(workspace),
            "--object",
            obj,
            "--expected-sessions",
            str(a.expected_sessions),
        ],
        [manifest],
        [captures_root],
    )

    for session in sessions:
        cap = captures_root / session
        sb = session_base(workspace, obj, session)

        depth_dir = sb / DEPTH
        depth_summary = depth_dir / "resumen_02_estimacion_profundidad.json"
        script = base / "02_estimar_profundidad_crestereo.py"
        yield make_stage(
            f"02:{session}",
            f"Paso 02 | {session} | Estimar profundidad CREStereo",
            script,
            [
                py,
                str(script),
                "--model",
                str(model),
                "--root",
                str(workspace),
                "--object",
                obj,
                "--session",
                session,
                "--provider",
                a.provider,
                "--output-name",
                DEPTH,
                "--calibration-dir",
                str(stereo),
                "--background-dir",
                str(background),
            ],
            [depth_summary],
            [cap, model, stereo, background],
        )

        sil_dir = sb / SIL
        sil_summary = sil_dir / "resumen_03_mascara_objeto.json"
        script = base / "03_crear_mascara_objeto.py"
        yield make_stage(
            f"03:{session}",
            f"Paso 03 | {session} | Crear máscara del objeto",
            script,
            [
                py,
                str(script),
                "--root",
                str(workspace),
                "--object",
                obj,
                "--session",
                session,
                "--depth-source",
                DEPTH,
                "--output-name",
                SIL,
            ],
            [sil_summary],
            [depth_dir],
        )

        reg_dir = sb / REG
        reg_summary = reg_dir / "resumen_04_validacion_disparidad.json"
        script = base / "04_validar_disparidad.py"
        yield make_stage(
            f"04:{session}",
            f"Paso 04 | {session} | Validar profundidad",
            script,
            [
                py,
                str(script),
                "--root",
                str(workspace),
                "--object",
                obj,
                "--session",
                session,
                "--depth-source",
                DEPTH,
                "--silhouette-source",
                SIL,
                "--output-name",
                REG,
                "--calibration-dir",
                str(stereo),
            ],
            [reg_summary],
            [depth_dir, sil_dir, stereo],
        )

    cons_dir = multi / CONS
    cons_summary = cons_dir / "resumen_05_consenso_multisesion.json"
    script = base / "05_crear_consenso_multisesion.py"
    reg_inputs = []
    for session in sessions:
        sb = session_base(workspace, obj, session)
        reg_inputs.extend([sb / DEPTH, sb / SIL, sb / REG])
    yield make_stage(
        "05",
        "Paso 05 | Crear consenso multisesión",
        script,
        [
            py,
            str(script),
            "--root",
            str(workspace),
            "--object",
            obj,
            "--manifest",
            str(manifest),
            "--depth-source",
            DEPTH,
            "--silhouette-source",
            SIL,
            "--regional-source",
            REG,
            "--output-name",
            CONS,
        ],
        [cons_summary],
        [manifest, *reg_inputs],
    )

    cloud_dir = multi / CLOUD
    cloud_summary = cloud_dir / "resumen_06_nubes_puntos.json"
    script = base / "06_crear_nubes_puntos.py"
    yield make_stage(
        "06",
        "Paso 06 | Crear nubes de puntos",
        script,
        [
            py,
            str(script),
            "--root",
            str(workspace),
            "--object",
            obj,
            "--consensus-source",
            CONS,
            "--output-name",
            CLOUD,
            "--calibration-dir",
            str(stereo),
        ],
        [cloud_summary],
        [cons_dir, stereo],
    )


def run_pipeline(a, base, workspace, obj, py, stereo, background, model):
    """Ejecuta o reanuda las etapas y detiene el flujo si falla un contrato."""
    manager = CheckpointManager(workspace, obj, a.mode, a.resume)
    if not a.resume:
        clean_current_results(workspace, obj)
        if a.mode == "calibrar-plataforma":
            local_calibration_result = workspace / "resultado_calibracion_plataforma"
            if local_calibration_result.exists():
                print(
                    "[CLEAN] Se elimina resultado_calibracion_plataforma previo:",
                    local_calibration_result,
                )
                shutil.rmtree(local_calibration_result)
        manager.reset()
        manager.save()
        print("[PIPELINE] Modo: DESDE CERO")
    else:
        manager.reconstruction.mkdir(parents=True, exist_ok=True)
        manager.state_dir.mkdir(parents=True, exist_ok=True)
        print("[PIPELINE] Modo: REANUDAR MISMO TRABAJO")
        print("[PIPELINE] No se borrarán resultados válidos previos de esta campaña.")

    for stage in build_preprocess_stages(a, base, workspace, obj, py, stereo, background, model):
        manager.run_stage(stage)

    multi = multi_base(workspace, obj)
    cloud_dir = multi / CLOUD

    if a.mode == "calibrar-plataforma":
        if not a.platform_calibration_output_dir:
            raise ValueError("--platform-calibration-output-dir es obligatorio al calibrar.")
        outcal = Path(a.platform_calibration_output_dir).expanduser().resolve()
        outcal.mkdir(parents=True, exist_ok=True)

        geom_dir = multi / GEOM
        geom_summary = geom_dir / "resumen_07_validacion_geometrica.json"
        script = base / "07_validar_geometria_nubes.py"
        manager.run_stage(
            make_stage(
                "07",
                "Paso 07 | Validar geometría de las nubes",
                script,
                [
                    py,
                    str(script),
                    "--root",
                    str(workspace),
                    "--object",
                    obj,
                    "--session",
                    "multisesion",
                    "--cloud-source",
                    CLOUD,
                    "--output-name",
                    GEOM,
                ],
                [geom_summary],
                [cloud_dir],
            )
        )

        reg_dir = multi / "08_registro_referencia"
        reg_summary = reg_dir / "resumen_08_registro_referencia.json"
        reg_poses = reg_dir / "poses_registradas_v3_2.csv"
        reg_edges = reg_dir / "calidad_aristas_v3_2.csv"
        reg_obs = reg_dir / "observabilidad_angular_v3_2.csv"
        script = base / "08_registrar_vistas_referencia.py"
        manager.run_stage(
            make_stage(
                "08",
                "Paso 08 | Registrar vistas de referencia",
                script,
                [
                    py,
                    str(script),
                    "--root",
                    str(workspace),
                    "--object",
                    obj,
                    "--session",
                    "multisesion",
                    "--cloud-source",
                    CLOUD,
                    "--geometry-source",
                    GEOM,
                ],
                [reg_summary, reg_poses, reg_edges, reg_obs],
                [cloud_dir, geom_dir],
                quality_json=reg_summary,
            )
        )

        candidate = outcal / "calibracion_plataforma_candidata.json"
        canonical = outcal / "calibracion_plataforma.json"
        transforms = outcal / "transformaciones_mecanicas_25_poses.csv"
        audit = outcal / "auditoria_congelacion_calibracion.json"
        script = base / "09_guardar_calibracion_plataforma.py"
        stage = make_stage(
            "09",
            "Paso 09 | Guardar calibración de plataforma",
            script,
            [
                py,
                str(script),
                "--root",
                str(workspace),
                "--source-object",
                obj,
                "--output-dir",
                str(outcal),
                "--stereo-calibration-dir",
                str(stereo),
            ],
            # La candidata se conserva como sentinel; la GUI la copia de forma
            # atómica al nombre canónico sin retirarla del checkpoint.
            [candidate, transforms, audit],
            [reg_dir, stereo],
            extra_validator=lambda: audit_platform_calibration(
                candidate,
                canonical,
                audit,
            ),
            allow_code2=False,
            allow_bootstrap=False,
        )
        manager.run_stage(stage)
        if a.storage_mode == "reducido":
            report = compact_reconstruction(workspace, reason="platform_calibration_complete")
            print_compaction_report(report)
        return 0

    if not a.platform_calibration:
        raise ValueError("--platform-calibration es obligatorio para reconstruir.")
    platform = require_file(a.platform_calibration, "Calibración de plataforma")

    d3 = multi / "10_registro_calibrado"
    s3 = d3 / "resumen_10_registro_calibrado.json"
    script = base / "10_registrar_vistas_calibradas.py"
    rc10 = manager.run_stage(
        make_stage(
            "10",
            "Paso 10 | Registrar vistas calibradas y evaluar A/B del eje",
            script,
            [
                py,
                str(script),
                "--root",
                str(workspace),
                "--object",
                obj,
                "--calibration",
                str(platform),
                # Se evalúa el candidato A/B, pero Paso 10 V3.3 conserva
                # la calibración congelada salvo autorización explícita.
                "--refine-axis-line",
            ],
            [s3],
            [cloud_dir, platform],
            quality_json=s3,
            # No reutilizar una salida antigua sin constancia del nuevo comando.
            allow_bootstrap=False,
            allow_code2=True,
            controlled_stop_on_reject=True,
        )
    )
    if rc10 == 2:
        print("\n========== PIPELINE DETENIDO DE FORMA CONTROLADA ==========", flush=True)
        print(
            "[DETENIDO] El Paso 10 produjo diagnósticos válidos, pero el registro NO es apto para fusión.",
            flush=True,
        )
        print("[DETENIDO] No se ejecutarán los pasos 11–18.", flush=True)
        result_dir = workspace / "resultado_final"
        if result_dir.exists():
            print(
                "[AVISO] Existe resultado_final de una ejecución anterior. "
                "Se conserva, pero NO pertenece a esta corrida detenida.",
                flush=True,
            )
        print("[PIPELINE] Finalización limpia sin modelo nuevo (código de salida 0).", flush=True)
        print("============================================================", flush=True)
        return 0

    d4a = multi / "11_fusion_multivista"
    s4a = d4a / "resumen_11_fusion_multivista.json"
    script = base / "11_fusionar_nubes.py"
    manager.run_stage(
        make_stage(
            "11",
            "Paso 11 | Fusionar nubes con consenso jerárquico",
            script,
            [
                py,
                str(script),
                "--root",
                str(workspace),
                "--object",
                obj,
                "--calibration",
                str(platform),
            ],
            [s4a],
            [d3, platform],
        )
    )

    d4b = multi / "12_regularizacion_nube"
    s4b = d4b / "resumen_12_regularizacion_nube.json"
    script = base / "12_regularizar_nube.py"
    manager.run_stage(
        make_stage(
            "12",
            "Paso 12 | Regularizar nube de puntos",
            script,
            [py, str(script), "--root", str(workspace), "--object", obj],
            [s4b],
            [d4a],
        )
    )

    d4c = multi / "13_reconstruccion_superficie"
    s4c = d4c / "resumen_13_reconstruccion_superficie.json"
    mesh4c = d4c / "malla_final_seleccionada.ply"
    script = base / "13_reconstruir_superficie.py"
    manager.run_stage(
        make_stage(
            "13",
            "Paso 13 | Reconstruir superficie",
            script,
            [py, str(script), "--root", str(workspace), "--object", obj],
            [s4c, mesh4c],
            [d4b],
            quality_json=s4c,
            allow_code2=False,
        )
    )

    d4c2 = multi / "14_limpieza_topologica"
    s4c2 = d4c2 / "resumen_14_limpieza_topologica.json"
    mesh_topo = d4c2 / "malla_final_topologica.ply"
    script = base / "14_limpiar_topologia.py"
    manager.run_stage(
        make_stage(
            "14",
            "Paso 14 | Limpiar topología",
            script,
            [py, str(script), "--root", str(workspace), "--object", obj],
            [s4c2, mesh_topo],
            [d4c],
            allow_code2=False,
            extra_validator=lambda: validate_topology_summary(s4c2),
        )
    )

    d4c3 = multi / "15_pulido_final"
    s4c3 = d4c3 / "resumen_15_pulido_final.json"
    mesh4c3 = d4c3 / "malla_final_topologica.ply"
    script = base / "15_pulir_modelo.py"
    manager.run_stage(
        make_stage(
            "15",
            "Paso 15 | Pulir modelo con guardas locales",
            script,
            [py, str(script), "--root", str(workspace), "--object", obj],
            [s4c3, mesh4c3],
            [d4c2, d4b],
            quality_json=s4c3,
            allow_code2=False,
        )
    )

    d4v = multi / "16_validacion_intersecciones"
    s4v = d4v / "resumen_16_validacion_intersecciones.json"
    script = base / "16_validar_intersecciones.py"
    manager.run_stage(
        make_stage(
            "16",
            "Paso 16 | Validar intersecciones",
            script,
            [py, str(script), "--root", str(workspace), "--object", obj],
            [s4v],
            [d4c3, d4c2],
            allow_code2=False,
            extra_validator=lambda: ContractResult(
                str(load_json(s4v).get("status", "")).lower()
                in {"ready_for_step_17", "accepted", "warning"},
                f"status={load_json(s4v).get('status')}" if s4v.is_file() else "sin resumen",
            ),
        )
    )

    d4d = multi / "17_validacion_modelo"
    s4d = d4d / "resumen_17_validacion_modelo.json"
    script = base / "17_validar_modelo.py"
    manager.run_stage(
        make_stage(
            "17",
            "Paso 17 | Validar modelo y núcleo multivista",
            script,
            [py, str(script), "--root", str(workspace), "--object", obj],
            [s4d],
            [d4v, d4c3, d4b],
            quality_json=s4d,
            allow_code2=True,
        )
    )

    d4e = multi / "18_exportacion_modelo"
    s4e = d4e / "resumen_18_exportacion_modelo.json"
    result_dir = workspace / "resultado_final"
    final_obj = result_dir / "modelo_final.obj"
    final_obj_mm = result_dir / "modelo_final_metric_mm.obj"
    final_ply = result_dir / "modelo_final_original_mm.ply"
    pre_polish_obj = result_dir / "modelo_pre_pulido_metric_mm.obj"
    pre_polish_ply = result_dir / "modelo_pre_pulido_mm.ply"

    script = base / "18_exportar_modelo_blender.py"
    manager.run_stage(
        make_stage(
            "18",
            "Paso 18 | Exportar modelo para Blender",
            script,
            [py, str(script), "--root", str(workspace), "--object", obj],
            [
                s4e,
                final_obj,
                final_obj_mm,
                final_ply,
                pre_polish_obj,
                pre_polish_ply,
            ],
            [d4d, d4c3, d4c2],
            quality_json=s4e,
            allow_code2=False,
            allow_bootstrap=False,
        )
    )

    if a.storage_mode == "reducido":
        report = compact_reconstruction(workspace, reason="reconstruction_complete")
        print_compaction_report(report)

    return 0


def main():
    """Valida los recursos del trabajo y ejecuta el flujo solicitado."""
    print(
        "[PROGRESO] Se está ejecutando: 00_ejecutar_pipeline.py",
        flush=True,
    )
    print(
        "[PROGRESO] Preparando y validando el flujo completo",
        flush=True,
    )
    self_check_v72_angle_parser()
    a = parser().parse_args()
    base = Path(__file__).resolve().parent
    py = sys.executable

    workspace = require_dir(a.workspace, "Workspace")
    obj = a.object.strip().lower()

    # Campañas nuevas llevan dentro una copia inmutable de sus referencias.
    # Los argumentos CLI se conservan como fallback únicamente para trabajos
    # creados antes de esta arquitectura.
    refs = resolve_campaign_references(
        workspace,
        a.mode,
        fallback_model=Path(a.model),
        fallback_stereo=Path(a.stereo_calibration_dir),
        fallback_background=Path(a.background_dir),
        fallback_platform=(Path(a.platform_calibration) if a.platform_calibration else None),
        verify_hashes=True,
    )

    model = require_file(refs["model"], "Modelo ONNX")
    stereo = require_dir(refs["stereo"], "Calibración estéreo")
    require_file(stereo / "stereo_initial.yaml", "stereo_initial.yaml")
    require_file(stereo / "rectification_maps.npz", "rectification_maps.npz")
    background = require_dir(refs["background"], "Fondo vacío")
    require_file(background / "background_left.png", "background_left.png")
    require_file(background / "background_right.png", "background_right.png")

    if refs.get("frozen"):
        print(
            "[REFERENCIAS] Campaña congelada: modelo, estéreo y fondo se leen "
            "desde el propio trabajo.",
            flush=True,
        )
        if a.mode == "reconstruir":
            if refs.get("platform") is None:
                raise FileNotFoundError("La campaña no contiene calibración de plataforma congelada.")
            a.platform_calibration = str(refs["platform"])
            print(
                "[REFERENCIAS] Calibración de plataforma: copia congelada del trabajo.",
                flush=True,
            )
        else:
            # La calibración producida por una campaña de calibración se guarda
            # primero dentro del trabajo. La GUI puede promoverla después al sistema.
            a.platform_calibration_output_dir = str(
                workspace / "resultado_calibracion_plataforma"
            )
    else:
        print(
            "[REFERENCIAS] Campaña heredada sin snapshot: se usarán los recursos "
            "globales indicados por línea de comandos.",
            flush=True,
        )

    return run_pipeline(a, base, workspace, obj, py, stereo, background, model)


def _entrada_con_diagnostico():
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    raise SystemExit(main())


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
