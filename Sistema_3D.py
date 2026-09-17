#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sistema integrado de captura estéreo y reconstrucción tridimensional.

La interfaz administra recursos del montaje, trabajos independientes,
adquisición de imágenes y ejecución o reanudación del pipeline. Las cámaras,
Arduino y los procesos de cálculo conservan sus controles de estado y sus
diagnósticos. La bitácora registra capturas y cierres de vuelta.

Modelo angular del montaje
--------------------------
2055 pasos calibrados completan una vuelta. Se capturan 25 pares por sesión,
con etiquetas nominales de 0.0° a 345.6° cada 14.4°. La secuencia de movimiento
es [82, 82, 83, 82, 82] repetida cinco veces; el ángulo físico se deriva de
los pasos acumulados. La última transición se ejecuta con CLOSE, sin captura,
y completa la vuelta antes de iniciar otra sesión.

RESET define la posición actual como origen lógico; no mueve el motor ni
busca una referencia física. El protocolo serie debe corresponder al firmware
ROT2055_V7_2. El flujo no aplica calibración visual ni corrección automática
del movimiento.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import serial
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from openpyxl import Workbook, load_workbook

PRODUCT_VERSION = "3.1.0"
APP_TITLE = "Sistema Integrado de Construcción 3D"
IMAGE_EXT = "png"

CAPTURE_PLAN = []  # El plan real se crea dinámicamente desde la interfaz.

MODEL_FILENAME = "crestereo_init_iter10_480x640.onnx"
PLATFORM_CALIBRATION_FILENAME = "calibracion_plataforma.json"
JOB_METADATA_FILENAME = "trabajo_sistema_3d.json"
VALID_JOB_MODES = {"calibrar-plataforma", "reconstruir"}

# ---------------------------------------------------------------------
# Modelo angular de la plataforma
# ---------------------------------------------------------------------

STEPS_PER_REVOLUTION = 2055
MOVES_PER_REVOLUTION = 25
CAPTURE_VIEWS = 25
NOMINAL_STEP_DEG = 360.0 / MOVES_PER_REVOLUTION

STEP_SEQUENCE = (
    82,
    82,
    83,
    82,
    82,
    82,
    82,
    83,
    82,
    82,
    82,
    82,
    83,
    82,
    82,
    82,
    82,
    83,
    82,
    82,
    82,
    82,
    83,
    82,
    82,
)

assert len(STEP_SEQUENCE) == MOVES_PER_REVOLUTION
assert sum(STEP_SEQUENCE) == STEPS_PER_REVOLUTION

# ---------------------------------------------------------------------
# Adquisición
# ---------------------------------------------------------------------

RESOLUTION_W = 1920
RESOLUTION_H = 1080

SETTLE_TIME_S = 1.0

BACKGROUND_SAMPLES = 20
BACKGROUND_SAMPLE_INTERVAL_S = 0.08

SERIAL_BAUD_DEFAULT = 115200
SERIAL_COMMAND_TIMEOUT_S = 20.0
FRESH_FRAME_TIMEOUT_S = 3.0

# Después del movimiento final que completa la vuelta se exige un tiempo
# adicional y, posteriormente, un par nuevo antes de iniciar la sesión
# siguiente. Así S02/S03 nunca reutilizan un frame anterior al retorno.
SESSION_RETURN_SETTLE_S = 1.0
SESSION_START_GUARD_S = 0.35


def cumulative_steps_before_pose(pose_index: int) -> int:
    """Pasos acumulados desde pose 0 hasta la pose indicada."""
    if pose_index < 0 or pose_index >= CAPTURE_VIEWS:
        raise ValueError("pose_index fuera de rango.")
    return int(sum(STEP_SEQUENCE[:pose_index]))


def exact_angle_from_steps(pose_index: int) -> float:
    """
    Ángulo físico inferido desde los pasos acumulados y 2055 pasos/vuelta.

    El ángulo nominal sigue siendo pose_index * 14.4°. Esta segunda magnitud
    describe la pequeña cuantización inevitable por usar pasos enteros.
    """
    steps = cumulative_steps_before_pose(pose_index)
    return float(steps * 360.0 / STEPS_PER_REVOLUTION)


def build_job_metadata(mode: str, object_name: str) -> dict:
    """Contrato persistente para reabrir un trabajo sin inferir por su nombre."""
    mode = str(mode).strip().lower()
    object_name = str(object_name).strip().lower()
    if mode not in VALID_JOB_MODES:
        raise ValueError(f"Modo de trabajo inválido: {mode}")
    if not object_name:
        raise ValueError("El objeto del trabajo no puede estar vacío.")
    return {
        "schema_version": 1,
        "product_version": PRODUCT_VERSION,
        "mode": mode,
        "object": object_name,
        "expected_sessions": 3,
        "poses_per_session": CAPTURE_VIEWS,
        "steps_per_revolution": STEPS_PER_REVOLUTION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def write_job_metadata(workspace: Path, mode: str, object_name: str) -> Path:
    """Guarda el contrato de trabajo de forma atómica."""
    path = Path(workspace) / JOB_METADATA_FILENAME
    payload = build_job_metadata(mode, object_name)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def resolve_existing_job_mode(
    workspace: Path,
    object_name: str,
) -> Tuple[str, str]:
    """Recupera el modo desde metadatos; admite trabajos heredados.

    El nombre de la carpeta queda como último fallback únicamente para campañas
    anteriores que tampoco conservan ``documentacion/protocolo_captura.md``.
    """
    workspace = Path(workspace)
    object_name = str(object_name).strip().lower()
    metadata_path = workspace / JOB_METADATA_FILENAME

    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(
                f"Metadatos de trabajo inválidos: {metadata_path.name}: {exc}"
            ) from exc

        mode = str(metadata.get("mode", "")).strip().lower()
        metadata_object = str(metadata.get("object", "")).strip().lower()
        if mode not in VALID_JOB_MODES:
            raise ValueError(f"Modo inválido en {metadata_path.name}: {mode!r}")
        if metadata_object != object_name:
            raise ValueError(
                "El objeto declarado en los metadatos no coincide con la "
                f"carpeta encontrada: {metadata_object!r} != {object_name!r}."
            )
        if int(metadata.get("expected_sessions", 3)) != 3:
            raise ValueError("El trabajo no declara exactamente 3 sesiones.")
        if int(metadata.get("poses_per_session", CAPTURE_VIEWS)) != CAPTURE_VIEWS:
            raise ValueError("El trabajo no corresponde a 25 poses por sesión.")
        if int(metadata.get("steps_per_revolution", STEPS_PER_REVOLUTION)) != STEPS_PER_REVOLUTION:
            raise ValueError("El trabajo no corresponde a 2055 pasos por vuelta.")
        return mode, "metadata"

    protocol = workspace / "documentacion" / "protocolo_captura.md"
    if protocol.is_file():
        text = protocol.read_text(encoding="utf-8", errors="replace")
        match = re.search(
            r"(?mi)^\s*-\s*Modo:\s*(calibrar-plataforma|reconstruir)\s*$",
            text,
        )
        if match:
            return match.group(1).lower(), "legacy_protocol"

    mode = (
        "calibrar-plataforma"
        if "CALIBRACION_PLATAFORMA" in workspace.name.upper()
        else "reconstruir"
    )
    return mode, "legacy_name"


class OperationCancelled(RuntimeError):
    """La operación recibió una solicitud de parada del usuario."""


def terminate_process_tree(proc):
    """Finaliza el grupo exclusivo del coordinador y sus procesos de cálculo."""
    if os.name == "nt":
        if proc.poll() is None:
            result = subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode and proc.poll() is None:
                raise RuntimeError("Windows no pudo detener el árbol de procesamiento.")
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        # Un hijo puede seguir vivo aunque el coordinador ya haya terminado.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=5)


def capture_readiness(workspace):
    """Verifica los pares y el cierre físico antes de habilitar procesamiento."""
    if workspace is None:
        return False, "Selecciona un trabajo."
    try:
        for session in ("S01", "S02", "S03"):
            directory = Path(workspace) / "capturas" / session
            pairs = []
            for side, suffix in (("izquierda", "L"), ("derecha", "R")):
                files = list((directory / side).glob(f"*_{suffix}.png"))
                if len(files) != CAPTURE_VIEWS or any(p.stat().st_size == 0 for p in files):
                    return (
                        False,
                        f"{session}: faltan imágenes válidas en {side}; se requieren {CAPTURE_VIEWS}.",
                    )
                keys = {p.stem[:-2] for p in files}
                poses = set()
                for key in keys:
                    match = re.search(rf"_{session}_V(\d{{3}})_A(\d{{4}})$", key)
                    if not match:
                        return False, f"{session}: nombre de captura no reconocido."
                    view, angle = map(int, match.groups())
                    if not 1 <= view <= CAPTURE_VIEWS or angle != round(
                        (view - 1) * NOMINAL_STEP_DEG * 10
                    ):
                        return False, f"{session}: vista o ángulo nominal incoherente."
                    poses.add(view)
                if len(poses) != CAPTURE_VIEWS:
                    return False, f"{session}: hay vistas duplicadas."
                pairs.append(keys)
            if pairs[0] != pairs[1]:
                return (
                    False,
                    f"{session}: las imágenes izquierda y derecha no forman los mismos pares.",
                )
            record = json.loads(
                (directory / "control_origen" / "movimiento_sesion.json").read_text(
                    encoding="utf-8"
                )
            )
            before = record.get("status_before_close", {})
            after = record.get("status_after_close", {})
            close = record.get("close_command_result", {})
            if not (
                record.get("captured_views") == CAPTURE_VIEWS
                and record.get("steps_per_revolution") == STEPS_PER_REVOLUTION
                and record.get("final_return_move_executed") is True
                and record.get("close_command_used") is True
                and before.get("phase") == 24
                and before.get("cumulative") == 1973
                and close.get("total") == STEPS_PER_REVOLUTION
                and close.get("delta") == STEP_SEQUENCE[-1]
                and close.get("phase") == 0
                and close.get("cumulative") == 0
                and after.get("phase") == 0
                and after.get("cumulative") == 0
            ):
                return False, f"{session}: falta confirmar el cierre completo de la vuelta."
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return False, f"No se pudo validar la captura: {exc}"
    return True, "3 sesiones verificadas: 75 pares y sus cierres de vuelta."


class PipelineProgress:
    """Conserva la actividad específica entre mensajes y calcula tiempos en la GUI."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = self.step_started = self.activity_started = clock()
        self.stopped = None
        self.stage = "Preparando procesamiento"
        self.activity = "Comprobando entradas y checkpoints"
        self.specific = False

    @staticmethod
    def duration(seconds):
        seconds = max(0, int(seconds))
        hours, rest = divmod(seconds, 3600)
        minutes, seconds = divmod(rest, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def feed(self, line):
        now = self.clock()
        if line.startswith("[ETAPA] "):
            self.stage = line[len("[ETAPA] ") :].strip()
            self.step_started = self.activity_started = now
            self.activity = "Preparando entradas del paso"
            self.specific = False
            return
        if line.startswith("[RESUME]") and "reutilizado" in line:
            if not self.specific:
                self.activity = line.removeprefix("[RESUME]").strip()
            return
        match = re.match(r"\[(PROGRESO|INICIO|EN CURSO|CPU)\]\s*(.*)", line)
        if not match:
            return
        tag, detail = match.groups()
        if detail.startswith(("Se está ejecutando:", "Completado:")):
            return
        parts = [part.strip() for part in detail.split("|")]
        if parts and (
            re.fullmatch(r"Paso\s+[\w.]+", parts[0], re.IGNORECASE) or parts[0].endswith(".py")
        ):
            parts.pop(0)
        parts = [
            part
            for part in parts
            if not re.match(r"Tiempo\s+(actividad|total)|^\d+(?:\.\d+)?\s*s$", part, re.IGNORECASE)
        ]
        detail = " | ".join(parts).strip()
        if not detail or detail == self.stage.split(" | ")[-1]:
            return
        # Los latidos de una función exterior no deben borrar el detalle interno.
        if tag == "EN CURSO" and self.specific:
            return
        if re.fullmatch(r"Procesando paso\s+[\w.]+[.…]*", detail, re.IGNORECASE):
            return
        if detail != self.activity:
            self.activity = detail
            self.activity_started = now
        self.specific = True

    def render(self):
        now = self.stopped if self.stopped is not None else self.clock()
        status = f"{self.stage} · Paso: {self.duration(now-self.step_started)} · Total: {self.duration(now-self.started)}"
        detail = f"{self.activity} · Actividad: {self.duration(now-self.activity_started)}"
        return status, detail


class CaptureApp:
    """Interfaz de captura estéreo, gestión de trabajos y ejecución del pipeline.

    La vista previa actualiza el par de imágenes bajo frame_lock. Los trabajadores
    comunican su progreso mediante status_queue, atendida periódicamente por Tk.
    Cada trabajo conserva su modo, capturas, reconstrucción y resultado final.
    """

    def __init__(self, root: tk.Tk):
        """Inicializa recursos, sincronización, estado del trabajo e interfaz principal."""
        self.root = root
        self.root.title(APP_TITLE)
        self._configure_window()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # La carpeta que contiene ESTE archivo es la raíz instalada del producto.
        # El usuario no necesita introducir rutas internas.
        self.install_root = Path(__file__).resolve().parent
        self.processing_dir = self.install_root / "procesamiento"
        self.models_dir = self.install_root / "modelos"
        self.system_dir = self.install_root / "sistema"
        self.stereo_dir = self.system_dir / "calibracion_estereo"
        self.platform_dir = self.system_dir / "calibracion_plataforma"
        self.background_dir = self.system_dir / "fondo_vacio"
        self.jobs_dir = self.install_root / "trabajos"
        self.records_dir = self.install_root / "registros"

        self.capture_plan = []
        self.job_mode = None
        self.current_job_name = None
        self.current_object = None
        self.pipeline_thread = None
        self.pipeline_running = False
        self._operation = None
        self._worker_thread = None
        self._stop_thread = None
        self._terminal_event = None
        self._finished_event = False
        self._closing = False
        self._destroyed = False
        self._pipeline_proc = None
        self._pipeline_lock = threading.Lock()
        self._serial_lock = threading.RLock()
        self._status_job = None

        self._initialize_product_structure()

        self.ser = None
        self.cap_left = None
        self.cap_right = None

        self.preview_job = None
        self.preview_running = False

        self.capture_thread = None
        self.background_capture_thread = None

        self.frame_lock = threading.Lock()
        self.latest_left = None
        self.latest_right = None
        self.latest_pair_time = 0.0

        self.stop_event = threading.Event()
        self.object_change_event = threading.Event()
        self.status_queue = queue.Queue()

        self.accepted = False
        self.background_captured = False
        self.capture_started = False
        self.capture_completed = False
        self.waiting_object_change = False

        self.root_dir = None
        self.workbook_path = None

        # Momento a partir del cual debe haberse adquirido el primer frame de
        # una nueva sesión. Se actualiza después del movimiento de retorno.
        self.last_session_return_time = None

        self._build_vars()
        self._build_ui()

        self._status_job = self.root.after(100, self._process_status_queue)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _configure_window(self):
        """Abre maximizada en Windows y conserva una geometría restaurable centrada."""
        self._ui_scale = max(1.0, float(self.root.tk.call("tk", "scaling")) * 72 / 96)
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        width, height = min(1280, screen_w - 60), min(820, screen_h - 100)
        self.root.geometry(
            f"{width}x{height}+{max(0, (screen_w-width)//2)}+{max(0, (screen_h-height)//2-20)}"
        )
        self.root.minsize(
            min(round(900 * self._ui_scale), width), min(round(580 * self._ui_scale), height)
        )
        if os.name == "nt":
            self.root.state("zoomed")

    def _layout_cameras(self, event):
        """Elige columnas o filas según el espacio disponible; mantiene ambas vistas."""
        horizontal = event.width / max(1, event.height) >= 1.7
        if horizontal == self._camera_layout:
            return
        self._camera_layout = horizontal
        for i in (0, 1):
            self._preview_frame.columnconfigure(
                i, weight=1 if horizontal or i == 0 else 0, uniform="camera" if horizontal else ""
            )
            self._preview_frame.rowconfigure(
                i,
                weight=1 if not horizontal or i == 0 else 0,
                uniform="camera" if not horizontal else "",
            )
        for i, (card, canvas) in enumerate(self._camera_cards):
            card.grid(
                row=0 if horizontal else i,
                column=i if horizontal else 0,
                sticky="nsew",
                padx=(0, 8) if horizontal and i == 0 else 0,
                pady=(0, 8) if not horizontal and i == 0 else 0,
            )

    @staticmethod
    def _thread_alive(thread):
        return thread is not None and thread.is_alive()

    def _busy(self):
        return bool(
            self._operation
            or self._thread_alive(self._worker_thread)
            or self._thread_alive(self._stop_thread)
            or (self._pipeline_proc is not None and self._pipeline_proc.poll() is None)
        )

    def _update_controls(self):
        """Deriva todos los botones de un único estado de operación."""
        if self._destroyed:
            return
        idle = not self._busy() and not self._closing
        for widget in self._idle_widgets + self._hardware_entries:
            widget.configure(state="normal" if idle else "disabled")
        connected = (
            self.cap_left is not None and self.cap_right is not None and self.ser is not None
        )
        self.btn_accept.configure(state="normal" if idle and connected else "disabled")
        self.btn_background.configure(state="normal" if idle and self.accepted else "disabled")
        can_capture = (
            idle
            and self.accepted
            and self.root_dir is not None
            and self.background_captured
            and not self.capture_completed
        )
        self.btn_start.configure(
            state="normal" if can_capture else "disabled",
            text=(
                "Reiniciar captura de 3 sesiones"
                if self.capture_started and not self.capture_completed
                else "Capturar 3 sesiones"
            ),
        )
        for button in (self.btn_process, self.btn_resume):
            button.configure(state="normal" if idle and self.capture_completed else "disabled")
        self.btn_continue.configure(
            state=(
                "normal"
                if self.waiting_object_change and not self.stop_event.is_set()
                else "disabled"
            )
        )
        if self.waiting_object_change:
            self.btn_continue.grid(row=8, column=0, columnspan=2, sticky="ew", pady=3)
        else:
            self.btn_continue.grid_remove()
        self.btn_stop.configure(
            state=(
                "normal"
                if not self._closing
                and not self._thread_alive(self._stop_thread)
                and (self._operation or connected)
                else "disabled"
            )
        )
        self.btn_close.configure(state="disabled" if self._closing else "normal")
        active = self._busy()
        if active != getattr(self, "_bar_active", False):
            self._activity_bar.start(15) if active else self._activity_bar.stop()
            self._bar_active = active

    def _check_cancelled(self):
        if self.stop_event.is_set():
            raise OperationCancelled("Operación detenida por el usuario.")

    def _wait_cancelable(self, seconds):
        if self.stop_event.wait(seconds):
            self._check_cancelled()

    def _begin_operation(self, kind, target, *args):
        if self._busy() or self._closing:
            return
        self.stop_event.clear()
        self._operation = kind
        self._terminal_event = None
        self._finished_event = False
        self._stop_warning = None
        self.pipeline_running = kind == "pipeline"
        if self.pipeline_running:
            self._pipeline_progress = PipelineProgress()
        self.status_var.set(
            {
                "pipeline": "Iniciando procesamiento…",
                "capture": "Iniciando captura…",
                "background": "Capturando fondo vacío…",
            }[kind]
        )
        self.progress_var.set("Preparando recursos…")
        self._worker_thread = threading.Thread(
            target=self._operation_worker, args=(kind, target, args), daemon=True
        )
        if kind == "pipeline":
            self.pipeline_thread = self._worker_thread
        elif kind == "capture":
            self.capture_thread = self._worker_thread
        else:
            self.background_capture_thread = self._worker_thread
        try:
            self._worker_thread.start()
        except RuntimeError:
            self._operation = None
            self.pipeline_running = False
            raise
        self._update_controls()

    def _operation_worker(self, kind, target, args):
        try:
            self._check_cancelled()
            target(*args)
        except OperationCancelled as exc:
            self.status_queue.put((f"{kind}_stopped", str(exc)))
        except Exception as exc:
            self.status_queue.put((f"{kind}_error", str(exc)))
        finally:
            try:
                with self._pipeline_lock:
                    proc = self._pipeline_proc
                    if proc is not None:
                        if proc.poll() is None:
                            terminate_process_tree(proc)
                        if proc.stdout is not None:
                            proc.stdout.close()
                        self._pipeline_proc = None
                if kind == "capture" and not self.stop_event.is_set():
                    self._send_motor_stop()
            except Exception as exc:
                self.status_queue.put(("stop_error", str(exc)))
            finally:
                self.status_queue.put(("operation_finished", ""))

    def _send_motor_stop(self):
        if self.ser is not None and self.ser.is_open:
            self.serial_command("STOP", "STOPPED", timeout=5.0, ignore_cancel=True)

    def _stop_resources(self):
        try:
            with self._pipeline_lock:
                if self._pipeline_proc is not None:
                    terminate_process_tree(self._pipeline_proc)
            self._send_motor_stop()
        except Exception as exc:
            self.status_queue.put(("stop_error", f"No se pudo confirmar la parada: {exc}"))
        finally:
            self.status_queue.put(("resources_stopped", ""))

    def _finish_operation(self):
        kind, message = self._terminal_event or (f"{self._operation}_stopped", "")
        operation = self._operation
        self._operation = None
        self.pipeline_running = False
        self._finished_event = False
        self.waiting_object_change = False
        if operation == "pipeline":
            self._pipeline_progress.stopped = time.monotonic()
        if operation == "capture":
            self.capture_completed, reason = capture_readiness(self.root_dir)
            if kind == "capture_complete" and not self.capture_completed:
                kind, message = "capture_error", reason
        self._refresh_system_status()
        if self._closing:
            return
        if kind.endswith("_error"):
            self.status_var.set("La operación terminó con un error.")
            self.progress_var.set(
                str(message).splitlines()[0]
                if str(message).strip()
                else "Consulta el diagnóstico de la operación."
            )
            messagebox.showerror("Error de operación", str(message), parent=self.root)
        elif kind.endswith("_stopped"):
            self.status_var.set("Operación detenida.")
            self.progress_var.set(
                "Puedes reanudar el procesamiento desde los pasos verificados."
                if operation == "pipeline"
                else (
                    "La captura interrumpida debe reiniciarse con el objeto en su posición inicial."
                    if operation == "capture"
                    else "Puedes volver a capturar el fondo vacío."
                )
            )
        elif kind == "pipeline_complete":
            elapsed = self._pipeline_progress.duration(
                self._pipeline_progress.stopped - self._pipeline_progress.started
            )
            if self.job_mode == "calibrar-plataforma":
                self.status_var.set("Calibración de plataforma terminada y guardada.")
                self.progress_var.set(
                    f"Tiempo total: {elapsed} · Ya puedes crear un trabajo de objeto."
                )
                messagebox.showinfo(
                    "Calibración terminada",
                    f"Calibración guardada:\n{message.get('platform_calibration')}",
                    parent=self.root,
                )
            else:
                quality = message.get("quality") or "desconocida"
                self.status_var.set(f"Reconstrucción y exportación terminadas · Calidad: {quality}")
                self.progress_var.set(
                    f"Tiempo total: {elapsed} · {message.get('result_dir') or self.root_dir}"
                )
                messagebox.showinfo(
                    "Reconstrucción terminada",
                    f"Calidad de validación: {quality}\n\nOBJ para Blender:\n{message.get('obj')}\n\nOBJ científico en mm:\n{message.get('obj_mm')}\n\nPLY original:\n{message.get('mesh')}\n\nResumen:\n{message.get('summary')}",
                    parent=self.root,
                )
        elif kind == "capture_complete":
            self.status_var.set("Captura completada y verificada.")
            self.progress_var.set("75 pares guardados. Ya puedes procesar el trabajo.")
        elif kind == "background_complete":
            self.status_var.set("Fondo vacío actualizado.")
            self.progress_var.set("Listo para capturar un trabajo.")
        if getattr(self, "_stop_warning", None):
            self.progress_var.set(self._stop_warning)

    def _finalize_close(self):
        self._destroyed = True
        self._activity_bar.stop()
        if self._status_job is not None:
            self.root.after_cancel(self._status_job)
        self.close_hardware()
        self.root.destroy()

    def _build_vars(self):
        """Crea las variables Tk que enlazan configuración y estado con la interfaz."""
        self.port_var = tk.StringVar(value="COM6")
        self.baud_var = tk.StringVar(value=str(SERIAL_BAUD_DEFAULT))
        self.cam_left_var = tk.StringVar(value="0")
        self.cam_right_var = tk.StringVar(value="1")

        self.root_dir_var = tk.StringVar(value="Ningún trabajo seleccionado")
        self.job_mode_var = tk.StringVar(value="—")
        self.plan_var = tk.StringVar(value="Sin campaña activa")

        self.model_status_var = tk.StringVar()
        self.stereo_status_var = tk.StringVar()
        self.platform_status_var = tk.StringVar()
        self.background_status_var = tk.StringVar()

        self.status_var = tk.StringVar(
            value="Sistema iniciado. Revisa el estado y abre cámaras + Arduino."
        )
        self.progress_var = tk.StringVar(value="Esperando.")

        self._refresh_system_status()

    def _plan_text(self):
        """Resume el objeto, las sesiones y las vistas de la campaña activa."""
        if not self.capture_plan:
            return "Sin campaña activa"
        return " | ".join(
            f"{item['name']} — {item['sessions']} sesiones × {CAPTURE_VIEWS} vistas"
            for item in self.capture_plan
        )

    def _build_ui(self):
        """Distribuye un panel fijo de controles y las vistas de las cámaras."""
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#edf2f7")
        style.configure("TLabel", background="#edf2f7", foreground="#233247", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 9), padding=(8, 6))
        style.configure("Card.TLabelframe", background="#edf2f7", padding=10)
        style.configure(
            "Card.TLabelframe.Label",
            background="#edf2f7",
            foreground="#233247",
            font=("Segoe UI", 10, "bold"),
        )
        style.configure("Header.TLabel", font=("Segoe UI", 9, "bold"))
        style.configure("Title.TLabel", font=("Segoe UI", 15, "bold"))
        style.configure("Stop.TButton", foreground="#9b2626")
        self.root.configure(background="#edf2f7")

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)
        ttk.Label(main, text=APP_TITLE, style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )

        sidebar = ttk.Frame(main, width=round(350 * self._ui_scale))
        sidebar.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(0, weight=1)

        # El panel lateral cabe completo en la interfaz maximizada, por lo que
        # no necesita Canvas ni barra de desplazamiento. Mantenerlo como un
        # Frame normal evita una interacción innecesaria y deja los controles
        # siempre visibles.
        controls = ttk.Frame(sidebar)
        controls.grid(row=0, column=0, sticky="nsew")
        self._idle_widgets = []
        self._hardware_entries = []

        system = ttk.LabelFrame(controls, text="Estado del sistema", style="Card.TLabelframe")
        system.pack(fill="x", pady=(0, 10))
        system.columnconfigure(1, weight=1)
        for row, (label, variable) in enumerate(
            (
                ("Modelo", self.model_status_var),
                ("Estéreo", self.stereo_status_var),
                ("Plataforma", self.platform_status_var),
                ("Fondo", self.background_status_var),
            )
        ):
            ttk.Label(system, text=label, style="Header.TLabel").grid(
                row=row, column=0, sticky="nw", padx=(0, 8), pady=4
            )
            ttk.Label(system, textvariable=variable, wraplength=round(210 * self._ui_scale)).grid(
                row=row, column=1, sticky="w", pady=4
            )
        for row, (label, command) in enumerate(
            (
                ("Actualizar estado", self._refresh_system_status),
                ("Importar calibración estéreo", self.import_stereo_calibration),
            ),
            start=4,
        ):
            button = ttk.Button(system, text=label, command=command)
            button.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(5, 0))
            self._idle_widgets.append(button)

        hardware = ttk.LabelFrame(controls, text="Cámaras y plataforma", style="Card.TLabelframe")
        hardware.pack(fill="x", pady=(0, 10))
        hardware.columnconfigure(1, weight=1)
        hardware.columnconfigure(3, weight=1)
        for i, (label, var) in enumerate(
            (
                ("Puerto", self.port_var),
                ("Baudios", self.baud_var),
                ("Cám. izq.", self.cam_left_var),
                ("Cám. der.", self.cam_right_var),
            )
        ):
            row, col = divmod(i, 2)
            ttk.Label(hardware, text=label).grid(
                row=row, column=col * 2, sticky="w", pady=4, padx=(0, 5)
            )
            entry = ttk.Entry(hardware, textvariable=var, width=8)
            entry.grid(row=row, column=col * 2 + 1, sticky="ew", padx=(0, 5))
            self._hardware_entries.append(entry)
        self.btn_open_hardware = ttk.Button(
            hardware, text="Conectar cámaras y Arduino", command=self.open_hardware
        )
        self.btn_open_hardware.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(6, 3))
        self.btn_accept = ttk.Button(
            hardware, text="Confirmar cámaras listas", command=self.accept_setup
        )
        self.btn_accept.grid(row=3, column=0, columnspan=4, sticky="ew", pady=3)
        self.btn_background = ttk.Button(
            hardware, text="Capturar fondo vacío", command=self.capture_empty_background
        )
        self.btn_background.grid(row=4, column=0, columnspan=4, sticky="ew", pady=3)
        ttk.Label(
            hardware,
            text=f"{RESOLUTION_W} × {RESOLUTION_H} · {CAPTURE_VIEWS} vistas por sesión",
            foreground="#64748b",
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(5, 0))
        self._idle_widgets.append(self.btn_open_hardware)

        workflow = ttk.LabelFrame(controls, text="Trabajo actual", style="Card.TLabelframe")
        workflow.pack(fill="x", pady=(0, 4))
        workflow.columnconfigure(0, weight=1)
        workflow.columnconfigure(1, weight=1)
        ttk.Label(workflow, textvariable=self.job_mode_var, style="Header.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        ttk.Entry(workflow, textvariable=self.root_dir_var, state="readonly", width=20).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=3
        )
        ttk.Label(
            workflow, textvariable=self.plan_var, wraplength=round(280 * self._ui_scale)
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 8))
        for i, (label, command) in enumerate(
            (
                ("Calibrar plataforma", self.new_platform_calibration_job),
                ("Nuevo objeto", self.new_object_job),
                ("Reabrir trabajo", self.reopen_existing_job),
                ("Abrir carpeta", self.open_current_job_folder),
            )
        ):
            button = ttk.Button(workflow, text=label, command=command)
            button.grid(
                row=3 + i // 2,
                column=i % 2,
                sticky="ew",
                padx=(0, 4) if i % 2 == 0 else (4, 0),
                pady=3,
            )
            self._idle_widgets.append(button)
        self.btn_start = ttk.Button(
            workflow, text="Capturar 3 sesiones", command=self.start_capture
        )
        self.btn_process = ttk.Button(
            workflow, text="Procesar desde cero", command=self.process_current_job
        )
        self.btn_resume = ttk.Button(
            workflow, text="Reanudar procesamiento", command=self.resume_current_job
        )
        for row, button in enumerate((self.btn_start, self.btn_process, self.btn_resume), start=5):
            button.grid(row=row, column=0, columnspan=2, sticky="ew", pady=3)
        self.btn_continue = ttk.Button(
            workflow, text="Continuar", command=self.continue_after_object_change
        )

        view = ttk.Frame(main)
        view.grid(row=1, column=1, sticky="nsew")
        view.rowconfigure(1, weight=1)
        view.columnconfigure(0, weight=1)
        ttk.Label(view, text="Vista previa estéreo", style="Header.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self._preview_frame = ttk.Frame(view)
        self._preview_frame.grid(row=1, column=0, sticky="nsew")
        self._camera_cards = []
        for title in ("Cámara izquierda", "Cámara derecha"):
            card = ttk.LabelFrame(self._preview_frame, text=title, style="Card.TLabelframe")
            canvas = tk.Canvas(card, width=1, height=1, bg="#101b29", highlightthickness=0, bd=0)
            canvas.pack(fill="both", expand=True)
            self._camera_cards.append((card, canvas))
        self.left_canvas, self.right_canvas = [item[1] for item in self._camera_cards]
        self._preview_items = {self.left_canvas: None, self.right_canvas: None}
        self._preview_images = {self.left_canvas: None, self.right_canvas: None}
        self._camera_layout = None
        self._preview_frame.bind("<Configure>", self._layout_cameras)

        footer = ttk.Frame(main, padding=(0, 10, 0, 0))
        footer.grid(row=2, column=0, columnspan=2, sticky="ew")
        footer.columnconfigure(0, weight=1)
        self._status_label = ttk.Label(
            footer, textvariable=self.status_var, foreground="#195fa5", wraplength=650
        )
        self._status_label.grid(row=0, column=0, sticky="w")
        self._progress_label = ttk.Label(
            footer, textvariable=self.progress_var, foreground="#64748b", wraplength=650
        )
        self._progress_label.grid(row=1, column=0, sticky="w", pady=(3, 0))
        footer.bind(
            "<Configure>",
            lambda e: [
                label.configure(wraplength=max(200, e.width - round(220 * self._ui_scale)))
                for label in (self._status_label, self._progress_label)
            ],
        )
        self.btn_stop = ttk.Button(
            footer, text="Parar", command=self.stop_capture, style="Stop.TButton"
        )
        self.btn_stop.grid(row=0, column=1, rowspan=2, padx=(12, 6))
        self.btn_close = ttk.Button(footer, text="Cerrar", command=self.on_close)
        self.btn_close.grid(row=0, column=2, rowspan=2)
        self._activity_bar = ttk.Progressbar(footer, mode="indeterminate")
        self._activity_bar.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self._update_controls()

    # ------------------------------------------------------------------
    # Estructura del producto / trabajos
    # ------------------------------------------------------------------

    def _initialize_product_structure(self):
        """Prepara los directorios del sistema y guarda su manifiesto de estructura."""
        for folder in (
            self.models_dir,
            self.stereo_dir,
            self.platform_dir,
            self.background_dir,
            self.jobs_dir,
            self.records_dir,
            self.processing_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)

        config = self.system_dir / "estructura_sistema.json"
        payload = {
            "schema_version": 2,
            "model_filename": MODEL_FILENAME,
            "platform_calibration_filename": PLATFORM_CALIBRATION_FILENAME,
            "steps_per_revolution": STEPS_PER_REVOLUTION,
            "poses_per_session": CAPTURE_VIEWS,
            "sessions_per_job": 3,
            "directorios_persistentes": {
                "modelos": "modelos",
                "calibracion_estereo": "sistema/calibracion_estereo",
                "calibracion_plataforma": "sistema/calibracion_plataforma",
                "fondo_vacio": "sistema/fondo_vacio",
            },
            "directorio_trabajos": "trabajos",
            "estructura_trabajo": {
                "capturas": "capturas/S01..S03/{izquierda,derecha}",
                "reconstruccion": "reconstruccion",
                "resultado_final": "resultado_final",
                "documentacion": "documentacion",
            },
            "principle": (
                "Las calibraciones y el modelo pertenecen al sistema. "
                "Cada trabajo contiene únicamente sus capturas y resultados."
            ),
        }
        config.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _model_path(self):
        """Devuelve la ruta del modelo ONNX dentro de la instalación."""
        return self.models_dir / MODEL_FILENAME

    def _platform_calibration_path(self):
        """Devuelve la ruta canónica de la calibración de plataforma activa."""
        return self.platform_dir / PLATFORM_CALIBRATION_FILENAME

    def _read_stereo_report(self):
        """Lee el informe estéreo o devuelve None si falta o no puede interpretarse."""
        report_path = self.stereo_dir / "calibration_report.json"
        if not report_path.is_file():
            return None
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _stereo_ready(self):
        """Comprueba los archivos estéreo y el informe, con compatibilidad para datos heredados."""
        required_ok = (self.stereo_dir / "stereo_initial.yaml").is_file() and (
            self.stereo_dir / "rectification_maps.npz"
        ).is_file()
        if not required_ok:
            return False
        report = self._read_stereo_report()
        if report is None:
            # Compatibilidad con calibraciones heredadas: los dos archivos
            # científicos siguen siendo suficientes, aunque no habrá métricas
            # visibles en la GUI.
            return True
        quality = str(report.get("quality", "")).strip().lower()
        return quality == "accepted"

    def _background_ready(self):
        """Comprueba que existan las dos imágenes del fondo vacío."""
        return (self.background_dir / "background_left.png").is_file() and (
            self.background_dir / "background_right.png"
        ).is_file()

    def _refresh_system_status(self):
        """Actualiza los indicadores de recursos y la disponibilidad de acciones."""
        model_ok = self._model_path().is_file()
        stereo_ok = self._stereo_ready()
        platform_ok = self._platform_calibration_path().is_file()
        background_ok = self._background_ready()

        self.model_status_var.set(
            "✓ listo" if model_ok else f"✗ copiar {MODEL_FILENAME} en modelos\\"
        )
        stereo_report = self._read_stereo_report()
        if stereo_ok and stereo_report:
            try:
                srms = float(stereo_report.get("stereo_rms_px"))
                baseline = float(stereo_report.get("baseline_mm"))
                epi = stereo_report.get("epipolar", {}) or {}
                emed = float(epi.get("median_abs_dy_px"))
                self.stereo_status_var.set(
                    f"✓ accepted | RMS {srms:.3f}px | epi {emed:.3f}px | B {baseline:.2f}mm"
                )
            except Exception:
                self.stereo_status_var.set("✓ lista | reporte disponible")
        elif stereo_ok:
            self.stereo_status_var.set("✓ lista | calibración heredada sin reporte")
        else:
            self.stereo_status_var.set("✗ no instalada / no aceptada")
        self.platform_status_var.set("✓ lista" if platform_ok else "✗ aún no calibrada")
        self.background_status_var.set("✓ listo" if background_ok else "✗ falta capturarlo")

        self.background_captured = background_ok

        if hasattr(self, "btn_start"):
            self._update_controls()

        return {
            "model": model_ok,
            "stereo": stereo_ok,
            "platform": platform_ok,
            "background": background_ok,
        }

    def import_stereo_calibration(self):
        """Importa una calibración seleccionada y actualiza los recursos activos.

        Valida el contenido antes de copiarlo. Una nueva geometría estéreo invalida la
        calibración de plataforma anterior, que se archiva en registros.
        """
        if self._busy() or self._closing:
            return
        source = filedialog.askdirectory(title="Selecciona la carpeta de calibración estéreo")
        if not source:
            return

        source = Path(source).expanduser().resolve()
        required = (
            source / "stereo_initial.yaml",
            source / "rectification_maps.npz",
        )
        missing = [str(p.name) for p in required if not p.is_file()]
        if missing:
            messagebox.showerror(
                "Calibración estéreo",
                "La carpeta seleccionada no contiene:\n" + "\n".join(missing),
            )
            return

        candidate_report = None
        report_path = source / "calibration_report.json"
        if report_path.is_file():
            try:
                candidate_report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception as exc:
                messagebox.showerror(
                    "Calibración estéreo",
                    f"calibration_report.json no es legible:\n{exc}",
                )
                return
            quality = str(candidate_report.get("quality", "")).strip().lower()
            if quality and quality != "accepted":
                messagebox.showerror(
                    "Calibración estéreo",
                    f"La calibración declara quality={quality}.\n"
                    "No se instalará una calibración científicamente no aceptada.",
                )
                return
            size = candidate_report.get("image_size")
            if size and list(size) != [RESOLUTION_W, RESOLUTION_H]:
                messagebox.showerror(
                    "Calibración estéreo",
                    f"Resolución incompatible en el reporte: {size}.\n"
                    f"El sistema captura a {RESOLUTION_W}x{RESOLUTION_H}.",
                )
                return

        proceed = messagebox.askyesno(
            "Importar calibración estéreo",
            (
                "Se reemplazará la calibración estéreo vigente del sistema.\n\n"
                "Esto debe hacerse únicamente cuando la calibración seleccionada "
                "corresponda a la posición física actual de las cámaras.\n\n"
                "¿Continuar?"
            ),
            icon="warning",
        )
        if not proceed:
            return

        if self.stereo_dir.exists():
            shutil.rmtree(self.stereo_dir)
        self.stereo_dir.mkdir(parents=True, exist_ok=True)

        allowed = {".yaml", ".yml", ".xml", ".npz", ".npy", ".json", ".csv", ".txt", ".png", ".md"}
        copied = 0
        for p in source.iterdir():
            if p.is_file() and p.suffix.lower() in allowed:
                shutil.copy2(p, self.stereo_dir / p.name)
                copied += 1

        # Una nueva geometría estéreo invalida cualquier calibración de
        # plataforma 3D previa. Se archiva en registros en lugar de borrarla.
        platform_files = [p for p in self.platform_dir.rglob("*") if p.is_file()]
        archived_platform = False
        if platform_files:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            archive_dir = self.records_dir / f"calibracion_plataforma_invalidada_{stamp}"
            archive_dir.mkdir(parents=True, exist_ok=True)
            for p in platform_files:
                rel = p.relative_to(self.platform_dir)
                target = archive_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(target))
            archived_platform = True

        self._refresh_system_status()
        detail = f"Calibración importada correctamente.\nArchivos copiados: {copied}"
        if candidate_report:
            try:
                detail += (
                    f"\nRMS estéreo: {float(candidate_report.get('stereo_rms_px')):.3f} px"
                    f"\nBaseline: {float(candidate_report.get('baseline_mm')):.2f} mm"
                    f"\nEpipolar mediana: "
                    f"{float((candidate_report.get('epipolar') or {}).get('median_abs_dy_px')):.3f} px"
                )
            except Exception:
                pass
        if archived_platform:
            detail += (
                "\n\nLa calibración de plataforma anterior fue archivada en registros "
                "porque una nueva calibración estéreo cambia el marco métrico. "
                "Debe calibrarse nuevamente la plataforma."
            )
        detail += (
            "\n\nSi también cambió físicamente la cámara, enfoque, exposición o iluminación, "
            "actualiza el fondo vacío antes de procesar."
        )
        messagebox.showinfo("Calibración estéreo", detail)

    @staticmethod
    def _safe_token(text: str) -> str:
        """Normaliza un nombre para rutas usando letras ASCII, cifras, guiones y subrayados."""
        text = str(text).strip()
        text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text or "objeto"

    def _remove_path(self, path: Path):
        """Elimina una ruta existente, distinguiendo directorios de archivos y enlaces."""
        if not path.exists() and not path.is_symlink():
            return
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()

    def _create_capture_documentation(self):
        """Guarda el protocolo del trabajo y crea la bitácora con hojas de capturas y cierres."""
        documentation = self.root_dir / "documentacion"
        documentation.mkdir(parents=True, exist_ok=True)

        protocol = documentation / "protocolo_captura.md"
        protocol.write_text(
            (
                "# Protocolo automático de captura\n\n"
                f"- Trabajo: {self.current_job_name}\n"
                f"- Objeto: {self.current_object}\n"
                f"- Modo: {self.job_mode}\n"
                f"- Pasos físicos por vuelta: {STEPS_PER_REVOLUTION}\n"
                f"- Vistas por sesión: {CAPTURE_VIEWS}\n"
                f"- Sesiones: 3\n"
                f"- Paso nominal: {NOMINAL_STEP_DEG:.6f}°\n"
                f"- Secuencia de pasos: {list(STEP_SEQUENCE)}\n"
                f"- Resolución: {RESOLUTION_W}x{RESOLUTION_H}\n"
                "- Cada sesión ejecuta CLOSE y retorna a pose 0.\n"
                "- El fondo y las calibraciones pertenecen al sistema, no al trabajo.\n"
            ),
            encoding="utf-8",
        )

        self.workbook_path = documentation / "bitacora_captura.xlsx"
        workbook = Workbook()

        ws = workbook.active
        ws.title = "Bitacora"
        ws.append(
            [
                "Fecha",
                "Objeto_ID",
                "Objeto_Nombre",
                "Sesion",
                "Vista",
                "Pose_index",
                "Angulo_nominal_deg",
                "Angulo_desde_pasos_deg",
                "Archivo_L",
                "Archivo_R",
                "Pasos_movimiento",
                "Pasos_acumulados",
                "Pasos_vuelta",
                "Tiempo_asentamiento_s",
                "Resolucion",
                "Estado",
                "Observaciones",
            ]
        )

        closures = workbook.create_sheet("Cierres")
        closures.append(
            [
                "Fecha",
                "Objeto",
                "Sesion",
                "Pasos_antes_cierre",
                "Pasos_movimiento_cierre",
                "Total_pasos_vuelta",
                "Arduino_revolution_flag",
                "Estado",
            ]
        )
        workbook.save(self.workbook_path)

    def _prepare_job(self, mode: str, object_name: str, suggested_name: str):
        """Crea un trabajo de un objeto con tres sesiones y actualiza el estado de la interfaz.

        Solicita confirmación antes de reemplazar un trabajo existente. Devuelve False
        si el usuario cancela y True cuando el trabajo queda preparado.
        """
        if self._busy() or self._closing:
            return
        job_name = simpledialog.askstring(
            "Nombre del trabajo",
            (
                "Escribe un nombre para esta campaña.\n\n"
                "Ejemplo: CALIBRACION_NUEVA o CUBO_VALIDACION_01"
            ),
            initialvalue=suggested_name,
            parent=self.root,
        )
        if not job_name:
            return False

        job_name = self._safe_token(job_name)
        object_folder = self._safe_token(object_name).lower()
        workspace = self.jobs_dir / job_name

        if workspace.exists():
            proceed = messagebox.askyesno(
                "Trabajo existente",
                (f"Ya existe:\n{workspace}\n\n" "¿Borrar SOLO ese trabajo y comenzar desde cero?"),
                icon="warning",
            )
            if not proceed:
                return False
            shutil.rmtree(workspace)

        # Un trabajo representa exactamente un objeto y contiene directamente
        # sus capturas, su reconstrucción y su resultado final.
        captures = workspace / "capturas"
        for s in range(1, 4):
            (captures / f"S{s:02d}" / "izquierda").mkdir(parents=True, exist_ok=True)
            (captures / f"S{s:02d}" / "derecha").mkdir(parents=True, exist_ok=True)

        self.root_dir = workspace.resolve()
        self.current_job_name = job_name
        self.current_object = object_folder
        self.job_mode = mode
        self.capture_plan = [
            {
                "id": "CAL01" if mode == "calibrar-plataforma" else "OBJ01",
                "name": object_folder.upper(),
                "folder": object_folder,
                "sessions": 3,
            }
        ]

        write_job_metadata(
            self.root_dir,
            self.job_mode,
            self.current_object,
        )

        self.root_dir_var.set(str(self.root_dir))
        self.job_mode_var.set(
            "CALIBRAR PLATAFORMA" if mode == "calibrar-plataforma" else "RECONSTRUIR OBJETO"
        )
        self.plan_var.set(self._plan_text())

        self.capture_started = False
        self.capture_completed = False
        self.waiting_object_change = False
        self.btn_process.config(state="disabled")
        self.btn_resume.config(state="disabled")

        self._create_capture_documentation()
        self._refresh_system_status()

        self.status_var.set("Trabajo creado. Coloca el objeto y captura S01 + S02 + S03.")
        return True

    def new_platform_calibration_job(self):
        """Valida los recursos e inicia un trabajo de calibración de plataforma."""
        if self._busy() or self._closing:
            return
        if not self._model_path().is_file():
            messagebox.showwarning(
                "Modelo CREStereo",
                f"Primero copia {MODEL_FILENAME} dentro de:\n{self.models_dir}",
            )
            return
        if not self._stereo_ready():
            messagebox.showwarning(
                "Calibración estéreo",
                "Primero importa una calibración estéreo válida.",
            )
            return
        if not self._background_ready():
            messagebox.showwarning(
                "Fondo vacío",
                "Primero abre las cámaras y captura el fondo vacío del montaje actual.",
            )
            return

        stamp = time.strftime("%Y%m%d_%H%M%S")
        self._prepare_job(
            "calibrar-plataforma",
            "cubo",
            f"CALIBRACION_PLATAFORMA_{stamp}",
        )

    def new_object_job(self):
        """Valida los recursos e inicia un trabajo de reconstrucción de objeto."""
        if self._busy() or self._closing:
            return
        status = self._refresh_system_status()
        missing = []
        if not status["model"]:
            missing.append("modelo CREStereo")
        if not status["stereo"]:
            missing.append("calibración estéreo")
        if not status["platform"]:
            missing.append("calibración de plataforma")
        if not status["background"]:
            missing.append("fondo vacío")
        if missing:
            messagebox.showwarning(
                "Sistema incompleto",
                "Antes de reconstruir falta:\n- " + "\n- ".join(missing),
            )
            return

        object_name = simpledialog.askstring(
            "Nuevo objeto",
            "Escribe un nombre corto para el objeto:",
            initialvalue="objeto",
            parent=self.root,
        )
        if not object_name:
            return

        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe_obj = self._safe_token(object_name)
        self._prepare_job(
            "reconstruir",
            safe_obj,
            f"{safe_obj}_{stamp}",
        )

    def reopen_existing_job(self):
        """Recupera el modo y las capturas de un trabajo para continuar su procesamiento."""
        if self._busy() or self._closing:
            return
        folder = filedialog.askdirectory(
            title="Selecciona una carpeta existente dentro de trabajos"
        )
        if not folder:
            return

        workspace = Path(folder).expanduser().resolve()
        try:
            workspace.relative_to(self.jobs_dir.resolve())
        except Exception:
            messagebox.showerror(
                "Trabajo inválido",
                "Selecciona una carpeta que esté dentro de la carpeta trabajos.",
            )
            return

        metadata_path = workspace / JOB_METADATA_FILENAME
        if not metadata_path.is_file():
            messagebox.showerror(
                "Trabajo inválido",
                f"No existe {JOB_METADATA_FILENAME} dentro del trabajo.",
            )
            return

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            object_name = self._safe_token(metadata.get("object", "")).lower()
        except Exception as exc:
            messagebox.showerror(
                "Trabajo inválido",
                f"No se pudieron leer los metadatos del trabajo:\n{exc}",
            )
            return

        try:
            mode, metadata_source = resolve_existing_job_mode(
                workspace,
                object_name,
            )
        except Exception as exc:
            messagebox.showerror(
                "Trabajo inválido",
                f"No se pudo validar el tipo de trabajo:\n{exc}",
            )
            return

        # Migra campañas anteriores para que las siguientes reaperturas ya no
        # dependan del nombre de carpeta ni de interpretar documentación libre.
        if metadata_source != "metadata":
            try:
                write_job_metadata(workspace, mode, object_name)
            except Exception as exc:
                messagebox.showerror(
                    "Trabajo inválido",
                    f"No se pudieron guardar los metadatos del trabajo:\n{exc}",
                )
                return

        self.root_dir = workspace
        self.current_job_name = workspace.name
        self.current_object = object_name
        self.job_mode = mode
        self.capture_plan = [
            {
                "id": "CAL01" if mode == "calibrar-plataforma" else "OBJ01",
                "name": self.current_object.upper(),
                "folder": self.current_object,
                "sessions": 3,
            }
        ]

        self.root_dir_var.set(str(self.root_dir))
        self.job_mode_var.set(
            "CALIBRAR PLATAFORMA" if mode == "calibrar-plataforma" else "RECONSTRUIR OBJETO"
        )
        self.plan_var.set(self._plan_text())

        self.workbook_path = workspace / "documentacion" / "bitacora_captura.xlsx"
        self.capture_started = any((workspace / "capturas").rglob("*.png"))
        self.capture_completed, reason = capture_readiness(workspace)
        self.last_session_return_time = None
        self._update_controls()
        if self.capture_completed:
            self.status_var.set(
                "Trabajo reabierto. Puedes procesar o reanudar desde el último checkpoint válido."
            )
        else:
            self.status_var.set(
                "Trabajo con captura incompleta. Conecta el equipo para reiniciar las tres sesiones."
            )
        self.progress_var.set(reason)

    def open_current_job_folder(self):
        """Abre el directorio del trabajo activo con el explorador del sistema."""
        if self.root_dir is None or not self.root_dir.exists():
            messagebox.showinfo("Trabajo", "Todavía no hay un trabajo seleccionado.")
            return
        try:
            if os.name == "nt":
                os.startfile(str(self.root_dir))
            else:
                subprocess.Popen(["xdg-open", str(self.root_dir)])
        except Exception as exc:
            messagebox.showerror("Abrir carpeta", str(exc))

    def process_current_job(self):
        """Solicita el procesamiento del trabajo desde cero."""
        self._start_pipeline(resume=False)

    def resume_current_job(self):
        """Solicita la reanudación del trabajo con sus checkpoints verificables."""
        self._start_pipeline(resume=True)

    def _start_pipeline(self, resume=False):
        """Valida el entorno y lanza el trabajador de procesamiento con el modo solicitado."""
        if self._busy() or self._closing:
            return
        if self.pipeline_running:
            return

        if self.root_dir is None or self.job_mode is None or not self.current_object:
            messagebox.showwarning("Trabajo", "Primero crea y captura un trabajo.")
            return
        if not self.capture_completed:
            messagebox.showwarning(
                "Capturas",
                "Primero termina S01 + S02 + S03 de este trabajo.",
            )
            return

        status = self._refresh_system_status()
        required = ("model", "stereo", "background")
        if any(not status[x] for x in required):
            messagebox.showerror(
                "Sistema",
                "Modelo, calibración estéreo y fondo deben estar disponibles.",
            )
            return
        if self.job_mode == "reconstruir" and not status["platform"]:
            messagebox.showerror(
                "Sistema",
                "No existe calibración de plataforma vigente.",
            )
            return

        if not resume:
            reconstruction = self.root_dir / "reconstruccion"
            has_previous = reconstruction.is_dir() and any(reconstruction.iterdir())
            if has_previous:
                proceed = messagebox.askyesno(
                    "Procesar desde cero",
                    (
                        "Este modo eliminará SOLO los resultados de reconstrucción "
                        "de este trabajo y volverá a ejecutar desde el paso 01.\n\n"
                        "Las capturas S01/S02/S03 se conservarán.\n\n"
                        "¿Continuar desde cero?"
                    ),
                    icon="warning",
                )
                if not proceed:
                    return

        complete, reason = capture_readiness(self.root_dir)
        if not complete:
            self.capture_completed = False
            self._update_controls()
            messagebox.showwarning("Capturas incompletas", reason)
            return
        self._begin_operation("pipeline", self._pipeline_worker, bool(resume))

    def _pipeline_worker(self, resume=False):
        """Ejecuta el coordinador en un subproceso y registra su salida.

        Envía estados a la cola de la interfaz. En modo de calibración, gestiona la
        promoción de la candidata una vez superadas las comprobaciones del flujo.
        """
        log_path = self.records_dir / (
            f"{self.current_job_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )

        try:
            runner = self.processing_dir / "00_ejecutar_pipeline.py"
            if not runner.is_file():
                raise FileNotFoundError(runner)

            cmd = [
                sys.executable,
                str(runner),
                self.job_mode,
                "--workspace",
                str(self.root_dir),
                "--object",
                self.current_object,
                "--model",
                str(self._model_path()),
                "--stereo-calibration-dir",
                str(self.stereo_dir),
                "--background-dir",
                str(self.background_dir),
            ]

            if self.job_mode == "calibrar-plataforma":
                cmd += [
                    "--platform-calibration-output-dir",
                    str(self.platform_dir),
                ]
            else:
                cmd += [
                    "--platform-calibration",
                    str(self._platform_calibration_path()),
                ]

            if resume:
                cmd.append("--resume")

            self.status_queue.put(
                (
                    "pipeline_status",
                    (
                        "Reanudación iniciada: auditando checkpoints del mismo trabajo."
                        if resume
                        else "Procesamiento desde cero iniciado. No cierres la aplicación."
                    ),
                )
            )

            with log_path.open("w", encoding="utf-8") as log:
                log.write("COMANDO INTERNO:\n" + " ".join(map(str, cmd)) + "\n\n")
                child_env = os.environ.copy()
                child_env["PYTHONUNBUFFERED"] = "1"
                child_env["PYTHONUTF8"] = "1"
                child_env["PYTHONIOENCODING"] = "utf-8"

                self._check_cancelled()
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=str(self.install_root),
                    bufsize=1,
                    env=child_env,
                    start_new_session=(os.name != "nt"),
                    creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
                )
                with self._pipeline_lock:
                    self._pipeline_proc = proc
                if self.stop_event.is_set():
                    with self._pipeline_lock:
                        terminate_process_tree(proc)
                if proc.stdout is not None:
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                        line = line.strip()
                        if line:
                            # La consola interna se conserva completa en el log.
                            # La GUI omite separadores sin información.
                            if re.fullmatch(r"[=\-#_\s]+", line):
                                continue

                            self.status_queue.put(("pipeline_progress", line))

                code = proc.wait()

            self._check_cancelled()
            if code != 0:
                raise RuntimeError(f"El pipeline terminó con código {code}. Revisa:\n{log_path}")

            result = {
                "mode": self.job_mode,
                "log": str(log_path),
            }

            if self.job_mode == "calibrar-plataforma":
                candidate = self.platform_dir / "calibracion_plataforma_candidata.json"
                canonical = self._platform_calibration_path()

                # El paso 09 conserva la candidata como parte verificable del
                # checkpoint. La promoción copia de forma atómica al nombre
                # canónico, de modo que una reanudación pueda volver a comprobar
                # el mismo contenido y restaurarlo si el archivo activo cambió.
                if candidate.is_file():
                    temporary = canonical.with_suffix(canonical.suffix + ".tmp")
                    shutil.copy2(candidate, temporary)
                    temporary.replace(canonical)
                elif not canonical.is_file():
                    raise FileNotFoundError(
                        "La calibración terminó pero no existe ni "
                        "calibracion_plataforma_candidata.json ni "
                        "calibracion_plataforma.json."
                    )

                result["platform_calibration"] = str(canonical)

            else:
                validation_summary = (
                    self.root_dir
                    / "reconstruccion"
                    / "multisesion"
                    / "17_validacion_modelo"
                    / "resumen_17_validacion_modelo.json"
                )
                if not validation_summary.is_file():
                    raise FileNotFoundError(
                        "La reconstrucción terminó sin el resumen exacto del paso 17."
                    )

                export_summary = (
                    self.root_dir
                    / "reconstruccion"
                    / "multisesion"
                    / "18_exportacion_modelo"
                    / "resumen_18_exportacion_modelo.json"
                )
                if not export_summary.is_file():
                    raise FileNotFoundError(
                        "La reconstrucción terminó sin la exportación final del paso 18."
                    )

                data = json.loads(export_summary.read_text(encoding="utf-8"))
                result["quality"] = data.get(
                    "validation_quality",
                    data.get("quality"),
                )
                result["summary"] = str(export_summary)
                result["validation_summary"] = str(validation_summary)

                result_dir = self.root_dir / "resultado_final"
                obj_path = result_dir / "modelo_final.obj"
                obj_mm_path = result_dir / "modelo_final_metric_mm.obj"
                ply_path = result_dir / "modelo_final_original_mm.ply"

                result["result_dir"] = str(result_dir) if result_dir.is_dir() else None
                result["obj"] = str(obj_path) if obj_path.is_file() else None
                result["obj_mm"] = str(obj_mm_path) if obj_mm_path.is_file() else None
                result["mesh"] = str(ply_path) if ply_path.is_file() else None

            self.status_queue.put(("pipeline_complete", result))

        except Exception as exc:
            if self.stop_event.is_set():
                self.status_queue.put(("pipeline_stopped", ""))
                return
            tail = ""
            try:
                if log_path.is_file():
                    lines = log_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    ).splitlines()
                    useful = [line for line in lines if line.strip()]
                    tail = "\n".join(useful[-12:])
            except Exception:
                tail = ""

            detail = str(exc)
            if tail:
                detail += "\n\nÚltimas líneas útiles:\n" + tail
            detail += f"\n\nLog completo:\n{log_path}"

            self.status_queue.put(
                (
                    "pipeline_error",
                    detail,
                )
            )

    # ------------------------------------------------------------------
    # Cámara
    # ------------------------------------------------------------------

    def _validate_frame_resolution(self, frame, label):
        """Rechaza imágenes vacías o incompatibles con la resolución configurada."""
        if frame is None:
            raise RuntimeError(f"{label}: cuadro vacío.")

        if frame.shape[:2] != (
            RESOLUTION_H,
            RESOLUTION_W,
        ):
            raise RuntimeError(
                f"{label}: resolución real "
                f"{frame.shape[1]}x{frame.shape[0]}, "
                f"esperada {RESOLUTION_W}x{RESOLUTION_H}."
            )

    def open_hardware(self):
        """Abre ambas cámaras y el puerto serie y prepara la vista previa."""
        if self._busy() or self._closing:
            return
        self.close_hardware()
        self.stop_event.clear()

        try:
            cam_l = int(self.cam_left_var.get())
            cam_r = int(self.cam_right_var.get())
            baud = int(self.baud_var.get())
        except ValueError:
            messagebox.showerror(
                "Error",
                "Revisa baudios e índices de cámara.",
            )
            return

        backend = getattr(cv2, "CAP_DSHOW", 0)

        self.cap_left = cv2.VideoCapture(
            cam_l,
            backend,
        )
        self.cap_right = cv2.VideoCapture(
            cam_r,
            backend,
        )

        for cap in (
            self.cap_left,
            self.cap_right,
        ):
            cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                RESOLUTION_W,
            )
            cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                RESOLUTION_H,
            )

        if not self.cap_left.isOpened() or not self.cap_right.isOpened():
            self.close_hardware()

            messagebox.showerror(
                "Cámaras",
                "No se pudieron abrir ambas cámaras.",
            )
            return

        for label, cap in (
            ("izquierda", self.cap_left),
            ("derecha", self.cap_right),
        ):
            width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            if (
                width,
                height,
            ) != (
                RESOLUTION_W,
                RESOLUTION_H,
            ):
                self.close_hardware()

                messagebox.showerror(
                    "Resolución",
                    f"{label}: {width}x{height}; " f"esperada {RESOLUTION_W}x{RESOLUTION_H}.",
                )
                return

        try:
            self.open_serial(baud)
        except Exception as exc:
            self.close_hardware()

            messagebox.showerror(
                "Arduino",
                f"No se pudo preparar el Arduino:\n{exc}",
            )
            return

        self.preview_running = True

        self.status_var.set("Cámaras y Arduino listos. Revisa la vista previa.")

        self._update_preview()

    def _update_preview(self):
        """Lee ambas cámaras, publica el par bajo bloqueo y programa la siguiente actualización.

        El bloqueo conserva la asociación de imágenes de una misma iteración; no
        implica sincronización física de los sensores.
        """
        if not self.preview_running or self._closing:
            return

        # El par se actualiza de forma atómica: una captura nunca mezcla
        # la izquierda de una iteración con la derecha de otra.
        ok_l, frame_l = self.cap_left.read() if self.cap_left is not None else (False, None)

        ok_r, frame_r = self.cap_right.read() if self.cap_right is not None else (False, None)

        if ok_l and ok_r:
            now = time.monotonic()

            with self.frame_lock:
                self.latest_left = frame_l.copy()
                self.latest_right = frame_r.copy()
                self.latest_pair_time = now

            self._show_frame(
                self.left_canvas,
                frame_l,
            )
            self._show_frame(
                self.right_canvas,
                frame_r,
            )

        else:
            with self.frame_lock:
                self.latest_left = None
                self.latest_right = None
                self.latest_pair_time = 0.0

        self.preview_job = self.root.after(
            30,
            self._update_preview,
        )

    def _show_frame(self, canvas, frame):
        """
        Dibuja un frame sin permitir que el tamaño de la imagen modifique la
        geometría de la interfaz.

        La relación de aspecto original se conserva siempre. El Canvas actúa
        como viewport: la imagen se centra y las zonas sobrantes permanecen
        oscuras.
        """
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        # El Canvas puede reportar 1x1 durante sus primeros ciclos antes de
        # que Tk resuelva la geometría. Se usan tamaños razonables únicamente
        # como fallback; no alteran el tamaño solicitado del Canvas.
        target_w = int(canvas.winfo_width())
        target_h = int(canvas.winfo_height())

        if target_w <= 2:
            target_w = 760
        if target_h <= 2:
            target_h = 428

        h, w = rgb.shape[:2]

        scale = min(
            target_w / float(w),
            target_h / float(h),
        )

        new_w = max(
            1,
            int(round(w * scale)),
        )
        new_h = max(
            1,
            int(round(h * scale)),
        )

        # Nunca se estira la imagen por separado en X/Y.
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

        rgb = cv2.resize(
            rgb,
            (new_w, new_h),
            interpolation=interpolation,
        )

        image = ImageTk.PhotoImage(Image.fromarray(rgb))

        item = self._preview_items.get(canvas)

        center_x = target_w / 2.0
        center_y = target_h / 2.0

        if item is None:
            item = canvas.create_image(
                center_x,
                center_y,
                anchor="center",
                image=image,
            )
            self._preview_items[canvas] = item
        else:
            canvas.coords(
                item,
                center_x,
                center_y,
            )
            canvas.itemconfigure(
                item,
                image=image,
            )

        # PhotoImage se destruye si Python pierde esta referencia.
        self._preview_images[canvas] = image

    def _current_pair(
        self,
        minimum_timestamp: Optional[float] = None,
        timeout: float = FRESH_FRAME_TIMEOUT_S,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve copias del par disponible que cumpla la antigüedad requerida.

        Comprueba la resolución de ambas imágenes y espera como máximo timeout
        segundos. Lanza RuntimeError si no llega un par válido dentro del plazo.
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            self._check_cancelled()
            with self.frame_lock:
                timestamp = self.latest_pair_time
                left = None if self.latest_left is None else self.latest_left.copy()
                right = None if self.latest_right is None else self.latest_right.copy()

            if (
                left is not None
                and right is not None
                and (minimum_timestamp is None or timestamp >= minimum_timestamp)
            ):
                self._validate_frame_resolution(
                    left,
                    "Cámara izquierda",
                )
                self._validate_frame_resolution(
                    right,
                    "Cámara derecha",
                )
                return left, right

            time.sleep(0.01)

        raise RuntimeError("No llegó un par estéreo fresco dentro del tiempo esperado.")

    # ------------------------------------------------------------------
    # Serial
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_key_values(line: str) -> Dict[str, str]:
        """Extrae los campos clave=valor de una respuesta del firmware."""
        values = {}

        for token in line.strip().split():
            if "=" not in token:
                continue

            key, value = token.split(
                "=",
                1,
            )
            values[key] = value

        return values

    def serial_command(
        self,
        command: str,
        expected_prefix: str,
        timeout: float = SERIAL_COMMAND_TIMEOUT_S,
        *,
        ignore_cancel=False,
    ) -> str:
        """Serializa comandos y respuestas; STOP puede completar la cancelación."""
        with self._serial_lock:
            if not ignore_cancel:
                self._check_cancelled()
            if self.ser is None or not self.ser.is_open:
                raise RuntimeError("Arduino no conectado.")
            self.ser.write((command.strip() + "\n").encode("ascii"))
            self.ser.flush()
            deadline = time.monotonic() + timeout
            lines = []
            while time.monotonic() < deadline:
                if not ignore_cancel:
                    self._check_cancelled()
                raw = self.ser.readline()
                if not raw:
                    continue
                line = raw.decode(errors="replace").strip()
                if not line:
                    continue
                lines.append(line)
                if line.startswith("ERR"):
                    raise RuntimeError(f"Arduino respondió: {line}")
                if line.startswith(expected_prefix):
                    return line
            raise TimeoutError(
                f"Sin respuesta '{expected_prefix}' a '{command}'. Recibido: {lines[-5:]}"
            )

    def open_serial(self, baud):
        """Abre el puerto y comprueba la identidad y el estado lógico del firmware.

        Espera el reinicio del Arduino, verifica PING y ejecuta RESET. Este último
        reinicia los contadores; no realiza una búsqueda física del origen.
        """
        if self.ser and self.ser.is_open:
            return

        port = self.port_var.get().strip()

        self.ser = serial.Serial(
            port,
            baudrate=baud,
            timeout=0.2,
            write_timeout=2,
        )

        # Arduino Uno reinicia al abrir el puerto.
        time.sleep(2.2)

        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception:
            pass

        pong = self.serial_command(
            "PING",
            expected_prefix="PONG",
            timeout=4.0,
        )

        if "ROT2055_V7_2" not in pong:
            raise RuntimeError(
                "El Arduino respondió, pero no tiene cargado "
                "el firmware definitivo ROT2055_V7_2."
            )

        self.serial_command(
            "RESET",
            expected_prefix="RESET_OK",
            timeout=3.0,
        )

        status = self._get_motor_status()

        if status["revsteps"] != STEPS_PER_REVOLUTION:
            raise RuntimeError(
                f"Firmware reporta revsteps={status['revsteps']}; "
                f"se esperaban {STEPS_PER_REVOLUTION}."
            )

        if status["moves"] != MOVES_PER_REVOLUTION:
            raise RuntimeError(
                f"Firmware reporta moves={status['moves']}; "
                f"se esperaban {MOVES_PER_REVOLUTION}."
            )

        if status["phase"] != 0 or status["cumulative"] != 0:
            raise RuntimeError(
                "El firmware no quedó en estado inicial después de RESET: "
                f"phase={status['phase']}, cumulative={status['cumulative']}."
            )

    def _get_motor_status(self) -> Dict[str, int]:
        """
        Lee el estado lógico del firmware y lo convierte a enteros.
        No mueve físicamente la plataforma.
        """
        line = self.serial_command(
            "STATUS",
            expected_prefix="STATUS",
            timeout=3.0,
        )

        values = self._parse_key_values(line)

        try:
            return {
                "phase": int(values["phase"]),
                "cumulative": int(values["cumulative"]),
                "revsteps": int(values["revsteps"]),
                "moves": int(values["moves"]),
            }
        except Exception as exc:
            raise RuntimeError(f"Respuesta STATUS inválida: {line}") from exc

    def _close_revolution(self) -> Dict[str, int]:
        """
        Ejecuta EXCLUSIVAMENTE el movimiento final de cierre.

        El firmware solo acepta CLOSE cuando:
            phase == 24
            cumulative == 1973

        La respuesta debe ser:
            CLOSE_DONE delta=82 total=2055 phase=0 cumulative=0
        """
        line = self.serial_command(
            "CLOSE",
            expected_prefix="CLOSE_DONE",
            timeout=20.0,
        )

        values = self._parse_key_values(line)

        try:
            return {
                "delta": int(values["delta"]),
                "total": int(values["total"]),
                "phase": int(values["phase"]),
                "cumulative": int(values["cumulative"]),
            }
        except Exception as exc:
            raise RuntimeError(f"Respuesta CLOSE inválida: {line}") from exc

    def _next_transition(self) -> Dict[str, int]:
        """Solicita NEXT y comprueba los datos de la transición informada por el firmware."""
        line = self.serial_command(
            "NEXT",
            expected_prefix="DONE",
            timeout=20.0,
        )

        values = self._parse_key_values(line)

        try:
            return {
                "delta": int(values["delta"]),
                "phase": int(values["phase"]),
                "cumulative": int(values["cumulative"]),
                "revolution": int(values["revolution"]),
            }
        except Exception as exc:
            raise RuntimeError(f"Respuesta NEXT inválida: {line}") from exc

    def release_motor(self):
        """Solicita liberar las bobinas cuando el puerto está abierto."""
        if self.ser is not None and self.ser.is_open:
            try:
                self.serial_command(
                    "RELEASE",
                    expected_prefix="RELEASED",
                    timeout=3.0,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Reinicio y fondo
    # ------------------------------------------------------------------

    def accept_setup(self):
        """Confirma la disponibilidad de cámaras y Arduino para comenzar la adquisición."""
        if self._busy() or self._closing:
            return
        self.stop_event.clear()
        if self.cap_left is None or self.cap_right is None or self.ser is None:
            messagebox.showwarning(
                "Aviso",
                "Primero abre cámaras + Arduino.",
            )
            return

        try:
            self._current_pair()
            self.serial_command(
                "RESET",
                expected_prefix="RESET_OK",
                timeout=3.0,
            )
        except Exception as exc:
            messagebox.showwarning("Aviso", str(exc))
            return

        self.accepted = True
        self._refresh_system_status()

        self.btn_background.config(state="normal")

        self.status_var.set(
            "Cámaras y Arduino listos. Puedes actualizar el fondo o crear un trabajo."
        )

    def capture_empty_background(self):
        """Valida el estado y lanza la adquisición del fondo vacío."""
        if self._busy() or self._closing:
            return
        if not self.accepted:
            messagebox.showwarning(
                "Aviso",
                "Primero confirma cámaras listas.",
            )
            return

        if self.background_capture_thread is not None and self.background_capture_thread.is_alive():
            return

        proceed = messagebox.askokcancel(
            "Fondo vacío",
            (
                "Retira completamente cualquier objeto de la plataforma.\n\n"
                f"Se tomarán {BACKGROUND_SAMPLES} pares y el fondo vigente "
                "del sistema será reemplazado."
            ),
        )

        if not proceed:
            return

        self._begin_operation("background", self._background_worker)

    def _background_worker(self):
        """Adquiere las muestras de fondo y guarda las referencias y sus metadatos."""
        try:
            left_frames = []
            right_frames = []

            for i in range(BACKGROUND_SAMPLES):
                if self.stop_event.is_set():
                    raise RuntimeError("Captura de fondo cancelada.")

                left, right = self._current_pair()

                left_frames.append(left)
                right_frames.append(right)

                self.status_queue.put(
                    (
                        "progress",
                        f"Fondo vacío {i + 1}/{BACKGROUND_SAMPLES}",
                    )
                )

                self._wait_cancelable(BACKGROUND_SAMPLE_INTERVAL_S)

            background_left = np.median(
                np.stack(left_frames),
                axis=0,
            ).astype(np.uint8)

            background_right = np.median(
                np.stack(right_frames),
                axis=0,
            ).astype(np.uint8)

            self._check_cancelled()
            directory = self.background_dir
            directory.mkdir(parents=True, exist_ok=True)

            path_l = directory / "background_left.png"
            path_r = directory / "background_right.png"
            path_pair = directory / "background_pair.png"

            if not cv2.imwrite(
                str(path_l),
                background_left,
            ):
                raise OSError("No se pudo guardar background_left.png.")

            if not cv2.imwrite(
                str(path_r),
                background_right,
            ):
                raise OSError("No se pudo guardar background_right.png.")

            if not cv2.imwrite(
                str(path_pair),
                np.hstack(
                    [
                        background_left,
                        background_right,
                    ]
                ),
            ):
                raise OSError("No se pudo guardar background_pair.png.")

            (directory / "background_metadata.json").write_text(
                json.dumps(
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "samples": BACKGROUND_SAMPLES,
                        "resolution": [
                            RESOLUTION_W,
                            RESOLUTION_H,
                        ],
                        "method": "median_stereo_background",
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            self.status_queue.put(
                (
                    "background_complete",
                    str(directory),
                )
            )

        except Exception as exc:
            kind = "background_stopped" if self.stop_event.is_set() else "background_error"
            self.status_queue.put((kind, str(exc)))

    # ------------------------------------------------------------------
    # Campaña
    # ------------------------------------------------------------------

    def start_capture(self):
        """Comprueba las condiciones iniciales e inicia la captura de la campaña."""
        if self._busy() or self._closing:
            return
        if not self.accepted:
            messagebox.showwarning(
                "Aviso",
                "Primero abre y confirma cámaras + Arduino.",
            )
            return

        if self.root_dir is None or not self.capture_plan:
            messagebox.showwarning(
                "Trabajo",
                "Primero pulsa 'Nueva calibración de plataforma' o 'Nuevo objeto'.",
            )
            return

        if not self._background_ready():
            messagebox.showwarning(
                "Fondo vacío",
                "Primero captura el fondo vacío correspondiente al montaje actual.",
            )
            return

        if self.capture_thread is not None and self.capture_thread.is_alive():
            return

        restart = self.capture_started
        proceed = messagebox.askokcancel(
            "Reiniciar captura" if restart else "Capturar S01 + S02 + S03",
            (
                (
                    "La captura anterior está incompleta. Se reemplazarán las tres sesiones, "
                    "su bitácora y los resultados de este trabajo.\n\n"
                    if restart
                    else ""
                )
                + f"Objeto: {self.current_object.upper()}\n"
                f"Trabajo: {self.current_job_name}\n\n"
                "Coloca el objeto en la posición que deseas llamar 0°.\n\n"
                "La aplicación capturará automáticamente:\n"
                "S01 → 25 vistas\n"
                "S02 → 25 vistas\n"
                "S03 → 25 vistas\n\n"
                "Entre sesiones la plataforma regresará automáticamente al origen.\n\n"
                "¿Iniciar?"
            ),
        )
        if not proceed:
            return

        self.stop_event.clear()
        self.object_change_event.clear()
        try:
            if self.ser is not None:
                self.ser.reset_input_buffer()
            self.serial_command("RESET", expected_prefix="RESET_OK", timeout=3.0)
            if restart:
                for folder in ("capturas", "reconstruccion", "resultado_final"):
                    self._remove_path(self.root_dir / folder)
            for item in self.capture_plan:
                for index in range(1, item["sessions"] + 1):
                    for side in ("izquierda", "derecha"):
                        (self.root_dir / "capturas" / f"S{index:02d}" / side).mkdir(
                            parents=True, exist_ok=True
                        )
            self._create_capture_documentation()
        except Exception as exc:
            messagebox.showerror("Preparar captura", str(exc))
            return
        self.capture_started = True
        self.capture_completed = False
        self._begin_operation("capture", self.capture_all_plan)

    def capture_all_plan(self):
        """Recorre objetos y sesiones del plan y comunica progreso, cancelación o errores."""
        total_sessions = sum(item["sessions"] for item in self.capture_plan)

        completed = 0

        try:
            for item_index, item in enumerate(self.capture_plan):
                if self.stop_event.is_set():
                    break

                obj_id = item["id"].upper()
                obj_name = item["name"].upper()
                obj_folder = item["folder"]

                # El paso debe comenzar en 0 para el nuevo objeto.
                self.serial_command(
                    "RESET",
                    expected_prefix="RESET_OK",
                    timeout=3.0,
                )

                # El nuevo objeto define un nuevo origen físico.
                self.last_session_return_time = None

                for session_index in range(
                    1,
                    item["sessions"] + 1,
                ):
                    if self.stop_event.is_set():
                        break

                    session = f"S{session_index:02d}"

                    session_dir = self.root_dir / "capturas" / session

                    self.status_queue.put(
                        (
                            "status",
                            f"Capturando {obj_name} {session}",
                        )
                    )

                    self.capture_one_session(
                        obj_id=obj_id,
                        obj_name=obj_name,
                        obj_folder=obj_folder,
                        session=session,
                        session_dir=session_dir,
                    )

                    completed += 1

                    self.status_queue.put(
                        (
                            "progress",
                            f"Sesiones completadas: {completed}/{total_sessions}",
                        )
                    )

                if not self.stop_event.is_set() and item_index < len(self.capture_plan) - 1:
                    self.release_motor()

                    next_item = self.capture_plan[item_index + 1]

                    self.waiting_object_change = True
                    self.object_change_event.clear()

                    message = (
                        f"{obj_name} terminado.\n\n"
                        f"Cambia al siguiente objeto: "
                        f"{next_item['name']}.\n"
                        "Colócalo en su posición 0 y pulsa Continuar."
                    )

                    self.status_queue.put(
                        (
                            "object_pause",
                            message,
                        )
                    )

                    while not self.object_change_event.is_set():
                        if self.stop_event.is_set():
                            break
                        time.sleep(0.1)

            self.waiting_object_change = False

            if self.stop_event.is_set():
                self.status_queue.put(
                    (
                        "capture_stopped",
                        "",
                    )
                )
            else:
                self.capture_completed = True
                self.status_queue.put(
                    (
                        "capture_complete",
                        "",
                    )
                )

        except Exception as exc:
            kind = "capture_stopped" if self.stop_event.is_set() else "capture_error"
            self.status_queue.put((kind, str(exc)))

    def capture_one_session(
        self,
        obj_id,
        obj_name,
        obj_folder,
        session,
        session_dir,
    ):
        # Verifica que la sesión empieza exactamente con la secuencia
        # lógica en paso 0. No mueve físicamente nada.
        """Captura 25 pares y completa la vuelta con el comando CLOSE.

        Exige estado lógico inicial, verifica cada movimiento y espera imágenes
        posteriores al asentamiento. El cierre se valida antes de habilitar la
        siguiente sesión; no se captura una imagen adicional al completar la vuelta.
        """
        status = self._get_motor_status()

        if status["phase"] != 0 or status["cumulative"] != 0:
            raise RuntimeError(
                f"{obj_name} {session}: el motor no está listo para iniciar. "
                f"phase={status['phase']}, cumulative={status['cumulative']}. "
                "La sesión NO comenzará."
            )

        control_dir = session_dir / "control_origen"
        control_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # S01 puede comenzar con el frame actual. S02/S03 deben esperar un
        # frame adquirido DESPUÉS del movimiento final de retorno de la sesión
        # anterior.
        if self.last_session_return_time is None:
            origin_left, origin_right = self._current_pair()
        else:
            origin_left, origin_right = self._current_pair(
                minimum_timestamp=self.last_session_return_time,
            )

        cv2.imwrite(
            str(control_dir / "inicio_left.png"),
            origin_left,
        )
        cv2.imwrite(
            str(control_dir / "inicio_right.png"),
            origin_right,
        )

        for pose_index in range(CAPTURE_VIEWS):
            if self.stop_event.is_set():
                raise RuntimeError("Campaña detenida.")

            view_num = pose_index + 1

            nominal_angle = pose_index * NOMINAL_STEP_DEG

            steps_angle = exact_angle_from_steps(pose_index)

            movement_steps = 0

            if pose_index > 0:
                expected_delta = STEP_SEQUENCE[pose_index - 1]

                move = self._next_transition()

                if move["delta"] != expected_delta:
                    raise RuntimeError(
                        f"{obj_name} {session}: transición incorrecta. "
                        f"Esperada={expected_delta}, Arduino={move['delta']}."
                    )

                movement_steps = move["delta"]

                # Espera mecánica.
                self._wait_cancelable(SETTLE_TIME_S)

                # Exige un par que haya sido adquirido después del settle.
                fresh_after = time.monotonic()
                frame_l, frame_r = self._current_pair(
                    minimum_timestamp=fresh_after,
                )

            else:
                frame_l, frame_r = self._current_pair()

            nominal_tenths = int(round(nominal_angle * 10.0))

            fname_l = (
                f"{obj_id}_{obj_name}_{session}_"
                f"V{view_num:03d}_"
                f"A{nominal_tenths:04d}_L.{IMAGE_EXT}"
            )

            fname_r = (
                f"{obj_id}_{obj_name}_{session}_"
                f"V{view_num:03d}_"
                f"A{nominal_tenths:04d}_R.{IMAGE_EXT}"
            )

            path_l = session_dir / "izquierda" / fname_l
            path_r = session_dir / "derecha" / fname_r

            ok_l = cv2.imwrite(
                str(path_l),
                frame_l,
            )

            ok_r = cv2.imwrite(
                str(path_r),
                frame_r,
            )

            if not (ok_l and ok_r):
                if path_l.exists():
                    path_l.unlink()

                if path_r.exists():
                    path_r.unlink()

                raise OSError(f"No se pudo guardar {obj_name} " f"{session} V{view_num:03d}.")

            self.append_log_row(
                obj_id=obj_id,
                obj_name=obj_name,
                session=session,
                view_num=view_num,
                pose_index=pose_index,
                nominal_angle=nominal_angle,
                steps_angle=steps_angle,
                file_l=fname_l,
                file_r=fname_r,
                movement_steps=movement_steps,
            )

            self.status_queue.put(
                (
                    "progress",
                    (
                        f"{obj_name} {session} | "
                        f"V{view_num:03d}/{CAPTURE_VIEWS} | "
                        f"{nominal_angle:.1f}° nominales | "
                        f"{steps_angle:.3f}° por pasos"
                    ),
                )
            )

        # ==============================================================
        # MOVIMIENTO FINAL OBLIGATORIO ENTRE SESIONES — CLOSE
        # ==============================================================
        #
        # Después de V025, la sesión NO puede terminar con un NEXT normal.
        # Antes del cierre se consulta STATUS y debe existir exactamente:
        #
        #     phase=24
        #     cumulative=1973
        #
        # Solo entonces Python envía CLOSE.
        #
        # CLOSE ejecuta físicamente los últimos 82 pasos, reporta
        # explícitamente total=2055 y reinicia la secuencia lógica a:
        #
        #     phase=0
        #     cumulative=0
        #
        # Después se consulta STATUS nuevamente. La siguiente sesión solo
        # comienza si ambas verificaciones son correctas.
        accumulated_before_closure = sum(STEP_SEQUENCE[:24])
        closure_expected = STEP_SEQUENCE[24]

        if accumulated_before_closure != 1973:
            raise RuntimeError("Error interno: antes del cierre deberían existir 1973 pasos.")

        if accumulated_before_closure + closure_expected != STEPS_PER_REVOLUTION:
            raise RuntimeError("Error interno: 1973 + 82 no suma 2055.")

        before = self._get_motor_status()

        self.status_queue.put(
            (
                "status",
                (
                    f"{obj_name} {session} — antes del cierre: "
                    f"phase={before['phase']}, "
                    f"cumulative={before['cumulative']}"
                ),
            )
        )

        if before["phase"] != 24 or before["cumulative"] != accumulated_before_closure:
            raise RuntimeError(
                f"{obj_name} {session}: estado incorrecto antes de CLOSE. "
                f"Esperado phase=24 cumulative={accumulated_before_closure}; "
                f"obtenido phase={before['phase']} "
                f"cumulative={before['cumulative']}. "
                "No se ejecutará el cierre."
            )

        self.status_queue.put(
            (
                "progress",
                (f"{obj_name} {session}: ejecutando CLOSE " f"(+{closure_expected} pasos)..."),
            )
        )

        close_result = self._close_revolution()

        if close_result["delta"] != closure_expected:
            raise RuntimeError(
                f"CLOSE ejecutó {close_result['delta']} pasos; " f"se esperaban {closure_expected}."
            )

        if close_result["total"] != STEPS_PER_REVOLUTION:
            raise RuntimeError(
                f"CLOSE reportó total={close_result['total']}; "
                f"se esperaban {STEPS_PER_REVOLUTION}."
            )

        if close_result["phase"] != 0 or close_result["cumulative"] != 0:
            raise RuntimeError("CLOSE no terminó en phase=0 cumulative=0.")

        self.status_queue.put(
            (
                "progress",
                (f"CLOSE_DONE delta={close_result['delta']} " f"total={close_result['total']}"),
            )
        )

        # Asentamiento mecánico después del cierre físico.
        self._wait_cancelable(SESSION_RETURN_SETTLE_S)

        # Exige un par estéreo posterior al movimiento final.
        self.last_session_return_time = time.monotonic()

        close_left, close_right = self._current_pair(
            minimum_timestamp=self.last_session_return_time,
        )

        # Segunda comprobación: el firmware debe seguir en origen lógico.
        after = self._get_motor_status()

        self.status_queue.put(
            (
                "status",
                (
                    f"{obj_name} {session} — después del cierre: "
                    f"phase={after['phase']}, "
                    f"cumulative={after['cumulative']}"
                ),
            )
        )

        if after["phase"] != 0 or after["cumulative"] != 0:
            raise RuntimeError(
                f"{obj_name} {session}: estado incorrecto después de CLOSE. "
                f"phase={after['phase']}, cumulative={after['cumulative']}. "
                "La siguiente sesión NO comenzará."
            )

        cv2.imwrite(
            str(control_dir / "cierre_2055_left.png"),
            close_left,
        )
        cv2.imwrite(
            str(control_dir / "cierre_2055_right.png"),
            close_right,
        )

        (control_dir / "movimiento_sesion.json").write_text(
            json.dumps(
                {
                    "steps_per_revolution": STEPS_PER_REVOLUTION,
                    "step_sequence": list(STEP_SEQUENCE),
                    "sequence_sum": sum(STEP_SEQUENCE),
                    "captured_views": CAPTURE_VIEWS,
                    "nominal_step_deg": NOMINAL_STEP_DEG,
                    "status_before_close": before,
                    "close_command_result": close_result,
                    "status_after_close": after,
                    "steps_before_closure": accumulated_before_closure,
                    "closure_steps": closure_expected,
                    "final_return_move_executed": True,
                    "close_command_used": True,
                    "automatic_correction": False,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        self.append_closure_row(
            obj_name=obj_name,
            session=session,
            before=accumulated_before_closure,
            closure_steps=closure_expected,
            revolution_flag=1,
        )

        self._wait_cancelable(SESSION_START_GUARD_S)

        self.status_queue.put(
            (
                "status",
                (
                    f"{obj_name} {session}: cierre físico completado. "
                    "Ahora sí puede comenzar la siguiente sesión."
                ),
            )
        )

    def append_log_row(
        self,
        obj_id,
        obj_name,
        session,
        view_num,
        pose_index,
        nominal_angle,
        steps_angle,
        file_l,
        file_r,
        movement_steps,
    ):
        """Añade a la bitácora los archivos y ángulos nominal y físico de una captura."""
        workbook = load_workbook(self.workbook_path)
        ws = workbook["Bitacora"]

        cumulative = cumulative_steps_before_pose(pose_index)

        ws.append(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                obj_id,
                obj_name,
                session,
                f"V{view_num:03d}",
                pose_index,
                float(nominal_angle),
                float(steps_angle),
                file_l,
                file_r,
                int(movement_steps),
                cumulative,
                STEPS_PER_REVOLUTION,
                (SETTLE_TIME_S if movement_steps else 0.0),
                f"{RESOLUTION_W}x{RESOLUTION_H}",
                "Capturado",
                "",
            ]
        )

        workbook.save(self.workbook_path)

    def append_closure_row(
        self,
        obj_name,
        session,
        before,
        closure_steps,
        revolution_flag,
    ):
        """Registra el movimiento de cierre y su estado en la hoja Cierres."""
        workbook = load_workbook(self.workbook_path)

        ws = workbook["Cierres"]

        ws.append(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                obj_name,
                session,
                int(before),
                int(closure_steps),
                int(before + closure_steps),
                int(revolution_flag),
                (
                    "OK"
                    if (before + closure_steps == STEPS_PER_REVOLUTION and revolution_flag == 1)
                    else "ERROR"
                ),
            ]
        )

        workbook.save(self.workbook_path)

    # ------------------------------------------------------------------
    # Controles
    # ------------------------------------------------------------------

    def continue_after_object_change(self):
        """Define la posición actual como origen lógico y continúa el plan pendiente."""
        if not self.waiting_object_change:
            return

        try:
            # La posición física actual del nuevo objeto se define como
            # pose 0. RESET solo reinicia la secuencia lógica.
            self.serial_command(
                "RESET",
                expected_prefix="RESET_OK",
                timeout=3.0,
            )
        except Exception as exc:
            messagebox.showerror(
                "Arduino",
                str(exc),
            )
            return

        self.waiting_object_change = False
        self.object_change_event.set()
        self.btn_continue.config(state="disabled")

    def stop_capture(self):
        """Solicita una parada y espera la liberación real de trabajadores y subprocesos."""
        if self._stop_thread is not None and self._stop_thread.is_alive():
            return
        self._stop_warning = None
        self.stop_event.set()
        self.object_change_event.set()
        self.waiting_object_change = False
        self.status_var.set(
            "Deteniendo la operación…" if self._operation else "Liberando la plataforma…"
        )
        self._stop_thread = threading.Thread(target=self._stop_resources, daemon=True)
        self._stop_thread.start()
        self._update_controls()

    # ------------------------------------------------------------------
    # Cola UI
    # ------------------------------------------------------------------

    def _process_status_queue(self):
        """Aplica eventos en Tk y libera los controles solo al finalizar los trabajadores."""
        if self._destroyed:
            return
        try:
            for _ in range(150):
                kind, message = self.status_queue.get_nowait()
                if kind == "operation_finished":
                    self._finished_event = True
                elif kind == "stop_error":
                    self._stop_warning = str(message)
                    self.progress_var.set(str(message))
                elif kind == "resources_stopped":
                    if not self._operation and not self._closing:
                        self.status_var.set(
                            "Plataforma detenida."
                            if not getattr(self, "_stop_warning", None)
                            else "Revisa la conexión con Arduino."
                        )
                elif kind.endswith(("_complete", "_error", "_stopped")):
                    self._terminal_event = (kind, message)
                elif kind in ("status", "pipeline_status", "object_pause"):
                    if not self.stop_event.is_set() and not self._closing:
                        self.status_var.set(str(message))
                elif kind == "pipeline_progress":
                    if self._operation == "pipeline" and not self.stop_event.is_set():
                        self._pipeline_progress.feed(str(message))
                elif kind == "progress":
                    if not self.stop_event.is_set() and not self._closing:
                        self.progress_var.set(str(message))
        except queue.Empty:
            pass
        if self._operation == "pipeline" and not self.stop_event.is_set() and not self._closing:
            status, detail = self._pipeline_progress.render()
            self.status_var.set(status)
            self.progress_var.set(detail)
        if (
            self._finished_event
            and not self._thread_alive(self._worker_thread)
            and not self._thread_alive(self._stop_thread)
        ):
            self._finish_operation()
        self._update_controls()
        if self._closing:
            if not self._busy():
                self._finalize_close()
                return
        self._status_job = self.root.after(100, self._process_status_queue)

    # ------------------------------------------------------------------
    # Cierre
    # ------------------------------------------------------------------

    def close_hardware(self):
        """Cancela la vista previa, libera cámaras y cierra el puerto serie."""
        self.preview_running = False

        if self.preview_job is not None:
            try:
                self.root.after_cancel(self.preview_job)
            except Exception:
                pass

            self.preview_job = None

        for cap in (
            self.cap_left,
            self.cap_right,
        ):
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

        self.cap_left = None
        self.cap_right = None

        if self.ser is not None:
            try:
                if self.ser.is_open:
                    try:
                        self.ser.write(b"STOP\n")
                        self.ser.flush()
                        time.sleep(0.05)
                    except Exception:
                        pass

                    self.ser.close()

            except Exception:
                pass

        self.ser = None
        self.accepted = False
        self.last_session_return_time = None
        with self.frame_lock:
            self.latest_left = None
            self.latest_right = None
            self.latest_pair_time = 0.0
        if hasattr(self, "btn_start"):
            self._update_controls()

    def on_close(self):
        """Detiene la operación antes de destruir Tk y cerrar los dispositivos."""
        if self._closing:
            return
        if self._busy():
            if not messagebox.askyesno(
                "Cerrar sistema",
                "Hay una operación activa. Se detendrá antes de cerrar.\n\nEl procesamiento podrá reanudarse desde sus checkpoints. Una captura interrumpida deberá reiniciarse.\n\n¿Detener y cerrar?",
                parent=self.root,
            ):
                return
        self._closing = True
        self.stop_capture()
        self.status_var.set("Cerrando: esperando la finalización de los recursos…")
        self._update_controls()


if __name__ == "__main__":
    print(
        "[PROGRESO] Se está ejecutando: Sistema_3D.py",
        flush=True,
    )
    print(
        "[PROGRESO] Iniciando la aplicación principal",
        flush=True,
    )
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    app = CaptureApp(root)
    root.mainloop()
