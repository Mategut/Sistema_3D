"""Acciones de mantenimiento y calibración de la interfaz Tk.

Los procesos envían eventos a la cola; únicamente el hilo principal modifica Tk.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from procesamiento.utilidades_estereo import validate_stereo_calibration

CANDIDATE = "calibracion_plataforma_candidata.json"


def install_stereo(source, stereo_dir, platform_dir, records_dir):
    """Copia primero y conserva ambas referencias ante un fallo del intercambio."""
    import shutil
    source, stereo_dir, platform_dir, records_dir = map(lambda p: Path(p).resolve(), (source, stereo_dir, platform_dir, records_dir))
    if source == stereo_dir or stereo_dir in source.parents:
        raise ValueError("Selecciona una calibración externa a la carpeta estéreo activa.")
    validate_stereo_calibration(source)
    records_dir.mkdir(parents=True, exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix="cambio_estereo_", dir=records_dir))
    staged = archive / "nueva"
    staged.mkdir()
    allowed = {".yaml", ".yml", ".xml", ".npz", ".npy", ".json", ".csv", ".txt", ".png", ".md"}
    copied = 0
    for path in source.iterdir():
        if path.is_file() and path.suffix.lower() in allowed:
            shutil.copy2(path, staged / path.name)
            copied += 1
    validate_stereo_calibration(staged)
    old_stereo, old_platform = archive / "estereo_anterior", archive / "plataforma_anterior"
    installed = False
    try:
        if stereo_dir.exists():
            stereo_dir.rename(old_stereo)
        if platform_dir.exists():
            platform_dir.rename(old_platform)
        staged.rename(stereo_dir)
        installed = True
        platform_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        if installed:
            stereo_dir.rename(staged)
        if old_stereo.exists():
            old_stereo.rename(stereo_dir)
        if old_platform.exists():
            old_platform.rename(platform_dir)
        raise
    return copied, old_platform.exists() and any(old_platform.rglob("*"))


class CalibrationToolsMixin:
    def _scroll_tab(self, title):
        page = ttk.Frame(self.control_tabs)
        self.control_tabs.add(page, text=title)
        canvas = tk.Canvas(page, highlightthickness=0, background="#edf2f7")
        scroll = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        content = ttk.Frame(canvas, padding=(0, 6, 5, 0))
        item = canvas.create_window(0, 0, window=content, anchor="nw")
        content.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(item, width=event.width))
        # Bind only this page, so scrolling a form does not move other tabs.
        def wheel(event):
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        def bind_children(widget):
            widget.bind("<MouseWheel>", wheel, add="+")
            for child in widget.winfo_children():
                bind_children(child)
        content.after_idle(lambda: bind_children(content))
        return content

    def _build_tool_tabs(self):
        calibration = self._scroll_tab("Calibración")
        diagnostics = self._scroll_tab("Herramientas")
        def card(parent, title, note, buttons):
            frame = ttk.LabelFrame(parent, text=title, style="Card.TLabelframe")
            frame.pack(fill="x", pady=(0, 10))
            ttk.Label(frame, text=note, wraplength=round(275 * self._ui_scale)).pack(fill="x", pady=(0, 6))
            for label, command in buttons:
                button = ttk.Button(frame, text=label, command=command)
                button.pack(fill="x", pady=3)
                self._idle_widgets.append(button)
            return frame
        stereo_card = card(calibration, "1 · Calibración estéreo",
             "Primero calibra las cámaras con el tablero. Importar una nueva geometría invalida la plataforma anterior.", [
                 ("Crear calibración con tablero…", self.stereo_wizard),
                 ("Importar calibración estéreo…", self.import_stereo_calibration),
                 ("Verificar estéreo y fondo", lambda: self.run_tool("verificar_calibracion_estereo.py")),
             ])
        self._last_stereo_result = None
        self._stereo_result_buttons = []
        for title, action in (("Abrir último resultado estéreo", lambda: self.open_tool_path(self._last_stereo_result)), ("Importar última candidata estéreo", lambda: self.import_stereo_calibration(self._last_stereo_result))):
            button = ttk.Button(stereo_card, text=title, command=action, state="disabled")
            button.pack(fill="x", pady=3)
            self._stereo_result_buttons.append(button)
        card(calibration, "2 · Calibración de plataforma",
             "Crea la campaña, captura las sesiones y procesa. Si el paso 09 aprueba, la calibración se instala automáticamente.", [
                 ("Crear campaña de calibración", self.new_platform_calibration_job),
                 ("Instalar calibración guardada…", lambda: self.platform_action(True)),
                 ("Cómo calibrar la plataforma", lambda: self.open_tool_path(self.install_root / "documentacion/validacion_independiente_plataforma.md")),
             ])
        card(diagnostics, "Diagnóstico", "Las comprobaciones se ejecutan en segundo plano. Su salida queda disponible en el registro.", [
            ("Verificar dependencias", lambda: self.run_tool("verificar_dependencias.py", ["--no-install"])),
            ("Auditar evidencia izquierda/derecha…", self.audit_lr),
            ("Ver registro de la última herramienta", self.show_tool_log),
            ("Abrir carpeta de registros", lambda: self.open_tool_path(self.records_dir)),
        ])
        options = card(diagnostics, "Procesamiento", "Estas opciones se aplican al iniciar o reanudar. Completo conserva los archivos intermedios para auditorías LR; ocupa más espacio.", [])
        self.provider_var = tk.StringVar(value="auto")
        self.storage_var = tk.StringVar(value="reducido")
        self._option_widgets = []
        for label, variable, values in (("Motor de profundidad", self.provider_var, ("auto", "cuda", "directml", "cpu")), ("Conservación de resultados", self.storage_var, ("reducido", "completo"))):
            ttk.Label(options, text=label).pack(anchor="w", pady=(5, 0))
            combo = ttk.Combobox(options, textvariable=variable, values=values, state="readonly")
            combo.pack(fill="x", pady=3)
            self._option_widgets.append(combo)
        self._tool_log = "Todavía no se ha ejecutado ninguna herramienta."
        self._log_widget = None

    def open_tool_path(self, path):
        try:
            path = Path(path).resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            os.startfile(str(path))
        except Exception as exc:
            messagebox.showerror("Abrir archivo", str(exc), parent=self.root)

    def show_tool_log(self):
        if self._log_widget is not None and self._log_widget.winfo_exists():
            self._log_widget.winfo_toplevel().lift()
            return
        window = tk.Toplevel(self.root)
        window.title("Registro de herramientas")
        window.geometry("900x540")
        frame = ttk.Frame(window, padding=10)
        frame.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(frame)
        scroll.pack(side="right", fill="y")
        self._log_widget = tk.Text(frame, wrap="word", yscrollcommand=scroll.set)
        self._log_widget.pack(fill="both", expand=True)
        scroll.configure(command=self._log_widget.yview)
        self._log_widget.insert("end", self._tool_log)
        self._log_widget.configure(state="disabled")

    def append_tool_log(self, line):
        self._tool_log = (self._tool_log + line)[-100000:]
        widget = self._log_widget
        if widget is not None and widget.winfo_exists():
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("end", self._tool_log)
            widget.see("end")
            widget.configure(state="disabled")

    def run_tool(self, script, arguments=(), output=None):
        if self._busy() or self._closing:
            return
        self._tool_log = ""
        self.append_tool_log(f"Herramienta: {script}\n")
        self.show_tool_log()
        self._begin_operation("tool", self._tool_worker, script, list(arguments), output)

    def _tool_worker(self, script, arguments, output):
        path = self.install_root / "herramientas" / script
        if not path.is_file():
            raise FileNotFoundError(path)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        fd, logfile = tempfile.mkstemp(prefix=Path(script).stem + "_", suffix=".log", dir=self.records_dir)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        with os.fdopen(fd, "w", encoding="utf-8") as log:
            with self._pipeline_lock:
                self._check_cancelled()
                self._pipeline_proc = subprocess.Popen(
                    [sys.executable, "-u", str(path), *map(str, arguments)],
                    cwd=self.install_root, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                    creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0,
                    start_new_session=os.name != "nt",
                )
                proc = self._pipeline_proc
            for line in proc.stdout:
                log.write(line)
                log.flush()
                self.status_queue.put(("tool_output", line))
            code = proc.wait()
        self._check_cancelled()
        if code:
            raise RuntimeError(f"{script} terminó con código {code}.\nRevisa el diagnóstico en la ventana de registro.\nRegistro completo: {logfile}")
        self.status_queue.put(("tool_complete", {"script": script, "log": logfile, "output": output}))

    def platform_action(self, activate):
        if self._busy() or self._closing:
            return
        selected = filedialog.askopenfilename(parent=self.root, title="Selecciona calibracion_plataforma_candidata.json", filetypes=[("Candidata JSON", "*.json")], initialdir=self.jobs_dir)
        if not selected:
            return
        candidate = Path(selected)
        if candidate.name != CANDIDATE:
            messagebox.showerror("Candidata", f"Selecciona el archivo {CANDIDATE} de resultado_calibracion_plataforma.", parent=self.root)
            return
        args = ["activar" if activate else "evaluar", "--candidate-dir", str(candidate.parent)]
        note = ("Se comprobarán la aprobación del paso 09, su auditoría y la correspondencia estéreo antes de instalar. No necesitas informes adicionales."
                if activate else "La candidata quedará disponible para nuevas campañas de evaluación. Seguirá pendiente de validación independiente.")
        if messagebox.askyesno("Activar calibración" if activate else "Evaluar candidata", note + "\n\nLa referencia vigente se archivará. Los trabajos existentes conservan sus referencias congeladas.\n\n¿Continuar?", parent=self.root):
            self.run_tool("promover_calibracion_plataforma.py", args)

    def stereo_wizard(self):
        if self._busy() or self._closing:
            return
        window = tk.Toplevel(self.root)
        window.title("Calibración estéreo con tablero")
        window.transient(self.root)
        window.grab_set()
        frame = ttk.Frame(window, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Introduce las esquinas INTERNAS del tablero y las medidas reales en mm.\nCaptura a 1920 × 1080 con los índices de cámara del panel principal.", wraplength=500).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        fields = {}
        for row, (key, title, default) in enumerate([
            ("cols", "Esquinas horizontales", "6"), ("rows", "Esquinas verticales", "8"),
            ("square-mm", "Lado de cada cuadro (mm)", "23"),
            ("expected-baseline-mm", "Separación de cámaras (mm; 0 omite control)", "81"),
            ("baseline-tolerance-mm", "Tolerancia de separación (mm)", "20"),
        ], 1):
            ttk.Label(frame, text=title).grid(row=row, column=0, sticky="w", pady=5)
            fields[key] = tk.StringVar(value=default)
            ttk.Entry(frame, textvariable=fields[key], width=12).grid(row=row, column=1, padx=10)
        ttk.Label(frame, text="Captura nueva: espacio guarda un par; Q termina y calcula; R borra\nlos pares de esta captura. Varía la inclinación y posición del tablero.\nEl resultado se guarda como candidato y se importa por separado.", wraplength=500).grid(row=6, column=0, columnspan=2, pady=12)
        def start(existing):
            try:
                values = {key: (int(var.get()) if key in {"cols", "rows"} else float(var.get())) for key, var in fields.items()}
                if any(not math.isfinite(value) for value in values.values()) or values["cols"] < 2 or values["rows"] < 2 or values["square-mm"] <= 0 or values["expected-baseline-mm"] < 0 or values["baseline-tolerance-mm"] <= 0:
                    raise ValueError("Revisa las esquinas y las medidas: deben ser finitas y positivas (separación admite 0).")
                left, right = (0, 1) if existing else (int(self.cam_left_var.get()), int(self.cam_right_var.get()))
                if not existing and (left < 0 or right < 0 or left == right):
                    raise ValueError("Selecciona dos índices de cámara diferentes y no negativos.")
                dataset = filedialog.askdirectory(parent=window, title="Dataset existente con carpetas left y right") if existing else None
                if existing and not dataset:
                    return
                self.jobs_dir.mkdir(parents=True, exist_ok=True)
                folder = Path(tempfile.mkdtemp(prefix="calibracion_estereo_", dir=self.jobs_dir))
                args = ["--dataset", dataset or str(folder / "dataset"), "--output", str(folder / "resultado"), "--width", "1920", "--height", "1080", "--left-camera", str(left), "--right-camera", str(right)]
                for key, value in values.items():
                    args += ["--" + key, str(value)]
                if existing:
                    args.append("--calibrate-only")
                else:
                    self.close_hardware()
                window.destroy()
                self.run_tool("calibrar_estereo_checkerboard.py", args, str(folder / "resultado"))
            except (ValueError, OSError) as exc:
                messagebox.showerror("Calibración estéreo", str(exc), parent=window)
        ttk.Button(frame, text="Capturar y calcular", command=lambda: start(False)).grid(row=7, column=0, sticky="ew")
        ttk.Button(frame, text="Calcular con imágenes existentes…", command=lambda: start(True)).grid(row=7, column=1, sticky="ew")

    def audit_lr(self):
        selected = filedialog.askopenfilename(parent=self.root, title="Selecciona un archivo *_lr_state.npy (requiere evidencia conservada)", filetypes=[("Estados LR", "*_lr_state.npy")])
        if not selected:
            return
        path = Path(selected)
        suffix = "_lr_state.npy"
        if not path.name.endswith(suffix):
            messagebox.showerror("Auditoría LR", "Selecciona un archivo terminado en _lr_state.npy.", parent=self.root)
            return
        args = ["--depth-dir", str(path.parent), "--stem", path.name[:-len(suffix)]]
        regional = filedialog.askdirectory(parent=self.root, title="Carpeta regional para comprobar contradicciones (Cancelar omite esta parte)")
        if regional:
            args += ["--regional-dir", regional]
        self.run_tool("auditar_evidencia_lr.py", args)
