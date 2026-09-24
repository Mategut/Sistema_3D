"""Medición de recursos por etapa; no cambia cálculos ni salidas científicas."""
import json
import os
import subprocess
import threading
import time


def run_measured(command, env, output_path, stage_id):
    samples = []
    stop = threading.Event()
    def monitor():
        try:
            import psutil
        except ImportError:
            return
        psutil.cpu_percent(None)
        while not stop.wait(2.0):
            sample = {"elapsed_s": time.perf_counter()-started,
                      "cpu_percent": psutil.cpu_percent(None),
                      "available_ram_bytes": psutil.virtual_memory().available}
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True,
                    timeout=2, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if result.returncode == 0:
                    sample["gpu_csv"] = result.stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                pass
            samples.append(sample)
    started = time.perf_counter()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    result = None
    try:
        result = subprocess.run(command, env=env, check=False)
        return result
    finally:
        stop.set()
        thread.join(timeout=3)
        report = {"stage": stage_id, "elapsed_s": time.perf_counter()-started,
                  "return_code": result.returncode if result else None,
                  "scope": "whole_computer_not_process_attribution", "samples": samples}
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"[RENDIMIENTO] No se pudo guardar telemetría: {exc}", flush=True)
