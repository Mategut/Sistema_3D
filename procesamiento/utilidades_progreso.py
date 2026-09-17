#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mensajes de progreso uniformes para los ejecutables del pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


def _argument_value(name: str) -> str | None:
    """Obtiene el valor que sigue a una opción de consola o None si no está disponible."""
    try:
        index = sys.argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(sys.argv):
        return None
    value = str(sys.argv[index + 1]).strip()
    return value or None


def informar_inicio(script_file: str, paso: str, descripcion: str) -> None:
    """Informa el ejecutable, su paso y la sesión cuando está disponible."""
    nombre = Path(script_file).name
    import os

    print(
        f"[RECURSOS] CPU nativa: {os.environ.get('OMP_NUM_THREADS', 'auto')} hilos | BLAS: {os.environ.get('OPENBLAS_NUM_THREADS', 'auto')} hilos",
        flush=True,
    )
    session = _argument_value("--session")
    parte = f" | Sesión {session.upper()}" if session else ""
    print(f"[PROGRESO] Se está ejecutando: {nombre}", flush=True)
    print(f"[PROGRESO] Paso {paso} | {descripcion}{parte}", flush=True)


import functools
import threading
import time
import traceback


def operacion(descripcion):
    """Actividad concreta y latido informativo; no estima porcentajes ficticios."""

    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            started = time.monotonic()
            stopped = threading.Event()
            label = f"{Path(fn.__code__.co_filename).name} | {descripcion}"
            print(f"[INICIO] {label}", flush=True)

            def heartbeat():
                while not stopped.wait(30):
                    print(f"[EN CURSO] {label} | {time.monotonic()-started:.0f} s", flush=True)

            monitor = threading.Thread(target=heartbeat, daemon=True)
            monitor.start()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                print(f"[ERROR] {label} | {type(exc).__name__}: {exc}", flush=True)
                raise
            else:
                print(f"[COMPLETADO] {label} | {time.monotonic()-started:.1f} s", flush=True)
                return result
            finally:
                stopped.set()
                monitor.join()

        return wrapped

    return decorate


def ejecutar_con_diagnostico(entry, script):
    """Conserva códigos de salida y publica el traceback en la consola del proceso."""
    name = Path(script).name
    print(f"[EJECUTANDO] {name}: {name[3:-3].replace(chr(95), chr(32))}", flush=True)
    try:
        return entry()
    except KeyboardInterrupt:
        print(f"[CANCELADO] {name}: ejecución interrumpida por el usuario.", flush=True)
        raise SystemExit(130)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            print(
                f"[ERROR] {name}: finalizó con salida {exc.code}. Revise el diagnóstico anterior y el resumen del paso.",
                flush=True,
            )
        raise
    except Exception as exc:
        print(f"[ERROR] {name} | {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc(file=sys.stdout)
        raise SystemExit(1)
