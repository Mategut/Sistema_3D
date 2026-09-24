"""Recursos compartidos y tareas independientes; sin modificar geometría."""

from __future__ import annotations
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

_CONTEXT = {}


def cpu_threads():
    """Obtiene el presupuesto de hilos de CPU, con un mínimo de uno."""
    return max(1, int(os.environ.get("SISTEMA3D_CPU_THREADS", os.cpu_count() or 1)))


def configurar(script):
    # Incluye fallos de importación en stdout, incluso antes de main().
    """Configura diagnóstico, salida inmediata y límites de hilos del proceso.

    Debe ejecutarse antes de importar las bibliotecas de cálculo para que sus
    runtimes reciban las variables de entorno. Los procesos hijos usan un solo
    hilo nativo; el presupuesto BLAS se configura de forma independiente.
    """

    def uncaught(exc_type, exc, tb):
        import traceback

        print(f"[ERROR] {os.path.basename(script)} | {exc_type.__name__}: {exc}", flush=True)
        traceback.print_exception(exc_type, exc, tb, file=sys.stdout)

    sys.excepthook = uncaught
    # Cada hijo configura su propio presupuesto antes de importar NumPy/Open3D.
    step = os.path.basename(script)[:2]
    internal = (
        1 if step in {"04", "15"} or mp.current_process().name != "MainProcess" else cpu_threads()
    )
    for key in ("OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(internal)
    # Miles de sistemas 6x6 no deben crear un equipo BLAS por cada ajuste.
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = os.environ.get("SISTEMA3D_BLAS_THREADS", "1")
    os.environ["PYTHONUNBUFFERED"] = "1"
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True, write_through=True)


def _initialize(context):
    """Instala el contexto privado del trabajador y limita OpenCV en procesos hijos."""
    global _CONTEXT
    _CONTEXT = _restore_context(context)
    if mp.current_process().name != "MainProcess":
        try:
            import cv2

            cv2.setNumThreads(1)
        except ImportError:
            pass


def contexto():
    """Devuelve el contexto instalado en el proceso trabajador actual."""
    return _CONTEXT


def _available_memory():
    """Consulta memoria disponible o usa un presupuesto de 2 GiB si falta psutil."""
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        return 2 * 1024**3  # presupuesto conservador si no hay psutil


def ejecutar_bloques(worker, context, outputs, count, block, label):
    """Ejecuta bloques independientes y escribe sus resultados en los arrays de salida.

    worker recibe (inicio, fin) y devuelve (inicio, fin, valores), donde valores
    es un diccionario de arrays. El número de procesos se limita por CPU, memoria
    y cantidad de tareas. Con spawn, el worker debe poder importarse en el hijo.
    La ventana de tareas pendientes evita acumular copias innecesarias.
    """
    import pickle

    block = max(1, int(block))
    tasks = [(a, min(a + block, count)) for a in range(0, count, block)]
    if not tasks:
        return
    requested = max(1, int(os.environ.get("SISTEMA3D_WORKERS", cpu_threads())))
    # Contexto privado por proceso + arrays temporales + importaciones.
    context_bytes = _serialized_size(context)
    memory_workers = max(1, int(_worker_memory_budget() / max(256 * 1024**2, 3 * context_bytes)))
    workers = min(requested, memory_workers, len(tasks))
    if count < 1500:
        workers = 1
    start = time.perf_counter()
    print(f"[CPU] {label}: {workers} procesos | {count:,} vértices", flush=True)
    completed = 0

    def collect(result):
        nonlocal completed
        a, b, values = result
        for name, value in values.items():
            outputs[name][a:b] = value
        completed += b - a

    if workers == 1:
        _initialize(context)
        try:
            for task in tasks:
                collect(worker(task))
                print(f"[CPU] {label}: {completed:,}/{count:,}", flush=True)
        finally:
            _initialize({})
    else:
        # Ventana limitada: no acumular resultados ni copias de la malla.
        with _mapped_context(context) as mapped, ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize,
            initargs=(mapped,),
        ) as pool:
            remaining = iter(tasks)
            pending = {
                pool.submit(worker, t)
                for t in [next(remaining, None) for _ in range(2 * workers)]
                if t is not None
            }
            while pending:
                done, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                if not done:
                    print(
                        f"[CPU] {label}: trabajando | {completed:,}/{count:,} | {time.perf_counter()-start:.0f} s",
                        flush=True,
                    )
                for future in done:
                    collect(future.result())
                    nxt = next(remaining, None)
                    if nxt is not None:
                        pending.add(pool.submit(worker, nxt))
                if done:
                    print(
                        f"[CPU] {label}: {completed:,}/{count:,} | {time.perf_counter()-start:.1f} s",
                        flush=True,
                    )
    print(f"[CPU] {label} completado en {time.perf_counter()-start:.1f} s", flush=True)


def ejecutar_items(worker, context, tasks, label, reserve_mb=512):
    """Tareas con ficheros propios; devuelve metadatos en orden de entrada."""
    import pickle

    tasks = list(tasks)
    if not tasks:
        return []
    requested = max(1, int(os.environ.get("SISTEMA3D_WORKERS", cpu_threads())))
    context_bytes = _serialized_size(context)
    budget = max(int(reserve_mb) * 1024**2, 3 * context_bytes)
    workers = min(requested, len(tasks), max(1, int(_worker_memory_budget() / budget)))
    start = time.perf_counter()
    print(f"[PROGRESO] {label}: {len(tasks)} tareas | {workers} procesos", flush=True)
    results = [None] * len(tasks)
    if workers == 1:
        _initialize(context)
        try:
            for index, task in enumerate(tasks):
                results[index] = worker(task)
                print(f"[PROGRESO] {label}: {index+1}/{len(tasks)} completadas", flush=True)
        finally:
            _initialize({})
    else:
        with _mapped_context(context) as mapped, ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize,
            initargs=(mapped,),
        ) as pool:
            remaining = iter(enumerate(tasks))
            pending = {}
            for _ in range(2 * workers):
                item = next(remaining, None)
                if item is not None:
                    index, task = item
                    pending[pool.submit(worker, task)] = index
            completed = 0
            while pending:
                done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                if not done:
                    print(
                        f"[PROGRESO] {label}: calculando | {completed}/{len(tasks)} | {time.perf_counter()-start:.0f} s",
                        flush=True,
                    )
                for future in done:
                    index = pending.pop(future)
                    try:
                        results[index] = future.result()
                    except BaseException:
                        print(f"[ERROR] {label}: falló la tarea {index+1}/{len(tasks)}", flush=True)
                        for other in pending:
                            other.cancel()
                        raise
                    completed += 1
                    item = next(remaining, None)
                    if item is not None:
                        idx, task = item
                        pending[pool.submit(worker, task)] = idx
                if done:
                    print(
                        f"[PROGRESO] {label}: {completed}/{len(tasks)} completadas | {time.perf_counter()-start:.1f} s",
                        flush=True,
                    )
    return results


def query_threads():
    """Devuelve un hilo para procesos hijos y el presupuesto de CPU para el principal."""
    return 1 if mp.current_process().name != "MainProcess" else cpu_threads()


def _worker_memory_budget():
    """Equipo dedicado: 85% de RAM disponible, dejando al menos 1 GiB.

    La reserva evita paginacion y protege al coordinador; no reserva aplicaciones.
    """
    available = _available_memory()
    return max(0, min(int(0.85 * available), available - 1024**3))


def _serialized_size(value):
    """Mide la serializacion sin construir otra copia gigante en memoria."""
    import pickle
    class Counter:
        def __init__(self):
            self.size = 0
        def write(self, data):
            size = memoryview(data).nbytes
            self.size += size
            return size
    counter = Counter()
    pickle.Pickler(counter, protocol=5).dump(value)
    return counter.size


def prefetch_items(loader, items):
    """Lee una tarea por adelantado; orden estable y excepciones propagadas."""
    from concurrent.futures import ThreadPoolExecutor
    iterator = iter(items)
    first = next(iterator, None)
    if first is None:
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="lectura") as pool:
        pending = pool.submit(loader, first)
        for item in iterator:
            value = pending.result()
            pending = pool.submit(loader, item)
            yield value
        yield pending.result()


class _MappedArray:
    """Descriptor privado; nunca se confunde con datos del usuario."""
    def __init__(self, path):
        self.path = path


def _restore_context(value, memo=None):
    if memo is None:
        memo = {}
    if isinstance(value, _MappedArray):
        import numpy as np
        # copy-on-write conserva la semántica privada si un worker modifica datos.
        if value.path not in memo:
            memo[value.path] = np.load(value.path, mmap_mode="c", allow_pickle=False)
        return memo[value.path]
    if isinstance(value, dict):
        return {k: _restore_context(v, memo) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_context(v, memo) for v in value]
    if isinstance(value, tuple):
        return tuple(_restore_context(v, memo) for v in value)
    return value


from contextlib import contextmanager

@contextmanager
def _mapped_context(context):
    """Una copia en disco por array grande; páginas compartidas entre workers.

    Los archivos viven hasta después de cerrar el pool. Arrays pequeños y
    objetos arbitrarios conservan el transporte original por pickle.
    """
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory(prefix="sistema3d_context_") as directory:
        memo = {}
        def pack(value):
            if isinstance(value, np.ndarray) and not value.dtype.hasobject and value.nbytes >= 1024**2:
                key = id(value)
                if key not in memo:
                    path = os.path.join(directory, str(len(memo)) + ".npy")
                    np.save(path, value, allow_pickle=False)
                    memo[key] = _MappedArray(path)
                return memo[key]
            if isinstance(value, dict):
                return {k: pack(v) for k, v in value.items()}
            if isinstance(value, list):
                return [pack(v) for v in value]
            if isinstance(value, tuple):
                return tuple(pack(v) for v in value)
            return value
        yield pack(context)
