#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Manifiestos angulares y correspondencia de capturas entre sesiones.

Cada sesión contiene 25 pares y comienza en la pose cero. Los 2055 pasos de
una vuelta se distribuyen mediante STEP_SEQUENCE; los ángulos físicos se
calculan a partir de los pasos acumulados. Las etiquetas de archivo conservan
el ángulo nominal y sirven para comprobar la captura, no para sustituir el
ángulo físico en la reconstrucción.
"""

from __future__ import annotations
import csv, json, re
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Tuple
import cv2
import numpy as np

STEPS_PER_REVOLUTION = 2055
MOVES_PER_REVOLUTION = 25
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
NOMINAL_STEP_DEG = 360.0 / MOVES_PER_REVOLUTION
assert len(STEP_SEQUENCE) == MOVES_PER_REVOLUTION and sum(STEP_SEQUENCE) == STEPS_PER_REVOLUTION
SESSION_RE = re.compile(r"^S(\d+)$", re.IGNORECASE)
VIEW_RE = re.compile(r"_V(\d+)", re.IGNORECASE)
ANGLE_RE = re.compile(r"_A(\d+)", re.IGNORECASE)


def natural_session_key(name: str) -> Tuple[int, str]:
    """Ordena sesiones S## por su número y deja otros nombres al final."""
    m = SESSION_RE.match(name.strip())
    return (int(m.group(1)), name) if m else (10**9, name)


def discover_sessions(root: Path, object_name: str) -> List[str]:
    """Encuentra y ordena las sesiones S## dentro de capturas/.

    El parámetro object_name se conserva por compatibilidad con los llamadores;
    la búsqueda se realiza exclusivamente dentro de la raíz del trabajo.
    """
    captures = root / "capturas"
    if not captures.is_dir():
        raise FileNotFoundError(f"No existe: {captures}")
    sessions = [
        p.name.upper() for p in captures.iterdir() if p.is_dir() and SESSION_RE.match(p.name)
    ]
    sessions.sort(key=natural_session_key)
    if not sessions:
        raise FileNotFoundError(f"No se encontraron sesiones S## en {captures}")
    return sessions


def parse_view_index(stem: str) -> int:
    """Convierte la etiqueta V001..V025 al índice de pose con base cero."""
    m = VIEW_RE.search(stem)
    if not m:
        raise ValueError(f"No se encontró V### en {stem}")
    return int(m.group(1)) - 1


def parse_filename_angle(stem: str) -> float:
    """Interpreta etiquetas A#### en décimas de grado y etiquetas cortas en grados."""
    m = ANGLE_RE.search(stem)
    if not m:
        raise ValueError(f"No se encontró A#### en {stem}")
    token = m.group(1)
    value = int(token)
    return float(value) / 10.0 if len(token) >= 4 else float(value)


def cumulative_steps_for_pose(pose_index: int) -> int:
    """Suma los pasos anteriores a la pose y valida su índice en la vuelta."""
    if not 0 <= pose_index < MOVES_PER_REVOLUTION:
        raise ValueError(f"pose_index inválido: {pose_index}")
    return int(sum(STEP_SEQUENCE[:pose_index]))


def physical_angle_for_pose(pose_index: int) -> float:
    """Convierte los pasos acumulados de una pose a grados físicos."""
    return float(cumulative_steps_for_pose(pose_index) * 360.0 / STEPS_PER_REVOLUTION)


def nominal_angle_for_pose(pose_index: int) -> float:
    """Calcula el ángulo nominal de la pose a intervalos uniformes."""
    return float(pose_index * NOMINAL_STEP_DEG)


def pair_session_images(root: Path, object_name: str, session: str) -> List[dict]:
    """Empareja izquierda/derecha y exige las 25 poses completas y ordenadas.

    object_name se conserva en la firma para mantener compatibilidad. Las rutas
    se resuelven a partir de root y session.
    """
    base = root / "capturas" / session
    left_dir = base / "izquierda"
    right_dir = base / "derecha"
    if not left_dir.is_dir() or not right_dir.is_dir():
        raise FileNotFoundError(f"Sesión incompleta: {base}")
    right_map = {p.name.replace("_R.png", ""): p for p in right_dir.glob("*.png")}
    records = []
    for left in left_dir.glob("*.png"):
        stem = left.name.replace("_L.png", "")
        right = right_map.get(stem)
        if right is None:
            continue
        pose = parse_view_index(stem)
        records.append(
            {
                "stem": stem,
                "view_index": pose,
                "view_number": pose + 1,
                "filename_angle_deg": parse_filename_angle(stem),
                "left_path": str(left),
                "right_path": str(right),
            }
        )
    records.sort(key=lambda r: r["view_index"])
    if len(records) != MOVES_PER_REVOLUTION:
        raise RuntimeError(f"{session}: se esperaban 25 pares y se encontraron {len(records)}.")
    expected = list(range(MOVES_PER_REVOLUTION))
    received = [r["view_index"] for r in records]
    if received != expected:
        raise RuntimeError(
            f"{session}: V001..V025 no están completos/ordenados. Índices encontrados: {received}"
        )
    return records


def build_angular_manifest(
    root: Path, object_name: str, sessions: Sequence[str], expected_sessions: int = 3
) -> dict:
    """Construye el manifiesto, valida las etiquetas y agrupa capturas por pose física."""
    if len(sessions) != expected_sessions:
        raise RuntimeError(
            f"Se esperaban exactamente {expected_sessions} sesiones para {object_name}; encontradas: {list(sessions)}"
        )
    all_records = []
    session_summaries = []
    for session_index, session in enumerate(sessions):
        views = pair_session_images(root, object_name, session)
        for view in views:
            pose = int(view["view_index"])
            nominal = nominal_angle_for_pose(pose)
            physical = physical_angle_for_pose(pose)
            cumulative = cumulative_steps_for_pose(pose)
            filename_angle = float(view["filename_angle_deg"])
            filename_error = filename_angle - nominal
            if abs(filename_error) > 0.051:
                raise RuntimeError(
                    f"{session}/{view['stem']}: etiqueta angular inesperada {filename_angle}°, esperada {nominal}°."
                )
            all_records.append(
                {
                    "session": session,
                    "session_index": session_index,
                    "stem": view["stem"],
                    "view_index": pose,
                    "view_number": pose + 1,
                    "pose_index": pose,
                    "filename_angle_deg": filename_angle,
                    "nominal_angle_deg": nominal,
                    "cumulative_steps": cumulative,
                    "physical_angle_deg": physical,
                    "angle_quantization_error_deg": physical - nominal,
                    "independent_capture_id": f"{session}:P{pose:02d}",
                    "left_path": view["left_path"],
                    "right_path": view["right_path"],
                }
            )
        session_summaries.append(
            {
                "session": session,
                "views": len(views),
                "start_pose_index": 0,
                "end_pose_index": 24,
                "start_physical_angle_deg": physical_angle_for_pose(0),
                "end_physical_angle_deg": physical_angle_for_pose(24),
                "steps_before_close": cumulative_steps_for_pose(24),
                "close_steps": STEP_SEQUENCE[24],
                "steps_per_revolution": STEPS_PER_REVOLUTION,
                "session_closes_to_origin": True,
            }
        )
    by_pose = defaultdict(list)
    for record in all_records:
        by_pose[int(record["pose_index"])].append(record)
    pose_groups = []
    for pose in range(MOVES_PER_REVOLUTION):
        members = by_pose[pose]
        pose_groups.append(
            {
                "pose_index": pose,
                "nominal_angle_deg": nominal_angle_for_pose(pose),
                "physical_angle_deg": physical_angle_for_pose(pose),
                "cumulative_steps": cumulative_steps_for_pose(pose),
                "independent_observations": len(members),
                "members": [
                    {
                        "session": r["session"],
                        "stem": r["stem"],
                        "filename_angle_deg": r["filename_angle_deg"],
                    }
                    for r in members
                ],
            }
        )
    return {
        "schema_version": 3,
        "capture_format": "V7.2_2055_steps_closed_sessions",
        "object": object_name,
        "sessions": list(sessions),
        "expected_sessions": expected_sessions,
        "steps_per_revolution": STEPS_PER_REVOLUTION,
        "moves_per_revolution": MOVES_PER_REVOLUTION,
        "step_sequence": list(STEP_SEQUENCE),
        "nominal_step_deg": NOMINAL_STEP_DEG,
        "maximum_angle_quantization_error_deg": max(
            abs(physical_angle_for_pose(i) - nominal_angle_for_pose(i))
            for i in range(MOVES_PER_REVOLUTION)
        ),
        "session_model": "Cada sesión comienza en pose 0 y ejecuta CLOSE después de V025. Por ello S01/S02/S03 comparten directamente el mismo pose_index.",
        "important_note": "filename_angle_deg es trazabilidad. Para registro y geometría usar physical_angle_deg derivado de pasos acumulados/2055.",
        "session_summaries": session_summaries,
        "records": all_records,
        "pose_groups": pose_groups,
    }


def save_manifest(output_dir: Path, manifest: dict):
    """Guarda el manifiesto en JSON y CSV y devuelve las rutas generadas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mapa_angular_multisesion.json"
    csv_path = output_dir / "mapa_angular_multisesion.csv"
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = [
        "session",
        "view_number",
        "stem",
        "pose_index",
        "filename_angle_deg",
        "nominal_angle_deg",
        "cumulative_steps",
        "physical_angle_deg",
        "angle_quantization_error_deg",
        "independent_capture_id",
        "left_path",
        "right_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        [w.writerow({k: r.get(k) for k in fields}) for r in manifest["records"]]
    return json_path, csv_path


def load_manifest(path: Path) -> dict:
    """Lee el manifiesto angular desde un archivo JSON UTF-8."""
    return json.loads(path.read_text(encoding="utf-8"))


def central_image_similarity(path_a: Path, path_b: Path) -> dict:
    """Compara intensidad y gradiente en la región central de dos imágenes.

    Devuelve available=False cuando alguna imagen no se puede leer. Esta medida
    es diagnóstica; no estima ni modifica poses de la plataforma.
    """
    a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return {"available": False, "score": None}
    size = (384, 216)
    a = cv2.resize(a, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(b, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    h, w = a.shape
    a = a[int(0.08 * h) : int(0.95 * h), int(0.20 * w) : int(0.80 * w)]
    b = b[int(0.08 * h) : int(0.95 * h), int(0.20 * w) : int(0.80 * w)]

    def corr(x, y):
        x = x.reshape(-1).astype(np.float64)
        y = y.reshape(-1).astype(np.float64)
        x -= np.mean(x)
        y -= np.mean(y)
        den = float(np.linalg.norm(x) * np.linalg.norm(y))
        return 0.0 if den < 1e-12 else float(np.clip(np.dot(x, y) / den, -1, 1))

    ga = cv2.magnitude(
        cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    )
    gb = cv2.magnitude(
        cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    )
    intensity = corr(a, b)
    gradient = corr(ga, gb)
    return {
        "available": True,
        "score": 0.45 * intensity + 0.55 * gradient,
        "intensity_correlation": intensity,
        "gradient_correlation": gradient,
        "mean_abs_difference": float(np.mean(np.abs(a - b))),
    }


def pairwise_iou(masks: Sequence[np.ndarray]) -> List[float]:
    """Calcula IoU entre pares de máscaras y omite pares con unión vacía."""
    values = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            a = masks[i] > 0
            b = masks[j] > 0
            union = int(np.count_nonzero(a | b))
            if union:
                values.append(float(np.count_nonzero(a & b) / union))
    return values


def finite_stats(values: np.ndarray) -> dict:
    """Resume valores finitos; devuelve count=0 y estadísticas nulas si no hay datos."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
            "mad": None,
        }
    med = float(np.median(arr))
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "median": med,
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mad": float(np.median(np.abs(arr - med))),
    }
