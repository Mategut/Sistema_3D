#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calibración estéreo con patrón de tablero y consenso de pose relativa.

Admite captura de pares o un conjunto existente. Detecta esquinas internas
en formato (N, 1, 2), calibra cada cámara y resuelve la ambigüedad de orden
de 180° entre las detecciones izquierda y derecha mediante PnP y consenso.
Los pares incompatibles se retiran antes de la calibración final.

La calidad combina reproyección, consistencia epipolar y baseline esperado.
La escala depende del lado físico del cuadrado, expresado en milímetros.
Las columnas y filas del patrón indican esquinas internas, no cuadrados.

Exporta stereo_initial.yaml, rectification_maps.npz y calibration_report.json,
además del estado de aceptación y las visualizaciones. Los mapas conservan
las claves map1x, map1y, map2x y map2y requeridas por el procesamiento.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080

DEFAULT_LEFT_CAMERA = 0
DEFAULT_RIGHT_CAMERA = 1

# ESQUINAS INTERNAS, no cuadrados.
DEFAULT_COLS = 6
DEFAULT_ROWS = 8

# Cambiar por el tamaño físico REAL del cuadrado.
DEFAULT_SQUARE_MM = 23.0

# Baseline de referencia del montaje, en milímetros.
DEFAULT_EXPECTED_BASELINE_MM = 81.0
DEFAULT_BASELINE_TOLERANCE_MM = 20.0

MIN_VALID_PAIRS = 15
RECOMMENDED_PAIRS = 25

# Filtro monocular por vista.
MAX_MONO_VIEW_RMSE_PX = 1.50

# Para consenso de pose relativa L->R durante la resolución
# de la ambigüedad 180°.
POSE_CLUSTER_ROT_DEG = 4.0
POSE_CLUSTER_TRANS_MM = 20.0

# Rechazo de un par si, incluso con la orientación elegida,
# queda demasiado lejos del consenso.
PAIR_MAX_ROT_FROM_CLUSTER_DEG = 7.0
PAIR_MAX_TRANS_FROM_CLUSTER_MM = 35.0

# Segunda limpieza usando geometría epipolar ya estimada.
PAIR_EPIPOLAR_MEDIAN_REJECT_PX = 2.5
PAIR_EPIPOLAR_P95_REJECT_PX = 5.0

# Calidad final.
STEREO_RMS_ACCEPT_PX = 1.50
STEREO_RMS_REJECT_PX = 3.00

EPIPOLAR_ACCEPT_MEDIAN_PX = 0.75
EPIPOLAR_ACCEPT_P95_PX = 2.00

EPIPOLAR_REJECT_MEDIAN_PX = 2.00
EPIPOLAR_REJECT_P95_PX = 4.00


# ============================================================
# UTILIDADES
# ============================================================


def ensure_dir(path: Path) -> None:
    """Crea el directorio y sus padres si todavía no existen."""
    path.mkdir(parents=True, exist_ok=True)


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """Abre la cámara con DirectShow o el backend disponible y configura la resolución."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )
    except Exception:
        pass

    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    except Exception:
        pass

    return cap


def camera_size(cap: cv2.VideoCapture) -> tuple[int, int]:
    """Devuelve el ancho y el alto que informa la cámara."""
    return (
        int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )


def detect_checkerboard(
    image: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    """Detecta las esquinas internas y devuelve sus coordenadas en formato (N, 1, 2)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE

    found, corners = cv2.findChessboardCornersSB(
        gray,
        pattern_size,
        flags=flags,
    )

    if not found or corners is None:
        return False, None

    # Formato canónico compatible con OpenCV 4.x / 5.x.
    corners = np.asarray(
        corners,
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    return True, corners


def build_object_points(
    cols: int,
    rows: int,
    square_mm: float,
) -> np.ndarray:
    """Genera las esquinas del patrón en su plano Z=0, con escala en milímetros."""
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj *= float(square_mm)
    return obj


def reprojection_rmse(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> float:
    """Calcula el error cuadrático medio de reproyección expresado como RMSE en píxeles."""
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64),
        np.asarray(rvec, dtype=np.float64),
        np.asarray(tvec, dtype=np.float64),
        np.asarray(K, dtype=np.float64),
        np.asarray(D, dtype=np.float64),
    )

    observed = np.asarray(
        image_points,
        dtype=np.float64,
    ).reshape(-1, 2)

    predicted = np.asarray(
        projected,
        dtype=np.float64,
    ).reshape(-1, 2)

    residual = observed - predicted

    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def calculate_view_errors(
    object_points,
    image_points,
    rvecs,
    tvecs,
    K,
    D,
) -> np.ndarray:
    """Calcula el RMSE de reproyección de cada vista monocular."""
    errors = []

    for obj, img, rv, tv in zip(
        object_points,
        image_points,
        rvecs,
        tvecs,
    ):
        errors.append(reprojection_rmse(obj, img, rv, tv, K, D))

    return np.asarray(errors, dtype=np.float64)


def rotation_distance_deg(
    R_a: np.ndarray,
    R_b: np.ndarray,
) -> float:
    """Mide la separación angular en grados entre dos matrices de rotación."""
    R_delta = np.asarray(R_a) @ np.asarray(R_b).T

    value = (float(np.trace(R_delta)) - 1.0) / 2.0

    value = float(np.clip(value, -1.0, 1.0))

    return float(np.degrees(np.arccos(value)))


def invert_board_order_180(
    corners: np.ndarray,
) -> np.ndarray:
    """
    Para un checkerboard rectangular sin marcador, la ambigüedad
    relevante entre dos detecciones es normalmente 180°.

    Si OpenCV comienza por la esquina opuesta en una de las cámaras,
    el vector completo queda invertido.
    """
    pts = np.asarray(
        corners,
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    return pts[::-1].copy()


def corner_variants(
    corners: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    """Genera las variantes de orden original e invertido de las esquinas."""
    pts = np.asarray(
        corners,
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    return [
        ("identity", pts.copy()),
        ("rot180", invert_board_order_180(pts)),
    ]


# ============================================================
# CAPTURA
# ============================================================


def capture_pairs(
    dataset_dir: Path,
    left_camera: int,
    right_camera: int,
    width: int,
    height: int,
    pattern_size: tuple[int, int],
) -> None:
    """Captura interactivamente pares del patrón en los directorios left y right."""
    left_dir = dataset_dir / "left"
    right_dir = dataset_dir / "right"

    ensure_dir(left_dir)
    ensure_dir(right_dir)

    cap_l = open_camera(left_camera, width, height)
    cap_r = open_camera(right_camera, width, height)

    size_l = camera_size(cap_l)
    size_r = camera_size(cap_r)

    print(f"\nCámara izquierda: {size_l}")
    print(f"Cámara derecha:   {size_r}")

    if size_l != size_r:
        cap_l.release()
        cap_r.release()
        raise RuntimeError("Las cámaras entregan resoluciones diferentes.")

    print("\nCONTROLES")
    print("SPACE : guardar par")
    print("Q     : terminar captura")
    print("R     : borrar dataset capturado\n")

    pair_index = len(sorted(left_dir.glob("left_*.png")))

    try:
        while True:
            ok_l, frame_l = cap_l.read()
            ok_r, frame_r = cap_r.read()

            if not ok_l or not ok_r:
                continue

            found_l, corners_l = detect_checkerboard(
                frame_l,
                pattern_size,
            )
            found_r, corners_r = detect_checkerboard(
                frame_r,
                pattern_size,
            )

            vis_l = frame_l.copy()
            vis_r = frame_r.copy()

            if found_l:
                cv2.drawChessboardCorners(
                    vis_l,
                    pattern_size,
                    corners_l,
                    True,
                )

            if found_r:
                cv2.drawChessboardCorners(
                    vis_r,
                    pattern_size,
                    corners_r,
                    True,
                )

            text = (
                f"Par {pair_index:03d} | "
                f"L={'OK' if found_l else 'NO'} | "
                f"R={'OK' if found_r else 'NO'}"
            )

            cv2.putText(
                vis_l,
                text,
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0) if found_l and found_r else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            scale = 900.0 / vis_l.shape[1]

            prev_l = cv2.resize(
                vis_l,
                None,
                fx=scale,
                fy=scale,
            )
            prev_r = cv2.resize(
                vis_r,
                None,
                fx=scale,
                fy=scale,
            )

            cv2.imshow(
                "Calibracion estereo",
                np.hstack((prev_l, prev_r)),
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                shutil.rmtree(
                    left_dir,
                    ignore_errors=True,
                )
                shutil.rmtree(
                    right_dir,
                    ignore_errors=True,
                )
                ensure_dir(left_dir)
                ensure_dir(right_dir)
                pair_index = 0
                print("Dataset reiniciado.")
                continue

            if key == 32:
                if not (found_l and found_r):
                    print("No guardado: el checkerboard no fue " "detectado en ambas cámaras.")
                    continue

                cv2.imwrite(
                    str(left_dir / f"left_{pair_index:03d}.png"),
                    frame_l,
                )
                cv2.imwrite(
                    str(right_dir / f"right_{pair_index:03d}.png"),
                    frame_r,
                )

                pair_index += 1
                print(f"Par {pair_index:03d} guardado.")

    finally:
        cap_l.release()
        cap_r.release()
        cv2.destroyAllWindows()


# ============================================================
# DATASET
# ============================================================


def load_dataset(
    dataset_dir: Path,
    pattern_size: tuple[int, int],
    square_mm: float,
):
    """Carga los pares y reúne detecciones, puntos físicos y nombres para calibrar."""
    left_files = sorted((dataset_dir / "left").glob("*.png"))
    right_files = sorted((dataset_dir / "right").glob("*.png"))

    if not left_files or not right_files:
        raise RuntimeError("No se encontraron imágenes de calibración.")

    if len(left_files) != len(right_files):
        raise RuntimeError("La cantidad de imágenes izquierda/derecha " "es diferente.")

    obj_template = build_object_points(
        pattern_size[0],
        pattern_size[1],
        square_mm,
    )

    object_points = []
    image_l = []
    image_r = []
    names = []

    image_size = None

    print("\n===== DETECCIÓN DEL CHECKERBOARD =====")

    for f_l, f_r in zip(
        left_files,
        right_files,
    ):
        img_l = cv2.imread(str(f_l))
        img_r = cv2.imread(str(f_r))

        if img_l is None or img_r is None:
            print(f"FAIL {f_l.name}: lectura")
            continue

        if img_l.shape[:2] != img_r.shape[:2]:
            print(f"FAIL {f_l.name}: tamaño L/R distinto")
            continue

        h, w = img_l.shape[:2]

        if image_size is None:
            image_size = (w, h)

        if image_size != (w, h):
            print(f"FAIL {f_l.name}: resolución inconsistente")
            continue

        ok_l, c_l = detect_checkerboard(
            img_l,
            pattern_size,
        )
        ok_r, c_r = detect_checkerboard(
            img_r,
            pattern_size,
        )

        if not (ok_l and ok_r):
            print(f"FAIL {f_l.name}")
            continue

        object_points.append(obj_template.copy())
        image_l.append(c_l)
        image_r.append(c_r)
        names.append(f_l.stem)

        print(f"OK   {f_l.name}")

    if image_size is None:
        raise RuntimeError("No se pudo determinar la resolución.")

    if len(names) < MIN_VALID_PAIRS:
        raise RuntimeError(
            f"Solo hay {len(names)} pares válidos. " f"Se requieren al menos {MIN_VALID_PAIRS}."
        )

    print(f"\nPares válidos: {len(names)}")

    return (
        object_points,
        image_l,
        image_r,
        names,
        image_size,
    )


# ============================================================
# CALIBRACIÓN MONOCULAR
# ============================================================


def mono_calibrate(
    object_points,
    image_points,
    image_size,
):
    """Estima los parámetros intrínsecos y las poses del patrón de una cámara."""
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    errors = calculate_view_errors(
        object_points,
        image_points,
        rvecs,
        tvecs,
        K,
        D,
    )

    return (
        float(rms),
        K,
        D,
        rvecs,
        tvecs,
        errors,
    )


# ============================================================
# RESOLUCIÓN ROBUSTA DEL ORDEN DE ESQUINAS
# ============================================================


@dataclass
class PoseCandidate:
    """Hipótesis de pose relativa estéreo asociada al orden de esquinas de un par."""

    pair_index: int
    left_variant: str
    right_variant: str
    corners_l: np.ndarray
    corners_r: np.ndarray
    R_lr: np.ndarray
    t_lr: np.ndarray
    reproj_l: float
    reproj_r: float


def solve_board_pose(
    obj: np.ndarray,
    img: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
):
    """Estima mediante PnP la transformación del patrón a la cámara."""
    ok, rvec, tvec = cv2.solvePnP(
        np.asarray(obj, dtype=np.float64),
        np.asarray(img, dtype=np.float64).reshape(-1, 1, 2),
        np.asarray(K, dtype=np.float64),
        np.asarray(D, dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)

    err = reprojection_rmse(
        obj,
        img,
        rvec,
        tvec,
        K,
        D,
    )

    return R, tvec.reshape(3, 1), err


def relative_pose_from_board_poses(
    R_l: np.ndarray,
    t_l: np.ndarray,
    R_r: np.ndarray,
    t_r: np.ndarray,
):
    """
    Board -> Left:
        X_L = R_l X_B + t_l

    Board -> Right:
        X_R = R_r X_B + t_r

    Por tanto:
        X_R = R_lr X_L + t_lr
    """
    R_lr = R_r @ R_l.T
    t_lr = t_r - R_lr @ t_l

    return R_lr, t_lr


def build_pose_candidates(
    object_points,
    image_l,
    image_r,
    K1,
    D1,
    K2,
    D2,
) -> list[PoseCandidate]:
    """Construye hipótesis de pose relativa para las variantes de orden del patrón."""
    candidates: list[PoseCandidate] = []

    for i, (obj, raw_l, raw_r) in enumerate(zip(object_points, image_l, image_r)):
        for name_l, c_l in corner_variants(raw_l):
            pose_l = solve_board_pose(obj, c_l, K1, D1)

            if pose_l is None:
                continue

            R_l, t_l, err_l = pose_l

            for name_r, c_r in corner_variants(raw_r):
                pose_r = solve_board_pose(obj, c_r, K2, D2)

                if pose_r is None:
                    continue

                R_r, t_r, err_r = pose_r

                R_lr, t_lr = relative_pose_from_board_poses(R_l, t_l, R_r, t_r)

                candidates.append(
                    PoseCandidate(
                        pair_index=i,
                        left_variant=name_l,
                        right_variant=name_r,
                        corners_l=c_l,
                        corners_r=c_r,
                        R_lr=R_lr,
                        t_lr=t_lr,
                        reproj_l=err_l,
                        reproj_r=err_r,
                    )
                )

    return candidates


def candidate_distance(
    a: PoseCandidate,
    b: PoseCandidate,
):
    """Devuelve la diferencia angular en grados y la distancia de traslación en milímetros."""
    rot = rotation_distance_deg(
        a.R_lr,
        b.R_lr,
    )

    trans = float(np.linalg.norm(a.t_lr.reshape(3) - b.t_lr.reshape(3)))

    return rot, trans


def find_pose_cluster_medoid(
    candidates: list[PoseCandidate],
) -> PoseCandidate:
    """Selecciona la hipótesis con mayor respaldo de pares y menor coste de consenso."""
    if not candidates:
        raise RuntimeError("No se pudieron generar candidatos de pose.")

    best = None
    best_pair_support = -1
    best_cost = float("inf")

    for center in candidates:
        supported_pairs = set()
        costs = []

        for other in candidates:
            rot, trans = candidate_distance(
                center,
                other,
            )

            if rot <= POSE_CLUSTER_ROT_DEG and trans <= POSE_CLUSTER_TRANS_MM:
                supported_pairs.add(other.pair_index)

                costs.append(rot + trans / 5.0)

        support = len(supported_pairs)

        cost = float(np.median(costs)) if costs else float("inf")

        if support > best_pair_support or (support == best_pair_support and cost < best_cost):
            best = center
            best_pair_support = support
            best_cost = cost

    if best is None:
        raise RuntimeError("No se encontró consenso de pose L->R.")

    print("\n===== CONSENSO DE POSE L->R =====")
    print(f"Vistas soportadas por el clúster: " f"{best_pair_support}")
    print("Baseline PnP del centro del clúster: " f"{np.linalg.norm(best.t_lr):.3f} mm")

    return best


def choose_consistent_corner_orders(
    object_points,
    image_l,
    image_r,
    names,
    K1,
    D1,
    K2,
    D2,
):
    """Selecciona órdenes de esquinas compatibles con el consenso de pose estéreo."""
    candidates = build_pose_candidates(
        object_points,
        image_l,
        image_r,
        K1,
        D1,
        K2,
        D2,
    )

    medoid = find_pose_cluster_medoid(candidates)

    chosen = []
    rejected_indices = []

    print("\n===== ORDEN L/R POR PAR =====")

    for pair_index in range(len(names)):
        pair_candidates = [c for c in candidates if c.pair_index == pair_index]

        if not pair_candidates:
            rejected_indices.append(pair_index)
            print(f"{names[pair_index]:20s} " "REJECT: sin candidatos")
            continue

        scored = []

        for c in pair_candidates:
            rot, trans = candidate_distance(
                medoid,
                c,
            )

            reproj = max(
                c.reproj_l,
                c.reproj_r,
            )

            score = rot + trans / 5.0 + reproj * 0.25

            scored.append((score, rot, trans, c))

        scored.sort(key=lambda x: x[0])

        _, rot, trans, best = scored[0]

        if rot > PAIR_MAX_ROT_FROM_CLUSTER_DEG or trans > PAIR_MAX_TRANS_FROM_CLUSTER_MM:
            rejected_indices.append(pair_index)
            print(f"{names[pair_index]:20s} " f"REJECT | ΔR={rot:.2f}° " f"ΔT={trans:.2f} mm")
            continue

        chosen.append(best)

        print(
            f"{names[pair_index]:20s} "
            f"L={best.left_variant:8s} "
            f"R={best.right_variant:8s} | "
            f"ΔR={rot:.2f}° "
            f"ΔT={trans:.2f} mm"
        )

    if len(chosen) < MIN_VALID_PAIRS:
        raise RuntimeError(
            "Después de resolver el orden de corners " f"solo quedaron {len(chosen)} pares."
        )

    return chosen, rejected_indices


# ============================================================
# ESTÉREO
# ============================================================


def run_stereo_calibration(
    object_points,
    image_l,
    image_r,
    K1,
    D1,
    K2,
    D2,
    image_size,
):
    """Estima la geometría relativa del par con los intrínsecos suministrados."""
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        300,
        1e-8,
    )

    (
        stereo_rms,
        K1_out,
        D1_out,
        K2_out,
        D2_out,
        R,
        T,
        E,
        F,
    ) = cv2.stereoCalibrate(
        object_points,
        image_l,
        image_r,
        K1,
        D1,
        K2,
        D2,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    return {
        "stereo_rms": float(stereo_rms),
        "K1": K1_out,
        "D1": D1_out,
        "K2": K2_out,
        "D2": D2_out,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "baseline_mm": float(np.linalg.norm(T)),
    }


def rectify(stereo: dict, image_size):
    """Calcula la rectificación estéreo y los mapas de remuestreo de ambas cámaras."""
    (
        R1,
        R2,
        P1,
        P2,
        Q,
        roi1,
        roi2,
    ) = cv2.stereoRectify(
        stereo["K1"],
        stereo["D1"],
        stereo["K2"],
        stereo["D2"],
        image_size,
        stereo["R"],
        stereo["T"],
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        stereo["K1"],
        stereo["D1"],
        R1,
        P1,
        image_size,
        cv2.CV_32FC1,
    )

    map2x, map2y = cv2.initUndistortRectifyMap(
        stereo["K2"],
        stereo["D2"],
        R2,
        P2,
        image_size,
        cv2.CV_32FC1,
    )

    stereo.update(
        {
            "R1": R1,
            "R2": R2,
            "P1": P1,
            "P2": P2,
            "Q": Q,
            "roi1": roi1,
            "roi2": roi2,
            "map1x": map1x,
            "map1y": map1y,
            "map2x": map2x,
            "map2y": map2y,
        }
    )

    return stereo


def epipolar_per_pair(
    stereo: dict,
    image_l,
    image_r,
    names,
):
    """Mide las discrepancias verticales rectificadas por par y en conjunto."""
    rows = []
    all_dy = []

    for name, pts_l, pts_r in zip(
        names,
        image_l,
        image_r,
    ):
        rect_l = cv2.undistortPoints(
            np.asarray(
                pts_l,
                dtype=np.float64,
            ).reshape(-1, 1, 2),
            stereo["K1"],
            stereo["D1"],
            R=stereo["R1"],
            P=stereo["P1"],
        ).reshape(-1, 2)

        rect_r = cv2.undistortPoints(
            np.asarray(
                pts_r,
                dtype=np.float64,
            ).reshape(-1, 1, 2),
            stereo["K2"],
            stereo["D2"],
            R=stereo["R2"],
            P=stereo["P2"],
        ).reshape(-1, 2)

        dy = np.abs(rect_l[:, 1] - rect_r[:, 1])

        all_dy.extend(dy.tolist())

        rows.append(
            {
                "name": name,
                "median_abs_dy_px": float(np.median(dy)),
                "p95_abs_dy_px": float(np.percentile(dy, 95)),
                "max_abs_dy_px": float(np.max(dy)),
            }
        )

    all_dy = np.asarray(
        all_dy,
        dtype=np.float64,
    )

    return rows, {
        "median_abs_dy_px": float(np.median(all_dy)),
        "mean_abs_dy_px": float(np.mean(all_dy)),
        "p95_abs_dy_px": float(np.percentile(all_dy, 95)),
        "max_abs_dy_px": float(np.max(all_dy)),
    }


def remove_stereo_outliers_and_recalibrate(
    stereo,
    object_points,
    image_l,
    image_r,
    names,
    K1,
    D1,
    K2,
    D2,
    image_size,
):
    """Retira pares con error epipolar excesivo y vuelve a estimar la calibración."""
    stereo = rectify(
        stereo,
        image_size,
    )

    per_pair, _ = epipolar_per_pair(
        stereo,
        image_l,
        image_r,
        names,
    )

    keep = []

    print("\n===== RESIDUAL EPIPOLAR POR PAR =====")

    for row in per_pair:
        ok = (
            row["median_abs_dy_px"] <= PAIR_EPIPOLAR_MEDIAN_REJECT_PX
            and row["p95_abs_dy_px"] <= PAIR_EPIPOLAR_P95_REJECT_PX
        )

        keep.append(ok)

        print(
            f"{row['name']:20s} "
            f"med={row['median_abs_dy_px']:.3f}px "
            f"P95={row['p95_abs_dy_px']:.3f}px "
            f"{'KEEP' if ok else 'REJECT'}"
        )

    if all(keep):
        return (
            stereo,
            object_points,
            image_l,
            image_r,
            names,
        )

    if sum(keep) < MIN_VALID_PAIRS:
        print("\nNo se aplicará el segundo filtro porque " "dejaría muy pocos pares.")
        return (
            stereo,
            object_points,
            image_l,
            image_r,
            names,
        )

    print(f"\nRecalibrando después del filtro estéreo: " f"{sum(keep)}/{len(keep)} pares.")

    obj2 = [x for x, k in zip(object_points, keep) if k]
    l2 = [x for x, k in zip(image_l, keep) if k]
    r2 = [x for x, k in zip(image_r, keep) if k]
    n2 = [x for x, k in zip(names, keep) if k]

    stereo2 = run_stereo_calibration(
        obj2,
        l2,
        r2,
        K1,
        D1,
        K2,
        D2,
        image_size,
    )

    stereo2 = rectify(
        stereo2,
        image_size,
    )

    return (
        stereo2,
        obj2,
        l2,
        r2,
        n2,
    )


# ============================================================
# CALIDAD FINAL
# ============================================================


def classify_quality(
    stereo: dict,
    epipolar: dict,
    expected_baseline_mm: float | None,
    baseline_tolerance_mm: float,
):
    """Evalúa RMS, error epipolar y baseline y devuelve calidad y motivos."""
    reject_reasons = []
    warnings = []

    stereo_rms = float(stereo["stereo_rms"])
    baseline = float(stereo["baseline_mm"])

    med = float(epipolar["median_abs_dy_px"])
    p95 = float(epipolar["p95_abs_dy_px"])

    if stereo_rms > STEREO_RMS_REJECT_PX:
        reject_reasons.append(f"RMS estéreo demasiado alto: " f"{stereo_rms:.3f}px")
    elif stereo_rms > STEREO_RMS_ACCEPT_PX:
        warnings.append(f"RMS estéreo elevado: " f"{stereo_rms:.3f}px")

    if med > EPIPOLAR_REJECT_MEDIAN_PX or p95 > EPIPOLAR_REJECT_P95_PX:
        reject_reasons.append(
            f"Residual epipolar excesivo: " f"mediana={med:.3f}px, " f"P95={p95:.3f}px"
        )
    elif not (med <= EPIPOLAR_ACCEPT_MEDIAN_PX and p95 <= EPIPOLAR_ACCEPT_P95_PX):
        warnings.append(
            f"Residual epipolar con margen: " f"mediana={med:.3f}px, " f"P95={p95:.3f}px"
        )

    if expected_baseline_mm is not None:
        delta = abs(baseline - expected_baseline_mm)

        if delta > baseline_tolerance_mm:
            reject_reasons.append(
                f"Baseline incompatible con el montaje: "
                f"{baseline:.3f}mm vs esperado "
                f"{expected_baseline_mm:.3f}±"
                f"{baseline_tolerance_mm:.3f}mm"
            )

    if reject_reasons:
        quality = "rejected"
    elif warnings:
        quality = "warning"
    else:
        quality = "accepted"

    return quality, reject_reasons, warnings


# ============================================================
# GUARDADO COMPATIBLE CON EL PIPELINE
# ============================================================


def write_stereo_yaml(
    path: Path,
    stereo: dict,
    rms_left: float,
    rms_right: float,
    epipolar: dict,
    quality: str,
):
    """Guarda las matrices y los diagnósticos con las claves esperadas por el pipeline."""
    fs = cv2.FileStorage(
        str(path),
        cv2.FILE_STORAGE_WRITE,
    )

    if not fs.isOpened():
        raise RuntimeError(f"No se pudo crear {path}")

    # Claves de calibración consumidas por el pipeline.
    fs.write(
        "rms_stereo",
        float(stereo["stereo_rms"]),
    )
    fs.write(
        "baseline_mm",
        float(stereo["baseline_mm"]),
    )

    for key in (
        "R",
        "T",
        "E",
        "F",
        "R1",
        "R2",
        "P1",
        "P2",
        "Q",
    ):
        fs.write(key, stereo[key])

    # Diagnósticos adicionales; no interfieren con el pipeline.
    fs.write(
        "rms_left",
        float(rms_left),
    )
    fs.write(
        "rms_right",
        float(rms_right),
    )
    fs.write(
        "epipolar_median_abs_dy_px",
        float(epipolar["median_abs_dy_px"]),
    )
    fs.write(
        "epipolar_p95_abs_dy_px",
        float(epipolar["p95_abs_dy_px"]),
    )
    fs.write(
        "epipolar_max_abs_dy_px",
        float(epipolar["max_abs_dy_px"]),
    )
    fs.write(
        "epipolar_quality",
        quality,
    )

    fs.release()


def save_outputs(
    output_dir: Path,
    stereo: dict,
    rms_left: float,
    rms_right: float,
    report: dict,
):
    """Escribe YAML, mapas NPZ, informe JSON y estado de aceptación de la calibración."""
    ensure_dir(output_dir)

    yaml_path = output_dir / "stereo_initial.yaml"

    npz_path = output_dir / "rectification_maps.npz"

    json_path = output_dir / "calibration_report.json"

    write_stereo_yaml(
        yaml_path,
        stereo,
        rms_left,
        rms_right,
        report["epipolar"],
        report["quality"],
    )

    # Claves de mapas consumidas por 02_estimar_profundidad_crestereo.py.
    np.savez_compressed(
        npz_path,
        map1x=np.asarray(
            stereo["map1x"],
            dtype=np.float32,
        ),
        map1y=np.asarray(
            stereo["map1y"],
            dtype=np.float32,
        ),
        map2x=np.asarray(
            stereo["map2x"],
            dtype=np.float32,
        ),
        map2y=np.asarray(
            stereo["map2y"],
            dtype=np.float32,
        ),
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    status_file = output_dir / (
        "CALIBRACION_ACEPTADA.txt"
        if report["quality"] == "accepted"
        else "CALIBRACION_NO_ACEPTADA.txt"
    )

    status_file.write_text(
        (
            f"quality={report['quality']}\n"
            + "\n".join(report["reject_reasons"] + report["warnings"])
        ),
        encoding="utf-8",
    )

    return yaml_path, npz_path, json_path


# ============================================================
# PREVIEW
# ============================================================


def create_preview(
    dataset_dir: Path,
    output_dir: Path,
    stereo: dict,
):
    """Genera una vista diagnóstica de la rectificación con las imágenes del conjunto."""
    left_files = sorted((dataset_dir / "left").glob("*.png"))
    right_files = sorted((dataset_dir / "right").glob("*.png"))

    if not left_files or not right_files:
        return None

    idx = (
        min(
            len(left_files),
            len(right_files),
        )
        // 2
    )

    img_l = cv2.imread(str(left_files[idx]))
    img_r = cv2.imread(str(right_files[idx]))

    if img_l is None or img_r is None:
        return None

    rect_l = cv2.remap(
        img_l,
        stereo["map1x"],
        stereo["map1y"],
        cv2.INTER_LINEAR,
    )

    rect_r = cv2.remap(
        img_r,
        stereo["map2x"],
        stereo["map2y"],
        cv2.INTER_LINEAR,
    )

    joined = np.hstack((rect_l, rect_r))

    h, w = joined.shape[:2]

    for y in range(40, h, 40):
        cv2.line(
            joined,
            (0, y),
            (w - 1, y),
            (0, 255, 0),
            1,
        )

    out = output_dir / "preview_rectificacion.png"

    cv2.imwrite(
        str(out),
        joined,
    )

    return out


# ============================================================
# MAIN
# ============================================================


def main():
    """Coordina captura opcional, calibración, filtrado de pares y exportación de resultados."""
    print(
        "[PROGRESO] Se está ejecutando: 01_calibrar_estereo_checkerboard.py",
        flush=True,
    )
    print("[PROGRESO] Preparando calibración estéreo checkerboard", flush=True)
    parser = argparse.ArgumentParser(
        description=("Calibración estéreo robusta con checkerboard " "para Sistema 3D Integrado.")
    )

    parser.add_argument(
        "--left-camera",
        type=int,
        default=DEFAULT_LEFT_CAMERA,
    )
    parser.add_argument(
        "--right-camera",
        type=int,
        default=DEFAULT_RIGHT_CAMERA,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=DEFAULT_COLS,
        help="Esquinas internas horizontales.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help="Esquinas internas verticales.",
    )
    parser.add_argument(
        "--square-mm",
        type=float,
        default=DEFAULT_SQUARE_MM,
        help="Lado físico real de cada cuadro.",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("calibracion_checkerboard"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calibracion_estereo_candidata"),
    )

    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Usar imágenes ya capturadas.",
    )

    parser.add_argument(
        "--expected-baseline-mm",
        type=float,
        default=DEFAULT_EXPECTED_BASELINE_MM,
        help=("Baseline físico aproximado del montaje. " "Usa 0 para desactivar la comprobación."),
    )

    parser.add_argument(
        "--baseline-tolerance-mm",
        type=float,
        default=DEFAULT_BASELINE_TOLERANCE_MM,
    )

    args = parser.parse_args()

    if args.cols <= 1 or args.rows <= 1:
        raise ValueError("cols/rows deben representar esquinas internas.")

    if args.square_mm <= 0:
        raise ValueError("square-mm debe ser > 0.")

    expected_baseline = None if args.expected_baseline_mm <= 0 else float(args.expected_baseline_mm)

    pattern_size = (
        args.cols,
        args.rows,
    )

    print("\n==============================================")
    print(" CALIBRACIÓN ESTÉREO ROBUSTA V2.1")
    print("==============================================")
    print(f"Checkerboard : " f"{args.cols} x {args.rows} esquinas")
    print(f"Cuadro       : " f"{args.square_mm:.3f} mm")
    print(f"Resolución   : " f"{args.width} x {args.height}")
    print(f"Baseline ref.: " f"{expected_baseline if expected_baseline else 'desactivado'}")
    print("==============================================")

    if not args.calibrate_only:
        capture_pairs(
            args.dataset,
            args.left_camera,
            args.right_camera,
            args.width,
            args.height,
            pattern_size,
        )

    (
        object_points,
        raw_l,
        raw_r,
        names,
        image_size,
    ) = load_dataset(
        args.dataset,
        pattern_size,
        args.square_mm,
    )

    # --------------------------------------------------------
    # 1. MONOCULAR INICIAL
    # --------------------------------------------------------

    print("\n===== CALIBRACIÓN MONOCULAR IZQUIERDA =====")

    (
        rms_l,
        K1,
        D1,
        rvecs_l,
        tvecs_l,
        errors_l,
    ) = mono_calibrate(
        object_points,
        raw_l,
        image_size,
    )

    print(f"RMS izquierda: {rms_l:.6f} px")

    print("\n===== CALIBRACIÓN MONOCULAR DERECHA =====")

    (
        rms_r,
        K2,
        D2,
        rvecs_r,
        tvecs_r,
        errors_r,
    ) = mono_calibrate(
        object_points,
        raw_r,
        image_size,
    )

    print(f"RMS derecha: {rms_r:.6f} px")

    mono_keep = (
        np.maximum(
            errors_l,
            errors_r,
        )
        <= MAX_MONO_VIEW_RMSE_PX
    )

    print("\n===== FILTRO MONOCULAR =====")

    for name, el, er, keep in zip(
        names,
        errors_l,
        errors_r,
        mono_keep,
    ):
        print(f"{name:20s} " f"L={el:.3f}px " f"R={er:.3f}px " f"{'KEEP' if keep else 'REJECT'}")

    if int(np.sum(mono_keep)) < MIN_VALID_PAIRS:
        raise RuntimeError("El filtro monocular deja menos del mínimo " "de pares válidos.")

    obj_m = [x for x, k in zip(object_points, mono_keep) if k]
    l_m = [x for x, k in zip(raw_l, mono_keep) if k]
    r_m = [x for x, k in zip(raw_r, mono_keep) if k]
    names_m = [x for x, k in zip(names, mono_keep) if k]

    # Recalibración intrínseca con pares monoculares buenos.
    (
        rms_l,
        K1,
        D1,
        _,
        _,
        _,
    ) = mono_calibrate(
        obj_m,
        l_m,
        image_size,
    )

    (
        rms_r,
        K2,
        D2,
        _,
        _,
        _,
    ) = mono_calibrate(
        obj_m,
        r_m,
        image_size,
    )

    print(f"\nRMS mono final L/R: " f"{rms_l:.6f} / {rms_r:.6f} px")

    # --------------------------------------------------------
    # 2. RESOLVER ORDEN 180° L/R
    # --------------------------------------------------------

    chosen, _ = choose_consistent_corner_orders(
        obj_m,
        l_m,
        r_m,
        names_m,
        K1,
        D1,
        K2,
        D2,
    )

    obj_s = []
    l_s = []
    r_s = []
    names_s = []
    orientation_log = []

    for c in chosen:
        obj_s.append(obj_m[c.pair_index])
        l_s.append(c.corners_l)
        r_s.append(c.corners_r)
        names_s.append(names_m[c.pair_index])

        orientation_log.append(
            {
                "name": names_m[c.pair_index],
                "left_order": c.left_variant,
                "right_order": c.right_variant,
                "pnp_relative_baseline_mm": float(np.linalg.norm(c.t_lr)),
                "pnp_reprojection_left_px": float(c.reproj_l),
                "pnp_reprojection_right_px": float(c.reproj_r),
            }
        )

    # --------------------------------------------------------
    # 3. ESTÉREO INICIAL
    # --------------------------------------------------------

    print("\n===== CALIBRACIÓN ESTÉREO =====")

    stereo = run_stereo_calibration(
        obj_s,
        l_s,
        r_s,
        K1,
        D1,
        K2,
        D2,
        image_size,
    )

    print(f"RMS estéreo inicial : " f"{stereo['stereo_rms']:.6f} px")
    print(f"Baseline inicial    : " f"{stereo['baseline_mm']:.6f} mm")

    # --------------------------------------------------------
    # 4. LIMPIEZA ESTÉREO Y RECALIBRACIÓN
    # --------------------------------------------------------

    (
        stereo,
        obj_s,
        l_s,
        r_s,
        names_s,
    ) = remove_stereo_outliers_and_recalibrate(
        stereo,
        obj_s,
        l_s,
        r_s,
        names_s,
        K1,
        D1,
        K2,
        D2,
        image_size,
    )

    # Asegura mapas finales.
    if "map1x" not in stereo:
        stereo = rectify(
            stereo,
            image_size,
        )

    per_pair_epi, epi = epipolar_per_pair(
        stereo,
        l_s,
        r_s,
        names_s,
    )

    quality, reject_reasons, warnings = classify_quality(
        stereo,
        epi,
        expected_baseline,
        args.baseline_tolerance_mm,
    )

    print("\n==============================================")
    print(" RESULTADO FINAL")
    print("==============================================")
    print(f"RMS izquierda : {rms_l:.6f} px")
    print(f"RMS derecha   : {rms_r:.6f} px")
    print(f"RMS estéreo   : " f"{stereo['stereo_rms']:.6f} px")
    print(f"Baseline      : " f"{stereo['baseline_mm']:.6f} mm")
    print(f"Pares usados  : {len(names_s)}")
    print(f"Epipolar med. : " f"{epi['median_abs_dy_px']:.6f} px")
    print(f"Epipolar P95  : " f"{epi['p95_abs_dy_px']:.6f} px")
    print(f"Epipolar max  : " f"{epi['max_abs_dy_px']:.6f} px")
    print(f"Calidad       : {quality}")

    if reject_reasons:
        print("\nRECHAZOS:")
        for reason in reject_reasons:
            print(f"  - {reason}")

    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")

    report = {
        "version": "2.1",
        "image_size": list(image_size),
        "checkerboard": {
            "cols_internal": args.cols,
            "rows_internal": args.rows,
            "square_mm": args.square_mm,
        },
        "pairs_after_mono_filter": len(obj_m),
        "pairs_used_final": len(names_s),
        "rms_left_px": float(rms_l),
        "rms_right_px": float(rms_r),
        "stereo_rms_px": float(stereo["stereo_rms"]),
        "baseline_mm": float(stereo["baseline_mm"]),
        "expected_baseline_mm": expected_baseline,
        "baseline_tolerance_mm": float(args.baseline_tolerance_mm),
        "epipolar": epi,
        "epipolar_per_pair": per_pair_epi,
        "corner_orientation": orientation_log,
        "quality": quality,
        "reject_reasons": reject_reasons,
        "warnings": warnings,
    }

    yaml_path, npz_path, json_path = save_outputs(
        args.output,
        stereo,
        rms_l,
        rms_r,
        report,
    )

    preview = create_preview(
        args.dataset,
        args.output,
        stereo,
    )

    print("\nARCHIVOS:")
    print(f"  {yaml_path}")
    print(f"  {npz_path}")
    print(f"  {json_path}")
    if preview is not None:
        print(f"  {preview}")

    print("\n==============================================")

    if quality == "accepted":
        print("CALIBRACIÓN ACEPTADA.")
        print("Estos archivos ya tienen el formato " "esperado por el pipeline.")
    elif quality == "warning":
        print("CALIBRACIÓN CON WARNING.")
        print("No reemplaces aún la calibración activa " "sin revisar métricas y preview.")
    else:
        print("CALIBRACIÓN RECHAZADA.")
        print("NO reemplaces los archivos activos del sistema.")

    print("==============================================\n")


if __name__ == "__main__":
    main()
