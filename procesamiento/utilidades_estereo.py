"""Validación estructural de referencias estéreo antes de utilizarlas o instalarlas."""
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np


def validate_stereo_calibration(directory):
    directory = Path(directory)
    try:
        report = json.loads((directory / "calibration_report.json").read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report.get("quality") != "accepted":
            raise ValueError("El informe debe declarar quality=accepted.")
        if report.get("image_size") != [1920, 1080] or report.get("reject_reasons") != []:
            raise ValueError("Resolución incompatible o motivos de rechazo ausentes/no vacíos.")
        for key in ("baseline_mm", "stereo_rms_px"):
            value = float(report[key])
            if not np.isfinite(value) or value < 0 or (key == "baseline_mm" and value == 0):
                raise ValueError(f"Valor no válido: {key}")
        with np.load(directory / "rectification_maps.npz", allow_pickle=False) as maps:
            for key in ("map1x", "map1y", "map2x", "map2y"):
                array = maps[key]
                if array.shape != (1080, 1920) or array.dtype not in (np.float32, np.float64) or not np.isfinite(array).all():
                    raise ValueError(f"Mapa {key} incompatible o no finito.")
            for prefix in ("map1", "map2"):
                x, y = maps[prefix + "x"], maps[prefix + "y"]
                valid = (x >= 0) & (x <= 1919) & (y >= 0) & (y <= 1079)
                if not np.any(valid):
                    raise ValueError(f"{prefix}: ningún píxel remite al interior de la imagen.")
                points = np.column_stack((x[valid], y[valid]))[::64].astype(np.float64)
                if len(points) < 3:
                    raise ValueError(f"{prefix}: soporte insuficiente para rectificación bidimensional.")
                spread = np.linalg.eigvalsh(np.cov(points, rowvar=False))
                if spread[0] <= max(1e-8, spread[-1] * 1e-10):
                    raise ValueError(f"{prefix}: mapa degenerado, colapsado en un punto o una línea.")
        fs = cv2.FileStorage(str(directory / "stereo_initial.yaml"), cv2.FILE_STORAGE_READ)
        try:
            if not fs.isOpened():
                raise ValueError("YAML estéreo ilegible.")
            values = {}
            for key, shape in (("R", (3, 3)), ("R1", (3, 3)), ("R2", (3, 3)), ("T", (3, 1)), ("P1", (3, 4)), ("P2", (3, 4)), ("Q", (4, 4))):
                value = fs.getNode(key).mat()
                if value is None or value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Matriz {key} ausente, incompatible o no finita.")
                values[key] = value
            for key in ("R", "R1", "R2"):
                matrix = values[key]
                if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(matrix), 1, atol=1e-4):
                    raise ValueError(f"{key} no es una rotación válida.")
            for key in ("P1", "P2"):
                if values[key][0, 0] <= 0 or values[key][1, 1] <= 0:
                    raise ValueError(f"Focal no positiva en {key}.")
            if abs(values["Q"][3, 2]) < 1e-12:
                raise ValueError("Q contiene una separación estéreo nula.")
            baseline_node, rms_node = fs.getNode("baseline_mm"), fs.getNode("rms_stereo")
            if baseline_node.empty() or rms_node.empty():
                raise ValueError("Faltan baseline_mm o rms_stereo en YAML.")
            baseline, rms = baseline_node.real(), rms_node.real()
            baselines = [baseline, float(report["baseline_mm"]), np.linalg.norm(values["T"]), -values["P2"][0, 3] / values["P2"][0, 0], 1 / values["Q"][3, 2]]
            if not np.isfinite(baselines).all() or min(baselines) <= 0 or np.ptp(baselines) > 1e-3:
                raise ValueError("Separación estéreo incoherente entre T, P2, Q e informe.")
            if not np.isfinite(rms) or not np.isclose(rms, report["stereo_rms_px"], atol=1e-6):
                raise ValueError("RMS incoherente entre YAML e informe.")
        finally:
            fs.release()
        return report
    except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile, cv2.error, SystemError) as exc:
        raise ValueError(f"Calibración estéreo no válida: {exc}") from exc
