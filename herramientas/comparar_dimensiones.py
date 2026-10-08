"""Compara descriptores geométricos de los PLY publicados con medidas aproximadas.

No modifica mallas ni informes científicos históricos. Métodos y parámetros fijos
para todas las repeticiones; no utiliza las medidas físicas para ajustar geometría.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display(value, signed=False):
    return "No disponible" if value is None else format(value, "+.2f" if signed else ".2f")


def dimensions(vertices, kind):
    """Eje Y vertical del sistema; orientación de base estimada sin referencias reales."""
    y = vertices[:, 1]
    height = float(np.ptp(y))
    result = {} if kind == "piramide" else {"alto" if kind == "cubo" else "altura": height}
    diagnostics = {"height_method": "extent_y_calibrated_axis", "height_robust_005_995_mm": float(np.diff(np.quantile(y, [0.005, 0.995]))[0]), "aabb_extents_mm": np.ptp(vertices, axis=0).tolist()}
    if kind == "cubo":
        projected = vertices[:, [0, 2]].astype(np.float32)
        rect = cv2.minAreaRect(projected.reshape(-1, 1, 2))
        short, long = sorted(map(float, rect[1]))
        result.update(fondo=short, ancho=long)
        diagnostics.update(base_method="minimum_area_enclosing_rectangle_xz", angle_degrees=float(rect[2]), assignment="Lado menor=fondo (94 mm), mayor=ancho (95 mm); los ejes físicos frente/fondo no están etiquetados en las capturas.")
    elif kind == "cilindro":
        # Franja central evita que el espesor del borde superior o inferior domine.
        mask = (y >= y.min() + 0.2 * height) & (y <= y.min() + 0.8 * height)
        points = vertices[mask][:, [0, 2]]
        if len(points) < 20:
            raise ValueError("Puntos insuficientes para estimar el diámetro.")
        center = np.mean(points, axis=0)
        centered = points - center
        design = np.column_stack((2 * centered, np.ones(len(points))))
        solution = np.linalg.lstsq(design, np.sum(centered ** 2, axis=1), rcond=None)[0]
        initial_center = center + solution[:2]
        initial_radius = np.sqrt(max(0, solution[2] + np.sum(solution[:2] ** 2)))
        fit = least_squares(lambda params: np.linalg.norm(points - params[:2], axis=1) - params[2], [*initial_center, initial_radius], bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]), loss="soft_l1", f_scale=1.0)
        if not fit.success:
            raise ValueError("No convergió el ajuste circular.")
        residual = np.linalg.norm(points - fit.x[:2], axis=1) - fit.x[2]
        result["diametro"] = float(2 * fit.x[2])
        diagnostics.update(diameter_method="robust_circle_fit_xz_middle_20_to_80_percent_height", radial_residual_p95_mm=float(np.quantile(np.abs(residual), .95)), circle_points=int(len(points)))
    else:
        # La foto identifica una base triangular. No se deducen medidas de píxeles.
        bands = [vertices[y <= y.min() + 2], vertices[y >= y.max() - 2]]
        areas = [cv2.contourArea(cv2.convexHull(b[:, [0, 2]].astype(np.float32))) for b in bands]
        base_index = int(np.argmax(areas))
        base, tip_band = bands[base_index], bands[1 - base_index]
        center = base.mean(axis=0)
        _, _, axes = np.linalg.svd(base - center, full_matrices=False)
        projected = (base - center) @ axes[:2].T
        hull = cv2.convexHull(projected.astype(np.float32))
        polygon = cv2.approxPolyDP(hull, .04 * cv2.arcLength(hull, True), True).reshape(-1, 2)
        residual = float(np.quantile(np.abs((base - center) @ axes[2]), .95))
        diagnostics.update(base_band_mm=2, polygon_epsilon_fraction=.04, base_plane_residual_p95_mm=residual, vertical_extent_y_mm=height)
        if len(polygon) != 3 or residual > 2:
            diagnostics.update(not_measured=["lado_base", "altura_cara", "arista_lateral_hasta_punta"], reason="No se identificó una base triangular suficientemente plana.")
            return result, diagnostics
        corners = center + polygon @ axes[:2]
        tip = tip_band.mean(axis=0)
        sides, edges, face_heights, feet = [], [], [], []
        for i in range(3):
            a, b = corners[i], corners[(i + 1) % 3]
            direction = b - a
            t = float(np.dot(tip - a, direction) / np.dot(direction, direction))
            foot = a + t * direction
            sides.append(float(np.linalg.norm(direction)))
            edges.append(float(np.linalg.norm(tip - a)))
            face_heights.append(float(np.linalg.norm(tip - foot)))
            feet.append(t)
        if not all(0 <= t <= 1 for t in feet):
            diagnostics.update(not_measured=["lado_base", "altura_cara", "arista_lateral_hasta_punta"], reason="La proyección de la punta queda fuera de un lado; requiere identificación manual.")
            return result, diagnostics
        result.update(lado_base=float(np.mean(sides)), altura_cara=float(np.mean(face_heights)), arista_lateral_hasta_punta=float(np.mean(edges)))
        diagnostics.update(method="triangular_base_plane_and_rounded_tip_band", base_vertices_mm=corners.tolist(), tip_mm=tip.tolist(), base_sides_mm=sides, lateral_edges_mm=edges, face_altitudes_mm=face_heights, foot_fractions=feet, perpendicular_height_mm=float(abs(np.dot(tip - center, axes[2]))), interpretation="Descriptores aproximados: media de tres lados/aristas/alturas; sin correspondencia de caras físicas. Punta promediada en banda de 2 mm y esquinas redondeadas pueden subestimar longitudes.")
    if not all(np.isfinite(v) and v > 0 for v in result.values()):
        raise ValueError("Dimensiones vacías, no finitas o degeneradas.")
    return result, diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "resultados")
    args = parser.parse_args(argv)
    root = args.results.resolve()
    physical_path = root / "medidas_fisicas.json"
    physical = json.loads(physical_path.read_text(encoding="utf-8"))
    rows, runs = [], []
    for record in json.loads((root / "catalogo.json").read_text(encoding="utf-8")):
        name = record["public_name"]
        kind = name.rstrip("0123456789")
        for variant, filename in (("final", "modelo_final_original_mm.ply"), ("pre_pulido", "modelo_pre_pulido_mm.ply")):
            path = root / name / filename
            mesh = o3d.io.read_triangle_mesh(str(path))
            vertices = np.asarray(mesh.vertices)
            if vertices.size == 0 or not np.isfinite(vertices).all():
                raise ValueError(f"Malla inválida: {path}")
            values, diagnostics = dimensions(vertices, kind)
            runs.append({"campaign": name, "variant": variant, "source": path.relative_to(root).as_posix(), "sha256": sha(path), "dimensions_mm": values, "diagnostics": diagnostics})
            for dimension in physical["objects"][kind]:
                value = values.get(dimension)
                reference = float(physical["objects"][kind][dimension])
                difference = value - reference if value is not None else None
                rows.append({"campaign": name, "object": kind, "variant": variant, "dimension": dimension, "reference_mm": reference, "reconstructed_mm": value, "difference_mm": difference, "absolute_difference_mm": abs(difference) if difference is not None else None, "difference_percent": 100 * difference / reference if difference is not None else None, "status": "measured" if value is not None else "not_available", "reason": "" if value is not None else diagnostics.get("reason", "Magnitud no disponible")})
    aggregate = []
    for kind, dimension in sorted({(r["object"], r["dimension"]) for r in rows}):
        expected = [r for r in rows if r["object"] == kind and r["dimension"] == dimension and r["variant"] == "final"]
        selected = [r for r in expected if r["reconstructed_mm"] is not None]
        values = np.array([r["reconstructed_mm"] for r in selected])
        aggregate.append({"object": kind, "dimension": dimension, "campaigns": len(selected), "expected_campaigns": len(expected), "mean_mm": float(values.mean()) if len(values) else None, "sample_std_mm": float(values.std(ddof=1)) if len(values) > 1 else None, "mean_absolute_difference_mm": float(np.mean([r["absolute_difference_mm"] for r in selected])) if selected else None})
    with (root / "comparacion_dimensional.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {"schema_version": 1, "physical_reference_sha256": sha(physical_path), "script_sha256": sha(Path(__file__)), "reference_precision": physical["precision"], "runs": runs, "rows": rows, "aggregate_final": aggregate, "scope": "Comparación exploratoria con regla; no modifica calidad ni constituye validación metrológica. Las desviaciones entre campañas no son incertidumbre del instrumento."}
    (root / "comparacion_dimensional.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Comparación dimensional exploratoria", "", "Esta comparación utiliza las referencias aproximadas tomadas con regla y las mallas publicadas, sin modificarlas. Para cada tipo de objeto se aplica el mismo método, sin ajustar los parámetros a sus dimensiones reales. Las discrepancias describen las diferencias obtenidas; no certifican la exactitud del sistema.", "", "## Método y alcance", "", "- Altura de cubo y cilindro: extensión sobre Y, eje vertical del sistema calibrado. Requiere objeto apoyado y alineado con el montaje; no equivale a ajustar un plano de base independiente. El JSON incluye también la extensión entre percentiles 0.5–99.5 como diagnóstico de extremos, sin sustituir la medida principal.", "- Cubo: rectángulo envolvente de área mínima en XZ, corrigiendo el giro horizontal. Lado menor se compara con fondo y mayor con ancho; esa asignación por orden no identifica físicamente las caras. Inclinación y puntos extremos pueden aumentar las dimensiones.", "- Cilindro: ajuste circular robusto en XZ sobre los vértices situados entre el 20 % y el 80 % de la altura. Se guarda el residual radial P95. El ajuste supone eje aproximadamente paralelo a Y y la densidad de vértices influye en la ponderación.", "- Pirámide triangular: plano por SVD en la banda extrema de 2 mm con mayor área horizontal; esquinas por envolvente convexa simplificada al 4 % del perímetro; punta promediada en la banda extrema opuesta. Se guardan los puntos, los tres lados, aristas y alturas de cara, y se comparan sus medias con las referencias aproximadas. No se supone regularidad ni correspondencia individual de caras. Redondeo y banda de punta pueden subestimar medidas. La altura perpendicular al plano y la extensión Y son diagnósticos sin referencia física; los 85 mm corresponden a altura de cara. La fotografía solo identifica segmentos, no aporta medidas por píxeles.", "- Diferencia = reconstruido − referencia. Porcentaje = 100 × diferencia / referencia. La desviación estándar muestral de las tres campañas describe variación de resultados, no incertidumbre de la regla ni independencia experimental acreditada.", "", "## Mallas finales", "", "| Campaña | Magnitud | Regla (mm) | Reconstrucción (mm) | Diferencia (mm) | Diferencia (%) |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        if r["variant"] == "final":
            lines.append(f"| {r['campaign']} | {r['dimension']} | {r['reference_mm']:.1f} | {display(r['reconstructed_mm'])} | {display(r['difference_mm'], True)} | {display(r['difference_percent'], True)} |")
    lines += ["", "Las medidas ausentes se muestran como **No disponible** y no participan en las estadísticas. La desviación estándar requiere al menos dos campañas válidas. CSV y JSON conservan el motivo de cada ausencia.", "", "## Variación entre campañas", "", "| Objeto | Magnitud | Campañas válidas/esperadas | Media (mm) | Desviación estándar (mm) | Discrepancia absoluta media (mm) |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in aggregate:
        lines.append(f"| {r['object']} | {r['dimension']} | {r['campaigns']}/{r['expected_campaigns']} | {display(r['mean_mm'])} | {display(r['sample_std_mm'])} | {display(r['mean_absolute_difference_mm'])} |")
    lines += ["", "El [CSV](comparacion_dimensional.csv) incluye todas las medidas de las mallas finales y previas al pulido. El [JSON](comparacion_dimensional.json) conserva métodos, diagnósticos y huellas. El modelo previo al pulido puede contener relleno anterior: esta comparación no separa observación de inferencia.", "", "Reproducir: `python herramientas/comparar_dimensiones.py`. No actualiza el estado de aceptación ni activa calibraciones. Las tablas son resultados numéricos del repositorio; su discusión y conclusiones corresponden al documento de tesis."]
    (root / "comparacion_dimensional.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Vincular la comparación sin alterar los informes históricos del paso 17.
    catalog = json.loads((root / "catalogo.json").read_text(encoding="utf-8"))
    for record in catalog:
        record["physical_reference_mm"] = physical["objects"][record["public_name"].rstrip("0123456789")]
        campaign_rows = [row for row in rows if row["campaign"] == record["public_name"]]
        measured = sum(row["reconstructed_mm"] is not None for row in campaign_rows)
        record["dimensional_comparison_performed"] = measured > 0
        record["dimensional_comparison_status"] = "complete" if measured == len(campaign_rows) and measured else "partial" if measured else "not_available"
        record["dimensional_comparison_scope"] = "Exploratoria; magnitudes y limitaciones en comparacion_dimensional.json"
        (root / record["public_name"] / "procedencia.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        path = root / record["public_name"] / "README.md"
        text = path.read_text(encoding="utf-8").replace("Se incluyen como referencia; no se ha calculado aquí una discrepancia dimensional.", "La [comparación dimensional exploratoria](../comparacion_dimensional.md) explica cómo se estimaron las dimensiones y cuáles son las limitaciones del método.")
        path.write_text(text, encoding="utf-8")
    (root / "catalogo.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    intro = root / "README.md"
    if "comparacion_dimensional.md" not in intro.read_text(encoding="utf-8"):
        with intro.open("a", encoding="utf-8") as stream:
            stream.write("\n## Comparación dimensional\n\n[Tabla de comparación con regla](comparacion_dimensional.md), con resultados finales y previos al pulido, métodos, discrepancias y variación entre campañas.\n")
    manifest = {p.relative_to(root).as_posix(): {"sha256": sha(p), "size_bytes": p.stat().st_size} for p in sorted(root.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}
    (root / "MANIFEST.json").write_text(json.dumps({"schema_version": 1, "excludes": ["MANIFEST.json"], "files": manifest}, indent=2), encoding="utf-8")
    print(f"Comparadas {len(runs)} mallas; {len(rows)} medidas. Manifiesto actualizado.")


if __name__ == "__main__":
    main()
