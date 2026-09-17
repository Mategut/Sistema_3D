#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PASO 18 V1.4 — EXPORTACIÓN FINAL DEL MODELO PARA BLENDER

Objetivo
--------
Cerrar el pipeline científico después de paso 17 generando una carpeta
resultado_final fácil de usar, sin modificar la geometría validada.

Entradas
--------
- Malla seleccionada por el paso 13, limpiada por el paso 14, pulida por el paso 15 y validada:
    malla_final_topologica.ply
- Validación final del paso 17 V1.4:
    resumen_17_validacion_modelo.json

Política
--------
- quality=accepted  -> exportar
- quality=warning   -> exportar y documentar advertencias
- quality=rejected  -> NO exportar como modelo final

Salidas
-------
resultado_final/
    modelo_final.obj                 # Blender-ready: metros, Z-up
    modelo_final.mtl
    modelo_final_metric_mm.obj       # coordenadas científicas originales, mm
    modelo_final_original_mm.ply     # copia exacta del PLY validado
    README_IMPORTAR_EN_BLENDER.txt
    resumen_exportacion_18.json
    preview_validacion.png           # si existe

Además:
reconstruccion/multisesion/18_exportacion_modelo/
    resumen_18_exportacion_modelo.json

Importante
----------
OBJ no define unidades formalmente. Para evitar ambigüedad:
- modelo_final.obj se transforma a metros y a sistema Z-up de Blender.
- modelo_final_metric_mm.obj conserva exactamente el marco científico
  previo: coordenadas en milímetros y ejes originales.

Transformación Blender:
    X_blender =  X_camera * 0.001
    Y_blender = -Z_camera * 0.001
    Z_blender =  Y_camera * 0.001

La matriz de rotación tiene determinante +1, por lo que preserva la
orientación de las caras.
"""

from __future__ import annotations
from utilidades_progreso import operacion
from utilidades_rendimiento import configurar as _configurar_recursos

# Los runtimes numéricos leen estos límites al importarse.
_configurar_recursos(__file__)


import argparse
import hashlib
import json
import math
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple

import numpy as np

PLY_SCALARS = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


@dataclass
class PlyProperty:
    """Descripción de una propiedad escalar o de lista en la cabecera PLY."""

    name: str
    scalar_type: Optional[str] = None
    list_count_type: Optional[str] = None
    list_item_type: Optional[str] = None

    @property
    def is_list(self) -> bool:
        return self.list_count_type is not None


@dataclass
class PlyElement:
    """Descripción de un elemento PLY y sus propiedades."""

    name: str
    count: int
    properties: List[PlyProperty]


@dataclass
class MeshData:
    """Arrays de geometría y atributos leídos del archivo PLY."""

    vertices: np.ndarray
    triangles: np.ndarray
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None


def parser() -> argparse.ArgumentParser:
    """Construye las opciones de línea de comandos de este paso."""
    p = argparse.ArgumentParser(description="Paso 18: exportación OBJ final para Blender.")
    p.add_argument("--root", required=True)
    p.add_argument("--object", required=True)
    p.add_argument(
        "--mesh-source",
        default="15_pulido_final",
    )
    p.add_argument(
        "--validation-source",
        default="17_validacion_modelo",
    )
    p.add_argument(
        "--pre-polish-source",
        default="14_limpieza_topologica",
        help=(
            "Fuente topológica anterior al pulido final. Se conserva para "
            "cuantificar el efecto exclusivo del paso 15 V1.8."
        ),
    )
    p.add_argument(
        "--pre-polish-fallback-source",
        default="14_limpieza_topologica",
        help=("Fallback canónico al mismo producto topológico del paso 14 V1.3."),
    )
    p.add_argument(
        "--output-name",
        default="18_exportacion_modelo",
    )
    p.add_argument(
        "--final-folder-name",
        default="resultado_final",
    )
    p.add_argument(
        "--blender-scale",
        type=float,
        default=0.001,
        help="Conversión de mm científicos a unidades Blender. 0.001 => metros.",
    )
    p.add_argument(
        "--require-evidence-aware-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Exige que paso 17 haya validado el núcleo multivista usando "
            "evidencia independiente/held-out, no solo conteo bruto de poses."
        ),
    )
    return p


def sha256_file(path: Path) -> str:
    """Calcula la huella SHA-256 del archivo mediante lectura por bloques."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    """Escribe el JSON en un temporal y reemplaza el destino al finalizar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def read_ply_header(fh: BinaryIO) -> Tuple[str, List[PlyElement]]:
    """Lee la cabecera PLY y valida su formato, elementos y propiedades."""
    first = fh.readline().decode("ascii", errors="strict").strip()
    if first != "ply":
        raise ValueError("El archivo no es PLY.")

    fmt = None
    elements: List[PlyElement] = []
    current: Optional[PlyElement] = None

    while True:
        raw = fh.readline()
        if not raw:
            raise ValueError("PLY truncado antes de end_header.")
        line = raw.decode("ascii", errors="strict").strip()

        if not line or line.startswith("comment") or line.startswith("obj_info"):
            continue

        parts = line.split()

        if parts[0] == "format":
            if len(parts) < 3:
                raise ValueError("Cabecera PLY format inválida.")
            fmt = parts[1]

        elif parts[0] == "element":
            if len(parts) != 3:
                raise ValueError("Cabecera PLY element inválida.")
            current = PlyElement(parts[1], int(parts[2]), [])
            elements.append(current)

        elif parts[0] == "property":
            if current is None:
                raise ValueError("property antes de element en PLY.")

            if len(parts) >= 5 and parts[1] == "list":
                current.properties.append(
                    PlyProperty(
                        name=parts[4],
                        list_count_type=parts[2],
                        list_item_type=parts[3],
                    )
                )
            elif len(parts) == 3:
                current.properties.append(
                    PlyProperty(
                        name=parts[2],
                        scalar_type=parts[1],
                    )
                )
            else:
                raise ValueError(f"Propiedad PLY no soportada: {line}")

        elif parts[0] == "end_header":
            break

    if fmt not in {
        "ascii",
        "binary_little_endian",
        "binary_big_endian",
    }:
        raise ValueError(f"Formato PLY no soportado: {fmt}")

    return fmt, elements


def scalar_ascii(token: str, type_name: str):
    """Convierte un token ASCII según el tipo escalar declarado en el PLY."""
    if type_name not in PLY_SCALARS:
        raise ValueError(f"Tipo PLY no soportado: {type_name}")
    fmt, _ = PLY_SCALARS[type_name]
    if fmt in {"f", "d"}:
        return float(token)
    return int(token)


def read_binary_scalar(fh: BinaryIO, type_name: str, endian: str):
    """Lee un escalar PLY binario respetando el tipo y el orden de bytes."""
    if type_name not in PLY_SCALARS:
        raise ValueError(f"Tipo PLY no soportado: {type_name}")
    fmt, size = PLY_SCALARS[type_name]
    raw = fh.read(size)
    if len(raw) != size:
        raise EOFError("PLY binario truncado.")
    return struct.unpack(endian + fmt, raw)[0]


def _triangulate_polygon(indices: List[int], triangles: List[Tuple[int, int, int]]) -> None:
    """Añade una triangulación en abanico desde el primer vértice del polígono."""
    if len(indices) < 3:
        return
    a = int(indices[0])
    for i in range(1, len(indices) - 1):
        triangles.append((a, int(indices[i]), int(indices[i + 1])))


@operacion("Leer geometría PLY")
def read_triangle_ply(path: Path) -> MeshData:
    """Carga una malla PLY ASCII o binaria con sus atributos disponibles."""
    vertices: List[List[float]] = []
    normals: List[List[float]] = []
    colors: List[List[float]] = []
    triangles: List[Tuple[int, int, int]] = []

    with path.open("rb") as fh:
        fmt, elements = read_ply_header(fh)
        endian = "<" if fmt == "binary_little_endian" else ">"

        for element in elements:
            if fmt == "ascii":
                for _ in range(element.count):
                    line = fh.readline().decode("ascii", errors="strict").strip()
                    if not line:
                        raise EOFError("PLY ASCII truncado.")
                    tokens = line.split()
                    pos = 0
                    values: Dict[str, object] = {}

                    for prop in element.properties:
                        if prop.is_list:
                            count = int(scalar_ascii(tokens[pos], prop.list_count_type))
                            pos += 1
                            vals = [
                                scalar_ascii(tokens[pos + i], prop.list_item_type)
                                for i in range(count)
                            ]
                            pos += count
                            values[prop.name] = vals
                        else:
                            values[prop.name] = scalar_ascii(
                                tokens[pos],
                                prop.scalar_type,
                            )
                            pos += 1

                    if element.name == "vertex":
                        vertices.append(
                            [
                                float(values["x"]),
                                float(values["y"]),
                                float(values["z"]),
                            ]
                        )
                        if all(k in values for k in ("nx", "ny", "nz")):
                            normals.append(
                                [
                                    float(values["nx"]),
                                    float(values["ny"]),
                                    float(values["nz"]),
                                ]
                            )
                        if all(k in values for k in ("red", "green", "blue")):
                            colors.append(
                                [
                                    float(values["red"]) / 255.0,
                                    float(values["green"]) / 255.0,
                                    float(values["blue"]) / 255.0,
                                ]
                            )

                    elif element.name == "face":
                        idx = values.get("vertex_indices") or values.get("vertex_index")
                        if idx is not None:
                            _triangulate_polygon(list(idx), triangles)

            else:
                for _ in range(element.count):
                    values: Dict[str, object] = {}

                    for prop in element.properties:
                        if prop.is_list:
                            count = int(
                                read_binary_scalar(
                                    fh,
                                    prop.list_count_type,
                                    endian,
                                )
                            )
                            vals = [
                                read_binary_scalar(
                                    fh,
                                    prop.list_item_type,
                                    endian,
                                )
                                for _ in range(count)
                            ]
                            values[prop.name] = vals
                        else:
                            values[prop.name] = read_binary_scalar(
                                fh,
                                prop.scalar_type,
                                endian,
                            )

                    if element.name == "vertex":
                        vertices.append(
                            [
                                float(values["x"]),
                                float(values["y"]),
                                float(values["z"]),
                            ]
                        )
                        if all(k in values for k in ("nx", "ny", "nz")):
                            normals.append(
                                [
                                    float(values["nx"]),
                                    float(values["ny"]),
                                    float(values["nz"]),
                                ]
                            )
                        if all(k in values for k in ("red", "green", "blue")):
                            colors.append(
                                [
                                    float(values["red"]) / 255.0,
                                    float(values["green"]) / 255.0,
                                    float(values["blue"]) / 255.0,
                                ]
                            )

                    elif element.name == "face":
                        idx = values.get("vertex_indices") or values.get("vertex_index")
                        if idx is not None:
                            _triangulate_polygon(list(idx), triangles)

    v = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)

    if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
        raise ValueError("PLY sin vértices 3D válidos.")
    if t.ndim != 2 or t.shape[1] != 3 or len(t) == 0:
        raise ValueError("PLY sin triángulos válidos.")
    if np.min(t) < 0 or np.max(t) >= len(v):
        raise ValueError("La malla contiene índices de cara fuera de rango.")
    if not np.all(np.isfinite(v)):
        raise ValueError("La malla contiene vértices no finitos.")

    n = None
    if len(normals) == len(vertices):
        n = np.asarray(normals, dtype=np.float64)
        if not np.all(np.isfinite(n)):
            n = None

    c = None
    if len(colors) == len(vertices):
        c = np.asarray(colors, dtype=np.float64)

    return MeshData(v, t, n, c)


def compute_vertex_normals(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Acumula las normales de las caras incidentes y normaliza por vértice."""
    normals = np.zeros_like(vertices, dtype=np.float64)
    tri = vertices[triangles]
    face_n = np.cross(
        tri[:, 1] - tri[:, 0],
        tri[:, 2] - tri[:, 0],
    )
    lengths = np.linalg.norm(face_n, axis=1)
    good = lengths > 1e-15
    face_n[good] /= lengths[good, None]
    face_n[~good] = 0.0

    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_n)

    ln = np.linalg.norm(normals, axis=1)
    good_v = ln > 1e-15
    normals[good_v] /= ln[good_v, None]
    normals[~good_v] = np.array([0.0, 0.0, 1.0])

    return normals


def normalize_normals(normals: np.ndarray) -> np.ndarray:
    """Normaliza las normales válidas y conserva el tratamiento de filas degeneradas."""
    n = np.asarray(normals, dtype=np.float64).copy()
    ln = np.linalg.norm(n, axis=1)
    good = np.isfinite(ln) & (ln > 1e-15)
    n[good] /= ln[good, None]
    n[~good] = np.array([0.0, 0.0, 1.0])
    return n


@operacion("Exportar geometría OBJ para Blender")
def write_obj(
    path: Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
    normals: np.ndarray,
    *,
    title: str,
    units: str,
    coordinate_system: str,
    mtl_name: Optional[str] = None,
) -> None:
    """Escribe vértices, normales y caras OBJ con los metadatos de escala y ejes."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Sistema 3D Integrado — Paso 18\n")
        fh.write(f"# {title}\n")
        fh.write(f"# units: {units}\n")
        fh.write(f"# coordinate_system: {coordinate_system}\n")
        fh.write(f"# vertices: {len(vertices)}\n")
        fh.write(f"# triangles: {len(triangles)}\n")

        if mtl_name:
            fh.write(f"mtllib {mtl_name}\n")
            fh.write("usemtl reconstruccion_3d\n")

        for x, y, z in vertices:
            fh.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")

        for nx, ny, nz in normals:
            fh.write(f"vn {nx:.9f} {ny:.9f} {nz:.9f}\n")

        # Correspondencia 1:1 vértice-normal.
        for a, b, c in triangles:
            a1, b1, c1 = int(a) + 1, int(b) + 1, int(c) + 1
            fh.write(f"f {a1}//{a1} {b1}//{b1} {c1}//{c1}\n")


def write_mtl(path: Path) -> None:
    """Escribe el material neutro utilizado por la exportación OBJ."""
    path.write_text(
        "\n".join(
            [
                "# Material simple para la reconstrucción 3D",
                "newmtl reconstruccion_3d",
                "Ka 0.180000 0.180000 0.180000",
                "Kd 0.720000 0.720000 0.720000",
                "Ks 0.120000 0.120000 0.120000",
                "Ns 32.000000",
                "d 1.000000",
                "illum 2",
                "",
            ]
        ),
        encoding="utf-8",
    )


def resolve_pre_polish_mesh(
    multi,
    source,
    fallback_source,
):
    """Resuelve la malla topológica inmediatamente anterior al paso 15 V1.8."""
    multi = Path(multi)

    candidates = [
        (
            multi / str(source) / "malla_final_topologica.ply",
            "pre_polish_source",
        ),
        (
            multi / str(fallback_source) / "malla_final_topologica.ply",
            "pre_polish_fallback",
        ),
    ]

    checked = []
    for path, mode in candidates:
        checked.append(str(path))
        if path.is_file():
            return path.resolve(), mode, checked

    raise FileNotFoundError(
        "No se encontró la malla topológica pre-pulido. "
        "Rutas comprobadas:\n- " + "\n- ".join(checked)
    )


def main() -> int:
    """Exporta la malla validada y sus metadatos en los formatos finales."""
    args = parser().parse_args()

    root = Path(args.root).expanduser().resolve()
    obj = args.object.strip().lower()

    multi = root / "reconstruccion" / "multisesion"

    mesh_path = multi / args.mesh_source / "malla_final_topologica.ply"
    validation_path = multi / args.validation_source / "resumen_17_validacion_modelo.json"
    preview_source = multi / args.validation_source / "preview_validacion_modelo_17.png"
    pre_polish_mesh_path, pre_polish_resolution_mode, pre_polish_checked_paths = (
        resolve_pre_polish_mesh(
            multi,
            args.pre_polish_source,
            args.pre_polish_fallback_source,
        )
    )

    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)

    print(
        "[paso 18] Malla topológica pre-pulido:",
        pre_polish_mesh_path,
    )
    print(
        "[paso 18] Resolución de fuente pre-pulido:",
        pre_polish_resolution_mode,
    )

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    quality = str(validation.get("quality", "unknown")).strip().lower()
    evidence_validation = validation.get("evidence_contract_validation", {})
    evidence_aware = bool(
        evidence_validation.get("available", False)
        and evidence_validation.get("strong_core_mode") == "pose_diverse_heldout_validated_evidence"
    )
    if bool(args.require_evidence_aware_validation) and not evidence_aware:
        raise RuntimeError(
            "Paso 18 V1.4 no exporta como resultado final una validación que no "
            "acredite el contrato de evidencia pose-diversa de 11/12/17."
        )

    if quality in {"rejected", "reject", "failed", "error"}:
        print("paso 18 NO exporta un modelo final porque paso 17 " f"declaró quality={quality}.")
        return 2

    mesh = read_triangle_ply(mesh_path)
    pre_polish_mesh = read_triangle_ply(pre_polish_mesh_path)

    normals = mesh.normals
    if normals is None or normals.shape != mesh.vertices.shape:
        normals = compute_vertex_normals(
            mesh.vertices,
            mesh.triangles,
        )
        normals_source = "computed_from_triangles"
    else:
        normals = normalize_normals(normals)
        normals_source = "source_ply"

    pre_polish_normals = pre_polish_mesh.normals
    if pre_polish_normals is None or pre_polish_normals.shape != pre_polish_mesh.vertices.shape:
        pre_polish_normals = compute_vertex_normals(
            pre_polish_mesh.vertices,
            pre_polish_mesh.triangles,
        )
        pre_polish_normals_source = "computed_from_triangles"
    else:
        pre_polish_normals = normalize_normals(pre_polish_normals)
        pre_polish_normals_source = "source_ply"

    source_min = np.min(mesh.vertices, axis=0)
    source_max = np.max(mesh.vertices, axis=0)
    source_extent = source_max - source_min

    # ------------------------------------------------------------
    # Exportación científica original: mm, ejes originales.
    # ------------------------------------------------------------
    scientific_vertices = mesh.vertices.copy()
    scientific_normals = normals.copy()

    # ------------------------------------------------------------
    # Exportación Blender:
    # Xb = X
    # Yb = -Z
    # Zb = Y
    # y mm -> m.
    # ------------------------------------------------------------
    rotation = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12):
        raise RuntimeError("La transformación Blender debe preservar orientación (det=+1).")

    blender_scale = float(args.blender_scale)
    if not math.isfinite(blender_scale) or blender_scale <= 0:
        raise ValueError("--blender-scale debe ser > 0.")

    blender_vertices = (scientific_vertices @ rotation.T) * blender_scale
    blender_normals = normalize_normals(scientific_normals @ rotation.T)

    # ------------------------------------------------------------
    # Crear salidas en temporal para evitar resultado_final parcial.
    # ------------------------------------------------------------
    object_dir = root
    final_dir = object_dir / args.final_folder_name
    temp_final = object_dir / (args.final_folder_name + "_temporal_paso_18")

    if temp_final.exists():
        shutil.rmtree(temp_final)
    temp_final.mkdir(parents=True)

    blender_obj = temp_final / "modelo_final.obj"
    blender_mtl = temp_final / "modelo_final.mtl"
    scientific_obj = temp_final / "modelo_final_metric_mm.obj"
    ply_copy = temp_final / "modelo_final_original_mm.ply"
    pre_polish_obj = temp_final / "modelo_pre_pulido_metric_mm.obj"
    pre_polish_ply = temp_final / "modelo_pre_pulido_mm.ply"
    readme = temp_final / "README_IMPORTAR_EN_BLENDER.txt"
    export_report_copy = temp_final / "resumen_exportacion_18.json"

    write_mtl(blender_mtl)

    write_obj(
        blender_obj,
        blender_vertices,
        mesh.triangles,
        blender_normals,
        title="Modelo final listo para Blender",
        units="meters",
        coordinate_system=("right-handed Z-up; X=X_original, " "Y=-Z_original, Z=Y_original"),
        mtl_name=blender_mtl.name,
    )

    write_obj(
        scientific_obj,
        scientific_vertices,
        mesh.triangles,
        scientific_normals,
        title="Modelo científico original",
        units="millimeters",
        coordinate_system="original reconstruction coordinate system",
        mtl_name=None,
    )

    shutil.copy2(mesh_path, ply_copy)

    # Producto científico PRE-PULIDO:
    # permite medir el efecto exclusivo del pulido final.
    write_obj(
        pre_polish_obj,
        pre_polish_mesh.vertices,
        pre_polish_mesh.triangles,
        pre_polish_normals,
        title="Modelo topológico antes del pulido final",
        units="millimeters",
        coordinate_system="original reconstruction coordinate system",
        mtl_name=None,
    )
    shutil.copy2(
        pre_polish_mesh_path,
        pre_polish_ply,
    )

    if preview_source.is_file():
        shutil.copy2(
            preview_source,
            temp_final / "preview_validacion.png",
        )

    blender_min = np.min(blender_vertices, axis=0)
    blender_max = np.max(blender_vertices, axis=0)

    report = {
        "schema_version": "1.0",
        "step": "18",
        "method": "lossless_mesh_export_obj_plus_blender_coordinate_conversion",
        "object": obj,
        "status": "exported",
        "quality": quality,
        "validation_quality": quality,
        "validation_closure_status": validation.get("closure_status"),
        "validation_warning_reasons": validation.get("warning_reasons", []),
        "validation_reject_reasons": validation.get("reject_reasons", []),
        "evidence_aware_validation": {
            "required": bool(args.require_evidence_aware_validation),
            "verified": bool(evidence_aware),
            "details": evidence_validation,
        },
        "export_allowed_for_warning": True,
        "geometry_modified": False,
        "topology_modified": False,
        "source": {
            "mesh": str(mesh_path),
            "mesh_sha256": sha256_file(mesh_path),
            "validation": str(validation_path),
            "validation_sha256": sha256_file(validation_path),
            "vertex_count": int(len(mesh.vertices)),
            "triangle_count": int(len(mesh.triangles)),
            "normals_source": normals_source,
            "bbox_min_mm": source_min.astype(float).tolist(),
            "bbox_max_mm": source_max.astype(float).tolist(),
            "bbox_extent_mm": source_extent.astype(float).tolist(),
        },
        "blender_export": {
            "filename": "modelo_final.obj",
            "units": "meters",
            "up_axis": "Z",
            "handedness": "right-handed",
            "scale_from_mm": blender_scale,
            "rotation_original_to_blender": rotation.astype(float).tolist(),
            "bbox_min_m": blender_min.astype(float).tolist(),
            "bbox_max_m": blender_max.astype(float).tolist(),
            "obj_sha256": sha256_file(blender_obj),
            "mtl_sha256": sha256_file(blender_mtl),
        },
        "scientific_obj_export": {
            "filename": "modelo_final_metric_mm.obj",
            "units": "millimeters",
            "coordinate_system": "original",
            "obj_sha256": sha256_file(scientific_obj),
        },
        "ply_copy": {
            "filename": "modelo_final_original_mm.ply",
            "sha256": sha256_file(ply_copy),
            "identical_to_source": (sha256_file(ply_copy) == sha256_file(mesh_path)),
        },
        "pre_polish_reference": {
            "source": str(pre_polish_mesh_path),
            "resolution_mode": str(pre_polish_resolution_mode),
            "checked_paths": list(pre_polish_checked_paths),
            "source_sha256": sha256_file(pre_polish_mesh_path),
            "vertex_count": int(len(pre_polish_mesh.vertices)),
            "triangle_count": int(len(pre_polish_mesh.triangles)),
            "normals_source": pre_polish_normals_source,
            "scientific_obj_filename": ("modelo_pre_pulido_metric_mm.obj"),
            "scientific_obj_sha256": sha256_file(pre_polish_obj),
            "ply_filename": ("modelo_pre_pulido_mm.ply"),
            "ply_sha256": sha256_file(pre_polish_ply),
            "interpretation": (
                "Geometría topológica previa al paso 15 V1.8. "
                "Se conserva para cuantificar el efecto del pulido final."
            ),
        },
        "blender_import_note": (
            "Importar modelo_final.obj. Sus coordenadas ya están en metros "
            "y Z-up; no requiere aplicar factor 0.001 manualmente."
        ),
    }

    atomic_json(export_report_copy, report)

    readme.write_text(
        (
            "RESULTADO FINAL — SISTEMA 3D\n"
            "========================================\n\n"
            "ARCHIVO RECOMENDADO PARA BLENDER\n"
            "----------------------------------------\n"
            "modelo_final.obj\n\n"
            "Este archivo ya fue preparado para Blender:\n"
            "- unidades numéricas: metros;\n"
            "- eje vertical: Z;\n"
            "- sistema de coordenadas: mano derecha;\n"
            "- la geometría/topología NO fue suavizada ni cerrada por paso 18.\n\n"
            "IMPORTAR EN BLENDER\n"
            "----------------------------------------\n"
            "File -> Import -> Wavefront (.obj)\n"
            "Selecciona: modelo_final.obj\n\n"
            "Puedes trabajar con Scene Units = Metric y Unit Scale = 1.0.\n\n"
            "ARCHIVO CIENTÍFICO\n"
            "----------------------------------------\n"
            "modelo_final_metric_mm.obj conserva las coordenadas originales\n"
            "de la reconstrucción en milímetros.\n\n"
            "PLY FINAL COMPLETADO/PULIDO\n"
            "----------------------------------------\n"
            "modelo_final_original_mm.ply es la malla final validada.\n\n"
            "MODELO TOPOLÓGICO PRE-PULIDO\n"
            "----------------------------------------\n"
            "modelo_pre_pulido_metric_mm.obj y\n"
            "modelo_pre_pulido_mm.ply conservan la geometría\n"
            "antes de que el paso 15 V1.8 realizara el pulido final.\n"
            "Estos archivos permiten diferenciar datos observados de la\n"
            "geometría inferida para completar el modelo final.\n\n"
            f"CALIDAD paso 17: {quality}\n"
            f"CIERRE: {validation.get('closure_status')}\n\n"
            "Un estado warning NO modifica ni invalida automáticamente la\n"
            "exportación. Las advertencias completas están en\n"
            "resumen_exportacion_18.json.\n"
        ),
        encoding="utf-8",
    )

    # Recalcular hashes después del README/reporte.
    report["files"] = {}
    for f in sorted(temp_final.iterdir()):
        if f.is_file():
            report["files"][f.name] = {
                "size_bytes": int(f.stat().st_size),
                "sha256": sha256_file(f),
            }
    atomic_json(export_report_copy, report)

    # Publicación atómica a nivel de carpeta.
    if final_dir.exists():
        shutil.rmtree(final_dir)
    temp_final.replace(final_dir)

    # Resumen del paso dentro de multisesion para el checkpoint.
    phase_dir = multi / args.output_name
    phase_dir.mkdir(parents=True, exist_ok=True)

    final_obj = final_dir / "modelo_final.obj"
    final_metric_obj = final_dir / "modelo_final_metric_mm.obj"
    final_ply = final_dir / "modelo_final_original_mm.ply"
    final_pre_polish_obj = final_dir / "modelo_pre_pulido_metric_mm.obj"
    final_pre_polish_ply = final_dir / "modelo_pre_pulido_mm.ply"

    phase_report = dict(report)
    phase_report["result_directory"] = str(final_dir)
    phase_report["outputs"] = {
        "blender_obj": str(final_obj),
        "scientific_obj_mm": str(final_metric_obj),
        "original_ply_mm": str(final_ply),
        "pre_polish_obj_mm": str(final_pre_polish_obj),
        "pre_polish_ply_mm": str(final_pre_polish_ply),
        "readme": str(final_dir / "README_IMPORTAR_EN_BLENDER.txt"),
        "export_report": str(final_dir / "resumen_exportacion_18.json"),
    }

    phase_summary = phase_dir / "resumen_18_exportacion_modelo.json"
    atomic_json(phase_summary, phase_report)

    print("\n========== PASO 18 V1.4 COMPLETADA ==========")
    print("Calidad heredada de paso 17:", quality)
    print("Vértices:", len(mesh.vertices))
    print("Triángulos:", len(mesh.triangles))
    print("OBJ Blender:", final_obj)
    print("OBJ científico mm:", final_metric_obj)
    print("PLY original:", final_ply)
    print("resultado_final:", final_dir)
    print("================================================\n")

    return 0


def _entrada_con_diagnostico():
    """Invoca el paso desde consola con el diagnóstico de ejecución configurado."""
    from utilidades_progreso import informar_inicio

    informar_inicio(__file__, "18", "Exportar modelo para Blender")
    raise SystemExit(main())


if __name__ == "__main__":
    from utilidades_progreso import ejecutar_con_diagnostico

    ejecutar_con_diagnostico(_entrada_con_diagnostico, __file__)
