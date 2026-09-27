"""Prepara una muestra pequeña y trazable; nunca altera campañas originales."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent.parent
SELECTION = {
    "cilindro1": "cilindro1_20260919_132249",
    "cilindro2": "cilindro2_20260919_132519",
    "cilindro3": "cilindro3_20260919_132734",
    "cubo1": "cubo1_20260919_130526",
    "cubo2": "cubo2_20260919_131206",
    "cubo3": "ubo3_20260919_131506",
    "piramide1": "piramide1_20260920_183454",
    "piramide3": "Piramide3_20260922_211146",
    "piramide4": "piramide4_20260922_175555",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str):
        # Los informes históricos contienen rutas del equipo del autor.
        if re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("/"):
            return "ruta_local_omitida/" + value.replace("\\", "/").rsplit("/", 1)[-1]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "resultados")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("La carpeta destino ya existe. Use --output con una carpeta nueva para no sobrescribir evidencia publicada.")
    # Comprueba que todos los recursos existen antes de crear el destino.
    plans = []
    for name, campaign in SELECTION.items():
        job = ROOT / "trabajos" / campaign
        final = job / "resultado_final"
        validation = job / "reconstruccion/multisesion/17_validacion_modelo/resumen_17_validacion_modelo.json"
        left = sorted((job / "capturas/S01/izquierda").glob("*_L.png"))[0]
        right = job / "capturas/S01/derecha" / left.name.replace("_L.png", "_R.png")
        files = {file: final / file for file in ("modelo_final_original_mm.ply", "modelo_pre_pulido_mm.ply", "preview_validacion.png")}
        files.update({"captura_izquierda.png": left, "captura_derecha.png": right})
        for path in [*files.values(), validation, final / "resumen_exportacion_18.json"]:
            if not path.is_file():
                raise FileNotFoundError(path)
        plans.append((name, campaign, files, validation, final / "resumen_exportacion_18.json"))
    args.output.mkdir(parents=True)
    for filename in ("medidas_fisicas.md", "medidas_fisicas.json"):
        shutil.copy2(ROOT / "documentacion" / filename, args.output / filename)
    measurements = json.loads((args.output / "medidas_fisicas.json").read_text(encoding="utf-8"))
    rows = []
    for name, campaign, files, validation, export in plans:
        dest = args.output / name
        dest.mkdir()
        origins = {}
        for filename, source in files.items():
            shutil.copy2(source, dest / filename)
            origins[filename] = {"source": source.relative_to(ROOT).as_posix(), "sha256": digest(source)}
        data = json.loads(validation.read_text(encoding="utf-8"))
        (dest / "validacion_17.json").write_text(json.dumps(portable(data), indent=2, ensure_ascii=False), encoding="utf-8")
        exported = json.loads(export.read_text(encoding="utf-8"))
        summary = {"campaign": campaign, "quality": data.get("quality"), "warnings": data.get("warning_reasons", []), "reject_reasons": data.get("reject_reasons", []), "units": "mm", "source_validation_sha256": digest(validation), "source_export_report_sha256": digest(export), "export_quality": exported.get("quality"), "origins": origins, "validation_transformation": "Solo rutas absolutas anonimizadas; métricas y advertencias conservadas.", "scope": "Muestra histórica ilustrativa. No es una campaña completa ni una validación dimensional externa."}
        object_type = re.sub(r"\d+$", "", name)
        summary["public_name"] = name
        summary["physical_reference_mm"] = measurements["objects"][object_type]
        summary["physical_reference_method"] = "Regla; valores aproximados, incertidumbre no especificada."
        summary["dimensional_comparison_performed"] = False
        (dest / "procedencia.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (dest / "README.md").write_text(f"# {name.capitalize()}\n\nCampaña: `{campaign}`. Calidad interna: **{summary['quality']}**.\n\n![Validación](preview_validacion.png)\n\nSe incluyen el PLY final en milímetros, el PLY anterior al pulido, un par estéreo ilustrativo, las métricas del paso 17 y la procedencia. El modelo previo al pulido puede contener relleno de etapas anteriores: no equivale a una malla puramente observada.\n\n## Advertencias del resultado\n\n" + ("\n".join("- " + w for w in summary['warnings']) or "Sin advertencias internas reportadas.") + "\n\nConsulte las [medidas físicas aproximadas con regla](../medidas_fisicas.md). Se incluyen como referencia; no se ha calculado aquí una discrepancia dimensional. Accepted indica superar controles internos, no exactitud dimensional demostrada.\n", encoding="utf-8")
        rows.append(summary)
    (args.output / "catalogo.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    selection_text = "Selección solicitada: cilindros 1, 2 y 3; cubos 1, 2 y 3; pirámides 1, 3 y 4. Se mantienen las calidades y advertencias históricas. `cubo3` corresponde a la carpeta original `ubo3_20260919_131506`, cuyo nombre se escribió sin la C. Las campañas originales no se renombran.\n\n" + "\n".join(f"- [{name}]({name}/README.md): `{campaign}`" for name, campaign in SELECTION.items()) + "\n\nConsulte las [medidas físicas aproximadas](medidas_fisicas.md) y su [registro JSON](medidas_fisicas.json). Las nueve campañas se asocian a las referencias por tipo de objeto según la selección confirmada por el responsable."
    (args.output / "README.md").write_text("# Resultados seleccionados\n\n" + selection_text + "\n\n## Uso y reproducción\n\nAbra los PLY con un visor compatible. Sus coordenadas están en milímetros: para trabajar en metros aplique escala 0.001. Las imágenes son evidencia ilustrativa, no las 75 parejas de una campaña. Las mallas no se simplificaron ni se recalcularon al preparar este paquete.\n\nPara repetir el procesamiento se necesitan las capturas completas S01/S02/S03, sus referencias congeladas y el modelo correspondiente, que permanecen en trabajos y no se duplican aquí. Abra la campaña original desde la aplicación y procese con conservación completa; un cambio de código puede modificar los resultados. Este paquete permite inspección de los resultados publicados, no reproducción numérica completa por sí solo.\n\nPara volver a generar esta selección local: `python herramientas/preparar_resultados_publicos.py --output resultados_nuevos`. Consulte `procedencia.json` en cada ejemplo y `catalogo.json`. Las rutas personales se omiten de los informes publicados; por ello sus huellas difieren de los originales. No se inventaron medidas físicas.\n\n`MANIFEST.json` registra los bytes de todos los archivos del paquete excepto él mismo. Git conserva estos archivos sin conversión de finales de línea.\n", encoding="utf-8")
    manifest = {p.relative_to(args.output).as_posix(): {"sha256": digest(p), "size_bytes": p.stat().st_size} for p in sorted(args.output.rglob("*")) if p.is_file()}
    (args.output / "MANIFEST.json").write_text(json.dumps({"schema_version": 1, "excludes": ["MANIFEST.json"], "files": manifest}, indent=2), encoding="utf-8")
    from comparar_dimensiones import main as comparar
    comparar(["--results", str(args.output)])
    print(f"Selección y comparación preparadas en {args.output}")


if __name__ == "__main__":
    main()
