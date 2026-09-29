"""Construye el sitio estático sin modificar los resultados científicos.

Ejecutar desde cualquier directorio: python herramientas/preparar_visor_web.py
La salida _site/ es generada; servir con python -m http.server 8000 --directory _site.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "_site"
    # Solo se limpia esta ruta fija de salida, nunca las campañas ni los originales.
    if output.is_symlink():
        raise RuntimeError("_site no puede ser un enlace simbólico")
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(ROOT / "visor", output)
    shutil.copy2(ROOT / "LICENSE", output / "LICENSE")
    results = output / "resultados"
    results.mkdir()
    for name in ("catalogo.json", "comparacion_dimensional.json"):
        shutil.copy2(ROOT / "resultados" / name, results / name)
    catalog = json.loads((results / "catalogo.json").read_text(encoding="utf-8"))
    names = set()
    for item in catalog:
        name = item["public_name"]
        if not name.isalnum() or name in names:
            raise ValueError(f"Nombre público inválido o duplicado: {name}")
        names.add(name)
        folder = results / name
        folder.mkdir()
        for filename in ("modelo_final_original_mm.ply", "modelo_pre_pulido_mm.ply",
                         "captura_izquierda.png", "captura_derecha.png",
                         "preview_validacion.png", "validacion_17.json"):
            shutil.copy2(ROOT / "resultados" / name / filename, folder / filename)
    (output / ".nojekyll").touch()
    print(f"Sitio preparado en {output}: {len(names)} campañas. Originales intactos.")


if __name__ == "__main__":
    main()
