"""Documenta argumentos y valores declarados sin importar etapas ni abrir hardware."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    lines = ["# Referencia de parámetros declarados", "", "Esta referencia se extrae de los `add_argument` de los scripts. Muestra los valores declarados en el código; el coordinador puede sustituirlos al ejecutar una etapa. Para conocer los valores utilizados en una campaña, consulte sus informes y checkpoints. `sin valor explícito` no implica necesariamente `None`, porque argparse puede obtener el valor de la acción. Las expresiones se transcriben sin ejecutarlas.", "", "Regenerar: `python herramientas/generar_referencia_parametros.py`.", ""]
    for path in sorted((ROOT / "procesamiento").glob("[0-9][0-9]_*.py")) + sorted((ROOT / "herramientas").glob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        rows = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            flags = [v.value for v in node.args if isinstance(v, ast.Constant) and isinstance(v.value, str)]
            if not flags:
                continue
            kw = {v.arg: v.value for v in node.keywords}
            def value(name, fallback="—"):
                if name not in kw:
                    return fallback
                try:
                    return str(ast.literal_eval(kw[name]))
                except (ValueError, TypeError):
                    return ast.unparse(kw[name])
            def clean(text):
                return text.replace("|", "\\|").replace("\n", " ")
            rows.append((node.lineno, f"| `{', '.join(flags)}` | {clean(value('default', 'sin valor explícito'))} | {clean(value('choices'))} | {clean(value('required', 'False'))} | {clean(value('help'))} |"))
        if rows:
            lines += [f"## [{path.name}](../{path.relative_to(ROOT).as_posix()})", "", "| Argumento | Valor declarado | Opciones | Obligatorio | Descripción del script |", "| --- | --- | --- | --- | --- |", *[row for _, row in sorted(rows)], ""]
    destination = ROOT / "documentacion/referencia_parametros.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
