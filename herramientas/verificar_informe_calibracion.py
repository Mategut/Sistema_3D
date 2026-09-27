"""Verifica evidencia independiente sin modificar la calibración activa."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from procesamiento.utilidades_calibracion import validate_promotion_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate_promotion_evidence(args.candidate, args.evidence)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Informe no validado: {exc}")
        return 2
    print(f"Informe verificado: {len(result['campaigns'])} campañas. Responsable: {result['reviewer']}")
    print("La calibración activa no se ha modificado. Usa Activar con informe revisado para instalarla.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
