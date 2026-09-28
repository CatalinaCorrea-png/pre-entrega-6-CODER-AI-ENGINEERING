"""Crea el índice Serverless de Pinecone si no existe, y lo valida si ya existe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import correr, INDEX_NAME, PINECONE_CLOUD, PINECONE_REGION
from rag.pinecone import crear_indice_si_falta, estadisticas, DIMENSION, METRICA


def main() -> int:
    info = crear_indice_si_falta()
    estado = "creado" if info["creado"] else "ya existía y es compatible"
    print(
        f"Índice '{INDEX_NAME}' {estado}: {DIMENSION} dims, {METRICA}, "
        f"serverless {PINECONE_CLOUD}/{PINECONE_REGION}"
    )

    stats = estadisticas()
    detalle = ", ".join(f"{n}={c}" for n, c in sorted(stats["namespaces"].items()))
    print(f"{stats['vectores_totales']} vectores" + (f" ({detalle})" if detalle else ""))

    if not stats["namespaces"]:
        print("Todavía sin vectores: python scripts/ingestar.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(correr(main))
