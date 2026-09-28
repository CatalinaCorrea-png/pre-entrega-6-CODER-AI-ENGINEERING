"""Ingesta el corpus de data/ al índice de Pinecone.

    python scripts/ingestar.py
    python scripts/ingestar.py --limpiar   # vacía el namespace antes de subir

Los IDs de los vectores son determinísticos, así que reingestar el mismo corpus actualiza
en vez de duplicar. `--limpiar` hace falta solo cuando cambió el chunking y quedan
fragmentos viejos huérfanos en el índice.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import correr, INDEX_NAME, NAMESPACE
from rag.pinecone import crear_indice_si_falta, estadisticas
from rag.ingesta import (
    aplicar_metadata,
    borrar_namespace,
    cargar_documentos,
    embeber,
    fragmentar,
    subir,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--limpiar", action="store_true", help="vacía el namespace antes de subir")
    args = parser.parse_args()

    # Validar el índice antes de generar un solo embedding: si la dimensión no coincide el
    # upsert va a fallar igual, pero para entonces ya se pagaron los embeddings.
    crear_indice_si_falta()

    documentos = cargar_documentos()
    chunks = aplicar_metadata(fragmentar(documentos))
    print(
        f"{len(documentos)} documentos -> {len(chunks)} fragmentos "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )

    if args.limpiar:
        borrar_namespace(args.namespace)

    # printeamos el progreso del embedding porque son varios minutos 
    # por la cuota por minuto de Gemini, y sin señal parece colgado.
    vectores = embeber(
        chunks, al_terminar_lote=lambda hechos, total: print(f"  embeddings {hechos}/{total}")
    )
    subir(chunks, vectores, namespace=args.namespace)

    stats = estadisticas()
    print(f"Listo: {stats['vectores_totales']} vectores en '{INDEX_NAME}' / '{args.namespace}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(correr(main))
