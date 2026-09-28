"""El modelo de embeddings y el índice de Pinecone.

Viven juntos porque `DIMENSION` tiene que ser la misma para los dos: una sola copia, a la
vista, no puede divergir. Es lo que descarta el "mismatch de dimensiones".

La ingesta upserta con el SDK nativo, porque necesita controlar sus lotes; la consulta usa
`PineconeVectorStore`, que es lo que `EnsembleRetriever` sabe consumir.
"""

import time
from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from .config import (
    INDEX_NAME,
    NAMESPACE,
    PINECONE_CLOUD,
    PINECONE_REGION,
    TEXT_KEY,
    ErrorDeUso,
    variable_obligatoria,
)


# --- El modelo de embeddings y la dimensión del índice ----------------------

MODELO = "models/gemini-embedding-001"
DIMENSION = 1536
METRICA = "cosine"

# Gemini pide declarar para qué se va a usar el vector. El modelo proyecta
# distinto un pasaje que se va a indexar que una pregunta que lo busca, y usar el par
# correcto mejora la recuperación sobre usar el mismo tipo para ambos lados.
TASK_DOCUMENTO = "RETRIEVAL_DOCUMENT"
TASK_CONSULTA = "RETRIEVAL_QUERY"


@lru_cache(maxsize=2)
def obtener_embeddings(task_type: str = TASK_DOCUMENTO) -> GoogleGenerativeAIEmbeddings:
    """Cliente de embeddings, cacheado por task_type."""
    return GoogleGenerativeAIEmbeddings(
        model=MODELO,
        google_api_key=variable_obligatoria("GOOGLE_API_KEY"),
        output_dimensionality=DIMENSION,
        task_type=task_type,
    )


# --- Cliente e índice Serverless, idempotente -------------------------------

ESPERA_MAXIMA_SEGUNDOS = 120


class IndiceIncompatible(ErrorDeUso):
    """El índice existe pero no coincide con la configuración del proyecto."""


@lru_cache(maxsize=1)
def obtener_cliente() -> Pinecone:
    """Cliente de Pinecone, cacheado por proceso."""
    return Pinecone(api_key=variable_obligatoria("PINECONE_API_KEY"))


def listar_indices() -> list[str]:
    return [indice["name"] for indice in obtener_cliente().list_indexes()]


def verificar_compatibilidad(nombre: str = INDEX_NAME) -> dict:
    """Compara el índice existente contra la configuración local."""
    descripcion = obtener_cliente().describe_index(nombre)
    problemas = []

    if descripcion.dimension != DIMENSION:
        problemas.append(
            f"dimensión del índice = {descripcion.dimension}, "
            f"pero el modelo produce vectores de {DIMENSION}"
        )
    if descripcion.metric != METRICA:
        problemas.append(f"métrica del índice = '{descripcion.metric}', esperada '{METRICA}'")

    if problemas:
        raise IndiceIncompatible(
            f"El índice '{nombre}' ya existe pero no sirve para este proyecto:\n"
            + "".join(f"  - {p}\n" for p in problemas)
            + "Elegí una: borralo desde app.pinecone.io, o usá otro nombre con "
            "INDEX_NAME en tu .env."
        )

    return {
        "nombre": nombre,
        "dimension": descripcion.dimension,
        "metrica": descripcion.metric,
        "host": descripcion.host,
        "listo": descripcion.status.get("ready", False),
    }


def crear_indice_si_falta(nombre: str = INDEX_NAME) -> dict:
    """Crea el índice Serverless si no existe; si existe, lo valida. Idempotente."""
    if nombre in listar_indices():
        return {"creado": False, **verificar_compatibilidad(nombre)}

    obtener_cliente().create_index(
        name=nombre,
        dimension=DIMENSION,
        metric=METRICA,
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    esperar_a_que_este_listo(nombre)
    return {"creado": True, **verificar_compatibilidad(nombre)}


def esperar_a_que_este_listo(nombre: str = INDEX_NAME) -> None:
    """El índice tarda unos segundos en aprovisionarse; escribirle antes falla."""
    limite = time.monotonic() + ESPERA_MAXIMA_SEGUNDOS
    while time.monotonic() < limite:
        if obtener_cliente().describe_index(nombre).status.get("ready"):
            return
        time.sleep(2)
    raise TimeoutError(
        f"El índice '{nombre}' no quedó listo en {ESPERA_MAXIMA_SEGUNDOS}s. "
        "Revisá su estado en app.pinecone.io."
    )


def obtener_indice(nombre: str = INDEX_NAME):
    """Handle del índice para operaciones del SDK nativo (upsert, list, fetch)."""
    return obtener_cliente().Index(nombre)


def estadisticas(nombre: str = INDEX_NAME) -> dict:
    """`describe_index_stats()` como dict plano, para imprimir."""
    stats = obtener_indice(nombre).describe_index_stats()
    return {
        "vectores_totales": stats.get("total_vector_count", 0),
        "dimension": stats.get("dimension"),
        "namespaces": {
            nombre_ns: datos.get("vector_count", 0)
            for nombre_ns, datos in (stats.get("namespaces") or {}).items()
        },
    }


# --- El vector store del lado de la recuperación ----------------------------

__all__ = ["obtener_vectorstore", "TASK_CONSULTA", "TASK_DOCUMENTO"]


@lru_cache(maxsize=4)
def obtener_vectorstore(
    task_type: str = TASK_CONSULTA,
    namespace: str = NAMESPACE,
) -> PineconeVectorStore:
    """Vector store apuntado al índice y namespace configurados."""
    return PineconeVectorStore(
        index=obtener_indice(),
        embedding=obtener_embeddings(task_type),
        text_key=TEXT_KEY,
        namespace=namespace,
    )
