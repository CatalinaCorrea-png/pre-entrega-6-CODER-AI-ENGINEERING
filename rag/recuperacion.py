"""De una consulta a los k fragmentos más relevantes.

    consulta -> BM25 (léxico) + Pinecone (vectorial) -> RRF -> top-k

`RAGSystem`, al final del archivo, es la fachada; arriba están sus piezas en el orden en
que las necesita. Sus tres modos existen para que la evaluación pueda medir cada mitad por
separado: sin ese contrafáctico.
"""

import re
import unicodedata
from functools import cached_property

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .config import NAMESPACE, TEXT_KEY, ErrorDeUso
from .pinecone import TASK_CONSULTA, obtener_indice, obtener_vectorstore


# --- Tokenizador de BM25 ----------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenizar(texto: str) -> list[str]:
    """Normaliza y parte en términos comparables."""
    descompuesto = unicodedata.normalize("NFKD", texto.lower())
    sin_acentos = "".join(c for c in descompuesto if not unicodedata.combining(c))
    return _TOKEN.findall(sin_acentos)


# --- El corpus léxico, reconstruido desde Pinecone --------------------------

TAMANO_LOTE_FETCH = 100


class CorpusVacioEnPinecone(ErrorDeUso):
    """El namespace existe pero no tiene vectores."""


def _metadata_de(vector) -> dict:
    """El SDK devuelve objetos o dicts según la versión; normalizamos a dict."""
    if isinstance(vector, dict):
        return vector.get("metadata") or {}
    return getattr(vector, "metadata", None) or {}


def corpus_desde_pinecone(namespace: str = NAMESPACE) -> list[Document]:
    """Pagina el namespace y reconstruye los fragmentos desde la metadata."""
    indice = obtener_indice()
    documentos: list[Document] = []

    for lote_ids in indice.list(namespace=namespace):
        for inicio in range(0, len(lote_ids), TAMANO_LOTE_FETCH):
            ids = lote_ids[inicio : inicio + TAMANO_LOTE_FETCH]
            respuesta = indice.fetch(ids=ids, namespace=namespace)
            vectores = respuesta.get("vectors") if isinstance(respuesta, dict) else respuesta.vectors
            for vector in (vectores or {}).values():
                metadata = dict(_metadata_de(vector))
                texto = metadata.pop(TEXT_KEY, "")
                if texto:
                    documentos.append(Document(page_content=texto, metadata=metadata))

    if not documentos:
        raise CorpusVacioEnPinecone(
            f"El namespace '{namespace}' no tiene vectores.\n"
            "Corré primero: python scripts/ingestar.py"
        )

    return _ordenar(documentos)


def corpus_desde_disco() -> list[Document]:
    """Re-deriva los fragmentos desde `data/`, sin tocar la red."""
    # Import diferido a propósito: `rag.ingesta` arrastra los loaders de LangChain, y
    # consultar el índice no tiene por qué pagar esa importación.
    from .ingesta import aplicar_metadata, cargar_documentos, fragmentar

    return _ordenar(aplicar_metadata(fragmentar(cargar_documentos())))


def obtener_corpus(origen: str = "pinecone", namespace: str = NAMESPACE) -> list[Document]:
    if origen == "pinecone":
        return corpus_desde_pinecone(namespace)
    if origen == "local":
        return corpus_desde_disco()
    raise ValueError(f"Origen desconocido: {origen!r}. Usá 'pinecone' o 'local'.")


def _ordenar(documentos: list[Document]) -> list[Document]:
    """Orden estable: BM25 desempata por posición, así que el orden altera resultados."""
    return sorted(
        documentos,
        key=lambda d: (d.metadata.get("source", ""), d.metadata.get("chunk_index", 0)),
    )


# --- Los tres recuperadores: léxico, vectorial e híbrido --------------------

K_POR_DEFECTO = 5

# 50/50, el punto neutro. `evaluate.py --sweep` mide el efecto de moverlo: sobre este corpus
# el barrido es monótono a favor del vectorial y 0.3/0.7 da mejores números en las cuatro
# métricas. Aun así el default no se mueve — ajustarlo para ganar en un benchmark de 10
# preguntas es ajustar el sistema al examen, y la diferencia no se distingue del ruido con
# esa muestra. Está a un flag de distancia: `--pesos 0.3 0.7`.
PESOS_POR_DEFECTO = (0.5, 0.5)


def retriever_bm25(corpus: list[Document], k: int = K_POR_DEFECTO) -> BM25Retriever:
    """Recuperador léxico sobre el corpus completo, en memoria."""
    return BM25Retriever.from_documents(corpus, k=k, preprocess_func=tokenizar)


def retriever_vectorial(
    k: int = K_POR_DEFECTO,
    namespace: str = NAMESPACE,
    filtro: dict | None = None,
) -> BaseRetriever:
    """Recuperador semántico contra Pinecone."""
    search_kwargs: dict = {"k": k, "namespace": namespace}
    if filtro:
        search_kwargs["filter"] = filtro

    return obtener_vectorstore(task_type=TASK_CONSULTA, namespace=namespace).as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )


def retriever_hibrido(
    corpus: list[Document],
    k: int = K_POR_DEFECTO,
    pesos: tuple[float, float] = PESOS_POR_DEFECTO,
    namespace: str = NAMESPACE,
    filtro: dict | None = None,
) -> EnsembleRetriever:
    """BM25 + vectorial, fusionados por RRF ponderado."""
    return EnsembleRetriever(
        retrievers=[
            retriever_bm25(corpus, k=k),
            retriever_vectorial(k=k, namespace=namespace, filtro=filtro),
        ],
        weights=list(pesos),
    )


# --- RAGSystem: la fachada del módulo ---------------------------------------

MODOS = ("hibrido", "vectorial", "bm25")


class RAGSystem:
    """Sistema de recuperación sobre el índice de Pinecone."""

    def __init__(
        self,
        k: int = K_POR_DEFECTO,
        modo: str = "hibrido",
        pesos: tuple[float, float] = PESOS_POR_DEFECTO,
        namespace: str = NAMESPACE,
        origen_corpus: str = "pinecone",
        corpus: list[Document] | None = None,
        filtro: dict | None = None,
    ) -> None:
        if modo not in MODOS:
            raise ValueError(f"Modo desconocido: {modo!r}. Opciones: {', '.join(MODOS)}")

        self.k = k
        self.modo = modo
        self.pesos = tuple(pesos)
        self.namespace = namespace
        self.origen_corpus = origen_corpus
        self.filtro = filtro
        # El corpus se puede inyectar para compartirlo entre varias instancias: en la
        # evaluación se comparan tres modos y bajarlo tres veces sería tiempo tirado.
        self._corpus = corpus

    @cached_property
    def corpus(self) -> list[Document]:
        if self._corpus is None:
            self._corpus = obtener_corpus(self.origen_corpus, self.namespace)
        return self._corpus

    @cached_property
    def retriever(self):
        if self.modo == "bm25":
            return retriever_bm25(self.corpus, k=self.k)
        if self.modo == "vectorial":
            return retriever_vectorial(k=self.k, namespace=self.namespace, filtro=self.filtro)
        return retriever_hibrido(
            self.corpus, k=self.k, pesos=self.pesos, namespace=self.namespace, filtro=self.filtro
        )

    def recuperar(self, consulta: str) -> list[Document]:
        """Los `k` fragmentos más relevantes para la consulta."""
        return self.retriever.invoke(consulta)[: self.k]

    def obtener_top_k(self, consulta: str) -> list[dict]:
        """Igual que `recuperar`, en dicts planos listos para imprimir o serializar."""
        return [
            {
                "contenido": documento.page_content,
                "fuente": documento.metadata.get("source", "desconocida"),
                "titulo": documento.metadata.get("titulo", ""),
                "categoria": documento.metadata.get("categoria", "sin_categoria"),
                "chunk_index": documento.metadata.get("chunk_index"),
                "url": documento.metadata.get("url", ""),
            }
            for documento in self.recuperar(consulta)
        ]

    def fuentes(self, consulta: str) -> list[str]:
        """Los nombres de archivo del top-k, en orden. Es lo que consume la evaluación."""
        return [d.metadata.get("source", "desconocida") for d in self.recuperar(consulta)]

    def __repr__(self) -> str:
        detalle = f", pesos={self.pesos}" if self.modo == "hibrido" else ""
        return f"RAGSystem(modo={self.modo!r}, k={self.k}{detalle}, namespace={self.namespace!r})"
