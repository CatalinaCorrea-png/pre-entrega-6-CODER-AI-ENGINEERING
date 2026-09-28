"""El pipeline de ingesta, en orden de ejecución.

    data/*.txt -> limpiar -> fragmentar -> metadata -> embeber -> subir a Pinecone

Es el pipeline de la pre-entrega 4 con otro corpus: las políticas internas de TechCorp de la
pre-entrega 3. Lo que cambia es solo lo que depende de los datos —la limpieza, la carga y la
metadata—; el embedding y la subida a Pinecone (etapa 5) son los mismos.
"""

import re
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DATA_DIR, NAMESPACE, TEXT_KEY, ErrorDeUso
from .pinecone import TASK_DOCUMENTO, obtener_embeddings, obtener_indice


# --- 1. Limpieza: sacar los artefactos de la extracción de PDF --------------

# Tomada de la pre-entrega 3, que indexaba estos mismos documentos. Vienen como los devuelve
# un extractor de PDF: encabezados y pies repetidos en cada carilla, palabras cortadas con
# guión, guiones blandos, ligaduras y espacios duros. Nada de eso aporta significado, pero
# todo termina dentro del embedding y del índice de BM25: una palabra con un guión blando
# adentro no coincide con la misma palabra escrita en la consulta.

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")
_PIE_DE_PAGINA = re.compile(r"^\s*p[áa]gina\s+\d+\s+de\s+\d+\s*$", re.IGNORECASE | re.MULTILINE)
# Invisibles: ancho cero, guión blando, BOM.
_INVISIBLES = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)


def _reparar_mojibake(texto: str) -> str:
    """Revierte un texto UTF-8 que fue decodificado como Latin-1."""
    if "Ã" not in texto and "Â" not in texto:
        return texto
    try:
        return texto.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto  # no era mojibake, o no es reversible: mejor no tocar


def _quitar_lineas_repetidas(texto: str, minimo: int = 3) -> str:
    """Elimina encabezados y pies que se repiten carilla tras carilla.

    Una línea corta que aparece 3 o más veces en un mismo documento es casi con certeza un
    artefacto de paginado, no contenido.
    """
    lineas = texto.split("\n")
    conteo = Counter(linea.strip() for linea in lineas if linea.strip())
    repetidas = {linea for linea, n in conteo.items() if n >= minimo and len(linea) <= 120}
    return "\n".join(linea for linea in lineas if linea.strip() not in repetidas)


def limpiar(texto: str) -> str:
    """Devuelve el texto sin los artefactos de la extracción de PDF."""
    texto = _reparar_mojibake(texto)
    # NFKC normaliza ligaduras y espacios duros (U+00A0 -> espacio normal).
    texto = unicodedata.normalize("NFKC", texto)
    texto = texto.translate(_INVISIBLES)
    texto = _CONTROL.sub("\n", texto)  # el form feed pasa a ser un salto de línea común

    texto = _PIE_DE_PAGINA.sub("", texto)
    texto = _quitar_lineas_repetidas(texto)

    # Palabra cortada por el ajuste de línea: "antigüe-\ndad" -> "antigüedad".
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)

    texto = re.sub(r"[ \t]{2,}", " ", texto)
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


# --- 2. Carga: leer el corpus desde data/ -----------------------------------

class CorpusVacio(ErrorDeUso):
    """No hay documentos que ingestar."""


# Título legible de cada documento, para la metadata. Explícito y no extraído del texto: la
# primera línea de estos archivos es un encabezado de paginado ("TechCorp — Documento
# Interno") que la limpieza elimina, así que "la primera línea" no es un criterio estable.
TITULOS = {
    "politica_vacaciones.txt": "Política de Licencias y Vacaciones",
    "politica_teletrabajo.txt": "Política de Teletrabajo",
    "politica_seguridad_informatica.txt": "Política de Seguridad Informática",
    "onboarding_nuevos_empleados.txt": "Proceso de Onboarding",
}


def titulo_de(source: str) -> str:
    return TITULOS.get(source, Path(source).stem.replace("_", " ").capitalize())


def cargar_documentos(directorio: Path = DATA_DIR, patron: str = "*.txt") -> list[Document]:
    """Lee el corpus y lo devuelve limpio, con `source` normalizado al nombre de archivo."""
    if not directorio.exists() or not any(directorio.glob(patron)):
        raise CorpusVacio(
            f"No hay archivos {patron} en {directorio}.\n"
            "Las políticas de TechCorp vienen en el repositorio: verificá la carpeta data/."
        )

    loader = DirectoryLoader(
        str(directorio),
        glob=patron,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documentos = []
    for doc in loader.load():
        nombre = Path(doc.metadata["source"]).name
        documentos.append(
            Document(
                page_content=limpiar(doc.page_content),
                metadata={"source": nombre, "titulo": titulo_de(nombre)},
            )
        )

    return sorted(documentos, key=lambda d: d.metadata["source"])


# --- 3. Fragmentación -------------------------------------------------------

# Los de la pre-entrega 3, medidos sobre este mismo corpus. Los de la pre-entrega 4 (800/140)
# estaban calibrados para documentación técnica con bloques de código largos, que acá no hay.
CHUNK_SIZE = 500      # tokens
CHUNK_OVERLAP = 70    # tokens

# Se prueban en orden: el splitter usa el primero que logre respetar el tamaño.
SEPARADORES = ["\n\n", "\n", " ", ""]


def obtener_splitter() -> RecursiveCharacterTextSplitter:
    """Splitter que mide en tokens reales, no en caracteres."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARADORES,
        keep_separator=True,
    )


def fragmentar(documentos: list[Document]) -> list[Document]:
    """Divide los documentos en fragmentos, preservando su metadata de origen."""
    return obtener_splitter().split_documents(documentos)


# --- 4. Metadatos: el contrato, en un solo lugar ----------------------------

# Contrato de metadatos. Cualquier pipeline que escriba en este índice escribe esto.
# Sin `url`, a diferencia de la pre-entrega 4: son documentos internos, no tienen una página
# pública que citar. La cita es `source`.
CAMPOS = (
    "text",           # str  - contenido del fragmento (lo inyecta PineconeVectorStore)
    "source",         # str  - nombre de archivo, ej. "politica_vacaciones.txt"
    "titulo",         # str  - título legible del documento de origen
    "categoria",      # str  - agrupación temática, para filtrar por metadata
    "doc_type",       # str  - tipo de documento
    "chunk_index",    # int  - posición del fragmento dentro de su documento
    "total_chunks",   # int  - cuántos fragmentos produjo ese documento
    "n_caracteres",   # int  - tamaño del fragmento
)

# Pinecone rechaza el vector entero si su metadata supera 40 KB. Un fragmento de 500 tokens
# ronda los 2 KB, así que hay margen de sobra, pero el corte explícito evita que un
# documento raro tire abajo un lote completo.
LIMITE_TEXTO_BYTES = 30_000

DOC_TYPE = "politica_interna"

# Habilita el "hard filter" por metadata: `filter={"categoria": {"$eq": "licencias"}}`
# acota el espacio de búsqueda antes de comparar similitud.
CATEGORIAS = {
    "politica_vacaciones.txt": "licencias",
    "politica_teletrabajo.txt": "modalidad_de_trabajo",
    "politica_seguridad_informatica.txt": "seguridad",
    "onboarding_nuevos_empleados.txt": "recursos_humanos",
}


def categoria_de(source: str) -> str:
    return CATEGORIAS.get(source, "sin_categoria")


def id_vector(source: str, chunk_index: int) -> str:
    """ID determinístico del vector: `archivo.txt::7`."""
    return f"{source}::{chunk_index}"


def aplicar_metadata(chunks: list[Document]) -> list[Document]:
    """Completa la metadata de cada fragmento según el contrato de `CAMPOS`."""
    por_documento: dict[str, list[Document]] = {}
    for chunk in chunks:
        por_documento.setdefault(chunk.metadata.get("source", "desconocido"), []).append(chunk)

    resultado = []
    for source, fragmentos in por_documento.items():
        total = len(fragmentos)
        for indice, chunk in enumerate(fragmentos):
            texto = chunk.page_content[:LIMITE_TEXTO_BYTES]
            chunk.page_content = texto
            chunk.metadata = {
                "source": source,
                "titulo": chunk.metadata.get("titulo", titulo_de(source)),
                "categoria": categoria_de(source),
                "doc_type": DOC_TYPE,
                "chunk_index": indice,
                "total_chunks": total,
                "n_caracteres": len(texto),
            }
            resultado.append(chunk)

    return sorted(resultado, key=lambda d: (d.metadata["source"], d.metadata["chunk_index"]))


def ids_de(chunks: list[Document]) -> list[str]:
    return [id_vector(c.metadata["source"], c.metadata["chunk_index"]) for c in chunks]


# --- 5. Embedding y upsert: dos lotes, dos límites --------------------------

# Acotado por los tokens por minuto de Gemini, no por Pinecone: ~15 fragmentos de este
# corpus son ~8.000 tokens, que es lo que la capa gratuita acepta por llamada.
LOTE_EMBEDDING = 15

LOTE_UPSERT = 100

# Pausa entre lotes de embedding, para no agotar la cuota por minuto. Con lotes de ~8.000
# tokens cada 20 segundos quedan ~24.000 tokens por minuto, debajo del límite medido.
PAUSA_ENTRE_LOTES = 20.0

INTENTOS = 4

# La cuota de Gemini se repone por ventana de un minuto, así que el primer reintento tiene
# que esperar más de un minuto. Un backoff que arranca en 4s agota los intentos dentro de
# la misma ventana y falla igual, solo que más tarde.
ESPERA_INICIAL = 65.0

SENALES_TRANSITORIAS = (
    "429", "rate limit", "resource_exhausted", "quota",
    "503", "unavailable", "timeout", "deadline",
)


def _es_transitorio(error: Exception) -> bool:
    mensaje = str(error).lower()
    return any(senal in mensaje for senal in SENALES_TRANSITORIAS)


def _con_reintentos(accion: Callable, descripcion: str):
    """Reintenta con backoff exponencial solo ante fallas transitorias."""
    espera = ESPERA_INICIAL
    for intento in range(1, INTENTOS + 1):
        try:
            return accion()
        except Exception as error:
            if intento == INTENTOS or not _es_transitorio(error):
                raise
            print(
                f"   ⏳ {descripcion}: cuota agotada, "
                f"reintento {intento}/{INTENTOS - 1} en {espera:.0f}s"
            )
            time.sleep(espera)
            espera *= 2


def _en_lotes(elementos: list, tamano: int) -> Iterator[list]:
    for inicio in range(0, len(elementos), tamano):
        yield elementos[inicio : inicio + tamano]


def embeber(
    chunks: list[Document],
    al_terminar_lote: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """Genera los embeddings de todos los fragmentos, respetando la cuota por minuto."""
    modelo = obtener_embeddings(TASK_DOCUMENTO)
    vectores: list[list[float]] = []
    lotes = list(_en_lotes(chunks, LOTE_EMBEDDING))

    for numero, lote in enumerate(lotes, start=1):
        textos = [chunk.page_content for chunk in lote]
        vectores.extend(
            _con_reintentos(lambda: modelo.embed_documents(textos), f"embedding {numero}/{len(lotes)}")
        )
        if al_terminar_lote:
            al_terminar_lote(len(vectores), len(chunks))
        if numero < len(lotes):
            time.sleep(PAUSA_ENTRE_LOTES)

    return vectores


def subir(
    chunks: list[Document],
    vectores: list[list[float]],
    namespace: str = NAMESPACE,
) -> int:
    """Hace upsert de los vectores con su metadata, en lotes de `LOTE_UPSERT`."""
    indice = obtener_indice()
    ids = ids_de(chunks)

    registros = [
        {
            "id": identificador,
            "values": vector,
            # El texto original viaja acá adentro: una sola consulta a Pinecone devuelve
            # vector, contenido y fuente, sin una segunda base de datos que consultar.
            "metadata": {**chunk.metadata, TEXT_KEY: chunk.page_content},
        }
        for identificador, chunk, vector in zip(ids, chunks, vectores, strict=True)
    ]

    for lote in _en_lotes(registros, LOTE_UPSERT):
        _con_reintentos(
            lambda lote=lote: indice.upsert(vectors=lote, namespace=namespace), "upsert"
        )

    return len(registros)


def borrar_namespace(namespace: str = NAMESPACE) -> None:
    """Vacía el namespace. Útil para reingestar desde cero tras cambiar el chunking."""
    indice = obtener_indice()
    stats = indice.describe_index_stats()
    if namespace in (stats.get("namespaces") or {}):
        indice.delete(delete_all=True, namespace=namespace)
