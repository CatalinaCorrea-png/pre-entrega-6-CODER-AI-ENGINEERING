"""Configuración, errores de uso y salida por consola.

Rutas resueltas desde este archivo y no desde el directorio de
trabajo, variables de entorno, `ErrorDeUso` para las fallas que quien las ve puede
arreglar, y `correr()`, que deja la consola en UTF-8 —Windows usa cp1252— y convierte un
`ErrorDeUso` en una línea legible en vez de un traceback.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


# --- Errores de uso, separados de los errores de programa -------------------

class ErrorDeUso(RuntimeError):
    """Falla que el usuario puede corregir; su mensaje dice cómo."""


# --- Rutas del proyecto y coordenadas del índice de Pinecone ----------------

RAIZ = Path(__file__).resolve().parent.parent

load_dotenv(RAIZ / ".env")

DATA_DIR = RAIZ / "data"                        # políticas internas de TechCorp (.txt)

# El nombre del índice y el namespace son configurables por entorno a propósito: es lo que
# permite levantar un índice de staging sin tocar el código.
#
# No es `techcorp-rag-hibrido`, el del notebook de la clase: ese se crea con embeddings de
# MiniLM (384 dims) y este proyecto usa los de Gemini (1536). Si el del notebook existe en la
# cuenta, compartir el nombre haría que `verificar_compatibilidad` lo rechace.
INDEX_NAME = os.getenv("INDEX_NAME", "techcorp-politicas")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "politicas-internas")

PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Clave donde viaja el texto original dentro de la metadata de cada vector. Vive acá porque
# la escriben tres módulos distintos y tienen que coincidir: la ingesta la usa para guardar
# el contenido, el vector store como `text_key` para leerlo, y la recuperación para
# reconstruir el corpus de BM25. Si divergieran, los documentos volverían con
# `page_content` vacío y sin ningún error visible.
TEXT_KEY = "text"


class ConfiguracionFaltante(ErrorDeUso):
    """Falta una variable de entorno obligatoria."""


def variable_obligatoria(nombre: str) -> str:
    """Devuelve la variable de entorno o falla con un mensaje accionable."""
    valor = (os.getenv(nombre) or "").strip()
    if not valor:
        raise ConfiguracionFaltante(
            f"Falta la variable {nombre}.\n"
            f"  1. copy .env.example .env      (cp en Linux / macOS)\n"
            f"  2. completá {nombre} en el archivo .env\n"
            f"El .env está en .gitignore: nunca se sube al repositorio."
        )
    return valor


# --- Salida por consola, también en Windows ---------------------------------

def configurar() -> None:
    """Fuerza UTF-8 en stdout y stderr, sin fallar si el flujo no lo soporta."""
    for flujo in (sys.stdout, sys.stderr):
        reconfigurar = getattr(flujo, "reconfigure", None)
        if reconfigurar is None:
            continue
        try:
            reconfigurar(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def correr(main) -> int:
    """Ejecuta el `main` de un script con la consola lista y sin tracebacks inútiles."""
    configurar()
    try:
        return main()
    except ErrorDeUso as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Cancelado.", file=sys.stderr)
        return 130
