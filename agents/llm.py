"""El modelo de lenguaje que comparten los cuatro roles del orquestador.

Una sola instancia a propósito: el `rate_limiter` vive adentro del cliente, así que para que
limite de verdad las llamadas por minuto tiene que ser el mismo objeto para el supervisor,
los dos especialistas y la síntesis.
"""

import logging
import os
from functools import lru_cache

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.config import variable_obligatoria

# flash-lite porque la capa gratuita le da 15 pedidos por minuto (contra 5 de
# gemini-flash-latest), y una consulta completa hace ~8 llamadas al LLM.
MODELO = os.getenv("MODELO_LLM", "gemini-flash-lite-latest")

# 0.2 pedidos por segundo = 12 por minuto, debajo del tope de 15 de la capa gratuita. En la
# pre-entrega 5 la corrida se cortó con un 429 a mitad de camino: acá el limitador espera
# antes de pedir, en vez de fallar después. Hace la corrida más lenta, no más frágil.
PEDIDOS_POR_SEGUNDO = 0.2

# google-genai avisa en cada llamada con herramientas que prefiere su propia API de chat
# ("Direct use of automatic function calling..."). Es un aviso sobre el SDK subyacente, que
# LangChain envuelve: no hay nada que corregir de este lado y ensucia la traza de delegación.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)


@lru_cache(maxsize=1)
def obtener_llm() -> ChatGoogleGenerativeAI:
    """El cliente de Gemini, uno por proceso."""
    return ChatGoogleGenerativeAI(
        model=MODELO,
        temperature=0,
        google_api_key=variable_obligatoria("GOOGLE_API_KEY"),
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=PEDIDOS_POR_SEGUNDO,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        ),
    )
