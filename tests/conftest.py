"""Configuración común de la suite.

Ningún test llama a una API: ni a Gemini ni a Pinecone. Los cuatro roles del orquestador y el
sistema de recuperación se reemplazan por dobles guionados, así que la suite corre gratis,
sin red y con resultados deterministas.
"""

import os

# Antes de importar cualquier módulo del proyecto: `agents/*` instancian el cliente de Gemini
# al importarse, y eso exige una GOOGLE_API_KEY. Una de mentira alcanza —construir el cliente
# no hace ninguna llamada— y `setdefault` gana sobre el `.env` real, porque `load_dotenv` no
# pisa variables que ya existen. Así es imposible gastar cuota por accidente.
os.environ.setdefault("GOOGLE_API_KEY", "clave-de-prueba-sin-uso")
os.environ.setdefault("PINECONE_API_KEY", "clave-de-prueba-sin-uso")

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

import agents.analyst_agent
import agents.research_agent
import agents.supervisor
from agents.supervisor import DecisionSupervisor


# --- Red de seguridad: nada real se puede llamar -----------------------------

class LlamadaProhibida(BaseException):
    """Un test llegó a un cliente real.

    Hereda de BaseException y no de Exception a propósito. Los nodos del orquestador capturan
    `Exception` para que un fallo de un agente no rompa el grafo, y esa resiliencia se tragaba
    una AssertionError común: el supervisor la registraba como "no pudo decidir", el grafo
    cerraba prolijo y el test pasaba en silencio. Una BaseException atraviesa esos `except`
    sin que haya que debilitarlos, y el test falla donde tiene que fallar.
    """


class _Prohibido:
    """Ocupa el lugar de un cliente real y falla si alguien lo usa."""

    def __init__(self, que: str):
        self.que = que

    def _explotar(self, *args, **kwargs):
        raise LlamadaProhibida(
            f"Un test intentó usar {self.que} real. Instalá el doble correspondiente con su "
            "fixture antes de ejecutar el grafo."
        )

    async def ainvoke(self, *args, **kwargs):
        self._explotar()

    invoke = __call__ = obtener_top_k = _explotar


@pytest.fixture(autouse=True)
def sin_apis_reales(monkeypatch):
    """Se aplica a TODOS los tests. Los que necesitan un rol lo reemplazan por su guion
    (esta fixture corre antes por ser autouse); cualquier otro que llegue a un cliente real
    falla con un mensaje que dice qué le falta, en vez de gastar tokens en silencio.

    Funciona porque cada nodo resuelve su agente o su LLM como global del módulo en cada
    llamada: pisar el global desvía el grafo real, sin reconstruirlo en el test.
    """
    monkeypatch.setattr(agents.research_agent, "agente_investigador", _Prohibido("el investigador"))
    monkeypatch.setattr(agents.analyst_agent, "agente_analista", _Prohibido("el analista"))
    monkeypatch.setattr(agents.supervisor, "llm_supervisor", _Prohibido("el LLM del supervisor"))
    monkeypatch.setattr(agents.supervisor, "llm_sintesis", _Prohibido("el LLM de la síntesis"))
    monkeypatch.setattr(agents.research_agent, "obtener_rag", _Prohibido("Pinecone"))


# --- Guiones de respuesta de un especialista ---------------------------------

def con_herramienta(herramienta: str, salida: str, final: str) -> list[BaseMessage]:
    """Lo que devuelve un agente que ejecutó una herramienta y después respondió."""
    return [
        AIMessage(content="", tool_calls=[{"name": herramienta, "args": {}, "id": "c1"}]),
        ToolMessage(content=salida, name=herramienta, tool_call_id="c1"),
        AIMessage(content=final),
    ]


def sin_herramienta(final: str) -> list[BaseMessage]:
    """Lo que devuelve un agente que respondió de memoria, sin ejecutar nada."""
    return [AIMessage(content=final)]


INVESTIGACION_OK = con_herramienta(
    "buscar_en_politicas_internas",
    "[Fuente: politica_vacaciones.txt] menos de 5 años: 14 días; de 5 a 10: 21; más de 10: 28.",
    "2 años: 14 días, 7 años: 21 días, 12 años: 28 días. Fuente: politica_vacaciones.txt",
)
ANALISIS_OK = con_herramienta("calculadora", "Resultado: 112", "El total es 112 días.")


# --- Dobles ------------------------------------------------------------------

class AgenteGuionado:
    """Especialista falso: devuelve sus respuestas en orden y registra qué tarea recibió."""

    def __init__(self, *respuestas):
        self.respuestas = list(respuestas)
        self.tareas: list[str] = []

    async def ainvoke(self, entrada: dict) -> dict:
        mensajes = entrada["messages"]
        assert len(mensajes) == 1, "el especialista tiene que recibir solo su instrucción"
        self.tareas.append(mensajes[0].content)
        respuesta = self.respuestas.pop(0)
        if isinstance(respuesta, Exception):
            raise respuesta
        return {"messages": [HumanMessage(content=mensajes[0].content), *respuesta]}


class SupervisorGuionado:
    """Supervisor falso: devuelve sus decisiones en orden y registra los prompts que recibió."""

    def __init__(self, *decisiones):
        self.decisiones = list(decisiones)
        self.prompts: list[str] = []

    async def ainvoke(self, mensajes: list[dict]):
        self.prompts.append(mensajes[0]["content"])
        decision = self.decisiones.pop(0)
        if isinstance(decision, Exception):
            raise decision
        return decision


class LLMGuionado:
    """LLM de síntesis falso."""

    def __init__(self, respuesta: str | Exception = "Respuesta final."):
        self.respuesta = respuesta
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str):
        self.prompts.append(prompt)
        if isinstance(self.respuesta, Exception):
            raise self.respuesta
        return AIMessage(content=self.respuesta)


class RAGFalso:
    def __init__(self, resultados=None, error: Exception | None = None):
        self.resultados = resultados or []
        self.error = error

    def obtener_top_k(self, consulta: str) -> list[dict]:
        if self.error:
            raise self.error
        return self.resultados


def decidir(siguiente: str, instruccion: str = "", razon: str = "porque sí") -> DecisionSupervisor:
    return DecisionSupervisor(next=siguiente, razon=razon, instruccion=instruccion)


# --- Fixtures que instalan los dobles ----------------------------------------

@pytest.fixture
def supervisor(monkeypatch):
    def instalar(*decisiones) -> SupervisorGuionado:
        doble = SupervisorGuionado(*decisiones)
        monkeypatch.setattr(agents.supervisor, "llm_supervisor", doble)
        return doble
    return instalar


@pytest.fixture
def investigador(monkeypatch):
    def instalar(*respuestas) -> AgenteGuionado:
        doble = AgenteGuionado(*respuestas)
        monkeypatch.setattr(agents.research_agent, "agente_investigador", doble)
        return doble
    return instalar


@pytest.fixture
def analista(monkeypatch):
    def instalar(*respuestas) -> AgenteGuionado:
        doble = AgenteGuionado(*respuestas)
        monkeypatch.setattr(agents.analyst_agent, "agente_analista", doble)
        return doble
    return instalar


@pytest.fixture
def sintesis(monkeypatch):
    def instalar(respuesta: str | Exception = "Respuesta final.") -> LLMGuionado:
        doble = LLMGuionado(respuesta)
        monkeypatch.setattr(agents.supervisor, "llm_sintesis", doble)
        return doble
    return instalar


@pytest.fixture
def rag(monkeypatch):
    def instalar(resultados=None, error: Exception | None = None) -> RAGFalso:
        doble = RAGFalso(resultados, error)
        monkeypatch.setattr(agents.research_agent, "obtener_rag", lambda: doble)
        return doble
    return instalar
