"""Agente especialista de investigación: busca datos en las políticas internas de TechCorp.

Su única herramienta consulta el índice híbrido (BM25 + Pinecone) de la pre-entrega 4,
cargado con las políticas de la pre-entrega 3. No calcula: si la pregunta necesita una
cuenta, ese es trabajo del analista.
"""

from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agents.comun import invocar_agente, tarea_del_estado
from agents.llm import obtener_llm
from rag.recuperacion import RAGSystem
from state import AgentState, Contribucion

NOMBRE = "investigador"
HERRAMIENTA = "buscar_en_politicas_internas"


# --- La herramienta ----------------------------------------------------------

@lru_cache(maxsize=1)
def obtener_rag() -> RAGSystem:
    """El sistema de recuperación, construido recién en la primera búsqueda.

    Perezoso a propósito: el corpus de BM25 se reconstruye bajando el namespace de Pinecone,
    y importar este módulo —en los tests, por ejemplo— no tiene por qué tocar la red.
    """
    return RAGSystem(k=5, modo="hibrido")


@tool
def buscar_en_politicas_internas(pregunta: str) -> str:
    """Busca información en las políticas internas de TechCorp (vacaciones y licencias,
    teletrabajo, seguridad informática, onboarding). Usar para cualquier pregunta sobre reglas
    o números concretos de la empresa. Devuelve los fragmentos encontrados con su fuente, o un
    mensaje que empieza con ERROR si la búsqueda falla."""
    try:
        resultados = obtener_rag().obtener_top_k(pregunta)
    except Exception as error:  # red, cuota, índice vacío: el agente tiene que enterarse
        return f"ERROR: la búsqueda en las políticas falló ({type(error).__name__}: {error})"

    if not resultados:
        return "No se encontró información relevante en las políticas."
    return "\n\n".join(f"[Fuente: {r['fuente']}] {r['contenido']}" for r in resultados)


# --- El agente ---------------------------------------------------------------

SYSTEM_PROMPT = (
    "Sos un agente de investigación. Tu única función es buscar datos concretos en "
    "las políticas internas de TechCorp usando tu herramienta. NO hagas cálculos. "
    "Respondé de forma breve con los datos encontrados y su fuente."
)

agente_investigador = create_agent(
    model=obtener_llm(),
    tools=[buscar_en_politicas_internas],
    system_prompt=SYSTEM_PROMPT,
)


# --- Validación determinística -----------------------------------------------

SIN_RESULTADOS = "No se encontró información relevante"


def validar_investigacion(tarea: str, respuesta: str, observaciones: list[tuple[str, str]]) -> Contribucion:
    """Decide con código, no con el juicio de un LLM, si la investigación es utilizable.

    Un texto que "suena" a dato puede ser inventado: lo único que prueba que el dato salió de
    las políticas es que la herramienta se haya ejecutado y haya devuelto fragmentos.
    """
    usadas = [nombre for nombre, _ in observaciones]
    salidas = [salida for nombre, salida in observaciones if nombre == HERRAMIENTA]

    motivo = None
    if not respuesta.strip():
        motivo = "el investigador devolvió una respuesta vacía"
    elif not salidas:
        motivo = "respondió sin consultar las políticas internas: el dato no está respaldado"
    elif all(s.startswith("ERROR") for s in salidas):
        motivo = f"la búsqueda falló: {salidas[-1]}"
    elif all(s.startswith(SIN_RESULTADOS) for s in salidas):
        motivo = "la búsqueda no encontró información relevante: hay que reformular la consulta"

    return Contribucion(
        agente=NOMBRE,
        tarea=tarea,
        aporte=respuesta.strip() or "(respuesta vacía)",
        valido=motivo is None,
        motivo=motivo,
        herramientas_usadas=usadas,
    )


# --- El nodo del grafo -------------------------------------------------------

async def nodo_investigador(state: AgentState) -> dict:
    """Envuelve al agente: le pasa solo su instrucción y valida lo que devuelve.

    Si el agente explota (API caída, cuota), el error no rompe el grafo: vuelve al
    supervisor como una contribución inválida, que decide si reintentar o cerrar.
    """
    tarea = tarea_del_estado(state)
    try:
        respuesta, observaciones = await invocar_agente(agente_investigador, tarea)
        contribucion = validar_investigacion(tarea, respuesta, observaciones)
    except Exception as error:
        contribucion = Contribucion(
            agente=NOMBRE,
            tarea=tarea,
            aporte=f"(el investigador falló: {type(error).__name__})",
            valido=False,
            motivo=f"el agente lanzó una excepción: {type(error).__name__}: {error}",
        )

    return {
        "messages": [AIMessage(content=contribucion.aporte, name=NOMBRE)],
        "contribuciones": [contribucion.model_dump()],
        "pasos": state.get("pasos", 0) + 1,
    }
