"""Piezas que comparten los dos especialistas: leer su respuesta y lo que hicieron."""

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage


def extraer_texto(mensaje: BaseMessage) -> str:
    """Normaliza el content de un mensaje.

    Gemini, a través de langchain-google-genai, a veces devuelve el content como una lista de
    bloques (`[{'type': 'text', 'text': '...', 'extras': {...}}]`) en vez de un string. El
    notebook de la clase describe el problema en su última celda; esto lo resuelve. Viene de
    la pre-entrega 5, donde apareció por primera vez.
    """
    contenido = mensaje.content
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        return "".join(b.get("text", "") for b in contenido if isinstance(b, dict))
    return str(contenido)


def observaciones(mensajes: list[BaseMessage]) -> list[tuple[str, str]]:
    """Las herramientas que ejecutó un agente y lo que devolvió cada una, en orden.

    Es la materia prima de la validación determinística: no le preguntamos al especialista
    si buscó o calculó, miramos si hay un ToolMessage que lo pruebe.
    """
    return [(m.name or "", extraer_texto(m)) for m in mensajes if isinstance(m, ToolMessage)]


async def invocar_agente(agente, tarea: str) -> tuple[str, list[tuple[str, str]]]:
    """Corre un especialista sobre una tarea y devuelve (respuesta final, observaciones).

    El especialista recibe **solo** la tarea, en un historial nuevo: no ve la conversación del
    usuario, ni las decisiones del supervisor, ni lo que aportó el otro especialista salvo lo
    que el supervisor haya puesto adentro de la instrucción.
    """
    resultado = await agente.ainvoke({"messages": [HumanMessage(content=tarea)]})
    mensajes = resultado["messages"]
    return extraer_texto(mensajes[-1]), observaciones(mensajes)


def tarea_del_estado(state: dict) -> str:
    """La instrucción del supervisor; si no hay, la pregunta original del usuario."""
    return state.get("instruccion") or state["messages"][0].content
