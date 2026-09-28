"""Agente especialista de análisis: hace las cuentas con los datos que le pasan.

No busca información nueva: trabaja sobre los números que el supervisor le incluye en la
instrucción, que a su vez salieron del investigador.
"""

import ast
import operator as op

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agents.comun import invocar_agente, tarea_del_estado
from agents.llm import obtener_llm
from state import AgentState, Contribucion

NOMBRE = "analista"
HERRAMIENTA = "calculadora"


# --- La herramienta: calculadora segura --------------------------------------

# `eval()` a secas sobre texto generado por un LLM es una puerta abierta a ejecución de
# código arbitrario. Acá se parsea la expresión con `ast` y solo se permiten operaciones
# aritméticas: nada de imports, llamadas a funciones ni acceso a builtins.
_OPERADORES = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg,
}

# Dos agregados al evaluador del notebook, los dos por la misma razón: que "segura" también
# signifique que no puede colgar el proceso.
#   - Solo constantes numéricas. El original aceptaba cualquier constante, así que
#     "'a' * 10**9" armaba un string de 1 GB.
#   - Tope al exponente. "9**9**9" es aritmética válida y tarda horas en calcularse.
EXPONENTE_MAXIMO = 100


def _evaluar_nodo(nodo):
    if isinstance(nodo, ast.Constant):
        if isinstance(nodo.value, bool) or not isinstance(nodo.value, (int, float)):
            raise ValueError(f"solo se admiten números, no {nodo.value!r}")
        return nodo.value
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _OPERADORES:
        izquierda, derecha = _evaluar_nodo(nodo.left), _evaluar_nodo(nodo.right)
        if isinstance(nodo.op, ast.Pow) and abs(derecha) > EXPONENTE_MAXIMO:
            raise ValueError(f"exponente demasiado grande (máximo {EXPONENTE_MAXIMO})")
        return _OPERADORES[type(nodo.op)](izquierda, derecha)
    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _OPERADORES:
        return _OPERADORES[type(nodo.op)](_evaluar_nodo(nodo.operand))
    raise ValueError("Expresión no permitida")


@tool
def calculadora(expresion: str) -> str:
    """Evalúa una expresión matemática (+, -, *, /, **, paréntesis). Ejemplo: '5*14 + 5*21 + 2*28'.
    Devuelve 'Resultado: <número>', o un mensaje que empieza con 'Error' si la expresión no
    es válida."""
    try:
        resultado = _evaluar_nodo(ast.parse(expresion, mode="eval").body)
        return f"Resultado: {resultado}"
    except Exception as e:
        return f"Error al calcular '{expresion}': {e}"


# --- El agente ---------------------------------------------------------------

SYSTEM_PROMPT = (
    "Sos un agente de análisis matemático. Tu única función es calcular usando tu "
    "herramienta a partir de los números que te den. NO busques información nueva."
)

agente_analista = create_agent(
    model=obtener_llm(),
    tools=[calculadora],
    system_prompt=SYSTEM_PROMPT,
)


# --- Validación determinística -----------------------------------------------

def validar_analisis(tarea: str, respuesta: str, observaciones: list[tuple[str, str]]) -> Contribucion:
    """Un número es válido si lo produjo la calculadora, no si el LLM "hizo la cuenta".

    Los LLMs se equivocan en aritmética con total seguridad; lo que se verifica es que el
    resultado haya pasado por la herramienta y que la herramienta no haya devuelto un error.
    """
    usadas = [nombre for nombre, _ in observaciones]
    salidas = [salida for nombre, salida in observaciones if nombre == HERRAMIENTA]

    motivo = None
    if not respuesta.strip():
        motivo = "el analista devolvió una respuesta vacía"
    elif not salidas:
        motivo = "no usó la calculadora: el resultado no está verificado"
    elif not any(s.startswith("Resultado:") for s in salidas):
        motivo = f"la calculadora devolvió un error: {salidas[-1]}"

    return Contribucion(
        agente=NOMBRE,
        tarea=tarea,
        aporte=respuesta.strip() or "(respuesta vacía)",
        valido=motivo is None,
        motivo=motivo,
        herramientas_usadas=usadas,
    )


# --- El nodo del grafo -------------------------------------------------------

async def nodo_analista(state: AgentState) -> dict:
    """Envuelve al agente: le pasa solo su instrucción y valida lo que devuelve."""
    tarea = tarea_del_estado(state)
    try:
        respuesta, observaciones = await invocar_agente(agente_analista, tarea)
        contribucion = validar_analisis(tarea, respuesta, observaciones)
    except Exception as error:
        contribucion = Contribucion(
            agente=NOMBRE,
            tarea=tarea,
            aporte=f"(el analista falló: {type(error).__name__})",
            valido=False,
            motivo=f"el agente lanzó una excepción: {type(error).__name__}: {error}",
        )

    return {
        "messages": [AIMessage(content=contribucion.aporte, name=NOMBRE)],
        "contribuciones": [contribucion.model_dump()],
        "pasos": state.get("pasos", 0) + 1,
    }
