"""El Supervisor (router inteligente y control de calidad) y la síntesis final.

El supervisor coordina, no ejecuta: nunca busca datos ni hace cuentas. En cada turno lee las
contribuciones —ya marcadas como válidas o inválidas por la validación determinística de cada
especialista— y decide quién interviene ahora, con qué instrucción, o si es momento de cerrar.
"""

from typing import Literal

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from agents.comun import extraer_texto
from agents.llm import obtener_llm
from state import AgentState

# Freno determinístico del "supervisor infinito": pasadas estas ejecuciones de
# especialistas, se cierra sin consultar al LLM, aunque insista en seguir delegando.
MAX_PASOS = 6


# --- El contrato de la decisión ----------------------------------------------

class DecisionSupervisor(BaseModel):
    next: Literal["investigador", "analista", "FINISH"] = Field(
        description="Próximo agente a invocar, o FINISH si la tarea ya está completa"
    )
    razon: str = Field(description="Breve justificación de la decisión")
    instruccion: str = Field(
        default="",
        description=(
            "Tarea concreta y autocontenida para el especialista elegido, con todos los datos "
            "que necesita. Vacía si next es FINISH."
        ),
    )


# --- El prompt: reglas de ruteo + rúbrica de validación ----------------------

SUPERVISOR_PROMPT = """Sos el Supervisor de un equipo con dos especialistas:
- investigador: busca datos concretos en las políticas internas de TechCorp.
- analista: hace cálculos matemáticos a partir de datos ya obtenidos.

Reglas de ruteo:
1. Si falta un dato de la empresa, mandá a 'investigador'.
2. Si ya se buscó el dato pero falta calcular, mandá a 'analista'.
3. Si ya se buscó Y se calculó lo necesario, respondé 'FINISH'.
4. Nunca repitas el mismo agente dos veces seguidas si ya cumplió su parte.
5. Si la pregunta no necesita ningún cálculo, no mandes al 'analista'.

Rúbrica de validación (aplicala antes de decidir):
- Cada contribución viene marcada por un validador automático como [VÁLIDA] o
  [INVÁLIDA: motivo]. Una contribución INVÁLIDA no cuenta como hecha.
- Si el último aporte de un especialista es INVÁLIDO, volvé a mandar a ESE especialista con
  una instrucción que corrija el problema puntual que indica el motivo. No repitas la
  instrucción anterior tal cual.
- Respondé 'FINISH' solo cuando los aportes VÁLIDOS alcanzan para responder la pregunta
  original completa.
- Nunca hagas vos el trabajo de los especialistas: no busques datos ni hagas cuentas.

Cómo escribir la instrucción:
- Tiene que ser autocontenida: el especialista NO ve la pregunta original ni los aportes de
  los demás. Incluí todo lo que necesita.
- Al 'analista' pasale los números concretos que trajo el investigador y qué hay que calcular.
- Si respondés 'FINISH', dejá la instrucción vacía.

Contribuciones hasta ahora:
{contribuciones}
"""

# Cuánto de cada aporte ve el supervisor. El notebook cortaba en 200 caracteres, pero el
# supervisor tiene que poder copiar los números del investigador en la instrucción del
# analista, y en 200 caracteres a veces no llegaban.
LARGO_APORTE = 600

llm_supervisor = obtener_llm().with_structured_output(DecisionSupervisor)


def formatear_contribuciones(contribuciones: list[dict]) -> str:
    """Las contribuciones con su marca de validez y la tarea que respondía cada una.

    La tarea va al lado del aporte porque el especialista contesta sin repetir la pregunta:
    "Corresponde 1 día por año calendario" no dice que se trata de la licencia por mudanza, y
    sin eso el supervisor no puede saber que el dato ya está.
    """
    lineas = []
    for c in contribuciones:
        marca = "VÁLIDA" if c["valido"] else f"INVÁLIDA: {c['motivo']}"
        lineas.append(
            f"- {c['agente']} [{marca}]\n"
            f"    tarea: {c['tarea']}\n"
            f"    aporte: {c['aporte'][:LARGO_APORTE]}"
        )
    return "\n".join(lineas) or "(ninguna todavía)"


def _normalizar(texto: str) -> str:
    return " ".join(texto.lower().split()).rstrip(".?! ")


def tarea_ya_resuelta(contribuciones: list[dict], agente: str, instruccion: str) -> dict | None:
    """La contribución válida en la que `agente` ya respondió exactamente esta instrucción.

    Mandar la misma tarea al mismo especialista después de que la resolvió no puede traer
    información nueva: es el síntoma de un supervisor en bucle. Compara el texto normalizado
    (mayúsculas, espacios, puntuación final), así que solo detecta repeticiones literales; una
    instrucción reformulada pasa, y para eso queda MAX_PASOS.
    """
    buscada = _normalizar(instruccion)
    return next(
        (
            c for c in contribuciones
            if c["agente"] == agente and c["valido"] and _normalizar(c["tarea"]) == buscada
        ),
        None,
    )


def _decision(state: AgentState, siguiente: str, razon: str, instruccion: str = "") -> dict:
    """Arma la actualización del estado: el supervisor escribe solo en su parcela."""
    return {
        "next_agent": siguiente,
        "instruccion": instruccion or None,
        "task_completed": siguiente == "FINISH",
        "decisiones": [
            {
                "turno": len(state.get("decisiones", [])) + 1,
                "next": siguiente,
                "razon": razon,
                "instruccion": instruccion,
            }
        ],
    }


# --- El nodo del supervisor --------------------------------------------------

async def nodo_supervisor(state: AgentState) -> dict:
    if state.get("pasos", 0) >= MAX_PASOS:
        return _decision(
            state, "FINISH", f"tope de {MAX_PASOS} pasos alcanzado: se cierra con lo que haya"
        )

    prompt_sistema = SUPERVISOR_PROMPT.format(
        contribuciones=formatear_contribuciones(state.get("contribuciones", []))
    )
    pregunta_original = state["messages"][0].content

    try:
        decision = await llm_supervisor.ainvoke([
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": f"Tarea original: {pregunta_original}"},
        ])
        if decision is None:  # Gemini a veces no devuelve la llamada estructurada
            raise ValueError("el modelo no devolvió una decisión estructurada")
    except Exception as error:
        # Un supervisor que falla no puede dejar el grafo colgado: se cierra y la síntesis
        # trabaja con lo que haya, diciendo qué no se pudo verificar.
        return _decision(
            state,
            "FINISH",
            f"el supervisor no pudo decidir ({type(error).__name__}: {error}); "
            "se cierra con lo que haya",
        )

    # Freno determinístico contra el bucle: la decisión del LLM se revisa con código antes
    # de ejecutarla, igual que los aportes de los especialistas.
    if decision.next != "FINISH" and tarea_ya_resuelta(
        state.get("contribuciones", []), decision.next, decision.instruccion
    ):
        return _decision(
            state,
            "FINISH",
            f"se pidió repetirle a {decision.next} una tarea que ya resolvió con un aporte "
            "válido; repetirla no trae información nueva, así que se cierra con lo que hay "
            f"(el supervisor proponía: {decision.razon})",
        )

    return _decision(state, decision.next, decision.razon, decision.instruccion)


# --- Síntesis: la respuesta final al usuario ---------------------------------

llm_sintesis = obtener_llm()


def _listar(contribuciones: list[dict]) -> str:
    return "\n".join(
        f"- [{c['agente']}] (tarea: {c['tarea']}) {c['aporte']}" for c in contribuciones
    )


async def nodo_sintesis(state: AgentState) -> dict:
    """Redacta la respuesta final usando solo los aportes que pasaron la validación."""
    contribuciones = state.get("contribuciones", [])
    validas = [c for c in contribuciones if c["valido"]]
    invalidas = [c for c in contribuciones if not c["valido"]]
    pregunta_original = state["messages"][0].content

    prompt = (
        f"Pregunta original: {pregunta_original}\n\n"
        f"Aportes verificados del equipo:\n{_listar(validas) or '(ninguno)'}\n\n"
    )
    if invalidas:
        prompt += (
            "Aportes que NO pasaron la validación (no los uses como dato):\n"
            f"{_listar(invalidas)}\n\n"
        )
    prompt += (
        "Redactá una respuesta final clara y breve combinando los aportes verificados, en "
        "texto plano (sin LaTeX). Si con ellos no alcanza para responder, decilo "
        "explícitamente en vez de completar con suposiciones."
    )

    try:
        respuesta = extraer_texto(await llm_sintesis.ainvoke(prompt))
    except Exception as error:
        respuesta = (
            f"No se pudo redactar la síntesis ({type(error).__name__}). "
            f"Aportes verificados:\n{_listar(validas) or '(ninguno)'}"
        )

    return {"messages": [AIMessage(content=respuesta, name="supervisor")]}
