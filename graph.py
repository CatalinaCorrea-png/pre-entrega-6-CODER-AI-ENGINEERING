"""El grafo del orquestador: topología jerárquica con un Supervisor en el centro.

    START -> supervisor -> investigador -> supervisor -> analista -> supervisor -> sintesis -> END
                  ^______________|    ^__________________|

Los especialistas nunca hablan entre sí: todo pasa por el supervisor, que es el único con
aristas condicionales. Esa es la topología en una línea.
"""

from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from agents.analyst_agent import nodo_analista
from agents.research_agent import nodo_investigador
from agents.supervisor import nodo_sintesis, nodo_supervisor
from state import AgentState

# Segunda red de seguridad, además de MAX_PASOS: si algo en el ruteo fallara y el
# supervisor dejara de contar pasos, LangGraph corta igual. 6 pasos de especialista + 7
# decisiones + la síntesis son 14 supersteps; 25 deja margen sin permitir un bucle caro.
RECURSION_LIMIT = 25


def enrutar(state: AgentState) -> Literal["investigador", "analista", "sintesis"]:
    """Traduce la decisión del supervisor al nombre del próximo nodo."""
    if state["next_agent"] == "FINISH":
        return "sintesis"
    return state["next_agent"]


grafo = StateGraph(AgentState)
grafo.add_node("supervisor", nodo_supervisor)
grafo.add_node("investigador", nodo_investigador)
grafo.add_node("analista", nodo_analista)
grafo.add_node("sintesis", nodo_sintesis)

grafo.add_edge(START, "supervisor")
grafo.add_conditional_edges("supervisor", enrutar, {
    "investigador": "investigador",
    "analista": "analista",
    "sintesis": "sintesis",
})
grafo.add_edge("investigador", "supervisor")
grafo.add_edge("analista", "supervisor")
grafo.add_edge("sintesis", END)

app = grafo.compile()

CONFIG = {"recursion_limit": RECURSION_LIMIT}


def estado_inicial(pregunta: str) -> dict:
    """El estado con el que arranca cada consulta."""
    return {
        "messages": [HumanMessage(content=pregunta)],
        "next_agent": None,
        "instruccion": None,
        "contribuciones": [],
        "decisiones": [],
        "pasos": 0,
        "task_completed": False,
    }
