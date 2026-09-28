"""El estado compartido del orquestador y el contrato de lo que escribe cada especialista.

Dos piezas:

- `Contribucion` es el contrato de interfaz entre un especialista y el supervisor. Se valida
  con Pydantic **antes** de entrar al estado: si un nodo intentara inyectar un aporte vacío o
  firmado por un agente que no existe, el error salta ahí mismo y no tres pasos después,
  cuando el supervisor lo lea.

- `AgentState` es la memoria de trabajo del grafo. Hereda `messages` de `MessagesState` (con
  su reducer `add_messages`) y suma los campos de coordinación.

Aislamiento de escritura: cada nodo es dueño de una parcela del estado y solo escribe ahí.

    | Campo            | Lo escribe             | Reducer                    |
    |------------------|------------------------|----------------------------|
    | messages         | especialistas, síntesis| add_messages (concatena)   |
    | contribuciones   | especialistas          | operator.add (concatena)   |
    | pasos            | especialistas          | reemplaza                  |
    | next_agent       | supervisor             | reemplaza                  |
    | instruccion      | supervisor             | reemplaza                  |
    | decisiones       | supervisor             | operator.add (concatena)   |
    | task_completed   | supervisor             | reemplaza                  |

Así ningún agente puede pisar lo que otro acaba de calcular: el investigador no tiene cómo
escribir en `next_agent`, ni el supervisor en `contribuciones`.
"""

import operator
from typing import Annotated, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

Especialista = Literal["investigador", "analista"]


class Contribucion(BaseModel):
    """Lo que un especialista aporta al estado compartido, ya validado.

    `valido` y `motivo` los completa una validación **determinística** (código, no el juicio
    de un LLM): ¿usó su herramienta?, ¿la herramienta devolvió un error? El supervisor lee
    `motivo` para mandar a refinar con un error concreto, en vez de un "intentá de nuevo"
    que probablemente repita el mismo sesgo.
    """

    agente: Especialista
    # La tarea que respondía el aporte. Sin ella, el aislamiento de contexto corta en las dos
    # direcciones: el especialista contesta corto porque en su conversación la pregunta es
    # obvia ("Corresponde 1 día por año calendario"), y el supervisor recibe esa respuesta sin
    # saber a qué pregunta corresponde. En una corrida real eso lo llevó a pedir cinco veces el
    # mismo dato que ya tenía.
    tarea: str = Field(description="La instrucción que recibió el especialista")
    aporte: str = Field(min_length=1, description="Respuesta final del especialista")
    valido: bool = Field(description="Si pasó la validación determinística de su rol")
    motivo: str | None = Field(
        default=None, description="Por qué no pasó la validación; None si es válida"
    )
    herramientas_usadas: list[str] = Field(
        default_factory=list, description="Nombres de las herramientas que ejecutó, en orden"
    )


class AgentState(MessagesState):
    """Hereda `messages` (con su propio reducer add_messages) y suma campos propios."""

    next_agent: str | None
    # Lo que el supervisor le encarga al próximo especialista: la tarea puntual con los datos
    # que necesita, no toda la conversación (evita la "contaminación de contexto").
    instruccion: str | None
    # `operator.add` concatena en vez de pisar: queda el registro completo de quién aportó
    # qué. Se guardan como dicts (`Contribucion.model_dump()`) para que el estado siga siendo
    # serializable a JSON en la traza.
    contribuciones: Annotated[list[dict], operator.add]
    # El log de ruteo: cada decisión del supervisor con su razón. Es lo que permite ver, en la
    # traza, exactamente quién decidió qué y por qué.
    decisiones: Annotated[list[dict], operator.add]
    pasos: int
    task_completed: bool
