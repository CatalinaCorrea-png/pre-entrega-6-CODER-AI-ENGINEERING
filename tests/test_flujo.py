"""El flujo de delegación completo, sobre el grafo real con los cuatro roles guionados."""

import pytest
from langchain_core.messages import HumanMessage

from agents.analyst_agent import nodo_analista
from agents.research_agent import nodo_investigador
from agents.supervisor import MAX_PASOS, nodo_sintesis, nodo_supervisor
from main import correr_consulta

from tests.conftest import ANALISIS_OK, INVESTIGACION_OK, decidir, sin_herramienta

PREGUNTA = "3 personas con 2 años, 2 con 7 y 1 con 12: ¿cuántos días de vacaciones suman?"


@pytest.mark.asyncio
async def test_delega_investigacion_despues_calculo_y_cierra(supervisor, investigador, analista, sintesis):
    """El recorrido que pide la consigna: supervisor -> investigador -> supervisor ->
    analista -> supervisor -> síntesis."""
    supervisor(
        decidir("investigador", "Buscá los días de vacaciones por antigüedad"),
        decidir("analista", "Calculá 3*14 + 2*21 + 1*28"),
        decidir("FINISH"),
    )
    investigador(INVESTIGACION_OK)
    analista(ANALISIS_OK)
    sintesis("En total suman 112 días.")

    traza = await correr_consulta(PREGUNTA, mostrar=False)

    assert traza["flujo"] == [
        "supervisor", "investigador", "supervisor", "analista", "supervisor", "sintesis",
    ]
    assert traza["respuesta_final"] == "En total suman 112 días."
    assert [c["agente"] for c in traza["contribuciones"]] == ["investigador", "analista"]
    assert all(c["valido"] for c in traza["contribuciones"])
    assert [d["next"] for d in traza["decisiones"]] == ["investigador", "analista", "FINISH"]


@pytest.mark.asyncio
async def test_cada_especialista_recibe_solo_su_instruccion(supervisor, investigador, analista, sintesis):
    """Contra la contaminación de contexto: el especialista no ve la pregunta del usuario ni
    los aportes de los otros, solo lo que el supervisor le encargó. El doble además falla si
    recibe más de un mensaje."""
    supervisor(
        decidir("investigador", "Buscá los días por antigüedad"),
        decidir("analista", "Calculá 3*14 + 2*21 + 1*28"),
        decidir("FINISH"),
    )
    inv, ana = investigador(INVESTIGACION_OK), analista(ANALISIS_OK)
    sintesis()

    await correr_consulta(PREGUNTA, mostrar=False)

    assert inv.tareas == ["Buscá los días por antigüedad"]
    assert ana.tareas == ["Calculá 3*14 + 2*21 + 1*28"]


@pytest.mark.asyncio
async def test_una_pregunta_de_un_solo_dominio_no_pasa_por_el_analista(supervisor, investigador, sintesis):
    """Delegación dinámica, no un pipeline fijo: si no hay nada que calcular, el analista ni
    se entera. Acá su doble es la bomba: si lo llamaran, el test fallaría."""
    supervisor(decidir("investigador", "Buscá los días presenciales mínimos"), decidir("FINISH"))
    investigador(INVESTIGACION_OK)
    sintesis("Mínimo 2 días presenciales.")

    traza = await correr_consulta("¿Cuántos días presenciales exige el esquema?", mostrar=False)
    assert "analista" not in traza["flujo"]


@pytest.mark.asyncio
async def test_un_aporte_invalido_se_manda_a_refinar(supervisor, investigador, analista, sintesis):
    """El bucle de reparación: el investigador responde de memoria, la validación lo marca
    inválido, y el supervisor lo ve —con el motivo— antes de volver a delegar."""
    sup = supervisor(
        decidir("investigador", "Buscá los días por antigüedad"),
        decidir("investigador", "Usá la herramienta de búsqueda: no respondas de memoria"),
        decidir("analista", "Calculá 3*14 + 2*21 + 1*28"),
        decidir("FINISH"),
    )
    investigador(sin_herramienta("Creo que son 15 días."), INVESTIGACION_OK)
    analista(ANALISIS_OK)
    sintesis()

    traza = await correr_consulta(PREGUNTA, mostrar=False)

    assert traza["flujo"].count("investigador") == 2
    primera, segunda = traza["contribuciones"][:2]
    assert not primera["valido"] and segunda["valido"]
    # el segundo turno del supervisor ya vio la contribución inválida con su motivo
    assert "INVÁLIDA: respondió sin consultar" in sup.prompts[1]


@pytest.mark.asyncio
async def test_un_especialista_que_explota_no_rompe_el_grafo(supervisor, investigador, sintesis):
    """El camino de fallo de la clase (slide 36): el especialista lanza una excepción, llega
    al supervisor como contribución inválida, y el grafo igual termina en la síntesis."""
    supervisor(decidir("investigador", "Buscá"), decidir("FINISH", razon="no hay cómo seguir"))
    investigador(ConnectionError("Gemini no responde"))
    sintesis("No pude verificar el dato.")

    traza = await correr_consulta(PREGUNTA, mostrar=False)

    contribucion = traza["contribuciones"][0]
    assert not contribucion["valido"]
    assert "ConnectionError" in contribucion["motivo"]
    assert traza["flujo"][-1] == "sintesis"


@pytest.mark.asyncio
async def test_un_supervisor_que_nunca_cierra_se_frena_en_MAX_PASOS(supervisor, investigador, sintesis):
    """El "supervisor infinito": el guion delega para siempre. El tope de pasos corta sin
    consultar al LLM y el grafo cierra por la síntesis, sin llegar al recursion_limit."""
    supervisor(*[decidir("investigador", "Buscá otra vez")] * MAX_PASOS)
    investigador(*[sin_herramienta("no sé")] * MAX_PASOS)
    sintesis()

    traza = await correr_consulta(PREGUNTA, mostrar=False)

    assert traza["flujo"].count("investigador") == MAX_PASOS
    assert traza["decisiones"][-1]["next"] == "FINISH"
    assert "tope" in traza["decisiones"][-1]["razon"]
    assert traza["flujo"][-1] == "sintesis"


# --- Aislamiento de escritura ------------------------------------------------

def _estado() -> dict:
    return {"messages": [HumanMessage(content=PREGUNTA)], "instruccion": "Buscá",
            "contribuciones": [], "decisiones": [], "pasos": 0}


@pytest.mark.asyncio
async def test_cada_nodo_escribe_solo_en_su_parcela_del_estado(supervisor, investigador, analista, sintesis):
    """Si dos nodos pudieran escribir el mismo campo, uno podría pisar lo que el otro acaba de
    calcular. Cada uno devuelve solo sus claves."""
    supervisor(decidir("investigador", "Buscá"))
    investigador(INVESTIGACION_OK)
    analista(ANALISIS_OK)
    sintesis()

    assert set(await nodo_supervisor(_estado())) == {"next_agent", "instruccion", "task_completed", "decisiones"}
    assert set(await nodo_investigador(_estado())) == {"messages", "contribuciones", "pasos"}
    assert set(await nodo_analista(_estado())) == {"messages", "contribuciones", "pasos"}
    assert set(await nodo_sintesis(_estado())) == {"messages"}


@pytest.mark.asyncio
async def test_un_supervisor_que_repite_la_misma_tarea_se_corta_al_primer_intento(supervisor, investigador, sintesis):
    """Regresión de la corrida real de "¿Cuántos días de licencia por mudanza da TechCorp?":
    el supervisor pidió cinco veces la misma instrucción. Con el guion repitiéndola, el
    investigador corre una sola vez y la segunda pedida ya cierra por la síntesis."""
    tarea = "Buscá cuántos días de licencia por mudanza corresponden"
    supervisor(*[decidir("investigador", tarea, "falta el dato")] * 5)
    inv = investigador(*[INVESTIGACION_OK] * 5)
    sintesis("1 día por año calendario.")

    traza = await correr_consulta("¿Cuántos días de licencia por mudanza da TechCorp?", mostrar=False)

    assert traza["flujo"] == ["supervisor", "investigador", "supervisor", "sintesis"]
    assert inv.tareas == [tarea]
    assert "ya resolvió" in traza["decisiones"][-1]["razon"]
