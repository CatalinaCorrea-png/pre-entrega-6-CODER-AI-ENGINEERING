"""El nodo Supervisor y la síntesis, probados de forma aislada."""

import pytest
from langchain_core.messages import HumanMessage

from agents.supervisor import MAX_PASOS, formatear_contribuciones, nodo_sintesis, nodo_supervisor

from tests.conftest import decidir

VALIDA = {"agente": "investigador", "tarea": "Buscá los días por antigüedad", "aporte": "14 días", "valido": True, "motivo": None,
          "herramientas_usadas": ["buscar_en_politicas_internas"]}
INVALIDA = {"agente": "analista", "tarea": "Calculá el total", "aporte": "112", "valido": False,
            "motivo": "no usó la calculadora: el resultado no está verificado",
            "herramientas_usadas": []}


def estado(**campos) -> dict:
    return {"messages": [HumanMessage(content="¿Cuántos días?")], "contribuciones": [],
            "decisiones": [], "pasos": 0, **campos}


# --- Decisiones --------------------------------------------------------------

@pytest.mark.asyncio
async def test_la_decision_se_traduce_al_estado(supervisor):
    supervisor(decidir("investigador", "Buscá los días por antigüedad", "falta el dato"))
    actualizacion = await nodo_supervisor(estado())

    assert actualizacion["next_agent"] == "investigador"
    assert actualizacion["instruccion"] == "Buscá los días por antigüedad"
    assert actualizacion["task_completed"] is False


@pytest.mark.asyncio
async def test_cada_decision_queda_registrada_con_su_razon(supervisor):
    """Es lo que permite ver en la traza quién decidió qué y por qué."""
    supervisor(decidir("analista", "3*14", "ya está el dato, falta la cuenta"))
    actualizacion = await nodo_supervisor(estado(decisiones=[{"turno": 1}]))

    assert actualizacion["decisiones"] == [{
        "turno": 2, "next": "analista", "razon": "ya está el dato, falta la cuenta",
        "instruccion": "3*14",
    }]


@pytest.mark.asyncio
async def test_FINISH_marca_la_tarea_completa_y_no_deja_instruccion(supervisor):
    supervisor(decidir("FINISH", razon="ya está todo"))
    actualizacion = await nodo_supervisor(estado())
    assert actualizacion["task_completed"] is True
    assert actualizacion["instruccion"] is None


# --- Frenos y fallos ---------------------------------------------------------

@pytest.mark.asyncio
async def test_el_tope_de_pasos_cierra_sin_consultar_al_LLM():
    """Acá no hay supervisor guionado: el LLM real está reemplazado por una bomba. Que el test
    pase prueba que, con el tope alcanzado, el LLM ni se consulta."""
    actualizacion = await nodo_supervisor(estado(pasos=MAX_PASOS))
    assert actualizacion["next_agent"] == "FINISH"
    assert "tope" in actualizacion["decisiones"][0]["razon"]


@pytest.mark.asyncio
async def test_si_el_LLM_no_devuelve_una_decision_se_cierra_en_vez_de_romper(supervisor):
    """Gemini a veces no emite la llamada estructurada y `with_structured_output` devuelve
    None. Sin el control, el nodo reventaría con AttributeError."""
    supervisor(None)
    actualizacion = await nodo_supervisor(estado())
    assert actualizacion["next_agent"] == "FINISH"
    assert "no pudo decidir" in actualizacion["decisiones"][0]["razon"]


@pytest.mark.asyncio
async def test_si_el_LLM_lanza_una_excepcion_se_cierra_en_vez_de_romper(supervisor):
    supervisor(TimeoutError("sin respuesta"))
    actualizacion = await nodo_supervisor(estado())
    assert actualizacion["next_agent"] == "FINISH"
    assert "TimeoutError" in actualizacion["decisiones"][0]["razon"]


# --- La rúbrica que lee el supervisor ----------------------------------------

def test_las_contribuciones_llegan_marcadas_con_su_validez():
    texto = formatear_contribuciones([VALIDA, INVALIDA])
    assert "investigador [VÁLIDA]" in texto
    assert "analista [INVÁLIDA: no usó la calculadora" in texto


def test_cada_aporte_llega_con_la_tarea_que_respondia():
    """El arreglo del bucle de la licencia por mudanza: el investigador contestaba "Corresponde
    1 día por año calendario" sin decir de qué, y el supervisor, sin ver la tarea, no podía
    saber que el dato ya estaba."""
    texto = formatear_contribuciones([VALIDA])
    assert "tarea: Buscá los días por antigüedad" in texto
    assert "aporte: 14 días" in texto


def test_sin_contribuciones_lo_dice():
    assert formatear_contribuciones([]) == "(ninguna todavía)"


@pytest.mark.asyncio
async def test_el_prompt_incluye_el_motivo_de_cada_invalida(supervisor):
    """El bucle de reparación depende de esto: el supervisor tiene que ver el error concreto
    para poder pedir una corrección concreta."""
    doble = supervisor(decidir("analista", "usá la calculadora"))
    await nodo_supervisor(estado(contribuciones=[VALIDA, INVALIDA]))
    assert "no usó la calculadora" in doble.prompts[0]


# --- Síntesis ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_la_sintesis_separa_lo_verificado_de_lo_que_no(sintesis):
    doble = sintesis("112 días.")
    actualizacion = await nodo_sintesis(estado(contribuciones=[VALIDA, INVALIDA]))

    prompt = doble.prompts[0]
    verificados = prompt.split("Aportes verificados")[1].split("NO pasaron")[0]
    assert "14 días" in verificados and "112" not in verificados
    assert "NO pasaron la validación" in prompt
    assert actualizacion["messages"][0].content == "112 días."


@pytest.mark.asyncio
async def test_si_la_sintesis_falla_igual_hay_respuesta(sintesis):
    """Si el LLM de la síntesis falla, la respuesta cae a los aportes verificados en crudo en
    vez de dejar al usuario sin nada."""
    sintesis(ConnectionError("sin red"))
    actualizacion = await nodo_sintesis(estado(contribuciones=[VALIDA]))
    contenido = actualizacion["messages"][0].content
    assert "No se pudo redactar" in contenido
    assert "14 días" in contenido


# --- Freno de repeticiones ---------------------------------------------------

@pytest.mark.asyncio
async def test_repetir_una_tarea_ya_resuelta_cierra_en_vez_de_ejecutarla(supervisor):
    """El bucle de la licencia por mudanza: el supervisor mandaba la misma instrucción, letra
    por letra, a un investigador que ya la había respondido con un aporte válido."""
    supervisor(decidir("investigador", VALIDA["tarea"], "falta el dato"))
    actualizacion = await nodo_supervisor(estado(contribuciones=[VALIDA]))

    assert actualizacion["next_agent"] == "FINISH"
    razon = actualizacion["decisiones"][0]["razon"]
    assert "ya resolvió" in razon
    assert "falta el dato" in razon  # queda registrado qué proponía el LLM


@pytest.mark.asyncio
async def test_la_repeticion_se_detecta_aunque_cambien_mayusculas_espacios_y_puntuacion(supervisor):
    supervisor(decidir("investigador", "  BUSCÁ los   días por antigüedad. "))
    actualizacion = await nodo_supervisor(estado(contribuciones=[VALIDA]))
    assert actualizacion["next_agent"] == "FINISH"


@pytest.mark.asyncio
async def test_reintentar_una_tarea_que_fallo_sigue_permitido(supervisor):
    """El freno no puede romper el bucle de reparación: si el aporte anterior fue INVÁLIDO,
    volver a pedir la misma tarea es exactamente lo que corresponde."""
    fallida = {**VALIDA, "valido": False, "motivo": "la búsqueda falló: timeout"}
    supervisor(decidir("investigador", VALIDA["tarea"]))
    actualizacion = await nodo_supervisor(estado(contribuciones=[fallida]))
    assert actualizacion["next_agent"] == "investigador"


@pytest.mark.asyncio
async def test_una_tarea_distinta_al_mismo_especialista_pasa(supervisor):
    supervisor(decidir("investigador", "Buscá además los días de licencia por mudanza"))
    actualizacion = await nodo_supervisor(estado(contribuciones=[VALIDA]))
    assert actualizacion["next_agent"] == "investigador"


@pytest.mark.asyncio
async def test_la_misma_tarea_a_otro_especialista_pasa(supervisor):
    """Lo que resolvió el investigador no cuenta como resuelto para el analista."""
    supervisor(decidir("analista", VALIDA["tarea"]))
    actualizacion = await nodo_supervisor(estado(contribuciones=[VALIDA]))
    assert actualizacion["next_agent"] == "analista"
