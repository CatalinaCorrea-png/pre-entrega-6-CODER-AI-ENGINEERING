"""El contrato `Contribucion` y la validación determinística de cada especialista."""

import json

import pytest
from pydantic import ValidationError

from agents.analyst_agent import validar_analisis
from agents.research_agent import validar_investigacion
from state import Contribucion

BUSCAR = "buscar_en_politicas_internas"
FRAGMENTO = "[Fuente: politica_vacaciones.txt] menos de 5 años: 14 días"
TAREA = "Buscá los días de vacaciones por antigüedad"


# --- El contrato -------------------------------------------------------------

def test_un_agente_que_no_existe_no_puede_firmar_una_contribucion():
    with pytest.raises(ValidationError):
        Contribucion(agente="redactor", tarea=TAREA, aporte="algo", valido=True)


def test_una_contribucion_sin_la_tarea_que_respondia_no_entra_al_estado():
    """Sin la tarea, el supervisor recibe una respuesta suelta y no puede saber a qué pregunta
    corresponde: es lo que lo hizo pedir cinco veces el mismo dato."""
    with pytest.raises(ValidationError):
        Contribucion(agente="investigador", aporte="1 día por año calendario", valido=True)


def test_la_contribucion_guarda_la_tarea():
    assert validar_investigacion(TAREA, "14 días", [(BUSCAR, FRAGMENTO)]).tarea == TAREA


def test_un_aporte_vacio_no_entra_al_estado():
    with pytest.raises(ValidationError):
        Contribucion(agente="investigador", tarea=TAREA, aporte="", valido=True)


def test_lo_que_entra_al_estado_es_serializable():
    """El estado guarda `model_dump()`, no el objeto: tiene que poder volcarse a la traza."""
    c = Contribucion(agente="analista", tarea=TAREA, aporte="112", valido=True,
                     herramientas_usadas=["calculadora"])
    assert json.loads(json.dumps(c.model_dump())) == c.model_dump()


# --- Investigador ------------------------------------------------------------

def test_investigacion_respaldada_por_la_herramienta_es_valida():
    c = validar_investigacion(TAREA, "14 días", [(BUSCAR, FRAGMENTO)])
    assert c.valido and c.motivo is None
    assert c.herramientas_usadas == [BUSCAR]


def test_responder_de_memoria_es_invalido():
    """El caso central: un dato que suena bien pero que no salió de las políticas."""
    c = validar_investigacion(TAREA, "Son 15 días.", [])
    assert not c.valido
    assert "sin consultar" in c.motivo


def test_una_busqueda_fallida_es_invalida_y_el_motivo_trae_el_error():
    c = validar_investigacion(TAREA, "No pude buscar.", [(BUSCAR, "ERROR: la búsqueda falló (Timeout)")])
    assert not c.valido
    assert "Timeout" in c.motivo  # el error concreto viaja al supervisor


def test_una_busqueda_sin_resultados_pide_reformular():
    c = validar_investigacion(TAREA, "Nada.", [(BUSCAR, "No se encontró información relevante en las políticas.")])
    assert not c.valido
    assert "reformular" in c.motivo


def test_si_un_reintento_de_la_busqueda_funciono_es_valida():
    """El agente puede reintentar solo dentro de su ciclo ReAct: alcanza con que una
    búsqueda haya devuelto fragmentos."""
    c = validar_investigacion(TAREA, "14 días", [(BUSCAR, "ERROR: timeout"), (BUSCAR, FRAGMENTO)])
    assert c.valido


def test_una_respuesta_vacia_es_invalida_y_aun_asi_cumple_el_contrato():
    c = validar_investigacion(TAREA, "   ", [(BUSCAR, FRAGMENTO)])
    assert not c.valido
    assert c.aporte  # Contribucion exige aporte no vacío: no revienta, se marca


# --- Analista ----------------------------------------------------------------

def test_un_resultado_de_la_calculadora_es_valido():
    c = validar_analisis(TAREA, "El total es 112.", [("calculadora", "Resultado: 112")])
    assert c.valido


def test_hacer_la_cuenta_de_cabeza_es_invalido():
    """Los LLMs se equivocan en aritmética con total seguridad: un número que no pasó por la
    calculadora no está verificado, aunque sea correcto."""
    c = validar_analisis(TAREA, "El total es 112.", [])
    assert not c.valido
    assert "calculadora" in c.motivo


def test_si_la_calculadora_solo_devolvio_errores_es_invalido():
    c = validar_analisis(TAREA, "No pude.", [("calculadora", "Error al calcular 'x': Expresión no permitida")])
    assert not c.valido
    assert "Expresión no permitida" in c.motivo


def test_si_corrigio_la_expresion_y_calculo_es_valido():
    c = validar_analisis(TAREA, "112", [("calculadora", "Error al calcular"), ("calculadora", "Resultado: 112")])
    assert c.valido
