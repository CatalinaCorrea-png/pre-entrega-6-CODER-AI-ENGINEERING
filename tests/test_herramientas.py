"""Las herramientas de los especialistas: la calculadora segura y la búsqueda."""

import time

import pytest

from agents.analyst_agent import calculadora
from agents.research_agent import buscar_en_politicas_internas


def calcular(expresion: str) -> str:
    return calculadora.invoke({"expresion": expresion})


# --- Calculadora: aritmética correcta ----------------------------------------

@pytest.mark.parametrize(
    "expresion, esperado",
    [
        ("3*14 + 2*21 + 1*28", "Resultado: 112"),  # la pregunta de demo
        ("2 + 3 * 4", "Resultado: 14"),            # precedencia
        ("(2 + 3) * 4", "Resultado: 20"),          # paréntesis
        ("-3 + 5", "Resultado: 2"),                # unario
        ("7 / 2", "Resultado: 3.5"),
        ("2 ** 10", "Resultado: 1024"),
    ],
)
def test_calcula_bien(expresion, esperado):
    assert calcular(expresion) == esperado


def test_un_error_de_calculo_vuelve_como_texto():
    """Una división por cero no puede romper el agente: vuelve como un string que empieza
    con Error, que es lo que la validación del analista detecta."""
    assert calcular("5 / 0").startswith("Error")


# --- Calculadora: lo que tiene que rechazar ----------------------------------

@pytest.mark.parametrize(
    "expresion",
    [
        "__import__('os').system('dir')",  # ejecución de código: el motivo de usar ast
        "open('secreto.txt').read()",
        "x + 1",                            # nombres
        "(1).__class__",                    # atributos
    ],
)
def test_rechaza_todo_lo_que_no_es_aritmetica(expresion):
    assert calcular(expresion).startswith("Error")


def test_rechaza_constantes_que_no_son_numeros():
    """El evaluador del notebook aceptaba cualquier constante: "'a' * 10**9" armaba un string
    de 1 GB en memoria."""
    assert calcular("'a' * 3").startswith("Error")
    assert calcular("True + 1").startswith("Error")


def test_un_exponente_gigante_se_rechaza_sin_colgar_el_proceso():
    """9**9**9 es aritmética válida y calcularla tardaría horas. El tope hace que se rechace
    al instante."""
    inicio = time.perf_counter()
    assert calcular("9 ** 9 ** 9").startswith("Error")
    assert time.perf_counter() - inicio < 1


# --- Búsqueda ----------------------------------------------------------------

def buscar(pregunta: str = "vacaciones") -> str:
    return buscar_en_politicas_internas.invoke({"pregunta": pregunta})


def test_cita_la_fuente_de_cada_fragmento(rag):
    rag([
        {"contenido": "menos de 5 años: 14 días", "fuente": "politica_vacaciones.txt"},
        {"contenido": "2 días presenciales", "fuente": "politica_teletrabajo.txt"},
    ])
    resultado = buscar()
    assert "[Fuente: politica_vacaciones.txt] menos de 5 años: 14 días" in resultado
    assert "[Fuente: politica_teletrabajo.txt]" in resultado


def test_sin_resultados_lo_dice(rag):
    rag([])
    assert buscar().startswith("No se encontró información relevante")


def test_si_pinecone_falla_devuelve_ERROR_en_vez_de_explotar(rag):
    """Un nodo que lanza una excepción sin capturar deja el estado en un limbo. La falla de
    la búsqueda vuelve como texto: el agente la lee y la validación la marca inválida."""
    rag(error=ConnectionError("sin red"))
    resultado = buscar()
    assert resultado.startswith("ERROR:")
    assert "ConnectionError" in resultado


# --- El contrato que ven los LLMs --------------------------------------------

@pytest.mark.parametrize("herramienta", [calculadora, buscar_en_politicas_internas], ids=lambda h: h.name)
def test_la_descripcion_avisa_como_se_ve_un_error(herramienta):
    """El docstring es lo único que el LLM lee de la herramienta. Si no dice cómo se ve una
    falla, el agente puede tomar el mensaje de error como si fuera un dato."""
    assert "error" in herramienta.description.lower()
