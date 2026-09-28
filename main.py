"""Corre una consulta contra el orquestador y muestra el flujo de delegación en vivo.

    python main.py                                  # la pregunta de demo
    python main.py "¿Cuántos días presenciales exige el esquema híbrido?"

Cada paso se imprime a medida que ocurre —quién decidió qué, con qué razón, y si el aporte
de cada especialista pasó la validación— y al final se guarda la traza en
`traza_delegacion.json`.
"""

import argparse
import asyncio
import json
from pathlib import Path

from agents.llm import MODELO
from graph import CONFIG, app, estado_inicial
from rag.config import correr

RAIZ = Path(__file__).resolve().parent
TRAZA_PATH = RAIZ / "traza_delegacion.json"

# Sin los números adentro a propósito: el notebook de la clase los daba en la pregunta, y así
# el investigador sobraba. Acá el dato solo está en las políticas, así que el supervisor
# tiene que delegar la búsqueda y después la cuenta. Respuesta esperada:
# 3 x 14 + 2 x 21 + 1 x 28 = 112 días.
PREGUNTA_DEMO = (
    "En un equipo de TechCorp hay 3 personas con 2 años de antigüedad, 2 con 7 años y 1 con "
    "12 años. ¿Cuántos días de vacaciones suman en total este año según la política vigente?"
)


def _mostrar(nodo: str, actualizacion: dict) -> None:
    if nodo == "supervisor":
        decision = actualizacion["decisiones"][-1]
        print(f"\n[supervisor] -> {decision['next']}")
        print(f"   razón: {decision['razon']}")
        if decision["instruccion"]:
            print(f"   instrucción: {decision['instruccion']}")
    elif nodo in ("investigador", "analista"):
        c = actualizacion["contribuciones"][-1]
        marca = "VÁLIDA" if c["valido"] else f"INVÁLIDA: {c['motivo']}"
        print(f"\n[{nodo}] {marca}  (herramientas: {', '.join(c['herramientas_usadas']) or 'ninguna'})")
        print(f"   {c['aporte'][:400]}")
    elif nodo == "sintesis":
        print(f"\n[síntesis]\n{actualizacion['messages'][-1].content}")


async def correr_consulta(pregunta: str, mostrar: bool = True) -> dict:
    """Ejecuta el grafo con `astream` y devuelve la traza completa de la corrida.

    Solo `astream`: el notebook de la clase corría `stream` para mostrar el flujo y después
    `invoke` para la respuesta, o sea todo el grafo dos veces, pagando dos veces cada llamada.
    Acá la traza se arma con lo que ya pasó por el stream.
    """
    flujo: list[str] = []
    decisiones: list[dict] = []
    contribuciones: list[dict] = []
    respuesta_final = ""

    async for evento in app.astream(estado_inicial(pregunta), config=CONFIG, stream_mode="updates"):
        for nodo, actualizacion in evento.items():
            flujo.append(nodo)
            decisiones += actualizacion.get("decisiones", [])
            contribuciones += actualizacion.get("contribuciones", [])
            if nodo == "sintesis":
                respuesta_final = actualizacion["messages"][-1].content
            if mostrar:
                _mostrar(nodo, actualizacion)

    return {
        "pregunta": pregunta,
        "modelo": MODELO,
        "flujo": flujo,
        "decisiones": decisiones,
        "contribuciones": contribuciones,
        "respuesta_final": respuesta_final,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pregunta", nargs="?", default=PREGUNTA_DEMO)
    args = parser.parse_args()

    print(f"Pregunta: {args.pregunta}")
    traza = asyncio.run(correr_consulta(args.pregunta))

    print(f"\nFlujo: {' -> '.join(traza['flujo'])}")
    TRAZA_PATH.write_text(json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Traza guardada en {TRAZA_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(correr(main))
