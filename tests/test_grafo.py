"""La topología: quién está conectado con quién."""

from graph import app, enrutar


def _aristas():
    return {(a.source, a.target): a.conditional for a in app.get_graph().edges}


def test_los_nodos_son_supervisor_dos_especialistas_y_sintesis():
    assert {"supervisor", "investigador", "analista", "sintesis"} <= set(app.get_graph().nodes)


def test_todo_arranca_por_el_supervisor():
    assert ("__start__", "supervisor") in _aristas()


def test_los_especialistas_siempre_vuelven_al_supervisor():
    """La topología jerárquica en una línea: los especialistas nunca hablan entre sí ni
    terminan el grafo por su cuenta."""
    aristas = _aristas()
    assert aristas[("investigador", "supervisor")] is False
    assert aristas[("analista", "supervisor")] is False
    assert ("investigador", "analista") not in aristas
    assert ("analista", "investigador") not in aristas


def test_el_supervisor_es_el_unico_que_decide_y_lo_hace_con_aristas_condicionales():
    condicionales = {destino for (origen, destino), cond in _aristas().items() if cond}
    assert condicionales == {"investigador", "analista", "sintesis"}
    assert all(origen == "supervisor" for (origen, _), cond in _aristas().items() if cond)


def test_solo_la_sintesis_llega_a_END():
    assert [o for (o, d) in _aristas() if d == "__end__"] == ["sintesis"]


def test_enrutar_traduce_FINISH_a_la_sintesis():
    assert enrutar({"next_agent": "FINISH"}) == "sintesis"
    assert enrutar({"next_agent": "investigador"}) == "investigador"
    assert enrutar({"next_agent": "analista"}) == "analista"
