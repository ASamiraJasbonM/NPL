import json
import tempfile
from src.clasificador import ClasificadorTexto


DICCIONARIO = {
    "familia": ["custodia", "alimentos", "divorcio"],
    "penal": ["delito", "prision", "homicidio"],
    "laboral": ["contrato", "despido", "salario"],
}


def test_cargar_diccionario():
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, encoding="utf-8", suffix=".json"
    ) as f:
        json.dump(DICCIONARIO, f)
        ruta = f.name
    try:
        c = ClasificadorTexto(ruta)
        assert c.diccionario == DICCIONARIO
    finally:
        import os

        os.unlink(ruta)


def test_predecir_familia():
    c = ClasificadorTexto()
    c.diccionario = DICCIONARIO
    cat, score = c.predecir("El padre solicita la custodia de su hija")
    assert cat == "familia"
    assert score == 1


def test_predecir_penal():
    c = ClasificadorTexto()
    c.diccionario = DICCIONARIO
    cat, score = c.predecir("El delito de homicidio fue cometido")
    assert cat == "penal"
    assert score == 2


def test_predecir_sin_coincidencias():
    c = ClasificadorTexto()
    c.diccionario = DICCIONARIO
    cat, score = c.predecir("El gato camina por el tejado")
    assert cat is None
    assert score == 0


def test_predecir_muchos():
    c = ClasificadorTexto()
    c.diccionario = DICCIONARIO
    textos = [
        "Custodia del menor",
        "Contrato de trabajo",
        "El gato camina",
    ]
    resultados = c.predecir_muchos(textos)
    assert len(resultados) == 3
    assert resultados[0][0] == "familia"
    assert resultados[1][0] == "laboral"
    assert resultados[2][0] is None
