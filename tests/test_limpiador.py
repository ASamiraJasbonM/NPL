import json
import tempfile
import os
from src.limpiador import LimpiadorTexto


def test_quitar_tildes():
    l = LimpiadorTexto()
    assert l.quitar_tildes("constitución") == "constitucion"
    assert l.quitar_tildes("acción") == "accion"
    assert l.quitar_tildes("jurídico") == "juridico"


def test_limpiar_texto():
    l = LimpiadorTexto()
    assert l.limpiar_texto("Artículo 15!") == "articulo"
    assert l.limpiar_texto("¡Hola, mundo!") == "hola mundo"


def test_preprocesar():
    l = LimpiadorTexto()
    r = l.preprocesar("El derecho a la vida es inviolable")
    assert "derecho" in r
    assert "vida" in r
    assert "el" not in r


def test_lematizar_conservador():
    l = LimpiadorTexto()
    assert l.lematizar_conservador("derechos") == "derecho"
    assert l.lematizar_conservador("leyes") == "ley"
    assert l.lematizar_conservador("el") == "el"


def test_limpiar_textos():
    l = LimpiadorTexto()
    textos = ["Hola mundo", "El derecho penal"]
    r = l.limpiar_textos(textos)
    assert len(r) == 2


def test_limpiar_archivo():
    l = LimpiadorTexto()
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, encoding="utf-8", suffix=".txt"
    ) as f:
        f.write("El derecho a la vida\n")
        f.write("La custodia del menor\n")
        path_in = f.name
    path_out = path_in.replace(".txt", ".json")
    try:
        r = l.limpiar_archivo(path_in, path_out)
        assert len(r) == 2
        assert os.path.exists(path_out)
        with open(path_out, encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) == 2
    finally:
        os.unlink(path_in)
        if os.path.exists(path_out):
            os.unlink(path_out)
