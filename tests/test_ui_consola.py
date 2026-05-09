from src.ui_consola import (
    mostrar_menu,
    opcion_limpiar,
    opcion_clasificar,
    opcion_limpiar_y_clasificar,
    opcion_procesar_archivo,
    main,
)


def test_mostrar_menu(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "5")
    opcion = mostrar_menu()
    captured = capsys.readouterr()
    assert "JUSTIA" in captured.out
    assert opcion == "5"


def test_opcion_limpiar(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "derecho penal")
    opcion_limpiar()
    captured = capsys.readouterr()
    assert "derecho penal" in captured.out


def test_opcion_clasificar_con_coincidencia(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "custodia del menor")
    opcion_clasificar()
    captured = capsys.readouterr()
    assert "familia" in captured.out


def test_opcion_clasificar_sin_coincidencia(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "el gato camina")
    opcion_clasificar()
    captured = capsys.readouterr()
    assert "No se pudo clasificar" in captured.out


def test_opcion_limpiar_y_clasificar(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "contrato de trabajo")
    opcion_limpiar_y_clasificar()
    captured = capsys.readouterr()
    assert "laboral" in captured.out


def test_opcion_procesar_archivo_no_encontrado(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "no_existe.txt")
    opcion_procesar_archivo()
    captured = capsys.readouterr()
    assert "Archivo no encontrado" in captured.out


def test_opcion_procesar_archivo_exitoso(monkeypatch, capsys, tmp_path):
    archivo = tmp_path / "test.txt"
    archivo.write_text("custodia del menor\ncontrato de trabajo", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda _: str(archivo))
    opcion_procesar_archivo()
    captured = capsys.readouterr()
    assert "Procesados" in captured.out


def test_main_salir(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "5")
    main()
    captured = capsys.readouterr()
    assert "Hasta luego" in captured.out
