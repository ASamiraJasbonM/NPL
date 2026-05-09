# JustIA - Consultorio Virtual

Procesamiento de lenguaje natural para textos jurídicos colombianos. Limpieza, corrección y clasificación automática de documentos legales.

## Estructura

```
.
├── src/
│   ├── limpiador.py      # Limpieza y preprocesamiento de texto
│   ├── clasificador.py    # Clasificación por categorías legales
│   ├── ui_consola.py      # Interfaz de consola
│   └── main.py            # Punto de entrada
├── tests/
│   ├── test_limpiador.py
│   ├── test_clasificador.py
│   ├── test_ui_consola.py
│   └── test_main.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Requisitos

- Python >= 3.11
- Dependencias: `pip install -r requirements.txt`

## Uso

```bash
python -m src.main
```

Menú interactivo con opciones para limpiar textos, clasificarlos por área legal (familia, laboral, penal, civil) y procesar archivos.

## Tests

```bash
python -m pytest tests/ -v
```

## Docker

```bash
docker build -t justia-consola .
docker run -it justia-consola
```
