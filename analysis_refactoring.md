# Análisis de factorización: Actividad 1 (BETO) vs Actividad 2 (NER spaCy)

## Código compartido / duplicado

### 1. Carga de datos desde `data/`

| Aspecto | Actividad 1 | Actividad 2 |
|---------|-------------|-------------|
| Ruta | `DATA_DIR = "data"` | `DATA_DIR = os.path.join(os.path.dirname(__file__), "data")` |
| Formato | `{categoria}.txt` (1 archivo = 1 categoría) | `texto_{n}.txt` (1 archivo = 1 documento) |
| Lectura | `open()` directo con path relativo | `os.listdir()` + filtro, `os.path.join()` |

Ambas recorren `data/` y leen archivos `.txt` línea por línea. La diferencia principal es que Act1 asume un archivo por categoría (penal.txt, civil.txt...) y Act2 busca archivos con prefijo `texto_`. La lógica de iteración de archivos y lectura es prácticamente idéntica.

**Factorizable**: Una función `cargar_textos(patron_archivo="*.txt")` que devuelva un dict `{nombre_archivo_sin_ext: [lineas]}`.

---

### 2. Semilla de aleatoriedad

Act1 define `SEED = 42` y fija `random`, `numpy` y `torch`. Act2 **no** fija semilla. Si Act2 quisiera reproducibilidad, necesitaría el mismo boilerplate.

**Factorizable**: `utils.set_seed(seed=42)` que fije `random`, `numpy` y opcionalmente `torch`.

---

### 3. Importaciones compartidas

Ambos archivos importan:
- `pandas as pd` — igual, sin alias.
- `matplotlib.pyplot as plt` — igual.
- Ambos usan `print()` como logging.

Act1 además importa: `torch`, `transformers`, `datasets`, `sklearn`, `seaborn`, `LimpiadorTexto`.
Act2 además importa: `spacy`, `EntityRuler`, `displacy`, `Counter`, `os`, `warnings`.

**Factorizable**: No hay imports duplicados significativos más allá de `pandas` y `matplotlib`, pero se podría centralizar la configuración de matplotlib (estilo, DPI por defecto).

---

### 4. Visualización con matplotlib

Ambos scripts generan gráficas y las guardan como PNG:

| Aspecto | Actividad 1 | Actividad 2 |
|---------|-------------|-------------|
| Import | `import matplotlib.pyplot as plt` + `import seaborn as sns` | `import matplotlib.pyplot as plt` |
| DPI | `dpi=150` | `dpi=150` |
| Guardado | `plt.savefig("matriz_confusion_justia.png", dpi=150)` | `plt.savefig("ner_estadisticas_justia.png", dpi=150, bbox_inches="tight")` |
| Visualización | `plt.tight_layout()` + `plt.show()` | `plt.tight_layout()` + `plt.show()` |
| Estilo | Seaborn heatmap | Barras con `axes[0].bar()` |
| Títulos | `fontweight="bold"` | `fontweight="bold"` |

Patrón repetido:
```python
plt.tight_layout()
plt.savefig("...", dpi=150)
plt.show()
```

**Factorizable**: Función `guardar_grafica(fig, nombre, dpi=150, bbox_inches="tight")` que centralice el guardado con parámetros consistentes.

---

### 5. Constantes compartidas (dominio legal)

Categorías legales que aparecen en ambos:

| Categoría | Act1 (clasificación) | Act2 (patrones EntityRuler) |
|-----------|---------------------|----------------------------|
| `penal` | Sí (clase) | Sí (NORMA "código penal", DELITO "hurto calificado", etc.) |
| `civil` | Sí (clase) | Sí (NORMA "código civil", JURISDICCION "juzgado civil") |
| `laboral` | Sí (clase) | Sí (NORMA "código sustantivo del trabajo", DELITO "acoso laboral", JURISDICCION) |
| `familia` | Sí (clase) | Sí (NORMA "código de infancia y adolescencia", JURISDICCION "comisaría de familia") |

**Factorizable**: Un archivo `src/constantes.py` o `config.py` con `CATEGORIAS_JURIDICAS = ["penal", "civil", "laboral", "familia"]` y mapas relacionados.

---

### 6. Limpieza de texto

Act1 usa `LimpiadorTexto` del módulo `src.limpiador` para preprocesar (minúsculas, quitar tildes, stopwords, stemming conservador). Act2 **no limpia** los textos antes de pasarlos a spaCy (el pipeline de spaCy ya tokeniza internamente).

Sin embargo, la función `extraer_entidades` de Act2 aplica `texto.strip()` únicamente.

**Factorizable**: Si se quisiera preprocesar también en Act2, se podría reutilizar `LimpiadorTexto`. Por ahora es una oportunidad latente, no duplicación activa.

---

### 7. Exportación de resultados

| Aspecto | Actividad 1 | Actividad 2 |
|---------|-------------|-------------|
| A archivo | Solo imagen PNG | Imagen PNG + CSV + HTML |
| Formato tabla | `classification_report` (sklearn) | `df_ents.head().to_string()` + `value_counts()` |
| Consola | `print(f"Accuracy: ...")` | `print(df_ents.groupby(...))` |

Act2 exporta a CSV (`df_ents.to_csv("entidades_juridicas_justia.csv", index=False, encoding="utf-8-sig")`). Act1 no exporta datos a CSV.

**Factorizable**: Una función `exportar_csv(df, nombre)` compartida.

---

### 8. Mensajes de estado / logging

Ambos scripts usan `print()` extensivamente para indicar progreso. Ejemplos:
- `print("Tokenización completada")` (Act1)
- `print(f"Corpus cargado: {len(textos_juridicos)} textos jurídicos")` (Act2)
- `print("Imagen guardada: ...")` (ambos)

**Factorizable**: Un logger simple (`logging.basicConfig`) o envoltorio `info(msg)` que unifique formato.

---

### 9. Detección de dispositivo

Act1 tiene:
```python
print(f"Dispositivo: {'GPU' if torch.cuda.is_available() else 'CPU'}")
```

Act2 no tiene código de detección de GPU (spaCy corre en CPU principalmente). No es duplicado, pero podría integrarse en una utilidad compartida.

---

## Resumen de oportunidades de refactorización

| # | Componente propuesto | Ubicación sugerida | Archivos afectados |
|---|---------------------|-------------------|-------------------|
| 1 | `cargar_textos(patron)` — carga unificada desde `data/` | `src/data_loader.py` | Ambos |
| 2 | `config.py` — constantes compartidas (SEED, categorías, output dir, DPI) | `src/config.py` | Ambos |
| 3 | `set_seed(seed)` — reproducibilidad | `src/utils.py` | Act1 (y opcionalmente Act2) |
| 4 | `guardar_grafica(fig, nombre)` — matplotlib wrapper | `src/visualizacion.py` | Ambos |
| 5 | `exportar_csv(df, nombre)` — exportación CSV | `src/utils.py` | Act2 (y opcionalmente Act1) |
| 6 | Logger unificado en vez de `print()` | `src/utils.py` | Ambos |

### Dependencias entre archivos

```
src/
├── config.py          ← constantes (SEED, CATEGORIAS, OUTPUT_DIR, DPI)
├── utils.py           ← set_seed(), exportar_csv(), logger
├── data_loader.py     ← cargar_textos()
├── visualizacion.py   ← guardar_grafica()
├── limpiador.py       ← ya existe (LimpiadorTexto)
├── clasificador.py    ← ya existe
├── main.py            ← ya existe
└── ui_consola.py      ← ya existe
```

### Notas adicionales

- **Act1** ya tiene `src/limpiador.py`, `src/clasificador.py`, `src/main.py` y `src/ui_consola.py`. La estructura de módulos ya está iniciada. La factorización sugerida integraría las actividades como pipelines independientes que comparten utilidades base.
- **Act2** es autocontenido (sin imports a `src/`). Sería el que más ganaría con la factorización.
- Ambos scripts son *scripts lineales* (no funciones reutilizables). Convertir la lógica central en funciones (`main()`, `entrenar()`, `evaluar()`) dentro de `src/` permitiría importarlas desde cualquier lado.
