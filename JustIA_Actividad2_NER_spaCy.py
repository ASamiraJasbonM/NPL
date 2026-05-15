# JustIA – Actividad 2
# Extractor de Entidades Nombradas (NER) en Texto Jurídico con spaCy
# **Corporación Universitaria de Asturias**
#
# Objetivo: Construir un extractor de entidades nombradas (NER) en texto jurídico colombiano para identificar:
# - Nombres de personas (jueces, partes procesales)
# - Fechas de hechos, sentencias y actuaciones
# - Normas jurídicas (leyes, artículos, decretos)
# - Tipos de violencia / delitos
# - Jurisdicciones y entidades (juzgados, tribunales)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 0. Instalación de dependencias (ejecutar solo la primera vez)
# ---------------------------------------------------------------------------
# Descomenta las siguientes líneas si es la primera vez que ejecutas:
# import subprocess, sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy", "--quiet"])
# subprocess.check_call([sys.executable, "-m", "spacy", "download", "es_core_news_md", "--quiet"])
# print("spaCy y modelo es_core_news_md instalados")

# ---------------------------------------------------------------------------
# 1. Importaciones
# ---------------------------------------------------------------------------
import spacy
from spacy.pipeline import EntityRuler
from spacy import displacy
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import warnings
from src.config import SEED, DATA_DIR, OUTPUT_DPI
from src.utils import set_seed, exportar_csv
from src.data_loader import cargar_textos_por_documento
from src.visualizacion import guardar_grafica

warnings.filterwarnings("ignore")

set_seed(SEED)

print(f"spaCy versión: {spacy.__version__}")

# ---------------------------------------------------------------------------
# 2. Carga del modelo base en español
# ---------------------------------------------------------------------------
nlp = spacy.load("es_core_news_md")

print("Pipeline original:", nlp.pipe_names)
print(f"Idioma: {nlp.meta['lang']}")

# ---------------------------------------------------------------------------
# 3. Corpus simulado: 10 textos de sentencias jurídicas colombianas
# ---------------------------------------------------------------------------
textos_juridicos = cargar_textos_por_documento(DATA_DIR)

print(f"Corpus cargado: {len(textos_juridicos)} textos jurídicos desde {DATA_DIR}")

# ---------------------------------------------------------------------------
# 4. Definición de patrones jurídicos con EntityRuler
# ---------------------------------------------------------------------------
if "entity_ruler" in nlp.pipe_names:
    nlp.remove_pipe("entity_ruler")

ruler = nlp.add_pipe("entity_ruler", before="ner")

patrones = [
    # -- NORMAS JURÍDICAS --
    {
        "label": "NORMA",
        "pattern": [
            {"LOWER": "ley"},
            {"IS_DIGIT": True},
            {"LOWER": "de"},
            {"IS_DIGIT": True},
        ],
    },
    {
        "label": "NORMA",
        "pattern": [
            {"LOWER": "decreto"},
            {"IS_DIGIT": True},
            {"LOWER": "de"},
            {"IS_DIGIT": True},
        ],
    },
    {
        "label": "NORMA",
        "pattern": [
            {"LOWER": "decreto"},
            {"LOWER": "ley"},
            {"IS_DIGIT": True},
            {"LOWER": "de"},
            {"IS_DIGIT": True},
        ],
    },
    {
        "label": "NORMA",
        "pattern": [
            {"LOWER": {"IN": ["artículo", "articulo", "art."]}},
            {"IS_DIGIT": True},
        ],
    },
    {
        "label": "NORMA",
        "pattern": [
            {"LOWER": "código"},
            {"LOWER": {"IN": ["penal", "civil", "laboral", "general"]}},
        ],
    },
    {
        "label": "NORMA",
        "pattern": [
            {"LOWER": "código"},
            {"LOWER": "general"},
            {"LOWER": "del"},
            {"LOWER": "proceso"},
        ],
    },
    {
        "label": "NORMA",
        "pattern": [
            {"LOWER": "código"},
            {"LOWER": "de"},
            {"LOWER": "infancia"},
            {"LOWER": "y"},
            {"LOWER": "adolescencia"},
        ],
    },
    {
        "label": "NORMA",
        "pattern": [
            {"LOWER": "código"},
            {"LOWER": "sustantivo"},
            {"LOWER": "del"},
            {"LOWER": "trabajo"},
        ],
    },
    # -- TIPOS DE VIOLENCIA / DELITOS --
    {
        "label": "DELITO",
        "pattern": [{"LOWER": "violencia"}, {"LOWER": "intrafamiliar"}],
    },
    {"label": "DELITO", "pattern": [{"LOWER": "violencia"}, {"LOWER": "doméstica"}]},
    {"label": "DELITO", "pattern": [{"LOWER": "violencia"}, {"LOWER": "sexual"}]},
    {"label": "DELITO", "pattern": [{"LOWER": "violencia"}, {"LOWER": "económica"}]},
    {"label": "DELITO", "pattern": [{"LOWER": "violencia"}, {"LOWER": "patrimonial"}]},
    {"label": "DELITO", "pattern": [{"LOWER": "acoso"}, {"LOWER": "laboral"}]},
    {"label": "DELITO", "pattern": [{"LOWER": "desplazamiento"}, {"LOWER": "forzado"}]},
    {
        "label": "DELITO",
        "pattern": [{"LOWER": "tráfico"}, {"OP": "?"}, {"LOWER": "estupefacientes"}],
    },
    {
        "label": "DELITO",
        "pattern": [
            {"LOWER": "tráfico"},
            {"LOWER": ","},
            {"LOWER": "fabricación"},
            {"LOWER": "o"},
            {"LOWER": "porte"},
            {"LOWER": "de"},
            {"LOWER": "estupefacientes"},
        ],
    },
    {
        "label": "DELITO",
        "pattern": [{"LOWER": "acceso"}, {"LOWER": "carnal"}, {"LOWER": "violento"}],
    },
    {"label": "DELITO", "pattern": [{"LOWER": "homicidio"}, {"LOWER": "agravado"}]},
    {"label": "DELITO", "pattern": [{"LOWER": "secuestro"}, {"LOWER": "extorsivo"}]},
    {"label": "DELITO", "pattern": [{"LOWER": "feminicidio"}]},
    {
        "label": "DELITO",
        "pattern": [{"LOWER": "peculado"}, {"LOWER": "por"}, {"LOWER": "apropiación"}],
    },
    {"label": "DELITO", "pattern": [{"LOWER": "hurto"}, {"LOWER": "calificado"}]},
    {"label": "DELITO", "pattern": [{"LOWER": "extorsión"}]},
    {
        "label": "DELITO",
        "pattern": [{"LOWER": "lavado"}, {"LOWER": "de"}, {"LOWER": "activos"}],
    },
    {
        "label": "DELITO",
        "pattern": [{"LOWER": "concierto"}, {"LOWER": "para"}, {"LOWER": "delinquir"}],
    },
    # -- JURISDICCIONES / ENTIDADES --
    {
        "label": "JURISDICCION",
        "pattern": [
            {"LOWER": "corte"},
            {"LOWER": "suprema"},
            {"LOWER": "de"},
            {"LOWER": "justicia"},
        ],
    },
    {
        "label": "JURISDICCION",
        "pattern": [{"LOWER": "consejo"}, {"LOWER": "de"}, {"LOWER": "estado"}],
    },
    {
        "label": "JURISDICCION",
        "pattern": [{"LOWER": "corte"}, {"LOWER": "constitucional"}],
    },
    {
        "label": "JURISDICCION",
        "pattern": [
            {"LOWER": "fiscalía"},
            {"LOWER": "general"},
            {"LOWER": "de"},
            {"LOWER": "la"},
            {"LOWER": "nación"},
        ],
    },
    {
        "label": "JURISDICCION",
        "pattern": [{"LOWER": "tribunal"}, {"LOWER": "superior"}],
    },
    {
        "label": "JURISDICCION",
        "pattern": [{"LOWER": "juzgado"}, {"OP": "?"}, {"OP": "?"}, {"LOWER": "penal"}],
    },
    {
        "label": "JURISDICCION",
        "pattern": [{"LOWER": "juzgado"}, {"OP": "?"}, {"OP": "?"}, {"LOWER": "civil"}],
    },
    {
        "label": "JURISDICCION",
        "pattern": [
            {"LOWER": "juzgado"},
            {"OP": "?"},
            {"OP": "?"},
            {"LOWER": "laboral"},
        ],
    },
    {
        "label": "JURISDICCION",
        "pattern": [
            {"LOWER": "juzgado"},
            {"OP": "?"},
            {"OP": "?"},
            {"LOWER": "de"},
            {"LOWER": "familia"},
        ],
    },
    {
        "label": "JURISDICCION",
        "pattern": [{"LOWER": "ministerio"}, {"LOWER": "del"}, {"LOWER": "trabajo"}],
    },
    {
        "label": "JURISDICCION",
        "pattern": [{"LOWER": "ministerio"}, {"LOWER": "de"}, {"LOWER": "defensa"}],
    },
    {"label": "JURISDICCION", "pattern": [{"LOWER": "icbf"}]},
    {"label": "JURISDICCION", "pattern": [{"LOWER": "colpensiones"}]},
    {
        "label": "JURISDICCION",
        "pattern": [{"LOWER": "comisaría"}, {"LOWER": "de"}, {"LOWER": "familia"}],
    },
    {"label": "JURISDICCION", "pattern": [{"LOWER": "caivas"}]},
    {
        "label": "JURISDICCION",
        "pattern": [
            {"LOWER": "unidad"},
            {"LOWER": "para"},
            {"LOWER": "la"},
            {"LOWER": "atención"},
        ],
    },
]

ruler.add_patterns(patrones)

print(f"EntityRuler configurado con {len(patrones)} patrones jurídicos")
print("Pipeline actualizado:", nlp.pipe_names)

# ---------------------------------------------------------------------------
# 5. Función de extracción y procesamiento de entidades
# ---------------------------------------------------------------------------
ETIQUETAS_INTERES = {
    "PER": "Persona",
    "NORMA": "Norma juridica",
    "DELITO": "Delito / Tipo de violencia",
    "JURISDICCION": "Jurisdiccion / Entidad",
    "DATE": "Fecha",
    "ORG": "Organizacion",
    "LOC": "Lugar",
    "GPE": "Lugar (geopolitico)",
}


def extraer_entidades(texto, idx=None):
    """Procesa un texto y retorna sus entidades jurídicas relevantes."""
    doc = nlp(texto.strip())
    resultados = []
    for ent in doc.ents:
        if ent.label_ in ETIQUETAS_INTERES:
            resultados.append(
                {
                    "texto_id": idx if idx is not None else "-",
                    "entidad": ent.text.strip(),
                    "tipo": ent.label_,
                    "descripcion": ETIQUETAS_INTERES[ent.label_],
                    "inicio": ent.start_char,
                    "fin": ent.end_char,
                }
            )
    return doc, resultados


print("Funcion de extraccion definida")

# ---------------------------------------------------------------------------
# 6. Procesamiento del corpus completo
# ---------------------------------------------------------------------------
todos_los_registros = []
documentos_nlp = []

for i, texto in enumerate(textos_juridicos, start=1):
    doc, registros = extraer_entidades(texto, idx=i)
    documentos_nlp.append(doc)
    todos_los_registros.extend(registros)
    print(f"Texto {i:2d}: {len(registros):3d} entidades encontradas")

df_ents = pd.DataFrame(todos_los_registros)
print(f"\nTotal entidades extraídas: {len(df_ents)}")

# ---------------------------------------------------------------------------
# 7. Tabla resumen de entidades extraídas
# ---------------------------------------------------------------------------
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.max_rows", 60)

print("=" * 70)
print(" ENTIDADES EXTRAÍDAS POR DOCUMENTO")
print("=" * 70)
print(
    df_ents[["texto_id", "entidad", "tipo", "descripcion"]]
    .head(50)
    .to_string(index=False)
)

print("\nDistribucion por tipo de entidad:")
print(df_ents["tipo"].value_counts().to_string())

# ---------------------------------------------------------------------------
# 8. Visualización con displacy (renderizado HTML)
# ---------------------------------------------------------------------------
COLORES = {
    "PER": "#AED6F1",
    "NORMA": "#A9DFBF",
    "DELITO": "#F1948A",
    "JURISDICCION": "#F9E79F",
    "DATE": "#D7BDE2",
    "ORG": "#FAD7A0",
    "LOC": "#ABEBC6",
    "GPE": "#AED6F1",
}

opciones_displacy = {
    "ents": list(ETIQUETAS_INTERES.keys()),
    "colors": COLORES,
}

print("\nVisualización con displacy (los primeros 3 textos):")
for i in range(3):
    print(f"\n{'─' * 70}")
    print(f" TEXTO {i + 1}")
    print(f"{'─' * 70}")
    html = displacy.render(
        documentos_nlp[i],
        style="ent",
        options=opciones_displacy,
        jupyter=False,
    )
    # Guardar HTML a archivo para visualizar en navegador
    with open(f"texto_{i + 1}_entidades.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Renderizado HTML guardado en: texto_{i + 1}_entidades.html")

# ---------------------------------------------------------------------------
# 9. Visualización estadística: frecuencia de entidades
# ---------------------------------------------------------------------------
if not df_ents.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: distribución por tipo
    conteo_tipo = df_ents["tipo"].value_counts()
    colores_barras = [COLORES.get(t, "#CCCCCC") for t in conteo_tipo.index]
    axes[0].bar(
        conteo_tipo.index, conteo_tipo.values, color=colores_barras, edgecolor="#555"
    )
    axes[0].set_title("Entidades por tipo (corpus completo)", fontweight="bold")
    axes[0].set_xlabel("Tipo de entidad")
    axes[0].set_ylabel("Frecuencia")
    axes[0].tick_params(axis="x", rotation=30)
    for bar, v in zip(axes[0].patches, conteo_tipo.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(v),
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # Gráfico 2: top 15 entidades más frecuentes
    top15 = df_ents["entidad"].value_counts().head(15)
    axes[1].barh(
        top15.index[::-1], top15.values[::-1], color="#85C1E9", edgecolor="#555"
    )
    axes[1].set_title("Top 15 entidades mas frecuentes", fontweight="bold")
    axes[1].set_xlabel("Frecuencia")
    for bar, v in zip(axes[1].patches, top15.values[::-1]):
        axes[1].text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            str(v),
            va="center",
            fontsize=9,
        )

    plt.suptitle(
        "JustIA – Analisis NER sobre corpus juridico colombiano",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    guardar_grafica(fig, "ner_estadisticas_justia.png", dpi=OUTPUT_DPI)
else:
    print("No hay entidades para graficar.")

# ---------------------------------------------------------------------------
# 10. Prueba con texto libre ingresado por el usuario
# ---------------------------------------------------------------------------
texto_prueba = """
La señora Carmen Lucía Rojas presentó denuncia el 5 de marzo de 2024 ante la Comisaría
de Familia de Soacha, Cundinamarca, por violencia intrafamiliar y violencia económica
por parte de su cónyuge. La Fiscalía General de la Nación inició investigación conforme
al artículo 229 del Código Penal y la Ley 1257 de 2008 sobre no violencia contra
la mujer. El Juzgado Tercero Penal Municipal decretó medida de protección inmediata.
"""

doc_prueba, ents_prueba = extraer_entidades(texto_prueba)

print("\nEntidades encontradas en texto de prueba:")
print("-" * 60)
for e in ents_prueba:
    print(f"  {e['descripcion']:35s} -> '{e['entidad']}'")

print("\nVisualizacion resaltada guardada en: texto_prueba_entidades.html")
html_prueba = displacy.render(
    doc_prueba,
    style="ent",
    options=opciones_displacy,
    jupyter=False,
)
with open("texto_prueba_entidades.html", "w", encoding="utf-8") as f:
    f.write(html_prueba)

# ---------------------------------------------------------------------------
# 11. Exportar resultados a CSV
# ---------------------------------------------------------------------------
exportar_csv(df_ents, "entidades_juridicas_justia.csv")
print(df_ents.groupby("tipo")["entidad"].count().rename("total").to_frame().to_string())
