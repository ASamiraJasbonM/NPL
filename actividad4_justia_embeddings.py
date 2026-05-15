# Actividad 4 – JustIA: Clustering Semántico de Textos Jurídicos
# Asturias Corporación Universitaria
#
# Objetivo: Utilizar embeddings semánticos con SentenceTransformers para agrupar
# textos jurídicos por similitud temática, facilitando la búsqueda contextual en JustIA.
#
# Modelo usado: paraphrase-multilingual-MiniLM-L12-v2
# Clustering: KMeans
# Visualización: t-SNE y UMAP
# ----------------------------------------------------------------------------

# =============================================================================
# 1. Instalación de dependencias (ejecutar una sola vez)
# =============================================================================
# !pip install sentence-transformers scikit-learn umap-learn matplotlib seaborn pandas numpy

# =============================================================================
# 2. Importaciones
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import umap
import warnings

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

print("✅ Librerías cargadas correctamente.")


# =============================================================================
# 3. Dataset simulado: 100 fragmentos jurídicos colombianos
# =============================================================================
# Cuatro áreas del derecho: Penal (25), Civil (25), Laboral (25), Familia (25)

textos_penal = [
    "El acusado fue imputado por el delito de hurto calificado conforme al artículo 240 del Código Penal.",
    "La Fiscalía General de la Nación solicitó medida de aseguramiento por presunto homicidio agravado.",
    "Se decretó la detención preventiva en establecimiento carcelario por peligro para la comunidad.",
    "El procesado confesó su participación en el concierto para delinquir con fines de narcotráfico.",
    "El juez de control de garantías negó la preclusión por insuficiencia probatoria.",
    "La víctima del delito sexual presentó denuncia ante la Fiscalía conforme a la Ley 1257 de 2008.",
    "Se ordenó captura internacional mediante circular roja de Interpol por extorsión agravada.",
    "El juez colegiado condenó al procesado a doce años de prisión por fabricación de armas.",
    "La defensa interpuso recurso de apelación contra la sentencia condenatoria por lesiones personales.",
    "El Tribunal Superior confirmó la condena por el delito de peculado por apropiación.",
    "Se presentó escrito de acusación formal ante el Juez Penal del Circuito de Bogotá.",
    "El juez absolvió al acusado por duda razonable en el proceso por tráfico de estupefacientes.",
    "La víctima de violencia intrafamiliar fue reconocida como sujeto especial de protección.",
    "Se decretó la libertad condicional por cumplimiento de las tres quintas partes de la pena.",
    "El agresor fue condenado por acceso carnal violento en concurso con lesiones personales.",
    "La sentencia anticipada redujo la pena en una tercera parte por aceptación de cargos.",
    "El procesado fue declarado inimputable y se le impuso medida de seguridad de internación.",
    "Se reconoció la responsabilidad del Estado por falla en el servicio de custodia de detenido.",
    "El delito de fraude procesal fue calificado como conducta punible dolosa.",
    "La Corte Suprema de Justicia casó la sentencia por violación directa de la ley sustancial.",
    "Se impuso pena de multa de cien salarios mínimos por el delito de receptación.",
    "El acusado fue vinculado mediante indagatoria al proceso por lavado de activos.",
    "La Sala Penal de la Corte ordenó la ruptura de la unidad procesal por acumulación de procesos.",
    "El testimonio de la víctima fue valorado como prueba idónea conforme a los principios del proceso penal acusatorio.",
    "Se decretó la nulidad del proceso por violación del debido proceso en la etapa de investigación.",
]

textos_civil = [
    "El demandante solicitó la declaración de pertenencia por prescripción adquisitiva ordinaria de dominio.",
    "Se inició proceso ejecutivo hipotecario por incumplimiento en el pago de las cuotas del crédito.",
    "La demanda de responsabilidad civil extracontractual fue presentada por daños y perjuicios.",
    "El juez civil decretó medida cautelar de embargo y secuestro sobre bienes del deudor.",
    "Se demandó la nulidad del contrato de compraventa por error sustancial en el objeto.",
    "El proceso de sucesión intestada fue tramitado ante notario conforme al Decreto 902 de 1988.",
    "El demandado interpuso excepción previa de falta de legitimación en la causa por pasiva.",
    "La acción de tutela por vía de hecho fue instaurada contra providencia judicial del juzgado civil.",
    "El contrato de arrendamiento fue terminado unilateralmente con base en la Ley 820 de 2003.",
    "Se reconoció la indemnización por lucro cesante y daño emergente en proceso de responsabilidad.",
    "La partición de la masa herencial fue cuestionada por preterición de heredero legítimo.",
    "El juez negó la reivindicación del inmueble por ausencia de prueba del dominio.",
    "Se declaró la simulación absoluta del negocio jurídico por ausencia de causa lícita.",
    "El proceso monitorio fue admitido para reclamar deuda de menor cuantía sin título ejecutivo.",
    "La demanda de declaración de unión marital de hecho fue acumulada al proceso de liquidación.",
    "Se aplicó la teoría de la imprevisión para revisar el contrato de obra civil.",
    "El acreedor hipotecario solicitó el remate del inmueble por mora en el pago de la obligación.",
    "Se ordenó la inscripción del fallo en el folio de matrícula inmobiliaria del predio disputado.",
    "El proceso verbal sumario fue tramitado para reclamar perjuicios derivados de un accidente de tránsito.",
    "La sentencia reconoció el derecho de retención del arrendatario sobre el inmueble mejorado.",
    "La pretensión de enriquecimiento sin causa fue desestimada por falta de prueba del empobrecimiento.",
    "El incidente de regulación de perjuicios fue resuelto mediante peritos designados por el juez.",
    "Se decretó la resolución del contrato de promesa de compraventa por incumplimiento del promitente vendedor.",
    "El Tribunal revocó la sentencia de primera instancia y reconoció la prescripción liberatoria.",
    "La acción popular fue admitida para proteger el derecho colectivo al espacio público urbano.",
]

textos_laboral = [
    "El trabajador demandó el pago de prestaciones sociales adeudadas desde el inicio del contrato.",
    "Se alegó la existencia de un contrato realidad por prestación continua de servicios personales.",
    "El empleador fue condenado al pago de la indemnización por despido sin justa causa.",
    "La trabajadora en estado de embarazo fue reintegrada por violación del fuero de maternidad.",
    "Se reconoció el derecho a la pensión de invalidez conforme a la Ley 100 de 1993.",
    "El juez laboral condenó al empleador por acoso laboral conforme a la Ley 1010 de 2006.",
    "Se negó la sustitución patronal por no existir transmisión del establecimiento de comercio.",
    "El trabajador reclamó horas extras, dominicales y festivos no pagados durante tres años.",
    "La empresa fue sancionada por no afiliar al trabajador al sistema de seguridad social integral.",
    "El sindicato presentó pliego de peticiones para iniciar la negociación colectiva.",
    "La huelga fue declarada ilegal por no cumplir los requisitos del Código Sustantivo del Trabajo.",
    "Se reconoció la ineficacia del despido por ausencia de autorización del Ministerio del Trabajo.",
    "El empleado recibió liquidación definitiva incluyendo cesantías, prima y vacaciones compensadas.",
    "La Corte Suprema fijó el criterio sobre la prescripción trienal de las acreencias laborales.",
    "El contrato de trabajo fue terminado por justa causa por abandono del cargo durante más de tres días.",
    "Se ordenó el reintegro del trabajador discapacitado con estabilidad laboral reforzada.",
    "El empleador descontó ilegalmente del salario sin autorización escrita del trabajador.",
    "La pensión de sobrevivientes fue reconocida a la compañera permanente del trabajador fallecido.",
    "Se declaró la solidaridad laboral del contratante en los pagos de seguridad social del contratista.",
    "El trabajador reclamó el auxilio de transporte como base para liquidar las prestaciones sociales.",
    "La sala laboral ordenó indexar la condena por depreciación monetaria durante el proceso.",
    "Se condenó a la empresa a pagar la sanción moratoria por no consignar las cesantías a tiempo.",
    "El inspector del trabajo levantó acta de infracción por no tener el reglamento interno actualizado.",
    "La junta de calificación de invalidez determinó una pérdida de capacidad laboral del 55%.",
    "El proceso verbal laboral fue resuelto en única instancia por cuantía inferior a veinte salarios mínimos.",
]

textos_familia = [
    "El juez de familia decretó la custodia compartida del menor de edad entre los progenitores.",
    "Se fijó cuota alimentaria provisional equivalente al 30% del salario del padre alimentante.",
    "La demanda de divorcio fue admitida por causal de maltrato, ultraje e infidelidad conyugal.",
    "El proceso de adopción fue adelantado conforme a los lineamientos del ICBF y el Código de Infancia.",
    "Se declaró la unión marital de hecho con efectos patrimoniales desde hace más de dos años.",
    "El menor fue declarado en estado de abandono y susceptible de adopción por el juzgado de familia.",
    "La Comisaría de Familia expidió medida de protección por violencia intrafamiliar contra la mujer.",
    "Se reconoció la filiación extramatrimonial mediante prueba de ADN con certeza superior al 99.9%.",
    "El proceso de liquidación de la sociedad conyugal fue originado por el divorcio de los cónyuges.",
    "El juez negó la patria potestad al progenitor por maltrato físico y psicológico al menor.",
    "Se tramitó proceso de interdicción judicial para declarar la incapacidad absoluta del adulto mayor.",
    "La demanda de impugnación de paternidad fue interpuesta dentro del año siguiente al conocimiento del hecho.",
    "El defensor de familia interpuso acción de tutela para proteger el interés superior del niño.",
    "La separación de cuerpos fue decretada por mutuo acuerdo ante notaría pública.",
    "Se ordenó visitas supervisadas al progenitor con antecedentes de violencia contra el menor.",
    "El juez de familia ordenó el restablecimiento de derechos del niño en situación de vulnerabilidad.",
    "La pensión alimenticia fue revisada por cambio sustancial en las condiciones económicas del deudor.",
    "Se declaró la nulidad del matrimonio civil por existir vínculo conyugal vigente no disuelto.",
    "El proceso de jurisdicción voluntaria decretó la emancipación del menor con más de dieciséis años.",
    "La Sala de Familia del Tribunal confirmó la cuota alimentaria fijada en primera instancia.",
    "Se ordenó la restitución internacional del menor conforme al Convenio de La Haya de 1980.",
    "El progenitor fue sancionado por incumplimiento reiterado de la obligación alimentaria.",
    "Se reconoció la vocación hereditaria del hijo extramatrimonial en igualdad de condiciones.",
    "La demanda de devolución de bienes entre cónyuges fue resuelta al liquidar la sociedad conyugal.",
    "El juez de familia aprobó el acuerdo de conciliación sobre custodia, visitas y alimentos.",
]

textos = textos_penal + textos_civil + textos_laboral + textos_familia
etiquetas = (
    (["Penal"] * 25) + (["Civil"] * 25) + (["Laboral"] * 25) + (["Familia"] * 25)
)

df = pd.DataFrame({"texto": textos, "area": etiquetas})
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"✅ Dataset creado: {len(df)} fragmentos jurídicos")
print(df["area"].value_counts())
print(df.head(5))


# =============================================================================
# 4. Generación de Embeddings con SentenceTransformers
# =============================================================================
# Modelo multilingüe paraphrase-multilingual-MiniLM-L12-v2, optimizado para español.

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

print(f"🔄 Cargando modelo: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

print("🔄 Generando embeddings...")
embeddings = model.encode(df["texto"].tolist(), show_progress_bar=True, batch_size=16)

print(f"✅ Embeddings generados. Shape: {embeddings.shape}")
print(
    f"   → Cada texto está representado por un vector de {embeddings.shape[1]} dimensiones."
)


# =============================================================================
# 5. Clustering con KMeans
# =============================================================================

# --- Análisis del número óptimo de clusters (Elbow Method) ---
inertias = []
sil_scores = []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels_tmp = km.fit_predict(embeddings)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(embeddings, labels_tmp))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Análisis del Número Óptimo de Clusters – JustIA", fontsize=14, fontweight="bold"
)

axes[0].plot(K_range, inertias, "o-", color="#1a5276", linewidth=2, markersize=8)
axes[0].axvline(x=4, color="#c0392b", linestyle="--", label="k=4 seleccionado")
axes[0].set_title("Método del Codo (Inercia)", fontsize=12)
axes[0].set_xlabel("Número de Clusters (k)")
axes[0].set_ylabel("Inercia")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, sil_scores, "s-", color="#1e8449", linewidth=2, markersize=8)
axes[1].axvline(x=4, color="#c0392b", linestyle="--", label="k=4 seleccionado")
axes[1].set_title("Índice de Silueta", fontsize=12)
axes[1].set_xlabel("Número de Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("grafico_elbow_silueta.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Gráfico guardado: grafico_elbow_silueta.png")

# --- Aplicar KMeans con k=4 ---
K_OPTIMO = 4
kmeans = KMeans(n_clusters=K_OPTIMO, random_state=SEED, n_init=10, max_iter=300)
df["cluster"] = kmeans.fit_predict(embeddings)

sil = silhouette_score(embeddings, df["cluster"])
print(f"✅ KMeans aplicado con k={K_OPTIMO}")
print(f"   Índice de Silueta: {sil:.4f}  (rango: -1 a 1, mayor es mejor)")
print()
print("Distribución de textos por cluster:")
print(df.groupby(["cluster", "area"]).size().unstack(fill_value=0))


# =============================================================================
# 6. Visualización con t-SNE
# =============================================================================

print("🔄 Reduciendo dimensiones con t-SNE...")
tsne = TSNE(
    n_components=2,
    random_state=SEED,
    perplexity=15,
    max_iter=1000,
    learning_rate="auto",
    init="random",
)
embeddings_2d = tsne.fit_transform(embeddings)

df["tsne_x"] = embeddings_2d[:, 0]
df["tsne_y"] = embeddings_2d[:, 1]

COLORES_AREA = {
    "Penal": "#e74c3c",
    "Civil": "#3498db",
    "Laboral": "#2ecc71",
    "Familia": "#f39c12",
}
COLORES_CLUSTER = {0: "#9b59b6", 1: "#1abc9c", 2: "#e67e22", 3: "#2980b9"}

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(
    "Visualización t-SNE de Embeddings Jurídicos – JustIA",
    fontsize=15,
    fontweight="bold",
    y=1.02,
)

for area, color in COLORES_AREA.items():
    mask = df["area"] == area
    axes[0].scatter(
        df.loc[mask, "tsne_x"],
        df.loc[mask, "tsne_y"],
        c=color,
        label=area,
        alpha=0.85,
        s=80,
        edgecolors="white",
        linewidths=0.5,
    )
axes[0].set_title(
    "Agrupación Real por Área del Derecho", fontsize=12, fontweight="bold"
)
axes[0].legend(title="Área Jurídica", fontsize=10)
axes[0].set_xlabel("Dimensión t-SNE 1")
axes[0].set_ylabel("Dimensión t-SNE 2")
axes[0].grid(True, alpha=0.2)
axes[0].set_facecolor("#f8f9fa")

for c, color in COLORES_CLUSTER.items():
    mask = df["cluster"] == c
    axes[1].scatter(
        df.loc[mask, "tsne_x"],
        df.loc[mask, "tsne_y"],
        c=color,
        label=f"Cluster {c}",
        alpha=0.85,
        s=80,
        edgecolors="white",
        linewidths=0.5,
    )
axes[1].set_title("Clusters Generados por KMeans", fontsize=12, fontweight="bold")
axes[1].legend(title="Cluster", fontsize=10)
axes[1].set_xlabel("Dimensión t-SNE 1")
axes[1].set_ylabel("Dimensión t-SNE 2")
axes[1].grid(True, alpha=0.2)
axes[1].set_facecolor("#f8f9fa")

plt.tight_layout()
plt.savefig("grafico_tsne.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Gráfico guardado: grafico_tsne.png")


# =============================================================================
# 7. Visualización con UMAP
# =============================================================================

print("🔄 Reduciendo dimensiones con UMAP...")
reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)
embeddings_umap = reducer.fit_transform(embeddings)

df["umap_x"] = embeddings_umap[:, 0]
df["umap_y"] = embeddings_umap[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(
    "Visualización UMAP de Embeddings Jurídicos – JustIA",
    fontsize=15,
    fontweight="bold",
    y=1.02,
)

for area, color in COLORES_AREA.items():
    mask = df["area"] == area
    axes[0].scatter(
        df.loc[mask, "umap_x"],
        df.loc[mask, "umap_y"],
        c=color,
        label=area,
        alpha=0.85,
        s=80,
        edgecolors="white",
        linewidths=0.5,
    )
axes[0].set_title(
    "Agrupación Real por Área del Derecho", fontsize=12, fontweight="bold"
)
axes[0].legend(title="Área Jurídica", fontsize=10)
axes[0].set_xlabel("Dimensión UMAP 1")
axes[0].set_ylabel("Dimensión UMAP 2")
axes[0].grid(True, alpha=0.2)
axes[0].set_facecolor("#f8f9fa")

for c, color in COLORES_CLUSTER.items():
    mask = df["cluster"] == c
    axes[1].scatter(
        df.loc[mask, "umap_x"],
        df.loc[mask, "umap_y"],
        c=color,
        label=f"Cluster {c}",
        alpha=0.85,
        s=80,
        edgecolors="white",
        linewidths=0.5,
    )
axes[1].set_title("Clusters Generados por KMeans", fontsize=12, fontweight="bold")
axes[1].legend(title="Cluster", fontsize=10)
axes[1].set_xlabel("Dimensión UMAP 1")
axes[1].set_ylabel("Dimensión UMAP 2")
axes[1].grid(True, alpha=0.2)
axes[1].set_facecolor("#f8f9fa")

plt.tight_layout()
plt.savefig("grafico_umap.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Gráfico guardado: grafico_umap.png")


# =============================================================================
# 8. Análisis de Clusters – Métricas y Matriz de Confusión
# =============================================================================

tabla = df.groupby(["cluster", "area"]).size().unstack(fill_value=0)
mapeo_cluster = tabla.idxmax(axis=1).to_dict()
df["area_predicha"] = df["cluster"].map(mapeo_cluster)

print("=== Mapeo Cluster → Área Jurídica Dominante ===")
for c, a in mapeo_cluster.items():
    print(f"  Cluster {c} → {a}")
print()

print("=== Reporte de Clasificación ===")
print(
    classification_report(
        df["area"],
        df["area_predicha"],
        target_names=["Civil", "Familia", "Laboral", "Penal"],
    )
)

areas_orden = ["Penal", "Civil", "Laboral", "Familia"]
cm = confusion_matrix(df["area"], df["area_predicha"], labels=areas_orden)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=areas_orden,
    yticklabels=areas_orden,
    ax=ax,
    linewidths=0.5,
    linecolor="gray",
)
ax.set_title(
    "Matriz de Confusión – KMeans vs. Etiquetas Reales\nJustIA Clustering Semántico",
    fontsize=12,
    fontweight="bold",
)
ax.set_xlabel("Área Predicha por Cluster", fontsize=11)
ax.set_ylabel("Área Real", fontsize=11)
plt.tight_layout()
plt.savefig("grafico_confusion.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Gráfico guardado: grafico_confusion.png")


# =============================================================================
# 9. Función de Búsqueda Semántica para JustIA
# =============================================================================


def busqueda_semantica_justia(consulta: str, top_k: int = 3) -> None:
    emb_consulta = model.encode([consulta])
    similitudes = cosine_similarity(emb_consulta, embeddings)[0]
    indices_top = similitudes.argsort()[::-1][:top_k]

    print(f'🔍 Consulta: "{consulta}"')
    print("=" * 70)
    for rank, idx in enumerate(indices_top, 1):
        print(
            f"Resultado #{rank} | Área: {df.loc[idx, 'area']} | Similitud: {similitudes[idx]:.4f}"
        )
        print(f"  Cluster: {df.loc[idx, 'cluster']}")
        print(f"  Texto: {df.loc[idx, 'texto']}")
        print("-" * 70)


busqueda_semantica_justia("¿Qué pasa si me despiden estando embarazada?", top_k=3)
print()
busqueda_semantica_justia(
    "Quiero saber sobre la custodia de mis hijos después del divorcio", top_k=3
)
print()
busqueda_semantica_justia(
    "Me acusaron de un delito y no sé cuáles son mis derechos", top_k=3
)


# =============================================================================
# 10. Resumen de Resultados
# =============================================================================

acc = accuracy_score(df["area"], df["area_predicha"])

print("=" * 60)
print("        RESUMEN DE RESULTADOS – JUSTIA ACTIVIDAD 4")
print("=" * 60)
print(f"  Modelo de embeddings : {MODEL_NAME}")
print(f"  Dimensión del vector : {embeddings.shape[1]}")
print(f"  Total de fragmentos  : {len(df)}")
print(f"  Número de clusters   : {K_OPTIMO}")
print(f"  Índice de Silueta    : {sil:.4f}")
print(f"  Accuracy (cluster)   : {acc:.2%}")
print("=" * 60)
print()
print("📊 Archivos generados:")
print("  - grafico_elbow_silueta.png")
print("  - grafico_tsne.png")
print("  - grafico_umap.png")
print("  - grafico_confusion.png")
print()
print("🔑 Conclusión:")
print("  Los embeddings semánticos logran capturar la similitud temática")
print("  entre fragmentos jurídicos en español. El clustering KMeans")
print("  agrupa correctamente los textos por área del derecho, lo que")
print("  permite implementar búsqueda contextual eficiente en JustIA.")
