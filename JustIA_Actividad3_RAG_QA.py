# JustIA – Actividad 3
# Sistema de Preguntas y Respuestas Jurídicas con RAG (Retrieval-Augmented Generation)
# **Corporación Universitaria de Asturias**
#
# Objetivo: Construir un prototipo de sistema de preguntas y respuestas jurídicas
# automatizadas basado en documentos cargados, como modelo base para JustIA.
#
# Arquitectura RAG:
#   Pregunta del usuario
#        |
#        v
#   [Embeddings]  -->  Búsqueda semántica en base de conocimiento
#        |
#        v
#   [Contexto recuperado]  -->  Modelo generativo (LLM)
#        |
#        v
#   Respuesta con fuente citada
# -----------------------------------------------------------------------------

# =============================================================================
# 0. Instalación de dependencias (ejecutar una sola vez)
# =============================================================================
# !pip install sentence-transformers faiss-cpu transformers torch --quiet
print("✅ Dependencias instaladas")


# =============================================================================
# 1. Importaciones
# =============================================================================
import subprocess
import sys
import numpy as np
import textwrap
import warnings
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings("ignore")

print(f"✅ Librerías cargadas")
print(f"🖥️  Dispositivo: {'GPU' if torch.cuda.is_available() else 'CPU'}")


# =============================================================================
# 2. Base de conocimiento jurídica (10 documentos colombianos)
# =============================================================================
# Cada documento representa un fragmento clave de legislación, guías o sentencias.
# En producción estos textos se cargarían desde PDFs reales.

base_conocimiento = [
    {
        "id": "DOC-01",
        "titulo": "Ley 1257 de 2008 – No violencia contra la mujer",
        "contenido": """
        La Ley 1257 de 2008 tiene por objeto la adopción de normas que permitan garantizar para todas
        las mujeres una vida libre de violencia. Define la violencia contra la mujer como cualquier
        acción u omisión que le cause muerte, daño o sufrimiento físico, sexual, psicológico, económico
        o patrimonial. Establece medidas de sensibilización, prevención y sanción.
        Las víctimas tienen derecho a recibir atención médica, psicológica, jurídica y social de forma
        gratuita. El empleador que despida a una mujer víctima de violencia debe pagar una indemnización
        especial de 180 días de salario. La ley ordena al Estado adoptar medidas de protección como:
        desalojo del agresor del hogar, prohibición de acercamiento, y suspensión de la tenencia de armas.
        El artículo 17 establece medidas de atención para las víctimas en salud, incluyendo
        rehabilitación física y mental. Las entidades responsables incluyen el ICBF, la Fiscalía,
        las Comisarías de Familia y el sistema de salud.
        """,
    },
    {
        "id": "DOC-02",
        "titulo": "Ley 1010 de 2006 – Acoso laboral",
        "contenido": """
        La Ley 1010 de 2006 define el acoso laboral como toda conducta persistente y demostrable
        ejercida sobre un empleado por parte de un empleador, un jefe o superior jerárquico inmediato
        o mediato, con el fin de infundir miedo, intimidación, terror y angustia, o causar perjuicio
        laboral o desmotivar al trabajador. Las modalidades de acoso laboral son: maltrato laboral,
        persecución laboral, discriminación laboral, entorpecimiento laboral, inequidad y desprotección laboral.
        El trabajador víctima puede denunciar ante el Inspector del Trabajo o ante el Comité de
        Convivencia Laboral de la empresa. La empresa tiene la obligación de conformar el Comité de
        Convivencia Laboral. Las sanciones para el acosador van de 2 a 10 salarios mínimos mensuales
        vigentes. Si el acoso genera despido con justa causa, el trabajador tiene derecho a indemnización.
        La víctima puede solicitar el traslado o la terminación del contrato con justa causa imputable
        al empleador y exigir indemnización plena de perjuicios.
        """,
    },
    {
        "id": "DOC-03",
        "titulo": "Ley 1448 de 2011 – Ley de Víctimas y Restitución de Tierras",
        "contenido": """
        La Ley 1448 de 2011 establece medidas de atención, asistencia y reparación integral
        a las víctimas del conflicto armado interno en Colombia. Se consideran víctimas las personas
        que hayan sufrido daños por hechos ocurridos a partir del 1 de enero de 1985 como consecuencia
        de infracciones al Derecho Internacional Humanitario o violaciones graves a los derechos humanos.
        Los componentes de la reparación integral son: restitución, indemnización, rehabilitación,
        satisfacción y garantías de no repetición. La indemnización administrativa varía según el hecho
        victimizante: homicidio o desaparición forzada (40 SMLMV), desplazamiento forzado (17 SMLMV),
        secuestro (40 SMLMV), lesiones que causen incapacidad permanente (40 SMLMV).
        La Unidad para la Atención y Reparación Integral a las Víctimas (UARIV) es la entidad
        encargada de coordinar el Sistema Nacional de Atención y Reparación Integral a las Víctimas.
        El Registro Único de Víctimas (RUV) es el mecanismo oficial de inscripción.
        """,
    },
    {
        "id": "DOC-04",
        "titulo": "Ley 100 de 1993 – Sistema General de Seguridad Social",
        "contenido": """
        La Ley 100 de 1993 crea el Sistema de Seguridad Social Integral en Colombia compuesto por:
        el Sistema General de Pensiones, el Sistema General de Seguridad Social en Salud y
        el Sistema General de Riesgos Laborales. Para acceder a la pensión de vejez en el régimen
        de prima media (Colpensiones) se requieren 1.300 semanas de cotización y tener 62 años si
        es hombre o 57 años si es mujer. La pensión de invalidez se reconoce cuando la persona
        pierde el 50% o más de su capacidad laboral. La pensión de sobrevivientes beneficia al
        cónyuge, compañero permanente e hijos del afiliado fallecido. El monto mínimo de cualquier
        pensión es de un salario mínimo legal mensual vigente. Colpensiones administra el régimen
        de prima media con prestación definida. Los fondos privados de pensiones operan el régimen
        de ahorro individual con solidaridad (RAIS).
        """,
    },
    {
        "id": "DOC-05",
        "titulo": "Código Sustantivo del Trabajo – Derechos laborales básicos",
        "contenido": """
        El Código Sustantivo del Trabajo regula las relaciones laborales en Colombia. La jornada
        máxima ordinaria es de 8 horas diarias y 46 horas semanales. Las horas extras diurnas
        se remuneran con un recargo del 25% sobre el valor ordinario. Las horas extras nocturnas
        tienen un recargo del 75%. El trabajo nocturno (entre 9 pm y 6 am) tiene un recargo del 35%.
        Las prestaciones sociales obligatorias son: cesantías (un mes de salario por año trabajado),
        intereses sobre cesantías (12% anual), prima de servicios (15 días de salario en junio y
        15 días en diciembre), y vacaciones (15 días hábiles por año). El empleador debe afiliar
        al trabajador a salud, pensión y riesgos laborales desde el primer día. El despido sin justa
        causa genera derecho a indemnización: para contratos a término indefinido, 30 días de salario
        por el primer año y 20 días adicionales por cada año subsiguiente.
        """,
    },
    {
        "id": "DOC-06",
        "titulo": "Ley 1098 de 2006 – Código de Infancia y Adolescencia",
        "contenido": """
        El Código de Infancia y Adolescencia establece normas sustantivas y procesales para la
        protección integral de los niños, niñas y adolescentes en Colombia. El interés superior
        del menor es el principio rector de todas las decisiones judiciales y administrativas.
        Se prohíbe el trabajo de menores de 15 años. Los adolescentes entre 15 y 17 años pueden
        trabajar máximo 6 horas diarias con autorización del Inspector de Trabajo. El ICBF es
        la entidad encargada de garantizar los derechos de los menores. Las medidas de restablecimiento
        de derechos incluyen: amonestación a los padres, retiro del hogar, ubicación en familia de
        acogida, y adopción. La adopción es irrevocable y confiere al adoptado la condición de hijo
        con todos los derechos. En materia penal, los adolescentes entre 14 y 17 años son imputables
        ante el Sistema de Responsabilidad Penal para Adolescentes (SRPA).
        """,
    },
    {
        "id": "DOC-07",
        "titulo": "Derecho de petición y tutela – Mecanismos de protección",
        "contenido": """
        El derecho de petición está consagrado en el artículo 23 de la Constitución Política y
        regulado por la Ley 1755 de 2015. Toda persona tiene derecho a formular peticiones respetuosas
        ante autoridades o particulares que presten servicios públicos. El término de respuesta es
        de 15 días hábiles para peticiones generales, 10 días para peticiones de información y
        30 días para peticiones de consulta. La acción de tutela está consagrada en el artículo 86
        de la Constitución. Protege los derechos fundamentales cuando son vulnerados o amenazados
        por la acción u omisión de cualquier autoridad pública o particular. El juez debe fallar
        en 10 días. Si el fallo es favorable, la entidad debe cumplir en 48 horas. La tutela
        procede cuando no existe otro mecanismo de defensa judicial, salvo como mecanismo transitorio
        para evitar un perjuicio irremediable. La impugnación del fallo de tutela debe presentarse
        dentro de los 3 días siguientes a la notificación.
        """,
    },
    {
        "id": "DOC-08",
        "titulo": "Decreto Ley 4633 de 2011 – Víctimas pueblos indígenas",
        "contenido": """
        El Decreto Ley 4633 de 2011 adopta medidas de asistencia, atención, reparación integral
        y restitución de derechos territoriales a las víctimas pertenecientes a los pueblos y
        comunidades indígenas. Reconoce que el territorio es víctima cuando ha sido afectado por
        el conflicto armado. La reparación colectiva busca restablecer las condiciones que permitan
        mantener la cohesión social, cultural, la autonomía y la gobernabilidad de las comunidades.
        Los resguardos y territorios indígenas afectados tienen derecho a consulta previa libre
        e informada sobre todas las medidas de reparación. La UARIV debe coordinar con la Autoridad
        Nacional de Consulta Previa. Las comunidades indígenas tienen derecho a retornar colectivamente
        a sus territorios con garantías de seguridad. Se reconoce el derecho propio (derecho indígena)
        y la jurisdicción especial indígena para resolver conflictos internos.
        """,
    },
    {
        "id": "DOC-09",
        "titulo": "Ley 820 de 2003 – Arrendamiento de vivienda urbana",
        "contenido": """
        La Ley 820 de 2003 regula los contratos de arrendamiento de vivienda urbana en Colombia.
        El canon de arrendamiento no puede exceder el 1% del valor comercial del inmueble por mes.
        El incremento anual del canon no puede superar el IPC del año anterior. El contrato puede
        ser verbal o escrito. El arrendatario tiene derecho a recibir el inmueble en buen estado
        y a que se respete su tranquilidad. El arrendador puede dar por terminado el contrato con
        preaviso de 3 meses cuando requiere el inmueble para uso propio, o cuando el arrendatario
        incumple el pago por más de dos meses. El arrendatario en mora puede ser demandado en
        proceso de restitución de inmueble arrendado ante el juez civil municipal. La demanda
        procede cuando hay mora de dos o más cánones de arrendamiento.
        """,
    },
    {
        "id": "DOC-10",
        "titulo": "Guía de acceso a la justicia para población migrante en Colombia",
        "contenido": """
        Los migrantes en Colombia tienen derecho de acceso a la justicia independientemente de
        su estatus migratorio. El Permiso por Protección Temporal (PPT) otorga a los venezolanos
        regularizados acceso a servicios de salud, educación y trabajo. Los migrantes pueden
        interponer tutela para proteger sus derechos fundamentales. En materia laboral, los
        migrantes con permiso de trabajo tienen los mismos derechos que los nacionales. Los
        migrantes víctimas de trata de personas pueden acudir a la Fiscalía General y al
        Ministerio del Interior. Migración Colombia es la autoridad competente en asuntos
        migratorios. El Estatuto Temporal de Protección para Migrantes Venezolanos (ETPV)
        fue adoptado mediante el Decreto 216 de 2021. Los niños migrantes tienen derecho
        incondicional a la educación y a la salud. La Defensoría del Pueblo ofrece orientación
        jurídica gratuita a población migrante en situación de vulnerabilidad.
        """,
    },
]

print(f"✅ Base de conocimiento cargada: {len(base_conocimiento)} documentos")
for doc in base_conocimiento:
    print(f"  [{doc['id']}] {doc['titulo']}")


# =============================================================================
# 3. Generación de embeddings con SentenceTransformers
# =============================================================================
# Usamos paraphrase-multilingual-MiniLM-L12-v2, modelo multilingüe optimizado para español.

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

print(f"⏳ Cargando modelo de embeddings: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)

# Extraer textos y metadatos
textos_docs = [doc["contenido"].strip() for doc in base_conocimiento]
ids_docs = [doc["id"] for doc in base_conocimiento]
titulos_docs = [doc["titulo"] for doc in base_conocimiento]

print("⏳ Generando embeddings del corpus...")
embeddings = embedder.encode(
    textos_docs,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=True,
)

print(f"\n✅ Embeddings generados: {embeddings.shape}")
print(f"   Dimensión por documento: {embeddings.shape[1]}")


# =============================================================================
# 4. Construcción del índice vectorial con FAISS
# =============================================================================

DIM = embeddings.shape[1]

indice_faiss = faiss.IndexFlatIP(DIM)
indice_faiss.add(embeddings.astype(np.float32))

print(f"✅ Índice FAISS construido")
print(f"   Vectores indexados : {indice_faiss.ntotal}")
print(f"   Dimensión          : {DIM}")


# =============================================================================
# 5. Carga del modelo generativo (Reader)
# =============================================================================
# Usamos mrm8488/bert2bert_shared-spanish-finetuned-summarization, modelo seq2seq en
# español disponible en Hugging Face. En producción puede reemplazarse por un LLM más
# potente (GPT-4, Claude, Llama-3).

GEN_MODEL = "mrm8488/bert2bert_shared-spanish-finetuned-summarization"

print(f"⏳ Cargando modelo generativo: {GEN_MODEL}")
tokenizer_gen = AutoTokenizer.from_pretrained(GEN_MODEL)
model_gen = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)


def generar_con_modelo(prompt: str) -> str:
    inputs = tokenizer_gen(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model_gen.generate(
            inputs.input_ids,
            max_new_tokens=200,
            min_length=60,
            do_sample=False,
        )
    return tokenizer_gen.decode(outputs[0], skip_special_tokens=True)


print("✅ Modelo generativo listo")


# =============================================================================
# 6. Pipeline RAG completo: Retrieve -> Augment -> Generate
# =============================================================================


def buscar_documentos(pregunta: str, top_k: int = 3) -> list:
    q_embedding = embedder.encode(
        [pregunta],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = indice_faiss.search(q_embedding, top_k)

    resultados = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            resultados.append(
                {
                    "id": ids_docs[idx],
                    "titulo": titulos_docs[idx],
                    "contenido": textos_docs[idx],
                    "score": float(score),
                }
            )
    return resultados


def generar_respuesta(pregunta: str, contexto: str) -> str:
    prompt = (
        f"Pregunta: {pregunta}\n\n"
        f"Información relevante:\n{contexto}\n\n"
        "Respuesta resumida basada en la información anterior:"
    )
    tokens = tokenizer_gen(
        prompt,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    prompt_truncado = tokenizer_gen.decode(
        tokens["input_ids"][0], skip_special_tokens=True
    )
    salida = generar_con_modelo(prompt_truncado)
    return salida


def justia_qa(pregunta: str, top_k: int = 2, umbral_score: float = 0.25) -> dict:
    # -- PASO 1: Retrieve --
    docs_recuperados = buscar_documentos(pregunta, top_k=top_k)
    docs_relevantes = [d for d in docs_recuperados if d["score"] >= umbral_score]

    if not docs_relevantes:
        return {
            "pregunta": pregunta,
            "respuesta": "No se encontró información suficientemente relevante en la base de conocimiento. Por favor consulte directamente con un abogado del consultorio.",
            "fuentes": [],
            "scores": [],
        }

    # -- PASO 2: Augment --
    contexto = "\n\n".join(
        f"[{d['id']}] {d['titulo']}:\n{d['contenido'][:600]}" for d in docs_relevantes
    )

    # -- PASO 3: Generate --
    respuesta = generar_respuesta(pregunta, contexto)

    return {
        "pregunta": pregunta,
        "respuesta": respuesta,
        "fuentes": [d["titulo"] for d in docs_relevantes],
        "ids": [d["id"] for d in docs_relevantes],
        "scores": [round(d["score"], 4) for d in docs_relevantes],
    }


def mostrar_respuesta(resultado: dict):
    linea = "=" * 68
    print(f"\n{linea}")
    print(f" SISTEMA JustIA – Respuesta")
    print(f"{linea}")
    print(f"\nPregunta:\n   {resultado['pregunta']}")
    print(f"\nRespuesta preliminar:")
    for linea_resp in textwrap.wrap(resultado["respuesta"], width=66):
        print(f"   {linea_resp}")
    if resultado.get("fuentes"):
        print(f"\nFuentes consultadas:")
        for fid, ftitulo, fscore in zip(
            resultado["ids"], resultado["fuentes"], resultado["scores"]
        ):
            print(f"   [{fid}] {ftitulo}")
            print(f"          Similitud: {fscore:.4f}")
    print(f"\nAVISO: Esta respuesta es orientativa. Consulte siempre")
    print(f"   con un abogado del consultorio para su caso especifico.")
    print(f"{linea}\n")


print("✅ Pipeline RAG de JustIA definido y listo")


# =============================================================================
# 7. Pruebas con preguntas jurídicas reales
# =============================================================================

# -- PREGUNTA 1: Acoso laboral --
resultado1 = justia_qa("¿Qué derechos tengo si soy víctima de acoso laboral?")
mostrar_respuesta(resultado1)

# -- PREGUNTA 2: Violencia contra la mujer --
resultado2 = justia_qa(
    "¿A qué medidas de protección tiene derecho una mujer víctima de violencia doméstica?"
)
mostrar_respuesta(resultado2)

# -- PREGUNTA 3: Pensión de vejez --
resultado3 = justia_qa(
    "¿Cuántas semanas debo cotizar para obtener la pensión de vejez en Colombia?"
)
mostrar_respuesta(resultado3)

# -- PREGUNTA 4: Víctimas del conflicto --
resultado4 = justia_qa("Fui desplazado forzosamente, ¿a qué reparación tengo derecho?")
mostrar_respuesta(resultado4)

# -- PREGUNTA 5: Migrantes --
resultado5 = justia_qa(
    "Soy migrante venezolano, ¿tengo derechos laborales en Colombia?"
)
mostrar_respuesta(resultado5)

# -- PREGUNTA 6: Derecho fuera de la base de conocimiento --
resultado6 = justia_qa(
    "¿Cuáles son los requisitos para constituir una sociedad anónima simplificada?",
    umbral_score=0.4,
)
mostrar_respuesta(resultado6)


# =============================================================================
# 8. Comparación de similitudes: tabla de scoring
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt

preguntas_test = [
    "¿Qué derechos tengo si soy víctima de acoso laboral?",
    "¿A qué medidas de protección tiene derecho una mujer víctima de violencia doméstica?",
    "¿Cuántas semanas debo cotizar para obtener la pensión de vejez?",
    "Fui desplazado forzosamente, ¿a qué reparación tengo derecho?",
    "Soy migrante venezolano, ¿tengo derechos laborales en Colombia?",
    "¿Qué es el derecho de petición y en cuánto tiempo deben responderme?",
    "¿Cuáles son las prestaciones sociales que debe pagarme mi empleador?",
    "¿Puedo interponer tutela si no me responden una petición?",
]

filas = []
for preg in preguntas_test:
    docs = buscar_documentos(preg, top_k=1)
    filas.append(
        {
            "Pregunta": preg[:55] + "...",
            "Documento top-1": docs[0]["id"],
            "Título": docs[0]["titulo"][:45] + "...",
            "Score similitud": round(docs[0]["score"], 4),
        }
    )

df_scores = pd.DataFrame(filas)
print("\nTabla de recuperación semántica:")
print(df_scores.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4))
colores = [
    "#2ECC71" if s >= 0.4 else "#F39C12" if s >= 0.25 else "#E74C3C"
    for s in df_scores["Score similitud"]
]
bars = ax.barh(
    range(len(df_scores)), df_scores["Score similitud"], color=colores, edgecolor="#555"
)
ax.set_yticks(range(len(df_scores)))
ax.set_yticklabels([f"P{i + 1}" for i in range(len(df_scores))], fontsize=10)
ax.set_xlabel("Score de similitud coseno")
ax.set_title("JustIA RAG – Score de recuperación por pregunta", fontweight="bold")
ax.axvline(0.25, color="orange", linestyle="--", label="Umbral mínimo (0.25)")
ax.axvline(0.40, color="green", linestyle="--", label="Umbral alto (0.40)")
ax.legend()
for bar, v in zip(bars, df_scores["Score similitud"]):
    ax.text(
        bar.get_width() + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{v:.4f}",
        va="center",
        fontsize=9,
    )
plt.tight_layout()
plt.savefig("rag_scores_justia.png", dpi=150)
plt.show()
print("Imagen guardada: rag_scores_justia.png")


# =============================================================================
# 9. Modo interactivo: descomenta para ingresar tu propia pregunta
# =============================================================================

# pregunta_usuario = input("\nIngresa tu pregunta jurídica: ")
# resultado = justia_qa(pregunta_usuario)
# mostrar_respuesta(resultado)

pregunta_demo = "¿Qué derechos tienen los niños migrantes en Colombia?"
resultado_demo = justia_qa(pregunta_demo)
mostrar_respuesta(resultado_demo)


# =============================================================================
# 10. Observaciones técnicas y reflexión crítica para JustIA
# =============================================================================
#
# Arquitectura implementada
# -------------------------
# Componente          | Tecnología                                  | Función
# Embeddings          | paraphrase-multilingual-MiniLM-L12-v2       | Representación semántica
# Indice vectorial    | FAISS IndexFlatIP                           | Busqueda de similitud coseno
# Modelo generativo   | bert2bert-spanish-summarization             | Generación de respuesta
# Umbral relevancia   | Score >= 0.25                               | Filtro de respuestas sin sustento
#
# Ventajas del enfoque RAG sobre LLM puro:
# 1. Trazabilidad: Cada respuesta cita explícitamente la fuente.
# 2. Actualización sin reentrenamiento: solo indexar nuevo documento.
# 3. Control del dominio: reduce alucinaciones.
# 4. Escalabilidad: FAISS soporta millones de vectores.
#
# Limitaciones y riesgos eticos:
# 1. Alucinaciones residuales: el umbral y citas mitigan el riesgo.
# 2. Desactualización normativa: puede citar normas derogadas.
# 3. Poblacion vulnerable: incluir derivación a profesional humano.
# 4. Idiomas y dialectos: comunidades indigenas pueden no recibir respuestas adecuadas.
# 5. Confidencialidad: cumplir Ley 1581 de 2012 (Habeas Data).
#
# Recomendaciones para producción:
# - Usar LLM con instrucción en espanol (ej. Llama-3-8B-Instruct).
# - Retroalimentacion del practicante (pulgar arriba/abajo).
# - Limite de confianza: score < 0.25 deriva a abogado.
# - Registro de auditoria completo.
# - Cumplir Ley 1581 de 2012 para datos personales.
#
# ---
# Proyecto JustIA – Corporación Universitaria de Asturias | Caso Práctico Unidad 2
