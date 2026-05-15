import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from src.limpiador import LimpiadorTexto
from src.config import (
    SEED,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
    MODEL_NAME,
    DATA_DIR,
    OUTPUT_DPI,
)
from src.utils import set_seed
from src.data_loader import cargar_textos_por_categoria
from src.visualizacion import guardar_grafica

# ─── Configuración ──────────────────────────────────────────────────────────

set_seed(SEED)

limpiador = LimpiadorTexto(idioma="spanish")

print("Librerías cargadas correctamente")
print(f"Dispositivo: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# ─── 2. Carga de datos desde archivos .txt ───────────────────────────────────

categorias = cargar_textos_por_categoria(DATA_DIR, list(LABEL2ID.keys()))

registros = []
for etiqueta, textos in categorias.items():
    for t in textos:
        registros.append({"texto": t, "etiqueta": etiqueta})

df = pd.DataFrame(registros).sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"Total registros: {len(df)}")
print(df["etiqueta"].value_counts())

# ─── 3. Preprocesamiento y codificación de etiquetas ────────────────────────

df["label"] = df["etiqueta"].map(LABEL2ID)

# División train / test (80/20)
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=SEED
)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# ─── 4. Carga del tokenizador BETO ──────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(
        batch["texto"],
        truncation=True,
        padding=False,
        max_length=128,
    )


train_df["texto"] = train_df["texto"].apply(limpiador.preprocesar)
test_df["texto"] = test_df["texto"].apply(limpiador.preprocesar)

train_ds = Dataset.from_pandas(train_df[["texto", "label"]])
test_ds = Dataset.from_pandas(test_df[["texto", "label"]])

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

print("Tokenización completada")
print(train_ds)

# ─── 5. Carga del modelo BETO para clasificación ────────────────────────────

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

print(f"Modelo cargado: {MODEL_NAME}")
print(f"   Parámetros totales: {model.num_parameters():,}")

# ─── 6. Fine-tuning con Hugging Face Trainer ────────────────────────────────


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


training_args = TrainingArguments(
    output_dir="./justia_beto_output",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_steps=20,
    seed=SEED,
    report_to="none",
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Iniciando fine-tuning de BETO...")
trainer.train()

# ─── 7. Evaluación final en conjunto de prueba ──────────────────────────────

resultados = trainer.evaluate()
print("=" * 50)
print(f"  Accuracy : {resultados['eval_accuracy']:.4f}")
print(f"  F1 macro : {resultados['eval_f1_macro']:.4f}")
print("=" * 50)

predicciones = trainer.predict(test_ds)
preds_ids = np.argmax(predicciones.predictions, axis=-1)
true_ids = predicciones.label_ids

etiquetas_nombres = [ID2LABEL[i] for i in range(NUM_LABELS)]

print("\nReporte de clasificación por clase:")
print(classification_report(true_ids, preds_ids, target_names=etiquetas_nombres))

# ─── 8. Visualización de la matriz de confusión ─────────────────────────────

cm = confusion_matrix(true_ids, preds_ids)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=etiquetas_nombres,
    yticklabels=etiquetas_nombres,
    ax=ax,
)
ax.set_title("Matriz de Confusión – JustIA BETO", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicción")
ax.set_ylabel("Etiqueta real")
guardar_grafica(fig, "matriz_confusion_justia.png", dpi=OUTPUT_DPI)

# ─── 9. Predicción sobre ejemplos nuevos ────────────────────────────────────

ejemplos_nuevos = [
    "La víctima del conflicto solicitó reparación por los crímenes cometidos por grupos armados.",
    "Se ordenó el pago de los salarios adeudados al trabajador durante el período de incapacidad.",
    "El arrendatario demandó la devolución del depósito por terminación normal del contrato.",
    "Los menores fueron declarados en situación de abandono y se inició proceso de adoptabilidad.",
    "El procesado fue condenado por el delito de tráfico de estupefacientes en zona escolar.",
]

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Predicciones sobre textos nuevos:")
print("-" * 80)
for texto in ejemplos_nuevos:
    inputs = tokenizer(
        texto, return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    prob = torch.softmax(logits, dim=-1)[0]
    pred_id = torch.argmax(prob).item()
    pred_cat = ID2LABEL[pred_id]
    confianza = prob[pred_id].item()
    print(f"Texto    : {texto[:70]}...")
    print(f"Categoria: {pred_cat.upper():10s}  |  Confianza: {confianza:.2%}")
    print("-" * 80)
