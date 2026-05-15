SEED = 42
DATA_DIR = "data"
OUTPUT_DPI = 150
CATEGORIAS_JURIDICAS = ["penal", "civil", "laboral", "familia"]
LABEL2ID = {"civil": 0, "familia": 1, "laboral": 2, "penal": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
