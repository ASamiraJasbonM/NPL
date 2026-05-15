import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def exportar_csv(df, nombre, index=False, encoding="utf-8-sig"):
    df.to_csv(nombre, index=index, encoding=encoding)
    print(f"CSV exportado: {nombre} ({len(df)} filas)")
