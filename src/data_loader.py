import os


def cargar_textos_por_categoria(data_dir, categorias):
    datos = {}
    for cat in categorias:
        ruta = os.path.join(data_dir, f"{cat}.txt")
        with open(ruta, encoding="utf-8") as f:
            textos = [linea.strip() for linea in f if linea.strip()]
        datos[cat] = textos
    return datos


def cargar_textos_por_documento(data_dir, prefijo="texto_", sufijo=".txt"):
    archivos = sorted(
        f for f in os.listdir(data_dir) if f.startswith(prefijo) and f.endswith(sufijo)
    )
    textos = []
    for archivo in archivos:
        ruta = os.path.join(data_dir, archivo)
        with open(ruta, "r", encoding="utf-8") as f:
            textos.append(f.read().strip())
    return textos
