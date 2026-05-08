import json


class ClasificadorTexto:
    def __init__(self, ruta_diccionario=None):
        self.diccionario = {}
        if ruta_diccionario:
            self.cargar_diccionario(ruta_diccionario)

    def cargar_diccionario(self, ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            self.diccionario = json.load(f)
        return self

    def predecir(self, texto):
        texto_limpio = texto.lower()
        mejor_categoria = None
        max_puntaje = 0

        for cat, palabras in self.diccionario.items():
            puntaje = sum(1 for p in palabras if p in texto_limpio)
            if puntaje > max_puntaje:
                max_puntaje = puntaje
                mejor_categoria = cat

        return mejor_categoria, max_puntaje

    def predecir_muchos(self, textos):
        return [self.predecir(t) for t in textos]
