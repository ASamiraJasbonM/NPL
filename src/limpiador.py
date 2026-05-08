import re
import json
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class LimpiadorTexto:
    def __init__(self, idioma="spanish", descargar_recursos=True):
        self.idioma = idioma
        if descargar_recursos:
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
        self.stop_words = set(stopwords.words(idioma))

    def quitar_tildes(self, texto):
        texto = unicodedata.normalize("NFD", texto)
        texto = re.sub(r"[\u0300-\u036f]", "", texto)
        return unicodedata.normalize("NFC", texto)

    def limpiar_texto(self, texto):
        texto = texto.lower()
        texto = self.quitar_tildes(texto)
        texto = re.sub(r"[^\w\s]", "", texto)
        texto = re.sub(r"\d+", "", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto

    def lematizar_conservador(self, token):
        if len(token) <= 3:
            return token
        if token.endswith("es") and not token.endswith("ves"):
            return token[:-2]
        elif (
            token.endswith("s")
            and not token.endswith("as")
            and not token.endswith("os")
        ):
            if len(token) > 4:
                return token[:-1]
        return token

    def preprocesar(self, texto):
        texto_limpio = self.limpiar_texto(texto)
        tokens = word_tokenize(texto_limpio, language=self.idioma)
        tokens = [t for t in tokens if t and t not in self.stop_words]
        tokens = [self.lematizar_conservador(t) for t in tokens]
        return " ".join(tokens)

    def limpiar_textos(self, textos):
        return [self.preprocesar(t) for t in textos if self.preprocesar(t)]

    def limpiar_archivo(self, ruta_entrada, ruta_salida=None):
        with open(ruta_entrada, "r", encoding="utf-8") as f:
            lineas = f.readlines()
        textos = [linea.strip() for linea in lineas if linea.strip()]
        resultados = self.limpiar_textos(textos)
        if ruta_salida:
            with open(ruta_salida, "w", encoding="utf-8") as f:
                json.dump(resultados, f, indent=2, ensure_ascii=False)
        return resultados

    def limpiar_json(self, ruta_entrada, ruta_salida=None):
        with open(ruta_entrada, "r", encoding="utf-8") as f:
            datos = json.load(f)
        if isinstance(datos, list):
            textos = [str(item) for item in datos]
        elif isinstance(datos, dict):
            textos = [str(v) for v in datos.values()]
        else:
            textos = [str(datos)]
        resultados = self.limpiar_textos(textos)
        if ruta_salida:
            with open(ruta_salida, "w", encoding="utf-8") as f:
                json.dump(resultados, f, indent=2, ensure_ascii=False)
        return resultados
