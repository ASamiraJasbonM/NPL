# ================================================================
# CÓDIGO CORREGIDO - Lematización conservadora para español jurídico
# ================================================================

import re
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import unicodedata

nltk.download('punkt')
nltk.download('stopwords')

# 1. Cargar corpus
with open('corpus_original.txt', 'r', encoding='utf-8') as f:
    lineas = f.readlines()

textos_originales = [linea.strip() for linea in lineas if linea.strip()]

# 2. Limpieza básica (SIN lematización agresiva)
def quitar_tildes(texto):
    texto = unicodedata.normalize('NFD', texto)
    texto = re.sub(r'[\u0300-\u036f]', '', texto)
    return unicodedata.normalize('NFC', texto)

def limpiar_texto(texto):
    texto = texto.lower()
    texto = quitar_tildes(texto)
    texto = re.sub(r'[^\w\s]', '', texto)   # elimina puntuación
    texto = re.sub(r'\d+', '', texto)       # elimina números
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# 3. Stopwords (solo las necesarias)
stop_words = set(stopwords.words('spanish'))
# NO agregamos términos jurídicos importantes como "derecho", "ley", "juez"

# 4. Lematización MUY conservadora (solo plurales simples)
def lematizar_conservador(token):
    """Solo convierte plurales regulares a singular"""
    # Solo si el token tiene más de 3 caracteres
    if len(token) <= 3:
        return token
    
    # Plurales terminados en 's' (regla más segura)
    if token.endswith('es') and not token.endswith('ves'):
        return token[:-2]
    elif token.endswith('s') and token.endswith('as') is False and token.endswith('os') is False:
        # Evitar convertir 'casos' → 'caso' (sí queremos), pero no 'las' → 'la'
        if len(token) > 4:
            return token[:-1]
    return token

def preprocesar(texto):
    texto_limpio = limpiar_texto(texto)
    tokens = word_tokenize(texto_limpio, language='spanish')
    tokens = [t for t in tokens if t and t not in stop_words]
    tokens = [lematizar_conservador(t) for t in tokens]
    return ' '.join(tokens)

# 5. Aplicar y guardar
corpus_procesado = []
for texto in textos_originales:
    resultado = preprocesar(texto)
    if resultado:
        corpus_procesado.append(resultado)

with open('corpus_limpio_corregido.json', 'w', encoding='utf-8') as f:
    json.dump(corpus_procesado, f, indent=2, ensure_ascii=False)

print(f"✅ Procesados {len(corpus_procesado)} fragmentos")
print("\n📝 Ejemplo corregido:")
print("Original:", textos_originales[1][:80])
print("Limpio: ", corpus_procesado[0][:80])