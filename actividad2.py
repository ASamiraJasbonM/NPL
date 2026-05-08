import json

def cargar_diccionario(ruta):
    with open(ruta, 'r', encoding='utf-8') as f:
        return json.load(f)

def predecir_categoria(texto, diccionario):
    texto_limpio = texto.lower()
    mejor_categoria = None
    max_puntaje = 0

    for cat, palabras in diccionario.items():
        puntaje = sum(1 for p in palabras if p in texto_limpio)
        if puntaje > max_puntaje:
            max_puntaje = puntaje
            mejor_categoria = cat

    return mejor_categoria, max_puntaje

# Ejemplo
dic = cargar_diccionario('diccionario_justia.json')
texto_legal = "El padre solicita la custodia de su hija y una pensión de alimentos"
print(predecir_categoria(texto_legal, dic))  # ('familia', 2)