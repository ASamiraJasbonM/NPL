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

def mostrar_menu():
    print("\n=== JUSTIA - CONSULTORIO VIRTUAL ===")
    print("1. Ingresar una pregunta legal")
    print("2. Cargar documento (simulado)")
    print("3. Clasificar caso por texto")
    print("4. Salir")
    return input("Seleccione una opción: ")

def respuesta_simulada(pregunta):
    return f"[SIMULACIÓN] Análisis preliminar para: '{pregunta}'. Se recomienda derivar al área correspondiente."

def cargar_documento_simulado():
    print("(Simulación) Se ha recibido un archivo: caso_123.pdf")
    return "Texto simulado del documento"

def clasificar_simulada(texto):
    dic = cargar_diccionario('diccionario_justia.json')
    cat, score = predecir_categoria(texto, dic)
    if score == 0:
        return "No se pudo clasificar. Derivar a evaluación humana."
    return f"Clasificación sugerida: {cat} (confianza: {score} coincidencias)"

def main():
    while True:
        opcion = mostrar_menu()
        if opcion == '1':
            pregunta = input("Escriba su pregunta legal: ")
            print(respuesta_simulada(pregunta))
        elif opcion == '2':
            texto_doc = cargar_documento_simulado()
            print("Documento cargado (simulación)")
        elif opcion == '3':
            texto = input("Ingrese el texto del caso a clasificar: ")
            print(clasificar_simulada(texto))
        elif opcion == '4':
            print("Gracias por usar JustIA. ¡Hasta luego!")
            break
        else:
            print("Opción no válida")

if __name__ == "__main__":
    main()