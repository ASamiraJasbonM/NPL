from src.limpiador import LimpiadorTexto
from src.clasificador import ClasificadorTexto


limpiador = LimpiadorTexto()
clasificador = ClasificadorTexto("diccionario_justia.json")


def mostrar_menu():
    print("\n=== JUSTIA - CONSULTORIO VIRTUAL ===")
    print("1. Limpiar y preprocesar un texto")
    print("2. Clasificar un texto legal")
    print("3. Limpiar y clasificar un texto")
    print("4. Cargar y procesar archivo .txt")
    print("5. Salir")
    return input("Seleccione una opción: ")


def opcion_limpiar():
    texto = input("Ingrese el texto a limpiar: ")
    resultado = limpiador.preprocesar(texto)
    print("\nTexto limpio:")
    print(resultado if resultado else "(vacio tras limpieza)")


def opcion_clasificar():
    texto = input("Ingrese el texto legal a clasificar: ")
    cat, score = clasificador.predecir(texto)
    if score == 0:
        print("No se pudo clasificar. Derivar a evaluación humana.")
    else:
        print(f"Clasificación sugerida: {cat} (confianza: {score} coincidencias)")


def opcion_limpiar_y_clasificar():
    texto = input("Ingrese el texto: ")
    limpio = limpiador.preprocesar(texto)
    print(f"\nLimpio: {limpio}")
    cat, score = clasificador.predecir(limpio)
    if score == 0:
        print("No se pudo clasificar.")
    else:
        print(f"Clasificación: {cat} ({score} coincidencias)")


def opcion_procesar_archivo():
    ruta = input("Ruta del archivo .txt: ")
    try:
        resultados = limpiador.limpiar_archivo(ruta)
        print(f"Procesados {len(resultados)} textos.")
        for i, r in enumerate(resultados[:3], 1):
            cat, sc = clasificador.predecir(r)
            print(f"  {i}. [{cat or '?'}] {r[:60]}...")
    except FileNotFoundError:
        print("Archivo no encontrado.")


def main():
    while True:
        opcion = mostrar_menu()
        if opcion == "1":
            opcion_limpiar()
        elif opcion == "2":
            opcion_clasificar()
        elif opcion == "3":
            opcion_limpiar_y_clasificar()
        elif opcion == "4":
            opcion_procesar_archivo()
        elif opcion == "5":
            print("Gracias por usar JustIA. ¡Hasta luego!")
            break
        else:
            print("Opción no válida")


if __name__ == "__main__":
    main()
