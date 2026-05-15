import matplotlib.pyplot as plt


def guardar_grafica(fig, nombre, dpi=150, bbox_inches="tight"):
    fig.tight_layout()
    fig.savefig(nombre, dpi=dpi, bbox_inches=bbox_inches)
    plt.show()
    print(f"Imagen guardada: {nombre}")
