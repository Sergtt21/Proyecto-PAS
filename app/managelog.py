import warnings
import sys

def manejo_errores(nivel_warning="ignore"):
    try:
        if nivel_warning in ["ignore", "once", "always", "error", "default"]:
            warnings.filterwarnings(nivel_warning)
            print(f"Configuración de Warnings establecida a: '{nivel_warning}'")
        else:
            print("Nivel de warning no reconocido. Usando 'ignore'.", file=sys.stderr)
            warnings.filterwarnings("ignore")
    except Exception as e:
        print(f"Error al configurar warnings: {e}", file=sys.stderr)