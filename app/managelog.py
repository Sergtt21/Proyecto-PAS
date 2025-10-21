# app/managelog.py
import warnings
import sys

_config_aplicada = False  # <-- guard

def manejo_errores(nivel_warning="ignore", verbose=False):
    global _config_aplicada
    if _config_aplicada:
        return
    try:
        if nivel_warning in ["ignore", "once", "always", "error", "default"]:
            warnings.filterwarnings(nivel_warning)
            if verbose:
                print(f"Configuración de Warnings establecida a: '{nivel_warning}'")
        else:
            print("Nivel de warning no reconocido. Usando 'ignore'.", file=sys.stderr)
            warnings.filterwarnings("ignore")
    except Exception as e:
        print(f"Error al configurar warnings: {e}", file=sys.stderr)
    finally:
        _config_aplicada = True