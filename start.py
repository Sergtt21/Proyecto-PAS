import os
import sys
import subprocess
import logging
from dotenv import load_dotenv

#configuracion de logs
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/startup.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.info("Inicio del script de automatizacion")

#crear entorno virtual si no existe
venv_path = ".venv"
if not os.path.exists(venv_path):
    print(f"Creando entorno virtual con {sys.version.split()[0]}...")
    subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
else:
    print("Entorno virtual encontrado.")

#instalar dependencias solo si no estan instaladas
pip_executable = os.path.join(venv_path, "Scripts", "pip.exe") if os.name == "nt" else os.path.join(venv_path, "bin", "pip")

try:
    import cv2  # OpenCV
    import numpy
    import mediapipe
except ImportError:
    print("Instalando dependencias---------")
    subprocess.run([pip_executable, "install", "-r", "requirements.txt"], check=True)
    logging.info("Dependencias instaladas correctamente")
else:
    print("Dependencias ya instaladas")
    logging.info("las dependencias ya estaban instaladas")

#verificar archivo .env
print("Verificando elarchivo .env...")
load_dotenv()

required_vars = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_DB",
    "BUS_QUEUE",
    "DEMO_MODE",
    "DEBUG"
]

missing = [v for v in required_vars if not os.getenv(v)]
if missing:
    print(f"Faltan variables en .env: {', '.join(missing)}")
    logging.error(f"Faltan variables en .env: {', '.join(missing)}")
    sys.exit(1)
else:
    print("Todas las variables .env encontradas")
    logging.info("Archivo .env verificado correctamente")

python_exec = os.path.join(venv_path, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv_path, "bin", "python")

try:
    logging.info("Ejecutando app.main...")
    subprocess.run([python_exec, "-m", "app.main"], check=True)
    logging.info("Sistema ejecutado correctamente.")
except subprocess.CalledProcessError as e:
    print("Error al ejecutar main.py. Revisa logs/startup.log")
    logging.error(f"Error al ejecutar main.py: {e}")
except KeyboardInterrupt:
    print("\nEjecucion interrumpida por el usuario.")
    logging.warning("Ejecución interrumpida por el usuario.")
finally:
    print("Proceso finalizado.")
    logging.info("Proceso finalizado correctamente.")

#mensajes de activacion del entorno virtual
activate_script = os.path.join(venv_path, "Scripts", "activate") if os.name == "nt" else os.path.join(venv_path, "bin", "activate")
print(f"\nPara activar el entorno virtual manualmente:")
if os.name == "nt":
    print(f"PowerShell: .\\{venv_path}\\Scripts\\Activate.ps1")
else:
    print(f"Shell: source {activate_script}")
