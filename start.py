import os
import sys
import subprocess

# 1. Crear venv si no existe
venv_path = ".venv"
if not os.path.exists(venv_path):
    print(f"Creando entorno virtual con {sys.version.split()[0]}...")
    subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)

# 2. Activación (solo mensaje informativo)
activate_script = os.path.join(venv_path, "Scripts", "activate")
print(f"Para activar el entorno, ejecutar:\n"
      f"Git Bash: source {activate_script}\n"
      f"PowerShell: .\\.venv\\Scripts\\Activate.ps1")

# 3. Instalar dependencias
pip_executable = os.path.join(venv_path, "Scripts", "pip")
print("Instalando dependencias...")
subprocess.run([pip_executable, "install", "-r", "requirements.txt"], check=True)
