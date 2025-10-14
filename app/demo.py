import sys
import time
import os
from dotenv import load_dotenv

load_dotenv()
#El archivo .env contiene DEMO_MODE
MODO_PRUEBA = os.getenv("DEMO_MODE") == "True"

if MODO_PRUEBA:
    print("Modo prueba activado, camara desactivada")
else:
    print("DEMO_MODE no esta en True")
    sys.exit(1)
ESTADO_IDLE = 0
ESTADO_DETECTADO = 1
ESTADO_CONFIRMADO = 2

ESTADO_ACTUAL = ESTADO_IDLE
GESTO_ACTUAL = None
ULTIMO_GESTO = time.time()
INTERVALO_GESTO = 5.0

SECUENCIA_GESTOS = ["CEJAS ARRIBA", "SONRISA", "BOCA ABIERTA", "PARPADEO"]
INDICE_GESTO = 0

def simular_gesto():
    global INDICE_GESTO
    
    gesto = SECUENCIA_GESTOS[INDICE_GESTO]
    
    INDICE_GESTO = (INDICE_GESTO + 1) % len(SECUENCIA_GESTOS)
    
    return gesto

def enviar_gesto(gesto):
    print(f"El gesto valido ha sido enviado: {gesto}")
    
if MODO_PRUEBA:
    try:
        while True:
            tiempo_actual = time.time()
            if ESTADO_ACTUAL == ESTADO_IDLE:
                print(f"[{tiempo_actual:.2f}] Estado: IDLE - Esperando simulacion...")
                
                if tiempo_actual - ULTIMO_GESTO >= INTERVALO_GESTO:
                    GESTO_ACTUAL = simular_gesto()
                    ESTADO_ACTUAL = ESTADO_DETECTADO
                    print(f"[{tiempo_actual:.2f}] Detectado: {GESTO_ACTUAL}")
            elif ESTADO_ACTUAL == ESTADO_DETECTADO:
                  ESTADO_ACTUAL = ESTADO_CONFIRMADO
                  print(f"[{tiempo_actual:.2f}] Confirmado: {GESTO_ACTUAL}. Enviando...")
            elif ESTADO_ACTUAL == ESTADO_CONFIRMADO:
                enviar_gesto(GESTO_ACTUAL)
                ESTADO_ACTUAL = ESTADO_IDLE
                GESTO_ACTUAL = None
                ULTIMO_GESTO = tiempo_actual
            time.sleep(5.0)
    except KeyboardInterrupt:
        print("\nSimulacion detenida")
        sys.exit(0)