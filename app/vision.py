# reconocimiento_facial.py
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import euclidean 
import sys # Necesario para el manejo directo de excepciones y sys.exit(0)

# Importar el módulo de manejo de logs
from managelog import manejo_errores

# ----------------------------------------------------
# 1. CONFIGURACIÓN INICIAL (Llamada al módulo externo)
# ----------------------------------------------------
# Se ignoran todos los warnings generados por librerías (como MediaPipe)
manejo_errores(nivel_warning="ignore") 


# --- Constantes y Variables Iniciales (Sin Cambios) ---
ESTADO_IDLE = 0
ESTADO_DETECTADO = 1
ESTADO_CONFIRMADO = 2
UMBRAL_EAR_CERRADO = 0.17 
UMBRAL_MAR_ABIERTO = 0.70 
UMBRAL_CONFIRMACION = 8 
UMBRAL_INCLINACION_Y = 0.65 
estado_actual = ESTADO_IDLE
contador_frames = 0
GESTO_ACTUAL = None
ultimo_gesto_enviado = None

def calcular_ratio(p1, p2, p3, p4, p5, p6):
    try:
        d_v1 = euclidean([p2.x, p2.y], [p6.x, p6.y])
        d_v2 = euclidean([p3.x, p3.y], [p5.x, p5.y])
        d_h = euclidean([p1.x, p1.y], [p4.x, p4.y])
        
        if d_h == 0: return 0.01 
        ratio = (d_v1 + d_v2) / (2.0 * d_h)
        return ratio
    except Exception as e:
        print(f"Error al calcular ratio en función: {e}", file=sys.stderr)
        return 0.01 

def obtener_ratios(rostro_landmarks):
    # Extrae puntos y calcula el EAR (izquierdo) y el MAR.
    L_OJO = [33, 160, 158, 133, 153, 144] 
    BOCA = [61, 0, 17, 291, 14, 13] 
    
    try:
        puntos_ojo = [rostro_landmarks.landmark[i] for i in L_OJO]
        puntos_boca = [rostro_landmarks.landmark[i] for i in BOCA]
        
        ear_izquierdo = calcular_ratio(*puntos_ojo)
        mar = calcular_ratio(*puntos_boca)
        
        return ear_izquierdo, mar
    except Exception as e:
        print(f"Error al obtener ratios de landmarks: {e}", file=sys.stderr)
        return 0.5, 0.5 

def check_inclinacion(rostro_landmarks):
    # verifica si la nariz está por debajo del umbral Y.
    try:
        nariz = rostro_landmarks.landmark[1]
        return nariz.y > UMBRAL_INCLINACION_Y
    except Exception as e:
        print(f"Error al chequear inclinación: {e}", file=sys.stderr)
        return False 

#exepciones
captura = None
try:
    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        raise IOError("No se pudo abrir la cámara. Verifica si está conectada o disponible.")
    
    captura.set(3, 1280)  # Ancho
    captura.set(4, 720)   # Alto

    mpfm = mp.solutions.drawing_utils
    cdj = mpfm.DrawingSpec(thickness=2, circle_radius=2)
    mp_face_mesh = mp.solutions.face_mesh
    mpmalla = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) 

    while True:
        ret, frame = captura.read()
        if not ret:
            print("No se pudo recibir el frame. Cerrando...", file=sys.stderr)
            break

        try:
            frame = cv2.flip(frame, 1) 
            al, an, c = frame.shape
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = mpmalla.process(RGB)

            px, py, lista = [], [], []
            r, t = 5, 3 
            gesto_detectado_actual = None 
            inclinacion_detectada = False
            ear_izquierdo, mar = 0.5, 0.5 

            if resultados.multi_face_landmarks:
                rostros = resultados.multi_face_landmarks[0]
                
                ear_izquierdo, mar = obtener_ratios(rostros)
                inclinacion_detectada = check_inclinacion(rostros)
                
                # Lógica de Gesto
                if ear_izquierdo < UMBRAL_EAR_CERRADO and not inclinacion_detectada:
                    gesto_detectado_actual = "PARPADEO"
                elif mar > UMBRAL_MAR_ABIERTO:
                    gesto_detectado_actual = "BOCA ABIERTA"
                    
                # Dibujo (Malla, ratios, caja, etc.)
                mpfm.draw_landmarks(image=frame, landmark_list=rostros, connections=mp_face_mesh.FACEMESH_TESSELATION, 
                                    landmark_drawing_spec=cdj, connection_drawing_spec=mpfm.DrawingSpec((255, 0, 0), thickness=t, circle_radius=r))
                
                for id, lm in enumerate(rostros.landmark):
                    x, y = int(lm.x * an), int(lm.y * al)
                    px.append(x); py.append(y); lista.append([id, x, y])
                    if id in [1, 4, 61, 291, 199]: cv2.circle(frame, (x, y), r, (0, 0, 255), -1)

                if len(px) != 0:
                    x1, y1 = min(px), min(py); x2, y2 = max(px), max(py) 
                    cv2.rectangle(frame, (x1 - 20, y1 - 20), (x2 + 20, y2 + 20), (0, 255, 0), 2)
                    cv2.putText(frame, 'Rostro', (x1 - 30, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.putText(frame, f"EAR: {ear_izquierdo:.2f}", (an - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (an - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if inclinacion_detectada:
                    cv2.putText(frame, "INCLINACION! (Parpadeo Bloqueado)", (an // 2 - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
            else:
                cv2.putText(frame, 'No se detecta rostro', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Lógica de Máquina de Estados (IDLE, DETECTADO, CONFIRMADO)
            if estado_actual == ESTADO_IDLE:
                cv2.putText(frame, "ESTADO: IDLE", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if gesto_detectado_actual:
                    estado_actual, contador_frames, GESTO_ACTUAL = ESTADO_DETECTADO, 1, gesto_detectado_actual
            elif estado_actual == ESTADO_DETECTADO:
                cv2.putText(frame, f"ESTADO: DETECTADO ({contador_frames}/{UMBRAL_CONFIRMACION})", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if gesto_detectado_actual == GESTO_ACTUAL:
                    contador_frames += 1
                    if contador_frames >= UMBRAL_CONFIRMACION: estado_actual = ESTADO_CONFIRMADO
                else:
                    estado_actual, contador_frames, GESTO_ACTUAL = ESTADO_IDLE, 0, None
            elif estado_actual == ESTADO_CONFIRMADO:
                cv2.putText(frame, "ESTADO: CONFIRMADO", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if gesto_detectado_actual != GESTO_ACTUAL:
                    print(f"GESTO VÁLIDO ENVIADO: {GESTO_ACTUAL}_OK")
                    estado_actual, contador_frames, GESTO_ACTUAL = ESTADO_IDLE, 0, None
            
            if GESTO_ACTUAL:
                color = (0, 0, 255) if estado_actual == ESTADO_CONFIRMADO else (0, 255, 0)
                cv2.putText(frame, f"GESTO: {GESTO_ACTUAL}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow('Reconocimiento Facial', frame)

            if cv2.waitKey(1) == ord('q'):
                break
                
        except cv2.error as e:
            # Captura errores específicos de OpenCV
            print(f"Error de OpenCV durante el procesamiento: {e}", file=sys.stderr)
            continue 
        except Exception as e:
            # Captura cualquier otro error en el bucle
            print(f"Error inesperado en el bucle: {e}", file=sys.stderr)
            break 

except IOError as e:
    # Captura errores críticos de inicialización (ej: cámara no disponible)
    print(f"ERROR CRÍTICO: {e}", file=sys.stderr)
except Exception as e:
    # Captura cualquier otro error de inicialización
    print(f"ERROR CRÍTICO INESPERADO: {e}", file=sys.stderr)

finally:
    # 2.3. Bloque de Limpieza (Se ejecuta SIEMPRE)
    if captura and captura.isOpened():
        print("Liberando recursos de la cámara...")
        captura.release()
    print("Cerrando todas las ventanas de OpenCV...")
    cv2.destroyAllWindows()
    sys.exit(0)