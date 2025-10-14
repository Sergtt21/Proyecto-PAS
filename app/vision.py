# app/vision.py
import os
import time
import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv

from .bus import put, Event
from .gestures import GestureDetector, LEFT_EYE, RIGHT_EYE, eye_aspect_ratio, mouth_aspect_ratio, LEFT_BROW_POINT, RIGHT_BROW_POINT, LEFT_EYE_CENTER, RIGHT_EYE_CENTER

# Importar el módulo de manejo de logs
from .managelog import manejo_errores

manejo_errores(nivel_warning="ignore", verbose=False) 

load_dotenv()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CHAT_ID = int(CHAT_ID) if CHAT_ID and CHAT_ID.isdigit() else None

# Solo para HUD 
GESTO_A_TEXTO = {
    "DOBLE_PARPADEO":   "Hola 👋",
    "CEJAS_ARRIBA":     "Ya voy 🚗",
    "SONRISA":          "Todo bien 😄",
    "CABEZA_DERECHA":   "OK ✅",
    "CABEZA_IZQUIERDA": "No ❌",
}

class FSM:
    def __init__(self):
        self.estado = "IDLE"
        self.ultimo_gesto = ""
        self.ultimo_mensaje = ""
        self._ts = time.time()

    def set(self, estado, gesto="", mensaje=""):
        self.estado = estado
        if gesto:
            self.ultimo_gesto = gesto
        if mensaje:
            self.ultimo_mensaje = mensaje
        self._ts = time.time()

    def volver_a_idle(self, segundos=1.2):
        if self.estado == "ENVIADO" and (time.time() - self._ts) > segundos:
            self.set("IDLE")

def _to_xy(landmarks, shape):
    h, w = shape[:2]
    return [(int(l.x * w), int(l.y * h)) for l in landmarks]

def _dibujar_hud(frame, fsm: FSM, metricas):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (w-20, 190), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    y = 50
    cv2.putText(frame, f"Estado: {fsm.estado}", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2); y+=30
    cv2.putText(frame, f"Gesto:  {fsm.ultimo_gesto}", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2); y+=30
    cv2.putText(frame, f"Mensaje:{fsm.ultimo_mensaje}", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,255,0), 2)

    if metricas:
        s = f"EAR:{metricas.get('EAR',0):.3f}  MAR:{metricas.get('MAR',0):.3f}  BROW:{metricas.get('BROW',0):.1f}  YAW:{metricas.get('YAW',0):.1f}°"
        cv2.putText(frame, s, (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    if "FPS" in metricas:
        fps = metricas["FPS"]
        if fps >= 30:
            fps_color = (0, 255, 0)      # Verde
        elif fps >= 25:
            fps_color = (0, 255, 255)    # Amarillo
        else:
            fps_color = (0, 0, 255)      # Rojo

        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 170, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)

def start_gesture_detection():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_face = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    fsm = FSM()
    detector = GestureDetector()

    # ---------- Calibración (3 s) ----------
    print("Calibrando... rostro neutro por 3s.")
    t0 = time.time()
    ear_vals, brow_vals, mar_vals = [], [], []
    while time.time() - t0 < 3.0:
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        if res.multi_face_landmarks:
            pts = _to_xy(res.multi_face_landmarks[0].landmark, frame.shape)
            # EAR baseline
            le = [pts[i] for i in LEFT_EYE]; re = [pts[i] for i in RIGHT_EYE]
            ear_vals.append( (eye_aspect_ratio(le) + eye_aspect_ratio(re)) / 2.0 )
            # Cejas-ojo baseline
            lb, rb = pts[LEFT_BROW_POINT], pts[RIGHT_BROW_POINT]
            lec, rec = pts[LEFT_EYE_CENTER], pts[RIGHT_EYE_CENTER]
            brow_vals.append( (abs(lb[1]-lec[1]) + abs(rb[1]-rec[1]))/2.0 )
            # MAR baseline
            mar_vals.append( mouth_aspect_ratio(pts) )

        cv2.putText(frame, "Calibrando... (ESC para saltar)", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow("Vision", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    if ear_vals and brow_vals and mar_vals:
        detector.set_baselines(
            float(np.median(ear_vals)),
            float(np.median(brow_vals)),
            float(np.median(mar_vals)),
        )
    print("Calibración OK.")

    # ---------- Loop principal ----------
    metricas = {}
    # Variables para cálculo de FPS
    prev_time = time.time()
    fps = 0.0
    alpha = 0.2
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)

        if res.multi_face_landmarks:
            pts = _to_xy(res.multi_face_landmarks[0].landmark, frame.shape)
            gestos, metricas = detector.process(pts, frame.shape)

            for g in gestos:
                msg = GESTO_A_TEXTO.get(g, "")
                fsm.set("GESTO_DETECTADO", gesto=g, mensaje=msg)
                if CHAT_ID:
                    put(Event(kind="GESTO", payload={"name": g, "chat_id": CHAT_ID}))
                fsm.set("ENVIADO", gesto=g, mensaje=msg)

            fsm.volver_a_idle(1.2)
            _dibujar_hud(frame, fsm, metricas)
        else:
            cv2.putText(frame, "Rostro no detectado", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60,60,255), 2)

        cv2.putText(frame, "ESC para salir", (30, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        # --- Cálculo de FPS ---
        current_time = time.time()
        delta = current_time - prev_time
        prev_time = current_time

        if delta > 0:
            instant_fps = 1.0 / delta
            fps = (alpha * instant_fps) + (1 - alpha) * fps
        metricas["FPS"] = fps
        _dibujar_hud(frame, fsm, metricas)
        cv2.imshow("Vision", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
