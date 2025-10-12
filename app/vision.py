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
DEBUG = os.getenv("DEBUG", "false").strip().lower() == "true"  # NUEVO: modo debug

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
    cv2.rectangle(overlay, (20, 20), (w-20, 160), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    y = 50
    cv2.putText(frame, f"Estado: {fsm.estado}", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2); y+=30
    cv2.putText(frame, f"Gesto:  {fsm.ultimo_gesto}", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2); y+=30
    cv2.putText(frame, f"Mensaje:{fsm.ultimo_mensaje}", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,255,0), 2)

    if metricas:
        s = f"EAR:{metricas.get('EAR',0):.3f}  MAR:{metricas.get('MAR',0):.3f}  BROW:{metricas.get('BROW',0):.1f}  YAW:{metricas.get('YAW',0):.1f}°"
        cv2.putText(frame, s, (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # NUEVO: Banner visual de modo debug (no interfiere con nada)
    if DEBUG:
        cv2.putText(frame, "DEBUG ON", (w-160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

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

        # NUEVO: barra de progreso + texto "Calibrando rostro..."
        progress = min(1.0, (time.time() - t0) / 3.0)
        h, w = frame.shape[:2]
        bar_x1, bar_y1 = 50, h - 60
        bar_x2, bar_y2 = w - 50, h - 30
        filled = int(bar_x1 + progress * (bar_x2 - bar_x1))
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (40, 40, 40), -1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (filled, bar_y2), (0, 200, 0), -1)
        cv2.putText(frame, "Calibrando rostro...", (bar_x1, bar_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

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
        # NUEVO: logs de calibración y umbrales si DEBUG=True
        if DEBUG:
            print(f"[DEBUG] Calibracion OK → EAR_base={np.median(ear_vals):.3f} "
                  f"BROW_base={np.median(brow_vals):.3f} MAR_base={np.median(mar_vals):.3f} "
                  f"EAR_THRESH(aplicado)={detector.EAR_THRESH:.3f}")
    else:
        if DEBUG:
            print("[DEBUG] Calibracion incompleta; se usan valores por defecto.")
    print("Calibración OK.")

    # ---------- Loop principal ----------
    metricas = {}
    # NUEVO: contador FPS simple (solo logs si DEBUG)
    t_start = time.time()
    frames = 0

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

        cv2.imshow("Vision", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

        # NUEVO: FPS a consola si DEBUG cada ~60 frames
        if DEBUG:
            frames += 1
            if frames % 60 == 0:
                elapsed = time.time() - t_start
                fps = frames / max(1e-6, elapsed)
                print(f"[DEBUG] FPS ~ {fps:.1f}")

    cap.release()
    cv2.destroyAllWindows()
