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

# Parámetros de optimización
SKIP_N = 2                 # procesa 1 de cada N frames (>=1)
FPS_ALERT_THRESHOLD = 25.0 # color rojo si cae por debajo
ROI_MARGIN = 0.30          # margen extra alrededor del rostro para el ROI

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

# Convertir landmarks normalizados a coords absolutas cuando se procesa un ROI
def _to_xy_roi(landmarks, roi_offset, roi_shape):
    ox, oy = roi_offset  
    rh, rw = roi_shape[:2]
    pts = []
    for l in landmarks:
        x = int(l.x * rw) + ox
        y = int(l.y * rh) + oy
        pts.append((x, y))
    return pts

# Utilidades ROI
def _rect_from_points(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = max(min(xs), 0), max(xs)
    y0, y1 = max(min(ys), 0), max(ys)
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

def _expand_roi(rect, frame_w, frame_h, margin=0.3):
    x, y, w, h = rect
    cx, cy = x + w/2, y + h/2
    w2 = int(w * (1 + margin))
    h2 = int(h * (1 + margin))
    x2 = int(max(0, cx - w2/2))
    y2 = int(max(0, cy - h2/2))
    x2 = min(x2, frame_w - 1)
    y2 = min(y2, frame_h - 1)
    w2 = min(w2, frame_w - x2)
    h2 = min(h2, frame_h - y2)
    return (x2, y2, w2, h2)

def _dibujar_hud(frame, fsm: FSM, metricas, fps=None):
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

    # FPS en HUD con color de alerta
    if fps is not None:
        color = (0, 255, 0) if fps >= FPS_ALERT_THRESHOLD else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

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
    # Estado para skip-frames, ROI y FPS
    frame_idx = 0
    last_roi = None
    last_metricas = {}
    last_had_face = False
    fps_value = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]

        # Seleccionar ROI si existe; si no, usar full frame
        if last_roi is not None:
            x, y, rw, rh = _expand_roi(last_roi, w, h, margin=ROI_MARGIN)
            roi = frame[y:y+rh, x:x+rw]
            roi_offset = (x, y)
            proc_img = roi
        else:
            proc_img = frame
            roi_offset = (0, 0)

        # Temporización con getTickCount y control de skip-frames
        do_process = (frame_idx % max(1, SKIP_N) == 0)
        t_start = cv2.getTickCount()

        res = None
        had_face = False
        pts_abs = None

        if do_process:
            rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
            res = mp_face.process(rgb)

            if res.multi_face_landmarks:
                had_face = True
                # landmarks a coordenadas absolutas del frame completo
                pts_abs = _to_xy_roi(res.multi_face_landmarks[0].landmark, roi_offset, proc_img.shape)

                gestos, metricas = detector.process(pts_abs, frame.shape)

                for g in gestos:
                    msg = GESTO_A_TEXTO.get(g, "")
                    fsm.set("GESTO_DETECTADO", gesto=g, mensaje=msg)
                    if CHAT_ID:
                        put(Event(kind="GESTO", payload={"name": g, "chat_id": CHAT_ID}))
                    fsm.set("ENVIADO", gesto=g, mensaje=msg)

                # Actualizar métricas/estado para HUD y ROI
                last_metricas = metricas
                last_had_face = True

                # Actualizar ROI a partir de los puntos detectados
                last_roi = _rect_from_points(pts_abs)
            else:
                last_had_face = False
                last_roi = None  # Volver a full-frame si se pierde el rostro

        t_end = cv2.getTickCount()
        dt = (t_end - t_start) / cv2.getTickFrequency()
        if dt > 0:
            fps_value = 1.0 / dt  # FPS instantáneo del pipeline de procesamiento

        # === DIBUJO / HUD ===
        if do_process and had_face:
            fsm.volver_a_idle(1.2)
            _dibujar_hud(frame, fsm, metricas, fps=fps_value)
        elif (not do_process) and last_had_face:
            fsm.volver_a_idle(1.2)
            _dibujar_hud(frame, fsm, last_metricas, fps=fps_value)
        else:
            cv2.putText(frame, "Rostro no detectado", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60,60,255), 2)
            color = (0, 255, 0) if fps_value >= FPS_ALERT_THRESHOLD else (0, 0, 255)
            cv2.putText(frame, f"FPS: {fps_value:.1f}", (w - 200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.putText(frame, "ESC para salir", (30, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

        # Borde del ROI 
        if last_roi is not None:
            x, y, rw, rh = _expand_roi(last_roi, w, h, margin=ROI_MARGIN)
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), (200, 200, 0), 1)

        cv2.imshow("Vision", frame)
        frame_idx += 1

        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
