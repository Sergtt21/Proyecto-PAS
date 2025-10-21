import os
import time
import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv

from .bus import put, Event
from .gestures import (
    GestureDetector, LEFT_EYE, RIGHT_EYE, eye_aspect_ratio, mouth_aspect_ratio,
    LEFT_BROW_POINT, RIGHT_BROW_POINT, LEFT_EYE_CENTER, RIGHT_EYE_CENTER
)
from .managelog import manejo_errores

# ===== Config básica / entorno =====
manejo_errores(nivel_warning="ignore", verbose=False)
load_dotenv()

CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CHAT_ID = int(CHAT_ID) if CHAT_ID and CHAT_ID.isdigit() else None
DEBUG = os.getenv("DEBUG", "false").strip().lower() == "true"

# ===== Rate-limit (segundos) =====
RATE_LIMIT_SECONDS = float(os.getenv("RATE_LIMIT_SECONDS", "10.0") or 10.0)

# ===== Confirmación de intención =====
# 1) Gesto de confirmación (p.ej. DOBLE_PARPADEO)
CONFIRM_GESTURE = os.getenv("CONFIRM_GESTURE", "DOBLE_PARPADEO").strip() or "DOBLE_PARPADEO"
# 2) Confirmación por mantenimiento (“hold-to-confirm”) en segundos
HOLD_CONFIRM_SECONDS = float(os.getenv("HOLD_CONFIRM_SECONDS", "0.7") or 0.7)
# Tiempo máximo para decidir un candidato antes de reiniciar selección
SELECT_TIMEOUT_SECONDS = float(os.getenv("SELECT_TIMEOUT_SECONDS", "3.0") or 3.0)
# Histeresis: cuántos ms debe dominar un gesto nuevo para reemplazar al candidato actual
SWITCH_DOMINANCE_MS = int(float(os.getenv("SWITCH_DOMINANCE_MS", "300") or 300))

# ===== HUD: textos por gesto =====
GESTO_A_TEXTO = {
    "DOBLE_PARPADEO":   "Hola, como estas ?👋",
    "CEJAS_ARRIBA":     "Gracias, Hasta pronto🙌",
    "SONRISA":          "Todo bien 😄",
    "CABEZA_DERECHA":   "Listo ✅",
    "CABEZA_IZQUIERDA": "No puedo ❌",
}

# ===== Utilidades =====
def _to_xy(landmarks, shape):
    h, w = shape[:2]
    return [(int(l.x * w), int(l.y * h)) for l in landmarks]

def _dibujar_hud(frame, fsm, metricas):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (w-20, 240), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    y = 48
    cv2.putText(frame, f"Estado: {fsm.estado}", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2); y += 28
    cv2.putText(frame, f"Gesto detect.: {fsm.ultimo_gesto}", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2); y += 28
    cv2.putText(frame, f"Mensaje: {fsm.ultimo_mensaje}", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,255,0), 2); y += 28

    # Selección y confirmación
    if fsm.candidato:
        cv2.putText(frame, f"Candidato: {fsm.candidato}", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,255,180), 2); y += 26
        if fsm.modo_confirmacion == "HOLD":
            # barra de progreso del hold
            frac = np.clip((time.time() - fsm.candidato_desde) / HOLD_CONFIRM_SECONDS, 0.0, 1.0)
            cv2.putText(frame, f"Mantener gesto para confirmar ({int(frac*100)}%)", (40, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,225,255), 2); y += 6
            bx1, by1 = 40, y + 12
            bx2, by2 = w - 40, by1 + 16
            filled = int(bx1 + frac * (bx2 - bx1))
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (60,60,60), -1)
            cv2.rectangle(frame, (bx1, by1), (filled, by2), (0,200,200), -1)
            y += 26
        else:
            cv2.putText(frame, f"Confirma con {CONFIRM_GESTURE}", (40, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,225,255), 2); y += 26

    # Cooldown
    cd = metricas.get("CD_REMAIN", 0.0)
    if cd > 0:
        cv2.putText(frame, f"Cooldown: {cd:0.1f}s", (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2); y += 22
        total = RATE_LIMIT_SECONDS
        frac = min(1.0, max(0.0, (total - cd) / total))
        bx1, by1 = 40, y
        bx2, by2 = w - 40, y + 16
        filled = int(bx1 + frac * (bx2 - bx1))
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (60,60,60), -1)
        cv2.rectangle(frame, (bx1, by1), (filled, by2), (0,200,200), -1)
        y += 26

    # Métricas
    s = f"EAR:{metricas.get('EAR',0):.3f}  MAR:{metricas.get('MAR',0):.3f}  BROW:{metricas.get('BROW',0):.1f}  YAW:{metricas.get('YAW',0):.1f}°"
    cv2.putText(frame, s, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    if DEBUG:
        cv2.putText(frame, "DEBUG ON", (w - 170, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    if "FPS" in metricas:
        fps = metricas["FPS"]
        fps_color = (0,255,0) if fps >= 30 else (0,255,255) if fps >= 25 else (0,0,255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 170, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)


# ===== FSM con selección y confirmación =====
class FSM:
    def __init__(self):
        self.estado = "IDLE"             # IDLE, SELECCION, ENVIADO
        self.ultimo_gesto = ""
        self.ultimo_mensaje = ""
        self._ts = time.time()

        # para cooldown global
        self.cooldown_until = 0.0

        # selección/confirmación
        self.candidato = None
        self.candidato_desde = 0.0
        self.candidato_last_seen = 0.0
        self.modo_confirmacion = "HOLD"  # "HOLD" o "GESTO" (doble parpadeo, etc.)
        self._gesto_dominante_desde = {} # tracking para histeresis de cambio

    def set(self, estado, gesto="", mensaje=""):
        self.estado = estado
        if gesto:
            self.ultimo_gesto = gesto
        if mensaje:
            self.ultimo_mensaje = mensaje
        self._ts = time.time()

    def en_cooldown(self):
        return time.time() < self.cooldown_until

    def cooldown_restante(self):
        return max(0.0, self.cooldown_until - time.time())

    # --- Selección ---
    def proponer_candidato(self, gesture_name: str):
        now = time.time()
        if self.candidato is None:
            # arrancar selección
            self.candidato = gesture_name
            self.candidato_desde = now
            self.candidato_last_seen = now
            self._gesto_dominante_desde[gesture_name] = now
            self.estado = "SELECCION"
            return

        # si es el mismo gesto, refrescar last_seen
        if gesture_name == self.candidato:
            self.candidato_last_seen = now
            # reset punto dominante del candidato
            self._gesto_dominante_desde[gesture_name] = self._gesto_dominante_desde.get(gesture_name, now)
            return

        # si es distinto, aplicamos histeresis: solo cambia si domina X ms
        start = self._gesto_dominante_desde.get(gesture_name, None)
        if start is None:
            self._gesto_dominante_desde[gesture_name] = now
            return
        if (now - start) * 1000.0 >= SWITCH_DOMINANCE_MS:
            # cambiar candidato
            self.candidato = gesture_name
            self.candidato_desde = now
            self.candidato_last_seen = now
            # reiniciar temporizadores de otros gestos para evitar saltos
            for k in list(self._gesto_dominante_desde.keys()):
                self._gesto_dominante_desde[k] = now if k == gesture_name else 0.0

    def sin_candidato(self):
        self.candidato = None
        self.candidato_desde = 0.0
        self.candidato_last_seen = 0.0
        self._gesto_dominante_desde.clear()
        if self.estado == "SELECCION":
            self.estado = "IDLE"

    def timeout_seleccion(self):
        return self.candidato and (time.time() - self.candidato_desde) > SELECT_TIMEOUT_SECONDS

    def hold_confirmado(self):
        return self.candidato and (time.time() - self.candidato_desde) >= HOLD_CONFIRM_SECONDS

    def confirmar_por_gesto(self, detected_names):
        # confirma si aparece el gesto de confirmación mientras hay candidato
        return self.candidato and (CONFIRM_GESTURE in detected_names)

# ====== Loop principal ======
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

    # ---- Calibración (3s) ----
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
            le = [pts[i] for i in LEFT_EYE]; re = [pts[i] for i in RIGHT_EYE]
            ear_vals.append((eye_aspect_ratio(le) + eye_aspect_ratio(re)) / 2.0)
            lb, rb = pts[LEFT_BROW_POINT], pts[RIGHT_BROW_POINT]
            lec, rec = pts[LEFT_EYE_CENTER], pts[RIGHT_EYE_CENTER]
            brow_vals.append((abs(lb[1]-lec[1]) + abs(rb[1]-rec[1]))/2.0)
            mar_vals.append(mouth_aspect_ratio(pts))

        # barra de progreso
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
        detector.set_baselines(float(np.median(ear_vals)),
                               float(np.median(brow_vals)),
                               float(np.median(mar_vals)))
        if DEBUG:
            print(f"[DEBUG] Calibracion OK")
    else:
        if DEBUG:
            print("[DEBUG] Calibracion incompleta; se usan valores por defecto.")
    print("Calibración OK.")

    # ---- Loop ----
    metricas = {}
    prev_time = time.time()
    fps = 0.0
    alpha = 0.2

    last_publish_ts = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)

        detected_names = []
        if res.multi_face_landmarks:
            pts = _to_xy(res.multi_face_landmarks[0].landmark, frame.shape)
            detected_names, metricas = detector.process(pts, frame.shape)

            # Si está en cooldown, solo actualizar HUD y evitar selección/envío
            metricas["CD_REMAIN"] = fsm.cooldown_restante()
            if fsm.en_cooldown():
                # mostrar el último gesto detectado, pero sin seleccionar/enviar
                if detected_names:
                    g = detected_names[0]
                    fsm.ultimo_gesto = g
                    fsm.ultimo_mensaje = GESTO_A_TEXTO.get(g, "")
                # limpiar selección si existía
                fsm.sin_candidato()
            else:
                # Proceso de selección
                if detected_names:
                    # Si el gesto de confirmación aparece y hay candidato → confirmar envío
                    if fsm.confirmar_por_gesto(detected_names):
                        if fsm.candidato:
                            g = fsm.candidato
                            msg = GESTO_A_TEXTO.get(g, "")
                            # respetar rate-limit global
                            now = time.time()
                            if now - last_publish_ts >= RATE_LIMIT_SECONDS:
                                if CHAT_ID:
                                    put(Event(kind="GESTO", payload={"name": g, "chat_id": CHAT_ID}))
                                last_publish_ts = now
                                fsm.cooldown_until = now + RATE_LIMIT_SECONDS
                                fsm.set("ENVIADO", gesto=g, mensaje=msg)
                            else:
                                # aún en cooldown (carrera de ms)
                                fsm.cooldown_until = last_publish_ts + RATE_LIMIT_SECONDS
                            # salir de selección
                            fsm.sin_candidato()
                        # No dispares el propio DOBLE_PARPADEO como mensaje
                    else:
                        # proponer/actualizar candidato con el primer gesto detectado “útil”
                        # (evitamos usar el gesto de confirmación como candidato)
                        usable = [x for x in detected_names if x != CONFIRM_GESTURE]
                        if usable:
                            fsm.proponer_candidato(usable[0])
                            fsm.ultimo_gesto = usable[0]
                            fsm.ultimo_mensaje = GESTO_A_TEXTO.get(usable[0], "")
                        # Confirmación por HOLD si está activo
                        if fsm.modo_confirmacion == "HOLD" and fsm.hold_confirmado():
                            g = fsm.candidato
                            msg = GESTO_A_TEXTO.get(g, "")
                            now = time.time()
                            if now - last_publish_ts >= RATE_LIMIT_SECONDS:
                                if CHAT_ID:
                                    put(Event(kind="GESTO", payload={"name": g, "chat_id": CHAT_ID}))
                                last_publish_ts = now
                                fsm.cooldown_until = now + RATE_LIMIT_SECONDS
                                fsm.set("ENVIADO", gesto=g, mensaje=msg)
                            else:
                                fsm.cooldown_until = last_publish_ts + RATE_LIMIT_SECONDS
                            fsm.sin_candidato()
                else:
                    # No hay gestos detectados: si llevamos mucho en selección, limpiar
                    if fsm.timeout_seleccion():
                        fsm.sin_candidato()

        else:
            cv2.putText(frame, "Rostro no detectado", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60,60,255), 2)
            # si no hay rostro por un rato, se cancela selección
            if fsm.timeout_seleccion():
                fsm.sin_candidato()

        # ---- UI estática ----
        cv2.putText(frame, "ESC para salir", (30, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

        # ---- FPS ----
        now = time.time()
        delta = now - prev_time
        prev_time = now
        if delta > 0:
            fps = (alpha * (1.0/delta)) + (1 - alpha) * fps
        metricas["FPS"] = fps
        metricas["CD_REMAIN"] = fsm.cooldown_restante()

        _dibujar_hud(frame, fsm, metricas)

        cv2.imshow("Vision", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
