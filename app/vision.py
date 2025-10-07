# app/vision.py
import time
import os
import cv2
import mediapipe as mp
from dotenv import load_dotenv

from .bus import put, Event  # usamos la cola

load_dotenv()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CHAT_ID = int(CHAT_ID) if CHAT_ID and CHAT_ID.isdigit() else None

def _mp_landmarks_to_xy(landmarks, shape):
    h, w = shape[:2]
    return [(int(l.x * w), int(l.y * h)) for l in landmarks]

def start_gesture_detection():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_face = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("Calibrando... mira a la cámara con rostro neutro.")
    start_cal = time.time()
    while time.time() - start_cal < 2.5:
        ok, frame = cap.read()
        if not ok: break
        cv2.putText(frame, "Calibrando... mantente quieto", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow("Vision", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    print("Calibracion lista.")

    info = "Teclas: [1]=DOUBLE_BLINK, [2]=BROW_UP, [ESC]=Salir"
    last_text = ""
    while True:
        ok, frame = cap.read()
        if not ok: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        h, w = frame.shape[:2]

        if res.multi_face_landmarks:
            cv2.putText(frame, last_text, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
        else:
            cv2.putText(frame, "Rostro no detectado", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60,60,255), 2)

        cv2.putText(frame, info, (30, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        cv2.imshow("Vision", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break
        elif k == ord('1'):
            put(Event(kind="GESTO", payload={"name": "DOUBLE_BLINK", "chat_id": CHAT_ID}))
            last_text = "Enviando: Hola 👋"
        elif k == ord('2'):
            put(Event(kind="GESTO", payload={"name": "BROW_UP", "chat_id": CHAT_ID}))
            last_text = "Enviando: Ya voy 🚗"

    cap.release()
    cv2.destroyAllWindows()
