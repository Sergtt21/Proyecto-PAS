import time
import cv2
import numpy as np
import mediapipe as mp
from .bus import Event, bus

# Índices de MediaPipe FaceMesh para ojos (ejemplo común)
# Ojo izquierdo y derecho (6 puntos para EAR)
LEFT_EYE  = [33, 133, 160, 144, 159, 145]
RIGHT_EYE = [362, 263, 387, 373, 386, 374]

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(eye_pts):
    # Orden esperado: [p1(punta), p2(punta opuesta), p3, p4, p5, p6] según mapeo arriba
    # Usaremos pares verticales (p3-p5, p4-p6) y horizontal (p1-p2)
    A = euclidean(eye_pts[2], eye_pts[4])
    B = euclidean(eye_pts[3], eye_pts[5])
    C = euclidean(eye_pts[0], eye_pts[1])
    return (A + B) / (2.0 * C + 1e-6)

def landmarks_to_points(landmarks, shape):
    h, w = shape[:2]
    return [(int(l.x * w), int(l.y * h)) for l in landmarks]

def start_gesture_detection():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_face = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    spec = mp.solutions.drawing_styles

    EAR_THRESH = 0.22        # Umbral base (ajústalo con luz/gafas)
    MIN_BLINK_MS = 120       # Duración mínima para contar blink
    DOUBLE_BLINK_WINDOW = 700  # ms entre blinks para doble parpadeo

    last_blink_time = 0
    blink_count = 0
    ear_low_since = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)

        if res.multi_face_landmarks:
            face = res.multi_face_landmarks[0]
            pts = landmarks_to_points(face.landmark, frame.shape)

            # Obtener puntos de cada ojo
            left_eye_pts  = [pts[i] for i in LEFT_EYE]
            right_eye_pts = [pts[i] for i in RIGHT_EYE]

            # Calcular EAR por ojo y promedio
            ear_l = eye_aspect_ratio(left_eye_pts)
            ear_r = eye_aspect_ratio(right_eye_pts)
            ear   = (ear_l + ear_r) / 2.0

            # Dibujar ojos (opcional)
            for p in left_eye_pts + right_eye_pts:
                cv2.circle(frame, p, 2, (0, 255, 0), -1)

            now_ms = int(time.time() * 1000)

            # Detección de parpadeo simple con histéresis temporal
            if ear < EAR_THRESH:
                if ear_low_since is None:
                    ear_low_since = now_ms
            else:
                if ear_low_since is not None:
                    duration = now_ms - ear_low_since
                    ear_low_since = None
                    if duration >= MIN_BLINK_MS:
                        # Contar blink válido
                        if now_ms - last_blink_time <= DOUBLE_BLINK_WINDOW:
                            blink_count += 1
                        else:
                            blink_count = 1
                        last_blink_time = now_ms

                        # Si hay doble parpadeo → enviar “Hola”
                        if blink_count == 2:
                            cv2.putText(frame, "DOUBLE BLINK → SEND 'Hola'", (30, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                            # Publicar evento hacia el bot
                            bus.put(Event(kind="SEND_TEXT", payload={"text": "Hola"}))
                            blink_count = 0  # reset

            # Overlay de EAR
            cv2.putText(frame, f"EAR: {ear:.3f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Vision - Gestos (ESC para salir)", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
