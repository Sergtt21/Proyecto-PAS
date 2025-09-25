import cv2
import mediapipe as mp

captura = cv2.VideoCapture(0)
captura.set(3, 1280)  # Ancho
captura.set(4, 720)  # Alto

mpfm = mp.solutions.drawing_utils
cdj = mpfm.DrawingSpect(thickness=2, circle_radius=2)

mpcara = mp.solutions.face_detection
mpmalla = mpcara.FaceMesh(max_num_faces=1)

if not captura.isOpened():
    print("No se pudo abrir la cámara")
    exit()
    
while True:
    ret, frame = captura.read()
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = mpmalla.process(RGB)
    
    px = []
    py = []
    lista = []
    r=5
    t=3
    if resultados.multi_face_landmarks:
        for rostros in resultados.multi_face_landmarks:
            for id, lm in enumerate(rostros.landmark):
                al, an, c = frame.shape
                x, y = int(lm.x * an), int(lm.y * al)
                px.append(x)
                py.append(y)
                lista.append([id, x, y])
                if id == 1 or id == 4 or id == 61 or id == 291 or id == 199:
                        cv2.circle(frame, (x, y), r, (0, 0, 255), -1)
                mpfm.draw_landmarks(frame, rostros, mpcara.FACE_CONNECTIONS,
                                    cdj.DrawingSpec((0, 255, 0), thickness=t, circle_radius=r),
                                    cdj.DrawingSpec((255, 0, 0), thickness=t, circle_radius=r))
        if len(px) != 0 and len(py) != 0:
            x1, y1 = min(px), min(py)
            x2, y2 = max(px), max(py)
            cv2.rectangle(frame, (x1 - 20, y1 - 20), (x2 + 20, y2 + 20), (0, 255, 0), 2)
            cv2.putText(frame, f'Rostro', (x1 - 30, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'No se detecta rostro', (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
    if not ret:
        print("No se pudo recibir el frame. Cerrando...")
        break
    cv2.imshow('Reconocimiento Facial', frame)
    if cv2.waitKey(1) == ord('q'):
        break
captura.release()
cv2.destroyAllWindows()