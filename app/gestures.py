# app/gestures.py
import math
import time
import numpy as np
import cv2
from typing import Dict, List, Tuple

# Importar el módulo de manejo de logs
from .managelog import manejo_errores

manejo_errores(nivel_warning="ignore", verbose=False) 

# -----------------------------
# Índices MediaPipe Face Mesh
# -----------------------------
LEFT_EYE  = [33, 133, 160, 144, 159, 145]
RIGHT_EYE = [362, 263, 387, 373, 386, 374]

LEFT_BROW_POINT  = 105
RIGHT_BROW_POINT = 334
LEFT_EYE_CENTER  = 159
RIGHT_EYE_CENTER = 386

MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
MOUTH_UP     = 13
MOUTH_DOWN   = 14

PNP_POINTS = {
    "nose": 1,
    "l_eye": 33,
    "r_eye": 263,
    "l_mouth": 61,
    "r_mouth": 291,
    "chin": 152,
}

# -----------------------------
# Utilidades métricas
# -----------------------------
def _dist(a, b) -> float:
    a, b = np.array(a, float), np.array(b, float)
    return float(np.linalg.norm(a - b))

def eye_aspect_ratio(eye_pts: List[Tuple[int,int]]) -> float:
    A = _dist(eye_pts[2], eye_pts[4])
    B = _dist(eye_pts[3], eye_pts[5])
    C = _dist(eye_pts[0], eye_pts[1]) + 1e-6
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(pts_xy: List[Tuple[int,int]]) -> float:
    h = _dist(pts_xy[MOUTH_UP],   pts_xy[MOUTH_DOWN])
    w = _dist(pts_xy[MOUTH_LEFT], pts_xy[MOUTH_RIGHT]) + 1e-6
    return h / w

def head_pose_yaw_deg(pts_xy: List[Tuple[int,int]], frame_shape) -> float:
    """Estimación simple de yaw con solvePnP (grados)."""
    h, w = frame_shape[:2]
    model_points = np.array([
        [0.0,   0.0,   0.0],   # nose
        [-30.0, -30.0, -30.0], # left eye
        [ 30.0, -30.0, -30.0], # right eye
        [-40.0,  30.0, -30.0], # left mouth
        [ 40.0,  30.0, -30.0], # right mouth
        [ 0.0,   70.0, -20.0], # chin
    ], dtype=np.float64)

    image_points = np.array([
        pts_xy[PNP_POINTS["nose"]],
        pts_xy[PNP_POINTS["l_eye"]],
        pts_xy[PNP_POINTS["r_eye"]],
        pts_xy[PNP_POINTS["l_mouth"]],
        pts_xy[PNP_POINTS["r_mouth"]],
        pts_xy[PNP_POINTS["chin"]],
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0,             center[0]],
        [0,            focal_length,  center[1]],
        [0,            0,             1        ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0

    R, _ = cv2.Rodrigues(rvec)
    # Nota: el signo puede variar según la cámara (mirror/no mirror).
    yaw = math.degrees(math.atan2(R[2,0], R[2,2]))
    return yaw

# -----------------------------
# Detector de gestos (puro)
# -----------------------------
class GestureDetector:
    """
    Devuelve:
    - lista de gestos detectados (strings en español)
    - dict de métricas para HUD
    Gestos:
    DOBLE_PARPADEO, CEJAS_ARRIBA, SONRISA, CABEZA_DERECHA, CABEZA_IZQUIERDA
    """
    def __init__(self) -> None:
        # Baselines (calibración)
        self.ear_base: float = 0.24
        self.brow_eye_base: float = 20.0
        self.mar_base: float = 0.15

        # Parámetros/umbrales
        self.EAR_THRESH = 0.22
        self.MIN_BLINK_MS = 120
        self.DOUBLE_WIN_MS = 700

        self.BROW_GAIN = 0.20
        self.BROW_MIN_MS = 250
        self.BROW_COOLDOWN_MS = 1000

        self.SMILE_GAIN = 0.35
        self.SMILE_MIN_MS = 250
        self.SMILE_COOLDOWN_MS = 1200

        self.YAW_THRESH = 16.0
        self.YAW_COOLDOWN_MS = 900

        # Estado temporal
        self._ear_low_since = None
        self._last_blink_ms = 0
        self._blink_count = 0

        self._brow_since = None
        self._last_brow_ms = 0

        self._smile_since = None
        self._last_smile_ms = 0

        self._last_yaw_ms = 0

    # ---- Calibración / baselines ----
    def set_baselines(self, ear_b: float, brow_b: float, mar_b: float) -> None:
        self.ear_base = ear_b
        self.brow_eye_base = brow_b
        self.mar_base = mar_b
        # Umbral EAR adaptativo
        self.EAR_THRESH = max(0.18, min(0.28, self.ear_base * 0.75))

    # ---- Procesamiento por frame ----
    def process(self, pts_xy: List[Tuple[int,int]], frame_shape) -> Tuple[List[str], Dict[str, float]]:
        if not pts_xy:
            self._ear_low_since = None
            self._brow_since = None
            self._smile_since = None
            return [], {}

        now = int(time.time() * 1000)
        gestures: List[str] = []

        # EAR (parpadeo / doble parpadeo)
        left_eye_pts  = [pts_xy[i] for i in LEFT_EYE]
        right_eye_pts = [pts_xy[i] for i in RIGHT_EYE]
        ear_l = eye_aspect_ratio(left_eye_pts)
        ear_r = eye_aspect_ratio(right_eye_pts)
        ear = (ear_l + ear_r) / 2.0

        if ear < self.EAR_THRESH:
            if self._ear_low_since is None:
                self._ear_low_since = now
        else:
            if self._ear_low_since is not None:
                dur = now - self._ear_low_since
                self._ear_low_since = None
                if dur >= self.MIN_BLINK_MS:
                    # Parpadeo válido
                    if (now - self._last_blink_ms) <= self.DOUBLE_WIN_MS:
                        self._blink_count += 1
                    else:
                        self._blink_count = 1
                    self._last_blink_ms = now

                    if self._blink_count == 2:
                        gestures.append("DOBLE_PARPADEO")
                        self._blink_count = 0

        # Cejas arriba (distancia ceja-ojo vs baseline)
        lb, rb = pts_xy[LEFT_BROW_POINT], pts_xy[RIGHT_BROW_POINT]
        le, re = pts_xy[LEFT_EYE_CENTER], pts_xy[RIGHT_EYE_CENTER]
        brow_eye = (abs(lb[1]-le[1]) + abs(rb[1]-re[1])) / 2.0

        if brow_eye > self.brow_eye_base * (1.0 + self.BROW_GAIN):
            if self._brow_since is None:
                self._brow_since = now
            elif (now - self._brow_since) >= self.BROW_MIN_MS and (now - self._last_brow_ms) > self.BROW_COOLDOWN_MS:
                gestures.append("CEJAS_ARRIBA")
                self._last_brow_ms = now
                self._brow_since = None
        else:
            self._brow_since = None

        # Sonrisa (MAR) vs baseline
        mar = mouth_aspect_ratio(pts_xy)
        if mar > self.mar_base * (1.0 + self.SMILE_GAIN):
            if self._smile_since is None:
                self._smile_since = now
            elif (now - self._smile_since) >= self.SMILE_MIN_MS and (now - self._last_smile_ms) > self.SMILE_COOLDOWN_MS:
                gestures.append("SONRISA")
                self._last_smile_ms = now
                self._smile_since = None
        else:
            self._smile_since = None

        # Yaw (cabeza izq/der)
        yaw = head_pose_yaw_deg(pts_xy, frame_shape)
        # Convención usada:
        #   yaw > +TH → CABEZA_IZQUIERDA
        #   yaw < -TH → CABEZA_DERECHA
        if (now - self._last_yaw_ms) > self.YAW_COOLDOWN_MS:
            if yaw > self.YAW_THRESH:
                gestures.append("CABEZA_IZQUIERDA")
                self._last_yaw_ms = now
            elif yaw < -self.YAW_THRESH:
                gestures.append("CABEZA_DERECHA")
                self._last_yaw_ms = now

        metrics = {"EAR": ear, "MAR": mar, "BROW": brow_eye, "YAW": yaw}
        return gestures, metrics