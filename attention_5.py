
import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import json
import asyncio
import threading
from collections import deque
from pythonosc.udp_client import SimpleUDPClient
import websockets

# =========================
# Config
# =========================

# 如果你已经知道内建摄像头编号，就填数字，比如 0 / 1
# 如果不确定，先保持 None，它会自动扫描 0~4
CAM_ID = None

# Mac 推荐 AVFoundation
USE_AVFOUNDATION = True

# 自动扫描摄像头
AUTO_SCAN_IF_CAM_NONE = True
MAX_CAMERA_INDEX_TO_SCAN = 5

WINDOW_NAME = "Attention Detector"

# 平滑窗口
SMOOTH_WINDOW = 15

# 状态阈值（调宽松一点，更容易进入 focused）
FOCUS_THRESHOLD = 0.55
DRIFT_THRESHOLD = 0.35

# EAR thresholds
EAR_MIN = 0.18
EAR_MAX = 0.30
EYES_OPEN_THRESHOLD = 0.20

# Gaze proxy thresholds
GAZE_X_MIN = 0.35
GAZE_X_MAX = 0.65
GAZE_Y_MIN = 0.25
GAZE_Y_MAX = 0.75

# CSV logging
SAVE_LOG = True
LOG_DIR = "logs"

# 是否画全部人脸点
DRAW_ALL_FACE_POINTS = False

# =========================
# Clarity / Reset Config
# =========================
# clarity: 1.0 = 最清晰, 0.0 = 最模糊
CLARITY_INITIAL = 1.0

# 调整后的速度：
# focused 时恢复更慢；drifting / distracted 时更快变模糊
CLARITY_RECOVER_FOCUSED = 0.5
CLARITY_DECAY_DRIFTING = 0.22
CLARITY_DECAY_DISTRACTED = 0.70
CLARITY_DECAY_NO_FACE = 0.10

# 短暂丢脸容忍时间：低于这个时长仍视为有人
NO_FACE_GRACE_SECONDS = 1.20

# 连续无人多久后触发 reset
NO_FACE_RESET_SECONDS = 6.0

# 直接给 TD 的 blur 强度范围
BLUR_MAX = 30.0

# 启动保护期：前几秒不要让 clarity 掉太快
STARTUP_GRACE_SECONDS = 2.0

# =========================
# OSC / TouchDesigner
# =========================
SEND_OSC = True
TD_IP = "127.0.0.1"   # 如果 TD 在同一台机器上就用这个
TD_PORT = 9000        # 要和 TouchDesigner 里的 OSC In CHOP 一致

osc_client = None
if SEND_OSC:
    try:
        osc_client = SimpleUDPClient(TD_IP, TD_PORT)
        print(f"[INFO] OSC ready -> {TD_IP}:{TD_PORT}")
    except Exception as e:
        print(f"[WARN] OSC init failed: {e}")
        osc_client = None

# =========================
# WebSocket / HTML
# =========================
SEND_WS = True
WS_HOST = "127.0.0.1"
WS_PORT = 8765
WS_SEND_INTERVAL = 0.03  # ~33fps

latest_state = {
    "attention_score": 0.0,
    "raw_attention": 0.0,
    "fixation_duration": 0.0,
    "status": "no face",
    "status_value": -1.0,
    "ear": 0.0,
    "left_ear": 0.0,
    "right_ear": 0.0,
    "gaze_left_x": 0.0,
    "gaze_left_y": 0.0,
    "gaze_right_x": 0.0,
    "gaze_right_y": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "roll": 0.0,
    "eyes_open": 0,
    "gaze_centered": 0,
    "face_detected": 0,
    "clarity": 1.0,
    "blur_strength": 0.0,
    "reset": 0,
    "timestamp": time.time(),
}

state_lock = threading.Lock()

# =========================
# MediaPipe
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# Landmark indices
# =========================
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_EYE_LIDS = (159, 145)
RIGHT_EYE_LIDS = (386, 374)

POSE_LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291
}

# Generic 3D model points for solvePnP
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),         # nose tip
    (0.0, -63.6, -12.5),     # chin
    (-43.3, 32.7, -26.0),    # left eye outer
    (43.3, 32.7, -26.0),     # right eye outer
    (-28.9, -28.9, -24.1),   # mouth left
    (28.9, -28.9, -24.1)     # mouth right
], dtype=np.float64)

attention_history = deque(maxlen=SMOOTH_WINDOW)
focus_start_time = None

# =========================
# Runtime state
# =========================
clarity = CLARITY_INITIAL
last_face_time = time.time()
last_frame_time = time.time()
startup_time = time.time()
reset_flag = 0
no_face_reset_armed = False
absence_started_at = None


# =========================
# Utility
# =========================
def ensure_log_dir():
    if SAVE_LOG and not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

def create_log_file():
    ensure_log_dir()
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"attention_log_{ts}.csv")
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "attention_score",
        "raw_attention",
        "status",
        "fixation_duration",
        "ear",
        "left_ear",
        "right_ear",
        "left_gaze_x",
        "left_gaze_y",
        "right_gaze_x",
        "right_gaze_y",
        "pitch",
        "yaw",
        "roll",
        "eyes_open",
        "gaze_centered",
        "face_detected",
        "clarity",
        "blur_strength",
        "reset_flag"
    ])
    return f, writer, path

def draw_text(frame, text, org, color=(255, 255, 255), scale=0.65, thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def get_backend_flag():
    return cv2.CAP_AVFOUNDATION if USE_AVFOUNDATION else cv2.CAP_ANY

def open_camera_with_index(index):
    backend = get_backend_flag()
    return cv2.VideoCapture(index, backend)

def scan_cameras(max_index=5):
    print("\n[INFO] Scanning cameras...")
    working = []
    for idx in range(max_index):
        cap = open_camera_with_index(idx)
        opened = cap.isOpened()
        ok, frame = cap.read() if opened else (False, None)
        shape = None if frame is None else frame.shape
        print(f"  index={idx}, opened={opened}, read={ok}, shape={shape}")
        if opened and ok and frame is not None:
            working.append(idx)
        cap.release()
    return working

def choose_camera():
    if CAM_ID is not None:
        print(f"[INFO] Using fixed CAM_ID={CAM_ID}")
        cap = open_camera_with_index(CAM_ID)
        return cap, CAM_ID

    if AUTO_SCAN_IF_CAM_NONE:
        working = scan_cameras(MAX_CAMERA_INDEX_TO_SCAN)
        if not working:
            return None, None
        chosen = working[0]
        print(f"[INFO] Auto-selected camera index {chosen}")
        cap = open_camera_with_index(chosen)
        return cap, chosen

    cap = open_camera_with_index(0)
    return cap, 0


# =========================
# Geometry helpers
# =========================
def lm_to_xy(landmarks, idx, w, h):
    return np.array([landmarks[idx].x * w, landmarks[idx].y * h], dtype=np.float64)

def eye_aspect_ratio(landmarks, ids, w, h):
    p1 = lm_to_xy(landmarks, ids[0], w, h)
    p2 = lm_to_xy(landmarks, ids[1], w, h)
    p3 = lm_to_xy(landmarks, ids[2], w, h)
    p4 = lm_to_xy(landmarks, ids[3], w, h)
    p5 = lm_to_xy(landmarks, ids[4], w, h)
    p6 = lm_to_xy(landmarks, ids[5], w, h)

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4) + 1e-6
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def iris_ratio(landmarks, iris_ids, corner_ids, lid_ids, w, h):
    iris_pts = np.array([lm_to_xy(landmarks, idx, w, h) for idx in iris_ids], dtype=np.float64)
    iris_center = iris_pts.mean(axis=0)

    left_corner = lm_to_xy(landmarks, corner_ids[0], w, h)
    right_corner = lm_to_xy(landmarks, corner_ids[1], w, h)
    top_lid = lm_to_xy(landmarks, lid_ids[0], w, h)
    bottom_lid = lm_to_xy(landmarks, lid_ids[1], w, h)

    if left_corner[0] > right_corner[0]:
        left_corner, right_corner = right_corner, left_corner

    if top_lid[1] > bottom_lid[1]:
        top_lid, bottom_lid = bottom_lid, top_lid

    x_ratio = (iris_center[0] - left_corner[0]) / (right_corner[0] - left_corner[0] + 1e-6)
    y_ratio = (iris_center[1] - top_lid[1]) / (bottom_lid[1] - top_lid[1] + 1e-6)

    return float(x_ratio), float(y_ratio), iris_center

def estimate_head_pose(landmarks, w, h):
    image_points = np.array([
        lm_to_xy(landmarks, POSE_LANDMARK_IDS["nose_tip"], w, h),
        lm_to_xy(landmarks, POSE_LANDMARK_IDS["chin"], w, h),
        lm_to_xy(landmarks, POSE_LANDMARK_IDS["left_eye_outer"], w, h),
        lm_to_xy(landmarks, POSE_LANDMARK_IDS["right_eye_outer"], w, h),
        lm_to_xy(landmarks, POSE_LANDMARK_IDS["mouth_left"], w, h),
        lm_to_xy(landmarks, POSE_LANDMARK_IDS["mouth_right"], w, h),
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return False, 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    proj_matrix = np.hstack((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    pitch, yaw, roll = euler_angles.flatten()
    return True, float(pitch), float(yaw), float(roll)

def closeness_to_center(r, center=0.5, half_range=0.2):
    d = abs(r - center)
    score = 1.0 - min(d / half_range, 1.0)
    return float(np.clip(score, 0.0, 1.0))

def normalize_ear(ear):
    return float(np.clip((ear - EAR_MIN) / (EAR_MAX - EAR_MIN + 1e-6), 0.0, 1.0))

def angle_score(angle, max_abs_angle):
    return float(np.clip(1.0 - abs(angle) / max_abs_angle, 0.0, 1.0))

def get_status(attention_score):
    if attention_score >= FOCUS_THRESHOLD:
        return "focused"
    elif attention_score >= DRIFT_THRESHOLD:
        return "drifting"
    return "distracted"

def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))

def update_clarity(current_clarity, current_status, dt):
    if current_status == "focused":
        current_clarity += CLARITY_RECOVER_FOCUSED * dt
    elif current_status == "drifting":
        current_clarity -= CLARITY_DECAY_DRIFTING * dt
    elif current_status == "distracted":
        current_clarity -= CLARITY_DECAY_DISTRACTED * dt
    else:  # no face
        current_clarity -= CLARITY_DECAY_NO_FACE * dt

    return clamp01(current_clarity)

def clarity_to_blur_strength(current_clarity):
    return float((1.0 - clamp01(current_clarity)) * BLUR_MAX)


# =========================
# OSC sender
# =========================
def send_osc_data(
    attention_score,
    raw_attention,
    fixation_duration,
    status,
    ear,
    left_ear,
    right_ear,
    lx,
    ly,
    rx,
    ry,
    pitch,
    yaw,
    roll,
    eyes_open,
    gaze_centered,
    face_detected,
    clarity_value,
    blur_strength,
    reset_value
):
    if osc_client is None:
        return

    status_map = {
        "focused": 1.0,
        "drifting": 0.5,
        "distracted": 0.0,
        "no face": -1.0
    }

    try:
        osc_client.send_message("/attention/score", float(attention_score))
        osc_client.send_message("/attention/raw", float(raw_attention))
        osc_client.send_message("/attention/fixation", float(fixation_duration))
        osc_client.send_message("/attention/status", float(status_map.get(status, -1.0)))

        osc_client.send_message("/attention/ear", float(ear))
        osc_client.send_message("/attention/left_ear", float(left_ear))
        osc_client.send_message("/attention/right_ear", float(right_ear))

        osc_client.send_message("/attention/gaze_left_x", float(lx))
        osc_client.send_message("/attention/gaze_left_y", float(ly))
        osc_client.send_message("/attention/gaze_right_x", float(rx))
        osc_client.send_message("/attention/gaze_right_y", float(ry))

        osc_client.send_message("/attention/pitch", float(pitch))
        osc_client.send_message("/attention/yaw", float(yaw))
        osc_client.send_message("/attention/roll", float(roll))

        osc_client.send_message("/attention/eyes_open", int(eyes_open))
        osc_client.send_message("/attention/gaze_centered", int(gaze_centered))
        osc_client.send_message("/attention/face_detected", int(face_detected))

        osc_client.send_message("/attention/clarity", float(clarity_value))
        osc_client.send_message("/attention/blur_strength", float(blur_strength))
        osc_client.send_message("/attention/reset", int(reset_value))
    except Exception as e:
        print(f"[WARN] OSC send failed: {e}")


# =========================
# WebSocket helpers
# =========================
def update_latest_state(
    attention_score,
    raw_attention,
    fixation_duration,
    status,
    ear,
    left_ear,
    right_ear,
    lx,
    ly,
    rx,
    ry,
    pitch,
    yaw,
    roll,
    eyes_open,
    gaze_centered,
    face_detected,
    clarity_value,
    blur_strength,
    reset_value
):
    status_map = {
        "focused": 1.0,
        "drifting": 0.5,
        "distracted": 0.0,
        "no face": -1.0
    }

    with state_lock:
        latest_state["attention_score"] = float(attention_score)
        latest_state["raw_attention"] = float(raw_attention)
        latest_state["fixation_duration"] = float(fixation_duration)
        latest_state["status"] = status
        latest_state["status_value"] = float(status_map.get(status, -1.0))
        latest_state["ear"] = float(ear)
        latest_state["left_ear"] = float(left_ear)
        latest_state["right_ear"] = float(right_ear)
        latest_state["gaze_left_x"] = float(lx)
        latest_state["gaze_left_y"] = float(ly)
        latest_state["gaze_right_x"] = float(rx)
        latest_state["gaze_right_y"] = float(ry)
        latest_state["pitch"] = float(pitch)
        latest_state["yaw"] = float(yaw)
        latest_state["roll"] = float(roll)
        latest_state["eyes_open"] = int(eyes_open)
        latest_state["gaze_centered"] = int(gaze_centered)
        latest_state["face_detected"] = int(face_detected)
        latest_state["clarity"] = float(clarity_value)
        latest_state["blur_strength"] = float(blur_strength)
        latest_state["reset"] = int(reset_value)
        latest_state["timestamp"] = time.time()

def get_latest_state_json():
    with state_lock:
        payload = dict(latest_state)
    return json.dumps(payload)

async def ws_handler(websocket):
    print("[INFO] HTML client connected")
    try:
        while True:
            await websocket.send(get_latest_state_json())
            await asyncio.sleep(WS_SEND_INTERVAL)
    except Exception:
        print("[INFO] HTML client disconnected")

async def ws_main():
    server = await websockets.serve(ws_handler, WS_HOST, WS_PORT)
    print(f"[INFO] WebSocket ready -> ws://{WS_HOST}:{WS_PORT}")
    await server.wait_closed()

def start_ws_server():
    if not SEND_WS:
        return

    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(ws_main())
        except Exception as e:
            print(f"[WARN] WebSocket server failed: {e}")

    t = threading.Thread(target=runner, daemon=True)
    t.start()


# =========================
# Main
# =========================
def main():
    global focus_start_time
    global DRAW_ALL_FACE_POINTS
    global clarity
    global last_face_time
    global last_frame_time
    global startup_time
    global reset_flag
    global no_face_reset_armed
    global absence_started_at

    if SAVE_LOG:
        log_file, log_writer, log_path = create_log_file()
        print(f"[INFO] Logging to: {log_path}")
    else:
        log_file, log_writer, log_path = None, None, None

    if SEND_WS:
        start_ws_server()

    cap, chosen_cam = choose_camera()

    if cap is None or not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        print("[TIP] Check:")
        print("  1. macOS Camera permission")
        print("  2. Close FaceTime / Photo Booth / Zoom / browser tabs")
        print("  3. Try another camera index")
        return

    print(f"[INFO] Camera ready. Using index: {chosen_cam}")
    print("[INFO] Press q or ESC to quit.")
    print("[INFO] Press r to reset smoothing/fixation.")
    print("[INFO] Press c to reset clarity.")
    print("[INFO] Press l to toggle full landmark drawing.")

    while True:
        reset_flag = 0

        now = time.time()
        dt = now - last_frame_time
        last_frame_time = now
        dt = max(1e-4, min(dt, 0.1))

        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] Camera frame read failed.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        attention_score = 0.0
        raw_attention = 0.0
        fixation_duration = 0.0
        status = "no face"

        face_detected = False
        left_ear = right_ear = ear = 0.0
        lx = ly = rx = ry = 0.0
        pitch = yaw = roll = 0.0
        eyes_open = False
        gaze_centered = False

        # center box
        box_w, box_h = int(w * 0.28), int(h * 0.28)
        x1, y1 = w // 2 - box_w // 2, h // 2 - box_h // 2
        x2, y2 = w // 2 + box_w // 2, h // 2 + box_h // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 2)

        if result.multi_face_landmarks:
            face_detected = True
            last_face_time = now
            no_face_reset_armed = True
            absence_started_at = None

            face_landmarks = result.multi_face_landmarks[0].landmark

            if DRAW_ALL_FACE_POINTS:
                for lm in face_landmarks:
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (px, py), 1, (120, 220, 120), -1)

            # EAR
            left_ear = eye_aspect_ratio(face_landmarks, LEFT_EYE_EAR, w, h)
            right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE_EAR, w, h)
            ear = (left_ear + right_ear) / 2.0
            eye_open_score = normalize_ear(ear)
            eyes_open = ear > EYES_OPEN_THRESHOLD

            # Gaze proxy
            lx, ly, lcenter = iris_ratio(face_landmarks, LEFT_IRIS, LEFT_EYE_CORNERS, LEFT_EYE_LIDS, w, h)
            rx, ry, rcenter = iris_ratio(face_landmarks, RIGHT_IRIS, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, w, h)

            gaze_centered = (
                GAZE_X_MIN <= lx <= GAZE_X_MAX and
                GAZE_X_MIN <= rx <= GAZE_X_MAX and
                GAZE_Y_MIN <= ly <= GAZE_Y_MAX and
                GAZE_Y_MIN <= ry <= GAZE_Y_MAX
            )

            gaze_score = np.mean([
                closeness_to_center(lx, 0.5, 0.18),
                closeness_to_center(rx, 0.5, 0.18),
                closeness_to_center(ly, 0.5, 0.28),
                closeness_to_center(ry, 0.5, 0.28),
            ])

            # Head pose
            ok_pose, pitch, yaw, roll = estimate_head_pose(face_landmarks, w, h)
            if ok_pose:
                yaw_score = angle_score(yaw, 25.0)
                pitch_score = angle_score(pitch, 20.0)
            else:
                yaw_score = 0.0
                pitch_score = 0.0

            # Combined attention score
            raw_attention = (
                0.35 * eye_open_score +
                0.35 * gaze_score +
                0.15 * yaw_score +
                0.15 * pitch_score
            )
            raw_attention = float(np.clip(raw_attention, 0.0, 1.0))

            attention_history.append(raw_attention)
            attention_score = float(np.mean(attention_history))

            # Fixation duration
            if attention_score >= FOCUS_THRESHOLD and eyes_open and gaze_centered:
                if focus_start_time is None:
                    focus_start_time = now
                fixation_duration = now - focus_start_time
            else:
                focus_start_time = None
                fixation_duration = 0.0

            status = get_status(attention_score)

            # Draw iris centers
            cv2.circle(frame, tuple(lcenter.astype(int)), 3, (0, 255, 0), -1)
            cv2.circle(frame, tuple(rcenter.astype(int)), 3, (0, 255, 0), -1)

            # Draw key points
            for idx in [1, 152, 33, 263, 61, 291]:
                p = lm_to_xy(face_landmarks, idx, w, h).astype(int)
                cv2.circle(frame, tuple(p), 2, (255, 200, 0), -1)

        else:
            # 短暂丢脸容忍：避免观众稍微偏头/低头就立刻进入 no face
            time_since_face = now - last_face_time

            if time_since_face < NO_FACE_GRACE_SECONDS:
                attention_score = float(np.mean(attention_history)) if len(attention_history) > 0 else FOCUS_THRESHOLD
                fixation_duration = 0.0
                status = get_status(attention_score)
                face_detected = True
            else:
                # 没检测到脸时，不再疯狂往 history 里灌 0
                # 避免一开始和短暂丢脸时 attention 被瞬间拉崩
                if len(attention_history) == 0:
                    attention_history.append(FOCUS_THRESHOLD)

                attention_score = float(np.mean(attention_history)) if len(attention_history) > 0 else FOCUS_THRESHOLD
                focus_start_time = None
                fixation_duration = 0.0
                status = "no face"

        # clarity 更新：启动保护期内强制按 focused 处理
        if now - startup_time < STARTUP_GRACE_SECONDS:
            status_for_clarity = "focused"
        else:
            status_for_clarity = status

        clarity = update_clarity(clarity, status_for_clarity, dt)

        # 连续无人达到阈值后触发 reset；短暂丢脸不算真正无人
        time_since_face = now - last_face_time
        effective_face_detected = face_detected or (time_since_face < NO_FACE_GRACE_SECONDS)

        if effective_face_detected:
            absence_started_at = None
        else:
            if absence_started_at is None:
                absence_started_at = now

            absence_duration = now - absence_started_at

            if no_face_reset_armed and absence_duration >= NO_FACE_RESET_SECONDS:
                clarity = CLARITY_INITIAL
                reset_flag = 1
                no_face_reset_armed = False

        blur_strength = clarity_to_blur_strength(clarity)

        # status color
        if status == "focused":
            color = (0, 255, 0)
        elif status == "drifting":
            color = (0, 200, 255)
        elif status == "distracted":
            color = (0, 0, 255)
        else:
            color = (180, 180, 180)

        draw_text(frame, f"CAM: {chosen_cam}", (20, 35), (200, 200, 200), 0.6, 2)
        draw_text(frame, f"Attention Score: {attention_score:.2f}", (20, 70), color, 0.8, 2)
        draw_text(frame, f"Status: {status}", (20, 105), color, 0.8, 2)
        draw_text(frame, f"Fixation: {fixation_duration:.2f}s", (20, 140), (255, 255, 255), 0.7, 2)

        draw_text(frame, f"Clarity: {clarity:.2f}", (20, 175), (255, 255, 0), 0.7, 2)
        draw_text(frame, f"Blur Strength: {blur_strength:.2f}", (20, 210), (255, 120, 120), 0.7, 2)
        draw_text(frame, f"Reset: {reset_flag}", (20, 245), (200, 200, 255), 0.7, 2)

        draw_text(frame, f"EAR: {ear:.3f}", (20, h - 110), (255, 255, 255), 0.6, 2)
        draw_text(frame, f"Gaze L({lx:.2f},{ly:.2f}) R({rx:.2f},{ry:.2f})", (20, h - 80), (255, 255, 255), 0.55, 2)
        draw_text(frame, f"Head pitch:{pitch:.1f} yaw:{yaw:.1f} roll:{roll:.1f}", (20, h - 50), (255, 255, 255), 0.55, 2)
        draw_text(frame, "q/ESC quit | r reset smooth | c reset clarity | l landmarks", (20, h - 20), (180, 180, 180), 0.5, 1)

        # 发给 TouchDesigner
        send_osc_data(
            attention_score=attention_score,
            raw_attention=raw_attention,
            fixation_duration=fixation_duration,
            status=status,
            ear=ear,
            left_ear=left_ear,
            right_ear=right_ear,
            lx=lx,
            ly=ly,
            rx=rx,
            ry=ry,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            eyes_open=eyes_open,
            gaze_centered=gaze_centered,
            face_detected=face_detected,
            clarity_value=clarity,
            blur_strength=blur_strength,
            reset_value=reset_flag
        )

        # 发给 HTML
        update_latest_state(
            attention_score=attention_score,
            raw_attention=raw_attention,
            fixation_duration=fixation_duration,
            status=status,
            ear=ear,
            left_ear=left_ear,
            right_ear=right_ear,
            lx=lx,
            ly=ly,
            rx=rx,
            ry=ry,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            eyes_open=eyes_open,
            gaze_centered=gaze_centered,
            face_detected=face_detected,
            clarity_value=clarity,
            blur_strength=blur_strength,
            reset_value=reset_flag
        )

        cv2.imshow(WINDOW_NAME, frame)

        if SAVE_LOG and log_writer is not None:
            log_writer.writerow([
                time.time(),
                round(attention_score, 4),
                round(raw_attention, 4),
                status,
                round(fixation_duration, 4),
                round(ear, 4),
                round(left_ear, 4),
                round(right_ear, 4),
                round(lx, 4),
                round(ly, 4),
                round(rx, 4),
                round(ry, 4),
                round(pitch, 4),
                round(yaw, 4),
                round(roll, 4),
                int(eyes_open),
                int(gaze_centered),
                int(face_detected),
                round(clarity, 4),
                round(blur_strength, 4),
                int(reset_flag)
            ])

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord("q")]:
            break
        elif key == ord("r"):
            attention_history.clear()
            focus_start_time = None
            print("[INFO] Reset smoothing and fixation timer.")
        elif key == ord("c"):
            clarity = CLARITY_INITIAL
            reset_flag = 1
            no_face_reset_armed = True
            print("[INFO] Manual clarity reset.")
        elif key == ord("l"):
            DRAW_ALL_FACE_POINTS = not DRAW_ALL_FACE_POINTS
            print(f"[INFO] DRAW_ALL_FACE_POINTS = {DRAW_ALL_FACE_POINTS}")

    cap.release()
    cv2.destroyAllWindows()

    if log_file is not None:
        log_file.close()
        print(f"[INFO] Log saved to: {log_path}")


if __name__ == "__main__":
    main()