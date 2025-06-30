from pathlib import Path
from typing import Optional
import cv2, numpy as np, onnxruntime as ort, mediapipe as mp

# ───── макет ─────
R_TOP, R_HEAD, R_BOTTOM = 0.10, 0.25, 0.65
ASPECT = 2 / 3
OUT_SIZE = (1200, 1800)
THR = 0.05
DENSITY = 0.02
MODEL = "u2net.onnx"
# ─────────────────

# U²-Net ─────────────────────────────────────────────
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name


def hair_mask(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (320, 320)).astype(np.float32) / 255
    x = (x.transpose(2, 0, 1)[None] - 0.5) / 0.5
    pr = sess.run(None, {inp: x})[0][0][0]
    return cv2.resize(pr, (w, h), interpolation=cv2.INTER_LINEAR)


def hair_top(mask: np.ndarray) -> Optional[int]:
    h, w = mask.shape
    binm = (mask > THR).astype(np.uint8)
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), 1)
    need = int(DENSITY * w)
    rows = np.where(binm.sum(1) >= need)[0]
    return int(rows.min()) if rows.size else None


# ───────────────────────────────────────────────────

# Face Mesh + Detection ─────────────────────────────
mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
mp_det = mp.solutions.face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
)
LM_CHIN = 152


def chin_mesh(img: np.ndarray) -> Optional[int]:
    h = img.shape[0]
    res = mp_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    return int(res.multi_face_landmarks[0].landmark[LM_CHIN].y * h)


def chin_bbox(img: np.ndarray) -> Optional[int]:
    h = img.shape[0]
    det = mp_det.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not det.detections:
        return None
    bb = det.detections[0].location_data.relative_bounding_box
    return int((bb.ymin + bb.height) * h)


# ───────────────────────────────────────────────────


def crop_one(img: np.ndarray):
    h, w = img.shape[:2]
    pr = hair_mask(img)
    y_top = hair_top(pr)
    y_chin = chin_mesh(img) or chin_bbox(img)

    if y_top is None or y_chin is None or y_chin <= y_top:
        return img, False

    # ───── вертикаль (как было, без изменений) ─────
    h_head = y_chin - y_top
    H = h_head / R_HEAD
    W = int(round(H * ASPECT))

    y1 = int(round(y_top - R_TOP * H))
    y2 = y1 + int(round(H))
    if y1 < 0:
        y2 += -y1
        y1 = 0
    if y2 > h:
        y1 -= y2 - h
        y2 = h
    # ───────────────────────────────────────────────

    # ───── гор. центрирование ТОЛЬКО по голове ─────
    binm = (pr > THR).astype(np.uint8)
    head_mask = binm[y_top:y_chin, :]  # зона макушка-подбородок
    need_col = int(0.10 * (y_chin - y_top))  # ≥10 % высоты головы
    xs = np.where(head_mask.sum(0) >= need_col)[0]

    if xs.size:  # границы головы
        x_left, x_right = xs.min(), xs.max()
    else:  # fallback
        x_left, x_right = 0, w - 1

    w_head = x_right - x_left
    if w_head >= W:
        x_c = (x_left + x_right) / 2
        x1 = int(round(x_c - W / 2))
        x2 = x1 + W
    else:
        pad = (W - w_head) / 2
        x1 = int(round(x_left - pad))
        x2 = x1 + W

    if x1 < 0:
        x2 += -x1
        x1 = 0
    if x2 > w:
        x1 -= x2 - w
        x2 = w
    # ───────────────────────────────────────────────

    crop = cv2.resize(img[y1:y2, x1:x2], OUT_SIZE, interpolation=cv2.INTER_LANCZOS4)
    return crop, True


# ───── пакет ─────
def batch(src="raw_photos", dst="cropped_photos"):
    Path(dst).mkdir(exist_ok=True)
    for p in Path(src).glob("*"):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(p))
        tag = "asis"
        if img is not None:
            out, ok = crop_one(img)
            tag = "auto" if ok else "asis"
            cv2.imwrite(f"{dst}/{p.stem}_{tag}.jpg", out)
            print(f"{p.name} → {tag.upper()}")


if __name__ == "__main__":
    batch()
