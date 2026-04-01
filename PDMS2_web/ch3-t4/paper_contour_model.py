import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


_MODEL_CACHE = {}


def resolve_weights_path(weights_path=None):
    if weights_path is not None:
        candidate = Path(weights_path)
        if candidate.exists():
            return candidate

    current_dir = Path(__file__).resolve().parent
    candidates = [
        current_dir / "models" / "best.pt",
        current_dir.parent / "ch3-t3" / "models" / "best.pt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _load_model(weights_path=None):
    resolved = resolve_weights_path(weights_path)
    if resolved is None:
        raise FileNotFoundError("找不到紙張輪廓模型權重 best.pt")

    key = str(resolved)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = YOLO(key)

    return _MODEL_CACHE[key], resolved


def detect_paper_contour_by_model(image, conf=0.5, device=None, weights_path=None):
    if image is None:
        return None, None, 0, None

    model, resolved_path = _load_model(weights_path)
    results = model.predict(source=image, conf=conf, device=device, verbose=False)

    if not results:
        return None, None, 0, resolved_path

    result = results[0]
    if result.masks is None or len(result.masks.data) == 0:
        return None, None, 0, resolved_path

    if result.boxes is not None and len(result.boxes) > 0:
        mask_idx = int(np.argmax(result.boxes.conf.cpu().numpy()))
    else:
        mask_areas = result.masks.data.cpu().numpy().sum(axis=(1, 2))
        mask_idx = int(np.argmax(mask_areas))

    mask = result.masks.data[mask_idx].cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255

    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)

    # 直接用 mask 白色像素數計算面積，避免輪廓近似造成誤差
    mask_area = int(np.count_nonzero(mask))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask, mask_area, resolved_path

    paper_contour = max(contours, key=cv2.contourArea)
    return paper_contour, mask, mask_area, resolved_path
