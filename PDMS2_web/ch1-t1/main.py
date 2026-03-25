import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
from pathlib import Path

# ================== 裁剪設定 ==================
CROP_RATIO = 0.85

# ================== YOLO 模型 ==================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "toybrick.pt"

print(f"[DEBUG] 模型路徑：{MODEL_PATH}", flush=True)
print(f"[DEBUG] 模型是否存在：{MODEL_PATH.exists()}", flush=True)

try:
    print("[DEBUG] 開始加載 YOLO 模型...", flush=True)
    model = YOLO(str(MODEL_PATH))
    print("[DEBUG] YOLO 模型加載完成", flush=True)
except Exception as e:
    print(f"[ERROR] 模型加載失敗：{e}", flush=True)
    sys.exit(1)

CONF = 0.5


def return_score(score):
    sys.exit(int(score))


# ================== 輔助函數：中心裁剪 ==================
def crop_center(frame, ratio=CROP_RATIO):
    """
    將圖像裁剪至中心指定比例 (ratio) 的區域。
    """
    h, w = frame.shape[:2]

    # 計算邊緣需要裁剪掉的比例
    margin_ratio = (1 - ratio) / 2

    # 計算起始和結束座標
    x_start = int(w * margin_ratio)
    x_end = int(w * (1 - margin_ratio))

    y_start = int(h * margin_ratio)
    y_end = int(h * (1 - margin_ratio))

    # 裁剪圖像
    cropped_frame = frame[y_start:y_end, x_start:x_end]

    return cropped_frame


# ================== YOLO 偵測方塊 & 取得 mask ==================
def detect_blocks_mask(frame, CONF=0.5):
    results = model.predict(source=frame, conf=CONF, verbose=False)
    boxes, masks = [], []

    for r in results:
        if r.boxes is None:
            continue
        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls_id == 0:  # 方塊類別
                boxes.append((x1, y1, x2, y2))
                if r.masks is not None:
                    # masks 數據需要與當前幀的尺寸匹配
                    mask = r.masks.data.cpu().numpy()[i]
                    masks.append(mask)
    return boxes, masks, results


# ================== 遮掉方塊 ==================
def remove_blocks_with_mask(binary, masks, extra_px=10):
    h, w = binary.shape
    # 建立一個結構元素（核心），用於膨脹
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (extra_px * 2, extra_px * 2))

    for mask in masks:
        # 調整 mask 大小以符合 binary 影像
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 膨脹：讓遮擋範圍往外擴張 extra_px 距離
        # 先轉成 uint8 (0 或 1)，再進行膨脹
        mask_binary = (mask_resized > 0).astype(np.uint8)
        mask_dilated = cv2.dilate(mask_binary, kernel)

        # 3. 直接遮蔽：把 mask 區域對應到 binary 的地方直接設為 0 (黑色)
        # 只要 mask_dilated 為 1 的位置，binary 就變 0
        binary[mask_dilated > 0] = 0

    return binary


# ================== 骨架化 ==================
def extract_line_skeleton(binary):
    skeleton = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
    temp = np.copy(binary)
    while True:
        open_img = cv2.morphologyEx(temp, cv2.MORPH_OPEN, element)
        temp2 = cv2.subtract(temp, open_img)
        eroded = cv2.erode(temp, element)
        skeleton = cv2.bitwise_or(skeleton, temp2)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return skeleton


# ================== 判斷是否在骨架線點附近 ==================
def is_mask_near_skeleton(mask, skeleton, tol=5):
    """
    mask: 二值 mask (0/1 或 0/255)
    skeleton: 骨架二值圖
    tol: 搜尋半徑
    """
    mask_resized = cv2.resize(
        mask, (skeleton.shape[1], skeleton.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    ys, xs = np.where(mask_resized > 0)
    h, w = skeleton.shape
    for x, y in zip(xs, ys):
        x0 = max(0, x - tol)
        x1 = min(w, x + tol + 1)
        y0 = max(0, y - tol)
        y1 = min(h, y + tol + 1)
        if np.any(skeleton[y0:y1, x0:x1] > 0):
            return True
    return False


# ================== 在方塊中心標記點 ==================
def draw_block_markers(frame, boxes, masks, is_correct):
    """
    在原始圖像上標記所有偵測到的方塊，並顯示半透明 Mask。
    """
    if len(boxes) != len(is_correct):
        print("警告: boxes 和 is_correct 列表長度不匹配。")
        return frame

    # 建立一個與原圖一樣大的空白圖層，用來畫 Mask
    overlay = frame.copy()
    h, w = frame.shape[:2]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = (0, 255, 0) if is_correct[i] else (0, 0, 255)

        # --- 新增：繪製半透明 Mask ---
        if i < len(masks):
            mask = masks[i]
            # 將 YOLO 的 mask 縮放回目前 frame 的尺寸
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            # 只要 mask > 0 的地方，就填上顏色
            overlay[mask_resized > 0] = color

        # --- 繪製中心點與 Bounding Box ---
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.circle(frame, center, radius=10, color=color, thickness=-1)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 將繪製了 Mask 的圖層與原圖依照比例混合 (0.4 是透明度)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame


# ================== 封裝：讀取 img_path 並回傳 score ==================
def score_from_image(img_path, conf=CONF):
    """
    輸入：img_path（圖片路徑）
    輸出：score（int）→ 4 個正確=2 分；3 個正確=1 分；其他=0 分
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"讀取圖片失敗：{img_path}")

    # ===== 新增: 中心裁剪 75% 區域 =====
    img = crop_center(img, CROP_RATIO)
    # ==================================
    display_frame = img.copy()

    # 灰階 + 模糊
    gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # 自適應二值化：將深色的繩子凸顯出來
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 10
    )

    # 閉運算去雜點
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # YOLO 偵測方塊 & 取得 mask
    # 由於 YOLO 模型會自動縮放圖片，因此這裡傳入裁剪後的 display_frame 即可
    boxes, masks, _ = detect_blocks_mask(display_frame, CONF=conf)

    # 遮掉方塊
    binary_masked = binary.copy()
    if masks:
        binary_masked = remove_blocks_with_mask(binary_masked, masks)

    # 骨架化
    skeleton = extract_line_skeleton(binary_masked)
    # 檢查每個方塊是否靠近骨架
    is_correct = []
    correct_num = 0
    for mask in masks:
        is_near = is_mask_near_skeleton(mask, skeleton, tol=10)
        is_correct.append(is_near)
        if is_near:
            correct_num += 1

    # 在所有圖像上繪製標記
    # 原始圖 (BGR)
    display_frame_with_markers = draw_block_markers(
        display_frame, boxes, masks, is_correct
    )

    # 二值遮罩圖 (GRAY/BGR)
    # 將單通道的二值圖轉為三通道才能繪製彩色標記
    binary_bgr = cv2.cvtColor(binary_masked, cv2.COLOR_GRAY2BGR)
    binary_with_markers = draw_block_markers(binary_bgr, boxes, masks, is_correct)
    # 骨架圖 (GRAY/BGR)
    skeleton_bgr = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    skeleton_with_markers = draw_block_markers(skeleton_bgr, boxes, masks, is_correct)

    # 這裡的顯示程式碼已經被註釋，如果需要預覽，請取消註釋
    # 顯示所有結果圖
    # display_frame_resized = cv2.resize(display_frame_with_markers, (0, 0), fx=0.7, fy=0.7)
    # cv2.imshow('Original with Markers', display_frame_resized)
    # binary_resized = cv2.resize(binary_with_markers, (0, 0), fx=0.7, fy=0.7)
    # cv2.imshow('Binary Masked', binary_resized)
    # skeleton_resized = cv2.resize(skeleton_with_markers, (0, 0), fx=0.7, fy=0.7)
    # cv2.imshow('Skeleton Line', skeleton_resized)

    # 計算分數 (沿用您的計分邏輯)
    correct_num_for_score = correct_num

    if correct_num_for_score >= 2:  # 有 2 個是基礎
        correct_num_for_score -= 2

    if correct_num_for_score >= 2:
        score = 2
    elif correct_num_for_score >= 1:
        score = 1
    else:
        score = 0

    return score, correct_num, display_frame_with_markers


if __name__ == "__main__":
    print("[DEBUG] ch1-t1 main.py 開始執行", flush=True)

    if len(sys.argv) > 2:
        # 使用傳入的 uid 和 id 作為圖片路徑
        uid = sys.argv[1]
        img_id = sys.argv[2]

        # print(f"UID : {uid}\nimg_id : {img_id}")
        image_path = os.path.join("kid", uid, f"{img_id}.jpg")
        print(f"[DEBUG] 圖片路徑：{image_path}", flush=True)
        print(f"[DEBUG] 圖片是否存在：{os.path.exists(image_path)}", flush=True)

        # for i, name in enumerate(sys.argv):
        #     print(f"{i} : {name}")
        # print(f"img_path : {image_path}")
    else:
        # 測試圖片路徑 (請替換為實際測試路徑)
        print("請提供 uid 和 img_id 參數或在程式碼中設定測試路徑。")
        sys.exit(0)

    # image_path = r"ch1-t1.jpg"  # 讀取圖片
    print("[DEBUG] 開始分析圖片", flush=True)
    score, num, result_img = score_from_image(image_path)
    print(f"[DEBUG] 分析完成，score={score}, num={num}", flush=True)

    # print(f"result: kid\{uid}\{img_id}_result.jpg")
    result_path = os.path.join("kid", uid, f"{img_id}_result.jpg")
    print(f"[DEBUG] 儲存結果到：{result_path}", flush=True)
    cv2.imwrite(result_path, result_img)

    # score, num = score_from_image(test_img)
    print("score =", score)
    print("num =", num)
    print("[DEBUG] ch1-t1 main.py 執行完成", flush=True)
    return_score(score)
