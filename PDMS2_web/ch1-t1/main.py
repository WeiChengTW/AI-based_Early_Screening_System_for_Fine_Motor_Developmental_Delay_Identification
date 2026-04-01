import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os

# ================== 裁剪設定 ==================
CROP_RATIO = 0.85

# ================== YOLO 模型 ==================
# 請確保模型路徑正確
model = YOLO(r"ch1-t1/toybrick.pt")
# model = YOLO(r"toybrick.pt")
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
                    # 注意: masks 數據需要與當前幀的尺寸匹配
                    mask = r.masks.data.cpu().numpy()[i]
                    masks.append(mask)
    return boxes, masks, results


# ================== 遮掉方塊 ==================
def remove_blocks_with_mask(binary, masks, extra_px=10):
    h, w = binary.shape
    for mask in masks:
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        # 膨脹 mask，增加遮擋範圍
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (extra_px * 2, extra_px * 2)
        )
        mask_dilated = cv2.dilate((mask_resized > 0).astype(np.uint8), kernel)

        # 使用 bitwise_and 替代直接賦值，使邏輯更清晰
        mask_inverted = cv2.bitwise_not(mask_dilated * 255)
        binary = cv2.bitwise_and(binary, mask_inverted)

    return binary


# ================== 骨架化 ==================
def extract_line_skeleton(binary):
    skeleton = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
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
    在原始圖像上標記所有偵測到的方塊。
    :param frame: 原始圖像 (BGR)
    :param boxes: YOLO 偵測到的 bounding box 列表 [(x1, y1, x2, y2), ...]
    :param masks: YOLO 偵測到的 mask 列表
    :param is_correct: 每個方塊是否正確穿線的布林值列表
    """
    # 檢查列表長度是否匹配
    if len(boxes) != len(is_correct):
        print("警告: boxes 和 is_correct 列表長度不匹配。")
        return frame

    # 繪製標記點時使用一個副本，避免修改到傳入的 frame (視需求而定，這裡為了安全起見使用副本)
    frame_to_draw = frame.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        # 計算中心點
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        center = (center_x, center_y)

        # 根據是否正確穿線來設定顏色
        # 綠色 (0, 255, 0) 代表正確 (Correct)
        # 紅色 (0, 0, 255) 代表錯誤 (Incorrect)
        color = (0, 255, 0) if is_correct[i] else (0, 0, 255)

        # 繪製中心點 (圓形)，稍微加大半徑以便觀察
        cv2.circle(frame_to_draw, center, radius=5, color=color, thickness=-1)

        # 繪製邊界框 (可選，用於確認偵測範圍)
        # cv2.rectangle(frame_to_draw, (x1, y1), (x2, y2), color, 2)

    return frame_to_draw


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
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # 自適應二值化：將深色的繩子凸顯出來
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 33, 10
    )

    # 閉運算去雜點
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

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
    # 注意：這裡的 tol 可以根據實際圖片解析度調整。50 可能有點大，如果誤判多請調小。
    is_correct = []
    correct_num = 0
    for mask in masks:
        is_near = is_mask_near_skeleton(mask, skeleton, tol=50)
        is_correct.append(is_near)
        if is_near:
            correct_num += 1

    # ================== 新增功能：繪製骨架 ==================
    # 準備一個用於視覺化的底圖副本
    visualization_base = display_frame.copy()

    # 為了讓骨架在結果圖中更容易看清，先稍微膨脹一下骨架線條
    # 使用 3x3 的核進行膨脹
    kernel_disp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skeleton_dilated_for_display = cv2.dilate(skeleton, kernel_disp)

    # 找出骨架 (白色像素) 的位置
    skel_ys, skel_xs = np.where(skeleton_dilated_for_display > 0)

    # 在底圖上將這些位置塗成顯眼的顏色 (例如黃色 BGR: 0, 255, 255)
    # 這樣骨架就會顯示在原圖上
    visualization_base[skel_ys, skel_xs] = [0, 255, 255]
    # ========================================================

    # 在已經畫好黃色骨架的圖像上繪製紅綠標記點
    final_result_img = draw_block_markers(
        visualization_base, boxes, masks, is_correct
    )

    # (以下為舊的除錯顯示程式碼，保持註解)
    # # 二值遮罩圖 (GRAY/BGR)
    # binary_bgr = cv2.cvtColor(binary_masked, cv2.COLOR_GRAY2BGR)
    # binary_with_markers = draw_block_markers(binary_bgr, boxes, masks, is_correct)
    #
    # # 骨架圖 (GRAY/BGR)
    # skeleton_bgr = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    # skeleton_with_markers = draw_block_markers(skeleton_bgr, boxes, masks, is_correct)

    # 計算分數 (沿用您的計分邏輯)
    correct_num_for_score = correct_num
    if correct_num_for_score >= 2:  # 假設有 2 個是基礎
        correct_num_for_score -= 2

    if correct_num_for_score == 4:
        score = 2
    elif correct_num_for_score == 3:
        score = 1
    else:
        score = 0

    # 如果在伺服器端運行，通常不需要 waitKey 和 destroyAllWindows
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return score, correct_num, final_result_img


if __name__ == "__main__":
    if len(sys.argv) > 2:
        # 使用傳入的 uid 和 id 作為圖片路徑
        uid = sys.argv[1]
        img_id = sys.argv[2]
        # 使用 os.path.join 處理跨平台路徑問題
        image_path = os.path.join("kid", uid, f"{img_id}.jpg")
    else:
        # 測試圖片路徑 (請替換為實際測試路徑)
        # image_path = "test.jpg" # 如果沒有參數，請在這裡指定一個存在的圖片
        print("請提供 uid 和 img_id 參數，例如: python main.py 1202 ch1-t1")
        sys.exit(0)

    print(f"Processing image: {image_path}")

    try:
        score, num, result_img = score_from_image(image_path)

        # 確保輸出目錄存在
        output_dir = os.path.join("kid", uid)
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, f"{img_id}_result.jpg")
        cv2.imwrite(save_path, result_img)

        print(f"Result saved to: {save_path}")
        print("score =", score)
        print("num =", num)
        return_score(score)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)