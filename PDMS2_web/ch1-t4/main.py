import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os

def return_score(score):
    sys.exit(int(score))

# ====================================================================
# === 俯視圖 (TOP View) 分析函數
# ====================================================================

CONF_TOP = 0.8
CROP_RATIO = 0.5
GAP_RATIO = 1.2
def analyze_image_top(frame, model):
    """
    分析俯視圖 (TOP View) 影像，檢查旋轉和偏移。
    返回: (cropped_frame, summary_string, final_score)
    """
    # 確保 model 是 YOLO 物件
    if isinstance(model, str):
        model = YOLO(model)
    
    H, W, _ = frame.shape
    crop_w, crop_h = int(W * CROP_RATIO), int(H * CROP_RATIO)
    x1 = (W - crop_w) // 2
    y1 = (H - crop_h) // 2
    x2, y2 = x1 + crop_w, y1 + crop_h

    cropped = frame[y1:y2, x1:x2].copy()

    results = model.predict(source=cropped, conf=CONF_TOP, verbose=False, show=False)
    masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []

    centers = []
    max_mask_side = 0
    rotate_ok_list = []
    GET_POINT = 2

    for mask in masks:
        binary_mask = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            max_mask_side = max(max_mask_side, max(w, h))

            mask_H, mask_W = binary_mask.shape
            scale_x = crop_w / mask_W
            scale_y = crop_h / mask_H

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"] * scale_x)
                cy = int(M["m01"] / M["m00"] * scale_y)
                centers.append((cx, cy))
                cv2.circle(cropped, (cx, cy), 5, (0, 0, 0), -1)

            if len(cnt) >= 5:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box[:, 0] = box[:, 0] * scale_x
                box[:, 1] = box[:, 1] * scale_y
                box = np.intp(box)

                max_len = -1
                main_angle = 0
                for i in range(4):
                    pt1 = box[i]
                    pt2 = box[(i + 1) % 4]
                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]
                    length = dx**2 + dy**2
                    angle = np.arctan2(dy, dx) * 180 / np.pi

                    if length > max_len:
                        max_len = length
                        main_angle = angle

                main_angle = abs(main_angle)
                angle_diff_to_horizontal = abs(main_angle)
                angle_diff_to_vertical = abs(main_angle - 90)

                rotate_ok = (
                    angle_diff_to_horizontal <= 10 or angle_diff_to_vertical <= 10
                )
                rotate_ok_list.append(rotate_ok)

                color = (0, 255, 0) if rotate_ok else (0, 0, 255)
                cv2.drawContours(cropped, [box], 0, color, 2)

    offset = False
    if len(centers) >= 2 and max_mask_side > 0:
        threshold = max_mask_side // 8
        x_vals = [pt[0] for pt in centers]
        y_vals = [pt[1] for pt in centers]
        std_x = np.std(x_vals)
        std_y = np.std(y_vals)
        offset = std_x < threshold or std_y < threshold

        cv2.putText(cropped, f"std_x = {std_x:.2f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        cv2.putText(cropped, f"std_y = {std_y:.2f}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        cv2.putText(cropped, f"threshold = {threshold:.2f}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    status_rotate = "?"
    is_rotate_ng = False
    if rotate_ok_list:
        if all(rotate_ok_list):
            status_rotate = "No Rotate"
        else:
            status_rotate = "Rotate !"
            is_rotate_ng = True

    status_offset = "Offset !" if not offset else "No Offset"
    is_offset_ng = not offset

    summary = f"{status_offset} | {status_rotate}"

    if status_rotate == "?":
        GET_POINT = 0
        color = (0, 0, 0)
    elif is_offset_ng or is_rotate_ng:
        GET_POINT = 1
        color = (0, 0, 255)

    cv2.putText(cropped, summary, (230, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    return cropped, summary, GET_POINT


# ====================================================================
# === 側視圖 (SIDE View) 分析函數 - 簡化版
# ====================================================================

CONF_SIDE = 0.8

def analyze_image_side(IMG_PATH, model):
    """
    簡化版側視圖分析：
    - 必須恰好偵測到 4 個積木
    - 必須分成 2 層
    - 每層恰好 2 個積木
    - 有空隙 = 1 分，無空隙 = 0 分
    """
    # 確保 model 是 YOLO 物件
    if isinstance(model, str):
        model = YOLO(model)
    
    frame = cv2.imread(IMG_PATH)
    if frame is None:
        raise ValueError(f"讀不到圖片：{IMG_PATH}")

    annotated_frame = frame.copy()
    results = model.predict(source=frame, conf=CONF_SIDE, verbose=False, show=False)
    r0 = results[0]

    boxes = r0.boxes.xyxy.cpu().numpy() if r0.boxes is not None else np.empty((0, 4))
    
    # 計算質心
    centroids = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centroids.append((cx, cy))

    num_blocks = len(boxes)
    SCORE = 0
    
    # ========== 繪製所有偵測到的框框 ==========
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255,255,255), 5)
        cv2.putText(annotated_frame, f"Block {i+1}", (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ========== 第一階段：檢查積木數量 ==========
    if num_blocks != 4:
        msg = f"NG: Found {num_blocks} blocks (Need exactly 4)"
        cv2.putText(annotated_frame, msg, (20, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(annotated_frame, f"Score: {SCORE}/2", 
                   (10, annotated_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        return annotated_frame, SCORE

    # ========== 第二階段：自動分層檢查 ==========
    # 按 Y 座標排序
    sorted_centroids = sorted(enumerate(centroids), key=lambda x: x[1][1])
    
    # 計算平均積木高度，用於判斷是否為同一層
    avg_block_height = np.mean([boxes[i][3] - boxes[i][1] for i in range(len(boxes))])
    layer_threshold = avg_block_height * 0.3  # Y 座標差距小於此值視為同一層
    
    # 自動分層
    layers = []
    current_layer = [sorted_centroids[0]]
    
    for i in range(1, len(sorted_centroids)):
        curr_idx, curr_centroid = sorted_centroids[i]
        prev_idx, prev_centroid = current_layer[-1]
        
        # 如果 Y 座標差距小，歸為同一層
        if abs(curr_centroid[1] - prev_centroid[1]) < layer_threshold:
            current_layer.append(sorted_centroids[i])
        else:
            # 開新層
            layers.append(current_layer)
            current_layer = [sorted_centroids[i]]
    
    # 加入最後一層
    layers.append(current_layer)
      
    # 檢查每層是否恰好 2 個積木
    for i, layer in enumerate(layers):
        if len(layer) != 2:
            msg = f"NG: Layer {i+1} has {len(layer)} blocks (Need exactly 2)"
            cv2.putText(annotated_frame, msg, (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.putText(annotated_frame, f"Score: {SCORE}/2", 
                       (10, annotated_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            return annotated_frame, SCORE
    
    # 提取兩層的索引和質心
    layer1_indices = [item[0] for item in layers[0]]
    layer2_indices = [item[0] for item in layers[1]]
    layer1 = [centroids[i] for i in layer1_indices]
    layer2 = [centroids[i] for i in layer2_indices]
    
    # ========== 繪製分層標註 ==========
    colors = [(255, 0, 0), (0, 0, 255)]  # 紅色=上層, 藍色=下層
    
    for layer_idx, (layer, indices) in enumerate([(layer1, layer1_indices), 
                                                    (layer2, layer2_indices)]):
        color = colors[layer_idx]
        for i, centroid in enumerate(layer):
            cx, cy = centroid
            box_idx = indices[i]
            x1, y1, x2, y2 = map(int, boxes[box_idx])
            
            # 繪製分層框框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 8)
            cv2.putText(annotated_frame, f"L{layer_idx+1}-{i+1}", (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 4)
            cv2.circle(annotated_frame, (int(cx), int(cy)), 8, (255, 0, 0), -1)

    # ========== 第三階段：檢查空隙 ==========
    # 計算每層內兩個積木的 X 座標距離
    layer1_x_gap = abs(layer1[0][0] - layer1[1][0])
    layer2_x_gap = abs(layer2[0][0] - layer2[1][0])
    
    avg_block_width = np.mean([boxes[i][2] - boxes[i][0] for i in range(len(boxes))])
    gap_threshold = avg_block_width * GAP_RATIO  # 空隙閾值：大於 1.2 倍寬度視為有空隙
    
    has_gap = layer1_x_gap > gap_threshold or layer2_x_gap > gap_threshold
    
    # ========== 繪製空隙標註 ==========
    if has_gap:
        SCORE = 1
        gap_msg = "GAP DETECTED"
        gap_color = (0, 255, 255)
        
        # 標註哪一層有空隙
        if layer1_x_gap > gap_threshold:
            mid_x = (layer1[0][0] + layer1[1][0]) / 2
            mid_y = (layer1[0][1] + layer1[1][1]) / 2
            cv2.line(annotated_frame, 
                    (int(layer1[0][0]), int(layer1[0][1])), 
                    (int(layer1[1][0]), int(layer1[1][1])), 
                    (0, 0, 255), 3)
            cv2.putText(annotated_frame, "GAP", (int(mid_x), int(mid_y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        if layer2_x_gap > gap_threshold:
            mid_x = (layer2[0][0] + layer2[1][0]) / 2
            mid_y = (layer2[0][1] + layer2[1][1]) / 2
            cv2.line(annotated_frame, 
                    (int(layer2[0][0]), int(layer2[0][1])), 
                    (int(layer2[1][0]), int(layer2[1][1])), 
                    (0, 0, 255), 3)
            cv2.putText(annotated_frame, "GAP", (int(mid_x), int(mid_y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        SCORE = 2
        gap_msg = "NO GAP"
        gap_color = (0, 255, 0)

    # ========== 顯示最終結果 ==========
    cv2.putText(annotated_frame, f"OK: 4 blocks, 2 layers", (20, 45), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.putText(annotated_frame, gap_msg, (20, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, gap_color, 3)
    cv2.putText(annotated_frame, f"Score: {SCORE}/2", 
               (10, annotated_frame.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, gap_color, 3)

    return annotated_frame, SCORE


# ====================================================================
# === 主程式執行區塊
# ====================================================================

if __name__ == "__main__":
    if len(sys.argv) > 2:
        uid = sys.argv[1]
        img_id = sys.argv[2]
        
        # SIDE_IMG_PATH = rf"kid\{uid}\{img_id}-side.jpg"
        SIDE_IMG_PATH = os.path.join('kid', uid, f"{img_id}-side.jpg")
        # TOP_IMG_PATH = rf"kid\{uid}\{img_id}-top.jpg"
        TOP_IMG_PATH = os.path.join('kid', uid, f"{img_id}-top.jpg")
        MODEL_PATH = r"ch1-t4/toybrick.pt"
    else:
        print("請提供 uid 和 img_id 參數")
        sys.exit(1)

    # --- 載入模型 ---
    try:
        yolo_model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"錯誤：載入 YOLO 模型失敗 (路徑: {MODEL_PATH})。請確保檔案存在。")
        sys.exit(1)

    # --- 1. 執行側視圖分析 ---
    score_side = -1
    try:
        annotated_side, score_side = analyze_image_side(SIDE_IMG_PATH, yolo_model)
        print(f"側視圖 ({SIDE_IMG_PATH}) 得分: {score_side}")
        
        # side_result_path = rf"kid\{uid}\{img_id}-side_result.jpg"
        side_result_path = os.path.join('kid',uid, f"{img_id}-side_result.jpg")
        cv2.imwrite(side_result_path, annotated_side)
        print(f"側視圖結果已儲存至: {side_result_path}")
        
    except ValueError as e:
        print(f"側視圖分析失敗: {e}")
    except Exception as e:
        print(f"側視圖分析時發生錯誤: {e}")

    # --- 2. 執行俯視圖分析 ---
    score_top = -1
    try:
        frame_top = cv2.imread(TOP_IMG_PATH)
        if frame_top is None:
            raise ValueError("讀取俯視圖失敗")

        analyzed_frame, summary, score_top = analyze_image_top(frame_top, yolo_model)
        print(f"俯視圖 ({TOP_IMG_PATH}) 檢測結果: {summary}")
        print(f"俯視圖得分: {score_top}")

        # top_result_path = rf"kid\{uid}\{img_id}-top_result.jpg"
        top_result_path = os.path.join('kid',uid, f"{img_id}-top_result.jpg")
        cv2.imwrite(top_result_path, analyzed_frame)
        print(f"俯視圖結果已儲存至: {top_result_path}")

    except ValueError as e:
        print(f"俯視圖分析失敗: {e}")
    except Exception as e:
        print(f"俯視圖分析時發生錯誤: {e}")

    # --- 3. 輸出最低得分 ---
    if score_side == 0 or score_top == 0:
        print("\n總結：有一項分析得分為 0，最終得分為 0。")
        return_score(0)
    
    valid_scores = [s for s in [score_side, score_top] if s != -1]

    if not valid_scores:
        print("\n總結：兩項分析皆失敗或未執行，無法計算最低得分。")
        final_score = -1
    else:
        final_score = min(valid_scores)
        print(f"\n最終最低得分：{final_score}")
        
    return_score(final_score if final_score != -1 else 0)

    # # === test ===
    # MODEL_PATH = r"toybrick.pt"

    # for i in range(1, 5):

    #     img_url = fr"{i}.jpg"
    #     result, score = analyze_image_side(img_url, model=MODEL_PATH)
    #     print(f'{img_url} : {score}')
        
    #     result = cv2.resize(result, (0, 0), fx = 0.3, fy = 0.3)
    #     cv2.imshow('result', result)
    #     cv2.waitKey(0)
