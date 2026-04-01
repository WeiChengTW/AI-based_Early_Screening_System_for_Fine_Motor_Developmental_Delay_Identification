import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys

# 載入 main_side.py 所需的模組
try:
    from check_gap import CheckGap
    from MaskAnalyzer import MaskAnalyzer
    from StairChecker import StairChecker
    from PyramidChecker import PyramidCheck
    from LayerGrouping import LayerGrouping
except ImportError as e:
    print(f"錯誤：缺少側視圖分析所需的模組，請確保這些檔案存在：{e}")
    sys.exit(1)


def return_score(score):
    sys.exit(int(score))

MODE_SIDE = 0  # 0 = 階梯, 1 = 金字塔

# ====================================================================
# === 俯視圖 (TOP View) 分析函數
# ====================================================================

CONF_TOP = 0.8
CROP_RATIO = 0.5

def analyze_image_top(frame, model, initial_get_point=2):
    """
    分析俯視圖 (TOP View) 影像，檢查旋轉和偏移。
    返回: (cropped_frame, summary_string, final_score)
    """
    H, W, _ = frame.shape
    crop_w, crop_h = int(W * CROP_RATIO), int(H * CROP_RATIO)
    x1 = (W - crop_w) // 2
    y1 = (H - crop_h) // 2
    x2, y2 = x1 + crop_w, y1 + crop_h

    cropped = frame[y1:y2, x1:x2].copy()

    results = model.predict(source=cropped, conf=CONF_TOP, verbose=False)
    masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []

    centers = []
    max_mask_side = 0
    rotate_ok_list = []
    GET_POINT = initial_get_point

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

        cv2.putText(
            cropped,
            f"std_x = {std_x:.2f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 100, 100),
            2,
        )
        cv2.putText(
            cropped,
            f"std_y = {std_y:.2f}",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 100, 255),
            2,
        )
        cv2.putText(
            cropped,
            f"threshold = {threshold:.2f}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 255, 100),
            2,
        )

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
# === 側視圖 (SIDE View) 分析函數
# ====================================================================


CONF_SIDE = 0.8
GAP_THRESHOLD_RATIO = 1.05

def analyze_image_side(IMG_PATH, model):
    """
    分析側視圖 (SIDE View) 影像，檢查間隙和結構。
    返回: (annotated_frame, score_side)
    """
    frame = cv2.imread(IMG_PATH)
    if frame is None:
        raise ValueError(f"讀不到圖片：{IMG_PATH}")

    annotated_frame = frame.copy()

    results = model.predict(source=frame, conf=CONF_SIDE, verbose=False)
    r0 = results[0]

    masks = r0.masks.data.cpu().numpy() if r0.masks is not None else []
    boxes = r0.boxes.xyxy.cpu().numpy() if r0.boxes is not None else np.empty((0, 4))
    
    # ✅ 修正：直接從 boxes 計算質心，確保座標一致
    centroids = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centroids.append((cx, cy))

    SCORE = 2
    IS_GAP = False

    # ========== 繪製所有偵測到的框框 ==========
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Block {i+1}", (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ========== 繪製質心（先不繪製，避免遮擋） ==========
    # for i, (cx, cy) in enumerate(centroids):
    #     cv2.circle(annotated_frame, (int(cx), int(cy)), 8, (255, 0, 0), -1)
    #     cv2.putText(annotated_frame, f"C{i+1}", (int(cx)+10, int(cy)-10), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 空隙檢測
    bbox_widths = [b[2] - b[0] for b in boxes] if len(boxes) > 0 else []
    avg_width = np.mean(bbox_widths) if len(bbox_widths) > 0 else 1.0
    GAP_THRESHOLD = GAP_THRESHOLD_RATIO * avg_width

    if len(centroids) >= 2:
        gap_checker = CheckGap(gap_threshold=GAP_THRESHOLD, y_layer_threshold=30)
        gap_pairs = gap_checker.check(centroids)

        # ========== 繪製空隙 ==========
        if gap_pairs:
            IS_GAP = True
            print(f"偵測到空隙: {len(gap_pairs) // 2} 組")
            
            try:
                for pair in gap_pairs:
                    # gap_pairs 格式: ((x1, y1), (x2, y2), distance)
                    if isinstance(pair, tuple) and len(pair) >= 2:
                        point1, point2 = pair[0], pair[1]
                        
                        # 確保 point1 和 point2 是座標元組
                        if isinstance(point1, tuple) and isinstance(point2, tuple):
                            cx1, cy1 = float(point1[0]), float(point1[1])
                            cx2, cy2 = float(point2[0]), float(point2[1])
                            
                            # 繪製空隙連線
                            cv2.line(annotated_frame, (int(cx1), int(cy1)), (int(cx2), int(cy2)), 
                                    (0, 0, 255), 3)
                            
                            # 在中點標註 GAP
                            mid_x, mid_y = int((cx1 + cx2) / 2), int((cy1 + cy2) / 2)
                            cv2.putText(annotated_frame, "GAP", (mid_x, mid_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
            except Exception as e:
                print(f"繪製空隙時發生錯誤: {e}")
                import traceback
                traceback.print_exc()
            
            if MODE_SIDE == 0:
                SCORE = 1
        else:
            print("無空隙")
            if MODE_SIDE == 1:
                SCORE = 1

    # 分層並做模式判定
    grouper = LayerGrouping(layer_ratio=0.2)
    layers = grouper.group_by_y(centroids, boxes=boxes)

    # ========== 繪製分層（同時繪製質心）==========
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255)]
    
    for layer_idx, layer in enumerate(layers):
        color = colors[layer_idx % len(colors)]
        
        # layer 中的每個元素是質心座標
        for centroid in layer:
            if isinstance(centroid, tuple) and len(centroid) == 2:
                cx, cy = centroid
                
                # 找到這個質心對應的 box 索引
                for box_idx, (bcx, bcy) in enumerate(centroids):
                    if abs(bcx - cx) < 5 and abs(bcy - cy) < 5:  # 找到匹配的質心
                        if box_idx < len(boxes):
                            x1, y1, x2, y2 = map(int, boxes[box_idx])
                            # 繪製分層框框（粗框）
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                            # 繪製分層標籤
                            cv2.putText(annotated_frame, f"L{layer_idx+1}", (x1, y2+20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            # 繪製質心點（在框框中心）
                            cv2.circle(annotated_frame, (int(cx), int(cy)), 8, (255, 0, 0), -1)
                            # 繪製質心編號
                            cv2.putText(annotated_frame, f"C{box_idx+1}", (int(cx)+10, int(cy)-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            break

    if MODE_SIDE == 0:  # 階梯模式
        stair_checker = StairChecker()
        result, msg = stair_checker.check(layers)
        if not result:
            SCORE = 0
        
        # 顯示階梯檢測結果
        cv2.putText(annotated_frame, f"Stair: {msg}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
    elif MODE_SIDE == 1:  # 金字塔模式
        block_width = (
            (np.mean([b[2] - b[0] for b in boxes]) // 2) if len(boxes) > 0 else 0
        )
        pyramid_checker = PyramidCheck()
        is_pyramid, pyramid_msg = pyramid_checker.check_pyramid(layers, block_width, IS_GAP)
        if not is_pyramid:
            SCORE = 0
        
        # 顯示金字塔檢測結果
        cv2.putText(annotated_frame, f"Pyramid: {pyramid_msg}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ========== 顯示得分 ==========
    score_text = f"Score: {SCORE}/2"
    cv2.putText(annotated_frame, score_text, (10, annotated_frame.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

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
        MODEL_PATH = r"ch1-t3/toybrick.pt"
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
        
        # ✅ 儲存側視圖結果
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

        initial_score = 2
        analyzed_frame, summary, score_top = analyze_image_top(
            frame_top, yolo_model, initial_score
        )
        print(f"俯視圖 ({TOP_IMG_PATH}) 檢測結果: {summary}")
        print(f"俯視圖得分: {score_top}")

        # ✅ 儲存俯視圖結果
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