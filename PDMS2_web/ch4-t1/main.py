import cv2
import glob
import os
import numpy as np
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def return_score(score):
    sys.exit(int(score))


# 找最大四邊形輪廓
def get_largest_quadrilateral_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 200)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    max_quad, max_area = None, 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area, max_quad = area, approx
    return max_quad


# 合併輪廓
def merge_close_contours(contours, threshold=10):
    merged, used, centers = [], [False] * len(contours), []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        else:
            centers.append((0, 0))
    for i, cnt1 in enumerate(contours):
        if used[i]:
            continue
        group = [cnt1]
        used[i] = True
        for j, cnt2 in enumerate(contours):
            if i == j or used[j]:
                continue
            if np.linalg.norm(np.array(centers[i]) - np.array(centers[j])) < threshold:
                group.append(cnt2)
                used[j] = True
        merged.append(np.vstack(group))
    return merged


# 角點排序（環狀）
def sort_corners(pts):
    pts = np.array(pts)
    center = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return [tuple(p) for p in pts[np.argsort(ang)]]


# 主流程
def analyze_image(img_path):
    # 確保結果目錄存在
    # result_dir = "PDMS2_web/ch4-t1/result"
    result_dir = os.path.join("PDMS2_web", 'ch4-t1', 'result')
    os.makedirs(result_dir, exist_ok=True)

    try:
        json_path = BASE_DIR.parent / "px2cm.json"
        with open(json_path, "r") as f:
            data = json.load(f)
            px2cm = data["pixel_per_cm"]
    except FileNotFoundError:
        px2cm = 47.4416628993705  # 預設值
    cm_per_pixel = 1 / px2cm
    actual_length_cm = 7.5

    img_color_raw = cv2.imread(img_path)
    if img_color_raw is None:
        raise ValueError("無法讀取圖片")

    max_contour = get_largest_quadrilateral_contour(img_color_raw)

    if max_contour is None:
        hsv = cv2.cvtColor(img_color_raw, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 80, 255]))
        edges_white = cv2.Canny(white_mask, 80, 200)
        dilated_white = cv2.dilate(edges_white, np.ones((5, 5), np.uint8), iterations=1)
        contours_white, _ = cv2.findContours(
            dilated_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_white = [cnt for cnt in contours_white if cv2.contourArea(cnt) > 500]
        if not contours_white:
            return 0
        merged_white = merge_close_contours(contours_white, threshold=5)
        max_contour = max(merged_white, key=cv2.contourArea)
        peri = cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, 0.05 * peri, True)
        if len(approx) != 4:
            return 0
    else:
        peri = cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, 0.05 * peri, True)
        if len(approx) != 4:
            return 0

    # 計算短邊
    corners = [np.array(p) for p in sort_corners([pt[0] for pt in approx])]
    edge_pts = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]

    edges_info = []
    for p1, p2 in edge_pts:
        length_px = np.linalg.norm(p1 - p2)
        edges_info.append((p1, p2, length_px * cm_per_pixel))

    edges_info.sort(key=lambda x: x[2])
    short_edges = edges_info[:2]

    s1, s2 = short_edges[0][2], short_edges[1][2]
    d1, d2 = abs(s1 - actual_length_cm), abs(s2 - actual_length_cm)

    if d1 <= 0.3 and d2 <= 0.3:
        score = 2
    elif d1 <= 1.2 and d2 <= 1.2:
        score = 1
    else:
        score = 0

    # 儲存分析結果
    result_path = os.path.join("PDMS2_web/ch4-t1/result", os.path.basename(img_path))
    result_img = img_color_raw.copy()

    # 繪製偵測到的邊界和詳細資訊
    if max_contour is not None:
        # 繪製邊界
        cv2.drawContours(result_img, [max_contour], -1, (0, 255, 0), 2)

        # 在圖片上標示每個邊的長度
        for i, (p1, p2, length_cm) in enumerate(edges_info):
            center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(
                result_img,
                f"{length_cm:.1f}cm",
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # 在圖片左上角添加詳細資訊
        info_text = [
            f"得分: {score}",
            f"比例尺: {px2cm:.2f} px/cm",
            "邊長(由短到長):",
            *[
                f"邊 {i+1}: {length:.1f}cm"
                for i, (_, _, length) in enumerate(edges_info)
            ],
        ]

        for i, text in enumerate(info_text):
            cv2.putText(
                result_img,
                text,
                (20, 30 + i * 30),  # 位置從左上角開始，每行間隔30像素
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),  # 黃色文字
                2,
            )

    # 儲存結果圖片
    # result_path = rf"kid\{uid}\{img_id}_result.jpg"
    result_path = os.path.join('kid', uid, f"{img_id}_result.jpg")
    cv2.imwrite(result_path, result_img)
    return score


if __name__ == "__main__":
    if len(sys.argv) > 2:
        uid = sys.argv[1]
        img_id = sys.argv[2]
        # image_path = rf"kid\{uid}\{img_id}.jpg"
        image_path = os.path.join('kid', uid, f"{img_id}.jpg")

        score = analyze_image(image_path)
        print(f"score : {score}")
        return_score(score)

    # image_path = rf"PDMS2_web\ch4-t1\ch4-t1.jpg"
    # image_path = os.path.join('kid', uid, f"{img_id}.jpg")
    score = analyze_image(image_path)
    print(f"score : {score}")
