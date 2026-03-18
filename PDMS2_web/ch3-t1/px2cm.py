# get_pixel_per_cm.py

import cv2
import numpy as np
import json


def get_pixel_per_cm_from_a4(image_path, real_width_cm=29.7, show_debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("❌ 圖片讀取失敗，請確認路徑正確")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    a4_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(a4_contour, True)
    approx = cv2.approxPolyDP(a4_contour, epsilon, True)

    if len(approx) != 4:
        raise ValueError("❌ 無法偵測 A4 紙四邊形輪廓")

    if show_debug:
        debug_img = img.copy()
        cv2.drawContours(debug_img, [approx], -1, (0, 0, 255), 3)
        cv2.imshow("Detected A4 Contour", cv2.resize(debug_img, (800, 600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    pts = approx.reshape(4, 2)
    pts = sorted(pts, key=lambda p: p[0])  # 先左右
    left = sorted(pts[0:2], key=lambda p: p[1])
    right = sorted(pts[2:4], key=lambda p: p[1])
    tl, bl = left
    tr, br = right

    a4_pixel_width = np.linalg.norm(tr - tl)
    pixel_per_cm = a4_pixel_width / real_width_cm

    json_path = "px2cm.json"
    data = {"pixel_per_cm": pixel_per_cm, "image_path": image_path}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return pixel_per_cm, json_path


# ✅ 單獨執行這個檔案時顯示紙張輪廓
if __name__ == "__main__":
    image_path = r"ch3-t1\raw\img1.jpg"
    pixel_per_cm, _ = get_pixel_per_cm_from_a4(image_path, show_debug=True)
    print(f"每公分像素：{pixel_per_cm:.2f}")
