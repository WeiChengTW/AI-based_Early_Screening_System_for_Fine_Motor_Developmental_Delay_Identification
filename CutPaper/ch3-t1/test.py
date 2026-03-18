import cv2
import numpy as np

# 設定圖片路徑
image_path = "a4_aruco.png"  # 你可以改成你要測試的圖片

# 載入圖片
img = cv2.imread(image_path)
if img is None:
    raise ValueError("無法讀取圖片")

# 偵測 ArUco marker
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corners, ids, _ = detector.detectMarkers(img)

if ids is None or len(ids) == 0:
    raise ValueError("找不到 ArUco marker")

# ArUco 實際邊長（公分）
aruco_length_cm = 2.0

# 對每個 ArUco 畫圓
for marker_corners in corners:
    pts = marker_corners[0]  # shape: (4,2)
    # 計算四邊長度
    side_lengths = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    avg_side_px = np.mean(side_lengths)
    pixel_per_cm = avg_side_px / aruco_length_cm
    # 中心點
    center_x = int(np.mean(pts[:, 0]))
    center_y = int(np.mean(pts[:, 1]))
    radius_cm = 4
    radius_px = int(radius_cm * pixel_per_cm)
    cv2.circle(img, (center_x, center_y), radius_px, (0, 0, 255), 2)  # 紅色細圓

# 顯示與儲存
cv2.imshow("ArUco with 4cm Circle", img)
cv2.imwrite("aruco_with_4cm_circle.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
