import cv2
import numpy as np
import os


def detect_aruco_and_draw_quarter_a4(image_path, output_path=None, aruco_size_cm=2.8):
    """
    檢測ArUco標記，畫出1/4 A4大小的矩形（綠色框線），並計算比例尺
    """
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取圖片: {image_path}")
        return None, None, None

    # 複製原圖用於繪製結果
    result_image = image.copy()

    # 初始化ArUco字典和檢測器 (適配不同OpenCV版本)
    try:
        # OpenCV 4.7+
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(aruco_dict)
        corners, ids, _ = detector.detectMarkers(image)
    except AttributeError:
        try:
            # OpenCV 4.0-4.6
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(
                image, aruco_dict, parameters=aruco_params
            )
        except AttributeError:
            # 更舊版本或其他情況
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict)

    if len(corners) > 0:
        # 使用第一個檢測到的ArUco標記
        aruco_corners = corners[0][0]

        # 計算ArUco的中心點和邊長（像素）
        center_x = np.mean(aruco_corners[:, 0])
        center_y = np.mean(aruco_corners[:, 1])

        # 計算ArUco標記的邊長（使用前兩個頂點的距離）
        aruco_pixel_size = np.linalg.norm(aruco_corners[1] - aruco_corners[0])

        # 計算像素到厘米的比例尺
        pixels_per_cm = aruco_pixel_size / aruco_size_cm

        print(f"ArUco中心點: ({center_x:.1f}, {center_y:.1f})")
        print(f"ArUco像素邊長: {aruco_pixel_size:.1f} pixels")
        print(f"比例尺: {pixels_per_cm:.2f} pixels/cm")

        # A4紙尺寸：21cm x 29.7cm，1/4 A4約為 10.5cm x 14.85cm
        quarter_a4_width_cm = 10.5
        quarter_a4_height_cm = 14.85

        # 轉換為像素尺寸
        quarter_a4_width_px = int(quarter_a4_width_cm * pixels_per_cm)
        quarter_a4_height_px = int(quarter_a4_height_cm * pixels_per_cm)

        print(f"1/4 A4尺寸: {quarter_a4_width_px} x {quarter_a4_height_px} pixels")

        # 以 ArUco 的兩條邊作為基底向量，確保方框四邊與 ArUco 邊平行
        width_vec = aruco_corners[1] - aruco_corners[0]
        height_vec = aruco_corners[3] - aruco_corners[0]

        width_len = np.linalg.norm(width_vec)
        height_len = np.linalg.norm(height_vec)
        if width_len < 1e-6 or height_len < 1e-6:
            print("ArUco 邊長異常，無法建立標準方框")
            return None, None, result_image

        width_dir = width_vec / width_len
        height_dir = height_vec / height_len

        half_width_vec = width_dir * (quarter_a4_width_px / 2.0)
        half_height_vec = height_dir * (quarter_a4_height_px / 2.0)
        center = np.array([center_x, center_y], dtype=np.float32)

        top_left = center - half_width_vec - half_height_vec
        top_right = center + half_width_vec - half_height_vec
        bottom_right = center + half_width_vec + half_height_vec
        bottom_left = center - half_width_vec + half_height_vec

        quarter_a4_corners = np.array(
            [top_left, top_right, bottom_right, bottom_left], dtype=np.int32
        )

        # 繪製ArUco標記（紅色框線）
        cv2.aruco.drawDetectedMarkers(result_image, corners, ids)

        # 繪製1/4 A4矩形（綠色框線）
        cv2.polylines(
            result_image, [quarter_a4_corners], True, (0, 255, 0), 1
        )  # 綠色，線寬1

        # 添加文字說明
        cv2.putText(
            result_image,
            f"Scale: {pixels_per_cm:.2f} px/cm",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            result_image,
            f"ArUco: {aruco_size_cm}cm",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            result_image,
            f"1/4 A4: {quarter_a4_width_cm}x{quarter_a4_height_cm}cm",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        print(f"✓ 成功檢測ArUco並繪製1/4 A4矩形")

        # 保存結果圖片
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"結果已保存到: {output_path}")

        return quarter_a4_corners, pixels_per_cm, result_image
    else:
        print("未檢測到ArUco標記")
        return None, None, result_image


def main():
    """測試函數"""
    input_dir = "img"
    output_dir = "result"

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 處理所有圖片
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"step2_{filename}")

            print(f"\n處理圖片: {filename}")
            corners, scale, result = detect_aruco_and_draw_quarter_a4(
                input_path, output_path
            )

            if corners is not None:
                print(f"✓ 成功檢測ArUco並繪製1/4 A4矩形")
            else:
                print("✗ 未能檢測到ArUco標記")


if __name__ == "__main__":
    main()
