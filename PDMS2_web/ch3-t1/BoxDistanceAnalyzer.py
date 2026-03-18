import cv2
import numpy as np


def BoxDistanceAnalyzer(img_path=None, output_path="ch3-t1\result"):
    img = cv2.imread(img_path)
    if img is None:
        print("讀取圖片失敗，請確認檔案路徑正確！")
        return

    # 先偵測最大輪廓並畫黃線
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blur, 20, 100)
    # cv2.imshow("edges", edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(img, [max_contour], -1, (0, 255, 255), 1)  # 黃色線

    # 偵測 ArUco marker
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(img)

    if ids is None or len(ids) == 0:
        raise ValueError("找不到 ArUco marker")

    # ArUco 實際邊長（公分）
    aruco_length_cm = 2.0
    if corners:
        # 對每個 ArUco 畫圓並計算中心到黃色邊框的最短最長距離
        for marker_corners in corners:
            pts = marker_corners[0]  # shape: (4,2)
            # 計算四邊長度
            side_lengths = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
            avg_side_px = np.mean(side_lengths)
            pixel_per_cm = avg_side_px / aruco_length_cm
            # 中心點
            center_x = int(np.mean(pts[:, 0]))
            center_y = int(np.mean(pts[:, 1]))
            center = np.array([center_x, center_y])

            # 計算中心點到黃色邊框(最大輪廓)的所有點的距離
            contour_points = max_contour.reshape(-1, 2)
            dists = np.linalg.norm(contour_points - center, axis=1)
            min_idx = np.argmin(dists)
            max_idx = np.argmax(dists)
            min_dist = dists[min_idx]
            max_dist = dists[max_idx]
            min_dist_cm = min_dist / pixel_per_cm
            max_dist_cm = max_dist / pixel_per_cm

            # 畫最短線（紅色）
            min_point = tuple(contour_points[min_idx])
            cv2.line(img, (center_x, center_y), min_point, (0, 0, 255), 2)
            # 畫最長線（藍色）
            max_point = tuple(contour_points[max_idx])
            cv2.line(img, (center_x, center_y), max_point, (255, 0, 0), 2)

            # 左上角顯示距離（紅字、藍字）
            cv2.putText(
                img,
                f"min: {min_dist_cm:.2f}cm",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                img,
                f"max: {max_dist_cm:.2f}cm",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )

            name = img_path.split("\\")[-1].split("_")[0]
            path = f"{output_path}/{name}.png"
            cv2.imwrite(path, img)

        print(f"結果已儲存為 '{path}'")
        print(
            f"ArUco中心到黃色邊框\n最短距離: {min_dist_cm:.2f}cm, 最長距離: {max_dist_cm:.2f}cm"
        )
        return img, min_dist_cm, max_dist_cm
    else:
        print("未偵測到任何 ArUco marker")
        return


if __name__ == "__main__":
    path = r"extracted\img1_extracted_paper.jpg"
    path = BoxDistanceAnalyzer(path)
