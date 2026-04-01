import cv2
import numpy as np
from Draw_square import Draw_square
import os

class BoxDistanceAnalyzer:
    def __init__(self, box1=None, image_path=None):
        self.box1 = box1
        self.image_path = image_path
        self.box2 = None

    def detect_main_contour_points(self, image_path, show_debug=False):
        """
        只取來自面積大於1000的輪廓的邊緣點
        """
        img = image_path
        if img is None:
            print(f"無法讀取圖片: {image_path}")
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 20, 100, apertureSize=3)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        points = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 3000:
                continue
            contour_points = contour.reshape(-1, 2)
            points.append(contour_points)
        if points:
            points = np.concatenate(points, axis=0)
        else:
            points = np.empty((0, 2), dtype=int)

        if show_debug:
            debug_image = img.copy()
            for pt in points:
                cv2.circle(debug_image, tuple(pt), 1, (0, 255, 255), 1)
            cv2.imshow("debug_edges_points", debug_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return points

    def detect_largest_contour(self, image_path, area_threshold=3000):
        """回傳最大外部輪廓的有序點集 (N,2)；若無則回傳 None。"""
        img = image_path
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 20, 100, apertureSize=3)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        # 取面積最大且超過門檻者
        largest = None
        largest_area = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area > area_threshold and area > largest_area:
                largest = c
                largest_area = area
        if largest is None:
            return None
        return largest.reshape(-1, 2)

    @staticmethod
    def _closest_point_on_segment(p, a, b):
        """取得點 p 到線段 ab 的最近點（垂足，若超出端點則回端點）。
        p, a, b: np.array([x, y])
        回傳: (closest_point, distance)
        """
        ap = p - a
        ab = b - a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq == 0.0:
            # 退化為單點
            return a, float(np.linalg.norm(p - a))
        t = float(np.dot(ap, ab)) / ab_len_sq
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return proj, float(np.linalg.norm(p - proj))

    def _min_distance_to_polygon_edges(self, p, poly):
        """計算點 p 到多邊形 poly 的各邊的最短距離與對應最近點。
        poly: (N,2) 順時針或逆時針
        回傳: (min_dist, closest_point)
        """
        min_dist = float("inf")
        closest = None
        n = len(poly)
        for i in range(n):
            a = poly[i].astype(float)
            b = poly[(i + 1) % n].astype(float)
            foot, d = self._closest_point_on_segment(p.astype(float), a, b)
            if d < min_dist:
                min_dist = d
                closest = foot
        return min_dist, closest

    def draw_main_contour(self, img, color=(0, 255, 255), thickness=1):
        """
        在 img 上畫出最大輪廓的邊緣點（黃色）
        """
        points = self.detect_main_contour_points(self.image_path)
        if points is not None:
            for pt in points:
                cv2.circle(img, tuple(pt), 1, color, thickness)
        return img

    @staticmethod
    def _point_to_line_distance_and_foot(p, a, b):
        """計算點 p 到由 a->b 定義之直線(無限延伸)的最短距離與垂足。
        p, a, b: np.array([x, y])
        回傳: (distance, foot_point)
        """
        ab = b - a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq == 0.0:
            # 若線段退化為點，則距離為 p 到 a 的距離，垂足視為 a
            return float(np.linalg.norm(p - a)), a.copy()
        t = float(np.dot(p - a, ab)) / ab_len_sq
        foot = a + t * ab
        dist = float(np.linalg.norm(p - foot))
        return dist, foot

    def max_distance_between_boxes(self, box1, box2):
        """
        box1, box2: shape (4,2) numpy array
        計算兩個四邊形所有點對的最大距離
        """
        max_dist = 0
        max_pair = (None, None)
        for pt1 in box1:
            for pt2 in box2:
                dist = np.linalg.norm(pt1 - pt2)
                if dist > max_dist:
                    max_dist = dist
                    max_pair = (tuple(pt1), tuple(pt2))
        return max_dist, max_pair

    def analyze(self, pixel_per_cm=None, out_path="result"):
        if pixel_per_cm is None:
            print("未提供每公分像素數")
            return
        if self.box1 is None or self.image_path is None:
            print("box1 或 image_path 尚未設定")
            return
        # 偵測最大輪廓的邊緣點（黃色點集合）
        main_points = self.detect_main_contour_points(self.image_path, show_debug=False)
        if main_points is None:
            print("未偵測到任何邊緣")
            return
        # 內凹與外凸皆支援：對所有黃色點計算到紅框的有號距離，取絕對值最大者
        red_poly = self.box1.astype(np.int32)
        red_contour = red_poly.reshape(-1, 1, 2)

        best_abs_signed = 0.0
        best_sign = 0.0
        best_yellow = None
        best_foot = None
        for y in main_points:
            # 有號距離：>0 在多邊形內，<0 在外
            signed_d = cv2.pointPolygonTest(
                red_contour, (float(y[0]), float(y[1])), True
            )
            # 無號最近距離與垂足
            d_unsigned, foot = self._min_distance_to_polygon_edges(
                np.array(y, dtype=float), red_poly
            )
            if abs(signed_d) > best_abs_signed:
                best_abs_signed = abs(signed_d)
                best_sign = 1.0 if signed_d >= 0 else -1.0
                best_yellow = tuple(map(int, y))
                best_foot = tuple(map(int, foot))
        direction = "內凹" if best_sign >= 0 else "外凸"
        print(f"{direction}方向的最大最短距離: {(best_abs_signed/pixel_per_cm):.2f}")

        img = self.image_path
        img_draw = img.copy()
        # 在左上角寫上最大最短距離（單位：cm）
        text = f"{(best_abs_signed/pixel_per_cm):.2f} cm"
        cv2.putText(
            img_draw,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        # 畫 box1（紅色）
        cv2.polylines(
            img_draw,
            [self.box1.astype(np.int32)],
            isClosed=True,
            color=(0, 0, 255),
            thickness=1,
        )
        # 畫最大輪廓邊緣點（黃色）
        self.draw_main_contour(img_draw, color=(0, 255, 255), thickness=1)
        # 畫最長距離線段（綠色）
        if best_yellow is not None and best_foot is not None:
            cv2.line(img_draw, best_yellow, best_foot, (0, 255, 0), 1)
            cv2.circle(img_draw, best_yellow, 2, (255, 0, 0), -1)
            cv2.circle(img_draw, best_foot, 2, (255, 0, 255), -1)
        # 進一步：計算「紅框四角到黃框線」的最短距離，最後取四角中之最大者
        yellow_contour = self.detect_largest_contour(self.image_path)
        if yellow_contour is not None and len(yellow_contour) >= 2:
            # 將黃框視為封閉多邊形，逐邊做最短距離
            n_y = len(yellow_contour)
            best_corner_idx = -1
            best_corner_point = None
            best_corner_foot = None
            best_corner_min_dist = -1.0  # 紀錄四角各自最短距離中的最大者

            for idx, corner in enumerate(red_poly):
                p = corner.astype(float)
                # 角點到黃框線的最短距離
                min_d = float("inf")
                min_foot = None
                for i in range(n_y):
                    a = yellow_contour[i].astype(float)
                    b = yellow_contour[(i + 1) % n_y].astype(float)
                    foot, d = self._closest_point_on_segment(p, a, b)
                    if d < min_d:
                        min_d = d
                        min_foot = foot
                # 更新四角中的最大者（最短的最長距離）
                if min_d > best_corner_min_dist:
                    best_corner_min_dist = min_d
                    best_corner_point = tuple(map(int, corner))
                    best_corner_foot = tuple(map(int, min_foot))
                    best_corner_idx = idx

            if best_corner_point is not None and best_corner_foot is not None:
                # 視覺化：青色線段 + 標記點 + 文字
                cv2.line(
                    img_draw, best_corner_point, best_corner_foot, (255, 255, 0), 1
                )
                cv2.circle(img_draw, best_corner_point, 3, (0, 165, 255), -1)
                cv2.circle(img_draw, best_corner_foot, 3, (0, 255, 255), -1)

                text2 = f"{(best_corner_min_dist/pixel_per_cm):.2f} cm"
                cv2.putText(
                    img_draw,
                    text2,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                print(
                    f"紅框四角到黃框線之最短的最長距離: {(best_corner_min_dist/pixel_per_cm):.2f}"
                )
        # name = self.image_path.split("\\")[-1].split("_")[0]
        # path = os.path.join("ch3-t2", out_path, "ch3-t2", "_result.png")

        # cv2.imwrite(path, img_draw)
        # print(f"最長距離線段與方框、最大輪廓邊緣已畫出並存檔於 {path}")
        return img_draw, max(
            best_abs_signed / pixel_per_cm, best_corner_min_dist / pixel_per_cm
        )


if __name__ == "__main__":
    detector_path = r"extracted\img2_extracted_paper.jpg"
    D_sq_path, black_corners_int = Draw_square(detector_path)

    analyzer = BoxDistanceAnalyzer(box1=black_corners_int, image_path=detector_path)

    analyzer.analyze(pixel_per_cm=19.597376925845985)
