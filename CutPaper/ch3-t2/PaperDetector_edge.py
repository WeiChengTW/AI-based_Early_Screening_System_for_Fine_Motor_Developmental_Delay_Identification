import cv2
import numpy as np
import os


class PaperDetector_edges:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original = None
        self.result = None
        self.contour = None
        self.paper_region = None

    def detect_paper_by_color(self):
        image = cv2.imread(self.image_path)
        if image is None:
            print("無法讀取圖像檔案")
            return None, None, None

        # 轉換為灰度圖像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Canny邊緣檢測
        edges = cv2.Canny(blurred, 20, 100, apertureSize=3)

        # 形態學操作來連接斷開的邊緣
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)

        # cv2.imshow("edges", edges)
        # 尋找輪廓
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 找到最大面積的輪廓作為紙張
        paper_contour = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            # 設定最小面積閾值，過濾掉太小的輪廓
            if area > 1000:
                if area > max_area:
                    max_area = area
                    paper_contour = contour

        if paper_contour is not None:
            # 多邊形近似，簡化輪廓
            epsilon = 0.02 * cv2.arcLength(paper_contour, True)
            paper_contour = cv2.approxPolyDP(paper_contour, epsilon, True)

        result_image = image.copy()
        if paper_contour is not None:
            cv2.drawContours(result_image, [paper_contour], -1, (0, 0, 255), 3)
            print(f"檢測到紙張區域，面積: {max_area}")
            print(f"紙張輪廓點數: {len(paper_contour)}")
        else:
            print("未檢測到明顯的紙張區域")

        # 可選：顯示邊緣檢測結果
        # cv2.imshow("edges", edges)
        self.original = image
        self.result = result_image
        self.contour = paper_contour
        return image, result_image, paper_contour

    def show_results(self):
        if self.original is None or self.result is None:
            print("尚未有檢測結果")
            return
        # 背景執行時不可阻塞等待鍵盤事件
        print("已略過 show_results 視窗顯示（背景任務模式）")

    def extract_paper_region(self):
        if self.original is None or self.contour is None:
            print("無法提取紙張區域")
            return None
        image = self.original
        contour = self.contour
        if len(contour) != 4:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            contour = np.intp(box)
        points = contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype=np.float32)
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        width_a = np.linalg.norm(rect[0] - rect[1])
        width_b = np.linalg.norm(rect[2] - rect[3])
        max_width = max(int(width_a), int(width_b))
        height_a = np.linalg.norm(rect[0] - rect[3])
        height_b = np.linalg.norm(rect[1] - rect[2])
        max_height = max(int(height_a), int(height_b))
        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

        # 邊框往內修10像素
        padding = 10
        h, w = warped.shape[:2]
        if h > 2 * padding and w > 2 * padding:  # 確保圖像足夠大才進行裁切
            warped = warped[padding : h - padding, padding : w - padding]

        self.paper_region = warped
        return warped

    def save_results(self):
        # if self.result is not None:
        #     cv2.imwrite(f"{self.image_path}detected_paper.jpg", self.result)
        name = self.image_path.split(os.sep)[-1].split(".")[0]
        result_path = os.path.join("ch3-t2", "extracted", f"{name}_extracted_paper.jpg")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        if self.paper_region is not None:
            ok = cv2.imwrite(result_path, self.paper_region)
            if not ok:
                raise IOError(f"無法寫入檔案: {result_path}")
        print(f"結果已儲存為 '{result_path}'")
        return result_path


if __name__ == "__main__":
    image_path = r"raw\img1.jpg"
    detector = PaperDetector_edges(image_path)
    detector.detect_paper_by_color()
    if detector.original is not None:
        detector.show_results()
        region = detector.extract_paper_region()
        if region is not None:
            height, width = region.shape[:2]
            if width > 600:
                scale = 600 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                region = cv2.resize(region, (new_width, new_height))
            # cv2.imshow("提取的紙張區域", region)
            # print("按任意鍵關閉視窗...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            detector.save_results()
