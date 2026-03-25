import cv2
import numpy as np
import os
from pathlib import Path
import torch
from ultralytics import YOLO


class PaperDetector_yolo:
    def __init__(self, image_path, weights_path=None):
        self.image_path = image_path
        self.original = None
        self.result = None
        self.contour = None
        self.paper_region = None

        # 預設權重路徑（本地 models 目錄）
        if weights_path is None:
            ch3_t1_dir = Path(__file__).resolve().parent
            weights_path = ch3_t1_dir / "models" / "best.pt"

        self.weights_path = str(weights_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLO model."""
        if os.path.exists(self.weights_path):
            self.model = YOLO(self.weights_path)
        else:
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")

    def detect_paper_by_yolo(self, conf=0.5, device=None):
        """Detect paper using YOLO model."""
        image = cv2.imread(self.image_path)
        if image is None:
            print("無法讀取圖像檔案")
            return None, None, None

        if device is None:
            device = "0" if torch.cuda.is_available() else "cpu"

        # Run YOLO inference
        try:
            results = self.model.predict(
                source=image, conf=conf, device=device, verbose=False
            )
        except Exception as e:
            # macOS/CPU-only 環境常見 CUDA 裝置錯誤，改用 CPU 再試一次
            if device != "cpu" and "cuda" in str(e).lower():
                print("GPU 不可用，切換到 CPU 重新嘗試")
                results = self.model.predict(
                    source=image, conf=conf, device="cpu", verbose=False
                )
            else:
                raise

        result_image = image.copy()
        paper_contour = None

        if len(results) > 0:
            result = results[0]

            # Get detections
            if result.boxes is not None and len(result.boxes) > 0:
                # Get the box with highest confidence
                max_conf_idx = np.argmax(result.boxes.conf.cpu().numpy())
                box = result.boxes.xyxy[max_conf_idx].cpu().numpy()

                # Convert box to contour format
                x1, y1, x2, y2 = box.astype(int)
                paper_contour = np.array(
                    [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32
                )

                # Draw on result image
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                print(
                    f"檢測到紙張，置信度: {result.boxes.conf[max_conf_idx].item():.2f}"
                )

        self.original = image
        self.result = result_image
        self.contour = paper_contour
        return image, result_image, paper_contour

    def show_results(self):
        if self.original is None or self.paper_region is None:
            print("尚未有檢測結果")
            return
        original = self.original
        result = self.paper_region
        height, width = original.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            original_resized = cv2.resize(original, (new_width, new_height))
        else:
            original_resized = original
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def extract_paper_region(self):
        """Extract paper region from the image using detected contour."""
        if self.original is None or self.contour is None:
            print("無法提取紙張區域")
            return None

        image = self.original
        contour = self.contour

        # If contour is not 4 points, fit a rectangle
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

        # Remove padding
        padding = 10
        h, w = warped.shape[:2]
        if h > 2 * padding and w > 2 * padding:
            warped = warped[padding : h - padding, padding : w - padding]

        self.paper_region = warped
        return warped

    def save_results(self):
        """Save extracted paper region."""
        name = self.image_path.split(os.sep)[-1].split(".")[0]
        result_path = os.path.join("ch3-t1", "extracted", f"{name}_extracted_paper.jpg")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        if self.paper_region is not None:
            cv2.imwrite(result_path, self.paper_region)
        print(f"結果已儲存為 '{result_path}'")
        return result_path


if __name__ == "__main__":
    image_path = r"raw\img1.jpg"
    detector = PaperDetector_yolo(image_path)
    detector.detect_paper_by_yolo()
    if detector.original is not None:
        region = detector.extract_paper_region()
        if region is not None:
            height, width = region.shape[:2]
            if width > 600:
                scale = 600 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                region = cv2.resize(region, (new_width, new_height))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            detector.save_results()
            detector.show_results()
