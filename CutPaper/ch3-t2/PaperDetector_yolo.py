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

        if weights_path is None:
            base_dir = Path(__file__).resolve().parents[1]
            weights_path = base_dir / "ch3-t1" / "models" / "best.pt"

        self.weights_path = str(weights_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.weights_path):
            self.model = YOLO(self.weights_path)
        else:
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")

    def detect_paper_by_yolo(self, conf=0.5, device=None):
        image = cv2.imread(self.image_path)
        if image is None:
            print("無法讀取圖像檔案")
            return None, None, None

        if device is None:
            device = "0" if torch.cuda.is_available() else "cpu"

        names = self.model.names if self.model is not None else {}
        paper_class_ids = [
            int(cls_id)
            for cls_id, cls_name in names.items()
            if "paper" in str(cls_name).lower()
        ]
        if not paper_class_ids:
            print("警告: 模型類別中找不到 'paper'")
            self.original = image
            self.result = image.copy()
            self.contour = None
            return image, self.result, None

        predict_kwargs = {
            "source": image,
            "conf": conf,
            "device": device,
            "verbose": False,
            "classes": paper_class_ids,
        }

        try:
            results = self.model.predict(**predict_kwargs)
        except Exception as e:
            if device != "cpu" and "cuda" in str(e).lower():
                print("GPU 不可用，切換到 CPU 重新嘗試")
                predict_kwargs["device"] = "cpu"
                results = self.model.predict(**predict_kwargs)
            else:
                raise

        result_image = image.copy()
        paper_contour = None

        if len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                max_conf_idx = int(np.argmax(result.boxes.conf.cpu().numpy()))
                box = result.boxes.xyxy[max_conf_idx].cpu().numpy()
                x1, y1, x2, y2 = box.astype(int)
                paper_contour = np.array(
                    [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32
                )

                cls_id = int(result.boxes.cls[max_conf_idx].item())
                cls_name = names.get(cls_id, str(cls_id))
                conf_score = result.boxes.conf[max_conf_idx].item()

                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    result_image,
                    f"{cls_name} {conf_score:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                print(f"檢測到紙張類別 '{cls_name}'，置信度: {conf_score:.2f}")

        self.original = image
        self.result = result_image
        self.contour = paper_contour
        return image, result_image, paper_contour

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

        # 往外擴一點，避免紙張被切太緊
        expand_ratio = 0.04
        center = np.mean(rect, axis=0)
        rect = center + (rect - center) * (1.0 + expand_ratio)
        h, w = image.shape[:2]
        rect[:, 0] = np.clip(rect[:, 0], 0, w - 1)
        rect[:, 1] = np.clip(rect[:, 1], 0, h - 1)

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

        self.paper_region = warped
        return warped

    def save_results(self):
        name = self.image_path.split(os.sep)[-1].split(".")[0]
        result_path = os.path.join("ch3-t2", "extracted", f"{name}_extracted_paper.jpg")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        if self.paper_region is not None:
            ok = cv2.imwrite(result_path, self.paper_region)
            if not ok:
                raise IOError(f"無法寫入檔案: {result_path}")
        print(f"結果已儲存為 '{result_path}'")
        return result_path
