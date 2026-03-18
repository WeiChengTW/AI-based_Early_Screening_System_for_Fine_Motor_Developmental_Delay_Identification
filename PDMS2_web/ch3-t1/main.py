from PaperDetector_edge import PaperDetector_edges
from BoxDistanceAnalyzer import BoxDistanceAnalyzer
import cv2
import sys


def return_score(score):
    sys.exit(int(score))


def crop_center_region(image, ratio=0.5):
    """Crop the center area by width/height ratio (0~1]."""
    if image is None:
        return None

    ratio = max(0.01, min(1.0, float(ratio)))
    h, w = image.shape[:2]
    crop_w = int(w * ratio)
    crop_h = int(h * ratio)

    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    return image[y1:y2, x1:x2]


def draw_crop_preview(image, ratio=0.5):
    """Draw crop rectangle on image for preview."""
    if image is None:
        return None

    ratio = max(0.01, min(1.0, float(ratio)))
    h, w = image.shape[:2]
    crop_w = int(w * ratio)
    crop_h = int(h * ratio)

    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    preview = image.copy()
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        preview,
        f"Crop: {int(ratio * 100)}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return preview


if __name__ == "__main__":
    # 檢查是否有傳入 id 參數
    if len(sys.argv) > 2:
        # 使用傳入的 uid 和 id 作為圖片路徑
        uid = sys.argv[1]
        img_id = sys.argv[2]
        try:
            crop_percent = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
        except ValueError:
            crop_percent = 60.0
        crop_ratio = max(1.0, min(100.0, crop_percent)) / 100.0
        score = 0
        detector_path = None
        min_dist_cm = None
        max_dist_cm = None
        # uid = "lull222"
        # img_id = "ch3-t1"
        image_path = rf"kid\{uid}\{img_id}.jpg"
        # image_path = rf"C:\Users\chang\Downloads\web\kid\lull222\ch3-t1.jpg"
        # 提取紙張區域
        print(f"\n正在處理圖片: {image_path}")
        print("====提取紙張區域====")
        detector = PaperDetector_edges(image_path)
        detector.detect_paper_by_color()
        if detector.original is not None:

            region = detector.extract_paper_region()
            if region is not None:
                preview = draw_crop_preview(region, crop_ratio)
                if preview is not None:
                    cv2.imshow("Crop Preview", preview)
                region = crop_center_region(region, crop_ratio)
                detector.paper_region = region
                height, width = region.shape[:2]
                if width > 600:
                    scale = 600 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    region = cv2.resize(region, (new_width, new_height))
                # cv2.imshow("提取的紙張區域", region)
                detector_path = detector.save_results()
                detector.show_results()

            if detector_path:

                result_img, min_dist_cm, max_dist_cm = BoxDistanceAnalyzer(
                    detector_path
                )
                result_path = rf"kid\{uid}\{img_id}_result.jpg"
                cv2.imwrite(result_path, result_img)

            if min_dist_cm is not None and max_dist_cm is not None:
                correct = 4.0
                kid = max(abs(min_dist_cm - correct), abs(max_dist_cm - correct))
                if kid < 0.6:
                    print(f"kid = {kid:.2f}, score = 2")
                    score = 2
                elif kid < 1.2:
                    print(f"kid = {kid:.2f}, score = 1")
                    score = 1
                else:
                    print(f"kid = {kid:.2f}, score = 0")
                    score = 0
        return_score(score)
