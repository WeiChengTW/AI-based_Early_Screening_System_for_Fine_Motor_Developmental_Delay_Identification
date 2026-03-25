from PaperDetector_yolo import PaperDetector_yolo
from BoxDistanceAnalyzer import BoxDistanceAnalyzer
from Draw_square import Draw_square

import cv2
import json
import sys
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def return_score(score):
    sys.exit(int(score))


if __name__ == "__main__":
    score = 0
    if len(sys.argv) <= 2:
        print("缺少參數，使用方式: python main.py <uid> <img_id>")
        return_score(score)

    uid = sys.argv[1]
    img_id = sys.argv[2]
    image_path = os.path.join("kid", uid, f"{img_id}.jpg")
    if not os.path.exists(image_path):
        print(f"找不到圖片: {image_path}")
        return_score(score)

    print(f"\n正在處理圖片: {image_path}")
    print("====使用 YOLO 提取紙張區域====")

    kid = None
    try:
        json_path = BASE_DIR.parent / "px2cm.json"
        with open(json_path, "r") as f:
            data = json.load(f)
            pixel_per_cm = data.get("pixel_per_cm", 19.597376925845985)

        detector = PaperDetector_yolo(image_path)
        detector.detect_paper_by_yolo()
        if detector.result is not None:
            detected_path = os.path.join("kid", uid, f"{img_id}_detected.jpg")
            cv2.imwrite(detected_path, detector.result)
            print(f"偵測框圖片已儲存: {detected_path}")

        detector_path = None
        if detector.original is not None:
            region = detector.extract_paper_region()
            if region is not None:
                detector_path = detector.save_results()

        print("====使用 object mask + ArUco 紅框評分====")
        if detector_path:
            if detector.object_mask_points_warped is None:
                print("原圖未偵測到 object mask，無法進行方形評分")
                detector_path = None

        if detector_path:
            draw_result = Draw_square(detector_path)
            if draw_result is not None:
                D_sq_path, black_corners_int = draw_result
                if D_sq_path is not None:
                    analyzer = BoxDistanceAnalyzer(
                        box1=black_corners_int,
                        image_path=detector_path,
                        mask_points=detector.object_mask_points_warped,
                        largest_mask_contour=detector.object_mask_largest_contour_warped,
                    )
                    result = analyzer.analyze(pixel_per_cm=pixel_per_cm)
                    if result is not None:
                        result_img, kid = result
                        result_path = os.path.join("kid", uid, f"{img_id}_result.jpg")
                        cv2.imwrite(result_path, result_img)

        if kid is not None:
            if kid < 0.6:
                print(f"kid = {kid:.2f}, score = 2")
                score = 2
            elif kid < 1.2:
                print(f"kid = {kid:.2f}, score = 1")
                score = 1
            else:
                print(f"kid = {kid:.2f}, score = 0")
                score = 0
        else:
            print("裁切或距離分析失敗，score = 0")
    except Exception as e:
        print(f"流程執行失敗: {e}")

    return_score(score)
