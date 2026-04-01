from PaperDetector_edge import PaperDetector_edges

from BoxDistanceAnalyzer import BoxDistanceAnalyzer

# from BoxDistanceAnalyzer_2 import BoxDistanceAnalyzer
from Draw_square import Draw_square

import cv2
import json
import sys
import os
from pathlib import Path

# base directory for resolving relative resources (parent of this file's directory)
BASE_DIR = Path(__file__).resolve().parent


def return_score(score):
    sys.exit(int(score))


if __name__ == "__main__":
    # 檢查是否有傳入 id 參數
    if len(sys.argv) > 2:
        # 使用傳入的 uid 和 id 作為圖片路徑
        uid = sys.argv[1]
        img_id = sys.argv[2]
        # uid = "lull222"
        # img_id = "ch3-t1"
        # image_path = rf"kid\{uid}\{img_id}.jpg"
        image_path = os.path.join('kid', uid, f'{img_id}.jpg')
        # img = 1
        # for img in range(1, 5):
        #     image_path = rf"raw\img{img}.jpg"
        # _, json_path = get_pixel_per_cm_from_a4(rf"a4.jpg", show_debug=False)
        json_path = os.path.join(BASE_DIR.parent, "px2cm.json")
        if json_path is not None:
            with open(json_path, "r") as f:
                data = json.load(f)
                pixel_per_cm = data.get("pixel_per_cm", 19.597376925845985)

        # 提取紙張區域
        print(f"\n正在處理圖片: {image_path}")
        print("====提取紙張區域====")
        detector = PaperDetector_edges(image_path)
        detector.detect_paper_by_color()
        if detector.original is not None:

            region = detector.extract_paper_region()
            if region is not None:
                height, width = region.shape[:2]
                if width > 600:
                    scale = 600 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    region = cv2.resize(region, (new_width, new_height))
                # cv2.imshow("提取的紙張區域", region)
                detector_img = detector.save_results()
                detector.show_results()
                D_sq_img, black_corners_int = Draw_square(detector_img)
                if D_sq_img is not None:
                    
                    # cv2.imwrite(os.path.join('kid', uid, f'{img_id}_result.jpg'), D_sq_img)
                    analyzer = BoxDistanceAnalyzer(
                        box1=black_corners_int, image_path=detector_img
                    )
                    result_img, kid = analyzer.analyze(pixel_per_cm=pixel_per_cm)
                    # result_path = rf"kid\{uid}\{img_id}_result.jpg"
                    result_path = os.path.join('kid', uid, f'{img_id}_result.jpg')
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
                return_score(score)
