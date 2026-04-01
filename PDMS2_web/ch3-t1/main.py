from PaperDetector_edge import PaperDetector_edges
from BoxDistanceAnalyzer import BoxDistanceAnalyzer
from px2cm import get_pixel_per_cm_from_a4
import cv2
import sys
import os
import json


def return_score(score):
    sys.exit(int(score))


if __name__ == "__main__":
    # 檢查是否有傳入 id 參數
    if len(sys.argv) > 2:
        # 使用傳入的 uid 和 id 作為圖片路徑
        uid = sys.argv[1]
        img_id = sys.argv[2]
        # uid = "lull222"
        # # img_id = "ch3-t1"
        # image_path = rf"kid\{uid}\{img_id}.jpg"
        image_path = os.path.join('kid', uid, f'{img_id}.jpg')
        # image_path = rf"ch3-t1.jpg"
        # a4_path = os.path.join('ch3-t1', 'a4_2.jpg')
        # _, json_path = get_pixel_per_cm_from_a4(a4_path, show_debug=False)
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
                # detector.show_results()

            if detector_img is not None:
                result_img, min_dist_cm, max_dist_cm = BoxDistanceAnalyzer(
                    detector_img
                )
                # result_path = rf"kid\{uid}\{img_id}_result.jpg"
                result_path = os.path.join('kid', uid, f'{img_id}_result.jpg')
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
        
        
        
