from PaperDetector_edge import PaperDetector_edges
from find_max_area import MaxAreaQuadFinder
import cv2
import os
import sys


def return_score(score):
    sys.exit(int(score))


def calculate_score_from_short_sides(side_lengths, px2cm):
    if not side_lengths or not px2cm:
        return 0

    side_cm = [
        side_lengths["top"] / px2cm,
        side_lengths["right"] / px2cm,
        side_lengths["bottom"] / px2cm,
        side_lengths["left"] / px2cm,
    ]
    short_edges = sorted(side_cm)[:2]
    d1 = abs(short_edges[0] - 7.5)
    d2 = abs(short_edges[1] - 7.5)

    if d1 <= 0.3 and d2 <= 0.3:
        return 2
    if d1 <= 1.2 and d2 <= 1.2:
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2:
        uid = sys.argv[1]
        img_id = sys.argv[2]
        image_path = os.path.join("kid", uid, f"{img_id}.jpg")

        score = 0
        detector = PaperDetector_edges(image_path)
        edges_path = detector.detect_paper_by_color()

        if edges_path is not None:
            finder = MaxAreaQuadFinder(edges_path)
            finder.find_max_area_quad()
            draw_result = finder.draw_and_show()

            if draw_result is not None:
                result_img, _ = draw_result
                result_path = os.path.join("kid", uid, f"{img_id}_result.jpg")
                cv2.imwrite(result_path, result_img)

                side_lengths = finder.get_side_lengths()
                score = calculate_score_from_short_sides(side_lengths, finder.px2cm)
                print(f"score = {score}")
            else:
                print("找不到有效輪廓，score = 0")
        else:
            print("無法產生邊緣圖，score = 0")

        return_score(score)
