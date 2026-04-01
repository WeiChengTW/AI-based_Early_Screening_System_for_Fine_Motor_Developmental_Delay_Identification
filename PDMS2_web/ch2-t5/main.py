import cv2
import os
from cut_paper import get_pixel_per_cm_from_a4
from exceed_correct import detect_horizontal_lines
from draw_range_correct import analyze_paint
import json
import sys


def return_score(score):
    sys.exit(int(score))


if __name__ == "__main__":
    # === 必須有 uid + img_id ===
    if len(sys.argv) > 2:
        uid = sys.argv[1]
        img_id = sys.argv[2]
        image_path = os.path.join("kid", uid, f'{img_id}.jpg')
    else:
        print("參數不足，需要 uid 與 img_id")
        return_score(0)

    # === 中途輸出路徑（裁紙後）===
    # out_dir = r"PDMS2_web\ch2-t5\new"
    out_dir = os.path.join("ch2-t5", "new")

    os.makedirs(out_dir, exist_ok=True)
    # out_path = rf"{out_dir}\new{img_id}.jpg"
    out_path = os.path.join(out_dir, f"new{img_id}.jpg")

    # ========= 1) 裁紙（偵測 A4） =========
    warped = get_pixel_per_cm_from_a4(image_path, show_debug=False)

    # 如果函式只把檔案寫到硬碟，但沒回傳 ndarray ⇒ fallback 読取 out_path
    if warped is None:
        warped = cv2.imread(out_path)

    if warped is None:
        print(f"裁切失敗：讀不到 {out_path}")
        return_score(0)

    # 寫出裁好的紙
    cv2.imwrite(out_path, warped)

    # ========= 2) 找兩條水平線 =========
    y_top, y_bot = detect_horizontal_lines(warped, show_debug=False)
    if y_top is None or y_bot is None:
        print("偵測不到兩條主線")
        return_score(0)

    # ========= 3) 分析塗色 + 超出區域 =========
    score, result_img = analyze_paint(
        warped,
        int(y_top),
        int(y_bot),
        show_windows=False
    )

    # ========= 4) 最終結果圖（給 admin 預覽的）=========
    result_dir = os.path.join("kid", uid)
    os.makedirs(result_dir, exist_ok=True)
    
    result_path = os.path.join(result_dir, f"{img_id}_result.jpg")
    cv2.imwrite(result_path, result_img)

    print("完成結果圖：", result_path)
    print("得分：", score)

    return_score(score)
