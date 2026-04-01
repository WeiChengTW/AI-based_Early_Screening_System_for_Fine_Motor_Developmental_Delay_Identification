import cv2
import numpy as np
import glob
import os
from pathlib import Path
from get_cm1 import auto_crop_paper, auto_crop_wood_board
from correct import analyze_image
import sys
import json


def return_score(score):
    sys.exit(int(score))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        # 使用傳入的 uid 和 id 作為圖片路徑
        uid = sys.argv[1]
        img_id = sys.argv[2]
        # uid = "lull222"
        # img_id = "ch2-t6"
        image_path = Path(os.path.join("kid", uid, f"{img_id}.jpg"))
    else:
        print("參數不足，需傳入 uid 與 img_id")
        return_score(0)

    # === 輸出路徑：中間處理用 ===
    output_dir = Path(os.path.join("ch2-t6", "new"))  # e.g. .../ch2-t6/new/
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, f"new{img_id}.jpg")   # e.g. .../ch2-t6/new/newch2-t6.jpg

    # === 讀圖 ===
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"無法讀取：{image_path}")
        return_score(0)  # 或者 sys.exit(1)

    # === 木板偵測：失敗就用原圖 ===
    try:
        board = auto_crop_wood_board(image, debug=False)
    except Exception:
        print(f"⚠️ {image_path.name} 木板偵測失敗 → 改用原圖裁紙")
        board = image

    # === 裁紙（透視校正）===
    try:
        paper = auto_crop_paper(board, trim=12, debug=False)
    except Exception as e:
        print(f"裁紙失敗：{e}")
        return_score(0)

    # === 先寫裁好的紙，成功才做 analyze ===
    ok = cv2.imwrite(str(out_path), paper)
    if not ok:
        print(f"⚠️ 影像儲存失敗：{out_path}")
        return_score(0)

    # print(f"{image_path.name} → {out_path.name}")

    # === 呼叫分析，並把結果圖存成 kid\uid\ch2-t6_result.jpg（跟 ch3-t1 一樣規則）===
    try:
        score, result_img = analyze_image(str(out_path), dot_distance_cm=10.0)

        # 注意：這裡才是真正給 admin 預覽的「結果圖」位置
        result_dir = Path(os.path.join("kid", uid))
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = os.path.join(result_dir, f"{img_id}_result.jpg")

        # 寫入結果圖
        cv2.imwrite(str(result_path), result_img)

        print("得分：", score)
        print("結果圖：", result_path)  # 這行純粹方便你 debug，看路徑對不對

        return_score(score)
    except Exception as e:
        print(f"分析失敗：{e}")
        return_score(0)
