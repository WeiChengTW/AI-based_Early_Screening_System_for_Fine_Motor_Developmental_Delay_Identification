# main.py — 單線分析
# 輸入：kid\<uid>\<img_id>.jpg
# 輸出結果圖：kid\<uid>\<img_id>_result.jpg，並以 exit code 回傳分數

import os
import json
import cv2
import sys
from final import find_baseline_and_show_all

PXCM_JSON = "px2cm.json"


def return_score(score: int):
    """用 exit code 回傳分數（與 ch2-t5 / ch3-t1 一致）"""
    sys.exit(int(score))


def main():
    # ========= 1) 讀比例檔 =========
    if not os.path.exists(PXCM_JSON):
        print(f"找不到比例檔：{PXCM_JSON}", file=sys.stderr)
        return_score(0)

    try:
        with open(PXCM_JSON, "r", encoding="utf-8") as f:
            obj = json.load(f)
        pixel_per_cm = float(obj.get("pixel_per_cm", 0.0))
    except Exception as e:
        print(f"讀取 {PXCM_JSON} 失敗：{e}", file=sys.stderr)
        return_score(0)

    if pixel_per_cm <= 0:
        print(f"{PXCM_JSON} 內的 pixel_per_cm 非正值：{obj}", file=sys.stderr)
        return_score(0)

    # ========= 2) 解析參數（與其他關卡介面一致） =========
    if len(sys.argv) > 2:
        uid = sys.argv[1]
        img_id = sys.argv[2]
        image_path = os.path.join('kid', uid, f"{img_id}.jpg")
    else:
        print("參數不足，需要 uid 與 img_id", file=sys.stderr)
        return_score(0)

    if not os.path.exists(image_path):
        print(f"找不到圖片：{image_path}", file=sys.stderr)
        return_score(0)

    # ========= 3) 讀影像 =========
    img = cv2.imread(image_path)
    if img is None:
        print(f"圖片讀取失敗：{image_path}", file=sys.stderr)
        return_score(0)

    # ========= 4) 分析（盡量相容：優先取回 (score, result_img)） =========
    result_img = None
    try:
        # 新版：回傳 (score, result_img)
        score, result_img = find_baseline_and_show_all(img, pixel_per_cm)
    except TypeError:
        # 舊版只回傳 score
        score = find_baseline_and_show_all(img, pixel_per_cm)
        result_img = img  # 至少把原圖當成結果圖

    # ========= 5) 準備輸出路徑：kid\<uid>\<img_id>_result.jpg =========
    result_dir = os.path.join('kid',uid)
    os.makedirs(result_dir, exist_ok=True)
    out_path = os.path.join(result_dir, f"{img_id}_result.jpg")

    # 沒有 result_img 的話就存原圖
    to_save = result_img if result_img is not None else img
    ok = cv2.imwrite(out_path, to_save)
    if not ok:
        print(f"⚠️ 影像儲存失敗：{out_path}", file=sys.stderr)
        # 但還是回傳分數，避免整個流程中斷

    print("完成結果圖：", out_path)
    print("得分：", score)

    # ========= 6) 用 exit code 回傳分數 =========
    return_score(score)


if __name__ == "__main__":
    main()
