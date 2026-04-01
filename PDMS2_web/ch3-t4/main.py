import cv2
import numpy as np
import os
import sys


def return_score(score):
    sys.exit(int(score))


def judge_score(target_img_path, standard_area):
    """
    核心邏輯函式：讀取圖片 -> 排除反光 -> 計算面積 -> 判定分數
    回傳一個字典 (result_data)，包含所有需要的資訊供顯示使用
    """
    # 1. 讀取圖片
    img = cv2.imread(target_img_path)
    if img is None:
        print(f"錯誤: 無法讀取圖片 {target_img_path}")
        return None

    # 2. 影像預處理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 二值化 (使用 OTSU 自動找閾值)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 尋找輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"error": "未偵測到任何輪廓", "img": img}

    # --- 關鍵改良：智慧過濾 (Smart Filter) ---
    # 目標：找出「面積夠大」且「形狀像矩形」的輪廓，避開不規則的光斑
    best_cnt = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # 濾除太小的雜訊
        if area < 1000:
            continue

        # 計算矩形充滿度 (Extent)
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        extent = float(area) / rect_area

        # 紙張通常是矩形，Extent 會比較高 (設定 > 0.65 比較保險，允許稍微剪歪)
        # 反光通常是不規則形狀，Extent 會較低
        if extent > 0.65:
            if area > best_area:
                best_area = area
                best_cnt = cnt

    # 如果都沒找到像矩形的，只好退而求其次找最大的 (並標記警告)
    is_rectangular = True
    if best_cnt is None:
        print("警告：找不到矩形物體，可能反光嚴重或紙張變形，使用最大輪廓代替。")
        best_cnt = max(contours, key=cv2.contourArea)
        best_area = cv2.contourArea(best_cnt)
        is_rectangular = False

    # 4. 計算比例與評分
    ratio = best_area / standard_area
    score = -1
    desc = ""

    # 評分邏輯 (嚴格版)
    if ratio > 0.90:
        score = 0
        desc = "Score 0 (Full)"
    elif 0.40 <= ratio <= 0.60:
        score = 2
        desc = "Score 2 (Half)"
    else:
        score = 1
        # 區分是拿到小張還是大張
        percent = ratio * 100
        desc = f"Score 1 ({percent:.1f}%)"

    # 5. 打包結果回傳
    result_data = {
        "img": img,  # 原圖 (供顯示用)
        "contour": best_cnt,  # 找到的輪廓
        "area": best_area,  # 當前面積
        "ratio": ratio,  # 比例
        "score": score,  # 分數
        "desc": desc,  # 描述文字
        "is_rectangular": is_rectangular,  # 是否通過形狀檢查
    }

    return result_data, score


def show_result(result_data):
    """
    顯示函式：接收 judge_score 的結果 -> 繪圖 -> 顯示視窗
    """
    if result_data is None or "error" in result_data:
        print("無法顯示結果 (資料錯誤或無影像)")
        return

    img = result_data["img"]
    cnt = result_data["contour"]
    score = result_data["score"]
    desc = result_data["desc"]
    ratio = result_data["ratio"]
    area = result_data["area"]

    # 複製圖片以免破壞原圖
    display_img = img.copy()

    # 1. 畫出輪廓 (綠色)
    if cnt is not None:
        # 如果形狀檢查未通過(可能是反光)，改用黃色警示；通過用綠色
        color = (0, 255, 0) if result_data["is_rectangular"] else (0, 255, 255)
        cv2.drawContours(display_img, [cnt], -1, color, 4)

        # 畫出包圍框 (紅色)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 2. 準備顯示文字
        text_info = f"{desc} | Ratio: {ratio:.2f}"

        # 為了讓文字清楚，加個黑色背景條
        cv2.rectangle(display_img, (x, y - 40), (x + w, y), (0, 0, 0), -1)
        cv2.putText(
            display_img,
            text_info,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    # 3. 縮放顯示 (避免圖片太大超出螢幕)
    h, w = display_img.shape[:2]
    max_width = 800
    if w > max_width:
        scale = max_width / w
        new_dim = (max_width, int(h * scale))
        final_view = cv2.resize(display_img, new_dim, interpolation=cv2.INTER_AREA)
    else:
        final_view = display_img

    # 4. 顯示視窗
    print(f"--- 詳細數據 ---")
    print(f"面積: {area:.0f}")
    print(f"比例: {ratio:.2f}")
    print(f"判定: {desc}")

    # cv2.imshow("Judge Result", final_view)
    # print("按下任意鍵關閉視窗...")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return final_view


# 1. 設定基準面積 (請填入你之前測得的數值)
STANDARD_AREA = 217270

if __name__ == "__main__":
    # 使用方式範例: python main.py 1125 ch3-t3

    if len(sys.argv) > 2:
        # 使用傳入的 uid 和 id 作為圖片路徑
        uid = sys.argv[1]
        img_id = sys.argv[2]
        # uid = "1125"
        # img_id = "ch3-t3"
        # image_path = rf"kid\{uid}\{img_id}.jpg"
        image_path = os.path.join('kid', uid, f'{img_id}.jpg')
        

        # result_path = rf"kid\{uid}\{img_id}_result.jpg"
        result_path = os.path.join('kid', uid, f'{img_id}_result.jpg')

    # image_path = rf"PDMS2_web\kid\1125\ch3-t4.jpg"
    # result_path = rf"PDMS2_web\kid\1125\ch3-t4_result.jpg"
    # 執行主程式
    if os.path.exists(image_path):
        # 步驟一：計算與判定
        result, score = judge_score(image_path, STANDARD_AREA)

        # 步驟二：顯示結果
        result_img = show_result(result)
        cv2.imwrite(result_path, result_img)

    else:
        print("找不到檔案")

    return_score(score)
