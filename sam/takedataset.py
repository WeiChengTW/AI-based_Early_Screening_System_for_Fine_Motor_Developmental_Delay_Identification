import cv2
import os

def main():
    # 1. 設定資料夾名稱
    folder_name = "datasetpictrue"
    
    # 2. 檢查資料夾是否存在
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"已建立資料夾: {folder_name}")

    # 3. 開啟鏡頭 (若 1 無法開啟，請試著改回 0)
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("無法開啟鏡頭")
        return

    print("程式已啟動：")
    print("- 按下 't' 拍照並儲存 (僅儲存裁切後的部分)")
    print("- 按下 'q' 離開程式")

    count = 0
    crop_percent = 0.9

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取畫面")
            break

        # --- 裁切邏輯開始 ---
        h, w = frame.shape[:2]
        
        # 計算裁切後的寬高
        new_w = int(w * crop_percent)
        new_h = int(h * crop_percent)
        
        # 計算起始座標 (中心點向外推)
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        
        # 進行裁切 [y1:y2, x1:x2]
        cropped_frame = frame[start_y:start_y+new_h, start_x:start_x+new_w]
        # --- 裁切邏輯結束 ---

        # 顯示視窗 (預覽裁切後的畫面)
        cv2.imshow('Camera System (Cropped 65%)', cropped_frame)

        key = cv2.waitKey(1) & 0xFF

        # 按下 't' 拍照
        if key == ord('t'):
            img_name = os.path.join(folder_name, f"image_{count}.jpg")
            # 儲存裁切後的影像
            cv2.imwrite(img_name, cropped_frame)
            print(f"已儲存裁切照片: {img_name} (解析度: {new_w}x{new_h})")
            count += 1

        # 按下 'q' 離開
        elif key == ord('q'):
            print("正在關閉程式...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()