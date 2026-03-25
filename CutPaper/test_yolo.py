import cv2
import os
from ultralytics import YOLO
from pathlib import Path


def main():
    # 1. 設定模型路徑 (請確認 best.pt 的正確位置)
    # 通常在 runs/segment/run_13_fixed/weights/best.pt
    model_path = r"/Users/william/AI-based_Early_Screening_System_for_Fine_Motor_Developmental_Delay_Identification/PDMS2_web/ch3-t1/models/best.pt"

    if not os.path.exists(model_path):
        print(f"錯誤：找不到模型檔案 {model_path}")
        return

    # 2. 載入模型
    model = YOLO(model_path)

    # 3. 設定資料夾路徑
    input_folder = r"CutPaper/input"
    output_folder = r"CutPaper/output"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 4. 取得資料夾內所有圖片
    img_formats = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = [
        f for f in os.listdir(input_folder) if Path(f).suffix.lower() in img_formats
    ]

    print(f"開始處理 {len(image_files)} 張圖片...")

    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)

        # 進行預測
        # conf=0.5 代表置信度大於 50% 才顯示
        results = model.predict(source=img_path, conf=0.5, save=False)

        for r in results:
            # 取得繪製好標籤與遮罩的影像 (bgr 格式)
            annotated_frame = r.plot()

            # 顯示結果
            cv2.imshow("YOLOv8 Segmentation Inference", annotated_frame)

            # 儲存結果到新資料夾
            save_path = os.path.join(output_folder, f"res_{img_name}")
            cv2.imwrite(save_path, annotated_frame)
            print(f"已儲存預測結果: {save_path}")

        # 按下 'q' 鍵可以提早結束預覽
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("所有圖片處理完成！")


if __name__ == "__main__":
    main()
