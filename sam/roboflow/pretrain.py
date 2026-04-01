import cv2
import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# 設定根目錄
ROOT_DIR = r'C:\Users\hiimd\Desktop\vscode\Peabody-Developmental-Motor-Scales\sam\roboflow'

def fix_dataset_and_train():
    mapping = {2: 0, 3: 1} 

    for split in ['train', 'valid', 'test']:
        split_path = Path(ROOT_DIR) / split
        img_dir = split_path / 'images'
        mask_dir = split_path / 'masks'
        label_dir = split_path / 'labels'
        
        # 自動建立 masks 和 labels 資料夾
        mask_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        # --- 自動搬移：把 images 裡的 mask 移到 masks 資料夾 ---
        for mask_file in img_dir.glob('*_mask.*'):  
            os.replace(mask_file, mask_dir / mask_file.name)

        # --- 開始轉換 ---
        print(f"正在處理 {split} 資料集...")
        for img_file in img_dir.glob('*.jpg'):
            # 尋找對應的 mask 檔案 (例如 image_1.jpg 對應 image_1_mask.png)
            # 注意：Roboflow 的 mask 通常是 .png
            mask_name = img_file.stem + "_mask.png" 
            mask_path = mask_dir / mask_name
            
            if not mask_path.exists():
                # 嘗試尋找 .jpg 結尾的 mask
                mask_path = mask_dir / (img_file.stem + "_mask.jpg")

            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                shape_info = mask.shape
                h, w = shape_info[0], shape_info[1]
                
                with open(label_dir / f"{img_file.stem}.txt", 'w') as f:
                    for pixel_val, yolo_id in mapping.items():
                        class_mask = np.where(mask == pixel_val, 255, 0).astype(np.uint8)
                        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            if len(cnt) < 3: continue
                            pts = cnt.reshape(-1, 2)
                            norm_pts = [f"{x/w:.6f} {y/h:.6f}" for x, y in pts]
                            f.write(f"{yolo_id} {' '.join(norm_pts)}\n")

    print("✅ 資料夾整理完成，標籤轉換成功！")

    # --- 啟動訓練 ---
    model = YOLO('yolov8n-seg.pt')
    yaml_path = os.path.join(ROOT_DIR, 'data.yaml')
    model.train(data=yaml_path.replace('\\', '/'), epochs=100, imgsz=640, device=0)

if __name__ == "__main__":
    fix_dataset_and_train()