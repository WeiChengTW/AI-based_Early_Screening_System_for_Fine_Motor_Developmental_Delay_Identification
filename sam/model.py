import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

def main():
    # 1. 載入 YOLO 模型
    yolo_model = YOLO('best.pt')
    
    # 2. 載入 SAM 模型
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"正在載入 SAM 模型 ({device})...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 3. 資料夾設定
    input_folder = "datasetpictrue"
    output_folder = "yolo_sam_results"
    os.makedirs(output_folder, exist_ok=True)

    # 定義你想要分割的類別名稱
    TARGET_CLASS = "line" 

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- 步驟 A: YOLO 偵測 ---
        yolo_results = yolo_model.predict(source=image, conf=0.5, iou=0.7)
        
        # --- 步驟 B: 把圖片餵給 SAM ---
        predictor.set_image(image_rgb)
        
        annotated_frame = image.copy()

        for r in yolo_results:
            # 取得類別字典 {ID: Name}
            class_names = r.names
            # 取得框的座標、類別 ID
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            
            for box, cls_id in zip(boxes, classes):
                current_name = class_names[int(cls_id)]
                color = np.random.randint(0, 255, (3,)).tolist()
                
                # --- 步驟 C: 判斷是否為 line ---
                if current_name == TARGET_CLASS:
                    # 如果是 line，執行 SAM 分割
                    masks, _, _ = predictor.predict(
                        box=box,
                        multimask_output=False
                    )
                    mask = masks[0]
                    
                    # 繪製遮罩 (Mask)
                    overlay = annotated_frame.copy()
                    overlay[mask] = color
                    cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
                    
                    # (選做) 遮罩邊框加強
                    # cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1)
                else:
                    # 如果不是 line，只畫 YOLO 的框 (Box)
                    cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3)
                    
                # 寫上標籤名稱
                cv2.putText(annotated_frame, current_name, (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('YOLO (Box) + SAM (Mask for Line)', annotated_frame)
        cv2.imwrite(os.path.join(output_folder, f"yolo_sam_{img_name}"), annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()