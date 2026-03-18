from ultralytics import YOLO

def train():
    model = YOLO('yolov8n-seg.pt') 

    # 開始訓練
    results = model.train(
        data='data.yaml', 
        epochs=100,      # 訓練回合數
        imgsz=640,       # 圖片輸入大小
        device=0,        # 指定 GPU ID，若無 GPU 則填 'cpu'
        project='block_segmentation', # 儲存專案名稱
        name='v1'        # 本次實驗名稱
    )

if __name__ == "__main__":
    train()