import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2


# -------------------------
# 參數設定
# -------------------------
# MODEL_PATH = r'model/check_circle.h5'
MODEL_PATH = os.path.join('model', "check_circle.h5")
# IMAGE_DIR = r'pic_result\Result'   # 放要辨識的圖片資料夾
IMAGE_DIR = os.path.join('pic_result','Result')
IMAGE_SIZE = (224, 224)         # 跟訓練模型時一樣
CLASS_NAMES = ['Other', 'circle_or_oval']  # 對應你的類別名稱

# -------------------------
# 載入模型
# -------------------------
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# -------------------------
# 讀取資料夾內所有圖片
# -------------------------
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)
    
    # 讀取並預處理圖片
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0            # normalize
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, C)
    
    # 模型預測
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]
    
    # 印出結果
    print(f"{img_name} → {CLASS_NAMES[predicted_class]} ({confidence*100:.2f}%)")
    # print(f"{img_name} → {predicted_class} ({confidence*100:.2f}%)")
