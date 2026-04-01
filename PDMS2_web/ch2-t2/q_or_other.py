import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tflite_runtime.interpreter as tflite


class ImageClassifier:
    def __init__(self, model_path, class_names, image_size=(224, 224)):
        """
        model_path: 模型檔案路徑
        class_names: 類別名稱列表，順序要跟訓練時一致
        image_size: 模型輸入大小
        """
        self.model_path = model_path
        self.class_names = class_names
        self.image_size = image_size

        # 載入模型
        self.model = load_model(model_path)
        print("Model loaded successfully!")

    def predict(self, img_path):
        """
        接收單張圖片路徑，回傳 (類別名稱, 信心度)
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"圖片不存在: {img_path}")

        img = image.load_img(img_path, target_size=self.image_size)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # normalize
        img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, C)

        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class_idx]
        predicted_class_name = self.class_names[predicted_class_idx]

        return predicted_class_name, confidence

# -------------------------
# 使用範例
# -------------------------
if __name__ == "__main__":
    MODEL_PATH = r'circle_or_oval\Final_model.h5'
    CLASS_NAMES = ['Other', 'quadrilateral']

    classifier = ImageClassifier(MODEL_PATH, CLASS_NAMES)

    img_path = r'pic_result\Result\example.jpg'
    cls, conf = classifier.predict(img_path)
    print(f"{img_path} → {cls} ({conf*100:.2f}%)")
