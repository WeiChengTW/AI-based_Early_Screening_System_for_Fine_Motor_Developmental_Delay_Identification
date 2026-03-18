import cv2
import os


class ScreenCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # 使用預設攝影機
        self.index = 1
        self.setup_camera()

    def setup_camera(self):
        """設定攝影機參數"""
        if not self.cap.isOpened():
            print("無法開啟攝影機")
            return False

        # 設定解析度 (可選)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return True

    def find_next_index(self):
        """找到下一個可用的檔案索引"""
        while os.path.exists(f"ch3-t1/raw/img{self.index}.jpg"):
            self.index += 1
        return self.index

    def save_frame(self, frame):
        """保存當前畫面並回傳路徑"""
        index = self.find_next_index()
        filename = f"ch3-t1/raw/img{index}.jpg"
        cv2.imwrite(filename, frame)
        abs_path = os.path.abspath(filename)
        print(f"圖片已保存為: {abs_path}")
        self.index += 1
        return abs_path

    def run(self):
        """主要執行迴圈"""
        print("攝影機已啟動")
        print("按 ENTER 鍵保存圖片")
        print("按 'q' 鍵退出程式")
        last_img_path = None  # 初始化變數
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("無法讀取攝影機畫面")
                break

            # 計算紅色框（拍攝範圍）的位置
            height, width = frame.shape[:2]
            crop_width = int(width * 0.6)
            crop_height = int(height * 0.6)
            start_x = (width - crop_width) // 2
            start_y = (height - crop_height) // 2
            end_x = start_x + crop_width
            end_y = start_y + crop_height

            # 複製一份 frame 來畫紅色框
            preview_frame = frame.copy()
            # 畫紅色矩形框（線寬2）
            cv2.rectangle(
                preview_frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2
            )

            # 顯示畫面（含紅色框）
            cv2.imshow("Camera - Press ENTER to save, Q to quit", preview_frame)

            # 檢查按鍵
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # ENTER 鍵 (ASCII 13)
                # 裁切畫面 (取中央區域)

                height, width = frame.shape[:2]
                crop_width = int(width * 0.6)  # 裁切寬度為原本的60%
                crop_height = int(height * 0.6)  # 裁切高度為原本的60%

                # 計算裁切區域的起始位置 (置中)
                start_x = (width - crop_width) // 2
                start_y = (height - crop_height) // 2

                # 執行裁切
                cropped_frame = frame[
                    start_y : start_y + crop_height, start_x : start_x + crop_width
                ]

                # 放大1.5倍
                new_width = int(crop_width * 1.5)
                new_height = int(crop_height * 1.5)
                enlarged_frame = cv2.resize(
                    cropped_frame,
                    (new_width, new_height),
                    interpolation=cv2.INTER_CUBIC,
                )

                last_img_path = self.save_frame(enlarged_frame)
                # break
            elif key == ord("q") or key == ord("Q"):  # 'q' 或 'Q' 鍵退出
                break
        return last_img_path

    def cleanup(self):
        """清理資源"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("攝影機已關閉")


if __name__ == "__main__":
    screen_capture = ScreenCapture()
    try:
        screen_capture.run()
    except KeyboardInterrupt:
        print("\n程式被中斷")
    finally:
        screen_capture.cleanup()
