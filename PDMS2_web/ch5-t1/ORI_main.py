from ultralytics import YOLO
import cv2
import time
import sys
import os
from pathlib import Path

game_started = False
start_time = None

# 按鈕設定
button_x, button_y, button_w, button_h = 10, 300, 150, 50

video_writer = None
recording = False

# 滑鼠回調函數
def mouse_callback(event, x, y, flags, param):
    global game_started, start_time, recording, video_writer
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 檢查是否點擊按鈕區域
        if button_x <= x <= button_x + button_w and button_y <= y <= button_y + button_h:
            if not game_started:
                game_started = True
                start_time = time.time()
                recording = True  # 開始錄影
                print("遊戲開始！")

def calculate_score(cur_count, remain_time, WARNING):
    if cur_count >= 10 and remain_time >= 30:
        return 2 if not WARNING else 1
    elif cur_count >= 5 and remain_time == 0:
        return 1
    elif remain_time == 0:
        return 0
    else:
        return -1

def main(CAMERA_INDEX, VIDEO_PATH):
    global video_writer, recording, game_started, start_time

    # 初始化模型
    model = YOLO(r'bean_model.pt')

    cap = cv2.VideoCapture(CAMERA_INDEX)  # 開啟攝影機

    # 檢查攝影機是否成功開啟
    if not cap.isOpened():
        print("錯誤：無法開啟攝影機")
        return -1

    # 預熱模型（讀取第一幀進行空推論）
    ret, warmup_frame = cap.read()
    if ret:
        # 裁切預熱畫面
        scale = 0.8
        height, width = warmup_frame.shape[:2]
        crop_height = int(height * scale)
        crop_width = int(width * scale)
        start_x = (width - crop_width) // 2
        start_y = (height - crop_height) // 2
        warmup_frame = warmup_frame[start_y:start_y + crop_height, start_x:start_x + crop_width]
        
        # 執行一次推論來預熱
        _ = model.predict(source=warmup_frame, conf=0.6, verbose=False)
        print("模型預熱完成！")

    frame_count = 0
    PER_FRAME = 5
    prev_box_count = 0
    CONF = 0.6
    game_duration = 60

    # 分數相關
    WARNING = False
    SCORE = -1

    # 在進入迴圈前設定視窗
    # cv2.namedWindow('Pick Bean Game')
    # cv2.setMouseCallback('Pick Bean Game', mouse_callback)
    game_started = True
    start_time = time.time()
    recording = True  # 開始錄影
    print("遊戲開始！")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取畫面")
            break
        
        frame_count += 1
        
        # 裁切中間區域
        scale = 0.8
        height, width = frame.shape[:2]

        crop_height = int(height * scale)
        crop_width = int(width * scale)

        start_x = (width - crop_width) // 2
        start_y = (height - crop_height) // 2

        # 裁切畫面
        frame = frame[start_y:start_y + crop_height, start_x:start_x + crop_width]

        # 初始化 VideoWriter
        if recording and video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            frame_size = (frame.shape[1], frame.shape[0])
            video_writer = cv2.VideoWriter(VIDEO_PATH, fourcc, fps, frame_size)
            print(f"開始錄影: {frame_size} @ {fps} FPS")

        # 只在遊戲開始後才進行偵測
        current_box_count = 0
        if game_started:
            results = model.predict(source=frame, conf=CONF, verbose=False)

            for result in results:
                boxes = result.boxes
                current_box_count = len(boxes)

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = int((x1 + x2) // 2)
                    cy = int((y1 + y2) // 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # 檢查是否突然增加 2 個或以上
            if prev_box_count > 0:
                increase = current_box_count - prev_box_count
                if frame_count % PER_FRAME == 0 and increase >= 2:
                    print("WARNING !")
                    WARNING = True
            
            prev_box_count = current_box_count

        # 計算剩餘時間
        if game_started:
            elapsed_time = time.time() - start_time
            remaining_time = max(0, game_duration - elapsed_time)

            SCORE = calculate_score(current_box_count, remaining_time, WARNING)
            if SCORE != -1:
                cv2.putText(frame, f"GAME OVER !", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)

                if video_writer is not None:
                    video_writer.write(frame)
                
                cv2.imshow('Pick Bean Game', frame)
                cv2.waitKey(3000)
                print(f"score : {SCORE}\n")
                break
        
        # 顯示資訊
        if game_started:
            cv2.putText(frame, f'Beans: {current_box_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Time: {int(remaining_time)}s', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 繪製按鈕
        # if not game_started:
        #     cv2.rectangle(frame, (button_x, button_y), 
        #                 (button_x + button_w, button_y + button_h), 
        #                 (0, 255, 0), -1)
        #     cv2.rectangle(frame, (button_x, button_y), 
        #                 (button_x + button_w, button_y + button_h), 
        #                 (0, 0, 0), 2)
        #     cv2.putText(frame, 'START', (button_x + 25, button_y + 35), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # 錄影
        if recording and video_writer is not None:
            video_writer.write(frame)

        cv2.imshow('Pick Bean Game', frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

    # 清理資源
    if video_writer is not None:
        video_writer.release()
        print(f"錄影已儲存: {VIDEO_PATH}")

    cap.release()
    cv2.destroyAllWindows()
    return 0 if SCORE == -1 else SCORE

if __name__ == "__main__":

    # UID = None
    # CAMERA_INDEX = 2
    
    # # 建立輸出路徑
    # BASE_DIR = Path(__file__).parent.parent
    # OUTPUT_DIR = BASE_DIR / "kid" / UID
    # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # VIDEO_PATH = OUTPUT_DIR / "Ch5-t1_result.mp4"

    # if len(sys.argv) >= 3:
    #     try:
    #         UID = sys.argv[1]
    #         CAMERA_INDEX = int(sys.argv[2])
    #         print(f"從 app.py 接收到 UID: {UID}, 相機索引: {CAMERA_INDEX}")
    #     except Exception as e:
    #         print(f"錯誤：無法解析參數: {e}")
    #         sys.exit(-1)
    # else:
    #     print(f"錯誤：缺少 UID 和相機索引參數")
    #     sys.exit(-1)

    

    #test
    VIDEO_PATH = 'result.mp4'
    CAMERA_INDEX = 1

    score = main(CAMERA_INDEX, VIDEO_PATH)
    print(f"最終分數: {score}")