from ultralytics import YOLO
import cv2
import time
import sys
import os
from pathlib import Path
import json

game_started = False
start_time = None
video_writer = None
recording = False

# 新增：遊戲狀態（供前端查詢）
game_state = {
    "running": False,
    "bean_count": 0,
    "remaining_time": 60,
    "warning": False,
    "game_over": False,
    "score": -1
}

def return_score(score):
    sys.exit(int(score))

def calculate_score(cur_count, remain_time, WARNING):
    if cur_count >= 10 and remain_time >= 30:
        return 2 if not WARNING else 1
    elif (cur_count >= 10 or cur_count >= 5) and remain_time < 30:
        return 1
    elif remain_time == 0:
        return 0
    else:
        return -1

def save_game_state(uid, state_data):
    """儲存遊戲狀態到檔案，供前端讀取"""
    state_file = Path(__file__).parent.parent / "kid" / uid / "Ch5-t1_state.json"
    try:
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False)
    except Exception as e:
        print(f"儲存狀態失敗: {e}")

def main(CAMERA_INDEX, VIDEO_PATH, UID):
    global video_writer, recording, game_started, start_time, game_state

    # ===== 重要：每次開始前完全重置遊戲狀態 =====
    game_state["running"] = False
    game_state["bean_count"] = 0
    game_state["remaining_time"] = 60
    game_state["warning"] = False
    game_state["game_over"] = False
    game_state["score"] = -1
    
    # 立即儲存初始狀態到檔案
    save_game_state(UID, game_state)
    print("遊戲狀態已初始化並儲存")

    # 初始化模型
    # 注意：請確認模型路徑是否正確，網頁呼叫時路徑可能需要絕對路徑
    model_path = r'ch5-t1/bean_model.pt'
    if not os.path.exists(model_path):
         # 嘗試用絕對路徑 (如果在 run.py 同層目錄下有 ch5-t1 資料夾)
         model_path = str(Path(__file__).parent / 'bean_model.pt')
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # 嘗試開啟相機（加入重試機制）
    max_retries = 3
    retry_delay = 1  # 秒
    
    cap = None
    for attempt in range(max_retries):
        print(f"嘗試開啟相機 (索引 {CAMERA_INDEX})，第 {attempt + 1}/{max_retries} 次...")
        cap = cv2.VideoCapture(CAMERA_INDEX)  # Windows 使用 DirectShow
        
        if cap.isOpened():
            print("相機開啟成功！")
            break
        
        print(f"開啟失敗，等待 {retry_delay} 秒後重試...")
        # cap.release() # 如果沒開成，其實不用 release，但加了也無妨
        time.sleep(retry_delay)
    else:
        # 所有嘗試都失敗
        print(f"錯誤：嘗試 {max_retries} 次後仍無法開啟相機索引 {CAMERA_INDEX}")
        print("可用的相機索引：")
        for i in range(5):
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                print(f"  - 相機索引 {i} 可用")
                test_cap.release()
        return -1

    # 預熱模型
    ret, warmup_frame = cap.read()
    if ret:
        scale = 0.8
        height, width = warmup_frame.shape[:2]
        crop_height = int(height * scale)
        crop_width = int(width * scale)
        start_x = (width - crop_width) // 2
        start_y = (height - crop_height) // 2
        warmup_frame = warmup_frame[start_y:start_y + crop_height, start_x:start_x + crop_width]
        _ = model.predict(source=warmup_frame, conf=0.6, verbose=False)
        print("模型預熱完成！")

    frame_count = 0
    PER_FRAME = 10
    prev_box_count = 0
    CONF = 0.4
    game_duration = 60

    WARNING = False
    SCORE = -1

    # 自動開始遊戲
    game_started = True
    start_time = time.time()
    recording = True
    game_state["running"] = True
    print("遊戲開始！")
    
    # [修正] 移除 namedWindow，避免 headless 環境報錯
    # cv2.namedWindow('Bean Detection - Press Q to Quit', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取畫面")
            break
        
        frame_count += 1
        
        # 裁切中間區域
        scale = 0.7
        height, width = frame.shape[:2]
        crop_height = int(height * scale)
        crop_width = int(width * scale)
        start_x = (width - crop_width) // 2
        start_y = (height - crop_height) // 2
        frame = frame[start_y:start_y + crop_height, start_x:start_x + crop_width]

        # 初始化 VideoWriter
        if recording and video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            frame_size = (frame.shape[1], frame.shape[0])
            video_writer = cv2.VideoWriter(str(VIDEO_PATH), fourcc, fps, frame_size)
            print(f"開始錄影: {frame_size} @ {fps} FPS, 路徑: {VIDEO_PATH}")

        current_box_count = 0
        display_frame = frame.copy()  # 複製一份用於顯示(雖然現在不顯示了，但保留繪圖邏輯)
        
        if game_started:
            results = model.predict(source=frame, conf=CONF, verbose=False)

            for result in results:
                boxes = result.boxes
                current_box_count = len(boxes)
                
                # 繪製偵測框和標籤
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # 繪製邊界框
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 顯示信心度
                    label = f'Bean {conf:.2f}'
                    cv2.putText(display_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 檢查是否突然增加 2 個或以上
            if prev_box_count > 0:
                increase = current_box_count - prev_box_count
                if frame_count % PER_FRAME == 0 and increase >= 2:
                    print("WARNING !")
                    WARNING = True
                    game_state["warning"] = True
            
            prev_box_count = current_box_count

        # 計算剩餘時間
        if game_started:
            elapsed_time = time.time() - start_time
            remaining_time = max(0, game_duration - elapsed_time)

            # 更新遊戲狀態
            game_state["bean_count"] = current_box_count
            game_state["remaining_time"] = int(remaining_time)
            
            # 在畫面上顯示資訊 (雖然不顯示視窗，但會被錄進影片)
            info_y = 30
            cv2.putText(display_frame, f'Bean Count: {current_box_count}', (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(display_frame, f'Time: {int(remaining_time)}s', (10, info_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            if WARNING:
                cv2.putText(display_frame, 'WARNING!', (10, info_y + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # 每0.1秒儲存一次狀態 (約每3幀)
            if frame_count % 3 == 0:
                save_game_state(UID, game_state)

            SCORE = calculate_score(current_box_count, remaining_time, WARNING)
            if SCORE != -1:
                game_state["game_over"] = True
                game_state["score"] = SCORE
                game_state["running"] = False
                save_game_state(UID, game_state)

                # 顯示最終分數 (寫入影片)
                cv2.putText(display_frame, f'GAME OVER - Score: {SCORE}', 
                            (display_frame.shape[1]//2 - 200, display_frame.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

                if video_writer is not None:
                    video_writer.write(display_frame) # 寫入最後一幀帶有 Game Over 的畫面
                
                # [修正] 移除 imshow 和 waitKey，改用 sleep
                # cv2.imshow('Bean Detection - Press Q to Quit', display_frame)
                # cv2.waitKey(2000)
                time.sleep(2)
                
                print(f"score : {SCORE}\n")
                break
        
        # [修正] 移除主迴圈顯示
        # cv2.imshow('Bean Detection - Press Q to Quit', display_frame)
        
        # 錄影（錄製原始畫面 + 標註）
        # 注意：你原本是 write(frame)，這樣錄進去的是乾淨畫面
        # 如果你想錄製有框線的畫面，請改用 write(display_frame)
        if recording and video_writer is not None:
            video_writer.write(display_frame) 

        # 檢查是否按下 Q 鍵退出 (Web模式下無法按鍵，故註解)
        # if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        #     print("使用者按下 Q 鍵，結束遊戲")
        #     break

    # 清理資源
    if video_writer is not None:
        video_writer.release()
        print(f"錄影已儲存: {VIDEO_PATH}")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    return 0 if SCORE == -1 else SCORE

if __name__ == "__main__":
    UID = None
    # [注意] 這裡你硬編碼了 CAMERA_INDEX = 6
    # 如果這不是故意的，請改回 int(sys.argv[2]) 或是 0
    CAMERA_INDEX = 6
    
    if len(sys.argv) >= 3:
        try:
            UID = sys.argv[1]
            # CAMERA_INDEX = int(sys.argv[2])
            CAMERA_INDEX = 6 # 強制覆蓋為 6
            print(f"從 run.py 接收到 UID: {UID}, 相機索引: {CAMERA_INDEX}")
        except Exception as e:
            print(f"錯誤：無法解析參數: {e}")
            sys.exit(-1)
    else:
        # 方便手動測試用
        print(f"警告：缺少參數，使用預設值測試")
        UID = "test_user"
        CAMERA_INDEX = 6 # 測試用

    # 建立輸出路徑
    BASE_DIR = Path(__file__).parent.parent
    OUTPUT_DIR = BASE_DIR / "kid" / UID
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # VIDEO_PATH = OUTPUT_DIR / "Ch5-t1_result.mp4"

    VIDEO_PATH = os.path.join(BASE_DIR, 'kid', UID, "Ch5-t1_result.mp4")
    
    print(f"影片將儲存至: {VIDEO_PATH}")

    score = main(CAMERA_INDEX, str(VIDEO_PATH), UID)
    print(f"最終分數: {score}")
    return_score(score)