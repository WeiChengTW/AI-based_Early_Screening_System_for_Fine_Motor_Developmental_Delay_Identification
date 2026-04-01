import cv2
import time

camera = cv2.VideoCapture(0, cv2.CAP_V4L2)

# 先設定 MJPG
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
camera.set(cv2.CAP_PROP_FOURCC, fourcc)

# 再設定解析度
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_FPS, 30)

time.sleep(0.5)

# 確認設定
w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = camera.get(cv2.CAP_PROP_FPS)

print(f"設定結果：{w}x{h} @ {fps} FPS")

# 實際拍一張確認
ret, frame = camera.read()
if ret:
    print(f"實際畫面：{frame.shape[1]}x{frame.shape[0]}")
    cv2.imwrite("test_720p.jpg", frame)
    print("已儲存 test_720p.jpg")
else:
    print("讀取畫面失敗")

camera.release()
