# $ sudo apt update
# $ sudo apt install python3-pip -y
# $ pip install -U pip

# Please install Ultralytics Package, 由 Ultralytics 提供的 YOLOv8 套件，功能包括訓練、驗證、推論等。
# $ pip install ultralytics[export] 
# Install torch 2.5.0 and torchvision 0.20 according to JP6.1
# $ pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
# $ pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
# Install cuSPARSELt to fix a dependency issue with torch 2.5.0
# $ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
# $ sudo dpkg -i cuda-keyring_1.1-1_all.deb
# $ sudo apt-get update
# $ sudo apt-get -y install libcusparselt0 libcusparselt-dev
# install onnxruntime-gpu 1.20.0 with Python3.10 support.
# $ pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
# $  pip install numpy==1.23.5
# $  vim Test_SingleCamera_Inspection.py

from ultralytics import YOLO
import cv2
import time

# 模型檔案對應表
model_paths = {
    '1': "yolo11n.pt",        # 偵測模型
    '2': "yolo11n-seg.pt",    # 分割模型
    '3': "yolo11n-pose.pt",   # 姿態模型
    '4': "yolo11n-cls.pt",    # 分類模型
}

# 預設載入偵測模型
model_key = '1'
trt_model = YOLO(model_paths[model_key])
current_model_name = "Detect"

# 攝影機設定
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

# 初始化參數
prev_time = time.time()
fps = 0.0
mirror = True  # 預設鏡像開啟

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # ✅ 鏡像處理
    if mirror:
        frame = cv2.flip(frame, 1)

    # 計時
    start_time = time.time()

    # 推論
    results = trt_model.predict(frame, verbose=False)
    plotted_frame = results[0].plot()

    # FPS 平滑計算
    curr_time = time.time()
    instant_fps = 1.0 / (curr_time - prev_time)
    fps = 0.9 * fps + 0.1 * instant_fps
    prev_time = curr_time

    # 顯示 FPS（中上方）
    fps_text = f"FPS: {fps:.2f} | Model: {current_model_name} | Mirror: {'ON' if mirror else 'OFF'}"
    (text_width, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x = int((960 - text_width) / 2)
    y = 30
    cv2.putText(plotted_frame, fps_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 顯示畫面
    cv2.imshow("YOLOv8 TensorRT Webcam Inference", plotted_frame)

    # 鍵盤控制
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif chr(key) in model_paths:
        model_key = chr(key)
        trt_model = YOLO(model_paths[model_key])
        current_model_name = {
            '1': "Detect",
            '2': "Segment",
            '3': "Pose",
            '4': "Classify"
        }[model_key]
        print(f"[INFO] Switched to model: {current_model_name}")
    elif key == ord('m'):
        mirror = not mirror
        print(f"[INFO] Mirror mode: {'ON' if mirror else 'OFF'}")

cap.release()
cv2.destroyAllWindows()

# $  vim Test_MultiCamera_Inspection.py
from ultralytics import YOLO
import cv2
import time

# 各攝影機對應模型（Detect、Segment、Pose）
camera_configs = [
    {"id": 0, "model_path": "yolo11n.pt", "model_name": "Detect"},
    {"id": 2, "model_path": "yolo11n-seg.pt", "model_name": "Segment"},
    {"id": 4, "model_path": "yolo11n-pose.pt", "model_name": "Pose"},
]

# 初始化攝影機與模型
cameras = []
for cfg in camera_configs:
    cap = cv2.VideoCapture(cfg["id"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    model = YOLO(cfg["model_path"])
    cameras.append({
        "cap": cap,
        "model": model,
        "name": cfg["model_name"]
    })

# 初始化參數
prev_time = time.time()
fps = 0.0
mirror = True  # 預設鏡像開啟

while all([cam["cap"].isOpened() for cam in cameras]):
    frames = []
    for cam in cameras:
        success, frame = cam["cap"].read()
        if not success:
            frame = None
        elif mirror:
            frame = cv2.flip(frame, 1)
        frames.append(frame)

    # 跳出回圈如果有任一攝影機失敗
    if any(f is None for f in frames):
        break

    # 推論與畫圖
    plotted_frames = []
    for i, cam in enumerate(cameras):
        results = cam["model"].predict(frames[i], verbose=False)
        plotted = results[0].plot()
        plotted_frames.append(plotted)

    # FPS 平滑計算
    curr_time = time.time()
    instant_fps = 1.0 / (curr_time - prev_time)
    fps = 0.9 * fps + 0.1 * instant_fps
    prev_time = curr_time

    # 顯示每個攝影機的畫面與 FPS
    for i, frame in enumerate(plotted_frames):
        fps_text = f"FPS: {fps:.2f} | Model: {cameras[i]['name']} | Mirror: {'ON' if mirror else 'OFF'}"
        (text_width, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x = int((frame.shape[1] - text_width) / 2)
        y = 30
        cv2.putText(frame, fps_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(f"Camera {i} - {cameras[i]['name']}", frame)

    # 鍵盤操作
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        mirror = not mirror
        print(f"[INFO] Mirror mode: {'ON' if mirror else 'OFF'}")

# 釋放資源
for cam in cameras:
    cam["cap"].release()
cv2.destroyAllWindows()
