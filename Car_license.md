# Study for Car License inspection
## 安裝 EasyOCR
```bash
pip3 install easyocr
```
## 安裝 GUI 支援套件（適合你要在螢幕顯示）
```bash
sudo apt update
sudo apt install -y libgtk2.0-dev pkg-config
sudo apt install -y libcanberra-gtk-module libcanberra-gtk3-module
```
## 如果出現 NumPy 2.x / PyTorch 不相容 的錯誤，可以先降級：
```bash
pip3 install "numpy<2"
```
## 確認你安裝的 OpenCV 有 GUI 支援 (GTK+ 或 QT)。
```bash
sudo apt-get install libgtk-3-dev
pip3 uninstall opencv-python opencv-python-headless -y
pip3 install --no-cache-dir opencv-python==4.6.0.66
```

## 轉換模型到 TensorRT, 成功後會產生一個： license_plate_detector.engine
```bash
yolo export model=license_plate_detector.pt format=engine device=0
```

## Code
```bash
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter
import threading, queue
import easyocr
import torch

# --- 初始化 YOLO (TensorRT engine) ---
yolo = YOLO("license_plate_detector.engine")

# --- 攝影機影像緩衝區 ---
frame_queue = queue.Queue(maxsize=2)

def cam_reader(cam_id=2):   # 使用 camera 2
    cap = cv2.VideoCapture(cam_id)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not frame_queue.full():
            frame_queue.put(frame)

threading.Thread(target=cam_reader, daemon=True).start()

# --- OCR (EasyOCR with GPU, 單執行緒) ---
reader = easyocr.Reader(["en"], gpu=True)
ocr_queue = queue.Queue(maxsize=16)
ocr_results = {}
plate_hist = {}

def smooth_text(track_id, raw, maxlen=32):
    if track_id not in plate_hist:
        plate_hist[track_id] = deque(maxlen=maxlen)
    hist = plate_hist[track_id]
    hist.append(raw)

    valid_hist = [s for s in hist if s]
    if not valid_hist:
        return raw

    cand = Counter(valid_hist).most_common(1)[0][0]
    if all(len(s) == len(cand) for s in valid_hist):
        final = ""
        for i in range(len(cand)):
            final += Counter(s[i] for s in valid_hist).most_common(1)[0][0]
        return final
    return cand

def ocr_worker():
    while True:
        try:
            batch = []
            while not ocr_queue.empty() and len(batch) < 4:
                track_id, gray = ocr_queue.get()
                if gray is not None and gray.size > 0:
                    batch.append((track_id, gray))

            if not batch:
                continue

            # 批次 OCR
            for track_id, gray in batch:
                text = reader.readtext(
                    gray,
                    detail=0,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )
                text = "".join(text).strip()
                text = smooth_text(track_id, text)
                ocr_results[track_id] = text

            # 避免 CUDA 記憶體累積
            torch.cuda.empty_cache()

        except Exception as e:
            print("OCR error:", e)

threading.Thread(target=ocr_worker, daemon=True).start()

# --- 主推論迴圈 ---
cv2.namedWindow("Jetson Plate OCR", cv2.WINDOW_NORMAL)

while True:
    if frame_queue.empty():
        continue
    frame = frame_queue.get()

    # ⚡ 使用 track=True 啟用 ByteTrack
    results = yolo.track(frame, imgsz=640, device=0, persist=True, tracker="bytetrack.yaml")

    if results[0].boxes.id is None:
        continue

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        track_id = int(box.id.cpu().numpy())  # ByteTrack 的 ID

        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if not ocr_queue.full():
            try:
                ocr_queue.put_nowait((track_id, gray))
            except queue.Full:
                pass

        text = ocr_results.get(track_id, "")

        # --- 繪製車牌框 + OCR 結果 ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id} {text}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Jetson Plate OCR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

```

