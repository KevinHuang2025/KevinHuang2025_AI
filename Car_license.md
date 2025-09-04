# Study for Car License inspection
```bash
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter
import threading, queue
import easyocr

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
ocr_queue = queue.Queue(maxsize=4)
ocr_results = {}

plate_hist = {}

def smooth_text(plate_id, raw, maxlen=8):
    if plate_id not in plate_hist:
        plate_hist[plate_id] = deque(maxlen=maxlen)
    hist = plate_hist[plate_id]
    hist.append(raw)
    if not hist:
        return raw
    cand = Counter(hist).most_common(1)[0][0]
    if all(len(s) == len(cand) for s in hist):
        final = ""
        for i in range(len(cand)):
            final += Counter(s[i] for s in hist).most_common(1)[0][0]
        return final
    return cand

def ocr_worker():
    while True:
        try:
            plate_id, gray = ocr_queue.get()
            if gray.size == 0:
                continue
            text = reader.readtext(
                gray,
                detail=0,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )
            text = "".join(text)
            text = smooth_text(plate_id, text)
            ocr_results[plate_id] = text
        except Exception as e:
            print("OCR error:", e)

threading.Thread(target=ocr_worker, daemon=True).start()

# --- 主推論迴圈 ---
while True:
    if frame_queue.empty():
        continue
    frame = frame_queue.get()

    results = yolo(frame, imgsz=640, device=0)
    for r in results:
        for i, b in enumerate(r.boxes.xyxy.cpu().numpy()):
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, b[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue  # 無效框

            crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            if not ocr_queue.full():
                ocr_queue.put((i, gray))

            text = ocr_results.get(i, "")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Jetson Plate OCR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```
