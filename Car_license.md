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
## Test code for AAA+1111 formate
```bash
import cv2
import numpy as np
import time
import re
from ultralytics import YOLO
from collections import deque, Counter
import threading
import queue
import easyocr
import torch

# --- 全域設定 (Global Settings) ---
IMG_SIZE = 640
DET_CONF = 0.40
OCR_INTERVAL = 3  # 每 3 幀對同一個 ID 觸發一次 OCR
OCR_BATCH_MAX = 16

# --- 初始化模型 (Initialize Models) ---
# 使用 TensorRT 引擎檔以獲得最佳效能
yolo = YOLO("license_plate_detector.engine")

# 初始化 EasyOCR Reader
reader = easyocr.Reader(["en"], gpu=True)

# --- 線程共享資源 (Thread-shared Resources) ---
latest_frame = None
lock = threading.Lock()
stop_event = threading.Event()  # 用於通知所有線程停止的事件

ocr_queue = queue.Queue(maxsize=64)
ocr_results = {}  # 儲存每個 track_id 的最終車牌號碼
plate_hist = {}   # 儲存每個 track_id 的近期原始辨識字串歷史紀錄

# --- 字元轉換與正規表示式 (Character Mapping & Regex) ---
NUM2CHAR = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"}
CHAR2NUM = {v: k for k, v in NUM2CHAR.items()}
ALLOWED_CHARS_REGEX = re.compile(r"[^A-Z0-9]")

# --- 影像預處理函式 ---
def multi_preprocess(gray):
    """對灰階影像應用多種預處理技術，以提高 OCR 成功率。"""
    outs = []
    # 1. CLAHE (限制對比度自適應直方圖等化)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g1 = clahe.apply(gray)
    _, bw1 = cv2.threshold(g1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    outs.append(bw1)

    # 2. 自適應閾值 (對極端光線有幫助)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw2 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 19, 9
    )
    bw2 = cv2.bitwise_not(bw2)
    outs.append(bw2)

    # 3. 銳化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)
    _, bw3 = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    outs.append(bw3)

    # 對較小的圖片進行放大，有助於辨識
    for im in list(outs):
        h, w = im.shape[:2]
        if max(h, w) < 200:
            im2 = cv2.resize(im, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            outs.append(im2)

    # 去除重複的影像變體
    unique = []
    seen = set()
    for im in outs:
        key = (im.shape, im.tobytes()[:40]) # 使用形狀和部分 byte 內容作為唯一標識
        if key not in seen:
            seen.add(key)
            unique.append(im)
    return unique

# --- 文字後處理與平滑化函式 ---
def clean_text(s):
    """清理 OCR 字串，只保留大寫字母和數字。"""
    return ALLOWED_CHARS_REGEX.sub("", s.upper()) if s else ""

def generate_7_substrings(s):
    """從字串中生成所有長度為 7 的子字串。"""
    return [s[i:i+7] for i in range(len(s) - 6)] if len(s) >= 7 else []

def attempt_compose_from_parts(s):
    """嘗試從混亂的字串中組合出 3 個字母 + 4 個數字的格式。"""
    letters = re.findall(r"[A-Z]", s)
    digits = re.findall(r"[0-9]", s)
    if len(letters) >= 3 and len(digits) >= 4:
        return "".join(letters[:3] + digits[-4:])
    return None

def reconstruct_plate_from_history(track_id):
    """從歷史辨識結果中，透過投票機制重建最可能的車牌號碼。"""
    if track_id not in plate_hist or not plate_hist[track_id]:
        return ""

    hist = list(plate_hist[track_id])
    candidate_subs = []

    for s in hist:
        if not s:
            continue
        candidate_subs.extend(generate_7_substrings(s))
        if not candidate_subs:
            comp = attempt_compose_from_parts(s)
            if comp:
                candidate_subs.append(comp)

    if not candidate_subs:
        all_concat = "".join(hist)
        comp = attempt_compose_from_parts(all_concat)
        if comp:
            candidate_subs.append(comp)

    if not candidate_subs:
        return ""

    # 逐字位置投票
    pos_counters = [Counter() for _ in range(7)]
    for sub in candidate_subs:
        if len(sub) == 7:
            for i, ch in enumerate(sub):
                pos_counters[i][ch] += 1

    result_chars = []
    for i in range(7):
        ctr = pos_counters[i]
        if not ctr:
            result_chars.append("A" if i < 3 else "0") # Fallback
            continue

        # 前 3 位偏好字母，後 4 位偏好數字
        is_letter_pos = i < 3
        items = list(ctr.items())
        
        # 根據位置類型排序，偏好的類型優先
        items.sort(key=lambda x: (
            (x[0].isalpha() if is_letter_pos else x[0].isdigit()), # 類型是否匹配
            x[1] # 票數
        ), reverse=True)
        
        best_char = items[0][0]
        result_chars.append(best_char)

    candidate = "".join(result_chars)

    # *** BUG FIX ***: 將這段格式強制轉換的邏輯移到 return 之前
    pref = list(candidate[:3])
    for i in range(3):
        if not pref[i].isalpha():
            pref[i] = NUM2CHAR.get(pref[i], "A")

    suf = list(candidate[3:])
    for i in range(4):
        if not suf[i].isdigit():
            suf[i] = CHAR2NUM.get(suf[i], "0")

    forced = "".join(pref + suf)
    if re.match(r"^[A-Z]{3}[0-9]{4}$", forced):
        return forced

    # 如果強制格式化後仍不匹配，返回空字串，表示辨識失敗
    return ""

def smooth_and_get_final(track_id, raw, maxlen=32):
    """將新的辨識結果加入歷史紀錄，並觸發重建以獲得最終結果。"""
    cleaned = clean_text(raw)
    if track_id not in plate_hist:
        plate_hist[track_id] = deque(maxlen=maxlen)
    plate_hist[track_id].append(cleaned)
    final = reconstruct_plate_from_history(track_id)
    return final

# --- 線程工作函式 ---
def cam_reader(cam_id=0):
    """相機讀取線程：持續從相機抓取最新影像。"""
    global latest_frame
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"錯誤: 無法開啟相機 {cam_id}")
        stop_event.set() # 通知主線程停止
        return
        
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            print("無法讀取相機影像，串流可能已結束。")
            time.sleep(1)
            continue
        with lock:
            latest_frame = frame
    cap.release()
    print("相機讀取線程已停止。")


def ocr_worker():
    """OCR 工作線程：從佇列中獲取任務並執行辨識。"""
    while not stop_event.is_set():
        try:
            # 使用 timeout 讓線程有機會檢查 stop_event
            track_id, gray = ocr_queue.get(timeout=0.1)
            
            variants = multi_preprocess(gray)
            frame_candidates = []

            for var in variants:
                try:
                    res = reader.readtext(var, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ01258")
                    if res:
                        # 簡單地將所有辨識到的文字片段合併
                        full_text = "".join([item[1] for item in res])
                        # 使用 confidence 作為權重 (這裡簡化為平均值)
                        avg_conf = np.mean([item[2] for item in res if len(item) >=3]) if res else 0.0
                        frame_candidates.append((full_text, avg_conf))
                except Exception:
                    pass
            
            chosen_text = ""
            if frame_candidates:
                # 選擇信心度最高的結果
                frame_candidates.sort(key=lambda x: -x[1])
                chosen_text = frame_candidates[0][0]

            final_plate = smooth_and_get_final(track_id, chosen_text)
            ocr_results[track_id] = final_plate

            torch.cuda.empty_cache() # 清理 GPU 記憶體
            ocr_queue.task_done()

        except queue.Empty:
            continue # 佇列為空，繼續等待
        except Exception as e:
            print(f"OCR 工作線程發生錯誤: {e}")
    print("OCR 工作線程已停止。")


# --- 主程式 ---
if __name__ == "__main__":
    # 啟動背景線程
    cam_thread = threading.Thread(target=cam_reader, args=(0,), daemon=True)
    ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
    cam_thread.start()
    ocr_thread.start()

    cv2.namedWindow("License Plate Recognition", cv2.WINDOW_NORMAL)
    prev_time = time.time()
    frame_idx = 0

    try:
        while not stop_event.is_set():
            with lock:
                if latest_frame is None:
                    continue
                frame = latest_frame.copy()

            frame_idx += 1
            do_ocr = (frame_idx % OCR_INTERVAL == 0)

            # 執行 YOLO 偵測與追蹤
            results = yolo.track(
                frame, imgsz=IMG_SIZE, conf=DET_CONF,
                device=0, persist=True, tracker="bytetrack.yaml", verbose=False
            )

            # 在畫面上繪製結果
            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes
                for box in boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        track_id = int(box.id[0])

                        # 裁切車牌區域
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        
                        # 判斷是否需要進行 OCR
                        need_ocr = do_ocr or (track_id not in ocr_results) or (not ocr_results.get(track_id, ""))
                        if need_ocr and not ocr_queue.full():
                            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            ocr_queue.put_nowait((track_id, gray))

                        # 取得並顯示文字
                        text = ocr_results.get(track_id, "...")
                        label = f"ID:{track_id} {text}"
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    except (ValueError, IndexError) as e:
                        print(f"處理 box 時出錯: {e}")
                        continue
            
            # *** 優化 ***: 將 FPS 計算和顯示邏輯移到迴圈末端
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_time))
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (16, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            cv2.imshow("License Plate Recognition", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("偵測到 'q' 鍵，準備關閉程式...")
                break # 跳出迴圈，觸發 finally 區塊

    except KeyboardInterrupt:
        print("偵測到 Ctrl+C，準備關閉程式...")
    finally:
        # *** 新增 ***: 優雅關閉機制
        print("正在停止所有線程...")
        stop_event.set() # 發送停止信號
        
        # 等待線程結束
        cam_thread.join(timeout=2)
        ocr_thread.join(timeout=2)
        
        cv2.destroyAllWindows()
        print("程式已完全關閉。")
```
