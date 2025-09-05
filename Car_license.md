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

___
# 🔧 訓練流程建議
## 1. 解壓縮資料集, download "EnglishFnt.tgz"
```bash
cd /data
tar -xvzf EnglishFnt.tgz
```
### 這些資料依字元類別編號存放。

## 2. 轉成 PyTorch Dataset
### 你可以寫一個 Dataset 類別，將這些字元影像轉成 label（A–Z, 0–9）。
```bash
import os
import cv2
from torch.utils.data import Dataset

class EnglishFntDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform

        for class_dir in sorted(os.listdir(root)):
            class_path = os.path.join(root, class_dir)
            if not os.path.isdir(class_path):
                continue

            # Sample001 → 0, Sample002 → 1, ...
            class_id = int(class_dir.replace("Sample", "")) - 1

            for fname in os.listdir(class_path):
                if fname.endswith(".png"):
                    path = os.path.join(class_path, fname)
                    self.samples.append((path, class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            img = self.transform(img)
        return img, label


# Label 對照表
idx_to_char = (
    [chr(i) for i in range(ord('A'), ord('Z')+1)] +   # 0-25 : A-Z
    [chr(i) for i in range(ord('a'), ord('z')+1)] +   # 26-51 : a-z
    [chr(i) for i in range(ord('0'), ord('9')+1)]     # 52-61 : 0-9
)


# 測試
if __name__ == "__main__":
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    dataset = EnglishFntDataset("/home/aopen/data/English/Fnt", transform)
    print("資料筆數:", len(dataset))
    img, label = dataset[0]
    print("第一筆標籤:", label, "字元:", idx_to_char[label])

```

## 3. Train 模型
```bash
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# =============== Dataset ===============
class EnglishFntDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform

        for class_dir in sorted(os.listdir(root)):
            class_path = os.path.join(root, class_dir)
            if not os.path.isdir(class_path):
                continue
            class_id = int(class_dir.replace("Sample", "")) - 1
            for fname in os.listdir(class_path):
                if fname.endswith(".png"):
                    path = os.path.join(class_path, fname)
                    self.samples.append((path, class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            img = self.transform(img)
        return img, label


# =============== Model ===============
class OCRNet(nn.Module):
    def __init__(self, num_classes=62):
        super(OCRNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # (1,32,32) -> (32,32,32)
            nn.ReLU(),
            nn.MaxPool2d(2),                 # (32,16,16)

            nn.Conv2d(32, 64, 3, padding=1), # (64,16,16)
            nn.ReLU(),
            nn.MaxPool2d(2),                 # (64,8,8)

            nn.Conv2d(64, 128, 3, padding=1),# (128,8,8)
            nn.ReLU(),
            nn.MaxPool2d(2),                 # (128,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============== Training ===============
def train():
    # 路徑
    data_root = "/home/aopen/data/English/Fnt"

    # Transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Dataset
    dataset = EnglishFntDataset(data_root, transform)

    # Train / Val split (9:1)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OCRNet(num_classes=62).to(device)

    # Loss / Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Acc: {val_acc:.2f}%")

    # Save model
    torch.save(model.state_dict(), "ocr_englishfnt.pth")
    print("模型已儲存為 ocr_englishfnt.pth")


if __name__ == "__main__":
    train()

```

## 強化版 train.py
### 換 ResNet18（最有效）
### 增加 epoch 到 30–50
### 加 data augmentation（旋轉、平移、噪聲）
```bash
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

# =============== Dataset ===============
class EnglishFntDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform

        for class_dir in sorted(os.listdir(root)):
            class_path = os.path.join(root, class_dir)
            if not os.path.isdir(class_path):
                continue
            class_id = int(class_dir.replace("Sample", "")) - 1
            for fname in os.listdir(class_path):
                if fname.endswith(".png"):
                    path = os.path.join(class_path, fname)
                    self.samples.append((path, class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            img = self.transform(img)
        return img, label


# =============== Model ===============
def build_resnet18(num_classes=62):
    model = models.resnet18(weights=None)   # 不用 ImageNet 預訓練
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 改成單通道
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# =============== Training ===============
def train():
    # 路徑 (建議確認實際位置)
    data_root = os.path.expanduser("~/data/English/Fnt")

    # Transform + Augmentation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),         # 隨機旋轉 ±10 度
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 平移 & 縮放
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = EnglishFntDataset(data_root, transform)

    # Train / Val split (9:1)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(num_classes=62).to(device)

    # Loss / Optimizer / Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Training loop
    epochs = 30
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = 100 * val_correct / val_total

        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Acc: {val_acc:.2f}% "
              f"LR: {scheduler.get_last_lr()[0]:.5f}")

    # Save model
    torch.save(model.state_dict(), "ocr_resnet18.pth")
    print("✅ 模型已儲存為 ocr_resnet18.pth")


if __name__ == "__main__":
    train()
```
