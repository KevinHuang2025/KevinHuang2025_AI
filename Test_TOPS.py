# $ pip3 install pycuda

#vim Test_TOPS.py
from ultralytics import YOLO
import numpy as np
import torch
import time

model = YOLO("yolo11n.pt")
#model = YOLO("yolo11n-seg.pt")
#model = YOLO("yolo11n-pose.pt")
#model = YOLO("yolo11n-cls.pt")
dummy_input = torch.rand(1, 3, 640, 640)  # 模型大小視情況調整

# 預熱
for _ in range(10):
    model(dummy_input)

# 測試
start = time.time()
num_runs = 50
for _ in range(num_runs):
    model(dummy_input)
end = time.time()

avg_time = (end - start) / num_runs
fps = 1.0 / avg_time
gflops = 6.5
tops = gflops / avg_time

print(f"\n📊 Model: yolo11n.engine")
print(f"⏱️  Average Inference Time: {avg_time*1000:.2f} ms")
print(f"🎯 Estimated FPS: {fps:.2f}")
print(f"⚙️  Estimated TOPS: {tops:.2f} TOPS\n")
