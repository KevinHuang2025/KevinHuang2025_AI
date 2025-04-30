# AOPEN_AI
## - Studying for NVidia Orin Solution
## - For JetPack6.2 versopm. running on AGX Orin, Orin NX, and Orin Nano series
#### --- Study 'https://elinux.org/Jetson_Zoo'
#### --- Model Zoo → Object Detection → YOLO11 → Quickstart
#### --- Study 'https://docs.ultralytics.com/guides/nvidia-jetson/#start-with-native-installation'

### # Setup Jetpack Environment
```bash
echo "🔄 Updating package lists..."
sudo apt update

echo "⬆️ Performing system upgrade..."
sudo apt dist-upgrade -y

echo "⚠️ System will reboot to apply upgrades. Please re-run the second script after reboot."
read -p "Press Enter to reboot..."
sudo reboot
```
#### --- After press "Enter" to do system reboot, then

```bash
echo "🧠 Installing NVIDIA JetPack..."
sudo apt install -y nvidia-jetpack

echo "🌐 Installing Chromium browser..."
sudo apt install -y chromium-browser

echo "✅ All done!"

# 檢查是否已經存在相關行，避免重複添加
if ! grep -q 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64' ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# cuda" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64" >> ~/.bashrc
    echo "export PATH=\$PATH:/usr/local/cuda/bin" >> ~/.bashrc
    echo "✅ CUDA 路徑已寫入 ~/.bashrc"
else
    echo "⚠️ CUDA 環境變數已存在於 ~/.bashrc，略過添加"
fi

# 重新載入 .bashrc
source ~/.bashrc
echo "🔄 ~/.bashrc 已重新加載"
```

### # Run JTOP
```bash
echo "🐍 安裝 python3-pip..."
sudo apt update
sudo apt install -y python3-pip

echo "🔍 pip3 版本確認："
pip3 -V

echo "📦 安裝 jetson-stats..."
sudo pip3 install jetson-stats

echo "✅ 安裝完成！你可以用以下指令查看 Jetson 狀態："
echo "👉 sudo jtop"
```
#### --- After install related API, then execute
```bash
sudo jtop
```

### # Run CPU and GPU performance test
```bash
echo "🚀 設定 Jetson 為最高效能模式..."
sudo nvpmodel -m 0
sudo jetson_clocks

echo "🛠️ 下載並編譯 GPU 壓力測試工具..."
git clone https://github.com/anseeto/jetson-gpu-burn.git
cd jetson-gpu-burn
make

echo "🔥 啟動 CPU 壓力測試 (背景執行)..."
sudo apt install -y stress
stress -c $(nproc) &

echo "⏳ 等待 3 秒後執行 GPU 壓力測試..."
sleep 3

echo "⚙️ 啟動 GPU 壓力測試（持續 1000 秒）..."
sudo ./gpu_burn 1000

echo "✅ 測試完成。你可以使用 'htop' 或 'jtop' 觀察資源使用狀況。"
```

### # Run yolo in USB camera
#### --- Setup environment
```bash
echo "🔄 更新套件列表..."
sudo apt update

echo "🐍 安裝 pip..."
sudo apt install -y python3-pip

echo "📦 更新 pip 自身..."
pip install -U pip

echo "🧠 安裝 Ultralytics YOLOv8..."
pip install ultralytics

echo "🔥 安裝 PyTorch 2.5.0 for JetPack 6.1..."
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl

echo "🖼️ 安裝 torchvision 0.20.0..."
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl

echo "🔧 安裝 cuSPARSELt 以解決 torch 相依問題..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev

echo "⚙️ 安裝 ONNX Runtime GPU 1.20.0 (Python 3.10)..."
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl

echo "🔢 安裝 numpy 1.23.5（特定相容版本）..."
pip install numpy==1.23.5

pip3 install pycuda

echo "✅ YOLOv8 + PyTorch 2.5 + ONNX 環境安裝完成！"
```

#### --- using only 1 USB camera in pyton3
#### --- make "1_USB_camera.py" file
```bash
gedit 1_USB_camera.py
```
#### --- copy following in "1_USB_camera.py" file 
```bash
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
```
#### --- then execute
```bash
python3 1_USB_camera.py
```

#### --- using 3 USB camera in python3
#### --- make "3_USB_camera.py" file
```bash
gedit 3_USB_camera.py
```
#### --- copy following in "3_USB_camera.py" file 
```bash
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
```
#### --- then execute
```bash
python3 3_USB_camera.py
```

### # Calculate TOPS by yolo
#### --- Please refer "# Run yolo in USB camera" to setup environment
#### --- make "TOPS.py" file
```bash
gedit TOPS.py
```
#### --- copy following in "TOPS.py" file 
```bash
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

print(f"\n📊 Model: yolo11n.pt")
print(f"⏱️  Average Inference Time: {avg_time*1000:.2f} ms")
print(f"🎯 Estimated FPS: {fps:.2f}")
print(f"⚙️  Estimated TOPS: {tops:.2f} TOPS\n")
```
#### --- then execute
```bash
python3 TOPS.py
```

### # LLM benchmark
#### --- setup LLM environment
#### --- directly copy from "https://www.jetson-ai-lab.com/benchmarks.html"
```bash
#!/usr/bin/env bash
#
# Llama benchmark with MLC. This script should be invoked from the host and will run 
# the MLC container with the commands to download, quantize, and benchmark the models.
# It will add its collected performance data to jetson-containers/data/benchmarks/mlc.csv 
#
# Set the HUGGINGFACE_TOKEN environment variable to your HuggingFace account token 
# that has been granted access to the Meta-Llama models.  You can run it like this:
#
#    HUGGINGFACE_TOKEN=hf_abc123 ./benchmark.sh meta-llama/Llama-2-7b-hf
#
# If a model is not specified, then the default set of models will be benchmarked.
# See the environment variables below and their defaults for model settings to change.
#
# These are the possible quantization methods that can be set like QUANTIZATION=q4f16_ft
#
#  (MLC 0.1.0) q4f16_0,q4f16_1,q4f16_2,q4f16_ft,q4f16_ft_group,q4f32_0,q4f32_1,q8f16_ft,q8f16_ft_group,q8f16_1
#  (MLC 0.1.1) q4f16_0,q4f16_1,q4f32_1,q4f16_2,q4f16_autoawq,q4f16_ft,e5m2_e5m2_f16
#
set -ex

: "${HUGGINGFACE_TOKEN:=SET_YOUR_HUGGINGFACE_TOKEN}"
: "${MLC_VERSION:=0.1.4}"

: "${QUANTIZATION:=q4f16_ft}"
: "${SKIP_QUANTIZATION:=no}"
: "${USE_SAFETENSORS:=yes}"

#: "${MAX_CONTEXT_LEN:=4096}"
: "${MAX_NUM_PROMPTS:=4}"
: "${CONV_TEMPLATE:=llama-2}"
: "${PROMPT:=/data/prompts/completion_16.json}"

: "${OUTPUT_CSV:=/data/benchmarks/mlc.csv}"


function benchmark() 
{
    local model_repo=$1
    local model_name=$(basename $model_repo)
    local model_root="/data/models/mlc/${MLC_VERSION}"
    
    local download_flags="--ignore-patterns='*.pth,*.bin'"

    if [ $USE_SAFETENSORS != "yes" ]; then
      download_flags="--skip-safetensors"
    fi
    
    if [ ${MLC_VERSION:4} -ge 4 ]; then
      if [ -n "$HF_USER" ]; then
        hf_user="$HF_USER"
      else
        if [ $QUANTIZATION = "q4f16_ft" ]; then
          hf_user="dusty-nv"
        else
          hf_user="mlc-ai"
        fi
      fi
      
      mkdir -p $(jetson-containers data)/models/mlc/cache || true ;
      
      run_cmd="\
        python3 benchmark.py \
          --model HF://${hf_user}/${model_name}-${QUANTIZATION}-MLC \
          --max-new-tokens 128 \
          --max-num-prompts 4 \
          --prompt $PROMPT \
          --save ${OUTPUT_CSV} "
        
      if [ -n "$MAX_CONTEXT_LEN" ]; then
        run_cmd="$run_cmd --max-context-len $MAX_CONTEXT_LEN"
      fi
      
      if [ -n "$PREFILL_CHUNK_SIZE" ]; then
        run_cmd="$run_cmd --prefill-chunk-size $PREFILL_CHUNK_SIZE"
      fi
      
      run_cmd="$run_cmd ; rm -rf /data/models/mlc/cache/* || true ; "
    else
      run_cmd="\
        if [ ! -d \${MODEL_REPO} ]; then \
            MODEL_REPO=\$(huggingface-downloader ${download_flags} \${MODEL_REPO}) ; \
        fi ; \
        bash test.sh $model_name \${MODEL_REPO} "
    fi
    
    jetson-containers run \
        -e HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
        -e QUANTIZATION=${QUANTIZATION} \
        -e SKIP_QUANTIZATION=${SKIP_QUANTIZATION} \
        -e USE_SAFETENSORS=${USE_SAFETENSORS} \
        -e MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN} \
        -e MAX_NUM_PROMPTS=${MAX_NUM_PROMPTS} \
        -e CONV_TEMPLATE=${CONV_TEMPLATE} \
        -e PROMPT=${PROMPT} \
        -e OUTPUT_CSV=${OUTPUT_CSV} \
        -e MODEL_REPO=${model_repo} \
        -e MODEL_ROOT=${model_root} \
        -v $(jetson-containers root)/packages/llm/mlc:/test \
        -w /test \
        dustynv/mlc:0.1.4-r36.4.2 /bin/bash -c "$run_cmd"
}
            
   
if [ "$#" -gt 0 ]; then
    benchmark "$@"
    exit 0 
fi


#MLC_VERSION="0.1.0" MAX_CONTEXT_LEN=4096 USE_SAFETENSORS=off benchmark "princeton-nlp/Sheared-LLaMA-1.3B"
#MLC_VERSION="0.1.0" MAX_CONTEXT_LEN=4096 USE_SAFETENSORS=off benchmark "princeton-nlp/Sheared-LLaMA-2.7B"

#MLC_VERSION="0.1.0" MAX_CONTEXT_LEN=4096 benchmark "meta-llama/Llama-2-7b-hf"
#MLC_VERSION="0.1.1" MAX_CONTEXT_LEN=8192 benchmark "meta-llama/Meta-Llama-3-8B"

benchmark "meta-llama/Llama-3.2-1B-Instruct"
benchmark "meta-llama/Llama-3.2-3B-Instruct"
benchmark "meta-llama/Llama-3.1-8B-Instruct"
benchmark "meta-llama/Llama-2-7b-chat-hf"

MAX_CONTEXT_LEN=4096 PREFILL_CHUNK_SIZE=4096 benchmark "Qwen/Qwen2.5-0.5B-Instruct"
MAX_CONTEXT_LEN=4096 PREFILL_CHUNK_SIZE=4096 benchmark "Qwen/Qwen2.5-1.5B-Instruct"
MAX_CONTEXT_LEN=2048 PREFILL_CHUNK_SIZE=1024 benchmark "Qwen/Qwen2.5-7B-Instruct"

QUANTIZATION="q4f16_1" benchmark "google/gemma-2-2b-it"
#QUANTIZATION="q4f16_1" benchmark "google/gemma-2-9b-it"

benchmark "microsoft/Phi-3.5-mini-instruct"

benchmark "HuggingFaceTB/SmolLM2-135M-Instruct"
benchmark "HuggingFaceTB/SmolLM2-360M-Instruct"
benchmark "HuggingFaceTB/SmolLM2-1.7B-Instruct"
```
#### --- make "benchmark.py" file
```bash
gedit benchmark.py
```
#### --- copy following in "benchmark.py" file 
```bash
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 讀取與清理資料
df = pd.read_csv('/home/aopen/jetson-containers/data/benchmarks/mlc.csv')
df.columns = df.columns.str.strip()

df['model'] = df['model'].str.replace('HF://dusty-nv/', '', regex=False)
df['model'] = df['model'].str.replace('HF://mlc-ai/', '', regex=False)
df['model'] = df['model'].str.replace('-q4f16_ft-MLC', '', regex=False)
df['model'] = df['model'].str.replace('-q4f16_l-MLC', '', regex=False)

# 計算平均並依解碼率排序
df_unique = df.groupby('model', as_index=False).mean(numeric_only=True)
df_unique = df_unique.sort_values(by='decode_rate', ascending=False)

models = df_unique['model']
n = len(models)
y_pos = np.arange(n)

fig, ax1 = plt.subplots(figsize=(12, n * 0.6))

# 主軸：記憶體用量（紅色 bar）
memory_pos = y_pos - 0.2
ax1.barh(memory_pos, df_unique['memory'], height=0.35, color='lightcoral', label='Memory Usage (MB)')
ax1.set_xlabel('Memory Usage (MB)', color='red')
ax1.tick_params(axis='x', labelcolor='red')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(models)
ax1.invert_yaxis()

# twin x 軸：解碼率（藍色 bar）
ax2 = ax1.twiny()
decode_pos = y_pos + 0.2
ax2.barh(decode_pos, df_unique['decode_rate'], height=0.35, color='skyblue', label='Decode Rate (tokens/sec)')
ax2.set_xlabel('Decode Rate (tokens/sec)', color='blue')
ax2.tick_params(axis='x', labelcolor='blue')

# 數值標籤（記憶體）
for i, val in enumerate(df_unique['memory']):
    ax1.text(val + 1, memory_pos[i], f'{val:.0f} MB', va='center', color='darkred')

# 數值標籤（解碼率）
for i, val in enumerate(df_unique['decode_rate']):
    ax2.text(val + 1, decode_pos[i], f'{val:.0f} tokens/sec', va='center', color='navy')

# 標題與格式
plt.title('Models comparison', pad=20)
ax1.grid(True, axis='x', linestyle='--', alpha=0.4)
ax2.grid(False)
plt.tight_layout()
plt.show()
```
#### --- then execute
```bash
python3 benchmark.py
```
