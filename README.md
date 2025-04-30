# AOPEN_AI
## Studying for NVidia Orin Solution
## For JetPack6.2 versopm. running on AGX Orin, Orin NX, and Orin Nano series
#### Study 'https://elinux.org/Jetson_Zoo'
#### Model Zoo → Object Detection → YOLO11 → Quickstart
#### Study 'https://docs.ultralytics.com/guides/nvidia-jetson/#start-with-native-installation'

### #Setup Jetpack Environment
```bash
echo "🔄 Updating package lists..."
sudo apt update

echo "⬆️ Performing system upgrade..."
sudo apt dist-upgrade -y

echo "⚠️ System will reboot to apply upgrades. Please re-run the second script after reboot."
read -p "Press Enter to reboot..."
sudo reboot
```
#### After press "Enter" to do system reboot, then

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

### #Run JTOP
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
#### After install related API, then execute
```bash
sudo jtop
```

### #Run CPU and GPU performance test
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
