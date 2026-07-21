# nemoclaw

## 安裝 NVIDIA Container Toolkit 並產生 CDI spec(這是裝好之後 Docker/OpenShell 才能把 Thor GPU 遞送進 sandbox 的關鍵):
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo mkdir -p /etc/cdi
sudo systemctl enable --now nvidia-cdi-refresh.path nvidia-cdi-refresh.service
sudo systemctl start nvidia-cdi-refresh.service

nvidia-ctk cdi list   # 確認有出現 nvidia.com/gpu
```
