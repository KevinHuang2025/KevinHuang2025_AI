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
## 如果跑完 nvidia-ctk cdi list 還是沒看到 nvidia.com/gpu 這個項目,再手動補產生一次:
```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list
```
```bash
nemoclaw onboard
```
## link webstie http://127.0.0.1:18790/ and run
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
```bash
nemoclaw onboard --resume
docker run --rm --runtime=nvidia nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```
## 重新拿一份「當下有效」的 dashboard 網址(帶新 token)：
```bash
nemoclaw dev2730 dashboard-url --quiet
```
# Install Codex Cli
## Codex Cli
## Switch to vllm
```bash
ollama pull gpt-oss:120b
```
## 在另一个终端启动本地 llm：
```bash
ollama serve
```
### 之后每次使用只需：--oss 表示 Codex 使用本机 Ollama；不使用云端模型。
```bash
systemctl --user start ollama-local
cd ~/projects
codex --oss -m gpt-oss:120b
```
