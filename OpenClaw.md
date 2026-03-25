# Study from Nvidia AI Lab
### 'https://www.jetson-ai-lab.com/tutorials/openclaw/'
# Study from chatgpt codex
### 'https://chatgpt.com/codex'
### 'https://chatgpt.com/codex'
### install codex
```bash
npm i -g @openai/codex
````
### check update version
```bash
npm i -g @openai/codex@latest
````
### Run codex
```bash
codex
```
### 匯出您的 Hugging Face 令牌
```bash
export HF_TOKEN=your_huggingface_token_here
```
## Nemotron 3 Nano 30B-A3B
### 步驟 1：使用 vLLM 提供本機模型
```bash
sudo docker run -it --rm --pull always \
  --runtime=nvidia --network host \
  -e HF_TOKEN=$HF_TOKEN \
  -e VLLM_USE_FLASHINFER_MOE_FP4=1 \
  -e VLLM_FLASHINFER_MOE_BACKEND=throughput \
  -v $HOME/.cache/huggingface:/data/models/huggingface \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
  bash -c "wget -q -O /tmp/nano_v3_reasoning_parser.py \
  --header=\"Authorization: Bearer \$HF_TOKEN\" \
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4/resolve/main/nano_v3_reasoning_parser.py \
  && vllm serve stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ \
  --gpu-memory-utilization 0.8 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin /tmp/nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3 \
  --kv-cache-dtype fp8"
```
### 提示：這些模型需要大量記憶體。運行前，請確保沒有其他進程佔用 GPU 記憶體。最好先清除記憶體緩存，以確保有盡可能多的可用記憶體：
```bash
sudo sysctl -w vm.drop_caches=3
```
### vLLM啟動並運行後，打開另一個終端並進行驗證：
```bash
curl -s http://127.0.0.1:8000/v1/models
```
### 步驟 2：安裝 Node.js 22+
```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
```
### 快速核查：
```bash
node --version   # should be v22.x.x or higher
```
### 步驟 3：安裝 OpenClaw
```bash
sudo npm install -g openclaw@latest
```
### 快速核查：
```bash
openclaw --version
```
### 步驟 4：執行新使用者引導程式
```bash
openclaw onboard --skip-daemon
```
### 模型/身份驗證提供程序
### 基本 URL	http://127.0.0.1:8000/v1（保留預設設定）
### API金鑰	任何隨機字串（例如vllm-local），但不要留空。
### 型號名稱	vLLM 所服務的確切模型名稱（例如nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4）
### 啟動網關
```bash
nohup openclaw gateway run > /tmp/openclaw-gateway.log 2>&1 &
```
### 請稍等幾秒鐘使其啟動，然後檢查狀態：
```bash
openclaw channels status --probe
```
