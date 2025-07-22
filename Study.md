# Linux commands
## 查看某個插件有哪些版本/ check apt version
### EX: check 'vsftpd'
```bash
sudo apt-cache policy vsftpd
```
## 安裝插件/ Install/Upgrade the latest package
### EX: install 'vsftpd'
```bash
sudo apt-get install vsftpd
```
## 安裝指定版本的插件/ Install specific package version
```bash
sudo apt-get install vsftpd=2.3.5-ubuntu1
```

## 更新軟體的最新資訊及列表/ renew apt list
```bash
sudo apt-get update
```
## 更新目前已安裝的軟體到最新版本/ update apt latest version
```bash
sudo apt-get upgrade
```
## 移除插件/ remove apt
### EX: remove 'vsftpd'
```bash
sudo apt-get remove vsftpd
```
## 移除插件，並同時移除設定檔/ remove apt and configuration file
### EX: remove 'vsftpd' and configuration files
```bash
sudo apt-get purge vsftpd
```
## 移除插件相依套件
```bash
sudo apt autoremove
```
## 清除之前下載的安裝檔 (*.deb)/ Remove all package files
```bash
sudo apt clean
```
## 清除過期的安裝檔/ Remove package files is not installed
```bash
sudo apt autoclean
```
# NVIDIA Jetson
## Jetson_release
```bash
sudo pip3 install -U jetson-stats
jetson_release
```
## Jetson Orin Nano 8G - JetPack 6.0
![image](https://github.com/user-attachments/assets/cfd8af4e-0716-48ef-a365-57bc2ffc047c)
![image](https://github.com/user-attachments/assets/c56460f8-d36f-4d83-98c7-4cde01d70fd4)
## Jetson Orin Nano 8G - JetPack 6.2
![image](https://github.com/user-attachments/assets/78cb939f-ebf6-4869-afb6-2d4b0dc22c52)
![image](https://github.com/user-attachments/assets/c14ad60d-f19a-4c95-a797-4810b3fb726a)
![image](https://github.com/user-attachments/assets/c4f1072a-2cb0-411b-953e-3788d86fd7a3)
![image](https://github.com/user-attachments/assets/1b09c171-eff4-4600-b07e-8c4f94411b39)

___
# LLM
## 安裝 jetson-containers 工具
```bash
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers
bash install.sh
```
## 啟動 Stable Diffusion Web UI 容器
```bash
./run.sh $(./autotag stable-diffusion-webui)
```
## 訪問 Web UI
### 打開瀏覽器並訪問 http://localhost:7860

___
# NVLLM
## check NV containers
```bash
jetson-containers list
```
## Run NV LLM
```bash
jetson-containers run dustynv/stable-diffusion-webui:r36.2.0
```
## use browser to open "http://0.0.0.0:7860"
## you can type txt to generate a photo now

___
# NVLLM
## NV Ollama, https://hub.docker.com/r/dustynv/ollama
## Ollama Server
```bash
# models cached under jetson-containers/data
jetson-containers run --name ollama $(autotag ollama)

# models cached under your user's home directory
docker run --runtime nvidia -it -rm --network=host -v ~/ollama:/ollama -e OLLAMA_MODELS=/ollama dustynv/ollama:r36.2.0
```
## Ollama Client: mistral
```bash
# if running inside the same container as launched above
/bin/ollama run mistral

# if launching a new container for the client in another terminal
jetson-containers run $(autotag ollama) /bin/ollama run mistral
```
## Ollama Client: llama3.2:3b
```bash
ollama run llama3.2:3b
```
___
# 中文輸入法
```bash
sudo apt update
sudo apt install fcitx5 fcitx5-chinese-addons fcitx5-config-qt fcitx5-frontend-gtk3
```
