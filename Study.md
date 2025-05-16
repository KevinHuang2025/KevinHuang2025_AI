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



