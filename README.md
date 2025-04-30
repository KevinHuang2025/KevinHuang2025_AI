# AOPEN_AI
# Studying for NVidia Orin Solution
# For JetPack6.2 versopm. running on AGX Orin, Orin NX, and Orin Nano series

		# Setup Jetpack Environment
		```bash
		echo "🔄 Updating package lists..."
		sudo apt update
		
		echo "⬆️ Performing system upgrade..."
		sudo apt dist-upgrade -y
		
		echo "⚠️ System will reboot to apply upgrades. Please re-run the second script after reboot."
		read -p "Press Enter to reboot..."
		sudo reboot
		```
		# After press "Enter" to do system reboot, then
		
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
		
		
		#------------------------------------------------------------------------------------------------------------------------------------------------------------------
		# Study 'https://elinux.org/Jetson_Zoo'
		# Model Zoo → Object Detection → YOLO11 → Quickstart
		# Study 'https://docs.ultralytics.com/guides/nvidia-jetson/#start-with-native-installation'
		
		```bash
		sudo apt update
		```

