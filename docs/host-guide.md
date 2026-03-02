# 主機設定手冊（給朋友看的）

你提供了 GB10 給朋友部署 AI 語音合成服務。這份手冊說明你需要在自己的機器上做什麼事，讓外部可以打 request 進來。

---

## 你需要做的事（一次性設定）

### 1. 確認 Docker 和 NVIDIA Container Toolkit 已安裝

```bash
# 確認 Docker
docker --version        # 需要 ≥ 24

# 確認 Docker Compose
docker compose version  # 需要 ≥ 2.x

# 確認 NVIDIA driver
nvidia-smi
```

如果 Docker 未安裝：

```bash
# Ubuntu / Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

如果 NVIDIA Container Toolkit 未安裝（讓 Docker 能用 GPU）：

```bash
# 加 NVIDIA 套件庫
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 設定 Docker 使用 NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 驗證
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

### 2. 開放防火牆 port

服務會用到兩個 port：
- **8000** — REST API（朋友的程式/工具用）
- **7860** — Gradio 網頁介面（瀏覽器用）

```bash
# 如果用 ufw（Ubuntu 預設）
sudo ufw allow 8000/tcp comment 'cosyvoice3 API'
sudo ufw allow 7860/tcp comment 'cosyvoice3 Gradio UI'
sudo ufw reload

# 確認規則
sudo ufw status verbose
```

如果沒有 ufw，用 iptables：

```bash
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 7860 -j ACCEPT

# 讓規則重開機後仍然有效
sudo apt-get install -y iptables-persistent
sudo netfilter-persistent save
```

---

### 3. 確認公網 IP 並告訴朋友

```bash
# 查你的公網 IP
curl ifconfig.me
# 或
curl ipinfo.io/ip
```

把這個 IP 告訴朋友（操作者）。他會用這個 IP 從外部連進來。

如果你的 GB10 在**路由器後面**（家用 / 辦公室網路），還需要做 port forwarding（見第 4 步）。

如果你的 GB10 是**直接連網路的伺服器**（有公網 IP 直接在機器上），跳過第 4 步。

---

### 4. 路由器 Port Forwarding（若 GB10 在 NAT 後面）

登入你的路由器管理介面（通常是 `http://192.168.1.1` 或 `http://192.168.0.1`），找到「Port Forwarding」、「虛擬伺服器」或「NAT」設定，新增兩條規則：

| 外部 Port | 內部 IP | 內部 Port | 協定 |
|-----------|---------|-----------|------|
| 8000 | GB10 的區域網路 IP | 8000 | TCP |
| 7860 | GB10 的區域網路 IP | 7860 | TCP |

查 GB10 的區域網路 IP：

```bash
ip addr show | grep 'inet ' | grep -v '127.0.0.1'
# 找 192.168.x.x 或 10.x.x.x 那行
```

設定好後，從**外部網路**測試（可以用手機熱點）：

```bash
curl http://你的公網IP:8000/health
```

---

### 5. WireGuard 路由問題（選做，但建議修）

你的 WireGuard 設定目前把朋友連進來後的**所有流量都導向 VPN tunnel**，導致 GB10 對外的網路部分不通。這是因為 `AllowedIPs = 0.0.0.0/0` 的設定。

**修法：改成 split tunnel，只讓 VPN 的流量走 tunnel，其他保持正常路由**

找到你的 WireGuard server 設定（通常在 `/etc/wireguard/wg0.conf`），找到 peer 的 `[Peer]` 區段：

```ini
# 現在可能長這樣（全路由）
[Peer]
PublicKey = 朋友的公鑰
AllowedIPs = 0.0.0.0/0, ::/0   # ← 這行是問題所在
```

改成只允許 VPN 子網路：

```ini
# 改成 split tunnel
[Peer]
PublicKey = 朋友的公鑰
AllowedIPs = 10.0.0.2/32   # ← 只允許這個 peer 的 VPN IP
```

套用設定：

```bash
sudo wg syncconf wg0 <(wg-quick strip wg0)
# 或直接重啟
sudo systemctl restart wg-quick@wg0
```

> 如果不確定 VPN IP 範圍，執行 `sudo wg show` 查看目前連線狀況。

---

### 6. 讓服務開機自動啟動（選做）

確認服務已在運作後，讓它開機自動啟動：

```bash
cd ~/cosyvoice3-api   # 專案目錄

# 建立 systemd service
sudo tee /etc/systemd/system/cosyvoice3.service > /dev/null <<EOF
[Unit]
Description=CosyVoice3 API + Gradio
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable cosyvoice3.service
```

驗證：

```bash
sudo systemctl start cosyvoice3.service
sudo systemctl status cosyvoice3.service
```

---

## 你需要給朋友的資訊

設定完成後，把以下資訊傳給朋友（操作者）：

```
公網 IP：____________
SSH user：____________
GB10 內網 IP（VPN 連線後用）：____________
WireGuard VPN 設定檔（如果需要）
```

---

## 快速確認清單

```
[ ] Docker ≥ 24 已安裝
[ ] NVIDIA Container Toolkit 已安裝，docker run --gpus 能看到 GPU
[ ] ufw / iptables 開放了 8000 和 7860
[ ] 如果在 NAT 後面：路由器 port forwarding 已設定
[ ] 公網 IP 已告知朋友（操作者）
[ ] （建議）WireGuard 改成 split tunnel
[ ] （選做）cosyvoice3.service 已設開機自啟
```
