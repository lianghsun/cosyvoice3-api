# 操作者部署手冊（給你看的）

目標機器：朋友的 GB10（Linux + NVIDIA GPU）
連線方式：WireGuard VPN
已知問題：VPN 進去後部分對外網站不通（路由問題），因此**所有需要下載的東西都要在自己電腦上先準備好**。

---

## 階段一：在自己電腦上準備好所有東西（連 VPN 之前）

### 1-1. Clone repo

```bash
git clone https://github.com/lianghsun/cosyvoice3-api.git
cd cosyvoice3-api
```

### 1-2. 建立 .env

```bash
cp .env.example .env
```

編輯 `.env`，至少填好：

```env
HF_TOKEN=hf_你的token
HF_REPO_ID=你的帳號/tts-outputs
LOAD_FP16=true
LOAD_VLLM=false   # 先關掉，等環境穩定再開
LOAD_TRT=false
```

### 1-3. 下載模型權重（~9.75 GB）

在**自己電腦**上下載，之後 scp 過去：

```bash
mkdir -p models
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    local_dir='models/Fun-CosyVoice3-0.5B',
    ignore_patterns=['*.git*'],
)
print('Done.')
"
```

> 如果本機也沒裝 huggingface_hub：`pip install huggingface_hub`

### 1-4. 預先拉 Docker images 並存成 tar

這是最關鍵的一步。VPN 進去後如果 docker pull 不通，就靠這些 tar 檔。

```bash
# 拉 base images
docker pull pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
docker pull python:3.11-slim

# 把它們存成 tar（可能各需要幾分鐘）
docker save pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime | gzip > pytorch_2.3.1_cuda12.1.tar.gz
docker save python:3.11-slim | gzip > python_3.11_slim.tar.gz

ls -lh *.tar.gz   # 確認檔案有產生
```

> `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` 約 6–7 GB，壓縮後約 5 GB。

### 1-5. 確認準備清單

連上 VPN 之前，確認本機目錄裡有以下東西：

```
cosyvoice3-api/
├── .env                          ← 已填好
├── models/Fun-CosyVoice3-0.5B/  ← 模型權重（~9.75 GB）
├── pytorch_2.3.1_cuda12.1.tar.gz ← Docker base image
└── python_3.11_slim.tar.gz       ← Docker base image
```

---

## 階段二：連上 VPN，把東西傳過去

### 2-1. 連 WireGuard VPN

```bash
# 視你的設定而定，大概是：
wg-quick up 你的config名稱
# 或 macOS 用 WireGuard app 手動點連線
```

確認 VPN 通了：

```bash
ping 朋友的GB10內網IP   # 確認能 ping 到
ssh user@GB10內網IP     # 確認 SSH 能進去
```

### 2-2. 把東西傳到 GB10

在自己電腦上執行（替換 `user` 和 `GB10_IP` 為實際值）：

```bash
GB10="user@GB10_IP"
REMOTE_DIR="~/cosyvoice3-api"

# 建遠端目錄
ssh $GB10 "mkdir -p $REMOTE_DIR"

# 傳整個 repo（排除 .git 和大型目錄）
rsync -avz --progress \
  --exclude='.git' \
  --exclude='models' \
  --exclude='.venv' \
  --exclude='CosyVoice' \
  ./ $GB10:$REMOTE_DIR/

# 傳模型權重（最大，可能需要很長時間）
rsync -avz --progress \
  models/ $GB10:$REMOTE_DIR/models/

# 傳 Docker image tar 檔
scp pytorch_2.3.1_cuda12.1.tar.gz $GB10:~/
scp python_3.11_slim.tar.gz $GB10:~/
```

---

## 階段三：SSH 進 GB10，部署

```bash
ssh user@GB10_IP
cd ~/cosyvoice3-api
```

### 3-1. 確認 Docker + NVIDIA Container Toolkit 已安裝

```bash
docker --version          # 需要 ≥ 24
docker compose version    # 需要 ≥ 2.x
nvidia-smi                # 確認 GPU 可見
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
# ↑ 最後這行確認 Docker 能使用 GPU
```

如果 `docker run --gpus` 報錯，表示 NVIDIA Container Toolkit 未安裝，請通知朋友（見 host-guide.md）。

### 3-2. 載入預先準備的 Docker images（避免重新下載）

```bash
docker load < ~/pytorch_2.3.1_cuda12.1.tar.gz
docker load < ~/python_3.11_slim.tar.gz

# 確認 images 已載入
docker images | grep -E 'pytorch|python'
```

### 3-3. 如果網路通的話：Build images

```bash
docker compose build
```

如果網路不通，CosyVoice clone 會失敗。這時候有兩個選項：

**選項 A（推薦）：在 Dockerfile 裡改用本機已有的 CosyVoice**

先把 CosyVoice 也 rsync 過來，再改一行 Dockerfile：

在自己電腦上：
```bash
# 先 clone CosyVoice（如果沒有的話）
git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice && git submodule update --init --recursive && cd ..

# 傳到 GB10
rsync -avz --progress \
  --exclude='.git' \
  CosyVoice/ user@GB10_IP:~/cosyvoice3-api/CosyVoice/
```

在 GB10 上，修改 `Dockerfile`，把 `git clone` 那段改成直接 `COPY`：

```bash
# 備份原始 Dockerfile
cp Dockerfile Dockerfile.orig

# 用 sed 把 git clone 整段替換成 COPY
# 找到這兩行：
#   RUN git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice \
#       && cd /app/CosyVoice && git submodule update --init --recursive
# 改成：
#   COPY CosyVoice /app/CosyVoice
```

用編輯器修改（`nano Dockerfile` 或 `vim Dockerfile`）：
找到 `RUN git clone` 那幾行，整個替換成：
```dockerfile
COPY CosyVoice /app/CosyVoice
```

然後 build：
```bash
docker compose build
```

**選項 B：等網路通了再 build**

如果朋友確認某些網站通的話，可以先試試：
```bash
curl -I https://github.com    # 測試 GitHub 是否可達
curl -I https://pypi.org      # 測試 PyPI 是否可達
docker compose build --no-cache 2>&1 | tee build.log
```

### 3-4. 啟動服務

```bash
docker compose up -d

# 確認兩個容器都起來了
docker compose ps

# 看 API 的 log（等 model 載入完成，約 1–3 分鐘）
docker compose logs -f api
# 看到 "Model loaded. Sample rate: 24000 Hz" 就代表好了
```

### 3-5. 本機驗證（在 GB10 上測試）

```bash
# Health check
curl http://localhost:8000/health

# 應該回傳類似：
# {"status":"ok","model_loaded":true,"queue_depth":0,...}

# Gradio UI
curl -I http://localhost:7860
# 應該回 200
```

---

## 階段四：從外部驗證

朋友把防火牆和 port forward 設好後（見 host-guide.md），從**自己電腦**（或任何外部）測試：

```bash
PUBLIC_HOST="朋友給你的公網IP或域名"

# API health
curl http://$PUBLIC_HOST:8000/health

# Gradio UI（瀏覽器開）
open http://$PUBLIC_HOST:7860
```

---

## 常見問題排除

### `docker compose build` 在 git clone 步驟掛掉

→ 用上面的「選項 A」，把 CosyVoice 改成 COPY 本機目錄。

### `nvidia-smi` 在容器裡看不到 GPU

```bash
# 確認 host 的 nvidia-smi 能跑
nvidia-smi

# 確認 Container Toolkit 設定了 default runtime
cat /etc/docker/daemon.json
# 應該要有 "default-runtime": "nvidia"
# 如果沒有，通知朋友執行 host-guide.md 的安裝步驟
```

### Port 連不上

```bash
# 在 GB10 上確認 container 有監聽
docker compose ps
ss -tlnp | grep -E '8000|7860'

# 確認防火牆沒擋
sudo ufw status
```

### 模型載入時 OOM（GPU 記憶體不足）

GB10 的 GPU 記憶體很大，通常不會發生。如果真的 OOM：
```bash
# 先關掉 TRT 和 vLLM，只開 fp16
# 編輯 .env：
LOAD_FP16=true
LOAD_VLLM=false
LOAD_TRT=false

docker compose down && docker compose up -d
```

### WireGuard 網路問題（在 GB10 上 ping 外部失敗）

這是朋友的路由設定問題。臨時繞過的方法（在 GB10 上執行）：

```bash
# 確認預設路由
ip route show

# 如果對外流量走了 wg 介面，可以暫時加一條直接路由
# （這只是測試用，重啟後失效）
ip route add 1.1.1.1/32 via $(ip route | grep default | awk '{print $3}') dev eth0
curl -I https://1.1.1.1    # 測試
```

長期解法要請朋友修 WireGuard 設定（見 host-guide.md）。
