# https://www.jetson-ai-lab.com/tutorials/openclaw/
## context overflow issue, switch model 從 qwen3.5:2b 切到 gemma4:e4b

### 整理成一支可執行的 modify_whatsapp.sh，把目前讓 WhatsApp 正確回 stats / recent 所需的設定與重啟
```bash
cd /home/aopen/.openclaw/workspace
gedit  ./modify_whatsapp.sh
```
```bash
#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/home/aopen/.openclaw/workspace"
USER_SYSTEMD_DIR="${HOME}/.config/systemd/user"

echo "[1/5] Refresh BOOTSTRAP.md from ad_stats.db"
python3 "${WORKSPACE}/stats_prompt_context.py"

echo "[2/5] Install user-level stats context service/timer"
mkdir -p "${USER_SYSTEMD_DIR}"
cp "${WORKSPACE}/openclaw-stats-context.service" \
   "${USER_SYSTEMD_DIR}/openclaw-stats-context.service"
cp "${WORKSPACE}/openclaw-stats-context.timer" \
   "${USER_SYSTEMD_DIR}/openclaw-stats-context.timer"

echo "[3/5] Reload user systemd and enable timer"
systemctl --user daemon-reload
systemctl --user enable --now openclaw-stats-context.timer
systemctl --user start openclaw-stats-context.service

echo "[4/5] Restart OpenClaw gateway"
systemctl --user restart openclaw-gateway

echo "[5/5] Current BOOTSTRAP snapshot"
sed -n '1,80p' "${WORKSPACE}/BOOTSTRAP.md"

cat <<'EOF'

Done.

How to test in WhatsApp:
1. Send /reset
2. Send stats
3. Send recent

Expected:
- stats -> 今日資料庫統計
- recent -> 最近 5 筆觀看紀錄

Useful checks:
- systemctl --user status openclaw-gateway --no-pager -l
- systemctl --user status openclaw-stats-context.timer --no-pager -l
- tail -n 20 /home/aopen/.openclaw/agents/main/sessions/4918aa06-22db-4b81-ab66-1d169a48d4cb.jsonl

If WhatsApp still replies with old behavior, send /reset again and retry.
EOF
```
### 跑完後到 WhatsApp 測
```bash
  /reset
  stats
  recent
```
