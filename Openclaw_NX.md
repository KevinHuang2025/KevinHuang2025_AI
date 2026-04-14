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
## 人臉偵測撥AD
### main1.py
```bash
"""
廣告播放系統主程式
功能：USB 相機人臉偵測 → 視線判斷 → 性別/年齡分類 → 播放對應廣告 → SQLite 統計
"""

import cv2
import time
import sqlite3
import logging
import threading
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ── 可選：InsightFace（若未安裝則 fallback 到 OpenCV 內建模型）──────────────
try:
    from insightface.app import FaceAnalysis
    USE_INSIGHTFACE = True
except ImportError:
    USE_INSIGHTFACE = False
    logging.warning("InsightFace 未安裝，使用 OpenCV 內建模型（精度較低）")

# ── 可選：MediaPipe（視線判斷用）────────────────────────────────────────────
try:
    import mediapipe as mp
    USE_MEDIAPIPE = True
except ImportError:
    USE_MEDIAPIPE = False
    logging.warning("MediaPipe 未安裝，視線判斷將使用簡化版（正臉即視為看鏡頭）")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ad_system.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  設定
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # 相機
    camera_index: int = 0
    frame_width: int = 1280 #1280
    frame_height: int = 720 #720
    fps_limit: int = 5           # 節省 CPU/GPU，廣告場景 15fps 足夠
    display_width: int = 1920
    display_height: int = 1080
    sidebar_width: int = 460
    camera_panel_height: int = 260

    # 偵測
    face_confidence: float = 0.6  # 人臉偵測最低信心值
    gaze_yaw_threshold: float = 25.0   # 臉部水平偏轉角度上限（度），超過視為未看鏡頭
    gaze_pitch_threshold: float = 20.0 # 臉部垂直偏轉角度上限（度）

    # 防重複觸發（同一個人離開畫面後需等多久才重新計數）
    cooldown_seconds: float = 5.0
    # 最短「看鏡頭」時間才算一次有效觀看
    min_gaze_duration: float = 1.0

    # 年齡切割
    age_threshold: int = 20

    # 廣告檔案路徑（MP4 / 任何 mpv 支援的格式）
    ad_files: dict = field(default_factory=lambda: {
        "male_adult":   "ads/male_adult.mp4",
        "male_minor":   "ads/male_minor.mp4",
        "female_adult": "ads/female_adult.mp4",
        "female_minor": "ads/female_minor.mp4",
    })

    # 待機畫面（無人時播放）
    idle_video: str = "ads/idle.mp4"

    # 資料庫
    db_path: str = "ad_stats.db"

    # 顯示視窗（開發時用，部署時可設 False）
    show_preview: bool = True


CFG = Config()


# ═══════════════════════════════════════════════════════════════════
#  資料庫
# ═══════════════════════════════════════════════════════════════════

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS views (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,          -- ISO8601 時間戳記
            gender      TEXT    NOT NULL,          -- 'male' | 'female'
            age_group   TEXT    NOT NULL,          -- 'adult' | 'minor'
            age_est     REAL,                      -- 估計年齡
            ad_key      TEXT    NOT NULL,          -- 廣告 key
            gaze_secs   REAL                       -- 注視秒數
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            report_date TEXT PRIMARY KEY,
            total       INTEGER,
            male_adult  INTEGER,
            male_minor  INTEGER,
            female_adult INTEGER,
            female_minor INTEGER,
            sent_wa     INTEGER DEFAULT 0          -- 是否已傳 WhatsApp
        )
    """)
    conn.commit()
    log.info("資料庫初始化完成：%s", db_path)
    return conn


def record_view(conn: sqlite3.Connection, gender: str, age_group: str,
                age_est: float, ad_key: str, gaze_secs: float):
    ts = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO views (ts, gender, age_group, age_est, ad_key, gaze_secs) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (ts, gender, age_group, age_est, ad_key, gaze_secs)
    )
    conn.commit()
    log.info("記錄觀看：%s %s 年齡~%.0f  廣告=%s  注視=%.1fs",
             gender, age_group, age_est, ad_key, gaze_secs)


def get_today_summary(conn: sqlite3.Connection) -> dict:
    today = date.today().isoformat()
    row = conn.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(gender='male'   AND age_group='adult')  AS male_adult,
            SUM(gender='male'   AND age_group='minor')  AS male_minor,
            SUM(gender='female' AND age_group='adult')  AS female_adult,
            SUM(gender='female' AND age_group='minor')  AS female_minor
        FROM views
        WHERE ts LIKE ?
    """, (today + "%",)).fetchone()
    return {
        "date":         today,
        "total":        row[0] or 0,
        "male_adult":   row[1] or 0,
        "male_minor":   row[2] or 0,
        "female_adult": row[3] or 0,
        "female_minor": row[4] or 0,
    }


def get_recent_views(conn: sqlite3.Connection, limit: int = 6) -> list[dict]:
    rows = conn.execute("""
        SELECT ts, gender, age_group, age_est, ad_key, gaze_secs
        FROM views
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()
    return [
        {
            "ts": row[0],
            "gender": row[1],
            "age_group": row[2],
            "age_est": row[3],
            "ad_key": row[4],
            "gaze_secs": row[5],
        }
        for row in rows
    ]


# ═══════════════════════════════════════════════════════════════════
#  人臉分析（InsightFace 版）
# ═══════════════════════════════════════════════════════════════════

class FaceAnalyzer:
    """
    封裝 InsightFace（優先）或 OpenCV DNN（fallback）的人臉分析。
    回傳 list of FaceResult。
    """

    def __init__(self):
        if USE_INSIGHTFACE:
            self._app = FaceAnalysis(
                name="buffalo_l",          # 包含人臉偵測+年齡+性別
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            log.info("InsightFace buffalo_l 載入完成（CUDA if available）")
        else:
            # OpenCV DNN fallback
            self._face_net = cv2.dnn.readNetFromCaffe(
                "models/deploy.prototxt",
                "models/res10_300x300_ssd_iter_140000.caffemodel"
            )
            self._age_net = cv2.dnn.readNetFromCaffe(
                "models/age_deploy.prototxt",
                "models/age_net.caffemodel"
            )
            self._gender_net = cv2.dnn.readNetFromCaffe(
                "models/gender_deploy.prototxt",
                "models/gender_net.caffemodel"
            )
            log.info("OpenCV DNN fallback 模型載入完成")

    def analyze(self, frame: np.ndarray) -> list:
        """回傳 [{'bbox': (x1,y1,x2,y2), 'gender': str, 'age': float, 'yaw': float, 'pitch': float}]"""
        if USE_INSIGHTFACE:
            return self._analyze_insightface(frame)
        return self._analyze_opencv(frame)

    def _analyze_insightface(self, frame: np.ndarray) -> list:
        faces = self._app.get(frame)
        results = []
        for f in faces:
            if f.det_score < CFG.face_confidence:
                continue
            x1, y1, x2, y2 = f.bbox.astype(int)
            gender = "male" if f.gender == 1 else "female"
            age    = float(f.age)
            # pose: [yaw, pitch, roll] 單位度
            yaw   = float(f.pose[0]) if hasattr(f, "pose") else 0.0
            pitch = float(f.pose[1]) if hasattr(f, "pose") else 0.0
            results.append({
                "bbox":   (x1, y1, x2, y2),
                "gender": gender,
                "age":    age,
                "yaw":    yaw,
                "pitch":  pitch,
            })
        return results

    def _analyze_opencv(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self._face_net.setInput(blob)
        detections = self._face_net.forward()

        AGE_BUCKETS    = ["(0-2)", "(4-6)", "(8-12)", "(15-20)",
                          "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
        AGE_MIDPOINTS  = [1, 5, 10, 17, 28, 40, 50, 75]
        GENDER_LABELS  = ["female", "male"]

        results = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < CFG.face_confidence:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = frame[y1:y2, x1:x2]
            face_blob = cv2.dnn.blobFromImage(
                face_crop, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            # 性別
            self._gender_net.setInput(face_blob)
            gender_preds = self._gender_net.forward()
            gender = GENDER_LABELS[gender_preds[0].argmax()]
            # 年齡
            self._age_net.setInput(face_blob)
            age_preds = self._age_net.forward()
            age = float(AGE_MIDPOINTS[age_preds[0].argmax()])

            results.append({
                "bbox":   (x1, y1, x2, y2),
                "gender": gender,
                "age":    age,
                "yaw":    0.0,   # fallback 無法算 pose
                "pitch":  0.0,
            })
        return results


# ═══════════════════════════════════════════════════════════════════
#  MediaPipe 視線精細判斷（可選強化）
# ═══════════════════════════════════════════════════════════════════

class GazeEstimator:
    """
    使用 MediaPipe Face Mesh 精確計算眼球相對位置，
    判斷是否注視鏡頭。
    若 MediaPipe 未安裝，退回以 yaw/pitch 角度判斷。
    """

    def __init__(self):
        if USE_MEDIAPIPE:
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=4,
                refine_landmarks=True,   # 啟用虹膜 landmark
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            log.info("MediaPipe Face Mesh 載入完成（含虹膜追蹤）")

    def is_looking(self, frame: np.ndarray,
                   face_info: dict) -> tuple[bool, float]:
        """
        回傳 (is_looking, gaze_score)
        gaze_score: 0.0（完全沒看）~ 1.0（正視鏡頭）
        """
        if USE_MEDIAPIPE:
            return self._gaze_mediapipe(frame)
        # fallback：直接用 InsightFace 的 yaw/pitch
        yaw   = abs(face_info.get("yaw", 0.0))
        pitch = abs(face_info.get("pitch", 0.0))
        looking = (yaw < CFG.gaze_yaw_threshold and
                   pitch < CFG.gaze_pitch_threshold)
        score = max(0.0, 1.0 - yaw / 90.0 - pitch / 90.0)
        return looking, score

    def _gaze_mediapipe(self, frame: np.ndarray) -> tuple[bool, float]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return False, 0.0

        # 取第一張臉的虹膜中心偏移量
        # landmark 468-472: 左虹膜，473-477: 右虹膜
        lm = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        def pt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        # 左眼：外角=33, 內角=133, 虹膜中心=468
        # 右眼：外角=362, 內角=263, 虹膜中心=473
        for outer, inner, iris in [(33, 133, 468), (362, 263, 473)]:
            left_pt  = pt(outer)
            right_pt = pt(inner)
            iris_pt  = pt(iris)
            eye_center = (left_pt + right_pt) / 2
            eye_width  = np.linalg.norm(right_pt - left_pt)
            if eye_width < 1:
                continue
            offset = np.linalg.norm(iris_pt - eye_center) / eye_width
            # offset < 0.15 視為正視
            if offset > 0.20:
                return False, max(0.0, 1.0 - offset * 4)

        return True, 1.0


# ═══════════════════════════════════════════════════════════════════
#  廣告播放器
# ═══════════════════════════════════════════════════════════════════

class AdPlayer:
    """用 OpenCV 解碼影片，將廣告與待機畫面輸出到同一個全視窗。"""

    def __init__(self):
        self._dw = CFG.display_width - CFG.sidebar_width
        self._dh = CFG.display_height
        self._frame = np.zeros((self._dh, self._dw, 3), dtype=np.uint8)
        self._frame_lock = threading.Lock()
        self._ctrl_lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._next_ad: Optional[str] = None
        self._is_playing_ad = False
        threading.Thread(target=self._run, daemon=True).start()

    def play_ad(self, ad_key: str):
        ad_path = CFG.ad_files.get(ad_key)
        if not ad_path or not Path(ad_path).exists():
            log.warning("廣告檔案不存在：%s → %s", ad_key, ad_path)
            return
        with self._ctrl_lock:
            self._next_ad = ad_path
        log.info("▶ 播放廣告：%s (%s)", ad_key, ad_path)

    def is_playing_ad(self) -> bool:
        with self._ctrl_lock:
            return self._is_playing_ad

    def get_frame(self) -> np.ndarray:
        with self._frame_lock:
            return self._frame.copy()

    def stop(self):
        self._stop_evt.set()

    def _set_frame(self, frame: np.ndarray):
        resized = cv2.resize(frame, (self._dw, self._dh))
        with self._frame_lock:
            self._frame = resized

    def _run(self):
        idle_path = CFG.idle_video if Path(CFG.idle_video).exists() else None
        if idle_path is None:
            log.warning("找不到待機畫面：%s", CFG.idle_video)

        while not self._stop_evt.is_set():
            with self._ctrl_lock:
                next_ad = self._next_ad
                self._next_ad = None

            if next_ad:
                with self._ctrl_lock:
                    self._is_playing_ad = True
                self._play_file(next_ad, loop=False)
                with self._ctrl_lock:
                    self._is_playing_ad = False
                continue

            if idle_path:
                self._play_file(idle_path, loop=True)
            else:
                self._set_frame(np.zeros((self._dh, self._dw, 3), dtype=np.uint8))
                time.sleep(0.05)

    def _play_file(self, path: str, loop: bool):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            log.warning("無法開啟影片：%s", path)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        spf = 1.0 / max(fps, 1.0)

        while not self._stop_evt.is_set():
            with self._ctrl_lock:
                if self._next_ad is not None and loop:
                    break
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            self._set_frame(frame)
            wait = spf - (time.time() - start)
            if wait > 0:
                time.sleep(wait)
        cap.release()


# ═══════════════════════════════════════════════════════════════════
#  防重複觸發管理
# ═══════════════════════════════════════════════════════════════════

class PersonTracker:
    """
    以臉部 bounding box 中心做簡單的同人追蹤，
    確保同一個人不在 cooldown 期間重複觸發廣告。
    """

    def __init__(self):
        # key: (cx, cy) 近似座標bucket, value: 最後觸發時間
        self._last_trigger: dict[tuple, float] = {}
        self._gaze_start: dict[tuple, float] = {}

    def bucket(self, bbox: tuple) -> tuple:
        """把臉部位置量化成格子，避免微小抖動產生不同 key"""
        x1, y1, x2, y2 = bbox
        cx = ((x1 + x2) // 2) // 80  # 每 80px 一格
        cy = ((y1 + y2) // 2) // 80
        return (cx, cy)

    def start_gaze(self, bbox: tuple) -> None:
        key = self.bucket(bbox)
        if key not in self._gaze_start:
            self._gaze_start[key] = time.time()

    def check_and_trigger(self, bbox: tuple) -> bool:
        """
        回傳 True 代表：這個人已注視足夠久，且過了 cooldown，可觸發廣告。
        """
        key = self.bucket(bbox)
        now = time.time()

        gaze_start = self._gaze_start.get(key)
        if gaze_start is None:
            return False
        gaze_duration = now - gaze_start
        if gaze_duration < CFG.min_gaze_duration:
            return False

        last = self._last_trigger.get(key, 0)
        if now - last < CFG.cooldown_seconds:
            return False

        self._last_trigger[key] = now
        self._gaze_start.pop(key, None)
        return True

    def reset_gaze(self, bbox: tuple) -> None:
        self._gaze_start.pop(self.bucket(bbox), None)

    def cleanup(self, active_buckets: set):
        """清除已離開的臉部 key"""
        for key in list(self._gaze_start.keys()):
            if key not in active_buckets:
                del self._gaze_start[key]


# ═══════════════════════════════════════════════════════════════════
#  主迴圈
# ═══════════════════════════════════════════════════════════════════

def classify(gender: str, age: float) -> tuple[str, str]:
    """性別 + 年齡 → (age_group, ad_key)"""
    age_group = "adult" if age >= CFG.age_threshold else "minor"
    ad_key    = f"{gender}_{age_group}"
    return age_group, ad_key


def draw_debug(frame: np.ndarray, faces: list, active: list):
    """在預覽視窗畫出偵測結果"""
    for fi, face in enumerate(faces):
        x1, y1, x2, y2 = face["bbox"]
        looking = fi in active
        color = (0, 255, 80) if looking else (80, 80, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = (f"{face['gender'][0].upper()} "
                 f"~{face['age']:.0f}y "
                 f"{'👁' if looking else ''}")
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def fit_to_panel(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    scale = min(width / max(src_w, 1), height / max(src_h, 1))
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = cv2.resize(frame, (new_w, new_h))
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    panel[y:y + new_h, x:x + new_w] = resized
    return panel


def draw_sidebar(sidebar: np.ndarray, conn: sqlite3.Connection):
    summary = get_today_summary(conn)
    recent = get_recent_views(conn)
    x = 28
    y = 46

    def text(line: str, size: float = 0.75, color=(245, 245, 245), gap: int = 34,
             thickness: int = 1):
        nonlocal y
        cv2.putText(sidebar, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    size, color, thickness, cv2.LINE_AA)
        y += gap

    text("Realtime Dashboard", 0.9, (255, 255, 255), 42, 2)
    text(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0.55, (180, 210, 255), 30)
    y += 12
    text(f"Today Total: {summary['total']}", 0.8, (120, 220, 255), 40, 2)
    text(f"Male Adult: {summary['male_adult']}", 0.65)
    text(f"Male Minor: {summary['male_minor']}", 0.65)
    text(f"Female Adult: {summary['female_adult']}", 0.65)
    text(f"Female Minor: {summary['female_minor']}", 0.65)
    y += 18
    text("Recent Records", 0.75, (255, 220, 120), 36, 2)

    for item in recent:
        ts = item["ts"][11:19] if "T" in item["ts"] else item["ts"][-8:]
        age = f"{item['age_est']:.0f}" if item["age_est"] is not None else "-"
        gaze = f"{item['gaze_secs']:.1f}s" if item["gaze_secs"] is not None else "-"
        line = f"{ts} {item['gender'][0].upper()} {item['age_group']} {age}y {gaze}"
        text(line[:46], 0.5, (210, 210, 210), 26)
        text(f"Ad: {item['ad_key']}", 0.48, (150, 190, 255), 24)

    footer_y = sidebar.shape[0] - 24
    cv2.putText(sidebar, "Q: quit   F: toggle fullscreen", (x, footer_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)


def render_dashboard(ad_frame: np.ndarray, cam_frame: np.ndarray, conn: sqlite3.Connection,
                     faces: list, looking_indices: list, player: AdPlayer) -> np.ndarray:
    screen = np.zeros((CFG.display_height, CFG.display_width, 3), dtype=np.uint8)

    main_w = CFG.display_width - CFG.sidebar_width
    main_h = CFG.display_height
    screen[:, :main_w] = fit_to_panel(ad_frame, main_w, main_h)

    overlay_w = min(main_w - 40, max(280, main_w // 3))
    overlay_h = min(CFG.camera_panel_height, CFG.display_height // 3)
    debug_frame = draw_debug(cam_frame.copy(), faces, looking_indices)
    camera_panel = fit_to_panel(debug_frame, overlay_w, overlay_h)

    margin = 20
    ox = main_w - overlay_w - margin
    oy = margin
    cv2.rectangle(screen, (ox - 3, oy - 3), (ox + overlay_w + 3, oy + overlay_h + 3),
                  (255, 255, 255), 1)
    screen[oy:oy + overlay_h, ox:ox + overlay_w] = camera_panel
    cv2.putText(screen, "Camera", (ox, oy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    status = "PLAYING AD" if player.is_playing_ad() else "IDLE"
    cv2.putText(screen, status, (28, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (120, 255, 180) if player.is_playing_ad() else (180, 180, 180),
                2, cv2.LINE_AA)

    sidebar = np.full((CFG.display_height, CFG.sidebar_width, 3), 24, dtype=np.uint8)
    draw_sidebar(sidebar, conn)
    screen[:, main_w:] = sidebar
    cv2.line(screen, (main_w, 0), (main_w, CFG.display_height), (70, 70, 70), 2)
    return screen


def main():
    log.info("═══ 廣告播放系統啟動 ═══")

    # 初始化各模組
    conn    = init_db(CFG.db_path)
    analyzer = FaceAnalyzer()
    gaze     = GazeEstimator()
    player   = AdPlayer()
    tracker  = PersonTracker()

    # 開啟相機
    cap = cv2.VideoCapture(CFG.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.frame_height)
    cap.set(cv2.CAP_PROP_FPS,          CFG.fps_limit)

    if not cap.isOpened():
        log.error("無法開啟相機（index=%d）", CFG.camera_index)
        return

    log.info("相機已啟動 %dx%d @%dfps",
             CFG.frame_width, CFG.frame_height, CFG.fps_limit)

    frame_interval = 1.0 / CFG.fps_limit
    last_frame_time = 0.0

    win_name = "Ad System Dashboard"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            # FPS 節流
            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()

            ret, frame = cap.read()
            if not ret:
                log.warning("讀取畫面失敗，重試中...")
                time.sleep(0.5)
                continue

            # 廣告播放中 → 跳過偵測（避免打斷正在播的廣告）
            if player.is_playing_ad():
                dashboard = render_dashboard(player.get_frame(), frame, conn, [], [], player)
                cv2.imshow(win_name, dashboard)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("f"):
                    fs = cv2.getWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty(
                        win_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL if fs else cv2.WINDOW_FULLSCREEN,
                    )
                continue

            # ── 人臉分析 ────────────────────────────────────────
            faces = analyzer.analyze(frame)

            active_buckets = set()
            looking_indices = []

            for i, face in enumerate(faces):
                looking, score = gaze.is_looking(frame, face)

                if looking:
                    tracker.start_gaze(face["bbox"])
                    active_buckets.add(tracker.bucket(face["bbox"]))
                    looking_indices.append(i)

                    # 判斷是否達到觸發條件
                    if tracker.check_and_trigger(face["bbox"]):
                        age_group, ad_key = classify(face["gender"], face["age"])
                        key = tracker.bucket(face["bbox"])
                        gaze_start = tracker._gaze_start.get(key, time.time())
                        gaze_secs = time.time() - gaze_start

                        # 記錄資料庫
                        record_view(conn, face["gender"], age_group,
                                    face["age"], ad_key, gaze_secs)

                        # 播放廣告（只取第一個觸發的人）
                        player.play_ad(ad_key)
                        break  # 同一畫面只播一支廣告
                else:
                    tracker.reset_gaze(face["bbox"])

            tracker.cleanup(active_buckets)

            # ── 預覽視窗 ─────────────────────────────────────────
            dashboard = render_dashboard(
                player.get_frame(),
                frame,
                conn,
                faces,
                looking_indices,
                player,
            )
            cv2.imshow(win_name, dashboard)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                log.info("使用者按 Q 停止")
                break
            if key == ord("f"):
                fs = cv2.getWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(
                    win_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL if fs else cv2.WINDOW_FULLSCREEN,
                )

    except KeyboardInterrupt:
        log.info("收到 Ctrl+C，正在關閉...")
    finally:
        player.stop()
        cap.release()
        cv2.destroyAllWindows()
        conn.close()
        log.info("系統已關閉")


if __name__ == "__main__":
    main()
```
    
