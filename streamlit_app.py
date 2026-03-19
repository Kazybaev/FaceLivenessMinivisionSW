"""
МГНОВЕННЫЙ КОНТРОЛЬ ДОСТУПА — каждый кадр = отдельный вердикт.

Логика:
  ✅ Каждый кадр с движением + резкостью → сразу на API → сразу ответ
  ✅ Никаких окон, голосований, сессий, задержек
  ✅ Человек не стоит — просто прошёл перед камерой → мгновенный 🟢/🔴
  ✅ Зелёный = реальный человек | Красный = телефон/фото/маска
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque

import av
import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, RTCConfiguration, webrtc_streamer

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# ─── Настройки ────────────────────────────────────────────────────────────────
DEFAULT_API   = "http://localhost:8000"
DOOR_OPEN_URL = "http://localhost:8001/door/open"

SHARPNESS_MIN = 80.0      # мин. резкость (Laplacian variance)
MOTION_MIN    = 0.03      # мин. движение (доля пикселей)
JPEG_QUALITY  = 92        # качество JPEG для отправки
MAX_QUEUE     = 3         # буфер кадров, чтобы не терять при пике

# Если API возвращает уверенность в % (0-100), поставьте True:
CONF_IS_PERCENT = False

st.set_page_config(
    page_title="⚡ Мгновенный доступ",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS: минималистичный интерфейс с крупными индикаторами ───────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Крупные индикаторы-светофоры */
.big-light {
    width: 120px; height: 120px; border-radius: 50%;
    margin: 0 auto 1rem; border: 6px solid rgba(0,0,0,0.1);
    transition: all 0.15s ease;
}
.big-light.off {
    background: #e2e8f0; box-shadow: inset 0 4px 8px rgba(0,0,0,0.1);
}
.big-light.green.on {
    background: #22c55e;
    box-shadow: 0 0 40px #22c55e, 0 0 80px rgba(34,197,94,0.5);
    border-color: #16a34a; transform: scale(1.05);
}
.big-light.red.on {
    background: #ef4444;
    box-shadow: 0 0 40px #ef4444, 0 0 80px rgba(239,68,68,0.5);
    border-color: #dc2626; transform: scale(1.05);
}

/* Текст статуса */
.status-text {
    text-align: center; font-size: 1.8rem; font-weight: 800;
    margin: 0.5rem 0; padding: 0.5rem 1rem; border-radius: 12px;
}
.status-live  { background: #dcfce7; color: #166534; }
.status-fake  { background: #fff1f2; color: #9f1239; }
.status-wait  { background: #f1f5f9; color: #64748b; }

/* Прогресс уверенности */
.conf-wrap { background:#e2e8f0; border-radius:99px; height:16px; margin:0.8rem 0; overflow:hidden; }
.conf-fill { height:16px; border-radius:99px; transition:width 0.2s ease; font-size:0.75rem; font-weight:700; color:white; text-align:center; line-height:16px; }

/* Бейджи */
.badge { display:inline-block; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600; margin:2px; }
.badge-ok  { background:#dcfce7; color:#166534; }
.badge-no  { background:#fff1f2; color:#9f1239; }
.badge-wait{ background:#f1f5f9; color:#64748b; }

/* Скрыть лишнее в Streamlit */
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "last_verdict": None,      # "live" | "fake" | None
    "last_conf": 0.0,
    "door_triggered": False,   # чтобы дверь не открывалась 100 раз подряд
    "stats_sent": 0,
    "stats_drop": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Утилиты ──────────────────────────────────────────────────────────────────
def normalize_conf(conf) -> float:
    try:
        f = float(conf)
    except (TypeError, ValueError):
        return 0.0
    return f / 100.0 if CONF_IS_PERCENT else max(0.0, min(1.0, f))


def is_live(verdict: str) -> bool:
    v = verdict.lower()
    return any(w in v for w in ("live", "real", "genuine", "bona"))


def is_fake(verdict: str) -> bool:
    v = verdict.lower()
    return any(w in v for w in ("spoof", "fake", "attack", "print", "replay", "mask"))


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ─── API ──────────────────────────────────────────────────────────────────────
def check_health(api_url: str) -> bool:
    try:
        return requests.get(f"{api_url}/health", timeout=2).status_code == 200
    except Exception:
        return False


def send_frame_fast(api_url: str, jpg_bytes: bytes) -> dict | None:
    """Отправка одного кадра без сессии — максимально быстро."""
    try:
        r = requests.post(
            f"{api_url}/api/v1/liveness/analyze",
            files={"file": ("f.jpg", jpg_bytes, "image/jpeg")},
            timeout=3  # ⚡ короткий таймаут для мгновенного ответа
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning("API error: %s", e)
        return None


def open_door(url: str) -> bool:
    try:
        return requests.post(url, timeout=2).status_code == 200
    except Exception:
        return False


# ─── Video Processor: мгновенный анализ каждого кадра ─────────────────────────
class InstantLivenessProcessor(VideoProcessorBase):
    """
    ⚡ КАЖДЫЙ КАДР = НЕЗАВИСИМЫЙ ВЕРДИКТ
    
    - Нет сессий, нет окон, нет голосований
    - Движение + резкость → кадр в очередь → воркер шлёт на API → сразу результат
    - Результат рисуется ПРЯМО НА КАДРЕ (цветной круг + текст)
    """

    def __init__(self):
        self.api_url       = DEFAULT_API
        self.motion_min    = MOTION_MIN
        self.sharp_min     = SHARPNESS_MIN
        
        self._prev_gray = None
        self._lock      = threading.Lock()
        
        # 🔥 Самый свежий результат (обновляется из воркера)
        self._latest_verdict = None  # "live" | "fake" | None
        self._latest_conf    = 0.0
        self._last_update_ts = 0
        
        # Статистика
        self.frames_sent  = 0
        self.frames_drop  = 0
        
        # 🔥 Очередь + воркер (без блокировок видео)
        self._q = queue.Queue(maxsize=MAX_QUEUE)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def __del__(self):
        self._stop.set()
        try: self._q.put_nowait(None)
        except: pass

    def _run_worker(self):
        """Фоновый воркер: берёт кадр из очереди → шлёт на API → обновляет результат."""
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
                
            jpg_bytes, frame_ts = item
            result = send_frame_fast(self.api_url, jpg_bytes)
            
            if result:
                verdict_raw = result.get("verdict", "")
                conf_raw    = result.get("confidence", 0)
                conf_norm   = normalize_conf(conf_raw)
                
                if is_live(verdict_raw):
                    v = "live"
                elif is_fake(verdict_raw):
                    v = "fake"
                else:
                    v = None  # неопределённо
                    
                # 🔥 Обновляем самый свежий результат (атомарно)
                with self._lock:
                    self._latest_verdict = v
                    self._latest_conf    = conf_norm
                    self._last_update_ts = time.time()
                    
                with self._lock:
                    self.frames_sent += 1

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 🔹 Резкость
        sharp = sharpness(gray)
        
        # 🔹 Движение (сравнение с предыдущим кадром)
        motion = 0.0
        if self._prev_gray is not None:
            diff = cv2.absdiff(self._prev_gray, gray)
            _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion = float(np.sum(mask > 0)) / mask.size
        self._prev_gray = gray.copy()
        
        # 🔹 Фильтр: только чёткие кадры с движением
        is_good = (motion >= self.motion_min) and (sharp >= self.sharp_min)
        
        if is_good:
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            try:
                # 🔥 Кидаем в очередь с меткой времени
                self._q.put_nowait((buf.tobytes(), time.time()))
            except queue.Full:
                with self._lock:
                    self.frames_drop += 1
        
        # ── 🔥 РИСУЕМ РЕЗУЛЬТАТ ПРЯМО НА КАДРЕ (мгновенно) ─────────────────
        # Берём самый свежий результат из воркера
        with self._lock:
            verdict = self._latest_verdict
            conf    = self._latest_conf
            upd_ts  = self._last_update_ts
        
        # 🔘 Цветной индикатор в правом верхнем углу
        cx, cy = w - 60, 60
        radius = 35
        
        if verdict == "live" and conf >= 0.5 and (time.time() - upd_ts) < 2.0:
            # 🟢 ЗЕЛЁНЫЙ — реальный человек (актуален последние 2 сек)
            cv2.circle(img, (cx, cy), radius+10, (0,0,0), -1)
            cv2.circle(img, (cx, cy), radius, (0, 255, 0), -1)
            cv2.circle(img, (cx, cy), radius, (255,255,255), 3)
            cv2.putText(img, "REAL", (cx-35, cy+5), cv2.FONT_HERSHEY_BOLD, 0.7, (255,255,255), 2)
            status_color = (0, 255, 0)
            status_text  = "✅ РЕАЛЬНЫЙ"
            
        elif verdict == "fake" and conf >= 0.5 and (time.time() - upd_ts) < 2.0:
            # 🔴 КРАСНЫЙ — подделка (актуален последние 2 сек)
            cv2.circle(img, (cx, cy), radius+10, (0,0,0), -1)
            cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
            cv2.circle(img, (cx, cy), radius, (255,255,255), 3)
            cv2.putText(img, "FAKE", (cx-35, cy+5), cv2.FONT_HERSHEY_BOLD, 0.7, (255,255,255), 2)
            status_color = (0, 0, 255)
            status_text  = "❌ ПОДДЕЛКА"
            
        else:
            # ⚪ СЕРЫЙ — ожидание / нет данных / устарело
            cv2.circle(img, (cx, cy), radius+8, (200,200,200), 3)
            cv2.circle(img, (cx, cy), radius, (128,128,128), -1)
            cv2.putText(img, "WAIT", (cx-32, cy+5), cv2.FONT_HERSHEY_BOLD, 0.6, (255,255,255), 2)
            status_color = (128, 128, 128)
            status_text  = "⏳ ОЖИДАНИЕ"
        
        # 🔹 Полоса статуса внизу кадра
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h-36), (w, h), status_color, -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        cv2.putText(img, f"{status_text}  |  {conf*100:.0f}%", 
                    (15, h-12), cv2.FONT_HERSHEY_BOLD, 0.75, (255,255,255), 2)
        
        # 🔹 Мини-статистика в углу
        cv2.putText(img, f"Sent:{self.frames_sent} Drop:{self.frames_drop}",
                    (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        # 🔹 Сохраняем текущий статус для UI
        with self._lock:
            st.session_state.last_verdict = verdict
            st.session_state.last_conf    = conf
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─── UI: минимализм, только суть ──────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;margin-bottom:0.5rem'>🚦 Мгновенный контроль</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#64748b;margin-top:0'>Прошёл → 🟢/🔴 → готово</p>", unsafe_allow_html=True)
st.divider()

# Боковая панель (скрыта по умолчанию, но доступна)
with st.sidebar:
    st.markdown("### ⚙️")
    api_url = st.text_input("API URL", value=DEFAULT_API)
    door_url = st.text_input("Door URL", value=DOOR_OPEN_URL)
    
    alive = check_health(api_url)
    st.metric("Сервер", "🟢 ОК" if alive else "🔴 Нет")
    
    st.markdown("---")
    st.caption("Пороги (перезапустите камеру после изменения):")
    motion_min = st.slider("Движение %", 1, 20, 3) / 100
    sharp_min  = st.slider("Резкость", 20, 300, 80)
    
    if st.button("🔄 Сброс"):
        st.session_state.last_verdict = None
        st.session_state.door_triggered = False
        st.rerun()

# ─── Основной layout ─────────────────────────────────────────────────────────
col_cam, col_status = st.columns([2.5, 1], gap="large")

with col_cam:
    RTC_CFG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    ctx = webrtc_streamer(
        key="instant",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CFG,
        video_processor_factory=InstantLivenessProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Обновляем параметры процессора
    if ctx.video_processor:
        proc = ctx.video_processor
        proc.api_url    = api_url
        proc.motion_min = motion_min
        proc.sharp_min  = float(sharp_min)

with col_status:
    st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
    
    # 🔴🟢 БОЛЬШОЙ ИНДИКАТОР
    verdict = st.session_state.last_verdict
    conf    = st.session_state.last_conf
    
    if verdict == "live" and conf >= 0.5:
        st.markdown(f'<div class="big-light green on"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-text status-live">✅ РЕАЛЬНЫЙ</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="conf-wrap">
            <div class="conf-fill" style="width:{conf*100}%;background:#22c55e">{conf*100:.0f}%</div>
        </div>""", unsafe_allow_html=True)
        
        # 🔓 Авто-открытие двери (только один раз при смене статуса)
        if not st.session_state.get("door_triggered"):
            if open_door(door_url):
                st.success("🚪 Дверь открыта!")
            st.session_state.door_triggered = True
            
    elif verdict == "fake" and conf >= 0.5:
        st.markdown(f'<div class="big-light red on"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-text status-fake">❌ ПОДДЕЛКА</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="conf-wrap">
            <div class="conf-fill" style="width:{conf*100}%;background:#ef4444">{conf*100:.0f}%</div>
        </div>""", unsafe_allow_html=True)
        st.session_state.door_triggered = False  # сброс, чтобы следующий LIVE сработал
        
    else:
        st.markdown(f'<div class="big-light off"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-text status-wait">⏳ Ждём кадр...</div>', unsafe_allow_html=True)
        st.markdown('<div class="conf-wrap"><div class="conf-fill" style="width:0%"></div></div>', unsafe_allow_html=True)
        st.session_state.door_triggered = False
    
    st.markdown("</div>", unsafe_allow_html=True)  # close center div
    
    # Мини-подсказки
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.9rem;color:#64748b'>
    <span class="badge badge-ok">🟢 Реальный</span>
    <span class="badge badge-no">🔴 Телефон/фото</span>
    <br><br>
    💡 Просто пройдите перед камерой — не нужно стоять
    </div>
    """, unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("⚡ Каждый кадр анализируется отдельно | Результат обновляется мгновенно | Без задержек и накоплений")