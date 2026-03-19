"""
Контроль доступа по лицу — максимальная скорость, без cooldown.

Алгоритм:
  - Каждый кадр с движением И достаточной резкостью немедленно отправляется на API
  - Никакого cooldown — реакция на первый же подходящий кадр
  - Параллельная очередь: пока идёт отправка одного кадра, новые не блокируют видео
  - Скользящее окно последних N результатов → majority vote для стабильности
  - Автооткрытие двери при LIVE

Исправления (v2):
  [BUG-1]  Session state frames_sent/skipped/blurry_dropped теперь реально синхронизируются
  [BUG-2]  Race condition при замене _window снаружи устранён через _lock
  [BUG-3]  Воркер корректно завершается через _stop_event при пересоздании контекста
  [BUG-4]  door_opened сбрасывается при смене use_session
  [BUG-5]  Нормализация confidence вынесена в отдельную функцию с явным контрактом
  [BUG-6]  open_door вызывается ровно один раз — через st.session_state-флаг + rerun
  [BUG-7]  Добавлен st.rerun() для real-time обновления UI
  [BUG-8]  Новый счётчик frames_dropped_queue для дропов из переполненной очереди
  [BUG-9]  Воркер логирует ошибки; send_frame пробрасывает исключения с logging
  [BUG-10] Комментарий о задержке применения слайдеров добавлен
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import Counter, deque

import av
import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, RTCConfiguration, webrtc_streamer

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# ─── Настройки ────────────────────────────────────────────────────────────────
DEFAULT_API      = "http://localhost:8000"
DOOR_OPEN_URL    = "http://localhost:8001/door/open"

WINDOW_SIZE      = 5      # скользящее окно для majority vote (последних N ответов)
SHARPNESS_MIN    = 80.0   # минимальная резкость (Laplacian variance)
JPEG_QUALITY     = 92     # качество JPEG
MAX_QUEUE        = 2      # макс. кадров в очереди отправки (дропаем лишние)

# Контракт API: confidence всегда float в диапазоне [0.0, 1.0]
# Если ваш API возвращает проценты (0–100), выставьте True:
CONF_IS_PERCENT  = False

st.set_page_config(
    page_title="Контроль доступа",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.card-granted {
    background: #f0fdf4; border: 2px solid #22c55e;
    border-radius: 18px; padding: 2rem; text-align: center; margin: 0.8rem 0;
}
.card-granted .icon { font-size: 3.5rem; }
.card-granted h2    { color: #16a34a; font-size: 2rem; font-weight: 700; margin: 0.3rem 0 0.2rem; }
.card-granted p     { color: #166534; margin: 0; font-size: 0.95rem; }

.card-denied {
    background: #fff1f2; border: 2px solid #f43f5e;
    border-radius: 18px; padding: 2rem; text-align: center; margin: 0.8rem 0;
}
.card-denied .icon { font-size: 3.5rem; }
.card-denied h2    { color: #e11d48; font-size: 2rem; font-weight: 700; margin: 0.3rem 0 0.2rem; }
.card-denied p     { color: #9f1239; margin: 0; font-size: 0.95rem; }

.card-checking {
    background: #fffbeb; border: 2px solid #f59e0b;
    border-radius: 18px; padding: 2rem; text-align: center; margin: 0.8rem 0;
}
.card-checking .icon { font-size: 3.5rem; }
.card-checking h2   { color: #b45309; font-size: 1.6rem; font-weight: 700; margin: 0.3rem 0 0.2rem; }
.card-checking p    { color: #92400e; margin: 0; font-size: 0.9rem; }

.card-waiting {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 18px; padding: 2rem; text-align: center;
    color: #94a3b8; margin: 0.8rem 0;
}
.card-waiting .icon { font-size: 2.5rem; }
.card-waiting p     { font-size: 1rem; margin: 0.5rem 0 0; }

/* Прогресс-бар уверенности */
.conf-bar-wrap { background:#e2e8f0; border-radius:99px; height:10px; margin:0.6rem 0 0; overflow:hidden; }
.conf-bar      { height:10px; border-radius:99px; transition: width 0.4s ease; }

/* Статистика */
.stats-row { display:flex; gap:10px; margin-top:1rem; }
.stat {
    flex:1; background:#f8fafc; border:1px solid #e2e8f0;
    border-radius:12px; padding:0.8rem; text-align:center;
}
.stat .val { font-size:1.5rem; font-weight:700; color:#1e293b; }
.stat .lbl { font-size:0.72rem; color:#64748b; margin-top:2px; }

/* Бейдж движения */
.badge-on  { background:#dcfce7; color:#166534; border-radius:20px; padding:4px 14px; font-size:0.82rem; font-weight:600; }
.badge-off { background:#f1f5f9; color:#94a3b8; border-radius:20px; padding:4px 14px; font-size:0.82rem; }
.badge-blur{ background:#fef9c3; color:#a16207; border-radius:20px; padding:4px 14px; font-size:0.82rem; }

/* Голосование */
.votes { display:flex; gap:8px; margin-top:0.6rem; justify-content:center; }
.vote-live  { background:#dcfce7; color:#166534; border-radius:8px; padding:4px 12px; font-size:0.8rem; font-weight:600; }
.vote-fake  { background:#fff1f2; color:#e11d48; border-radius:8px; padding:4px 12px; font-size:0.8rem; font-weight:600; }
.vote-na    { background:#f1f5f9; color:#94a3b8; border-radius:8px; padding:4px 12px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "session_id":        None,
    "final_result":      None,
    "door_opened":       False,
    "prev_use_session":  True,   # [BUG-4] отслеживаем смену use_session
    # Статистика — синхронизируется из proc в каждом рендере [BUG-1]
    "frames_sent":       0,
    "frames_skipped":    0,
    "blurry_dropped":    0,
    "frames_dropped_q":  0,      # [BUG-8] новый счётчик
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Утилиты ──────────────────────────────────────────────────────────────────
def normalize_conf(conf) -> float:
    """
    [BUG-5] Нормализация confidence в диапазон [0.0, 1.0].
    Контракт задаётся константой CONF_IS_PERCENT вверху файла,
    а не угадывается по значению — это устраняет двусмысленность при conf == 1.0.
    """
    try:
        f = float(conf)
    except (TypeError, ValueError):
        return 0.0
    return f / 100.0 if CONF_IS_PERCENT else max(0.0, min(1.0, f))


def is_live(verdict: str) -> bool:
    return any(w in verdict.lower() for w in ("live", "real"))


def is_fake(verdict: str) -> bool:
    return any(w in verdict.lower() for w in ("spoof", "fake", "attack"))


def sharpness(gray: np.ndarray) -> float:
    """Laplacian variance — мера резкости кадра. Чем выше, тем чётче."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def window_vote(results: deque) -> dict | None:
    """
    Majority vote по скользящему окну последних N ответов.
    Возвращает самый частый вердикт + среднюю уверенность окна.
    """
    if not results:
        return None
    verdicts = [r.get("verdict", "") for r in results]
    confs    = [normalize_conf(r.get("confidence", 0)) for r in results]
    winner   = Counter(verdicts).most_common(1)[0][0]
    avg_conf = sum(confs) / len(confs)
    return {"verdict": winner, "confidence": avg_conf, "votes": verdicts}


# ─── API-утилиты ──────────────────────────────────────────────────────────────
def check_health(api_url: str) -> tuple[bool, bool]:
    try:
        alive = requests.get(f"{api_url}/health", timeout=2).status_code == 200
    except Exception:
        return False, False
    try:
        models_loaded = requests.get(
            f"{api_url}/health/ready", timeout=2
        ).json().get("models_loaded", False)
    except Exception:
        models_loaded = False
    return alive, models_loaded


def create_session(api_url: str) -> str | None:
    try:
        r = requests.post(f"{api_url}/api/v1/liveness/session", timeout=5)
        if r.status_code == 200:
            return r.json()["session_id"]
    except Exception:
        pass
    return None


def delete_session_api(api_url: str, session_id: str):
    try:
        requests.delete(f"{api_url}/api/v1/liveness/session/{session_id}", timeout=3)
    except Exception:
        pass


def send_frame(api_url: str, session_id: str | None, jpg_bytes: bytes) -> dict:
    """
    [BUG-9] Теперь выбрасывает исключение при сетевой или HTTP-ошибке
    вместо тихого возврата None — воркер может залогировать реальную причину.
    """
    url = (
        f"{api_url}/api/v1/liveness/session/{session_id}/frame"
        if session_id
        else f"{api_url}/api/v1/liveness/analyze"
    )
    r = requests.post(
        url, files={"file": ("frame.jpg", jpg_bytes, "image/jpeg")}, timeout=8
    )
    r.raise_for_status()
    return r.json()


def open_door(door_url: str) -> bool:
    try:
        return requests.post(door_url, timeout=5).status_code == 200
    except Exception:
        return False


# ─── Video Processor ───────────────────────────────────────────────────────────
class LivenessProcessor(VideoProcessorBase):
    """
    Без cooldown — каждый чёткий кадр с движением немедленно уходит на API.

    Архитектура:
      - recv() кладёт кадр в queue.Queue(maxsize=MAX_QUEUE) и сразу возвращает управление
      - Фоновый воркер (_worker) берёт из очереди и отправляет по одному
      - Новые кадры при заполненной очереди дропаются (не блокируют видео)
      - Скользящее окно (_window) из последних WINDOW_SIZE ответов → majority vote

    Примечание [BUG-10]:
      motion_threshold и sharpness_min читаются из recv() напрямую.
      Изменения слайдеров применяются с задержкой в один ре-рендер Streamlit
      (главный поток обновляет атрибуты после рендера UI). Для float-значений
      это безопасно — промежуточное чтение даст лишь устаревшее значение.
    """

    def __init__(self):
        self.api_url          : str        = DEFAULT_API
        self.session_id       : str | None = None
        self.motion_threshold : float      = 0.03
        self.sharpness_min    : float      = SHARPNESS_MIN

        self._prev_gray  : np.ndarray | None = None
        self._lock       = threading.Lock()
        self._window     : deque             = deque(maxlen=WINDOW_SIZE)
        self._voted      : dict | None       = None
        self._sending    : bool              = False

        self._motion_score    : float = 0.0
        self._sharpness_score : float = 0.0
        self.frames_sent      : int   = 0
        self.frames_skipped   : int   = 0
        self.blurry_dropped   : int   = 0
        self.frames_dropped_q : int   = 0   # [BUG-8]

        # [BUG-3] Флаг остановки воркера
        self._stop_event = threading.Event()

        self._q = queue.Queue(maxsize=MAX_QUEUE)
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def __del__(self):
        """[BUG-3] Корректная остановка воркера при уничтожении объекта."""
        self._stop_event.set()
        # Разбудить воркер, если он ждёт на _q.get()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass

    # ── Фоновый воркер ──
    def _worker(self):
        """[BUG-3, BUG-9] Воркер с флагом остановки и логированием ошибок."""
        while not self._stop_event.is_set():
            try:
                jpg_bytes = self._q.get(timeout=1.0)
            except queue.Empty:
                continue

            # Сигнал остановки — None в очереди
            if jpg_bytes is None:
                break

            with self._lock:
                self._sending = True

            try:
                result = send_frame(self.api_url, self.session_id, jpg_bytes)
                with self._lock:
                    self._window.append(result)
                    self._voted = window_vote(self._window)
                    self.frames_sent += 1
            except requests.HTTPError as e:
                log.warning("send_frame HTTP error: %s", e)
            except requests.RequestException as e:
                log.warning("send_frame network error: %s", e)
            except Exception as e:
                log.error("send_frame unexpected error: %s", e)
            finally:
                with self._lock:
                    self._sending = False

    # ── Публичные свойства ──
    @property
    def last_result(self) -> dict | None:
        with self._lock:
            return self._voted

    @property
    def window_votes(self) -> list[str]:
        with self._lock:
            return [r.get("verdict", "?") for r in self._window]

    @property
    def motion_score(self) -> float:
        with self._lock:
            return self._motion_score

    @property
    def sharpness_score(self) -> float:
        with self._lock:
            return self._sharpness_score

    @property
    def is_sending(self) -> bool:
        with self._lock:
            return self._sending or not self._q.empty()

    def set_window_size(self, size: int):
        """[BUG-2] Потокобезопасное изменение размера окна."""
        with self._lock:
            new_window = deque(list(self._window), maxlen=size)
            self._window = new_window
            self._voted  = window_vote(self._window)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        sharp     = sharpness(gray)

        motion = 0.0
        if self._prev_gray is not None:
            diff = cv2.absdiff(self._prev_gray, gray_blur)
            _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion = float(np.sum(mask > 0)) / mask.size
        self._prev_gray = gray_blur

        with self._lock:
            self._motion_score    = motion
            self._sharpness_score = sharp

        is_sharp   = sharp  >= self.sharpness_min
        has_motion = motion >= self.motion_threshold

        if has_motion and is_sharp:
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            try:
                self._q.put_nowait(buf.tobytes())
            except queue.Full:
                # [BUG-8] Считаем дропы из-за переполненной очереди отдельно
                with self._lock:
                    self.frames_dropped_q += 1
        elif not has_motion:
            with self._lock:
                self.frames_skipped += 1
        else:
            with self._lock:
                self.blurry_dropped += 1

        # ── Оверлей ───────────────────────────────────────────────────────
        m_color = (0, 200, 80) if has_motion else (120, 120, 120)
        s_color = (0, 200, 80) if is_sharp   else (0, 180, 220)
        cv2.putText(img, f"Движение: {motion*100:.1f}%",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, m_color, 2)
        cv2.putText(img, f"Резкость: {sharp:.0f}",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.60, s_color, 2)
        if self.is_sending:
            cv2.putText(img, "Отправка...",
                        (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 200, 0), 2)

        result = self.last_result
        if result:
            verdict = result.get("verdict", "")
            conf    = normalize_conf(result.get("confidence", 0))
            pct     = f"{conf*100:.1f}%"
            if is_live(verdict):
                bar_color, label = (30, 180, 60), f"ЖИВОЙ  {pct}"
            elif is_fake(verdict):
                bar_color, label = (30, 30, 210), f"ПОДДЕЛКА  {pct}"
            else:
                bar_color, label = (160, 120, 0), f"{verdict.upper()}  {pct}"

            overlay = img.copy()
            cv2.rectangle(overlay, (0, h - 52), (w, h), bar_color, -1)
            cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
            cv2.putText(img, label, (14, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─── Боковая панель ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Настройки")
    api_url      = st.text_input("URL сервера liveness", value=DEFAULT_API)
    door_api_url = st.text_input("URL открытия двери", value=DOOR_OPEN_URL,
                                 help="POST-запрос открывает дверь/турникет")

    alive, models_loaded = check_health(api_url)
    c1, c2 = st.columns(2)
    c1.metric("Сервер", "🟢 ОК" if alive        else "🔴 Выкл")
    c2.metric("Модели", "🟢 ОК" if models_loaded else "🔴 Нет")
    if not alive:
        st.error("Запустите:\n```\npython -m app.main\n```")

    st.markdown("---")
    motion_threshold = st.slider("Порог движения (%)", 1, 15, 3) / 100
    sharpness_min    = st.slider("Мин. резкость кадра", 20, 300, int(SHARPNESS_MIN),
                                 help="Выше = принимаем только очень чёткие кадры")
    window_size      = st.slider("Окно majority vote (кадров)", 1, 10, WINDOW_SIZE,
                                 help="Сколько последних ответов учитывать при голосовании")
    use_session = st.checkbox("Использовать сессию", value=True)

    # [BUG-4] Сбрасываем door_opened при переключении use_session
    if use_session != st.session_state.get("prev_use_session", True):
        st.session_state.door_opened    = False
        st.session_state.final_result   = None
        st.session_state.prev_use_session = use_session

    st.markdown("---")
    if st.button("🔄 Новая сессия", disabled=not alive):
        if st.session_state.session_id:
            delete_session_api(api_url, st.session_state.session_id)
        sid = create_session(api_url)
        if sid:
            st.session_state.session_id  = sid
            st.session_state.door_opened = False
            st.session_state.final_result = None
            st.success("Сессия создана")
        else:
            st.error("Не удалось создать сессию")

    if st.session_state.session_id:
        st.caption(f"🔑 `{st.session_state.session_id[:24]}…`")
        if st.button("🗑️ Удалить сессию"):
            delete_session_api(api_url, st.session_state.session_id)
            st.session_state.session_id  = None
            st.session_state.door_opened = False
            st.rerun()

    st.markdown(f"---\n[📄 Swagger]({api_url}/docs)")


# ─── Автосоздание сессии ──────────────────────────────────────────────────────
if use_session and not st.session_state.session_id and alive:
    sid = create_session(api_url)
    if sid:
        st.session_state.session_id = sid


# ─── Заголовок ────────────────────────────────────────────────────────────────
st.markdown("# 🔐 Контроль доступа по лицу")
st.caption(
    f"Без cooldown · каждый чёткий кадр с движением → API · "
    f"majority vote по {WINDOW_SIZE} последним ответам · JPEG {JPEG_QUALITY}%"
)
st.divider()


# ─── Layout ───────────────────────────────────────────────────────────────────
col_cam, col_result = st.columns([3, 2], gap="large")

with col_cam:
    st.markdown("### 📷 Камера")
    # Для локального запуска STUN не нужен — убираем внешние ICE серверы.
    # Если деплой на удалённый сервер — добавьте свой TURN-сервер.
    RTC_CONFIG = RTCConfiguration({"iceServers": []})
    ctx = webrtc_streamer(
        key="liveness",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=LivenessProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    if ctx.video_processor:
        proc = ctx.video_processor
        # Обновляем параметры из слайдеров [BUG-10: задержка 1 рендер — ожидаемо]
        proc.api_url          = api_url
        proc.session_id       = st.session_state.session_id if use_session else None
        proc.motion_threshold = motion_threshold
        proc.sharpness_min    = float(sharpness_min)
        # [BUG-2] Потокобезопасная замена окна через метод
        if proc._window.maxlen != window_size:
            proc.set_window_size(window_size)

with col_result:
    st.markdown("### 📊 Результат")

    result_ph = st.empty()
    votes_ph  = st.empty()
    stats_ph  = st.empty()

    if ctx.video_processor:
        proc    = ctx.video_processor
        result  = proc.last_result
        votes   = proc.window_votes
        motion  = proc.motion_score
        sharp   = proc.sharpness_score
        sending = proc.is_sending

        # [BUG-1] Синхронизируем статистику из proc в session_state
        st.session_state.frames_sent      = proc.frames_sent
        st.session_state.frames_skipped   = proc.frames_skipped
        st.session_state.blurry_dropped   = proc.blurry_dropped
        st.session_state.frames_dropped_q = proc.frames_dropped_q

        # ── Карточка вердикта ────────────────────────────────────────────
        if sending and not result:
            result_ph.markdown("""
            <div class="card-checking">
                <div class="icon">🔄</div>
                <h2>Анализирую...</h2>
                <p>Отправляю кадры на API</p>
            </div>""", unsafe_allow_html=True)

        elif result:
            verdict = result.get("verdict", "")
            conf_f  = normalize_conf(result.get("confidence", 0))
            pct     = f"{conf_f*100:.1f}%"
            bar_w   = int(conf_f * 100)

            if is_live(verdict):
                result_ph.markdown(f"""
                <div class="card-granted">
                    <div class="icon">✅</div>
                    <h2>Доступ разрешён</h2>
                    <p>Живой человек · уверенность {pct}</p>
                    <div class="conf-bar-wrap">
                        <div class="conf-bar" style="width:{bar_w}%;background:#22c55e;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # [BUG-6] open_door вызывается ровно один раз:
                # только если дверь ещё не открыта и результат стал LIVE
                if not st.session_state.door_opened:
                    ok = open_door(door_api_url)
                    st.session_state.door_opened = True  # ставим флаг в любом случае
                    if not ok:
                        st.session_state._door_error = True
                    st.rerun()  # [BUG-7] принудительный ре-рендер для отображения статуса

                if st.session_state.get("_door_error"):
                    votes_ph.warning("⚠️ Не удалось открыть дверь — проверьте URL в настройках")
                else:
                    votes_ph.success("🚪 Дверь открыта автоматически!")

            elif is_fake(verdict):
                result_ph.markdown(f"""
                <div class="card-denied">
                    <div class="icon">🚫</div>
                    <h2>Доступ запрещён</h2>
                    <p>Обнаружена подделка · уверенность {pct}</p>
                    <div class="conf-bar-wrap">
                        <div class="conf-bar" style="width:{bar_w}%;background:#f43f5e;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
                st.session_state.door_opened = False
                st.session_state.pop("_door_error", None)
                votes_ph.empty()

            else:
                result_ph.markdown(f"""
                <div class="card-waiting">
                    <div class="icon">🔍</div>
                    <p>{verdict.upper()} · {pct}</p>
                </div>""", unsafe_allow_html=True)
                votes_ph.empty()

        else:
            result_ph.markdown("""
            <div class="card-waiting">
                <div class="icon">📸</div>
                <p>Нажмите <b>START</b> и встаньте перед камерой</p>
            </div>""", unsafe_allow_html=True)

        # ── Голоса серии ─────────────────────────────────────────────────
        if votes and not sending:
            vote_html = " ".join(
                f'<span class="vote-live">✅ {v}</span>' if is_live(v)
                else f'<span class="vote-fake">❌ {v}</span>'
                for v in votes
            )
            votes_ph.markdown(
                f'<div style="margin-top:0.5rem"><b style="font-size:0.8rem;color:#64748b;">'
                f'Окно ({len(votes)} ответов):</b>'
                f'<div class="votes">{vote_html}</div></div>',
                unsafe_allow_html=True,
            )

        # ── Статистика ────────────────────────────────────────────────────
        m_badge = (
            f'<span class="badge-on">Движение: {motion*100:.1f}%</span>'
            if motion >= motion_threshold
            else f'<span class="badge-off">Нет движения: {motion*100:.1f}%</span>'
        )
        s_badge = (
            f'<span class="badge-on">Резкость: {sharp:.0f}</span>'
            if sharp >= sharpness_min
            else f'<span class="badge-blur">Размыто: {sharp:.0f}</span>'
        )
        stats_ph.markdown(f"""
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:0.6rem">
            {m_badge} {s_badge}
        </div>
        <div class="stats-row">
            <div class="stat">
                <div class="val">{proc.frames_sent}</div>
                <div class="lbl">Кадров отправлено</div>
            </div>
            <div class="stat">
                <div class="val">{proc.frames_skipped}</div>
                <div class="lbl">Пропущено (нет движения)</div>
            </div>
            <div class="stat">
                <div class="val">{proc.blurry_dropped}</div>
                <div class="lbl">Размытых</div>
            </div>
            <div class="stat">
                <div class="val">{proc.frames_dropped_q}</div>
                <div class="lbl">Дроп (очередь полна)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        result_ph.markdown("""
        <div class="card-waiting">
            <div class="icon">📸</div>
            <p>Нажмите <b>START</b> на камере чтобы начать</p>
        </div>""", unsafe_allow_html=True)


# ─── Подсказка ────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Алгоритм:** движение + резкость → кадр сразу в очередь → воркер отправляет без cooldown "
    "→ скользящее окно majority vote → ЖИВОЙ = дверь открывается автоматически"
)
