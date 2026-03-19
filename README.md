# Face Liveness Detection

> Определяет — **живой человек** перед камерой или это **фото / экран / спуфинг** — в реальном времени, без облака, без платных API.

---

## Как это работает

Система проверяет **4 независимых признака**. Все 4 должны пройти, чтобы подтвердить живого человека:

| # | Проверка | Почему работает |
|---|----------|----------------|
| 1 | **Моргание** (EAR — Eye Aspect Ratio) | Фото и экраны никогда не моргают |
| 2 | **Движение головы** | Держащий фото двигает всю картинку как единый жёсткий объект |
| 3 | **Соотношение движения носа и глаз** | При повороте живой головы нос смещается иначе, чем уголки глаз — на плоском фото всё движется синхронно |
| 4 | **Текстура кожи** (Laplacian variance) | Живая кожа имеет поры и микротекстуру; печатные и экранные изображения слишком гладкие |

---

## Стек

- **MiniFASNet** (minivision) — нейросеть anti-spoofing, ~98% точность, ~10 мс на CPU
- **MediaPipe Face Mesh** — 478 landmarks лица в реальном времени
- **OpenCV** — захват видео, детектор лиц (Caffe / RetinaFace), UI
- **PyTorch** — инференс модели

---

## Структура репозитория

```
FaceLivenessMinivisionSW/
├── liveness_detection.py      # главный файл — запускать его
├── test.py                    # тест на одном изображении
├── train.py                   # дообучение модели
├── face_landmarker.task       # модель MediaPipe (скачивается автоматически)
├── req.txt                    # зависимости
├── resources/
│   ├── anti_spoof_models/     # веса MiniFASNet (.pth)
│   └── detection_model/       # детектор лиц (Caffe)
├── src/
│   ├── model_lib/             # архитектуры MiniFASNet
│   ├── data_io/               # трансформации
│   └── utility.py             # вспомогательные функции
├── datasets/                  # датасеты для обучения
└── images/                    # тестовые изображения
```

---

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/Kazybaev/FaceLivenessMinivisionSW.git
cd FaceLivenessMinivisionSW
```

### 2. Установить Git LFS и скачать модели

Веса моделей хранятся через Git LFS. Без этого шага `.pth` файлы будут пустышками (132 байта).

```bash
sudo apt-get install -y git-lfs   # Ubuntu/Debian
git lfs install
git lfs pull
```

Проверка — файлы должны весить ~1.8 МБ каждый:

```bash
ls -la resources/anti_spoof_models/
```

### 3. Создать виртуальное окружение и установить зависимости

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 4. Запуск тестов

```bash
# Все тесты
pytest tests/ -v

# Только unit-тесты
pytest tests/unit/ -v

# Только интеграционные тесты
pytest tests/integration/ -v
```

### 5a. Запуск — режим веб-камеры (standalone)

```bash
source .venv/bin/activate
python liveness_detection.py
```

При первом запуске файл `face_landmarker.task` (~2 МБ) скачается автоматически.

### 5b. Запуск — REST API сервер (FastAPI)

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Или через Docker:

```bash
docker compose up --build
```

После запуска API доступен на `http://localhost:8000`.

#### API-эндпоинты

| Метод | URL | Описание |
|-------|-----|----------|
| `GET` | `/health` | Проверка что сервер жив |
| `GET` | `/health/ready` | Проверка что модели загружены |
| `POST` | `/api/v1/liveness/analyze` | Анализ одного изображения (form-data, поле `file`) |
| `POST` | `/api/v1/liveness/session` | Создать сессию для видеопотока |
| `POST` | `/api/v1/liveness/session/{id}/frame` | Отправить кадр в сессию |
| `DELETE` | `/api/v1/liveness/session/{id}` | Удалить сессию |

#### Пример вызова

```bash
# Анализ одного фото
curl -X POST http://localhost:8000/api/v1/liveness/analyze \
  -F "file=@images/sample/image_T1.jpg"

# Ответ:
# {"verdict":"REAL","confidence":0.999,"method":"deep_learning","details":{...}}
```

---

## Управление

| Клавиша | Действие |
|---------|----------|
| `Q` | Выход |
| `R` | Сброс счётчиков (начать проверку заново) |

---

## Что отображается на экране

```
┌─────────────────────────────────────────┐
│  REAL PERSON          Score: 4/4        │
├──────────────────┬──────────────────────┤
│                  │  LIVENESS CHECK      │
│   [камера]       │  1. Blinks:    3/3 ✓ │
│                  │  2. Head move: 10/10✓ │
│   • точки глаз   │  3. Face depth: 6/6 ✓│
│   • точка носа   │  4. Skin texture: ✓  │
│                  │                      │
│                  │  RESULT: REAL PERSON │
└──────────────────┴──────────────────────┘
```

---

## Настройка порогов

Все параметры находятся в начале файла `liveness_detection.py`:

```python
BLINKS_NEEDED  = 3     # сколько морганий нужно
MOVES_NEEDED   = 10    # сколько движений головы нужно
EAR_THRESHOLD  = 0.22  # порог закрытого глаза (выше = чувствительнее)
MOVE_PIXELS    = 16    # минимальный сдвиг носа в пикселях
MOVE_MAX_JUMP  = 35    # максимальный сдвиг (больше = тряска камеры, игнорируется)
TEXTURE_MIN    = 90.0  # порог текстуры кожи (выше = строже)
RATIO_DIFF_MIN = 0.25  # минимальное различие движения носа и глаз
```

---

## Известные ограничения

| Атака | Статус |
|-------|--------|
| Фото (распечатка) | ✅ Блокируется |
| Видео на экране телефона | ✅ Блокируется (нет независимого моргания) |
| 3D-маска силиконовая | ⚠️ Частично (текстура) |
| Deepfake / GAN | ❌ Не поддерживается — нужна отдельная модель |

---

## Модель MiniFASNet

Проект использует [Silent Face Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) от Minivision.

- Точность: ~98% на NUAA
- Размер модели: ~600 КБ
- Лицензия: Apache 2.0

---

## Лицензия

Apache License 2.0 — см. файл [LICENSE](LICENSE)
