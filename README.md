# Face Liveness and Access Control MVP

Production-like MVP on Python for:
- continuous camera or video processing
- suspicious object detection before face recognition
- anti-spoofing on a short frame sequence
- visible preview window and FastAPI status endpoints

## What the project does now

The current runtime works in this order:

1. Read frames continuously from a camera or video file.
2. Run YOLO object detection first.
3. If a suspicious object is found, do not allow access.
4. If no suspicious object is found, run anti-spoofing on a short frame buffer.
5. Only if the face is confirmed as `real`, allow the face to move forward to the face-recognition gateway stub.

Suspicious objects include:
- phone
- cell phone
- smartphone
- mobile phone
- tablet
- iPad
- monitor
- display
- screen
- laptop screen
- printed photo
- paper photo
- photo sheet

## Current behavior

- `phone / screen / tablet / printed photo` in frame -> `DENY`
- suspicious object removed -> automatic retry starts quickly
- `anti-spoof == real` and no suspicious object -> `ALLOW`
- `anti-spoof == spoof/uncertain/low confidence` -> `DENY`, then short automatic retry
- preview window shows a simple user-facing state:
  - `REAL`
  - `FAKE / RETRY`

## YOLO model selection

At runtime the app tries to enable a local Ultralytics model automatically.

Current search order inside `models/`:

1. `yolo26n.pt`
2. `yolo26s.pt`
3. `yolo26m.pt`
4. `yolo11n.pt`
5. `yolov8n.pt`

If one of these files exists, runtime switches to `YOLO: ULTRALYTICS`.
If none of them exists, runtime falls back to `YOLO: MOCK`.

Important:
- `YOLO: MOCK` means real phone detection is not active.
- `YOLO: ULTRALYTICS` means suspicious-object detection is active.

The repository already contains:

- `models/yolov8n.pt`

So the project can already start with a real YOLO model without extra manual setup.

## Preview overlay

The preview window shows:

- current state
- current decision
- current reason
- suspicious labels
- YOLO backend
- loaded YOLO model

Examples:

- `YOLO: ULTRALYTICS`
- `MODEL: yolov8n.pt`
- `STATE: REAL`
- `STATE: FAKE / RETRY`

## Main runtime flow

1. `CameraService` reads the latest frame continuously.
2. `YoloDetector` checks suspicious objects first.
3. `OpenCvHaarFaceDetector` finds a face.
4. `FaceTracker` keeps the active face stable across frames.
5. `SessionManager` stores state and frame buffer.
6. `MockTemporalAntiSpoofModel` evaluates a short sequence.
7. `DecisionEngine` returns `ALLOW`, `DENY`, or retry behavior.
8. `RecognitionGateway` receives the face only after `ALLOW`.

## Project structure

```text
app/
  api/
    routes.py
  cli/
    access_preview.py
    turnstile_kiosk.py
  core/
    decision_engine.py
    detection_policy.py
    models.py
    pipeline.py
    schemas.py
    session_manager.py
  services/
    anti_spoof_service.py
    camera_service.py
    event_logger.py
    face_detector.py
    face_tracker.py
    recognition_gateway.py
    yolo_detector.py
  utils/
    geometry.py
    image_utils.py
    timers.py
  config.py
  main.py
tests/
requirements.txt
requirements-lock.txt
README.md
```

## Local setup

### 1. Create a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run modes

### Visible preview with camera

```powershell
python -m app.main --preview
```

### Visible preview with another camera index

```powershell
python -m app.main --preview --camera 1
```

### Visible preview with a video file

```powershell
python -m app.main --preview --video "D:\face_liveness_minivision\vid_1.mp4"
```

### API only

```powershell
python -m app.main
```

### Optional script entry point

```powershell
face-liveness-api
```

## Preview controls

- `Q` -> quit
- `R` -> reset the current session

## FastAPI endpoints

- `GET /health`
- `GET /status`
- `POST /session/reset`
- `GET /events`
- `GET /current-decision`
- `GET /health/runtime`

Legacy routes already present in the repository are still included too.

## Environment overrides

Examples:

```powershell
$env:ACCESS_CAMERA__INDEX="0"
$env:ACCESS_RUNTIME__AUTOSTART="true"
python -m app.main --preview
```

Force a specific YOLO model:

```powershell
$env:ACCESS_YOLO__BACKEND="ultralytics"
$env:ACCESS_YOLO__MODEL_PATH="models\\yolo26s.pt"
python -m app.main --preview
```

## Tests

Run the test suite with:

```powershell
python -m pytest -q
```

The current repository state passes the local test suite.

## Notes

- The anti-spoof module is still replaceable and currently uses a temporal mock implementation.
- Face recognition is a stub gateway for the next stage of integration.
- For a stronger phone detector, prefer `yolo26n.pt` for better speed or `yolo26s.pt` for higher accuracy.
- If preview becomes too slow on your machine, use a smaller camera resolution or a lighter YOLO model.
