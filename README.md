# Access Control Anti-Spoofing MVP

Production-like MVP on Python for continuous camera processing, session-based anti-spoofing, suspicious object screening, and FastAPI status/control endpoints.

## What this MVP does

- Continuously reads frames from a camera in real time
- Runs a session-based access check instead of a single-frame decision
- Uses a YOLO-compatible object detector layer to watch for suspicious objects such as phone, tablet, monitor, laptop screen, or paper/photo sheet
- Tracks one face across frames
- Sends a short frame sequence to an anti-spoofing model interface
- Allows handoff to face recognition only after anti-spoofing returns `real`
- Denies access on `spoof` and `uncertain`
- Blocks the whole session if a suspicious object appears even once during the session
- Exposes current runtime status, current decision, events, and session reset through FastAPI

## Current implementation

- `app/core`, `app/services`, and `app/utils` contain the new MVP skeleton
- Camera runs in a dedicated service thread
- Pipeline runs in a separate background thread
- YOLO detector is implemented as a pluggable service
- Default YOLO backend is `mock`, so the project starts even without a real YOLO model
- Anti-spoofing uses a temporal mock implementation that consumes a frame buffer
- Face recognition is a stub gateway for future integration

## File tree

```text
app/
  api/
    routes.py
  core/
    decision_engine.py
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
README.md
```

## Main runtime flow

1. `CameraService` continuously reads frames.
2. `AccessControlPipeline` pulls the latest frame in a loop.
3. `YoloDetector` checks suspicious objects.
4. `OpenCvHaarFaceDetector` finds faces.
5. `FaceTracker` keeps one tracked face across frames.
6. `SessionManager` maintains session state and frame buffer.
7. `MockTemporalAntiSpoofModel` evaluates a short frame sequence.
8. `DecisionEngine` decides `allow` or `deny`.
9. `RecognitionGateway` receives the tracked face only on `allow`.

## Session logic

Session states include:

- `idle`
- `observing`
- `suspicious_object_detected`
- `spoof_detected`
- `real_detected`
- `blocked`
- `cooldown`
- `allowed`

Rules:

- If YOLO sees a suspicious object during the current session, the session is denied.
- The same session cannot continue after that object disappears.
- A new attempt is possible only after cooldown.
- If anti-spoofing returns `uncertain`, access is denied.
- YOLO is only an extra security layer. It never proves that a person is real.

## FastAPI endpoints

- `GET /health`
- `GET /status`
- `POST /session/reset`
- `GET /events`
- `GET /current-decision`
- `GET /health/runtime`

The legacy routes already present in the repository remain available too.

## Local run

### 1. Create and activate a virtual environment

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

### 3. Start the API and continuous runtime

```powershell
python -m app.main
```

Then open:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/status`
- `http://127.0.0.1:8000/current-decision`
- `http://127.0.0.1:8000/events`

### 4. Start with a visible camera window

If you want to see the camera feed on screen, run preview mode:

```powershell
python -m app.main --preview
```

You can also choose a different camera:

```powershell
python -m app.main --preview --camera 1
```

Controls:

- `Q` to quit
- `R` to reset the current session

FastAPI stays available during preview unless you add `--no-api` to `python -m app.cli.access_preview`.

## Config via environment

Examples:

```powershell
$env:ACCESS_CAMERA__INDEX="0"
$env:ACCESS_YOLO__BACKEND="mock"
$env:ACCESS_RUNTIME__AUTOSTART="true"
python -m app.main
```

To switch to a real YOLO model later:

```powershell
$env:ACCESS_YOLO__BACKEND="ultralytics"
$env:ACCESS_YOLO__MODEL_PATH="models\\yolov8n.pt"
python -m app.main
```

## Tests

```powershell
python -m pytest -q
```

## Notes

- This is an MVP skeleton, not a finished high-security product.
- The anti-spoofing module is intentionally replaceable.
- The current face detector is lightweight and chosen to keep the MVP runnable out of the box.
- For production hardening, replace the mock anti-spoof and mock YOLO path with real models and deployment-specific calibration.
