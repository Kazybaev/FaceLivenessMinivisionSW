Plan

Windows Edge Turnstile v1 Production Plan
Summary
Make app/ the only production codepath; keep root liveness_*.py files as lab/demo scripts.
Ship one Windows edge application for one camera and one turnstile, offline after install, security-first.
Scope v1 to photo / phone screen / replay video defense only; do not include 3D mask / deepfake.
Release gate: attack acceptance <= 1% on the v1 spoof set, real-user pass rate >= 92% in target conditions, median decision <= 1.5s, p95 <= 2.5s.
Key Hypotheses
A single stateful turnstile engine will outperform the current split between single-image DL and frame-based heuristic logic.
Field-calibrated thresholds on the real deployment camera/light will improve security more than more demo scripts or more UI variants.
Bundled assets, fixed dependencies, and startup checks will remove more production failures than model tuning alone in the first release.
Implementation Changes
Build one TurnstileDecisionEngine in app/ with states NO_FACE, POSITIONING, ANALYZING, ACCESS_GRANTED, ACCESS_DENIED, COOLDOWN.
Use one frame-stream pipeline: quality gate first, MediaPipe landmarks every frame, heuristic signals every frame, MiniFASNet on stable good frames, then weighted fusion plus hard spoof overrides.
Deny on strong spoof evidence even if one other signal is weak: screen flicker, rigid flat motion, low parallax, or high fake confidence from MiniFASNet.
Bundle face_landmarker.task, anti-spoof weights, and detector assets inside the Windows build; remove production auto-download behavior and fail fast if any asset is missing.
Keep FastAPI only as a local control plane on 127.0.0.1; it is not the primary network product in v1.
Add a kiosk-ready UI with live camera, face guide, current state, confidence bar, reason text, and a short cooldown after the final decision.
Add structured JSON logs plus per-session audit records containing timestamp, verdict, confidence, reason codes, latency, camera id, and model versions.
Standardize the dev/runtime environment: pin Python 3.11, add a locked dependency file, install .[dev] in CI, and make python -m pytest the canonical test command.
Keep Docker for QA and API smoke tests only; do not use Docker as the Windows edge production runtime.
Public Interfaces
Add local loopback endpoints GET /health, GET /ready, and GET /decision/latest.
Add an outbound decision webhook configured by YAML/env. On each final verdict, send a DecisionEvent JSON payload with session_id, device_id, camera_id, timestamp_utc, verdict, confidence, reason_codes, latency_ms, and model_versions.
Standardize controller-facing verdicts to ACCESS_GRANTED and ACCESS_DENIED. Keep NO_FACE, POSITIONING, and ANALYZING as internal/UI states only.
Extend config with camera.index, camera.width, camera.height, turnstile.decision_window_ms, turnstile.cooldown_ms, turnstile.webhook_url, turnstile.audit_log_dir, and per-signal thresholds.
Test Plan
Unit test the state machine, fusion logic, hard spoof overrides, startup asset checks, and webhook payload generation.
Add integration tests with prerecorded clips for: real person, printed photo, phone still image, replay video on phone, and poor-quality/no-face input.
Add failure-path tests for missing camera, missing model files, corrupted frames, and unavailable webhook target; all must fail safe to ACCESS_DENIED or not_ready.
Build a field validation set from the actual deployment camera and lighting, then calibrate thresholds against that set before the pilot.
Pilot acceptance: zero false grants in the pilot spoof set, no crash in an 8-hour soak test, and repeatable startup on a clean Windows machine.
Assumptions
First release targets one Windows edge PC attached to one turnstile and one camera.
v1 defends only against 2D spoofing.
v1 shows status on screen and emits structured events; direct relay control stays in the downstream controller integration.
The right base for production is the existing app/ architecture, because it already has config, DI, API, and tests, while the root scripts are still experimental.
The main repo gaps to fix first are the split analyzer design, asset auto-downloads, non-standardized environment setup, and missing runnable test environment.



