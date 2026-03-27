from __future__ import annotations

from typing import Any
import threading
import time

from app.config import Settings
from app.core.detection_policy import should_block_session
from app.core.decision_engine import DecisionEngine
from app.core.models import (
    DecisionRecord,
    DecisionVerdict,
    EventSeverity,
    ObjectDetection,
    SessionState,
)
from app.core.session_manager import SessionManager
from app.services.anti_spoof_service import MockTemporalAntiSpoofModel
from app.services.camera_service import CameraService
from app.services.event_logger import EventLogger
from app.services.face_detector import OpenCvHaarFaceDetector
from app.services.face_tracker import FaceTracker
from app.services.recognition_gateway import RecognitionGateway
from app.services.yolo_detector import YoloDetector
from app.utils.timers import monotonic_seconds


class AccessControlPipeline:
    """Continuous real-time access-control loop driven by camera frames."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._event_logger = EventLogger(max_events=settings.logging.max_events)
        self._camera_service = CameraService(settings.camera, self._event_logger)
        self._yolo_detector = YoloDetector(settings.yolo)
        self._face_detector = OpenCvHaarFaceDetector(settings.face)
        self._face_tracker = FaceTracker(settings.face)
        self._anti_spoof_model = MockTemporalAntiSpoofModel(settings.anti_spoof)
        self._recognition_gateway = RecognitionGateway()
        self._session_manager = SessionManager(settings.session, settings.anti_spoof, self._event_logger)
        self._decision_engine = DecisionEngine(settings.session, settings.anti_spoof)
        self._running = False
        self._thread: threading.Thread | None = None
        self._latest_decision: DecisionRecord | None = None
        self._last_processed_frame_id = 0
        self._last_logged_decision_key: tuple | None = None
        self._snapshot_lock = threading.Lock()
        self._latest_preview_frame = None
        self._latest_preview_meta: dict[str, Any] = {
            "frame_id": 0,
            "session_id": None,
            "session_state": SessionState.IDLE.value,
            "blocked_by_suspicious_object": False,
            "blocked_reason": None,
            "suspicious_object_types": [],
            "tracked_face": None,
            "objects": [],
            "suspicious_objects": [],
            "anti_spoof_result": None,
            "decision": None,
            "buffered_frames": 0,
        }

    def start(self) -> None:
        if self._running:
            return
        self._camera_service.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._camera_service.stop()

    def reset_session(self) -> str:
        session = self._session_manager.reset(monotonic_seconds(), reason="api_reset")
        return session.session_id

    def get_events(self, limit: int = 100):
        return self._event_logger.list_events(limit=limit)

    def get_current_decision(self) -> DecisionRecord | None:
        return self._latest_decision

    def get_preview_snapshot(self) -> dict[str, object]:
        with self._snapshot_lock:
            snapshot = dict(self._latest_preview_meta)
            frame = self._latest_preview_frame
        snapshot["frame"] = None if frame is None else frame.copy()
        return snapshot

    def get_status(self) -> dict[str, object]:
        camera_status = self._camera_service.status()
        session_snapshot = self._session_manager.snapshot(monotonic_seconds())
        return {
            "runtime_running": self._running,
            "camera_running": bool(camera_status["running"]),
            "camera_error": camera_status["error"],
            "yolo_backend": self._yolo_detector.backend_name,
            "anti_spoof_backend": self._settings.anti_spoof.backend,
            "frame_counter": int(camera_status["frame_counter"]),
            "session": session_snapshot,
        }

    def _loop(self) -> None:
        self._event_logger.log("pipeline_started", "Continuous access-control pipeline started")
        while self._running:
            packet = self._camera_service.get_latest_frame()
            if packet is None or packet.frame_id == self._last_processed_frame_id:
                time.sleep(self._settings.runtime.loop_sleep_seconds)
                continue

            self._last_processed_frame_id = packet.frame_id
            self._process_packet(packet)
            time.sleep(self._settings.runtime.loop_sleep_seconds)

    def _process_packet(self, packet) -> None:
        objects = self._yolo_detector.detect(packet.frame)
        faces = self._face_detector.detect(packet.frame)
        tracked_face = self._face_tracker.update(faces, packet.timestamp)
        session = self._session_manager.on_frame(packet, tracked_face)

        suspicious_objects: list[ObjectDetection] = []
        if session.state != SessionState.IDLE and objects:
            should_block, suspicious_types = should_block_session(objects, self._settings.yolo.suspicious_labels)
            if should_block:
                suspicious_objects = [obj for obj in objects if obj.label in suspicious_types]
        self._session_manager.update_suspicious_state(suspicious_objects, packet.timestamp)

        anti_spoof_result = None
        if tracked_face is not None and not self._session_manager.current_session.suspicious_object_seen:
            anti_spoof_result = self._anti_spoof_model.predict(
                list(self._session_manager.current_session.frame_buffer),
                tracked_face,
            )
            self._session_manager.record_anti_spoof_result(anti_spoof_result)

        decision = self._decision_engine.evaluate(
            self._session_manager.current_session,
            suspicious_objects,
            anti_spoof_result,
            packet.timestamp,
        )
        if decision is None:
            self._update_preview_snapshot(
                packet,
                objects=objects,
                tracked_face=tracked_face,
                suspicious_objects=suspicious_objects,
                anti_spoof_result=anti_spoof_result,
                decision=self._latest_decision,
            )
            return

        self._session_manager.apply_decision(decision, packet.timestamp)
        self._latest_decision = decision

        self._update_preview_snapshot(
            packet,
            objects=objects,
            tracked_face=tracked_face,
            suspicious_objects=suspicious_objects,
            anti_spoof_result=anti_spoof_result,
            decision=decision,
        )

        # Recognition must stay behind the hard-block decision gate.
        if decision.verdict == DecisionVerdict.ALLOW and tracked_face is not None:
            self._recognition_gateway.submit(tracked_face, packet.frame, packet.timestamp)

        self._log_decision_if_needed(decision)

    def _log_decision_if_needed(self, decision: DecisionRecord) -> None:
        key = (
            decision.session_id,
            decision.state.value,
            decision.verdict.value,
            decision.reason,
        )
        if decision.verdict == DecisionVerdict.PENDING or key == self._last_logged_decision_key:
            return
        severity = EventSeverity.INFO if decision.verdict == DecisionVerdict.ALLOW else EventSeverity.WARNING
        self._event_logger.log(
            "decision",
            decision.reason,
            severity=severity,
            session_id=decision.session_id,
            payload={
                "state": decision.state.value,
                "verdict": decision.verdict.value,
                "confidence": decision.confidence,
                "details": decision.details,
            },
        )
        self._last_logged_decision_key = key

    def _update_preview_snapshot(
        self,
        packet,
        *,
        objects: list[ObjectDetection],
        tracked_face,
        suspicious_objects: list[ObjectDetection],
        anti_spoof_result,
        decision: DecisionRecord | None,
    ) -> None:
        session = self._session_manager.current_session
        with self._snapshot_lock:
            self._latest_preview_frame = packet.frame
            self._latest_preview_meta = {
                "frame_id": packet.frame_id,
                "session_id": session.session_id,
                "session_state": session.state.value,
                "blocked_by_suspicious_object": session.blocked_by_suspicious_object,
                "blocked_reason": session.blocked_reason,
                "suspicious_object_types": list(session.suspicious_object_types),
                "tracked_face": None if tracked_face is None else self._bbox_payload(tracked_face.bbox),
                "objects": [self._object_payload(obj) for obj in objects],
                "suspicious_objects": [self._object_payload(obj) for obj in suspicious_objects],
                "anti_spoof_result": self._anti_spoof_payload(anti_spoof_result),
                "decision": self._decision_payload(decision),
                "buffered_frames": len(session.frame_buffer),
            }

    @staticmethod
    def _bbox_payload(bbox) -> dict[str, int]:
        return {
            "x": bbox.x,
            "y": bbox.y,
            "width": bbox.width,
            "height": bbox.height,
            "x2": bbox.x2,
            "y2": bbox.y2,
        }

    def _object_payload(self, detection: ObjectDetection) -> dict[str, object]:
        return {
            "label": detection.label,
            "confidence": detection.confidence,
            "bbox": self._bbox_payload(detection.bbox),
        }

    def _anti_spoof_payload(self, result) -> dict[str, object] | None:
        if result is None:
            return None
        return {
            "label": result.label.value,
            "confidence": result.confidence,
            "model_name": result.model_name,
            "details": result.details,
        }

    def _decision_payload(self, decision: DecisionRecord | None) -> dict[str, object] | None:
        if decision is None:
            return None
        return {
            "session_id": decision.session_id,
            "verdict": decision.verdict.value,
            "state": decision.state.value,
            "allow_face_recognition": decision.allow_face_recognition,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "timestamp": decision.timestamp,
            "cooldown_until": decision.cooldown_until,
            "details": decision.details,
        }
