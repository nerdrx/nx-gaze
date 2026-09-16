"""Camera capture and gaze estimation, isolated from the UI thread."""

import sys
import threading
import time

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
from core import BlinkFilter


class CameraWorker(QThread):
    sample = pyqtSignal(object, object, bool)
    failed = pyqtSignal(str)
    ready = pyqtSignal()
    trained = pyqtSignal(int)
    prediction = pyqtSignal(object)
    pose_state_changed = pyqtSignal(bool)
    head_support_changed = pyqtSignal(int)
    comparison = pyqtSignal(object, object, bool)
    candidate_finished = pyqtSignal(bool)
    candidate_retention = pyqtSignal(bool)

    def __init__(self, camera_index: int, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.preview_enabled = False
        self._train_lock = threading.Lock()
        self._pending_train = None
        self._training_id = 0
        self._pending_decision = None

    def request_train(self, features, targets, candidate=False, baseline=None):
        """Queue a calibration snapshot; a newer request replaces a pending one."""
        with self._train_lock:
            self._training_id += 1
            self._pending_train = (
                self._training_id, candidate,
                None if baseline is None else tuple([[list(row) for row in part] for part in baseline]),
                [list(row) for row in features],
                [list(row) for row in targets],
            )
            return self._training_id

    def finish_candidate(self, accept):
        with self._train_lock:
            self._pending_decision = bool(accept)

    def run(self):
        cap = estimator = None
        try:
            # Import heavyweight dependencies here so opening the UI stays quick.
            import cv2
            import numpy as np
            from head_tracking import HeadAwareEstimator

            if self.isInterruptionRequested():
                return
            camera_api = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.camera_index, camera_api)
            if not cap.isOpened():
                raise RuntimeError(
                    f"Cannot open camera {self.camera_index}. Check its permissions "
                    "and close other apps using it."
                )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # EyeTrax downloads its landmark model on first use; no frames are uploaded.
            estimator = HeadAwareEstimator()
            if self.isInterruptionRequested():
                return
            self.ready.emit()
            is_trained = False
            candidate_active = False
            preview_at = 0.0
            previous_pose_state = None
            blink_filter = BlinkFilter()
            while not self.isInterruptionRequested():
                frame_started = time.monotonic()
                with self._train_lock:
                    training, self._pending_train = self._pending_train, None
                    decision, self._pending_decision = self._pending_decision, None
                if training is not None:
                    training_id, candidate, baseline, *values = training
                    features, targets = (np.asarray(value, dtype=float) for value in values)
                    if (
                        features.ndim != 2
                        or features.shape[0] < 3
                        or features.shape[1] == 0
                        or targets.shape != (features.shape[0], 2)
                        or not np.isfinite(features).all()
                        or not np.isfinite(targets).all()
                    ):
                        raise ValueError("Calibration samples are incomplete. Please calibrate again.")
                    if candidate:
                        estimator.begin_candidate(features, targets)
                        candidate_active = True
                        retained = True
                        if baseline is not None:
                            bx, by = (np.asarray(v, dtype=float) for v in baseline)
                            before, after = estimator.comparison(bx)
                            _, groups = np.unique(by, axis=0, return_inverse=True)
                            for group in np.unique(groups):
                                mask = groups == group
                                old = np.median(np.linalg.norm(before[mask]-by[mask], axis=1))
                                new = np.median(np.linalg.norm(after[mask]-by[mask], axis=1))
                                retained &= bool(new <= max(old*1.2, old+10))
                                old_axes = np.median(np.abs(before[mask]-by[mask]), axis=0)
                                new_axes = np.median(np.abs(after[mask]-by[mask]), axis=0)
                                retained &= bool((new_axes <= np.maximum(old_axes*1.2, old_axes+10)).all())
                        self.candidate_retention.emit(retained)
                    else:
                        estimator.train(features, targets)
                    self.head_support_changed.emit(int(np.count_nonzero(getattr(estimator, 'pose_weights', []))))
                    is_trained = True
                    self.trained.emit(training_id)
                if decision is not None:
                    if candidate_active:
                        estimator.finish_candidate(decision)
                        candidate_active = False
                    self.head_support_changed.emit(int(np.count_nonzero(getattr(estimator, 'pose_weights', []))))
                    self.candidate_finished.emit(decision)
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.sample.emit(None, None, False)
                    raise RuntimeError("Camera stopped providing frames. Reconnect it and start again.")
                features, blink = estimator.extract_features(frame)
                if features is not None and not np.isfinite(features).all():
                    features = None
                preview = None
                now = time.monotonic()
                if self.preview_enabled and now >= preview_at:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width = rgb.shape[:2]
                    preview = QImage(
                        rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888
                    ).copy()
                    preview_at = now + 0.1
                blink = blink_filter.update(features is not None, bool(blink), now)
                self.sample.emit(features, preview, blink)
                point = None
                if is_trained and features is not None and not blink:
                    xy = estimator.predict([features])[0]
                    pose_state = getattr(estimator, 'pose_in_range', True)
                    if pose_state != previous_pose_state:
                        self.pose_state_changed.emit(pose_state)
                        previous_pose_state = pose_state
                    if candidate_active:
                        before, after = estimator.comparison([features])
                        self.comparison.emit(tuple(before[0]), tuple(after[0]), pose_state)
                    if np.isfinite(xy).all():
                        point = (float(xy[0]), float(xy[1]))
                self.prediction.emit(point)
                remaining = 1 / 30 - (time.monotonic() - frame_started)
                if remaining > 0:
                    self.msleep(max(1, int(remaining * 1000 + 0.999)))
        except Exception as exc:
            if not self.isInterruptionRequested():
                if isinstance(exc, ImportError):
                    message = "Tracking dependencies are missing or incompatible. Run ./setup.sh to install them."
                else:
                    message = str(exc) or type(exc).__name__
                self.failed.emit(message)
        finally:
            self.prediction.emit(None)
            if cap is not None:
                cap.release()
            if estimator is not None:
                estimator.close()
