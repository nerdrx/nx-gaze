"""Camera capture and gaze estimation, isolated from the UI thread."""

import sys
import threading
import time

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage


class CameraWorker(QThread):
    sample = pyqtSignal(object, object, bool)
    failed = pyqtSignal(str)
    ready = pyqtSignal()
    trained = pyqtSignal(int)
    prediction = pyqtSignal(object)

    def __init__(self, camera_index: int, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.preview_enabled = False
        self._train_lock = threading.Lock()
        self._pending_train = None
        self._training_id = 0

    def request_train(self, features, targets):
        """Queue a calibration snapshot; a newer request replaces a pending one."""
        with self._train_lock:
            self._training_id += 1
            self._pending_train = (
                self._training_id,
                [list(row) for row in features],
                [list(row) for row in targets],
            )
            return self._training_id

    def run(self):
        cap = estimator = None
        try:
            # Import heavyweight dependencies here so opening the UI stays quick.
            import cv2
            import numpy as np
            from eyetrax import GazeEstimator

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
            estimator = GazeEstimator()
            if self.isInterruptionRequested():
                return
            self.ready.emit()
            is_trained = False
            preview_at = 0.0
            while not self.isInterruptionRequested():
                frame_started = time.monotonic()
                with self._train_lock:
                    training, self._pending_train = self._pending_train, None
                if training is not None:
                    training_id, *values = training
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
                    estimator.train(features, targets)
                    is_trained = True
                    self.trained.emit(training_id)
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
                self.sample.emit(features, preview, bool(blink))
                point = None
                if is_trained and features is not None and not blink:
                    xy = estimator.predict([features])[0]
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
