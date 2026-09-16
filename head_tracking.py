"""Preserve camera-relative head motion from EyeTrax's existing inference pass.

EyeTrax 0.4.0 supplies normalized eye landmarks and orientation angles. Its
normalization removes image-space translation and face scale; its Euler angles
also wrap at +/-pi. This adapter restores position/scale and encodes angles
continuously, without running a second detector. These are relative visual
features, not a metric head pose or a camera-calibrated 3D gaze ray.
"""
import numpy as np

FEATURE_SCHEMA = 'head-position-scale-cyclic-v2'


def compensate_features(features, landmarks, aspect):
    """Combine local eye shape, cyclic orientation, and camera-relative pose."""
    if features is None or landmarks is None or len(landmarks) <= 263:
        return None
    corners = np.asarray([[landmarks[i].x, landmarks[i].y, landmarks[i].z]
                          for i in (33, 263)], dtype=float)
    if not np.isfinite(corners).all():
        return None
    center = corners.mean(axis=0)[:2]
    # MediaPipe x/z use image-width units; convert y to the same units.
    span = np.linalg.norm((corners[1] - corners[0]) * [1, aspect, 1])
    if span < 1e-5:
        return None
    angles = np.asarray(features[-3:])
    result = np.concatenate([features[:-3], np.sin(angles), np.cos(angles),
                             center, [np.log(span)]])
    return result if np.isfinite(result).all() else None


class _LandmarkTap:
    """Forward the pinned upstream detector, retaining only this frame's result."""
    def __init__(self, detector):
        self.detector = detector
        self.landmarks = None

    def detect_for_video(self, image, timestamp):
        self.landmarks = None
        result = self.detector.detect_for_video(image, timestamp)
        if result.face_landmarks:
            self.landmarks = result.face_landmarks[0]
        return result

    def __getattr__(self, name):
        return getattr(self.detector, name)

    def close(self):
        self.landmarks = None
        self.detector.close()


class HeadAwareEstimator:
    def __init__(self):
        from eyetrax import GazeEstimator
        self.estimator = GazeEstimator()
        # Deliberate integration with EyeTrax 0.4.0's private detector. Keep the
        # version pinned; single-pass integration has a regression test.
        self.tap = _LandmarkTap(self.estimator._face_landmarker)
        self.estimator._face_landmarker = self.tap

    def extract_features(self, frame):
        self.tap.landmarks = None
        features, blink = self.estimator.extract_features(frame)
        return compensate_features(features, self.tap.landmarks,
                                   frame.shape[0] / frame.shape[1]), blink

    def train(self, features, targets):
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float)
        weights = np.ones(features.shape[1])
        pose = features[:, -9:]
        _, groups = np.unique(targets, axis=0, return_inverse=True)
        residual = pose.copy()
        for group in np.unique(groups):
            selected = groups == group
            residual[selected] -= pose[selected].mean(axis=0)
        # Between-target head motion says nothing about compensation direction.
        # Enable a pose feature only with meaningful movement at a fixed target.
        floors = np.array([.03] * 6 + [.025, .025, .05])
        within = residual.std(axis=0)
        pooled = pose.std(axis=0)
        weights[-9:] = np.where(within >= floors,
                                within / np.maximum(pooled, 1e-9), 0)
        self.pose_weights = weights[-9:].copy()
        # This is a supported-pose guard, not a confidence/accuracy score.
        margins = np.array([.06] * 3 + [.03] * 3 + [.025, .025, .10])
        self.pose_low = np.quantile(pose, .01, axis=0) - margins
        self.pose_high = np.quantile(pose, .99, axis=0) + margins
        self.estimator.train(features, targets, variable_scaling=weights)

    def predict(self, features):
        features = np.asarray(features, dtype=float)
        pose = features[:, -9:]
        supported = ((pose >= self.pose_low) & (pose <= self.pose_high)).all(axis=1)
        self.pose_in_range = bool(supported.all())
        result = np.asarray(self.estimator.predict(features), dtype=float)
        result[~supported] = np.nan
        return result

    def close(self):
        self.estimator.close()
