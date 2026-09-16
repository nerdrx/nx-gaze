from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest
import numpy as np
from head_tracking import HeadAwareEstimator, compensate_features


def face(dx=0, dy=0, scale=1):
    points = [SimpleNamespace(x=.5, y=.5, z=0) for _ in range(478)]
    for index, offset in [(33, -.1), (263, .1)]:
        points[index] = SimpleNamespace(x=.5+dx+offset*scale, y=.5+dy, z=0)
    return points


class HeadTrackingTests(unittest.TestCase):
    def test_translation_and_scale_are_preserved(self):
        base = np.array([.1, .2, 0, 0, np.pi])
        normal = compensate_features(base, face(), .75)
        shifted = compensate_features(base, face(.08, -.04, 1.2), .75)
        np.testing.assert_allclose(normal[:-3], shifted[:-3])
        np.testing.assert_allclose(shifted[-3:] - normal[-3:], [.08, -.04, np.log(1.2)])

    def test_orientation_has_no_pi_wrap_discontinuity(self):
        before = compensate_features(np.array([.1, 0, 0, np.pi-1e-6]), face(), .75)
        after = compensate_features(np.array([.1, 0, 0, -np.pi+1e-6]), face(), .75)
        self.assertLess(np.linalg.norm(before-after), 3e-6)

    def test_reject_missing_or_degenerate_geometry(self):
        self.assertIsNone(compensate_features(None, face(), .75))
        self.assertIsNone(compensate_features([1, 2, 3], None, .75))
        self.assertIsNone(compensate_features([1, 2, 3], face(scale=0), .75))
        self.assertIsNone(compensate_features([1, 2, 3], face(dx=float('nan')), .75))

    def model_without_camera(self):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        class Model:
            def train(self, x, y, variable_scaling):
                self.scaler = StandardScaler()
                self.weights = variable_scaling
                self.model = Ridge(alpha=1).fit(self.scaler.fit_transform(x)*self.weights, y)
            def predict(self, x):
                return self.model.predict(self.scaler.transform(x)*self.weights)
        estimator = HeadAwareEstimator.__new__(HeadAwareEstimator)
        estimator.estimator = Model()
        return estimator

    def test_target_correlated_head_pose_cannot_steer_gaze(self):
        estimator = self.model_without_camera()
        target = np.repeat([-1., 1.], 30)
        x = np.zeros((60, 10))
        x[:, 0] = target
        x[:, -9:] = target[:, None] * .04
        y = np.column_stack([target*100, target*40])
        estimator.train(x, y)
        np.testing.assert_array_equal(estimator.pose_weights, np.zeros(9))
        normal = x[0].copy()
        shifted = normal.copy()
        shifted[-9:] += .02
        np.testing.assert_allclose(estimator.predict([normal]), estimator.predict([shifted]))
        shifted[-9:] += 1
        self.assertTrue(np.isnan(estimator.predict([shifted])).all())
        self.assertFalse(estimator.pose_in_range)

    def test_within_target_motion_can_enable_supported_features(self):
        estimator = self.model_without_camera()
        target = np.repeat([-1., 1.], 60)
        head = np.tile(np.linspace(-.1, .1, 60), 2)
        x = np.zeros((120, 10))
        x[:, 0] = target - head
        x[:, -3] = head
        estimator.train(x, np.column_stack([target*100, target*40]))
        self.assertGreater(estimator.pose_weights[-3], .9)
        self.assertTrue(np.isfinite(estimator.predict(x)).all())

    def test_single_inference_and_no_stale_pose(self):
        class Detector:
            calls = 0
            closed = False
            def detect_for_video(self, image, timestamp):
                self.calls += 1
                return SimpleNamespace(face_landmarks=[face()] if self.calls == 1 else [])
            def close(self):
                self.closed = True
        class Estimator:
            def __init__(self):
                self._face_landmarker = Detector()
            def extract_features(self, frame):
                result = self._face_landmarker.detect_for_video(frame, 1)
                return (np.array([.1, 0, 0, np.pi]) if result.face_landmarks else None), False
            def train(self, x, y, variable_scaling):
                self.weights = variable_scaling
            def close(self):
                self._face_landmarker.close()
        with patch.dict(sys.modules, {'eyetrax': SimpleNamespace(GazeEstimator=Estimator)}):
            estimator = HeadAwareEstimator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        features, blink = estimator.extract_features(frame)
        self.assertIsNotNone(features)
        self.assertEqual(estimator.tap.calls, 1)
        self.assertIsNone(estimator.extract_features(frame)[0])
        self.assertIsNone(estimator.tap.landmarks)
        self.assertEqual(estimator.tap.calls, 2)
        estimator.train(np.tile(features, (10, 1)), np.zeros((10, 2)))
        self.assertTrue((estimator.estimator.weights[-9:] < 1e-10).all())
        estimator.close()
        self.assertTrue(estimator.tap.closed)


if __name__ == '__main__':
    unittest.main()
