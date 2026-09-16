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
