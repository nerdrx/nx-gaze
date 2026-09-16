"""Run candidate lifecycle through the real worker, without camera/inference."""
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PyQt6.QtCore import Qt

from backend import CameraWorker


class WorkerMotionTests(unittest.TestCase):
    def test_candidate_comparison_retention_and_decisions(self):
        for accept, offset in ((True, 2.), (False, 100.)):
            with self.subTest(accept=accept):
                class Capture:
                    released = False
                    def isOpened(self): return True
                    def set(self, *args): pass
                    def read(self): return True, np.zeros((2, 2, 3), dtype=np.uint8)
                    def release(self): self.released = True

                class Estimator:
                    frame = 0
                    offset = 0.
                    closed = False
                    pose_in_range = True
                    pose_weights = np.ones(9)
                    comparisons = []
                    decisions = []
                    def train(self, features, targets): self.offset = 0.
                    def begin_candidate(self, features, targets): self.offset = offset
                    def extract_features(self, image):
                        self.frame += 1
                        return np.array([self.frame, 1.]), False
                    def predict(self, features):
                        return np.asarray(features, dtype=float) + self.offset
                    def comparison(self, features):
                        values = np.asarray(features, dtype=float)
                        self.comparisons.append(values.copy())
                        return values.copy(), values + self.offset
                    def finish_candidate(self, accepted):
                        self.decisions.append(accepted)
                        if not accepted: self.offset = 0.
                    def close(self): self.closed = True

                capture, estimator = Capture(), Estimator()
                cv2 = SimpleNamespace(VideoCapture=lambda *args: capture,
                    CAP_V4L2=200, CAP_ANY=0, CAP_PROP_FRAME_WIDTH=3,
                    CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_FPS=5, CAP_PROP_BUFFERSIZE=38)
                worker = CameraWorker(0)
                samples = [[0., 0.], [1., 1.], [2., 2.]]
                targets = [[0., 0.]] * 3
                baseline_id = worker.request_train(samples, targets)
                trained, retained, pairs, finished, errors = [], [], [], [], []
                direct = Qt.ConnectionType.DirectConnection

                def on_trained(training_id):
                    trained.append(training_id)
                    if training_id == baseline_id:
                        worker.request_train(samples, targets, candidate=True,
                                             baseline=(samples, targets))

                def on_comparison(before, after, supported):
                    pairs.append((before, after, supported, estimator.frame))
                    worker.finish_candidate(accept)

                def on_finished(accepted):
                    finished.append(accepted)
                    worker.requestInterruption()

                worker.trained.connect(on_trained, direct)
                worker.comparison.connect(on_comparison, direct)
                worker.candidate_retention.connect(retained.append, direct)
                worker.candidate_finished.connect(on_finished, direct)
                worker.failed.connect(errors.append, direct)
                with patch.dict(sys.modules, {'cv2': cv2,
                        'head_tracking': SimpleNamespace(HeadAwareEstimator=lambda: estimator)}):
                    worker.start()
                    stopped = worker.wait(5000)
                    if not stopped:
                        worker.requestInterruption()
                        worker.wait(5000)
                    self.assertTrue(stopped)
                self.assertEqual(errors, [])
                self.assertEqual(len(trained), 2)
                self.assertEqual(retained, [accept])
                self.assertEqual(finished, [accept])
                self.assertEqual(estimator.decisions, [accept])
                self.assertEqual(estimator.offset, offset if accept else 0.)
                self.assertEqual(len(pairs), 1)
                before, after, supported, frame = pairs[0]
                np.testing.assert_array_equal(before, [frame, 1.])
                np.testing.assert_array_equal(after, np.asarray(before) + offset)
                self.assertTrue(supported)
                np.testing.assert_array_equal(estimator.comparisons[0], samples)
                np.testing.assert_array_equal(estimator.comparisons[1], [[frame, 1.]])
                self.assertTrue(capture.released and estimator.closed)


if __name__ == '__main__':
    unittest.main()
