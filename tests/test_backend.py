"""Exercise the actual worker thread without opening a camera or loading AI."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PyQt6.QtCore import Qt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import CameraWorker


class CameraWorkerTest(unittest.TestCase):
    def test_training_frames_and_cooperative_cleanup(self):
        class Capture:
            released = False
            settings = {}

            def isOpened(self):
                return True

            def set(self, key, value):
                self.settings[key] = value

            def read(self):
                return True, np.zeros((2, 2, 3), dtype=np.uint8)

            def release(self):
                self.released = True

        class Estimator:
            frame = 0
            closed = False
            training = None
            predicted_features = []

            def train(self, features, targets):
                self.training = (features.copy(), targets.copy())

            def extract_features(self, image):
                self.frame += 1
                if self.frame == 2:
                    return None, False
                return np.array([self.frame, 1.0]), self.frame == 3

            def predict(self, features):
                self.predicted_features.append(features[0].copy())
                return [[-100 * features[0][0], 200]]

            def close(self):
                self.closed = True

        capture, estimator = Capture(), Estimator()
        opened = []

        def open_camera(*args):
            opened.append(args)
            return capture

        cv2 = SimpleNamespace(
            VideoCapture=open_camera, CAP_V4L2=200, CAP_ANY=0,
            CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
            CAP_PROP_FPS=5, CAP_PROP_BUFFERSIZE=38,
        )
        worker = CameraWorker(7)
        features = [[1, 2], [2, 3], [3, 4]]
        targets = [[-1920, 0], [0, 0], [1920, 0]]
        obsolete_id = worker.request_train(features, targets)
        training_id = worker.request_train(features, targets)
        # Mutating caller-owned lists must not change the queued snapshot.
        features[0][0] = 999
        targets[0][0] = 999
        events, errors, trained, ready = [], [], [], []
        direct = Qt.ConnectionType.DirectConnection

        def receive_sample(feature, preview, blink):
            events.append(('sample', None if feature is None else tuple(feature), blink))
            if estimator.frame >= 3:
                worker.requestInterruption()

        worker.sample.connect(receive_sample, direct)
        worker.prediction.connect(lambda point: events.append(('prediction', point)), direct)
        worker.failed.connect(errors.append, direct)
        worker.trained.connect(trained.append, direct)
        worker.ready.connect(lambda: ready.append(True), direct)
        with patch.dict(sys.modules, {
            'cv2': cv2, 'head_tracking': SimpleNamespace(HeadAwareEstimator=lambda: estimator),
        }):
            worker.start()
            finished = worker.wait(5000)
            if not finished:
                worker.requestInterruption()
                worker.wait(5000)
            self.assertTrue(finished, 'Worker did not stop cooperatively')
        self.assertEqual(errors, [])
        self.assertEqual(ready, [True])
        self.assertEqual(trained, [training_id])
        self.assertGreater(training_id, obsolete_id)
        self.assertEqual(estimator.training[0][0, 0], 1)
        self.assertEqual(estimator.training[1][0, 0], -1920)
        self.assertEqual(events, [
            ('sample', (1.0, 1.0), False), ('prediction', (-100.0, 200.0)),
            ('sample', None, False), ('prediction', None),
            ('sample', (3.0, 1.0), True), ('prediction', None),
            ('prediction', None),
        ])
        self.assertEqual(len(estimator.predicted_features), 1)
        self.assertEqual(opened, [(7, cv2.CAP_V4L2 if sys.platform.startswith('linux') else cv2.CAP_ANY)])
        self.assertEqual(capture.settings[cv2.CAP_PROP_FPS], 30)
        self.assertTrue(capture.released)
        self.assertTrue(estimator.closed)


if __name__ == '__main__':
    unittest.main()
