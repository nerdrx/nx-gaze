import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from PyQt6.QtWidgets import QApplication
from core import Display, Smoother, targets, visible_display
from app import Controller


class GeometryTests(unittest.TestCase):
    def test_negative_origins_and_gaps(self):
        displays = [Display('left', -1920, 0, 1920, 1080), Display('top', 0, -1440, 2560, 1440)]
        self.assertEqual(visible_display(displays, (-1, 20)), 0)
        self.assertEqual(visible_display(displays, (0, -1)), 1)
        for point in [(0, 0), (2560, -10), (float('nan'), 0), None]:
            self.assertIsNone(visible_display(displays, point))
        points = targets(displays)
        self.assertEqual(len(points), 18)
        for index, x, y in points:
            self.assertTrue(displays[index].contains(x, y))

    def test_smoothing_resets_on_screen_crossing_and_loss(self):
        smoother = Smoother()
        self.assertEqual(smoother.update((-100, 20), 0, 0, .6), (-100, 20))
        self.assertEqual(smoother.update((100, 20), 1, .02, .6), (100, 20))
        self.assertEqual(smoother.update((200, 20), 1, 1, .6), (200, 20))
        self.assertEqual(smoother.update((210, 20), 1, 1.02, 0), (210, 20))


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.c = Controller(self.app)
        self.c.timer.stop()
        self.tmp = tempfile.TemporaryDirectory()
        self.c.state_dir = Path(self.tmp.name)
        self.c.profile = self.c.state_dir / 'calibration.npz'
        self.worker = Mock()
        self.worker.isRunning.return_value = True
        self.worker.isInterruptionRequested.return_value = False
        self.worker.request_train.return_value = 42
        self.c.worker = self.worker
        self.c.camera_ready = True

    def tearDown(self):
        self.c.worker = None
        self.c.panel.close()
        self.tmp.cleanup()

    def test_calibration_cancel_and_stale_completion(self):
        self.c.calibrated = True
        self.c.panel.set_calibrated(True)
        self.c.calibrate()
        self.assertFalse(self.c.calibrated)
        self.assertFalse(self.c.panel._calibrated)
        self.c.waiting_training = 42
        self.c.cancel_calibration()
        self.c.trained(42)
        self.assertFalse(self.c.calibrated)
        self.assertFalse(self.c.profile.exists())
        self.c.waiting_training = 43
        self.c.trained(42)
        self.assertFalse(self.c.calibrated)

    def test_validation_uses_current_prediction(self):
        self.c.phase = 'validate'
        self.c.cal_targets = [(0, 100, 100)]
        self.c.target_index = 0
        self.c.target_since = time.monotonic() - 2
        self.c.target_errors = []
        self.c.last_prediction = (500, 500)
        self.c.prediction((103, 104))
        self.assertEqual(self.c.target_errors, [5])
        self.c.prediction(None)
        self.assertEqual(self.c.target_errors, [5])

    def test_train_waits_for_validation_before_save(self):
        self.c.calibrate()
        self.c.phase = 'training'
        self.c.waiting_training = 42
        self.c.pending_save = ([[1, 2]] * 18, [[3, 4]] * 18)
        self.c.trained(42)
        self.assertEqual(self.c.phase, 'validate')
        self.assertFalse(self.c.calibrated)
        self.assertFalse(self.c.profile.exists())
        self.c.save_calibration()
        with np.load(self.c.profile, allow_pickle=False) as saved:
            self.assertEqual(saved['features'].shape, (18, 2))
        self.assertEqual(self.c.profile.stat().st_mode & 0o777, 0o600)

    def test_stop_ignores_queued_ready_and_train(self):
        self.worker.isInterruptionRequested.return_value = True
        self.c.waiting_training = 42
        self.c.ready()
        self.c.trained(42)
        self.assertFalse(self.c.calibrated)
        self.worker.request_train.assert_not_called()

    def test_layout_change_invalidates(self):
        self.c.calibrated = True
        self.c.layout_changed()
        self.assertFalse(self.c.calibrated)
        self.assertFalse(self.c.panel._calibrated)


if __name__ == '__main__':
    unittest.main()
