import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from PyQt6.QtWidgets import QApplication
from core import BlinkFilter, Display, Smoother, targets, visible_display
from app import Controller


class GeometryTests(unittest.TestCase):
    def test_negative_origins_and_gaps(self):
        displays = [Display('left', -1920, 0, 1920, 1080), Display('top', 0, -1440, 2560, 1440)]
        self.assertEqual(visible_display(displays, (-1, 20)), 0)
        self.assertEqual(visible_display(displays, (0, -1)), 1)
        for point in [(0, 0), (2560, -10), (float('nan'), 0), None]:
            self.assertIsNone(visible_display(displays, point))
        points = targets(displays)
        self.assertEqual(len(points), 10)
        for index, x, y in points:
            self.assertTrue(displays[index].contains(x, y))

    def test_smoothing_resets_on_screen_crossing_and_loss(self):
        smoother = Smoother()
        self.assertEqual(smoother.update((-100, 20), 0, 0, .6), (-100, 20))
        self.assertEqual(smoother.update((100, 20), 1, .02, .6), (100, 20))
        self.assertEqual(smoother.update((200, 20), 1, 1, .6), (200, 20))
        self.assertEqual(smoother.update((210, 20), 1, 1.02, 0), (210, 20))


class BlinkTests(unittest.TestCase):
    def test_reopening_requires_stable_frames(self):
        gate = BlinkFilter()
        self.assertFalse(gate.update(True, False, 0))
        self.assertTrue(gate.update(True, True, .1))
        self.assertTrue(gate.update(True, False, .2))
        self.assertTrue(gate.update(True, False, .25))
        self.assertFalse(gate.update(True, False, .31))
        self.assertTrue(gate.update(True, True, .4))
        self.assertFalse(gate.update(False, False, .5))
        self.assertFalse(gate.update(True, False, .6))

    def test_max_smoothing_is_slower(self):
        fast, slow = Smoother(), Smoother()
        fast.update((0, 0), 0, 0, .1)
        slow.update((0, 0), 0, 0, 1)
        self.assertGreater(fast.update((100, 0), 0, .1, .1)[0], 90)
        self.assertLess(slow.update((100, 0), 0, .1, 1)[0], 6)


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

    def test_short_blink_holds_but_lost_face_hides(self):
        self.c.calibrated = True
        point = self.c.displays[0].target(.5, .5)
        self.c.prediction(point)
        self.c.sample(np.array([1]), None, True)
        self.c.prediction(None)
        self.assertEqual(self.c.overlays[0].point, point)
        self.assertEqual(self.c.smoother.point, point)
        self.c.sample(None, None, False)
        self.c.prediction(None)
        self.assertIsNone(self.c.overlays[0].point)
        self.assertIsNone(self.c.smoother.point)

    def test_long_blink_and_stale_profile_are_rejected(self):
        self.c.calibrated = True
        self.c.prediction(self.c.displays[0].target(.5, .5))
        self.c.blink_since = time.monotonic() - .5
        self.c.sample(np.array([1]), None, True)
        self.c.prediction(None)
        self.assertIsNone(self.c.overlays[0].point)
        import json
        metadata = self.c.metadata()
        metadata['version'] = 1
        np.savez(self.c.profile, metadata=json.dumps(metadata), features=np.ones((10, 3)), targets=np.zeros((10, 2)))
        self.c.ready()
        self.worker.request_train.assert_not_called()
        self.assertEqual(self.c.panel.status_label.text(), 'Quick calibration needed')

    def test_marker_style_radius_and_opacity_render(self):
        from PyQt6.QtGui import QImage
        overlay = self.c.overlays[0]
        overlay.setGeometry(0, 0, 120, 120)
        overlay.point = (60, 60)
        overlay.radius = 30
        overlay.color = '#ff2200'
        overlay.opacity = 20
        rendered = []
        for shape in ('ring', 'dot', 'crosshair', 'diamond'):
            overlay.shape = shape
            image = QImage(120, 120, QImage.Format.Format_ARGB32)
            image.fill(0)
            overlay.render(image)
            bits = image.bits()
            bits.setsize(image.sizeInBytes())
            rendered.append(bytes(bits))
            self.assertGreater(image.pixelColor(60, 60).red(), 200)
            self.assertLessEqual(image.pixelColor(60, 60).alpha(), 60)
        self.assertEqual(len(set(rendered)), 4)

    def test_quick_calibration_nominal_duration(self):
        from app import SETTLE_SECONDS, CAPTURE_SECONDS
        # Five learned points + one held-out point per display.
        self.assertLessEqual(2 * 6 * (SETTLE_SECONDS + CAPTURE_SECONDS), 15)

    def test_layout_change_invalidates(self):
        self.c.calibrated = True
        self.c.layout_changed()
        self.assertFalse(self.c.calibrated)
        self.assertFalse(self.c.panel._calibrated)


if __name__ == '__main__':
    unittest.main()
