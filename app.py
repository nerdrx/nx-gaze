#!/usr/bin/env python3
"""NX Gaze: local webcam tracking across the desktop."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

# Global positioning and click-through overlays use X11/XWayland deliberately.
if sys.platform.startswith('linux') and os.environ.get('DISPLAY'):
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QRadialGradient, QShortcut, QKeySequence, QPolygonF
from PyQt6.QtWidgets import QApplication, QWidget
from core import Display, Smoother, signature, targets, visible_display
from ui import ControlPanel
from head_tracking import FEATURE_SCHEMA

SETTLE_SECONDS = .65
CAPTURE_SECONDS = .55
MIN_TARGET_SAMPLES = 6


class GazeOverlay(QWidget):
    def __init__(self, screen):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint |
                         Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowTransparentForInput |
                         Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setGeometry(screen.geometry())
        self.point, self.radius = None, 30
        self.color, self.shape, self.opacity = '#7700ff', 'ring', 80

    def paintEvent(self, event):
        if self.point is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        point = QPointF(self.point[0] - self.x(), self.point[1] - self.y())
        painter.setOpacity(self.opacity / 100)
        color = QColor(self.color)
        if self.shape in ('ring', 'dot'):
            gradient = QRadialGradient(point, self.radius)
            core, edge = QColor(color), QColor(color)
            core.setAlpha(90)
            edge.setAlpha(0)
            gradient.setColorAt(0, core)
            gradient.setColorAt(1, edge)
            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(point, self.radius, self.radius)
        # Keep opacity a cap rather than accumulating glow and marker alpha.
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(color, max(1.5, min(4, self.radius / 12))))
        r = self.radius * .62
        if self.shape == 'ring':
            painter.drawEllipse(point, r, r)
        elif self.shape == 'dot':
            painter.setBrush(color)
            painter.drawEllipse(point, r, r)
        elif self.shape == 'crosshair':
            gap = r * .25
            painter.drawLine(QPointF(point.x()-r, point.y()), QPointF(point.x()-gap, point.y()))
            painter.drawLine(QPointF(point.x()+gap, point.y()), QPointF(point.x()+r, point.y()))
            painter.drawLine(QPointF(point.x(), point.y()-r), QPointF(point.x(), point.y()-gap))
            painter.drawLine(QPointF(point.x(), point.y()+gap), QPointF(point.x(), point.y()+r))
        elif self.shape == 'diamond':
            painter.drawPolygon(QPolygonF([QPointF(point.x(), point.y()-r),
                QPointF(point.x()+r, point.y()), QPointF(point.x(), point.y()+r),
                QPointF(point.x()-r, point.y())]))
        if self.shape != 'dot':
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(point, min(3, self.radius*.2), min(3, self.radius*.2))



class CalibrationWindow(QWidget):
    def __init__(self, screen, cancel):
        super().__init__(None, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint |
                         Qt.WindowType.WindowStaysOnTopHint)
        self.setGeometry(screen.geometry())
        self.target = None
        self.title = 'Calibration'
        self.subtitle = ''
        self.cancel = cancel
        self.progress = 0
        QShortcut(QKeySequence('Escape'), self, activated=cancel)

    def closeEvent(self, event):
        event.ignore()
        self.cancel()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor('#fafafc'))
        painter.setPen(QColor('#1a1823'))
        painter.setFont(QFont('Sans Serif', 20, QFont.Weight.DemiBold))
        painter.drawText(QRectF(32, 24, self.width() - 64, 40), Qt.AlignmentFlag.AlignCenter, self.title)
        painter.setPen(QColor('#6c687e'))
        painter.setFont(QFont('Sans Serif', 12))
        painter.drawText(QRectF(32, self.height() - 72, self.width() - 64, 50),
                         Qt.AlignmentFlag.AlignCenter, self.subtitle + '  ·  Esc to cancel')
        if self.target:
            point = QPointF(self.target[0] - self.x(), self.target[1] - self.y())
            painter.setBrush(QColor('#f4edff'))
            painter.setPen(QPen(QColor('#7700ff'), 2))
            painter.drawEllipse(point, 26, 26)
            painter.setPen(QPen(QColor('#7700ff'), 4))
            painter.drawArc(QRectF(point.x()-32, point.y()-32, 64, 64), 90*16, -int(360*16*self.progress))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor('#7700ff'))
            painter.drawEllipse(point, 5, 5)


class Controller:
    def __init__(self, app):
        self.app = app
        self.panel = ControlPanel()
        self.worker = None
        self.camera_index = self.panel.camera_combo.currentData()
        self.camera_ready = False
        self.waiting_training = False
        self.preview = False
        self.calibrated = False
        self.panel.set_calibrated(False)
        self.paused = False
        self.panel.pause_button.setText('Pause overlay')
        self.closing = False
        self.radius = self.panel.radius_slider.value()
        self.smoothing = self.panel.smoothing_slider.value() / 100
        self.color, self.shape, self.opacity = self.panel.current_style()
        self.smoother = Smoother()
        self.last_sample = 0
        self.blink_since = None
        self.blink_hold = False
        self.cal_windows = []
        self.cal_targets = []
        self.samples, self.labels = [], []
        self.pending_save = None
        self.phase = None
        self.last_prediction = None
        self.validation_errors = []
        self.state_dir = Path(os.environ.get('XDG_STATE_HOME', Path.home()/'.local/state')) / 'nx-gaze'
        self.profile = self.state_dir / 'calibration.npz'
        self.save_error = ''
        self.overlays = []
        self.refresh_displays()
        self.panel.start_requested.connect(self.toggle_camera)
        self.panel.calibrate_requested.connect(self.calibrate)
        self.panel.pause_requested.connect(self.toggle_pause)
        self.panel.preview_requested.connect(self.set_preview)
        self.panel.camera_changed.connect(self.change_camera)
        self.panel.appearance_changed.connect(self.appearance)
        self.panel.style_changed.connect(self.style)
        QShortcut(QKeySequence('Escape'), self.panel, activated=self.cancel_calibration)
        app.screenAdded.connect(self.layout_changed)
        app.screenRemoved.connect(self.layout_changed)
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(50)
        self.panel.closeEvent = self.close_event
        self.panel.show()

    def refresh_displays(self):
        for overlay in self.overlays:
            overlay.close()
        screens = self.app.screens()
        self.displays = [Display(s.name(), s.geometry().x(), s.geometry().y(),
                                 s.geometry().width(), s.geometry().height()) for s in screens]
        self.overlays = [GazeOverlay(s) for s in screens]
        for screen in screens:
            try:
                screen.geometryChanged.disconnect(self.layout_changed)
            except TypeError:
                pass
            screen.geometryChanged.connect(self.layout_changed)
        self.panel.set_displays(signature(self.displays))

    def layout_changed(self, *args):
        self.cancel_calibration()
        self.calibrated = False
        self.panel.set_calibrated(False)
        self.refresh_displays()
        self.panel.set_status('Display layout changed', 'Calibrate again to match your screens.')

    def appearance(self, radius, smoothing):
        self.radius, self.smoothing = radius, smoothing
        for overlay in self.overlays:
            overlay.radius = radius
            overlay.update()

    def style(self, color, shape, opacity):
        self.color, self.shape, self.opacity = color, shape, opacity
        for overlay in self.overlays:
            overlay.color, overlay.shape, overlay.opacity = color, shape, opacity
            overlay.update()

    def set_preview(self, enabled):
        self.preview = enabled
        if self.worker:
            self.worker.preview_enabled = enabled

    def change_camera(self, index):
        if self.worker:
            self.panel.set_status('Stop the camera first', 'Then select a different input.')
            return
        self.camera_index = index
        self.calibrated = False
        self.panel.set_calibrated(False)

    def toggle_camera(self):
        if self.worker:
            self.stop_camera()
            return
        from backend import CameraWorker
        self.panel.set_status('Starting camera', 'First use downloads a face model. Frames stay on this computer.')
        self.camera_ready = False
        self.worker = CameraWorker(self.camera_index)
        self.worker.preview_enabled = self.preview
        self.worker.sample.connect(self.sample)
        self.worker.prediction.connect(self.prediction)
        self.worker.ready.connect(self.ready)
        self.worker.trained.connect(self.trained)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.stopped)
        self.panel.set_running(True)
        self.worker.start()

    def ready(self):
        if not self.worker or self.worker.isInterruptionRequested() or self.closing:
            return
        self.camera_ready = True
        self.panel.set_status('Camera ready', 'Quick calibration takes about 7 seconds per screen. Sit naturally.')
        if self.profile.exists():
            try:
                import numpy as np
                with np.load(self.profile, allow_pickle=False) as data:
                    meta = json.loads(str(data['metadata']))
                    if meta != self.metadata():
                        self.panel.set_status('Quick calibration needed', 'Head compensation needs a fresh calibration. About 7 seconds per screen.')
                        return
                    x, y = data['features'].copy(), data['targets'].copy()
                    if x.ndim != 2 or y.shape != (len(x), 2) or len(x) < 9 or not np.isfinite(x).all() or not np.isfinite(y).all():
                        raise ValueError('Invalid calibration data')
                self.waiting_training = self.worker.request_train(x, y)
                self.panel.set_status('Restoring calibration', 'Use recalibration if your camera or seating position moved.')
            except Exception:
                self.panel.set_status('Calibration needs replacing', 'Run a fresh calibration for these screens.')

    def metadata(self):
        device = Path(f'/sys/class/video4linux/video{self.camera_index}/device')
        return {'version': 2, 'features': FEATURE_SCHEMA, 'eyetrax': '0.4.0', 'camera': self.camera_index,
                'device': str(device.resolve()), 'displays': signature(self.displays)}

    def stop_camera(self):
        self.camera_ready = False
        self.cancel_calibration()
        self.hide_gaze()
        self.calibrated = False
        self.panel.set_calibrated(False)
        if self.worker:
            self.worker.requestInterruption()
            self.panel.set_status('Stopping camera', 'Releasing the webcam…')

    def stopped(self):
        self.worker = None
        self.camera_ready = False
        self.panel.set_running(False)
        self.panel.set_calibrated(False)
        if self.panel.status_label.text() == 'Stopping camera':
            self.panel.set_status('Camera stopped', 'Webcam released. Start again when you’re ready.')
        self.hide_gaze()
        if self.closing:
            self.panel.close()

    def failed(self, message):
        self.cancel_calibration()
        self.calibrated = False
        self.panel.set_calibrated(False)
        self.hide_gaze()
        self.panel.set_status('Camera unavailable', message)

    def toggle_pause(self):
        self.paused = not self.paused
        self.panel.pause_button.setText('Resume overlay' if self.paused else 'Pause overlay')
        self.hide_gaze()
        self.panel.set_status('Overlay paused' if self.paused else 'Overlay resumed',
                              'Camera remains active.' if self.paused else 'Look around your calibrated screens.')

    def hide_gaze(self):
        self.smoother.reset()
        self.panel.set_gaze(None, None)
        for overlay in self.overlays:
            overlay.point = None
            overlay.hide()

    def sample(self, features, preview, blink):
        self.last_sample = time.monotonic()
        if preview is not None:
            self.panel.set_preview(preview)
        if features is None:
            self.blink_since = None
            self.blink_hold = False
            self.hide_gaze()
            return
        if blink:
            if self.blink_since is None:
                self.blink_since = self.last_sample
            self.blink_hold = self.last_sample - self.blink_since < .4
            if not self.blink_hold:
                self.hide_gaze()
            elif self.smoother.point is not None:
                self.smoother.time = self.last_sample
            return
        self.blink_since = None
        self.blink_hold = False
        if self.phase and self.cal_targets and time.monotonic() - self.target_since >= SETTLE_SECONDS:
            if self.phase == 'collect':
                self.samples.append(features.copy())
                self.labels.append(self.cal_targets[self.target_index][1:])
                self.target_samples += 1

    def prediction(self, point):
        self.last_prediction = point
        if self.phase == 'validate' and point is not None and time.monotonic() - self.target_since >= SETTLE_SECONDS:
            import math
            self.target_errors.append(math.dist(self.cal_targets[self.target_index][1:], point))
        if point is None and self.blink_hold and self.calibrated and not self.paused and not self.phase:
            return
        if not self.calibrated or self.paused or self.phase or point is None:
            self.hide_gaze()
            return
        screen = visible_display(self.displays, point)
        if screen is None:
            self.hide_gaze()
            return
        point = self.smoother.update(point, screen, time.monotonic(), self.smoothing)
        for index, overlay in enumerate(self.overlays):
            if index == screen:
                overlay.point, overlay.radius = point, self.radius
                overlay.color, overlay.shape, overlay.opacity = self.color, self.shape, self.opacity
                overlay.show()
                overlay.update()
            else:
                overlay.hide()
        self.panel.set_gaze(*point)

    def calibrate(self):
        if not self.worker or not self.camera_ready or self.worker.isInterruptionRequested():
            self.panel.set_status('Start the camera first', 'Then calibrate all screens.')
            return
        if self.phase or self.waiting_training:
            return
        self.calibrated = False
        self.panel.set_calibrated(False)
        self.paused = False
        self.panel.pause_button.setText('Pause overlay')
        self.hide_gaze()
        self.samples, self.labels = [], []
        self.cal_targets = targets(self.displays)
        self.cal_windows = [CalibrationWindow(s, self.cancel_calibration) for s in self.app.screens()]
        self.phase = 'collect'
        self.target_index = 0
        self.show_target()

    def show_target(self):
        self.target_since = time.monotonic()
        self.target_samples = 0
        self.target_errors = []
        active, x, y = self.cal_targets[self.target_index]
        for i, window in enumerate(self.cal_windows):
            window.target = (x, y) if i == active else None
            window.title = ('Look at the violet dot' if i == active else f'Look at display {active + 1}')
            window.subtitle = f'{"Check" if self.phase == "validate" else "Point"} {self.target_index+1} of {len(self.cal_targets)} · Keep your eyes on the dot; sit naturally'
            window.show()
            window.update()
        self.cal_windows[active].activateWindow()

    def tick(self):
        now = time.monotonic()
        if now - self.last_sample > .35:
            self.hide_gaze()
        if not self.phase or self.phase == 'training':
            return
        elapsed = now - self.target_since
        count = self.target_samples if self.phase == 'collect' else len(self.target_errors)
        for window in self.cal_windows:
            window.progress = min(1, max(0, elapsed - SETTLE_SECONDS) / CAPTURE_SECONDS)
            window.update()
        if elapsed > 12:
            self.cancel_calibration()
            self.panel.set_status('Could not track this target', 'Improve lighting or camera angle, then try again.')
            return
        if elapsed < SETTLE_SECONDS + CAPTURE_SECONDS or count < MIN_TARGET_SAMPLES:
            return
        if self.phase == 'validate':
            import statistics
            self.validation_errors.append((self.cal_targets[self.target_index][0], statistics.median(self.target_errors)))
        self.target_index += 1
        if self.target_index < len(self.cal_targets):
            self.show_target()
        elif self.phase == 'collect':
            self.phase = 'training'
            for window in self.cal_windows:
                window.hide()
            self.pending_save = (self.samples.copy(), self.labels.copy())
            self.waiting_training = self.worker.request_train(self.samples, self.labels)
            self.panel.set_status('Learning your gaze', 'Training a local mapping for all screens.')
        else:
            import statistics
            results = [f'Display {i+1}: {statistics.median(e for s,e in self.validation_errors if s == i):.0f} px'
                       for i in range(len(self.displays))]
            self.save_calibration()
            self.cancel_calibration()
            self.calibrated = True
            self.panel.set_calibrated(True)
            self.panel.set_status('Calibration checked', 'Quick spot-check error (logical pixels) · ' + ' / '.join(results) + self.save_error)

    def save_calibration(self):
        self.save_error = ''
        if self.pending_save:
            import numpy as np
            features, labels = self.pending_save
            self.pending_save = None
            try:
                self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                temp = self.profile.with_suffix('.tmp')
                with temp.open('wb') as file:
                    os.chmod(temp, 0o600)
                    np.savez_compressed(file, features=features, targets=labels, metadata=json.dumps(self.metadata()))
                temp.replace(self.profile)
            except OSError as exc:
                self.save_error = ' · Could not save calibration: ' + str(exc)

    def trained(self, training_id):
        if training_id != self.waiting_training or not self.waiting_training or not self.worker or self.worker.isInterruptionRequested():
            return
        self.waiting_training = False
        if self.phase == 'training':
            self.phase = 'validate'
            self.validation_errors = []
            self.cal_targets = [(i, *d.target(u, v)) for i, d in enumerate(self.displays)
                                for u, v in [(.3, .7)]]
            self.target_index = 0
            self.show_target()
        else:
            self.calibrated = True
            self.panel.set_calibrated(True)
            self.panel.set_status('Tracking ready', 'Saved calibration loaded. Recalibrate after moving your camera.')

    def cancel_calibration(self):
        if self.phase:
            self.calibrated = False
            self.panel.set_calibrated(False)
            self.panel.set_status('Calibration cancelled', 'Run calibration again when you’re ready.')
        self.waiting_training = False
        self.phase = None
        self.cal_targets = []
        self.pending_save = None
        for window in self.cal_windows:
            window.hide()
            window.deleteLater()
        self.cal_windows = []

    def close_event(self, event):
        if self.worker and self.worker.isRunning():
            event.ignore()
            self.closing = True
            self.stop_camera()
        else:
            self.cancel_calibration()
            self.hide_gaze()
            event.accept()
            self.app.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', metavar='PNG', help='Save camera-free UI screenshot and exit')
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setApplicationName('NX Gaze')
    app.setOrganizationName('NX')
    controller = Controller(app)
    if args.smoke:
        def capture():
            controller.panel.grab().save(args.smoke)
            controller.panel.close()
        QTimer.singleShot(800, capture)
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
