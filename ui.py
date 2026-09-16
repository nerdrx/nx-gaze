"""Native NX Clear control surface. Camera ownership stays in the controller."""
from pathlib import Path
import re

from PyQt6.QtCore import Qt, QRectF, QPointF, QSettings, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPalette, QPen, QPixmap
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QComboBox, QSlider, QCheckBox, QSizePolicy, QGraphicsDropShadowEffect, QColorDialog,
)


class DesktopMap(QWidget):
    def __init__(self):
        super().__init__()
        self.displays = []
        self.gaze = None
        self.dark = False
        self.setMinimumHeight(140)
        self.setAccessibleName('Connected display arrangement and estimated gaze')

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        muted = QColor('#9c98ab' if self.dark else '#6c687e')
        if not self.displays:
            p.setPen(muted)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'No displays detected')
            return
        left = min(d['x'] for d in self.displays)
        top = min(d['y'] for d in self.displays)
        right = max(d['x'] + d['width'] for d in self.displays)
        bottom = max(d['y'] + d['height'] for d in self.displays)
        scale = min((self.width() - 48) / max(right-left, 1), (self.height()-46) / max(bottom-top, 1))
        ox = (self.width() - (right-left)*scale)/2
        oy = (self.height() - (bottom-top)*scale)/2
        for i, d in enumerate(self.displays):
            rect = QRectF(ox+(d['x']-left)*scale, oy+(d['y']-top)*scale,
                          d['width']*scale, d['height']*scale).adjusted(3, 3, -3, -3)
            p.setBrush(QColor('#17141f' if self.dark else '#faf9fd'))
            p.setPen(QPen(QColor('#51485f' if self.dark else '#ddd8ea'), 1.5))
            p.drawRoundedRect(rect, 10, 10)
            p.setPen(QColor('#f4f3f8' if self.dark else '#1a1823'))
            p.setFont(QFont(self.font().family(), 11, QFont.Weight.DemiBold))
            title = f"{i + 1}  ·  {d.get('name', 'Display')}"
            title = p.fontMetrics().elidedText(title, Qt.TextElideMode.ElideRight, max(10, int(rect.width()-14)))
            p.drawText(rect.adjusted(7, 0, -7, -12), Qt.AlignmentFlag.AlignCenter, title)
            if rect.height() > 58:
                p.setPen(muted)
                p.setFont(QFont(self.font().family(), 9))
                p.drawText(rect.adjusted(0, 28, 0, 0), Qt.AlignmentFlag.AlignCenter,
                           f"{d['width']} × {d['height']}")
        if self.gaze is not None:
            x, y = self.gaze
            center = QPointF(ox+(x-left)*scale, oy+(y-top)*scale)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(119, 0, 255, 28))
            p.drawEllipse(center, 17, 17)
            p.setBrush(QColor('#7700ff'))
            p.drawEllipse(center, 5, 5)


class ControlPanel(QWidget):
    start_requested = pyqtSignal()
    calibrate_requested = pyqtSignal()
    pause_requested = pyqtSignal()
    preview_requested = pyqtSignal(bool)
    camera_changed = pyqtSignal(int)
    appearance_changed = pyqtSignal(int, float)
    style_changed = pyqtSignal(str, str, int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('NX Gaze')
        self.resize(980, 880)
        self.setMinimumSize(840, 850)
        self.setObjectName('window')
        self._running = False
        self._calibrated = False
        self.settings = QSettings('NX', 'Gaze')
        self._color = str(self.settings.value('color', '#7700ff'))
        if not QColor(self._color).isValid():
            self._color = '#7700ff'
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 26, 32, 22)
        root.setSpacing(20)
        header = QHBoxLayout()
        self.mark = QSvgWidget()
        self.mark.setFixedSize(54, 30)
        header.addWidget(self.mark)
        product = QLabel('Gaze')
        product.setObjectName('product')
        header.addWidget(product)
        header.addStretch()
        self.theme = QCheckBox('Dark appearance')
        self.theme.toggled.connect(self._apply_theme)
        header.addWidget(self.theme)
        root.addLayout(header)
        hero = QVBoxLayout()
        hero.setSpacing(5)
        title = QLabel('Your attention, in view.')
        title.setObjectName('hero')
        hero.addWidget(title)
        subtitle = QLabel('A quiet gaze overlay. One webcam. Your whole desktop.')
        subtitle.setObjectName('muted')
        hero.addWidget(subtitle)
        root.addLayout(hero)
        columns = QHBoxLayout()
        columns.setSpacing(20)
        map_card, map_layout = self._card()
        map_card.setMinimumHeight(320)
        map_heading = QHBoxLayout()
        map_heading.addWidget(self._label('Your desktop', 'section'))
        map_heading.addStretch()
        self.display_count = self._label('Detecting displays', 'muted')
        map_heading.addWidget(self.display_count)
        map_layout.addLayout(map_heading)
        self.desktop_map = DesktopMap()
        map_layout.addWidget(self.desktop_map, 1)
        self.map_note = self._label('Five targets + one quick check. About 7 seconds per screen.', 'muted')
        self.map_note.setWordWrap(True)
        map_layout.addWidget(self.map_note)
        self.calibrate_button = QPushButton('Quick calibration')
        self.calibrate_button.clicked.connect(self.calibrate_requested.emit)
        map_layout.addWidget(self.calibrate_button)
        columns.addWidget(map_card, 3)
        camera_card, camera_layout = self._card()
        camera_card.setMinimumHeight(320)
        camera_layout.addWidget(self._label('Camera', 'section'))
        self.camera_combo = QComboBox()
        self.camera_combo.setAccessibleName('Webcam device')
        devices = sorted(Path('/dev').glob('video*'), key=lambda p: int(re.search(r'\d+$', p.name)[0]))
        for path in devices:
            index = int(re.search(r'\d+$', path.name)[0])
            name_file = Path('/sys/class/video4linux') / path.name / 'name'
            try:
                name = name_file.read_text().strip()
            except OSError:
                name = 'Camera'
            self.camera_combo.addItem(f'{name} · {path.name}', index)
        if not devices:
            self.camera_combo.addItem('No webcam detected', -1)
        self.camera_combo.currentIndexChanged.connect(lambda _: self.camera_changed.emit(self.camera_combo.currentData()))
        camera_layout.addWidget(self.camera_combo)
        self.preview_label = QLabel('Camera is off\nStart when you’re ready')
        self.preview_label.setObjectName('preview')
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(220, 100)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        camera_layout.addWidget(self.preview_label, 1)
        self.preview_toggle = QCheckBox('Show camera preview')
        self.preview_toggle.toggled.connect(self.preview_requested.emit)
        self.preview_toggle.toggled.connect(self._preview_visibility)
        camera_layout.addWidget(self.preview_toggle)
        self.start_button = QPushButton('Start camera')
        self.start_button.setObjectName('primary')
        self.start_button.clicked.connect(self.start_requested.emit)
        camera_layout.addWidget(self.start_button)
        columns.addWidget(camera_card, 2)
        root.addLayout(columns, 1)
        settings_card, settings = self._card()
        settings.addWidget(self._label('Make it feel right', 'section'))
        sliders = QHBoxLayout()
        sliders.setSpacing(32)
        self.radius_slider = self._slider(sliders, 'Bubble radius', 4, 160,
                                          self._saved_int('radius', 30, 4, 160), 'px')
        self.smoothing_slider = self._slider(sliders, 'Smoothing response', 0, 100,
                                             self._saved_int('smoothing', 35, 0, 100), 'ms')
        self.smoothing_slider.setToolTip('0 ms follows immediately. Higher values smooth more and react more slowly.')
        self.radius_slider.valueChanged.connect(self._appearance)
        self.smoothing_slider.valueChanged.connect(self._appearance)
        settings.addLayout(sliders)
        styles = QHBoxLayout()
        styles.setSpacing(24)
        color_layout = QVBoxLayout()
        color_layout.addWidget(QLabel('Colour'))
        self.color_button = QPushButton()
        self.color_button.setAccessibleName('Choose gaze bubble colour')
        self._update_color_button()
        self.color_button.clicked.connect(self._choose_color)
        color_layout.addWidget(self.color_button)
        styles.addLayout(color_layout, 1)
        shape_layout = QVBoxLayout()
        shape_layout.addWidget(QLabel('Shape'))
        self.shape_combo = QComboBox()
        self.shape_combo.setAccessibleName('Gaze bubble shape')
        for shape in ('ring', 'dot', 'crosshair', 'diamond'):
            self.shape_combo.addItem(shape.title(), shape)
        saved_shape = self.shape_combo.findData(str(self.settings.value('shape', 'ring')))
        self.shape_combo.setCurrentIndex(max(0, saved_shape))
        shape_layout.addWidget(self.shape_combo)
        styles.addLayout(shape_layout, 1)
        self.opacity_slider = self._slider(styles, 'Opacity', 10, 100,
                                           self._saved_int('opacity', 80, 10, 100), '%')
        self.opacity_slider.setToolTip('Lower opacity makes the overlay more transparent.')
        self.shape_combo.currentIndexChanged.connect(self._style)
        self.opacity_slider.valueChanged.connect(self._style)
        settings.addLayout(styles)
        root.addWidget(settings_card)
        status_row = QHBoxLayout()
        status_text = QVBoxLayout()
        status_text.setSpacing(3)
        self.status_label = self._label('Ready when you are', 'status')
        self.detail_label = self._label('Start your camera, then calibrate your displays.', 'muted')
        self.detail_label.setWordWrap(True)
        status_text.addWidget(self.status_label)
        status_text.addWidget(self.detail_label)
        status_row.addLayout(status_text, 1)
        self.pause_button = QPushButton('Pause overlay')
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.pause_requested.emit)
        status_row.addWidget(self.pause_button)
        root.addLayout(status_row)
        footer = self._label('ON YOUR DEVICE   ·   Local camera processing   ·   No video uploads', 'footer')
        root.addWidget(footer)
        dark = str(self.settings.value('dark', False)).strip().lower() in ('true', '1', 'yes', 'on')
        self.theme.setChecked(dark)
        self._apply_theme(dark)

    def _label(self, text, name):
        label = QLabel(text)
        label.setObjectName(name)
        return label

    def _card(self):
        card = QFrame()
        card.setObjectName('card')
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(14)
        return card, layout

    def _slider(self, layout, title, minimum, maximum, value, suffix):
        group = QVBoxLayout()
        heading = QHBoxLayout()
        heading.addWidget(QLabel(title))
        heading.addStretch()
        format_value = (lambda v: f'{round(2000 * (v / 100) ** 2)} ms') if suffix == 'ms' else (lambda v: f'{v} {suffix}')
        readout = self._label(format_value(value), 'muted')
        heading.addWidget(readout)
        group.addLayout(heading)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setAccessibleName(title)
        slider.valueChanged.connect(lambda v: readout.setText(format_value(v)))
        group.addWidget(slider)
        layout.addLayout(group, 1)
        return slider

    def _saved_int(self, key, default, minimum, maximum):
        try:
            return max(minimum, min(maximum, int(self.settings.value(key, default))))
        except (TypeError, ValueError):
            return default

    def _appearance(self):
        self.settings.setValue('radius', self.radius_slider.value())
        self.settings.setValue('smoothing', self.smoothing_slider.value())
        self.appearance_changed.emit(self.radius_slider.value(), self.smoothing_slider.value()/100)

    def current_style(self):
        return self._color, self.shape_combo.currentData(), self.opacity_slider.value()

    def _style(self):
        color, shape, opacity = self.current_style()
        for key, value in (('color', color), ('shape', shape), ('opacity', opacity)):
            self.settings.setValue(key, value)
        self.style_changed.emit(color, shape, opacity)

    def _update_color_button(self):
        swatch = QPixmap(16, 16)
        swatch.fill(QColor(self._color))
        self.color_button.setIcon(QIcon(swatch))
        self.color_button.setText(self._color.upper())

    def _choose_color(self):
        dialog = QColorDialog(QColor(self._color), self)
        dialog.setWindowTitle('Gaze bubble colour')
        dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog)
        if dialog.exec():
            self._color = dialog.selectedColor().name()
            self._update_color_button()
            self._style()

    def _preview_visibility(self, visible):
        if not visible:
            self.preview_label.clear()
            self.preview_label.setText('Preview hidden' if self._running else 'Camera is off\nStart when you’re ready')

    def set_displays(self, displays):
        self.desktop_map.displays = displays
        self.desktop_map.update()
        count = len(displays)
        self.display_count.setText(f'{count} display' + ('s' if count != 1 else ''))

    def set_status(self, text, detail=''):
        self.status_label.setText(text)
        self.detail_label.setText(detail)

    def set_running(self, running):
        self._running = running
        self.start_button.setText('Stop camera' if running else 'Start camera')
        self.camera_combo.setEnabled(not running)
        self.pause_button.setEnabled(running and self._calibrated)
        if not running:
            self.desktop_map.gaze = None
            self.desktop_map.update()
            self.preview_label.clear()
            self.preview_label.setText('Camera is off\nStart when you’re ready')
        elif not self.preview_toggle.isChecked():
            self.preview_label.setText('Preview hidden')

    def set_preview(self, image: QImage):
        if self.preview_toggle.isChecked() and not image.isNull():
            self.preview_label.setPixmap(QPixmap.fromImage(image).scaled(
                self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def set_gaze(self, x, y):
        self.desktop_map.gaze = None if x is None or y is None else (x, y)
        self.desktop_map.update()

    def set_calibrated(self, calibrated):
        self._calibrated = calibrated
        self.calibrate_button.setText('Recalibrate' if calibrated else 'Quick calibration')
        self.pause_button.setEnabled(self._running and calibrated)
        self.map_note.setText('Calibrated for your displays. Recalibrate after moving your camera.' if calibrated else
                              'Five targets + one quick check. About 7 seconds per screen.')

    def _apply_theme(self, dark):
        self.settings.setValue('dark', dark)
        bg, surface, tile, ink, muted, line, accent = (
            ('#000000', '#121017', '#17141f', '#f4f3f8', '#9c98ab', '#262330', '#a566ff')
            if dark else ('#fafafc', '#ffffff', '#faf9fd', '#1a1823', '#6c687e', '#ece9f4', '#7700ff'))
        strong = '#51485f' if dark else '#ddd8ea'
        on_accent = '#140b22' if dark else '#ffffff'
        palette = self.palette()
        for role, color in (
            (QPalette.ColorRole.Window, surface),
            (QPalette.ColorRole.WindowText, ink),
            (QPalette.ColorRole.Base, tile),
            (QPalette.ColorRole.AlternateBase, surface),
            (QPalette.ColorRole.Text, ink),
            (QPalette.ColorRole.Button, tile),
            (QPalette.ColorRole.ButtonText, ink),
            (QPalette.ColorRole.Highlight, accent),
            (QPalette.ColorRole.HighlightedText, on_accent),
            (QPalette.ColorRole.PlaceholderText, muted),
        ):
            palette.setColor(role, QColor(color))
        self.setPalette(palette)
        self.mark.load(str(Path(__file__).parent / 'assets' / ('nx-wordmark-light.svg' if dark else 'nx-wordmark-violet.svg')))
        self.desktop_map.dark = dark
        self.desktop_map.update()
        self.setStyleSheet(f'''
            QWidget {{ color: {ink}; font-family: "Inter", "Noto Sans", sans-serif; font-size: 13px; }}
            QWidget#window {{ background: {bg}; }}
            QDialog, QColorDialog {{ background: {surface}; color: {ink}; }}
            QLineEdit, QSpinBox, QDoubleSpinBox {{ background: {tile}; color: {ink}; border: 1px solid {strong}; border-radius: 6px; padding: 4px; selection-background-color: {accent}; selection-color: {on_accent}; }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {accent}; }}
            QLabel {{ background: transparent; border: none; }}
            QLabel#product {{ font-size: 23px; font-weight: 600; padding-left: 8px; }}
            QLabel#hero {{ font-size: 32px; font-weight: 650; letter-spacing: -1px; }}
            QLabel#section, QLabel#status {{ font-weight: 600; font-size: 15px; }}
            QLabel#muted {{ color: {muted}; font-size: 12px; }}
            QLabel#footer {{ color: {muted}; font-size: 10px; letter-spacing: 1px; }}
            QFrame#card {{ background: {surface}; border: 1px solid {line}; border-radius: 20px; }}
            QLabel#preview {{ background: {tile}; border: 1px solid {line}; border-radius: 12px; color: {muted}; }}
            QPushButton, QComboBox {{ background: {tile}; border: 1px solid {strong}; border-radius: 11px; padding: 11px 16px; min-height: 18px; }}
            QPushButton:hover {{ border-color: {accent}; }}
            QPushButton:pressed {{ background: {line}; }}
            QPushButton:focus, QComboBox:focus, QCheckBox:focus {{ border: 2px solid {accent}; }}
            QPushButton:disabled {{ color: {muted}; border-color: {line}; }}
            QPushButton#primary {{ background: {accent}; color: {on_accent}; border-color: {accent}; font-weight: 600; }}
            QPushButton#primary:hover {{ background: {'#b780ff' if dark else '#6810d9'}; }}
            QComboBox {{ padding: 8px 12px; }}
            QComboBox QAbstractItemView {{ background: {surface}; color: {ink}; selection-background-color: {accent}; selection-color: {on_accent}; }}
            QCheckBox {{ spacing: 8px; color: {muted}; }}
            QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {strong}; border-radius: 5px; background: {surface}; }}
            QCheckBox::indicator:checked {{ background: {accent}; border-color: {accent}; }}
            QSlider {{ min-height: 24px; }}
            QSlider::groove:horizontal {{ height: 4px; background: {strong}; border-radius: 2px; }}
            QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: {accent}; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; }}
        ''')
        for card in self.findChildren(QFrame):
            if card.objectName() == 'card':
                if dark:
                    card.setGraphicsEffect(None)
                else:
                    shadow = QGraphicsDropShadowEffect(card)
                    shadow.setBlurRadius(18)
                    shadow.setOffset(0, 4)
                    shadow.setColor(QColor(40, 20, 90, 15))
                    card.setGraphicsEffect(shadow)
