# Validation for v0.1.0-preview

This is an experimental build, not a claim of measured gaze accuracy.

Verified on Linux with Python 3.12:

- Local model initialization through EyeTrax 0.4.0 and MediaPipe 1.0.1.
- A four-second webcam worker smoke check: 68 processed frames, 62 face-feature frames, no reported camera errors. No images or video saved. This checks capture/inference only, not gaze position accuracy.
- Native application rendered inside headless Gamescope at 1280 × 960 and exited normally.
- Light and dark control-panel screenshots inspected. README images show the actual UI with an example two-display layout and generic camera label.
- Geometry checks cover negative origins, vertical layouts, screen edges, desktop gaps, non-finite coordinates, and smoothing resets.
- Lifecycle checks cover calibration cancellation, stale training completion, stop races, display layout changes, same-frame validation, and private calibration persistence.
- Dependency consistency checked with `pip check`; source files compile and launch scripts pass `bash -n`.
- `docs/DESIGN.md` is byte-identical to NX Hub's canonical v1.8 source at the time of creation.

Still requires hands-on validation:

- Personal calibration and held-out error measurements across physical monitors.
- Tracking stability while moving the head or turning toward peripheral screens.
- Actual desktop click-through behavior and overlay stacking above each application, especially fullscreen games.
- Mixed display scaling and compositor-specific positioning.

Run the automated checks with:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -v
```

Camera-free app smoke check:

```bash
gamescope --backend headless --force-windows-fullscreen -W 1280 -H 960 -- \
  ./run.sh --smoke /tmp/nx-gaze-ui.png
```
