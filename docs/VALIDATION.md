# v0.2.1-preview correction

The v0.2.0 pose weighting used pooled variation, which confounded head position with calibration target. The replacement uses within-target variation and disables unsupported pose inputs. A camera-relative pose guard suppresses extrapolation outside the calibrated range. This is a conservative fallback, not a claim of reliable free-head gaze tracking.

20 tests pass, including actual Ridge/StandardScaler regression: changing target-correlated unsupported pose features cannot steer the output; independent within-target motion can enable supported features; out-of-range pose returns no estimate. Existing v0.2 calibration samples are reused and retrained. No new live accuracy result is claimed.

# Validation for v0.2.0-preview

- 18 automated checks pass, covering head-feature continuity, camera-relative translation/scale, one inference per frame, no stale pose reuse, low-variance regularization, blink recovery/hold, marker colours/shapes/opacity, and prior lifecycle/geometry checks.
- Updated real webcam smoke: 66 frames processed, 66 with face features, 492 augmented features per valid frame, no errors. No images saved. This is inference evidence, not an accuracy result.
- Actual EyeTrax train/predict path accepts augmented features and bounded scaling; finite output checked with synthetic inputs.
- Calibration nominally takes 14.4 seconds across two screens (five targets and one held-out check per screen), excluding model training and extensions for blinks or tracking loss. Real completion time has not yet been measured with a person.
- Updated UI passes headless Gamescope smoke. Light/dark settings layout and persistent controls checked offscreen.
- No before/after head-motion accuracy claim: camera-relative compensation needs fresh personal calibration, and broad head movement outside the sampled range remains a limitation.
- Appearance settings live in the platform Qt settings location (Linux normally `~/.config/NX/Gaze.conf`). Calibration features remain local; old profiles are rejected by feature-schema metadata without deletion.

## Initial preview evidence

### v0.1.0-preview

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
