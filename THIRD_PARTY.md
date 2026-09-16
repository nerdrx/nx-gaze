# Third-party credits

NX Gaze combines an NX desktop interface and overlay with established open-source tracking and numerical libraries. These projects deserve the credit for their respective technology. Dependency installation retains their distributed license files; this source repository does not relicense them.

| Project | Role | License / upstream reference |
| --- | --- | --- |
| [EyeTrax](https://github.com/ck-zhang/EyeTrax), Chenkai Zhang | Webcam feature extraction and calibrated gaze estimation; dependency pinned to 0.4.0 | [MIT](https://github.com/ck-zhang/EyeTrax/blob/master/LICENSE) |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe), Google and contributors | Face landmark inference used by EyeTrax | [Apache-2.0 software license](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE); model terms discussed below |
| [PyQt6](https://www.riverbankcomputing.com/software/pyqt/), Riverbank Computing | Qt bindings for the control panel, calibration, and overlay | [GPL v3 or commercial](https://www.riverbankcomputing.com/software/pyqt/intro); NX Gaze uses the GPL distribution |
| [Qt](https://www.qt.io/) | Desktop windowing and rendering through PyQt6 | [Qt open-source licensing](https://www.qt.io/licensing/open-source-lgpl-obligations); individual Qt modules retain their applicable licenses |
| [OpenCV](https://opencv.org/) | Webcam capture and image handling | [Apache-2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE) for current OpenCV 4.x |
| [NumPy](https://numpy.org/) | Numerical arrays | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| [scikit-learn](https://scikit-learn.org/) | Calibration regression through EyeTrax | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) |
| [NX Hub](https://github.com/nerdrx/nx-hub) | Official NX wordmark and NX Clear v1.8 design source | [Canonical design document](https://github.com/nerdrx/nx-hub/blob/main/docs/DESIGN.md); NX project branding |

## Model assets

EyeTrax uses a MediaPipe face model. The model is downloaded on first camera start when it is absent from the local cache; it is not bundled as an NX-authored model. See the [MediaPipe Face Landmarker documentation](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) for upstream model information and links.

MediaPipe's software license does not by itself establish the license of every downloadable model asset. Model assets retain their own upstream terms. Check the terms accompanying the exact model before redistributing it.

## Attribution and distribution

NX Gaze uses EyeTrax as a dependency. Its gaze estimation is credited to EyeTrax and its underlying libraries; the NX contribution is the application interface, display geometry, calibration workflow, and overlay behavior. The head-feature adapter taps the pinned EyeTrax 0.4.0 detector once per frame, preserves camera-relative position/scale, and replaces wrapped angles with sine/cosine features; it does not run a separate tracking model.

NX Gaze application code is distributed under [GPL-3.0-or-later](LICENSE). The GPL PyQt6 distribution is used by this project. If packaging dependencies into a binary or installer, preserve each package's license and notice files, including notices for transitive dependencies, and meet the applicable source distribution requirements. This credit index does not replace those files.
