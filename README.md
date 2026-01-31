# Eye Tracking System (EAR + MediaPipe)

A real-time eye tracking system that uses MediaPipe Face Mesh to detect facial landmarks and compute the Eye Aspect Ratio (EAR) to classify eye state as OPEN or CLOSED. Includes EAR smoothing for stability and a blink counter.

# Features
- Real-time webcam capture via OpenCV
- MediaPipe Face Mesh face landmark detection
- Eye landmark extraction (6 points per eye)
- EAR metric + on-screen EAR display
- Threshold-based eye state (OPEN/CLOSED)
- Color display: green = OPEN, red = CLOSED
- EAR smoothing to reduces jitter
- Blink counter
- Edge case handling (no face detected)
- Optional toggle to display all face landmarks (`l` key)

# Limitations
- EAR thresholds vary for different kinds of people i.e. people with smaller eyes or those who are squinting could be classified as having CLOSED eyes when they are in fact OPEN
- The blink counter cannot capture many fast blinks in a row

# Requirements
- Any Python 3.9 to 3.12 recommended (or else MediaPipe unsupported)
- Webcam

# Python Packages
- opencv-python
- mediapipe
- numpy
- scipy
- (optional for tests) pytest

#Installation (Windows PowerShell)

1) Create and activate a virtual environment:
```powershell
py -V:3.11 -m venv .venv
.\.venv\Scripts\activate
