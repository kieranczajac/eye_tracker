import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import euclidean
import time
from collections import deque


class EyeTracker:
    """
    Real-Time Eye Tracking System
    - MediaPipe Face Mesh for landmarks
    - Eye Aspect Ratio (EAR) for eye state (OPEN/CLOSED)
    - EAR smoothing + blink counter
    - Optional full face landmark visualization toggle ('l')
    """

    def __init__(self):
        # MediaPipe landmark indices (6 points per eye)
        # Left eye: [p1,p2,p3,p4,p5,p6] and same for right
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        # EAR threshold for classification
        self.EAR_THRESHOLD = 0.21

        # Smoothing window for EAR (reduces jitter)
        self.ear_history = deque(maxlen=5)

        # Blink tracking
        self.blink_count = 0
        self.eye_closed_frames = 0
        self.MIN_CLOSED_FRAMES_FOR_BLINK = 2  # must be closed this many frames to count as blink

        # Toggle for drawing full face landmarks
        self.show_full_mesh = False

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,           # handle multiple faces by processing only one
            refine_landmarks=True,     # improves eye detail
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.prev_time = time.time()

        # Camera is opened in run() so unit tests don't trigger webcam usage
        self.cap = None

    # ----------------------------- Core Math -----------------------------

    def calculate_ear(self, eye_points: np.ndarray) -> float:
        """
        Eye Aspect Ratio (EAR)
        eye_points shape: (6, 2) in order [p1,p2,p3,p4,p5,p6]
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        """
        p1, p2, p3, p4, p5, p6 = eye_points

        v1 = euclidean(p2, p6)
        v2 = euclidean(p3, p5)
        h = euclidean(p1, p4)

        if h == 0:
            return 0.0

        return (v1 + v2) / (2.0 * h)

    def get_eye_landmarks(self, landmarks, indices, frame_w: int, frame_h: int) -> np.ndarray:
        """
        Convert normalized MediaPipe landmarks to pixel coords for selected indices.
        Returns np.array shape (len(indices), 2)
        """
        pts = []
        for idx in indices:
            lm = landmarks[idx]
            pts.append((int(lm.x * frame_w), int(lm.y * frame_h)))
        return np.array(pts, dtype=np.int32)

    # ----------------------------- Rendering -----------------------------

    def _draw_full_mesh(self, frame, face_landmarks, w, h):
        """
        Draw small dots for all face landmarks (optional).
        """
        for lm in face_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (200, 200, 200), -1)

    def process_frame(self, frame):
        """
        Process one video frame:
        - detect face
        - extract eye landmarks
        - compute EAR + smoothed EAR
        - classify OPEN/CLOSED
        - draw overlays
        Returns annotated frame.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        # Default UI text if no face
        status = "NO FACE"
        ear_text = "EAR: --"
        state_text = "STATE: --"
        state_color = (255, 255, 255)  # white default when no face

        if result.multi_face_landmarks:
            # Process only one face (first)
            face_landmarks = result.multi_face_landmarks[0].landmark

            if self.show_full_mesh:
                self._draw_full_mesh(frame, face_landmarks, w, h)

            # Extract eye points
            left_pts = self.get_eye_landmarks(face_landmarks, self.LEFT_EYE, w, h)
            right_pts = self.get_eye_landmarks(face_landmarks, self.RIGHT_EYE, w, h)

            # Compute EAR
            left_ear = self.calculate_ear(left_pts.astype(np.float32))
            right_ear = self.calculate_ear(right_pts.astype(np.float32))
            ear_raw = (left_ear + right_ear) / 2.0

            # Smooth EAR
            self.ear_history.append(ear_raw)
            ear = float(np.mean(self.ear_history))

            # Classify state
            state = "CLOSED" if ear < self.EAR_THRESHOLD else "OPEN"
            state_color = (0, 0, 255) if state == "CLOSED" else (0, 255, 0)

            # Blink detection
            if state == "CLOSED":
                self.eye_closed_frames += 1
            else:
                if self.eye_closed_frames > self.MIN_CLOSED_FRAMES_FOR_BLINK:
                    self.blink_count += 1
                self.eye_closed_frames = 0

            status = "FACE DETECTED"
            ear_text = f"EAR: {ear:.3f}"
            state_text = f"STATE: {state}"

            # Draw eye contours in state color (green=open, red=closed)
            for pts in [left_pts, right_pts]:
                for (x, y) in pts:
                    cv2.circle(frame, (x, y), 2, state_color, -1)
                cv2.polylines(frame, [pts], True, state_color, 1)

        # FPS
        now = time.time()
        fps = 1.0 / (now - self.prev_time) if (now - self.prev_time) > 0 else 0.0
        self.prev_time = now

        # UI overlays
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, ear_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, state_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {self.blink_count}", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Help text
        cv2.putText(frame, "q=quit | l=toggle landmarks", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

        return frame

    # ----------------------------- Main Loop -----------------------------

    def run(self, camera_index=0):
        """
        Main loop:
        - open camera
        - read frames
        - process + display
        - keyboard input (q quit, l toggle landmarks)
        """
        # Try DirectShow backend for Windows reliability
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        print("Camera opened?", self.cap.isOpened())

        if not self.cap.isOpened():
            print("ERROR: Webcam could not be opened.")
            print("Try: run(camera_index=1) or enable Windows camera permissions for desktop apps.")
            return

        while True:
            ok, frame = self.cap.read()

            if not ok or frame is None:
                print("ERROR: cap.read() failed (no frame). Camera permission denied or camera busy.")
                break

            annotated = self.process_frame(frame)
            cv2.imshow("Real-Time Eye Tracking", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l"):
                self.show_full_mesh = not self.show_full_mesh

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    EyeTracker().run()
