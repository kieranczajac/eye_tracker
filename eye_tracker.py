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
    - Eye Aspect Ratio (EAR) for OPEN/CLOSED classification
    Enhanced Features (no CSV logging):
    - Blink count
    - Blink frequency (blinks/min)
    - Blink duration (ms) + average blink duration
    - Independent left/right eye state (wink detection)
    - Temporal smoothing (reduces jitter)
    - Toggle full face landmark visualization (press 'l')
    """

    def __init__(self):
        # 6 landmark indices per eye (p1..p6)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        # Threshold for open/closed
        self.EAR_THRESHOLD = 0.21

        # Temporal smoothing (per-eye)
        self.left_hist = deque(maxlen=5)
        self.right_hist = deque(maxlen=5)

        # Blink tracking (both eyes together)
        self.blink_count = 0
        self.blink_start_time = None
        self.last_blink_duration_ms = None
        self.blink_durations_ms = []

        # Frequency tracking
        self.session_start_time = time.time()

        # Independent eye states (wink detection)
        self.left_state = "—"
        self.right_state = "—"

        # Toggle for drawing full face landmarks
        self.show_full_mesh = False

        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,          # process only one face
            refine_landmarks=True,    # improves eye landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.prev_time = time.time()

        # Camera opened in run() so tests don't invoke webcam
        self.cap = None

    # ---------------- Core Math ----------------

    def calculate_ear(self, eye_points: np.ndarray) -> float:
        """
        Eye Aspect Ratio (EAR) using 6 eye points:
        EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
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
        Convert normalized MediaPipe landmark coords to pixel coords.
        Returns np.array shape (len(indices), 2)
        """
        pts = []
        for idx in indices:
            lm = landmarks[idx]
            pts.append((int(lm.x * frame_w), int(lm.y * frame_h)))
        return np.array(pts, dtype=np.int32)

    # ---------------- Rendering helpers ----------------

    def _draw_full_mesh(self, frame, face_landmarks, w, h):
        for lm in face_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (200, 200, 200), -1)

    # ---------------- Frame processing ----------------

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        status = "NO FACE"
        both_state = "—"
        both_color = (255, 255, 255)

        left_ear_s = None
        right_ear_s = None
        avg_ear = None

        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0].landmark

            if self.show_full_mesh:
                self._draw_full_mesh(frame, face_landmarks, w, h)

            left_pts = self.get_eye_landmarks(face_landmarks, self.LEFT_EYE, w, h)
            right_pts = self.get_eye_landmarks(face_landmarks, self.RIGHT_EYE, w, h)

            left_ear = self.calculate_ear(left_pts.astype(np.float32))
            right_ear = self.calculate_ear(right_pts.astype(np.float32))

            # Smooth per eye
            self.left_hist.append(left_ear)
            self.right_hist.append(right_ear)
            left_ear_s = float(np.mean(self.left_hist))
            right_ear_s = float(np.mean(self.right_hist))
            avg_ear = (left_ear_s + right_ear_s) / 2.0

            # Independent eye state (wink detection)
            self.left_state = "CLOSED" if left_ear_s < self.EAR_THRESHOLD else "OPEN"
            self.right_state = "CLOSED" if right_ear_s < self.EAR_THRESHOLD else "OPEN"

            # Both-eyes state (for blink counting)
            both_closed = (self.left_state == "CLOSED" and self.right_state == "CLOSED")
            both_state = "CLOSED" if both_closed else "OPEN"
            both_color = (0, 0, 255) if both_state == "CLOSED" else (0, 255, 0)

            # Blink duration tracking (when BOTH eyes close then open)
            now = time.time()
            if both_closed and self.blink_start_time is None:
                self.blink_start_time = now

            if (not both_closed) and self.blink_start_time is not None:
                duration_ms = int((now - self.blink_start_time) * 1000)
                self.last_blink_duration_ms = duration_ms
                self.blink_durations_ms.append(duration_ms)
                self.blink_count += 1
                self.blink_start_time = None

            # Draw eye contours per-eye color
            left_color = (0, 255, 0) if self.left_state == "OPEN" else (0, 0, 255)
            right_color = (0, 255, 0) if self.right_state == "OPEN" else (0, 0, 255)

            for (x, y) in left_pts:
                cv2.circle(frame, (x, y), 2, left_color, -1)
            cv2.polylines(frame, [left_pts], True, left_color, 1)

            for (x, y) in right_pts:
                cv2.circle(frame, (x, y), 2, right_color, -1)
            cv2.polylines(frame, [right_pts], True, right_color, 1)

            status = "FACE DETECTED"

        # FPS
        now_t = time.time()
        fps = 1.0 / (now_t - self.prev_time) if (now_t - self.prev_time) > 0 else 0.0
        self.prev_time = now_t

        # Blink frequency
        elapsed_s = now_t - self.session_start_time
        blinks_per_min = (self.blink_count / elapsed_s) * 60.0 if elapsed_s > 1 else 0.0

        # UI overlays
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if left_ear_s is not None:
            cv2.putText(frame, f"L_EAR: {left_ear_s:.3f}  ({self.left_state})",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if right_ear_s is not None:
            cv2.putText(frame, f"R_EAR: {right_ear_s:.3f}  ({self.right_state})",
                        (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if avg_ear is not None:
            cv2.putText(frame, f"AVG_EAR: {avg_ear:.3f}  BOTH: {both_state}",
                        (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, both_color, 2)

        cv2.putText(frame, f"Blinks: {self.blink_count}  ({blinks_per_min:.1f}/min)",
                    (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.last_blink_duration_ms is not None:
            avg_dur = int(np.mean(self.blink_durations_ms)) if self.blink_durations_ms else 0
            cv2.putText(frame, f"Last blink: {self.last_blink_duration_ms} ms | Avg: {avg_dur} ms",
                        (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        cv2.putText(frame, "q=quit | l=toggle landmarks",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

        return frame

    # ---------------- Main loop ----------------

    def run(self, camera_index=0):
        self.session_start_time = time.time()

        # Use DirectShow backend for Windows reliability
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        print("Camera opened?", self.cap.isOpened())

        if not self.cap.isOpened():
            print("ERROR: Webcam could not be opened. Try camera_index=1 or check Windows camera permissions.")
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
