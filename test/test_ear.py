import os
import sys
import numpy as np

# Ensure project root is on sys.path so `from eye_tracker import EyeTracker` works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from eye_tracker import EyeTracker


def test_calculate_ear_open_eye():
    tracker = EyeTracker()

    # Open eye: larger vertical distance relative to horizontal
    eye_points = np.array([
        [0, 0],    # p1
        [1, 3],    # p2
        [2, 3],    # p3
        [4, 0],    # p4
        [2, -3],   # p5
        [1, -3]    # p6
    ], dtype=np.float32)

    ear = tracker.calculate_ear(eye_points)
    assert ear > 0.3


def test_calculate_ear_closed_eye():
    tracker = EyeTracker()

    # Closed eye: very small vertical distance relative to horizontal
    eye_points = np.array([
        [0, 0],     # p1
        [1, 0.2],   # p2
        [2, 0.2],   # p3
        [4, 0],     # p4
        [2, -0.2],  # p5
        [1, -0.2]   # p6
    ], dtype=np.float32)

    ear = tracker.calculate_ear(eye_points)
    assert ear < 0.2


