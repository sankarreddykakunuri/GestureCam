
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque, Counter

CAMERA_SOURCE = "http://192.168.1.72:8080/video"
WINDOW_NAME = "GestureCam - Filter Studio"

FILTERS = {
    0: "Original",
    1: "Grayscale",
    2: "Blur",
    3: "Edges",
    4: "Sepia",
    5: "Sharpen"
}

gesture_history = deque(maxlen=7)


def count_fingers(hand, handedness):
    lm = hand.landmark
    count = 0

    # Thumb
    if handedness == "Right":
        count += lm[4].x < lm[3].x
    else:
        count += lm[4].x > lm[3].x

    # Index, middle, ring and little fingers
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        count += lm[tip].y < lm[pip].y

    return int(count)


def apply_filter(frame, name):
    if name == "Grayscale":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if name == "Blur":
        return cv2.GaussianBlur(frame, (21, 21), 0)

    if name == "Edges":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if name == "Sepia":
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        return np.clip(
            cv2.transform(frame, kernel), 0, 255
        ).astype(np.uint8)

    if name == "Sharpen":
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        return cv2.filter2D(frame, -1, kernel)

    return frame.copy()


def draw_text(frame, text, position, scale=0.7,
              color=(255, 255, 255)):
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 2, cv2.LINE_AA
    )


def main():
    camera = cv2.VideoCapture(CAMERA_SOURCE)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    mp_hands = mp.solutions.hands
    drawing = mp.solutions.drawing_utils

    active_filter = "Original"
    fps = 0.0
    previous_time = time.time()
    fullscreen = False

    print("GestureCam Filter Studio started.")
    print("Gestures: 0-5 fingers select filters.")
    print("Keyboard: 0-5 filters | F fullscreen | Q quit")

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ) as hands:

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

            while True:
                success, frame = camera.read()
                if not success:
                    print("Unable to read camera frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                finger_count = None

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]

                    handedness = "Right"
                    if results.multi_handedness:
                        handedness = (
                            results.multi_handedness[0]
                            .classification[0].label
                        )

                    finger_count = count_fingers(hand, handedness)

                    drawing.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS
                    )

                    gesture_history.append(finger_count)
                else:
                    # Don't let an old gesture remain active forever.
                    gesture_history.clear()

                if gesture_history:
                    stable_count = Counter(
                        gesture_history
                    ).most_common(1)[0][0]

                    active_filter = FILTERS[stable_count]

                output = apply_filter(frame, active_filter)

                now = time.time()
                delta = now - previous_time
                previous_time = now

                if delta > 0:
                    instant_fps = 1.0 / delta
                    fps = instant_fps if fps == 0 else (
                        0.9 * fps + 0.1 * instant_fps
                    )

                gesture_label = (
                    str(finger_count)
                    if finger_count is not None
                    else "No hand"
                )

                draw_text(
                    output, "GestureCam",
                    (20, 35), 0.9, (0, 255, 0)
                )
                draw_text(
                    output, f"Fingers: {gesture_label}",
                    (20, 75)
                )
                draw_text(
                    output, f"Filter: {active_filter}",
                    (20, 115)
                )
                draw_text(
                    output, f"FPS: {fps:.1f}",
                    (20, 155)
                )
                draw_text(
                    output, "0-5: Filters | F: Fullscreen | Q: Quit",
                    (20, output.shape[0] - 20), 0.55
                )

                cv2.imshow(WINDOW_NAME, output)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                elif key == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen
                        else cv2.WINDOW_NORMAL
                    )

                elif ord("0") <= key <= ord("5"):
                    active_filter = FILTERS[key - ord("0")]
                    gesture_history.clear()

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("GestureCam stopped. Resources released.")


if __name__ == "__main__":
    main()
