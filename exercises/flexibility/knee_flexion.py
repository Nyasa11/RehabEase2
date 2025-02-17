import cv2
import mediapipe as mp
import math
import time
import pyttsx3
import threading

# -----------------------------
# 1) Helper Functions
# -----------------------------

def calculate_angle(a, b, c):
    """
    Calculates the angle at point b given three landmarks: a, b, c.
    Returns angle in degrees within [0, 180].
    """
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]
    
    radians = (math.atan2(c[1] - b[1], c[0] - b[0]) -
               math.atan2(a[1] - b[1], a[0] - b[0]))
    angle = abs(math.degrees(radians))
    if angle > 180.0:
        angle = 360 - angle
    return angle

def say_async(message):
    """
    Provides voice feedback in a separate thread to avoid blocking.
    """
    def _speak(msg):
        engine = pyttsx3.init()
        engine.say(msg)
        engine.runAndWait()
    threading.Thread(target=_speak, args=(message,)).start()

# -----------------------------
# 2) Main Exercise Function
# -----------------------------

def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    # ---- Rep Counting Variables ----
    rep_count = 0
    stage = None  # "up" (extended) or "down" (flexed)
    times_of_each_rep = []

    # ---- Performance Metrics ----
    start_time = time.time()
    total_paused_time = 0.0  # Total time spent in error state (paused)
    in_error = False
    error_start_time = None

    # ---- Voice Feedback Controls ----
    last_feedback_time = 0.0
    feedback_cooldown = 3.0  # seconds to wait before repeating an error voice

    # ---- Full-Screen Window ----
    window_name = 'Exercise Feedback'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Default feedback (empty until we determine)
        feedback_text = ""
        feedback_color = (255, 255, 255)  # white

        # Assume posture is correct unless an error is detected.
        posture_is_correct = True

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get landmarks for left side
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

            # Also get right hip for alignment check
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

            # Calculate knee angle
            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

            # ---------------------------
            # A) Error Check: Knee Angle
            # ---------------------------
            if knee_angle < 10 or knee_angle > 160:
                posture_is_correct = False
                feedback_text = "Incorrect posture. Please adjust your knee position."
                feedback_color = (0, 0, 255)  # red
                current_time = time.time()
                if (current_time - last_feedback_time) > feedback_cooldown:
                    say_async("Incorrect posture. Please adjust your knee position.")
                    last_feedback_time = current_time

            # ---------------------------
            # B) Error Check: Knee Alignment
            # ---------------------------
            if abs(left_hip.x - left_knee.x) > 0.15:
                posture_is_correct = False
                feedback_text = "Keep your knee aligned with your hip."
                feedback_color = (0, 0, 255)  # red
                current_time = time.time()
                if (current_time - last_feedback_time) > feedback_cooldown:
                    say_async("Keep your knee aligned with your hip.")
                    last_feedback_time = current_time

            # ---------------------------
            # C) Rep Counting State Machine
            #    (Only if posture is correct.)
            # ---------------------------
            if posture_is_correct:
                if knee_angle > 160:
                    if stage == "down":
                        rep_count += 1
                        times_of_each_rep.append(time.time())
                        if rep_count % 5 == 0:
                            say_async(f"Great job! You have completed {rep_count} reps. Keep going!")
                    stage = "up"
                elif knee_angle < 10:
                    stage = "down"

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

            # Show the angle (top left, now in blue)
            cv2.putText(frame,
                        f"Angle: {int(knee_angle)}",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),  # Blue color
                        2)

        # ---------------------------
        # E) Posture Feedback if Correct
        # ---------------------------
        if posture_is_correct:
            if not feedback_text:
                feedback_text = "You're doing great! Keep going."
                feedback_color = (0, 255, 0)  # green

            if in_error:
                in_error = False
                if error_start_time is not None:
                    total_paused_time += (time.time() - error_start_time)
                    error_start_time = None
        else:
            if not in_error:
                in_error = True
                error_start_time = time.time()

        # Display rep count (blue)
        cv2.putText(frame,
                    f"Reps: {rep_count}",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),  # Blue color
                    2)

        # Display feedback text
        cv2.putText(frame,
                    feedback_text,
                    (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    feedback_color,
                    2)

        # Show the frame
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ---------------------------
    # F) Performance Metrics
    # ---------------------------
    if rep_count > 0 and times_of_each_rep:
        last_rep_time = times_of_each_rep[-1]
        total_time = last_rep_time - start_time - total_paused_time
        avg_time_per_rep = total_time / rep_count
        print("===== PERFORMANCE METRICS =====")
        print(f"Total Reps: {rep_count}")
        print(f"Active Exercise Time: {total_time:.2f} seconds (excludes paused time)")
        print(f"Average Time Per Rep: {avg_time_per_rep:.2f} seconds")
    else:
        print("No reps completed.")

# -----------------------------
# 3) Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
