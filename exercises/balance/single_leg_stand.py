import cv2
import mediapipe as mp
import time
import pyttsx3
import threading
import numpy as np

def say_async(message):
    """
    Provides voice feedback in a separate thread to avoid blocking.
    """
    def _speak(msg):
        engine = pyttsx3.init()
        engine.say(msg)
        engine.runAndWait()
    threading.Thread(target=_speak, args=(message,)).start()


def check_leg_raise(landmarks, leg_to_raise):
    """
    Checks if the user is correctly performing a side leg raise (front-facing camera).
    
    Returns:
      is_correct (bool): True if posture is correct, False otherwise.
      feedback (str): A message for on-screen feedback.
      error_flags (dict): Dictionary with error types and messages.
    """
    error_flags = {}
    feedback = ""
    is_correct = True

    # Extract necessary landmarks
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # --- Error Check 1: Upper Body Upright ---
    # We expect the hip to be above the knee (a smaller y-value) for an upright posture.
    if (left_hip.y > left_knee.y) or (right_hip.y > right_knee.y):
        error_flags["upright"] = "Keep your upper body upright."
        is_correct = False

    # --- Error Check 2: Shoulder Alignment ---
    # If shoulders differ in y by more than 0.1, user might be leaning or tilting.
    if abs(left_shoulder.y - right_shoulder.y) > 0.1:
        error_flags["shoulders"] = "Keep your shoulders level."
        is_correct = False

    # --- Error Check 3: Side Leg Raise (Frontal View) ---
    # For a LEFT leg raise, check if left ankle is sufficiently far from left hip on the x-axis.
    # For a RIGHT leg raise, check if right ankle is sufficiently far from right hip on the x-axis.
    # We use a threshold of 0.06 as a starting point. Adjust if needed.
    threshold = 0.06

    if leg_to_raise == "left":
        distance_x = abs(left_ankle.x - left_hip.x)
        if distance_x < threshold:
            error_flags["leg_raise"] = "Raise your left leg more to the side."
            is_correct = False
        else:
            feedback = "Good job! Left leg raised correctly."

    elif leg_to_raise == "right":
        distance_x = abs(right_ankle.x - right_hip.x)
        if distance_x < threshold:
            error_flags["leg_raise"] = "Raise your right leg more to the side."
            is_correct = False
        else:
            feedback = "Good job! Right leg raised correctly."

    else:
        error_flags["leg_select"] = "Unknown leg selection."
        is_correct = False

    # If no specific feedback was set but posture is correct
    if is_correct and not feedback:
        feedback = "Posture is good."
    return is_correct, feedback, error_flags


def main():
    global mp_pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Exercise Timing Parameters (modified)
    active_duration = 10   # seconds of active (correct) hold per leg
    rest_duration = 5      # seconds rest between holds
    leg_to_raise = 'left'  # start with left leg

    # Performance Metrics
    total_active_hold = 0.0  # total time in correct posture (active exercise)
    successful_holds = 0     # number of successful holds

    # Voice feedback control
    last_error_voice_time = 0.0
    error_voice_cooldown = 3.0  # seconds

    # Active hold timer variables
    active_start_time = None  # when correct posture resumed
    current_active_hold = 0.0

    # For positive voice feedback (e.g., every 10 seconds)
    last_positive_voice_time = 0.0
    positive_voice_interval = 10.0

    # Set up full-screen window
    window_name = 'Single Leg Side Raise'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        feedback_text = ""
        feedback_color = (0, 255, 0)  # green for correct by default
        current_time = time.time()

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Check posture for the chosen leg
            is_correct, check_feedback, error_flags = check_leg_raise(landmarks, leg_to_raise)

            if is_correct:
                # Good posture: start/resume the active hold timer
                feedback_text = check_feedback
                feedback_color = (0, 255, 0)  # green

                if active_start_time is None:
                    active_start_time = current_time
                else:
                    current_active_hold = current_time - active_start_time

                # Occasional positive voice feedback
                if current_time - last_positive_voice_time > positive_voice_interval:
                    say_async(f"Good job! Keep holding your {leg_to_raise} leg.")
                    last_positive_voice_time = current_time

            else:
                # Posture is incorrect
                feedback_text = " ".join(error_flags.values())
                feedback_color = (0, 0, 255)  # red

                # Voice feedback for errors (throttled)
                if current_time - last_error_voice_time > error_voice_cooldown:
                    for err_msg in error_flags.values():
                        say_async(err_msg)
                    last_error_voice_time = current_time

                # Pause active hold timer (if it was running)
                if active_start_time is not None:
                    total_active_hold += (current_time - active_start_time)
                    active_start_time = None
                    current_active_hold = 0.0

            # Display current hold time (only while correct)
            hold_text = f"Hold Time: {int(current_active_hold)}s" if active_start_time else "Hold Paused"
            cv2.putText(frame, hold_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Display current leg
            cv2.putText(frame, f"Raise your {leg_to_raise} leg to the side.", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Display feedback
            cv2.putText(frame, feedback_text, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 2)

        # If the hold time for the correct posture reaches active_duration, proceed to rest
        if active_start_time is not None and current_active_hold >= active_duration:
            say_async(f"{leg_to_raise.capitalize()} leg hold complete! Rest for {rest_duration} seconds.")
            successful_holds += 1
            total_active_hold += current_active_hold

            # Show completion message briefly
            cv2.putText(frame, f"{leg_to_raise.capitalize()} leg hold complete!", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

            # Rest period
            rest_start = time.time()
            while time.time() - rest_start < rest_duration:
                rest_frame = 255 * np.ones_like(frame)
                remaining_rest = int(rest_duration - (time.time() - rest_start))
                cv2.putText(rest_frame, f"Rest: {remaining_rest}s", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(window_name, rest_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Switch legs
            leg_to_raise = "right" if leg_to_raise == "left" else "left"

            # Reset timers for the next hold
            active_start_time = None
            current_active_hold = 0.0
            last_positive_voice_time = 0.0

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Performance Summary
    session_end = time.time()
    print("===== SESSION SUMMARY =====")
    print(f"Total Active Hold Time (correct posture): {total_active_hold:.2f} seconds")
    print(f"Number of Successful Holds: {successful_holds}")

    say_async("Session ended. Well done!")

if __name__ == "__main__":
    main()
