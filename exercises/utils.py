import math
import threading
import pyttsx3

def calculate_angle(a, b, c):
    """
    Calculates angle between three points - used across different exercises
    for measuring joint angles (elbow, knee, shoulder etc.)
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
    Voice feedback function used across all exercises
    """
    def _speak(msg):
        engine = pyttsx3.init()
        engine.say(msg)
        engine.runAndWait()
    threading.Thread(target=_speak, args=(message,)).start()

def get_exercise_metrics(start_time, end_time, total_paused_time, rep_count):
    """
    Common function to calculate exercise metrics
    """
    active_time = end_time - start_time - total_paused_time
    return {
        'total_time': round(end_time - start_time, 2),
        'active_time': round(active_time, 2),
        'paused_time': round(total_paused_time, 2),
        'reps': rep_count,
        'avg_time_per_rep': round(active_time / rep_count, 2) if rep_count > 0 else 0
    }