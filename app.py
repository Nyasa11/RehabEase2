from flask import Flask, render_template, redirect, url_for, request
import os
from datetime import datetime

app = Flask(__name__)

# Exercise configurations
exercises = {
    'balance': {
        'static_balance': {
            'name': 'Static Balance Exercise',
            'description': 'Improve stability and control while maintaining a steady position.',
            'duration': 30,
            'sets': 3,
            'rest_time': 60,
            'video_id': 'xyz123',  # Replace with actual YouTube video ID
            'type': 'static_balance'
        },
        'tandem_stance': {
            'name': 'Tandem Stance',
            'description': 'Improve balance by standing heel-to-toe.',
            'duration': 30,
            'sets': 3,
            'rest_time': 60,
            'video_id': 'abc456',  # Replace with actual YouTube video ID
            'type': 'tandem_stance'
        },
        'single_leg_reach': {
            'name': 'Single Leg Reach',
            'description': 'Dynamic balance exercise with reaching movements.',
            'duration': 45,
            'sets': 3,
            'rest_time': 60,
            'video_id': 'def789',  # Replace with actual YouTube video ID
            'type': 'single_leg_reach'
        }
    },
    'flexibility': {
        'elbow_flexion': {
            'name': 'Elbow Flexion & Extension',
            'description': 'Improve elbow range of motion.',
            'reps': 15,
            'sets': 3,
            'hold_time': 5,
            'rest_time': 30,
            'video_id': 'ghi012',  # Replace with actual YouTube video ID
            'type': 'elbow_flexion'
        },
        'knee_flexion': {
            'name': 'Knee Flexion & Extension',
            'description': 'Enhance knee mobility and strength.',
            'reps': 12,
            'sets': 3,
            'hold_time': 5,
            'rest_time': 30,
            'video_id': 'jkl345',  # Replace with actual YouTube video ID
            'type': 'knee_flexion'
        },
        'shoulder_flexion': {
            'name': 'Shoulder Flexion',
            'description': 'Improve shoulder mobility and range.',
            'reps': 10,
            'sets': 3,
            'hold_time': 5,
            'rest_time': 30,
            'video_id': 'mno678',  # Replace with actual YouTube video ID
            'type': 'shoulder_flexion'
        }
    },
    'gait': {
        'heel_raises': {
            'name': 'Heel Raises',
            'description': 'Strengthen calf muscles to improve walking stability and push-off phase.',
            'distance': '10 meters',
            'reps': 5,
            'rest_time': 60,
            'video_id': 'pqr901',  # Replace with actual YouTube video ID
            'type': 'heel_raises'
        },
        'side_stepping': {
            'name': 'Side Stepping',
            'description': 'Enhance lateral stability and movement.',
            'distance': '5 meters',
            'reps': 8,
            'rest_time': 45,
            'video_id': 'stu234',  # Replace with actual YouTube video ID
            'type': 'side_stepping'
        },
        'backward_walking': {
            'name': 'Backward Walking',
            'description': 'Improve balance and coordination.',
            'distance': '5 meters',
            'reps': 5,
            'rest_time': 60,
            'video_id': 'vwx567',  # Replace with actual YouTube video ID
            'type': 'backward_walking'
        }
    }
}

@app.route('/')
def home():
    return render_template('home.html')  # or your landing page

@app.route('/categories')
def categories():
    return render_template('categories.html')  # or your categories page

@app.route('/<category>')
def category_page(category):
    if category in ['balance', 'gait', 'flexibility']:
        return render_template(f'{category}.html')
    return "Category not found", 404

@app.route('/exercises/<category>/<exercise_type>')
def exercise_instructions(category, exercise_type):
    """
    Dynamic route for instructions of any exercise in your dictionary.
    For example: /exercises/flexibility/knee_flexion
    """
    if category in exercises and exercise_type in exercises[category]:
        exercise_data = exercises[category][exercise_type]
        return render_template('exercise_instruction.html', 
                               category=category,
                               exercises=exercise_data)
    return "Exercise not found", 404

# Specific route for Elbow Flexion instructions
@app.route('/exercises/flexibility/elbow_flexion')
def elbow_flexion_instructions():
    # Renders the instructions file: elbow_inst.html
    return render_template('elbow_inst.html')

# Route for the camera-based exercise page
@app.route('/start-exercises/elbow_flexion')
def start_elbow_flexion():
    # Renders the camera-based page elbow_flexion.html
    return render_template('elbow_flexion.html')

@app.route('/exercises/flexibility/knee_flexion')
def knee_flexion_instructions():
    # Renders the instructions file: elbow_inst.html
    return render_template('knee_inst.html')

# Route for the camera-based exercise page
@app.route('/start-exercises/knee_flexion')
def start_knee_flexion():
    # Renders the camera-based page elbow_flexion.html
    return render_template('knee_flexion.html')

@app.route('/exercises/balance/single_leg_stand')
def single_leg_stand_instructions():
    # Renders the instructions file: elbow_inst.html
    return render_template('single_leg_stand_inst.html')

@app.route('/start-exercises/single_leg_stand')
def start_single_leg_stand():
    # Renders the camera-based page elbow_flexion.html
    return render_template('single_leg_stand.html')

@app.route('/exercises/balance/single_leg_stand_front')
def single_leg_stand_front_instructions():
    # Renders the instructions file: elbow_inst.html
    return render_template('single_leg_stand_front_inst.html')

@app.route('/start-exercises/single_leg_stand_front')
def start_single_leg_stand_front():
    # Renders the camera-based page elbow_flexion.html
    return render_template('single_leg_stand_front.html')

@app.route('/exercises/balance/arm_raise')
def arm_raise_instructions():
    # Renders the instructions file: elbow_inst.html
    return render_template('arm_raise_inst.html')

@app.route('/start-exercises/arm_raise')
def start_arm_raise():
    # Renders the camera-based page elbow_flexion.html
    return render_template('arm_raise.html')
@app.route('/exercises/gait/gait_analysis')
def gait_analysis_instructions():
    # Renders the instructions file: elbow_inst.html
    return render_template('gait_analysis_inst.html')

@app.route('/start-exercises/gait_analysis')
def start_gait_analysis():
    # Renders the camera-based page elbow_flexion.html
    return render_template('gait_analysis.html')

# Error handlers
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True, port=5000)


