import threading
import time
import io
import cv2
import logging
from flask import Flask, Response, render_template_string

logger = logging.getLogger(__name__)

# Global variable to store the latest frame
# In a more complex app, we might use a proper Queue or shared memory,
# but for a simple debug viewer, this is sufficient.
latest_frames = {}
lock = threading.Lock()

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>OpenPI OMY Viewer</title>
    <style>
        body { font-family: sans-serif; text-align: center; background: #222; color: #fff; }
        .container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }
        .camera-box { border: 2px solid #444; padding: 10px; background: #333; border-radius: 8px; }
        h1 { margin-bottom: 20px; }
        h3 { margin: 0 0 10px 0; color: #aaa; }
        img { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <h1>OpenPI Robotis OMY Viewer</h1>
    <div class="container">
        {% for cam_name in camera_names %}
        <div class="camera-box">
            <h3>{{ cam_name }}</h3>
            <img src="/video_feed/{{ cam_name }}" width="640" height="480">
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""

def update_frame(name, frame):
    """Updates the latest frame for a specific camera."""
    with lock:
        latest_frames[name] = frame

def generate_mjpeg(cam_name):
    """Generator function for MJPEG stream."""
    while True:
        with lock:
            frame = latest_frames.get(cam_name)
        
        if frame is None:
            time.sleep(0.1)
            continue
            
        # Ensure frame is BGR for encoding (OpenCV default) or handle RGB
        # Assumes input frame is already in BGR or compatible format for encoding
        
        # JPEG Encode
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03) # Limit to ~30fps loop check

@app.route('/')
def index():
    with lock:
        camera_names = list(latest_frames.keys())
    # If no frames yet, show a placeholder or empty list
    return render_template_string(HTML_TEMPLATE, camera_names=camera_names)

@app.route('/video_feed/<cam_name>')
def video_feed(cam_name):
    return Response(generate_mjpeg(cam_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def start_viewer(port=5000):
    """Starts the Flask server in a daemon thread."""
    def run():
        # run with use_reloader=False to avoid issues in threads/ros
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    
    t = threading.Thread(target=run, daemon=True)
    t.start()
    logger.info(f"Viewer started at http://0.0.0.0:{port}")
