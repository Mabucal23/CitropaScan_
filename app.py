#!/usr/bin/python3

from flask import Flask, render_template, redirect, url_for, session, Response, jsonify, request, flash
import cv2
import tensorflow as tf
import numpy as np
import firebase_admin
from firebase_admin import credentials, auth, firestore
import requests
import os
from werkzeug.utils import secure_filename
import time
from datetime import datetime
from flask_mail import Mail, Message
import threading
import json
import traceback

# -------------------------
# Config via environment
# -------------------------

FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "citropascan_secret_key_dev")
FLASK_SERVER_URL = os.getenv("FLASK_SERVER_URL", "http://localhost:5000")
PI_IP_ADDRESS = os.getenv("PI_IP_ADDRESS", "192.168.254.109")
MODEL_PATH = os.getenv("MODEL_PATH", "accDiseases.keras")
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "serviceAccountKey.json")
FIREBASE_WEB_API_KEY = os.getenv("FIREBASE_WEB_API_KEY", "")
START_BACKGROUND_THREAD = os.getenv("START_BACKGROUND_THREAD", "1") == "1"

MAIL_USERNAME = os.getenv("MAIL_USERNAME", "")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD", "")
MAIL_DEFAULT_SENDER = os.getenv("MAIL_DEFAULT_SENDER", MAIL_USERNAME)

# -------------------------
# Pi URLs
# -------------------------

PI_SERVO_URL = f"http://{PI_IP_ADDRESS}:5000/move_servo"
PI_SENSOR_URL = f"http://{PI_IP_ADDRESS}:5000/check_sensor"
PI_NGROK_URL = None
PI_NGROK_LOCK = threading.Lock()

# Prediction/upload intervals
PREDICTION_INTERVAL = 1.0
UPLOAD_INTERVAL = 20.0

# -------------------------
# Flask App
# -------------------------

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# -------------------------
# Globals
# -------------------------

camera = None
g_last_frame = None
g_frame_lock = threading.Lock()
g_last_prediction = {"class": "Loading...", "confidence": 0.0}
g_prediction_lock = threading.Lock()
g_current_user_id = None
db = None
firebase_available = False
model = None
model_available = False

CLASS_NAMES = ["Black Spot", "Canker", "Fresh", "Greening", "Scab"]

# -------------------------
# Utilities
# -------------------------

def safe_print(*args, **kwargs):
    print(*args, **kwargs)

# -------------------------
# Firebase initialization
# -------------------------

def init_firebase():
    global db, firebase_available
    try:
        firebase_json = os.getenv("FIREBASE_CREDENTIALS")
        if firebase_json:
            cred_dict = json.loads(firebase_json)
            if "private_key" in cred_dict:
                cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            firebase_available = True
            safe_print("✅ Firebase initialized from ENV.")
            return
        if os.path.exists(SERVICE_ACCOUNT_FILE):
            cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            firebase_available = True
            safe_print("✅ Firebase initialized from local file.")
            return
        safe_print("⚠️ Firebase credentials not found. Features disabled.")
        firebase_available = False
    except Exception as e:
        safe_print("❌ Firebase init error:", e)
        traceback.print_exc()
        firebase_available = False

init_firebase()

# -------------------------
# Mail Setup
# -------------------------

app.config.update(
    MAIL_SERVER=os.getenv("MAIL_SERVER", "smtp.gmail.com"),
    MAIL_PORT=int(os.getenv("MAIL_PORT", 587)),
    MAIL_USE_TLS=os.getenv("MAIL_USE_TLS", "True").lower() in ("1", "true", "yes"),
    MAIL_USE_SSL=os.getenv("MAIL_USE_SSL", "False").lower() in ("1", "true", "yes"),
    MAIL_USERNAME=MAIL_USERNAME,
    MAIL_PASSWORD=MAIL_PASSWORD,
    MAIL_DEFAULT_SENDER=MAIL_DEFAULT_SENDER
)
mail = Mail(app)

# -------------------------
# Load model
# -------------------------

def load_model_safe(path):
    global model, model_available
    try:
        if os.path.exists(path):
            model = tf.keras.models.load_model(path)
            model_available = True
            safe_print(f"✅ Model loaded: {path}")
        else:
            safe_print(f"⚠️ Model not found: {path}")
            model_available = False
    except Exception as e:
        safe_print(f"❌ Failed to load model: {e}")
        traceback.print_exc()
        model_available = False

load_model_safe(MODEL_PATH)

# -------------------------
# Camera init
# -------------------------

def initialize_camera_with_failover(sources):
    cam = None
    for source in sources:
        try:
            safe_print(f"Trying camera source: {source}")
            cap = cv2.VideoCapture(source)
            time.sleep(0.5)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    safe_print(f"✅ Connected to camera: {source}")
                    cam = cap
                    break
            cap.release()
        except Exception as e:
            safe_print(f"❌ Camera error {source}: {e}")
    if cam is None:
        safe_print("❌ No camera available.")
    return cam

def get_pi_video_source():
    with PI_NGROK_LOCK:
        return f"{PI_NGROK_URL}/video_feed" if PI_NGROK_URL else 0

def update_pi_ngrok_url():
    global PI_NGROK_URL
    try:
        resp = requests.get(f"http://{PI_IP_ADDRESS}:5000/ngrok_url", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            with PI_NGROK_LOCK:
                PI_NGROK_URL = data.get("ngrok_url")
            safe_print(f"✅ Updated Pi ngrok URL: {PI_NGROK_URL}")
    except Exception as e:
        safe_print(f"❌ Could not fetch Pi ngrok URL: {e}")

camera = initialize_camera_with_failover([get_pi_video_source(), 0, 1])

# -------------------------
# Capture frame
# -------------------------

def capture_frame_bytes():
    global g_last_frame
    with g_frame_lock:
        return g_last_frame.copy() if g_last_frame is not None else None

# -------------------------
# Servo control
# -------------------------

def send_servo_command(direction):
    try:
        with PI_NGROK_LOCK:
            if PI_NGROK_URL:
                try:
                    requests.post(f"{PI_NGROK_URL}/move_servo", json={"direction": direction}, timeout=2)
                    return
                except Exception as e:
                    safe_print(f"⚠️ PI_NGROK servo error: {e}")
            requests.post(PI_SERVO_URL, json={"direction": direction}, timeout=2)
    except Exception as e:
        safe_print(f"❌ Servo command error: {e}")

# -------------------------
# Frame preprocessing
# -------------------------

def preprocess_frame(frame):
    try:
        img = cv2.resize(frame, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        return np.expand_dims(img, axis=0).astype(np.float32)
    except Exception as e:
        safe_print("❌ preprocess_frame error:", e)
        return None

# -------------------------
# Background prediction thread
# -------------------------

def background_prediction_thread():
    global g_last_frame, g_last_prediction, g_current_user_id, db
    safe_print("🚀 Prediction thread started.")
    while True:
        if not model_available or camera is None:
            time.sleep(1)
            continue
        try:
            resp = requests.get(PI_SENSOR_URL, timeout=0.5)
            sensor_data = resp.json()
            if not sensor_data.get("detected"):
                time.sleep(0.1)
                continue
        except:
            time.sleep(0.2)
            continue

        frame = capture_frame_bytes()
        if frame is None:
            time.sleep(0.1)
            continue
        img = preprocess_frame(frame)
        if img is None:
            time.sleep(0.1)
            continue
        try:
            preds = model.predict(img, verbose=0)
            pred_class = CLASS_NAMES[int(np.argmax(preds))]
            confidence = float(np.max(preds) * 100)
            with g_prediction_lock:
                g_last_prediction = {"class": pred_class, "confidence": confidence}
            # Servo movement
            send_servo_command("right" if pred_class.lower() == "fresh" else "left")
            time.sleep(1 if pred_class.lower() == "fresh" else 5)
            send_servo_command("center")
            # Save to Firebase
            if firebase_available and db and g_current_user_id:
                timestamp_obj = datetime.now()
                data = {
                    "predicted_class": pred_class,
                    "confidence": confidence,
                    "timestamp": timestamp_obj,
                    "user_id": g_current_user_id
                }
                db.collection("predictions").add(data)
                db.collection("users").document(g_current_user_id).collection("scans").add(data)
        except Exception as e:
            safe_print("❌ Prediction error:", e)
            traceback.print_exc()
        time.sleep(0.2)

# -------------------------
# Video streaming
# -------------------------

def generate_frames(user_id):
    global g_last_frame, g_last_prediction, g_current_user_id
    g_current_user_id = user_id
    if camera is None:
        return
    while True:
        success, frame = camera.read()
        if not success:
            safe_print("⚠️ Camera read failed")
            time.sleep(1)
            continue
        with g_frame_lock:
            g_last_frame = frame.copy()
        with g_prediction_lock:
            pred = g_last_prediction.copy()
        try:
            cv2.putText(
                frame,
                f"{pred['class']} ({pred['confidence']:.1f}%)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        except:
            pass
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() +
               b'\r\n')
        time.sleep(1/30)

# -------------------------
# Routes
# -------------------------

@app.route('/video_feed')
def video_feed():
    if "user" not in session:
        return redirect(url_for("login"))
    if camera is None:
        return "Camera not available", 503
    return Response(generate_frames(session["user"]["uid"]),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def landing():
    return render_template("landing.html")

# -------------------------
# Run app
# -------------------------

if __name__ == "__main__":
    if START_BACKGROUND_THREAD:
        threading.Thread(target=background_prediction_thread, daemon=True).start()
    safe_print("Starting Flask app on 0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
