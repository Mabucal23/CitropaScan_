#!/usr/bin/python3
"""
Improved and hardened app.py for CitroPaScan

Environment variables to set (Render / production):
- FIREBASE_CREDENTIALS : JSON string of your service account (recommended)
- SERVICE_ACCOUNT_FILE : optional path to local serviceAccountKey.json (used when FIREBASE_CREDENTIALS not set)
- FIREBASE_WEB_API_KEY : your Firebase web API key (used for REST signin)
- FLASK_SECRET_KEY : secret key for Flask sessions
- MAIL_USERNAME, MAIL_PASSWORD : credentials for sending mail (Gmail or other SMTP)
- FLASK_SERVER_URL : remote model server (PC ngrok) e.g. https://your-ngrok-url.ngrok.io
- PI_IP_ADDRESS : Raspberry Pi local IP (optional)
- MODEL_PATH : path to tf.keras model (optional)
- START_BACKGROUND_THREAD : "1" to start background thread (optional; helpful in dev)
"""

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
FLASK_SERVER_URL = os.getenv("FLASK_SERVER_URL", "https://semicaricatural-elizabet-proximal.ngrok-free.dev")
PI_IP_ADDRESS = os.getenv("PI_IP_ADDRESS", "192.168.254.109")
MODEL_PATH = os.getenv("MODEL_PATH", "accDiseases.keras")
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "serviceAccountKey.json")
FIREBASE_WEB_API_KEY = os.getenv("FIREBASE_WEB_API_KEY", "AIzaSyAVAZemmqpkBuAmEqppKQ0URWAIzAVP3m4")  # keep in env in prod
START_BACKGROUND_THREAD = os.getenv("START_BACKGROUND_THREAD", "1") == "1"

# Mail config from env (recommended)
MAIL_USERNAME = os.getenv("MAIL_USERNAME", "te.st.emmail02.gmail.com")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD", "testemail02")
MAIL_DEFAULT_SENDER = os.getenv("MAIL_DEFAULT_SENDER", MAIL_USERNAME)

# -------------------------
# Pi URLs
# -------------------------
PI_SERVO_URL = f"http://{PI_IP_ADDRESS}:5000/move_servo"
PI_SENSOR_URL = f"http://{PI_IP_ADDRESS}:5000/check_sensor"
# Note: The Pi's ngrok URL may be fetched by querying the Pi (if it exposes /ngrok_url)
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
db = None  # Firestore client (if available)
firebase_available = False
model = None
model_available = False

CLASS_NAMES = [
    "Black Spot", "Canker", "Fresh", "Greeening", "Scab"
]

# -------------------------
# Utilities
# -------------------------
def safe_print(*args, **kwargs):
    """Wrap prints so we can easily replace or augment later."""
    print(*args, **kwargs)

# -------------------------
# Firebase initialization (env-friendly)
# -------------------------
def init_firebase():
    global db, firebase_available
    try:
        firebase_json = os.getenv("FIREBASE_CREDENTIALS")
        if firebase_json:
            safe_print("🔒 Loading Firebase credentials from FIREBASE_CREDENTIALS env var.")
            cred_dict = json.loads(firebase_json)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            firebase_available = True
            safe_print("✅ Firebase initialized from env.")
            return
        # Fall back to local file if exists
        if os.path.exists(SERVICE_ACCOUNT_FILE):
            safe_print(f"🔒 Loading Firebase credentials from local file {SERVICE_ACCOUNT_FILE}.")
            cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            firebase_available = True
            safe_print("✅ Firebase initialized from local file.")
            return
        safe_print("⚠️ Firebase credentials not provided. Firebase features will be disabled.")
        firebase_available = False
    except Exception as e:
        safe_print("❌ Firebase initialization error:", e)
        traceback.print_exc()
        firebase_available = False

# initialize firebase at import/startup
init_firebase()

# -------------------------
# Mail Setup (from env for security)
# -------------------------
app.config['MAIL_SERVER'] = os.getenv("MAIL_SERVER", 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv("MAIL_PORT", 587))
app.config['MAIL_USE_TLS'] = os.getenv("MAIL_USE_TLS", "True").lower() in ("1", "true", "yes")
app.config['MAIL_USE_SSL'] = os.getenv("MAIL_USE_SSL", "False").lower() in ("1", "true", "yes")
app.config['MAIL_USERNAME'] = MAIL_USERNAME
app.config['MAIL_PASSWORD'] = MAIL_PASSWORD
app.config['MAIL_DEFAULT_SENDER'] = MAIL_DEFAULT_SENDER

mail = Mail(app)

# -------------------------
# Model loading (graceful)
# -------------------------
def load_model_safe(path):
    global model, model_available
    try:
        if os.path.exists(path):
            model = tf.keras.models.load_model(path)
            model_available = True
            safe_print(f"✅ Model loaded from {path}")
        else:
            safe_print(f"⚠️ Model file {path} not found. Model-based features will be disabled.")
            model_available = False
    except Exception as e:
        safe_print(f"❌ Failed to load model {path}: {e}")
        traceback.print_exc()
        model_available = False

load_model_safe(MODEL_PATH)

# -------------------------
# Camera init with failover
# -------------------------
def initialize_camera_with_failover(sources):
    cam = None
    for source in sources:
        try:
            safe_print(f"Attempting to connect to camera/source: {source}")
            cap = cv2.VideoCapture(source)
            time.sleep(0.8)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    safe_print(f"✅ Connected to camera source: {source}")
                    cam = cap
                    break
                else:
                    cap.release()
                    safe_print(f"⚠️ Camera opened but could not read a frame from: {source}")
            else:
                safe_print(f"⚠️ VideoCapture could not open source: {source}")
        except Exception as e:
            safe_print(f"❌ Exception while opening camera source {source}: {e}")
    if cam is None:
        safe_print("❌ CRITICAL: Could not connect to ANY camera source.")
    return cam

def get_pi_video_source():
    # If we have a PI_NGROK_URL, return its /video_feed route; otherwise fallback to local camera index 0
    with PI_NGROK_LOCK:
        if PI_NGROK_URL:
            return f"{PI_NGROK_URL}/video_feed"
    return 0

# Try to update pi ngrok URL from Pi (non-critical routine)
def update_pi_ngrok_url():
    global PI_NGROK_URL
    try:
        resp = requests.get(f"http://{PI_IP_ADDRESS}:5000/ngrok_url", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            with PI_NGROK_LOCK:
                PI_NGROK_URL = data.get("ngrok_url")
            safe_print(f"✅ Updated Pi ngrok URL: {PI_NGROK_URL}")
    except Exception as e:
        safe_print(f"❌ Could not fetch Pi ngrok URL: {e}")

# Build list of video sources to try
video_sources = [get_pi_video_source(), 0, 1]  # try pi feed, then local indexes

camera = initialize_camera_with_failover(video_sources)

# -------------------------
# Frame capture helper
# -------------------------
def capture_frame_bytes():
    """Return a copy of the last frame (numpy array) or None."""
    global g_last_frame
    with g_frame_lock:
        if g_last_frame is None:
            return None
        try:
            return g_last_frame.copy()
        except Exception:
            return None

# -------------------------
# Servo control
# -------------------------
def send_servo_command(direction):
    """Sends a move command to the Raspberry Pi. Tries PI_NGROK_URL then PI_SERVO_URL."""
    try:
        # Try to use PI_NGROK_URL first if known
        with PI_NGROK_LOCK:
            if PI_NGROK_URL:
                url = f"{PI_NGROK_URL}/move_servo"
                try:
                    requests.post(url, json={"direction": direction}, timeout=2.0)
                    return
                except Exception as e:
                    safe_print(f"⚠️ Could not reach PI_NGROK_URL servo endpoint ({url}): {e}")
        # Fall back to direct IP endpoint
        try:
            requests.post(PI_SERVO_URL, json={"direction": direction}, timeout=2.0)
            return
        except Exception as e:
            safe_print(f"❌ Servo error to {PI_SERVO_URL}: {e}")
    except Exception as e:
        safe_print(f"❌ Unexpected send_servo_command error: {e}")

# -------------------------
# Request local PC model (via FLASK_SERVER_URL)
# -------------------------
def request_model_prediction(image_bytes):
    try:
        response = requests.post(
            f"{FLASK_SERVER_URL}/predict",
            files={'image': ('frame.jpg', image_bytes, 'image/jpeg')},
            timeout=6
        )
        return response.json()
    except Exception as e:
        safe_print("Prediction request error:", e)
        return {"result": "error"}

# -------------------------
# Image preprocess for local model
# -------------------------
def preprocess_frame(frame):
    """Resize and normalize frame for prediction"""
    try:
        target_size = (256, 256)
        img = cv2.resize(frame, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img
    except Exception as e:
        safe_print("❌ preprocess_frame error:", e)
        return None

# -------------------------
# Background prediction thread
# -------------------------
def background_prediction_thread():
    global g_last_frame, g_last_prediction, g_current_user_id, db

    safe_print("🚀 Background Smart Detection thread started.")

    while True:
        # if model or camera not available, sleep and continue
        if not model_available or camera is None:
            time.sleep(1.0)
            continue

        # 1. Check sensor on Pi
        try:
            resp = requests.get(PI_SENSOR_URL, timeout=0.5)
            sensor_data = resp.json()
            if not sensor_data.get("detected"):
                time.sleep(0.1)
                continue
            safe_print(f"👀 Object detected at {sensor_data.get('distance_cm', 'unknown')}cm")
        except Exception:
            time.sleep(0.2)
            continue

        # 2. Capture frame
        frame_to_process = capture_frame_bytes()
        if frame_to_process is None:
            time.sleep(0.1)
            continue

        try:
            img = preprocess_frame(frame_to_process)
            if img is None:
                time.sleep(0.1)
                continue
            preds = model.predict(img, verbose=0)
            predicted_class = CLASS_NAMES[int(np.argmax(preds))]
            confidence = float(np.max(preds) * 100.0)
            with g_prediction_lock:
                g_last_prediction = {"class": predicted_class, "confidence": confidence}

            # Move servo
            if predicted_class.lower() == "fresh":
                safe_print("Prediction: Fresh. Moving servo RIGHT.")
                send_servo_command("right")
                time.sleep(1.0)
            else:
                safe_print(f"Prediction: {predicted_class}. Moving servo LEFT.")
                send_servo_command("left")
                time.sleep(5.0)

            # Return to center
            safe_print("Returning servo to CENTER.")
            send_servo_command("center")

            # Save to Firebase if enabled
            if firebase_available and db and g_current_user_id:
                try:
                    timestamp_obj = datetime.now()
                    data = {
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "timestamp": timestamp_obj,
                        "user_id": g_current_user_id
                    }
                    db.collection("predictions").add(data)
                    db.collection("users").document(g_current_user_id).collection("scans").add({
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "timestamp": timestamp_obj
                    })
                    safe_print(f"✅ Saved to Firebase for user {g_current_user_id}")
                except Exception as e:
                    safe_print("❌ Firebase save error:", e)
        except Exception as e:
            safe_print("❌ Prediction error:", e)
            traceback.print_exc()

        # cooldown before next detection loop
        time.sleep(0.2)

# -------------------------
# Video streaming generator
# -------------------------
def generate_frames(user_id):
    global g_last_frame, g_last_prediction, g_current_user_id
    g_current_user_id = user_id

    if camera is None:
        # Return nothing if camera not available
        return

    while True:
        success, frame = camera.read()
        if not success:
            safe_print("⚠️ Camera read failed; attempting to reinitialize.")
            # Try a re-init sequence
            try:
                camera.release()
            except:
                pass
            time.sleep(1.0)
            # attempt reinit once
            new_cam = initialize_camera_with_failover(video_sources)
            if new_cam:
                # replace global camera reference
                global camera
                camera = new_cam
            else:
                # stop generator
                break
            continue

        with g_frame_lock:
            g_last_frame = frame.copy()

        with g_prediction_lock:
            prediction = g_last_prediction.copy()

        try:
            cv2.putText(frame,
                        f"{prediction['class']} ({prediction['confidence']:.1f}%)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception:
            pass

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        # Limit frame rate
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

@app.route("/dashboard")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    user_id = session["user"]["uid"]

    # Fetch latest scan documents (safe if no firebase)
    predictions = []
    total_scans = 0
    fresh_count = 0
    if firebase_available and db:
        try:
            scans_ref = db.collection("users").document(user_id).collection("scans").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(500)
            docs = scans_ref.stream()
            for doc in docs:
                d = doc.to_dict()
                predictions.append(d)
            total_scans = len(predictions)
            fresh_count = sum(1 for p in predictions if p.get("predicted_class", "").lower() == "fresh")
        except Exception as e:
            safe_print("❌ Error fetching scans for dashboard:", e)
    else:
        safe_print("ℹ️ Firebase not available - dashboard counts will be empty.")

    diseased_count = total_scans - fresh_count
    good_percentage = (fresh_count / total_scans * 100) if total_scans > 0 else 0
    diseased_percentage = (diseased_count / total_scans * 100) if total_scans > 0 else 0

    return render_template(
        "index.html",
        total_scans=total_scans,
        healthy_count=fresh_count,
        diseased_count=diseased_count,
        good_percentage=round(good_percentage, 2),
        diseased_percentage=round(diseased_percentage, 2),
    )

@app.route("/api/counters")
def get_counters():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session["user"]["uid"]

    total_scans = 0
    fresh_count = 0
    if firebase_available and db:
        try:
            scans_ref = db.collection("users").document(user_id).collection("scans")
            docs = scans_ref.stream()
            rows = [d.to_dict() for d in docs]
            total_scans = len(rows)
            fresh_count = sum(1 for r in rows if r.get("predicted_class", "").lower() == "fresh")
        except Exception as e:
            safe_print("❌ Error getting counters:", e)
            return jsonify({"error": "Could not retrieve counters"}), 500
    else:
        safe_print("ℹ️ Firebase not available - returning zeroed counters.")

    diseased_count = total_scans - fresh_count
    good_percentage = round((fresh_count / total_scans * 100), 2) if total_scans > 0 else 0
    diseased_percentage = round((diseased_count / total_scans * 100), 2) if total_scans > 0 else 0

    return jsonify({
        "total_scans": total_scans,
        "healthy_count": fresh_count,
        "diseased_count": diseased_count,
        "good_percentage": good_percentage,
        "diseased_percentage": diseased_percentage
    })

@app.route("/services")
def services():
    return render_template("services.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/learnmore")
def learnmore():
    return render_template("learnmore.html")

@app.route("/report")
def report():
    if "user" not in session:
        return redirect(url_for("login"))

    user_id = session["user"]["uid"]
    predictions = []
    disease_counts = {}
    monthly_counts = {}
    total_confidence = 0
    total_scans = 0
    fresh_count = 0
    disease_count = 0

    if firebase_available and db:
        try:
            scans_ref = db.collection("users").document(user_id).collection("scans").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(100)
            docs = scans_ref.stream()
            for idx, doc in enumerate(docs, start=1):
                data = doc.to_dict()
                disease = data.get("predicted_class", "Unknown")
                confidence = float(data.get("confidence", 0))
                fs_timestamp = data.get("timestamp")
                timestamp = datetime.now()
                try:
                    if hasattr(fs_timestamp, "to_datetime"):
                        timestamp = fs_timestamp.to_datetime()
                except Exception:
                    pass

                predictions.append({
                    "id": idx,
                    "disease": disease,
                    "confidence": confidence,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "fresh" if disease.lower() == "fresh" else "disease"
                })
                total_scans += 1
                total_confidence += confidence
                if disease.lower() == "fresh":
                    fresh_count += 1
                else:
                    disease_count += 1
                disease_counts[disease] = disease_counts.get(disease, 0) + 1
                month = timestamp.strftime("%b")
                monthly_counts[month] = monthly_counts.get(month, 0) + 1
        except Exception as e:
            safe_print("❌ Error building report:", e)
            traceback.print_exc()
    else:
        safe_print("ℹ️ Firebase not available - report will be empty.")

    avg_confidence = round(total_confidence / total_scans, 2) if total_scans > 0 else 0

    return render_template("report.html",
                           predictions=predictions,
                           total_scans=total_scans,
                           healthy_count=fresh_count,
                           disease_count=disease_count,
                           avg_confidence=avg_confidence,
                           disease_counts=disease_counts,
                           monthly_counts=monthly_counts)

# -------------------------
# Profile & Upload
# -------------------------
UPLOAD_FOLDER_RELATIVE = "uploads"
UPLOAD_FOLDER_FULL = os.path.join("static", UPLOAD_FOLDER_RELATIVE)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER_FULL
os.makedirs(UPLOAD_FOLDER_FULL, exist_ok=True)

@app.route("/upload_profile_pic", methods=["POST"])
def upload_profile_pic():
    if "user" not in session:
        return redirect(url_for("login"))
    file = request.files.get("profile_pic")
    if file:
        filename = secure_filename(file.filename)
        user_id = session["user"]["uid"]
        unique_filename = f"{user_id}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(filepath)
        web_filepath_relative = os.path.join(UPLOAD_FOLDER_RELATIVE, unique_filename).replace("\\", "/")
        if firebase_available and db:
            try:
                db.collection("users").document(user_id).update({"profile_pic": web_filepath_relative})
            except Exception as e:
                safe_print("❌ Error updating profile pic in Firestore:", e)
        session["user"]["profile_pic"] = web_filepath_relative
        session.modified = True
    return redirect(url_for("profile"))

@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect(url_for("login"))
    user_id = session["user"]["uid"]
    user_data = {"fullname": "Unknown", "email": "Not Found", "phone": "", "address": ""}
    if firebase_available and db:
        try:
            user_ref = db.collection("users").document(user_id)
            user_doc = user_ref.get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                user_data['phone'] = user_data.get('phone', '')
                user_data['address'] = user_data.get('address', '')
                db_last_login = user_data.get("last_login")
                if db_last_login:
                    try:
                        if hasattr(db_last_login, "to_datetime"):
                            user_data["last_login"] = db_last_login.to_datetime()
                    except:
                        user_data["last_login"] = session["user"].get("last_login")
        except Exception as e:
            safe_print("❌ Error reading profile from Firestore:", e)

    session["user"]["profile_pic"] = user_data.get("profile_pic")
    session["user"]["phone"] = user_data.get("phone")
    session["user"]["address"] = user_data.get("address")
    session.modified = True

    scan_history = []
    if firebase_available and db:
        try:
            scans = db.collection("users").document(user_id).collection("scans").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(5).stream()
            for scan in scans:
                data = scan.to_dict()
                fs_timestamp = data.get("timestamp")
                py_datetime = None
                if fs_timestamp and hasattr(fs_timestamp, "to_datetime"):
                    try:
                        py_datetime = fs_timestamp.to_datetime()
                    except:
                        py_datetime = None
                scan_history.append({
                    "disease": data.get("predicted_class"),
                    "confidence": data.get("confidence"),
                    "timestamp": py_datetime,
                })
        except Exception as e:
            safe_print("❌ Error fetching scan history:", e)

    return render_template("profile.html", user=user_data, scan_history=scan_history)

@app.route("/update_profile", methods=["POST"])
def update_profile():
    if "user" not in session:
        return redirect(url_for("login"))
    user_id = session["user"]["uid"]
    fullname = request.form.get("fullname")
    email = request.form.get("email")
    phone = request.form.get("phone")
    address = request.form.get("address")
    try:
        current_email = session["user"]["email"]
        if email != current_email:
            if firebase_available:
                auth.update_user(user_id, email=email)
            flash("Login email updated. Please check your inbox to verify.", "info")
        update_data = {"fullname": fullname, "email": email, "phone": phone, "address": address}
        if firebase_available and db:
            db.collection("users").document(user_id).update(update_data)
        session["user"].update({"email": email, "fullname": fullname, "phone": phone, "address": address})
        session.modified = True
        flash("Profile updated successfully!", "success")
    except Exception as e:
        safe_print("❌ Update profile failed:", e)
        flash(f"Update failed: {str(e)}", "error")
    return redirect(url_for("profile"))

# -------------------------
# Auth Routes (Signin via REST)
# -------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    if request.method == "POST":
        fullname = request.form.get("fullname")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        if not fullname or not email or not password or not confirm_password:
            return render_template("signup.html", error="All fields are required.")
        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match.")
        try:
            if firebase_available:
                user = auth.create_user(email=email, password=password, display_name=fullname)
                created_at_obj = datetime.now()
                db.collection("users").document(user.uid).set({
                    "fullname": fullname,
                    "email": email,
                    "created_at": created_at_obj,
                    "is_logged_in": False,
                    "phone": "",
                    "address": ""
                })
                session["user"] = {"uid": user.uid, "email": email, "fullname": fullname, "last_login": created_at_obj, "phone": "", "address": ""}
                safe_print(f"✅ New user saved: {session['user']}")
                return redirect(url_for("index"))
            else:
                error = "Auth is not available (Firebase not initialized)."
                return render_template("signup.html", error=error)
        except Exception as e:
            safe_print("❌ Signup error:", e)
            return render_template("signup.html", error=f"Signup failed: {str(e)}")
    return render_template("signup.html", error=error)

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        try:
            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
            payload = {"email": email, "password": password, "returnSecureToken": True}
            r = requests.post(url, json=payload, timeout=6)
            data = r.json()
            if "error" in data:
                error_message = data["error"]["message"]
                if error_message in ("INVALID_PASSWORD", "Wrong Password"):
                    error = "❌ Wrong password. Please try again."
                elif error_message in ("EMAIL_NOT_FOUND", "Email does not exist"):
                    error = "❌ Email not found. Please sign up first."
                else:
                    error = f"Login failed: {error_message}"
                return render_template("login.html", error=error)
            if not all(k in data for k in ("localId", "email", "idToken")):
                return render_template("login.html", error="❌ Invalid login response. Please try again.")
            user_id = data["localId"]
            user_ref = None
            user_doc = None
            if firebase_available and db:
                user_ref = db.collection("users").document(user_id)
                user_doc = user_ref.get()
            last_login_obj = datetime.now()
            user_data = user_doc.to_dict() if user_doc and user_doc.exists else {}
            session["user"] = {"uid": user_id, "email": data["email"], "idToken": data["idToken"], "last_login": last_login_obj, "fullname": user_data.get("fullname", "User"), "profile_pic": user_data.get("profile_pic")}
            if firebase_available and db:
                user_ref.set({"is_logged_in": True, "last_login": last_login_obj}, merge=True)
            return redirect(url_for("index"))
        except Exception as e:
            safe_print("❌ Login failed:", e)
            return render_template("login.html", error=f"Login failed: {str(e)}")
    return render_template("login.html", error=error)

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email")
        if not email:
            flash("Please enter your email address.", "error")
            return redirect(url_for('forgot_password'))
        try:
            if not firebase_available:
                flash("Password reset not available (Firebase not initialized).", "error")
                return redirect(url_for('login'))
            user = auth.get_user_by_email(email)
            link = auth.generate_password_reset_link(email)
            try:
                msg = Message(subject="Reset Your CitroPaScan Password", recipients=[email], html=f"""
                    <p>Hello,</p>
                    <p>You requested a password reset for your CitroPaScan account.</p>
                    <p>Click the link below to set a new password:</p>
                    <p><a href="{link}">Reset Password</a></p>
                    <p>If you did not request this, please ignore this email.</p>
                """)
                mail.send(msg)
                flash("Password reset email sent. Please check your inbox (and spam folder).", "success")
            except Exception as mail_error:
                safe_print("❌ Mail sending error:", mail_error)
                flash("Could not send the reset email. Please try again later or contact support.", "error")
            return redirect(url_for('login'))
        except auth.UserNotFoundError:
            flash("If an account exists for that email, a reset link has been sent.", "info")
            return redirect(url_for('login'))
        except Exception as e:
            safe_print("❌ Password reset error:", e)
            flash(f"An error occurred: {e}", "error")
            return redirect(url_for('forgot_password'))
    return render_template("forget_password.html")

@app.route("/logout")
def logout():
    global g_current_user_id
    if "user" in session:
        user_id = session["user"]["uid"]
        if firebase_available and db:
            try:
                db.collection("users").document(user_id).update({"is_logged_in": False})
            except Exception as e:
                safe_print("❌ Error writing logout state to Firestore:", e)
        session.pop("user", None)
        g_current_user_id = None
    return redirect(url_for("login"))

# -------------------------
# Optional /predict_and_sort endpoint (calls external model server)
# -------------------------
@app.route("/predict_and_sort", methods=["POST"])
def predict_and_sort():
    try:
        frame = capture_frame_bytes()
        if frame is None:
            return jsonify({"status": "error", "message": "No frame captured"}), 400

        # Convert frame to JPEG bytes
        ret, buf = cv2.imencode(".jpg", frame)
        if not ret:
            return jsonify({"status": "error", "message": "Could not encode frame"}), 500
        result = request_model_prediction(buf.tobytes())
        safe_print("Model response:", result)
        prediction = result.get("result", "unknown")
        if prediction == "good":
            send_servo_command("right")
            return jsonify({"status": "sorted", "direction": "right"})
        elif prediction == "diseased":
            send_servo_command("left")
            return jsonify({"status": "sorted", "direction": "left"})
        else:
            return jsonify({"status": "error", "message": "Unknown prediction"})
    except Exception as e:
        safe_print("❌ predict_and_sort error:", e)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    try:
        if START_BACKGROUND_THREAD:
            pred_thread = threading.Thread(target=background_prediction_thread, daemon=True)
            pred_thread.start()
        safe_print("Starting Flask app at http://0.0.0.0:5000")
        app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        safe_print("❌ App failed to start:", e)
        traceback.print_exc()
