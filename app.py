from flask import Flask, render_template, redirect, url_for, session, Response, jsonify, request, flash
import cv2
import tensorflow as tf
import numpy as np
import firebase_admin
from firebase_admin import credentials, auth, firestore
import requests
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import time
from datetime import datetime
from flask_mail import Mail, Message
import threading

# Load local environment variables (must be first)
load_dotenv()

REQUIRED_ENV_VARS = [
    "GOOGLE_CREDENTIALS_JSON",
    "FIREBASE_API_KEY",
    "FLASK_SECRET_KEY",
    "MAIL_SERVER",
    "MAIL_PORT",
    "MAIL_USE_TLS",
    "MAIL_USE_SSL",
    "MAIL_USERNAME",
    "MAIL_PASSWORD",
    "MAIL_DEFAULT_SENDER",
    "PI_IP_ADDRESS"
]

missing_vars = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
if missing_vars:
    print(f"❌ CRITICAL ERROR: Missing environment variables: {missing_vars}")
    exit(1)

# =========================
# Flask app initialization
# =========================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")


# OpenCV capture option (example)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "stimeout;3000000"

# Raspberry Pi servo base IP (change to your Pi IP)
PI_IP_ADDRESS = os.environ.get("PI_IP_ADDRESS", "192.168.254.109")
PI_SERVO_URL = f"http://{PI_IP_ADDRESS}:5000/move_servo"

def send_servo_command(direction):
    """Send a move command to the Raspberry Pi (with timeout)."""
    try:
        requests.post(PI_SERVO_URL, json={"direction": direction}, timeout=2.0)
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not send servo command: {e}")

# Options
PREDICTION_INTERVAL = 1.0  # seconds (not currently used directly)
UPLOAD_INTERVAL = 20.0     # seconds



# Mail setup using env vars
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'True').lower() in ('true', '1', 't')
app.config['MAIL_USE_SSL'] = os.environ.get('MAIL_USE_SSL', 'False').lower() in ('true', '1', 't')
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER')

mail = Mail(app)

# Firebase setup using env var JSON
SERVICE_ACCOUNT_PATH = os.environ.get("GOOGLE_CREDENTIALS_JSON")
if not os.path.exists(SERVICE_ACCOUNT_PATH):
    print(f"❌ CRITICAL ERROR: serviceAccountKey.json not found at {SERVICE_ACCOUNT_PATH}")
    exit(1)

try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase initialized")
except Exception as e:
    print(f"❌ ERROR initializing Firebase: {e}")
    exit(1)

FIREBASE_WEB_API_KEY = os.environ.get("FIREBASE_API_KEY")
if not FIREBASE_WEB_API_KEY:
    print("❌ CRITICAL ERROR: Missing FIREBASE_API_KEY in .env")
    exit(1)


# Load ML model
MODEL_PATH = "accDiseases.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    print("Expected input shape:", model.input_shape)
except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to load Keras model from {MODEL_PATH}: {e}")
    model = None

CLASS_NAMES = [
    "Black Spot", "Canker", "Fresh", "Greening", "Scab"
]

# Camera initialization with failover
def initialize_camera_with_failover(sources, wait_seconds=1.0):
    """
    Try each source until one opens AND returns a frame.
    Returns cv2.VideoCapture or None.
    """
    for source in sources:
        try:
            print(f"Attempting to connect to camera: {source}")
            cam = cv2.VideoCapture(source)
            # give it a moment
            time.sleep(wait_seconds)
            if not cam.isOpened():
                print(f"❌ Failed to open camera connection: {source}")
                try:
                    cam.release()
                except Exception:
                    pass
                continue
            # test a read
            ret, _ = cam.read()
            if ret:
                print(f"✅ Success! Connected and receiving frames from: {source}")
                return cam
            else:
                print(f"❌ Connected but failed to read frame from: {source}")
                try:
                    cam.release()
                except Exception:
                    pass
        except Exception as e:
            print(f"❌ Error when trying source {source}: {e}")
    print("❌ CRITICAL: Could not connect to ANY camera source.")
    return None

video_sources = [
    f"http://{PI_IP_ADDRESS}:5000/video_feed",
    "http://10.32.181.201:5000/video_feed",
    0  # local camera fallback
]

camera = initialize_camera_with_failover(video_sources)

# Global vars for threading
g_last_frame = None
g_last_prediction = {"class": "Loading...", "confidence": 0.0}
g_frame_lock = threading.Lock()
g_prediction_lock = threading.Lock()
g_current_user_id = None
g_last_upload_time = 0

def preprocess_frame(frame):
    """Resize and normalize frame for prediction."""
    target_size = (256, 256)
    img = cv2.resize(frame, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Motion detection constants
MIN_MOTION_AREA = 5000
MOTION_THRESHOLD = 25

def background_prediction_thread():
    """
    Background thread:
    - detects motion
    - runs prediction when motion found
    - moves servo and optionally saves to Firestore (rate-limited)
    """
    global g_last_frame, g_last_prediction, g_current_user_id, g_last_upload_time

    print("🚀 Background prediction thread started with Motion Detection.")
    prev_gray = None

    while True:
        frame_to_process = None
        with g_frame_lock:
            if g_last_frame is not None:
                frame_to_process = g_last_frame.copy()

        if frame_to_process is None:
            time.sleep(0.5)
            continue

        try:
            gray = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_gray is None:
                prev_gray = gray
                # wait a short moment to get a second frame for comparison
                time.sleep(0.2)
                continue

            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            contours_info = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.findContours returns either (contours, hierarchy) or (image, contours, hierarchy)
            if len(contours_info) == 3:
                _, contours, _ = contours_info
            else:
                contours, _ = contours_info

            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) > MIN_MOTION_AREA:
                    motion_detected = True
                    break

            # update baseline
            prev_gray = gray

            if not motion_detected:
                time.sleep(0.2)
                continue

            print("📸 Motion Detected! Running Prediction...")

            img = preprocess_frame(frame_to_process)

            if model is None:
                print("❌ Model not loaded, skipping prediction.")
                time.sleep(1.0)
                continue

            preds = model.predict(img, verbose=0)
            predicted_class = CLASS_NAMES[np.argmax(preds)]
            confidence = float(np.max(preds) * 100.0)

            with g_prediction_lock:
                g_last_prediction = {"class": predicted_class, "confidence": confidence}

            # Servo actions
            if predicted_class.lower() == "fresh":
                print("Prediction: Fresh. Moving servo RIGHT.")
                send_servo_command('right')
                time.sleep(1.0)
            else:
                print(f"Prediction: {predicted_class}. Moving servo LEFT.")
                send_servo_command('left')
                print("Waiting 5 seconds for disease sort...")
                time.sleep(5.0)

            # Return center
            print("Returning servo to CENTER.")
            send_servo_command('center')

            # Firestore saving (rate-limited)
            current_time = time.time()
            user_id = g_current_user_id

            if user_id and (current_time - g_last_upload_time >= UPLOAD_INTERVAL):
                timestamp_obj = datetime.now()
                data = {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "timestamp": timestamp_obj,
                    "user_id": user_id
                }
                try:
                    db.collection("predictions").add(data)
                    db.collection("users").document(user_id).collection("scans").add({
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "timestamp": timestamp_obj
                    })
                    print(f"✅ [BackgroundThread] Saved to Firebase for user {user_id}")
                    g_last_upload_time = current_time
                except Exception as e:
                    print(f"❌ [BackgroundThread] Firebase save error: {e}")

            # Reset motion baseline after servo movement to avoid false triggers
            print("🔄 Resetting motion detector to ignore servo movement...")
            time.sleep(1.0)
            with g_frame_lock:
                if g_last_frame is not None:
                    reset_gray = cv2.cvtColor(g_last_frame, cv2.COLOR_BGR2GRAY)
                    reset_gray = cv2.GaussianBlur(reset_gray, (21, 21), 0)
                    prev_gray = reset_gray
            print("✅ Motion detector reset. Ready for next fruit.")

        except Exception as e:
            print(f"❌ [BackgroundThread] Prediction error: {e}")

        # Rate limit loop iterations
        time.sleep(0.5)

def generate_frames(user_id):
    """
    Generator yields MJPEG frames. Updates global frame buffer used by predictor.
    """
    global g_last_frame, g_last_prediction, g_current_user_id, g_last_upload_time

    g_current_user_id = user_id
    g_last_upload_time = time.time()

    if camera is None or not camera.isOpened():
        print("❌ Camera is not available. Stopping frame generation.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            print("❌ Camera read failed. Stream may have ended.")
            break

        with g_frame_lock:
            g_last_frame = frame.copy()

        with g_prediction_lock:
            prediction = g_last_prediction.copy()

        # Draw label (fast)
        try:
            label = f"{prediction['class']} ({prediction['confidence']:.2f}%)"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"❌ DrawText error: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("❌ Failed to encode frame")
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(1/60)

@app.route('/video_feed')
def video_feed():
    if "user" not in session:
        return redirect(url_for("login"))
    user_id = session["user"]["uid"]
    return Response(generate_frames(user_id), mimetype='multipart/x-mixed-replace; boundary=frame')

# Routes (mostly unchanged)
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/dashboard")
def index():
    if "user" not in session:
        return redirect(url_for("login"))

    user_id = session["user"]["uid"]
    scans_ref = db.collection("users").document(user_id).collection("scans")
    try:
        docs = scans_ref.stream()
        predictions = [doc.to_dict() for doc in docs]
    except Exception as e:
        print(f"❌ Error reading scans: {e}")
        predictions = []

    total_scans = len(predictions)
    fresh_count = sum(1 for p in predictions if p.get("predicted_class", "").lower() == "fresh")
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
    scans_ref = db.collection("users").document(user_id).collection("scans")

    try:
        # Some firebase-admin versions don't support count() this way, so stream and count.
        docs = scans_ref.stream()
        total_scans = 0
        fresh_count = 0
        for doc in docs:
            total_scans += 1
            data = doc.to_dict()
            if data.get("predicted_class", "").lower() == "fresh":
                fresh_count += 1

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
    except Exception as e:
        print(f"❌ Error getting counters: {e}")
        return jsonify({"error": f"Could not retrieve counters: {str(e)}"}), 500

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
    scans_ref = (
        db.collection("users").document(user_id)
        .collection("scans")
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(100)
    )
    try:
        docs = scans_ref.stream()
    except Exception as e:
        print(f"❌ Error fetching report docs: {e}")
        docs = []

    predictions = []
    disease_counts = {}
    monthly_counts = {}
    total_confidence = 0
    total_scans = 0
    fresh_count = 0
    disease_count = 0

    for idx, doc in enumerate(docs, start=1):
        data = doc.to_dict()
        disease = data.get("predicted_class", "Unknown")
        confidence = float(data.get("confidence", 0))
        fs_timestamp = data.get("timestamp")
        timestamp = datetime.now()
        if fs_timestamp:
            try:
                timestamp = fs_timestamp.to_datetime()
            except Exception:
                try:
                    # If you saved plain datetime objects, they might already be python datetimes
                    if isinstance(fs_timestamp, datetime):
                        timestamp = fs_timestamp
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

    avg_confidence = round(total_confidence / total_scans, 2) if total_scans > 0 else 0

    return render_template("report.html",
                           predictions=predictions,
                           total_scans=total_scans,
                           healthy_count=fresh_count,
                           disease_count=disease_count,
                           avg_confidence=avg_confidence,
                           disease_counts=disease_counts,
                           monthly_counts=monthly_counts)

# Profile & Upload
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
        try:
            db.collection("users").document(user_id).update({
                "profile_pic": web_filepath_relative
            })
            session["user"]["profile_pic"] = web_filepath_relative
            session.modified = True
        except Exception as e:
            print(f"❌ Error updating profile_pic in DB: {e}")

    return redirect(url_for("profile"))

@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect(url_for("login"))

    user_id = session["user"]["uid"]
    user_ref = db.collection("users").document(user_id)
    try:
        user_doc = user_ref.get()
    except Exception as e:
        print(f"❌ Error retrieving user doc: {e}")
        user_doc = None

    if user_doc and user_doc.exists:
        user_data = user_doc.to_dict()
        user_data['phone'] = user_data.get('phone')
        user_data['address'] = user_data.get('address')
        db_last_login = user_data.get("last_login")
        if db_last_login:
            try:
                user_data["last_login"] = db_last_login.to_datetime()
            except Exception:
                user_data["last_login"] = session["user"].get("last_login")
        else:
            user_data["last_login"] = session["user"].get("last_login")
    else:
        user_data = {"fullname": "Unknown", "email": "Not Found", "phone": None, "address": None}

    session["user"]["profile_pic"] = user_data.get("profile_pic")
    session["user"]["phone"] = user_data.get("phone")
    session["user"]["address"] = user_data.get("address")
    session.modified = True

    scans_ref = user_ref.collection("scans")
    try:
        scans = scans_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(5).stream()
    except Exception as e:
        print(f"❌ Error fetching scans for profile: {e}")
        scans = []

    scan_history = []
    for scan in scans:
        data = scan.to_dict()
        fs_timestamp = data.get("timestamp")
        py_datetime = None
        if fs_timestamp:
            try:
                py_datetime = fs_timestamp.to_datetime()
            except Exception:
                try:
                    if isinstance(fs_timestamp, datetime):
                        py_datetime = fs_timestamp
                except Exception:
                    pass
        scan_history.append({
            "disease": data.get("predicted_class"),
            "confidence": data.get("confidence"),
            "timestamp": py_datetime,
        })

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
        if email and email != current_email:
            auth.update_user(user_id, email=email)
            flash("Login email updated. Please check your inbox to verify.", "info")

        update_data = {
            "fullname": fullname,
            "email": email,
            "phone": phone,
            "address": address
        }
        db.collection("users").document(user_id).update(update_data)

        session["user"]["email"] = email
        session["user"]["fullname"] = fullname
        session["user"]["phone"] = phone
        session["user"]["address"] = address
        session.modified = True

        flash("Profile updated successfully!", "success")
    except Exception as e:
        flash(f"Update failed: {str(e)}", "error")

    return redirect(url_for("profile"))

# Auth routes
@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    if request.method == "POST":
        fullname = request.form.get("fullname")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not fullname or not email or not password or not confirm_password:
            error = "All fields are required."
            return render_template("signup.html", error=error)

        if password != confirm_password:
            error = "Passwords do not match."
            return render_template("signup.html", error=error)

        try:
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
            session["user"] = {
                "uid": user.uid,
                "email": email,
                "fullname": fullname,
                "last_login": created_at_obj,
                "phone": "",
                "address": ""
            }
            print(f"✅ New user saved: {session['user']}")
            return redirect(url_for("index"))
        except Exception as e:
            error = f"Signup failed: {str(e)}"
            print("❌ Signup error:", e)
            return render_template("signup.html", error=error)

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
            r = requests.post(url, json=payload)
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
            user_ref = db.collection("users").document(user_id)
            user_doc = user_ref.get()

            if user_doc.exists and user_doc.to_dict().get("is_logged_in", False):
                print("Warning: User is already marked as logged in.")

            last_login_obj = datetime.now()
            user_data = user_doc.to_dict() or {}

            session["user"] = {
                "uid": user_id,
                "email": data["email"],
                "idToken": data["idToken"],
                "last_login": last_login_obj,
                "fullname": user_data.get("fullname", "User"),
                "profile_pic": user_data.get("profile_pic")
            }

            user_ref.set({"is_logged_in": True, "last_login": last_login_obj}, merge=True)
            return redirect(url_for("index"))

        except Exception as e:
            error = f"Login failed: {str(e)}"
            return render_template("login.html", error=error)

    return render_template("login.html", error=error)

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email")
        if not email:
            flash("Please enter your email address.", "error")
            return redirect(url_for('forgot_password'))

        try:
            user = auth.get_user_by_email(email)
            link = auth.generate_password_reset_link(email)
            try:
                msg = Message(
                    subject="Reset Your CitroPaScan Password",
                    recipients=[email],
                    html=f"""
                        <p>Hello,</p>
                        <p>You requested a password reset for your CitroPaScan account.</p>
                        <p>Click the link below to set a new password:</p>
                        <p><a href="{link}">Reset Password</a></p>
                        <p>If you did not request this, please ignore this email.</p>
                        <br>
                        <p>Thanks,</p>
                        <p>The CitroPaScan Team</p>
                    """
                )
                mail.send(msg)
                flash("Password reset email sent. Please check your inbox (and spam folder).", "success")
            except Exception as mail_error:
                print(f"❌ Mail sending error: {mail_error}")
                flash("Could not send the reset email. Please try again later or contact support.", "error")
            return redirect(url_for('login'))
        except Exception:
            # Do not reveal whether an email exists
            flash("If an account exists for that email, a reset link has been sent.", "info")
            return redirect(url_for('login'))

    return render_template("forget_password.html")

@app.route("/logout")
def logout():
    global g_current_user_id
    if "user" in session:
        user_id = session["user"]["uid"]
        try:
            db.collection("users").document(user_id).update({"is_logged_in": False})
        except Exception as e:
            print(f"❌ Error updating is_logged_in on logout: {e}")
        session.pop("user", None)
        g_current_user_id = None
    return redirect(url_for("login"))

if __name__ == "__main__":
    # Start background thread (daemon)
    pred_thread = threading.Thread(target=background_prediction_thread, daemon=True)
    pred_thread.start()

    print("Starting Flask app at http://0.0.0.0:5000")
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000, threaded=True)
