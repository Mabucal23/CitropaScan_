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

FLASK_SERVER_URL = "https://semicaricatural-elizabet-proximal.ngrok-free.dev"

import requests

def request_model_prediction(image_bytes):
    """
    Sends captured frame to the remote Flask app (running on your PC via ngrok)
    """
    try:
        response = requests.post(
            f"{FLASK_SERVER_URL}/predict",
            files={'image': ('frame.jpg', image_bytes, 'image/jpeg')},
            timeout=5
        )
        return response.json()  # expected {result: "good"/"diseased"}
    except Exception as e:
        print("Prediction error:", e)
        return {"result": "error"}

# --- Set OpenCV Timeout ---
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "stimeout;3000000"

# ===============================
# --- Project Settings ---
# ===============================
PI_IP_ADDRESS = "192.168.254.109" 

# URLs for the Pi
PI_SERVO_URL = f"http://{PI_IP_ADDRESS}:5000/move_servo"
PI_SENSOR_URL = f"http://{PI_IP_ADDRESS}:5000/check_sensor"

# Camera Sources




PREDICTION_INTERVAL = 1.0 
UPLOAD_INTERVAL = 20.0

def send_servo_command(direction):
    if not PI_NGROK_URL:
        update_pi_ngrok_url()
    if PI_NGROK_URL:
        try:
            requests.post(f"{PI_NGROK_URL}/move_servo", json={"direction": direction}, timeout=2.0)
        except requests.exceptions.RequestException as e:
            print(f"❌ Servo error: {e}")

# Sensor
def check_pi_sensor():
    try:
        resp = requests.get(PI_SENSOR_URL, timeout=2.0)
        return resp.json()
    except Exception as e:
        print(f"Sensor check failed: {e}")
        return {"detected": False, "distance_cm": 999}


# Video feed source
def get_pi_video_source():
    if not PI_NGROK_URL:
        update_pi_ngrok_url()
    if PI_NGROK_URL:
        return f"{PI_NGROK_URL}/video_feed"
    return 0 


PI_NGROK_URL = None

def update_pi_ngrok_url():
    global PI_NGROK_URL
    try:
        resp = requests.get(f"http://{PI_IP_ADDRESS}:5000/ngrok_url", timeout=2.0)
        if resp.status_code == 200:
            PI_NGROK_URL = resp.json().get("ngrok_url")
            print(f"✅ Updated Pi ngrok URL: {PI_NGROK_URL}")
    except Exception as e:
        print(f"❌ Could not fetch Pi ngrok URL: {e}")


def send_servo_command(direction):
    """Sends a move command to the Raspberry Pi."""
    try:
        requests.post(PI_SERVO_URL, json={"direction": direction}, timeout=2.0)
    except requests.exceptions.RequestException as e:
        print(f"❌ Servo error: {e}")


# Initialize Flask app
app = Flask(__name__)
app.secret_key = "citropascan_secret_key"

video_sources = [get_pi_video_source(), 0]

def capture_frame_bytes():
    with output.condition:
        output.condition.wait()
        frame = output.frame
    return frame if frame else None

# ===============================
# Mail Setup
# ===============================
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'te.st.emmail02.gmail.com' # TODO: Change this
app.config['MAIL_PASSWORD'] = 'testemail02' # TODO: Change this
app.config['MAIL_DEFAULT_SENDER'] = ('te.st.emmail02.gmail.com') # TODO: Change this

mail = Mail(app)

# ===============================
# Firebase Setup (REVERTED TO FILE-BASED)
# ===============================
SERVICE_ACCOUNT_FILE = "serviceAccountKey.json"

try:
    # 1. Load the key directly from the file in the project folder
    cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase initialized successfully!")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to load Firebase certificate from file: {e}")
    print(f"Check if {SERVICE_ACCOUNT_FILE} is in the current directory and has valid JSON content.")
    exit(1)


FIREBASE_WEB_API_KEY = "AIzaSyAVAZemmqpkBuAmEqppKQ0URWAIzAVP3m4" # Used only for client-side REST login

# ===============================
# Load Model
# ===============================
MODEL_PATH = "accDiseases.keras" # Assume .keras model for now
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to load ML model {MODEL_PATH}: {e}")
    exit(1)


CLASS_NAMES = [
    "Black Spot", "Canker", "Fresh", "Greeening", "Scab"
]

@app.route("/predict_and_sort", methods=['POST'])
def predict_and_sort():
    try:
        # 1 — Capture frame from camera
        frame = capture_frame_bytes()
        if frame is None:
            return jsonify({"status": "error", "message": "No frame captured"})

        # 2 — Send to PC Flask via Ngrok
        result = request_model_prediction(frame)
        print("Model response:", result)

        prediction = result.get("result", "unknown")

        # 3 — Move servo based on result
        if prediction == "good":
            requests.post("http://localhost:5000/move_servo", json={"direction": "right"})
            return jsonify({"status": "sorted", "direction": "right"})

        elif prediction == "diseased":
            requests.post("http://localhost:5000/move_servo", json={"direction": "left"})
            return jsonify({"status": "sorted", "direction": "left"})

        else:
            return jsonify({"status": "error", "message": "Unknown prediction"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ===============================
# Camera Setup
# ===============================
def initialize_camera_with_failover(sources):
    camera = None
    for source in sources:
        print(f"Attempting to connect to camera: {source}")
        camera = cv2.VideoCapture(source)
        time.sleep(1) 
        if camera.isOpened():
            ret, _ = camera.read() 
            if ret:
                print(f"✅ Connected to: {source}")
                break 
            else:
                camera.release()
                camera = None
        else:
            if camera:
                camera.release()
            camera = None 
    
    if camera is None:
        print("❌ CRITICAL: Could not connect to ANY camera source.")
        
    return camera

# Try to initialize the camera
camera = initialize_camera_with_failover(video_sources)

# ==================================
# Global Vars for Threading
# ==================================
g_last_frame = None
g_last_prediction = { "class": "Loading...", "confidence": 0.0 }
g_frame_lock = threading.Lock()
g_prediction_lock = threading.Lock()
g_current_user_id = None
g_last_upload_time = 0

def preprocess_frame(frame):
    """Resize and normalize frame for prediction"""
    target_size = (256, 256)
    img = cv2.resize(frame, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

# ==================================
# Background Smart Detection Thread
# ==================================
def background_prediction_thread():
    global g_last_frame, g_last_prediction, g_current_user_id, g_last_upload_time

    print("🚀 Background Smart Detection thread started.")
    
    while True:
        # --- 1. WAIT FOR SENSOR TRIGGER ---
        try:
            resp = requests.get(PI_SENSOR_URL, timeout=0.5)
            sensor_data = resp.json()
            
            if not sensor_data.get("detected"):
                time.sleep(0.1) 
                continue 
            
            # OBJECT DETECTED!
            print(f"👀 Object at {sensor_data['distance_cm']}cm! Predicting...")
            time.sleep(0.5) 

        except Exception as e:
            print(f"Sensor check failed: {e}")
            time.sleep(1)
            continue
            
        # --- 2. CAPTURE FRAME ---
        frame_to_process = None
        with g_frame_lock:
            if g_last_frame is not None:
                frame_to_process = g_last_frame.copy()
        
        if frame_to_process is None:
            time.sleep(0.1)
            continue
            
        try:
            # --- 3. PERFORM PREDICTION ---
            img = preprocess_frame(frame_to_process)
            preds = model.predict(img, verbose=0)
            predicted_class = CLASS_NAMES[np.argmax(preds)]
            confidence = float(np.max(preds) * 100)
            
            with g_prediction_lock:
                g_last_prediction = {
                    "class": predicted_class,
                    "confidence": confidence
                }
            
            # --- 4. SEND SERVO COMMAND ---
            if predicted_class == "Fresh":
                print("Prediction: Fresh. Moving servo RIGHT.")
                send_servo_command('right')
                time.sleep(1.0) 
            else:
                print(f"Prediction: {predicted_class}. Moving servo LEFT.")
                send_servo_command('left')
                print("Waiting 5 seconds for disease sort...")
                time.sleep(5.0) 
            
            # --- 5. RETURN TO CENTER ---
            print("Returning servo to CENTER.")
            send_servo_command('center')
            
            # --- 6. SAVE TO FIREBASE ---
            current_time = time.time()
            user_id = g_current_user_id
            
            if user_id:
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
                    print(f"✅ Saved to Firebase for user {user_id}")
                except Exception as e:
                    print(f"❌ Firebase save error: {e}")

            # --- 7. COOLDOWN (Wait for object to clear) ---
            print("Waiting for object to clear sensor...")
            while True:
                try:
                    r = requests.get(PI_SENSOR_URL, timeout=0.5)
                    if not r.json().get("detected"):
                        break 
                except:
                    break
                time.sleep(0.2)
            print("Ready for next fruit.")

        except Exception as e:
            print(f"❌ Prediction error: {e}")
        
        time.sleep(0.1)

def generate_frames(user_id):
    """
    Generator function that yields JPEG frames.
    """
    global g_current_user_id
    g_current_user_id = user_id
    
    if camera is None or not camera.isOpened():
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        with g_frame_lock:
            g_last_frame = frame.copy()
        
        with g_prediction_lock:
            prediction = g_last_prediction.copy()

        try:
            cv2.putText(frame, f"{prediction['class']} ({prediction['confidence']:.1f}%)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except: pass

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(1/60)

@app.route('/video_feed')
def video_feed():
    if "user" not in session:
        return redirect(url_for("login"))
    return Response(generate_frames(session["user"]["uid"]), mimetype='multipart/x-mixed-replace; boundary=frame')

# ===============================
# Standard Routes
# ===============================
@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/dashboard")
def index():
    if "user" not in session:
        return redirect(url_for("login"))

    user_id = session["user"]["uid"]

    scans_ref = db.collection("users").document(user_id).collection("scans")
    docs = scans_ref.stream()
    predictions = [doc.to_dict() for doc in docs]

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
        # Use efficient count() aggregation
        total_query = scans_ref.count()
        total_result = total_query.get()
        total_scans = total_result[0][0].value if total_result else 0

        # Get healthy ("Fresh") count
        healthy_query = scans_ref.where("predicted_class", "==", "Fresh").count() 
        healthy_result = healthy_query.get()
        fresh_count = healthy_result[0][0].value if healthy_result else 0
        
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
    docs = scans_ref.stream()

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
        timestamp = datetime.now() # Fallback
        if fs_timestamp:
            try:
                timestamp = fs_timestamp.to_datetime() 
            except Exception as e:
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

# ===============================
# Profile & Upload
# ===============================
UPLOAD_FOLDER_RELATIVE = "uploads"  # Path INSIDE static
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
        
        db.collection("users").document(user_id).update({
            "profile_pic": web_filepath_relative 
        })
        session["user"]["profile_pic"] = web_filepath_relative 
        session.modified = True

    return redirect(url_for("profile"))

@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect(url_for("login"))

    user_id = session["user"]["uid"]
    user_ref = db.collection("users").document(user_id)
    user_doc = user_ref.get()

    if user_doc.exists:
        user_data = user_doc.to_dict()
        user_data['phone'] = user_data.get('phone', '') 
        user_data['address'] = user_data.get('address', '')
        
        db_last_login = user_data.get("last_login")
        if db_last_login:
            try:
                user_data["last_login"] = db_last_login.to_datetime()
            except Exception as e:
                user_data["last_login"] = session["user"].get("last_login")
        else:
            user_data["last_login"] = session["user"].get("last_login")
            
    else:
        user_data = {"fullname": "Unknown", "email": "Not Found", "phone": "", "address": ""}

    session["user"]["profile_pic"] = user_data.get("profile_pic")
    session["user"]["phone"] = user_data.get("phone")
    session["user"]["address"] = user_data.get("address")
    session.modified = True

    scans = user_ref.collection("scans").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(5).stream()
    
    scan_history = []
    for scan in scans:
        data = scan.to_dict()
        fs_timestamp = data.get("timestamp")
        py_datetime = None 
        if fs_timestamp:
            try:
                py_datetime = fs_timestamp.to_datetime() 
            except Exception as e:
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
        if email != current_email:
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

# ===============================
# Auth Routes
# ===============================
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
            user = auth.create_user(
                email=email,
                password=password,
                display_name=fullname
            )

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
            # FIXED URL HERE
            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }

            r = requests.post(url, json=payload)
            data = r.json()

            if "error" in data:
                error_message = data["error"]["message"]
                if error_message == "INVALID_PASSWORD" or error_message == "Wrong Password":
                    error = "❌ Wrong password. Please try again."
                elif error_message == "EMAIL_NOT_FOUND" or error_message == "Email does not exist":
                    error = "❌ Email not found. Please sign up first."
                else:
                    error = f"Login failed: {error_message}"
                return render_template("login.html", error=error)

            if not all(k in data for k in ("localId", "email", "idToken")):
                return render_template("login.html", error="❌ Invalid login response. Please try again.")

            # FIXED: Removed syntax error +
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

        except auth.UserNotFoundError:
            flash("If an account exists for that email, a reset link has been sent.", "info")
            return redirect(url_for('login'))
        except Exception as e:
            print(f"❌ Password reset error: {e}")
            flash(f"An error occurred: {e}", "error")
            return redirect(url_for('forgot_password'))

    return render_template("forget_password.html")

@app.route("/logout")
def logout():
    global g_current_user_id
    if "user" in session:
        user_id = session["user"]["uid"]

        db.collection("users").document(user_id).update({
            "is_logged_in": False
        })

        session.pop("user", None)
        g_current_user_id = None 
        
    return redirect(url_for("login"))


# ===============================
# Run Flask App
# ===============================
if __name__ == "__main__":
    pred_thread = threading.Thread(target=background_prediction_thread, daemon=True)
    pred_thread.start()
    
    print("Starting Flask app at http://0.0.0.0:5000")
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000, threaded=True)