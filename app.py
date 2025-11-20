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

# --- NEW: Set the environment variable for OpenCV's timeout ---
# This tells FFMPEG to time out after 3 seconds (3,000,000 microseconds)
# This MUST be set before any cv2.VideoCapture call
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "stimeout;3000000"
# ---------------------------------------------------------------


# PI_SERVO_URL = "http://10.32.181.201:5000/move_servo"
PI_SERVO_URL = "http://192.168.254.109:5000/move_servo"


def send_servo_command(direction):
    """Sends a move command to the Raspberry Pi."""
    try:
        # --- MODIFICATION HERE: Timeout increased to 2.0 seconds ---
        requests.post(PI_SERVO_URL, json={"direction": direction}, timeout=2.0)
        # --- END MODIFICATION ---
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not send servo command: {e}")
        
# =Options
PREDICTION_INTERVAL = 1.0  # (seconds) How often to run the model
UPLOAD_INTERVAL = 20.0     # (seconds) How often to save to Firebase


# Initialize Flask app
app = Flask(__name__)
app.secret_key = "citropascan_secret_key"

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
# Firebase Setup
# ===============================
cred = credentials.Certificate("serviceAccountKey.json")  # service account JSON
firebase_admin.initialize_app(cred)
db = firestore.client()

FIREBASE_WEB_API_KEY = "AIzaSyAVAZemmqpkBuAmEqppKQ0URWAIzAVP3m4" # TODO: Check if this is still valid

# ===============================
# Load Model
# ===============================
MODEL_PATH = "accDiseases.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")
print("Expected input shape:", model.input_shape)

CLASS_NAMES = [
    "Black Spot", "Canker", "Fresh", "Greeening", "Scab"
]

# ===============================
# Camera Setup (FIXED)
# ===============================

def initialize_camera_with_failover(sources):
    """
    Tries to open video sources from a list, one by one, until one succeeds.
    This version also tries to read one frame to confirm the stream is active.
    
    Args:
        sources (list): A list of video sources to try in order.

    Returns:
        cv2.VideoCapture: The first successfully opened AND read VideoCapture object.
        None: If all sources in the list fail.
    """
    camera = None
    
    for source in sources:
        print(f"Attempting to connect to camera: {source}")
        
        # This call will now honor the 3-second timeout
        camera = cv2.VideoCapture(source)
        
        # Give the connection a moment to establish
        time.sleep(1) 
        
        if camera.isOpened():
            # Now, test if we can actually READ from the camera
            ret, _ = camera.read() 
            if ret:
                # SUCCESS! We have a camera AND a frame.
                print(f"✅ Success! Connected and receiving frames from: {source}")
                break # Exit the 'for' loop
            else:
                # Connection opened, but read failed
                print(f"❌ Connected but failed to read frame from: {source}")
                camera.release()
                camera = None
        else:
            # Connection failed to open
            print(f"❌ Failed to open camera connection: {source}")
            if camera:
                camera.release()
            camera = None 
    
    if camera is None:
        print("❌ CRITICAL: Could not connect to ANY camera source.")
        
    return camera


video_sources = [
    "http://192.168.254.109:5000/video_feed",
    "http://10.32.181.201:5000/video_feed", 
     # (This line is what you're using)
    0 
]

# In your app.py file:

# The IP of your Raspberry Pi
PI_IP_ADDRESS = "192.168.254.109" 

# Correct URL for sending commands
PI_SERVO_URL = f"http://{PI_IP_ADDRESS}:5000/move_servo"
# Try to initialize the camera
camera = initialize_camera_with_failover(video_sources)

# --- THE BLOCKING 'while True: ...' LOOP HAS BEEN DELETED ---
# The Flask app will now continue past this point and start the server.
# The 'camera' object (or None) will be used by 'generate_frames()'.

# ==================================
# Global Vars for Threading
# ==================================
g_last_frame = None            # The most recent frame from the camera
g_last_prediction = {          # The most recent prediction from the model
    "class": "Loading...", 
    "confidence": 0.0
}
g_frame_lock = threading.Lock()      # Lock for g_last_frame
g_prediction_lock = threading.Lock() # Lock for g_last_prediction
g_current_user_id = None       # To tell the background thread who is logged in
g_last_upload_time = 0
# ==================================


def preprocess_frame(frame):
    """Resize and normalize frame for prediction"""
    target_size = (256, 256)  # Match model input
    img = cv2.resize(frame, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR → RGB
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

# ==================================
# NEW: Motion Detection Constants
# ==================================
MIN_MOTION_AREA = 5000  # Adjust this: Smaller = more sensitive, Larger = ignores small objects
MOTION_THRESHOLD = 25   # Pixel intensity difference required to count as motion

# ==================================
# UPDATED: Background Prediction Thread
# ==================================
def background_prediction_thread():
    """
    Runs in a separate thread. 
    1. Detects Motion.
    2. If Motion -> Predicts Disease -> Moves Servo -> Saves to Firebase.
    """
    global g_last_frame, g_last_prediction, g_current_user_id, g_last_upload_time

    print("🚀 Background prediction thread started with Motion Detection.")
    
    # Variable to store the previous frame for comparison
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
            # --- 1. Motion Detection Logic ---
            
            # Convert to grayscale and blur to remove noise
            gray = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # If this is the very first frame, initialize prev_gray and skip
            if prev_gray is None:
                prev_gray = gray
                continue

            # Compute the absolute difference between current frame and previous frame
            frame_delta = cv2.absdiff(prev_gray, gray)
            
            # Threshold the delta image (convert to black and white)
            thresh = cv2.threshold(frame_delta, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
            
            # Dilate the thresholded image to fill in holes
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Find contours (shapes) of the moving areas
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            
            # Check if any contour is large enough to be a fruit
            for contour in contours:
                if cv2.contourArea(contour) > MIN_MOTION_AREA:
                    motion_detected = True
                    break # We found at least one big moving object
            
            # Update prev_gray for the next loop iteration
            prev_gray = gray

            # --- 2. Decision Logic ---
            
            if not motion_detected:
                # If no motion, just print a status (optional) and skip prediction
                # print("No motion detected... sleeping")
                time.sleep(0.2) # Short sleep to check again soon
                continue 
            
            # IF WE REACH HERE, MOTION WAS DETECTED!
            print("📸 Motion Detected! Running Prediction...")

            # --- 3. Perform Heavy Prediction ---
            img = preprocess_frame(frame_to_process)
            preds = model.predict(img, verbose=0)
            predicted_class = CLASS_NAMES[np.argmax(preds)]
            confidence = float(np.max(preds) * 100)
            
            with g_prediction_lock:
                g_last_prediction = {
                    "class": predicted_class,
                    "confidence": confidence
                }
            
            # --- 4. Send Servo Command (with delay logic) ---
            if predicted_class == "Fresh":
                print("Prediction: Fresh. Moving servo RIGHT.")
                send_servo_command('right')
                time.sleep(1.0) 
            else:
                print(f"Prediction: {predicted_class}. Moving servo LEFT.")
                send_servo_command('left')
                print("Waiting 5 seconds for disease sort...")
                time.sleep(5.0) 
            
            # --- 5. Return to Center ---
            print("Returning servo to CENTER.")
            send_servo_command('center')
            
            # --- 6. Handle Firebase Saving ---
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

            # ====================================================
            # CRITICAL FIX: RESET MOTION BASELINE
            # ====================================================
            print("🔄 Resetting motion detector to ignore servo movement...")
            
            # 1. Wait a tiny bit for the servo to actually finish returning to center
            time.sleep(1.0) 

            # 2. Grab the absolute latest frame to be the new "background"
            with g_frame_lock:
                if g_last_frame is not None:
                    # Create a new "previous" frame based on the scene RIGHT NOW
                    # This prevents the "Old Scene vs New Scene" ghost motion trigger
                    reset_gray = cv2.cvtColor(g_last_frame, cv2.COLOR_BGR2GRAY)
                    reset_gray = cv2.GaussianBlur(reset_gray, (21, 21), 0)
                    prev_gray = reset_gray
            
            print("✅ Motion detector reset. Ready for next fruit.")
            # ====================================================

        except Exception as e:
            print(f"❌ [BackgroundThread] Prediction error: {e}")
            
        # --- 7. Rate Limit ---
        # We can use a shorter sleep here because the motion detector acts as the main filter
        time.sleep(0.5)
        
def generate_frames(user_id):
    """
    Generator function that yields JPEG frames *fast*.
    It only draws the latest prediction from the background thread.
    """
    global g_last_frame, g_last_prediction, g_current_user_id, g_last_upload_time

    # Tell the background thread who is logged in
    g_current_user_id = user_id
    # Reset upload timer for new user session
    g_last_upload_time = time.time() 
    
    # Check if camera initialization failed globally
    if camera is None or not camera.isOpened():
        print("❌ Camera is not available. Stopping frame generation.")
        # Optionally, you could yield a "Camera Offline" image
        return

    while True:
        success, frame = camera.read()
        if not success:
            print("❌ Camera read failed. Stream may have ended.")
            # We will stop the generator here. 
            # The client will need to refresh to try reconnecting.
            break
        
        # --- Update the global frame (for the prediction thread) ---
        with g_frame_lock:
            g_last_frame = frame.copy()
        
        # --- Get the latest prediction (don't wait) ---
        with g_prediction_lock:
            prediction = g_last_prediction.copy()

        # --- Draw on frame (fast) ---
        try:
            cv2.putText(frame,
                        f"{prediction['class']} ({prediction['confidence']:.2f}%)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)
        except Exception as e:
            print(f"❌ DrawText error: {e}")

        # --- Encode and yield (fast) ---
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("❌ Failed to encode frame")
            continue
        
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Optional: slightly yield to prevent 100% CPU
        time.sleep(1/60) # Aim for 60 FPS stream


@app.route('/video_feed')
def video_feed():
    if "user" not in session:
        return redirect(url_for("login"))

    user_id = session["user"]["uid"]
    return Response(generate_frames(user_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ===============================
# Routes (Unchanged)
# ===============================
@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/dashboard")
def index():
    if "user" not in session:
        return redirect(url_for("login"))

    user_id = session["user"]["uid"]

    # This part is slow if there are many scans. 
    # Consider using the /api/counters logic instead.
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
        
        # Get total count
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
                # Convert Firestore Timestamp to Python datetime
                timestamp = fs_timestamp.to_datetime() 
            except Exception as e:
                print(f"Warning: Could not parse timestamp {fs_timestamp}: {e}")

        predictions.append({
            "id": idx,
            "disease": disease,
            "confidence": confidence,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"), # Format for display
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
# Profile & Upload (Unchanged)
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

        # Path to store in DB (relative to 'static')
        web_filepath_relative = os.path.join(UPLOAD_FOLDER_RELATIVE, unique_filename).replace("\\", "/") 
        
        db.collection("users").document(user_id).update({
            "profile_pic": web_filepath_relative  # Save the relative path
        })
        session["user"]["profile_pic"] = web_filepath_relative # Also save to session
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
        user_data['phone'] = user_data.get('phone') 
        user_data['address'] = user_data.get('address')
        
        # --- EDITED: Robust last_login handling ---
        # Get last_login from the database (it's a Firestore Timestamp)
        db_last_login = user_data.get("last_login")
        if db_last_login:
            try:
                # Convert it to a Python datetime object
                user_data["last_login"] = db_last_login.to_datetime()
            except Exception as e:
                print(f"Warning: Could not parse DB last_login timestamp: {e}")
                # Fallback to the login time from this session
                user_data["last_login"] = session["user"].get("last_login")
        else:
            # Fallback if no value is in the database yet
            user_data["last_login"] = session["user"].get("last_login")
        # --- END EDIT ---
            
    else:
        user_data = {"fullname": "Unknown", "email": "Not Found", "phone": None, "address": None}

    # Sync DB data to session
    session["user"]["profile_pic"] = user_data.get("profile_pic")
    session["user"]["phone"] = user_data.get("phone")
    session["user"]["address"] = user_data.get("address")
    session.modified = True

    scans_ref = user_ref.collection("scans")
    scans = scans_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(5).stream()
    
    scan_history = []
    for scan in scans:
        data = scan.to_dict()
        fs_timestamp = data.get("timestamp")
        py_datetime = None 
        if fs_timestamp:
            try:
                py_datetime = fs_timestamp.to_datetime() 
            except Exception as e:
                print(f"Warning: Could not parse scan timestamp {fs_timestamp}: {e}")
                
        scan_history.append({
            "disease": data.get("predicted_class"),
            "confidence": data.get("confidence"),
            "timestamp": py_datetime, 
        })
    
    # This line is no longer needed, as we handle it above
    # user_data["last_login"] = session["user"].get("last_login") 

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
        # --- CRITICAL EDIT: Update Firebase Auth email ---
        # This keeps the login email in sync with the database email
        current_email = session["user"]["email"]
        if email != current_email:
            auth.update_user(user_id, email=email)
            flash("Login email updated. Please check your inbox to verify.", "info")
        # --- END CRITICAL EDIT ---

        update_data = {
            "fullname": fullname,
            "email": email,
            "phone": phone,
            "address": address
        }
        db.collection("users").document(user_id).update(update_data)

        # Update session data as well
        session["user"]["email"] = email
        session["user"]["fullname"] = fullname
        session["user"]["phone"] = phone
        session["user"]["address"] = address
        session.modified = True # Mark session as modified
        
        # --- NEW: Add success message ---
        flash("Profile updated successfully!", "success")
        # --- END NEW ---

    except Exception as e:
        # --- NEW: Add error message ---
        flash(f"Update failed: {str(e)}", "error")
        # --- END NEW ---
        
    return redirect(url_for("profile"))

# ===============================
# Auth Routes (Unchanged)
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
            error = "All fields are required."
            return render_template("signup.html", error=error)

        if password != confirm_password:
            error = "Passwords do not match."
            return render_template("signup.html", error=error)

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

            user_id = data["localId"]
            user_ref = db.collection("users").document(user_id)
            user_doc = user_ref.get()

            # Note: This "is_logged_in" check is not 100% reliable.
            # A user could close their browser without logging out.
            # Don't rely on it for critical security.
            if user_doc.exists and user_doc.to_dict().get("is_logged_in", False):
                # pass # Allow login even if logged in elsewhere
                print("Warning: User is already marked as logged in.")

            last_login_obj = datetime.now()
            user_data = user_doc.to_dict() or {} # Handle new user login

            session["user"] = {
                "uid": user_id,
                "email": data["email"],
                "idToken": data["idToken"],
                "last_login": last_login_obj,
                "fullname": user_data.get("fullname", "User"),
                "profile_pic": user_data.get("profile_pic")
                # Phone/address will be added on profile page load
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
            # Check if user exists
            user = auth.get_user_by_email(email)
            
            # Generate the password reset link
            link = auth.generate_password_reset_link(email)
            
            # --- Send the email ---
            try:
                msg = Message(
                    subject="Reset Your CitroPaScan Password",
                    recipients=[email], # Send to the user's email
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
            
            return redirect(url_for('login')) # Redirect after attempting send

        except auth.UserNotFoundError:
            # Don't reveal if the email exists for security
            flash("If an account exists for that email, a reset link has been sent.", "info")
            return redirect(url_for('login'))
        except Exception as e:
            print(f"❌ Password reset error: {e}")
            flash(f"An error occurred: {e}", "error")
            return redirect(url_for('forgot_password'))

    # Handle GET request
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
        g_current_user_id = None # Stop logging for background thread
        
    return redirect(url_for("login"))


# ===============================
# Run Flask App
# ===============================
if __name__ == "__main__":
    # Start the background prediction thread
    pred_thread = threading.Thread(target=background_prediction_thread, daemon=True)
    pred_thread.start()
    
    # Start the Flask app
    
    # IMPORTANT: use_reloader=False is critical when running threads
    # host='0.0.0.0' makes it accessible on your network
    print("Starting Flask app at http://0.0.0.0:5000")
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000, threaded=True)