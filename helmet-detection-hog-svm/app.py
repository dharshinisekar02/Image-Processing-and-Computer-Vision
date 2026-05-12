"""
Helmet Detection — Flask Web App
=================================
Features:
  - Upload image → detect → show result
  - Upload video → process → download result
  - Live webcam detection via browser

Run:
  pip install flask
  python app.py
  Open http://localhost:5000
"""

import os
import cv2
import uuid
import joblib
import numpy as np
from flask import (Flask, render_template, request,
                   jsonify, send_from_directory, Response)
from werkzeug.utils import secure_filename
from skimage.feature import hog, local_binary_pattern
from ultralytics import YOLO

# ── Config ───────────────────────────────────
UPLOAD_FOLDER  = "static/uploads"
RESULT_FOLDER  = "static/results"
ALLOWED_IMG    = {"png", "jpg", "jpeg", "bmp", "webp"}
ALLOWED_VID    = {"mp4", "avi", "mov", "mkv"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB

# ── HOG+SVM params (same as training) ────────
IMG_SIZE     = (64, 64)
HOG_PARAMS   = dict(orientations=9, pixels_per_cell=(8,8),
                    cells_per_block=(2,2), block_norm="L2-Hys",
                    visualize=False, feature_vector=True)
LBP_RADIUS   = 3
LBP_N_POINTS = 24
GREEN = (34,  197, 94)
RED   = (30,  100, 255)
WHITE = (255, 255, 255)
BLACK = (0,   0,   0)

# ── Load models once at startup ───────────────
print("Loading models...")
svm_model  = joblib.load("helmet_svm.pkl")
svm_scaler = joblib.load("helmet_scaler.pkl")
svm_pca    = joblib.load("helmet_pca.pkl")
try:
    THRESHOLD = joblib.load("helmet_threshold.pkl")
except:
    THRESHOLD = 0.75
yolo_model = YOLO("yolov8n.pt")
print(f"Models loaded! Threshold = {THRESHOLD:.4f}")


# ─────────────────────────────────────────────
#  DETECTION HELPERS
# ─────────────────────────────────────────────
def preprocess(img_bgr):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)


def extract_features(img_gray):
    hog_feat = hog(img_gray, **HOG_PARAMS)
    lbp      = local_binary_pattern(img_gray, LBP_N_POINTS,
                                    LBP_RADIUS, method="uniform")
    n_bins   = LBP_N_POINTS + 2
    hist,_   = np.histogram(lbp.ravel(), bins=n_bins,
                             range=(0,n_bins), density=True)
    return np.concatenate([hog_feat, hist])


def classify_head(crop_bgr):
    if crop_bgr.size == 0:
        return "Unknown", 0.0
    resized = cv2.resize(crop_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    gray    = preprocess(resized)
    feat    = extract_features(gray).reshape(1,-1)
    feat    = svm_pca.transform(svm_scaler.transform(feat))
    prob    = svm_model.predict_proba(feat)[0][1]
    label   = "With Helmet" if prob >= THRESHOLD else "Without Helmet"
    return label, prob


def get_head_rois(frame, yolo_conf=0.40, head_ratio=0.30):
    h, w    = frame.shape[:2]
    results = yolo_model(frame, verbose=False)[0]
    rois    = []
    for det in results.boxes:
        if int(det.cls[0]) != 0 or float(det.conf[0]) < yolo_conf:
            continue
        px1,py1,px2,py2 = map(int, det.xyxy[0])
        px1=max(0,px1); py1=max(0,py1)
        px2=min(w,px2); py2=min(h,py2)
        ph=py2-py1; pw=px2-px1
        ex  = int(pw*0.08)
        hx1 = max(0, px1-ex)
        hy1 = max(0, py1-int(ph*0.05))
        hx2 = min(w, px2+ex)
        hy2 = min(h, py1+int(ph*head_ratio))
        if (hx2-hx1)>10 and (hy2-hy1)>10:
            rois.append((hx1,hy1,hx2,hy2))
    return rois


def process_frame(frame, head_ratio=0.30, yolo_conf=0.40):
    rois   = get_head_rois(frame, yolo_conf, head_ratio)
    result = frame.copy()
    hc=0; nhc=0

    for (x1,y1,x2,y2) in rois:
        crop       = frame[y1:y2, x1:x2]
        label,conf = classify_head(crop)
        color      = GREEN if label=="With Helmet" else RED

        cv2.rectangle(result, (x1,y1), (x2,y2), color, 2)
        text = f"{'Helmet' if 'With' in label else 'No Helmet'} {conf:.2f}"
        (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(result,(x1,max(y1-22,0)),(x1+tw+6,max(y1,22)),color,-1)
        cv2.putText(result,text,(x1+3,max(y1-5,17)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.52,WHITE,1)

        if label=="With Helmet": hc+=1
        else: nhc+=1

    # Summary bar
    summary = f"With Helmet: {hc}  |  Without: {nhc}"
    bw = len(summary)*10+12
    cv2.rectangle(result,(0,0),(bw,28),BLACK,-1)
    cv2.putText(result,summary,(6,19),
                cv2.FONT_HERSHEY_SIMPLEX,0.58,WHITE,1)

    return result, hc, nhc


def allowed_file(filename, allowed):
    return "." in filename and filename.rsplit(".",1)[1].lower() in allowed


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect_image", methods=["POST"])
def detect_image():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if not f or not allowed_file(f.filename, ALLOWED_IMG):
        return jsonify({"error": "Invalid image file"}), 400

    uid      = str(uuid.uuid4())[:8]
    filename = secure_filename(f.filename)
    in_path  = os.path.join(UPLOAD_FOLDER, uid+"_"+filename)
    out_name = uid+"_result.jpg"
    out_path = os.path.join(RESULT_FOLDER, out_name)

    f.save(in_path)
    frame = cv2.imread(in_path)
    if frame is None:
        return jsonify({"error": "Cannot read image"}), 400

    result, hc, nhc = process_frame(frame)
    cv2.imwrite(out_path, result)

    return jsonify({
        "result_url": f"/static/results/{out_name}",
        "with_helmet": hc,
        "without_helmet": nhc,
        "total": hc + nhc
    })


@app.route("/detect_video", methods=["POST"])
def detect_video():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if not f or not allowed_file(f.filename, ALLOWED_VID):
        return jsonify({"error": "Invalid video file"}), 400

    uid      = str(uuid.uuid4())[:8]
    filename = secure_filename(f.filename)
    in_path  = os.path.join(UPLOAD_FOLDER, uid+"_"+filename)
    out_name = uid+"_result.mp4"
    out_path = os.path.join(RESULT_FOLDER, out_name)

    f.save(in_path)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video"}), 400

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    total_hc=0; total_nhc=0; frame_count=0

    while True:
        ret, frame = cap.read()
        if not ret: break
        result, hc, nhc = process_frame(frame)
        out.write(result)
        total_hc  = max(total_hc,  hc)
        total_nhc = max(total_nhc, nhc)
        frame_count += 1

    cap.release(); out.release()

    return jsonify({
        "result_url":     f"/static/results/{out_name}",
        "with_helmet":    total_hc,
        "without_helmet": total_nhc,
        "frames":         frame_count
    })


# ── Webcam streaming ──────────────────────────
def gen_webcam_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        result,_,_ = process_frame(frame)
        _, buffer  = cv2.imencode(".jpg", result)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")
    cap.release()


@app.route("/webcam_feed")
def webcam_feed():
    return Response(gen_webcam_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True, threaded=True)