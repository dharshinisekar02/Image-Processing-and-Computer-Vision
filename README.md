#  SENTINEL — Helmet & Safety Gear Detection System

> **HOG + LBP + SVM Engine** | Two-Wheeler Rider Compliance Monitoring  
> Course: AD23B31 - Image Processing and Computer Vision | Rajalakshmi Engineering College

---

##  Overview

**SENTINEL** is an automated helmet detection system for two-wheeler riders, built to assist traffic enforcement with real-time compliance monitoring. It uses a **two-stage hybrid pipeline** — YOLOv8n for person detection, and a classical HOG + LBP + SVM pipeline for helmet classification — deployed as a **Flask web application**.

- ✅ **94.27% accuracy** at optimal threshold (0.5752)
- ✅ **AUC-ROC: 0.9832**
- ✅ **CPU-only** — no GPU required
- ✅ Supports **image, video, and live webcam** inputs

---

##  How It Works

```
Input Frame
    │
    ▼
[YOLOv8n] ──── Person Detection (bounding boxes)
    │
    ▼
[ROI Extraction] ── Top 25% of bbox + asymmetric padding (30% up, 15% sides) → 64×64 crop
    │
    ▼
[Preprocessing] ── Grayscale + CLAHE (illumination normalization)
    │
    ▼
[Feature Extraction]
    ├── HOG: 9 orientations, 8×8 px/cell, 2×2 cells/block, L2-Hys
    └── LBP: radius=3, 24 points, 26-bin histogram
         └── Concatenated feature vector
    │
    ▼
[StandardScaler → PCA (95% variance)] ── Dimensionality reduction
    │
    ▼
[SVM Classifier] ── RBF kernel, C=10, CalibratedClassifierCV
    │
    ▼
Output: "With Helmet" (green box) / "Without Helmet" (red box) + confidence score
```

---

##  Project Structure

```
sentinel/
├── app.py                  # Flask application (routes: /image, /video, /webcam)
├── detector.py             # Core detection + classification pipeline
├── feature_extractor.py    # HOG + LBP feature computation
├── train.py                # Model training script
├── models/
│   ├── svm_model.pkl       # Trained SVM classifier
│   ├── scaler.pkl          # Fitted StandardScaler
│   └── pca.pkl             # Fitted PCA object
├── datasets/
│   ├── BikesHelmets/       # Pascal VOC XML format
│   └── IndianRoad/         # YOLO txt format
├── static/                 # CSS, JS, UI assets
├── templates/
│   └── index.html          # Flask frontend
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
flask>=2.0
opencv-python>=4.5
scikit-learn>=1.0
scikit-image
numpy>=1.21
ultralytics
joblib
```

### Run the app

```bash
python app.py
```

Then open `http://localhost:5000` in your browser.

---

##  Usage

The web interface has three modes, selectable via tabs:

| Mode | Route | Description |
|------|-------|-------------|
| Image Scan | `/image` | Upload a single JPG/PNG/WEBP/BMP image |
| Video Analysis | `/video` | Upload a video file for frame-by-frame analysis |
| Live Surveillance | `/webcam` | Real-time webcam stream detection |

Each output shows:
- Color-coded bounding boxes (🟢 helmet / 🔴 no helmet)
- Classification label + confidence score
- Summary count: `With Helmet: N | Without: N`

---

##  Model Training

Training uses two combined datasets:
- **BikesHelmets** dataset (Pascal VOC XML annotations)
- **Indian Road dataset** (YOLO txt format annotations)

### Data Augmentation (7 variants per sample)
- Original
- Horizontal flip
- Rotations: ±10°, ±12°
- Brightness shifts: ±25, ±30
- Gaussian blur

### Training config

```python
SVM(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
CalibratedClassifierCV(cv=5, method='sigmoid')  # probability calibration
PCA(n_components=0.95)                           # retain 95% variance
optimal_threshold = 0.5752                       # Youden's J optimal
```

To retrain:
```bash
python train.py
```

---

##  Performance

| Metric | Value |
|--------|-------|
| Accuracy (default threshold 0.5) | 93.72% |
| Accuracy (optimal threshold 0.5752) | **94.27%** |
| AUC-ROC | **0.9832** |
| Precision — With Helmet | 0.95 |
| Recall — With Helmet | 0.93 |
| F1-Score — With Helmet | 0.94 |
| Precision — Without Helmet | 0.94 |
| Recall — Without Helmet | 0.95 |
| F1-Score — Without Helmet | 0.94 |

**Confusion Matrix (Combined Test Set):**
```
                  Predicted
                  No Helmet   Helmet
Actual No Helmet    781         40
       Helmet        54        766
```

---

##  Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Processor | Intel Core i3 | Core i5+ |
| RAM | 4 GB | 8 GB |
| Storage | 1 GB free | — |
| Camera | 720p webcam | — |

> ✅ No GPU required — fully CPU-deployable.

---

##  Known Limitations

- **Heuristic ROI**: Head region computed from body bbox — may fail for extreme poses or partial frames
- **Person detection dependency**: Missed detections by YOLOv8 = missed helmet checks
- **Low-light degradation**: Performance drops in nighttime/poorly lit conditions (despite CLAHE)
- **Profile/occluded riders**: Side-facing or partially hidden riders may not classify accurately
- **Binary only**: Detects helmet presence/absence — does NOT check if helmet is correctly fastened
- **No license plate capture**: Violations aren't linked to vehicle IDs

---

##  Future Enhancements

- [ ] Dedicated head detector (RetinaFace / fine-tuned YOLOv8-head) to replace heuristic ROI
- [ ] Lightweight CNN classifier (MobileNetV3 / EfficientNet-B0) as SVM replacement
- [ ] ALPR (Automatic License Plate Recognition) integration for violation logging
- [ ] Night-vision support via Zero-DCE or Retinex enhancement
- [ ] Multi-rider detection (driver + pillion passenger)
- [ ] Edge deployment (Raspberry Pi / NVIDIA Jetson Nano)
- [ ] Helmet fastening detection (correctly vs incorrectly fastened)
- [ ] Cloud dashboard for real-time violation aggregation and reporting

---

##  References

1. Dalal & Triggs — *HOG for Human Detection*, CVPR 2005
2. Ojala et al. — *Local Binary Patterns*, IEEE TPAMI 2002
3. Cortes & Vapnik — *Support-Vector Networks*, Machine Learning 1995
4. Redmon et al. — *You Only Look Once*, CVPR 2016
5. Ultralytics — [YOLOv8](https://github.com/ultralytics/ultralytics), 2023
6. [BikesHelmets Dataset — Kaggle](https://www.kaggle.com/datasets/andrewmvd/helmet-detection)
7. Scikit-learn — [scikit-learn.org](https://scikit-learn.org)
8. OpenCV — [docs.opencv.org](https://docs.opencv.org)

---

##  Author

**Dharshini R S** (2116230701076)  
Department of Computer Science Engineering  
Rajalakshmi Engineering College  
Supervised by: Dr. S. Madhusudhanan, Associate Professor  

---

*March 2026*
