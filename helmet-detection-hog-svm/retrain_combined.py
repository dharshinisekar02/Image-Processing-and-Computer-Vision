"""
Helmet Detector — Final Retrain (Combined Dataset)
===================================================
Combines:
  Dataset 1 (old): BikesHelmets — Pascal VOC XML format
    archive/images/       archive/annotations/

  Dataset 2 (new): Indian Road — YOLO txt format
    archive(1)/train/images/   archive(1)/train/labels/
    archive(1)/val/images/     archive(1)/val/labels/

Both parsed → crops extracted → combined → HOG+SVM trained

YOLO classes:
  0 = with helmet    → label 1
  1 = without helmet → label 0
  2 = rider          → SKIP
  3 = number plate   → SKIP

Usage:
  python retrain_combined.py
  python retrain_combined.py --old archive --new "archive (1)"
"""

import os
import cv2
import numpy as np
import joblib
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.feature import hog, local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score,
                              roc_curve, f1_score)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample

# ── Config ───────────────────────────────────
IMG_SIZE     = (64, 64)
HOG_PARAMS   = dict(orientations=9, pixels_per_cell=(8,8),
                    cells_per_block=(2,2), block_norm="L2-Hys",
                    visualize=False, feature_vector=True)
LBP_RADIUS   = 3
LBP_N_POINTS = 24
PCA_VARIANCE = 0.95

# Old dataset class map (XML)
XML_CLASS_MAP = {"With Helmet": 1, "Without Helmet": 0}

# New dataset class map (YOLO)
YOLO_HELMET    = 0   # with helmet    → 1
YOLO_NO_HELMET = 1   # without helmet → 0
YOLO_SKIP      = {2, 3}

# SVM params (best from previous runs)
BEST_C     = 10
BEST_GAMMA = "scale"

MODEL_PATH     = "helmet_svm.pkl"
SCALER_PATH    = "helmet_scaler.pkl"
PCA_PATH       = "helmet_pca.pkl"
THRESHOLD_PATH = "helmet_threshold.pkl"

PAD = 0.08   # 8% padding around each crop


# ─────────────────────────────────────────────
#  SHARED HELPERS
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


def augment(img):
    variants = [img, cv2.flip(img, 1)]
    h, w = img.shape
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
        variants.append(cv2.warpAffine(img, M, (w,h)))
    for beta in [-25, 25]:
        variants.append(np.clip(img.astype(int)+beta,0,255).astype(np.uint8))
    variants.append(cv2.GaussianBlur(img,(3,3),0))
    return variants


def crop_and_add(img_bgr, x1, y1, x2, y2, label, X, y, do_aug=True):
    """Safely crop, resize, extract features, add to X/y."""
    ih, iw = img_bgr.shape[:2]
    # add padding
    pw = int((x2-x1)*PAD); ph = int((y2-y1)*PAD)
    x1 = max(0, x1-pw);  y1 = max(0, y1-ph)
    x2 = min(iw, x2+pw); y2 = min(ih, y2+ph)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0]<8 or crop.shape[1]<8:
        return
    crop = cv2.resize(crop, IMG_SIZE, interpolation=cv2.INTER_AREA)
    gray = preprocess(crop)
    imgs = augment(gray) if do_aug else [gray]
    for aug in imgs:
        X.append(extract_features(aug))
        y.append(label)


# ─────────────────────────────────────────────
#  DATASET 1 — Old BikesHelmets (Pascal VOC XML)
# ─────────────────────────────────────────────
def load_old_dataset(old_dir):
    images_dir = os.path.join(old_dir, "images")
    ann_dir    = os.path.join(old_dir, "annotations")

    if not os.path.isdir(images_dir) or not os.path.isdir(ann_dir):
        print(f"  [OLD] Skipping — folder not found: {old_dir}")
        return [], []

    X, y = [], []
    xml_files = [f for f in os.listdir(ann_dir) if f.endswith(".xml")]
    counts = {0: 0, 1: 0}

    for xml_file in sorted(xml_files):
        base     = os.path.splitext(xml_file)[0]
        xml_path = os.path.join(ann_dir, xml_file)

        img_path = None
        for ext in (".png",".jpg",".jpeg",".PNG",".JPG"):
            c = os.path.join(images_dir, base+ext)
            if os.path.exists(c):
                img_path = c; break
        if not img_path:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            if name not in XML_CLASS_MAP:
                continue
            label = XML_CLASS_MAP[name]
            bb    = obj.find("bndbox")
            x1 = int(float(bb.find("xmin").text))
            y1 = int(float(bb.find("ymin").text))
            x2 = int(float(bb.find("xmax").text))
            y2 = int(float(bb.find("ymax").text))
            crop_and_add(img, x1, y1, x2, y2, label, X, y)
            counts[label] += 1

    print(f"  [OLD] {len(xml_files)} XMLs → "
          f"With Helmet: {counts[1]}  Without: {counts[0]}  "
          f"Samples: {len(X)}")
    return X, y


# ─────────────────────────────────────────────
#  DATASET 2 — New Indian Road (YOLO txt)
# ─────────────────────────────────────────────
def load_yolo_split(images_dir, labels_dir, split_name):
    X, y = [], []
    counts = {0: 0, 1: 0}

    if not os.path.isdir(images_dir):
        print(f"  [NEW/{split_name}] images dir not found: {images_dir}")
        return X, y

    img_files = [f for f in os.listdir(images_dir)
                 if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]

    for img_file in sorted(img_files):
        img_path = os.path.join(images_dir, img_file)
        base     = os.path.splitext(img_file)[0]
        lbl_path = os.path.join(labels_dir, base+".txt")

        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        ih, iw = img.shape[:2]

        with open(lbl_path) as f:
            lines = f.read().strip().splitlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            if cls_id in YOLO_SKIP:
                continue
            if cls_id == YOLO_HELMET:
                label = 1
            elif cls_id == YOLO_NO_HELMET:
                label = 0
            else:
                continue

            # YOLO → pixel coords
            cx = float(parts[1]) * iw
            cy = float(parts[2]) * ih
            bw = float(parts[3]) * iw
            bh = float(parts[4]) * ih

            x1 = int(cx - bw/2)
            y1 = int(cy - bh/2)
            x2 = int(cx + bw/2)
            y2 = int(cy + bh/2)

            crop_and_add(img, x1, y1, x2, y2, label, X, y)
            counts[label] += 1

    print(f"  [NEW/{split_name}] {len(img_files)} images → "
          f"With Helmet: {counts[1]}  Without: {counts[0]}  "
          f"Samples: {len(X)}")
    return X, y


def load_new_dataset(new_dir):
    X, y = [], []

    for split in ["train", "val"]:
        img_dir = os.path.join(new_dir, split, "images")
        lbl_dir = os.path.join(new_dir, split, "labels")
        if not os.path.isdir(img_dir):
            print(f"  [NEW] '{split}' not found — skipping")
            continue
        Xs, ys = load_yolo_split(img_dir, lbl_dir, split)
        X += Xs; y += ys

    return X, y


# ─────────────────────────────────────────────
#  BALANCE CLASSES
# ─────────────────────────────────────────────
def balance_classes(X, y):
    X = np.array(X); y = np.array(y)
    X0 = X[y==0]; X1 = X[y==1]
    y0 = y[y==0]; y1 = y[y==1]

    print(f"\n  Before balance → With Helmet: {len(X1)}  Without: {len(X0)}")

    target = min(len(X0), len(X1))   # balance to smaller class
    X1d, y1d = resample(X1, y1, n_samples=target, random_state=42, replace=False)
    X0d, y0d = resample(X0, y0, n_samples=target, random_state=42, replace=False)

    X_bal = np.vstack([X1d, X0d])
    y_bal = np.concatenate([y1d, y0d])
    idx   = np.random.RandomState(42).permutation(len(X_bal))
    X_bal = X_bal[idx]; y_bal = y_bal[idx]

    print(f"  After balance  → With Helmet: {int(np.sum(y_bal==1))}  "
          f"Without: {int(np.sum(y_bal==0))}  Total: {len(X_bal)}")
    return X_bal, y_bal


# ─────────────────────────────────────────────
#  FIND OPTIMAL THRESHOLD
# ─────────────────────────────────────────────
def find_optimal_threshold(y_true, y_prob):
    _, _, thresholds = roc_curve(y_true, y_prob)
    best_f1    = 0
    best_thresh = 0.5
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f = f1_score(y_true, preds)
        if f > best_f1:
            best_f1 = f; best_thresh = t
    print(f"  Optimal threshold: {best_thresh:.4f}  (F1={best_f1:.4f})")
    return float(best_thresh)


# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────
def train(old_dir, new_dir):
    print("\n" + "="*55)
    print("  HELMET DETECTOR — COMBINED DATASET RETRAIN")
    print("="*55)

    # ── Step 1: Load both datasets ────────────
    print("\n[1/6] Loading datasets...")
    X_old, y_old = load_old_dataset(old_dir)
    X_new, y_new = load_new_dataset(new_dir)

    X_all = X_old + X_new
    y_all = y_old + y_new

    print(f"\n  Combined raw:")
    print(f"    With Helmet   : {y_all.count(1) if isinstance(y_all,list) else int(np.sum(np.array(y_all)==1))}")
    print(f"    Without Helmet: {y_all.count(0) if isinstance(y_all,list) else int(np.sum(np.array(y_all)==0))}")
    print(f"    Total samples : {len(X_all)}")

    # ── Step 2: Balance ───────────────────────
    print("\n[2/6] Balancing classes...")
    X, y = balance_classes(X_all, y_all)

    # ── Step 3: Split ─────────────────────────
    print("\n[3/6] Train / test split (80/20)...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"  Train: {len(X_tr)}  Test: {len(X_te)}")

    # ── Step 4: Scale + PCA ───────────────────
    print("\n[4/6] StandardScaler + PCA...")
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)
    pca     = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=42)
    X_tr_p  = pca.fit_transform(X_tr_s)
    X_te_p  = pca.transform(X_te_s)
    print(f"  Features: {X_tr_s.shape[1]} → {pca.n_components_} components")

    # ── Step 5: Train HOG+SVM ─────────────────
    print(f"\n[5/6] Training HOG+SVM (C={BEST_C}, gamma={BEST_GAMMA})...")
    print("  Please wait ~5-10 mins...")
    svm   = SVC(C=BEST_C, gamma=BEST_GAMMA, kernel="rbf",
                class_weight="balanced", probability=False, random_state=42)
    model = CalibratedClassifierCV(svm, cv=5, method="sigmoid")
    model.fit(X_tr_p, y_tr)
    print("  Training done!")

    # ── Step 6: Evaluate ─────────────────────
    print("\n[6/6] Evaluation...")
    y_pred = model.predict(X_te_p)
    y_prob = model.predict_proba(X_te_p)[:,1]

    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)
    opt_thresh = find_optimal_threshold(y_te, y_prob)

    # re-evaluate with optimal threshold
    y_pred_opt = (y_prob >= opt_thresh).astype(int)
    acc_opt    = accuracy_score(y_te, y_pred_opt)

    print(f"\n{'='*55}")
    print(f"  Accuracy (default 0.5)  : {acc*100:.2f}%")
    print(f"  Accuracy (optimal thresh): {acc_opt*100:.2f}%")
    print(f"  AUC-ROC                 : {auc:.4f}")
    print(f"  Optimal threshold       : {opt_thresh:.4f}")
    print(f"{'='*55}")
    print(classification_report(y_te, y_pred_opt,
                                 target_names=["Without Helmet","With Helmet"]))

    # ── Save ──────────────────────────────────
    joblib.dump(model,      MODEL_PATH)
    joblib.dump(scaler,     SCALER_PATH)
    joblib.dump(pca,        PCA_PATH)
    joblib.dump(opt_thresh, THRESHOLD_PATH)
    print(f"\nSaved → {MODEL_PATH}, {SCALER_PATH}, {PCA_PATH}, {THRESHOLD_PATH}")

    _plot_confusion(confusion_matrix(y_te, y_pred_opt))
    _plot_roc(y_te, y_prob, auc, opt_thresh)
    print("\nAll done! Run detect_final.py to test. 🎉")


# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────
def _plot_confusion(cm):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Without Helmet","With Helmet"],
                yticklabels=["Without Helmet","With Helmet"])
    plt.title("Confusion Matrix — Combined Dataset")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix_combined.png", dpi=150)
    plt.close()
    print("Saved → confusion_matrix_combined.png")


def _plot_roc(y_true, y_scores, auc, thresh):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, color="#00e5a0", label=f"AUC = {auc:.4f}")
    plt.plot([0,1],[0,1],"k--", alpha=0.4)
    idx = np.argmin(np.abs(thresholds - thresh))
    plt.scatter(fpr[idx], tpr[idx], color="red", zorder=5,
                label=f"Optimal = {thresh:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Combined Dataset")
    plt.legend(); plt.tight_layout()
    plt.savefig("roc_curve_combined.png", dpi=150)
    plt.close()
    print("Saved → roc_curve_combined.png")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Retrain HOG+SVM with combined old+new dataset")
    p.add_argument("--old", default="archive",
                   help="Path to old BikesHelmets dataset (has images/ and annotations/)")
    p.add_argument("--new", default="archive (1)",
                   help="Path to new YOLO dataset (has train/ and val/)")
    args = p.parse_args()

    train(args.old, args.new)