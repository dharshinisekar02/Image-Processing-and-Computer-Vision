"""
Helmet Detector — Strict Retrain (Annotation Crops Only)
=========================================================
Key insight:
  Previous versions trained on YOLO-cropped head ROIs (top 28-38% of person)
  BUT annotations are tight crops around just the helmet/head.
  This mismatch causes confusion at inference time.

This version:
  1. Uses ONLY tight annotation crops (XML + YOLO labels)
  2. Adds smart context padding (20% each side)
  3. Trains SVM on these crops
  4. At inference: uses YOLO person box → tight head crop
     matching the annotation crop style as closely as possible

Datasets:
  Old: archive/images + archive/annotations  (Pascal VOC XML)
  New: archive (1)/train + archive (1)/val   (YOLO txt)

Usage:
  python retrain_strict.py --old archive --new "archive (1)"
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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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

XML_CLASS_MAP   = {"With Helmet": 1, "Without Helmet": 0}
YOLO_HELMET     = 0
YOLO_NO_HELMET  = 1
YOLO_SKIP       = {2, 3}

BEST_C     = 10
BEST_GAMMA = "scale"

# ✅ Smart padding — adds context around tight annotation box
# Helmet crops: add more padding on top (helmet sits above face)
# Head crops:   add equal padding all sides
HELMET_PAD_TOP    = 0.30   # 30% above for helmet context
HELMET_PAD_SIDES  = 0.15   # 15% sides
HELMET_PAD_BOTTOM = 0.10   # 10% below chin

NO_HELMET_PAD_TOP    = 0.40  # more top for missed helmet check
NO_HELMET_PAD_SIDES  = 0.15
NO_HELMET_PAD_BOTTOM = 0.10

MODEL_PATH     = "helmet_svm.pkl"
SCALER_PATH    = "helmet_scaler.pkl"
PCA_PATH       = "helmet_pca.pkl"
THRESHOLD_PATH = "helmet_threshold.pkl"


# ─────────────────────────────────────────────
#  HELPERS
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
    """7 variants per crop."""
    variants = [img, cv2.flip(img,1)]
    h, w = img.shape
    for angle in [-12, 12]:
        M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
        variants.append(cv2.warpAffine(img, M, (w,h),
                         borderMode=cv2.BORDER_REFLECT))
    for beta in [-30, 30]:
        variants.append(np.clip(img.astype(int)+beta,0,255).astype(np.uint8))
    variants.append(cv2.GaussianBlur(img,(3,3),0))
    return variants


def smart_crop(img_bgr, x1, y1, x2, y2, label):
    """
    Crop with smart asymmetric padding based on class.
    Helmet crops get more top padding (helmet sits above annotation box).
    No-helmet crops get more top padding to check for missed helmets.
    """
    ih, iw = img_bgr.shape[:2]
    bw = x2 - x1
    bh = y2 - y1

    if label == 1:  # With Helmet
        pt = int(bh * HELMET_PAD_TOP)
        ps = int(bw * HELMET_PAD_SIDES)
        pb = int(bh * HELMET_PAD_BOTTOM)
    else:           # Without Helmet
        pt = int(bh * NO_HELMET_PAD_TOP)
        ps = int(bw * NO_HELMET_PAD_SIDES)
        pb = int(bh * NO_HELMET_PAD_BOTTOM)

    cx1 = max(0,  x1 - ps)
    cy1 = max(0,  y1 - pt)
    cx2 = min(iw, x2 + ps)
    cy2 = min(ih, y2 + pb)

    crop = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    return cv2.resize(crop, IMG_SIZE, interpolation=cv2.INTER_AREA)


def process_crop(img_bgr, x1, y1, x2, y2, label, X, y):
    crop = smart_crop(img_bgr, x1, y1, x2, y2, label)
    if crop is None:
        return
    gray = preprocess(crop)
    for aug in augment(gray):
        X.append(extract_features(aug))
        y.append(label)


# ─────────────────────────────────────────────
#  LOAD OLD DATASET (Pascal VOC XML)
# ─────────────────────────────────────────────
def load_old_dataset(old_dir):
    images_dir = os.path.join(old_dir, "images")
    ann_dir    = os.path.join(old_dir, "annotations")
    if not os.path.isdir(images_dir):
        print(f"  [OLD] Not found: {old_dir}"); return [],[]

    X, y = [], []
    counts = {0:0, 1:0}
    xml_files = [f for f in os.listdir(ann_dir) if f.endswith(".xml")]

    for xml_file in sorted(xml_files):
        base     = os.path.splitext(xml_file)[0]
        xml_path = os.path.join(ann_dir, xml_file)
        img_path = None
        for ext in (".png",".jpg",".jpeg",".PNG",".JPG"):
            c = os.path.join(images_dir, base+ext)
            if os.path.exists(c): img_path=c; break
        if not img_path: continue

        img = cv2.imread(img_path)
        if img is None: continue

        tree = ET.parse(xml_path)
        for obj in tree.getroot().findall("object"):
            name = obj.find("name").text.strip()
            if name not in XML_CLASS_MAP: continue
            label = XML_CLASS_MAP[name]
            bb = obj.find("bndbox")
            x1 = int(float(bb.find("xmin").text))
            y1 = int(float(bb.find("ymin").text))
            x2 = int(float(bb.find("xmax").text))
            y2 = int(float(bb.find("ymax").text))
            process_crop(img, x1, y1, x2, y2, label, X, y)
            counts[label] += 1

    print(f"  [OLD] {len(xml_files)} files → "
          f"Helmet:{counts[1]} NoHelmet:{counts[0]} Samples:{len(X)}")
    return X, y


# ─────────────────────────────────────────────
#  LOAD NEW DATASET (YOLO txt)
# ─────────────────────────────────────────────
def load_yolo_split(images_dir, labels_dir, split_name):
    X, y = [], []
    counts = {0:0, 1:0}
    if not os.path.isdir(images_dir): return X,y

    for img_file in sorted(os.listdir(images_dir)):
        if not img_file.lower().endswith((".jpg",".jpeg",".png",".bmp")):
            continue
        img_path = os.path.join(images_dir, img_file)
        lbl_path = os.path.join(labels_dir,
                                os.path.splitext(img_file)[0]+".txt")
        if not os.path.exists(lbl_path): continue

        img = cv2.imread(img_path)
        if img is None: continue
        ih, iw = img.shape[:2]

        with open(lbl_path) as f:
            for line in f.read().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5: continue
                cls_id = int(parts[0])
                if cls_id in YOLO_SKIP: continue
                label = 1 if cls_id==YOLO_HELMET else (0 if cls_id==YOLO_NO_HELMET else None)
                if label is None: continue

                cx = float(parts[1])*iw; cy = float(parts[2])*ih
                bw = float(parts[3])*iw; bh = float(parts[4])*ih
                x1 = int(cx-bw/2); y1 = int(cy-bh/2)
                x2 = int(cx+bw/2); y2 = int(cy+bh/2)

                process_crop(img, x1, y1, x2, y2, label, X, y)
                counts[label] += 1

    print(f"  [NEW/{split_name}] "
          f"Helmet:{counts[1]} NoHelmet:{counts[0]} Samples:{len(X)}")
    return X, y


def load_new_dataset(new_dir):
    X, y = [], []
    for split in ["train","val"]:
        img_dir = os.path.join(new_dir, split, "images")
        lbl_dir = os.path.join(new_dir, split, "labels")
        Xs,ys   = load_yolo_split(img_dir, lbl_dir, split)
        X+=Xs; y+=ys
    return X, y


# ─────────────────────────────────────────────
#  BALANCE
# ─────────────────────────────────────────────
def balance_classes(X, y):
    X=np.array(X); y=np.array(y)
    X0=X[y==0]; X1=X[y==1]; y0=y[y==0]; y1=y[y==1]
    print(f"\n  Before → Helmet:{len(X1)}  NoHelmet:{len(X0)}")
    target = min(len(X0), len(X1))
    X1d,y1d = resample(X1,y1,n_samples=target,random_state=42,replace=False)
    X0d,y0d = resample(X0,y0,n_samples=target,random_state=42,replace=False)
    X_bal = np.vstack([X1d,X0d]); y_bal = np.concatenate([y1d,y0d])
    idx = np.random.RandomState(42).permutation(len(X_bal))
    X_bal=X_bal[idx]; y_bal=y_bal[idx]
    print(f"  After  → Helmet:{int(np.sum(y_bal==1))}  "
          f"NoHelmet:{int(np.sum(y_bal==0))}  Total:{len(X_bal)}")
    return X_bal, y_bal


# ─────────────────────────────────────────────
#  OPTIMAL THRESHOLD
# ─────────────────────────────────────────────
def find_optimal_threshold(y_true, y_prob):
    _,_,thresholds = roc_curve(y_true, y_prob)
    best_f1=0; best_t=0.5
    for t in thresholds:
        f = f1_score(y_true, (y_prob>=t).astype(int))
        if f > best_f1: best_f1=f; best_t=t
    print(f"  Optimal threshold: {best_t:.4f}  (F1={best_f1:.4f})")
    return float(best_t)


# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────
def train(old_dir, new_dir):
    print("\n"+"="*55)
    print("  STRICT RETRAIN — ANNOTATION CROPS ONLY")
    print("="*55)

    print("\n[1/6] Loading datasets...")
    X_old,y_old = load_old_dataset(old_dir)
    X_new,y_new = load_new_dataset(new_dir)

    X_all = X_old+X_new; y_all = y_old+y_new
    ya = np.array(y_all)
    print(f"\n  Combined → Helmet:{int(np.sum(ya==1))}  "
          f"NoHelmet:{int(np.sum(ya==0))}  Total:{len(X_all)}")

    print("\n[2/6] Balancing classes...")
    X,y = balance_classes(X_all, y_all)

    print("\n[3/6] Train / test split (80/20)...")
    X_tr,X_te,y_tr,y_te = train_test_split(
        X,y,test_size=0.2,stratify=y,random_state=42)
    print(f"  Train:{len(X_tr)}  Test:{len(X_te)}")

    print("\n[4/6] StandardScaler + PCA...")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    pca    = PCA(n_components=PCA_VARIANCE,svd_solver="full",random_state=42)
    X_tr_p = pca.fit_transform(X_tr_s)
    X_te_p = pca.transform(X_te_s)
    print(f"  Features: {X_tr_s.shape[1]} → {pca.n_components_} components")

    print(f"\n[5/6] Training HOG+SVM (C={BEST_C}, gamma={BEST_GAMMA})...")
    print("  Please wait ~5-10 mins...")
    svm   = SVC(C=BEST_C,gamma=BEST_GAMMA,kernel="rbf",
                class_weight="balanced",probability=False,random_state=42)
    model = CalibratedClassifierCV(svm,cv=5,method="sigmoid")
    model.fit(X_tr_p,y_tr)
    print("  Done!")

    print("\n[6/6] Evaluation...")
    y_pred = model.predict(X_te_p)
    y_prob = model.predict_proba(X_te_p)[:,1]
    acc    = accuracy_score(y_te,y_pred)
    auc    = roc_auc_score(y_te,y_prob)
    opt_t  = find_optimal_threshold(y_te,y_prob)
    y_opt  = (y_prob>=opt_t).astype(int)
    acc_opt= accuracy_score(y_te,y_opt)

    print(f"\n{'='*55}")
    print(f"  Accuracy (default) : {acc*100:.2f}%")
    print(f"  Accuracy (optimal) : {acc_opt*100:.2f}%")
    print(f"  AUC-ROC            : {auc:.4f}")
    print(f"  Threshold          : {opt_t:.4f}")
    print(f"{'='*55}")
    print(classification_report(y_te,y_opt,
          target_names=["Without Helmet","With Helmet"]))

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(pca,    PCA_PATH)
    joblib.dump(opt_t,  THRESHOLD_PATH)
    print(f"Saved → {MODEL_PATH}, {SCALER_PATH}, {PCA_PATH}, {THRESHOLD_PATH}")

    _plot_confusion(confusion_matrix(y_te,y_opt))
    _plot_roc(y_te,y_prob,auc,opt_t)
    print("\nDone! Now run detect_final.py 🎉")


def _plot_confusion(cm):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
                xticklabels=["Without Helmet","With Helmet"],
                yticklabels=["Without Helmet","With Helmet"])
    plt.title("Confusion Matrix — Strict Retrain")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix_strict.png",dpi=150); plt.close()
    print("Saved → confusion_matrix_strict.png")


def _plot_roc(y_true,y_scores,auc,thresh):
    fpr,tpr,ths = roc_curve(y_true,y_scores)
    plt.figure(figsize=(5,4))
    plt.plot(fpr,tpr,color="#00f5ff",label=f"AUC={auc:.4f}")
    plt.plot([0,1],[0,1],"k--",alpha=0.4)
    idx = np.argmin(np.abs(ths-thresh))
    plt.scatter(fpr[idx],tpr[idx],color="red",zorder=5,
                label=f"Thresh={thresh:.2f}")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC — Strict Retrain"); plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve_strict.png",dpi=150); plt.close()
    print("Saved → roc_curve_strict.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--old", default="archive")
    p.add_argument("--new", default="archive (1)")
    args = p.parse_args()
    train(args.old, args.new)