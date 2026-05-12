"""
Helmet Detection — Final (Strict Head Crop Version)
====================================================
Matches training: tight head crop with asymmetric padding
  head_ratio = 0.25 (tighter — just head+helmet area)
  expand top more than bottom

Usage:
  python detect_final.py --image yourimage.png
  python detect_final.py --video yourvideo.mp4
  python detect_final.py --video 0   (webcam)
"""

import os, cv2, numpy as np, joblib
from skimage.feature import hog, local_binary_pattern

IMG_SIZE     = (64,64)
HOG_PARAMS   = dict(orientations=9,pixels_per_cell=(8,8),
                    cells_per_block=(2,2),block_norm="L2-Hys",
                    visualize=False,feature_vector=True)
LBP_RADIUS=3; LBP_N_POINTS=24

MODEL_PATH="helmet_svm.pkl"; SCALER_PATH="helmet_scaler.pkl"
PCA_PATH="helmet_pca.pkl";   THRESHOLD_PATH="helmet_threshold.pkl"

GREEN=(34,197,94); RED=(30,100,255); WHITE=(255,255,255); BLACK=(0,0,0)


def load_svm():
    model=joblib.load(MODEL_PATH); scaler=joblib.load(SCALER_PATH)
    pca=joblib.load(PCA_PATH)
    try:    thresh=joblib.load(THRESHOLD_PATH); print(f"  Threshold: {thresh:.4f}")
    except: thresh=0.60; print("  Using default threshold: 0.60")
    return model,scaler,pca,thresh


def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


def preprocess(img_bgr):
    gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)


def extract_features(img_gray):
    hf=hog(img_gray,**HOG_PARAMS)
    lbp=local_binary_pattern(img_gray,LBP_N_POINTS,LBP_RADIUS,method="uniform")
    n=LBP_N_POINTS+2; hist,_=np.histogram(lbp.ravel(),bins=n,range=(0,n),density=True)
    return np.concatenate([hf,hist])


def classify_head(crop_bgr,model,scaler,pca,threshold):
    if crop_bgr is None or crop_bgr.size==0: return "Unknown",0.0
    r=cv2.resize(crop_bgr,IMG_SIZE,interpolation=cv2.INTER_AREA)
    g=preprocess(r)
    f=extract_features(g).reshape(1,-1)
    f=pca.transform(scaler.transform(f))
    prob=model.predict_proba(f)[0][1]
    return ("With Helmet" if prob>=threshold else "Without Helmet"), prob


def get_head_rois(frame, yolo, yolo_conf=0.40):
    """
    Tight head crop matching training annotation style.
    Takes top 25% of person box + expands 30% upward for helmet.
    """
    h,w=frame.shape[:2]
    results=yolo(frame,verbose=False)[0]
    rois=[]
    for det in results.boxes:
        if int(det.cls[0])!=0 or float(det.conf[0])<yolo_conf: continue
        px1,py1,px2,py2=map(int,det.xyxy[0])
        px1=max(0,px1); py1=max(0,py1)
        px2=min(w,px2); py2=min(h,py2)
        ph=py2-py1; pw=px2-px1

        # tight head region — top 25% of person
        head_h = int(ph*0.25)
        # expand: 30% upward, 15% sides, 10% below
        ex_top  = int(head_h*0.30)
        ex_side = int(pw*0.15)
        ex_bot  = int(head_h*0.10)

        hx1=max(0,   px1-ex_side)
        hy1=max(0,   py1-ex_top)
        hx2=min(w,   px2+ex_side)
        hy2=min(h,   py1+head_h+ex_bot)

        if (hx2-hx1)>10 and (hy2-hy1)>10:
            rois.append((hx1,hy1,hx2,hy2))
    return rois


def draw_box(img,x1,y1,x2,y2,label,conf,color):
    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
    text=f"{'Helmet' if 'With' in label else 'No Helmet'} {conf:.2f}"
    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.52,1)
    cv2.rectangle(img,(x1,max(y1-22,0)),(x1+tw+6,max(y1,22)),color,-1)
    cv2.putText(img,text,(x1+3,max(y1-5,17)),cv2.FONT_HERSHEY_SIMPLEX,0.52,WHITE,1)


def draw_summary(img,hc,nhc):
    text=f"With Helmet: {hc}  |  Without: {nhc}"
    bw=len(text)*10+12
    cv2.rectangle(img,(0,0),(bw,28),BLACK,-1)
    cv2.putText(img,text,(6,19),cv2.FONT_HERSHEY_SIMPLEX,0.58,WHITE,1)


def process_frame(frame,yolo,model,scaler,pca,threshold,yolo_conf=0.40):
    rois=get_head_rois(frame,yolo,yolo_conf)
    result=frame.copy(); hc=0; nhc=0
    for (x1,y1,x2,y2) in rois:
        crop=frame[y1:y2,x1:x2]
        label,conf=classify_head(crop,model,scaler,pca,threshold)
        color=GREEN if label=="With Helmet" else RED
        draw_box(result,x1,y1,x2,y2,label,conf,color)
        if label=="With Helmet": hc+=1
        else: nhc+=1
    draw_summary(result,hc,nhc)
    return result,hc,nhc


def detect_image(image_path,yolo_conf=0.40,save_path=None):
    print("Loading models..."); yolo=load_yolo(); model,scaler,pca,thresh=load_svm()
    frame=cv2.imread(image_path)
    if frame is None: raise FileNotFoundError(image_path)
    result,hc,nhc=process_frame(frame,yolo,model,scaler,pca,thresh,yolo_conf)
    out=save_path or ("final_"+os.path.basename(image_path))
    cv2.imwrite(out,result)
    print(f"  With Helmet: {hc}  Without: {nhc}  Saved → {out}")


def detect_video(source,yolo_conf=0.40):
    print("Loading models..."); yolo=load_yolo(); model,scaler,pca,thresh=load_svm()
    src=int(source) if str(source).isdigit() else source
    cap=cv2.VideoCapture(src)
    if not cap.isOpened(): raise IOError(f"Cannot open: {source}")
    print("Press Q to quit.")
    while True:
        ret,frame=cap.read()
        if not ret: break
        result,_,_=process_frame(frame,yolo,model,scaler,pca,thresh,yolo_conf)
        cv2.imshow("Helmet Detection",result)
        if cv2.waitKey(1)&0xFF==ord("q"): break
    cap.release(); cv2.destroyAllWindows()


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--image",default=None)
    p.add_argument("--video",default=None)
    p.add_argument("--yolo_conf",type=float,default=0.40)
    p.add_argument("--save",default=None)
    args=p.parse_args()
    if args.image:
        detect_image(args.image,args.yolo_conf,args.save)
    elif args.video is not None:
        detect_video(args.video,args.yolo_conf)
    else:
        print("Provide --image <path> or --video <path/0>")