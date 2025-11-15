"""
Lightweight annotation loader & applicator for video_processor + CrewTracker.

Place CSVs in refs/annotations.csv with columns:
frame,x1,y1,x2,y2,label

Example CSV lines (no header required; '#' lines are ignored):
# frame,x1,y1,x2,y2,label
123,950,120,1010,300,fueler
"""

import os
import csv
import cv2
import numpy as np

def load_annotations(csv_path='refs/annotations.csv'):
    """
    Returns dict: frame_idx -> list of {bbox:(x1,y1,x2,y2), label:str}
    """
    ann = {}
    if not os.path.exists(csv_path):
        return ann
    with open(csv_path, newline='') as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            if row[0].strip().startswith('#'):
                continue
            try:
                frame = int(row[0])
                x1 = int(float(row[1])); y1 = int(float(row[2]))
                x2 = int(float(row[3])); y2 = int(float(row[4]))
                label = row[5].strip() if len(row) > 5 else 'unknown'
            except Exception:
                # skip malformed rows
                continue
            ann.setdefault(frame, []).append({'bbox':(x1,y1,x2,y2),'label':label})
    return ann

def iou_xyxy(a, b):
    """IoU for two xyxy boxes"""
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0, (ax2-ax1)) * max(0, (ay2-ay1))
    area_b = max(0, (bx2-bx1)) * max(0, (by2-by1))
    denom = (area_a + area_b - inter)
    if denom <= 0:
        return 0.0
    return inter / denom

def apply_annotations_to_tracker(tracker, ann_list, frame, frame_idx, iou_thresh=0.2, add_template=True):
    """
    ann_list: list of {'bbox':(x1,y1,x2,y2),'label':str} for this frame
    tracker: CrewTracker instance (from crew_tracker.py)
    frame: full BGR frame (so we can crop to build a template/embedding)
    frame_idx: current frame index (for diagnostics)
    iou_thresh: minimum IoU to consider a match between annotation bbox and track bbox
    add_template: if True and we can crop a person, also add embedding+hist to tracker.crew_templates
    """
    if not ann_list:
        return

    # For each annotation, find the best matching track (by IoU)
    for ann in ann_list:
        abox = ann['bbox']
        alabel = ann['label']
        best_tid = None
        best_iou = 0.0
        for tid, t in tracker.tracks.items():
            if 'bbox' not in t:
                continue
            tb = t['bbox']
            i = iou_xyxy(abox, tb)
            if i > best_iou:
                best_iou = i
                best_tid = tid
        if best_tid is not None and best_iou >= iou_thresh:
            # force label/role on this track
            t = tracker.tracks[best_tid]
            t['label'] = alabel
            # set role to label if there isn't a role already
            if t.get('role') is None:
                t['role'] = alabel
            t['label_score'] = 1.0
            t.setdefault('manual_annotations', []).append({'frame':frame_idx, 'label':alabel, 'iou':best_iou})
            # optionally add this crop embedding to crew_templates so tracker learns this appearance
            if add_template and frame is not None:
                x1,y1,x2,y2 = abox
                h,w = frame.shape[:2]
                x1,y1 = max(0, x1), max(0, y1)
                x2,y2 = min(w-1, x2), min(h-1, y2)
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2].copy()
                    # compute HSV hist
                    try:
                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        hist = cv2.calcHist([hsv],[0,1],None,[32,32],[0,180,0,256])
                        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                    except Exception:
                        hist = None
                    emb = None
                    try:
                        # tracker is expected to have embedder attribute (Embedder.embed returns normalized vector)
                        if hasattr(tracker, 'embedder') and tracker.embedder is not None:
                            emb = tracker.embedder.embed(crop)
                    except Exception:
                        emb = None
                    # append a manual template entry to tracker.crew_templates
                    tracker.crew_templates.append({'label': f"{alabel}_manual_{frame_idx}", 'hist': hist, 'emb': emb, 'role': alabel})
        else:
            # no matching track -> ignore for now (could create synthetic track if desired)
            pass
