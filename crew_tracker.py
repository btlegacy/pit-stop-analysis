"""
CrewTracker - lightweight person re-id + track state machine for pit crew tracking.

Usage:
    from crew_tracker import CrewTracker
    tracker = CrewTracker(device='cpu', crew_dir='refs/crew', crew_wall_roi=(0,200,360,720))
    # in loop:
    tracker.update(person_dets, person_crops, frame, frame_idx, stop_start_frame)
    tracks = tracker.tracks  # dict of track_id -> info (role, state, hop_time, carrying, bbox, label_score...)
"""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
from collections import deque, OrderedDict

# ---- Parameters (tuneable) ----
DEFAULT_EMBED_DIM = 512
EMBED_MATCH_THRESH = 0.65      # cosine similarity threshold to re-id to templates/tracks
IOU_ASSOC_THRESH = 0.35
MAX_MISSING = 5
HISTORY_LEN = 64

# ---- Utilities ----
def l2_normalize(x):
    x = x / (np.linalg.norm(x) + 1e-8)
    return x

def center_of_box(b):
    x1,y1,x2,y2 = b
    return int((x1+x2)/2), int((y1+y2)/2)

# ---- Simple tire detection helper (Hough) ----
def detect_tire_in_crop(crop_gray):
    # Returns True if circular dark object (tire) is detected
    # small heuristic; tuned for top-down pit images where tire is dark circle
    try:
        img = cv2.medianBlur(crop_gray, 5)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                   param1=50, param2=18, minRadius=10, maxRadius=120)
        return circles is not None
    except Exception:
        return False

# ---- Appearance embedder (ResNet18 head) ----
class Embedder:
    def __init__(self, device='cpu'):
        self.device = device
        # Use small resnet18 pretrained for speed
        model = models.resnet18(pretrained=True)
        # remove classifier
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.to(self.device)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128,128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def embed(self, bgr_crop):
        if bgr_crop is None or getattr(bgr_crop, "size", 0) == 0:
            return None
        img = self.transform(bgr_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(img).cpu().numpy().squeeze()
        feat = l2_normalize(feat)
        return feat

# ---- CrewTracker class ----
class CrewTracker:
    def __init__(self, device='cpu', crew_dir='refs/crew', crew_wall_roi=(0,200,360,720),
                 embed_device='cpu'):
        self.device = device
        self.embedder = Embedder(device=embed_device)
        self.crew_templates = []
        self.load_crew_templates(crew_dir)
        self.tracks = OrderedDict()
        self.next_id = 1
        self.iou_thresh = IOU_ASSOC_THRESH
        self.max_missing = MAX_MISSING
        self.history_len = HISTORY_LEN
        self.crew_wall_roi = crew_wall_roi
        self.embed_match_thresh = EMBED_MATCH_THRESH

    def load_crew_templates(self, crew_dir):
        if not os.path.isdir(crew_dir):
            return
        for fname in sorted(os.listdir(crew_dir)):
            full = os.path.join(crew_dir, fname)
            if not os.path.isfile(full):
                continue
            img = cv2.imread(full)
            if img is None:
                continue
            label = os.path.splitext(fname)[0]
            # compute both HSV hist and embedding (embedding on color crop)
            hist = None
            try:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0,1], None, [32,32], [0,180,0,256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            except Exception:
                hist = None
            emb = self.embedder.embed(img)
            # role hint
            rl = None
            low = label.lower()
            if 'front' in low:
                rl='front_tire_changer'
            elif 'rear' in low:
                rl='rear_tire_changer'
            elif 'carrier' in low or 'tirecarrier' in low:
                rl='tire_carrier'
            elif 'fuel' in low or 'refu' in low:
                rl='fueler'
            self.crew_templates.append({'label':label,'hist':hist,'emb':emb,'role':rl})

    # internal track creation
    def _new_track(self, bbox, frame_idx, crop=None):
        tid = self.next_id; self.next_id += 1
        cx,cy = center_of_box(bbox)
        t = {
            'id': tid,
            'bbox': bbox,
            'last_seen': frame_idx,
            'missing': 0,
            'history': deque(maxlen=self.history_len),
            'label': None,
            'role': None,
            'label_score': 0.0,
            'emb': None,
            'tire_frames': {},
            'tire_cumulative': {},
            'was_on_wall': False,
            'hopped': False,
            'hop_time': None,
            'carrying_tire': False,
            'state': 'on_wall'  # or hopped, carrying, installing, done
        }
        t['history'].append((cx,cy))
        if crop is not None:
            emb = self.embedder.embed(crop)
            t['emb'] = emb
            # match to templates
            if emb is not None and self.crew_templates:
                best = None; best_score=-1.0; best_role=None; best_label=None
                for ct in self.crew_templates:
                    if ct.get('emb') is None:
                        continue
                    score = np.dot(emb, ct['emb'])
                    if score > best_score:
                        best_score=score; best=ct; best_role=ct.get('role'); best_label=ct.get('label')
                if best_score >= self.embed_match_thresh:
                    t['label'] = best_label; t['label_score']=best_score; t['role']=best_role
        # was_on_wall
        if self._point_in_roi((cx,cy), self.crew_wall_roi):
            t['was_on_wall'] = True
        self.tracks[tid] = t
        return t

    def _point_in_roi(self, point, roi):
        x,y = point
        x1,y1,x2,y2 = roi
        return (x >= x1 and x <= x2 and y >= y1 and y <= y2)

    # update existing track with det
    def _update_track(self, tid, bbox, frame_idx, crop=None):
        t = self.tracks.get(tid)
        if t is None:
            return self._new_track(bbox, frame_idx, crop)
        x1,y1,x2,y2 = bbox
        cx,cy = center_of_box(bbox)
        t['bbox'] = bbox
        t['last_seen'] = frame_idx
        t['missing'] = 0
        t['history'].append((cx,cy))
        if crop is not None:
            emb = self.embedder.embed(crop)
            if emb is not None:
                # update emb on moving average
                if t.get('emb') is None:
                    t['emb'] = emb
                else:
                    t['emb'] = l2_normalize(0.6 * t['emb'] + 0.4 * emb)
                # try to improve label
                for ct in self.crew_templates:
                    if ct.get('emb') is None: continue
                    score = np.dot(t['emb'], ct['emb'])
                    if score > t.get('label_score', 0.0) and score >= self.embed_match_thresh:
                        t['label'] = ct['label']; t['role'] = ct.get('role'); t['label_score'] = score
        return t

    # association helper: greedy IoU first, then embed fallback
    def update(self, person_dets, person_crops, frame, frame_idx, stop_start_frame=None):
        """
        person_dets - list of xyxy boxes
        person_crops - corresponding color crops (BGR)
        frame - BGR frame (for extra analysis)
        frame_idx - current frame index
        stop_start_frame - frame index when car stopped (or None)
        """
        matched = set()
        # prepare track list
        track_ids = list(self.tracks.keys())
        if track_ids and person_dets:
            iou_mat = np.zeros((len(track_ids), len(person_dets)), dtype=np.float32)
            for ti, tid in enumerate(track_ids):
                tb = self.tracks[tid]['bbox']
                for di, det in enumerate(person_dets):
                    iou_mat[ti,di] = self._iou_np(tb, det)
            # greedy
            while True:
                if iou_mat.size == 0:
                    break
                tdi = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                best_val = iou_mat[tdi]
                if best_val <= self.iou_thresh:
                    break
                ti, di = tdi
                tid = track_ids[ti]
                if tid in matched or di in matched:
                    iou_mat[ti,di] = -1.0
                    continue
                # update
                self._update_track(tid, person_dets[di], frame_idx, crop=person_crops[di])
                matched.add(di)
                # invalidate
                iou_mat[ti,:] = -1.0; iou_mat[:,di] = -1.0

        # unmatched detections -> create new tracks
        for di, det in enumerate(person_dets):
            if di in matched:
                continue
            self._new_track(det, frame_idx, crop=person_crops[di])

        # bump missing counters and prune
        for tid in list(self.tracks.keys()):
            t = self.tracks[tid]
            if t['last_seen'] != frame_idx:
                t['missing'] = t.get('missing',0) + 1
            else:
                t['missing'] = 0
            if t['missing'] > MAX_MISSING:
                del self.tracks[tid]

        # detect hop and carrying / roles and update states
        for tid, t in self.tracks.items():
            if 'bbox' not in t:
                continue
            bx1,by1,bx2,by2 = t['bbox']
            cx,cy = t['history'][-1]
            # hop detection
            if t.get('was_on_wall') and not t.get('hopped'):
                if not self._point_in_roi((cx,cy), self.crew_wall_roi):
                    t['hopped'] = True
                    if stop_start_frame is not None:
                        t['hop_time'] = (frame_idx - stop_start_frame) / (1.0 if frame is None else 1.0)  # caller should pass fps separately if desired
                    t['state'] = 'hopped'
            # carrying/tire detection (simple heuristics)
            carrying = False
            # check crop for tire shape
            crop = t.get('last_person_crop')
            if crop is not None:
                g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if detect_tire_in_crop(g):
                    carrying = True
            # update flags & state machine
            if carrying:
                t['carrying_tire'] = True
                if t['state'] != 'carrying':
                    t['state'] = 'carrying'
            # labelling via role_hint persists (done earlier)

    def _iou_np(self, a, b):
        ax1,ay1,ax2,ay2 = a
        bx1,by1,bx2,by2 = b
        interw = max(0, min(ax2,bx2) - max(ax1,bx1))
        interh = max(0, min(ay2,by2) - max(ay1,by1))
        inter = interw * interh
        areaA = max(0, (ax2-ax1) * (ay2-ay1))
        areaB = max(0, (bx2-bx1) * (by2-by1))
        union = areaA + areaB - inter
        return inter/(union+1e-6)
