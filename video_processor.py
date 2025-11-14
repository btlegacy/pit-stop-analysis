import cv2
import numpy as np
import os
from ultralytics import YOLO
import math
from collections import deque

# -------------------- Utilities --------------------
def get_patch_signature(frame, roi):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, int(roi[0])), max(0, int(roi[1])), min(w, int(roi[2])), min(h, int(roi[3]))
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return np.array([0, 0, 0])
    return np.mean(patch, axis=(0, 1))

def boxes_overlap_area(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)

def iou(boxA, boxB):
    # boxes are (x1,y1,x2,y2) or (x,y,w,h) — we assume xyxy here
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    interW = max(0, min(ax2, bx2) - max(ax1, bx1))
    interH = max(0, min(ay2, by2) - max(ay1, by1))
    inter = interW * interH
    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = (areaA + areaB - inter) if (areaA + areaB - inter) > 0 else 1e-6
    return inter / union

# ---------------------- Crew template helpers ----------------------
def load_crew_templates(dir_path):
    templates = []
    if not os.path.isdir(dir_path):
        return templates
    for fname in sorted(os.listdir(dir_path)):
        full = os.path.join(dir_path, fname)
        if not os.path.isfile(full):
            continue
        img = cv2.imread(full)
        if img is None:
            continue
        label = os.path.splitext(fname)[0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = calc_hsv_hist(img)
        templates.append({'label': label, 'gray': gray, 'hist': hist})
    return templates

def calc_hsv_hist(bgr_patch):
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def match_crew_by_hist(bgr_patch, crew_templates):
    if bgr_patch is None or bgr_patch.size == 0 or not crew_templates:
        return None, 0.0
    patch_hist = calc_hsv_hist(bgr_patch)
    best_score = -1.0
    best_label = None
    for ct in crew_templates:
        score = float(cv2.compareHist(ct['hist'], patch_hist, cv2.HISTCMP_CORREL))
        if score > best_score:
            best_score = score
            best_label = ct['label']
    return best_label, best_score

# ---------------------- Template helpers ----------------------
def load_templates_from_dir(dir_path):
    templates = []
    if not os.path.isdir(dir_path):
        return templates
    for fname in sorted(os.listdir(dir_path)):
        full = os.path.join(dir_path, fname)
        if not os.path.isfile(full):
            continue
        img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        templates.append(img)
    return templates

def best_match_template(search_area_gray, templates):
    if not templates:
        return None, None, None
    best_score = -1.0
    best_idx = None
    best_t = None
    for i, t in enumerate(templates):
        th, tw = t.shape[:2]
        if search_area_gray.shape[0] < th or search_area_gray.shape[1] < tw:
            continue
        res = cv2.matchTemplate(search_area_gray, t, cv2.TM_CCOEFF_NORMED)
        if res is None:
            continue
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = float(max_val)
            best_idx = max_loc
            best_t = t
    if best_idx is None:
        return None, None, None
    return best_score, best_idx, best_t

# ---------------------- Tire-area helpers ----------------------
def load_tire_rois_from_image(ref_path, frame_width, frame_height):
    if not os.path.exists(ref_path):
        return []
    img = cv2.imread(ref_path)
    if img is None:
        return []
    ref_h, ref_w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 100, 100]); upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 100, 100]); upper2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1); mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 8 or h < 8:
            continue
        scale_x = frame_width / ref_w
        scale_y = frame_height / ref_h
        fx1 = int(x * scale_x); fy1 = int(y * scale_y)
        fx2 = int((x + w) * scale_x); fy2 = int((y + h) * scale_y)
        fx1, fy1 = max(0, fx1), max(0, fy1)
        fx2, fy2 = min(frame_width, fx2), min(frame_height, fy2)
        if fx2 > fx1 and fy2 > fy1:
            rois.append((fx1, fy1, fx2, fy2))
    rois = sorted(rois, key=lambda r: (r[1], r[0]))
    return rois

# ---------------------- Main processing ----------------------
def process_video(video_path, output_path, progress_callback):
    model = YOLO('yolov8n.pt')

    # --- Crew templates (optional) ---
    crew_templates = load_crew_templates('refs/crew')
    MIN_CREW_MATCH = 0.45

    # --- Probe multi-template support ---
    probe_in_templates = load_templates_from_dir('refs/probein')
    probe_out_templates = load_templates_from_dir('refs/probeout')
    if not probe_in_templates and os.path.exists('refs/probe_in.png'):
        t = cv2.imread('refs/probe_in.png', cv2.IMREAD_GRAYSCALE)
        if t is not None:
            probe_in_templates = [t]
    if not probe_out_templates and os.path.exists('refs/probe_out.png'):
        t = cv2.imread('refs/probe_out.png', cv2.IMREAD_GRAYSCALE)
        if t is not None:
            probe_out_templates = [t]
    if not probe_in_templates or not probe_out_templates:
        print("Error: probe-in or probe-out templates missing. Place images in refs/probein and refs/probeout or use fallback files.")
        return [0.0] * 3

    # --- ROIs and other setup (defaults preserved) ---
    ref_roi = (1042, 463, 1059, 487)
    tire_rois_fallback = [(1210, 30, 1370, 150), (1210, 400, 1400, 550), (685, 10, 830, 100), (685, 430, 780, 500)]
    refuel_roi_in_air = (803, 328, 920, 460)

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Attempt load tire ROIs from user-provided image
    tire_rois_from_image = load_tire_rois_from_image('refs/tirechangeareas.png', width, height)
    tire_rois = tire_rois_from_image if tire_rois_from_image else tire_rois_fallback

    # Car stop detection params (kept from earlier)
    unobstructed_signature, last_car_stop_sig = None, None
    CAR_STOP_THRESH = 15
    STOP_CONFIRM_FRAMES = int(fps / 5)
    stopped_frames_count, is_car_stopped, stop_start_frame = 0, False, 0

    # global timers
    total_stopped_time, tire_change_time, refuel_time = 0.0, 0.0, 0.0

    # probe/refuel tracking state
    refuel_bbox = None
    is_refueling_state = False
    TRACK_LOST_THRESHOLD = 0.60
    active_probe_template = None
    active_template_w = None
    active_template_h = None

    # ---------------- Person-tracking state ----------------
    tracks = {}            # track_id -> track dict
    next_track_id = 1
    IOU_THRESH = 0.35
    MAX_MISSING_FRAMES = max(2, int(fps * 0.15))  # tolerate short occlusions
    TRAJ_HISTORY = int(fps * 1.5)  # keep ~1.5s history
    TIRES_SECONDS_TO_COUNT = 0.5   # sustained presence seconds to count
    TIRES_FRAMES_TO_COUNT = max(1, int(fps * TIRES_SECONDS_TO_COUNT))

    # per-track structure:
    # { 'id', 'bbox'(xyxy), 'last_seen', 'missing', 'history' deque of centers,
    #   'label', 'label_score', 'tire_frames': dict roi_idx->consecutive frames,
    #   'tire_cumulative': dict roi_idx->seconds }
    for roi in tire_rois:
        pass

    # optional: prepare crew matching use
    has_crew = True if crew_templates else False

    # helper to create new track
    def create_track(bbox, frame_idx, crop=None):
        nonlocal next_track_id
        tid = next_track_id
        next_track_id += 1
        x1,y1,x2,y2 = bbox
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        t = {
            'id': tid,
            'bbox': bbox,
            'last_seen': frame_idx,
            'missing': 0,
            'history': deque(maxlen=TRAJ_HISTORY),
            'label': None,
            'label_score': 0.0,
            'tire_frames': {i:0 for i in range(len(tire_rois))},
            'tire_cumulative': {i:0.0 for i in range(len(tire_rois))},
            'last_person_crop': crop
        }
        t['history'].append((cx, cy))
        # attempt crew label immediately if crop available
        if crop is not None and has_crew:
            lbl, sc = match_crew_by_hist(crop, crew_templates)
            if lbl is not None and sc >= MIN_CREW_MATCH:
                t['label'] = lbl
                t['label_score'] = sc
        tracks[tid] = t
        return t

    # helper: update a track with matched bbox
    def update_track(t, bbox, frame_idx, crop=None):
        x1,y1,x2,y2 = bbox
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        t['bbox'] = bbox
        t['last_seen'] = frame_idx
        t['missing'] = 0
        t['history'].append((cx, cy))
        if crop is not None:
            t['last_person_crop'] = crop
            if has_crew:
                lbl, sc = match_crew_by_hist(crop, crew_templates)
                if lbl is not None and sc >= MIN_CREW_MATCH and sc > t.get('label_score', 0.0):
                    t['label'] = lbl
                    t['label_score'] = sc

    # main loop
    last_gray = None
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        progress_callback(frame_idx / total_frames)

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run YOLO person detection (reuse existing call pattern)
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        annotated_frame = results[0].plot()

        # prepare person detections as xyxy int list and keep crops
        person_dets = []
        person_crops = []
        if results[0].boxes:
            for b in results[0].boxes:
                if int(b.cls) != 0:
                    continue
                xyxy = b.xyxy[0].cpu().numpy()
                x1,y1,x2,y2 = [int(v) for v in xyxy]
                x1,y1 = max(0, x1), max(0, y1)
                x2,y2 = min(width-1, x2), min(height-1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                person_dets.append((x1,y1,x2,y2))
                crop = frame[y1:y2, x1:x2].copy()
                person_crops.append(crop)

        # Car stopped logic (same as before)
        if frame_idx == 0:
            unobstructed_signature = get_patch_signature(frame, ref_roi)

        current_car_stop_sig = get_patch_signature(frame, ref_roi)
        is_obstructed = np.linalg.norm(current_car_stop_sig - unobstructed_signature) > CAR_STOP_THRESH
        car_is_moving = True
        if is_obstructed:
            if last_car_stop_sig is not None and np.linalg.norm(current_car_stop_sig - last_car_stop_sig) < CAR_STOP_THRESH:
                stopped_frames_count += 1
            else:
                stopped_frames_count = 0
            if stopped_frames_count > STOP_CONFIRM_FRAMES:
                car_is_moving = False
        else:
            stopped_frames_count = 0
        last_car_stop_sig = current_car_stop_sig

        if not car_is_moving and not is_car_stopped:
            is_car_stopped, stop_start_frame = True, frame_idx
            # reset track per-stop counters if desired
            for t in tracks.values():
                for k in t['tire_frames']:
                    t['tire_frames'][k] = 0
        elif car_is_moving and is_car_stopped:
            is_car_stopped, is_refueling_state, refuel_bbox = False, False, None
            total_stopped_time += (frame_idx - stop_start_frame) / fps
            stop_start_frame = 0

        # ---------- Person tracking update (IoU matching) ----------
        # Build IoU matrix between existing tracks and detections
        matched_det_idx = set()
        matched_track_ids = set()
        if person_dets:
            track_ids = list(tracks.keys())
            iou_mat = np.zeros((len(track_ids), len(person_dets)), dtype=np.float32)
            for ti, tid in enumerate(track_ids):
                tb = tracks[tid]['bbox']
                for di, det in enumerate(person_dets):
                    iou_mat[ti, di] = iou(tb, det)
            # Greedy match: pick highest IoU pairs above threshold
            while True:
                tdi = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                best_val = iou_mat[tdi]
                if best_val <= IOU_THRESH:
                    break
                ti, di = tdi
                tid = track_ids[ti]
                if tid in matched_track_ids or di in matched_det_idx:
                    iou_mat[ti, di] = -1.0
                    continue
                # assign
                update_track(tracks[tid], person_dets[di], frame_idx, crop=person_crops[di])
                matched_track_ids.add(tid)
                matched_det_idx.add(di)
                # invalidate row and column
                iou_mat[ti, :] = -1.0
                iou_mat[:, di] = -1.0

        # create tracks for unmatched detections
        for di, det in enumerate(person_dets):
            if di in matched_det_idx:
                continue
            crop = person_crops[di]
            t = create_track(det, frame_idx, crop=crop)

        # increase missing counters for unmatched tracks
        for tid, t in list(tracks.items()):
            if t['last_seen'] != frame_idx:
                t['missing'] += 1
            else:
                t['missing'] = 0
            if t['missing'] > MAX_MISSING_FRAMES:
                # remove track
                del tracks[tid]

        # ---------- If car stopped, evaluate tire activity using tracks ----------
        if is_car_stopped:
            # legacy person-based ROI check (kept for robustness)
            person_bboxes = person_dets
            if sum(boxes_overlap_area(p, troi) for p in person_bboxes for troi in tire_rois) > 500:
                tire_change_time += 1.0 / fps

            # use tracks to attribute activity
            if last_gray is not None:
                for tid, t in tracks.items():
                    # compute person bbox and crop if present (use last_person_crop)
                    if 'bbox' not in t:
                        continue
                    bx1, by1, bx2, by2 = t['bbox']
                    # ensure in bounds
                    bx1, by1, bx2, by2 = max(0, bx1), max(0, by1), min(width-1, bx2), min(height-1, by2)
                    if bx2 <= bx1 or by2 <= by1:
                        continue
                    # person motion within their bbox
                    prev_patch = last_gray[by1:by2, bx1:bx2]
                    curr_patch = curr_gray[by1:by2, bx1:bx2]
                    motion_amt = 0.0
                    if prev_patch.size and curr_patch.size and prev_patch.shape == curr_patch.shape:
                        motion_amt = float(np.mean(cv2.absdiff(prev_patch, curr_patch)))

                    # check each tire ROI for overlap with this track
                    for ridx, troi in enumerate(tire_rois):
                        tx1, ty1, tx2, ty2 = troi
                        overlap_area = boxes_overlap_area((bx1,by1,bx2,by2), troi)
                        if overlap_area > 0:
                            # increment person presence frames in this ROI
                            t['tire_frames'][ridx] += 1
                            # if motion exceeds small threshold, consider it active motion
                            MOTION_THRESH = 6.0
                            if motion_amt > MOTION_THRESH:
                                # count frame as active
                                t['tire_cumulative'][ridx] += 1.0 / fps
                                tire_change_time += 1.0 / fps
                                # annotate ROI as active on frame
                                cv2.rectangle(annotated_frame, (tx1, ty1), (tx2, ty2), (0,255,0), 2)
                                # annotate person track
                                cv2.putText(annotated_frame, f"ID{tid}:{t.get('label','')}", (bx1, by1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)
                                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0,255,0), 2)
                            else:
                                # presence but low motion; only mark as active if presence sustained
                                if t['tire_frames'][ridx] >= TIRES_FRAMES_TO_COUNT:
                                    t['tire_cumulative'][ridx] += 1.0 / fps
                                    tire_change_time += 1.0 / fps
                                    cv2.rectangle(annotated_frame, (tx1, ty1), (tx2, ty2), (0,255,0), 2)
                                    cv2.putText(annotated_frame, f"ID{tid}:{t.get('label','')}", (bx1, by1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)
                                    cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0,255,0), 2)
                                else:
                                    cv2.rectangle(annotated_frame, (tx1, ty1), (tx2, ty2), (0,255,255), 1)
                        else:
                            # no overlap for this ROI; reset that ROI's presence counter if desired
                            # (we keep cumulative time persistent)
                            pass

        # ---------- Probe/refuel detection (kept from prior logic) ----------
        if is_car_stopped:
            if refuel_bbox is None:
                x, y, w, h = refuel_roi_in_air
                sx1, sy1 = max(0, int(x)), max(0, int(y))
                sx2, sy2 = min(width, int(x + w)), min(height, int(y + h))
                if sx2 > sx1 and sy2 > sy1:
                    search_area_gray = cv2.cvtColor(frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2GRAY)
                    best_in_score, best_in_idx, best_in_t = best_match_template(search_area_gray, probe_in_templates)
                    best_out_score, best_out_idx, best_out_t = best_match_template(search_area_gray, probe_out_templates)
                    best_in_score = best_in_score if best_in_score is not None else -1.0
                    best_out_score = best_out_score if best_out_score is not None else -1.0
                    DECISION_THRESH = 0.7
                    if best_in_score > best_out_score and best_in_score >= DECISION_THRESH:
                        active_probe_template = best_in_t
                        # best_in_idx is max_loc
                        mx, my = best_in_idx
                        active_template_h, active_template_w = active_probe_template.shape[:2]
                        refuel_bbox = (sx1 + int(mx), sy1 + int(my), active_template_w, active_template_h)
                        is_refueling_state = True
            else:
                search_pad = 40
                x, y, w, h = refuel_bbox
                sx1, sy1 = max(0, int(x - search_pad)), max(0, int(y - search_pad))
                sx2, sy2 = min(width, int(x + w + search_pad)), min(height, int(y + h + search_pad))
                win_w = sx2 - sx1; win_h = sy2 - sy1
                if active_probe_template is None or win_w < active_template_w or win_h < active_template_h:
                    is_refueling_state = False
                    refuel_bbox = None
                    active_probe_template = None
                    active_template_w = None
                    active_template_h = None
                else:
                    search_window = frame[sy1:sy2, sx1:sx2]
                    search_window_gray = cv2.cvtColor(search_window, cv2.COLOR_BGR2GRAY)
                    res = cv2.matchTemplate(search_window_gray, active_probe_template, cv2.TM_CCOEFF_NORMED)
                    if res is None:
                        is_refueling_state = False
                        refuel_bbox = None
                        active_probe_template = None
                    else:
                        _, score, _, max_loc = cv2.minMaxLoc(res)
                        if score > TRACK_LOST_THRESHOLD:
                            is_refueling_state = True
                            refuel_bbox = (sx1 + int(max_loc[0]), sy1 + int(max_loc[1]), active_template_w, active_template_h)
                            p1 = (int(refuel_bbox[0]), int(refuel_bbox[1])); p2 = (int(refuel_bbox[0] + refuel_bbox[2]), int(refuel_bbox[1] + refuel_bbox[3]))
                            cv2.rectangle(annotated_frame, p1, p2, (255, 0, 255), 3, 1)
                        else:
                            is_refueling_state = False
                            refuel_bbox = None
            if is_refueling_state:
                refuel_time += 1.0 / fps

        # ---------- Drawing overlays ----------
        cv2.rectangle(annotated_frame, ref_roi, (0,255,255), 2)
        cv2.rectangle(annotated_frame, refuel_roi_in_air, (0,0,255), 2)

        # draw tire ROIs and per-ROI cumulative times derived from tracks
        overlay_x = 20
        overlay_y = height // 2 - 80
        for ridx, troi in enumerate(tire_rois):
            rx1, ry1, rx2, ry2 = troi
            # default draw
            cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (255,255,0), 1)
            # compute sum of per-track cumulative times for this ROI for overlay
            total_roi_time = 0.0
            last_label = None
            for t in tracks.values():
                total_roi_time += t['tire_cumulative'].get(ridx, 0.0)
                if t.get('label'):
                    last_label = t.get('label')
            text = f"R{ridx+1} {total_roi_time:.2f}s"
            if last_label:
                text = f"{last_label}: {total_roi_time:.2f}s"
            cv2.putText(annotated_frame, text, (overlay_x, overlay_y + ridx*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        # status overlay
        rect_x, rect_y, rect_w, rect_h = 20, height // 2 + 20, 520, 120
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0,255,255), -1)
        annotated_frame = cv2.addWeighted(overlay, 0.45, annotated_frame, 0.55, 0)
        current_display_stop_time = total_stopped_time + ((frame_idx - stop_start_frame)/fps if is_car_stopped else 0)
        cv2.putText(annotated_frame, f'Car Stopped: {current_display_stop_time:.2f}s', (rect_x + 10, rect_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
        cv2.putText(annotated_frame, f'Tire Change: {tire_change_time:.2f}s', (rect_x + 10, rect_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        cv2.putText(annotated_frame, f'Refueling: {refuel_time:.2f}s', (rect_x + 10, rect_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

        out.write(annotated_frame)

        # persist gray frame
        last_gray = curr_gray.copy()

    # finalize
    if is_car_stopped:
        total_stopped_time += (total_frames - stop_start_frame) / fps

    cap.release()
    out.release()
    return total_stopped_time, tire_change_time, refuel_time
