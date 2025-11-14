import cv2
import numpy as np
import os
from ultralytics import YOLO

def get_patch_signature(frame, roi):
    """Calculates a simple signature (mean color) of a region."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, int(roi[0])), max(0, int(roi[1])), min(w, int(roi[2])), min(h, int(roi[3]))
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return np.array([0, 0, 0])
    return np.mean(patch, axis=(0, 1))

def boxes_overlap_area(box1, box2):
    """Calculates the area of overlap between two boxes."""
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)

# ---------------------- Crew template helpers ----------------------
def load_crew_templates(dir_path):
    """
    Load all images from refs/crew and return list of (label, template_gray, hist).
    Label is filename without extension.
    """
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
    """
    Compute a normalized HSV 2D histogram (H,S) for matching.
    """
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def match_crew_by_hist(bgr_patch, crew_templates):
    """
    Compare a person crop patch (BGR) with all crew_templates.
    Returns (best_label, best_score) or (None, 0.0) if none matched.
    """
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

# ---------------------- Tire-area helpers ----------------------
def load_tire_rois_from_image(ref_path, frame_width, frame_height):
    """
    Load refs/tirechangeareas.png and detect red-marked boxes.
    Returns list of ROI tuples (x1,y1,x2,y2) in frame coordinates.
    """
    if not os.path.exists(ref_path):
        return []

    img = cv2.imread(ref_path)
    if img is None:
        return []

    ref_h, ref_w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 100, 100])
    upper2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
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
        fx1 = int(x * scale_x)
        fy1 = int(y * scale_y)
        fx2 = int((x + w) * scale_x)
        fy2 = int((y + h) * scale_y)
        fx1, fy1 = max(0, fx1), max(0, fy1)
        fx2, fy2 = min(frame_width, fx2), min(frame_height, fy2)
        if fx2 > fx1 and fy2 > fy1:
            rois.append((fx1, fy1, fx2, fy2))
    rois = sorted(rois, key=lambda r: (r[1], r[0]))
    return rois

# ---------------------- Main processing ----------------------
def process_video(video_path, output_path, progress_callback):
    model = YOLO('yolov8n.pt')

    # crew templates (optional)
    CREW_DIR = 'refs/crew'
    crew_templates = load_crew_templates(CREW_DIR)
    MIN_CREW_MATCH = 0.45

    # probe templates (multi-dir fallback)
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

    probe_in_templates = load_templates_from_dir('refs/probein')
    probe_out_templates = load_templates_from_dir('refs/probeout')
    # fallback single files (backwards compat)
    if not probe_in_templates and os.path.exists('refs/probe_in.png'):
        t = cv2.imread('refs/probe_in.png', cv2.IMREAD_GRAYSCALE)
        if t is not None:
            probe_in_templates = [t]
    if not probe_out_templates and os.path.exists('refs/probe_out.png'):
        t = cv2.imread('refs/probe_out.png', cv2.IMREAD_GRAYSCALE)
        if t is not None:
            probe_out_templates = [t]
    if not probe_in_templates or not probe_out_templates:
        print("Error: probe-in/probe-out templates missing. Place in refs/probein & refs/probeout or use fallback files.")
        return [0.0] * 3

    # tire ROIs: try to load from user-provided image, fall back to previous hard-coded ROIs
    tire_rois_from_image = None

    # --- ROIs and other setup (unchanged defaults) ---
    ref_roi = (1042, 463, 1059, 487)
    tire_rois_fallback = [(1210, 30, 1370, 150), (1210, 400, 1400, 550), (685, 10, 830, 100), (685, 430, 780, 500)]
    refuel_roi_in_air = (803, 328, 920, 460)

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Attempt load tire ROIs from refs/tirechangeareas.png
    tire_rois_from_image = load_tire_rois_from_image('refs/tirechangeareas.png', width, height)
    tire_rois = tire_rois_from_image if tire_rois_from_image else tire_rois_fallback

    unobstructed_signature, last_car_stop_sig = None, None
    CAR_STOP_THRESH = 15
    STOP_CONFIRM_FRAMES = int(fps / 5)
    stopped_frames_count, is_car_stopped, stop_start_frame = 0, False, 0

    MIN_TIRE_OVERLAP_AREA = 500
    total_stopped_time, tire_change_time, refuel_time = 0.0, 0.0, 0.0

    # probe tracking state (same logic as before)
    refuel_bbox = None
    is_refueling_state = False
    TRACK_LOST_THRESHOLD = 0.60

    # ---------------- Improved tire-change detection state ----------------
    # per-ROI state dictionary
    tire_states = []
    for roi in tire_rois:
        tire_states.append({
            'roi': roi,                    # (x1,y1,x2,y2)
            'motion_counter': 0,           # consecutive frames with motion
            'person_counter': 0,           # consecutive frames with a person overlapping
            'active': False,               # currently considered active
            'cumulative_time': 0.0,        # seconds
            'last_active_frame': -1,
            'last_person_label': None      # crew label if matched
        })

    # thresholds (tune if needed)
    TIRE_ACTIVITY_MEAN_THRESH = 8.0            # mean absolute diff threshold for motion
    TIRE_ACTIVITY_CONFIRM_FRAMES = max(1, int(fps * 0.03))  # ~1 frame at 30fps
    TIRE_PERSON_CONFIRM_FRAMES = max(1, int(fps * 0.05))   # require person present for ~1-2 frames
    TIRE_MIN_COUNT_FRAMES_TO_START = 1         # require at least this many confirm frames (combined rule) to mark active
    PERSON_MATCH_MIN_SCORE = MIN_CREW_MATCH    # for crew label identification
    last_gray = None

    # helper to check person overlap for a given ROI
    def any_person_overlaps(people_xyxy, roi):
        for (x1,y1,x2,y2) in people_xyxy:
            if boxes_overlap_area((x1,y1,x2,y2), roi) > 0:
                return True
        return False

    # main frame loop
    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        progress_callback(frame_count / total_frames)

        # precompute gray frame for motion detection
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = model.track(frame, persist=True, classes=[0], verbose=False)
        annotated_frame = results[0].plot()

        # prepare person detections list in xyxy ints
        person_bboxes_xyxy = []
        if results[0].boxes:
            for b in results[0].boxes:
                if int(b.cls) != 0:
                    continue
                xyxy = b.xyxy[0].cpu().numpy()
                x1,y1,x2,y2 = [int(v) for v in xyxy]
                x1,y1 = max(0,x1), max(0,y1)
                x2,y2 = min(width-1,x2), min(height-1,y2)
                if x2<=x1 or y2<=y1:
                    continue
                person_bboxes_xyxy.append((x1,y1,x2,y2))

        # car stopped logic unchanged
        if frame_count == 0:
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
            is_car_stopped, stop_start_frame = True, frame_count
            # reset tire activity counters on stop start
            for t in tire_states:
                t['motion_counter'] = 0
                t['person_counter'] = 0
                t['active'] = False
                t['last_active_frame'] = -1
                t['last_person_label'] = None
        elif car_is_moving and is_car_stopped:
            # finalize stop normally
            is_car_stopped, is_refueling_state, refuel_bbox = False, False, None
            total_stopped_time += (frame_count - stop_start_frame) / fps
            stop_start_frame = 0

        # If stopped, evaluate each tire ROI for combined activity signal
        if is_car_stopped:
            # compute person-box overlap for quick checks and build person crops for crew matching
            person_crops = {}  # bbox -> crop
            for (x1,y1,x2,y2) in person_bboxes_xyxy:
                person_crops[(x1,y1,x2,y2)] = frame[y1:y2, x1:x2].copy()

            # For motion we need last_gray
            if last_gray is not None:
                for idx, tstate in enumerate(tire_states):
                    x1,y1,x2,y2 = tstate['roi']
                    # validate roi
                    if x2<=x1 or y2<=y1 or x1>=width or y1>=height:
                        tstate['motion_counter'] = 0
                        tstate['person_counter'] = 0
                        tstate['active'] = False
                        continue
                    prev_patch = last_gray[y1:y2, x1:x2]
                    curr_patch = curr_gray[y1:y2, x1:x2]
                    if prev_patch.size == 0 or curr_patch.size == 0 or prev_patch.shape != curr_patch.shape:
                        tstate['motion_counter'] = 0
                    else:
                        mean_diff = float(np.mean(cv2.absdiff(prev_patch, curr_patch)))
                        if mean_diff > TIRE_ACTIVITY_MEAN_THRESH:
                            tstate['motion_counter'] += 1
                        else:
                            tstate['motion_counter'] = 0

                    # person overlap
                    person_overlap = any_person_overlaps(person_bboxes_xyxy, (x1,y1,x2,y2))
                    if person_overlap:
                        tstate['person_counter'] += 1
                    else:
                        tstate['person_counter'] = 0

                    # attempt crew label if a person overlaps and crew templates exist
                    assigned_label = None
                    if person_overlap and crew_templates:
                        # find the best overlapping detection and match by histogram
                        best_label = None
                        best_score = -1.0
                        for bbox, crop in person_crops.items():
                            bx1,by1,bx2,by2 = bbox
                            # only consider if overlaps
                            if boxes_overlap_area((bx1,by1,bx2,by2), (x1,y1,x2,y2)) == 0:
                                continue
                            label, score = match_crew_by_hist(crop, crew_templates)
                            if score > best_score:
                                best_score = score
                                best_label = label
                        if best_label and best_score >= PERSON_MATCH_MIN_SCORE:
                            assigned_label = best_label

                    tstate['last_person_label'] = assigned_label

                    # Decide active: we require confirmation of either:
                    # - motion_counter >= TIRE_ACTIVITY_CONFIRM_FRAMES AND (person_counter >=1 OR person_counter >= TIRE_PERSON_CONFIRM_FRAMES)
                    # - OR person_counter >= TIRE_PERSON_CONFIRM_FRAMES (someone present for sustained frames)
                    motion_ok = tstate['motion_counter'] >= TIRE_ACTIVITY_CONFIRM_FRAMES
                    person_ok = tstate['person_counter'] >= TIRE_PERSON_CONFIRM_FRAMES
                    # combined rule
                    new_active = False
                    if (motion_ok and (tstate['person_counter'] >= 1 or person_ok)) or person_ok:
                        new_active = True

                    # start counting only when new_active and was not active before OR continue if active
                    if new_active:
                        # mark active and increment timer for this frame
                        tstate['active'] = True
                        tstate['cumulative_time'] += 1.0 / fps
                        tstate['last_active_frame'] = frame_count
                        # add to global tire_change_time
                        tire_change_time += 1.0 / fps
                        # draw active ROI
                        cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), (0, 255, 0), 2)
                    else:
                        # not active: draw faint ROI for visibility
                        cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), (0, 255, 255), 1)
            else:
                # no prev frame yet: draw ROIs but do not count
                for tstate in tire_states:
                    x1,y1,x2,y2 = tstate['roi']
                    cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), (0, 255, 255), 1)

            # legacy person-based tire change (additive) — still counted if many persons overlap ROIs
            if sum(boxes_overlap_area(p, t['roi']) for p in person_bboxes_xyxy for t in tire_states) > MIN_TIRE_OVERLAP_AREA:
                tire_change_time += 1.0 / fps

            # --- existing probe/refuel detection --- (kept as before)
            if refuel_bbox is None:
                x, y, w, h = refuel_roi_in_air
                sx1, sy1 = max(0, int(x)), max(0, int(y))
                sx2, sy2 = min(width, int(x + w)), min(height, int(y + h))
                if sx2 > sx1 and sy2 > sy1:
                    search_area = cv2.cvtColor(frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2GRAY)
                    # best match among probe_in templates
                    best_in_score, best_in_idx, best_in_t = -1.0, None, None
                    for t in probe_in_templates:
                        if search_area.shape[0] < t.shape[0] or search_area.shape[1] < t.shape[1]:
                            continue
                        res = cv2.matchTemplate(search_area, t, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        if max_val > best_in_score:
                            best_in_score = float(max_val); best_in_idx = max_loc; best_in_t = t
                    best_out_score = -1.0
                    for t in probe_out_templates:
                        if search_area.shape[0] < t.shape[0] or search_area.shape[1] < t.shape[1]:
                            continue
                        res = cv2.matchTemplate(search_area, t, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(res)
                        if max_val > best_out_score:
                            best_out_score = float(max_val)
                    DECISION_THRESH = 0.7
                    if best_in_score > best_out_score and best_in_score >= DECISION_THRESH:
                        active_t = best_in_t
                        max_loc = best_in_idx
                        refuel_bbox = (sx1 + int(max_loc[0]), sy1 + int(max_loc[1]), active_t.shape[1], active_t.shape[0])
                        is_refueling_state = True
            else:
                search_pad = 40
                x, y, w, h = refuel_bbox
                sx1, sy1 = max(0, int(x - search_pad)), max(0, int(y - search_pad))
                sx2, sy2 = min(width, int(x + w + search_pad)), min(height, int(y + h + search_pad))
                win_w = sx2 - sx1
                win_h = sy2 - sy1
                if win_w <= 0 or win_h <= 0:
                    is_refueling_state = False
                    refuel_bbox = None
                else:
                    search_window = frame[sy1:sy2, sx1:sx2]
                    search_window_gray = cv2.cvtColor(search_window, cv2.COLOR_BGR2GRAY)
                    # choose a probe template (first) as active if not stored
                    active_probe_template = probe_in_templates[0] if probe_in_templates else None
                    if active_probe_template is not None and search_window_gray.shape[0] >= active_probe_template.shape[0] and search_window_gray.shape[1] >= active_probe_template.shape[1]:
                        res = cv2.matchTemplate(search_window_gray, active_probe_template, cv2.TM_CCOEFF_NORMED)
                        _, score, _, max_loc = cv2.minMaxLoc(res)
                        if score > TRACK_LOST_THRESHOLD:
                            is_refueling_state = True
                            refuel_bbox = (sx1 + int(max_loc[0]), sy1 + int(max_loc[1]), active_probe_template.shape[1], active_probe_template.shape[0])
                            p1 = (int(refuel_bbox[0]), int(refuel_bbox[1]))
                            p2 = (int(refuel_bbox[0] + refuel_bbox[2]), int(refuel_bbox[1] + refuel_bbox[3]))
                            cv2.rectangle(annotated_frame, p1, p2, (255, 0, 255), 3, 1)
                        else:
                            is_refueling_state = False
                            refuel_bbox = None
                    else:
                        is_refueling_state = False
                        refuel_bbox = None

            if is_refueling_state:
                refuel_time += 1.0 / fps

        # Drawing: overlay core ROIs and per-ROI labels/times
        cv2.rectangle(annotated_frame, ref_roi, (0, 255, 255), 2)
        cv2.rectangle(annotated_frame, refuel_roi_in_air, (0, 0, 255), 2)

        # draw tire ROI boxes + per-ROI time and last person label
        overlay_x = 20
        overlay_y = height // 2 - 80
        for idx, tstate in enumerate(tire_states):
            rx1, ry1, rx2, ry2 = tstate['roi']
            # draw rectangle already drawn during active logic; ensure label overlay
            lbl = f"R{idx+1}"
            if tstate['last_person_label']:
                lbl = f"{tstate['last_person_label']}"
            time_s = tstate['cumulative_time']
            text = f"{lbl}: {time_s:.2f}s"
            cv2.putText(annotated_frame, text, (overlay_x, overlay_y + idx*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        # draw status display (car/tire/refuel)
        rect_x, rect_y, rect_w, rect_h = 20, height // 2 + 20, 520, 120
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 255), -1)
        annotated_frame = cv2.addWeighted(overlay, 0.45, annotated_frame, 0.55, 0)
        current_display_stop_time = total_stopped_time + ((frame_count - stop_start_frame) / fps if is_car_stopped else 0)
        cv2.putText(annotated_frame, f'Car Stopped: {current_display_stop_time:.2f}s', (rect_x + 10, rect_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(annotated_frame, f'Tire Change: {tire_change_time:.2f}s', (rect_x + 10, rect_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(annotated_frame, f'Refueling: {refuel_time:.2f}s', (rect_x + 10, rect_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        out.write(annotated_frame)

        # save current gray for next frame diff
        last_gray = curr_gray.copy()

    # finalize
    if is_car_stopped:
        total_stopped_time += (total_frames - stop_start_frame) / fps

    cap.release()
    out.release()

    # return global totals (per-ROI cumulative_time still available in local state if you want)
    return total_stopped_time, tire_change_time, refuel_time
