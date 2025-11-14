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
        # store grayscale template for possible visualization and histogram for matching
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = calc_hsv_hist(img)
        templates.append({'label': label, 'gray': gray, 'hist': hist})
    return templates

def calc_hsv_hist(bgr_patch):
    """
    Compute a normalized HSV 2D histogram (H,S) for matching.
    Returns a flattened normalized histogram.
    """
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    # parameters tuned for coarse but robust matching
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def match_crew_by_hist(bgr_patch, crew_templates):
    """
    Compare a person crop patch (BGR) with all crew_templates (list of dict).
    Returns (best_label, best_score) or (None, 0.0) if none matched.
    Uses correlation of HSV histograms.
    """
    if bgr_patch is None or bgr_patch.size == 0:
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

# ---------------------- Main processing ----------------------
def process_video(video_path, output_path, progress_callback):
    model = YOLO('yolov8n.pt')

    # --- Crew templates directory (user-provided) ---
    CREW_DIR = 'refs/crew'  # expect files like refs/crew/fueler.png, refs/crew/tire_left.png, etc.
    crew_templates = load_crew_templates(CREW_DIR)

    # -------------------- Config (tune these) --------------------
    # crew-match threshold: require histogram correlation >= this to accept a label
    MIN_MATCH_SCORE = 0.45

    # define wall area where crew jump off (update coordinates to suit your camera)
    # Format: (x1, y1, x2, y2)
    # Default is a conservative guess; please adjust to match your camera view
    CREW_WALL_ROI = (0, 200, 360, 720)  # example left-side wall; change to your scene

    # --- Existing files/templates (probe templates etc.) ---
    probe_in_dir = 'refs/probein'
    probe_out_dir = 'refs/probeout'
    probe_in_fallback = 'refs/probe_in.png'
    probe_out_fallback = 'refs/probe_out.png'

    # load probe templates as before (existing logic or fallback)
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

    probe_in_templates = load_templates_from_dir(probe_in_dir)
    probe_out_templates = load_templates_from_dir(probe_out_dir)
    if not probe_in_templates and os.path.exists(probe_in_fallback):
        t = cv2.imread(probe_in_fallback, cv2.IMREAD_GRAYSCALE)
        if t is not None:
            probe_in_templates = [t]
    if not probe_out_templates and os.path.exists(probe_out_fallback):
        t = cv2.imread(probe_out_fallback, cv2.IMREAD_GRAYSCALE)
        if t is not None:
            probe_out_templates = [t]

    # make sure we have probe templates as before
    if not probe_in_templates or not probe_out_templates:
        print("Error: probe-in or probe-out templates not found. Place images in refs/probein and refs/probeout or use refs/probe_in.png and refs/probe_out.png.")
        return [0.0] * 3

    # --- ROIs and other setup (unchanged) ---
    ref_roi = (1042, 463, 1059, 487)
    tire_rois = [(1210, 30, 1370, 150), (1210, 400, 1400, 550), (685, 10, 830, 100), (685, 430, 780, 500)]
    refuel_roi_in_air = (803, 328, 920, 460)

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    unobstructed_signature, last_car_stop_sig = None, None
    CAR_STOP_THRESH = 15
    STOP_CONFIRM_FRAMES = int(fps / 5)
    stopped_frames_count, is_car_stopped, stop_start_frame = 0, False, 0

    MIN_TIRE_OVERLAP_AREA = 500
    total_stopped_time, tire_change_time, refuel_time = 0.0, 0.0, 0.0

    # probe tracking state (as before)
    refuel_bbox = None  # Will store (x, y, w, h) of the probe
    is_refueling_state = False
    TRACK_LOST_THRESHOLD = 0.60  # If match score drops below this, we lose the lock

    # ---------------- Crew tracking / hop timers ----------------
    # crew_state[label] = {'on_wall':bool, 'seen':bool, 'hop_time':float_or_None, 'last_seen_frame': int}
    crew_state = {}
    for ct in crew_templates:
        crew_state[ct['label']] = {'on_wall': False, 'seen': False, 'hop_time': None, 'last_seen_frame': -1}

    # helper: test overlap between person bbox and wall ROI
    def bbox_overlaps_wall(bbox, wall_roi):
        # bbox: (x1,y1,x2,y2) or (x,y,w,h)
        if len(bbox) == 4 and bbox[2] - bbox[0] > 0 and bbox[3] - bbox[1] > 0:
            # xyxy
            bx1, by1, bx2, by2 = bbox
        else:
            bx1, by1, bw, bh = bbox
            bx2, by2 = bx1 + bw, by1 + bh
        wx1, wy1, wx2, wy2 = wall_roi
        inter = boxes_overlap_area((bx1, by1, bx2, by2), (wx1, wy1, wx2, wy2))
        return inter > 0

    # For display: sorted crew labels
    crew_labels_sorted = sorted(crew_state.keys())

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        progress_callback(frame_count / total_frames)

        results = model.track(frame, persist=True, classes=[0], verbose=False)
        annotated_frame = results[0].plot()

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

        # normal transitions
        if not car_is_moving and not is_car_stopped:
            is_car_stopped, stop_start_frame = True, frame_count
            # reset crew seen flags at stop start (we'll capture who is on the wall at the moment)
            for lbl in crew_state:
                crew_state[lbl]['seen'] = False
                crew_state[lbl]['on_wall'] = False
                crew_state[lbl]['last_seen_frame'] = -1
        elif car_is_moving and is_car_stopped:
            # finalize stop normally
            is_car_stopped, is_refueling_state, refuel_bbox = False, False, None
            total_stopped_time += (frame_count - stop_start_frame) / fps
            stop_start_frame = 0

        # When stopped: detect crew, assign labels, and collect hop-off times
        if is_car_stopped:
            # get all detected person boxes (xyxy) and their crops
            person_bboxes = []
            if results[0].boxes:
                for b in results[0].boxes:
                    if int(b.cls) != 0:
                        continue
                    xyxy = b.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2]
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    # clip
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width - 1, x2), min(height - 1, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    person_bboxes.append((x1, y1, x2, y2))

            # For each person detection, compute crop and match to crew templates
            # We'll pick best label per detection, and for labels we pick best detection (by match score)
            label_best_for_frame = {}  # label -> (score, bbox)
            for bbox in person_bboxes:
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                label, score = match_crew_by_hist(crop, crew_templates)
                if label is None:
                    continue
                if score < MIN_MATCH_SCORE:
                    continue
                # keep best detection for that label in this frame
                prev = label_best_for_frame.get(label)
                if prev is None or score > prev[0]:
                    label_best_for_frame[label] = (score, bbox)

            # Mark seen labels and wall overlap at this frame
            for lbl, state in crew_state.items():
                entry = label_best_for_frame.get(lbl)
                if entry:
                    score, bbox = entry
                    state['seen'] = True
                    state['last_seen_frame'] = frame_count
                    # check overlap with wall ROI
                    # convert bbox to (x,y,w,h)
                    bx1, by1, bx2, by2 = bbox
                    bw, bh = bx2 - bx1, by2 - by1
                    overlaps = bbox_overlaps_wall((bx1, by1, bw, bh), CREW_WALL_ROI)
                    # If this is the first time we see the label after the stop started, set on_wall accordingly
                    if state['last_seen_frame'] == frame_count and state['on_wall'] is False and state['hop_time'] is None:
                        # if we haven't set on_wall before in this stop period, treat initial seen overlapping as on_wall
                        # (this helps capture who was on the wall at stop start)
                        state['on_wall'] = overlaps
                    else:
                        # update on_wall for subsequent frames
                        state['on_wall'] = overlaps
                else:
                    # if not seen in this frame, do not modify last_seen_frame, but check if they were previously on_wall and now gone
                    pass

            # detect hop-off transitions: if a label was on_wall (and seen around stop start) and now no longer overlaps -> record hop_time
            for lbl, state in crew_state.items():
                # only record hop_time if we haven't recorded it yet and we saw them at stop start or during stop while on wall
                if state['hop_time'] is None:
                    # conditions to be considered eligible for hop detection:
                    # - previously seen and on_wall at some point after stop started (we use last_seen_frame to approximate)
                    # We'll consider them "eligible" if last_seen_frame >= stop_start_frame
                    if state['last_seen_frame'] >= stop_start_frame and state['on_wall'] == False:
                        # If they were last seen on the wall (we stored on_wall True earlier) and now the last detection indicates off-wall,
                        # or if they were seen on wall at the beginning and now the label is not detected at all (we consider that hop-off too).
                        # To detect change robustly, we check if they were seen earlier (stop_start_frame) and are now not overlapping.
                        # Record hop time when we first detect on_wall->off_wall
                        # To approximate "was on wall at start", check if the first seen frame after stop_start_frame had them overlapping.
                        # We'll approximate by assuming if last_seen_frame >= stop_start_frame and current on_wall == False then they hopped.
                        # Record the time:
                        state['hop_time'] = (frame_count - stop_start_frame) / fps
                # Note: if desired we could refine by remembering the 'on_wall' value at the very first detection after stop_start_frame.

            # --- existing tire-change logic remains unchanged (uses person_bboxes) ---
            if sum(boxes_overlap_area(p, t) for p in person_bboxes for t in tire_rois) > MIN_TIRE_OVERLAP_AREA:
                tire_change_time += 1/fps

            # --- Probe detection and tracking (unchanged from previous logic) ---
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
                # active template size might vary; for safety we accept matching if window fits at least minimally
                if win_w <= 0 or win_h <= 0:
                    is_refueling_state = False
                    refuel_bbox = None
                else:
                    search_window = frame[sy1:sy2, sx1:sx2]
                    search_window_gray = cv2.cvtColor(search_window, cv2.COLOR_BGR2GRAY)
                    # use previously chosen active_probe_template if available; else fall back to first probe_in_templates
                    active_probe_template = None
                    if 'active_probe_template' in locals() and locals().get('active_probe_template') is not None:
                        active_probe_template = locals().get('active_probe_template')
                    elif probe_in_templates:
                        active_probe_template = probe_in_templates[0]
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
                refuel_time += 1 / fps

        # ---------------- Drawing & overlay crew hop times ----------------
        # draw ref ROI and tire ROIs as before
        cv2.rectangle(annotated_frame, ref_roi, (0, 255, 255), 2)
        for roi in tire_rois:
            cv2.rectangle(annotated_frame, roi, (255, 255, 0), 2)
        cv2.rectangle(annotated_frame, refuel_roi_in_air, (0, 0, 255), 2)

        # overlay: car timers and crew hop times
        rect_x, rect_y, rect_w, rect_h = 20, height // 2 - 80, 520, 200
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

        # crew hop times: list labels and their state/time
        y_offset = 120
        for lbl in crew_labels_sorted:
            st = crew_state.get(lbl, {})
            display = "—"
            if st.get('hop_time') is not None:
                display = f"{st['hop_time']:.2f}s"
            elif st.get('on_wall'):
                display = "on wall"
            elif st.get('seen'):
                display = "seen"
            cv2.putText(annotated_frame, f"{lbl}: {display}", (rect_x + 10, rect_y + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y_offset += 22

        out.write(annotated_frame)

    # finalize
    if is_car_stopped:
        total_stopped_time += (total_frames - stop_start_frame) / fps

    cap.release()
    out.release()

    return total_stopped_time, tire_change_time, refuel_time
