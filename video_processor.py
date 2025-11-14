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

def load_templates_from_dir(dir_path):
    """Load all grayscale images from a directory into a list of numpy arrays."""
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
    """
    Given a grayscale search area and a list of grayscale templates,
    return (best_score, best_idx, best_template) or (None, None, None) if no templates.
    """
    if not templates:
        return None, None, None
    best_score = -1.0
    best_idx = None
    best_t = None
    for i, t in enumerate(templates):
        th, tw = t.shape[:2]
        # If template is larger than search area, skip
        if search_area_gray.shape[0] < th or search_area_gray.shape[1] < tw:
            continue
        res = cv2.matchTemplate(search_area_gray, t, cv2.TM_CCOEFF_NORMED)
        if res is None:
            continue
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = float(max_val)
            best_idx = i
            best_t = t
    if best_idx is None:
        return None, None, None
    return best_score, best_idx, best_t

def process_video(video_path, output_path, progress_callback):
    model = YOLO('yolov8n.pt')

    # --- Load Templates for Initial Detection ---
    # Prefer directories refs/probein and refs/probeout; fallback to single files for compatibility.
    probe_in_dir = 'refs/probein'
    probe_out_dir = 'refs/probeout'
    probe_in_fallback = 'refs/probe_in.png'
    probe_out_fallback = 'refs/probe_out.png'

    probe_in_templates = load_templates_from_dir(probe_in_dir)
    probe_out_templates = load_templates_from_dir(probe_out_dir)

    # fallback single files if directories are empty
    if not probe_in_templates and os.path.exists(probe_in_fallback):
        t = cv2.imread(probe_in_fallback, cv2.IMREAD_GRAYSCALE)
        if t is not None:
            probe_in_templates = [t]
    if not probe_out_templates and os.path.exists(probe_out_fallback):
        t = cv2.imread(probe_out_fallback, cv2.IMREAD_GRAYSCALE)
        if t is not None:
            probe_out_templates = [t]

    if not probe_in_templates or not probe_out_templates:
        print("Error: probe-in or probe-out templates not found. Place images in refs/probein and refs/probeout or use refs/probe_in.png and refs/probe_out.png.")
        return [0.0] * 3

    # --- ROIs and other setup ---
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

    # --- Custom Tracker State ---
    refuel_bbox = None  # Will store (x, y, w, h) of the probe
    is_refueling_state = False
    TRACK_LOST_THRESHOLD = 0.60  # If match score drops below this, we lose the lock

    # Active probe template (selected from probe_in_templates) used while tracking
    active_probe_template = None
    active_template_w = None
    active_template_h = None

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
        # (Car stop logic remains the same)
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
        elif car_is_moving and is_car_stopped:
            is_car_stopped, is_refueling_state, refuel_bbox = False, False, None
            active_probe_template = None
            active_template_w = None
            active_template_h = None
            total_stopped_time += (frame_count - stop_start_frame) / fps
            stop_start_frame = 0

        if is_car_stopped:
            # person boxes => tire change timer (same as before)
            person_bboxes = [b.xyxy[0].cpu().numpy() for b in results[0].boxes if int(b.cls) == 0] if results[0].boxes else []
            if sum(boxes_overlap_area(p, t) for p in person_bboxes for t in tire_rois) > MIN_TIRE_OVERLAP_AREA:
                tire_change_time += 1 / fps

            # --- Probe detection and tracking ---
            if refuel_bbox is None:
                # initial detection: search in the refuel roi using all templates and pick the best
                x, y, w, h = refuel_roi_in_air
                sx1, sy1 = max(0, int(x)), max(0, int(y))
                sx2, sy2 = min(width, int(x + w)), min(height, int(y + h))
                if sx2 > sx1 and sy2 > sy1:
                    search_area_gray = cv2.cvtColor(frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2GRAY)

                    # best match among probe_in templates
                    best_in_score, best_in_idx, best_in_t = best_match_template(search_area_gray, probe_in_templates)
                    # best match among probe_out templates
                    best_out_score, best_out_idx, best_out_t = best_match_template(search_area_gray, probe_out_templates)

                    # ensure numeric values (None -> very low)
                    best_in_score = best_in_score if best_in_score is not None else -1.0
                    best_out_score = best_out_score if best_out_score is not None else -1.0

                    # decision rule: selected probe-in must beat probe-out and exceed threshold
                    DECISION_THRESH = 0.7
                    if best_in_score > best_out_score and best_in_score >= DECISION_THRESH:
                        # pick the template that matched best
                        active_probe_template = best_in_t
                        active_template_h, active_template_w = active_probe_template.shape[:2]
                        # determine location (we need top-left location from matchTemplate)
                        # recompute res to get max_loc
                        res = cv2.matchTemplate(search_area_gray, active_probe_template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        # lock bbox in full-frame coordinates
                        refuel_bbox = (sx1 + int(max_loc[0]), sy1 + int(max_loc[1]), active_template_w, active_template_h)
                        is_refueling_state = True
            else:
                # we have a lock; track by searching in a local window around last bbox using the active template
                search_pad = 40
                x, y, w, h = refuel_bbox
                sx1, sy1 = max(0, int(x - search_pad)), max(0, int(y - search_pad))
                sx2, sy2 = min(width, int(x + w + search_pad)), min(height, int(y + h + search_pad))

                win_w = sx2 - sx1
                win_h = sy2 - sy1
                if active_probe_template is None or win_w < active_template_w or win_h < active_template_h:
                    # can't match -> lose lock
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
                        active_template_w = None
                        active_template_h = None
                    else:
                        _, score, _, max_loc = cv2.minMaxLoc(res)
                        if score > TRACK_LOST_THRESHOLD:
                            is_refueling_state = True
                            # update bbox to new location
                            refuel_bbox = (sx1 + int(max_loc[0]), sy1 + int(max_loc[1]), active_template_w, active_template_h)
                            p1 = (int(refuel_bbox[0]), int(refuel_bbox[1]))
                            p2 = (int(refuel_bbox[0] + refuel_bbox[2]), int(refuel_bbox[1] + refuel_bbox[3]))
                            cv2.rectangle(annotated_frame, p1, p2, (255, 0, 255), 3, 1)  # Draw magenta box
                        else:
                            # lost lock
                            is_refueling_state = False
                            refuel_bbox = None
                            active_probe_template = None
                            active_template_w = None
                            active_template_h = None

            if is_refueling_state:
                refuel_time += 1 / fps

        # --- Drawing Logic ---
        cv2.rectangle(annotated_frame, ref_roi, (0, 255, 255), 2)
        for roi in tire_rois:
            cv2.rectangle(annotated_frame, roi, (255, 255, 0), 2)
        cv2.rectangle(annotated_frame, refuel_roi_in_air, (0, 0, 255), 2)

        rect_x, rect_y, rect_w, rect_h = 20, height // 2 - 60, 450, 130
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 255), -1)
        annotated_frame = cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0)

        current_display_stop_time = total_stopped_time + ((frame_count - stop_start_frame) / fps if is_car_stopped else 0)
        cv2.putText(annotated_frame, f'Car Stopped: {current_display_stop_time:.2f}s', (rect_x + 10, rect_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(annotated_frame, f'Tire Change: {tire_change_time:.2f}s', (rect_x + 10, rect_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(annotated_frame, f'Refueling: {refuel_time:.2f}s', (rect_x + 10, rect_y + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        out.write(annotated_frame)

    if is_car_stopped:
        total_stopped_time += (total_frames - stop_start_frame) / fps

    cap.release()
    out.release()

    return total_stopped_time, tire_change_time, refuel_time
