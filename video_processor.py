import cv2
import numpy as np
import os
from ultralytics import YOLO
import math
from collections import deque

# Integration: CrewTracker (must be present as crew_tracker.py in repo)
from crew_tracker import CrewTracker

# Integration: optional annotation helper (for seeding labels like "fueler")
from annotation_helper import load_annotations, apply_annotations_to_tracker

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

    # Integration: instantiate CrewTracker
    CREW_WALL_ROI = (0, 200, 360, 720)  # tune for your camera if needed
    tracker = CrewTracker(device='cpu', crew_dir='refs/crew', crew_wall_roi=CREW_WALL_ROI, embed_device='cpu')

    # Apply tuned tracker settings based on observed frames
    # (these override defaults in crew_tracker to better handle sprint pit movements)
    tracker.iou_thresh = 0.25
    # MAX_MISSING will be set after fps is known below

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

    # --- ROIs and other setup ---
    ref_roi = (1042, 463, 1059, 487)
    tire_rois_fallback = [(1210, 30, 1370, 150), (1210, 400, 1400, 550), (685, 10, 830, 100), (685, 430, 780, 500)]
    refuel_roi_in_air = (803, 328, 920, 460)
    tire_areas_path = 'refs/tirechangeareas.png'

    # Load optional annotations (manual labels) to seed tracker (refs/annotations.csv)
    annotations = load_annotations('refs/annotations.csv') if os.path.exists('refs/annotations.csv') else {}

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # now set tracker parameters dependent on fps
    tracker.max_missing = max(4, int(fps * 0.5))
    tracker.history_len = int(fps * 2.0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Attempt load tire ROIs from the tirechangeareas image
    tire_rois = load_tire_rois_from_image(tire_areas_path, width, height)
    if not tire_rois:
        tire_rois = tire_rois_fallback

    # Car stop detection params
    unobstructed_signature, last_car_stop_sig = None, None
    CAR_STOP_THRESH = 15
    STOP_CONFIRM_FRAMES = int(fps / 5)
    stopped_frames_count, is_car_stopped, stop_start_frame = 0, False, 0

    # Tire latch/release settings
    TIRE_LATCH_SECONDS = 1.2
    TIRE_RELEASE_FRAMES = max(1, int(fps * TIRE_LATCH_SECONDS))
    MOTION_THRESH = 4.0
    TIRES_SECONDS_TO_COUNT = 0.5
    TIRES_FRAMES_TO_COUNT = max(1, int(fps * TIRES_SECONDS_TO_COUNT))

    # Fueler-specific settings
    FUELER_RELEASE_SECONDS = 1.5
    FUELER_RELEASE_FRAMES = max(1, int(fps * FUELER_RELEASE_SECONDS))

    # initialize per-ROI states (latched, release counter, cumulative time, initiator role)
    tire_roi_states = []
    for roi in tire_rois:
        tire_roi_states.append({
            'latched': False,
            'release_counter': 0,
            'last_active_frame': -1,
            'active_track_id': None,
            'cumulative_time': 0.0,
            'initiating_role': None
        })

    # fueler state
    fueler_state = {
        'in_fuel': False,
        'start_frame': None,
        'release_counter': 0,
        'initiating_track': None,
        'completed_fills': []  # list of (track_id, duration_s)
    }

    # global timers
    total_stopped_time, tire_change_time, refuel_time = 0.0, 0.0, 0.0

    # probe/refuel state
    refuel_bbox = None
    is_refueling_state = False
    TRACK_LOST_THRESHOLD = 0.60
    active_probe_template = None
    active_template_w = None
    active_template_h = None

    last_gray = None

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        progress_callback(frame_idx / total_frames)

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # YOLO person detection
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        annotated_frame = results[0].plot()

        # build person detection boxes and crops
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
                person_crops.append(frame[y1:y2, x1:x2].copy())

        # Car stop detection (same approach)
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
            # reset per-ROI release counters and fueler release counter
            for s in tire_roi_states:
                s['release_counter'] = 0
            fueler_state['release_counter'] = 0
        elif car_is_moving and is_car_stopped:
            is_car_stopped, is_refueling_state, refuel_bbox = False, False, None
            total_stopped_time += (frame_idx - stop_start_frame) / fps
            stop_start_frame = 0

        # Update crew tracker with detections (integration point)
        tracker.update(person_dets, person_crops, frame, frame_idx, stop_start_frame if is_car_stopped else None)

        # Apply manual annotations for this frame (if any). This will force-track labeling and seed templates.
        ann_list = annotations.get(frame_idx, [])
        if ann_list:
            try:
                apply_annotations_to_tracker(tracker, ann_list, frame, frame_idx, iou_thresh=0.25, add_template=True)
            except Exception:
                # don't crash on annotation failures; continue processing
                pass

        counted_refuel_this_frame = False  # ensure we only add refuel_time once per frame

        # ---------- If car stopped, evaluate tire activity using tracker.tracks ----------
        if is_car_stopped:
            # legacy person-ROI fallback counting (keeps previous behavior)
            if sum(boxes_overlap_area(p, troi) for p in person_dets for troi in tire_rois) > 500:
                tire_change_time += 1.0 / fps

            # Fueler-based fueling detection (prefer this over probe-template when crew tracker identifies a fueler)
            fueler_tid = None
            for tid, t in tracker.tracks.items():
                if t.get('role') == 'fueler' or (t.get('label') and 'fuel' in t.get('label', '').lower()):
                    fueler_tid = tid
                    break

            # If a fueler track exists, check overlap with refuel ROI
            if fueler_tid is not None:
                t = tracker.tracks[fueler_tid]
                if 'bbox' in t:
                    bx1, by1, bx2, by2 = t['bbox']
                    overlap_area = boxes_overlap_area((bx1,by1,bx2,by2), refuel_roi_in_air)
                    if overlap_area > 0:
                        # fueler is at the refuel ROI -> start or continue fueling
                        if not fueler_state['in_fuel']:
                            fueler_state['in_fuel'] = True
                            fueler_state['start_frame'] = frame_idx
                            fueler_state['initiating_track'] = fueler_tid
                            fueler_state['release_counter'] = 0
                        else:
                            fueler_state['release_counter'] = 0
                        # count fueling time this frame
                        refuel_time += 1.0 / fps
                        counted_refuel_this_frame = True
                        # annotate
                        display_label = t.get('role') or t.get('label') or f"ID{fueler_tid}"
                        cv2.putText(annotated_frame, f"FUELER {display_label}", (bx1, by1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                        cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0,0,255), 2)
                    else:
                        # not overlapping: if currently fueling, start release countdown
                        if fueler_state['in_fuel']:
                            fueler_state['release_counter'] += 1
                            if fueler_state['release_counter'] >= FUELER_RELEASE_FRAMES:
                                # finalize fueling
                                fueler_state['in_fuel'] = False
                                # compute duration
                                if fueler_state['start_frame'] is not None:
                                    duration = (frame_idx - fueler_state['start_frame']) / fps
                                    fueler_state['completed_fills'].append((fueler_state['initiating_track'], duration))
                                fueler_state['start_frame'] = None
                                fueler_state['initiating_track'] = None
                                fueler_state['release_counter'] = 0
                            else:
                                # still in release window, keep counting as fueling (optional)
                                refuel_time += 1.0 / fps
                                counted_refuel_this_frame = True
                                # visual feedback
                                cv2.rectangle(annotated_frame, (int(refuel_roi_in_air[0]), int(refuel_roi_in_air[1])),
                                              (int(refuel_roi_in_air[2]), int(refuel_roi_in_air[3])), (0,150,200), 2)

            # If no fueler-based detection counted fueling this frame, fall back to probe-template detection
            if not counted_refuel_this_frame:
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
                            mx, my = best_in_idx
                            active_template_h, active_template_w = active_probe_template.shape[:2]
                            refuel_bbox = (sx1 + int(mx), sy1 + int(my), active_template_w, active_template_h)
                            is_refueling_state = True
                            # count probe-based fuel time
                            refuel_time += 1.0 / fps
                            counted_refuel_this_frame = True
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
                                # count probe-based refuel time
                                refuel_time += 1.0 / fps
                                counted_refuel_this_frame = True
                            else:
                                is_refueling_state = False
                                refuel_bbox = None

            # ---------- Tire activity attribution using tracker.tracks ----------
            if last_gray is not None:
                for tid, t in tracker.tracks.items():
                    if 'bbox' not in t:
                        continue
                    bx1, by1, bx2, by2 = t['bbox']
                    bx1, by1, bx2, by2 = max(0, bx1), max(0, by1), min(width-1, bx2), min(height-1, by2)
                    if bx2 <= bx1 or by2 <= by1:
                        continue
                    prev_patch = last_gray[by1:by2, bx1:bx2]
                    curr_patch = curr_gray[by1:by2, bx1:bx2]
                    motion_amt = 0.0
                    if prev_patch.size and curr_patch.size and prev_patch.shape == curr_patch.shape:
                        motion_amt = float(np.mean(cv2.absdiff(prev_patch, curr_patch)))

                    # check each tire ROI for overlap with this track
                    for ridx, troi in enumerate(tire_rois):
                        tx1, ty1, tx2, ty2 = troi
                        overlap_area = boxes_overlap_area((bx1,by1,bx2,by2), troi)
                        active_signal = False
                        if overlap_area > 0:
                            if motion_amt > MOTION_THRESH:
                                active_signal = True
                            if t.get('tire_frames', {}).get(ridx, 0) >= TIRES_FRAMES_TO_COUNT:
                                active_signal = True

                        roi_state = tire_roi_states[ridx]

                        if active_signal:
                            roi_state['latched'] = True
                            roi_state['release_counter'] = 0
                            roi_state['last_active_frame'] = frame_idx
                            roi_state['active_track_id'] = tid
                            if roi_state.get('initiating_role') is None and t.get('role'):
                                roi_state['initiating_role'] = t.get('role')
                            roi_state['cumulative_time'] += 1.0 / fps
                            tire_change_time += 1.0 / fps
                            if 'tire_cumulative' in t:
                                t['tire_cumulative'][ridx] = t.get('tire_cumulative', {}).get(ridx, 0.0) + 1.0 / fps
                            display_label = t.get('role') or t.get('label') or f"ID{tid}"
                            cv2.rectangle(annotated_frame, (tx1, ty1), (tx2, ty2), (0,255,0), 2)
                            cv2.putText(annotated_frame, f"{display_label}", (bx1, by1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)
                            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0,255,0), 2)
                        else:
                            if roi_state['latched']:
                                roi_state['release_counter'] += 1
                                if roi_state['release_counter'] >= TIRE_RELEASE_FRAMES:
                                    roi_state['latched'] = False
                                    roi_state['release_counter'] = 0
                                    roi_state['active_track_id'] = None
                                    cv2.rectangle(annotated_frame, (tx1, ty1), (tx2, ty2), (0,255,255), 1)
                                else:
                                    cv2.rectangle(annotated_frame, (tx1, ty1), (tx2, ty2), (0,200,50), 2)
                            else:
                                cv2.rectangle(annotated_frame, (tx1, ty1), (tx2, ty2), (0,255,255), 1)

        # ---------- Drawing overlays ----------
        cv2.rectangle(annotated_frame, ref_roi, (0,255,255), 2)
        cv2.rectangle(annotated_frame, refuel_roi_in_air, (0,0,255), 2)

        # Draw tracks with role & hop info (from tracker)
        for tid, t in tracker.tracks.items():
            if 'bbox' not in t:
                continue
            bx1,by1,bx2,by2 = t['bbox']
            label = t.get('role') or t.get('label') or f"ID{tid}"
            info = f"{label}"
            if t.get('hopped'):
                hop_time = t.get('hop_time')
                try:
                    hop_display = f" hop:{hop_time:.2f}s" if hop_time is not None else ""
                except Exception:
                    hop_display = f" hop:{hop_time}"
                info += hop_display
            cv2.putText(annotated_frame, info, (bx1, by1-18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            cv2.rectangle(annotated_frame, (bx1,by1), (bx2,by2), (200,200,200), 1)

        # Per-ROI overlays & cumulative times
        overlay_x = 20
        overlay_y = height // 2 - 80
        for ridx, troi in enumerate(tire_rois):
            rx1, ry1, rx2, ry2 = troi
            cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (255,255,0), 1)
            text_time = tire_roi_states[ridx]['cumulative_time']
            initiator = tire_roi_states[ridx].get('initiating_role')
            text = f"R{ridx+1} {text_time:.2f}s"
            if initiator:
                text = f"{initiator}: {text_time:.2f}s"
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

        last_gray = curr_gray.copy()

    # finalize
    if is_car_stopped:
        total_stopped_time += (total_frames - stop_start_frame) / fps

    cap.release()
    out.release()

    return total_stopped_time, tire_change_time, refuel_time
