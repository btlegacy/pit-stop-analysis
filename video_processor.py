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

def load_tire_rois_from_image(ref_path, frame_width, frame_height):
    """
    Load refs/tirechangeareas.png and detect red-marked boxes.
    Returns list of ROI tuples (x1,y1,x2,y2) in frame coordinates.
    If detection fails, returns an empty list.
    """
    if not os.path.exists(ref_path):
        return []

    img = cv2.imread(ref_path)
    if img is None:
        return []

    ref_h, ref_w = img.shape[:2]

    # convert to hsv and detect red (two hue ranges)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower red range
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([10, 255, 255])
    # upper red range
    lower2 = np.array([160, 100, 100])
    upper2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # morphology to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # ignore tiny boxes
        if w < 8 or h < 8:
            continue
        # convert to full-frame coordinates by scaling from ref image to frame dimensions
        scale_x = frame_width / ref_w
        scale_y = frame_height / ref_h
        fx1 = int(x * scale_x)
        fy1 = int(y * scale_y)
        fx2 = int((x + w) * scale_x)
        fy2 = int((y + h) * scale_y)
        # clamp
        fx1, fy1 = max(0, fx1), max(0, fy1)
        fx2, fy2 = min(frame_width, fx2), min(frame_height, fy2)
        if fx2 > fx1 and fy2 > fy1:
            rois.append((fx1, fy1, fx2, fy2))
    # sort rois left-to-right, top-to-bottom for consistency
    rois = sorted(rois, key=lambda r: (r[1], r[0]))
    return rois

def process_video(video_path, output_path, progress_callback):
    model = YOLO('yolov8n.pt')

    # --- Load Templates for Initial Detection ---
    probe_in_path = 'refs/probe_in.png'
    probe_out_path = 'refs/probe_out.png'
    # probe multi-template dirs fallback handled externally; here we keep the original single-file support
    if not os.path.exists(probe_in_path) or not os.path.exists(probe_out_path):
        print("Error: Please create 'refs/probe_in.png' and 'refs/probe_out.png' template files.")
        return [0.0] * 3
    
    probe_in_template = cv2.imread(probe_in_path, cv2.IMREAD_GRAYSCALE)
    probe_out_template = cv2.imread(probe_out_path, cv2.IMREAD_GRAYSCALE)
    if probe_in_template is None or probe_out_template is None:
        print("Error: Could not read 'probe_in.png' or 'probe_out.png'.")
        return [0.0] * 3

    # Template dims (grayscale)
    template_h, template_w = probe_in_template.shape[:2]

    # --- ROIs and other setup ---
    # default/static tire ROIs (fallback if no tirechangeareas.png or detection fails)
    tire_rois_fallback = [(1210, 30, 1370, 150), (1210, 400, 1400, 550), (685, 10, 830, 100), (685, 430, 780, 500)]
    ref_roi = (1042, 463, 1059, 487)
    refuel_roi_in_air = (803, 328, 920, 460)
    
    # Tire-change image path (user-provided)
    tire_areas_path = 'refs/tirechangeareas.png'

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Attempt to load tire ROIs from the tirechangeareas image
    tire_rois = load_tire_rois_from_image(tire_areas_path, width, height)
    if not tire_rois:
        # fallback to hard-coded ROIs (these are in absolute frame coordinates used previously)
        tire_rois = tire_rois_fallback

    unobstructed_signature, last_car_stop_sig = None, None
    CAR_STOP_THRESH = 15
    STOP_CONFIRM_FRAMES = int(fps / 5)
    stopped_frames_count, is_car_stopped, stop_start_frame = 0, False, 0
    
    # Tire activity detection thresholds & per-ROI state
    TIRE_ACTIVITY_MEAN_THRESH = 8.0            # mean absolute diff threshold for activity
    TIRE_ACTIVITY_CONFIRM_FRAMES = 1           # how many consecutive frames of motion to accept
    tire_activity_counters = [0] * len(tire_rois)

    MIN_TIRE_OVERLAP_AREA = 500
    total_stopped_time, tire_change_time, refuel_time = 0.0, 0.0, 0.0
    
    # --- Custom Tracker State (probe) ---
    refuel_bbox = None # Will store (x, y, w, h) of the probe
    is_refueling_state = False
    TRACK_LOST_THRESHOLD = 0.60 # If match score drops below this, we lose the lock

    last_gray = None

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        progress_callback(frame_count / total_frames)

        # compute gray once for motion detection
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
            total_stopped_time += (frame_count - stop_start_frame) / fps
            stop_start_frame = 0
            # reset tire activity counters when stop ends
            tire_activity_counters = [0] * len(tire_rois)

        if is_car_stopped:
            # Use motion in tire ROIs (from tirechangeareas.png or fallback) to measure tire-change activity
            if last_gray is not None:
                for i, roi in enumerate(tire_rois):
                    x1, y1, x2, y2 = roi
                    # ensure valid coordinates
                    if x2 <= x1 or y2 <= y1:
                        continue
                    prev_patch = last_gray[y1:y2, x1:x2]
                    curr_patch = curr_gray[y1:y2, x1:x2]
                    if prev_patch.size == 0 or curr_patch.size == 0 or prev_patch.shape != curr_patch.shape:
                        tire_activity_counters[i] = 0
                        continue
                    mean_diff = float(np.mean(cv2.absdiff(prev_patch, curr_patch)))
                    if mean_diff > TIRE_ACTIVITY_MEAN_THRESH:
                        tire_activity_counters[i] += 1
                    else:
                        tire_activity_counters[i] = 0

                    # If confirmed activity (>= confirm frames), count it as tire-change time for this frame
                    if tire_activity_counters[i] >= TIRE_ACTIVITY_CONFIRM_FRAMES:
                        tire_change_time += 1.0 / fps
                        # draw ROI in green to indicate active
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else:
                        # draw ROI in yellow when inactive
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            else:
                # draw ROIs (first frame after stop start) in yellow
                for (x1, y1, x2, y2) in tire_rois:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

            # existing person-based tire-change detection (legacy behavior) - keep as additive
            person_bboxes = [b.xyxy[0].cpu().numpy() for b in results[0].boxes if int(b.cls)==0] if results[0].boxes else []
            if sum(boxes_overlap_area(p, t) for p in person_bboxes for t in tire_rois) > MIN_TIRE_OVERLAP_AREA:
                # also count as tire-change activity if persons overlap those ROIs
                tire_change_time += 1.0 / fps

            # --- Custom "Track-by-Search" Refueling Logic (unchanged) ---
            if refuel_bbox is None: # We haven't locked on yet
                x, y, w, h = refuel_roi_in_air
                # initial detection uses grayscale search area
                sx1, sy1 = max(0, int(x)), max(0, int(y))
                sx2, sy2 = min(width, int(x + w)), min(height, int(y + h))
                if sx2 > sx1 and sy2 > sy1:
                    search_area = cv2.cvtColor(frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2GRAY)
                    _, score_in, _, max_loc_in = cv2.minMaxLoc(cv2.matchTemplate(search_area, probe_in_template, cv2.TM_CCOEFF_NORMED))
                    _, score_out, _, _ = cv2.minMaxLoc(cv2.matchTemplate(search_area, probe_out_template, cv2.TM_CCOEFF_NORMED))
                    
                    if score_in > score_out and score_in > 0.7:
                        is_refueling_state = True
                        # Lock On: Store the bounding box of the probe (use template dims)
                        refuel_bbox = (sx1 + int(max_loc_in[0]), sy1 + int(max_loc_in[1]), template_w, template_h)
            else: # We have a lock, now track it by searching in a local window
                search_pad = 40
                x, y, w, h = refuel_bbox
                sx1, sy1 = max(0, int(x - search_pad)), max(0, int(y - search_pad))
                sx2, sy2 = min(width, int(x + w + search_pad)), min(height, int(y + h + search_pad))

                # If the search window is smaller than the template, we cannot match -> lose the lock
                win_w = sx2 - sx1
                win_h = sy2 - sy1
                if win_w < template_w or win_h < template_h:
                    is_refueling_state = False
                    refuel_bbox = None
                else:
                    search_window = frame[sy1:sy2, sx1:sx2]
                    # Convert search window to grayscale before matchTemplate (fixes the type error)
                    search_window_gray = cv2.cvtColor(search_window, cv2.COLOR_BGR2GRAY)

                    res = cv2.matchTemplate(search_window_gray, probe_in_template, cv2.TM_CCOEFF_NORMED)
                    _, score, _, max_loc = cv2.minMaxLoc(res)

                    if score > TRACK_LOST_THRESHOLD:
                        is_refueling_state = True
                        # Update the bounding box to the new location (use template dims)
                        refuel_bbox = (sx1 + int(max_loc[0]), sy1 + int(max_loc[1]), template_w, template_h)
                        p1 = (int(refuel_bbox[0]), int(refuel_bbox[1]))
                        p2 = (int(refuel_bbox[0] + refuel_bbox[2]), int(refuel_bbox[1] + refuel_bbox[3]))
                        cv2.rectangle(annotated_frame, p1, p2, (255, 0, 255), 3, 1) # Draw magenta box
                    else:
                        # Lost lock
                        is_refueling_state = False
                        refuel_bbox = None
            
            if is_refueling_state:
                refuel_time += 1 / fps

        # --- Drawing Logic (non-stopped frames) ---
        if not is_car_stopped:
            # draw ROIs in dimmed color so they are visible even when not stopped
            for (x1, y1, x2, y2) in tire_rois:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

        cv2.rectangle(annotated_frame, ref_roi, (0, 255, 255), 2)
        # legacy tire_rois overlay (if you want them highlighted regardless)
        # Note: During stopped frames, they are drawn as active/inactive above.
        for roi in tire_rois:
            # if already drawn active/inactive, skip; otherwise draw faint rectangle (handled above)
            pass
        cv2.rectangle(annotated_frame, refuel_roi_in_air, (0, 0, 255), 2)
        
        rect_x, rect_y, rect_w, rect_h = 20, height // 2 - 60, 450, 130
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 255), -1)
        annotated_frame = cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0)
        
        current_display_stop_time = total_stopped_time + ((frame_count - stop_start_frame) / fps if is_car_stopped else 0)
        cv2.putText(annotated_frame, f'Car Stopped: {current_display_stop_time:.2f}s', (rect_x + 10, rect_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(annotated_frame, f'Tire Change: {tire_change_time:.2f}s', (rect_x + 10, rect_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(annotated_frame, f'Refueling: {refuel_time:.2f}s', (rect_x + 10, rect_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        
        out.write(annotated_frame)

        # save current gray frame as previous for next iteration
        last_gray = curr_gray.copy()

    if is_car_stopped:
        total_stopped_time += (total_frames - stop_start_frame) / fps

    cap.release()
    out.release()
    
    return total_stopped_time, tire_change_time, refuel_time
