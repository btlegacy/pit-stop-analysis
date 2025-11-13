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

def process_video(video_path, output_path, progress_callback):
    model = YOLO('yolov8n.pt')

    # --- Load Templates for Initial Detection ---
    probe_in_path = 'refs/probe_in.png'
    probe_out_path = 'refs/probe_out.png'
    if not os.path.exists(probe_in_path) or not os.path.exists(probe_out_path):
        print("Error: Please create 'refs/probe_in.png' and 'refs/probe_out.png' template files.")
        return [0.0] * 3
    
    probe_in_template = cv2.imread(probe_in_path, cv2.IMREAD_GRAYSCALE)
    probe_out_template = cv2.imread(probe_out_path, cv2.IMREAD_GRAYSCALE)
    if probe_in_template is None or probe_out_template is None:
        print("Error: Could not read probe template images.")
        return [0.0] * 3

    # Template dims (grayscale)
    template_h, template_w = probe_in_template.shape[:2]

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
    STOP_CONFIRM_FRAMES = int(fps / 5)  # keep arrival confirmation as before
    stopped_frames_count, is_car_stopped, stop_start_frame = 0, False, 0
    
    # --- Early-Movement Detection (to end the stop promptly but avoid false positives) ---
    CAR_MOVE_EXPAND_X = 120
    CAR_MOVE_EXPAND_Y = 50

    # Tuned for 30 fps: 2-frame confirmation (~67 ms)
    MOVEMENT_MEAN_THRESH = 4.0                 # mean absolute difference threshold (slightly higher to ignore tiny noise)
    MOVEMENT_PIXEL_DIFF = 8                    # per-pixel intensity diff threshold
    MOVEMENT_PCT_THRESH = 0.003                # fraction of pixels changed to consider motion (0.3%)
    MOVING_CONFIRM_FRAMES = 2                  # require movement in 2 consecutive frames to end stop
    moving_frames_count = 0
    prev_frame_gray = None

    MIN_TIRE_OVERLAP_AREA = 500
    total_stopped_time, tire_change_time, refuel_time = 0.0, 0.0, 0.0
    
    # --- Custom Tracker State ---
    refuel_bbox = None # Will store (x, y, w, h) of the probe
    is_refueling_state = False
    TRACK_LOST_THRESHOLD = 0.60 # If match score drops below this, we lose the lock

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        progress_callback(frame_count / total_frames)

        # Precompute grayscale for movement checks
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = model.track(frame, persist=True, classes=[0], verbose=False)
        annotated_frame = results[0].plot()

        if frame_count == 0:
            unobstructed_signature = get_patch_signature(frame, ref_roi)

        current_car_stop_sig = get_patch_signature(frame, ref_roi)
        # (Car stop detection unchanged for arrival)
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

        # --- Early movement detection to end stop promptly ---
        if not car_is_moving and not is_car_stopped:
            # just transitioned to stopped state on this frame; initialize movement counters
            is_car_stopped, stop_start_frame = True, frame_count
            moving_frames_count = 0
        elif car_is_moving and is_car_stopped:
            # car started moving in the normal path: finalize stop
            is_car_stopped, is_refueling_state, refuel_bbox = False, False, None
            total_stopped_time += (frame_count - stop_start_frame) / fps
            stop_start_frame = 0
            moving_frames_count = 0

        if is_car_stopped:
            # compute expanded car movement ROI around ref_roi
            cx1 = max(0, ref_roi[0] - CAR_MOVE_EXPAND_X)
            cy1 = max(0, ref_roi[1] - CAR_MOVE_EXPAND_Y)
            cx2 = min(width, ref_roi[2] + CAR_MOVE_EXPAND_X)
            cy2 = min(height, ref_roi[3] + CAR_MOVE_EXPAND_Y)

            if prev_frame_gray is not None:
                prev_patch = prev_frame_gray[cy1:cy2, cx1:cx2]
                curr_patch = curr_gray[cy1:cy2, cx1:cx2]
                if prev_patch.size > 0 and curr_patch.size == prev_patch.size:
                    # mean absolute difference
                    mean_diff = float(np.mean(cv2.absdiff(prev_patch, curr_patch)))

                    # pixel-wise motion percentage
                    diff = cv2.absdiff(prev_patch, curr_patch)
                    motion_mask = diff > MOVEMENT_PIXEL_DIFF
                    motion_pct = float(np.count_nonzero(motion_mask)) / (diff.size + 1e-9)

                    # consider this frame "moving" if either metric passes
                    if mean_diff > MOVEMENT_MEAN_THRESH or motion_pct > MOVEMENT_PCT_THRESH:
                        moving_frames_count += 1
                    else:
                        moving_frames_count = 0

                    # If movement persists for the short confirmation window, end the stop immediately
                    if moving_frames_count >= MOVING_CONFIRM_FRAMES:
                        # finalize stop and reset states
                        is_car_stopped, is_refueling_state, refuel_bbox = False, False, None
                        total_stopped_time += (frame_count - stop_start_frame) / fps
                        stop_start_frame = 0
                        moving_frames_count = 0
                else:
                    moving_frames_count = 0

            # -- existing tire-change using person boxes --
            person_bboxes = [b.xyxy[0].cpu().numpy() for b in results[0].boxes if int(b.cls)==0] if results[0].boxes else []
            if sum(boxes_overlap_area(p, t) for p in person_bboxes for t in tire_rois) > MIN_TIRE_OVERLAP_AREA:
                tire_change_time += 1/fps

            # --- Custom "Track-by-Search" Refueling Logic ---
            if refuel_bbox is None: # initial detection
                x, y, w, h = refuel_roi_in
