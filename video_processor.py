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
    
    # --- Adaptive Tracker State ---
    refuel_tracker = None
    is_refueling_state = False

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        progress_callback(frame_count / total_frames)

        results = model.track(frame, persist=True, classes=[0], verbose=False)
        annotated_frame = results[0].plot()

        if frame_count == 0:
            unobstructed_signature = get_patch_signature(frame, ref_roi)

        current_car_stop_sig = get_patch_signature(frame, ref_roi)
        # Car stop logic
        is_obstructed = np.linalg.norm(current_car_stop_sig - unobstructed_signature) > CAR_STOP_THRESH
        car_is_moving = True
        if is_obstructed:
            if last_car_stop_sig is not None and np.linalg.norm(current_car_stop_sig - last_car_stop_sig) < CAR_STOP_THRESH:
                stopped_frames_count += 1
            else: stopped_frames_count = 0
            if stopped_frames_count > STOP_CONFIRM_FRAMES:
                car_is_moving = False
        else: stopped_frames_count = 0
        last_car_stop_sig = current_car_stop_sig

        if not car_is_moving and not is_car_stopped:
            is_car_stopped, stop_start_frame = True, frame_count
        elif car_is_moving and is_car_stopped:
            is_car_stopped, is_refueling_state, refuel_tracker = False, False, None
            total_stopped_time += (frame_count - stop_start_frame) / fps
            stop_start_frame = 0

        if is_car_stopped:
            person_bboxes = [b.xyxy[0].cpu().numpy() for b in results[0].boxes if int(b.cls)==0] if results[0].boxes else []
            if sum(boxes_overlap_area(p, t) for p in person_bboxes for t in tire_rois) > MIN_TIRE_OVERLAP_AREA:
                tire_change_time += 1/fps

            # --- Adaptive Refueling Logic ---
            if refuel_tracker is None: # We haven't started tracking yet
                x, y, w, h = refuel_roi_in_air
                search_area = cv2.cvtColor(frame[int(y):int(y+h), int(x):int(x+w)], cv2.COLOR_BGR2GRAY)
                
                _, score_in, _, max_loc_in = cv2.minMaxLoc(cv2.matchTemplate(search_area, probe_in_template, cv2.TM_CCOEFF_NORMED))
                _, score_out, _, _ = cv2.minMaxLoc(cv2.matchTemplate(search_area, probe_out_template, cv2.TM_CCOEFF_NORMED))
                
                if score_in > score_out and score_in > 0.7:
                    is_refueling_state = True
                    # --- Initialize Tracker ---
                    print("Probe detected. Initializing tracker.")
                    tracker_roi = (x + max_loc_in[0], y + max_loc_in[1], probe_in_template.shape[1], probe_in_template.shape[0])
                    refuel_tracker = cv2.TrackerCSRT_create()
                    refuel_tracker.init(frame, tracker_roi)
            else: # Tracker is active
                success, bbox = refuel_tracker.update(frame)
                if success:
                    is_refueling_state = True
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(annotated_frame, p1, p2, (255, 0, 255), 3, 1) # Draw magenta box for tracked probe
                else:
                    is_refueling_state = False
                    refuel_tracker = None # Tracker lost the object
            
            if is_refueling_state:
                refuel_time += 1 / fps

        # --- Drawing Logic ---
        cv2.rectangle(annotated_frame, ref_roi, (0, 255, 255), 2)
        for roi in tire_rois: cv2.rectangle(annotated_frame, roi, (255, 255, 0), 2)
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

    if is_car_stopped:
        total_stopped_time += (total_frames - stop_start_frame) / fps

    cap.release()
    out.release()
    
    return total_stopped_time, tire_change_time, refuel_time
