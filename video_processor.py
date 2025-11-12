import cv2
import numpy as np
from ultralytics import YOLO
import os

def get_patch_signature(frame, roi):
    """Calculates a simple signature (mean color) of a region."""
    x1, y1, x2, y2 = roi
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return np.array([0, 0, 0])
    return np.mean(patch, axis=(0, 1))

def boxes_overlap_area(box1, box2):
    """Calculates the area of overlap between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    return intersection

def process_video(video_path, output_path, progress_callback):
    model = YOLO('yolov8n.pt')
    
    # --- Load Both Refueling Templates ---
    active_template_path = 'refs/fuelerpluggedin.png'
    clean_template_path = 'refs/emptyfuelport.png'
    
    if not os.path.exists(active_template_path) or not os.path.exists(clean_template_path):
        print("Error: Ensure 'fuelerpluggedin.png' and 'emptyfuelport.png' are in the 'refs' directory.")
        return None, None, None
        
    active_template = cv2.imread(active_template_path, cv2.IMREAD_GRAYSCALE)
    clean_template = cv2.imread(clean_template_path, cv2.IMREAD_GRAYSCALE)
    
    if active_template is None or clean_template is None:
        print("Error: Could not read one or both template images.")
        return None, None, None

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # --- Core Variables ---
    ref_roi = (1042, 463, 1059, 487)
    unobstructed_signature, last_car_stop_sig = None, None
    CAR_STOP_THRESH = 15
    STOP_CONFIRM_FRAMES = int(fps / 5)
    stopped_frames_count = 0
    is_car_stopped, stop_start_frame = False, 0
    total_stopped_time, tire_change_time, refuel_time = 0.0, 0.0, 0.0

    is_refueling = False
    MIN_MATCH_THRESHOLD = 0.6 # Minimum confidence to consider a match valid

    tire_rois = [(1210, 30, 1370, 150), (1210, 400, 1400, 550), (685, 10, 830, 100), (685, 430, 780, 500)]
    refuel_roi = (803, 328, 920, 460)
    MIN_TIRE_OVERLAP_AREA = 500

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        progress_callback(frame_count / total_frames)

        if frame_count == 0:
            unobstructed_signature = get_patch_signature(frame, ref_roi)

        results = model.track(frame, persist=True, classes=[0], verbose=False)
        annotated_frame = results[0].plot()

        current_car_stop_sig = get_patch_signature(frame, ref_roi)
        is_obstructed = np.linalg.norm(current_car_stop_sig - unobstructed_signature) > CAR_STOP_THRESH
        car_is_moving = True
        if is_obstructed:
            if last_car_stop_sig is not None:
                if np.linalg.norm(current_car_stop_sig - last_car_stop_sig) < CAR_STOP_THRESH:
                    stopped_frames_count += 1
                else: stopped_frames_count = 0
            if stopped_frames_count > STOP_CONFIRM_FRAMES:
                car_is_moving = False
        else: stopped_frames_count = 0
        last_car_stop_sig = current_car_stop_sig

        if not car_is_moving and not is_car_stopped:
            is_car_stopped = True
            stop_start_frame = frame_count
        elif car_is_moving and is_car_stopped:
            is_car_stopped, is_refueling = False, False
            total_stopped_time += (frame_count - stop_start_frame) / fps
            stop_start_frame = 0

        if is_car_stopped:
            # --- Competitive Template Matching for Refueling ---
            x, y, w, h = refuel_roi[0], refuel_roi[1], refuel_roi[2]-refuel_roi[0], refuel_roi[3]-refuel_roi[1]
            if h > 0 and w > 0:
                search_area_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                
                # Match for the "active" template
                _, active_score, _, _ = cv2.minMaxLoc(cv2.matchTemplate(search_area_gray, active_template, cv2.TM_CCOEFF_NORMED))
                # Match for the "clean" template
                _, clean_score, _, _ = cv2.minMaxLoc(cv2.matchTemplate(search_area_gray, clean_template, cv2.TM_CCOEFF_NORMED))
                
                # The "active" state wins if its score is higher and above a minimum threshold
                if active_score > clean_score and active_score > MIN_MATCH_THRESHOLD:
                    is_refueling = True
                else:
                    is_refueling = False
            
            if is_refueling:
                refuel_time += 1 / fps

            person_bboxes = [b.xyxy[0].cpu().numpy() for b in results[0].boxes if int(b.cls)==0] if results[0].boxes and results[0].boxes.id is not None else []
            if sum(boxes_overlap_area(p, t) for p in person_bboxes for t in tire_rois) > MIN_TIRE_OVERLAP_AREA:
                tire_change_time += 1 / fps

        # Drawing logic...
        cv2.rectangle(annotated_frame, (ref_roi[0], ref_roi[1]), (ref_roi[2], ref_roi[3]), (0, 255, 255), 2)
        for roi in tire_rois: cv2.rectangle(annotated_frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 2)
        cv2.rectangle(annotated_frame, (refuel_roi[0], ref_roi[1]), (ref_roi[2], ref_roi[3]), (0, 0, 255), 2)
        
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
