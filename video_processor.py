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
    
    active_template = cv2.imread('refs/fuelerpluggedin.png', cv2.IMREAD_GRAYSCALE)
    clean_template = cv2.imread('refs/emptyfuelport.png', cv2.IMREAD_GRAYSCALE)
    if active_template is None or clean_template is None:
        print("Error: Ensure refueling templates are in 'refs' directory.")
        return [0.0] * 4

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # --- Core & Car Drop Variables ---
    ref_roi = (1042, 463, 1059, 487)
    unobstructed_signature, last_car_stop_sig = None, None
    CAR_STOP_THRESH = 15
    STOP_CONFIRM_FRAMES = int(fps / 5)
    stopped_frames_count, is_car_stopped, stop_start_frame = 0, False, 0
    is_car_on_ground = False
    prev_ref_patch_gray = None
    CAR_DROP_Y_MOVEMENT_THRESH = 2.0 

    # --- Timer & State Variables ---
    total_stopped_time, tire_change_time = 0.0, 0.0
    refuel_time_in_air, refuel_time_on_ground = 0.0, 0.0
    is_refueling = False
    MIN_MATCH_THRESHOLD = 0.6
    tire_rois = [(1210, 30, 1370, 150), (1210, 400, 1400, 550), (685, 10, 830, 100), (685, 430, 780, 500)]
    
    # --- Define both refueling ROIs ---
    refuel_roi_in_air = (803, 328, 920, 460)
    refuel_roi_on_ground = (refuel_roi_in_air[0], refuel_roi_in_air[1] + 20, refuel_roi_in_air[2], refuel_roi_in_air[3] + 20)
    
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
            if last_car_stop_sig is not None and np.linalg.norm(current_car_stop_sig - last_car_stop_sig) < CAR_STOP_THRESH:
                stopped_frames_count += 1
            else: stopped_frames_count = 0
            if stopped_frames_count > STOP_CONFIRM_FRAMES:
                car_is_moving = False
        else: stopped_frames_count = 0
        last_car_stop_sig = current_car_stop_sig

        if not car_is_moving and not is_car_stopped:
            is_car_stopped, stop_start_frame, is_car_on_ground = True, frame_count, False
        elif car_is_moving and is_car_stopped:
            is_car_stopped, is_refueling = False, False
            total_stopped_time += (frame_count - stop_start_frame) / fps
            stop_start_frame = 0

        if is_car_stopped:
            # --- Car Drop Detection ---
            if not is_car_on_ground:
                x1, y1, x2, y2 = ref_roi
                current_ref_patch_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                if prev_ref_patch_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_ref_patch_gray, current_ref_patch_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    if flow is not None and np.mean(flow[..., 1]) > CAR_DROP_Y_MOVEMENT_THRESH:
                        is_car_on_ground = True
                prev_ref_patch_gray = current_ref_patch_gray
            
            # --- Refueling Logic with Dynamic ROI ---
            current_refuel_roi = refuel_roi_on_ground if is_car_on_ground else refuel_roi_in_air
            x, y, w, h = current_refuel_roi[0], current_refuel_roi[1], current_refuel_roi[2]-current_refuel_roi[0], current_refuel_roi[3]-current_refuel_roi[1]

            if h > 0 and w > 0:
                search_area = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                _, active_score, _, _ = cv2.minMaxLoc(cv2.matchTemplate(search_area, active_template, cv2.TM_CCOEFF_NORMED))
                _, clean_score, _, _ = cv2.minMaxLoc(cv2.matchTemplate(search_area, clean_template, cv2.TM_CCOEFF_NORMED))
                is_refueling = active_score > clean_score and active_score > MIN_MATCH_THRESHOLD

            if is_refueling:
                if is_car_on_ground:
                    refuel_time_on_ground += 1 / fps
                else:
                    refuel_time_in_air += 1 / fps
            
            # Tire change logic
            person_bboxes = [b.xyxy[0].cpu().numpy() for b in results[0].boxes if int(b.cls)==0] if results[0].boxes else []
            if sum(boxes_overlap_area(p, t) for p in person_bboxes for t in tire_rois) > MIN_TIRE_OVERLAP_AREA:
                tire_change_time += 1/fps

        # Drawing logic...
        cv2.rectangle(annotated_frame, ref_roi_in_air, (0, 0, 255), 2)
        cv2.rectangle(annotated_frame, refuel_roi_on_ground, (0, 165, 255), 2) # Orange for on-ground
        
        total_refuel_time = refuel_time_in_air + refuel_time_on_ground
        rect_x, rect_y, rect_w, rect_h = 20, height // 2 - 80, 550, 180
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 255), -1)
        annotated_frame = cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0)
        
        current_display_stop_time = total_stopped_time + ((frame_count - stop_start_frame) / fps if is_car_stopped else 0)
        cv2.putText(annotated_frame, f'Car Stopped: {current_display_stop_time:.2f}s', (rect_x + 10, rect_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(annotated_frame, f'Tire Change: {tire_change_time:.2f}s', (rect_x + 10, rect_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(annotated_frame, f'Refueling (In Air): {refuel_time_in_air:.2f}s', (rect_x + 10, rect_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(annotated_frame, f'Refueling (On Ground): {refuel_time_on_ground:.2f}s', (rect_x + 10, rect_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(annotated_frame, f'Total Refueling: {total_refuel_time:.2f}s', (rect_x + 10, rect_y + 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        
        out.write(annotated_frame)

    if is_car_stopped:
        total_stopped_time += (total_frames - stop_start_frame) / fps

    cap.release()
    out.release()
    
    return total_stopped_time, tire_change_time, refuel_time_in_air, refuel_time_on_ground
