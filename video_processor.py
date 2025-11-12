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
    # --- Re-initialize YOLO model ---
    model = YOLO('yolov8n.pt')

    # --- Load Master Baseline Image ---
    baseline_path = 'refs/car.png'
    if not os.path.exists(baseline_path):
        print("Error: 'car.png' not found in 'refs' directory.")
        return [0.0] * 4
    baseline_img = cv2.imread(baseline_path)
    if baseline_img is None:
        print("Error: Could not read 'car.png'.")
        return [0.0] * 4

    # --- ROIs ---
    ref_roi = (1042, 463, 1059, 487)
    tire_rois = [(1210, 30, 1370, 150), (1210, 400, 1400, 550), (685, 10, 830, 100), (685, 430, 780, 500)]
    refuel_roi_in_air = (803, 328, 920, 460)
    refuel_roi_on_ground = (refuel_roi_in_air[0], refuel_roi_in_air[1] + 20, refuel_roi_in_air[2], refuel_roi_in_air[3] + 20)

    # --- Calculate Baseline Signatures for Refueling from car.png ---
    baseline_sig_refuel_air = get_patch_signature(baseline_img, refuel_roi_in_air)
    M = np.float32([[1, 0, 0], [0, 1, 20]])
    shifted_baseline = cv2.warpAffine(baseline_img, M, (baseline_img.shape[1], baseline_img.shape[0]))
    baseline_sig_refuel_ground = get_patch_signature(shifted_baseline, refuel_roi_on_ground)

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # --- Core & Car Drop Variables ---
    unobstructed_signature, last_car_stop_sig = None, None
    CAR_STOP_THRESH = 15
    STOP_CONFIRM_FRAMES = int(fps / 5)
    stopped_frames_count, is_car_stopped, stop_start_frame = 0, False, 0
    is_car_on_ground = False
    
    # --- Activity Thresholds & Timers ---
    ACTIVITY_THRESHOLD = 25
    MIN_TIRE_OVERLAP_AREA = 500
    total_stopped_time, tire_change_time = 0.0, 0.0
    refuel_time_in_air, refuel_time_on_ground = 0.0, 0.0

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        progress_callback(frame_count / total_frames)

        if frame_count == 0:
            unobstructed_signature = get_patch_signature(frame, ref_roi)
        
        # --- Person Tracking ---
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        annotated_frame = results[0].plot()

        # --- Car Stop/Drop Logic ---
        current_car_stop_sig = get_patch_signature(frame, ref_roi)
        is_obstructed = np.linalg.norm(current_car_stop_sig - unobstructed_signature) > CAR_STOP_THRESH
        car_is_moving = True
        if is_obstructed:
            if last_car_stop_sig is not None:
                sig_diff = np.linalg.norm(current_car_stop_sig - last_car_stop_sig)
                if sig_diff < CAR_STOP_THRESH:
                    stopped_frames_count += 1
                else:
                    if is_car_stopped and not is_car_on_ground and sig_diff > (CAR_STOP_THRESH + 5):
                        is_car_on_ground = True
                    stopped_frames_count = 0
            if stopped_frames_count > STOP_CONFIRM_FRAMES:
                car_is_moving = False
        else: stopped_frames_count = 0
        last_car_stop_sig = current_car_stop_sig

        if not car_is_moving and not is_car_stopped:
            is_car_stopped, stop_start_frame, is_car_on_ground = True, frame_count, False
        elif car_is_moving and is_car_stopped:
            is_car_stopped = False
            total_stopped_time += (frame_count - stop_start_frame) / fps
            stop_start_frame = 0

        # --- Activity Detection ---
        if is_car_stopped:
            # --- Tire Change Detection (using Person BBoxes) ---
            person_bboxes = [b.xyxy[0].cpu().numpy() for b in results[0].boxes if int(b.cls)==0] if results[0].boxes else []
            if sum(boxes_overlap_area(p, t) for p in person_bboxes for t in tire_rois) > MIN_TIRE_OVERLAP_AREA:
                tire_change_time += 1/fps

            # --- Refueling Detection (using Baseline Signatures) ---
            if is_car_on_ground:
                current_refuel_sig = get_patch_signature(frame, refuel_roi_on_ground)
                if np.linalg.norm(current_refuel_sig - baseline_sig_refuel_ground) > ACTIVITY_THRESHOLD:
                    refuel_time_on_ground += 1 / fps
            else:
                current_refuel_sig = get_patch_signature(frame, refuel_roi_in_air)
                if np.linalg.norm(current_refuel_sig - baseline_sig_refuel_air) > ACTIVITY_THRESHOLD:
                    refuel_time_in_air += 1 / fps

        # --- Drawing Logic ---
        cv2.rectangle(annotated_frame, ref_roi, (0, 255, 255), 2)
        for roi in tire_rois: cv2.rectangle(annotated_frame, roi, (255, 255, 0), 2)
        cv2.rectangle(annotated_frame, refuel_roi_in_air, (0, 0, 255), 2)
        cv2.rectangle(annotated_frame, refuel_roi_on_ground, (0, 165, 255), 2)
        
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
