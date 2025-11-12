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
    
    # intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    return intersection

def process_video(video_path, output_path, progress_callback):
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # --- Load Fuel Nozzle Template ---
    template_path = 'refs/fuelerpluggedin.png'  # Use the new, more accurate template
    if not os.path.exists(template_path):
        print(f"Error: Fuel nozzle template not found at {template_path}")
        print("Please ensure 'fuelerpluggedin.png' is in the 'refs' directory.")
        return None, None, None
    fuel_nozzle_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if fuel_nozzle_template is None:
        print(f"Error: Could not read template image from {template_path}")
        return None, None, None
    print(f"Successfully loaded fuel nozzle template from {template_path}") # Debug message
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # --- Static Reference Point & Analysis Variables ---
    ref_roi = (1042, 463, 1059, 487)
    unobstructed_signature, last_frame_signature = None, None
    SIGNATURE_CHANGE_THRESHOLD = 15
    STOPPED_CONFIRMATION_FRAMES = int(fps / 5)
    stopped_frames_count = 0
    is_car_stopped, stop_start_frame = False, 0
    total_stopped_time, tire_change_time, refuel_time = 0.0, 0.0, 0.0

    # ROIs
    tire_rois = [
        (1210, 30, 1370, 150), (1210, 400, 1400, 550),
        (685, 10, 830, 100), (685, 430, 780, 500)
    ]
    refuel_roi = (803, 328, 920, 460)
    MIN_TIRE_OVERLAP_AREA = 500

    # --- Template Matching Threshold ---
    NOZZLE_MATCH_THRESHOLD = 0.7

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        progress_callback(frame_count / total_frames)

        if frame_count == 0:
            unobstructed_signature = get_patch_signature(frame, ref_roi)

        results = model.track(frame, persist=True, classes=[0], verbose=False)
        annotated_frame = results[0].plot()

        # Car Stopped Logic (remains the same)
        current_signature = get_patch_signature(frame, ref_roi)
        is_obstructed = np.linalg.norm(current_signature - unobstructed_signature) > SIGNATURE_CHANGE_THRESHOLD
        car_is_moving = True
        if is_obstructed:
            if last_frame_signature is not None:
                movement_diff = np.linalg.norm(current_signature - last_frame_signature)
                if movement_diff < SIGNATURE_CHANGE_THRESHOLD:
                    stopped_frames_count += 1
                else: stopped_frames_count = 0
            if stopped_frames_count > STOPPED_CONFIRMATION_FRAMES:
                car_is_moving = False
        else: stopped_frames_count = 0
        last_frame_signature = current_signature
        
        if not car_is_moving and not is_car_stopped:
            is_car_stopped = True
            stop_start_frame = frame_count
        elif car_is_moving and is_car_stopped:
            is_car_stopped = False
            total_stopped_time += (frame_count - stop_start_frame) / fps
            stop_start_frame = 0

        # --- Action Logic (only when car is stopped) ---
        if is_car_stopped:
            person_bboxes = [box.xyxy[0].cpu().numpy() for box in results[0].boxes if int(box.cls) == 0] if results[0].boxes and results[0].boxes.id is not None else []
            
            total_overlap = sum(boxes_overlap_area(p, t) for p in person_bboxes for t in tire_rois)
            if total_overlap > MIN_TIRE_OVERLAP_AREA:
                tire_change_time += 1 / fps

            # --- Fuel nozzle logic with template matching ---
            x, y, w, h = refuel_roi[0], refuel_roi[1], refuel_roi[2]-refuel_roi[0], refuel_roi[3]-refuel_roi[1]
            if h > 0 and w > 0:
                refuel_search_area = frame[y:y+h, x:x+w]
                refuel_search_area_gray = cv2.cvtColor(refuel_search_area, cv2.COLOR_BGR2GRAY)
                
                # Ensure template is not larger than search area
                if refuel_search_area_gray.shape[0] >= fuel_nozzle_template.shape[0] and \
                   refuel_search_area_gray.shape[1] >= fuel_nozzle_template.shape[1]:
                    match_result = cv2.matchTemplate(refuel_search_area_gray, fuel_nozzle_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(match_result)

                    if max_val > NOZZLE_MATCH_THRESHOLD:
                        refuel_time += 1 / fps

        # --- Draw ROIs and Stats ---
        cv2.rectangle(annotated_frame, (ref_roi[0], ref_roi[1]), (ref_roi[2], ref_roi[3]), (0, 255, 255), 2)
        for roi in tire_rois: cv2.rectangle(annotated_frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 2)
        cv2.rectangle(annotated_frame, (refuel_roi[0], ref_roi[1]), (ref_roi[2], ref_roi[3]), (0, 0, 255), 2)

        current_display_stopped_time = total_stopped_time + ((frame_count - stop_start_frame) / fps if is_car_stopped else 0)
        cv2.putText(annotated_frame, f'Car Stopped: {current_display_stopped_time:.2f}s', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Tire Change: {tire_change_time:.2f}s', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Refueling: {refuel_time:.2f}s', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(annotated_frame)

    if is_car_stopped:
        total_stopped_time += (total_frames - stop_start_frame) / fps

    cap.release()
    out.release()
    
    return total_stopped_time, tire_change_time, refuel_time
