import cv2
import numpy as np
from ultralytics import YOLO

def get_patch_signature(frame, roi):
    """Calculates a simple signature (mean color) of a region."""
    x1, y1, x2, y2 = roi
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return np.array([0, 0, 0])
    return np.mean(patch, axis=(0, 1))

def process_video(video_path, output_path, progress_callback):
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, None

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # --- Static Reference Point Logic ---
    ref_roi = (1042, 463, 1059, 487)
    unobstructed_signature = None
    last_frame_signature = None
    SIGNATURE_CHANGE_THRESHOLD = 15  # How much the patch must change to be considered "moving"
    STOPPED_CONFIRMATION_FRAMES = int(fps / 5) # Require 1/5 second of no movement
    stopped_frames_count = 0

    # --- Analysis Variables ---
    is_car_stopped = False
    stop_start_frame = 0
    total_stopped_time = 0.0
    tire_change_time = 0.0
    refuel_time = 0.0
    
    # Regions of Interest for actions
    tire_rois = [
        (1210, 30, 1370, 150), (1210, 400, 1400, 550),
        (685, 10, 830, 100), (685, 430, 780, 500)
    ]
    refuel_roi = (803, 328, 920, 460)

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        progress_callback(frame_count / total_frames)

        if frame_count == 0:
            unobstructed_signature = get_patch_signature(frame, ref_roi)

        results = model.track(frame, persist=True, classes=[0], verbose=False) # Only track people now
        annotated_frame = results[0].plot()

        # --- Car Stopped Logic using Reference Point ---
        current_signature = get_patch_signature(frame, ref_roi)
        
        # Check if the reference point is obstructed by comparing to the initial empty state
        is_obstructed = np.linalg.norm(current_signature - unobstructed_signature) > SIGNATURE_CHANGE_THRESHOLD
        
        car_is_moving = True
        if is_obstructed:
            if last_frame_signature is not None:
                # Check for movement by comparing the patch between consecutive frames
                movement_diff = np.linalg.norm(current_signature - last_frame_signature)
                if movement_diff < SIGNATURE_CHANGE_THRESHOLD:
                    stopped_frames_count += 1
                else:
                    stopped_frames_count = 0
            
            if stopped_frames_count > STOPPED_CONFIRMATION_FRAMES:
                car_is_moving = False
        else:
            stopped_frames_count = 0

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
            person_bboxes = []
            if results[0].boxes is not None and results[0].boxes.id is not None:
                for box in results[0].boxes:
                    if int(box.cls) == 0:
                        person_bboxes.append(box.xyxy[0].cpu().numpy())
            
            if any(boxes_overlap(p, t) for p in person_bboxes for t in tire_rois):
                tire_change_time += 1 / fps
            if any(boxes_overlap(p, refuel_roi) for p in person_bboxes):
                refuel_time += 1 / fps

        # --- Draw ROIs and Stats ---
        cv2.rectangle(annotated_frame, (ref_roi[0], ref_roi[1]), (ref_roi[2], ref_roi[3]), (0, 255, 255), 2) # Yellow ref box
        for roi in tire_rois:
            cv2.rectangle(annotated_frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 2)
        cv2.rectangle(annotated_frame, (refuel_roi[0], refuel_roi[1]), (refuel_roi[2], ref_roi[3]), (0, 0, 255), 2)

        current_display_stopped_time = total_stopped_time + ((frame_count - stop_start_frame) / fps if is_car_stopped else 0)
        cv2.putText(annotated_frame, f'Car Stopped: {current_display_stopped_time:.2f}s', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Tire Change: {tire_change_time:.2f}s', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Refueling: {refuel_time:.2f}s', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(annotated_frame)

    if is_car_stopped:
        total_stopped_time += (total_frames - stop_start_frame) / fps

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return total_stopped_time, tire_change_time, refuel_time

def boxes_overlap(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])
