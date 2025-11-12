import cv2
import numpy as np
from ultralytics import YOLO

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

    # --- Analysis Variables ---
    is_car_stopped = False
    stop_start_frame = 0
    total_stopped_time = 0.0
    
    tire_change_time = 0.0
    refuel_time = 0.0

    last_car_center = None
    stopped_frames_count = 0
    MOVEMENT_THRESHOLD = 5  # pixels
    STOPPED_CONFIRMATION_FRAMES = int(fps / 4) # Require 1/4 second of no movement to confirm stop

    # Regions of Interest (ROIs) from user
    tire_rois = [
        (1210, 30, 1370, 150), (1210, 400, 1400, 550),
        (685, 10, 830, 100), (685, 430, 780, 500)
    ]
    refuel_roi = (803, 328, 920, 460)

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Update progress
        progress_callback(frame_count / total_frames)

        results = model.track(frame, persist=True, classes=[2, 0], verbose=False)
        annotated_frame = results[0].plot()
        
        car_bbox = None
        person_bboxes = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            for box in results[0].boxes:
                if int(box.cls) == 2:  # Car
                    car_bbox = box.xyxy[0].cpu().numpy()
                elif int(box.cls) == 0:  # Person
                    person_bboxes.append(box.xyxy[0].cpu().numpy())

        # --- New Car Stopped Logic ---
        car_is_moving = True
        if car_bbox is not None:
            car_center = ((car_bbox[0] + car_bbox[2]) / 2, (car_bbox[1] + car_bbox[3]) / 2)
            
            if last_car_center is not None:
                distance = np.sqrt((car_center[0] - last_car_center[0])**2 + (car_center[1] - last_car_center[1])**2)
                if distance < MOVEMENT_THRESHOLD:
                    stopped_frames_count += 1
                else:
                    stopped_frames_count = 0
            
            last_car_center = car_center
            
            if stopped_frames_count > STOPPED_CONFIRMATION_FRAMES:
                car_is_moving = False

        if not car_is_moving and not is_car_stopped:
            # Car just stopped
            is_car_stopped = True
            stop_start_frame = frame_count
        elif car_is_moving and is_car_stopped:
            # Car just started moving again
            is_car_stopped = False
            total_stopped_time += (frame_count - stop_start_frame) / fps
            stop_start_frame = 0

        # --- Tire Change and Refuel Logic (only when car is stopped) ---
        if is_car_stopped:
            person_in_tire_roi = any(boxes_overlap(p_bbox, t_roi) for p_bbox in person_bboxes for t_roi in tire_rois)
            if person_in_tire_roi:
                tire_change_time += 1 / fps

            person_in_refuel_roi = any(boxes_overlap(p_bbox, refuel_roi) for p_bbox in person_bboxes)
            if person_in_refuel_roi:
                refuel_time += 1 / fps

        # --- Draw ROIs and Stats ---
        for roi in tire_rois:
            cv2.rectangle(annotated_frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 2)
        cv2.rectangle(annotated_frame, (refuel_roi[0], refuel_roi[1]), (refuel_roi[2], refuel_roi[3]), (0, 0, 255), 2)

        # Calculate current stopped time for display
        current_display_stopped_time = total_stopped_time
        if is_car_stopped:
            current_display_stopped_time += (frame_count - stop_start_frame) / fps
        
        cv2.putText(annotated_frame, f'Car Stopped: {current_display_stopped_time:.2f}s', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Tire Change: {tire_change_time:.2f}s', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Refueling: {refuel_time:.2f}s', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(annotated_frame)

    # Handle case where car is still stopped when video ends
    if is_car_stopped:
        total_stopped_time += (total_frames - stop_start_frame) / fps

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return total_stopped_time, tire_change_time, refuel_time

def boxes_overlap(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])
