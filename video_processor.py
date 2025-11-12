import cv2
import numpy as np
from ultralytics import YOLO

def process_video(video_path, output_path):
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # --- Analysis Variables ---
    car_positions = []
    is_car_stopped = False
    stop_start_time = 0
    total_stopped_time = 0

    tire_change_time = 0
    refuel_time = 0

    # For simplicity, we'll define static regions of interest (ROIs) for tire changing and refueling
    # These would need to be adjusted for your specific video's camera angle
    # Format: (x_min, y_min, x_max, y_max)
    tire_rois = [
        (int(width * 0.2), int(height * 0.4), int(width * 0.4), int(height * 0.6)),  # Front-left
        (int(width * 0.6), int(height * 0.4), int(width * 0.8), int(height * 0.6)),  # Front-right
    ]
    refuel_roi = (int(width * 0.6), int(height * 0.2), int(width * 0.8), int(height * 0.4)) # Side of the car

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time_sec = frame_count / fps

        # Run YOLOv8 tracking on the frame
        results = model.track(frame, persist=True, classes=[2, 0]) # 2 is car, 0 is person

        # Get the annotated frame
        annotated_frame = results[0].plot()
        
        car_bbox = None
        person_bboxes = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls)
                if cls_id == 2: # Car
                    car_bbox = box.xyxy[0].cpu().numpy()
                elif cls_id == 0: # Person
                    person_bboxes.append(box.xyxy[0].cpu().numpy())

        # --- Car Stopped Logic ---
        if car_bbox is not None:
            car_center_x = (car_bbox[0] + car_bbox[2]) / 2
            car_positions.append(car_center_x)

            if len(car_positions) > int(fps * 0.5): # Analyze last 0.5 seconds
                car_positions.pop(0)
                movement = np.std(car_positions)

                if movement < 2.0 and not is_car_stopped: # Threshold for stopped
                    is_car_stopped = True
                    stop_start_time = current_time_sec
                elif movement >= 2.0 and is_car_stopped:
                    is_car_stopped = False
                    total_stopped_time += (current_time_sec - stop_start_time)
                    stop_start_time = 0
        
        # --- Tire Change and Refuel Logic (if car is stopped) ---
        if is_car_stopped:
            in_tire_roi = False
            for p_bbox in person_bboxes:
                for t_roi in tire_rois:
                    if boxes_overlap(p_bbox, t_roi):
                        in_tire_roi = True
                        break
            if in_tire_roi:
                tire_change_time += 1 / fps

            in_refuel_roi = False
            for p_bbox in person_bboxes:
                 if boxes_overlap(p_bbox, refuel_roi):
                    in_refuel_roi = True
                    break
            if in_refuel_roi:
                 refuel_time += 1/fps


        # --- Draw ROIs and Stats ---
        for roi in tire_rois:
            cv2.rectangle(annotated_frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 2)
            cv2.putText(annotated_frame, 'Tire Area', (roi[0], roi[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        cv2.rectangle(annotated_frame, (refuel_roi[0], refuel_roi[1]), (refuel_roi[2], refuel_roi[3]), (0, 255, 255), 2)
        cv2.putText(annotated_frame, 'Refuel Area', (refuel_roi[0], refuel_roi[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Display stats
        if is_car_stopped and stop_start_time > 0:
            current_stop_duration = current_time_sec - stop_start_time
            display_stopped_time = total_stopped_time + current_stop_duration
        else:
            display_stopped_time = total_stopped_time

        cv2.putText(annotated_frame, f'Car Stopped Time: {display_stopped_time:.2f}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Tire Change Time: {tire_change_time:.2f}s', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Refuel Time: {refuel_time:.2f}s', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to {output_path}")

def boxes_overlap(box1, box2):
    # box format: (x_min, y_min, x_max, y_max)
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])