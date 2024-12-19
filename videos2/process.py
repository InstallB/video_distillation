import cv2
import os
import torch
import config
import scoring
import numpy as np
import torchvision.transforms.functional as F

BOUNDING_BOX_SELECT_NUM = 2

def compute_new_bounding_box(x, y, w, h, target_ratio):
    current_ratio = w / h
    if current_ratio < target_ratio:
        new_w = (int)(target_ratio * h)
        new_h = h
    else:
        new_w = w
        new_h = (int)(w / target_ratio)

    # Center the new bounding box
    new_x = x + w // 2 - new_w // 2
    new_y = y + h // 2 - new_h // 2

    new_x = max(new_x,0)
    new_y = max(new_y,0)
    return new_x, new_y, new_w, new_h

def get_bounding_box(frame, model):
    results = model(frame)
    detections = results.xyxy[0]
    
    # If no objects detected, return empty list
    if detections.shape[0] == 0:
        return []

    count = 0
    bounding_boxes = []
    for detection in detections:
        x_min, y_min, x_max, y_max = map(int, detection[:4])
        width = x_max - x_min
        height = y_max - y_min
        bounding_boxes.append((x_min, y_min, width, height))
        count = count + 1
        if count == BOUNDING_BOX_SELECT_NUM:
            break

    return bounding_boxes

def track_and_crop(video_path, output_path, bbox, W, H):
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    print("output new video: ", output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (W, H))

    # processed_frames = []
    video_tensor = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update the tracker
        success, bbox = tracker.update(frame)
        if success:
            # Extract bounding box coordinates
            x, y, w, h = map(int, bbox)
            new_x, new_y, new_w, new_h = compute_new_bounding_box(x, y, w, h, W / H)

            # print(new_x,new_y,new_w,new_h)
            cropped_frame = frame[new_y:new_y + new_h, new_x:new_x + new_w]
            resized_frame = cv2.resize(cropped_frame, (W, H), interpolation=cv2.INTER_LINEAR)
            out.write(resized_frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
            # processed_frames.append(preprocess(frame))
            frame_tensor = (torch.from_numpy(resized_frame).permute(2, 0, 1).float()).to(config.device)  # .float() for tensor type consistency
            frame_tensor = F.resize(frame_tensor, (224, 224))  # Resize directly on tensor
            frame_tensor = frame_tensor.float() / 255.0  # Convert to float and scale to [0, 1]
            frame_tensor = F.normalize(frame_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize directly on tensor
            if video_tensor is None:
                video_tensor = frame_tensor.unsqueeze(0).to(config.device)  # Initialize tensor
            else:
                video_tensor = torch.cat((video_tensor, frame_tensor.unsqueeze(0)), dim=0)
    

    cap.release()
    out.release()

    # video_tensor = torch.stack(processed_frames) # Shape: [T, C, H, W]
    video_tensor = video_tensor.permute(1, 0, 2, 3) # Shape: [C, T, H, W]
    return video_tensor

def process_video(video_path, output_folder, id, model_object_detection, model_video_classification, W, H):
    print(f"process file: {video_path} {output_folder}")
    try: 
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        print("Error creating folder:", e)
    cap = cv2.VideoCapture(video_path)
    # Read video file
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return
    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    bounding_boxes = get_bounding_box(frame, model_object_detection)

    cap.release()

    scores = []
    for bbox in bounding_boxes:
        video_tensor = track_and_crop(video_path, os.path.join(output_folder, f"{id}.{config.EXT}"), bbox, W, H)
        score, label = scoring.rank_video(video_tensor.unsqueeze(0), model_video_classification)
        print(f"score is: {score}, label is {label}")
        scores.append((score,video_length))
        id = id + 1

    return scores