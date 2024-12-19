import os
from pathlib import Path
import torch
import process
import config
import cv2
import csv
import numpy as np

def get_subfolder_names(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")
    
    subfolders = [name for name in os.listdir(folder_path) 
                  if os.path.isdir(os.path.join(folder_path, name))]
    return subfolders

N = config.N
W = config.target_width // N
H = config.target_height // N

MERGE_GROUP_SIZE = N * N  # Number of videos to merge based on length

def combine_videos(video_paths, output_path):
    caps = [cv2.VideoCapture(video) for video in video_paths]
    lengths = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    min_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps)
    
    for cap, path in zip(caps, video_paths):
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {path}")
    
    fps = config.TARGET_FPS
    out_width = config.target_width
    out_height = config.target_height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    for frame_idx in range(min_frames):
        combined_frame = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        for i in range(N):
            for j in range(N):
                cap_idx = i * N + j
                if frame_idx > 0:
                    skip_frames = (frame_idx * lengths[cap_idx]) // min_frames - ((frame_idx - 1) * lengths[cap_idx]) // min_frames - 1
                    for _ in range(skip_frames):
                        ret, frame = caps[cap_idx].read()
                ret, frame = caps[cap_idx].read()
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
                if ret:
                    start_y = i * H
                    start_x = j * W
                    combined_frame[start_y:start_y + H, start_x:start_x + W] = frame
        
        out.write(combined_frame)
    
    for cap in caps:
        cap.release()
    out.release()

# model_object_detection = torch.hub.load('ultralytics/yolov5', "yolov5s", pretrained=True)
model_object_detection = torch.hub.load('/userhome/jiangruohong/cv/.cache/torch/hub/ultralytics_yolov5_master/', 'yolov5s', pretrained=True, source='local')
# model_video_classification = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
model_video_classification = torch.hub.load('/userhome/jiangruohong/cv/.cache/torch/hub/facebookresearch_pytorchvideo_main', 'i3d_r50', pretrained=True, source='local').to(config.device)
model_object_detection.eval()
model_video_classification.eval()

subfolders = get_subfolder_names(config.CROPPED_FOLDER)

for subfolder_name in subfolders:
    print(f"merge processing folder: {subfolder_name}")
    INPUT_FOLDER = os.path.join(config.CROPPED_FOLDER, subfolder_name)
    OUTPUT_FOLDER = os.path.join(config.DISTILLED_FOLDER, subfolder_name)
    try: 
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    except Exception as e:
        print("Error creating folder:", e)

    scores_list = []
    with open(os.path.join(INPUT_FOLDER, "scores.csv"), "r") as file:
        reader = csv.reader(file)
        scores_list = [(row[0], float(row[1]), float(row[2])) for row in reader] 

    count = (config.DISTILL_NUMBERS // (N * N)) * (N * N)
    if len(scores_list) < count:
        count = (len(scores_list) // (N * N)) * (N * N)
    best_list = sorted(scores_list, key=lambda x: x[1], reverse=True)[:count]
    best_list = sorted(best_list, key=lambda x: x[2])
    print(best_list)
    print(count,len(best_list))

    count = 0
    for i in range(0, len(best_list), MERGE_GROUP_SIZE):
        group = best_list[i:i + MERGE_GROUP_SIZE]
        video_paths = [os.path.join(INPUT_FOLDER, item[0]) for item in group]
        output_path = os.path.join(OUTPUT_FOLDER, f"{count}.{config.OUTPUT_EXT}")
        count = count + 1
        combine_videos(video_paths, output_path)