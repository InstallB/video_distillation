import os
from pathlib import Path
import torch
import process
import config
import csv
import multiprocessing
import random

def get_subfolder_names(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")
    
    subfolders = [name for name in os.listdir(folder_path) 
                  if os.path.isdir(os.path.join(folder_path, name))]
    return subfolders

N = config.N
W = config.target_width // N
H = config.target_height // N

MERGE_GROUP_SIZE = N * N

# model_object_detection = torch.hub.load('ultralytics/yolov5', "yolov5s", pretrained=True)
model_object_detection = torch.hub.load('/userhome/jiangruohong/cv/.cache/torch/hub/ultralytics_yolov5_master/', 'yolov5s', pretrained=True, source='local')
# model_video_classification = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
model_video_classification = torch.hub.load('/userhome/jiangruohong/cv/.cache/torch/hub/facebookresearch_pytorchvideo_main', 'i3d_r50', pretrained=True, source='local').to(config.device)
model_object_detection.eval()
model_video_classification.eval()

def process_subfolder(subfolder_name):
    print(f"processing folder: {subfolder_name}")
    input_folder = os.path.join(config.DATASET_FOLDER, subfolder_name)
    output_folder = os.path.join(config.CROPPED_FOLDER, subfolder_name)

    video_paths = []
    with open(config.splits_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['split'] == 'train' and row['label'] == subfolder_name:
                video_paths.append(os.path.join(input_folder, row['folder_name'] + f".{config.EXT}"))
    
    # video_paths = [str(path) for path in Path(input_folder).rglob(f"*.{config.EXT}")]
    print(video_paths)
    
    TOP_K = config.DISTILL_NUMBERS // MERGE_GROUP_SIZE
    # print(f"will generate {TOP_K} merged videos")

    # Process each video
    total_scores = []
    id = 0
    for video_path in video_paths:
        try:
            items = process.process_video(video_path, output_folder, id, model_object_detection, model_video_classification, W, H)
            for score, length in items:
                total_scores.append((f"{id}.{config.EXT}", score, length))
                id = id + 1
            with open(os.path.join(output_folder, "scores.csv"), "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(total_scores)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    subfolders = get_subfolder_names(config.DATASET_FOLDER)

    num_cpus = 8
    print(f"Using {num_cpus} CPUs")

    with multiprocessing.Pool(processes=num_cpus) as pool:
        pool.map(process_subfolder, subfolders)

# for subfolder_name in subfolders:
#     process_subfolder(subfolder_name)