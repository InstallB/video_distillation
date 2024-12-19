import torch

# N = 2 # each distilled video will contain N*N cropped videos
# DISTILL_NUMBERS = 16 # you will get floor(DISTILL_NUMBERS/(N*N)) videos in total
# DISTILL_NUMBERS = (DISTILL_NUMBERS // (N * N)) * (N * N)
# DISTILLED_FOLDER = "/userhome/jiangruohong/cv/UCF-101-Distilled/4_merged_videos_2x2"

# N = 2
# DISTILL_NUMBERS = 32
# DISTILL_NUMBERS = (DISTILL_NUMBERS // (N * N)) * (N * N)
# DISTILLED_FOLDER = "/userhome/jiangruohong/cv/UCF-101-Distilled/8_merged_videos_2x2"

N = 2
DISTILL_NUMBERS = 64
DISTILL_NUMBERS = (DISTILL_NUMBERS // (N * N)) * (N * N)
DISTILLED_FOLDER = "/userhome/jiangruohong/cv/UCF-101-Distilled/16_merged_videos_2x2"

# N = 1
# DISTILL_NUMBERS = 32
# DISTILL_NUMBERS = (DISTILL_NUMBERS // (N * N)) * (N * N)
# DISTILLED_FOLDER = "/userhome/jiangruohong/cv/UCF-101-Distilled/32_merged_videos_1x1"

# N = 3
# DISTILL_NUMBERS = 36
# DISTILL_NUMBERS = (DISTILL_NUMBERS // (N * N)) * (N * N)
# DISTILLED_FOLDER = "/userhome/jiangruohong/cv/UCF-101-Distilled/4_merged_videos_3x3"

train_percentage = 0.6

splits_file = "/userhome/jiangruohong/cv/videos2/ucf101_splits1.csv"

EXT = "avi" # extension of input videos
OUTPUT_EXT = "avi" # extension of output videos, avi or mp4
TARGET_FPS = 25 # recommend to be the same FPS as input videos

target_width = 320
target_height = 240 # the size of videos you *output*

DATASET_FOLDER = "/userhome/jiangruohong/cv/UCF-101/"
CROPPED_FOLDER = "/userhome/jiangruohong/cv/UCF-101-Distilled/cropped/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")