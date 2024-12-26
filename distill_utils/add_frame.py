import os
import shutil
import pandas as pd

def pad_frames_to_20_with_six_digits(root_dir, csv_path):
    """
    确保训练集视频文件夹中的帧数至少为20帧。
    如果帧数不足20帧，则复制最后一帧并重命名以达到20帧，文件名格式为 frameXXXXXX.jpg。

    参数:
    - root_dir: 包含所有视频文件夹的根目录路径。
    - csv_path: 包含视频元数据的 CSV 文件路径。
    """
    # 读取 CSV 文件
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"读取 CSV 文件时出错: {e}")
        return

    # 过滤出 split 为 'train' 的视频
    train_videos = df[df['split'] == 'train']

    for index, row in train_videos.iterrows():
        folder_name = row['folder_name']
        label = row['label']
        video_dir = os.path.join(root_dir, folder_name)

        if not os.path.isdir(video_dir):
            print(f"警告: 目录 {video_dir} 不存在，跳过。")
            continue

        # 列出视频目录下所有 jpg 文件
        frame_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.jpg')]
        
        if not frame_files:
            print(f"警告: 目录 {video_dir} 中没有找到帧文件，跳过。")
            continue

        # 过滤出符合 frameXXXXXX.jpg 格式的文件
        frame_files = [f for f in frame_files if f.startswith('frame') and f[5:11].isdigit()]
        frame_files.sort()

        num_frames = len(frame_files)

        if num_frames >= 20:
            print(f"{folder_name}: 已有 {num_frames} 帧，无需填充。")
            continue

        # 计算需要添加的帧数
        frames_needed = 20 - num_frames
        last_frame = frame_files[-1]
        last_frame_path = os.path.join(video_dir, last_frame)

        # 提取最后一帧的数字部分
        frame_num_str = last_frame[5:11]  # 假设格式为 frameXXXXXX.jpg
        frame_ext = os.path.splitext(last_frame)[1]

        # 确保最后一帧的数字部分为六位
        if len(frame_num_str) != 6 or not frame_num_str.isdigit():
            print(f"错误: 最后一帧文件名 {last_frame} 不符合 'frameXXXXXX.jpg' 格式，跳过 {folder_name} 的填充。")
            continue

        last_num = int(frame_num_str)

        for i in range(1, frames_needed + 1):
            new_num = last_num + i
            new_frame_name = f"frame{new_num:06d}{frame_ext}"
            new_frame_path = os.path.join(video_dir, new_frame_name)
            
            # 复制最后一帧到新的帧文件
            try:
                shutil.copy(last_frame_path, new_frame_path)
                print(f"在 {folder_name} 中添加帧 {new_frame_name}")
            except Exception as e:
                print(f"复制帧到 {new_frame_path} 时出错: {e}")

    print("帧填充过程完成。")

if __name__ == "__main__":
    # 示例用法
    # 定义包含所有视频文件夹的根目录路径
    root_directory = "/mnt/nas-new/home/yangnianzu/cyf/video_distillation/video_distillation/UCF101/2m_3x3_new/distill_rded_jpeg"  # 请替换为您的实际路径

    # 定义 CSV 文件的路径
    csv_file_path = "/mnt/nas-new/home/yangnianzu/cyf/video_distillation/video_distillation/UCF101/2m_3x3_new/rded_ucf101_splits.csv"  # 请替换为您的实际 CSV 文件路径

    pad_frames_to_20_with_six_digits(root_directory, csv_file_path)
