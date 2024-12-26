import os
import sys
import cv2
from tqdm import tqdm

def resize_to_target(image, target_size=(112, 112)):
    """
    直接缩放图像到目标尺寸，不保持宽高比。
    
    参数：
    - image: 输入图像（NumPy数组）。
    - target_size: 目标尺寸 (宽, 高)。
    
    返回：
    - resized_image: 缩放后的图像。
    """
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized_image

def extract_frames(video_path, output_folder, target_size=(112, 112)):
    """
    从视频中提取每一帧，直接缩放到目标大小112x112，并保存为 .jpg 文件。
    
    参数：
    - video_path: 视频文件的路径。
    - output_folder: 保存帧图像的目标文件夹。
    - target_size: 目标图像尺寸，默认(112, 112)。
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 使用 tqdm 显示进度条
    pbar = tqdm(total=total_frames, desc=f"处理 {os.path.basename(video_path)}")
    
    frame_num = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 直接缩放到目标尺寸
        resized_frame = resize_to_target(frame, target_size)
        
        # 构建帧文件名，例如 frame000001.jpg
        newframe_filename = f"frame{frame_num:06d}.jpg"
        new_path = os.path.join(output_folder, newframe_filename)
        
        # 保存帧图像为 JPEG 格式
        cv2.imwrite(new_path, resized_frame)
        
        frame_num += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    print(f"完成提取 {frame_num-1} 帧到文件夹: {output_folder}")

def main(source_dir, destination_dir, start_video=None):
    """
    从源文件夹中的所有视频文件提取帧，并保存到目标文件夹。
    
    参数：
    - source_dir: 包含视频文件的源文件夹。
    - destination_dir: 保存提取帧图像的目标文件夹。
    - start_video: 从哪个视频开始提取（可选）。如果未指定，则从第一个视频开始。
    """
    # 支持的常见视频文件扩展名
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    # 确保目标文件夹存在
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    # 遍历源目录中的所有子目录和文件
    video_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in video_extensions:
                full_path = os.path.join(root, file)
                video_files.append(full_path)
    
    # 按路径排序视频文件
    video_files = sorted(video_files)
    
    # 如果指定了起始视频，则找到其索引
    if start_video:
        # 如果 start_video 是相对路径或仅文件名，需要找到对应的全路径
        matched_videos = [vf for vf in video_files if os.path.basename(vf) == start_video]
        if not matched_videos:
            print(f"起始视频 '{start_video}' 未在源文件夹中找到。")
            sys.exit(1)
        start_index = video_files.index(matched_videos[0])
    else:
        start_index = 0
    
    # 遍历视频文件，从指定的视频开始
    for video_path in video_files[start_index:]:
        # 获取相对于源目录的路径
        relative_path = os.path.relpath(video_path, source_dir)
        # 分割路径以获取子目录和视频文件名
        parts = relative_path.split(os.sep)
        
        if len(parts) >= 2:
            # 假设视频文件在子目录中，组合子目录名和视频文件名（不含扩展名）
            subdir = parts[-2]
            video_name = os.path.splitext(parts[-1])[0]
            output_folder_name = f"{subdir}{video_name}"
        else:
            # 如果视频文件直接在源目录中
            video_name = os.path.splitext(parts[-1])[0]
            output_folder_name = video_name
        
        # 生成完整的目标文件夹路径
        output_folder = os.path.join(destination_dir, output_folder_name)
        
        # 创建目标文件夹（如果不存在）
        os.makedirs(output_folder, exist_ok=True)
        
        # 提取视频帧
        extract_frames(video_path, output_folder)
    
    print("所有视频已处理完毕，帧图像已保存！")

if __name__ == "__main__":
    """
    使用方法：
    python extract_frames.py 源文件夹 目标文件夹 [起始视频文件名]

    示例：
    python extract_frames.py /mnt/nas-new/home/yangnianzu/cyf/video_distillation/distill_rded /mnt/nas-new/home/yangnianzu/cyf/video_distillation/distill_rded_jpeg 0.avi
    """
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("用法: python extract_frames.py 源文件夹 目标文件夹 [起始视频文件名]")
        sys.exit(1)
    
    source_directory = sys.argv[1]
    destination_directory = sys.argv[2]
    
    start_video = sys.argv[3] if len(sys.argv) == 4 else None
    
    if not os.path.exists(source_directory):
        print(f"源文件夹不存在: {source_directory}")
        sys.exit(1)
    
    main(source_directory, destination_directory, start_video)
