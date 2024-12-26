import cv2
import os
import sys
from tqdm import tqdm

def resize_and_crop_images(output_folder, target_size=(240, 320), crop_size=(112, 112)):
    """
    遍历目标文件夹下的所有图片，先resize然后crop到目标大小112x112。
    
    参数：
    - output_folder: 保存帧图像的目标文件夹。
    - target_size: resize的目标大小，默认(160, 120)。
    - crop_size: crop的目标大小，默认(112, 112)。
    """
    print(f"Resizing and cropping images in {output_folder}")
    files = os.listdir(output_folder)
    
    # 使用正则表达式匹配重命名后的文件名
    # pattern = re.compile(r"frame(\d{6})\.(jpg|png)$", re.IGNORECASE)
    
    # matched_files = [file for file in files if pattern.match(file)]
    
    for file in tqdm(files, desc="Processing images"):
        img_path = os.path.join(output_folder, file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        # Resize图片
        img_resized = cv2.resize(img, target_size)
        
        # Crop图片
        h, w, _ = img_resized.shape
        top = (h - crop_size[0]) // 2
        left = (w - crop_size[1]) // 2
        img_cropped = img_resized[top:top+crop_size[0], left:left+crop_size[1]]
        
        # 保存处理后的图片
        cv2.imwrite(img_path, img_cropped)  # 覆盖原图，或者可以另存为新文件
        
    # print("Resizing and cropping completed.")

def process_video_frames(output_folder, target_size=(160, 120), crop_size=(112, 112)):
    """
    处理视频帧，包括重命名和裁剪调整大小。
    
    参数：
    - output_folder: 保存帧图像的目标文件夹。
    - target_size: resize的目标大小，默认(160, 120)。
    - crop_size: crop的目标大小，默认(112, 112)。
    """
    # rename_frames(output_folder)
    resize_and_crop_images(output_folder, target_size, crop_size)

def extract_frames(video_path, output_folder):
    """
    从视频中提取每一帧并保存为 .jpg 文件。

    参数：
    - video_path: 视频文件的路径。
    - output_folder: 保存帧图像的目标文件夹。
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # if not cap.isOpened():
    #     print(f"无法打开视频文件: {video_path}")
    #     return
    
    # # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # # 使用 tqdm 显示进度条
    pbar = tqdm(total=total_frames, desc=f"处理 {os.path.basename(video_path)}")
    files = os.listdir(output_folder)
    frame_num = 1
    while True:
    # for file in files:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 构建帧文件名，例如 frame1.jpg
        # frame_filename = f"frame{frame_num}.jpg"
        newframe_filename = f"frame{frame_num:06d}.jpg"
        # frame_path = os.path.join(output_folder, frame_filename)
        new_path = os.path.join(output_folder, newframe_filename)
        # 保存帧图像为 JPEG 格式
        # os.rename(frame_path, new_path)
        # print(f"{f} -> {new_name}")
        cv2.imwrite(new_path, frame)
        
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
    
    # 获取所有视频文件，并按名称排序
    video_files = sorted([f for f in os.listdir(source_dir) 
                          if os.path.isfile(os.path.join(source_dir, f)) and 
                          os.path.splitext(f)[1].lower() in video_extensions])
    
    # 如果指定了起始视频，则找到它的位置
    if start_video:
        if start_video not in video_files:
            print(f"起始视频 '{start_video}' 未在源文件夹中找到。")
            sys.exit(1)
        start_index = video_files.index(start_video)
    else:
        start_index = 0
    
    # 遍历视频文件，从指定的视频开始
    for filename in video_files[start_index:]:
        file_path = os.path.join(source_dir, filename)
        
        # 获取视频名（不包含扩展名）
        video_name = os.path.splitext(filename)[0]
        
        # 为每个视频创建一个对应的文件夹
        video_output_folder = os.path.join(destination_dir, video_name)
        
        # 如果输出文件夹已存在，跳过该视频
        # if os.path.exists(video_output_folder):
        #     print(f"输出文件夹已存在，跳过视频: {filename}")
        #     continue
        process_video_frames(video_output_folder)
        # os.makedirs(video_output_folder, exist_ok=True)
        
        # # 提取视频帧
        extract_frames(file_path, video_output_folder)
    # for subdir in os.listdir(destination_dir):
    #     video_output_folder = os.path.join(destination_dir, subdir)
    #     if os.path.isdir(video_output_folder):
    #         process_video_frames(video_output_folder)

    print("All videos processed and images resized and cropped!")

if __name__ == "__main__":
    """
    使用方法：
    python extract_frames.py 源文件夹 目标文件夹 [起始视频文件名]

    示例：
    python extract_frames.py /path/to/videos /path/to/output video3.avi
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
