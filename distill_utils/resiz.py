import os
import cv2
from tqdm import tqdm

def resize_and_crop_images(output_folder, target_size=(240, 320), crop_size=(112, 112)):
    """
    遍历目标文件夹下的所有图片，先resize然后crop到目标大小112x112。
    
    参数：
    - output_folder: 保存帧图像的目标文件夹。
    - target_size: resize的目标大小，默认(240, 320)。
    - crop_size: crop的目标大小，默认(112, 112)。
    """
    print(f"Resizing and cropping images in {output_folder}")
    files = os.listdir(output_folder)
    
    for file in tqdm(files, desc="Processing images"):
        img_path = os.path.join(output_folder, file)
        
        # 检查是否为图片文件
        if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Skipping non-image file: {img_path}")
            continue
        
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        # Resize图片
        img_resized = cv2.resize(img, target_size)
        
        # Crop图片
        h, w = img_resized.shape[:2]
        top = (h - crop_size[0]) // 2
        left = (w - crop_size[1]) // 2
        img_cropped = img_resized[top:top+crop_size[0], left:left+crop_size[1]]
        
        # 保存处理后的图片
        cv2.imwrite(img_path, img_cropped)  # 覆盖原图，或者可以另存为新文件
    print(f"Finished processing images in {output_folder}\n")

def process_all_subfolders(main_folder, target_size=(240, 320), crop_size=(112, 112)):
    """
    遍历主文件夹下的所有子文件夹，检查每个子文件夹中的图片大小。
    如果图片的大小不是crop_size，则调用resize_and_crop_images进行处理。
    
    参数：
    - main_folder: 主文件夹路径，包含多个子文件夹。
    - target_size: resize的目标大小，默认(240, 320)。
    - crop_size: crop的目标大小，默认(112, 112)。
    """
    # 遍历主文件夹下的所有子文件夹
    for root, dirs, files in os.walk(main_folder):
        # Skip the main_folder itself
        if root == main_folder:
            continue
        
        # Flag to determine if resizing and cropping is needed
        needs_processing = False
        
        # Check sizes of images in the current subfolder
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue  # Skip non-image files
            
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            if img.shape[0] != crop_size[0] or img.shape[1] != crop_size[1]:
                needs_processing = True
                break  # No need to check further, process this folder
        
        if needs_processing:
            resize_and_crop_images(root, target_size, crop_size)
        else:
            print(f"Skipping {root}: All images are already {crop_size[0]}x{crop_size[1]}\n")

if __name__ == "__main__":
    # 设置主文件夹路径
    main_folder_path = "/mnt/nas-new/home/yangnianzu/cyf/video_distillation/video_distillation/UCF101/2m_2x2/distill_rded_jpeg"  # 替换为你的主文件夹路径
    
    # 调用函数处理所有子文件夹
    process_all_subfolders(main_folder_path)
