import os
import sys
import shutil
import csv
from tqdm import tqdm

def copy_and_rename_test_folders(source_folder, destination_folder, csv_file_path):
    """
    根据CSV文件中标记为'test'的行，将对应的文件夹从源文件夹复制到目标文件夹。
    如果源文件夹名称大小写不匹配，则重命名源文件夹以匹配CSV中的名称。

    参数：
    - source_folder: 包含所有子文件夹的源文件夹路径（例如 /jpeg112）。
    - destination_folder: 目标文件夹路径，复制后的文件夹将放在这里。
    - csv_file_path: 包含folder_name,label,split的CSV文件路径。
    """
    # 确保目标文件夹存在
    os.makedirs(destination_folder, exist_ok=True)
    
    # 读取CSV文件并获取'split'为'test'的folder_names
    test_folders = []
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'].strip().lower() == 'test':
                test_folders.append(row['folder_name'].strip())
    
    print(f"找到 {len(test_folders)} 个 'test' 分割的文件夹。")
    
    # 获取源文件夹中的所有子文件夹，并创建一个不区分大小写的映射
    try:
        source_subfolders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]
    except FileNotFoundError:
        print(f"源文件夹不存在: {source_folder}")
        sys.exit(1)
    
    lower_mapping = {f.lower(): f for f in source_subfolders}
    
    # 使用 tqdm 显示处理进度
    pbar = tqdm(total=len(test_folders), desc="处理文件夹")
    
    # 遍历并处理文件夹
    for folder_name in test_folders:
        src_path = os.path.join(source_folder, folder_name)
        dest_path = os.path.join(destination_folder, folder_name)
        
        if os.path.exists(src_path):
            # 如果源路径存在，直接复制
            try:
                shutil.copytree(src_path, dest_path)
                # print(f"复制文件夹: {folder_name}")  # 可以注释掉以减少输出
            except FileExistsError:
                print(f"目标文件夹已存在，跳过复制: {folder_name}")
            except Exception as e:
                print(f"复制文件夹时出错 {folder_name}: {e}")
        else:
            # 如果源路径不存在，尝试不区分大小写地查找
            folder_name_lower = folder_name.lower()
            if folder_name_lower in lower_mapping:
                actual_folder = lower_mapping[folder_name_lower]
                actual_src_path = os.path.join(source_folder, actual_folder)
                new_src_path = src_path  # 正确的源路径
                try:
                    # 重命名源文件夹为正确的名称
                    os.rename(actual_src_path, new_src_path)
                    print(f"已重命名源文件夹 '{actual_folder}' 为 '{folder_name}'")
                    
                    # 复制到目标文件夹
                    shutil.copytree(new_src_path, dest_path)
                    # print(f"复制文件夹: {folder_name}")  # 可以注释掉以减少输出
                except FileExistsError:
                    print(f"目标文件夹已存在，跳过复制: {folder_name}")
                except Exception as e:
                    print(f"重命名或复制文件夹时出错 {folder_name} (实际文件夹名: {actual_folder}): {e}")
            else:
                print(f"源文件夹不存在: {src_path}，且未找到匹配的文件夹（不区分大小写）。")
        
        pbar.update(1)
    
    pbar.close()
    print("所有 'test' 文件夹已处理完成。")

def main():
    """
    使用方法：
    python script.py source_folder destination_folder csv_file_path

    示例：
    python script.py /mnt/nas-new/home/yangnianzu/cyf/video_distillation/distill_utils/data/UCF101/jpegs_112 /mnt/nas-new/home/yangnianzu/cyf/video_distillation/distill_utils/data/UCF101/test_jpeg /mnt/nas-new/home/yangnianzu/cyf/video_distillation/distill_utils/data/UCF101/labels.csv
    """
    if len(sys.argv) != 4:
        print("用法: python script.py source_folder destination_folder csv_file_path")
        sys.exit(1)
    
    source_folder = sys.argv[1]
    destination_folder = sys.argv[2]
    csv_file_path = sys.argv[3]
    
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"源文件夹不存在: {source_folder}")
        sys.exit(1)
    
    # 检查CSV文件是否存在
    if not os.path.isfile(csv_file_path):
        print(f"CSV文件不存在: {csv_file_path}")
        sys.exit(1)
    
    # 执行复制和重命名操作
    copy_and_rename_test_folders(source_folder, destination_folder, csv_file_path)

if __name__ == "__main__":
    main()
