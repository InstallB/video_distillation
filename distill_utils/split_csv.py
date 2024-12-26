import os
import sys
import csv
import re
import argparse
from tqdm import tqdm

def extract_label(folder_name, split):
    """
    根据文件夹名称和split类型提取标签。

    参数：
    - folder_name: 文件夹名称。
    - split: 'test' 或 'train'。

    返回：
    - label: 提取的标签。
    """
    if split == 'test':
        # 对于test文件夹，去掉v_前缀，然后提取第一个下划线前的部分
        match = re.match(r'^v_([^_]+)', folder_name)
        if match:
            return match.group(1)
    elif split == 'train':
        # 对于train文件夹，提取名称中首个数字或下划线之前的部分
        match = re.match(r'^([A-Za-z]+)', folder_name)
        if match:
            return match.group(1)
    return "Unknown"

def generate_split_csv(source_folder, output_csv):
    """
    生成 split.csv 文件，排除在 train 中不存在的 test 标签。

    参数：
    - source_folder: 源文件夹路径，包含所有子文件夹。
    - output_csv: 生成的 CSV 文件路径。
    """
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"源文件夹不存在: {source_folder}")
        sys.exit(1)
    
    # 获取所有子文件夹
    try:
        subfolders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]
    except Exception as e:
        print(f"无法读取源文件夹内容: {e}")
        sys.exit(1)
    
    # 分离 train 和 test 文件夹
    train_folders = []
    test_folders = []
    for folder in subfolders:
        if folder.startswith('v_'):
            test_folders.append(folder)
        else:
            train_folders.append(folder)
    
    # 提取 train 标签
    train_labels = set()
    for folder in train_folders:
        label = extract_label(folder, 'train')
        if label != "Unknown":
            train_labels.add(label)
    
    print(f"收集到 {len(train_labels)} 个 train 标签。")
    
    # 准备CSV数据
    csv_data = []
    csv_data.append(['folder_name', 'label', 'split'])
    
    print(f"正在处理 {len(train_folders)} 个 train 文件夹...")
    
    # 处理 train 文件夹
    for folder in tqdm(train_folders, desc="处理 train 文件夹"):
        label = extract_label(folder, 'train')
        csv_data.append([folder, label, 'train'])
    
    print(f"正在处理 {len(test_folders)} 个 test 文件夹（仅保留 train 中存在的标签）...")
    
    # 处理 test 文件夹，过滤掉 train 中不存在的标签
    for folder in tqdm(test_folders, desc="处理 test 文件夹"):
        label = extract_label(folder, 'test')
        if label in train_labels:
            csv_data.append([folder, label, 'test'])
        else:
            print(f"排除文件夹 '{folder}'，因为标签 '{label}' 不存在于 train 中。")
    
    # 写入CSV文件
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)
        print(f"成功生成 CSV 文件: {output_csv}")
    except Exception as e:
        print(f"无法写入CSV文件: {e}")
        sys.exit(1)

def main():
    """
    使用方法：
    python generate_split_csv.py --source_folder <源文件夹路径> --output_csv <输出CSV文件路径>

    示例：
    python generate_split_csv.py --source_folder /path/to/jpegs_112 --output_csv /path/to/split.csv
    """
    parser = argparse.ArgumentParser(description="生成 split.csv 文件，根据文件夹名称划分 train/test 并提取标签，排除 test 中不存在于 train 的标签。")
    parser.add_argument('--source_folder', type=str, required=True, help="源文件夹路径，包含所有子文件夹。")
    parser.add_argument('--output_csv', type=str, required=True, help="输出的 CSV 文件路径。")
    
    args = parser.parse_args()
    
    generate_split_csv(args.source_folder, args.output_csv)

if __name__ == "__main__":
    main()
