import os
import sys
import pandas as pd

def generate_splits(frame_data_dir, ucf_splits_csv, output_csv):
    """
    根据 ucf101_splits.csv 文件中的标签划分，为视频帧数据生成 splits.csv。

    参数：
    - frame_data_dir: 提取帧数据的目标文件夹路径。
    - ucf_splits_csv: ucf101_splits.csv 文件的路径。
    - output_csv: 生成的 splits.csv 文件的路径。
    """
    # 读取 ucf101_splits.csv
    try:
        splits_df = pd.read_csv(ucf_splits_csv)
    except Exception as e:
        print(f"无法读取文件 {ucf_splits_csv}: {e}")
        sys.exit(1)
    
    # 检查必要的列是否存在
    required_columns = {'folder_name', 'label', 'split'}
    if not required_columns.issubset(splits_df.columns):
        print(f"ucf101_splits.csv 文件缺少必要的列: {required_columns}")
        sys.exit(1)
    
    # 获取唯一的标签及其split
    label_split_map = splits_df.groupby('label')['split'].first().to_dict()
    
    # 获取所有帧数据目录
    frame_dirs = [d for d in os.listdir(frame_data_dir) 
                 if os.path.isdir(os.path.join(frame_data_dir, d))]
    
    if not frame_dirs:
        print(f"在目标文件夹中未找到任何子文件夹: {frame_data_dir}")
        sys.exit(1)
    
    # 获取所有标签并按长度降序排序，以避免标签前缀冲突
    labels = list(label_split_map.keys())
    labels_sorted = sorted(labels, key=lambda x: len(x), reverse=True)
    
    # 生成 splits 信息
    splits_list = []
    for dir_name in frame_dirs:
        label_found = False
        for label in labels_sorted:
            if dir_name.startswith(label):
                split = label_split_map[label]
                splits_list.append({'folder_name': dir_name, 'label': label, 'split': split})
                label_found = True
                break
        if not label_found:
            print(f"警告: 无法找到匹配的标签 for directory: {dir_name}")
            splits_list.append({'folder_name': dir_name, 'label': 'unknown', 'split': 'unknown'})
    
    # 创建 DataFrame
    splits_output_df = pd.DataFrame(splits_list)
    
    # 保存 splits.csv
    try:
        splits_output_df.to_csv(output_csv, index=False)
        print(f"成功生成 splits.csv 文件: {output_csv}")
    except Exception as e:
        print(f"无法保存文件 {output_csv}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    """
    使用方法：
    python generate_splits.py 提取帧数据的目标文件夹 ucf101_splits.csv 输出的 splits.csv
    
    示例：
    python generate_splits.py /mnt/nas-new/home/yangnianzu/cyf/video_distillation/distill_rded_jpeg /path/to/ucf101_splits.csv /path/to/splits.csv
    """
    if len(sys.argv) != 4:
        print("用法: python generate_splits.py 提取帧数据的目标文件夹 ucf101_splits.csv 输出的 splits.csv")
        sys.exit(1)
    
    frame_data_directory = sys.argv[1]
    ucf_splits_csv_path = sys.argv[2]
    splits_csv_output_path = sys.argv[3]
    
    # 检查路径是否存在
    if not os.path.exists(frame_data_directory):
        print(f"提取帧数据的目标文件夹不存在: {frame_data_directory}")
        sys.exit(1)
    
    if not os.path.exists(ucf_splits_csv_path):
        print(f"ucf101_splits.csv 文件不存在: {ucf_splits_csv_path}")
        sys.exit(1)
    
    generate_splits(frame_data_directory, ucf_splits_csv_path, splits_csv_output_path)
