import os

def add_avi_extension(directory):
    """
    为指定目录下所有没有扩展名的文件添加 .avi 扩展名。
    
    参数：
    - directory: 目标目录路径。
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件是否有扩展名
            if '.' not in file:
                old_path = os.path.join(root, file)
                new_path = old_path + '.avi'
                try:
                    os.rename(old_path, new_path)
                    print(f"已重命名: {old_path} -> {new_path}")
                except Exception as e:
                    print(f"无法重命名 {old_path}: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python add_avi_extension.py 目标目录路径")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    
    if not os.path.isdir(target_directory):
        print(f"目标目录不存在: {target_directory}")
        sys.exit(1)
    
    add_avi_extension(target_directory)
