import os
import shutil
import random

# 源目录和目标目录
src_dir = './train/'
dest_dir = './train2/mix_file/'

# 确保目标目录存在
os.makedirs(dest_dir, exist_ok=True)

# 每个子目录复制的文件数量
num_files_to_copy = 200  # 根据需要调整这个数量

# 遍历每个子目录
for i in range(1, 3):
    class_dir = os.path.join(src_dir, f'class{i}')
    all_files = os.listdir(class_dir)
    selected_files = random.sample(all_files, min(num_files_to_copy, len(all_files)))

    # 复制文件到新目录
    for file in selected_files:
        src_file_path = os.path.join(class_dir, file)
        dest_file_path = os.path.join(dest_dir, file)
        shutil.copy2(src_file_path, dest_file_path)

print("文件复制完成。")