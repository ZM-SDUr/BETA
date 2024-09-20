import os
from natsort import natsorted
import shutil


def move_first_n_files(src_dir, dest_dir, n=300):
    # 确保目标文件夹存在
    os.makedirs(dest_dir, exist_ok=True)

    # 获取源文件夹中的所有文件，并按文件名排序
    all_files = natsorted([f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))])

    # 获取前n个文件
    files_to_move = all_files[:n]

    # 打印要移动的文件名称
    print("Files to move:")
    for filename in files_to_move:
        print(filename)

    # 移动文件
    for filename in files_to_move:
        src_file = os.path.join(src_dir, filename)
        dest_file = os.path.join(dest_dir, filename)
        shutil.move(src_file, dest_file)

    print(f'Moved {len(files_to_move)} files from {src_dir} to {dest_dir}.')


# 示例使用
src_dir = './L1-5train/'
dest_dir = './L1-5test/'

move_first_n_files(src_dir, dest_dir, n=300)
