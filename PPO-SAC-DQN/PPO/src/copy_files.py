import shutil
import os


def copy_files(src_dir1, src_dir2, dest_dir):
    # 创建目标文件夹，如果不存在
    os.makedirs(dest_dir, exist_ok=True)

    # 从第一个源文件夹复制文件到目标文件夹
    for filename in os.listdir(src_dir1):
        src_file = os.path.join(src_dir1, filename)
        dest_file = os.path.join(dest_dir, filename)
        if os.path.isfile(src_file):
            shutil.copy(src_file, dest_file)

    # 从第二个源文件夹复制文件到目标文件夹
    for filename in os.listdir(src_dir2):
        src_file = os.path.join(src_dir2, filename)
        dest_file = os.path.join(dest_dir, filename)
        if os.path.isfile(src_file):
            shutil.copy(src_file, dest_file)

# 示例使用
src_dir1 = '../../classified_trace/L5/'
src_dir2 = '../../classified_trace/L14/'
dest_dir = './L1-5train/'

copy_files(src_dir1, src_dir2, dest_dir)
