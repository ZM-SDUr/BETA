import os
import shutil


def delete_files_in_directory(directory):
    # 检查目录是否存在
    if not os.path.exists(directory):
        print("Directory does not exist.")
        return

    # 遍历目录中的所有文件和子目录
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除目录及其所有内容
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


# 使用函数
delete_files_in_directory('./OPT_and_Model_con/test_log_file/')
