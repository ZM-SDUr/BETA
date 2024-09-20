import os

def delete_all_files_in_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

# 使用示例
#folder_to_delete = "./results5/"
#delete_all_files_in_folder(folder_to_delete)
