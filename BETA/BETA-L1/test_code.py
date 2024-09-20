import os
import shutil


def delete_files_in_directory(directory):

    if not os.path.exists(directory):
        print("Directory does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


delete_files_in_directory('./OPT_and_Model_con/test_log_file/')
