import os

def process_trace_files(folder_path, output_folder, x, xlen):
    for i, file_name in enumerate(sorted(os.listdir(folder_path))[x:][:xlen]):
        input_file_path = os.path.join(folder_path, file_name)
        output_file_path = os.path.join(output_folder, f"processed_{file_name}")
        output_lines = []
        with open(input_file_path, 'r') as f:
            lines = f.readlines()
            for j, line in enumerate(lines[300:], start=0):
                parts = line.split()
                output_lines.append(f"{j}.0 {parts[1]}\n")

        with open(output_file_path, 'w') as f:
            f.writelines(output_lines)
# 使用示例
process_trace_files("./trace_for_test", "./trace_for_300_onwards", 0, 1000)  # 从第0个文件开始处理，处理1个文件
