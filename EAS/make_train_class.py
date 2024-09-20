import os
import shutil

def read_and_calculate_average(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        values = [float(line.strip().split()[1]) for line in lines if len(line.strip().split()) > 1]
        if values:
            return sum(values) / len(values)
        else:
            return 0

def divide_files_into_parts(files, num_parts):
    total_files = len(files)
    part_size = total_files // num_parts
    remainder = total_files % num_parts

    parts = []
    start = 0
    for i in range(num_parts):
        end = start + part_size + (1 if i < remainder else 0)
        parts.append(files[start:end])
        start = end
    return parts


def main(source_directory, dest_directory, num_parts=2):
    file_averages = []

    for filename in os.listdir(source_directory):
        file_path = os.path.join(source_directory, filename)
        if os.path.isfile(file_path):
            avg_value = read_and_calculate_average(file_path)
            file_averages.append((filename, avg_value))

    file_averages.sort(key=lambda x: x[1])

    sorted_files = [filename for filename, avg_value in file_averages]

    parts = divide_files_into_parts(sorted_files, num_parts)

    for i in range(num_parts):
        os.makedirs(os.path.join(dest_directory, f'class{i + 1}'), exist_ok=True)

    for i, part in enumerate(parts):
        for filename in part:
            src_path = os.path.join(source_directory, filename)
            dest_path = os.path.join(dest_directory, f'class{i + 1}', filename)
            shutil.copy(src_path, dest_path)

    return parts


# Usage example
source_directory = '../Pensieve/L1-5test'
dest_directory = './test3'
parts = main(source_directory, dest_directory)

for i, part in enumerate(parts):
    print(f"Part {i + 1}:")
    for filename in part:
        print(f"  {filename}")