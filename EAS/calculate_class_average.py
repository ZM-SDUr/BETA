import os


def read_and_calculate_average(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        values = [float(line.strip().split()[1]) for line in lines if len(line.strip().split()) > 1]
        if values:
            return sum(values) / len(values)
        else:
            return 0

def calculate_total_average(directory_path):
    total_sum = 0
    total_count = 0

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            avg_value = read_and_calculate_average(file_path)
            total_sum += avg_value
            total_count += 1

    if total_count > 0:
        return total_sum / total_count
    else:
        return 0


# Usage example
directory_path = './train3/class5'
total_average = calculate_total_average(directory_path)
print(f"The total average bandwidth in {directory_path} is: {total_average}")