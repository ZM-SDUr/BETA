import subprocess

# 定义要启动的 Python 程序及其参数
interval = 25
start = 0
end = 825

# 定义要启动的 Python 程序及其参数
scripts = [('test_actor.py', [i, i + interval]) for i in range(start, end, interval)]
model_numbers = range(1, 3)  # 这里假设你有多个模型编号 #best 21

# 指定 Python 解释器路径
python_interpreter = '/home/ubuntu/miniconda3/envs/tf-wzm/bin/python3'

# 启动每个模型的所有程序并传递参数
for model_num in model_numbers:
    model_processes = []
    for script, args in scripts:
        # 将参数转换为字符串
        str_args = list(map(str, args))
        process = subprocess.Popen(
            [python_interpreter, script] + str_args + [str(model_num)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        model_processes.append(process)

    # 等待当前模型的所有子进程完成并收集输出
    model_rewards = []
    for process in model_processes:
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            try:
                # 假设子进程的输出是 "Model <model_num> Average Reward: <average_reward>"
                output = stdout.decode().strip().split()[-1]
                result = float(output)
                model_rewards.append(result)
            except ValueError as e:
                print(f"Error parsing output: {e}")
                print(f"Output was: {stdout.decode().strip()}")
        else:
            print(f"Process failed with error: {stderr.decode().strip()}")

    # 计算当前模型的平均值并打印
    if model_rewards:
        average_result = sum(model_rewards) / len(model_rewards)
        print(f"Model {model_num} Average result: {average_result}")
    else:
        print(f"No successful results collected for Model {model_num}.")
