import subprocess

interval = 25
start = 0
end = 933

scripts = [('test_actor.py', [i, i + interval]) for i in range(start, end, interval)]
model_numbers = range(73, 74)  # 这里假设你有多个模型编号 (Best 15)

python_interpreter = '/home/ubuntu/miniconda3/envs/tf-wzm/bin/python3'

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

    model_rewards = []
    for process in model_processes:
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            try:
                output = stdout.decode().strip().split()[-1]
                result = float(output)
                model_rewards.append(result)
            except ValueError as e:
                print(f"Error parsing output: {e}")
                print(f"Output was: {stdout.decode().strip()}")
        else:
            print(f"Process failed with error: {stderr.decode().strip()}")

    if model_rewards:
        average_result = sum(model_rewards) / len(model_rewards)
        print(f"Model {model_num} Average result: {average_result}")
    else:
        print(f"No successful results collected for Model {model_num}.")
