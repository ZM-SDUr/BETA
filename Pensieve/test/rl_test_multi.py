import subprocess

# 定义要启动的 Python 程序及其参数
interval = 100
start = 0
end = 1000

# 定义要启动的 Python 程序及其参数
scripts = [('rl_test.py', [i, i + interval]) for i in range(start, end, interval)]

processes = []
#/home/ubuntu/miniconda3/bin/conda run -n tf-wzm --no-capture-output python
# 指定 Python 解释器路径
python_interpreter = '/home/ubuntu/miniconda3/envs/tf-wzm/bin/python3'

# 启动所有程序并传递参数
for script, args in scripts:
    # 将参数转换为字符串
    str_args = list(map(str, args))
    process = subprocess.Popen([python_interpreter, script] + str_args)
    processes.append(process)

# 等待所有程序完成
for process in processes:
    process.wait()
