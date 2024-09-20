import matplotlib.pyplot as plt
import numpy as np
import os

Avg_reward_every_2000_list = []
Log_file_list   = os.listdir('log_file_4G/')
coun = 0
sum3reward =0
for i in range(1,124):#len(Log_file_list)+1
    reward_list = []
    action_list = []
    rebuf_list = []
    throughput_list = []
    critic_list = []
    time_list = []
    buffer_size_list = []
    with open("./log_file_4G/" + "recent_log" + str(i), "r") as f:
        for line in f:
            parse = line.split()
            action_list.append(float(parse[0]))
            buffer_size_list.append(float((parse[1])))
            reward_list.append(float(parse[2]))
            throughput_list.append(float(parse[3]))

    Avg_reward = np.mean(reward_list) * 48

    Avg_reward_every_2000_list.append(Avg_reward)

plt.ylabel("Avg_reward")
plt.plot(Avg_reward_every_2000_list)
for i in range(len(Avg_reward_every_2000_list)):
    print(i,Avg_reward_every_2000_list[i])
plt.xlabel("log_files")
plt.grid(True)
plt.show()
'''
plt.subplot(3, 1, 1)
plt.ylabel("Bitrate")
plt.plot(action_list)
plt.title("Avg_reward=" + str(Avg_reward))
plt.subplot(3, 1, 2)
plt.ylabel("buf_size")
plt.plot(buffer_size_list)

plt.subplot(3, 1, 3)
plt.ylabel("Throughput")
plt.plot(throughput_list)
plt.xlabel("Time(s)")

plt.show()

'''
