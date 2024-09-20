import os
import numpy as np
import matplotlib.pyplot as plt



MPCresult_name = './MPC_result-HD/mpc_Test-NewFile-HighDensity-4G'

rl_name = './results-HD/Test-NewFile-HighDensity-4G'

PSQA_name = './PSQA_result/PSQA_Test-NewFile-HighDensity-4G'

last_action = 200
ALl_reward_list = []
ALL_smooth_list = []
ALL_rebuff_list = []
ALl_reward_without_smoothp_list = []
for i in range(1000):
    action_list = []
    reward_list = []
    rebuf_list = []
    buff_size_list = []
    throughput_list = []
    smooth_list = []

    with open(rl_name + str(i), 'r') as f:
        for line in f:
            if len(line.split()) < 2:
                break
            action = float(line.split()[1])
            reward = float(line.split()[1])
            rebuf = float(line.split()[3])
            buff_size = float(line.split()[2])
            #throughput = 8*float(line.split()[4])/float(line.split()[5])

            action_list.append(action)
            reward_list.append(reward)
            rebuf_list.append(rebuf*50)
            buff_size_list.append(buff_size)
            #throughput_list.append(throughput)
            smooth_list.append(np.abs(action-last_action)/1000.)
            last_action = action

        f.close()

    ALl_reward_list.append(np.sum(reward_list))



plt.plot(ALl_reward_list)
plt.title('Reward-: '+str(np.mean(ALl_reward_list)))
plt.ylim(-2000,2000)
plt.show()
plt.savefig('fig1.png')
ALl_reward_list.sort()

sorted_list = sorted(ALl_reward_list)
print(np.mean(sorted_list))


