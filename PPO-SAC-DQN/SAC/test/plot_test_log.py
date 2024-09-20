import os
import numpy as np
import matplotlib.pyplot as plt


rl_name = './test_results-HD/log_sim_rl_Test-NewFile-HighDensity-4G'

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
            reward = float(line.split()[7])
            rebuf = float(line.split()[3])
            buff_size = float(line.split()[2])
            throughput = 8*float(line.split()[4])/float(line.split()[5])

            action_list.append(action)
            reward_list.append(reward)
            rebuf_list.append(rebuf*50)
            buff_size_list.append(buff_size)
            throughput_list.append(throughput)
            smooth_list.append(np.abs(action-last_action)/1000.)
            last_action = action

        f.close()
    #if np.sum(reward_list) >1200:
    if np.mean(throughput_list)>3000 and np.mean(throughput_list)>6000 or 1:
        ALl_reward_list.append(np.sum(reward_list[1:]))
        ALL_smooth_list.append(-np.sum(smooth_list[1:]))
        ALL_rebuff_list.append(np.sum(rebuf_list[1:]))

    if np.sum(reward_list) >1000  and 0:
        fig,fig1 = plt.subplots()
        fig1.plot(action_list[1:],label = 'action')
        fig1.plot(throughput_list[1:], label= 'throughput')
        fig1.set_ylabel('Left Axis')
        plt.legend()
        fig2 = fig1.twinx()
        fig2.plot(buff_size_list , color = 'green', linestyle = '-',linewidth = 0.3,label = 'rebuf')
        #fig2.plot(smooth_list, color = 'red' , linestyle = '-',linewidth = 0.3)
        plt.title('CUHK'+str(i)+'-Reward:  ' +
                  str(np.sum(reward_list[1:]))+'\n'+
                'Rebuff-P:  ' + str(np.sum(rebuf_list[1:]))+'\n'+
                'Smooth-P :  '+str(np.sum(smooth_list[1:])),fontsize = 10)
        plt.xlabel('action')
        plt.show()



plt.plot(ALl_reward_list)
plt.title('Reward-: '+str(np.mean(ALl_reward_list)))
plt.ylim(-5000,2000)
plt.show()

ALl_reward_list.sort()

sorted_list = sorted(ALl_reward_list)
print(np.mean(ALl_reward_list))

