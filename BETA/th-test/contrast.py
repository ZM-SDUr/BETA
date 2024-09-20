import os
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import shutil
OPT_file = '../DP_result/DP-result-1000/'
Model_file = './file_trans/TD3-N/'

source_dir = '../PPO-SAC-DQN/trace_for_test/'
thr = [100,200,300,500,700,1000,1500,2000,2500,3000]

for th in thr:
    dest_dir1 = './threshold/th-' + str(th) + '/L1/'
    dest_dir2 = './threshold/th-' + str(th) + '/L2/'
    if not os.path.exists(dest_dir1):
        os.makedirs(dest_dir1)
    if not os.path.exists(dest_dir2):
        os.makedirs(dest_dir2)
    file_list = natsorted(os.listdir(Model_file))
    file_list_OPT = natsorted(os.listdir(OPT_file))

    OPT_reward = []
    Model_reward = []
    for i in range(len(file_list_OPT)):
        One_Model_reward = 0
        with open(Model_file + file_list[i],'r') as f:
            for line in f:
                if(len(line.strip())>0):
                    One_Model_reward += float(line.split()[1])
            Model_reward.append(One_Model_reward)
        linecount =0
        with open(OPT_file  + file_list_OPT[i], 'r') as f:
            for line in f:
                if linecount == 0:
                    OPT_reward.append(float(line.split()[0]))
                linecount += 1

    diff_list = np.array(OPT_reward)-np.array(Model_reward)
    #plt.plot(OPT_reward)

    cou=0
    for i in range(len(diff_list)):

        if diff_list[i]<th:
            a=1
           # shutil.copy2(source_dir+file_list[i], dest_dir1 + file_list[i])
        elif diff_list[i]>=th:
            #shutil.copy2(source_dir+file_list[i], dest_dir2 + file_list[i])
            cou+=1
    print(th,cou)
