import numpy as np
import matplotlib.pyplot as plt


LOG_PATH = './results5/log_test'
PLOT_SAMPLES = 100


epoch = []
reward_min = []
reward_5per = []
reward_mean = []
reward_mid = []
reward_95per=[]
reward_max=[]

with open(LOG_PATH, 'rb') as f:
    for line in f:
        parse = line.split()
        epoch.append(float(parse[0]))
        reward_min.append(float(parse[1]))
        reward_5per.append(float(parse[2]))
        reward_mean.append(float(parse[3]))
        reward_mid.append(float(parse[4]))
        reward_95per.append(float(parse[5]))
        reward_max.append(float(parse[6]))




plt.plot(reward_mean)
plt.title("Recent_r:  "+str(np.mean(reward_mean[-4:])))
plt.show()