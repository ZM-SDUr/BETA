import os
import random
import numpy

Trace_path = './HighDensity/NewFile-HighDensity-4G.txt'
Save_path = 'trace_for_test/'
trace_file_num = 5000
trace_file_len = 1000
trace_list = []
with open(Trace_path,'r')as f:
    for line in f:
        trace = float(line.split()[0])*8/1024
        trace_list.append(trace)
    f.close()
len_data = len(trace_list) #600万 大概70天


for i in range(trace_file_num):
    start_num = random.randint(3, 6000000) 
    with open(Save_path+'trace_for_test_log-NewFile-HighDensity-4G'+str(i),'w') as f:
        for len in range(trace_file_len):
            f.write(str(float(len))+' ')
            f.write(str(trace_list[start_num+len])+'\n')
