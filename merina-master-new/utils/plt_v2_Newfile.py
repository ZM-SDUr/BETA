import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

RESULTS_FOLDER_FCC = '../Results/test/lin/fcc/'
RESULTS_FOLDER_OBE = '../Results/test/lin/oboe/'
RESULTS_FOLDER_3GP = '../Results/test/lin/3gp/'
RESULTS_FOLDER_PUF = '../Results/test/lin/puffer/'
RESULTS_FOLDER_PUF2 = '../Results/test/lin/puffer2/'
RESULTS_FOLDER_FH = '../Results/test/lin/fh/'
TEST_LOG_FILE_Newfile = '../Results_TEST_10_level/test/lin/Newfile_CUHK/'
TEST_LOG_FILE_Newfile_NEXT = '../Results_TEST_10_level/test_next/lin/Newfile_CUHK/'

RESULTS_FOLDER_FCC_LOG = '../Results/test/log/fcc/'
RESULTS_FOLDER_OBE_LOG = '../Results/test/log/oboe/'
RESULTS_FOLDER_3GP_LOG = '../Results/test/log/3gp/'
RESULTS_FOLDER_PUF_LOG = '../Results/test/log/puffer/'
RESULTS_FOLDER_FH_LOG = '../Results/test/log/fh/'
RESULTS_FOLDER_PUF2_LOG = '../Results/test/log/puffer2/'
TEST_LOG_FILE_Newfile_LOG = '../Results_TEST_10_level/test/log/Newfile_CUHK/'
TEST_LOG_FILE_Newfile_LOG_NEXT = '../Results_TEST_10_level/test_next/log/Newfile_CUHK/'

PIC_FOLDER = '../Results/pic/'
NUM_BINS = 200
BITS_IN_BYTE = 8.0
INIT_CHUNK = 4
M_IN_K = 1000.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 150
VIDEO_BIT_RATE = [200, 400 ,800, 1200, 2200, 3300, 5000, 6500, 8600, 10000, 12000]
K_IN_M = 1000.0
REBUF_P_LIN = 12
REBUF_P_LOG = 4.09
SMOOTH_P = 1
COLOR_MAP = plt.cm.rainbow#plt.cm.jet #nipy_spectral, Set1,Paired plt.cm.rainbow#
METRIC = 'log'
NORM = False #True 
PROPOSED_SCHEME = 'test_mpc'
PROPOSED_SCHEME_NAME = 'RobustMPC'
# SCHEMES = ['test_cmc', 'test_merina', 'test_bola'] # 
# METHOD_LABEL = ['Comyco', 'MERINA', 'BOLA'] # 'Penseive' 'BOLA', 'RobustMPC', 'BOLA', , 'Comyco'
LINE_STY = ['--', ':', '-.', '--', ':', '-.', '-', '-']


parser = argparse.ArgumentParser(description='PLOT RESULTS')
parser.add_argument('--log', action='store_true', help='Use logarithmic form QoE metric')
parser.add_argument('--tf', action='store_true', help='Use FCC traces')
parser.add_argument('--tfh', action='store_true', help='Use FCCand3GP traces')
parser.add_argument('--t3g', action='store_true', help='Use 3GP traces')
parser.add_argument('--to', action='store_true', help='Use Oboe traces')
parser.add_argument('--tp', action='store_true', help='Use Puffer traces')
parser.add_argument('--tp2', action='store_true', help='Use Puffer2 traces')
parser.add_argument('--tnf', action='store_true', help='Use Newfile traces')
parser.add_argument('--comyco', action='store_true', help='Show the results5 of Comyco')
parser.add_argument('--mpc', action='store_true', help='Show the results5 of RobustMPC')
parser.add_argument('--pensieve', action='store_true', help='Show the results5 of Penseive')
parser.add_argument('--imrl', action='store_true', help='Show the results5 of MERINA')
parser.add_argument('--bola', action='store_true', help='Show the results5 of BOLA')
parser.add_argument('--adp', action='store_true', help='Show the results5 of adaptation')
parser.add_argument('--fugu', action='store_true', help='Show the results5 of FUGU')
parser.add_argument('--bayes', action='store_true', help='Show the results5 of BayesMPC')
parser.add_argument('--genet', action='store_true', help='Show the results5 of Genet')
parser.add_argument('--pen_next', action='store_true', help='Show the results5 of Penseive next')
parser.add_argument('--im_next', action='store_true', help='Show the results5 of MERINA next')

def save_csv(data, file_name):
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(file_name,index=False,sep=',')

def main():

	args = parser.parse_args()
	# args = parser.parse_args(args = ['--tf', '--merina', '--bola', '--mpc'])
	if args.tf:
		results_folder = RESULTS_FOLDER_FCC_LOG if args.log else RESULTS_FOLDER_FCC
		save_folder = 'log/fcc/' if args.log else 'fcc/'
	elif args.t3g:
		results_folder = RESULTS_FOLDER_3GP_LOG if args.log else RESULTS_FOLDER_3GP
		save_folder = 'log/3gp/' if args.log else '3gp/'
	elif args.tfh:
		results_folder = RESULTS_FOLDER_FH_LOG if args.log else RESULTS_FOLDER_FH
		save_folder = 'log/fh/' if args.log else 'fh/'
	elif args.to:
		results_folder = RESULTS_FOLDER_OBE_LOG if args.log else RESULTS_FOLDER_OBE
		save_folder = 'log/oboe/' if args.log else 'oboe/'
	elif args.tp:
		results_folder = RESULTS_FOLDER_PUF_LOG if args.log else RESULTS_FOLDER_PUF
		save_folder = 'log/puffer/' if args.log else 'puffer/'
	elif args.tp2:
		results_folder = RESULTS_FOLDER_PUF2_LOG if args.log else RESULTS_FOLDER_PUF2
		save_folder = 'log/puffer2/' if args.log else 'puffer2/'
	elif args.tnf:
		if args.log:
			results_folder1 = TEST_LOG_FILE_Newfile_LOG
			results_folder2 = TEST_LOG_FILE_Newfile_LOG_NEXT
		else:
			results_folder1 = TEST_LOG_FILE_Newfile
			results_folder2 = TEST_LOG_FILE_Newfile_NEXT
			save_folder = 'log/Newfile/' if args.log else 'Newfile/'
	else:
		print("Please choose the throughput data traces!!!")
		#results_folder = None

	# if results_folder is not None:
	# 	if os.path.exists(results_folder):
	# 		os.system('rm -rf ' + results_folder)
	# 		os.makedirs(results_folder)

	schemes_show = []
	schemes_label = []

	if args.bola:
		schemes_show.append('test_bola')
		schemes_label.append('BOLA')
	if args.mpc:
		schemes_show.append('test_mpc')
		schemes_label.append('RobustMPC')
	if args.pensieve:
		schemes_show.append('test_a3c')
		schemes_label.append('Pensieve')
	if args.comyco:
		schemes_show.append('test_cmc')
		schemes_label.append('Comyco')
	if args.fugu:
		schemes_show.append('test_fugu')
		schemes_label.append('Fugu')
	if args.bayes:
		schemes_show.append('test_bayes')
		schemes_label.append('BayesMPC')
	if args.imrl:
		schemes_show.append('test_merina')
		schemes_label.append('MERINA')
	if args.genet:
		schemes_show.append('test_genet')
		schemes_label.append('Genet')
	if args.pen_next:
		schemes_show.append('test_a_next')
		schemes_label.append('pensieve_next')
	if args.im_next:
		schemes_show.append('test_mer_next')
		schemes_label.append('MERINA_next')

	mean_rewards = {}
	for scheme in schemes_show:
		mean_rewards[scheme] = {}


	i = 10
	for ii in range(i):

		qoe_metric = METRIC
		normalize = NORM
		norm_addition = 1.2
		time_all = {}
		bit_rate_all = {}
		buff_all = {}
		rebuf_all = {}
		bw_all = {}
		raw_reward_all = {}

		for scheme in schemes_show:
			time_all[scheme] = {}
			raw_reward_all[scheme] = {}
			bit_rate_all[scheme] = {}
			buff_all[scheme] = {}
			rebuf_all[scheme] = {}
			bw_all[scheme] = {}

		log_files = os.listdir(results_folder1 + str(ii+1) + '/')
		for log_file in log_files:

			time_ms = []
			bit_rate = []
			buff = []
			rebuf = []
			bw = []
			reward = []

			print(log_file)

			with open(results_folder1+ str(ii+1) + '/' + log_file, 'rb') as f:
				for line in f:
					parse = line.split()
					if len(parse) <= 1:
						break
					time_ms.append(float(parse[0]))
					bit_rate.append(int(parse[1]))
					buff.append(float(parse[2]))
					rebuf.append(float(parse[3]))
					bw.append(float(parse[4]) / float(parse[5]) * BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
					reward.append(float(parse[6]))


			#将时间序列数据"归一化"，即将时间数据的起点设置为0
			time_ms = np.array(time_ms)
			time_ms -= time_ms[0]

			# print log_file

			#将之前从日志文件中提取的数据（时间、比特率、缓冲区大小、重缓冲时间、带宽和奖励）根据不同的方案（scheme）分类并存储
			for scheme in schemes_show:
				if scheme in log_file:
					time_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = time_ms
					bit_rate_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
					buff_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = buff
					rebuf_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = rebuf
					bw_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw
					raw_reward_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = reward
					break
		
		
		
		
		
		log_files = os.listdir(results_folder2 + str(ii + 1) + '/')
		for log_file in log_files:

			time_ms = []
			bit_rate = []
			buff = []
			rebuf = []
			bw = []
			reward = []

			print(log_file)

			with open(results_folder2 + str(ii + 1) + '/' + log_file, 'rb') as f:
				for line in f:
					parse = line.split()
					if len(parse) <= 1:
						break
					time_ms.append(float(parse[0]))
					bit_rate.append(int(parse[1]))
					buff.append(float(parse[2]))
					rebuf.append(float(parse[3]))
					bw.append(float(parse[4]) / float(parse[5]) * BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
					reward.append(float(parse[6]))

			# 将时间序列数据"归一化"，即将时间数据的起点设置为0
			time_ms = np.array(time_ms)
			time_ms -= time_ms[0]

			# print log_file

			# 将之前从日志文件中提取的数据（时间、比特率、缓冲区大小、重缓冲时间、带宽和奖励）根据不同的方案（scheme）分类并存储
			for scheme in schemes_show:
				if scheme in log_file:
					time_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = time_ms
					bit_rate_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
					buff_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = buff
					rebuf_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = rebuf
					bw_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw
					raw_reward_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = reward
					break
		
		
		# ---- ---- ---- ----
		# Reward records
		# ---- ---- ---- ----

		log_file_all = []
		reward_all = {}
		reward_quality = {}
		reward_rebuf = {}
		reward_smooth = {}
		reward_improvement = {}
		rebuf_improvement = {}
		for scheme in schemes_show:
			reward_all[scheme] = []
			reward_quality[scheme] = []
			reward_rebuf[scheme] = []
			reward_smooth[scheme] = []
			if scheme != PROPOSED_SCHEME:
				reward_improvement[scheme]= []
				rebuf_improvement[scheme] = []

		#l是文件名
		for l in time_all[schemes_show[0]]:
			schemes_check = True
			for scheme in schemes_show:
				if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
					schemes_check = False
					break
			if schemes_check:
				log_file_all.append(l)
				for scheme in schemes_show:
					## record the total QoE data
					reward_all[scheme].append(np.mean(raw_reward_all[scheme][l][INIT_CHUNK:VIDEO_LEN]))
					##--------------------record the individual terms in QoE -------------------------
					## caculate the average video quality and quality smoothness penalty




		for scheme in schemes_show:
			mean_rewards[scheme][ii] = np.mean(reward_all[scheme])



	for ii in range(i):
		for scheme in schemes_show:
			print(scheme+"\t" + str(ii)+"\t" + str(mean_rewards[scheme][ii]))

	fig = plt.figure()
	ax = fig.add_subplot(111)
	width = 0.15
	x = np.arange(1,11)
	y={}
	for scheme in range(len(schemes_show)):
		y[scheme]=[]
		for ii in range(i):
			y[scheme].append(mean_rewards[schemes_show[scheme]][ii])
	barss=[]

	for scheme in range(len(schemes_show)):
		temp = ax.bar(x +(scheme - len(schemes_show)/2) * width, y[scheme], width, label=str(schemes_show[scheme]))
		barss.append(temp)

	# for scheme in schemes_show:
	# 	if qqq==1:
	# 		temp= ax.bar(x-width,y[scheme],width,label = str(scheme))
	# 		barss.append(temp)
	# 		qqq=qqq+1
	# 	elif qqq==2:
	# 		temp= ax.bar(x , y[scheme], width, label=str(scheme))
	# 		barss.append(temp)
	# 		qqq = qqq + 1
	# 	else:
	# 		temp= ax.bar(x+width, y[scheme], width, label=str(scheme))
	# 		barss.append(temp)

	# 在每个柱子上方添加文本
	for bars in barss:
		for bar in bars:
			y_val = bar.get_height()
			plt.text(bar.get_x() + bar.get_width() / 2, y_val, round(y_val, 3), ha='center', va='bottom',fontsize=5)


	plt.ylabel('Mearn reward')
	plt.xlabel('Throughput level')
	plt.legend()
	# if not os.path.exists(PIC_FOLDER + save_folder):
	# 	os.mkdir(PIC_FOLDER + save_folder)
	# plt.savefig(PIC_FOLDER + save_folder + "avg_QoE.pdf")
	plt.savefig("image_Newfile.png", dpi = 500)
	plt.show()

		# ---- ---- ---- ----
		# CDF
		# # ---- ---- ---- ----
		# SCHEMES_REW = []
		# for idx in range(len(schemes_show)):
		# 	# SCHEMES_REW.append(schemes_label[idx] + ': ' + str('%.3f' % mean_rewards[schemes_show[idx]]))
		# 	SCHEMES_REW.append(schemes_label[idx])
		#
		# fig = plt.figure()
		# ax = fig.add_subplot(111)
		#
		#
		# for scheme in schemes_show:
		# 	cdf_values = {}
		# 	values, base = np.histogram(reward_all[scheme], bins=NUM_BINS)
		# 	cumulative = np.cumsum(values)/float(len(reward_all[scheme]))
		# 	cumulative = np.insert(cumulative, 0, 0)
		# 	# ax.plot(base[:-1], cumulative)
		# 	# cdf_values[scheme] = {}
		# 	cdf_values['value'] = base
		# 	cdf_values['cumulative'] = cumulative
		# 	cdf_data_frame = pd.DataFrame(cdf_values)
		# 	sns.lineplot(x="value", y="cumulative", data=cdf_data_frame)
		# # cdf_data_frame = pd.pivot
		#
		#
		# # colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
		# for i,j in enumerate(ax.lines):
		# 	# j.set_color(colors[i])
		# 	plt.setp(j, linestyle = LINE_STY[i], linewidth = 2.6)
		# # sns.lineplot(x=)
		# # sns.lineplot(x="value", y="cumulative", data=cdf_data_frame)
		#
		# legend = ax.legend(SCHEMES_REW, fontsize = 18)
		# frame = legend.get_frame()
		# frame.set_alpha(0)
		# frame.set_facecolor('none')
		#
		# plt.ylabel('CDF (Perc. of sessions)', fontsize = 15)
		# if args.log:
		# 	plt.xlabel("Average Values of Chunk's $QoE_{log}$", fontsize = 15)
		# else:
		# 	plt.xlabel("Average Values of Chunk's $QoE_{lin}$", fontsize = 20)
		# plt.xticks(fontsize = 15)
		# plt.yticks(fontsize = 15)
		# ax.spines['top'].set_visible(False)
		# ax.spines['right'].set_visible(False)
		# ax.spines['bottom'].set_linewidth(2.8)
		# ax.spines['left'].set_linewidth(2.8)
		# # plt.savefig(PIC_FOLDER + save_folder + "CDF_QoE.pdf")
		# # plt.title('HSDPA and FCC') # HSDPA , FCC , Oboe
		# plt.show()
		# plt.savefig("image2.png")
		# #################################################################
		# # QoE reward_improvement
		# #################################################################
		#
		# fig = plt.figure()
		# ax = fig.add_subplot(111)
		#
		# for scheme in comparison_schemes:
		# 	cdf_values = {}
		# 	values, base = np.histogram(reward_improvement[scheme], bins=NUM_BINS)
		# 	cumulative = np.cumsum(values)/float(len(reward_improvement[scheme]))
		# 	cumulative = np.insert(cumulative, 0, 0)
		# 	# ax.plot(base[:-1], cumulative)
		# 	cdf_values['value'] = base[:]
		# 	cdf_values['cumulative'] = cumulative
		# 	cdf_data_frame = pd.DataFrame(cdf_values)
		# 	sns.lineplot(x="value", y="cumulative", data=cdf_data_frame)
		#
		# # colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
		# for i,j in enumerate(ax.lines):
		# 	# j.set_color(colors[i])
		# 	plt.setp(j, linestyle = LINE_STY[i+1], linewidth = 2.6) #, marker = HATCH[i]
		#
		# comparison_schemes_names = [schemes_label[i] for i in range(len(schemes_label))]
		# comparison_schemes_names.remove(PROPOSED_SCHEME_NAME)
		#
		# legend = ax.legend(comparison_schemes_names, loc='best', fontsize = 18)
		# 	# legend = ax.legend(SCHEMES_REW, loc=4, fontsize = 14)
		# frame = legend.get_frame()
		# frame.set_alpha(0)
		# frame.set_facecolor('none')
		# plt.ylabel('CDF (Perc. of sessions)', fontsize = 10)
		# if args.log:
		# 	plt.xlabel("Avg. $QoE_{log}$ improvement", fontsize = 10)
		# else:
		# 	plt.xlabel("Avg. $QoE_{lin}$ improvement", fontsize = 20)
		# # plt.xlabel("Avg. QoE improvement", fontsize = 18)
		# plt.xticks(fontsize = 10)
		# plt.yticks(fontsize = 10)
		# plt.ylim([0.0,1.0])
		# # plt.xlim(-0.2, 1)
		# plt.vlines(0, 0, 1, colors='k',linestyles='solid')
		# ax.spines['top'].set_visible(False)
		# ax.spines['right'].set_visible(False)
		# ax.spines['bottom'].set_linewidth(2.5)
		# ax.spines['left'].set_linewidth(2.5)
		# # plt.title('HSDPA and FCC') # HSDPA , FCC , Oboe
		# # plt.grid()
		# # plt.savefig(PIC_FOLDER + save_folder + "CDF_QoE_IM.pdf")
		# plt.show()
		# plt.savefig("image3.png")

if __name__ == '__main__':
	main()
