import os
import argparse
import random

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
TEST_FILE_Newfile = '../Results_TEST_10_level/test/lin/Newfile_CUHK/'
TEST_FILE_Newfile_NEXT = '../Results_TEST_10_level/test_next/lin/Newfile_CUHK/'

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
parser.add_argument('--pen_next', action='store_true', help='Show the results5 of Penseive')
parser.add_argument('--im_next', action='store_true', help='Show the results5 of MERINA')


def save_csv(data, file_name):
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(file_name,index=False,sep=',')

def main():
	np.random.seed(30)
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
			results_folder1 =  TEST_FILE_Newfile
			results_folder2 = TEST_FILE_Newfile_NEXT
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

	mean_rewards = {}
	for scheme in schemes_show:
		mean_rewards[scheme] = {}



	i = 10
	for ii in range(i):

		qoe_metric = METRIC
		normalize = NORM
		norm_addition = 1.2



		for scheme in schemes_show:
			time_all[scheme][ii] = {}
			raw_reward_all[scheme][ii] = {}
			bit_rate_all[scheme][ii] = {}
			buff_all[scheme][ii] = {}
			rebuf_all[scheme][ii] = {}
			bw_all[scheme][ii] = {}

		log_files = os.listdir(results_folder1 + str(ii+1) + '/')
		for log_file in log_files:

			time_ms = []
			bit_rate = []
			buff = []
			rebuf = []
			bw = []
			reward = []

			#print(log_file)

			with open(results_folder1 + str(ii+1) + '/' + log_file, 'rb') as f:
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
					time_all[scheme][ii][log_file[len('log_' + str(scheme) + '_'):]] = time_ms
					bit_rate_all[scheme][ii][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
					buff_all[scheme][ii][log_file[len('log_' + str(scheme) + '_'):]] = buff
					rebuf_all[scheme][ii][log_file[len('log_' + str(scheme) + '_'):]] = rebuf
					bw_all[scheme][ii][log_file[len('log_' + str(scheme) + '_'):]] = bw
					raw_reward_all[scheme][ii][log_file[len('log_' + str(scheme) + '_'):]] = reward
					break

		log_files = os.listdir(results_folder2 + str(ii+1) + '/')
		for log_file in log_files:

			time_ms = []
			bit_rate = []
			buff = []
			rebuf = []
			bw = []
			reward = []

			#print(log_file)

			with open(results_folder2  + str(ii+1) + '/' + log_file, 'rb') as f:
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
					time_all[scheme][ii][log_file[len('log_' + str(scheme) + '_'):]] = time_ms
					bit_rate_all[scheme][ii][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
					buff_all[scheme][ii][log_file[len('log_' + str(scheme) + '_'):]] = buff
					rebuf_all[scheme][ii][log_file[len('log_' + str(scheme) + '_'):]] = rebuf
					bw_all[scheme][ii][log_file[len('log_' + str(scheme) + '_'):]] = bw
					raw_reward_all[scheme][ii][log_file[len('log_' + str(scheme) + '_'):]] = reward
					#print(raw_reward_all[scheme][ii][log_file[len('log_' + str(scheme) + '_'):]])
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
		for l in time_all[schemes_show[0]][ii]:
			#print(l)
			schemes_check = True
			for scheme in schemes_show:
				if l not in time_all[scheme][ii] or len(time_all[scheme][ii][l]) < VIDEO_LEN:
					schemes_check = False
					break
			if schemes_check:
				log_file_all.append(l)
				for scheme in schemes_show:
					## record the total QoE data
					reward_all[scheme].append(np.mean(raw_reward_all[scheme][ii][l][INIT_CHUNK:VIDEO_LEN]))
					##--------------------record the individual terms in QoE -------------------------



		## calculate the average QoE and individual terms (mean value + std)

		mean_quality = []
		std_quality = []
		mean_rebuf = []
		std_rebuf = []
		mean_smooth = []
		std_smooth = []
		for scheme in schemes_show:
			#print(reward_all[scheme])
			mean_rewards[scheme][ii] = reward_all[scheme]
			##----------------mean value and std------------------


	# ## ------------------------------------load the data into texts------------------------------

	rewards_2 = {}
	for scheme in schemes_show:
		rewards_2[scheme]=[]
		for i in range(10):
			rewards_2[scheme].append(np.mean(mean_rewards[scheme][i]))

	rewards_3 = {}
	for scheme in schemes_show:
		rewards_3[scheme] = []
		for i in range(10):
			for lll in mean_rewards[scheme][i]:
				rewards_3[scheme].append(lll)

	mean_rewards_2 = {}
	for scheme in schemes_show:
		mean_rewards_2[scheme] = np.mean(rewards_2[scheme])

	fig = plt.figure()
	ax = fig.add_subplot(111)

	for scheme in schemes_show:
		ax.plot(rewards_3[scheme])

	SCHEMES_REW = []
	# for scheme in schemes_show:
	# 	SCHEMES_REW.append(scheme + ': ' + str('%.3f' % mean_rewards[scheme]))
	for idx in range(len(schemes_show)):
		SCHEMES_REW.append(schemes_label[idx] + ': ' + str('%.3f' % mean_rewards_2[schemes_show[idx]]))
		# SCHEMES_REW.append(schemes_label[idx])

	colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	for i,j in enumerate(ax.lines):
		j.set_color(colors[i])

	ax.legend(SCHEMES_REW, loc=6)

	plt.ylabel('total reward')
	plt.xlabel('trace index')
	# if not os.path.exists(PIC_FOLDER + save_folder):
	# 	os.mkdir(PIC_FOLDER + save_folder)
	# plt.savefig(PIC_FOLDER + save_folder + "avg_QoE.pdf")
	plt.show()
	plt.savefig("overview.png")


	##--------------------------------------计算细节------------------------------------------------

# ---- ---- ---- ----
# check each trace
# ---- ---- ---- ----
	Te = random.randint(0, 40)
	for level in range(10):

		l = "NewFile-HighDensity-CUHK" + str(Te)


		schemes_check = True
		for scheme in schemes_show:
			if l not in time_all[scheme][level] or len(time_all[scheme][level][l]) < VIDEO_LEN:
				schemes_check = False
				break

		if schemes_check:
			fig = plt.figure(figsize = (10,10))

			ax = fig.add_subplot(311)
			for scheme in schemes_show:
				ax.plot(time_all[scheme][level][l][:VIDEO_LEN], bit_rate_all[scheme][level][l][:VIDEO_LEN])
			colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
			for i, j in enumerate(ax.lines):
				j.set_color(colors[i])
			plt.title(l)
			plt.ylabel('bit rate selection (kbps)')

			ax = fig.add_subplot(312)
			for scheme in schemes_show:
				ax.plot(time_all[scheme][level][l][:VIDEO_LEN], buff_all[scheme][level][l][:VIDEO_LEN])
			colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
			for i, j in enumerate(ax.lines):
				j.set_color(colors[i])
			plt.ylabel('buffer size (sec)')

			ax = fig.add_subplot(313)
			for scheme in schemes_show:
				ax.plot(time_all[scheme][level][l][:VIDEO_LEN], bw_all[scheme][level][l][:VIDEO_LEN])
			colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
			for i, j in enumerate(ax.lines):
				j.set_color(colors[i])
			plt.ylabel('bandwidth (mbps)')
			plt.xlabel('time (sec)')

			SCHEMES_REW = []
			for scheme in schemes_show:
				SCHEMES_REW.append(scheme + ': ' + str(np.sum(raw_reward_all[scheme][level][l][1:VIDEO_LEN])))

			ax.legend(SCHEMES_REW, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(np.ceil(len(schemes_show) / 2.0)))
			plt.show()

			plt.savefig("trace_detail/"+ str(level)+"_"+str(Te) + ".png",dpi=300)
			plt.close(fig)

			#--------------------------------------detail_rebuff---------------------------------------------------
			fig = plt.figure(figsize=(10, 10))
			ax =fig.add_subplot(311)
			rebuf_nubm = {}
			for scheme in schemes_show:
				rebuf_nubm[scheme]=0
				ax.plot(time_all[scheme][level][l][:VIDEO_LEN], rebuf_all[scheme][level][l][:VIDEO_LEN])
				for kk in rebuf_all[scheme][level][l]:
					if float(kk)>0 :
						rebuf_nubm[scheme]= rebuf_nubm[scheme]+1

			colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
			for i, j in enumerate(ax.lines):
				j.set_color(colors[i])
			plt.title(l)
			plt.ylabel('rebuff numbers')
			SCHEMES_REW = []
			for scheme in schemes_show:
				SCHEMES_REW.append(scheme + ': ' + str(rebuf_nubm[scheme]))

			ax.legend(SCHEMES_REW, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(np.ceil(len(schemes_show) / 2.0)))
			plt.show()
			plt.savefig("trace_detail/"+"rebuff_"+ str(level)+"_"+str(Te) + ".png", dpi=300)

if __name__ == '__main__':
	main()
