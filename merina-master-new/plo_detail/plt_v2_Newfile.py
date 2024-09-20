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
TEST_LOG_FILE_Newfile = '../Results_10bit/test/lin/Newfile_CUHK/'
RESULTS_FOLDER_DP = '../cal_upper_bound/results_up_bound/lin/'

RESULTS_FOLDER_FCC_LOG = '../Results/test/log/fcc/'
RESULTS_FOLDER_OBE_LOG = '../Results/test/log/oboe/'
RESULTS_FOLDER_3GP_LOG = '../Results/test/log/3gp/'
RESULTS_FOLDER_PUF_LOG = '../Results/test/log/puffer/'
RESULTS_FOLDER_FH_LOG = '../Results/test/log/fh/'
RESULTS_FOLDER_PUF2_LOG = '../Results/test/log/puffer2/'
TEST_LOG_FILE_Newfile_LOG = '../Results_10bit/test/log/Newfile_CUHK/'
RESULTS_FOLDER_DP_LOG = '../cal_upper_bound/results_up_bound/log/'


PIC_FOLDER = '../Results/pic/'
NUM_BINS = 200
BITS_IN_BYTE = 8.0
INIT_CHUNK = 4
M_IN_K = 1000.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 150
VIDEO_BIT_RATE = [145, 365 ,730, 1100, 2000, 3000, 4500, 6000, 7800, 10000]
K_IN_M = 1000.0
REBUF_P_LIN = 6
REBUF_P_LOG = 4.23
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
parser.add_argument('--newval', action='store_true', help='Show the results5 of New_val')
parser.add_argument('--ppo', action='store_true', help='Show the results5 of New val ppo')
parser.add_argument('--cri_tur', action='store_true', help='Show the results5 of critic turmoil')
parser.add_argument('--ma_ge', action='store_true', help='Show the results5 of master gener')
parser.add_argument('--master', action='store_true', help='Show the results5 of master upper bound')
parser.add_argument('--imrl', action='store_true', help='Show the results5 of MERINA')
parser.add_argument('--bola', action='store_true', help='Show the results5 of BOLA')
parser.add_argument('--dp', action='store_true', help='Show the results5 of upper bound')
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
			results_folder2 = RESULTS_FOLDER_DP_LOG
		else:
			results_folder1 = TEST_LOG_FILE_Newfile
			results_folder2 = RESULTS_FOLDER_DP
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
	if args.newval:
		schemes_show.append('test_newval')
		schemes_label.append('New_val')
	if args.ppo:
		schemes_show.append('test_gener_ppo_4')
		schemes_label.append('ppo_4')
		schemes_show.append('test_gener_ppo_6')
		schemes_label.append('ppo_6')
	if args.cri_tur:
		schemes_show.append('test_cri_tur_3')
		schemes_label.append('Cri_Tur_3')
	if args.ma_ge:
		schemes_show.append('test_ma_ge_4')
		schemes_label.append('Ma_Ge_4')
		# schemes_show.append('test_ma_ge_5')
		# schemes_label.append('Ma_Ge_5')
		schemes_show.append('test_ma_ge_6')
		schemes_label.append('Ma_Ge_6')
		# schemes_show.append('test_ma_ge_7')
		# schemes_label.append('Ma_Ge_7')
	if args.master:
		schemes_show.append('test_master')
		schemes_label.append('master_ub')
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

	if args.dp:
		schemes_show.append('sim_dp')
		schemes_label.append('upper_bound')


	mean_rewards = {}
	mean_quality = {}
	mean_rebuf_time ={}
	mean_rebuf_number = {}
	mean_change_bit = {}
	mean_change_number = {}

	#存储每个跟踪文件的平均奖励，用于画出总图像的跟踪线条
	reward_trace = {}
	for scheme in schemes_show:
		mean_rewards[scheme] = {}
		reward_trace[scheme] = []
		mean_quality[scheme] ={}
		mean_rebuf_time[scheme] = {}
		mean_rebuf_number[scheme] = {}
		mean_change_bit[scheme] = {}
		mean_change_number[scheme] ={}


	if args.log:
		REBUF_P = REBUF_P_LOG
	else:
		REBUF_P = REBUF_P_LIN


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

			#print(log_file)

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
		
		#---------------------提取upper bound文件中的数据-------------------------------------

		if args.dp:
			log_files = os.listdir(results_folder2 + str(ii + 1) + '/')

			for log_file_dp in log_files:
				time_ms = []
				bit_rate = []
				buff = []
				rebuf = []
				bw = []
				reward = []

				#print(log_file_dp)

				with open(results_folder2 + str(ii + 1) + '/' + log_file_dp, 'rb') as f:
					last_t = 0
					last_b = 0
					last_q = 1
					lines = []
					for line in f:
						lines.append(line)
						parse = line.split()
						if len(parse) >= 6:
							time_ms.append(float(parse[3]))
							bit_rate.append(VIDEO_BIT_RATE[int(parse[6])])
							buff.append(float(parse[4]))
							bw.append(float(parse[5]))

					for line in reversed(lines):
						parse = line.split()
						r = 0

						if len(parse) > 1:
							t = float(parse[3])
							b = float(parse[4])
							q = int(parse[6])
							rebuff = 0
							if b == 2:
								rebuff = (t - last_t) - last_b
								rebuf.append(float(rebuff))
								assert rebuff >= -1e-4
							else:
								rebuf.append(float(0))
							if args.log:
								log_bit_rate = np.log(VIDEO_BIT_RATE[q] / \
													  float(VIDEO_BIT_RATE[0]))
								log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_q] / \
														   float(VIDEO_BIT_RATE[0]))
								r = log_bit_rate \
										 - REBUF_P * rebuff \
										 - SMOOTH_P * np.abs(log_bit_rate - log_last_bit_rate)
							else:
								r = VIDEO_BIT_RATE[q] / K_IN_M \
										 - REBUF_P * rebuff \
										 - SMOOTH_P * np.abs(VIDEO_BIT_RATE[q] -
																   VIDEO_BIT_RATE[last_q]) / K_IN_M
							reward.append(float(r))
							# print(r)

							last_t = t
							last_b = b
							last_q = q

					time_ms = time_ms[::-1]
					bit_rate = bit_rate[::-1]
					buff = buff[::-1]
					bw = bw[::-1]

					time_ms = np.array(time_ms)
					# if len(time_ms)>0:
					time_ms -= time_ms[0]

					for scheme in schemes_show:
						if scheme in log_file_dp:
							time_all[scheme][log_file_dp[len('log_' + str(scheme) + '_'):]] = time_ms
							bit_rate_all[scheme][log_file_dp[len('log_' + str(scheme) + '_'):]] = bit_rate
							buff_all[scheme][log_file_dp[len('log_' + str(scheme) + '_'):]] = buff
							rebuf_all[scheme][log_file_dp[len('log_' + str(scheme) + '_'):]] = rebuf
							bw_all[scheme][log_file_dp[len('log_' + str(scheme) + '_'):]] = bw
							raw_reward_all[scheme][log_file_dp[len('log_' + str(scheme) + '_'):]] = reward
							# if(scheme =="sim_rl"):
							# 	print(time_all[scheme])
							# 	print("\n\n---------------------------\n\n")
							break


		#print(1111111)
		# ---- ---- ---- ----
		# Reward records
		# ---- ---- ---- ----

		log_file_all = []
		reward_all = {}
		reward_quality = {}
		reward_rebuf_time = {}
		reward_rebuf_number = {}
		reward_change_bit = {}
		reward_change_number = {}
		temp_change_bit = {}
		reward_improvement = {}
		rebuf_improvement = {}
		for scheme in schemes_show:
			reward_all[scheme] = []
			reward_quality[scheme] = []
			reward_rebuf_time[scheme] = []
			reward_rebuf_number[scheme] = []
			reward_change_bit[scheme] = []
			reward_change_number[scheme] = []
			temp_change_bit[scheme] = {}
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
					temp_change_bit[scheme][l] = []
					#计算每一个吞吐量级别下，每个文件的平均值（rewoard，bit等）
					reward_all[scheme].append(np.mean(raw_reward_all[scheme][l][INIT_CHUNK:VIDEO_LEN]))
					#存储到全局的reward_trace中，用于画出总reward图像的跟踪线条
					reward_trace[scheme].append(np.mean(raw_reward_all[scheme][l][INIT_CHUNK:VIDEO_LEN]))

					reward_quality[scheme].append(np.mean(bit_rate_all[scheme][l][INIT_CHUNK:VIDEO_LEN]))
					reward_rebuf_time[scheme].append(np.sum(rebuf_all[scheme][l][INIT_CHUNK:VIDEO_LEN]))
					reward_rebuf_number[scheme].append(sum(1 for x in rebuf_all[scheme][l][INIT_CHUNK:VIDEO_LEN] if float(x) > 0))

					for sss in range(len(bit_rate_all[scheme][l][INIT_CHUNK:VIDEO_LEN])):
						if sss+INIT_CHUNK+1 < VIDEO_LEN:
							temp_change_bit[scheme][l].append(abs(bit_rate_all[scheme][l][sss+INIT_CHUNK+1]-bit_rate_all[scheme][l][sss+INIT_CHUNK]))
					reward_change_bit[scheme].append(sum(temp_change_bit[scheme][l]))
					reward_change_number[scheme].append(sum(1 for x in temp_change_bit[scheme][l] if float(x) > 0))
					##--------------------record the individual terms in QoE -------------------------
					## caculate the average video quality and quality smoothness penalty



		#计算每个吞吐量级别下，每个视频快的平均值（reward，bit等）
		for scheme in schemes_show:
			mean_rewards[scheme][ii] = np.mean(reward_all[scheme])
			mean_quality[scheme][ii] = np.mean(reward_quality[scheme])
			mean_rebuf_time[scheme][ii] = np.mean(reward_rebuf_time[scheme])
			mean_rebuf_number[scheme][ii] = np.mean(reward_rebuf_number[scheme])
			mean_change_bit[scheme][ii] = np.mean(reward_change_bit[scheme])
			mean_change_number[scheme][ii] = np.mean(reward_change_number[scheme])


	#文件中记录了结果的具体数据
	file_path = "./data.txt"

	# for ii in range(i):
	# 	for scheme in schemes_show:
	# 		print(scheme+"\t" + str(ii)+"\t" + str(mean_rewards[scheme][ii]))





	with open(file_path, "w") as file:

		# -------------------------计算总奖励-------------------------------------------------------------

		file.write("\ntotal reward")
		fig = plt.figure()
		ax = fig.add_subplot(111)
		width = 0.15
		x = np.arange(1, 11)
		y = {}
		for scheme in range(len(schemes_show)):
			file.write("\n" + str(schemes_show[scheme]) + "\t")
			temp_reward = []
			for iii in range(10):
				temp_reward.append(round(mean_rewards[schemes_show[scheme]][iii], 2))
			y[scheme] = round(np.mean(temp_reward), 2)
			file.write(str(y[scheme]))

		for scheme in schemes_show:
			ax.plot(reward_trace[scheme])

		SCHEMES_REW = []
		# for scheme in schemes_show:
		# 	SCHEMES_REW.append(scheme + ': ' + str('%.3f' % mean_rewards[scheme]))
		for idx in range(len(schemes_show)):
			SCHEMES_REW.append(schemes_label[idx] + ': ' + str('%.3f' % y[idx]))
		# SCHEMES_REW.append(schemes_label[idx])

		colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
		for i, j in enumerate(ax.lines):
			j.set_color(colors[i])

		ax.legend(SCHEMES_REW, loc=6)

		plt.ylabel('Mean reward')
		plt.xlabel('trace index')
		plt.show()
		# if not os.path.exists(PIC_FOLDER + save_folder):
		# 	os.mkdir(PIC_FOLDER + save_folder)
		# plt.savefig(PIC_FOLDER + save_folder + "avg_QoE.pdf")
		if args.log:
			plt.savefig("total_reward_log.png", dpi=300)
		else:
			plt.savefig("total_reward_lin.png", dpi=300)

		# -------------------计算性能的细节-----------------------------------------------------

		# -------------------计算reward细节-----------------------------------------------------
		file.write("\n\nreward")
		fig = plt.figure()
		ax = fig.add_subplot(111)
		width = 0.15
		x = np.arange(1, 11)
		y = {}
		for scheme in range(len(schemes_show)):
			file.write("\n" + str(schemes_show[scheme]) + "\t")
			y[scheme] = []
			for ii in range(10):
				y[scheme].append(round(mean_rewards[schemes_show[scheme]][ii], 2))
				file.write(str(round(mean_rewards[schemes_show[scheme]][ii], 2)) + "\t")

		barss = []

		for scheme in range(len(schemes_show)):
			temp = ax.bar(x + (scheme - len(schemes_show) / 2) * width, y[scheme], width,
						  label=str(schemes_show[scheme]))
			barss.append(temp)

		# 在每个柱子上方添加文本
		for bars in barss:
			for bar in bars:
				y_val = bar.get_height()
				plt.text(bar.get_x() + bar.get_width() / 2, y_val, round(y_val, 3), ha='center', va='bottom',
						 fontsize=5)

		plt.ylabel('Mean reward')
		plt.xlabel('Throughput level')
		plt.legend()
		plt.show()
		# if not os.path.exists(PIC_FOLDER + save_folder):
		# 	os.mkdir(PIC_FOLDER + save_folder)
		# plt.savefig(PIC_FOLDER + save_folder + "avg_QoE.pdf")
		if args.log:
			plt.savefig("Mean_reward_log.png", dpi=300)
		else:
			plt.savefig("Mean_reward_lin.png", dpi=300)

		# -------------------计算reward再upper bound中的比例-----------------------------------------------------
		file.write("\n\nreward percent")
		fig = plt.figure()
		ax = fig.add_subplot(111)
		width = 0.15
		x = np.arange(1, 11)
		y = {}
		for scheme in range(len(schemes_show)):
			if schemes_show[scheme] !="sim_dp":
				file.write("\n" + str(schemes_show[scheme]) + "\t")
				y[scheme] = []
				for ii in range(10):
					y[scheme].append(round(mean_rewards[schemes_show[scheme]][ii]/mean_rewards["sim_dp"][ii]*100, 2))
					file.write(str(round(mean_rewards[schemes_show[scheme]][ii]/mean_rewards["sim_dp"][ii]*100, 2)) + "\t")

		barss = []

		for scheme in range(len(schemes_show)):
			if schemes_show[scheme] != "sim_dp":
				temp = ax.bar(x + (scheme - len(schemes_show) / 2) * width, y[scheme], width,
							  label=str(schemes_show[scheme]))
				barss.append(temp)

		# 在每个柱子上方添加文本
		for bars in barss:
			for bar in bars:
				y_val = bar.get_height()
				plt.text(bar.get_x() + bar.get_width() / 2, y_val, round(y_val, 3), ha='center', va='bottom',
						 fontsize=5)

		plt.ylabel('reward percent(%)')
		plt.xlabel('Throughput level')
		plt.legend()
		plt.show()
		# if not os.path.exists(PIC_FOLDER + save_folder):
		# 	os.mkdir(PIC_FOLDER + save_folder)
		# plt.savefig(PIC_FOLDER + save_folder + "avg_QoE.pdf")
		if args.log:
			plt.savefig("reward_percent_log.png", dpi=300)
		else:
			plt.savefig("reward_percent_lin.png", dpi=300)

		# -------------------计算bit细节-----------------------------------------------------

		file.write("\n\nqulity")
		fig = plt.figure()
		ax = fig.add_subplot(111)
		width = 0.15
		x = np.arange(1, 11)
		y = {}
		for scheme in range(len(schemes_show)):
			file.write("\n" + str(schemes_show[scheme]) + "\t")
			y[scheme] = []
			for ii in range(10):
				y[scheme].append(mean_quality[schemes_show[scheme]][ii])
				file.write(str(round(mean_quality[schemes_show[scheme]][ii],2)) + "\t")
		barss = []

		for scheme in range(len(schemes_show)):
			temp = ax.bar(x + (scheme - len(schemes_show) / 2) * width, y[scheme], width, label=str(schemes_show[scheme]))
			barss.append(temp)

		# 在每个柱子上方添加文本
		for bars in barss:
			for bar in bars:
				y_val = bar.get_height()
				plt.text(bar.get_x() + bar.get_width() / 2, y_val, round(y_val, 3), ha='center', va='bottom', fontsize=5)

		plt.ylabel('Mean quality')
		plt.xlabel('Throughput level')
		plt.legend()
		plt.show()
		# if not os.path.exists(PIC_FOLDER + save_folder):
		# 	os.mkdir(PIC_FOLDER + save_folder)
		# plt.savefig(PIC_FOLDER + save_folder + "avg_QoE.pdf")
		if args.log:
			plt.savefig("10level_details/Mean_quality_log.png", dpi=300)
		else:
			plt.savefig("10level_details/Mean_quality_lin.png", dpi=300)



		# -------------------计算细节-----------------------------------------------------
		file.write("\n\nrebuf time")
		fig = plt.figure()
		ax = fig.add_subplot(111)
		width = 0.15
		x = np.arange(1, 11)
		y = {}
		for scheme in range(len(schemes_show)):
			file.write("\n" + str(schemes_show[scheme]) + "\t")
			y[scheme] = []
			for ii in range(10):
				y[scheme].append(mean_rebuf_time[schemes_show[scheme]][ii])
				file.write(str(round(mean_rebuf_time[schemes_show[scheme]][ii], 2)) + "\t")
		barss = []

		for scheme in range(len(schemes_show)):
			temp = ax.bar(x + (scheme - len(schemes_show) / 2) * width, y[scheme], width, label=str(schemes_show[scheme]))
			barss.append(temp)

		# 在每个柱子上方添加文本
		for bars in barss:
			for bar in bars:
				y_val = bar.get_height()
				plt.text(bar.get_x() + bar.get_width() / 2, y_val, round(y_val, 3), ha='center', va='bottom', fontsize=5)

		plt.ylabel('Mean rebuf time')
		plt.xlabel('Throughput level')
		plt.legend()
		plt.show()
		# if not os.path.exists(PIC_FOLDER + save_folder):
		# 	os.mkdir(PIC_FOLDER + save_folder)
		# plt.savefig(PIC_FOLDER + save_folder + "avg_QoE.pdf")
		if args.log:
			plt.savefig("10level_details/Mean_rebuf_time_log.png", dpi=300)
		else:
			plt.savefig("10level_details/Mean_rebuf_time_lin.png", dpi=300)


		# -------------------计算细节-----------------------------------------------------
		file.write("\n\nrebuf number")
		fig = plt.figure()
		ax = fig.add_subplot(111)
		width = 0.15
		x = np.arange(1, 11)
		y = {}
		for scheme in range(len(schemes_show)):
			file.write("\n" + str(schemes_show[scheme]) + "\t")
			y[scheme] = []
			for ii in range(10):
				y[scheme].append(mean_rebuf_number[schemes_show[scheme]][ii])
				file.write(str(round(mean_rebuf_number[schemes_show[scheme]][ii], 2)) + "\t")
		barss = []

		for scheme in range(len(schemes_show)):
			temp = ax.bar(x + (scheme - len(schemes_show) / 2) * width, y[scheme], width, label=str(schemes_show[scheme]))
			barss.append(temp)

		# 在每个柱子上方添加文本
		for bars in barss:
			for bar in bars:
				y_val = bar.get_height()
				plt.text(bar.get_x() + bar.get_width() / 2, y_val, round(y_val, 3), ha='center', va='bottom', fontsize=5)

		plt.ylabel('Mean rebuf number')
		plt.xlabel('Throughput level')
		plt.legend()
		plt.show()
		# if not os.path.exists(PIC_FOLDER + save_folder):
		# 	os.mkdir(PIC_FOLDER + save_folder)
		# plt.savefig(PIC_FOLDER + save_folder + "avg_QoE.pdf")

		if args.log:
			plt.savefig("10level_details/Mean_rebuf_number_log.png", dpi=300)
		else:
			plt.savefig("10level_details/Mean_rebuf_number_lin.png", dpi=300)




		# -------------------计算细节-----------------------------------------------------
		file.write("\n\nbit change")
		fig = plt.figure()
		ax = fig.add_subplot(111)
		width = 0.15
		x = np.arange(1, 11)
		y = {}
		for scheme in range(len(schemes_show)):
			file.write("\n" + str(schemes_show[scheme]) + "\t")
			y[scheme] = []
			for ii in range(10):
				y[scheme].append(mean_change_bit[schemes_show[scheme]][ii])
				file.write(str(round(mean_change_bit[schemes_show[scheme]][ii], 2)) + "\t")
		barss = []

		for scheme in range(len(schemes_show)):
			temp = ax.bar(x + (scheme - len(schemes_show) / 2) * width, y[scheme], width, label=str(schemes_show[scheme]))
			barss.append(temp)

		# 在每个柱子上方添加文本
		for bars in barss:
			for bar in bars:
				y_val = bar.get_height()
				plt.text(bar.get_x() + bar.get_width() / 2, y_val, round(y_val, 3), ha='center', va='bottom', fontsize=5)

		plt.ylabel('Mean bit change')
		plt.xlabel('Throughput level')
		plt.legend()
		plt.show()
		# if not os.path.exists(PIC_FOLDER + save_folder):
		# 	os.mkdir(PIC_FOLDER + save_folder)
		# plt.savefig(PIC_FOLDER + save_folder + "avg_QoE.pdf")

		if args.log:
			plt.savefig("10level_details/Mean_bit_change_log.png", dpi=300)
		else:
			plt.savefig("10level_details/Mean_bit_change_lin.png", dpi=300)



		# -------------------计算细节-----------------------------------------------------
		file.write("\n\nbit change number")
		fig = plt.figure()
		ax = fig.add_subplot(111)
		width = 0.15
		x = np.arange(1, 11)
		y = {}
		for scheme in range(len(schemes_show)):
			file.write("\n" + str(schemes_show[scheme]) + "\t")
			y[scheme] = []
			for ii in range(10):
				y[scheme].append(mean_change_number[schemes_show[scheme]][ii])
				file.write(str(round(mean_change_number[schemes_show[scheme]][ii], 2)) + "\t")
		barss = []

		for scheme in range(len(schemes_show)):
			temp = ax.bar(x + (scheme - len(schemes_show) / 2) * width, y[scheme], width,
						  label=str(schemes_show[scheme]))
			barss.append(temp)

		# 在每个柱子上方添加文本
		for bars in barss:
			for bar in bars:
				y_val = bar.get_height()
				plt.text(bar.get_x() + bar.get_width() / 2, y_val, round(y_val, 3), ha='center', va='bottom',
						 fontsize=5)

		plt.ylabel('Mean bit change number')
		plt.xlabel('Throughput level')
		plt.legend()
		plt.show()
		# if not os.path.exists(PIC_FOLDER + save_folder):
		# 	os.mkdir(PIC_FOLDER + save_folder)
		# plt.savefig(PIC_FOLDER + save_folder + "avg_QoE.pdf")

		if args.log:
			plt.savefig("10level_details/Mean_bit_change_number_log.png", dpi=300)
		else:
			plt.savefig("10level_details/Mean_bit_change_number_lin.png", dpi=300)




if __name__ == '__main__':
	main()
