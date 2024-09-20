"""
In this file, ppo algorithm is adopted to fine-tune the policy of rate adaptation, gae advantage function and multi-step return are used to calculate the gradients.

!!!! Model-free manner, without prediction for VAE
"""

import argparse
import os
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from .AC_net_v6_11bit import Actor, Critic
from .beta_vae_v6 import BetaVAE
from .test_v5_next import valid
from .replay_memory import ReplayMemory

RANDOM_SEED = 28
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-4
MAX_GRAD_NORM = 5. #梯度裁剪的最大范数

REWARD_MAX = 12  #奖励函数中可能使用的最大奖励值
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
BITS_IN_BYTE = 8.0
DB_NORM_FACTOR = 25.0 #用于归一化分贝值的因子
DEFAULT_QUALITY = int(1)  # default video quality without agent

RAND_RANGE = 1000
ENTROPY_EPS = 1e-6

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# model_actor_para, model_vae_para,

def calculate_from_selection(selected ,last_bit_rate):
    # selected_action is 0-7
    # naive step implementation
    if selected == 1:
        bit_rate = last_bit_rate
    elif selected == 2:
        bit_rate = last_bit_rate + 1
    else:
        bit_rate = last_bit_rate - 1
    # bound
    if bit_rate < 0:
        bit_rate = 0
    if bit_rate > 10:
        bit_rate = 10

    # print(bit_rate)
    return bit_rate






def train_ppo_v6(model_actor_para, model_vae_para, train_env, valid_env, \
                    args, add_str, summary_dir):

    #初始化和日志记录
    log_file_name = summary_dir + '/' + add_str + '/log'    
    if not os.path.exists(summary_dir + '/' + add_str):
        os.mkdir(summary_dir + '/' + add_str)
    command = 'rm -rf ' + summary_dir + '/' + add_str + '/*'
    os.system(command)
    writer = SummaryWriter(summary_dir + '/' + add_str + '/')

    #获取环境信息
    s_info, s_len, c_len, total_chunk_num, \
            bitrate_versions, rebuffer_penalty, smooth_penalty = train_env.get_env_info()
    latent_dim = args.latent_dim
    kld_beta, kld_lambda, recon_gamma = args.kld_beta, args.kld_lambda, args.vae_gamma
    
    with open(log_file_name + '_record', 'w') as log_file, \
            open(log_file_name + '_test_2', 'w') as test_log_file:
        torch.manual_seed(RANDOM_SEED)

        act_dim = 3
        br_dim = len(bitrate_versions)

        # print hyperparameters
        test_log_file.write('Hyper_p: ')
        for k in args.__dict__:
            test_log_file.write(k + ':' + str(args.__dict__[k]) + '|')
        test_log_file.write('\n')
        test_log_file.flush()



        #VAE模型的初始化
        vae_in_channels = 2 # past throughput, downloading time and video chunk sizes

        # Load the vae model
        vae_net = BetaVAE(in_channels=vae_in_channels, hist_dim=c_len, \
                            latent_dim=latent_dim, beta=kld_beta, \
                                delta = kld_lambda, gamma =recon_gamma).type(dtype)
        vae_net.eval()
        vae_net_ori = BetaVAE(in_channels=vae_in_channels, hist_dim=c_len, \
                                latent_dim=latent_dim, beta=kld_beta, \
                                    delta = kld_lambda, gamma =recon_gamma).type(dtype)
        if model_vae_para is not None:
            vae_net.load_state_dict(model_vae_para)
            vae_net_ori.load_state_dict(model_vae_para)
        optimizer_vae = optim.Adam(vae_net.parameters(), lr=LEARNING_RATE_ACTOR)

        ## meta reinforcement learning with a latent-conditioned policy
        # establish the actor and critic model

        #演员-评论家模型的初始化
        model_actor = Actor(act_dim, latent_dim, s_info, s_len).type(dtype)
        # model_actor_nl = Actor_NoLatent(br_dim, s_info, s_len, hidden_layers).type(dtype)
        model_actor_ori = Actor(act_dim, latent_dim, s_info, s_len).type(dtype)
        model_critic = Critic(act_dim, latent_dim, s_info, s_len).type(dtype)
        # model_critic_nl = Critic_NoLatent(br_dim, s_info, s_len, hidden_layers).type(dtype)

        ## load the pretrain model, initialize with sub-optimal models
        if model_actor_para is not None:
            model_actor.load_state_dict(model_actor_para)
            model_actor_ori.load_state_dict(model_actor_para)
        #     model_critic.load_state_dict(model_critic_para)
        
        ## store the original parameters
        # model_actor_ori = Actor(br_dim, latent_dim, s_info, s_len, hidden_layers).type(dtype)
        # model_actor_ori.load_state_dict(model_actor_para)

        optimizer_actor = optim.Adam(model_actor.parameters(), lr=LEARNING_RATE_ACTOR)
        # optimizer_actor_nl = optim.Adam(model_actor_nl.parameters(), lr=LEARNING_RATE_ACTOR)
        optimizer_critic = optim.Adam(model_critic.parameters(), lr=LEARNING_RATE_CRITIC)
        # optimizer_critic_nl = optim.Adam(model_critic_nl.parameters(), lr=LEARNING_RATE_CRITIC)

        # max_grad_norm = MAX_GRAD_NORM 

        # define the observations for vae
        ob = np.zeros((vae_in_channels, c_len)) # observations for vae input

        # define the state for rl agent
        state = np.zeros((s_info, s_len))
        # state[-1][-1] = 1.0
        # state = torch.from_numpy(state)
        explo_bit_rate = last_bit_rate = DEFAULT_QUALITY
        time_stamp, end_flag, video_chunk_remain = 0., True, total_chunk_num

        # define the parameters
        steps_in_episode = args.ro_len #TRAIN_SEQ_LEN  
        minibatch_size = args.batch_size # BATCH_SIZE 
        # exploration_len = args.explo_num #EXPLORATION_LEN

        epoch = 0
        memory = ReplayMemory(64* steps_in_episode)


        #强化学习相关参数
        gamma, gae_param, ppo_ups, clip = args.gae_gamma, args.gae_lambda, args.ppo_ups, args.clip

        lc_alpha, lc_beta, lc_gamma, lc_mu = args.lc_alpha, args.lc_beta, args.lc_gamma, args.lc_mu

        #互信息样本数
        sample_num = int(args.sp_n)

        while True:
            # turn on the eval model
            #将actor和critic，vae_net网络设置为评估模式
            print("epoch:"+ str(epoch))
            model_actor.eval()
            model_critic.eval()
            vae_net.eval()
            # 当缓存中的数据少于指定数量时，继续收集数据
            while memory.return_size() < args.explo_num * args.ro_len -1:

                session_return = []
                states = []
                obs = []
                actions = []
                rewards = []
                values = []
                returns = []
                advantages = []

                # 对每个训练周期内的步骤进行遍历
                for _ in range(steps_in_episode):
                    # 记录当前状态、观察结果和行动
                    if not end_flag:
                        states.append(state_)
                        obs.append(ob_)
                        actions.append(action)
                        values.append(value)

                    bit_rate = explo_bit_rate

                    # execute a step forward
                    delay, sleep_time, buffer_size, rebuf, \
                        video_chunk_size, next_video_chunk_sizes, \
                            end_of_video, video_chunk_remain, \
                                _ = train_env.get_video_chunk(bit_rate)

                    # compute and record the reward of current chunk
                    time_stamp += delay  # in ms
                    time_stamp += sleep_time  # in ms


                    #计算奖励
                    if args.log:
                        log_bit_rate = np.log(bitrate_versions[bit_rate] / \
                                                float(bitrate_versions[0]))
                        log_last_bit_rate = np.log(bitrate_versions[last_bit_rate] / \
                                                    float(bitrate_versions[0]))
                        reward = log_bit_rate \
                                - rebuffer_penalty * rebuf \
                                - smooth_penalty * np.abs(log_bit_rate - log_last_bit_rate)
                    else:
                        reward = bitrate_versions[bit_rate] / M_IN_K \
                                - rebuffer_penalty * rebuf \
                                - smooth_penalty * np.abs(bitrate_versions[bit_rate] -
                                                        bitrate_versions[last_bit_rate]) / M_IN_K


                    # rewards.append(float(reward/REWARD_MAX))
                    #计算奖励，并将其标准化
                    reward_max = rebuffer_penalty
                    r_ = float(max(reward, -0.5*reward_max) / reward_max)
                    # r_ = float(max(reward, -reward_max))
                    rewards.append(r_)
                    last_bit_rate = bit_rate

                    # -------------- logging -----------------
                    # log time_stamp, bit_rate, buffer_size, reward
                    log_file.write(str(time_stamp) + '\t' +
                                str(bitrate_versions[bit_rate]) + '\t' +
                                str(buffer_size) + '\t' +
                                str(rebuf) + '\t' +
                                str(video_chunk_size) + '\t' +
                                str(delay) + '\t' +
                                str(reward) + '\n')
                    log_file.flush()

                    ## dequeue history record
                    # 更新状态和观测信息
                    state = np.roll(state, -1, axis=1)
                    ob = np.roll(ob, -1, axis=1)

                    # this should be S_INFO number of terms
                    # state[0, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec 
                    state[0, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
                    state[1, -1] = float(buffer_size / BUFFER_NORM_FACTOR)  # 10 sec
                    state[2, -1] = bitrate_versions[bit_rate] / float(np.max(bitrate_versions))  # last quality
                    state[3, -1] = np.minimum(video_chunk_remain, total_chunk_num) / float(total_chunk_num)
                    state[4 : 4 + br_dim, -1] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K # mega byte
                    state[15, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR

                    ob[0, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo bits / ms
                    ob[1, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR# seconds
                    # ob[2 : 2 + br_dim, -1] = np.array(curr_chunk_sizes) / M_IN_K / M_IN_K # mega byte
                    # ob[1 + br_dim : 1 + 2 * br_dim, -1] = np.array(curr_chunk_psnrs) / DB_NORM_FACTOR

                    # 为下一个动作选择计算潜在变量和概率
                    # compute the optimal choice for next video chunk
                    ob_ = np.array([ob]).transpose(0, 2, 1)# [N, C_LEN, in_channels]
                    ob_ = torch.from_numpy(ob_).type(dtype)

                    state_ = np.array([state])
                    state_ = torch.from_numpy(state_).type(dtype)

                    with torch.no_grad():
                        latent = vae_net.get_latent(ob_).detach()
                        prob = model_actor.forward(state_, latent).detach()
                        value = model_critic.forward(state_, latent).detach()
                    action = prob.multinomial(num_samples=1).detach()
                    selection = int(action.squeeze().cpu().numpy())
                    explo_bit_rate = calculate_from_selection(selection, last_bit_rate)
                    #explo_bit_rate = int(action.squeeze().cpu().numpy())

                    # retrieve the starting status
                    end_flag = end_of_video
                    if end_of_video:
                        # define the observations for vae
                        ob = np.zeros((vae_in_channels, c_len)) # observations for vae input

                        # define the state for rl agent
                        state = np.zeros((s_info, s_len))
                        # state[-1][-1] = 1.0
                        explo_bit_rate = DEFAULT_QUALITY
                        last_bit_rate = DEFAULT_QUALITY
                        time_stamp = 0.
                        total_chunk_num = train_env.total_chunk_num
                        video_chunk_remain = total_chunk_num

                        break

                # ==================== finish one episode ===================
                # one last step
                R = torch.zeros(1, 1)
                if end_of_video == False:
                    v = value.cpu()
                    R = v.data

                # compute returns and GAE(lambda) advantages:
                if len(states) != len(rewards):
                    if len(states) + 1 == len(rewards):
                        #print("------------------------------------")
                        rewards = rewards[1:]
                    else:
                        print('error in length of states!')
                        break

                #计算回报和优势
                values.append(Variable(R).type(dtype))
                R = Variable(R).type(dtype)
                A = Variable(torch.zeros(1, 1)).type(dtype)
                for i in reversed(range(len(rewards))):
                    td = rewards[i] + gamma * values[i + 1].data[0, 0] - values[i].data[0, 0]
                    A = float(td) + gamma * gae_param * A #计算优势A
                    advantages.insert(0, A)
                    # R = A + values[i]
                    R = gamma * R + rewards[i] #计算累计折扣回报R
                    returns.insert(0, R)

                ## store usefull info:
                session_return.append(np.mean(rewards[1:]))
                # obs_target = obs[:]
                # for i in range(3):
                #     obs_target.append(obs[-1])
                # obs_target = obs_target[3:]
                #将采样的信息放入缓冲B
                memory.push([states, obs, actions, returns, advantages])
                #print(memory.return_size())
        
            # policy grad updates:
            model_actor_old = Actor(act_dim, latent_dim, s_info, s_len).type(dtype)
            model_actor_old.load_state_dict(model_actor.state_dict())
            model_critic_old = Critic(act_dim, latent_dim, s_info, s_len).type(dtype)
            model_critic_old.load_state_dict(model_critic.state_dict())

            ## model update
            model_actor.train()
            model_actor_ori.eval()
            model_actor_old.eval()
            model_critic.eval()
            vae_net.train()
            model_critic.train()

            vae_kld_loss_ = []
            policy_loss_ = []
            value_loss_ = []
            entropy_loss_ = []
            policy_mi_loss_ = []

            # #-------------- VAE training ------------------------
            # for _ in range(vae_update_interval):
            #     batch_states, batch_obs, batch_obs_target, _, _, _ = memory.sample_cuda(minibatch_size)

            #     ## ----------------------- learn the VAE network -------------------------
            #     # retrieve obs and corresponding targets
            #     batch_obs_target = batch_obs_target[:, -3:, :] # [N, 1, in_channels]
            #     batch_states_ = batch_obs[:, -s_len:, :] #[N, 1, in_channels]

            #     x_train = batch_obs # (N, C_LEN, in_channels)
            #     x_decode = batch_states_ # 
            #     y_train = batch_obs_target.view(minibatch_size, vae_in_channels * 3) # (N, in_channels)

            #     # fit the model
            #     y_pre_mu, y_pre_sigma, z_mu, z_log_var = vae_net.forward(x_train, x_decode)
            #     vae_loss, recon_loss, kld_loss = vae_net.loss_function(y_pre_mu, y_pre_sigma, y_train, z_mu, z_log_var)

            #     # record loss infors
            #     vae_recon_loss_.append(recon_loss.detach().cpu().numpy())
            #     vae_kld_loss_.append(kld_loss.detach().cpu().numpy())

            #     # update parameter
            #     optimizer_vae.zero_grad()
            #     vae_loss.backward()
            #     clip_grad_norm_(vae_net.parameters(), max_norm = MAX_GRAD_NORM, norm_type = 2) ## clip
            #     optimizer_vae.step()

            # ------------------- Meta DRL training ---------------------------
            for _ in range(ppo_ups):

                # new mini_batch
                # priority_batch_size = int(memory.get_capacity()/10)
                batch_states, batch_obs, batch_actions, \
                        batch_returns, batch_advantages = memory.sample_cuda(minibatch_size)
                # batch_size = memory.return_size()
                # batch_states, batch_actions, batch_returns, batch_advantages = memory.pop(batch_size)

                # ------------------ VAE case -----------------------
                #使用 VAE 生成潜在变量 (batch_latents)
                batch_latents = vae_net.get_latent(batch_obs)
                batch_latents_ori = vae_net_ori.get_latent(batch_obs)

                # latent samples for p(z_i|s)
                # batch_s_ = batch_obs[:, -s_len:, :]

                x_train = batch_obs # (N, C_LEN, in_channels)

                # fit the model
                #使用从B中采样出来的x_train = batch_obs，得到潜在空间表示
                z_mu, z_log_var = vae_net.forward(x_train)
                #计算 VAE 的 KL 散度损失，并记录
                kld_loss = vae_net.loss_function(z_mu, z_log_var)
                vae_kld_loss_.append(kld_loss.detach().cpu().numpy())

                # kld_loss.backward(retain_graph=True)

                #     # record loss infors
                #     vae_recon_loss_.append(recon_loss.detach().cpu().numpy())
                #     vae_kld_loss_.append(kld_loss.detach().cpu().numpy())

                #     # update parameter
                #     optimizer_vae.zero_grad()
                #     vae_loss.backward()
                #     clip_grad_norm_(vae_net.parameters(), max_norm = MAX_GRAD_NORM, norm_type = 2) ## clip
                #     optimizer_vae.step()

                # ------------------- Latent case ------------------------
                probs_ori = model_actor_ori(batch_states, batch_latents_ori).detach()

                # old_prob
                probs_old = model_actor_old(batch_states, batch_latents).detach()
                v_pre_old = model_critic_old(batch_states, batch_latents).detach()
                prob_value_old = torch.gather(probs_old, dim=1, \
                                                index=batch_actions.type(dlongtype)).detach()

                # new prob
                probs = model_actor(batch_states, batch_latents)
                v_pre = model_critic(batch_states, batch_latents.detach())
                prob_value = torch.gather(probs, dim=1, index=batch_actions.type(dlongtype))

                # ratio  计算新旧策略之间的比率 (ratio）
                ratio = prob_value / (1e-6 + prob_value_old)

                ## non-clip loss
                # surrogate_loss = ratio * batch_advantages.type(dtype)

                # clip loss
                surr1 = ratio * batch_advantages.type(dtype)  # surrogate from conservative policy iteration
                surr2 = ratio.clamp(1 - clip, 1 + clip) * batch_advantages.type(dtype)
                #使用裁剪技术计算演员模型的损失 (loss_clip_actor)
                loss_clip_actor = -torch.mean(torch.min(surr1, surr2))
                # value loss  计算评论家模型的价值损失 (vfloss1)
                vfloss1 = (v_pre - batch_returns.type(dtype)) ** 2

                if epoch >= 150:
                    v_pred_clipped = v_pre_old + (v_pre - v_pre_old).clamp(-clip, clip)
                    vfloss2 = (v_pred_clipped - batch_returns.type(dtype)) ** 2
                    loss_value = 0.5 * torch.mean(torch.max(vfloss1, vfloss2))
                else:
                    loss_value = 0.5 * torch.mean(vfloss1)

                # entropy loss  熵损失
                ent_latent = - torch.mean(probs * torch.log(probs + 1e-6))

                # mutual information loss  互信息损失
                latent_samples = []
                for _ in range(sample_num):
                    latent_samples.append(torch.randn(minibatch_size, latent_dim).type(dtype)) #.detach()
                probs_samples = torch.zeros(minibatch_size, act_dim, 1).type(dtype)
                for idx in range(sample_num):
                    probs_ = model_actor(batch_states, latent_samples[idx])
                    probs_ = probs_.unsqueeze(2)
                    probs_samples = torch.cat((probs_samples, probs_), 2)
                probs_samples = probs_samples[:, :, 1:]
                probs_sa = torch.mean(probs_samples, dim=2) # p(a|s) = 1/L * \sum p(a|s, z_i) p(z_i|s)
                probs_sa = Variable(probs_sa)
                ent_noLatent = - torch.mean(probs_sa * torch.log(probs_sa + 1e-6))
                mutual_info = ent_noLatent - ent_latent

                # mutual_info = - ent_latent

                # cross entropy loss
                #衡量策略网络相对于基准网络性能的提升
                ce_latent = - torch.mean(probs_ori * torch.log(probs + 1e-6))

                # total loss
                # loss_actor = lc_alpha * loss_clip_actor - lc_beta * ent_latent - lc_gamma * mutual_info - lc_mu * ce_latent
                loss_actor = 4 * lc_alpha * loss_clip_actor - \
                                lc_gamma * mutual_info - \
                                    lc_mu * ce_latent
                loss_critic = loss_value 
                # loss_critic = loss_value + kld_loss

                # record
                policy_loss_.append(loss_clip_actor.detach().cpu().numpy())
                entropy_loss_.append(ent_latent.detach().cpu().numpy())
                value_loss_.append(loss_value.detach().cpu().numpy())
                policy_mi_loss_.append(mutual_info.detach().cpu().numpy())

                # update parameters via backpropagation
                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                optimizer_vae.zero_grad()

                loss_total = loss_actor + kld_loss
                # loss_actor.backward(retain_graph=False)
                loss_total.backward(retain_graph=False)
                loss_critic.backward(retain_graph=False)
                # 计算梯度范数并记录
                total_norm = 0
                for p in model_actor.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)

                writer.add_scalar('actor Gradient Norm', total_norm,epoch )
                total_norm = 0
                for p in model_critic.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)

                writer.add_scalar('critic Gradient Norm', total_norm, epoch)
                total_norm = 0
                for p in vae_net.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)

                writer.add_scalar('vae Gradient Norm', total_norm, epoch)



                #参数更新
                clip_grad_norm_(model_actor.parameters(), \
                                    max_norm = MAX_GRAD_NORM, norm_type = 2)
                clip_grad_norm_(model_critic.parameters(), \
                                    max_norm = MAX_GRAD_NORM, norm_type = 2)
                clip_grad_norm_(vae_net.parameters(), \
                                    max_norm = MAX_GRAD_NORM, norm_type = 2)

                # 计算梯度范数并记录
                total_norm = 0
                for p in model_actor.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)

                writer.add_scalar('actor Gradient Norm next', total_norm, epoch)
                total_norm = 0
                for p in model_critic.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)

                writer.add_scalar('critic Gradient Norm next', total_norm, epoch)
                total_norm = 0
                for p in vae_net.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)

                writer.add_scalar('vae Gradient Norm next', total_norm, epoch)





                if epoch >= 150:
                    optimizer_actor.step()
                    optimizer_vae.step()

                optimizer_critic.step()

            ## valid and save the model
            epoch += 1
            memory.clear()

            writer.add_scalar("Avg_VAE_kld_loss", np.mean(vae_kld_loss_), epoch)
            writer.add_scalar("Avg_Policy_loss", np.mean(policy_loss_), epoch)
            writer.add_scalar("Avg_Value_loss", np.mean(value_loss_), epoch)
            writer.add_scalar("Avg_Entropy_loss", np.mean(entropy_loss_), epoch)
            writer.add_scalar("Avg_MI_loss", np.mean(policy_mi_loss_), epoch)
            writer.add_scalar("Avg_Return", np.mean(session_return), epoch)
            writer.flush()

            if epoch % int(args.valid_i) == 0 and epoch > 0:
                logging.info("Model saved in file")
                valid(args, valid_env, model_actor, vae_net, epoch, \
                        test_log_file, add_str, summary_dir)
                lc_beta = args.anneal_p * lc_beta
                # lc_gamma = args.anneal_p * lc_gamma

                # save models
                actor_save_path = summary_dir + '/' + add_str + \
                                    "/%s_%s_%d.model" %(str('policy'), add_str, int(epoch))
                # critic_save_path = summary_dir + '/' + add_str + "/%s_%s_%d.model" %(str('critic'), add_str, int(epoch))
                vae_save_path = summary_dir + '/' + add_str + \
                                    "/%s_%s_%d.model" %(str('VAE'), add_str, int(epoch))
                if os.path.exists(actor_save_path): os.system('rm ' + actor_save_path)
                # if os.path.exists(critic_save_path): os.system('rm ' + critic_save_path)
                if os.path.exists(vae_save_path): os.system('rm ' + vae_save_path)
                torch.save(model_actor.state_dict(), actor_save_path)
                # torch.save(model_critic.state_dict(), critic_save_path)
                torch.save(vae_net.state_dict(), vae_save_path)

        writer.close()
    

