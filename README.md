BETA 中有关于n-step和阈值的敏感度分析实验和BETA本体，BETA分为BETA1和BETA2，首先训练TD3-N，将TD3-N的模型作为BETA的输入

BETA中buffer文件包括了训练经验缓冲区和训练过程，全局变量通过全局变量文件控制，main函数为主训练模型，每一定步长会保存模型。

MPC、PSQA在Pensieve test文件中

PPO、SAC、DQN三个算法在对应的文件夹下

Genet需要使用pensieve进行预训练

merina参考需要先用传统算法进行预训练
