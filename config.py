import os
import torch


class hyperParameters(object):
    def __init__(self, control_steering):

        # about agent's input and output dimention
        # 判断横纵向控制来得到action_dim
        if control_steering:
            self.action_dim = 2
            self.max_action = 1
            self.route_feature_num = 9
            self.ego_feature_num = 4
            self.action_type = "acc_steer"
        else:
            self.action_dim = 1
            self.max_action = 1
            self.route_feature_num = 8
            self.ego_feature_num = 5
            self.action_type = "speed"
        
        self.npc_num = 5
        self.npc_feature_num = 7
        self.mask_num = self.npc_num + 1 # indicating vehicles' present
        self.state_dim = self.route_feature_num + self.ego_feature_num + self.npc_num*self.npc_feature_num + self.mask_num

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


        # Online params
        self.online_train_maxsteps = int(2e5) # Max time steps to run environment
        self.collision_buffer_size = int(5e3)
        self.start_timesteps = 5e3   # Time steps initial random policy is used before training behavioral
        self.rand_action_p = 0.3     # Probability of selecting random action during batch generation
        self.gaussian_std = 0.1      # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
        self.online_eval_freq = int(85)

        # offline experience setting
        self.offline_buffer_size = 2e5 # offline buffer size  14118
        self.offline_timesteps = 2e5   
        self.eval_freq = int(5e3)
        self.batch_size = 512         # Batch size for both actor and critic  CQL 200  other 512
        
        # eval setting
        self.seed = 0               # Sets PyTorch and Numpy seeds  
        self.eval_episodes = 101      # How much episodes we evaluate each time  straight: 54 

        # EP0 : [74, 85, 92, 87, 101, 115, 105, 73]
        # CHN-Merging: [684, 1087, 1039, 1048] 
        

        self.demo_name = "CHN"


        # BCQ params
        self.BCQ_config = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_action": self.max_action,
            "device": self.device,
            "discount": 0.99,
            "tau": 0.005,                   # Target network update rate
            "lmbda": 0.75,                  # Weighting for clipped double Q-learning in BCQ
            "phi": 0.25                     # Max perturbation hyper-parameter for BC  {0, 0.005, 0.05, 0.25, 0.5} 
        }
        
        # BEAR params
        self.BEAR_config = {
            "num_qs": 2,              # the number of critics (Q_function)
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_action": self.max_action,
            "device": self.device,
            "delta_conf": 0.1,
            "use_bootstrap": True,    # actually we did not use this parameter
            "version": 0,             # Basically whether to do min(Q), max(Q), mean(Q) over multiple Q networks for policy updates
            "lambda_": 0.5,
            "threshold": 0.05,
            "mode":'hardcoded',             # Whether to do automatic lagrange dual descent or manually tune coefficient of the MMD loss (prefered "auto")
            "num_samples_match": 10,       # number of samples to do matching in MMD
            "mmd_sigma": 20.0,             # The bandwidth of the MMD kernel parameter
            "lagrange_thresh": 10.0,       # What is the threshold for the lagrange multiplier
            "use_kl": True,                # (True if args.distance_type == "KL" else False),
            "use_ensemble": False,         # Whether to use ensemble variance or not
            "kernel_type": "laplacian"     # kernel type for MMD ("laplacian" or "gaussian")
            # "num_random": 10
        }

        # TD3_BC params
        self.TD3_BC_config = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_action": self.max_action,
            "device": self.device,
            "discount": 0.99,               # Discount factor
            "tau": 0.005,                   # Target network update rate
            # TD3
            "policy_noise": 0.2,             # Noise added to target policy during critic update
            "noise_clip": 0.5,            # Range to clip target policy noise# Range to clip target policy noise
            "policy_freq": 2,             # Frequency of delayed policy updates
            # TD3_BC
            "alpha": 2.5                 # params about BC default 2.5
            }


        self.Morel_config = {
            "obs_dim": self.state_dim,
            "action_dim": self.action_dim,
            "tensorboard_writer": None,
            "comet_experiment": None
        }

        self.IQL_config = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_action": self.max_action,
            "discount": 0.99,
            "tau": 0.005,
            "expectile": 0.7,       # Expectile parameter Tau
            "beta": 3.0,            # Temperature parameter Beta
            "max_weight": 100.0,    # Max weight for actor update
            "actor_hidden": (256, 256),
            "critic_hidden": (256, 256),
            "value_hidden": (256, 256),
        }

    #     parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    # parser.add_argument("--discount", default=0.99)  # Discount factor
    # parser.add_argument("--tau", default=0.005)  # Target network update rate
    # parser.add_argument("--expectile", default=0.7)  
    # parser.add_argument("--beta", default=3.0)  
    # parser.add_argument("--max_weight", default=100.0)  
    # parser.add_argument("--normalize_data", default=True)



       
