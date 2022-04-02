import os
from pycparser.c_ast import Default
import zmq
import argparse
import pickle
import math
import numpy as np
from collections import deque
import random
import time
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter 

from queue import Queue, LifoQueue
from predict_trajectories.trajectory_loader import trajectory_loader
from interaction_env import InteractionEnv


from algo.DDPG import DDPG

from config import hyperParameters
import ReplayBuffer


class main_loop(object):
    def __init__(self, sim_args):
        
        self.interface = InteractionEnv(sim_args)
        self.args = sim_args
        self.train_model = sim_args.train_model
        self.algo_name = sim_args.algo_name
        self.buffer_name = sim_args.buffer_name

        self.current_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())


        # config setting
        self.config = hyperParameters(control_steering=sim_args.control_steering)
        self.action_type = self.config.action_type     # speed or acc_steer

        # common setting
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.state_dim = self.config.state_dim
        self.action_dim = self.config.action_dim
        self.max_action = self.config.max_action
        self.device = self.config.device

        # setting for print and save
        self.setting = f"action_type_{self.action_type}_seed_{self.config.seed}_buffer_name_{self.buffer_name}"
        print("---------------------------------------")
        print(f"Train_model: {self.train_model}, Algo_name: {self.algo_name}, setting = {self.setting}")
        print("---------------------------------------")



    def train_online(self):

        writer = SummaryWriter(f'./log/{self.algo_name}_{self.current_time}')

        # Initialize policy replay buffer
        policy = DDPG(self.state_dim, self.action_dim, self.max_action, self.device)
        self.online_buffer_size = self.config.online_train_maxsteps
        replay_buffer = ReplayBuffer.ReplayBuffer(self.state_dim, self.action_dim, self.device, self.config.online_buffer_size)


        self.config.eval_freq = int(5e3)

        if self.args.visualaztion:
            policy.load(f"./models/{self.algo_name}/{self.algo_name}_network_{self.setting}")
            eval = self.eval_policy(policy)


        training_iters = 0
        episode_num = 0
        episode_timesteps = 0
        episode_reward = 0

        results = dict()
        results['avg_reward'] = []
        results['success_rate'] = []
        results['time_exceed_rate'] = []
        results["collision_rate"] = []
        actor_loss = [0 for x in range(1,int(1e5))]
        critic_loss = [0 for x in range(1,int(1e5))]
        q_val =[0 for x in range(1,int(1e5))]

        # without train, initial eval
        print("Initial eval")
        # eval = self.eval_policy(policy)
        # results['avg_reward'].append(eval['avg_reward'])
        # results['success_rate'].append(eval['success_rate'])
        # results['time_exceed_rate'].append(eval['time_exceed_rate'])
        # results["collision_rate"].append(eval['collision_rate'])
        # np.save(f"./results/{self.algo_name}_results_{self.setting}", results)
        # writer.add_scalar(f'log/{self.algo_name}_avg_reward', float(eval['avg_reward']), training_iters)
        # writer.add_scalar(f'log/{self.algo_name}_success_rate', float(eval['success_rate']), training_iters)
        # writer.add_scalar(f'log/{self.algo_name}_collision_rate', float(eval['collision_rate']), training_iters)

        count = 1
        store_flag = 0
        state_dict = self.interface.reset(count)

        # Interact with the environment for online_buffer_size
        while not self.interface.socket.closed and training_iters < self.online_buffer_size:
            
            training_iters += 1
            episode_timesteps += 1
            # Select action with noise
            action_dict = dict()
            if training_iters < self.config.start_timesteps:
                # action = env.action_space.sample()
                for ego_id, ego_state in state_dict.items():
                    if self.action_type == "speed":
                        action = random.random()
                        action_dict[ego_id] = [action]
                    elif self.action_type == "acc_steer":
                        action1= random.random()
                        action2 = random.random()
                        action = [action1, action2]
                        action_dict[ego_id] = action

            else: 
                
                for ego_id, ego_state in state_dict.items():
                    action = (
                        policy.select_action(np.array(ego_state))
                        + np.random.normal(0, self.max_action * self.config.gaussian_std, size=self.action_dim)
                    ).clip(0, self.max_action)

                    action_dict[ego_id] = list(action)
                
            # Perform action
            next_state_dict, reward_dict, done_dict, _ = self.interface.step(action_dict)

            state = list(state_dict.values())[0]
            next_state = list(next_state_dict.values())[0]
            reward = list(reward_dict.values())[0]
            done_bool = list(done_dict.values())[0]
        
            # Store data in replay buffer

            # judge whether current_state is critical state
            def judge_state(state):
    
                # interaction_vehicles_state 35D  5 * [length, width, x_in_ego, y_in_ego, vehicle_speed, cos(vehicle_heading), sin(vehicle_heading)]
                # x_in_ego 表示的是F坐标系下左右的偏移（即近似y1-y0)
                # y_in_ego 表示的是F坐标系下车行驶的距离 （即近似x1-x0 表示在主车前还是主车后）
                # 关键问题在于计算前车和后车，同时因为考虑的主要是高速路，vehicle_heading意义不大

                flag = 0  # not critical state
    
                ego_length = state[6]
                ego_width = state[7]
                ego_speed = state[3]
                # 注意这里在考虑的时候需要想到我们会存在没有前车或者后车的情况这种情况下阈值要如何设定需要考虑
                dist_reward = 0
                front_dists = [60]; behind_dists = [60]
                front_speeds = [0]; behind_speeds = [0]
                for i in range(0,5):
                    vehicle_length = state[8+7*i]
                    vehicle_width = state[8+7*i+1]
                    x_in_ego_axis = state[8+7*i+2]
                    y_in_ego_axis = state[8+7*i+3]
                    vehicle_speed = state[8+7*i+4]

                    if  y_in_ego_axis > 0 and abs(x_in_ego_axis) < ego_width:
                        front_dists.append(y_in_ego_axis)
                        front_speeds.append(vehicle_speed)
                    elif y_in_ego_axis < 0 and abs(x_in_ego_axis) < ego_width:
                        behind_dists.append(abs(y_in_ego_axis))
                        behind_speeds.append(vehicle_speed)
                
                front_dist = min(front_dists)
                behind_dist = min(behind_dists)

                front_speed = front_speeds[front_dists.index(front_dist)]
                behind_speed = front_speeds[behind_dists.index(behind_dist)]

                a_max = 6
                react_time = 2
                # front_vehicle  safe_dist
                if front_speed <= math.sqrt(ego_speed**2 + 2*ego_speed*a_max*react_time) and front_speed !=0:
                    safe_dist1 = (ego_speed**2 - front_speed**2)/(2*a_max) + ego_speed*react_time
                else:
                    safe_dist1 = 0
                
                # behind_vehicle safe_dist
                if ego_speed <= math.sqrt(behind_speed**2 + 2*behind_speed*a_max*react_time) and behind_speed !=0:
                    safe_dist2 = (behind_speed**2 - ego_speed**2)/(2*a_max) + behind_speed*react_time
                else:
                    safe_dist2 = 0


                if front_dist == 60:
                    # no vehicle is in front of ego
                    if ego_speed > 20:
                        flag = 1
                else:
                    front_speed = front_speeds[front_dists.index(front_dist)]
                    ttc = front_dist/(ego_speed-front_speed)
                    if front_dist < safe_dist1:
                        flag = 1
                
                if behind_dist !=60 and behind_dist < safe_dist2:
                    flag = 1

                        

                return flag

            store_flag = judge_state(state)

            if store_flag == 1:
                replay_buffer.add(state, action, next_state, reward, done_bool)
                replay_buffer.save(f"./offlinedata/buffers/{self.buffer_name}")

            state_dict = next_state_dict
            episode_reward += reward

            # Train agent after collecting sufficient data
            if  training_iters >= self.config.start_timesteps:
                info = policy.train(replay_buffer)
                
                actor_loss[training_iters] = info['actor_loss']
                critic_loss[training_iters] = info['critic_loss']
                q_val[training_iters] = info['q_val']
                writer.add_scalar(f'log/{self.algo_name}_q_val', float(q_val[training_iters]), training_iters)
                writer.add_scalar(f'log/{self.algo_name}_actor_loss', float(actor_loss[training_iters]), training_iters)
                writer.add_scalar(f'log/{self.algo_name}_critic_loss', float(critic_loss[training_iters]), training_iters)


            if done_bool: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {training_iters+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")

                # episode ends, reset store_flag = 0
                store_flag = 0
                #replay_buffer.save(f"./offlinedata/buffers/{self.buffer_name}")
                policy.save(f"./models/DDPG/DDPG_network_{self.setting}")

                if episode_num % self.config.online_eval_freq == 0 and training_iters >= self.config.start_timesteps:
                    eval = self.eval_policy(policy)

                    results['avg_reward'].append(eval['avg_reward'])
                    results['success_rate'].append(eval['success_rate'])
                    results['time_exceed_rate'].append(eval['time_exceed_rate'])
                    results["collision_rate"].append(eval['collision_rate'])

                    np.save(f"./results/DDPG_results_{self.setting}", results)


                    writer.add_scalar(f'log/{self.algo_name}_avg_reward', float(eval['avg_reward']), training_iters)
                    writer.add_scalar(f'log/{self.algo_name}_success_rate', float(eval['success_rate']), training_iters)
                    writer.add_scalar(f'log/{self.algo_name}_collision_rate', float(eval['collision_rate']), training_iters)

                # Reset environment
                count += 1
                if count > 99:
                    count = count - 99
                # print(f"online_count :{online_count}")
                state_dict = self.interface.reset(count)
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1



    # Runs policy eval_episodes times and returns average reward, success_rate, collision_rate
    def eval_policy(self, policy):

        success = 0
        failure = 0
        collision = 0
        deflection = 0
        time_exceed = 0
        episode_num = 0
        avg_reward = 0

        
        # path for demonstration
        demonstration_dir = os.path.dirname(os.path.abspath(__file__))
        demonstration_dir = os.path.join(demonstration_dir, "offlinedata")
        demonstration_name = f"CHN_speed_demonstration_000"
        demonstration_path = os.path.join(demonstration_dir, "vehicle_demo", demonstration_name)
        with open(demonstration_path, 'rb') as fo:
            demo = pickle.load(fo, encoding='bytes')
            fo.close()

        while not self.interface.socket.closed and episode_num < self.config.eval_episodes:

            episode_reward = 0
            episode_step = 0
            episode_num += 1
            count = episode_num - 1  # count: [0, self.config.eval_episodes-1]

            # state reset 
            state_dict = self.interface.reset(count)

            ego_ids = []
            for i in range(102):
                if i+1 in [3,31]:
                    continue
                ego_ids.append(i+1)

            # prepare for no action
            for ego_id, ego_state in state_dict.items():
                initial_action  = np.array(ego_state)[8]/25
            
            # prepare for expert action 
            vehicle_id = ego_ids[count]
            # vehicle_id = 4
            traj = demo[vehicle_id]
            
                
            # start interaction and evaluate the policy 
            while True:
                # sars_tuple = traj[episode_step]
                # expert_action = sars_tuple[2]

                action_dict = dict()
                for ego_id, ego_state in state_dict.items():
                    # policy action
                    action = policy.select_action(np.array(ego_state))
                    
                    # print(f'policy_action: {action}')
  
                    # No action
                    # action = np.array([initial_action])
                    # print(f'initial_action: {initial_action}')

                    #  expert action
                    # action = expert_action
                    # print(f'expert_action: {expert_action}')

                    action_dict[ego_id] = list(action)

                next_state_dict, reward_dict, done_dict, aux_info_dict = self.interface.step(action_dict)

                # episode reward ++
                reward = list(reward_dict.values())[0]
                episode_reward += reward
                avg_reward += reward
                
                if False not in done_dict.values():  # all egos are done
                    aux_info = list(aux_info_dict.values())[0]
                    if aux_info['result'] == 'collision':
                        collision += 1
                        failure += 1
                    elif aux_info['result'] == 'time_exceed':
                        time_exceed += 1
                        # success += 1
                    else:
                        success += 1
                    break
                else:
                    episode_step += 1
                    state_dict = next_state_dict
    
        # After run eval_episodes = 10 episodes, we get avg_reward
        avg_reward /= self.config.eval_episodes
        success_rate = success/self.config.eval_episodes
        time_exceed_rate = time_exceed/self.config.eval_episodes
        collision_rate = collision/self.config.eval_episodes
        eval = {
            'avg_reward': avg_reward,
            'success_rate' : success_rate,
            'time_exceed_rate': time_exceed_rate,
            'collision_rate': collision_rate
        }

        print(f"Evaluation over {self.config.eval_episodes} episodes: {avg_reward:.3f}, success_rate = {success_rate:.3f}, time_exceed_tate: {time_exceed_rate:.3f}, collision_rate = {collision_rate:.3f}")
        return eval



if __name__ == "__main__":
    # simulation argument
    sim_parser = argparse.ArgumentParser()
    sim_parser.add_argument("port", type=int, help="Number of the port (int)", default=5560, nargs="?")
    sim_parser.add_argument("scenario_name", type=str, default="DR_CHN_Merging_ZS" ,help="Name of the scenario (to identify map and folder for track "
                            "files), DR_CHN_Merging_ZS, DR_USA_Intersection_EP0", nargs="?")
    sim_parser.add_argument("total_track_file_number", type=int, help="total number of the track file (int)", default=7, nargs="?")
    sim_parser.add_argument("load_mode", type=str, help="Dataset to load (vehicle, pedestrian, or both)", default="vehicle", nargs="?")
    sim_parser.add_argument("demo_collecting", type=bool, help="Collecting demo through interacting with env", default=False, nargs="?")    
    sim_parser.add_argument("continous_action", type=bool, help="Is the action type continous or discrete", default=True,nargs="?")

    sim_parser.add_argument("control_steering", type=bool, help="control both lon and lat motions", default=False,nargs="?")
    sim_parser.add_argument("route_type", type=str, help="predict, ground_truth or centerline", default='ground_truth',nargs="?")

    sim_parser.add_argument("visualaztion", type=bool, help="Visulize or not", default=False,nargs="?")
    sim_parser.add_argument("ghost_visualaztion", type=bool, help="Visulize or not", default=True,nargs="?")
    sim_parser.add_argument("route_visualaztion", type=bool, help="Visulize or not", default=True,nargs="?")
    sim_parser.add_argument("route_bound_visualaztion", type=bool, help="Visulize or not", default=False,nargs="?")

    sim_parser.add_argument("is_imitation_saving_demo",type=bool, help="Collect demo or not", default=False,nargs="?")
    sim_parser.add_argument("is_imitation_agent",type=bool, help="Use agent policy or not", default=True,nargs="?")

    sim_parser.add_argument('--train_model', type = str, default="online", help='offline, online, generate_buffer')
    sim_parser.add_argument('--buffer_name', type = str, default="collision_data", help='DDPG_CHN_0,DDPG_1e5 random_CHN_0, random_1e5')
    # DDPG_1e5_new： terminal reward = 0, speed_reward = 0.5*v/v_max; random_1e5_new: erminal reward = 0, speed_reward = 0.5*v/v_max
    # collision_data
    sim_parser.add_argument('--algo_name', type = str, default="DDPG", help='DDPG')


    sim_args = sim_parser.parse_args()

    if sim_args.scenario_name is None:
        raise IOError("You must specify a scenario. Type --help for help.")
    if sim_args.load_mode != "vehicle" and sim_args.load_mode != "pedestrian" and sim_args.load_mode != "both":
        raise IOError("Invalid load command. Use 'vehicle', 'pedestrian', or 'both'")


    if not os.path.exists("./results"):
        os.makedirs("./results")


    main_loop = main_loop(sim_args)
    main_loop.train_online()
