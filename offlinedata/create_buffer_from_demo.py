import numpy as np
import torch
import os
import time 
import pickle

import utils 


def create_buffer_from_one_demo_offline(action_type = 'speed', demo_name = "EP0"):

	# action_type decides buffer_name and action_dim
	if action_type == 'acc_steer':
		action_dim = 2
	else:
		action_dim = 1

	# client_interface_.py needs "buffer_name" to load buffer
	# client_interface_.py needs "buffer_name" to load buffer
	if demo_name == "EP0":
		buffer_name = "EP0_human_expert_0"
		max_size = int(13300)
		# max_size = int(20431)  "EP0_human_expert_0"
	elif demo_name == "CHN":
		buffer_name = "CHN_human_expert_0_new"
		max_size = int(108311)

	# 设置 replay_buffer的相应参数
	state_dim = 54-5
	device = torch.device('cuda:3' if torch.cuda.is_available() else "cpu")
	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device, max_size)

	# path for demonstration
	demonstration_dir = os.path.dirname(os.path.abspath(__file__))
	demonstration_name = f"{demo_name}_{action_type}_demonstration_000"
	demonstration_path = os.path.join(demonstration_dir, "vehicle_demo", demonstration_name)

	# 读取相应的demonstration，后面在进行修改的时候可以将更多的demo放进来
	with open(demonstration_path, 'rb') as fo:
		demo = pickle.load(fo, encoding='bytes')
		fo.close()

	# 从demo中读取相应的buffer 

	print("start to load buffer")
	avg_reward = 0
	count = 0
	dist_count  = 0

	for vehicle_id, vehicle_trajectory in demo.items():
		trajectory_len = len(vehicle_trajectory)

		count += 1
		total_reward = 0
		i = 0
		start_time = time.time()
		# vehicle_trajectory[trajectory_len - 1] represents the final state, we don't consider it.
		while i < trajectory_len-1:
			# get corresponding <s,a,r,s'>
			state = vehicle_trajectory[i][1]
			action = vehicle_trajectory[i][2]
			next_state = vehicle_trajectory[i+1][1]
			reward = vehicle_trajectory[i][3]
			# next_state != final state, done_bool = False
			if i != trajectory_len - 2:
				done_bool = False
			else:
				done_bool = True
			
			i = i + 1
			total_reward += reward

			# print(reward)
			# print(state)
			replay_buffer.add(state, action, next_state, reward, done_bool)
		end_time = time.time()
		# print(end_time - start_time)
		print(f'vehicle_id: {vehicle_id}, episode total reward: {total_reward}; trajectory_len: {trajectory_len}')
		if total_reward != 0:
			dist_count += 1
		avg_reward += total_reward

	print(f'count : {count}')
	print(f'dist_count: {dist_count}')
	print(f"avg_reward: {avg_reward/count}")	
	print(f"{buffer_name} has loaded")
	print(replay_buffer.size)
	# 保存收集好的buffer
	if not os.path.exists("./buffers"):
			os.makedirs("./buffers")


	replay_buffer.save(f"./buffers/{buffer_name}")



def create_buffer_from_all_demo_offline(action_type = 'speed', demo_name = "EP0"):

	# action_type decides buffer_name and action_dim
	if action_type == 'acc_steer':
		action_dim = 2
	else:
		action_dim = 1

	
	# client_interface_.py needs "buffer_name" to load buffer
	if demo_name == "EP0":
		buffer_name = "EP0_human_expert_without_4"
		# buffer_name = "EP0_human_expert"
		max_size = int(137363)
	elif demo_name == "CHN":
		buffer_name = "CHN_human_expert"
		max_size = int(2e5)
	

	# 设置 replay_buffer的相应参数
	state_dim = 54
	device = torch.device('cuda:3' if torch.cuda.is_available() else "cpu")
	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device, max_size)

	for demo_id in range(0,8):

		if demo_id == 4:
			continue
		# path for demonstration
		demonstration_dir = os.path.dirname(os.path.abspath(__file__))
		demonstration_name = f"{demo_name}_{action_type}_demonstration_00{demo_id}"
		demonstration_path = os.path.join(demonstration_dir, "vehicle_demo", demonstration_name)

		with open(demonstration_path, 'rb') as fo:
			demo = pickle.load(fo, encoding='bytes')
			fo.close()

		# 从demo中读取相应的buffer 
		print("-"*30)
		print(f"start to load {demo_name}_{action_type}_demonstration_00{demo_id} buffer")


		for vehicle_id, vehicile_trajectory in demo.items():
			trajectory_len = len(vehicile_trajectory)
			# print(trajectory_len)
			i = 0
			start_time = time.time()
			while i < trajectory_len:
				if i != trajectory_len - 1:
					state = vehicile_trajectory[i][1]
					action = vehicile_trajectory[i][2]																																																																																																																																																																																																																																																																																																
					next_state = vehicile_trajectory[i+1][1]
					reward = vehicile_trajectory[i][3]
					done_bool = False
				else:
					state = vehicile_trajectory[i][1]
					action = vehicile_trajectory[i][2]
					next_state = vehicile_trajectory[i][1]
					reward = vehicile_trajectory[i][3]
					done_bool = True
				
				i = i + 1
				# print(action)
				replay_buffer.add(state, action, next_state, reward, done_bool)
			end_time = time.time()
			# print(end_time - start_time)
		
		# check after loading current demon, the size of replay_buffer
		print(f"{demo_name}_{action_type}_demonstration_00{demo_id} buffer has loaded")
	
	print(replay_buffer.size)
		
	
	if not os.path.exists("./buffers"):
		os.makedirs("./buffers")

	replay_buffer.save(f"./buffers/{buffer_name}")

create_buffer_from_one_demo_offline(action_type = "speed", demo_name = "CHN")

# create_buffer_from_one_demo_offline(action_type = 'speed')