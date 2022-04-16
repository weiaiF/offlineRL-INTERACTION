import numpy as np

def calculate_lane_keeping_reward(observation_dict, ego_id):
    ego_x_in_point_axis = observation_dict['lane_observation'][ego_id][0]
    print('ego_x_in_point_axis:', ego_x_in_point_axis)
    # current_min_bound_distance = min(observation['distance_from_bound'][ego_id][0])
    current_heading_error = observation_dict['lane_observation'][ego_id][3]
    future_heading_errors = observation_dict['lane_observation'][ego_id][4:]
    print('futue_heading_errors:', future_heading_errors)

    l_1 = 0.75
    l_2 = 0.75
    lk_reward_current = np.cos(current_heading_error) - l_1*(np.sin(abs(current_heading_error))) - l_2*(abs(ego_x_in_point_axis))
    lk_reward_future = np.sum(np.cos(future_heading_errors))*0.25 - l_1*np.sum(np.sin(np.abs(future_heading_errors)))*0.25
    
    # if current_min_bound_distance < 0.5:
    #     lk_reward = lk_reward_current + lk_reward_future - 5
    # else:
    #     lk_reward = lk_reward_current + lk_reward_future

    lk_reward = lk_reward_current + lk_reward_future
    # print('lk_rewards:', lk_reward_current, lk_reward_future)
    return lk_reward

def calculate_trajectory_pos_reward(observation_dict, ego_id):
    ego_trajectory_distance = observation_dict['trajectory_distance'][ego_id][0]

    trajectory_pos_reward = 1 - 0.2*ego_trajectory_distance
    
    return trajectory_pos_reward


def calculate_speed_reward(observation_dict, control_steering=True):
    # ego_current_target_speed = observation_dict['target_speed'].values()[0][0]
    ego_current_target_speed = 25
    if control_steering:
        ego_speed = observation_dict['lane_observation'].values()[0][2]
    else:
        ego_speed = observation_dict['current_speed'].values()[0][0]

    # max_speed = 12 # m/s

    l_3 = 0.5
    if ego_current_target_speed != 0:
        if ego_speed <= ego_current_target_speed:
            speed_reward = ego_speed / ego_current_target_speed
        else:
            speed_reward = 1 - ((ego_speed - ego_current_target_speed) / ego_current_target_speed)
    else:
        speed_reward = -ego_speed
    
    # speed_reward = ego_current_target_speed - ego_speed
    speed_reward *= l_3

    return speed_reward

def calculate_steer_reward(previous_steer, current_steer):

    l_4 = 1
    # steer_reward = -l_4*abs(current_steer - previous_steer)
    steer_reward = -l_4*abs(current_steer)
    
    return steer_reward

def calculate_dist_reward(ego_width, observation_dict):

    ego_current_speed = observation_dict['current_speed'].values()[0][0]
    TWHFs = [10]; TWHBs = [10]
    # print(observation_dict['interaction_vehicles_observation'].values()[0])
    for i in range(0,5):
        
        vehicle_length = observation_dict['interaction_vehicles_observation'].values()[0][7*i]
        vehicle_width = observation_dict['interaction_vehicles_observation'].values()[0][7*i+1]
        x_in_ego_axis = observation_dict['interaction_vehicles_observation'].values()[0][7*i+2]
        y_in_ego_axis = observation_dict['interaction_vehicles_observation'].values()[0][7*i+3]
        vehicle_speed = observation_dict['interaction_vehicles_observation'].values()[0][7*i+4]
        

        if abs(x_in_ego_axis) < ego_width and y_in_ego_axis > 0 and ego_current_speed > 1:
            TWHF = y_in_ego_axis / ego_current_speed
            TWHFs.append(TWHF)
        elif abs(x_in_ego_axis) < ego_width and y_in_ego_axis < 0 and vehicle_speed > 1:
            TWHB = abs(y_in_ego_axis) / vehicle_speed
            TWHBs.append(TWHB)

    if TWHFs == [10] and ego_current_speed > 1:
        TWHF = 25 / ego_current_speed
        TWHFs.append(TWHF)
    if TWHBs == [10] and ego_current_speed > 1:
        TWHB = 12 / ego_current_speed
        TWHBs.append(TWHB)

    TWHF = np.exp(-min(TWHFs))
    TWHB = np.exp(-min(TWHBs))

    l1 = -4.5
    l2 = -3
    
    dist_reward = TWHF * l1 + TWHB * l2

    if TWHF < 1e-5:
        TWHF = 1
    if TWHB < 1e-5:
        TWHB = 1

    return dist_reward

def calculate_ttc_reward(observation):
    reward = 0

    for v in observation['following_vehicle_ttc'].values():
        if v < 0:
            r = 0
        elif v > 1:
            r = 0
        elif 0.5 < v <= 1:
            r = 0
        else:
            r = -1 * (0.5 - v)
        reward += r

    for v in observation['previous_vehicle_ttc'].values():
        if v < 0:
            r = 0
        elif v > 1:
            r = 0
        elif 0.5 < v <= 1:
            r = 0
        else:
            r = -1 * (0.5 - v)
        reward += r

    for v in observation['front_conflict_vehicle_ttc'].values():
        if v < 0:
            r = 0
        elif v > 1:
            r = 0
        elif 0.5 < v <= 1:
            r = 0
        else:
            r = -1 * (0.5 - v)
        reward += r

    return reward
