import sys
import os
from numpy.lib.ufunclike import fix
import zmq
import argparse
import pickle
import math
import numpy as np
from collections import deque
import random
import csv
import heapq




class Key:
    track_id = "track_id"
    frame_id = "frame_id"
    time_stamp_ms = "timestamp_ms"
    agent_type = "agent_type"
    x = "x"
    y = "y"
    vx = "vx"
    vy = "vy"
    psi_rad = "psi_rad"
    length = "length"
    width = "width"
    

class KeyEnum:
    track_id = 0
    frame_id = 1
    time_stamp_ms = 2
    agent_type = 3
    x = 4
    y = 5
    vx = 6
    vy = 7
    psi_rad = 8
    length = 9
    width = 10
    

class MotionState:
    def __init__(self, time_stamp_ms):
        assert isinstance(time_stamp_ms, int)
        self.time_stamp_ms = time_stamp_ms
        self.x = None
        self.y = None
        self.vx = None
        self.vy = None
        self.psi_rad = None

    def __str__(self):
        return "MotionState: " + str(self.__dict__)

    def get_dict_type_data(self):
        return {'x':self.x,'y':self.y,'vx':self.vx,'vy':self.vy,'psi_rad':self.psi_rad}


class Track:
    def __init__(self, id):
        # assert isinstance(id, int)
        self.track_id = id
        self.agent_type = None
        self.length = None
        self.width = None
        self.time_stamp_ms_first = None
        self.time_stamp_ms_last = None
        self.motion_states = dict()

    def __str__(self):
        string = "Track: track_id=" + str(self.track_id) + ", agent_type=" + str(self.agent_type) + \
               ", length=" + str(self.length) + ", width=" + str(self.width) + \
               ", time_stamp_ms_first=" + str(self.time_stamp_ms_first) + \
               ", time_stamp_ms_last=" + str(self.time_stamp_ms_last) + \
               "\n motion_states:"
        for key, value in sorted(self.motion_states.items()):
            string += "\n    " + str(key) + ": " + str(value)
        return string


def read_tracks(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        track_dict = dict()
        track_id = None

        for i, row in enumerate(list(csv_reader)):

            if i == 0:
                # check first line with key names
                assert(row[KeyEnum.track_id] == Key.track_id)
                assert(row[KeyEnum.frame_id] == Key.frame_id)
                assert(row[KeyEnum.time_stamp_ms] == Key.time_stamp_ms)
                assert(row[KeyEnum.agent_type] == Key.agent_type)
                assert(row[KeyEnum.x] == Key.x)
                assert(row[KeyEnum.y] == Key.y)
                assert(row[KeyEnum.vx] == Key.vx)
                assert(row[KeyEnum.vy] == Key.vy)
                assert(row[KeyEnum.psi_rad] == Key.psi_rad)
                assert(row[KeyEnum.length] == Key.length)
                assert(row[KeyEnum.width] == Key.width)
                continue

            if int(row[KeyEnum.track_id]) != track_id:
                # new track
                track_id = int(row[KeyEnum.track_id])
                assert(track_id not in track_dict.keys()), \
                    "Line %i: Track id %i already in dict, track file not sorted properly" % (i+1, track_id)
                track = Track(track_id)
                track.agent_type = row[KeyEnum.agent_type]
                track.length = float(row[KeyEnum.length])
                track.width = float(row[KeyEnum.width])
                track.time_stamp_ms_first = int(row[KeyEnum.time_stamp_ms])
                track.time_stamp_ms_last = int(row[KeyEnum.time_stamp_ms])
                track_dict[track_id] = track

            track = track_dict[track_id]
            track.time_stamp_ms_last = int(row[KeyEnum.time_stamp_ms])
            ms = MotionState(int(row[KeyEnum.time_stamp_ms]))
            ms.x = float(row[KeyEnum.x])
            ms.y = float(row[KeyEnum.y])
            ms.vx = float(row[KeyEnum.vx])
            ms.vy = float(row[KeyEnum.vy])
            ms.psi_rad = float(row[KeyEnum.psi_rad])
            track.motion_states[ms.time_stamp_ms] = ms

        return track_dict
    
    
def get_trajectory_from_ms_dict(ms_dict):
    # a list [[x, y, vehicle_yaw, vehicle_vx, vehicle_vy]...]
    trajectory_list = []
    for time in sorted(ms_dict):
        ms = ms_dict[time]
        trajectory_list.append([ms.x, ms.y, ms.psi_rad, ms.vx, ms.vy])

    return trajectory_list


def get_route_from_trajectory(trajectory_list, interval_distance):
    # a list [[x, y, point_yaw, point_speed]]
    # first make them equal distance
    average_trajectory_list = [[trajectory_list[0][0], trajectory_list[0][1]]]
    for index, point in enumerate(trajectory_list[1:]):
        if index != (len(trajectory_list) - 1):
            point_previous = average_trajectory_list[-1]
            if point == point_previous:
                continue
            else:
                distance_to_previous = math.sqrt((point[0] - point_previous[0])**2 + (point[1] - point_previous[1])**2)
                average_trajectory_list.append([point[0], point[1]])
                if distance_to_previous >= interval_distance:
                    average_trajectory_list.append([point[0], point[1]])
                else:
                    continue
        else:
            if point == average_trajectory_list[-1]:
                continue
            else:
                average_trajectory_list.append([point[0], point[1]])

    # then calculate yaw
    average_trajectory_with_heading_list = []
    previous_point_yaw = None
    
    for index, point in enumerate(average_trajectory_list):
        if index == (len(average_trajectory_list) - 1): # last point
            point_yaw = average_trajectory_with_heading_list[-1][-1]
        else:
            point_next = average_trajectory_list[index + 1]
            point_vector = np.array((point_next[0] - point[0], point_next[1] - point[1]))

            point_vector_length =  np.sqrt(point_vector.dot(point_vector))
            cos_angle = point_vector.dot(np.array(([1,0])))/(point_vector_length*1 + 1e-8) # angle with x positive (same with carla)
            point_yaw = np.arccos(cos_angle) # rad
            if point_yaw != point_yaw:
                print(f"cos_angle: {cos_angle}")
            if point_vector[1] < 0: # in the upper part of the axis, yaw is a positive value
                point_yaw = - point_yaw
            if previous_point_yaw:
                if (abs(point_yaw - previous_point_yaw) > np.pi/2 and abs(point_yaw - previous_point_yaw) < np.pi* (3/2)):
                    continue
                else:
                    previous_point_yaw = point_yaw
            else:
                previous_point_yaw = point_yaw

        average_trajectory_with_heading_list.append((point[0], point[1], point_yaw))

    # at last the recommend speed value is the nearest trajectory point's speed value
    average_trajectory_with_heading_and_speed_list = []
    for point in average_trajectory_with_heading_list:
        min_distance = 100
        min_distance_point = None
        # get closest point in trajectory
        for point_with_speed in trajectory_list:
            distance = math.sqrt((point[0] - point_with_speed[0])**2 + (point[1] - point_with_speed[1])**2)
            if distance < min_distance:
                min_distance = distance
                min_distance_point = point_with_speed

        # calculate speed value
        point_speed = math.sqrt(min_distance_point[3] ** 2 + min_distance_point[4] ** 2)

        # if point[2] != point[2]:
        #     print(point)
        average_trajectory_with_heading_and_speed_list.append([point[0], point[1], point[2], point_speed])

    return average_trajectory_with_heading_and_speed_list


def get_closest_front_point_index(vehicle_pos, vehicle_route_list):
    min_distance = 100
    closet_point_index = 0
    for index, point in enumerate(vehicle_route_list):
        vehicle_to_point_distance = math.sqrt((point[0] - vehicle_pos[0])**2 + (point[1] - vehicle_pos[1])**2)
        vehicle_y_in_point_axis = (vehicle_pos[1] - point[1])*np.sin(point[2]) + (vehicle_pos[0] - point[0])*np.cos(point[2])
        if min_distance > vehicle_to_point_distance and vehicle_y_in_point_axis < 0:
            min_distance = vehicle_to_point_distance
            closet_point_index = index

    if closet_point_index == 0:
        closet_point_index = len(vehicle_route_list)-1

    return closet_point_index


def get_heading_errors_and_target_speed(vehicle_pos, vehicle_heading, vehicle_route_list):
    # get front points
    closet_point_index = get_closest_front_point_index(vehicle_pos, vehicle_route_list)
    front_point = vehicle_route_list[closet_point_index]

    # get speed value in the cloeset front point
    target_speed = [front_point[3]]

    # get closet_point distance with the vehicle
    closet_dis = math.sqrt((front_point[0] - vehicle_pos[0])**2 + (front_point[1] - vehicle_pos[1])**2)  
    
    # get ego_x_in_point_axis
    ego_x_in_point_axis = (vehicle_pos[1] - front_point[1])*np.cos(front_point[2]) - (vehicle_pos[0] - front_point[0])*np.sin(front_point[2])

    # get corresponding heading errors
    ego_heading_error_0 = front_point[2] - vehicle_heading
    ego_heading_errors_list = [ego_heading_error_0]
    require_num = 4
    remain_point_num = len(vehicle_route_list) - 1 - closet_point_index 
    if remain_point_num < require_num:
        for i in range(closet_point_index + 1, len(vehicle_route_list)):
            point_heading = vehicle_route_list[i][2]
            ego_heading_error = point_heading - vehicle_heading
            # if ego_heading_error != ego_heading_error:
                # print("nan")
                # print(f'point_heading: {point_heading}')
                # print(f'vehicle_heading: {vehicle_heading}')
            ego_heading_errors_list.append(ego_heading_error)
        while len(ego_heading_errors_list) -1  < require_num:
            ego_heading_errors_list.append(ego_heading_errors_list[-1])

    else:
        for i in range(closet_point_index + 1, closet_point_index + require_num + 1):
            point_heading = vehicle_route_list[i][2]
            ego_heading_error = point_heading - vehicle_heading
            ego_heading_errors_list.append(ego_heading_error)

    return ego_heading_errors_list, target_speed, ego_x_in_point_axis, closet_dis



def get_next_pos(vehicle_pos, vehicle_heading, vehicle_speed):
    next_point_pos_x = vehicle_pos[0] + vehicle_speed * math.cos(vehicle_heading)
    next_point_pos_y = vehicle_pos[1] + vehicle_speed * math.sin(vehicle_heading)
    next_point_pos = (next_point_pos_x, next_point_pos_y)

    next_x_in_ego_axis = (next_point_pos[1] - vehicle_pos[1])*np.cos(vehicle_heading) - (next_point_pos[0] - vehicle_pos[0])*np.sin(vehicle_heading)
    next_y_in_ego_axis = (next_point_pos[1] - vehicle_pos[1])*np.sin(vehicle_heading) + (next_point_pos[0] - vehicle_pos[0])*np.cos(vehicle_heading)
    return [next_x_in_ego_axis, next_y_in_ego_axis]



def get_other_vehicle_id(ego_id, crrent_time, track_file):
    other_vehicle_id_list = []
    for vehicle_id, vehicle_dict in track_file.items():
        veh_start = vehicle_dict.time_stamp_ms_first
        veh_end = vehicle_dict.time_stamp_ms_last
        if crrent_time >= veh_start and crrent_time <= veh_end:
            other_vehicle_id_list.append(vehicle_id)
    other_vehicle_id_list.remove(ego_id)
    return other_vehicle_id_list



def get_interaction_vehicles_observation(ego_pos, ego_heading, crrent_time, track_file, all_other_vehicle_id_list):
    surrounding_vehicles = []
    # 1. check if this vehicle within ego's detective range, and put them together
    ego_detective_range = 30 # m
    for other_id in all_other_vehicle_id_list:
        # motion state
        ms = track_file[other_id].motion_states[crrent_time]
        other_vehicle_pos = [ms.x, ms.y]
        other_vehicle_heading = ms.psi_rad
        other_vehicle_speed = math.sqrt(ms.vx ** 2 + ms.vy ** 2)
        
        distance_with_ego = math.sqrt((other_vehicle_pos[0] - ego_pos[0])**2 + (other_vehicle_pos[1] - ego_pos[1])**2)
        y_relative = (other_vehicle_pos[1] - ego_pos[1])*np.sin(ego_heading) + (other_vehicle_pos[0] - ego_pos[0])*np.cos(ego_heading)
        if distance_with_ego <= ego_detective_range and y_relative > -12:
            add_dict = {'vehicle_id': other_id, 'distance': distance_with_ego, 'pos': other_vehicle_pos, 'speed': other_vehicle_speed, 'heading': other_vehicle_heading}
            surrounding_vehicles.append(add_dict)

    # 2. get interaction vehicles and their basic observation
    interaction_vehicles = heapq.nsmallest(5, surrounding_vehicles, key=lambda s: s['distance'])

    # 3. get their full observation
    interaction_vehicles_id = []
    interaction_vehicles_observation = []
    for vehicle_dict in interaction_vehicles:
        # id
        interaction_vehicles_id.append(vehicle_dict['vehicle_id'])
        # basic observation
        # shape
        vehicle_length = track_file[vehicle_dict['vehicle_id']].length
        vehicle_width = track_file[vehicle_dict['vehicle_id']].width
        # motion state
        vehicle_pos = vehicle_dict['pos']
        vehicle_speed = vehicle_dict['speed']
        vehicle_heading = vehicle_dict['heading']

        # ture observation
        x_in_ego_axis = (vehicle_pos[1] - ego_pos[1])*np.cos(ego_heading) - (vehicle_pos[0] - ego_pos[0])*np.sin(ego_heading)
        y_in_ego_axis = (vehicle_pos[1] - ego_pos[1])*np.sin(ego_heading) + (vehicle_pos[0] - ego_pos[0])*np.cos(ego_heading)
        # heading_error_with_ego = vehicle_heading - ego_heading

        single_observation = [vehicle_length, vehicle_width, x_in_ego_axis, y_in_ego_axis, vehicle_speed, np.cos(vehicle_heading), np.sin(vehicle_heading)]
        interaction_vehicles_observation += single_observation
        
    # 4. zero padding and attention mask
    npc_obs_size = 5 * 7
    if len(interaction_vehicles_observation) < npc_obs_size:
        while len(interaction_vehicles_observation) < npc_obs_size:
            interaction_vehicles_observation.append(0)
    # and append attention mask, we didnt use it in discriminator, so set it all 0
    interaction_vehicles_observation += [0, 0, 0, 0, 0, 0]
    return interaction_vehicles_id, interaction_vehicles_observation


def calculate_lane_keeping_reward(ego_x_in_point_axis, ego_heading_errors_list):
    # current_min_bound_distance = min(observation['distance_from_bound'][ego_id][0])
    current_heading_error = ego_heading_errors_list[0]
    future_heading_errors = ego_heading_errors_list[-4:]
    # print('futue_heading_errors:', future_heading_errors)

    l_1 = 0.75
    l_2 = 0.75
    lk_reward_current = np.cos(current_heading_error) - l_1*(np.sin(abs(current_heading_error))) - l_2*(abs(ego_x_in_point_axis))
    # lk_reward_future = np.sum(np.cos(future_heading_errors))*0.25 - l_1*np.sum(np.sin(np.abs(future_heading_errors)))*0.25
    
    # if current_min_bound_distance < 0.5:
    #     lk_reward = lk_reward_current + lk_reward_future - 5
    # else:
    #     lk_reward = lk_reward_current + lk_reward_future

    lk_reward = lk_reward_current
    # print('lk_rewards:', lk_reward_current, lk_reward_future)
    return lk_reward


def calculate_steer_reward(current_steer):

    l_4 = 1
    # steer_reward = -l_4*abs(current_steer - previous_steer)
    steer_reward = -l_4*abs(current_steer)
    
    return steer_reward


def calculate_speed_reward(ego_current_speed, ego_heading_error_0, ego_current_target_speed):

    # ego_speed = ego_current_speed * np.cos(ego_heading_error_0)
    ego_speed = ego_current_speed

    # max_speed = 12 # m/s

    l_3 = 0.5
    if ego_current_target_speed != 0:
        if ego_speed <= ego_current_target_speed:
            speed_reward = ego_speed / ego_current_target_speed
        else:
            speed_reward = 1 - ((ego_speed - ego_current_target_speed) / ego_current_target_speed)
    else:
        speed_reward = -ego_speed
    
    speed_reward *= l_3

    return speed_reward



def calculate_ttc_reward(ego_shape, ego_current_speed, interaction_vehicles_state):
    
    # interaction_vehicles_state 35D  5 * [length, width, x_in_ego, y_in_ego, vehicle_speed, cos(heading_error_with_ego), sin(heading_error_with_ego)]

    # 这里我们默认ttc的阈值是 2.4s
    ttc_thred = 2.4
    ttc_reward = 0
    ego_length = ego_shape[0]
    ego_width = ego_shape[1]
    for i in range(0,5):
        
        vehicle_length = interaction_vehicles_state[7*i]
        vehicle_width = interaction_vehicles_state[7*i+1]
        x_in_ego_axis = interaction_vehicles_state[7*i+2]
        y_in_ego_axis = interaction_vehicles_state[7*i+3]
        vehicle_speed = interaction_vehicles_state[7*i+4]

        if x_in_ego_axis > 0 and ego_current_speed > vehicle_speed:
            # ploy_dis = x_in_ego_axis
            ploy_dis = x_in_ego_axis - 0.5*ego_length - 0.5*vehicle_length
            ttc = ploy_dis/(ego_current_speed - vehicle_speed)
        elif x_in_ego_axis < 0 and ego_current_speed < vehicle_speed:
            # ploy_dis = -x_in_ego_axis
            ploy_dis = -x_in_ego_axis - 0.5*ego_length - 0.5*vehicle_length
            ttc = ploy_dis / (vehicle_speed- ego_current_speed)
        else:
            # other case, ego and vehicle will not have collision
            ttc = 100

        # action type == speed, ttc_reward
        # 0.5*vehicle_width + 0.5*ego_width
        if ttc < ttc_thred and abs(y_in_ego_axis) < 0.5:
            ttc_reward = -500*(1 + ttc/ttc_thred)
            print("--------------------------------------------------")
            print(f"y_in_ego_axis: {y_in_ego_axis}, ttc_reward: {ttc_reward}")
            print(f"x_in_ego_axis: {x_in_ego_axis}, ego_current_speed: {ego_current_speed}, vehicle_speed: {vehicle_speed}")
            print(f"vehicle_length: {vehicle_length}, vehicle_width: {vehicle_width}")
            return ttc_reward

    return ttc_reward


    def in_front_of_ego_vehicle(ego_vehicle,other_vehicle_pos):
        # determine whether it is in front of the ego vehicle
        ego_to_other_vector = np.array([other_vehicle_pos[0] - ego_vehicle._current_state.x, other_vehicle_pos[1] - ego_vehicle._current_state.y])

        ego_velocity_vector = np.array([ego_vehicle._current_state.vx * 10, ego_vehicle._current_state.vy * 10])

        L_ego_to_other = np.sqrt(ego_to_other_vector.dot(ego_to_other_vector))
        L_ego_vel = np.sqrt(ego_velocity_vector.dot(ego_velocity_vector))
        cos_angle = ego_to_other_vector.dot(ego_velocity_vector)/(L_ego_to_other*L_ego_vel)

        radian = np.arccos(cos_angle)
        angle = radian * 180 / np.pi

        if abs(angle) >90:
            return False
        else:
            return True
    
    def calculate_inlane_ttc(self,min_dist,infront_of_ego,ego_vehicle_ms,other_vehicle_ms):
        # in case the rel_speed is zero, ttc = 10
        ego_heading_vector = np.array([ego_vehicle_ms.vx,ego_vehicle_ms.vy])


        other_heading_vector = np.array([other_vehicle_ms.vx,other_vehicle_ms.vy])
        
        
        L_ego_heading = np.sqrt(ego_heading_vector.dot(ego_heading_vector))
        L_other_heading = np.sqrt(other_heading_vector.dot(other_heading_vector))
        
        same_direction = False
        # check whether the other vehicle is in front of ego
        if infront_of_ego:
            # project other_heading_vector along ego_heading_vector 
            project_speed_vector = other_heading_vector.dot(ego_heading_vector) / L_ego_heading
            rel_speed_vector = ego_heading_vector - project_speed_vector 
            print('rel_speed_vector: ',rel_speed_vector)
            L_rel_speed = np.sqrt(rel_speed_vector.dot(rel_speed_vector))
          
            cos_angle = rel_speed_vector.dot(ego_heading_vector)/(L_rel_speed*L_ego_heading)

            radian = np.arccos(cos_angle)
            
            angle = radian * 180 / np.pi
            if abs(angle) <90:
                # rel speed is positive, dist decrese 
                same_direction = True

        else:
            # project ego_heading_vector along other_heading_vector 
            project_speed_vector = ego_heading_vector.dot(other_heading_vector) / L_other_heading
            rel_speed_vector = other_heading_vector - project_speed_vector
            print('rel_speed_vector: ',rel_speed_vector)
            L_rel_speed = np.sqrt(rel_speed_vector.dot(rel_speed_vector))
          
            cos_angle = rel_speed_vector.dot(other_heading_vector)/(L_rel_speed*L_other_heading)

            radian = np.arccos(cos_angle)

            angle = radian * 180 / np.pi
            if abs(angle) <90:
                # rel speed is positive, dist decrese 
                same_direction = True

        min_dist = min_dist * self.lane_dist_limit
        
        print('rel speed: ',np.linalg.norm(rel_speed_vector))
        if same_direction:
            ttc = min_dist / np.linalg.norm(rel_speed_vector)
        else:
            ttc = - min_dist / np.linalg.norm(rel_speed_vector)

        ttc /= self.ttc_nomalization
        if np.isnan(ttc):
            print('ttc is nan, let it be one')
            ttc = 1

        print('min_dist: ',min_dist)
        print('ttc: ',ttc)

        return ttc
    
    
    def calculate_surroundlane_ttc(self,min_dist,infront_of_ego,ego_vehicle_ms,other_vehicle_ms):
        ego_speed_vector = np.array([ego_vehicle_ms.vx,ego_vehicle_ms.vy])
        other_speed_vector = np.array([other_vehicle_ms.vx,other_vehicle_ms.vy])


        # check whether the other vehicle is in front of ego
        if infront_of_ego:
            rel_speed_vector = ego_speed_vector - other_speed_vector
            mass_center_vector = np.array([other_vehicle_ms.x - ego_vehicle_ms.x,other_vehicle_ms.y - ego_vehicle_ms.y])
        else:
            rel_speed_vector = other_speed_vector - ego_speed_vector
            mass_center_vector = np.array([ego_vehicle_ms.x - other_vehicle_ms.x ,ego_vehicle_ms.y - other_vehicle_ms.y])
        # determine whether there is a possibility of collision 
        L_rel_speed = np.sqrt(rel_speed_vector.dot(rel_speed_vector))
        L_mass_center = np.sqrt(mass_center_vector.dot(mass_center_vector))
        cos_angle = rel_speed_vector.dot(mass_center_vector)/(L_rel_speed*L_mass_center)

        radian = np.arccos(cos_angle)
        angle = radian * 180 / np.pi

        min_dist = min_dist * self.lane_dist_limit

        if abs(angle) <90:
            # can collision in future
            # project rel_speed along mass center line 
            rel_speed_along_ceneter_vector = rel_speed_vector.dot(mass_center_vector) / L_mass_center
            ttc = min_dist / np.linalg.norm(rel_speed_along_ceneter_vector)
            # normalized into 10s
            ttc /= self.ttc_nomalization
        else:
            # two vehicle move away
            # project rel_speed along mass center line
            rel_speed_along_ceneter_vector = rel_speed_vector.dot(mass_center_vector) / L_mass_center
            ttc = - min_dist / np.linalg.norm(rel_speed_along_ceneter_vector)
            # normalized into 10s
            ttc /= self.ttc_nomalization

        if np.isnan(ttc):
            print('ttc is nan, let it be one')
            ttc = 1

        return ttc


    def calculate_intersection_ttc(self,conflict_point,ego_vehicle_ms,other_vehicle_ms,ego_lanelet,other_lanelet):
        # calculate the time reach the conflict point
        ego_lanelet_to_conflict_length = geometry.get_centerline_length_to_conflict_point(conflict_point,ego_lanelet,self.min_interval_distance) 

        other_lanelet_to_conflict_length = geometry.get_centerline_length_to_conflict_point(conflict_point,other_lanelet,self.min_interval_distance)

        # calculate the velocity along frenet coordinate

def create_demonstrations_pkl(action_type='acc_steer'):
    root_dir = os.path.dirname((os.path.abspath(__file__)))
    map_dir = os.path.join(root_dir, "maps")
    tracks_dir = os.path.join(root_dir, "recorded_trackfiles")
    scenario_dir = os.path.join(tracks_dir, "DR_USA_Intersection_EP0")

    demonstration_dir = os.path.join(root_dir, "vehicle_demo")
    
    csv_name_list = os.listdir(scenario_dir)
    print(csv_name_list)

    interval_distance = 2

    for csv_name in csv_name_list:
        # for each csv, create a trajectory dict: vehicle_id:[[timestamp, state, action(normalized next tick speed)]....]
        demo_trajectories_dict = dict()
        # load csv file
        csv_file_name = os.path.join(scenario_dir, csv_name)
        csv_file = read_tracks(csv_file_name)
        count = 0
        # make dict
        for vehicle_id, vehicle_dict in csv_file.items():
            vehicle_length = vehicle_dict.length
            vehicle_width = vehicle_dict.width
            ms_dict = vehicle_dict.motion_states
            # first we make vehicle's trajectory to a route
            vehicle_trajectory_list = get_trajectory_from_ms_dict(ms_dict)
            vehicle_route_list = get_route_from_trajectory(vehicle_trajectory_list, interval_distance) # list of [x, y, route_yaw, route_speed]

            # make single trajectory
            demo_trajectories_dict[vehicle_id] = list()
            for time, ms in ms_dict.items():
                vehicle_current_pos = [ms.x, ms.y]
                vehicle_current_heading = ms.psi_rad
                vehicle_current_speed = math.sqrt(ms.vx ** 2 + ms.vy ** 2)

                # 1.route state
                trajectory_pos = [0,0] # trajectory = ego current position in expert data
                heading_errors_list, target_speed, ego_x_in_point_axis, closet_dis = get_heading_errors_and_target_speed(vehicle_current_pos, vehicle_current_heading, vehicle_route_list)
                route_state = trajectory_pos + heading_errors_list + target_speed

                # 2.ego state
                ego_speed = [vehicle_current_speed]
                ego_next_pos_list = get_next_pos(vehicle_current_pos, vehicle_current_heading, vehicle_current_speed)
                ego_shape = [vehicle_length, vehicle_width]
                ego_state = ego_speed + ego_next_pos_list + ego_shape

                # 3.npc state
                all_other_vehicle_id_list = get_other_vehicle_id(vehicle_id, time, csv_file) # first pick other vehicles which in current time stamp
                interaction_vehicles_id, interaction_vehicles_state = get_interaction_vehicles_observation(vehicle_current_pos, vehicle_current_heading, time, csv_file, all_other_vehicle_id_list)

                # total state
                all_state = route_state + ego_state + interaction_vehicles_state

                # current action
                if action_type == 'speed':
                    time_next = time + 100
                    if time_next in ms_dict.keys():
                        target_vx = ms_dict[time_next].vx
                        target_vy = ms_dict[time_next].vy
                        action = math.sqrt(target_vx ** 2 + target_vy ** 2) / 10
                    else:
                        action = vehicle_current_speed / 10

                    action_norm = [np.clip(action, 0, 1)]

                elif action_type == 'acc_steer':
                    wheelbase_scale = 0.6
                    wheelbase = vehicle_length * wheelbase_scale
                    gravity_core_scale = 0.4
                    f_len = wheelbase * gravity_core_scale
                    r_len = wheelbase - f_len

                    time_next = time + 100
                    if time_next in ms_dict.keys():
                        vehicle_next_pos = [ms_dict[time_next].x, ms_dict[time_next].y]
                        vehicle_next_heading = ms_dict[time_next].psi_rad
                        
                        delta_x = vehicle_next_pos[0] - vehicle_current_pos[0]
                        delta_y = vehicle_next_pos[1] - vehicle_current_pos[1]
                        if delta_x == 0 or delta_y == 0 or vehicle_current_speed == 0:
                            beta = 0
                        else:
                            delta_rad = vehicle_next_heading - vehicle_current_heading
                            if delta_rad < -3.14:
                                delta_rad += 6.28
                            elif delta_rad > 3.14:
                                delta_rad -= 6.28

                            sin_beta = delta_rad / ((vehicle_current_speed / r_len) * 0.1)
                            # print(sin_beta)
                            # print("%", sin_beta, delta_rad, vehicle_current_speed, r_len)
                            beta = math.asin(sin_beta) # rad
                            xx = math.tan(beta)
                            tan_rad = xx / (r_len / (r_len + f_len))
                            rad = math.atan(tan_rad)

                        vehicle_next_speed = math.sqrt(ms_dict[time_next].vx ** 2 + ms_dict[time_next].vy ** 2)
                        acc = (vehicle_next_speed - vehicle_current_speed) * 10
                        steering = math.degrees(rad)

                        # normalization
                        max_acc = 3
                        max_steering = 30
                        acc_norm = acc/max_acc
                        steering_norm = steering/max_steering
                        action_norm = [acc_norm, steering_norm]

                # calculate reward
            
                # step reward
                position_reward = calculate_lane_keeping_reward(ego_x_in_point_axis, heading_errors_list)
                steer_reward = calculate_steer_reward(steering_norm)
                ego_heading_error_0 = heading_errors_list[0]
                ego_current_speed = ego_speed[0]
                new_target_speed = 10
                speed_reward = calculate_speed_reward(ego_current_speed, ego_heading_error_0, new_target_speed)

                # ttc_reward = calculate_ttc_reward(ego_shape, ego_current_speed, interaction_vehicles_state)
                step_reward = -0.5 
                # position_reward = 0
                env_reward =  speed_reward + step_reward + position_reward + steer_reward


                demo_trajectories_dict[vehicle_id].append([time, all_state, action_norm, env_reward])
        
        if action_type == 'speed':
            demonstration_name = "EP0_speed_demonstration_" + csv_name[-7:-4]
        elif action_type == 'acc_steer':
            demonstration_name = "EP0_acc_steer_demonstration_" + csv_name[-7:-4]
        
        demonstration_path = os.path.join(demonstration_dir, demonstration_name)
        with open(demonstration_path, 'wb') as fo:
            pickle.dump(demo_trajectories_dict, fo)
            fo.close()

create_demonstrations_pkl(action_type='acc_steer')

# demonstration_dir = os.path.dirname(os.path.abspath(__file__))
# demonstration_name = "demonstration_000"
# demonstration_path = os.path.join(demonstration_dir, demonstration_name)
# with open(demonstration_path, 'rb') as fo:
#     demo = pickle.load(fo, encoding='bytes')
#     fo.close()

# print(len(a[1]))

# a ->  dictionary 
# a[1] ->  track_id = 1 , this vehicle's trajectory
# a[1][k] ->  timestamp = k this time's <s,a,r> [time, [all_state], [action-normalize], reward]
# vehicle_trajectory = demo[1]
# sara_tuple = vehicle_trajectory[1]
# test_state = sara_tuple[1]
# state_len = len(test_state)

# print(state_len)
        
# convince
# cal_next_heading = vehicle_current_heading + (vehicle_current_speed / r_len) * math.sin(beta) * 0.1
# if vehicle_next_heading != cal_next_heading:
#     print(cal_next_heading, vehicle_next_heading)       

        