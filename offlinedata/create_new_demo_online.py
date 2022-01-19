import sys
import os
import zmq
import argparse
import pickle
import math
import numpy as np
from collections import deque
import random
import csv
import heapq

import sys 
sys.path.append("..") 
from client_interface_ import client_interface


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
            cos_angle = point_vector.dot(np.array(([1,0])))/(point_vector_length*1) # angle with x positive (same with carla)
            point_yaw = np.arccos(cos_angle) # rad
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
        point_speed = math.sqrt(point_with_speed[3] ** 2 + point_with_speed[4] ** 2)

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

    # get corresponding heading errors
    ego_heading_error_0 = front_point[2] - vehicle_heading
    ego_heading_errors_list = [ego_heading_error_0]
    require_num = 4
    remain_point_num = len(vehicle_route_list) - 1 - closet_point_index 
    if remain_point_num < require_num:
        for i in range(closet_point_index + 1, len(vehicle_route_list)):
            point_heading = vehicle_route_list[i][2]
            ego_heading_error = point_heading - vehicle_heading
            ego_heading_errors_list.append(ego_heading_error)
        while len(ego_heading_errors_list)-1 < require_num:
            ego_heading_errors_list.append(ego_heading_errors_list[-1])

    else:
        for i in range(closet_point_index + 1, closet_point_index + require_num + 1):
            point_heading = vehicle_route_list[i][2]
            ego_heading_error = point_heading - vehicle_heading
            ego_heading_errors_list.append(ego_heading_error)

    return ego_heading_errors_list, target_speed


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
    interaction_vehicles_observation = []
    for vehicle_dict in interaction_vehicles:
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
        heading_error_with_ego = vehicle_heading - ego_heading

        single_observation = [vehicle_length, vehicle_width, x_in_ego_axis, y_in_ego_axis, vehicle_speed, np.cos(heading_error_with_ego), np.sin(heading_error_with_ego)]
        interaction_vehicles_observation += single_observation
    
    # 4. zero padding and attention mask
    npc_obs_size = 5 * 7
    if len(interaction_vehicles_observation) < npc_obs_size:
        while len(interaction_vehicles_observation) < npc_obs_size:
            interaction_vehicles_observation.append(0)
    # and append attention mask, we didnt use it in discriminator, so set it all 0
    interaction_vehicles_observation += [0, 0, 0, 0, 0, 0]
    return interaction_vehicles_observation


def create_demonstrations_pkl_offline(action_type='speed'):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    map_dir = os.path.join(root_dir, "maps")
    tracks_dir = os.path.join(root_dir, "recorded_trackfiles")
    scenario_dir = os.path.join(tracks_dir, "DR_USA_Intersection_EP0")

    demonstration_dir = os.path.dirname(os.path.abspath(__file__))
    
    csv_name_list = os.listdir(scenario_dir)
    # print(csv_name_list)

    interval_distance = 2
    for csv_name in csv_name_list:
        # for each csv, create a trajectory dict: vehicle_id:[[timestamp, state, action(normalized next tick speed)]....]
        demo_trajectories_dict = dict()
        # load csv file
        csv_file_name = os.path.join(scenario_dir, csv_name)
        csv_file = read_tracks(csv_file_name)
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
                heading_errors_list, target_speed = get_heading_errors_and_target_speed(vehicle_current_pos, vehicle_current_heading, vehicle_route_list)
                route_state = trajectory_pos + heading_errors_list + target_speed

                # 2.ego state
                ego_speed = [vehicle_current_speed]
                ego_next_pos_list = get_next_pos(vehicle_current_pos, vehicle_current_heading, vehicle_current_speed)
                ego_shape = [vehicle_length, vehicle_width]
                ego_state = ego_speed + ego_next_pos_list + ego_shape

                # 3.npc state
                all_other_vehicle_id_list = get_other_vehicle_id(vehicle_id, time, csv_file) # first pick other vehicles which in current time stamp
                interaction_vehicles_state = get_interaction_vehicles_observation(vehicle_current_pos, vehicle_current_heading, time, csv_file, all_other_vehicle_id_list)

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

                    action_clip = np.clip(action, 0, 1)

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
                        acc = vehicle_next_speed - vehicle_current_speed
                        steering = math.degrees(rad)

                        acc_normalize = 10 * acc / 3 
                        steering_normalize = steering / 30
                        # if abs(steering) > 30:
                        #     print(steering)
                        # if abs(acc_normalize) > 1:
                        #     print(acc_normalize)

                        # convince
                        cal_next_heading = vehicle_current_heading + (vehicle_current_speed / r_len) * math.sin(beta) * 0.1
                        if vehicle_next_heading != cal_next_heading:
                            print(cal_next_heading, vehicle_next_heading)

                        action_clip = np.array([acc, steering])

                demo_trajectories_dict[vehicle_id].append([time, all_state, action_clip])

        demonstration_name = "demonstration_offline_" + csv_name[-7:-4]
        demonstration_path = os.path.join(demonstration_dir, demonstration_name)
        with open(demonstration_path, 'wb') as fo:
            pickle.dump(demo_trajectories_dict, fo)
            fo.close()


# we need make interaction with the env to get some map features as a part of observation
def load_expert_id_time(id_time_file_dir):
    id_and_time_file_list = os.listdir(id_time_file_dir)
    


    expert_id_time_dict = dict()
    for i in range(8):
        expert_id_time_dict[i] = dict()

    for name in id_and_time_file_list:
        name_list = name.split('_')


        # infomation in name list
        file_index = int(name_list[0])
        ego_start_end_time = [int(name_list[1]), int(name_list[2])]

        # id infomation in data
        trajectory_file = os.path.join(id_time_file_dir, name)
        with open(trajectory_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        # data dict['egos_track', 'others_track', 'gt_of_acc_violation', 'gt_of_collision']
        ego_id_list = data['egos_track'].keys()

        for ego_id in ego_id_list:
            if ego_id not in expert_id_time_dict[file_index].keys():
                expert_id_time_dict[file_index][ego_id] = []
                expert_id_time_dict[file_index][ego_id].append(ego_start_end_time)
            else:
                expert_id_time_dict[file_index][ego_id].append(ego_start_end_time)
    
    return expert_id_time_dict

def get_bicycle_model_action(ms_dict, vehicle_length, current_time, next_time, acc_norm_scale=3, steering_norm_scale=30, dt=0.1):
    # bicycle model

    wheelbase_scale = 0.6
    wheelbase = vehicle_length * wheelbase_scale
    gravity_core_scale = 0.4
    f_len = wheelbase * gravity_core_scale
    r_len = wheelbase - f_len

    # action
    current_ms = ms_dict[current_time]
    next_ms = ms_dict[next_time]

    vehicle_current_pos = [current_ms.x, current_ms.y]
    vehicle_current_heading = current_ms.psi_rad
    vehicle_current_speed = math.sqrt(current_ms.vx ** 2 + current_ms.vy ** 2)

    vehicle_next_pos = [next_ms.x, next_ms.y]
    vehicle_next_heading = next_ms.psi_rad
    vehicle_next_speed = math.sqrt(next_ms.vx ** 2 + next_ms.vy ** 2)
    
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

        sin_beta = delta_rad / ((vehicle_current_speed / r_len) * dt)
        beta = math.asin(sin_beta) # rad

    xx = math.tan(beta)
    tan_rad = xx / (r_len / (r_len + f_len))
    rad = math.atan(tan_rad)
    steering = math.degrees(rad)

    delta_speed = vehicle_next_speed - vehicle_current_speed
    acc = delta_speed / dt # m/s^2

    acc_normalize = acc / acc_norm_scale 
    steering_normalize = steering / steering_norm_scale

    # check if it's correct
    # cal_next_heading = vehicle_current_heading + (vehicle_current_speed / r_len) * math.sin(beta) * dt
    # if vehicle_next_heading != cal_next_heading:
    #     print(cal_next_heading, vehicle_next_heading)
    # cal_next_speed = vehicle_current_speed + acc_normalize * 3 * dt
    # if vehicle_next_speed != cal_next_speed:
    #     print(cal_next_speed, vehicle_next_speed)

    return acc_normalize, steering_normalize


def create_demonstrations_pkl_online():
    # pathes
    root_dir = os.path.dirname(os.path.abspath(__file__))
    map_dir = os.path.join(root_dir, "maps")
    tracks_dir = os.path.join(root_dir, "recorded_trackfiles")
    scenario_dir = os.path.join(tracks_dir, "DR_USA_Intersection_EP0")
    id_time_file_dir = os.path.join(root_dir, 'predict_trajectories', 'test_5')

    demonstration_dir = os.path.dirname(os.path.abspath(__file__))
    
    # simulation argument
    sim_parser = argparse.ArgumentParser()
    sim_parser.add_argument("port", type=int, help="Number of the port (int)", default=5557, nargs="?")
    sim_parser.add_argument("scenario_name", type=str, default="DR_USA_Intersection_EP0" ,help="Name of the scenario (to identify map and folder for track "
                            "files)", nargs="?")
    sim_parser.add_argument("total_track_file_number", type=int, help="total number of the track file (int)", default=7, nargs="?")
    sim_parser.add_argument("load_mode", type=str, help="Dataset to load (vehicle, pedestrian, or both)", default="vehicle", nargs="?")         
    sim_parser.add_argument("continous_action", type=bool, help="Is the action type continous or discrete", default=True,nargs="?")

    sim_parser.add_argument("control_steering", type=bool, help="control both lon and lat motions", default=True,nargs="?")
    sim_parser.add_argument("route_type", type=str, help="predict, ground_truth or centerline", default='ground_truth',nargs="?")

    sim_parser.add_argument("visualaztion", type=bool, help="Visulize or not", default=True,nargs="?")
    sim_parser.add_argument("ghost_visualaztion", type=bool, help="Visulize or not", default=True,nargs="?")
    sim_parser.add_argument("route_visualaztion", type=bool, help="Visulize or not", default=True,nargs="?")
    sim_parser.add_argument("route_bound_visualaztion", type=bool, help="Visulize or not", default=False,nargs="?")

    sim_parser.add_argument('--gail', action="store_true", default=False, help='testing single car')
    sim_parser.add_argument('--test', action="store_true", default=False, help='testing single car')
    sim_parser.add_argument('--test_traffic_flow', action="store_true", default=False, help='testing traffic flow')
    sim_args = sim_parser.parse_args()

    # collecting through interaction
    interface = client_interface(sim_args)
    csv_name_list = os.listdir(scenario_dir)
    # corresponding ego vehicles' id and time horizen
    expert_id_time_dict = load_expert_id_time(id_time_file_dir)
    # check
    # total_num = 0
    # for i in range(8):
    #     for vehicle_id in expert_id_time_dict[i].keys():
    #         num = len(expert_id_time_dict[i][vehicle_id])
    #         total_num += num
    # print(total_num)
    
    # interval_distance = 2
    trouble_acc = []
    trouble_steering = []
    for csv_name in csv_name_list:        
        # for each csv, create a demo dict, with the form of: {{vehicle_id:{start_timestamp:[state, action(acc & steering)]...}...}...}s
        demo_dict = dict()
        # load ego's id and time horizen
        track_id = csv_name[-7:-6]
        current_expert_id_time_dict = expert_id_time_dict[int(track_id)]
        print(current_expert_id_time_dict.keys())
        # print(current_expert_id_time_dict)
        # load csv file
        csv_file_name = os.path.join(scenario_dir, csv_name)
        csv_file = read_tracks(csv_file_name)
        print(csv_file.keys())

        # make expert action dict
        vehicle_action_dict = dict() # {id:{start_time:[action1, action2...]}}
        # for vehicle_id, vehicle_dict in csv_file.items():
        for vehicle_id, start_end_time_list in current_expert_id_time_dict.items():
            if vehicle_id not in vehicle_action_dict.keys():
                vehicle_action_dict[vehicle_id] = dict()
            # vehicle basic info
            vehicle_dict = csv_file[vehicle_id]
            vehicle_length = vehicle_dict.length
            vehicle_width = vehicle_dict.width
            ms_dict = vehicle_dict.motion_states

            for start_end_time in start_end_time_list:
                start_timestamp = start_end_time[0]
                end_timestamp = start_end_time[1]
                
                # an action list which stores the action of current vehicle in a specific time horizen
                vehicle_action_list = []
                # fill up the list
                for time, ms in ms_dict.items():
                    if time >= start_timestamp and time <= end_timestamp:
                        next_time = time + 100
                        if next_time in ms_dict.keys():
                            acc_normalize, steering_normalize = get_bicycle_model_action(ms_dict, vehicle_length, time, next_time)
                            action = [acc_normalize, steering_normalize]
                            vehicle_action_list.append(action)
                        else:
                            vehicle_action_list.append(action)

                        vehicle_action_dict[vehicle_id][start_timestamp] = vehicle_action_list

        # get current vehicles' demo through interaction with the env
        for vehicle_id in vehicle_action_dict.keys():
            # vehicle_id = 22
            if vehicle_id not in demo_dict.keys():
                demo_dict[vehicle_id] = dict()
            # collect one expert's experimence at one start timestamp
            vehicle_start_timestamp_list = vehicle_action_dict[vehicle_id].keys()
            for vehicle_start_timestamp in vehicle_start_timestamp_list:
                demo_dict[vehicle_id][vehicle_start_timestamp] = list()
                vehicle_action_list = vehicle_action_dict[vehicle_id][vehicle_start_timestamp]
                while not interface.socket.closed:
                    print(track_id, vehicle_id)
                    # state reset
                    state_dict = interface.reset(track_id, vehicle_id, vehicle_start_timestamp)
                    while True:
                        action_dict = dict()
                        for ego_id, ego_state in state_dict.items():
                            action = vehicle_action_list.pop(0)
                            if abs(action[0]) > 1:
                                print('acc over limit:', action[0])
                                trouble_acc.append(action[0])
                            if abs(action[1]) > 1:
                                print('steering over limit:', action[1])
                                trouble_steering.append(action[1])
                            action_dict[ego_id] = action
                        next_state_dict, _, done_dict, aux_info_dict = interface.step(action_dict)

                        state = list(state_dict.values())[0]
                        demo_dict[vehicle_id][vehicle_start_timestamp].append([state, action])
                        
                        if False not in done_dict.values(): # all egos are done
                            break
                        else:
                            state_dict = next_state_dict
                    break

        demonstration_name = "demonstration_online_" + csv_name[-7:-4]
        demonstration_path = os.path.join(demonstration_dir, demonstration_name)
        with open(demonstration_path, 'wb') as fo:
            pickle.dump(demo_dict, fo)
            fo.close()


    trouble_acc_name = "trouble_acc"
    trouble_steering_name = "trouble_steering"
    trouble_acc_path = os.path.join(demonstration_dir, trouble_acc_name)
    trouble_steering_path = os.path.join(demonstration_dir, trouble_steering_name)
    with open(trouble_acc_path, 'wb') as fo:
        pickle.dump(trouble_acc, fo)
        fo.close()
    with open(trouble_steering_path, 'wb') as fo:
        pickle.dump(trouble_steering, fo)
        fo.close()

# create_demonstrations_pkl_offline(action_type='acc_steer')
create_demonstrations_pkl_online()

    