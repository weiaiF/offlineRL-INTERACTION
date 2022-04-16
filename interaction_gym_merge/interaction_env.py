#-*- coding: UTF-8 -*- 
import sys
sys.path.append("..")
import os
import glob
import copy
import numpy as np
import math
import time
import argparse
import zmq

import lanelet2
import lanelet2_matching

import geometry
import reward
import data_collector

from interaction_map import interaction_map
from ego_vehicle import ego_vehicle
from observation import observation
from utils import dataset_types
from interaction_rl.config import hyperParameters

class interaction_env:
    def __init__(self, args):
        if isinstance(args, dict):
            self.route_type = args['route_type']
            self.scenario_name = args['scenario_name']
            # self.track_file_number = args['track_file_number']
            self._control_steering = args['control_steering']
            self._visualaztion = args['visualaztion']
            self._ghost_visualaztion = args['ghost_visualaztion']
            self._route_visualaztion = args['route_visualaztion']
            self._route_bound_visualaztion = args['route_bound_visualaztion']
            self._continous_action = args['continous_action']
            
        else:
            self.route_type = args.route_type
            self.scenario_name = args.scenario_name
            # self.track_file_number = args.track_file_number
            self._control_steering = args.control_steering
            self._visualaztion = args.visualaztion
            self._ghost_visualaztion = args.ghost_visualaztion
            self._route_visualaztion = args.route_visualaztion
            self._route_bound_visualaztion = args.route_bound_visualaztion
            self._continous_action = args.continous_action
        self._config = hyperParameters(self._control_steering)

        self._map = interaction_map(args, self.route_type)  # load map and track file
        self._delta_time = dataset_types.DELTA_TIMESTAMP_MS                 # 100 ms
        self._start_end_state = None     # ego vehicle start & end state (start_time,end_time,length,width,start motion_state,end motion_state)
        self._ego_vehicle_dict = dict()
        self._ego_route_dict = dict()
        
        self._ego_previous_route_points_dict = dict()
        self._ego_future_route_points_dict = dict()

        self.ego_trajectory_record = dict()

        # 目前场景结束时间选为 所有主车原始轨迹数据集中最早时刻,该值为一个预设值 实际场景结束还需要加入终止条件判断,如 出界\碰撞等
        self._scenario_start_time = None
        self._scenario_end_time = None # select the earliest end time
       
        self._observation = None
        self.current_time_observation = None
        
        self._stepnum = 0


    def __del__(self):
        self._map.__del__()


    def change_predict_track_file(self, trajectory_file_name=None):
        self._map.change_predict_track_file(trajectory_file_name = trajectory_file_name)

        
    def change_ground_truth_track_file(self, track_file_number=None):
        self._map.change_ground_truth_track_file(track_file_number = track_file_number)


    def choose_ego_and_init_map(self, ego_info_dict):
        print('')
        print('map init and choosing ego:')
        self._map.map_init()
        if ego_info_dict['ego_id_list'] is None or len(ego_info_dict['ego_id_list']) == 0:
            # random choose
            print('random choose ego')
            self._map.random_choose_ego_vehicle()
        else:
            # specify choose
            print('specify choose ego')
            self._map.specify_id_choose_ego_vehicle(ego_info_dict['ego_id_list'], ego_info_dict['ego_start_timestamp'])

        if self.route_type == 'predict':
            self._start_end_state = self._map._ego_vehicle_start_end_state_dict  # ego vehicle start & end state dict, key = ego_id, value = (start_time,end_time,length,width,start motion_state,end motion_state)
            self._ego_vehicle_dict.clear()
            self._ego_route_dict.clear()

            for ego_id, start_end_state in self._start_end_state.items():
                self._ego_vehicle_dict[ego_id] = ego_vehicle(ego_id, start_end_state, self._delta_time) # delta_time means tick-time length
                self._ego_route_dict[ego_id] = ego_info_dict['ego_route'][ego_id]
            
            self._scenario_start_time = max([self._start_end_state[i][0] for i in self._start_end_state]) # select the latest start time among all ego vehicle
            self._current_time = self._scenario_start_time
            print('current time:', self._current_time)
            self._scenario_end_time = self._scenario_start_time + 100 * self._config.max_steps # 10s = 100 * 0.1s
            # self._scenario_end_time = min([self._start_end_state[i][1] for i in self._start_end_state]) # select the earliest end time among all ego vehicle

        elif self.route_type == 'ground_truth':
            self._start_end_state = self._map._ego_vehicle_start_end_state_dict  # ego vehicle start & end state dict, key = ego_id, value = (start_time,end_time,length,width,start motion_state,end motion_state)
            self._ego_vehicle_dict.clear()
            self._ego_route_dict.clear()

            self._ground_truth_route = self.get_ground_truth_route(ego_info_dict['ego_start_timestamp'])

            for ego_id, start_end_state in self._start_end_state.items():
                self._ego_vehicle_dict[ego_id] = ego_vehicle(ego_id, start_end_state, self._delta_time) # delta_time means tick-time length
                self._ego_route_dict[ego_id] = self._ground_truth_route[ego_id]
            
            self._scenario_start_time = max([self._start_end_state[i][0] for i in self._start_end_state]) # select the latest start time among all ego vehicle
            self._current_time = self._scenario_start_time
            print('current time:', self._current_time)
            self._scenario_end_time = min([self._start_end_state[i][1] for i in self._start_end_state]) # select the earliest end time among all ego vehicle

        elif self.route_type == 'centerline':
            self._start_end_state = self._map._ego_vehicle_start_end_state_dict  # ego vehicle start & end state dict, key = ego_id, value = (start_time,end_time,length,width,start motion_state,end motion_state)
            self._ego_vehicle_dict.clear()
            self._ego_route_dict.clear()

            self._centerline_route = self.get_centerline_route(ego_info_dict['ego_start_timestamp'])

            for ego_id, start_end_state in self._start_end_state.items():
                self._ego_vehicle_dict[ego_id] = ego_vehicle(ego_id, start_end_state, self._delta_time) # delta_time means tick-time length
                self._ego_route_dict[ego_id] = self._centerline_route[ego_id]
            
            self._scenario_start_time = max([self._start_end_state[i][0] for i in self._start_end_state]) # select the latest start time among all ego vehicle
            self._current_time = self._scenario_start_time
            print('current time:', self._current_time)
            self._scenario_end_time = min([self._start_end_state[i][1] for i in self._start_end_state]) # select the earliest end time among all ego vehicle

        if self._scenario_start_time > self._scenario_end_time:
            print('start time > end time?')
            return False

        self._observation = observation(self._ego_vehicle_dict, self._map, self._config, self._control_steering)
        return self._map._ego_vehicle_id_list


    def reset(self):
        # reset ego state and trajectory record
        ego_state_dict = dict()
        self.ego_trajectory_record.clear()
        self._ego_previous_route_points_dict.clear()
        for ego_id, ego_state in self._start_end_state.items():
            print('ego vehicle id:', ego_id)
            ego_state_dict[ego_id] = ego_state[4] # now it only contains start motion_state: (time_stamp_ms, x, y, vx, vy, psi_rad)
            self.ego_trajectory_record[ego_id] = []
        for ego_id, ego_state in self._ego_vehicle_dict.items():
            print('reset ego state')
            ego_state.reset_state(self._start_end_state[ego_id][4])
            self.ego_trajectory_record[ego_id].append([ego_state._current_state.x, ego_state._current_state.y, ego_state._current_state.vx, ego_state._current_state.vy, ego_state._current_state.psi_rad])
            
        self._current_time = self._scenario_start_time

        # reset(clear) environment observation
        reset_success = self._observation.reset(self.route_type, self._ego_route_dict)
        if not reset_success:
            print('reset failure')
            return None
        
        # visualize map and vehicles
        self._map.update_param(self._current_time, self._scenario_start_time, self._scenario_end_time, ego_state_dict)

        # reset some params of the episode
        self._stepnum = 0
        self._total_stepnum = (self._scenario_end_time - self._scenario_start_time) / self._delta_time
        self._previous_steer = 0 # for calculate steer reward
        
        # get real observation (rl state)
        init_observation_dict = self._observation.get_scalar_observation(self._current_time)

        # make a deepcopy of init_observation
        self.current_time_observation = data_collector.copy_observation(init_observation_dict)

        # draw conflict point
        # self._map.render_conflict_centerline_point(init_observation['following_route_conflict_lanelet'])
        
        self._ego_future_route_points_dict = self._observation.get_future_route_points(init_observation_dict)
        if self._visualaztion:
            # specified ego(with/without ghost ego) and surrounding vehicle highlight
            # surrounding_vehicle_id_list = self._observation.get_surrounding_and_intersection_vehicle_id(init_observation_dict)
            interaction_vehicle_id_list = self._observation.get_intersection_vehicle_id(init_observation_dict)
            if not self._ghost_visualaztion:
                self._map.render_with_highlight(ego_state_dict, interaction_vehicle_id_list)
            else:
                self._map.render_with_highlight_and_ghost(ego_state_dict, interaction_vehicle_id_list)

            # render ego's route
            if self._route_visualaztion:
                if self.route_type == 'predict':
                    self._map.render_route(self._ego_route_dict)
                elif self.route_type == 'ground_truth':
                    # draw init planning route for ego
                    # self._map.render_planning_centerline(self._observation.ego_route_lanelet, self._observation._current_lanelet, self._ego_vehicle_dict)
                    self._map.render_route(self._ego_route_dict)
                    # if self._route_bound_visualaztion:
                    #     # draw route bounds
                    #     self._map.render_route_bounds(self._observation.ego_route_left_bound_points, self._observation.ego_route_right_bound_points)
                    #     # draw closet bound points
                    #     self.previous_bound_points_list = []
                    #     current_bound_points_list = self._observation.get_current_bound_points(init_observation_dict)
                    #     self._map.render_closet_bound_point(self.previous_bound_points_list, current_bound_points_list)
                elif self.route_type == 'centerline':
                    # draw init planning route for ego
                    # self._map.render_planning_centerline(self._observation.ego_route_lanelet, self._observation._current_lanelet, self._ego_vehicle_dict)
                    self._map.render_route(self._ego_route_dict)
                    # if self._route_bound_visualaztion:
                    #     # draw route bounds
                    #     self._map.render_route_bounds(self._observation.ego_route_left_bound_points, self._observation.ego_route_right_bound_points)
                    #     # draw closet bound points
                    #     self.previous_bound_points_list = []
                    #     current_bound_points_list = self._observation.get_current_bound_points(init_observation_dict)
                    #     self._map.render_closet_bound_point(self.previous_bound_points_list, current_bound_points_list)
                
                # render ego route's future points
                self._ego_future_route_points_dict = self._observation.get_future_route_points(init_observation_dict)
                self._map.render_future_route_points(self._ego_previous_route_points_dict, self._ego_future_route_points_dict)

            # ego_id = list(self._ego_vehicle_dict.keys())[0]
            # self._map.save_jpg(ego_id, self._current_time)

        # self._observation.get_visualization_observation(self._current_time,ego_state_dict)

        return init_observation_dict


    def step(self, action_dict):
        print('')
        print('step:', self._stepnum, '/', self._total_stepnum)
        
        ego_state_dict = dict()
        ego_action_dict = dict()

        reward_dict = dict()
        aux_info_dict = dict()

        if self._continous_action:
            for ego_id, action_list in action_dict.items():
                # step as policy action
                if self._control_steering:
                    ego_state, ego_action = self._ego_vehicle_dict[ego_id].step_continuous_action(action_list)
                else:
                    future_route_points_list = self._ego_future_route_points_dict[ego_id]
                    index = int(len(future_route_points_list)/2)
                    next_waypoint_position = [future_route_points_list[index][0], future_route_points_list[index][1]]

                    ego_state, ego_action = self._ego_vehicle_dict[ego_id].step_continuous_action(action_list, next_waypoint_position)

                ego_state_dict[ego_id] = ego_state
                ego_action_dict[ego_id] = ego_action
        
        self._current_time += self._delta_time
        self._stepnum += 1

        # first update map
        self._map.update_param(self._current_time, self._scenario_start_time, self._scenario_end_time, ego_state_dict)
        # then update observation
        next_time_observation_dict = self._observation.get_scalar_observation(self._current_time) 
        
        # calculate rewards and record results based on observation
        done_dict, result_dict = self.reach_terminate_condition(self._current_time, next_time_observation_dict)

        for ego_id in result_dict.keys():
            # record result, trajcetory, ade and fde
            aux_info_dict[ego_id] = dict()
            aux_info_dict[ego_id]['result'] = result_dict[ego_id]

            ego_state = ego_state_dict[ego_id]
            self.ego_trajectory_record[ego_id].append([ego_state.x, ego_state.y, ego_state.vx, ego_state.vy, ego_state.psi_rad])
            aux_info_dict[ego_id]['trajectory'] = self.ego_trajectory_record[ego_id]

            if (self._current_time - self._scenario_start_time) % (10 * self._delta_time) == 0:
                aux_info_dict[ego_id]['ade'] = next_time_observation_dict['trajectory_distance'][ego_id][0]
            else:
                aux_info_dict[ego_id]['ade'] = None
            if done_dict[ego_id] is True:
                aux_info_dict[ego_id]['fde'] = next_time_observation_dict['trajectory_distance'][ego_id][0]
            else:
                aux_info_dict[ego_id]['fde'] = None

            # calculate reward
            # terminal reward
            if aux_info_dict[ego_id]['result'] == 'success':
                terminal_reward = 50 # 0 # +30
            elif aux_info_dict[ego_id]['result'] == 'time_exceed':
                terminal_reward = 0 # -10 # -50
            elif aux_info_dict[ego_id]['result'] == 'collision':
                current_speed_norm = next_time_observation_dict['current_speed'][ego_id][0]/25
                # terminal_reward = -500 * (1 + current_speed_norm)
                terminal_reward = -100
            elif aux_info_dict[ego_id]['result'] == 'deflection':
                terminal_reward = -300
            else:
                terminal_reward = 0
            # step reward
            if self._control_steering:
                position_reward = reward.calculate_lane_keeping_reward(next_time_observation_dict, ego_id)
                steer_reward = reward.calculate_steer_reward(self._previous_steer, ego_action_dict[ego_id].steering)
                self._previous_steer = ego_action_dict[ego_id].steering
            else:
                # position_reward = reward.calculate_trajectory_pos_reward(next_time_observation_dict, ego_id)
                position_reward = 0
                steer_reward = 0

            speed_reward = reward.calculate_speed_reward(next_time_observation_dict, self._control_steering)
            # print(self._start_end_state[ego_id][3])
            ego_width = self._start_end_state[ego_id][3]
            dist_reward = reward.calculate_dist_reward(ego_width, next_time_observation_dict)
            
            env_reward = terminal_reward + speed_reward
            
            reward_dict[ego_id] = env_reward

            print(ego_id, 'env reward:', env_reward)
            print(ego_id, 'terminal_reward:', terminal_reward)
            # print(ego_id, 'dist_reward:', dist_reward)
            print(ego_id, 'speed_reward:', speed_reward)
            # print(ego_id, 'steer_reward:', steer_reward)
            print('speed:', next_time_observation_dict['current_speed'][ego_id][0])

        self._ego_future_route_points_dict = self._observation.get_future_route_points(next_time_observation_dict)
        # render road bound and future route points
        if self._visualaztion:
            # without highlight
            # self._map.render(ego_state_dict)
            
            # specified ego(with/without ghost ego) and surrounding vehicle highlight
            # surrounding_vehicle_id = self._observation.get_surrounding_and_intersection_vehicle_id(next_time_observation)
            interaction_vehicle_id_list = self._observation.get_intersection_vehicle_id(next_time_observation_dict)
            if not self._ghost_visualaztion:
                self._map.render_with_highlight(ego_state_dict, interaction_vehicle_id_list)
            else:
                self._map.render_with_highlight_and_ghost(ego_state_dict, interaction_vehicle_id_list)

            
            if self._route_visualaztion:
                if self._route_bound_visualaztion:
                    current_bound_points_list = self._observation.get_current_bound_points(next_time_observation_dict)
                    self._map.render_closet_bound_point(self.previous_bound_points_list, current_bound_points_list)
                    self.previous_bound_points_list = current_bound_points_list

                # render ego route's future points
                
                self._map.render_future_route_points(self._ego_previous_route_points_dict, self._ego_future_route_points_dict)
                self._ego_previous_route_points_dict = self._ego_future_route_points_dict

            # ego_id = list(self._ego_vehicle_dict.keys())[0]
            # self._map.save_jpg(ego_id, self._current_time)
            
            self.current_time_observation = data_collector.copy_observation(next_time_observation_dict)

        return next_time_observation_dict, reward_dict, done_dict, aux_info_dict

    def reach_terminate_condition(self, current_time, observation_dict):
        # done dict(id: True/False)
        done_dict = dict()

        # result dict(id : result)
        result_dict = dict()
        ego_id_list = observation_dict['ego_shape'].keys()
        ego_id_remove_list = []

        # running by default
        for ego_id in ego_id_list:
            done_dict[ego_id] = False
            result_dict[ego_id] = 'running'

        # reach end time
        if not (current_time + self._delta_time < self._scenario_end_time):
            print('END: reach end time')
            for ego_id in ego_id_list:
                done_dict[ego_id] = True
                result_dict[ego_id] = 'time_exceed'
        # success, collision or deflection 
        else:
            for observation_type, observation_content_dict in observation_dict.items():
                # successfully reach end point ego vehicles
                if observation_type == 'reach_end':
                    for ego_id in ego_id_list:
                        if observation_content_dict[ego_id] is True:
                            print(ego_id, 'Success: reach goal point')
                            done_dict[ego_id] = True
                            result_dict[ego_id] = 'success'
                            ego_id_remove_list.append(ego_id)
                # collision ego vehicles
                elif observation_type == 'collision':
                    for ego_id in ego_id_list:
                        if observation_content_dict[ego_id] is True:
                            print(ego_id, 'Fail: collision')
                            done_dict[ego_id] = True
                            result_dict[ego_id] = 'collision'
                            ego_id_remove_list.append(ego_id)
                # deflection ego vehicles
                elif observation_type == 'deflection':
                    for ego_id in ego_id_list:
                        if observation_content_dict[ego_id] is True:
                            print(ego_id, 'Fail: deflection (distance)')
                            done_dict[ego_id] = True
                            result_dict[ego_id] = 'deflection'
                            ego_id_remove_list.append(ego_id)

            # remove done ego from ego_id_list 
            for ego_remove_id in ego_id_remove_list:
                if ego_remove_id in ego_id_list:
                    ego_id_list.remove(ego_remove_id)

        return done_dict, result_dict


    def set_visualaztion(self, is_visulaztion):
        self._visualaztion = is_visulaztion


    # generate centerline routes
    def get_centerline_route(self, start_timestamp_list):
        centerline_route_dict = dict()
        track_dict = self._map.track_dict
        for vehicle_id in self._start_end_state.keys():
            vehicle_dict = track_dict[vehicle_id]
            
            # time horizen
            if len(start_timestamp_list) != 0:
                start_timestamp = int(start_timestamp_list[0])
                end_timestamp = start_timestamp + 100 * self._config.max_steps - 100
            else:
                start_timestamp = vehicle_dict.time_stamp_ms_first
                end_timestamp = vehicle_dict.time_stamp_ms_last
            # in order to get all of the lanelets
            initial_timestamp = vehicle_dict.time_stamp_ms_first
            terminal_timestamp = vehicle_dict.time_stamp_ms_last

            # get vehicle's whole lanelet
            ms_dict = vehicle_dict.motion_states
            start_lanelet, end_lanelet = self.get_start_end_lanelet_from_ms_dict_with_min_heading(ms_dict, initial_timestamp, terminal_timestamp)

            if not start_lanelet or not end_lanelet or not self._map.routing_graph.getRoute(start_lanelet, end_lanelet, 0):
                initial_lanelet, terminal_lanelet = start_lanelet, end_lanelet
                print('can\'t find route, try to use start time instead of initial time')
                start_lanelet, end_lanelet = self.get_start_end_lanelet_from_ms_dict_with_min_heading(ms_dict, start_timestamp, end_timestamp)
                if not start_lanelet or not end_lanelet or not self._map.routing_graph.getRoute(start_lanelet, end_lanelet, 0):
                    print('still can\'t find route, try to mix them up')
                    start_lanelet, end_lanelet = self.try_to_find_practicable_start_end_lanelet(start_lanelet, initial_lanelet, end_lanelet, terminal_lanelet)
                    if not start_lanelet or not end_lanelet or not self._map.routing_graph.getRoute(start_lanelet, end_lanelet, 0):
                        print('the centerline route doesn\'t exist, using ground truth route')
                        ground_truth_route_dict = self.get_ground_truth_route(start_timestamp_list)
                        return ground_truth_route_dict

            route_lanelet = self.get_route_lanelet(start_lanelet, end_lanelet)

            # get vehicle's route based on whole route lanelet and a specific time horizen
            vehicle_start_pos = [ms_dict[start_timestamp].x, ms_dict[start_timestamp].y]
            vehicle_end_pos = [ms_dict[end_timestamp].x, ms_dict[end_timestamp].y]
            vehicle_route_list = self.get_route_from_lanelet(route_lanelet, vehicle_start_pos, vehicle_end_pos)
            centerline_route_dict[vehicle_id] = vehicle_route_list

        return centerline_route_dict
    
    def try_to_find_practicable_start_end_lanelet(self, start_lanelet_1, start_lanelet_2, end_lanelet_1, end_lanelet_2):
        print(start_lanelet_1, start_lanelet_2, end_lanelet_1, end_lanelet_2)
        start_list = []
        end_list = []
        if start_lanelet_1:
            start_list.append(start_lanelet_1)
            start_lanelet_3_list = self._map.routing_graph.previous(start_lanelet_1)
            if start_lanelet_3_list:
                start_list.append(start_lanelet_3_list[0])
        if start_lanelet_2:
            start_list.append(start_lanelet_1)
            start_lanelet_4_list = self._map.routing_graph.previous(start_lanelet_2)
            if start_lanelet_4_list:
                start_list.append(start_lanelet_4_list[0])
        if end_lanelet_1:
            end_list.append(end_lanelet_1)
            end_lanelet_3_list = self._map.routing_graph.following(end_lanelet_1)
            if end_lanelet_3_list:
                end_list.append(end_lanelet_3_list[0])
        if end_lanelet_2:
            end_list.append(end_lanelet_2)
            end_lanelet_4_list = self._map.routing_graph.following(end_lanelet_2)
            if end_lanelet_4_list:
                end_list.append(end_lanelet_4_list[0])


        for start_lanelet in start_list:
            for end_lanelet in end_list:
                print(self._map.routing_graph.getRoute(start_lanelet, end_lanelet, 0))
                if self._map.routing_graph.getRoute(start_lanelet, end_lanelet, 0):
                    print('yes yes find route')
                    return start_lanelet, end_lanelet
                    
        return start_lanelet, end_lanelet

    def get_start_end_lanelet_from_ms_dict(self, ms_dict, start_timestamp, end_timestamp):
        # get the start and end lanelet set of ego vehicles
        traffic_rules = self._map.traffic_rules
        
        # start lanelet
        ms_initial = ms_dict[start_timestamp]
        vehicle_initial_pos = (ms_initial.x, ms_initial.y)

        obj_start = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_initial_pos[0], vehicle_initial_pos[1], 0), [])
        obj_start_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, obj_start, 0.2)
        obj_start_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(obj_start_matches, traffic_rules)

        if len(obj_start_matches_rule_compliant) > 0:
            # first matching principle
            start_lanelet = obj_start_matches_rule_compliant[0].lanelet

        # end lanelet
        ms_terminal = ms_dict[end_timestamp]
        vehicle_terminal_pos = (ms_terminal.x, ms_terminal.y)
        
        obj_end = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_terminal_pos[0], vehicle_terminal_pos[1], 0), [])
        obj_end_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, obj_end, 0.2)
        obj_end_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(obj_end_matches, traffic_rules)

        if obj_end_matches_rule_compliant:
            end_lanelet = obj_end_matches_rule_compliant[0].lanelet

        return start_lanelet, end_lanelet

    def get_start_end_lanelet_from_ms_dict_with_min_heading(self, ms_dict, start_timestamp, end_timestamp):
        start_lanelet = None
        end_lanelet = None

        traffic_rules = self._map.traffic_rules
        
        # start lanelet
        ms_initial = ms_dict[start_timestamp]
        vehicle_initial_pos = (ms_initial.x, ms_initial.y)
        vehicle_initial_velocity = (ms_initial.vx, ms_initial.vy)

        obj_start = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_initial_pos[0], vehicle_initial_pos[1], 0), [])
        obj_start_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, obj_start, 0.2)
        obj_start_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(obj_start_matches, traffic_rules)
        if len(obj_start_matches_rule_compliant) > 0:
            # similar min heading error matching principle
            min_heading_error = 90
            start_lanelet_index = 0

            for index, match in enumerate(obj_start_matches_rule_compliant):
                match_lanelet = match.lanelet
                heading_error = geometry.get_vehicle_and_lanelet_heading_error(vehicle_initial_pos, vehicle_initial_velocity, match_lanelet, 2)
                if min_heading_error > heading_error:
                    min_heading_error = heading_error
                    start_lanelet_index = index
            start_lanelet = obj_start_matches_rule_compliant[start_lanelet_index].lanelet
        
        # end lanelet
        ms_terminal = ms_dict[end_timestamp]
        vehicle_terminal_pos = (ms_terminal.x, ms_terminal.y)
        vehicle_terminal_velocity = (ms_terminal.vx, ms_terminal.vy)
        
        obj_end = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_terminal_pos[0], vehicle_terminal_pos[1], 0), [])
        obj_end_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, obj_end, 0.2)
        obj_end_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(obj_end_matches, traffic_rules)
        if len(obj_end_matches_rule_compliant) > 0:
            # similar min heading error matching principle
            min_heading_error = 90
            end_lanelet_index = 0

            for index,match in enumerate(obj_end_matches_rule_compliant):
                match_lanelet = match.lanelet
                heading_error = geometry.get_vehicle_and_lanelet_heading_error(vehicle_terminal_pos, vehicle_terminal_velocity, match_lanelet, 2)
                if min_heading_error > heading_error:
                    min_heading_error = heading_error
                    end_lanelet_index = index
            end_lanelet = obj_end_matches_rule_compliant[end_lanelet_index].lanelet

        # if not self._map.routing_graph.getRoute(start_lanelet, end_lanelet, 0):
        #     left_start_lanelet = self._map.routing_graph.lefts(start_lanelet,0)
        #     right_start_lanelet = self._map.routing_graph.lefts(start_lanelet,0)
        #     if left_start_lanelet and self._map.routing_graph.getRoute(left_start_lanelet, end_lanelet, 0):
        #         start_lanelet = left_start_lanelet
        #     elif right_start_lanelet and self._map.routing_graph.getRoute(right_start_lanelet, end_lanelet, 0):
        #         start_lanelet = right_start_lanelet

        return start_lanelet, end_lanelet

    def get_route_lanelet(self, start_lanelet, end_lanelet):
        lanelet_list = []
        if start_lanelet.id == end_lanelet.id:
            lanelet_list.append(start_lanelet)
        else:
            # print(start_lanelet.id, end_lanelet.id)
            lanelet_route = self._map.routing_graph.getRoute(start_lanelet, end_lanelet, 0)
            # print(lanelet_route)
            all_following_lanelet = lanelet_route.fullLane(start_lanelet)
            for lanelet in all_following_lanelet:
                lanelet_list.append(lanelet)
            if lanelet_list[0].id != start_lanelet.id:
                print('error route do not match start lanelet')
            if lanelet_list[-1].id != end_lanelet.id:
                print('error route do not match end lanelet')
                lanelet_list.append(end_lanelet)
        return lanelet_list

    def get_route_from_lanelet(self, route_lanelet, vehicle_start_pos, vehicle_end_pos):
        # we set the max speed of the vehicle as the recommand speed
        recommand_speed = 10 # m/s
        yaw_by_default = 0
        # all centerline points on the whole route
        centerline_point_list = []
        for lanelet in route_lanelet:
            if lanelet is route_lanelet[-1]:
                for index in range(len(lanelet.centerline)):
                    centerline_point_list.append([lanelet.centerline[index].x, lanelet.centerline[index].y, yaw_by_default, recommand_speed, 0]) # recommand_speed = sqrt(recommand_speed**2 + 0**2)
            else:
                for index in range(len(lanelet.centerline)-1):
                    centerline_point_list.append([lanelet.centerline[index].x, lanelet.centerline[index].y, yaw_by_default, recommand_speed, 0])

        # we just need a part of it
        condensed_centerline_point_list = []
        min_distance_with_start = 100
        min_distance_with_end = 100
        for index, point in enumerate(centerline_point_list):
            # find start centerline point's index
            distance_with_start = math.sqrt((point[0] - vehicle_start_pos[0])**2 + (point[1] - vehicle_start_pos[1])**2)
            if distance_with_start < min_distance_with_start:
                min_distance_with_start = distance_with_start
                start_index = index
            # find end centerline point's index
            distance_with_end = math.sqrt((point[0] - vehicle_end_pos[0])**2 + (point[1] - vehicle_end_pos[1])**2)
            if distance_with_end < min_distance_with_end:
                min_distance_with_end = distance_with_end
                end_index = index
        # make sure there are at least two points
        if start_index == end_index:
            end_index += 1

        for index in range(start_index, end_index + 1):
            condensed_centerline_point_list.append(centerline_point_list[index])
        
        # get route from the condensed centerline point list
        route = self.get_route_from_trajectory(condensed_centerline_point_list)

        return route


    # generate ground truth routes
    def get_ground_truth_route(self, start_timestamp_list, interval_distance=2):
        ground_truth_route_dict = dict()
        track_dict = self._map.track_dict
        for vehicle_id in self._start_end_state.keys():
            vehicle_dict = track_dict[vehicle_id]
            # time horizen
            if len(start_timestamp_list) != 0:
                start_timestamp = int(start_timestamp_list[0])
                end_timestamp = start_timestamp + 100 * self._config.max_steps - 100
            else:
                start_timestamp = vehicle_dict.time_stamp_ms_first
                end_timestamp = vehicle_dict.time_stamp_ms_last
            ms_dict = vehicle_dict.motion_states
            vehicle_trajectory_list = self.get_trajectory_from_ms_dict(ms_dict, start_timestamp, end_timestamp)
            if vehicle_trajectory_list:
                vehicle_route_list = self.get_route_from_trajectory(vehicle_trajectory_list, interval_distance)
                ground_truth_route_dict[vehicle_id] = vehicle_route_list

        return ground_truth_route_dict
        
    def get_route_from_trajectory(self, trajectory_list, interval_distance=2):
        # a list [[x, y, point_recommend_speed]]
        # first make them equal distance
        average_trajectory_list = []
        for index, point in enumerate(trajectory_list):
            # first point
            if index == 0:
                average_trajectory_list.append([point[0], point[1]])
            # middle points
            elif index != (len(trajectory_list) - 1):
                point_previous = average_trajectory_list[-1]
                distance_to_previous = math.sqrt((point[0] - point_previous[0])**2 + (point[1] - point_previous[1])**2)
                if distance_to_previous >= 0.75 * interval_distance and distance_to_previous <= 1.25 * interval_distance:
                    average_trajectory_list.append([point[0], point[1]])
                elif distance_to_previous < 0.75 * interval_distance:
                    continue
                elif distance_to_previous > 1.25 * interval_distance:
                    ratio = 1.25 * interval_distance / distance_to_previous
                    insert_point_x = point_previous[0] + ratio * (point[0] - point_previous[0])
                    insert_point_y = point_previous[1] + ratio * (point[1] - point_previous[1])
                    average_trajectory_list.append([insert_point_x, insert_point_y])
            # last point
            else:
                point_previous = average_trajectory_list[-1]
                distance_to_previous = math.sqrt((point[0] - point_previous[0])**2 + (point[1] - point_previous[1])**2)
                if point == point_previous and len(average_trajectory_list) > 1:
                    continue
                else:
                    while distance_to_previous > 1.25 * interval_distance:
                        ratio = 1.25 * interval_distance / distance_to_previous
                        insert_point_x = point_previous[0] + ratio * (point[0] - point_previous[0])
                        insert_point_y = point_previous[1] + ratio * (point[1] - point_previous[1])
                        average_trajectory_list.append([insert_point_x, insert_point_y])

                        point_previous = average_trajectory_list[-1]
                        distance_to_previous = math.sqrt((point[0] - point_previous[0])**2 + (point[1] - point_previous[1])**2)

                    average_trajectory_list.append([point[0], point[1]])

        # then the recommend speed value is the nearest trajectory point's speed value
        average_trajectory_with_speed_list = []
        for point in average_trajectory_list:
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
            average_trajectory_with_speed_list.append([point[0], point[1], point_speed])

        return average_trajectory_with_speed_list

    def get_trajectory_from_ms_dict(self, ms_dict, start_timestamp, end_timestamp):
        # a list [[x, y, vehicle_yaw, vehicle_vx, vehicle_vy]...]
        trajectory_list = []
        # sort mc_dict based on time
        sorted_time = sorted(ms_dict)
        for time in sorted_time:
            if time >= start_timestamp and time <= end_timestamp:
                ms = ms_dict[time]
                trajectory_list.append([ms.x, ms.y, ms.psi_rad, ms.vx, ms.vy])
        # make sure the end point and start point's interval distance is long enough
        if trajectory_list: # if vehicle exist in the time horizen
            start_point = [trajectory_list[0][0], trajectory_list[0][1]]
            end_point = [trajectory_list[-1][0], trajectory_list[-1][1]]
            if math.sqrt((start_point[0] - end_point[0])**2 + (start_point[1] - end_point[1])**2) < 2:
                ms = ms_dict[list(ms_dict.keys())[-1]]
                trajectory_list.append([ms.x, ms.y, ms.psi_rad, ms.vx, ms.vy])

        return trajectory_list

 

class sever_interface:
    def __init__(self, port):
        # communication related
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.port = port
        url = ':'.join(["tcp://*",str(self.port)])
        self.socket.bind(url)
        self.args = None
        self.gym_env = None

        # env statue flag
        self.env_init_flag = False
        self.ego_choose_and_map_init_flag = False
        self.can_change_track_file_flag = False
        self.env_reset_flag = False
        print('server init')
                
    def pop_useless_item(self, observation):
        # remove some useless item from the observation when send them to learning algorithm, to reduce communication costs
        useless_key = ['reach_end', 'collision', 'deflection',
                       'interaction_vehicles_id', 'future_route_points',
                      ]
        
        observation_key = observation.keys()

        for item in useless_key:
            if item in observation_key:
                observation.pop(item)

        return observation

    def start_communication(self):
        while not self.socket.closed:
            message = self.socket.recv()
            print('receive')
            str_message = bytes.decode(message)
            if str_message == 'close':
                self.socket.close()
                return
            print('decoded message:', message)
            message = eval(str_message)

            # env init
            if message['command'] == 'env_init':
                print('env init')
                self.gym_env = interaction_env(message['content'])
                self.args = message['content']
                self.socket.send_string('env_init_done')
                self.env_init_flag = True

            # choose ego & initialize map 
            elif message['command'] == 'ego_map_init':
                print('choose ego and initialize map')
                ego_id_list = self.gym_env.choose_ego_and_init_map(message['content'])
                self.socket.send_string(str(ego_id_list))
                self.ego_choose_and_map_init_flag = True

            # change track file
            elif message['command'] == 'track_init':
                track_type = message['content']['track_type']
                track_content = message['content']['track_content']
                if  track_type == 'predict':
                    self.gym_env.change_predict_track_file(trajectory_file_name=track_content)
                elif track_type == 'ground_truth':
                    self.gym_env.change_ground_truth_track_file(track_file_number=track_content)

                self.socket.send_string('change_file_done')
                self.can_change_track_file_flag = False

            # reset
            elif message['command'] == 'reset':
                print('reset')
                observation_dict = self.gym_env.reset()
                if observation_dict is not None:
                    self.env_reset_flag = True
                    # send_observation = copy.deepcopy(observation)
                    # remove some unuseable item
                    
                    condensed_observation_dict = self.pop_useless_item(observation_dict)
                    reset_message = {'observation': condensed_observation_dict, 'reward':0, 'done':False}
                    self.socket.send_string(str(reset_message).encode())
                    start_time = time.time()
                    
                else:
                    self.ego_choose_and_map_init_flag = False
                    self.socket.send_string(str(self.env_reset_flag).encode())
                
                end_time = time.time()
                print(end_time - start_time)

            # step
            elif message['command'] == 'step':
                action_dict = dict()
                # receiving action
                for ego_id in self.gym_env._ego_vehicle_dict.keys():
                    action_dict[ego_id] = message['content'][ego_id]

                observation_dict, reward_dict, done_dict, aux_info_dict = self.gym_env.step(action_dict)

                if False not in done_dict.values(): # all egos are done
                    self.can_change_track_file_flag = True
                    self.ego_choose_and_map_init_flag = False
                    self.env_reset_flag = False
                if observation_dict is not None:
                    condensed_observation_dict = self.pop_useless_item(observation_dict)  # remove following conflict route lanelet observation
                    step_message = {'observation':condensed_observation_dict, 'reward': reward_dict, 'done': done_dict, 'aux_info': aux_info_dict}
                    self.socket.send_string(str(step_message).encode())

            else:
                print('env_init:', self.env_init_flag)
                print('ego_choose_and_map_init:', self.ego_choose_and_map_init_flag)
                print('can_change_track_file', self.can_change_track_file_flag)
                print('env_reset:', self.env_reset_flag)
                self.socket.send_string('null type')

if __name__ == "__main__":

    # === for docker internal test ===

    # parser = argparse.ArgumentParser()
    # parser.add_argument("scenario_name", type=str, help="Name of the scenario (to identify map and folder for track "
    #                     "files)", nargs="?")
    # parser.add_argument("track_file_number", type=int, help="Number of the track file (int)", default=1, nargs="?")
    # parser.add_argument("load_mode", type=str, help="Dataset to load (vehicle, pedestrian, or both)", default="vehicle",
    #                     nargs="?")         
    # parser.add_argument("continous_action", type=bool, help="Is the action type continous or discrete", default=True,nargs="?")             
    # parser.add_argument("ego_vehicles_num", type=int, help="Number of the ego vehicles", default=1,nargs="?")
    # parser.add_argument("visualaztion", type=bool, help="Visulize or not", default=True,nargs="?")
    # parser.add_argument("--is_imitation_saving_demo", action='store_true',help="Collect demo or not", default="False")
    # args = parser.parse_args()

    # if args.scenario_name is None:
    #     raise IOError("You must specify a scenario. Type --help for help.")
    # if args.load_mode != "vehicle" and args.load_mode != "pedestrian" and args.load_mode != "both":
    #     raise IOError("Invalid load command. Use 'vehicle', 'pedestrian', or 'both'")

    # my_env = interaction_env(args)
    # track_num = my_env._map.track_num
    # ego_num = args.ego_vehicles_num
    # # inter_num = int(track_num / ego_num)
    # # straight_ego_id_list = [59,18,24,75,78,21,35,58,3,2,73]
    # # straight_ego_id_list = [30,31,36,37,42,43,48,49,54,55,58,59,61,69,73,77,82,88,92]
    # # straight_ego_id_list = [1,2,9,25,31,35,47,48,49,51,53,62,64,66,73,76,77,87,88,91,93,94,95,96]
    # straight_ego_id_list = [4,5,6,7,12,24,29,39,58,62,65,69,79,82,83,85,90]
    # # left_ego_id_list = [13,48,72,47,57,45,37,69,20,71,77,30]
    # # right_ego_id_list = [74,9,70,19,40,62,76,14,43,8,44,10,15,62,41,20,68]

    # inter_num = len(straight_ego_id_list)
    # # inter_num = 30
    # for i in range(inter_num):
    #     # ego_id_list = [my_env._map.track_dict.keys()[i+j] for j in range(ego_num)]
    #     ego_id_list = [straight_ego_id_list[i]]    # specified ego id
    #     print(str(i)+'times:')
    #     done = False
    #     if not my_env.choose_ego_and_map_init(ego_id_list):
    #         continue
    #     current_observation = my_env.reset()
    #     if current_observation is None:
    #         print('env_observation is none')
    #         continue
    #     # inner_inter = 0
    #     while not done:
    #     # while not done and inner_inter <10:
    #         # done = True
    #         action_dict = dict()
    #         # calculate action
    #         # inner_inter += 1
    #         for k,v in my_env._ego_vehicle_dict.items():
    #             action_dict[k] = (0,0)
    #         # print(action_dict)
    #         next_observation,env_reward,done = my_env.step(action_dict)

    #         for k,v in my_env._ego_vehicle_dict.items():
    #             print('ego state:',v._current_state.x,v._current_state.y)



    # === for docker external communication test ===
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--port", type=int, help="Number of the port (int)", default=None, nargs="?")
    # args = parser.parse_args()

    sever = sever_interface(5557)
    sever.start_communication()
