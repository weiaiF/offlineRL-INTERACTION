#-*- coding: UTF-8 -*- 
import geometry
import lanelet_relationship
import math
import matplotlib
import numpy as np
import heapq
try:
    import lanelet2
    import lanelet2_matching
    print("Lanelet2_matching import")
except:
    import warnings
    string = "Could not import lanelet2_matching"
    warnings.warn(string)

class observation:
    def __init__(self, ego_vehicles_dict, interaction_map, config, control_steering):
        self._ego_vehicles_dict = ego_vehicles_dict
        self._map = interaction_map

        self._ego_start_lanelet = dict()
        self._ego_end_lanelet = dict()
        self._current_lanelet = dict()

        self.ego_lanelet_route = dict() # {key:route}  ego_route may changes as ego vehicle change, right now the route may lose some lanelet, in case of mistake, use ego_route_lanelet as recommanded
        self.ego_route_lanelet = dict() # {key:list(lanelet)}  the lanelet along ego planning route  use this instead of self.ego_lanelet_route
        self.ego_route_left_lanelet = dict()
        self.ego_route_right_lanelet = dict()
        self.ego_route_dict = dict()
        self.ego_closet_bound_points = dict()

        self.ego_complete_ratio = dict()
        
        # termination judgement
        self.reach_goal = dict()
        self.collision = dict()
        self.deflection = dict()

        # observation terms
        self.trajectory_distance = dict()
        self.trajectory_pos = dict()
        self.trajectory_speed = dict()
        
        self.distance_from_bound = dict()
        self.distance_from_center = dict()
        self.lane_observation = dict()

        self.future_route_points = dict()
        # self.heading_errors = dict()

        self.ego_shape = dict()
        self.ego_route_points = dict()
        self.ego_route_target_speed = dict()
        self.ego_route_left_bound_points = dict()
        self.ego_route_right_bound_points = dict()
        self.ego_current_speed = dict()
        self.ego_current_target_speed = dict()
        self.ego_next_pos = dict()
        self.interaction_vehicles_id = dict()
        self.interaction_vehicles_observation = dict()
        self.attention_mask = dict()

        self.observation_dict = dict()

        # init
        for ego_id in self._ego_vehicles_dict.keys():
            self.reach_goal[ego_id] = False
            self.collision[ego_id] = False
            self.deflection[ego_id] = False

        self.exist_start_and_end = False
        self.exist_route = False
        self.virtual_route_bound = True

        self.lane_dist_limit = 20 # max distance, also for normalization
        self.intersection_dist_limit = 30

        self.min_interval_distance = 2 # 0.5 # minmum interval distance of waypoint (meter)
        self.ttc_nomalization = 10 # 10s
        self.speed_nomalization = 15 # 15m/s

        self.total_ahead_point_num = 50
        
        self._config = config
        self._control_steering = control_steering


    def register_observation_type(self, type):
        if type == 'center_line':
            self.observation_dict['center_line'] = dict()

    def get_start_end_lanelet(self):
        # get the start and end lanelet set of ego vehicles
        for k,v in self._map._ego_vehicle_start_end_state.items():
            obj_start = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(v[4].x, v[4].y, 0), [])
            obj_end = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(v[5].x, v[5].y, 0), [])

            traffic_rules = self._map.traffic_rules

            obj_start_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, obj_start, 0.2)
            obj_start_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(obj_start_matches, traffic_rules)

            obj_end_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, obj_end, 0.2)
            obj_end_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(obj_end_matches, traffic_rules)

            if len(obj_start_matches_rule_compliant)>0:
                # first matching principle
                self._ego_start_lanelet[k] = obj_start_matches_rule_compliant[0].lanelet
                self._current_lanelet[k] = obj_start_matches_rule_compliant[0].lanelet
            else:
                print('can not find ego:' + str(k) + ' start lanelet')
                return False

            if obj_end_matches_rule_compliant:
                self._ego_end_lanelet[k] = obj_end_matches_rule_compliant[0].lanelet
            else:
                print('can not find ego:' + str(k) + ' end lanelet')
                return False

        return True

    def get_start_end_lanelet_with_min_heading_error(self):
        # get the start and end lanelet set of ego vehicles
        for ego_id, ego_info in self._map._ego_vehicle_start_end_state.items():
            ego_start_pos = (ego_info[4].x, ego_info[4].y)
            ego_start_heading = (ego_info[4].vx, ego_info[4].vy)
            ego_end_pos = (ego_info[5].x, ego_info[5].y)
            ego_end_heading = (ego_info[5].vx, ego_info[5].vy)

            obj_start = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(ego_info[4].x, ego_info[4].y, 0), [])
            obj_end = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(ego_info[5].x, ego_info[5].y, 0), [])

            traffic_rules = self._map.traffic_rules

            obj_start_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, obj_start, 0.2)
            obj_start_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(obj_start_matches, traffic_rules)

            obj_end_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, obj_end, 0.2)
            obj_end_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(obj_end_matches, traffic_rules)

            # start lanelet
            if len(obj_start_matches_rule_compliant) > 0:
                # similar min heading error matching principle
                min_heading_error = 90
                start_lanelet_index = 0

                for index, match in enumerate(obj_start_matches_rule_compliant):
                    match_lanelet = match.lanelet
                    heading_error = geometry.get_vehicle_and_lanelet_heading_error(ego_start_pos, ego_start_heading, match_lanelet, self.min_interval_distance)
                    if min_heading_error > heading_error:
                        min_heading_error = heading_error
                        start_lanelet_index = index

                self._ego_start_lanelet[ego_id] = obj_start_matches_rule_compliant[start_lanelet_index].lanelet
                self._current_lanelet[ego_id] = obj_start_matches_rule_compliant[start_lanelet_index].lanelet
            else:
                print('can not find ego:' + str(ego_id) + ' start lanelet')
                return False

            # end lanelet
            if len(obj_end_matches_rule_compliant) > 0:
                # similar min heading error matching principle
                min_heading_error = 90
                end_lanelet_index = 0

                for index,match in enumerate(obj_end_matches_rule_compliant):
                    match_lanelet = match.lanelet
                    heading_error = geometry.get_vehicle_and_lanelet_heading_error(ego_end_pos,ego_end_heading, match_lanelet,self.min_interval_distance)
                    if min_heading_error > heading_error:
                        min_heading_error = heading_error
                        end_lanelet_index = index

                self._ego_end_lanelet[ego_id] = obj_end_matches_rule_compliant[end_lanelet_index].lanelet
            else:
                print('can not find ego:' + str(ego_id) + ' end lanelet')
                return False
                
        return True

    def get_max_occupy_lanelets(self,vehicle_center,vehicle_corner_1,vehicle_corner_2,vehicle_corner_3,vehicle_corner_4):
        # this function is used for position and lanelet matching
        # only when the all corner of the vehicle left the previous occupy lanelet of last time,
        # we think the vehicle left the previous lanelet
        # in case of the sitution that vehicle occupy two lanelet at same time
        center_obj = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_center[0],vehicle_center[1], 0), [])
        corner_1_obj = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_corner_1[0],vehicle_corner_1[1], 0), [])
        corner_2_obj = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_corner_2[0],vehicle_corner_2[1], 0), [])
        corner_3_obj = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_corner_3[0],vehicle_corner_3[1], 0), [])
        corner_4_obj = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_corner_4[0],vehicle_corner_4[1], 0), [])

        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                      lanelet2.traffic_rules.Participants.Vehicle)
          
        # === Match lanelet within 0.2m range ===
        center_obj_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, center_obj, 0.2)
        corner_1_obj_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, corner_1_obj, 0.2)
        corner_2_obj_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, corner_2_obj, 0.2)
        corner_3_obj_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, corner_3_obj, 0.2)
        corner_4_obj_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, corner_4_obj, 0.2)

        center_obj_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(center_obj_matches, traffic_rules)
        corner_1_obj_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(corner_1_obj_matches, traffic_rules)
        corner_2_obj_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(corner_2_obj_matches, traffic_rules)
        corner_3_obj_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(corner_3_obj_matches, traffic_rules)
        corner_4_obj_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(corner_4_obj_matches, traffic_rules)
        # find the lanelet with the max occupy number for 5 points(center and 4 corner points)
        # at least bigger than 3
         
        match_lanelet_id = dict()
        for match in center_obj_matches_rule_compliant:
            key = match.lanelet.id
            if key in match_lanelet_id.keys():
                match_lanelet_id[key][0] += 1
            else:
                match_lanelet_id[key] = [1,match.lanelet]

        for match in corner_1_obj_matches_rule_compliant:
            key = match.lanelet.id
            if key in match_lanelet_id.keys():
                match_lanelet_id[key][0] += 1
            else:
                match_lanelet_id[key] = [1,match.lanelet]

        
        for match in corner_2_obj_matches_rule_compliant:
            key = match.lanelet.id
            if key in match_lanelet_id.keys():
                match_lanelet_id[key][0] += 1
            else:
                match_lanelet_id[key] = [1,match.lanelet]


        for match in corner_3_obj_matches_rule_compliant:
            key = match.lanelet.id
            if key in match_lanelet_id.keys():
                match_lanelet_id[key][0] += 1
            else:
                match_lanelet_id[key] = [1,match.lanelet]
  

        for match in corner_4_obj_matches_rule_compliant:
            key = match.lanelet.id
            if key in match_lanelet_id.keys():
                match_lanelet_id[key][0] += 1
            else:
                match_lanelet_id[key] = [1,match.lanelet]

        main_occupy_limit = 1
        main_occupy_lanelet = []
        for k,v in match_lanelet_id.items():
            if v[0] >= main_occupy_limit:
                main_occupy_lanelet.append(v)

        return main_occupy_lanelet

    def get_scalar_observation(self, current_time):
        if self.exist_start_and_end and self.exist_route:
            for ego_id, ego_state in self._ego_vehicles_dict.items():
                # get ego shape, polygon and motion states
                ego_state_dict = dict()
                self.ego_shape[ego_id] = [ego_state._length, ego_state._width]
                ego_state_dict['polygon'] = self._map.ego_vehicle_polygon[ego_id]
                ego_state_dict['pos'] = [ego_state._current_state.x, ego_state._current_state.y]
                ego_state_dict['speed'] = math.sqrt(ego_state._current_state.vx ** 2 + ego_state._current_state.vy ** 2)
                ego_state_dict['heading'] = ego_state._current_state.psi_rad
                
                # get other vehicless state, first other egos, then the log npcs
                other_vehicles_state_dict = dict()
                for other_ego_id, other_ego_state in self._ego_vehicles_dict.items():
                    if other_ego_id == ego_id:
                        continue
                    else:
                        other_vehicles_state_dict[other_ego_id] = dict()
                        other_vehicles_state_dict[other_ego_id]['polygon'] = self._map.ego_vehicle_polygon[other_ego_id]
                        other_vehicles_state_dict[other_ego_id]['pos'] = [other_ego_state._current_state.x, other_ego_state._current_state.y]
                        other_vehicles_state_dict[other_ego_id]['speed'] = math.sqrt(other_ego_state._current_state.vx ** 2 + other_ego_state._current_state.vy ** 2)
                        other_vehicles_state_dict[other_ego_id]['heading'] = other_ego_state._current_state.psi_rad
                for other_npc_id, other_npc_polygon in self._map.other_vehicle_polygon.items():
                    other_vehicles_state_dict[other_npc_id] = dict()
                    other_npc_motion_state = self._map.other_vehicle_motion_state[other_npc_id]
                    other_vehicles_state_dict[other_npc_id]['pos'] = [other_npc_motion_state.x, other_npc_motion_state.y]
                    other_vehicles_state_dict[other_npc_id]['speed'] = math.sqrt(other_npc_motion_state.vx ** 2 + other_npc_motion_state.vy ** 2)
                    other_vehicles_state_dict[other_npc_id]['heading'] = other_npc_motion_state.psi_rad
                    other_vehicles_state_dict[other_npc_id]['polygon'] = other_npc_polygon

                # get current ego route point
                if self.route_type == 'predict':
                    self.ego_route_points[ego_id] = geometry.get_ego_route_point_with_heading_from_point_list(self.ego_route_dict[ego_id], self.min_interval_distance)
                    self.ego_route_target_speed[ego_id] = geometry.get_ego_target_speed_from_point_list(self.ego_route_dict[ego_id])
                    # do not need lane bound and distance if use predict route (for now)
                    # self.ego_route_left_bound_points[ego_id], self.ego_route_right_bound_points[ego_id] = None, None
                    # self.ego_closet_bound_points[ego_id], self.distance_from_bound[ego_id] = None, None

                elif self.route_type == 'ground_truth':
                    self.ego_route_points[ego_id] = geometry.get_ego_route_point_with_heading_from_point_list(self.ego_route_dict[ego_id], self.min_interval_distance)
                    self.ego_route_target_speed[ego_id] = geometry.get_ego_target_speed_from_point_list(self.ego_route_dict[ego_id])
                    # get current lane bound and distance
                    # self.ego_route_left_bound_points[ego_id], self.ego_route_right_bound_points[ego_id] = geometry.get_route_bounds_points(route_lanelet, self.min_interval_distance)
                    # self.ego_closet_bound_points[ego_id], self.distance_from_bound[ego_id] = geometry.get_closet_bound_point(ego_state_dict['pos'], self.ego_route_left_bound_points[ego_id], self.ego_route_right_bound_points[ego_id])
                
                elif self.route_type == 'centerline':
                    # self.ego_route_points[ego_id] = geometry.get_ego_route_point_with_heading_from_lanelet(route_lanelet, self.min_interval_distance)
                    self.ego_route_points[ego_id] = geometry.get_ego_route_point_with_heading_from_point_list(self.ego_route_dict[ego_id], self.min_interval_distance)
                    self.ego_route_target_speed[ego_id] = geometry.get_ego_target_speed_from_point_list(self.ego_route_dict[ego_id]) # we set the max speed of the vehicle as the target speed
                    # get current lane bound and distance
                    # self.ego_route_left_bound_points[ego_id], self.ego_route_right_bound_points[ego_id] = geometry.get_route_bounds_points(route_lanelet, self.min_interval_distance)
                    # self.ego_closet_bound_points[ego_id], self.distance_from_bound[ego_id] = geometry.get_closet_bound_point(ego_state_dict['pos'], self.ego_route_left_bound_points[ego_id], self.ego_route_right_bound_points[ego_id])

                # get ego's distance with ground truth trajectory (for ade and fde calculation)
                ego_trajectory = self._map._ego_vehicle_track_dict[ego_id]
                ego_trajectory_pos = [ego_trajectory.motion_states[current_time].x, ego_trajectory.motion_states[current_time].y]
                ego_trajectory_velocity = [ego_trajectory.motion_states[current_time].vx, ego_trajectory.motion_states[current_time].vy] 
                trajectory_distance = geometry.get_trajectory_distance(ego_state_dict['pos'], ego_trajectory_pos)
                self.trajectory_distance[ego_id] = [trajectory_distance]
                trajectory_pos = geometry.get_trajectory_pos(ego_state_dict, ego_trajectory_pos)
                self.trajectory_pos[ego_id] = trajectory_pos
                trajectory_speed = geometry.get_trajectory_speed(ego_trajectory_velocity)
                self.trajectory_speed[ego_id] = trajectory_speed

                # ego current speed value
                self.ego_current_speed[ego_id] = [ego_state_dict['speed']]

                # ego distance, heading errors and velocity from route center
                lane_observation, ego_current_target_speed, future_route_points = geometry.get_lane_observation_and_future_route_points(ego_state_dict, self.ego_route_points[ego_id], self.ego_route_target_speed[ego_id], self._control_steering)
                self.lane_observation[ego_id] = lane_observation
                self.ego_current_target_speed[ego_id] = [ego_current_target_speed]
                self.future_route_points[ego_id] = future_route_points
                
                # ego's next position raletive to current
                self.ego_next_pos[ego_id] = geometry.get_ego_next_pos(ego_state_dict)

                # get interaction social vehicles' id and observation
                interaction_vehicles_id, interaction_vehicles_observation, attention_mask = self.get_interaction_vehicles_id_and_observation(ego_state_dict, other_vehicles_state_dict)
                self.interaction_vehicles_id[ego_id] = interaction_vehicles_id
                self.interaction_vehicles_observation[ego_id] = interaction_vehicles_observation
                self.attention_mask[ego_id] = attention_mask

                # Finish judgement 1: reach goal
                goal_point = self.ego_route_points[ego_id][-1]
                reach_goal = self.ego_reach_goal(ego_state_dict, goal_point)
                self.reach_goal[ego_id] = reach_goal
                
                # Finish judgement 2: collision with other vehicles
                ego_collision = self.get_ego_collision(ego_state_dict, other_vehicles_state_dict, interaction_vehicles_id)
                self.collision[ego_id] = ego_collision

                # Finish judgement 3: deflection from current route/road
                if self._control_steering:
                    if self.virtual_route_bound:
                        ego_x_in_route_axis = self.lane_observation[ego_id][0]
                        limitation = 3
                        ego_deflection = self.get_ego_deflection(virtual_route_bound=True, limitation=limitation, distance_to_center=abs(ego_x_in_route_axis))
                    else: # actual route bound
                        ego_min_bound_distance = min(self.distance_from_bound[ego_id])
                        limitation = 0.25
                        ego_deflection = self.get_ego_deflection(virtual_route_bound=False, limitation=limitation, distance_bound=ego_min_bound_distance)
                else:
                    ego_deflection = False
                self.deflection[ego_id] = ego_deflection

            # Finish judgements
            self.observation_dict['reach_end'] = self.reach_goal
            self.observation_dict['collision'] = self.collision
            self.observation_dict['deflection'] = self.deflection

            # Observations - ego state
            self.observation_dict['ego_shape'] = self.ego_shape # 2-D
            self.observation_dict['current_speed'] = self.ego_current_speed      # 1-D
            self.observation_dict['ego_next_pos'] = self.ego_next_pos  # 2-D

            # Observations - others state
            self.observation_dict['interaction_vehicles_observation'] = self.interaction_vehicles_observation  # 25-D
            
            # Observations - route tracking
            self.observation_dict['trajectory_pos'] = self.trajectory_pos              # 2-D
            self.observation_dict['trajectory_speed'] = self.trajectory_speed          # 1-D
            self.observation_dict['trajectory_distance'] = self.trajectory_distance    # 1-D
            self.observation_dict['target_speed'] = self.ego_current_target_speed      # 1-D
            self.observation_dict['distance_from_bound'] = self.distance_from_bound    # 2-D
            self.observation_dict['lane_observation'] = self.lane_observation # 8-D or 5-D
            # self.observation_dict['distance_from_center'] = self.distance_from_center  # 1-D
            # self.observation_dict['heading_errors'] = self.heading_errors    # 10-D
            
            # Observations - attention mask
            self.observation_dict['attention_mask'] = self.attention_mask # 6-D

            # Observations - render
            self.observation_dict['interaction_vehicles_id'] = self.interaction_vehicles_id  # use for render
            # self.observation_dict['current_bound_points'] = self.ego_closet_bound_points     # use for render
            self.observation_dict['future_route_points'] = self.future_route_points                          # use for render
            
            return self.observation_dict


    def get_visualization_observation(self,current_time,ego_state_dict):
        self.get_ego_center_image(current_time, ego_state_dict)

        
    def get_surrounding_and_intersection_vehicle_id(self, current_time_observation):
        surrounding_vehicle_id = []
        for ego_key in self._ego_vehicles_dict.keys():
            if current_time_observation['upper_left_vehicle_id_ms_dict_lanelet'] and current_time_observation['upper_left_vehicle_id_ms_dict_lanelet'][ego_key]:
                upper_left_id = current_time_observation['upper_left_vehicle_id_ms_dict_lanelet'][ego_key][0]
                surrounding_vehicle_id.append(upper_left_id)
            if current_time_observation['lower_left_vehicle_id_ms_dict_lanelet'] and current_time_observation['lower_left_vehicle_id_ms_dict_lanelet'][ego_key]:
                lower_left_id = current_time_observation['lower_left_vehicle_id_ms_dict_lanelet'][ego_key][0]
                surrounding_vehicle_id.append(lower_left_id)
            if current_time_observation['upper_right_vehicle_id_ms_dict_lanelet'] and current_time_observation['upper_right_vehicle_id_ms_dict_lanelet'][ego_key]:
                upper_right_id = current_time_observation['upper_right_vehicle_id_ms_dict_lanelet'][ego_key][0]
                surrounding_vehicle_id.append(upper_right_id)
            if current_time_observation['lower_right_vehicle_id_ms_dict_lanelet'] and current_time_observation['lower_right_vehicle_id_ms_dict_lanelet'][ego_key]:
                lower_right_id = current_time_observation['lower_right_vehicle_id_ms_dict_lanelet'][ego_key][0]
                surrounding_vehicle_id.append(lower_right_id)
            if current_time_observation['previous_vehicle_id_ms_dict_lanelet'] and current_time_observation['previous_vehicle_id_ms_dict_lanelet'][ego_key]:
                prev_id = current_time_observation['previous_vehicle_id_ms_dict_lanelet'][ego_key][0]
                surrounding_vehicle_id.append(prev_id)
            if current_time_observation['following_vehicle_id_ms_dict_lanelet'] and current_time_observation['following_vehicle_id_ms_dict_lanelet'][ego_key]:
                follow_id = current_time_observation['following_vehicle_id_ms_dict_lanelet'][ego_key][0]
                surrounding_vehicle_id.append(follow_id)
            if current_time_observation['front_conflict_vehicle_id_ms_dict_lanelet'] and current_time_observation['front_conflict_vehicle_id_ms_dict_lanelet'][ego_key]:
                front_conflict_id = current_time_observation['front_conflict_vehicle_id_ms_dict_lanelet'][ego_key][0]
                surrounding_vehicle_id.append(front_conflict_id)

        return surrounding_vehicle_id

    # def get_intersection_vehicle_id(self, current_time_observation):
    #     intersection_vehicle_id = []
    #     for ego_key in self._ego_vehicles_dict.keys():
    #         if current_time_observation['current_conflict_vehicle'] and current_time_observation['current_conflict_vehicle'][ego_key]:
    #             current_conflict_id = current_time_observation['current_conflict_vehicle'][ego_key][0]
    #             intersection_vehicle_id.append(current_conflict_id)
    #         if current_time_observation['following_conflict_vehicle'] and current_time_observation['following_conflict_vehicle'][ego_key]:
    #             following_conflict_id = current_time_observation['following_conflict_vehicle'][ego_key][0]
    #             intersection_vehicle_id.append(following_conflict_id)

    #     return intersection_vehicle_id

    def get_intersection_vehicle_id(self, observation_dict):
        intersection_vehicle_id = []
        for ego_id in self._ego_vehicles_dict.keys():
            intersection_vehicle_id += observation_dict['interaction_vehicles_id'][ego_id]

        return intersection_vehicle_id

    def get_future_route_points(self, observation_dict):
        future_route_points_dict = dict()
        for ego_id in self._ego_vehicles_dict.keys():
            future_route_points_dict[ego_id] = observation_dict['future_route_points'][ego_id]

        return future_route_points_dict

    def get_current_bound_points(self, observation_dict):
        current_bound_points = []
        for ego_id in self._ego_vehicles_dict.keys():
            current_bound_points += observation_dict['current_bound_points'][ego_id]

        return current_bound_points

    def reset(self, route_type, route_dict):
        self.route_type = route_type
        self.ego_route_dict = route_dict
        self.ego_complete_ratio.clear()
        # self.ego_step_forward_distance.clear()

        for ego_id in self._ego_vehicles_dict.keys():
            # self.reach_end_lanelet[k] = False
            self.reach_goal[ego_id] = False
            self.collision[ego_id] = False
            self.deflection[ego_id] = False

            self.ego_complete_ratio[ego_id] = 0

        self.trajectory_distance.clear()
        self.trajectory_pos.clear()
        self.trajectory_speed.clear()

        self._current_lanelet.clear()
        if self.ego_lanelet_route: 
            self.ego_lanelet_route.clear()
        self._ego_start_lanelet.clear()
        self._ego_end_lanelet.clear()
        if self.ego_route_lanelet:
            self.ego_route_lanelet.clear()

        self.ego_shape.clear()
        self.ego_route_points.clear()
        self.ego_route_target_speed.clear()
        self.ego_route_left_bound_points.clear()
        self.ego_route_right_bound_points.clear()
        self.ego_closet_bound_points.clear()
        # self.heading_errors.clear()
        self.distance_from_bound.clear()
        self.distance_from_center.clear()
        self.lane_observation.clear()
        self.future_route_points.clear()
        self.ego_next_pos.clear()

        self.ego_current_speed.clear()
        self.ego_current_target_speed.clear()
        self.interaction_vehicles_id.clear()
        self.interaction_vehicles_observation.clear()
        self.attention_mask.clear()

        self.observation_dict.clear()

        self.exist_start_and_end = False
        self.exist_route = False
    
        # self.exist_start_and_end = self.get_start_end_lanelet()  # first matching principle
        self.exist_start_and_end = True # self.get_start_end_lanelet_with_min_heading_error() # min heading matching principle
            
        if self.exist_start_and_end:
            self.exist_route, self.ego_lanelet_route, self.ego_route_lanelet = lanelet_relationship.get_planning_route(self._map, self._ego_start_lanelet, self._ego_end_lanelet) 
            if self.exist_route:
                self.ego_route_left_lanelet, self.ego_route_right_lanelet = lanelet_relationship.get_surrounding_route_along_planning_route(self._map, self.ego_route_lanelet)
                return True
            else:
                print('do not exist route')
                return False
        else:
            print('do not exist start and end')
            return False

    def get_ego_current_lanelet(self):
        for k,v in self._ego_vehicles_dict.items():
            current_lanelet = None
            ego_state = v._current_state

            obj = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(ego_state.x,ego_state.y, 0), [])

            traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                      lanelet2.traffic_rules.Participants.Vehicle)
          
            # === Match lanelet within 0.5m range ===
            obj_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, obj, 0.2)

            obj_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(obj_matches, traffic_rules)

            # print('right now ego vehicle matching rule lanelet num:',len(obj_matches_rule_compliant))

            # print([i.lanelet.id for i in obj_matches_rule_compliant])

            # to slove more than one matching lanelet result problem
            # first we choose the most reasonable one that exisit in ego_route
            # second we choose the surrounding route
            # third we choose the first matching
            if len(obj_matches)>=1:
                # first find matching in ego route lanelet
                for match in obj_matches_rule_compliant:
                    if current_lanelet is not None:
                        break
                    for route_ll in self.ego_route_lanelet[k]:
                        if not isinstance(route_ll,str) and match.lanelet.id == route_ll.id:
                            current_lanelet = match.lanelet
                            break
                # second find matching in surrounding route
                for match in obj_matches_rule_compliant:
                    if current_lanelet is not None:
                        break
                    for route_ll in self.ego_route_left_lanelet[k]:
                        if not isinstance(route_ll,str) and match.lanelet.id == route_ll.id:
                            current_lanelet = match.lanelet
                            break

                for match in obj_matches_rule_compliant:
                    if current_lanelet is not None:
                        break
                    for route_ll in self.ego_route_right_lanelet[k]:
                        if not isinstance(route_ll,str) and match.lanelet.id == route_ll.id:
                            current_lanelet = match.lanelet
                            break
                
                if current_lanelet is None:
                    current_lanelet = obj_matches_rule_compliant[0].lanelet

            else:
                print('ego: ' + str(k) + ' can not find matched lanelet!!!')
                return False

            # print('rule matches lanelet:',obj_matches_rule_compliant[0].lanelet)
            # print('rule matches distance:',obj_matches_rule_compliant[0].distance)

            if current_lanelet:
                # print('lanelet:',type(current_lanelet))
                # print('current lanelet id:',current_lanelet.id)
                self._current_lanelet[k] = current_lanelet
            else:
                print('ego: ' + str(k) + ' can not find current lanelet!!!')
                return False

        return True

    def get_ego_current_lanelet_using_all_corner(self):
        for ego_id in self._ego_vehicles_dict.keys():
            current_lanelet = None

            vehicle_center = self._ego_vehicles_dict[ego_id]._current_state
            vehicle_corner_1 = self._map.ego_vehicle_polygon[ego_id][0]
            vehicle_corner_2 = self._map.ego_vehicle_polygon[ego_id][1]
            vehicle_corner_3 = self._map.ego_vehicle_polygon[ego_id][2]
            vehicle_corner_4 = self._map.ego_vehicle_polygon[ego_id][3]
            center_obj = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_center.x,vehicle_center.y, 0), [])
            corner_1_obj = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_corner_1[0],vehicle_corner_1[1], 0), [])
            corner_2_obj = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_corner_2[0],vehicle_corner_2[1], 0), [])
            corner_3_obj = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_corner_3[0],vehicle_corner_3[1], 0), [])
            corner_4_obj = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_corner_4[0],vehicle_corner_4[1], 0), [])

            traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                      lanelet2.traffic_rules.Participants.Vehicle)
          
            # === Match lanelet within 0.2m range ===
            center_obj_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, center_obj, 0.2)
            corner_1_obj_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, corner_1_obj, 0.2)
            corner_2_obj_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, corner_2_obj, 0.2)
            corner_3_obj_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, corner_3_obj, 0.2)
            corner_4_obj_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, corner_4_obj, 0.2)

            center_obj_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(center_obj_matches, traffic_rules)
            corner_1_obj_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(corner_1_obj_matches, traffic_rules)
            corner_2_obj_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(corner_2_obj_matches, traffic_rules)
            corner_3_obj_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(corner_3_obj_matches, traffic_rules)
            corner_4_obj_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(corner_4_obj_matches, traffic_rules)

            obj_matches_rule_compliant = []
            obj_matches_rule_compliant.extend(center_obj_matches_rule_compliant)
            obj_matches_rule_compliant.extend(corner_1_obj_matches_rule_compliant)
            obj_matches_rule_compliant.extend(corner_2_obj_matches_rule_compliant)
            obj_matches_rule_compliant.extend(corner_3_obj_matches_rule_compliant)
            obj_matches_rule_compliant.extend(corner_4_obj_matches_rule_compliant)
            
            if len(obj_matches_rule_compliant)>=1:
                # first find matching in ego route lanelet
                for match in obj_matches_rule_compliant:
                    if current_lanelet is not None:
                        break
                    for route_ll in self.ego_route_lanelet[ego_id]:
                        if not isinstance(route_ll, str) and match.lanelet.id == route_ll.id:
                            current_lanelet = match.lanelet
                            break
                # second find matching in surrounding route
                for match in obj_matches_rule_compliant:
                    if current_lanelet is not None:
                        break
                    for route_ll in self.ego_route_left_lanelet[ego_id]:
                        if not isinstance(route_ll, str) and match.lanelet.id == route_ll.id:
                            current_lanelet = match.lanelet
                            break
                for match in obj_matches_rule_compliant:
                    if current_lanelet is not None:
                        break
                    for route_ll in self.ego_route_right_lanelet[ego_id]:
                        if not isinstance(route_ll, str) and match.lanelet.id == route_ll.id:
                            current_lanelet = match.lanelet
                            break
             
                if current_lanelet is None:
                    current_lanelet = obj_matches_rule_compliant[0].lanelet

            else:
                print('ego: ' + str(ego_id) + ' can not find matched lanelet!!!')
                return False

            # print('rule matches lanelet:',obj_matches_rule_compliant[0].lanelet)
            # print('rule matches distance:',obj_matches_rule_compliant[0].distance)

            if current_lanelet:
                # print('lanelet:',type(current_lanelet))
                # print('current lanelet id:',current_lanelet.id)
                self._current_lanelet[ego_id] = current_lanelet
            else:
                print('ego: ' + str(ego_id) + ' can not find current lanelet!!!')
                return False

        return True


    def replanning_ego_route(self):
        for k,v in self.ego_route_lanelet.items():

            current_lanelet = self._current_lanelet[k]

            # first need to check whether the ego vehicle is inside the planning route
            # if not, means the ego vehicle change lane，but still in the drivable area(not the wrong direction lane)
            # we need to planning again
            if not geometry.ego_inside_planning_route(current_lanelet,v):
                self.exist_route, self.ego_lanelet_route[k], self.ego_route_lanelet[k] = lanelet_relationship.get_specified_ego_vehicle_replanning_route(self._map,current_lanelet,self._ego_end_lanelet[k]) 
                if self.exist_route:
                    self.ego_route_left_lanelet[k],self.ego_route_right_lanelet[k] = lanelet_relationship.get_specified_ego_vehicle_surrounding_route_along_planning_route(self._map, self.ego_route_lanelet[k])
                else:
                    print('ego vehicle: ' + str(k) + 'can not find replanning route')
                    return False
        
        return True

    def get_center_line_and_previous_following(self):

        center_line_dict = dict()
        following_lanelet_dict = dict()
        previous_lanelet_dict = dict()
        for k, v in self._ego_vehicles_dict.items():

            current_lanelet = self._current_lanelet[k]
            # print('current lanelet:',current_lanelet)

            # # first need to check whether the ego vehicle is inside the planning route
            # if not geometry.ego_inside_planning_route(current_lanelet,self.ego_route_lanelet[k]):
            #     # if not, means the ego vehicle change lane，but still in the drivable area (not the wrong direction lane)
            #     # we need to planning again
            #     self.exist_route, self.ego_lanelet_route, self.ego_route_lanelet = lanelet_relationship.get_planning_route(self._map, self._ego_start_lanelet,self._ego_end_lanelet) 
            #     if self.exist_route:
            #         self.ego_route_left_lanelet,self.ego_route_right_lanelet = lanelet_relationship.get_surrounding_route_along_planning_route(self._map, self.ego_route_lanelet)
            #     else:
            #         return None,None,None

            # print('lanelet:',type(current_lanelet))
            # print('current lanelet id:',current_lanelet.id)
            # print('end lanelet id:',self._ego_end_lanelet[k].id)
            # print('route lanelet list:')
            # for ll in self.ego_route_lanelet[k]:
            #     print('route lanelet:',ll.id)


            # current lanelet centerline
            current_center_line = current_lanelet.centerline
            extend_center_line = geometry.insert_node_to_meet_min_interval(current_center_line, self.min_interval_distance)
            current_center_line_xy = []
            for i in range(len(extend_center_line)):
                current_center_line_xy.append([extend_center_line[i].x, extend_center_line[i].y])

            center_line_dict[k] = current_center_line_xy

            for index, ll in enumerate(self.ego_route_lanelet[k]):
                if ll.id == current_lanelet.id:
                    current_lanelet_index = index

                    if 0 < current_lanelet_index < len(self.ego_route_lanelet[k]) - 1:
                        # print('current lanelet is not the first and last lanelet')
                        following_lanelet = self.ego_route_lanelet[k][current_lanelet_index+1]
                        following_lanelet_dict[k] = following_lanelet

                        previous_lanelet = self.ego_route_lanelet[k][current_lanelet_index-1]
                        previous_lanelet_dict[k] = previous_lanelet
                        

                    elif current_lanelet_index == 0 and current_lanelet_index < len(self.ego_route_lanelet[k]) - 1:
                        # print('current lanelet is the first lanelet')
                        # print('try to get a virtual previous lanelet')
                        following_lanelet = self.ego_route_lanelet[k][current_lanelet_index+1]
                        following_lanelet_dict[k] = following_lanelet

                        previous_lanelet = self._map.routing_graph.previous(current_lanelet)
                        if len(previous_lanelet):
                            previous_lanelet = previous_lanelet[0]
                            previous_lanelet_dict[k] = previous_lanelet
                            # print('virtual previous lanelet id:',previous_lanelet.id)
                            
                    else:
                        # print('current lanelet is the end lanelet')
                        # print('try to get a vitrual following lanelet')
                        previous_lanelet = self.ego_route_lanelet[k][current_lanelet_index-1]
                        previous_lanelet_dict[k] = previous_lanelet

                        following_lanelet = self._map.routing_graph.following(current_lanelet)
                        if len(following_lanelet):
                            following_lanelet = following_lanelet[0]
                            following_lanelet_dict[k] = following_lanelet
                            # print('virtual following lanelet id:',following_lanelet.id)
                            

        return center_line_dict, previous_lanelet_dict, following_lanelet_dict

            # center line point num
            # print(len(current_lanelet.centerline))

    def get_lanelet_closet_vehicles_distance_and_collision(self,related_lanelet_dict,ego_vehicle):
        min_dist = dict()
        dist_limit = 100 # max distance, also for normalization
        # find min_dist vehicle in dist_limit meter circle  min_dist[key] = (normalization_distance,[other_vehicle_id,motion state])
        
        min_dist['lower_left'] = (1,'')
        min_dist['upper_left'] = (1,'')
        min_dist['lower_right'] = (1,'')
        min_dist['upper_right'] = (1,'')
        min_dist['previous'] = (1,'')
        min_dist['following'] = (1,'')
        min_dist['front_conflict'] = (1,'')

        ego_collision = False

        # first determin whether the vehicle belongs to related lanelet
        for k,v in self._map.other_vehicle_polygon.items():
            other_vehicle_center = (v[0] + v[2])/2

            other_vehicle_obj = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(other_vehicle_center[0],other_vehicle_center[1], 0), [])

            traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                      lanelet2.traffic_rules.Participants.Vehicle)
          
            # === Match lanelet within 0.2m range ===
            obj_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, other_vehicle_obj, 0.2)
            # print('obj matches:',obj_matches)
            obj_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(obj_matches, traffic_rules)
            # print('obj matches rule compliant:',obj_matches_rule_compliant)

            # print('obj_matches_rule_compliant:',obj_matches_rule_compliant)
            if len(obj_matches) == 0:
                print('can not find other vehicle current lanelet!!!')
                return None,None

            # other_vehicle_current_lanelet = obj_matches_rule_compliant[0].lanelet
            other_vehicle_current_lanelet = None
            if len(obj_matches)>1:
                other_vehicle_current_lanelet = obj_matches[0].lanelet
                for match in obj_matches:
                    for rll_id in related_lanelet_dict.keys():
                        if match.lanelet.id == rll_id:
                            other_vehicle_current_lanelet = match.lanelet
                            break 
            else:
                other_vehicle_current_lanelet = obj_matches[0].lanelet
            
            # print('other vehicle:' + str(k) + ' ' + str(other_vehicle_current_lanelet.id))

            # === whether is in the surrounding & intersection lanelet collection
            if other_vehicle_current_lanelet.id in related_lanelet_dict.keys():
                # calculate distance or collision

                distance,collision = geometry.ego_other_distance_and_collision(ego_vehicle,v)
                distance = distance / dist_limit # normalization

                if not collision:
                    # following, previous and current need to calculate separately
                    # left, upper left, lower left need to calculate separately
                    # right, upper right, lower right need to calculate separately
                    key = related_lanelet_dict[other_vehicle_current_lanelet.id][0]
                    if key != 'current' and key != 'left' and key != 'right' and key != 'current_conflict' and key != 'following_conflict':
                        if min_dist[related_lanelet_dict[other_vehicle_current_lanelet.id][0]][0] > distance:
                            min_dist[related_lanelet_dict[other_vehicle_current_lanelet.id][0]] = (distance,[k,self._map.other_vehicle_motion_state[k]])
                    elif key == 'current':
                        if geometry.in_front_of_ego_vehicle(ego_vehicle,other_vehicle_center):
                            # in front of ego vehicle in current lanelet
                            if min_dist['following'][0] > distance:
                                min_dist['following'] = (distance,[k,self._map.other_vehicle_motion_state[k]])
                                # print('fo:',k)
                        else:
                            # behind of ego vehicle in current lanelet
                            if min_dist['previous'][0] > distance:
                                min_dist['previous'] = (distance,[k,self._map.other_vehicle_motion_state[k]])
                                # print('pr:',k)
                    elif key == 'left':
                        if geometry.in_front_of_ego_vehicle(ego_vehicle,other_vehicle_center):
                            # in front of ego vehicle in left lanelet
                            if min_dist['upper_left'][0] > distance:
                                min_dist['upper_left'] = (distance,[k,self._map.other_vehicle_motion_state[k]])
                        else:
                            # behind of ego vehicle in left lanelet
                            if min_dist['lower_left'][0] > distance:
                                min_dist['lower_left'] = (distance,[k,self._map.other_vehicle_motion_state[k]])
                    elif key == 'right':
                        if geometry.in_front_of_ego_vehicle(ego_vehicle,other_vehicle_center):
                            # in front of ego vehicle in right lanelet
                            if min_dist['upper_right'][0] > distance:
                                min_dist['upper_right'] = (distance,[k,self._map.other_vehicle_motion_state[k]])
                        else:
                            # behind of ego vehicle in right lanelet
                            if min_dist['lower_right'][0] > distance:
                                min_dist['lower_right'] = (distance,[k,self._map.other_vehicle_motion_state[k]])
                    elif key == 'current_conflict' or key == 'following_conflict':
                        # need careful judgement
                        if min_dist['front_conflict'][0] > distance:
                                min_dist['front_conflict'] = (distance,[k,self._map.other_vehicle_motion_state[k]])
                else:       
                    # distance = 0 when collision
                    min_dist[related_lanelet_dict[other_vehicle_current_lanelet.id][0]] = (0,[k,self._map.other_vehicle_motion_state[k]])
                    ego_collision = True

        return min_dist, ego_collision

    def get_interaction_vehicles_id_and_observation(self, ego_state_dict, other_vehicles_state_dict):
        ego_pos = ego_state_dict['pos']
        ego_heading = ego_state_dict['heading']

        surrounding_vehicles = []
        # 1. check if this vehicle within ego's detective range, and put them together
        ego_detective_range = 30 # m
        for other_id, other_state_dict in other_vehicles_state_dict.items():
            # motion state
            other_vehicle_pos = other_state_dict['pos']
            other_vehicle_speed = other_state_dict['speed']
            other_vehicle_heading = other_state_dict['heading']

            distance_with_ego = math.sqrt((other_vehicle_pos[0] - ego_pos[0])**2 + (other_vehicle_pos[1] - ego_pos[1])**2)
            y_relative = (other_vehicle_pos[1] - ego_pos[1])*np.sin(ego_heading) + (other_vehicle_pos[0] - ego_pos[0])*np.cos(ego_heading)
            if distance_with_ego <= ego_detective_range and y_relative > -12:
                add_dict = {'vehicle_id': other_id, 'distance': distance_with_ego, 'pos': other_vehicle_pos, 'speed': other_vehicle_speed, 'heading': other_vehicle_heading}
                surrounding_vehicles.append(add_dict)

        # 2. get interaction vehicles and their basic observation
        interaction_vehicles = heapq.nsmallest(self._config.npc_num, surrounding_vehicles, key=lambda s: s['distance'])

        # 3. get their ids and full observation
        interaction_vehicles_id = []
        interaction_vehicles_observation = []
        for vehicle_dict in interaction_vehicles:
            # id
            interaction_vehicles_id.append(vehicle_dict['vehicle_id'])
            # basic observation
            # shape
            other_vehicle_polygan = other_vehicles_state_dict[vehicle_dict['vehicle_id']]['polygon']
            poly_01 = [i - j for i, j in zip(other_vehicle_polygan[0], other_vehicle_polygan[1])]
            poly_12 = [i - j for i, j in zip(other_vehicle_polygan[1], other_vehicle_polygan[2])]
            vehicle_length = math.sqrt(poly_01[0]**2 + poly_01[1]**2)
            vehicle_width = math.sqrt(poly_12[0]**2 + poly_12[1]**2)
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
        attention_mask = list(np.ones(self._config.mask_num))
        npc_obs_size = self._config.npc_num * self._config.npc_feature_num
        if len(interaction_vehicles_observation) < npc_obs_size:
            zero_padding_num = int( (npc_obs_size - len(interaction_vehicles_observation)) / self._config.npc_feature_num)
            for _ in range(zero_padding_num):
                attention_mask.pop()
            for _ in range(zero_padding_num):
                attention_mask.append(0)
            while len(interaction_vehicles_observation) < npc_obs_size:
                interaction_vehicles_observation.append(0)

        return interaction_vehicles_id, interaction_vehicles_observation, attention_mask

    def ego_reach_goal(self, ego_state_dict, goal_point):
        ego_loc_x = ego_state_dict['pos'][0]
        ego_loc_y = ego_state_dict['pos'][1]

        goal_loc_x = goal_point[0]
        goal_loc_y = goal_point[1]

        ego_goal_distance = math.sqrt((ego_loc_x - goal_loc_x)**2 + (ego_loc_y - goal_loc_y)**2)
        return ego_goal_distance < 2

    def get_ego_collision(self, ego_state_dict, other_vehicles_state_dict, interaction_vehicles_id):
        ego_collision = False

        for other_id, other_state_dict in other_vehicles_state_dict.items():
            if other_id in interaction_vehicles_id:
                distance, collision = geometry.ego_other_distance_and_collision(ego_state_dict, other_state_dict)
                if collision:
                    return True
        return False

    def get_ego_deflection(self, virtual_route_bound, limitation, distance_bound=None, distance_to_center=None):
        deflection = False
        if virtual_route_bound:
            if distance_to_center > limitation:
                deflection = True
        else:
            if distance_bound < limitation:
                deflection = True
        return deflection
            
    def get_lanelet_closet_vehicles_distance_and_collision_along_route(self,upper_left_route,left,lower_left_route,upper_right_route,right,lower_right_route,previous_route,following_route,current, following_route_conflict_lanelet_dict,following_route_conflict_lanelet_previous_list,previous_route_conflict_lanelet_list,ego_vehicle,ego_polygon):
        min_dist = dict()
    
        # find min_dist vehicle in dist_limit meter circle  
        # min_dist[key] = (normalization_distance,[other_vehicle_id,motion state_dict,related_route_lanelet])
        
        min_dist['lower_left'] = (1,'')
        min_dist['upper_left'] = (1,'')
        min_dist['lower_right'] = (1,'')
        min_dist['upper_right'] = (1,'')
        min_dist['previous'] = (1,'')
        min_dist['following'] = (1,'')
        min_dist['front_conflict'] = (1,'')


        ego_collision = False
        
        # first determine whether the vehicle belongs to related lanelet
        # related lanelet includes left_route lanelet, right_route lanelet, previous_route lanelet following_route lanelet
        for k,v in self._map.other_vehicle_polygon.items():
            is_in_related_route = False
            related_route_key = ''
            related_route_lanelet = None
            other_vehicle_motion_state = self._map.other_vehicle_motion_state[k]
            other_vehicle_pos = (other_vehicle_motion_state.x, other_vehicle_motion_state.y)
            other_vehicle_heading = geometry.get_vehicle_heading(v)
         
            other_vehicle_obj = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(other_vehicle_pos[0],other_vehicle_pos[1], 0), [])
            obj_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, other_vehicle_obj, 0.2)
            lanelet_matches = [m.lanelet for m in obj_matches]

            # if k == 2:
            #     print('vehicle ' + str(k) + ' in :', [i.id for i in lanelet_matches])

            # import pdb
            # pdb.set_trace()
                    
            # obj_matches = self.get_max_occupy_lanelets(other_vehicle_pos,v[0],v[1],v[2],v[3])
            # print('obj matches:',obj_matches)
            # print(type(obj_matches))

            # print('obj_matches_rule_compliant:',obj_matches_rule_compliant)
            if len(lanelet_matches) == 0:
                print('can not find other vehicle: ' + str(k) + ' current lanelet!!!')
                # return None,None
                # right now
                continue

            other_vehicle_current_lanelet = None
            if len(lanelet_matches)>1:
                other_vehicle_current_lanelet = lanelet_matches[0]
                # for surrounding judgement
                for match_lanelet in lanelet_matches:

                    for upper_left_ll in upper_left_route:
                        if isinstance(upper_left_ll,lanelet2.core.ConstLanelet) and match_lanelet.id == upper_left_ll.id and geometry.is_vaild_velocity_direction_along_lanelet(other_vehicle_pos,other_vehicle_heading,upper_left_ll,self.min_interval_distance): 
                            other_vehicle_current_lanelet = match_lanelet
                            is_in_related_route = True
                            related_route_key = 'upper_left'
                            related_route_lanelet = upper_left_ll
                            break
                    
                    if not is_in_related_route:
                        for upper_right_ll in upper_right_route:
                            if isinstance(upper_right_ll,lanelet2.core.ConstLanelet) and match_lanelet.id == upper_right_ll.id and geometry.is_vaild_velocity_direction_along_lanelet(other_vehicle_pos,other_vehicle_heading,upper_right_ll,self.min_interval_distance): 
                                other_vehicle_current_lanelet = match_lanelet
                                is_in_related_route = True
                                related_route_key = 'upper_right'
                                related_route_lanelet = upper_right_ll
                                break
                    
                    if not is_in_related_route:
                        for following_ll in following_route:
                            if isinstance(following_ll,lanelet2.core.ConstLanelet) and match_lanelet.id == following_ll.id and match_lanelet.id != current.id and geometry.is_vaild_velocity_direction_along_lanelet(other_vehicle_pos,other_vehicle_heading,following_ll,self.min_interval_distance):
                                other_vehicle_current_lanelet = match_lanelet
                                is_in_related_route = True
                                related_route_key = 'following'
                                related_route_lanelet = following_ll
                                break

                    if not is_in_related_route:
                        if match_lanelet.id == current.id and geometry.is_vaild_velocity_direction_along_lanelet(other_vehicle_pos,other_vehicle_heading,current,self.min_interval_distance):
                            other_vehicle_current_lanelet = current
                            is_in_related_route = True
                            related_route_key = 'current'
                            related_route_lanelet = current
                    
                    if not is_in_related_route:
                        if isinstance(left,lanelet2.core.ConstLanelet) and match_lanelet.id == left.id and geometry.is_vaild_velocity_direction_along_lanelet(other_vehicle_pos,other_vehicle_heading,left,self.min_interval_distance):
                            other_vehicle_current_lanelet = left
                            is_in_related_route = True
                            related_route_key = 'left'
                            related_route_lanelet = left

                    if not is_in_related_route:
                        if isinstance(right,lanelet2.core.ConstLanelet) and match_lanelet.id == right.id and geometry.is_vaild_velocity_direction_along_lanelet(other_vehicle_pos,other_vehicle_heading,right,self.min_interval_distance):
                            other_vehicle_current_lanelet = right
                            is_in_related_route = True
                            related_route_key = 'right'
                            related_route_lanelet = right
                    
                    # for any previous(previous,lower right,lower left) judgement
                    # make sure the match lanelet only contain previous lanelet, do not contain any previous conflict lanelet 
                    # in other words, other vehicle must be inside the ego vehicle's previous route, not the crossing conflict situation
                    
                    # is_in_previous_conflict_ego_route = False
                    # for match in obj_matches:
                    #     if is_in_previous_conflict_ego_route:
                    #         break
                    #     else:
                    #         for previous_ll in previous_route_conflict_lanelet_list:
                    #             if match.lanelet.id == previous_ll.id:
                    #                 is_in_previous_conflict_ego_route = True
                    #                 break
                    
                    # if not is_in_previous_conflict_ego_route:

                    if not is_in_related_route:
                        for lower_left_ll in lower_left_route:
                            if isinstance(lower_left_ll,lanelet2.core.ConstLanelet) and match_lanelet.id == lower_left_ll.id and geometry.is_vaild_velocity_direction_along_lanelet(other_vehicle_pos,other_vehicle_heading,lower_left_ll,self.min_interval_distance): 
                                other_vehicle_current_lanelet = match_lanelet
                                is_in_related_route = True
                                related_route_key = 'lower_left'
                                related_route_lanelet = lower_left_ll
                                break
                
                    if not is_in_related_route:
                        for lower_right_ll in lower_right_route:
                            if isinstance(lower_right_ll,lanelet2.core.ConstLanelet) and match_lanelet.id == lower_right_ll.id and geometry.is_vaild_velocity_direction_along_lanelet(other_vehicle_pos,other_vehicle_heading,lower_right_ll,self.min_interval_distance): 
                                other_vehicle_current_lanelet = match_lanelet
                                is_in_related_route = True
                                related_route_key = 'lower_right'
                                related_route_lanelet = lower_right_ll
                                break
                    
                    if not is_in_related_route:
                        for previous_ll in previous_route:
                            if isinstance(previous_ll,lanelet2.core.ConstLanelet) and match_lanelet.id == previous_ll.id and geometry.is_vaild_velocity_direction_along_lanelet(other_vehicle_pos,other_vehicle_heading,previous_ll,self.min_interval_distance):
                                other_vehicle_current_lanelet = match_lanelet
                                is_in_related_route = True
                                related_route_key = 'previous'
                                related_route_lanelet = previous_ll
                                break

        
                # ==============================for intersection conflict judgement===========================================================================
                # make sure the match lanelet only contain conflict lanelet, do not contain any ego vehicle route lanelet (is_in_following_ego_route = False )
                # in other words, other vehicle must be outside the ego vehicle's route (the inside crossing situation is consider as following situation)
                if not is_in_related_route:
                    is_in_following_ego_route = False
                    for match_lanelet in lanelet_matches:
 
                        if is_in_following_ego_route:
                            break
                        else:
                            for following_ll in following_route:
                                if match_lanelet.id == following_ll.id:
                                    is_in_following_ego_route = True
                                    other_vehicle_current_lanelet = match_lanelet
                                    is_in_related_route = True
                                    related_route_key = 'following'
                                    related_route_lanelet = following_ll
                                    break
                    # make sure other vehicle is in front of the conflict region
                    if not is_in_following_ego_route:
                        for match_lanelet in lanelet_matches:

                            for f_ll_id,c_ll_pair in following_route_conflict_lanelet_dict.items():
                                for c_ll in c_ll_pair[1]:
                                    if match_lanelet.id == c_ll.id:
                                        # print('enter judgement id:',k)
                                        # is in front or inside the overlap region, also has a vaild speed direction
                                        # print('other id: ',k,' is in conflict region')
                                        if geometry.is_in_front_of_overlap_region(c_ll_pair[0],c_ll,other_vehicle_pos,self.min_interval_distance) and geometry.is_vaild_velocity_direction_along_lanelet(other_vehicle_pos,other_vehicle_heading,c_ll,self.min_interval_distance):
                                        # if geometry.is_in_front_of_overlap_region(c_ll_pair[0],c_ll,other_vehicle_pos,self.min_interval_distance):
                                            is_in_related_route = True
                                            related_route_key = 'front_conflict'
                                            related_route_lanelet = c_ll
 
            else:   
                other_vehicle_current_lanelet = lanelet_matches[0]

            # print 'other vehicle: ' + str(k) + ' ' + str(other_vehicle_current_lanelet.id)

            if is_in_related_route:
                # print('other vehicle: ' + str(k) + ' related')
                distance, collision = geometry.ego_other_distance_and_collision(ego_vehicle, v)
                # print('distance:',distance)
                if related_route_key != 'front_conflict':
                    distance = distance / self.lane_dist_limit # normalization
                else:
                    distance = distance / self.intersection_dist_limit
                # distance = distance / lane_dist_limit # normalization

                if not collision:
                    if related_route_key != 'current' and related_route_key != 'left' and related_route_key != 'right':
                        if min_dist[related_route_key][0] > distance:
                            min_dist[related_route_key] = (distance,[k,self._map.other_vehicle_motion_state[k].get_dict_type_data(),related_route_lanelet])
                    elif related_route_key == 'left':
                        if geometry.in_front_of_ego_vehicle(ego_vehicle,ego_polygon,other_vehicle_pos):
                            # in front of ego vehicle in left lanelet
                            if min_dist['upper_left'][0] > distance:
                                min_dist['upper_left'] = (distance,[k,self._map.other_vehicle_motion_state[k].get_dict_type_data(),related_route_lanelet])
                        else:
                            # behind of ego vehicle in left lanelet
                            if min_dist['lower_left'][0] > distance:
                                min_dist['lower_left'] = (distance,[k,self._map.other_vehicle_motion_state[k].get_dict_type_data(),related_route_lanelet])
                    elif related_route_key == 'right':
                        if geometry.in_front_of_ego_vehicle(ego_vehicle,ego_polygon,other_vehicle_pos):
                            # in front of ego vehicle in right lanelet
                            if min_dist['upper_right'][0] > distance:
                                min_dist['upper_right'] = (distance,[k,self._map.other_vehicle_motion_state[k].get_dict_type_data(),related_route_lanelet])
                        else:
                            # behind of ego vehicle in right lanelet
                            if min_dist['lower_right'][0] > distance:
                                min_dist['lower_right'] = (distance,[k,self._map.other_vehicle_motion_state[k].get_dict_type_data(),related_route_lanelet])
                    elif related_route_key == 'current':
                        if geometry.in_front_of_ego_vehicle(ego_vehicle,ego_polygon,other_vehicle_pos):
                            # in front of ego vehicle in current lanelet
                            if min_dist['following'][0] > distance:
                                min_dist['following'] = (distance,[k,self._map.other_vehicle_motion_state[k].get_dict_type_data(),related_route_lanelet])
                                # print('fo:',k)
                        else:
                            # behind of ego vehicle in current lanelet
                            if min_dist['previous'][0] > distance:
                                min_dist['previous'] = (distance,[k,self._map.other_vehicle_motion_state[k].get_dict_type_data(),related_route_lanelet])
                else:       
                    # distance = 0 when collision
                    min_dist[related_route_key] = (0, [k,self._map.other_vehicle_motion_state[k].get_dict_type_data(), related_route_lanelet])
                    ego_collision = True

        # print('min_dist front conflict:',min_dist['front_conflict'])

        return min_dist, ego_collision

    # def get_ego_center_image(self,current_time,ego_state_dict):
    #     image = np.asarray(self._map.fig.canvas.buffer_rgba())
    #     image_array = image.reshape(self._map.fig_height,self._map.fig_width,4)
    #     for k,v in ego_state_dict.items():
    #         ego_center_pixel_x = int((v.x - self._map.map_x_bound[0]) / self._map.map_width_ratio)
    #         ego_center_pixel_y = int((v.y - self._map.map_y_bound[0]) / self._map.map_height_ratio)
    #         image_min_x = ego_center_pixel_x - self._map.image_half_width
    #         image_max_x = ego_center_pixel_x + self._map.image_half_width
    #         image_min_y = ego_center_pixel_y - self._map.image_half_height
    #         image_max_y = ego_center_pixel_y + self._map.image_half_height
    #         ego_center_image = image_array[self._map.fig_height - image_max_y:self._map.fig_height- image_min_y,image_min_x:image_max_x,:].tolist()
    #         # matplotlib.image.imsave('ego_'+ str(k) + '_' + str(current_time) +'.png', ego_center_image)  
    #     # matplotlib.image.imsave('map.png', image_array.tolist()) 

    # def get_rotate_ego_center_image(self,current_time,ego_state_dict):
    #     pass

    def calculate_inlane_ttc(self, min_dist, infront_of_ego, ego_vehicle_ms, ego_vehicle_polygon, other_vehicle_ms_dict, other_vehicle_polygon):
        # in case the rel_speed is zero, ttc = 10
        ego_heading_vector = geometry.get_vehicle_heading(ego_vehicle_polygon)
        ego_speed_vector = np.array([ego_vehicle_ms.vx,ego_vehicle_ms.vy])

        other_heading_vector = geometry.get_vehicle_heading(other_vehicle_polygon)
        other_speed_vector = np.array([other_vehicle_ms_dict['vx'],other_vehicle_ms_dict['vy']])
        
        
        L_ego_heading = np.sqrt(ego_heading_vector.dot(ego_heading_vector))
        L_other_heading = np.sqrt(other_heading_vector.dot(other_heading_vector))
        
        same_direction = False
        # check whether the other vehicle is in front of ego
        if infront_of_ego:
            # project other_heading_vector along ego_heading_vector 
            project_speed_vector = other_speed_vector.dot(ego_heading_vector) / L_ego_heading
            rel_speed_vector = ego_speed_vector - project_speed_vector 
            # print('rel_speed_vector: ',rel_speed_vector)
            L_rel_speed = np.sqrt(rel_speed_vector.dot(rel_speed_vector))

            if L_rel_speed == 0:
                ttc = 1
                return 1
            # whether the rel_speed is positive along other  
            cos_angle = rel_speed_vector.dot(ego_heading_vector)/(L_rel_speed*L_ego_heading)

            radian = np.arccos(cos_angle)
            
            angle = radian * 180 / np.pi
            if abs(angle) <90:
                # rel speed is positive, dist decrese 
                same_direction = True

        else:
            # project ego_heading_vector along other_heading_vector 
            project_speed_vector = ego_speed_vector.dot(other_heading_vector) / L_other_heading
            rel_speed_vector = other_heading_vector - project_speed_vector
            # print('rel_speed_vector: ',rel_speed_vector)
            L_rel_speed = np.sqrt(rel_speed_vector.dot(rel_speed_vector))

            if L_rel_speed == 0:
                ttc = 1
                return 1
            # whether the rel_speed is positive along ego   
            cos_angle = rel_speed_vector.dot(other_heading_vector)/(L_rel_speed*L_other_heading)

            radian = np.arccos(cos_angle)

            angle = radian * 180 / np.pi
            if abs(angle) <90:
                # rel speed is positive, dist decrese 
                same_direction = True

        # as the min_dist is normalized
        # we need to re-scaled this value
        min_dist = min_dist * self.lane_dist_limit
        
        # print('rel speed: ',np.linalg.norm(rel_speed_vector))
        if same_direction:
            ttc = min_dist / np.linalg.norm(rel_speed_vector)
        else:
            # ttc = - min_dist / np.linalg.norm(rel_speed_vector)
            ttc = self.ttc_nomalization

        ttc /= self.ttc_nomalization

        # we need to make sure ttc is positive also not larger than 1
        if ttc > 1:
            ttc = 1

        if np.isnan(ttc):
            print('ttc is nan, let it be one')
            ttc = 1

        # print('min_dist: ',min_dist)
        # print('ttc: ',ttc)

        return ttc
        
     
    def calculate_surroundlane_ttc(self, min_dist, infront_of_ego, ego_vehicle_ms, other_vehicle_ms_dict):
        ego_speed_vector = np.array([ego_vehicle_ms.vx, ego_vehicle_ms.vy])
        other_speed_vector = np.array([other_vehicle_ms_dict['vx'], other_vehicle_ms_dict['vy']])


        # check whether the other vehicle is in front of ego
        if infront_of_ego:
            rel_speed_vector = ego_speed_vector - other_speed_vector
            mass_center_vector = np.array([other_vehicle_ms_dict['x'] - ego_vehicle_ms.x,other_vehicle_ms_dict['y'] - ego_vehicle_ms.y])
        else:
            rel_speed_vector = other_speed_vector - ego_speed_vector
            mass_center_vector = np.array([ego_vehicle_ms.x - other_vehicle_ms_dict['x'] ,ego_vehicle_ms.y - other_vehicle_ms_dict['y']])
        # determine whether there is a possibility of collision 
        L_rel_speed = np.sqrt(rel_speed_vector.dot(rel_speed_vector))
        if L_rel_speed == 0:
            return 1
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
            # rel_speed_along_ceneter_vector = rel_speed_vector.dot(mass_center_vector) / L_mass_center
            # ttc = - min_dist / np.linalg.norm(rel_speed_along_ceneter_vector)
            # # normalized into 10s
            # ttc /= self.ttc_nomalization   # need to fix
            ttc = 1

        # we need to make sure ttc is positive also not larger than 1
        if ttc > 1:
            ttc = 1

        if np.isnan(ttc):
            print('ttc is nan, let it be one')
            ttc = 1

        return ttc


    def calculate_intersection_ttc(self,ego_vehicle_ms_dict,other_vehicle_ms_dict,ego_current_lanelet,ego_following_route,ego_conflict_lanelet,other_conflict_lanelet):
        # === get the conflict_point of two lanelet ===
        print('calculate conflict point')
        print('ego lanelet:',ego_conflict_lanelet.id)
        print('other lanelet:',other_conflict_lanelet.id)
        conflict_point_list =  geometry.get_overlap_point_list(ego_conflict_lanelet,other_conflict_lanelet)
        conflict_point = conflict_point_list[0]


        # === calculate the time reach the conflict point ===

        ego_current_lanelet_to_conflict_point_length = 0
        current_lanelet_index = 0
        for ll in ego_following_route:
            if ll.id == ego_current_lanelet.id and ll.id == ego_conflict_lanelet.id:
                # current lanelet is the conflict lanelet
                ego_current_lanelet_to_conflict_point_length += geometry.get_centerline_length_between_vehicle_and_conflict_point(ego_vehicle_ms_dict,conflict_point,ego_current_lanelet,self.min_interval_distance)
                break
            elif ll.id == ego_current_lanelet.id and ll.id != ego_conflict_lanelet.id:
                ego_current_lanelet_to_conflict_point_length += geometry.get_centerline_length_ahead_vehicle(ego_vehicle_ms_dict,ego_current_lanelet,self.min_interval_distance)
            elif ll.id == ego_conflict_lanelet.id and ll.id != ego_current_lanelet.id:
                ego_current_lanelet_to_conflict_point_length += geometry.get_centerline_length_to_conflict_point(conflict_point,ll,self.min_interval_distance)
                break
            else:
                ego_current_lanelet_to_conflict_point_length += geometry.lanelet_length(ll)

         
        other_lanelet_to_conflict_point_length = geometry.get_centerline_length_between_vehicle_and_conflict_point(other_vehicle_ms_dict,conflict_point,other_conflict_lanelet,self.min_interval_distance)


        # calculate the velocity project along frenet coordinate
        ego_vel_along_frenet = geometry.get_velocity_vector_along_frenet_coordinate(ego_vehicle_ms_dict,ego_current_lanelet,self.min_interval_distance)

        other_vel_along_frenet = geometry.get_velocity_vector_along_frenet_coordinate(other_vehicle_ms_dict,other_conflict_lanelet,self.min_interval_distance)

        # in case of speed = 0
        if ego_vel_along_frenet == 0:
            ttc = 1
            return ttc
        else:
            time_of_ego_to_conflict_point = ego_current_lanelet_to_conflict_point_length / ego_vel_along_frenet

            if other_vel_along_frenet == 0:
                ttc = 1
                return ttc
            else:
                time_of_other_to_conflict_point = other_lanelet_to_conflict_point_length / other_vel_along_frenet

                # ttc is equal to the difference of two time (the value must be postive not negative)
                if time_of_ego_to_conflict_point > time_of_other_to_conflict_point:
                    ttc = time_of_ego_to_conflict_point - time_of_other_to_conflict_point
                    ttc /= self.ttc_nomalization
                else:
                    ttc = time_of_other_to_conflict_point - time_of_ego_to_conflict_point
                    ttc /= self.ttc_nomalization

                # we need to make sure ttc is positive also not larger than 1
                if ttc > 1:
                    ttc = 1

                return ttc



            


