import csv
import pickle
import os
import numpy as np
import copy
import math


class Key:
    track_id = "track_id"
    other_id = "other_id"
    relationship = 'relationship'
    interac_ttime_stamp = 'interact_time_stamp'
    start_time_stamp = 'start_time_stamp'
    end_time_stamp = 'end_time_stamp'
  
class KeyEnum:
    track_id = 0
    other_id = 1
    relationship = 2
    interac_ttime_stamp = 3
    start_time_stamp = 4
    end_time_stamp = 5

def find_and_save_critical_ttc_track(ttc_threshold,observation,current_time,ego_track_dict,other_track_dict,upper_left_id_dict,lower_left_id_dict,upper_right_id_dict,lower_right_id_dict,previous_id_dict,following_id_dict,front_conflict_id_dict):
    # save track of ttc in oberservation is less than ttc_threshold
    # we need to save the ego vehicle id, the other vehicle id pair
    # make sure the track duration of both vehicle in critical pair satisfied the duration threshold
    duration_threshold = 5  # 3s duration limit

    upper_left_vehicle_ttc = observation['upper_left_vehicle_ttc']
    lower_left_vehicle_ttc = observation['lower_left_vehicle_ttc']
    upper_right_vehicle_ttc = observation['upper_right_vehicle_ttc']
    lower_right_vehicle_ttc = observation['lower_right_vehicle_ttc']
    previous_vehicle_ttc = observation['previous_vehicle_ttc']
    following_vehicle_ttc = observation['following_vehicle_ttc']

    front_conflict_vehicle = observation['front_conflict_vehicle_id_ms_dict']

    print('front_conflict_vehicle:',front_conflict_vehicle)
    # print('ego_track_dict:',ego_track_dict)

    # for key in upper_left_vehicle_ttc.keys():
    #     if 0< upper_left_vehicle_ttc[key] < ttc_threshold:
    #         upper_left_id = observation['upper_left_vehicle_id_ms'][key][0]
    #         if upper_left_id not in upper_left_id_list[key]:
    #             upper_left_id_list[key].append(upper_left_id)

        # if 0< lower_left_vehicle_ttc[key] < ttc_threshold:
        #     lower_left_id = observation['lower_left_vehicle_id_ms'][key][0]
        #     if lower_left_id not in lower_left_id_list[key]:
        #         lower_left_id_list[key].append(lower_left_id)
                
        # if 0< upper_right_vehicle_ttc[key] < ttc_threshold:
        #     upper_right_id = observation['upper_right_vehicle_id_ms'][key][0]
        #     if upper_right_id not in upper_right_id_list[key]:
        #         upper_right_id_list[key].append(upper_right_id)

        # if 0< lower_right_vehicle_ttc[key] < ttc_threshold:
        #     lower_right_id = observation['lower_right_vehicle_id_ms'][key][0]
        #     if lower_right_id not in lower_right_id_list[key]:
        #         lower_right_id_list[key].append(lower_right_id)
    for ego_id in previous_vehicle_ttc.keys():
        if 0< previous_vehicle_ttc[ego_id] < ttc_threshold:
            previous_id = observation['previous_vehicle_id_ms_dict'][ego_id][0]
            if previous_id not in previous_id_dict[ego_id].keys():
                is_meet,start,end = is_meet_duration_require(ego_track_dict[ego_id],other_track_dict[previous_id],current_time,duration_threshold)
                if is_meet:
                    previous_id_dict[ego_id][previous_id] = [current_time,start,end]

    for ego_id in following_vehicle_ttc.keys():
        if 0< following_vehicle_ttc[ego_id] < ttc_threshold:
            # print('following_vehicle_id_ms: ',observation['following_vehicle_id_ms'])
            following_id = observation['following_vehicle_id_ms_dict'][ego_id][0]
            if following_id not in previous_id_dict[ego_id].keys():
                is_meet,start,end = is_meet_duration_require(ego_track_dict[ego_id],other_track_dict[following_id],current_time,duration_threshold)
                if is_meet:
                    following_id_dict[ego_id][following_id] = [current_time,start,end]

    for ego_id in front_conflict_vehicle.keys():
        if observation['front_conflict_vehicle_id_ms_dict'][ego_id]:
            front_conflict_id = observation['front_conflict_vehicle_id_ms_dict'][ego_id][0]
            if front_conflict_id not in previous_id_dict[ego_id].keys():
                is_meet,start,end = is_meet_duration_require(ego_track_dict[ego_id],other_track_dict[front_conflict_id],current_time,duration_threshold)
                if is_meet:
                    front_conflict_id_dict[ego_id][front_conflict_id] = [current_time,start,end]

def is_meet_duration_require(ego_track,other_track,current_time,duration_threshold):
    ego_track_start_time = ego_track.time_stamp_ms_first
    ego_track_end_time = ego_track.time_stamp_ms_last
    other_track_start_time = other_track.time_stamp_ms_first
    other_track_end_time = other_track.time_stamp_ms_last

    share_start_time = None
    share_end_time = None

    if ego_track_start_time <= other_track_start_time and ego_track_end_time <= other_track_end_time:
        share_time = ego_track_end_time - other_track_start_time
        share_start_time = other_track_start_time
        share_end_time = ego_track_end_time
    elif ego_track_start_time >= other_track_start_time and ego_track_end_time <= other_track_end_time:
        share_time = ego_track_end_time - ego_track_start_time
        share_start_time = ego_track_start_time
        share_end_time = ego_track_end_time
    elif ego_track_start_time >= other_track_start_time and ego_track_end_time >= other_track_end_time:
        share_time = other_track_end_time - ego_track_start_time
        share_start_time = ego_track_start_time
        share_end_time = other_track_end_time
    elif ego_track_start_time < other_track_start_time and ego_track_end_time > other_track_end_time:
        share_time = other_track_end_time - other_track_start_time
        share_start_time = other_track_start_time
        share_end_time = other_track_end_time


    if not share_time < duration_threshold:
        return True,share_start_time,share_end_time
    else:
        return False,None,None

def save_track_id_pair_to_dataset(csv_file_name,upper_left_id_dict,lower_left_id_dict,upper_right_id_dict,lower_right_id_dict,previous_id_dict,following_id_dict,front_conflict_id_dict):
    error_string = ""
    # dir_path = os.path.dirname(os.path.abspath(__file__))
    # csv_path = os.path.join(dir_path, csv_file_name)
    # if not os.path.isfile(csv_path):
    #     print('do not exist csv file, creat new one')
    #     with open(csv_file_name + '.csv','ab')as f:
    #         csv_writer = csv.writer(f)
    #         csv_writer.writerow(['track_id','other_id','relationship','time_stamp'])
  
    with open(csv_file_name + '.csv','ab')as f:
        writed_id_pair = []
        csv_writer = csv.writer(f)
        # for ego_id in upper_left_id_dict.keys():
        #     for other_id,time_stamp in upper_left_id_dict[ego_id].items():
        #         csv_writer.writerow([ego_id,other_id,time_stamp,'upper_left'])
        # for ego_id in lower_left_id_dict.keys():
        #     for other_id,time_stamp in lower_left_id_dict[ego_id].items():
        #         csv_writer.writerow([ego_id,other_id,time_stamp,'lower_left'])
        # for ego_id in upper_right_id_dict.keys():
        #     for other_id,time_stamp in upper_right_id_dict[ego_id].items():
        #         csv_writer.writerow([ego_id,other_id,time_stamp,'upper_right'])
        # for ego_id in lower_right_id_dict.keys():
        #     for other_id,time_stamp in lower_right_id_dict[ego_id].items():
        #         csv_writer.writerow([ego_id,other_id,time_stamp,'lower_right'])
        for ego_id in previous_id_dict.keys():
            for other_id,time_stamp in previous_id_dict[ego_id].items():
                if not is_duplicate_id_pair(writed_id_pair,ego_id,other_id):
                    csv_writer.writerow([ego_id,other_id,'previous',time_stamp[0],time_stamp[1],time_stamp[2]])
                    writed_id_pair.append([ego_id,other_id])
        for ego_id in following_id_dict.keys():
            for other_id,time_stamp in following_id_dict[ego_id].items():
                if not is_duplicate_id_pair(writed_id_pair,ego_id,other_id):
                    csv_writer.writerow([ego_id,other_id,'following',time_stamp[0],time_stamp[1],time_stamp[2]])
                    writed_id_pair.append([ego_id,other_id])
        for ego_id in front_conflict_id_dict.keys():
            for other_id,time_stamp in front_conflict_id_dict[ego_id].items():
                if not is_duplicate_id_pair(writed_id_pair,ego_id,other_id):
                    csv_writer.writerow([ego_id,other_id,'front_conflict',time_stamp[0],time_stamp[1],time_stamp[2]])
                    writed_id_pair.append([ego_id,other_id])

def is_duplicate_id_pair(writed_id_pair,id_1,id_2):
    # use for remove duplication 
    for item in writed_id_pair:
        if item[0] == id_1 and item[1] == id_2:
            return True
        elif item[0] == id_2 and item[1] == id_1:
            return True
        else:
            return False

def save_episode_expert_demo(ego_id_set,current_observation,ego_action_dict,next_observation,done,episode_trajectory,turing_episode_id):
    # print('current id:',id(current_observation))
    # print('next id:',id(next_observation))
    for ego_id in ego_id_set:
        if done:
            mask = 1
        else:
            mask = 0
        min_turning_rad = math.radians(30)
        if contain_turning(current_observation,ego_id,min_turning_rad) and ego_id not in turing_episode_id:
            turing_episode_id.append(ego_id)
        state_action_mask_tuple = [unfold_observation(current_observation,ego_id),[ego_action_dict[ego_id].acc,ego_action_dict[ego_id].steering],unfold_observation(next_observation,ego_id),mask]
        episode_trajectory[ego_id].append(state_action_mask_tuple)


def save_trajectory_to_pickle(episode_trajectory,turing_episode_id,filename):
    full_episode_np_trajectories = np.array([episode_trajectory[k] for k in episode_trajectory.keys()])
    turning_episode_np_trajectories = np.array([episode_trajectory[k] for k in turing_episode_id])
    no_turning_episode_id = [k for k in episode_trajectory.keys() if k not in turing_episode_id]
    no_turning_episode_np_trajectories = np.array([episode_trajectory[k] for k in no_turning_episode_id])
    # np_trajectories = np.array(trajectories,float)
    
    # print("np_trajectories shape", np_trajectories.shape)

    # save pickle
    print('')
    print('saving!!!')
    print('')
    filename = filename + '_straight.pkl'
    # turning_episode_filename =  filename + '_turning.pkl'
    # no_turning_episode_filename = filename + '_no_turning.pkl'
    # filename = filename + ".pkl"
    # with open(turning_episode_filename,'ab') as f:
    #     pickle.dump(turning_episode_np_trajectories, f)
    
    # with open(no_turning_episode_filename,'ab') as f:
    #     pickle.dump(no_turning_episode_np_trajectories, f)
    
    with open(filename,'ab') as f:
        pickle.dump(full_episode_np_trajectories, f)
    # filename = filename + ".npy"
    # np.save(filename, arr=np_trajectories)


def is_useless_for_imitation_learning(key):
    # remove some useless item
    useless_key = ['center_line','reach_end','collision','upper_left_vehicle_id_ms_dict_lanelet','lower_left_vehicle_id_ms_dict_lanelet',
                    'upper_right_vehicle_id_ms_dict_lanelet','lower_right_vehicle_id_ms_dict_lanelet','previous_vehicle_id_ms_dict_lanelet',
                    'following_vehicle_id_ms_dict_lanelet','front_conflict_vehicle_id_ms_dict_lanelet','following_route_conflict_lanelet',
                    'ego_complete_ratio']
   
    if key in useless_key:
        return True
    else:
        return False

def unfold_observation(observation,ego_id):
    observation_value = []
    for k,v in observation.items():
        if isinstance(v[ego_id],list):
            observation_value.extend(v[ego_id])
        else:
            observation_value.append(v[ego_id])

    return observation_value

def copy_observation(observation):
    copy_observation = dict()
    for k, v in observation.items():
        if not is_useless_for_imitation_learning(k):
            copy_observation[k] = copy.deepcopy(v)

    return copy_observation

def contain_turning(observation,ego_id,min_turning_rad):
    heading_errors = observation['heading_errors'][ego_id]
    for sin_theta in heading_errors:
        if sin_theta > math.sin(min_turning_rad) or sin_theta < - math.sin(min_turning_rad):
            return True

    else:
        return False
