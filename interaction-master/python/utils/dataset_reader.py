#!/usr/bin/env python

import csv
# import cPickle as pickle
import pickle
from .dataset_types import MotionState, Track


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

def read_trajectory(filename):
    # print(filename)
    with open(filename, 'rb') as f:
        trajectory_data = pickle.load(f, ) ## encoding='latin1')
        # egos_trajectory_data = trajectory_data['egos_track']
        vehicles_trajectory_data = trajectory_data['others_track']

        track_dict = dict()
        track_id = None

        for id, info in vehicles_trajectory_data.items():
            car_info = info[0]
            traj_info = info[1:]

            track_id = int(id)
            track = Track(track_id)
            track.agent_type = 'car'

            track.width = float(car_info[0])
            track.length = float(car_info[1])
            track.time_stamp_ms_first = int(car_info[2])
            track.time_stamp_ms_last = int(car_info[3]) - 100
            # print(track_id)
            
            for single_traj_info in traj_info:
                # print(int(single_traj_info[0]))
                ms = MotionState(int(single_traj_info[0]))
                ms.x = float(single_traj_info[1])
                ms.y = float(single_traj_info[2])
                ms.vx = float(single_traj_info[3])
                ms.vy = float(single_traj_info[4])
                ms.psi_rad = float(single_traj_info[5])
                # print(ms.psi_rad)
                track.motion_states[ms.time_stamp_ms] = ms

            track_dict[track_id] = track
            # print('look:', track.motion_states[int(car_info[2])].psi_rad, track.time_stamp_ms_last)

        return track_dict

def read_pedestrian(filename):

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        track_dict = dict()
        track_id = None

        for i, row in enumerate(list(csv_reader)):

            if i == 0:
                # check first line with key names
                assert (row[KeyEnum.track_id] == Key.track_id)
                assert (row[KeyEnum.frame_id] == Key.frame_id)
                assert (row[KeyEnum.time_stamp_ms] == Key.time_stamp_ms)
                assert (row[KeyEnum.agent_type] == Key.agent_type)
                assert (row[KeyEnum.x] == Key.x)
                assert (row[KeyEnum.y] == Key.y)
                assert (row[KeyEnum.vx] == Key.vx)
                assert (row[KeyEnum.vy] == Key.vy)
                continue

            if row[KeyEnum.track_id] != track_id:
                # new track
                track_id = row[KeyEnum.track_id]
                assert (track_id not in track_dict.keys()), \
                    "Line %i: Track id %s already in dict, track file not sorted properly" % (i + 1, track_id)
                track = Track(track_id)
                track.agent_type = row[KeyEnum.agent_type]
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
            track.motion_states[ms.time_stamp_ms] = ms

        return track_dict


def read_specified_id_track(id,track_dict):
    return track_dict[id]

def get_other_track_except_two_ego_car(ego1_id,ego2_id,start_time_stamp,end_time_stamp,track_dict):
    other_track_dict = dict()
    for k,v in track_dict.items():
        if k != ego1_id and k != ego2_id:
            other_start_time = v.time_stamp_ms_first
            other_end_time = v.time_stamp_ms_last
            if start_time_stamp <= other_start_time <= end_time_stamp or start_time_stamp<= other_end_time <= end_time_stamp:
                other_track_dict[k] = [start_time_stamp,end_time_stamp]

    return other_track_dict



      