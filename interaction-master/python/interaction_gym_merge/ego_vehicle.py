import sys
sys.path.append("..")
from utils import tracks_vis
import geometry
from utils.dataset_types import Track, MotionState, Action
import numpy as np
import math
import copy
from collections import deque


# colours = {
#     'w': (255, 255, 255),
#     'k': (000, 000, 000),
#     'r': (255, 000, 000),
#     'g': (000, 255, 000),
#     'm': (255, 000, 255),-
#     'b': (000, 000, 255),
#     'c': (000, 255, 255),
#     'y': (255, 255, 000),
# }


class ego_vehicle:
    def __init__(self, track_id, start_end_state, delta_time, max_speed=25):
        self.id = track_id
        self._length = start_end_state[2]
        self._width = start_end_state[3]
        self._current_state = MotionState(start_end_state[4].time_stamp_ms)        # MotionState type (time_stamp_ms,x,y,vx,vy,psi_rad)
        self._dt = delta_time                       # 100 ms
        # self._colour = colours['c']
        self._action_for_current_state = Action(self._current_state.time_stamp_ms)

        self.acc_normalize = 3 # 2.5 # 3
        self.dec_normalize = 3 # 4
        self.steering_normalize = 30 # degree
        self.max_speed = max_speed

        args_lateral_dict = {'K_P': 1.4,
                              'K_D': 0.05,
                              'K_I': 0.25,
                              'dt': 0.1}
        args_longitudinal_dict = {'K_P': 1.0,
                                  'K_D': 0,
                                  'K_I': 0.05,
                                  'dt': 0.1}
        self.pid_controller = VehiclePIDController(args_lateral=args_lateral_dict, 
                                                   args_longitudinal=args_longitudinal_dict, 
                                                   max_throttle=1,
                                                   max_steering=1)

    @property
    def width(self):
        return self._width

    @property
    def length(self):
        return self._length

    def step_continuous_action(self, action_list, next_waypoint_position=None):
        """
        Update current position, given current velocity and acceleration
        """
        # Actions: acceleration (a), steering (b)
        # acceleration (m/s^2 / self.acc_normalize) 
        # steering is based on the body coordinate (degree/self.steering_normalize)
        # to rotate vehicle direction using steering : 
        # right turn is equal to conter clockwise rotation b
        # left turn is equal to clockwise rotation b


        current_direction = np.array([math.cos(self._current_state.psi_rad), math.sin(self._current_state.psi_rad)])
        print('current_direction', self._current_state.psi_rad * 180 / np.pi)

        current_speed_value = math.sqrt(self._current_state.vx * self._current_state.vx + self._current_state.vy * self._current_state.vy)
        print('current_speed:', current_speed_value, self._current_state.vx, self._current_state.vy)

        current_pos = np.array([self._current_state.x,self._current_state.y])
        print('current_pos:', current_pos)


        if next_waypoint_position: # using pid controller controls steering
            if action_list[0] == -1: # stop mode
                acc_normalize = 'stop'
                steering_normalize = 0
            else:
                target_speed_value = action_list[0] * self.max_speed
                acc_normalize = self.pid_controller.run_lon_step(current_speed_value, target_speed_value)
                # acc_normalize = action
                # print('look', acc_normalize)
                steering_normalize = self.pid_controller.run_lat_step(current_position=[self._current_state.x,self._current_state.y], 
                                                                        waypoint_position=next_waypoint_position, current_direction=[math.cos(self._current_state.psi_rad), math.sin(self._current_state.psi_rad)])
        else: # controls steering by nets
            acc_normalize, steering_normalize = action_list


        if acc_normalize == 'stop':
            acc = -10000
        elif acc_normalize >= 0:
            acc = acc_normalize * self.acc_normalize
        else:
            acc = acc_normalize * self.dec_normalize
            
        steering = steering_normalize * self.steering_normalize
        steering_rad = math.radians(steering)
        # steering_rad = steering
        # print("acc:", acc)
        # print("steering:", steering)

        dt = 0.1
        # print('self dt:',self._dt)
        # print('dt:',dt)
        # State integration
        # partical motion model

        
        # gyt motion model
        # frozen_speed = 0.02
        # if current_speed_value > frozen_speed:
        #     new_direction_rad = self._current_state.psi_rad + steering_rad * dt
        # else:
        #     new_direction_rad = self._current_state.psi_rad
        # new_pos = current_pos + [current_speed_value * dt * math.cos(new_direction_rad), current_speed_value * dt * ma                                                                th.sin(new_direction_rad)] #+ 0.5 * acc * new_direction_norm * (dt**2)
        # print('new_pos:',new_pos)

        # if acc <= 0 and current_speed_value <= frozen_speed:
        #     new_speed = [0, 0]
        # elif acc > 0 and current_speed_value >= 15:
        #     new_speed = [self._current_state.vx, self._current_state.vy]
        # else: 
        #     new_speed = [(current_speed_value + acc * dt) * math.cos(new_direction_rad), (current_speed_value + acc * dt) * math.sin(new_direction_rad)] 


        # bicycle model
        print('length:', self._length)
        wheelbase_scale = 0.6
        wheelbase = self._length * wheelbase_scale
        gravity_core_scale = 0.4
        f_len = wheelbase * gravity_core_scale
        r_len = wheelbase - f_len
        # print('car length:', wheelbase, f_len, r_len)

        beta = math.atan((r_len / (r_len + f_len)) * math.tan(steering_rad)) # *dt?
        # print('steering and beta:', steering_rad, beta)
        new_pos_x = current_pos[0] + current_speed_value * math.cos(self._current_state.psi_rad + beta) * dt
        new_pos_y = current_pos[1] + current_speed_value * math.sin(self._current_state.psi_rad + beta) * dt
        new_pos = np.array([new_pos_x, new_pos_y])
        print('new_pos:', new_pos)
        new_direction_rad = self._current_state.psi_rad + (current_speed_value / r_len) * math.sin(beta) * dt
        if acc > 0 and current_speed_value >= 15:
            new_speed = [self._current_state.vx, self._current_state.vy]
        else:
            new_speed_value = current_speed_value + acc * dt
            if new_speed_value < 0:
                new_speed_value = 0
            new_speed = [new_speed_value * math.cos(new_direction_rad), new_speed_value * math.sin(new_direction_rad)]
        
        # 
        self._current_state.x = new_pos[0]
        self._current_state.y = new_pos[1]
        self._current_state.vx = new_speed[0]
        self._current_state.vy = new_speed[1]
        # self._current_state.psi_rad = math.atan2(new_direction_norm[1],new_direction_norm[0])
        self._current_state.psi_rad = new_direction_rad
        self._current_state.time_stamp_ms += self._dt 


        # record for discrim
        self._action_for_current_state.time_stamp_ms += self._dt 
        self._action_for_current_state.acc = acc_normalize
        self._action_for_current_state.steering = steering_normalize

        return self._current_state, self._action_for_current_state


    def step_continuous_action_as_record(self,current_time,track):
        """
        Update current postion as track record
        """
        # Actions: acceleration (a), steering (b)
        # acceleration (m/s^2)
        # steering is based on the body coordinate (degree)

        # as we use particle model we think velocity direction = heading direction 
        ms = track.motion_states
        next_time = current_time + self._dt

        # atan2 (-180,+180]
        # we set acc must in [-3,3]
        # we set steering must in [-15,15]
        current_direction_rad = ms[current_time].psi_rad
        next_time_direction_rad = ms[next_time].psi_rad


        current_direction_vector = np.array([math.cos(current_direction_rad),math.sin(current_direction_rad)])
        next_time_direction_vector = np.array([math.cos(next_time_direction_rad),math.sin(next_time_direction_rad)])

        TheNorm = np.linalg.norm(current_direction_vector) * np.linalg.norm(next_time_direction_vector)

        # cross 
        cross = np.cross(current_direction_vector, next_time_direction_vector) / TheNorm
        cross = np.clip(cross,-1,1)
        rho = np.rad2deg(np.arcsin(cross))
        # print('rho:',rho)
        # dot
        dot = np.dot(current_direction_vector, next_time_direction_vector) / TheNorm
        dot = np.clip(dot,-1,1)
        theta = np.rad2deg(np.arccos(dot))
        # print('theta:',theta)
        # if np.isnan(theta):
        #     import pdb
        #     pdb.set_trace()
        #     print('TheNorm:',TheNorm)
        #     print('dot:',np.dot(current_direction_vector, next_time_direction_vector))

        if rho < 0:
            # print('steering:',theta)
            steering_normalize = -theta / self.steering_normalize
        else:
            # print('steering:',-theta)
            steering_normalize = theta / self.steering_normalize
 
        current_speed = math.sqrt((ms[current_time].vx)**2 + (ms[current_time].vy)**2)
        next_time_speed = math.sqrt((ms[next_time].vx)**2 + (ms[next_time].vy)**2)
        acc = (next_time_speed - current_speed) / self._dt    # m/(s * ms)  need to convert to m/s^2
        acc = 1000 * acc
        print('acc:',acc)
        acc_normalize = acc / self.acc_normalize


        # compare
        self.step_continuous_action([acc_normalize, steering_normalize])

        # # record action for IL
        # self._current_state.time_stamp_ms = next_time
        # self._current_state.x = ms[next_time].x
        # self._current_state.y = ms[next_time].y
        # self._current_state.vx = ms[next_time].vx
        # self._current_state.vy = ms[next_time].vy
        # self._current_state.psi_rad = ms[next_time].psi_rad

        self._action_for_current_state.time_stamp_ms += self._dt 
        self._action_for_current_state.acc = acc_normalize
        self._action_for_current_state.steering = steering_normalize

        return self._current_state, self._action_for_current_state


    def reset_state(self,start_state):
        print('start state:', start_state.x, start_state.y, start_state.vx, start_state.vy)
        self._current_state.time_stamp_ms = start_state.time_stamp_ms
        self._current_state.x = start_state.x
        self._current_state.y = start_state.y
        self._current_state.vx = start_state.vx
        self._current_state.vy = start_state.vy
        self._current_state.psi_rad = start_state.psi_rad
        print('reset state:', self._current_state.x, self._current_state.y, self._current_state.vx, self._current_state.vy)

        
class VehiclePIDController(object):
    def __init__(self, args_lateral, args_longitudinal, offset=0, max_throttle=0.75, max_steering=0.8):

        self.max_throt = max_throttle
        self.max_steer = max_steering
        # print('look', args_lateral)

        self._lon_controller = PIDLongitudinalController(**args_longitudinal)
        self._lat_controller = PIDLateralController(offset, **args_lateral)

    def run_lon_step(self, current_speed, target_speed):
        acceleration = self._lon_controller.run_step(current_speed, target_speed)
        return acceleration
    
    def run_lat_step(self, current_position, waypoint_position, current_direction):
        current_steering = self._lat_controller.run_step(current_position, waypoint_position, current_direction)
        return current_steering

    def run_step(self, past_steering, current_speed, target_speed, current_position, waypoint_position, current_direction):

        acceleration = self.run_lon_step(current_speed, target_speed)
        current_steering = self.run_lat_step(current_position, waypoint_position, current_direction)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.

        if current_steering > past_steering + 0.1:
            current_steering = past_steering + 0.1
        elif current_steering < past_steering - 0.1:
            current_steering = past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return acceleration, current_steering


class PIDLongitudinalController():

    def __init__(self, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):

        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, current_speed, target_speed, debug=False):

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):

        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

class PIDLateralController():

    def __init__(self, offset=0, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):

        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._offset = offset
        self._e_buffer = deque(maxlen=10)

    def run_step(self, current_position, waypoint_position, current_direction):

        return self._pid_control(waypoint_position, current_position, current_direction)

    def _pid_control(self, waypoint_position, current_position, current_direction):

        # Get the ego's location and forward vector
        ego_loc_x = current_position[0]
        ego_loc_y = current_position[1]

        v_vec = np.array([current_direction[0], current_direction[1], 0.0])

        # Get the vector vehicle-target_wp
        if self._offset != 0:
            # Displace the wp to the side
            w_tran = waypoint.transform
            r_vec = w_tran.get_right_vector()
            w_loc = w_tran.location + carla.Location(x=self._offset*r_vec.x,
                                                         y=self._offset*r_vec.y)
        else:
            # w_loc = waypoint.transform.location
            w_loc_x = waypoint_position[0]
            w_loc_y = waypoint_position[1]

        w_vec = np.array([w_loc_x - ego_loc_x,
                          w_loc_y - ego_loc_y,
                          0.0])

        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))
        # print('look', w_vec, v_vec)
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        # print('look', _dot, _de, _ie)
        return np.clip((self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)
