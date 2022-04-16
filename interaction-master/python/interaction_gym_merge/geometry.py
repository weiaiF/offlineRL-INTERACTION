
import numpy as np
import lanelet2
from lanelet2.core import AttributeMap, getId, BasicPoint2d, LineString3d, LineString2d, Point3d, Polygon2d
from lanelet2.geometry import inside, distance, intersects2d, length2d, intersectCenterlines2d, follows
import sys
sys.path.append("..")
# print('2:',__file__)
from utils import tracks_vis
import math

def getAttributes():
    return AttributeMap({"key": "value"})

def is_equal_point(point1,point2):
    if point1.x == point2.x and point1.y == point2.y:
        return True
    else:
        return False

def lanelet_length(lanelet):
    # return centerline length
    length = length2d(lanelet)
    return length

def is_following_lanelet(previous_lanelet, next_lanelet):
    # check whether the next lanelet is the following lanelet of previous lanelet
    return follows(previous_lanelet,next_lanelet)

def get_trajectory_distance(ego_pos, ego_trajectory_pos):
    trajectory_distance = math.sqrt((ego_pos[0] - ego_trajectory_pos[0]) ** 2 + (ego_pos[1] - ego_trajectory_pos[1]) ** 2)
    return trajectory_distance

def get_trajectory_pos(ego_state_dict, ego_trajectory_pos):
    ego_pos = ego_state_dict['pos']
    ego_heading = ego_state_dict['heading']
    x_in_ego_axis = (ego_trajectory_pos[1] - ego_pos[1])*np.cos(ego_heading) - (ego_trajectory_pos[0] - ego_pos[0])*np.sin(ego_heading)
    y_in_ego_axis = (ego_trajectory_pos[1] - ego_pos[1])*np.sin(ego_heading) + (ego_trajectory_pos[0] - ego_pos[0])*np.cos(ego_heading)
    return [x_in_ego_axis, y_in_ego_axis]

def get_trajectory_speed(ego_trajectory_velocity):
    # print('ego_trajectory_velocity:', ego_trajectory_velocity)
    trajectory_vx = ego_trajectory_velocity[0]
    trajectory_vy = ego_trajectory_velocity[1]

    trajectory_speed = math.sqrt(trajectory_vx**2 + trajectory_vy**2)

    return [trajectory_speed]


def ego_reach_lanelet_region(ego_vehicle, lanelet):  
    ego_point2d = BasicPoint2d(ego_vehicle._current_state.x, ego_vehicle._current_state.y)
   
    # test if the center point of ego is inside the lanelet area
    reach = inside(lanelet, ego_point2d)

    return reach

def ego_inside_planning_route(current_lanelet,ego_route_lanelet_list):
    # check whether the ego vehicle is inside the planning route
    # as right now the real time planning function is not working, we need to fix the planning route of ego vehicle
    for ll in ego_route_lanelet_list:
        if current_lanelet.id == ll.id:
            return True

    print('ego outside the planning route!')
    return False

def ego_collision(ego_vehicle, other_vehicle_polygon):
    # test if the ego polygon is overlap with other vehicles
    ego_polypoint_np = tracks_vis.polygon_xy_from_motionstate(ego_vehicle._current_state, ego_vehicle._width, ego_vehicle._length)

    ego_polyPoint3d = [Point3d(getId(),p[0],p[1],0,getAttributes()) for p in ego_polypoint_np]

    # ego_polyLineString3d_1 = LineString3d(getId(), [ego_polyPoint3d[0], ego_polyPoint3d[1]], getAttributes())
    # ego_polyLineString3d_2 = LineString3d(getId(), [ego_polyPoint3d[1], ego_polyPoint3d[2]], getAttributes())
    # ego_polyLineString3d_3 = LineString3d(getId(), [ego_polyPoint3d[2], ego_polyPoint3d[3]], getAttributes())
    # ego_polyLineString3d_4 = LineString3d(getId(), [ego_polyPoint3d[3], ego_polyPoint3d[0]], getAttributes())

    ego_poly = Polygon2d(getId(),[ego_polyPoint3d[0],ego_polyPoint3d[1],ego_polyPoint3d[2],ego_polyPoint3d[3]],getAttributes())

    # ego_poly = Polygon2d(getId(),[ego_polyLineString3d_1,ego_polyLineString3d_2,ego_polyLineString3d_3,ego_polyLineString3d_4],getAttributes())

    for k,v in other_vehicle_polygon.items():
        other_polypoint_np = v
        # print(type(v))
        other_polyPoint3d = [Point3d(getId(),p[0],p[1],0,getAttributes()) for p in other_polypoint_np]
        other_poly = Polygon2d(getId(),[other_polyPoint3d[0],other_polyPoint3d[1],other_polyPoint3d[2],other_polyPoint3d[3]],getAttributes())


        if intersects2d(ego_poly, other_poly):
            return True

    return False

def ego_other_distance_and_collision(ego_state_dict, other_state_dict):
    # calculte the minmum distance between two polygon2d
    ego_polypoint_np = ego_state_dict['polygon'] # tracks_vis.polygon_xy_from_motionstate(ego_vehicle._current_state, ego_vehicle._width,ego_vehicle._length)
    ego_polyPoint3d = [Point3d(getId(),p[0],p[1],0,getAttributes()) for p in ego_polypoint_np]
    ego_poly = Polygon2d(getId(),[ego_polyPoint3d[0],ego_polyPoint3d[1],ego_polyPoint3d[2],ego_polyPoint3d[3]],getAttributes())

    
    other_polypoint_np = other_state_dict['polygon']
    # print(type(v))
    other_polyPoint3d = [Point3d(getId(),p[0],p[1],0,getAttributes()) for p in other_polypoint_np]
    other_poly = Polygon2d(getId(),[other_polyPoint3d[0],other_polyPoint3d[1],other_polyPoint3d[2],other_polyPoint3d[3]],getAttributes())

    if intersects2d(ego_poly, other_poly):
        return 0,True
    else:   
        poly_distance = distance(ego_poly,other_poly)
        return poly_distance,False

def in_front_of_ego_vehicle(ego_vehicle,ego_polygon,other_vehicle_pos):
    # determine whether it is in front of the ego vehicle
    # this function need to fix:(as the velocity vector can be 0)
    # we should use the vehicle heading instead of the velocity heading
    ego_to_other_vector = np.array([other_vehicle_pos[0] - ego_vehicle._current_state.x, other_vehicle_pos[1] - ego_vehicle._current_state.y])

    ego_heading_vector = get_vehicle_heading(ego_polygon)

    L_ego_to_other = np.sqrt(ego_to_other_vector.dot(ego_to_other_vector))
    L_ego_vel = np.sqrt(ego_heading_vector.dot(ego_heading_vector))
    cos_angle = ego_to_other_vector.dot(ego_heading_vector)/(L_ego_to_other*L_ego_vel)
    cos_angle = np.clip(cos_angle,-1,1)
    radian = np.arccos(cos_angle)
    angle = radian * 180 / np.pi

    if abs(angle) >90:
        return False
    else:
        return True

def ego_line_distance(ego_vehicle,linestring):
    # calculate the minmum distance between polygon2d and LineString3d
    ego_polypoint_np = tracks_vis.polygon_xy_from_motionstate(ego_vehicle._current_state,ego_vehicle._width,ego_vehicle._length)
    ego_polyPoint3d = [Point3d(getId(),p[0],p[1],0,getAttributes()) for p in ego_polypoint_np]
    ego_poly = Polygon2d(getId(),[ego_polyPoint3d[0],ego_polyPoint3d[1],ego_polyPoint3d[2],ego_polyPoint3d[3]],getAttributes())

    line = LineString2d(linestring)
    # print('linestring2d:',line)

    poly_line_distance = distance(ego_poly,line)

    return poly_line_distance

def get_overlap_point_list(ego_planning_lanelet,conflict_lanelet):
    overlap_centerline = intersectCenterlines2d(ego_planning_lanelet,conflict_lanelet)
    return overlap_centerline

def insert_node_to_meet_min_interval(centerline_point_list, min_interval):
    # convert point form
    point_list = []
    if not isinstance(centerline_point_list[0], (list, tuple)):
        for point in centerline_point_list:
            point_list.append([point.x, point.y])
    else:
        for point in centerline_point_list:
            point_list.append([point[0], point[1]])        
    # uniform insert node to meet the minmum interval distance requirement
    extend_centerline_point_list = [] 
    for index in range(len(point_list)-1):
        extend_centerline_point_list.append(point_list[index])
        # print('origin point type:',type(centerline_point_list[index]))
        current_interval_distance =  math.sqrt((point_list[index][0] - point_list[index+1][0])**2 + (point_list[index][1] -point_list[index+1][1])**2)
        if current_interval_distance > min_interval:
            interval_num = math.ceil(current_interval_distance / min_interval)
            interval_point_num = interval_num - 1
            # print('interval_point_num:',interval_point_num)
            for i in range(int(interval_point_num)):
                pt_x = point_list[index][0] + (i+1) * (point_list[index+1][0] - point_list[index][0]) / interval_num
                pt_y = point_list[index][1] + (i+1) * (point_list[index+1][1] - point_list[index][1]) / interval_num
                # interval_point = Point3d(getId(), pt_x, pt_y, 0, getAttributes())
                interval_point = [pt_x, pt_y]
                extend_centerline_point_list.append(interval_point)
        
    extend_centerline_point_list.append(point_list[-1])

    return extend_centerline_point_list


def is_in_front_of_overlap_region(ego_planning_lanelet,other_conflict_lanelet,other_vehicle_pos,min_interval_distance):
    conflict_point_list = get_overlap_point_list(ego_planning_lanelet,other_conflict_lanelet)
    conflict_point = conflict_point_list[0]

    
    # the overlap centerline is a point2d vector which is the same share point of two overlap lanelet centerline
    # so in order to determine the relative position of the conflict vehicle(before/after conflict region)
    # we compare the current index of the closet point and the index of conflict point
    # first find the closet centerline point
    centerline_point_list = other_conflict_lanelet.centerline
    extend_centerline_point_list = insert_node_to_meet_min_interval(centerline_point_list, min_interval_distance)
    vehicle_pos = [other_vehicle_pos[0],other_vehicle_pos[1]]
    closet_point_index = get_closet_centerline_point(vehicle_pos,extend_centerline_point_list)

    conflict_point_index = 0 
    for index,pt in enumerate(extend_centerline_point_list):
        if is_equal_point(pt,conflict_point):
            conflict_point_index = index
            break

    if closet_point_index < conflict_point_index:
        # print('is in front overlap')
        return True
    else:
        return False   

def get_vehicle_heading(vehicle_polygon):
    lowleft = vehicle_polygon[0]
    lowright = vehicle_polygon[1]

    heading = lowright - lowleft
    L_heading = np.sqrt(heading.dot(heading))
    uniform_heading_vector = heading / L_heading
 

    return uniform_heading_vector


def is_vaild_velocity_direction_along_lanelet(vehicle_pos,vehicle_heading,current_lanelet,min_interval_distance):
    heading_error = get_vehicle_and_lanelet_heading_error(vehicle_pos,vehicle_heading,current_lanelet,min_interval_distance)
    # print('heading error angle:',heading_error)
    if heading_error < 20:
        # print('valid direction')
        return True
    else:
        # print('not valid direction')
        return False

def get_centerline_point_list_with_heading_and_average_interval(centerline_point_list, min_interval_distance):
    # first we average the interval between points
    # centerline_point_list_average_interval = [centerline_point_list[0]]
    # for index, point in enumerate(centerline_point_list[1:]):
    #     if index != (len(centerline_point_list) - 1):
    #         point = centerline_point_list[index]
    #         point_previous = centerline_point_list_average_interval[-1]
    #         if point == point_previous:
    #             continue
    #         distance_to_previous = math.sqrt((point[0] - point_previous[0])**2 + (point[1] - point_previous[1])**2)
    #         if distance_to_previous > 2:
    #             centerline_point_list_average_interval.append(point)
    #         else:
    #             centerline_point_list_average_interval.append(point)
    #     else:
    #         if point == centerline_point_list_average_interval[-1]:
    #             continue
    #         centerline_point_list_average_interval.append(point)
        
    # then we calculate each point's heading
    previous_point_yaw = None
    centerline_point_list_with_heading = []
    # print('route:', centerline_point_list)
    for index, point in enumerate(centerline_point_list):
        if index == (len(centerline_point_list) - 1): # last centerlane point
            point_yaw = centerline_point_list_with_heading[-1][-1]
        else:
            point = centerline_point_list[index]
            point_next = centerline_point_list[index + 1]
            point_vector = np.array((point_next[0] - point[0], point_next[1] - point[1]))

            point_vector_length =  np.sqrt(point_vector.dot(point_vector))
            # print('look:', point_vector, point_vector_length)
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

        # print(point_yaw, cos_angle)
        centerline_point_list_with_heading.append((point[0], point[1], point_yaw))

    return centerline_point_list_with_heading

def get_closet_front_centerline_point(vehicle_pos, centerline_point_list_with_heading):
    min_dist = 100
    closet_point_index = 0
    # print('look:', centerline_point_list_with_heading)
    for index, point in enumerate(centerline_point_list_with_heading):
        vehicle_to_point_dist = math.sqrt((point[0] - vehicle_pos[0])**2 + (point[1] - vehicle_pos[1])**2)
        vehicle_y_in_point_axis = (vehicle_pos[1] - point[1])*np.sin(point[2]) + (vehicle_pos[0] - point[0])*np.cos(point[2])
        # print('vehicle_y_in_point_axis:', vehicle_y_in_point_axis)
        if min_dist > vehicle_to_point_dist and vehicle_y_in_point_axis < 0:
            min_dist = vehicle_to_point_dist
            closet_point_index = index

    # if closet_point_index == 0:
    #     closet_point_index = len(centerline_point_list_with_heading)-1

    return closet_point_index

def get_closet_centerline_point(vehicle_pos, centerline_point_list):
    min_dist = 100
    closet_point_index = 0
    for index, point in enumerate(centerline_point_list):
        if isinstance(point, (list, tuple)):
            vehicle_to_point_dist = math.sqrt((point[0] - vehicle_pos[0])**2 + (point[1] - vehicle_pos[1])**2)
        else:
            vehicle_to_point_dist = math.sqrt((point.x - vehicle_pos[0])**2 + (point.y - vehicle_pos[1])**2)
        if min_dist > vehicle_to_point_dist:
            min_dist = vehicle_to_point_dist
            closet_point_index = index

    return closet_point_index

def get_closet_bound_point(vehicle_pos, left_point_list, right_point_list):
    min_dist_l = 100
    min_dist_r = 100
    closet_left_point_index = 0
    closet_right_point_index = 0
    for index, point in enumerate(left_point_list):
        vehicle_to_point_dist = math.sqrt((point.x - vehicle_pos[0])**2 + (point.y - vehicle_pos[1])**2)
        if min_dist_l > vehicle_to_point_dist:
            min_dist_l = vehicle_to_point_dist
            closet_left_point_index = index
    closet_left_point = [left_point_list[closet_left_point_index].x, left_point_list[closet_left_point_index].y]

    for index, point in enumerate(right_point_list):
        vehicle_to_point_dist = math.sqrt((point.x - vehicle_pos[0])**2 + (point.y - vehicle_pos[1])**2)
        if min_dist_r > vehicle_to_point_dist:
            min_dist_r = vehicle_to_point_dist
            closet_right_point_index = index
    closet_right_point = [right_point_list[closet_right_point_index].x, right_point_list[closet_right_point_index].y]

    road_width = math.sqrt((closet_right_point[0] - closet_left_point[0])**2 + (closet_right_point[1] - closet_left_point[1])**2)
    if min_dist_l > road_width:
        min_dist_r = 0
    elif  min_dist_r > road_width:
        min_dist_l = 0

    return [closet_left_point, closet_right_point], [min_dist_l, min_dist_r]


def get_vehicle_and_lanelet_heading_error(vehicle_pos, vehicle_heading, current_lanelet, min_interval_distance):
    # first find the closet centerline point
    # (this function maybe need repair as the centerline points do not have uniform distance !!!)
    # print('### calculate heading error ###')

    centerline_point_list = current_lanelet.centerline
    extend_centerline_point_list = insert_node_to_meet_min_interval(centerline_point_list, min_interval_distance)
    # print('total centerline point:',len(extend_centerline_point_list))
    closet_point_index = get_closet_centerline_point(vehicle_pos, extend_centerline_point_list)
    # print('closet point index:',closet_point_index)
    # calculate the heading along the lanelet
    
    if closet_point_index < len(extend_centerline_point_list) - 1:
        # lanelet_heading_vector = np.array([extend_centerline_point_list[closet_point_index+1].x - extend_centerline_point_list[closet_point_index].x, extend_centerline_point_list[closet_point_index+1].y - extend_centerline_point_list[closet_point_index].y])
        lanelet_heading_vector = np.array([extend_centerline_point_list[closet_point_index+1][0] - extend_centerline_point_list[closet_point_index][0], extend_centerline_point_list[closet_point_index+1][1] - extend_centerline_point_list[closet_point_index][1]])
    else:
        # lanelet_heading_vector = np.array([extend_centerline_point_list[closet_point_index].x - extend_centerline_point_list[closet_point_index-1].x, extend_centerline_point_list[closet_point_index].y - extend_centerline_point_list[closet_point_index-1].y])
        lanelet_heading_vector = np.array([extend_centerline_point_list[closet_point_index][0] - extend_centerline_point_list[closet_point_index-1][0], extend_centerline_point_list[closet_point_index][1] - extend_centerline_point_list[closet_point_index-1][1]])

    # print('lanelet_heading_vector:',lanelet_heading_vector)
    vehicle_heading_vector = np.array([vehicle_heading[0],vehicle_heading[1]])
    # print('vehicle_shape_heading_vector:',vehicle_heading_vector)

    L_lanelet_heading = np.sqrt(lanelet_heading_vector.dot(lanelet_heading_vector))
    # print('L_lanelet_heading:',L_lanelet_heading)
    L_vehicle_heading = np.sqrt(vehicle_heading_vector.dot(vehicle_heading_vector))
    # print('L_vehicle_heading:',L_vehicle_heading)
    cos_angle = vehicle_heading_vector.dot(lanelet_heading_vector)/(L_lanelet_heading*L_vehicle_heading)
    # print('cos_angle:',cos_angle)
    cos_angle = np.clip(cos_angle,-1,1)
    radian = np.arccos(cos_angle)
    heading_error =  radian * 180 / np.pi

    return heading_error


def get_centerline_length_to_conflict_point(conflict_point,lanelet,min_interval_distance):
    centerline_point_list = lanelet.centerline
    extend_centerline_point_list = insert_node_to_meet_min_interval(centerline_point_list,min_interval_distance)
    length = 0
    for index in range(len(extend_centerline_point_list)-1):
        if is_equal_point(extend_centerline_point_list[index],conflict_point):
            break
        else:
            length += distance(extend_centerline_point_list[index],extend_centerline_point_list[index+1])

    return length

def get_centerline_length_ahead_vehicle(vehicle_ms_dict,current_lanelet,min_interval_distance):
    # first find the closet centerline point
    centerline_point_list = current_lanelet.centerline
    extend_centerline_point_list = insert_node_to_meet_min_interval(centerline_point_list,min_interval_distance)
    vehicle_pos = [vehicle_ms_dict['x'],vehicle_ms_dict['y']]
    closet_point_index = get_closet_centerline_point(vehicle_pos,extend_centerline_point_list)

    length = 0
    if closet_point_index == len(extend_centerline_point_list) - 1:
        return length
    else:
        # accumulate length
        for i in range(len(extend_centerline_point_list[closet_point_index:])-1):
            length += distance(extend_centerline_point_list[closet_point_index + i],extend_centerline_point_list[closet_point_index + i + 1])

        return length

def get_centerline_length_between_vehicle_and_conflict_point(vehicle_ms_dict,conflict_point,current_lanelet,min_interval_distance):
    # first find the closet centerline point
    centerline_point_list = current_lanelet.centerline
    extend_centerline_point_list = insert_node_to_meet_min_interval(centerline_point_list,min_interval_distance)
    vehicle_pos = [vehicle_ms_dict['x'],vehicle_ms_dict['y']]
    closet_point_index = get_closet_centerline_point(vehicle_pos,extend_centerline_point_list)

    length = 0
    if closet_point_index == len(extend_centerline_point_list) - 1 or is_equal_point(extend_centerline_point_list[closet_point_index],conflict_point):
        return length
    else:
        # accumulate length
        for i in range(len(extend_centerline_point_list[closet_point_index:])-1):
            if not is_equal_point(extend_centerline_point_list[closet_point_index + i + 1],conflict_point):
                length += distance(extend_centerline_point_list[closet_point_index + i],extend_centerline_point_list[closet_point_index + i + 1])
            else:
                return length


def get_velocity_vector_along_frenet_coordinate(vehicle_ms_dict,current_lanelet,min_interval_distance):
    # first find the closet centerline point
    centerline_point_list = current_lanelet.centerline
    extend_centerline_point_list = insert_node_to_meet_min_interval(centerline_point_list,min_interval_distance)
    vehicle_pos = [vehicle_ms_dict['x'],vehicle_ms_dict['y']]
    closet_point_index = get_closet_centerline_point(vehicle_pos,extend_centerline_point_list)
    vehicle_velocity_vector = np.array([vehicle_ms_dict['vx'],vehicle_ms_dict['vy']])

    
    # calculate the heading along the lanelet
    if closet_point_index < len(extend_centerline_point_list) - 1:
        lanelet_heading_vector = np.array([extend_centerline_point_list[closet_point_index+1].x - extend_centerline_point_list[closet_point_index].x, extend_centerline_point_list[closet_point_index+1].y - extend_centerline_point_list[closet_point_index].y])
    else:
        lanelet_heading_vector = np.array([extend_centerline_point_list[closet_point_index].x - extend_centerline_point_list[closet_point_index-1].x, extend_centerline_point_list[closet_point_index].y - extend_centerline_point_list[closet_point_index-1].y])

    # calculate the velocity along the frenet
    L_lanelet_heading = np.sqrt(lanelet_heading_vector.dot(lanelet_heading_vector))
    project_vehicle_velocity_vector = vehicle_velocity_vector.dot(lanelet_heading_vector) / (L_lanelet_heading)

    return project_vehicle_velocity_vector

def get_route_bounds_points(route_lanelet, min_interval_distance):
    min_interval_distance = min_interval_distance/2
    left_bound_points = []
    right_bound_points = []
    for lanelet in route_lanelet:
        left_bound = lanelet.leftBound
        right_bound = lanelet.rightBound
        for i in range(len(left_bound)):
            left_bound_points.append(left_bound[i])
        for j in range(len(right_bound)):
            right_bound_points.append(right_bound[j])
    left_bound_points = insert_node_to_meet_min_interval(left_bound_points, min_interval_distance)
    right_bound_points = insert_node_to_meet_min_interval(right_bound_points, min_interval_distance)

    return left_bound_points, right_bound_points

# def get_ego_route_point_with_heading_from_lanelet(route_lanelet, min_interval_distance):
#     centerline_point_list = [] # all centerline points on the route
#     for lanelet in route_lanelet:
#         if lanelet is route_lanelet[-1]:
#             for index in range(len(lanelet.centerline)):
#                 centerline_point_list.append(lanelet.centerline[index])
#         else:
#             for index in range(len(lanelet.centerline)-1):
#                 centerline_point_list.append(lanelet.centerline[index])
#     # print(centerline_point_list, type(centerline_point_list))
#     extend_centerline_point_list = insert_node_to_meet_min_interval(centerline_point_list, min_interval_distance)

#     centerline_point_list_with_heading = get_centerline_point_list_with_heading_and_average_interval(extend_centerline_point_list, min_interval_distance)
#     # print(centerline_point_list_with_heading)
#     return centerline_point_list_with_heading

def get_ego_route_point_with_heading_from_point_list(ego_predict_route, min_interval_distance):
    point_list = []
    point_previous = None
    for route_speedpoint in ego_predict_route:
        point_x = route_speedpoint[0]
        point_y = route_speedpoint[1]
        if [point_x, point_y] == point_previous:
            continue
        else:
            point_previous = [point_x, point_y]
        point_list.append([point_x, point_y])
    # print('look', ego_predict_route, point_list)
    route_point_with_heading = get_centerline_point_list_with_heading_and_average_interval(point_list, min_interval_distance)

    return route_point_with_heading

def get_ego_target_speed_from_point_list(ego_predict_route):
    route_point_speed_list = []
    for route_point in ego_predict_route:
        point_speed = route_point[2]
        route_point_speed_list.append(point_speed)
    
    return route_point_speed_list

def get_lane_observation_and_future_route_points(ego_state_dict, vehilce_route, vehilce_route_target_speed, control_steering):
    vehicle_pos = ego_state_dict['pos']
    vehicle_heading = ego_state_dict['heading']
    vehicle_speed = ego_state_dict['speed']
    
    future_points = [] # for render next 5 route points
    # first find the closet route point
    closet_point_index = get_closet_front_centerline_point(vehicle_pos, vehilce_route)

    center_point = vehilce_route[closet_point_index]
    current_target_speed = vehilce_route_target_speed[closet_point_index]

    future_points.append(vehilce_route[closet_point_index])
    print('current_lane_heading', center_point[2] * 180 / np.pi)
    # print(vehicle_pos, center_point)
    
    ego_x_in_point_axis = (vehicle_pos[1] - center_point[1])*np.cos(center_point[2]) - (vehicle_pos[0] - center_point[0])*np.sin(center_point[2])
    ego_y_in_point_axis = (vehicle_pos[1] - center_point[1])*np.sin(center_point[2]) + (vehicle_pos[0] - center_point[0])*np.cos(center_point[2])

    ego_heading_error_0 = center_point[2] - vehicle_heading
    ego_speed_x_in_point_axis = vehicle_speed * np.sin(ego_heading_error_0)
    ego_speed_y_in_point_axis = vehicle_speed * np.cos(ego_heading_error_0)

    # also get next 4 points' heading errors
    ego_heading_error_next_list = []
    require_num = 4
    remain_point_num = len(vehilce_route) - 1 - closet_point_index 
    if remain_point_num < require_num:
        for i in range(closet_point_index + 1, len(vehilce_route)):
            point_heading = vehilce_route[i][2]
            ego_heading_error_point = point_heading - vehicle_heading
            ego_heading_error_next_list.append(ego_heading_error_point)
            future_points.append(vehilce_route[i])
        while len(ego_heading_error_next_list) < require_num:
            if ego_heading_error_next_list:
                ego_heading_error_next_list.append(ego_heading_error_next_list[-1])
            else:
                ego_heading_error_next_list.append(ego_heading_error_0)
    else:
        for i in range(closet_point_index + 1, closet_point_index + require_num + 1):
            point_heading = vehilce_route[i][2]
            ego_heading_error_point = point_heading - vehicle_heading
            ego_heading_error_next_list.append(ego_heading_error_point)
            future_points.append(vehilce_route[i])

    # left or right judgement
    # right_bound = current_lanelet.rightBound
    # left_bound = current_lanelet.leftBound
    # if right_bound == 0 or left_bound == 0:
    #     return None
    
    # return
    if control_steering:
        lane_observation = [ego_x_in_point_axis, ego_speed_x_in_point_axis, ego_speed_y_in_point_axis, ego_heading_error_0] + ego_heading_error_next_list
    else:
        lane_observation = [ego_heading_error_0] + ego_heading_error_next_list

    return lane_observation, current_target_speed, future_points

def get_ego_next_pos(ego_state_dict):
    next_point_pos_x = ego_state_dict['pos'][0] + ego_state_dict['speed'] * math.cos(ego_state_dict['heading'])
    next_point_pos_y = ego_state_dict['pos'][1] + ego_state_dict['speed'] * math.sin(ego_state_dict['heading'])
    next_point_pos = (next_point_pos_x, next_point_pos_y)

    next_x_in_ego_axis = (next_point_pos[1] - ego_state_dict['pos'][1])*np.cos(ego_state_dict['heading']) - (next_point_pos[0] - ego_state_dict['pos'][0])*np.sin(ego_state_dict['heading'])
    next_y_in_ego_axis = (next_point_pos[1] - ego_state_dict['pos'][1])*np.sin(ego_state_dict['heading']) + (next_point_pos[0] - ego_state_dict['pos'][0])*np.cos(ego_state_dict['heading'])
    # print('next_pos_from_ego', [next_x_in_ego_axis, next_y_in_ego_axis])
    return [next_x_in_ego_axis, next_y_in_ego_axis]

def get_distance_from_center(vehicle_pos, current_lanelet, route_lanelet, min_interval_distance):
    # first find the closet centerline point
    centerline_point_list = current_lanelet.centerline
    extend_centerline_point_list = insert_node_to_meet_min_interval(centerline_point_list,min_interval_distance)
    closet_point_index = get_closet_centerline_point(vehicle_pos,extend_centerline_point_list)

    ego_center_point = Point3d(getId(),vehicle_pos[0],vehicle_pos[1],0,getAttributes())

    distance_from_center = distance(extend_centerline_point_list[closet_point_index],ego_center_point)

    # left or right judgement
    right_bound = current_lanelet.rightBound
    left_bound = current_lanelet.leftBound
    if right_bound == 0 or left_bound == 0:
        return None
    # extend_right_bound_point_list = insert_node_to_meet_min_interval(right_bound_point_list,min_interval_distance)
    # extend_left_bound_point_list = insert_node_to_meet_min_interval(left_bound_point_list,min_interval_distance)

    # closet_right_bound_point_index = get_closet_centerline_point(extend_centerline_point_list[closet_point_index],extend_right_bound_point_list)
    # closet_left_bound_point_index = get_closet_centerline_point(extend_centerline_point_list[closet_point_index],extend_left_bound_point_list)

    distance_to_right_bound = distance(right_bound,ego_center_point)
    # print('distance_to right_bound:',distance_to_right_bound)
    distance_to_left_bound = distance(left_bound,ego_center_point)
    # print('distance_to left_bound:',distance_to_left_bound)

    right_width = distance(right_bound,extend_centerline_point_list[closet_point_index])
    # print('right_width:',right_width)
    left_width = distance(left_bound,extend_centerline_point_list[closet_point_index])
    # print('left_width:',left_width)

    if right_width == 0 or left_width == 0:
        return None

    if distance_to_left_bound > distance_to_right_bound:
        normalize_distance_from_center = distance_from_center / right_width
    else:
        normalize_distance_from_center = -distance_from_center / left_width
   

    # print('distance_to normalize_distance_from_center:',normalize_distance_from_center)
    return normalize_distance_from_center

def get_route_heading_errors(vehicle_pos, vehicle_polygon, current_lanelet, route_lanelet, min_interval_distance, total_ahead_point_num):
    # use 50 ahead point (= total_ahead_point_num) to find ahead 10 heading errors
    # first find the closet centerline point
    calculate_ahead_point = []
    heading_error_angle = []
    heading_error_rad = []
    heading_errors = []
    centerline_point_list = current_lanelet.centerline
    extend_centerline_point_list = insert_node_to_meet_min_interval(centerline_point_list,min_interval_distance)
    closet_point_index = get_closet_centerline_point(vehicle_pos,extend_centerline_point_list)

    vehicle_heading = get_vehicle_heading(vehicle_polygon)

    # second find the current lanelet index
    current_lanelet_index = 0
    for index, ll in enumerate(route_lanelet):
        if ll.id == current_lanelet.id:
            current_lanelet_index = index
            break
    
    lanelet_point_num = len(extend_centerline_point_list) - closet_point_index - 1
    remain_to_insert_num = total_ahead_point_num

    # use ahead 50 point
    if lanelet_point_num < remain_to_insert_num:
        calculate_ahead_point.extend(extend_centerline_point_list[closet_point_index+1:])
        remain_to_insert_num -= lanelet_point_num
        for ll in route_lanelet[current_lanelet_index+1:]:
            centerline_point_list = ll.centerline
            extend_centerline_point_list = insert_node_to_meet_min_interval(centerline_point_list,min_interval_distance)
            # print('ex:',len(extend_centerline_point_list))
            # print('ah:',len(calculate_ahead_point))
            if len(calculate_ahead_point)>0 and is_equal_point(extend_centerline_point_list[0],calculate_ahead_point[-1]):
                extend_centerline_point_list.pop(0)
            lanelet_point_num = len(extend_centerline_point_list)
            if lanelet_point_num < remain_to_insert_num:
                calculate_ahead_point.extend(extend_centerline_point_list)
                remain_to_insert_num -= lanelet_point_num
            else:
                calculate_ahead_point.extend(extend_centerline_point_list[:remain_to_insert_num])
                break
            
    else:
        calculate_ahead_point.extend(extend_centerline_point_list[closet_point_index+1:closet_point_index+remain_to_insert_num+1])

    # print('lens of cal:',len(calculate_ahead_point))

    actual_ahead_point_num = len(calculate_ahead_point)

    if actual_ahead_point_num < total_ahead_point_num and len(calculate_ahead_point) > 1:
        # less than total num means the current lanelet is the last lanelet
        # we need to supplement by hand
        delta_x = calculate_ahead_point[-1].x - calculate_ahead_point[-2].x
        delta_y = calculate_ahead_point[-1].y - calculate_ahead_point[-2].y
        for i in range(total_ahead_point_num-actual_ahead_point_num):
            pt_x = calculate_ahead_point[actual_ahead_point_num-1].x + (i+1) * delta_x
            pt_y = calculate_ahead_point[actual_ahead_point_num-1].y + (i+1) * delta_y
            new_ahead_point = Point3d(getId(),pt_x,pt_y,0,getAttributes())
            calculate_ahead_point.append(new_ahead_point)
    elif 0 < actual_ahead_point_num < total_ahead_point_num and len(calculate_ahead_point) < 2:
        delta_x = extend_centerline_point_list[closet_point_index].x - extend_centerline_point_list[closet_point_index-1].x
        delta_y = extend_centerline_point_list[closet_point_index].y - extend_centerline_point_list[closet_point_index-1].y
        for i in range(total_ahead_point_num-actual_ahead_point_num):
            pt_x = calculate_ahead_point[actual_ahead_point_num-1].x + (i+1) * delta_x
            pt_y = calculate_ahead_point[actual_ahead_point_num-1].y + (i+1) * delta_y
            new_ahead_point = Point3d(getId(),pt_x,pt_y,0,getAttributes())
            calculate_ahead_point.append(new_ahead_point)
    elif actual_ahead_point_num == 0:
        return None



    # print('extend lens of cal:',len(calculate_ahead_point))


    # take 10 in 50 as order:[0, 1, 2, 3, 5, 8, 13, 21, 34, 49]
    order = [0, 1, 2, 3, 5, 8, 13, 21, 34, 49]
    for i in range(len(order)):
        if order[i] < len(calculate_ahead_point) -1: 
            waypoint_heading_vector = np.array([calculate_ahead_point[order[i+1]].x - calculate_ahead_point[order[i]].x, calculate_ahead_point[order[i+1]].y - calculate_ahead_point[order[i]].y])
        else:
            waypoint_heading_vector = np.array([calculate_ahead_point[order[i]].x - calculate_ahead_point[order[i-1]].x, calculate_ahead_point[order[i]].y - calculate_ahead_point[order[i-1]].y])
        
        angle_with_sign = calculate_angle_with_sign(vehicle_heading, waypoint_heading_vector)

        heading_error_angle.append(angle_with_sign)

        heading_error_rad.append(math.radians(angle_with_sign))

        h_error = math.sin(math.radians(angle_with_sign))

        heading_errors.append(h_error)

    # print('angle:',heading_error_angle)
    # print('angle_rad:',heading_error_rad)
    print('heading_errors:', np.arcsin(heading_errors[0]))

    # import pdb
    # pdb.set_trace()

    return heading_errors


def calculate_angle_with_sign(vehicle_heading_vector,waypoint_heading_vector):

    # v1 rotate to v2, clockwise is positive, unclockwise is negative
    TheNorm = np.linalg.norm(vehicle_heading_vector) * np.linalg.norm(waypoint_heading_vector)
    # cross
    cross = np.cross(vehicle_heading_vector, waypoint_heading_vector) / TheNorm
    cross = np.clip(cross,-1,1)
    rho = np.rad2deg(np.arcsin(cross))
    # dot
    dot = np.dot(vehicle_heading_vector, waypoint_heading_vector) / TheNorm
    dot = np.clip(dot,-1,1)
    theta = np.rad2deg(np.arccos(dot))

    if np.isnan(theta):
        import pdb
        pdb.set_trace()
        print('TheNorm:',TheNorm)
        print('waypoint norm:',np.linalg.norm(waypoint_heading_vector))

    if rho < 0:
        return - theta
    else:
        return theta

        

def calculate_complete_ratio_along_planning_route(vehicle_pos,current_lanelet,route_lanelet,min_interval_distance):
    # first find the closet centerline point
    centerline_point_list = current_lanelet.centerline
    extend_centerline_point_list = insert_node_to_meet_min_interval(centerline_point_list,min_interval_distance)
    closet_point_index = get_closet_centerline_point(vehicle_pos,extend_centerline_point_list)

    # second find the current lanelet index
    current_lanelet_index = 0
    for index,ll in enumerate(route_lanelet):
        if ll.id == current_lanelet.id:
            current_lanelet_index = index
            break
    
    total_forward_distance = 0
    # third calculate forward distance
    for ll in route_lanelet[0:current_lanelet_index+1]:
        if ll.id != current_lanelet.id:
            total_forward_distance += lanelet_length(ll)
        else:
            for i in range(closet_point_index):
                total_forward_distance += distance(extend_centerline_point_list[i],extend_centerline_point_list[i+1])
            
            break

    # final calculate the total route length
    total_route_length = 0
    for ll in route_lanelet:
        total_route_length += lanelet_length(ll)

    
    complete_ratio = total_forward_distance/ total_route_length


    return complete_ratio

def is_route_contain_turning(route_lanelet,min_turning_rad,min_interval_distance):
    route_center_line_point_list = []
    heading_list = []
    for ll in route_lanelet:
        centerline_point_list = ll.centerline
        # extend_centerline_point_list = insert_node_to_meet_min_interval(centerline_point_list,min_interval_distance)
        if len(route_center_line_point_list) > 0 and is_equal_point(centerline_point_list[0],route_center_line_point_list[-1]):
            centerline_point_list.pop(0)
        route_center_line_point_list.extend(centerline_point_list)

    for i in range(1,len(route_center_line_point_list)):
        heading_vector = np.array([route_center_line_point_list[i].x - route_center_line_point_list[i-1].x, route_center_line_point_list[i].y - route_center_line_point_list[i-1].y])
        heading_list.append(heading_vector)

    for h in  heading_list[1:]:
        angle_with_sign = calculate_angle_with_sign(h,heading_list[0])
        if angle_with_sign > 30 or angle_with_sign < -30:
            return True
    
    return False







