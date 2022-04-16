import sys
sys.path.append("..")
from visualize import update_ego_others_param, render_ego_others_param, render_ego_others_param_with_surrounding_highlight, render_ego_others_and_ghost_param_with_surrounding_highlight
# from main_visualize_data import update_plot_with_param
from utils import map_vis_lanelet2
from utils import dataset_reader
import random
try:
    import lanelet2
    use_lanelet2_lib = True
    print("Using Lanelet2 visualization")
except:
    import warnings
    string = "Could not import lanelet2. It must be built and sourced, " + \
             "see https://github.com/fzi-forschungszentrum-informatik/Lanelet2 for details."
    warnings.warn(string)
    print("Using visualization without lanelet2.")
    use_lanelet2_lib = False
    from utils import map_vis_without_lanelet


import argparse
import os
import glob

# import python3-tk
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import lanelet_relationship
import geometry

# the map data in intersection dataset
# x axis direction is from left to right
# y axis direction is from top to bottom

class interaction_map:
    def __init__(self, args, route_type):
        self._root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        print('root:',self._root_dir)
        self._map_dir = os.path.join(self._root_dir, "maps")
        self._tracks_dir = os.path.join(self._root_dir, "recorded_trackfiles")
        lanelet_map_ending = ".osm"

        # load files
        if isinstance(args, dict):
            self._lanelet_map_file = os.path.join(self._map_dir, args['scenario_name'] + lanelet_map_ending)
            self._scenario_dir = os.path.join(self._tracks_dir, args['scenario_name'])

            self._track_file_name = os.path.join(
                self._scenario_dir,
                "vehicle_tracks_" + str(args['track_file_number']).zfill(3) + ".csv"
            )
            self._pedestrian_file_name = os.path.join(
                self._scenario_dir,
                "pedestrian_tracks_" + str(args['track_file_number']).zfill(3) + ".csv"
            )

            self._trajectory_file_name = str(args['trajectory_file_name'])

            self.port = args['port']
             
        else:
            self._lanelet_map_file = os.path.join(self._map_dir, args.scenario_name + lanelet_map_ending)
            self._scenario_dir = os.path.join(self._tracks_dir, args.scenario_name)

            self._track_file_name = os.path.join(
                self._scenario_dir,
                "vehicle_tracks_" + str(args.track_file_number).zfill(3) + ".csv"
            )
            self._pedestrian_file_name = os.path.join(
                self._scenario_dir,
                "pedestrian_tracks_" + str(args.track_file_number).zfill(3) + ".csv"
            )

            self._trajectory_file_name = str(args.trajectory_file_name)

            self.port = args.port

        # check folders and files
        error_string = ""
        if not os.path.isdir(self._tracks_dir):
            error_string += "Did not find track file directory \"" + self._tracks_dir + "\"\n"
        if not os.path.isdir(self._map_dir):
            error_string += "Did not find map file directory \"" + self._tracks_dir + "\"\n"
        if not os.path.isdir(self._scenario_dir):
            error_string += "Did not find scenario directory \"" + self._scenario_dir + "\"\n"
        if not os.path.isfile(self._lanelet_map_file):
            error_string += "Did not find lanelet map file \"" + self._lanelet_map_file + "\"\n"
        if not os.path.isfile(self._track_file_name):
            error_string += "Did not find track file \"" + self._track_file_name + "\"\n"
        if not os.path.isfile(self._pedestrian_file_name):
            flag_ped = 0
        else:
            flag_ped = 1
        if error_string != "":
            error_string += "Type --help for help."
            raise IOError(error_string)

        # load vehicles' track based on pre-defined route type
        if route_type == 'predict':
            self.track_dict = dataset_reader.read_trajectory(self._trajectory_file_name)
        elif route_type == 'ground_truth' or route_type == 'centerline':
            self.track_dict = dataset_reader.read_tracks(self._track_file_name)
        self.route_type = route_type
        # total num of tracks
        self.track_num = len(self.track_dict.keys())
        # load pedestrian track
        self._pedestrian_dict = dict()
        if isinstance(args, dict):
            if args['load_mode'] == 'both' or args['load_mode'] == 'pedestrian' and flag_ped:
                self._pedestrian_dict = dataset_reader.read_pedestrian(self._pedestrian_file_name)
        else:
            if args.load_mode == 'both' or args.load_mode == 'pedestrian' and flag_ped:
                self._pedestrian_dict = dataset_reader.read_pedestrian(self._pedestrian_file_name)
        
        # initialize vehicles' id list and so on
        self._ego_vehicle_id_list = list()
        self._ego_vehicle_track_dict = dict()
        self._ego_vehicle_start_end_state_dict = dict()

        self._others_vehicle_id_list = list()
        self._others_vehicle_track_dict = dict()
        
        # create a figure
        self.fig, self.axes = plt.subplots(1, 1, facecolor = 'lightgray')  # figure backcolor (figure size > map render size)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # print('axes type:', type(self.axes))
        self.axes.set_facecolor('white')  # map render backcolor
        self.fig.canvas.set_window_title("Interaction Dataset Visualization " + str(self.port))

        self.ego_patches_dict = dict()
        self.other_patches_dict = dict()
        self.ghost_patches_dict = dict()

        # use for collison detection and distance calculation
        self.ego_vehicle_polygon = dict()
        self.other_vehicle_polygon = dict()
        self.ghost_vehicle_polygon = dict()

        self.other_vehicle_motion_state = dict() # for saving current step other vehicles motion state
        self.ghost_vehicle_motions_state = dict()

        # use for vehicle id visualaztion
        self.text_dict = dict()
        
        # use for time percentage visualization
        # self.title_text = self.fig.suptitle("")
        
        # load and draw the lanelet2 map, either with or without the lanelet2 library
        lat_origin = 0.  # origin is necessary to correctly project the lat lon values in the osm file to the local
        lon_origin = 0.  # coordinates in which the tracks are provided; we decided to use (0|0) for every scenario
        print("Loading map...")
        
        self.laneletmap = None
        
        self.rules_map = {"vehicle": lanelet2.traffic_rules.Participants.Vehicle}
        self.projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(lat_origin, lon_origin))
        
        # self.layers = {"Points": self.laneletmap.pointLayer, "Line Strings": self.laneletmap.lineStringLayer, "Polygons": self.laneletmap.polygonLayer,
        #   "Lanelets": self.laneletmap.laneletLayer, "Areas": self.laneletmap.areaLayer, "Regulatory Elements": self.laneletmap.regulatoryElementLayer}


    def __del__(self):
        plt.close('all')


    def change_predict_track_file(self, trajectory_file_name=None):
        self._trajectory_file_name = trajectory_file_name
        self.track_dict = dataset_reader.read_trajectory(self._trajectory_file_name)
        print('predict track file name:', track_file_number)
        self.track_num = len(self.track_dict.keys())
    

    def change_ground_truth_track_file(self, track_file_number=None, trajectory_file_name=None):
        self._track_file_name = os.path.join(
            self._scenario_dir,
            "vehicle_tracks_" + str(track_file_number).zfill(3) + ".csv")
        self.track_dict = dataset_reader.read_tracks(self._track_file_name)
        print('ground truth track file number:', track_file_number)
        self.track_num = len(self.track_dict.keys())


    def map_init(self):
        self._ego_vehicle_start_end_state_dict.clear()
        self._ego_vehicle_track_dict.clear()

        self._others_vehicle_track_dict.clear()

        self.ego_patches_dict.clear()
        self.other_patches_dict.clear()
        self.ghost_patches_dict.clear()

        self.ego_vehicle_polygon.clear()
        self.other_vehicle_polygon.clear()
        self.ghost_vehicle_polygon.clear()

        self.other_vehicle_motion_state.clear()
        self.ghost_vehicle_motions_state.clear()
        self.axes.clear()

        # initialize map
        self.laneletmap = lanelet2.io.load(self._lanelet_map_file, self.projector)

        # render static map and get pixel ratio
        self.map_x_bound, self.map_y_bound = map_vis_lanelet2.draw_lanelet_map(self.laneletmap, self.axes)
        self.fig_width, self.fig_height = self.fig.get_size_inches()*self.fig.dpi
        self.fig_width = int(self.fig_width) # figure size >= map render size as there exist empty space between figure bound and axes bound
        self.fig_height = int(self.fig_height)
        # print('rect:',self.axes.get_position())
        self.map_width_ratio = (self.map_x_bound[1] - self.map_x_bound[0])/self.fig_width
        self.map_height_ratio = (self.map_y_bound[1] - self.map_y_bound[0])/self.fig_height

        # pixel ration: (m/pixel)
        # print('map width ration:',self.map_width_ratio)
        # print('map height ratio:',self.map_height_ratio)

        # ego fixed pos image size 
        self.image_half_width = int(15 / self.map_width_ratio)
        self.image_half_height = int(15 / self.map_height_ratio)
        
        self.routing_cost = lanelet2.routing.RoutingCostDistance(0.) # zero cost for lane changes
        self.traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, self.rules_map['vehicle'])
        
        # route graph is used for lanelet relationship searching
        self.routing_graph = lanelet2.routing.RoutingGraph(self.laneletmap, self.traffic_rules, [self.routing_cost])

    
    def random_choose_ego_vehicle(self):
        # randome select vehicles whose length is less than 5.5m as egos
        vehicle_id_list = self.track_dict.keys()
        vehicle_limited_length_id_list = []
        for vehicle_id in vehicle_id_list:
            vehicle_info = self.track_dict[vehicle_id]
            if vehicle_info.length <= 5.5:
                vehicle_limited_length_id_list.append(vehicle_id)

        # split track into egos' and others'
        self._ego_vehicle_id_list = random.sample(vehicle_limited_length_id_list, self._ego_vehicle_num)
        self._others_vehicle_id_list = list(set(vehicle_id_list) - set(self._ego_vehicle_id_list))

        self._ego_vehicle_track_dict = {key:value for key,value in self.track_dict.items() if key in self._ego_vehicle_id_list}
        self._others_vehicle_track_dict = {key:value for key,value in self.track_dict.items() if key in self._others_vehicle_id_list}

        # read ego vehicles start end state for reset
        for index, vehicle_id in enumerate(self._ego_vehicle_id_list):
            ego_info = self.track_dict[vehicle_id]
            self._ego_vehicle_start_end_state_dict[vehicle_id] = [ego_info.time_stamp_ms_first, ego_info.time_stamp_ms_last, length, width, ego_info.motion_states[ego_info.time_stamp_ms_first], ego_info.motion_states[ego_info.time_stamp_ms_last]]
            

    def specify_id_choose_ego_vehicle(self, ego_id_list, ego_start_timestamp=None):
        # split track into egos' and others'
        vehicle_id_list = self.track_dict.keys()
        self._ego_vehicle_id_list = ego_id_list
        self._others_vehicle_id_list = list(set(vehicle_id_list) - set(self._ego_vehicle_id_list))

        self._ego_vehicle_track_dict = {key:value for key,value in self.track_dict.items() if key in self._ego_vehicle_id_list}
        self._others_vehicle_track_dict = {key:value for key,value in self.track_dict.items() if key in self._others_vehicle_id_list}
    
        # read ego vehicles start & end state
        for index, vehicle_id in enumerate(self._ego_vehicle_id_list):
            print('vehicle id:', vehicle_id)
            ego_info = self.track_dict[vehicle_id]

            # origin shape of the car
            width = ego_info.width
            length = ego_info.length
            # if we need to specify ego's shape for rl training
            # width = 1.8
            # length = 5

            if ego_start_timestamp:
                ego_timestamp_ms_first = int(ego_start_timestamp[0])
                ego_timestamp_ms_last = ego_timestamp_ms_first + 100 * 100 - 100
            else:
                ego_timestamp_ms_first = ego_info.time_stamp_ms_first
                ego_timestamp_ms_last = ego_info.time_stamp_ms_last

            print('time_stamp_ms_first:', ego_timestamp_ms_first)
            print('time_stamp_ms_last:', ego_timestamp_ms_last)
            self._ego_vehicle_start_end_state_dict[vehicle_id] = [ego_timestamp_ms_first, ego_timestamp_ms_last, length, width, ego_info.motion_states[ego_timestamp_ms_first], ego_info.motion_states[ego_timestamp_ms_last]]
            # print('_ego_vehicles_start_pos:',ego_info.motion_states[ego_info.time_stamp_ms_first].x,ego_info.motion_states[ego_info.time_stamp_ms_first].y)
            # print('_ego_vehicles_start_vel:',ego_info.motion_states[ego_info.time_stamp_ms_first].vx,ego_info.motion_states[ego_info.time_stamp_ms_first].vy)
            # print('_ego_vehicles_end_pos:',ego_info.motion_states[ego_info.time_stamp_ms_last].x,ego_info.motion_states[ego_info.time_stamp_ms_last].y)
            return True


    def update_param(self, current_time, timestamp_min, timestamp_max, ego_state_dict):
        # visualize map and vehicles
        ego_shape_dict = dict()
        for vehicle_id, vehicle_info in self._ego_vehicle_start_end_state_dict.items():
            ego_shape_dict[vehicle_id] = (vehicle_info[2], vehicle_info[3]) # length and width

        # update ego vehicles and other vehicles postion
        update_ego_others_param(current_time, timestamp_min, timestamp_max, self.other_vehicle_polygon, self.other_vehicle_motion_state, self._others_vehicle_track_dict, self.ego_vehicle_polygon, ego_state_dict, ego_shape_dict, self.ego_patches_dict, self.ghost_vehicle_polygon, self.ghost_vehicle_motions_state, self._ego_vehicle_track_dict, self._pedestrian_dict)
        # update_plot_with_param_new(current_time,timestamp_min,timestamp_max,self.other_patches_dict,self.other_vehicle_polygon,self.ego_patches_dict,self.text_dict,self.axes,self.fig,self.title_text,self._others_vehicle_track_dict,ego_state_dict,ego_shape_dict,self._pedestrian_dict)
        # update_plot_with_param(current_time,timestamp_min,timestamp_max,self.other_patches_dict,self.other_vehicle_polygon,self.other_vehicle_motion_state,self.ego_patches_dict,self.ego_vehicle_polygon,self.text_dict,self.axes,self.fig,self.title_text,self._others_vehicle_track_dict,ego_state_dict,ego_shape_dict,self._pedestrian_dict)
        # print('look:', self.ego_patches_dict)
        # update_ego_others_param(current_time, timestamp_min, timestamp_max, self.other_vehicle_polygon, self.other_vehicle_motion_state, self._others_vehicle_track_dict,self.ego_vehicle_polygon,ego_state_dict,ego_shape_dict,self.ego_patches_dict,self._pedestrian_dict)


    def render(self, ego_state_dict):
        plt.ion()

        render_ego_others_param(self.other_patches_dict, self.other_vehicle_polygon, self.other_vehicle_motion_state,self.ego_patches_dict,self.ego_vehicle_polygon,ego_state_dict,self.text_dict,self.axes,self.fig)
        
        plt.xticks([]) 
        plt.yticks([])
        plt.axis('off')

        plt.pause(0.05)
        plt.show()
        plt.ioff()


    def render_with_highlight(self, ego_state_dict, surrounding_vehicle_id_list):
        plt.ion()

        render_ego_others_param_with_surrounding_highlight(self.other_patches_dict, self.other_vehicle_polygon, self.other_vehicle_motion_state, self.ego_patches_dict, self.ego_vehicle_polygon, ego_state_dict, surrounding_vehicle_id_list, self.text_dict, self.axes, self.fig)

        plt.xticks([]) 
        plt.yticks([])
        plt.axis('off')

        plt.pause(0.02)
        plt.show()
        plt.ioff()


    def render_with_highlight_and_ghost(self, ego_state_dict, surrounding_vehicle_id_list):
        plt.ion()

        render_ego_others_and_ghost_param_with_surrounding_highlight(self.other_patches_dict,self.other_vehicle_polygon,self.other_vehicle_motion_state,self.ego_patches_dict,self.ego_vehicle_polygon,ego_state_dict,surrounding_vehicle_id_list,self.ghost_patches_dict, self.ghost_vehicle_polygon, self.ghost_vehicle_motions_state, self.text_dict, self.axes, self.fig)

        plt.xticks([]) 
        plt.yticks([])
        plt.axis('off')

        plt.pause(0.001)
        plt.savefig('')
        plt.show()
        plt.ioff()


    def render_planning_centerline(self, ego_route_lanelet, ego_current_lanelet, ego_vehicle_dict):
        
        plt.ion()
        for k,v in ego_route_lanelet.items():
            ego_current_state = ego_vehicle_dict[k]._current_state
            map_vis_lanelet2.draw_route_center_line(v, ego_current_lanelet[k], ego_current_state, self.axes)
        plt.xticks([]) 
        plt.yticks([])
        plt.axis('off')

        plt.show()
        plt.ioff()


    def render_route(self, route):
        plt.ion()
        for k,v in route.items():
            route_point_list = []
            for point in v:
                route_point_list.append([point[0], point[1]])
            map_vis_lanelet2.draw_route(route_point_list, self.axes)
        plt.xticks([]) 
        plt.yticks([])
        plt.axis('off')

        plt.show()
        plt.ioff()


    def render_route_bounds(self, route_left_bounds, route_right_bounds):
        plt.ion()
        for k,v in route_left_bounds.items():
            map_vis_lanelet2.draw_route_bounds(route_left_bounds[k], self.axes)
            map_vis_lanelet2.draw_route_bounds(route_right_bounds[k], self.axes)
        plt.xticks([]) 
        plt.yticks([])
        plt.axis('off')

        plt.show()
        plt.ioff()


    def render_closet_bound_point(self, previous_closet_points, current_closet_points):
        plt.ion()
        map_vis_lanelet2.draw_closet_bound_point(previous_closet_points, current_closet_points, self.axes)
        plt.xticks([]) 
        plt.yticks([])
        plt.axis('off')

        plt.show()
        plt.ioff()


    def render_future_route_points(self, ego_previous_points_dict, ego_future_points_dict):
        
        plt.ion()
        map_vis_lanelet2.draw_ego_future_route(ego_previous_points_dict, ego_future_points_dict, self.axes)
        plt.xticks([]) 
        plt.yticks([])
        plt.axis('off')

        plt.show()
        plt.ioff()


    def save_jpg(self, ego_id, current_time):
        # pass
        image_folder = '/home/developer/workspace/interaction-dataset-master/python/interaction_rl/results/saved_images/' 
        plt.savefig(image_folder  + str(ego_id) + '_' + str(current_time) + '.jpg')


    def render_conflict_centerline_point(self,following_route_conflict_lanelet):
        plt.ion()
        print('render_conflict_centerline_point')
        for k in following_route_conflict_lanelet.keys():
            
            # print('upupuiup')
            # following_route_conflict_lanelet_dict,following_route_conflict_lanelet_previous_list = lanelet_relationship.get_conflict_lanelet_dict_along_route(self,ego_route,following_route,previous_route)
            # print('upupuiup')
            for f_ll_id,c_ll_pair in following_route_conflict_lanelet[k].items():
                for c_ll in c_ll_pair[1]:
                    overlap_point_list = geometry.get_overlap_point_list(c_ll_pair[0],c_ll)
                    map_vis_lanelet2.draw_conflict_point(overlap_point_list,self.axes)


        plt.xticks([]) 
        plt.yticks([])
        plt.axis('off')

        plt.show()
        plt.ioff()