#!/usr/bin/env python

try:
    import lanelet2
    use_lanelet2_lib = True
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
import time

import matplotlib

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from utils import dataset_reader
from utils import dataset_types
from utils import map_vis_lanelet2
from utils import tracks_vis
from utils import dict_utils


# # a version needed param
# def update_plot_with_param(timestamp,timestamp_min,timestamp_max, other_patches_dict, other_vehicle_polygon,other_vehicle_motionstate ,ego_patches_dict,ego_vehicle_polygon,text_dict, axes,fig, title_text, others_track_dictionary, ego_state_dict,ego_shape_dict, pedestrian_dictionary):
#     # update text and tracks based on current timestamp
#     assert(timestamp <= timestamp_max), "timestamp=%i" % timestamp
#     assert(timestamp >= timestamp_min), "timestamp=%i" % timestamp
#     assert(timestamp % dataset_types.DELTA_TIMESTAMP_MS == 0), "timestamp=%i" % timestamp
#     percentage = (float(timestamp) / timestamp_max) * 100
#     title_text.set_text("\nts = {} / {} ({:.2f}%)".format(timestamp, timestamp_max, percentage))
#     # plot ego vehicles
#     tracks_vis.update_objects_plot_ego(ego_patches_dict, ego_vehicle_polygon, text_dict, axes,ego_shape_dict=ego_shape_dict,
#                                    ego_motionstate_dict=ego_state_dict)
#     # plot others
#     # tracks_vis.update_objects_plot_without_ego(timestamp, other_patches_dict, other_vehicle_polygon, text_dict, axes,
#     #                                other_track_dict=others_track_dictionary, pedest_dict=pedestrian_dictionary)

#     # plot others without postion conflict with ego vehicle
#     tracks_vis.update_objects_plot_without_ego_and_conflict(timestamp, other_patches_dict, other_vehicle_polygon,other_vehicle_motionstate,ego_vehicle_polygon, text_dict, axes,
#                                    other_track_dict=others_track_dictionary, pedest_dict=pedestrian_dictionary)
#     fig.canvas.draw()

# a version needed param without time percentage visualization
def update_ego_others_param(timestamp,timestamp_min,timestamp_max,other_vehicle_polygon, other_vehicle_motionstate,others_track_dictionary, ego_vehicle_polygon,ego_state_dict,ego_shape_dict, ego_patches_dict, ghost_vehicle_polygon,ghost_motionstate_dict, ghost_track_dict, pedestrian_dictionary):
    # # update text and tracks based on current timestamp
    # assert(timestamp <= timestamp_max), "timestamp=%i" % timestamp
    # assert(timestamp >= timestamp_min), "timestamp=%i" % timestamp
    # assert(timestamp % dataset_types.DELTA_TIMESTAMP_MS == 0), "timestamp=%i" % timestamp
    # # percentage = (float(timestamp) / timestamp_max) * 100
    # # title_text.set_text("\nts = {} / {} ({:.2f}%)".format(timestamp, timestamp_max, percentage))
    # # update ego vehicles
    # tracks_vis.update_objects_ego(ego_patches_dict=ego_patches_dict, ego_vehicle_polygon=ego_vehicle_polygon, ego_shape_dict=ego_shape_dict,
    #                                ego_motionstate_dict=ego_state_dict)
    # # plot others
    # # tracks_vis.update_objects_plot_without_ego(timestamp, other_patches_dict, other_vehicle_polygon, text_dict, axes,
    # #                                other_track_dict=others_track_dictionary, pedest_dict=pedestrian_dictionary)

    # # update others without postion conflict with ego vehicle
    # tracks_vis.update_objects_without_ego_and_conflict(timestamp, other_vehicle_polygon, other_vehicle_motionstate, ego_vehicle_polygon,
    #                                                     other_track_dict=others_track_dictionary, pedest_dict=pedestrian_dictionary)
    
    # update text and tracks based on current timestamp
    assert(timestamp <= timestamp_max), "timestamp=%i" % timestamp
    assert(timestamp >= timestamp_min), "timestamp=%i" % timestamp
    assert(timestamp % dataset_types.DELTA_TIMESTAMP_MS == 0), "timestamp=%i" % timestamp
    # percentage = (float(timestamp) / timestamp_max) * 100
    # title_text.set_text("\nts = {} / {} ({:.2f}%)".format(timestamp, timestamp_max, percentage))
    # update ego vehicles
    tracks_vis.update_objects_ego(ego_patches_dict=ego_patches_dict, ego_vehicle_polygon=ego_vehicle_polygon,ego_shape_dict=ego_shape_dict,
                                   ego_motionstate_dict=ego_state_dict)
    # plot others
    # tracks_vis.update_objects_plot_without_ego(timestamp, other_patches_dict, other_vehicle_polygon, text_dict, axes,
    #                                other_track_dict=others_track_dictionary, pedest_dict=pedestrian_dictionary)

    # update ghost vehicle
    tracks_vis.update_objects_ghost(timestamp, ghost_vehicle_polygon, ghost_motionstate_dict, ghost_track_dict)

    # update others without postion conflict with ego vehicle
    tracks_vis.update_objects_without_ego_and_conflict(timestamp, other_vehicle_polygon, other_vehicle_motionstate, ego_vehicle_polygon,
                                   other_track_dict=others_track_dictionary, pedest_dict=pedestrian_dictionary)
    
def render_ego_others_param(other_patches_dict, other_vehicle_polygon,other_vehicle_motionstate,ego_patches_dict,ego_vehicle_polygon,ego_state_dict,text_dict, axes,fig):
    # render ego vehicle
    tracks_vis.render_objects_ego(ego_patches_dict,ego_vehicle_polygon,ego_state_dict,text_dict, axes)

    # render others vehicle
    tracks_vis.render_objects_without_ego_and_conflict(other_patches_dict,other_vehicle_polygon,other_vehicle_motionstate,text_dict, axes)

    # update data
    # fig.canvas.draw()

def render_ego_others_param_with_surrounding_highlight(other_patches_dict, other_vehicle_polygon,other_vehicle_motionstate,ego_patches_dict, ego_vehicle_polygon,ego_state_dict,surrounding_vehicle_id_list,text_dict, axes,fig):
    # render ego vehicle
    tracks_vis.render_objects_ego(ego_patches_dict, ego_vehicle_polygon, ego_state_dict, text_dict, axes)

    # render others vehicle with highlight surrounding
    tracks_vis.render_objects_without_ego_and_conflict_with_highlight(other_patches_dict, other_vehicle_polygon, other_vehicle_motionstate, surrounding_vehicle_id_list, text_dict, axes)

    fig.canvas.draw()

def render_ego_others_and_ghost_param_with_surrounding_highlight(other_patches_dict, other_vehicle_polygon, other_vehicle_motionstate, ego_patches_dict,ego_vehicle_polygon,ego_state_dict,surrounding_vehicle_id_list, ghost_patches_dict, ghost_vehicle_polygon, ghost_vehicle_motionstate, text_dict, axes,fig):
    # render ego vehicle
    tracks_vis.render_objects_ego(ego_patches_dict,ego_vehicle_polygon,ego_state_dict,text_dict, axes)

    # render ego ghost vehicle
    tracks_vis.render_objects_ghost(ghost_patches_dict, ghost_vehicle_polygon, ghost_vehicle_motionstate, text_dict, axes, 'whitesmoke', False)

    # render others vehicle with highlight surrounding
    tracks_vis.render_objects_without_ego_and_conflict_with_highlight(other_patches_dict,other_vehicle_polygon,other_vehicle_motionstate,surrounding_vehicle_id_list,text_dict, axes)

    fig.canvas.draw()
