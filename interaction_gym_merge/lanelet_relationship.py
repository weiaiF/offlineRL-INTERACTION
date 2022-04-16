import geometry
import numpy as np

def get_route(interaction_map,current_lanelet,end_lanelet):
    return  interaction_map.routing_graph.getRoute(current_lanelet, end_lanelet, 0)

def get_planning_route(interaction_map, ego_start_lanelet_dict, ego_end_lanelet_dict):
    ego_route = dict()
    ego_route_lanelet = dict()
    is_route_contain_turning = dict()

    for k in ego_start_lanelet_dict.keys():

        # current_lanelet = self._current_lanelet[k]
        start_lanelet = ego_start_lanelet_dict[k]   
        # print('lanelet:',type(current_lanelet))
        print('start lanelet id:',start_lanelet.id)


        # get following lanelet of current lanelet
        # forward lanelet is calculated by routing
        end_lanelet = ego_end_lanelet_dict[k]
        print('end lanelet id:',end_lanelet.id)

        # need to remove the situation: current lanelet = end lanelet(means we filter out the stand still data)
        # in this situation still can find route but no more following lanelet in next process
        if start_lanelet.id == end_lanelet.id:
            print('start lanelet equals to end lanelet!!!')
            return False,None,None
        route = get_route(interaction_map, start_lanelet, end_lanelet)
        # print('route type:',type(route))
        if route:
            ego_route[k] = route

            all_following_lanelet = route.fullLane(start_lanelet)
            # print('---all following relations len---:',len(all_following_lanelet))
            lanelet_list = []
            for ll in all_following_lanelet:
                lanelet_list.append(ll)
                # print('route lanelet id:',ll.id)
                # print('route length:',geometry.lanelet_length(ll))
            if lanelet_list[0].id != start_lanelet.id:
                # print('ego: ' + str(k) + ' find a wrong route!!!')
                print('error route do not match start lanelet')
                return False,None,None
            if lanelet_list[-1].id != end_lanelet.id:
                # print('ego: ' + str(k) + ' find a  incomplete route!!!')
                print('error route do not match end lanelet')
                lanelet_list.append(end_lanelet)
            
            ego_route_lanelet[k] = lanelet_list

            # need to verify the planning route is a complete route
            # means that the end lanelet is the following lanelet of the last but one
            for i in range(len(lanelet_list)-1):
                if not geometry.is_following_lanelet(lanelet_list[i],lanelet_list[i+1]):
                    print('error route do not continue')
                    return False,None,None
        else:
            print('ego: ' + str(k) + ' can not find route!!!')
            return False,None,None
    
    return True, ego_route, ego_route_lanelet  

def get_specified_ego_vehicle_replanning_route(interaction_map,ego_current_lanelet,ego_end_lanelet):
    print('replanning')
    route = get_route(interaction_map,ego_current_lanelet,ego_end_lanelet)
    if route:
        all_following_lanelet = route.fullLane(ego_current_lanelet)
        lanelet_list = []
        for ll in all_following_lanelet:
            lanelet_list.append(ll)
        if lanelet_list[0].id != ego_current_lanelet.id:
            return False,None,None
        if lanelet_list[-1].id != ego_end_lanelet.id:
            lanelet_list.append(ego_end_lanelet)

        if not geometry.is_following_lanelet(lanelet_list[-2],lanelet_list[-1]):
            return False,None,None
            
        return True,route,lanelet_list
    else:
        return False,None,None


def get_surrounding_lanelets(interaction_map,current_lanelet,previous_lanelet,following_lanelet):

    # === lanelet relationship finding ===
    # get privious, following, left upper_left lower_left and right upper_right lower_right lanelet of current lanelet (routeable)

    upper_left = upper_right = lower_left = lower_right = None
    left = lanelet_map.routing_graph.lefts(current_lanelet,0)
    if len(left):
        left = left[0]
        # print('left adjcent:',left.id)
    right = lanelet_map.routing_graph.rights(current_lanelet,0)
    if len(right):
        right = right[0]
        # print('right adjcent:',right.id)
    if following_lanelet:
        upper_left = lanelet_map.routing_graph.lefts(following_lanelet,0)
        if len(upper_left):
            upper_left = upper_left[0]
            # print('upper_left adjcent:',upper_left.id)
        upper_right = lanelet_map.routing_graph.rights(following_lanelet,0)
        if len(upper_right):
            upper_right = upper_right[0]
            # print('upper_right adjcent:',upper_right.id)
    if previous_lanelet:
        lower_left = lanelet_map.routing_graph.lefts(previous_lanelet,0)
        if len(lower_left):
            lower_left = lower_left[0]
            # print('lower_left adjcent:',lower_left.id)
        lower_right = lanelet_map.routing_graph.rights(previous_lanelet,0)
        if len(lower_right):
            lower_right = lower_right[0]
            # print('lower_right adjcent:',lower_right.id)
    # previous and following are different from the left and right finding way
    # as them need to along the planning route

    return left,right,upper_left,upper_right,lower_left,lower_right

def get_surrounding_route_along_planning_route(interaction_map,route_lanelet_dict):
    left_route_lanelet_dict = dict()
    right_route_lanelet_dict = dict()
    for k,v in route_lanelet_dict.items():
        left_route_lanelet_dict[k] = []
        right_route_lanelet_dict[k] = []
        for rl in v:
            left = interaction_map.routing_graph.lefts(rl,0)
            if len(left):
                left = left[0]
                left_route_lanelet_dict[k].append(left)
            else:
                # if current lanelet do not have left lanelet
                left_route_lanelet_dict[k].append('')
            right = interaction_map.routing_graph.rights(rl,0)
            if len(right):
                right = right[0]
                right_route_lanelet_dict[k].append(right)
            else:
                # if current lanelet do not have right lanelet
                right_route_lanelet_dict[k].append('')

    
    return left_route_lanelet_dict,right_route_lanelet_dict

def get_specified_ego_vehicle_surrounding_route_along_planning_route(interaction_map,route_lanelet):
    left_route_lanelet = []
    right_route_lanelet = []

    for rl in route_lanelet:
        left = interaction_map.routing_graph.lefts(rl,0)
        if len(left):
            left = left[0]
            left_route_lanelet.append(left)
        else:
            # if current lanelet do not have left lanelet
            left_route_lanelet.append('')
        right = interaction_map.routing_graph.rights(rl,0)
        if len(right):
            right = right[0]
            right_route_lanelet.append(right)
        else:
            # if current lanelet do not have right lanelet
            right_route_lanelet.append('')

    
    return left_route_lanelet,right_route_lanelet

def get_surrounding_lanelets_along_route(interaction_map,current_lanelet,route_lanelet):
    # this function is used to repalce the get_surrounding_lanelets() function
    # the biggest different is surrounding lanlet now become a list along the route not a single lanelet
    # so there is no need to distinguish between upper and lower
    # the order is same as the route index (there may exist the left lanelet number is different from the right lanelet number or route lanelet number)
    left_list = []
    right_list = []
    upper_left_list = []
    upper_right_list = []
    lower_left_list = []
    lower_right_list = []
    # print('current_lanelet:',current_lanelet.id)
    # print('route_lanelet:',[r_l.id for r_l in route_lanelet])
    findresult = np.where([route_lanelet[i].id==current_lanelet.id for i in range(len(route_lanelet))])[0]
    if len(findresult) == 0:
        return None,None,None,None,None,None,None,None

    current_lanelet_index = findresult[0]
    # print('current_lanelet_index:',current_lanelet_index)
    
    
    following_list = route_lanelet[current_lanelet_index:]
    # print('following_list:',[fl.id for fl in following_list])
    previous_list = route_lanelet[:current_lanelet_index]
    # print('previous_list:',[pl.id for pl in previous_list])
    
      # find viruture lanelet for start and end situtaion

    # vitural_previous_lanelet = self._map.routing_graph.previous(current_lanelet)
    # if len(vitural_previous_lanelet):
    #     vitural_previous_lanelet = vitural_previous_lanelet[0]
    #     print('virtual previous lanelet id:',vitural_previous_lanelet.id)
    #     previous_list.insert(0,vitural_previous_lanelet)
    # else:
    #     previous_list.insert(0,'')

    
    # vitural_following_lanelet = self._map.routing_graph.following(current_lanelet)
    # if len(vitural_following_lanelet):
    #     vitural_following_lanelet = vitural_following_lanelet[0]
    #     print('virtual following lanelet id:',vitural_following_lanelet.id)
    #     following_list.append(vitural_following_lanelet)
    # else:
    #     following_list.append('')

    # current_lanelet_index += 1 
    
    for rl in route_lanelet:
        left = interaction_map.routing_graph.lefts(rl,0)
        if len(left):
            left = left[0]
            left_list.append(left)
        else:
            # if current lanelet do not have left lanelet
            left_list.append('')
        right = interaction_map.routing_graph.rights(rl,0)
        if len(right):
            right = right[0]
            right_list.append(right)
        else:
            # if current lanelet do not have right lanelet
            right_list.append('')

    upper_left_list = left_list[current_lanelet_index:]
    # print('upper_left_list:',upper_left_list)
    left = left_list[current_lanelet_index]
    # print('left:',left)
    lower_left_list = left_list[:current_lanelet_index]
    # print('lower_left_list:',lower_left_list)
    upper_right_list = right_list[current_lanelet_index:]
    # print('upper_right_list:',upper_right_list)
    right = right_list[current_lanelet_index]
    # print('right:',right)
    lower_right_list = right_list[:current_lanelet_index]

    # print('over')

    return upper_left_list,left,lower_left_list,upper_right_list,right,lower_right_list,previous_list,following_list

def get_conflict_lanelet(route,current_lanelet,following_lanelet):
    # print('current_lanelet:',type(current_lanelet))
    current_conflict_lanelet_list = route.conflictingInMap(current_lanelet)
    # print('current_conflict in map:',len(current_conflict_lanelet_list))
    
    # for i in current_conflict_lanelet_list:
        # print('current_conflict:',i.id)
    following_conflict_lanelet_list = []
    if following_lanelet:
        following_conflict_lanelet_list = route.conflictingInMap(following_lanelet)
        # print('following_conflict in map:',len(following_conflict_lanelet_list))
        following_regulate_element = following_lanelet.regulatoryElements
        # print('following regulat:',following_regulate_element)
        # for i in following_conflict_lanelet_list:
            # print('following_conflict:',i.id)

    return current_conflict_lanelet_list, following_conflict_lanelet_list

def get_conflict_lanelet_dict_along_route(interaction_map,route,following_route,previous_route):
    # only consider the current and following route conflict
    # the following route contain current lanelet
        
    following_route_conflict_lanelet_dict = dict()

    for index,following_ll in enumerate(following_route):
        following_ll_confilct_lanelet_list =  route.conflictingInMap(following_ll)

        filtered_following_ll_confilct_lanelet_list = []
        following_ll_confilct_lanelet_list_without_fork = []
        following_route_conflict_lanelet_previous_list = []
        # remove do not have conflict point case
        # remove following or previous relationship conflict
        for f_c_ll in following_ll_confilct_lanelet_list:    
            conflict_point_list = geometry.get_overlap_point_list(following_ll,f_c_ll)
            if len(conflict_point_list) != 0:
                if not geometry.is_following_lanelet(following_ll,f_c_ll) and not geometry.is_following_lanelet(f_c_ll,following_ll):
                    filtered_following_ll_confilct_lanelet_list.append(f_c_ll)


        # make sure that the conflict lanelet is not the following of lanelet of previous (brother relationship with current lanelet)
        # in other words, make sure the conflict lanelet direction is conflict with the current lanelet direction,
        # not the fork situation
        for c_ll in filtered_following_ll_confilct_lanelet_list:
            previous_lanelet = None
            # find the previous lanelet
            if index >0:
                previous_lanelet = following_route[index-1]
            else:
                if len(previous_route)>0:
                    # exist previous
                    previous_lanelet = previous_route[-1]                   
            
            if previous_lanelet is not None:
                # not the fork sitiuation
                if not geometry.is_following_lanelet(previous_lanelet,c_ll):
                    following_ll_confilct_lanelet_list_without_fork.append(c_ll)
            else:
                # can not find previous lanelet
                following_ll_confilct_lanelet_list_without_fork.append(c_ll)
        
        if len(following_ll_confilct_lanelet_list_without_fork) > 0:
            following_route_conflict_lanelet_dict[following_ll.id] = (following_ll,following_ll_confilct_lanelet_list_without_fork)

        
        # following_route_conflict_lanelet_dict[following_ll.id] = (following_ll,following_ll_confilct_lanelet_list)

        # to see far than conflict part we need the previous lanelet of the conflict part
        for f_c_ll in following_ll_confilct_lanelet_list_without_fork:
            f_c_previous_ll = interaction_map.routing_graph.previous(f_c_ll)
            if len(f_c_previous_ll)>0:
                previous_conflict_pair = {following_ll.id:(following_ll,f_c_previous_ll)}
                following_route_conflict_lanelet_previous_list.append(previous_conflict_pair)
                

    # for k,v in following_route_conflict_lanelet_dict.items():
    #     print(k,' conflict with : ', [ll.id for ll in v[1]])

    return following_route_conflict_lanelet_dict,following_route_conflict_lanelet_previous_list

def get_conflict_lanelet_id_dict_along_route(interaction_map,route,following_route,previous_route):
    # only consider the current and following route conflict
    # the following route contain current lanelet
        
    following_route_conflict_lanelet_id_dict = dict()
    following_ll_confilct_lanelet_list_without_fork = []
    following_route_conflict_lanelet_previous_list = []
    for index,following_ll in enumerate(following_route):
        following_ll_confilct_lanelet_list =  route.conflictingInMap(following_ll)
        # make sure that the conflict lanelet is not the following of lanelet of previous (brother relationship with current lanelet)
        # in other words, make sure the conflict lanelet direction is conflict with the current lanelet direction,
        # not the fork situation
        for c_ll in following_ll_confilct_lanelet_list:
            previous_lanelet = None
            # find the previous lanelet
            if index >0:
                previous_lanelet = following_route[index-1]
            else:
                if len(previous_route)>0:
                    # exist previous
                    previous_lanelet = previous_route[-1]                   
            
            if previous_lanelet is not None:
                # not the fork sitiuation
                if not geometry.is_following_lanelet(previous_lanelet,c_ll):
                    following_ll_confilct_lanelet_list_without_fork.append(c_ll)
            else:
                # can not find previous lanelet
                following_ll_confilct_lanelet_list_without_fork.append(c_ll)
        
        following_route_conflict_lanelet_id_dict[following_ll.id] = [f_c_l.id for f_c_l in following_ll_confilct_lanelet_list_without_fork]

        # to see far than conflict part we need the previous lanelet of the conflict part
        for f_c_ll in following_ll_confilct_lanelet_list_without_fork:
            f_c_previous_ll = interaction_map.routing_graph.previous(f_c_ll)
            if len(f_c_previous_ll)>0:
                previous_conflict_pair = {following_ll.id:(following_ll,f_c_previous_ll)}
                following_route_conflict_lanelet_previous_list.append(previous_conflict_pair)

    return following_route_conflict_lanelet_id_dict,following_route_conflict_lanelet_previous_list

def get_conflict_lanelet_in_previous_route(route,previous_route):
    # this function is used to remove the confict situation in previous vehicle judgement
    previous_route_conflict_lanelet_list = []
    for previous_ll in previous_route:
        previous_ll_confilct_lanelet_list =  route.conflictingInMap(previous_ll)

        previous_route_conflict_lanelet_list.extend(previous_ll_confilct_lanelet_list)

    return previous_route_conflict_lanelet_list