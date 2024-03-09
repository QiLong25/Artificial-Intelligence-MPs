# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function

    queue = (maze.start,)
    links = {maze.start: None}                  # {tail:head, }

    ### BFS Search ###
    while True:
        ## dequeue and search
        if len(queue) == 0:         # queue is empty
            print("Waypoint is not reachable!")
            return []
        cur = queue[0]          # dequeue current point
        queue = queue[1:]
        neighbors = maze.neighbors_all(cur[0], cur[1])           # neighbors to cur

        ## check waypoints
        if maze.waypoints[0] in neighbors:
            links[maze.waypoints[0]] = cur
            break

        # push valid candidate to queue
        for cand in neighbors:
            if cand in list(links.keys()):
                continue
            else:
                links[cand] = cur
                queue += (cand,)

    ### Build Path ###
    tail = maze.waypoints[0]
    path = [tail]
    while True:
        node_to_add = links[tail]
        if node_to_add is None:             # find start
            break
        else:
            path.append(node_to_add)
            tail = node_to_add
    path = list(reversed(path))

    return path

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single

    from queue import PriorityQueue

    ### Estimate h ###
    h_dict = {}
    for node in list(maze.indices()):
        y_dist = abs(node[0] - maze.waypoints[0][0])
        x_dist = abs(node[1] - maze.waypoints[0][1])
        if x_dist >= y_dist:
            h_dict[node] = x_dist
        else:
            h_dict[node] = y_dist

    ### Initialize g to inf ###
    g_dict = {}
    for node in list(maze.indices()):
        g_dict[node] = 50
    g_dict[maze.start] = 0

    ### Search ###
    queue = PriorityQueue()
    queue.put([g_dict[maze.start] + h_dict[maze.start], maze.start])
    links = {maze.start: None}                  # {tail:head, }

    ### BFS Search ###
    while True:
        ## dequeue and search
        if queue.empty():           # queue is empty
            print("Waypoint is not reachable!")
            return []
        cur = queue.get()[1]           # dequeue current point
        neighbors = maze.neighbors_all(cur[0], cur[1])           # neighbors to cur

        ## check waypoints
        if maze.waypoints[0] in neighbors:
            links[maze.waypoints[0]] = cur
            break

        # push valid candidate to queue
        for cand in neighbors:
            if cand in list(links.keys()):
                if g_dict[cand] > g_dict[cur] + 1:          # current parent is better
                    g_dict[cand] = g_dict[cur] + 1
                    links[cand] = cur
                    # queue.put([g_dict[cand] + h_dict[cand], cand])          # open node
                continue
            else:
                links[cand] = cur
                g_dict[cand] = g_dict[cur] + 1
                queue.put([g_dict[cand] + h_dict[cand], cand])

    ### Build Path ###
    tail = maze.waypoints[0]
    path = [tail]
    while True:
        node_to_add = links[tail]
        if node_to_add is None:  # find start
            break
        else:
            path.append(node_to_add)
            tail = node_to_add
    path = list(reversed(path))

    return path

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    ## Build Minimum spanning tree
    def mst(waypoints):             # input: remaining waypoints
        clusters = []
        total_dist = 0
        dist = {}               # {(head, tail):dist, }
        head_visited = []
        result = {}             # {waypoint:dist, }

        if len(waypoints) <= 1:
            return {waypoints[0]:0}

        ## Build graph
        for head in waypoints:
            for tail in waypoints:
                if head == tail:
                    continue
                elif tail in head_visited:
                    continue
                else:
                    y_dist = abs(head[0] - tail[0])
                    x_dist = abs(head[1] - tail[1])
                    if x_dist >= y_dist:
                        dist[(head, tail)] = x_dist
                    else:
                        dist[(head, tail)] = y_dist
            head_visited.append(head)
        dist = sorted(dist.items(), key=lambda x: x[1])

        ## Build tree
        tree = []                   # [((head, tail), dist), ]
        for edge in dist:           # ((head, tail), dist)
            loop_flag = 0           # check edge forming loop
            cluster0 = None
            cluster1 = None
            for idx in range(len(clusters)):
                if edge[0][0] in clusters[idx] and edge[0][1] in clusters[idx]:
                    loop_flag = 1
                    break
                elif edge[0][0] in clusters[idx]:
                    cluster0 = idx
                elif edge[0][1] in clusters[idx]:
                    cluster1 = idx
            if loop_flag == 1:
                continue
            if cluster0 is None and cluster1 is None:
                cluster_new = [edge[0][0], edge[0][1]]
                clusters.append(cluster_new)
            elif cluster0 is None and cluster1 is not None:
                clusters[cluster1].append(edge[0][0])
            elif cluster1 is None and cluster0 is not None:
                clusters[cluster0].append(edge[0][1])
            else:
                clusters[cluster0] += clusters[cluster1]
                del clusters[cluster1]
            total_dist += edge[1]
            tree.append(edge)

        def search_tree(tree, start_node, parent):
            max_dist = 0
            for edge in tree:
                if start_node in edge[0] and parent not in edge[0]:
                    if start_node == edge[0][0]:
                        temp_dist = edge[1] + search_tree(tree, edge[0][1], edge[0][0])
                        if temp_dist > max_dist:
                            max_dist = temp_dist
                    else:
                        temp_dist = edge[1] + search_tree(tree, edge[0][0], edge[0][1])
                        if temp_dist > max_dist:
                            max_dist = temp_dist
            return max_dist

        ## Compute traversal dist of each waypoint
        for start in waypoints:
            max_dist = 0

            # find longest path
            for edge in tree:
                if start in edge[0]:
                    if start == edge[0][0]:
                        temp_dist = edge[1] + search_tree(tree, edge[0][1], start)
                        if temp_dist > max_dist:
                            max_dist = temp_dist
                    else:
                        temp_dist = edge[1] + search_tree(tree, edge[0][0], start)
                        if temp_dist > max_dist:
                            max_dist = temp_dist

            # store dist to result
            result[start] = (total_dist - max_dist) * 2 + max_dist

        return result

    from queue import PriorityQueue

    waypoint_remain = list(maze.waypoints)
    start_cur = maze.start
    path_whole = [maze.start]

    while len(waypoint_remain) > 0:

        waypoint_cur = None

        ### Estimate h ###
        all_dest_dist = mst(waypoint_remain)
        # print(all_dest_dist)
        h_dict = {}
        for node in list(maze.indices()):
            dist = 100
            for waypoint in waypoint_remain:
                y_dist_try = abs(node[0] - waypoint[0])
                x_dist_try = abs(node[1] - waypoint[1])
                if y_dist_try <= x_dist_try:
                    dist_try = x_dist_try + all_dest_dist[waypoint]
                    if dist_try < dist:
                        dist = dist_try
                elif x_dist_try <= y_dist_try < dist:
                    dist_try = y_dist_try + all_dest_dist[waypoint]
                    if dist_try < dist:
                        dist = dist_try
            h_dict[node] = dist

        ### Initialize g to inf ###
        g_dict = {}
        for node in list(maze.indices()):
            g_dict[node] = 50
        g_dict[start_cur] = 0

        ### Search ###
        queue = PriorityQueue()
        queue.put([g_dict[start_cur] + h_dict[start_cur], start_cur])
        links = {start_cur: None}                  # {tail:head, }

        ### BFS Search ###
        while True:
            ## dequeue and search
            if queue.empty():           # queue is empty
                print("Waypoint is not reachable!")
                return []
            cur = queue.get()[1]           # dequeue current point
            if cur in waypoint_remain:
                waypoint_cur = cur
                break
            neighbors = maze.neighbors_all(cur[0], cur[1])           # neighbors to cur

            ### check waypoints
            # reach_goal = 0
            # for waypoint in waypoint_remain:
            #     if waypoint in neighbors:
            #         links[waypoint] = cur
            #         waypoint_cur = waypoint
            #         reach_goal = 1
            #         break
            # if reach_goal == 1:
            #     break

            # push valid candidate to queue
            for cand in neighbors:
                if cand in list(links.keys()):
                    if g_dict[cand] > g_dict[cur] + 1:          # current parent is better
                        g_dict[cand] = g_dict[cur] + 1
                        links[cand] = cur
                        # queue.put([g_dict[cand] + h_dict[cand], cand])          # open node
                    continue
                else:
                    links[cand] = cur
                    g_dict[cand] = g_dict[cur] + 1
                    queue.put([g_dict[cand] + h_dict[cand], cand])

        ### Build SubPath ###
        tail = waypoint_cur
        path = [tail]
        while True:
            node_to_add = links[tail]
            if node_to_add is None or links[node_to_add] is None:  # find start
                break
            else:
                path.append(node_to_add)
                tail = node_to_add
        path = list(reversed(path))

        ### State update ###
        path_whole += path
        waypoint_remain.remove(waypoint_cur)
        start_cur = waypoint_cur

    return path_whole
