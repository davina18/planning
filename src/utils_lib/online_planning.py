import numpy as np
import copy
import matplotlib.pyplot as plt
import rospy
import random
from scipy.ndimage import binary_dilation

#--------------------------------- State Validity Checker -----------------------------------#

class StateValidityChecker:
    def __init__(self, distance=0.1, is_unknown_valid=True):
        self.map = None
        self.resolution = None  
        self.origin = None
        self.there_is_map = False
        self.distance = distance    # minimum distance from obstacles
        self.is_unknown_valid = is_unknown_valid    
        self.max_recursion = 50  # recursion limit for finding a valid goal

    # Set the map and optionally inflate the obstacles
    def set(self, data, resolution, origin, inflate=False):
        self.map = data
        self.resolution = resolution
        self.origin = np.array(origin)
        self.there_is_map = True
        self.height = data.shape[0]
        self.width = data.shape[1]

        if inflate:
            inflated_map = binary_dilation(self.map == 100, iterations=2)
            self.map[inflated_map] = 100 # set inflated cells as obstacles

    # Check if a pose is valid
    def is_valid(self, pose):
        shape = [len(self.map), len(self.map[0])] # [height, width]
        if isinstance(pose[0], float) or isinstance(pose[1], float):
            pose = self.__position_to_map__(pose) # convert pose from world coords to map coords
            if pose is None:
                return False
            row, col = pose
        else:
            row, col = pose

        # Check if pose is within map boundaries
        if row < 0 or row >= shape[0] or col < 0 or col >= shape[1]:
            return False

        # Define area of minimum obstacle distance
        distance = round(self.distance / self.resolution)   # distance in grid cells
        i_min, i_max = max(0, row - distance), min(shape[0] - 1, row + distance)
        j_min, j_max = max(0, col - distance), min(shape[1] - 1, col + distance)

        # Check validity of cells in defined area
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                cell_value = self.map[i, j]
                if cell_value == 100:   # obstacle
                    return False
                if cell_value == -1 and not self.is_unknown_valid: # unknown
                    return False
        return True
    
    # Convert pose from world coords (x, y) to map coords (m = [row, col])
    def __position_to_map__(self, p):
        shape = [len(self.map), len(self.map[0])]  # [height, width]
        row = int((p[0] - self.origin[0]) / self.resolution)
        col = int((p[1] - self.origin[1]) / self.resolution)
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            return (row, col)
        else:
            return None
    
    # Convert pose from map coords (m = [row, col]) to world coords (x, y)
    def map_to_position(self, m):
        x = self.origin[0] + m[0] * self.resolution + self.resolution / 2.0
        y = self.origin[1] + m[1] * self.resolution + self.resolution / 2.0
        return [x, y]
        
    # Check if a path is valid
    '''def check_path(self, path):
        for i in range(len(path) - 1):
            # Check the path in increments
            spaced_x = np.linspace(path[i][0], path[i+1][0], num=10)
            spaced_y = np.linspace(path[i][1], path[i+1][1], num=10)
            for x, y in zip(spaced_x, spaced_y):
                if not self.is_valid([x, y]):
                    rospy.logwarn(f"[CHECK_PATH] Invalid point in path: ({x:.2f}, {y:.2f})")
                    return False 
        return True'''
    def check_path(self, path, tolerance=0.1):
        """
        Check if the entire path is valid.
        For each interpolated point between waypoints, if the point is invalid,
        try a few points within a small tolerance.
        """
        for i in range(len(path) - 1):
            spaced_x = np.linspace(path[i][0], path[i+1][0], num=10)
            spaced_y = np.linspace(path[i][1], path[i+1][1], num=10)
            for x, y in zip(spaced_x, spaced_y):
                if self.is_valid([x, y]):
                    continue

                # If the exact point isn't valid, check a few neighboring points within tolerance.
                neighbors = [
                    [x + tolerance, y],
                    [x - tolerance, y],
                    [x, y + tolerance],
                    [x, y - tolerance],
                ]
                if not any(self.is_valid(n) for n in neighbors):
                    rospy.logwarn(f"[CHECK_PATH] Invalid point in path: ({x:.2f}, {y:.2f}) between waypoints {i} and {i+1}")
                    return False
        return True

    
    #--------------------------- Used in frontier_exploration.py ------------------------#

    # Check if a pose is a valid frontier
    def is_valid_frontier(self, world_pos):

        # Get map pose of frontier
        if not self.there_is_map:
            raise ValueError("Occupancy map not set.")
        map_pos = self.__position_to_map__(world_pos)
        if map_pos is None:
            return False
        row, col = map_pos
        
        # Check if frontier is valid
        if not self.is_valid(world_pos):
            return False
        
        # Check if its adjacent to unknown space
        for i in range(max(0, row - 1), min(self.height, row + 2)):
            for j in range(max(0, col - 1), min(self.width, col + 2)):
                if self.map[i, j] == -1:  # unknown
                    return True
                    
        return False

    #----------------------------------- Used in planner.py -------------------------------#

    # Find a nearby valid goal if the current goal is invalid
    def find_alternative_goal(self, goal, c = 0):

        # Ensure recursion does not exceed the limit
        if c >= self.max_recursion:
            return None

        # Convert position to map coordinates
        map_coord = self.__position_to_map__(goal)
        if map_coord is None or np.any(map_coord == None):
            return None
        map_x, map_y = map_coord

        # Define the search radius in terms of grid cells
        search_radius = int(self.distance / self.resolution)
        
        # Scan the neighboring cells around the current goal for validity
        for x_offset in range(-search_radius, search_radius + 1):
            for y_offset in range(-search_radius, search_radius + 1):
                candidate_x = map_x + x_offset
                candidate_y = map_y + y_offset

                # Ensure the candidate pose is within boundaries
                if 0 <= candidate_x < self.height and 0 <= candidate_y < self.width:
                    candidate_position = self.map_to_position([candidate_x, candidate_y])

                    # Return the first valid candidate pose found
                    if self.is_valid(candidate_position):
                        return candidate_position  

        # If no valid pose is found, increase the search radius recursively
        return self.find_alternative_goal(goal, c + 1)


#---------------------------------------- RRT* Planner --------------------------------------#

class Planner:
    def __init__(self, state_validity_checker, max_iterations=1000, dominion=[-10.0, 10.0, -10.0, 10.0], delta_q=0.3, goal_bias=0.1):
        self.svc = state_validity_checker
        self.max_iterations = max_iterations
        self.dominion = dominion # planning bounds
        self.delta_q = delta_q
        self.goal_bias = goal_bias

    # Compute path from start to goal
    def compute_path(self, q_start, q_goal):
        q_start = tuple(q_start)
        q_goal = tuple(q_goal)
        G = {"vertices": [q_start], "edges": []}
        cost = {q_start: 0} # cost to reach each vertex

        r = 0.7  # radius for neighbors
        min_dist = 1.0 # minimum distance to connect to goal

        for _ in range(self.max_iterations):
            q_rand = self.rand_config(q_goal)
            q_near = self.nearest_vertex(q_rand, G)
            q_new = self.new_config(q_near, q_rand)

            # Check if edge between q_near and q_new is valid
            if self.is_segment_free_inc(q_near, q_new) and self.svc.is_valid(q_new):
                neighbors = [v for v in G["vertices"] if self.dist(v, q_new) <= r]
                q_min = q_near
                cost[q_new] = float('inf')

                # Check if a neighbour offers a shorter path to q_new
                for n in neighbors:
                    if self.is_segment_free_inc(n, q_new) and cost[n] + self.dist(n, q_new) < cost[q_new]:
                        q_min = n
                        cost[q_new] = cost[n] + self.dist(n, q_new)

                self.add_vertex(G, q_new)
                self.add_edge(G, q_min, q_new)

                # Check if any neighbours can shorten their path by being connected via q_new
                for n in neighbors:
                    if self.is_segment_free_inc(q_new, n) and cost[q_new] + self.dist(q_new, n) < cost[n]:
                        cost[n] = cost[q_new] + self.dist(q_new, n)
                        G["edges"] = [edge for edge in G["edges"] if edge[1] != n]
                        self.add_edge(G, q_new, n)
                
                # Check if we are close enough to connect to the goal
                if self.dist(q_new, q_goal) < min_dist and self.is_segment_free_inc(q_new, q_goal):

                    self.add_vertex(G, q_goal)
                    self.add_edge(G, q_new, q_goal)
                    total_path = self.reconstruct_path(G, q_start, q_goal)
                    #rospy.loginfo(f"[PLANNER] Reconstructed path with {len(total_path)} points.")
                    
                    if self.svc.check_path(total_path):
                        rospy.loginfo("[PLANNER] Path is valid after reconstruction.")
                        return total_path     
                           
        rospy.loginfo("[PLANNER] No path found.")
        return None

    # Sample random point within dominion bounds or the goal with goal bias
    def rand_config(self, q_goal):
        x_min, x_max, y_min, y_max = self.dominion
        if random.random() > self.goal_bias:
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            return (x, y)
        return q_goal
    
    # Find the nearest node in the graph to a given point
    def nearest_vertex(self, q_rand, G):
        return min(G["vertices"], key=lambda v: self.dist(q_rand, v))

    # Create a new point by stepping from q_rand to q_near by delta_q 
    def new_config(self, q_near, q_rand):
        delta_x, delta_y = q_rand[0] - q_near[0], q_rand[1] - q_near[1]
        dist = self.dist(q_near, q_rand)
        if dist == 0:
            return q_near
        direction = [delta_x / dist, delta_y / dist]
        new_x = q_near[0] + direction[0] * self.delta_q
        new_y = q_near[1] + direction[1] * self.delta_q
        return (new_x, new_y)
    
    # Check if the edge between q_near and q_new is valid
    def is_segment_free_inc(self, q_near, q_new):
        path = [q_near, q_new]
        step_size = 0.02
        for index_i in range(len(path) - 1):
            segment_start = path[index_i][0:2]      # start point of the segment
            segment_end = path[index_i + 1][0:2]    # end point of the segment
            
            # Calculate the distance between the segment start and end
            dist = np.linalg.norm(np.array(segment_end) - np.array(segment_start))
            
            # Calculate the number of steps required to interpolate between the points
            num_steps = max(int(dist / step_size), 1)
            
            # Interpolate along the segment
            for index_j in range(num_steps + 1):
                s = index_j / num_steps  # Interpolation parameter from 0 (start) to 1(end)
                new_point = (1 - s) * np.array(segment_start) + s * np.array(segment_end)
                
                # Check if the interpolated point is valid
                if not self.svc.is_valid(new_point):
                    return False # fails if any point along the segment is invalid
              
        return True
    
    # Add a node to the graph
    def add_vertex(self, G, vertex):
        G["vertices"].append(vertex)

    # Add an edge to the graph
    def add_edge(self, G, vertex_1, vertex_2):
        G["edges"].append((vertex_1, vertex_2))

    # Reconstruct the final path by traversing from goal to start
    def reconstruct_path(self, G, q_start, q_goal):
        rospy.loginfo(f"[PLANNER] Reconstructing path from {q_start} to {q_goal}")
        path = [q_goal]
        current = q_goal
        while current != q_start:
            for v1, v2 in G["edges"]:
                if v2 == current:
                    path.append(v1)
                    current = v1
                    break
        return path[::-1] # reverse so that the path is from start to goal
   
    '''def smooth_path(self, path, start, goal):
        rospy.loginfo(f"[SMOOTH] Input path: {path}")
        rospy.loginfo(f"[SMOOTH] Start: {start}, Goal: {goal}")

        if not path or len(path) < 2:
            rospy.logwarn("[SMOOTH] Path too short to smooth.")
            return path

        smoothed = [start]
        i = 0
        while i < len(path):
            j = len(path) - 1
            while j > i:
                if self.svc.check_path([path[i], path[j]]):
                    rospy.loginfo(f"[SMOOTH] Shortcut from {path[i]} to {path[j]}")
                    smoothed.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                # if no shortcut found, move to next
                i += 1

        smoothed.append(goal)
        rospy.loginfo(f"[SMOOTH] Final smoothed path: {smoothed}")
        return smoothed'''
    
    def smooth_path(self, path):
        #rospy.loginfo(f"[SMOOTH] Input path: {path}")
        n = len(path)
        smoothed_path = [path[0]]
        i = 0
        while i < n - 1:
            j = n - 1
            while j > i and not self.is_segment_free_inc(path[i], path[j]):
                #rospy.loginfo(f"[SMOOTH] Shortcut from {path[i]} to {path[j]}")
                j -= 1
            smoothed_path.append(path[j])
            i = j
        #rospy.loginfo(f"[SMOOTH] Final smoothed path: {smoothed_path}")
        return smoothed_path


    def is_close(self, point_a, point_b, tolerance=0.01):
         # If either point is None. Return False
        if point_a is None or point_b is None:
            return False
        #Return True if the points are within the tolerance
        return np.linalg.norm(np.array(point_a) - np.array(point_b)) <= tolerance

    # Compute Euclidean distance between two points
    def dist(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Initialises the planner with the map and computes a path from start to goal
def compute_path(start_p, goal_p, state_validity_checker, max_iterations=1000):
    origin = state_validity_checker.origin
    resolution = state_validity_checker.resolution
    height, width = state_validity_checker.map.shape
    planner = Planner(state_validity_checker, max_iterations,
                      dominion=[origin[0], origin[0] + (width-1) * resolution,
                                origin[1], origin[1] + (height-1) * resolution],
                      delta_q=1, goal_bias=0.1)
    path = planner.compute_path(start_p, goal_p)
    if path:
        #return [[p[0], p[1]] for p in path]
        smoothed_path = planner.smooth_path(path)
        rospy.loginfo(f"Path found and smoothed: {smoothed_path}")
        result = [[p[0], p[1]] for p in smoothed_path]
        return result
    else:
        rospy.logwarn("No path found!")
        return None

# Compute (v, w) velocities to drive the robot towards the goal
def move_to_point(current, goal, Kv=0.5, Kw=0.5):
    delta_x = goal[0] - current[0]
    delta_y = goal[1] - current[1]
    distance = np.sqrt(delta_x ** 2 + delta_y ** 2)
    desired_angle = np.arctan2(delta_y, delta_x)
    current_angle = wrap_angle(current[2])
    angle_err = wrap_angle(desired_angle - current_angle)
    w = -Kw * angle_err
    v = Kv * distance if abs(angle_err) < np.pi / 6 else 0 # move forward if roughly facing the goal
    return v, w

#----------------------------------- Helper Functions ---------------------------------------#

# Bound an angle to the range [0, 2*pi]
def wrap_angle(angle):
    return (angle + (2.0 * np.pi * np.floor((np.pi - angle) / (2.0 * np.pi))))



