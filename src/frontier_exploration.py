#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point, PoseArray, Pose
from std_msgs.msg import Bool, ColorRGBA, Header
from skimage import measure
import tf
from utils_lib.online_planning import StateValidityChecker
import math

class FrontierDetector:

#--------------------------------------------- Initialisation ------------------------------------------------#

    def __init__(self):
        # Initialise node
        rospy.init_node('frontier_detector')

        # Initialise variables
        self.resolution = None
        self.origin = None
        self.current_pose = None
        self.map = None
        self.svc = StateValidityChecker(0.3)  # svc with 0.3m obstacle clearance
        self.at_goal = True                   # whether the robot has reached the current goal
        self.current_clusters = None          # currently detected frontier cluseters
        self.visited_goals = set()
        self.visited_radius = 0.5             # meters
        #self.bounds = {'xmin': -2.0, 'xmax': 2.0, 'ymin': -2.0, 'ymax': 2.0} # virtual boundaries

        # Subscribers
        self.map_sub = rospy.Subscriber('/projected_map', OccupancyGrid, self.get_map, queue_size=10)
        self.odom_sub = rospy.Subscriber('/turtlebot/kobuki/odom', Odometry, self.get_odom)
        self.goal_sub = rospy.Subscriber('/at_goal', Bool, self.at_goal_callback)

        # Publishers
        self.new_goal_pub = rospy.Publisher('/new_goal', PoseStamped, queue_size=10)
        self.cluster_pub = rospy.Publisher('/frontier_clusters', MarkerArray, queue_size=10)
        self.frontier_pub = rospy.Publisher('/frontier_points', PoseArray, queue_size=10)
        self.viewpoint_pub = rospy.Publisher('/best_viewpoint_marker', Marker, queue_size=10)
        self.boundary_pub = rospy.Publisher('/exploration_boundary', Marker, queue_size=1, latch=True)

        rospy.Timer(rospy.Duration(1.0), self.delayed_start, oneshot=True)   # start exploration once map is ready

#------------------------------------------- Helper Functions ------------------------------------------------#

    # Bound an angle to the range [-pi, pi]
    def wrap_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    # Convert point from map coords (m = [row, col]) to world coords (x, y)
    def map_to_world(self, map_point):
        return [
            map_point[0] * self.resolution + self.origin[0],
            map_point[1] * self.resolution + self.origin[1]
        ]
    
    # Checks whether a point (x, y) is within a bounding box
    def is_within_bounds(self, world_point, bounds=5.0):
        x, y = world_point
        return -bounds <= x <= bounds and -bounds <= y <= bounds


#------------------------------------------- Callback Functions ----------------------------------------------#

    # Callback to get the map
    def get_map(self, msg):
        self.resolution = msg.info.resolution
        self.origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data).reshape((height, width))
        self.map = data.T # transpose to match the map coord system
        self.svc.set(self.map, self.resolution, self.origin)

    # Callback to get the odometry
    def get_odom(self, odom):
        # Convert the quarternion to a yaw angle
        _, _, yaw = tf.transformations.euler_from_quaternion([
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w
        ])
        # Set the current (x, y, theta) pose
        self.current_pose = np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            self.wrap_angle(yaw)
        ])

    # Callback when a goal status message is received
    def at_goal_callback(self, msg):
        if self.at_goal != msg.data: # if the goal status has chanegd
            self.at_goal = msg.data
            if self.at_goal: #  if the goal is reached
                self.clear_frontiers_and_clusters()
                rospy.sleep(0.5)
                # Begin exploration again
                self.exploration()

#--------------------------------------------- Exploration ---------------------------------------------------#

    # Begin exploration once map is ready
    def delayed_start(self, event):
        if self.map is not None:
            #self.publish_exploration_boundary()
            self.exploration()
        else:
            rospy.Timer(rospy.Duration(1.0), self.delayed_start, oneshot=True)

    # Explore the environment
    def exploration(self):
        """
        Check for frontiers:
        A frontier is a cell with value 0 surrounded by -1 (unknown) cells
        We use a 3x3 neighborhood to check for frontiers
        The map is padded with -1 to handle edges

        Find clusters of frontiers:
        We use skimage's measure.label to find connected components
        The clusters are represented as a list of regionprops objects
        Each regionprops object contains information about the cluster
        We only keep clusters with area greater than 5
        """

        rospy.loginfo("[EXPLORE] Starting exploration.")
        frontier_mask = np.zeros_like(self.map, dtype=bool)

        # Detect frontier points
        for i in range(1, self.map.shape[0] - 1):
            for j in range(1, self.map.shape[1] - 1):
                if self.map[i, j] == 0: # free cell
                    neighborhood = self.map[i-1:i+2, j-1:j+2] # 3x3 neighbourhood
                    if -1 in neighborhood:  # neighbouring unknown cells
                        frontier_mask[i, j] = True # mark as a frontier
    
        self.publish_frontiers(frontier_mask) # publish frontier points

        # Group frontier points into clusters
        clusters = self.cluster_frontiers(frontier_mask)
        rospy.loginfo(f"[EXPLORE] Found {len(clusters)} clusters")
        if not clusters:
            rospy.loginfo("[EXPLORE] No frontiers found.")
            return
        self.current_clusters = clusters
        self.publish_cluster_markers(clusters)
   
        # Select the best cluster
        best_cluster = self.select_best_cluster(clusters)
        if best_cluster is None:
            rospy.logwarn("[EXPLORE] No best cluster found")
            return
        
        # Select the best viewpoint within the best cluster
        best_point = self.select_best_viewpoint(best_cluster)
        if best_point is None:
            rospy.logwarn("[EXPLORE] No best viewpoint found")
            return
        
        # Retry exploration after 3 seconds of waiting for goal
        success = self.publish_valid_goal(best_point) # publish best viewpoint as the goal
        if not success:
            rospy.logwarn("[EXPLORE] Goal was skipped or invalid. Retrying in 3 seconds.")
            rospy.Timer(rospy.Duration(3.0), lambda event: self.exploration(), oneshot=True)

    # Publish visualisation markers for virtual boundaries
    def publish_exploration_boundary(self):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "exploration_bounds"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05  # line width
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Define corners of the bounding box
        points = [
            Point(self.bounds['xmin'], self.bounds['ymin'], 0),
            Point(self.bounds['xmax'], self.bounds['ymin'], 0),
            Point(self.bounds['xmax'], self.bounds['ymax'], 0),
            Point(self.bounds['xmin'], self.bounds['ymax'], 0),
            Point(self.bounds['xmin'], self.bounds['ymin'], 0) 
        ]
        marker.points = points

        self.boundary_pub.publish(marker)

#-------------------------------------- Frontiers and Clustering ---------------------------------------------#

    # Publish detected frontier points as a PoseArray
    def publish_frontiers(self, frontier_mask):
        frontier_points = np.argwhere(frontier_mask) # get coords of frontier points
        rospy.loginfo(f"[FRONTIERS] Detected {len(frontier_points)} frontier points")

        pose_array = PoseArray()
        pose_array.header.frame_id = "odom"
        pose_array.header.stamp = rospy.Time.now()

        for point in frontier_points:
            pose = Pose()
            world_point = self.map_to_world(point)
            pose.position.x = world_point[0]
            pose.position.y = world_point[1]
            pose_array.poses.append(pose)

        self.frontier_pub.publish(pose_array)

    # Group frontier points into clusters using skimage.measure
    def cluster_frontiers(self, frontier_mask):
        # Finds connected components (i.e. groups adjacent frontier cells)
        connected_components = measure.label(frontier_mask, connectivity=2)
        # Calculates properites (e.g. size, centroid, shape) for each connected component
        properties = measure.regionprops(connected_components)
        # Only keep clusters with an area (number of frontier cells) larger than 5
        return [region for region in properties if region.area > 5]

    # Publish visualisation markers for frontier clusters
    def publish_cluster_markers(self, clusters):
        marker_array = MarkerArray()
        
        for idx, cluster in enumerate(clusters):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "frontiers"
            marker.id = idx
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.color = ColorRGBA(*np.random.rand(3).tolist(), 1.0)

            for coord in cluster.coords:
                world_point = self.map_to_world(coord)
                marker.points.append(Point(world_point[0], world_point[1], 0))

            marker_array.markers.append(marker)

            # Text label for cluseter ID
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "cluster_labels"
            text_marker.id = idx
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.scale.z = 0.3
            text_marker.color = ColorRGBA(1.0, 1.0, 0.0, 1.0)
            centroid = self.map_to_world(cluster.centroid)
            text_marker.pose.position.x = centroid[0]
            text_marker.pose.position.y = centroid[1]
            text_marker.text = f"Cluster {idx}"
            marker_array.markers.append(text_marker)

        self.cluster_pub.publish(marker_array)

    # Clear visualisations of frontiers and clusters
    def clear_frontiers_and_clusters(self):
        empty_frontiers = PoseArray()
        empty_frontiers.header.stamp = rospy.Time.now()
        empty_frontiers.header.frame_id = "odom"
        self.frontier_pub.publish(empty_frontiers)
        
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array = MarkerArray()
        marker_array.markers.append(clear_marker)
        self.cluster_pub.publish(marker_array)


#------------------------------------------- Goal Selection --------------------------------------------------#
    
    # Select the best cluster based on distance, size, and information gain
    def select_best_cluster(self, clusters):
        best_cluster = None
        best_score = -np.inf
        
        robot_xy   = self.current_pose[:2]
        robot_yaw  = self.current_pose[2]
        for cluster in clusters:
            centroid = self.map_to_world(cluster.centroid)
            # Skip clusters whose centroids are too close to any visited goal
            centroid_np = np.array(centroid)
            too_close = False
            for visited in self.visited_goals:
                if np.linalg.norm(centroid_np - np.array(visited)) < self.visited_radius:
                    too_close = True
                    break
            if too_close:
                continue
            
            distance = np.linalg.norm(self.current_pose[:2] - centroid) # distance from robot to centroid
            distance_score = 1.0 / (1.0 + distance) # clusters distance score is inversely proportional to its distance
            size_score = cluster.area / 100.0 # clusters size score is proportional to its cluster size
            info_gain = self.compute_information_gain( # information gain around the centroid, the higher the better
                int(cluster.centroid[0]),
                int(cluster.centroid[1])
            )

             # --- NEW: heading cost -----------------------------------------------
            # angle between robot’s heading and vector to centroid
            vec_to_centroid = centroid - robot_xy
            target_yaw      = math.atan2(vec_to_centroid[1], vec_to_centroid[0])
            yaw_error       = abs(self.wrap_angle(target_yaw - robot_yaw))  # 0..π

            heading_penalty = yaw_error / math.pi            # 0 (straight ahead) … 1
            if yaw_error > math.pi / 2:                      # >90° ⇒ “behind”
                heading_penalty *= 1.5                      # extra penalty

            heading_score = 1.0 - heading_penalty           # higher is better

            # Combine scores as a weighted score
            # Tune the  α,β,γ,δ weights to taste (they must sum ≤ 1 for readability)
            score = (0.40 * distance_score +
                 0.25 * size_score +
                 0.25 * info_gain +
                 0.10 * heading_score)
            if score > best_score:
                best_score = score
                best_cluster = cluster

        return best_cluster # return the cluster with the best score

    # Select the best viewpoint within a cluster
    def select_best_viewpoint(self, cluster):
        best_point = None
        best_gain = -np.inf

        candidate_viewpoints = [cluster.centroid] # add the centroid as a candidate viewpoint
        for _ in range(20):  # add 10 random points inside the cluster as candidate viewpoints
            if len(cluster.coords) > 0:
                random_idx = np.random.randint(0, len(cluster.coords))
                candidate_viewpoints.append(cluster.coords[random_idx])

        for point in candidate_viewpoints:
            # Check if the candidate viewpoint is valid
            world_point = self.map_to_world(point)
            if not self.svc.is_valid_frontier(world_point):
                rospy.logdebug(f"[VIEWPOINT] Rejected invalid point: {world_point}")
                continue
            # Compute its information gain
            gain = self.compute_information_gain(int(point[0]), int(point[1]))
            if gain > best_gain:
                best_gain = gain
                best_point = point

        return best_point # return the viewpoint with the highest information gain

    # Publish a valid goal
    def publish_valid_goal(self, map_point, max_attempts=3):
        world_point = self.map_to_world(map_point)
        # Check if goal is within radius of any previously visited point
        for visited in self.visited_goals:
            visited_np = np.array(visited)
            if np.linalg.norm(np.array(world_point) - visited_np) < self.visited_radius:
                rospy.loginfo(f"[PUBLISH GOAL] Skipping goal near previously visited point: {world_point}")
                return False
        
        # Attempt n times
        for attempt in range(max_attempts):
            if self.svc.is_valid_frontier(world_point): # check if frontier is valid
                goal = PoseStamped()
                goal.header.frame_id = "odom"
                goal.header.stamp = rospy.Time.now()
                goal.pose.position.x = world_point[0]
                goal.pose.position.y = world_point[1]
                goal.pose.orientation.w = 1.0
                self.new_goal_pub.publish(goal)
                rospy.loginfo(f"[PUBLISH GOAL] Published valid goal at: {world_point}")
                self.visited_goals.add(tuple(world_point)) # mark as visited
                return True
            else:
                rospy.logwarn(f"[PUBLISH GOAL] Invalid goal: {world_point}")
                if self.current_clusters: # try the best viewpoint from another cluster
                    map_point = self.select_best_viewpoint(self.current_clusters[0])
                    if map_point is None:
                        break
                    world_point = self.map_to_world(map_point) # new goal to try to publish
        
        rospy.logerr("[PUBLISH GOAL] Failed to find valid goal after retries")
        return False

#-------------------------------------------- Information Gain ----------------------------------------------#
    
    # Compute information gain around a given point
    def compute_information_gain(self, x, y):
        """
        # Inf gain is the reduction in uncertainty about the occupancy of a cell after observing its value
        # We use a 3x3 neighborhood to compute the inf gain
        # The inf gain is computed as the entropy of the occupancy distribution of the neighborhood
        """
        # Radius to compute information gain
        radius = int(0.5 / self.resolution)
        x_min = max(0, x - radius)
        x_max = min(self.map.shape[1], x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(self.map.shape[0], y + radius + 1)
        subgrid = self.map[y_min:y_max, x_min:x_max] # radius area
        total = subgrid.size # number of cells in the radius area
        if total == 0:
            return 0

        # Calculate the probabilities of unknown, free and occupied cells
        p_unknown = np.count_nonzero(subgrid == -1) / total
        p_free = np.count_nonzero(subgrid == 0) / total
        p_occ = np.count_nonzero(subgrid == 100) / total

        # Compute the entropy (i.e. the information gain)
        probs = [p_unknown, p_free, p_occ]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)

        return entropy
    
#------------------------------------------------- Main ------------------------------------------------------#

if __name__ == '__main__':
    rospy.loginfo("[MAIN] Starting frontier detector node...")
    node = FrontierDetector()
    rospy.spin()
