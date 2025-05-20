#!/usr/bin/env python3
import numpy as np
import time
import rospy
import tf
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA 
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray 
from std_msgs.msg import Bool
from utils_lib.online_planning import StateValidityChecker, compute_path

class OnlinePlanner:

#------------------------------------------- Initialisation ----------------------------------------------#

    def __init__(self, gridmap_topic, odom_topic, cmd_vel_topic, bounds, distance_threshold):

        # Initialise variables
        self.svc = StateValidityChecker(distance_threshold)
        self.current_pose = None
        self.goal = None
        self.at_goal = False
        self.last_map_time = rospy.Time.now()
        self.bounds = bounds
        self.path = []
        self.tree = []

        # Initialise motion planner parameters
        self.Kv = 0.5 #0.5
        self.Kw = 0.5 #1.2
        self.v_max = 0.05 #0.3
        self.w_max = 0.5 #1.0
        self.distance_threshold = 0.1 #0.2

        # Initialise robot parameters
        self.wheel_radius = 0.035
        self.wheel_base_distance = 0.257

        # Publishers
        self.cmd_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        self.marker_pub = rospy.Publisher('~path_marker', Marker, queue_size=1)
        self.goal_marker_pub = rospy.Publisher('~goal_marker', Marker, queue_size=1)
        self.tree_marker_pub = rospy.Publisher('tree', MarkerArray, queue_size=10)
        self.atgoal_pub = rospy.Publisher('/at_goal', Bool, queue_size=10)

        # Subscribers
        self.gridmap_sub = rospy.Subscriber(gridmap_topic, OccupancyGrid, self.get_gridmap)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.get_odom)
        self.move_goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_pose)


        rospy.Timer(rospy.Duration(0.1), self.controller) # frequency to call controller()

#------------------------------------------- Helper Functions ----------------------------------------------#

    # Bound an angle to the range [-pi, pi]
    def wrap_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return -angle

#------------------------------------------- Callback Functions ----------------------------------------------#

    # Callback to get the odometry
    def get_odom(self, odom):
        # Convert the quarternion to a yaw angle
        _, _, yaw = tf.transformations.euler_from_quaternion([
            odom.pose.pose.orientation.x, 
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w])
        # Set the current (x, y, theta) pose
        self.current_pose = np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            yaw])
        
        #print("odom yaw", yaw)  # Callback to check goal validity when a new goal is received

    def goal_pose(self, goal):
        if self.current_pose is None:
             return     # wait until odometry is received
        
        if self.svc.there_is_map:
            self.goal = np.array([goal.pose.position.x, goal.pose.position.y])
            self.publish_goal_marker(self.goal) # publishes goal marker
            self.path = []
            self.path = self.plan() # plan a path to the goal
            self.at_goal = False
            self.atgoal_pub.publish(Bool(data=False)) # goal not reached yet


    # Callback to check path validity when a new occupancy grid map is received
    def get_gridmap(self, gridmap):
        if self.current_pose is None:
            return  # wait until odometry is received
        
        self.current_map = gridmap # store the current map

        if (gridmap.header.stamp - self.last_map_time).to_sec() > 1: # only update if its been more than 1 second
            self.last_map_time = gridmap.header.stamp
            env = np.array(gridmap.data).reshape(gridmap.info.height, gridmap.info.width).T # convert map data to 2d array
            origin = [gridmap.info.origin.position.x, gridmap.info.origin.position.y]
            self.svc.set(env, gridmap.info.resolution, origin) # update svc with new map

            # If a path exists
            if self.path is not None and len(self.path) > 0:
                # Take a small part of the path with just the current pose and next two waypoints
                path = [self.current_pose[:2]] + self.path[:2]
                # If that path is invalid, replan a new path
                if not self.svc.check_path(path):
                    rospy.logwarn("[MAP] Path invalid after remappping. Replanning.")
                    self.path = []
                    self.at_goal = False
                    goal_reached_msg = Bool()
                    goal_reached_msg.data = self.at_goal
                    self.atgoal_pub.publish(goal_reached_msg)
                    self.path = self.plan()
                # If the goal becomes invalid, choose a new goal along the path !!! should we do this?
                elif self.svc.is_valid(self.goal) == False:
                     self.goal = self.new_goal_in_path(path)
                     rospy.loginfo(f"[MAP] Alternative goal found in the path: {self.goal}")
                     self.at_goal = False
                     goal_reached_msg = Bool()
                     goal_reached_msg.data = self.at_goal
                     self.atgoal_pub.publish(goal_reached_msg)
                     self.path = self.plan() # replan a new path

#------------------------------------------- Fallback Functions ----------------------------------------------#     
    def trap_avoidance(self):
        rospy.logwarn("[TRAP] Executing trap avoidance...")

        # Get current map and pose
        grid = self.current_map
        res = grid.info.resolution
        ox, oy = grid.info.origin.position.x, grid.info.origin.position.y
        width, height = grid.info.width, grid.info.height

        # Convert robot's position to map index
        map_pos = self.svc.__position_to_map__((self.current_pose[0], self.current_pose[1]))
        if map_pos is None:
            rospy.logwarn("[TRAP] Robot is out of map bounds.")
            return
        mx, my = map_pos

        best_theta = None
        max_clear_dist = 0.0

        # Scan directions (in robot frame, not fixed world)
        for theta in np.linspace(0, 2*np.pi, 36, endpoint=False):
            dist = 0.0
            while dist < 1.0:
                # Raycasting in map index space
                tx = mx + int((dist * np.cos(theta)) / res)
                ty = my + int((dist * np.sin(theta)) / res)

                if not (0 <= tx < width and 0 <= ty < height):
                    break

                cell_value = grid.data[ty * width + tx]
                if cell_value > 50:
                    break  # hit an obstacle

                dist += res

            if dist > max_clear_dist:
                max_clear_dist = dist
                best_theta = theta

        if best_theta is None:
            rospy.logwarn("[TRAP] No clear direction found.")
            return

        # Convert best_theta (in local robot frame) to global yaw
        desired_yaw = self.wrap_angle(self.current_pose[2] + best_theta)
        yaw_tolerance = 0.1
        max_rotation_time = 3.0
        start_time = rospy.Time.now()

        rospy.loginfo(f"[TRAP] Rotating to clear direction: θ={best_theta:.2f} → yaw={desired_yaw:.2f}")

        # Rotate toward best direction
        while (rospy.Time.now() - start_time).to_sec() < max_rotation_time:
            current_yaw = self.current_pose[2]
            yaw_error = self.wrap_angle(desired_yaw - current_yaw)

            if abs(yaw_error) < yaw_tolerance:
                rospy.loginfo("[TRAP] Finished rotating.")
                break

            twist = Twist()
            twist.angular.z = 0.5 * yaw_error
            self.cmd_pub.publish(twist)
            rospy.sleep(0.05)
        else:
            rospy.logwarn("[TRAP] Rotation timed out.")

        self.cmd_pub.publish(Twist())  # stop rotation

        # Small forward nudge
        rospy.loginfo("[TRAP] Moving forward slightly to escape.")
        twist = Twist()
        twist.linear.x = 0.1
        self.cmd_pub.publish(twist)
        rospy.sleep(1.0)  # Move forward for 1 second
        self.cmd_pub.publish(Twist())  # Stop again

        rospy.loginfo("[TRAP] Trap avoidance complete.")


    # Choose another goal along the path
    def new_goal_in_path(self, path, index=-1):
        # If no index is provided, start at the last waypoint
        if index == -1:
            index = len(path) - 1

        # Step backward through the path to find a valid goal
        while index >= 0:
            candidate_goal = path[index][:2]  # get (x, y) position (ignore theta)
            if self.svc.is_valid(candidate_goal):
                return np.array(candidate_goal) # return candidate goal if valid
            index -= 1 # otherwise move one step back
        
        # If no valid point is found, return None
        return None
    
    def check_again(self):
        if not self.svc.is_valid(self.current_pose[0:2]):
            rospy.logwarn("Invalid current position, obstacle ahead")
            # Move backward to avoid the obstacle
            start_time = time.time()
            while time.time() - start_time < 1.0:
                self.__send_command__(-0.5, 0.0)
            self.__send_command__(0.0, 0.0)
            self.trap_avoidance()  # Call trap avoidance
            # Clear the existing path and re-plan
            del self.path[:]
            self.plan()
    
    

    def add_heading_to_path(self, path):
        path_with_yaw = []
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            yaw = math.atan2(dy, dx)
            path_with_yaw.append([path[i][0], path[i][1], yaw])
        # Add final point with same yaw as previous
        path_with_yaw.append([path[-1][0], path[-1][1], path_with_yaw[-1][2]])
        return path_with_yaw

    
#------------------------------------------- Path Planning ----------------------------------------------#

    # Compute valid path from current pose to goal
    def plan(self):
        attempt = 0
        rospy.loginfo(f"[PLAN] Planning a path...")

        while len(self.path) <= 1 and attempt < 6:
            rospy.loginfo(f"[PLAN] Attempt {attempt + 1} to find a path.")
            if not self.svc.is_valid(self.current_pose[:2]):
                rospy.logwarn("[PLAN] Current pose is invalid. Executing trap avoidance...")
                self.check_again()
                return []

            raw_path = compute_path(self.current_pose[:2], self.goal, self.svc, 1000)
            
            if raw_path:
                full_path = [self.current_pose[:2]] + raw_path
                self.path = self.add_heading_to_path(full_path)
                self.publish_path(self.path)
                self.publish_tree([])  # Optional: fill this with actual tree later
                break
            else:
                rospy.logwarn(f"[PLAN] No path found on attempt {attempt + 1}")
                self.at_goal = True
                self.atgoal_pub.publish(Bool(data=True))

            attempt += 1

        if attempt == 3 or not self.path:
            rospy.logwarn("[PLAN] All attempts failed. Triggering trap avoidance ?.")
            self.check_again()
            return []

        rospy.loginfo("[PLAN] Path found!")
        self.at_goal = True
        self.atgoal_pub.publish(Bool(data=False))
        del self.path[0]  # remove current pose
        return self.path
    
#------------------------------------------- Controller ----------------------------------------------#

    # Control loop called every 0.1 seconds
    def controller(self, event):
        
        if self.current_pose is None or not self.svc.there_is_map:
            return  # wait until odometry is received
        
        #self.check_again() # check if the robot is stuck
        v, w = 0, 0

        # If a path is available
        if len(self.path) > 0:
            #rospy.loginfo("[CTRL] Path available.")
            waypoint = np.array(self.path[0]) # set next waypoint as the target
            delta = np.array(waypoint[:2]) - self.current_pose[:2]
            dist = np.linalg.norm(delta)
            angle_to_goal = math.atan2(delta[1], delta[0])
            #print("Angle to goal: ", angle_to_goal)
            angle_err = self.wrap_angle(angle_to_goal - self.current_pose[2])

            # If close enough to the waypoint
            if dist < self.distance_threshold:
                rospy.loginfo(f"[CTRL] Reached waypoint: {waypoint}")
                #print("Velocities after waypoint reached: ", v, w)
                del self.path[0] # remove the reached waypoint
                if not self.path: # if the path is empty, the goal is reached
                    rospy.loginfo("[CTRL] Reached the goal. Stopping the robot.")
                    self.at_goal = True
                    goal_reached_msg = Bool()
                    goal_reached_msg.data = self.at_goal
                    self.atgoal_pub.publish(goal_reached_msg)
                    self.__send_command__(0, 0) # stop the robot
                    return
            else:
                # Calculate linear and angular velocities to move towards waypoint
                #print("Angle error = ", abs(angle_err))
                #print("Kv * dist = ", self.Kv * dist)
                #print("v_max = ", self.v_max)
                if abs(angle_err) > np.pi / 4:
                    #print("Angle err: ", angle_err)
                    v = 0.0
                else:
                    v = min(self.Kv * dist, self.v_max)
                w = np.clip(-self.Kw * angle_err, -self.w_max, self.w_max)
                self.at_goal = False
                goal_reached_msg = Bool()
                goal_reached_msg.data = self.at_goal
                self.atgoal_pub.publish(goal_reached_msg)

        #print("Velocities to move towards waypoint: ", v, w)
        self.__send_command__(v, w) # send calculated velocities

    # Send velocity commands to the robot
    def __send_command__(self, v, w):
        twist = Twist()
        twist.linear.x = np.clip(v, -self.v_max, self.v_max)    # clip linear velocity within limits
        twist.angular.z = np.clip(w, -self.w_max, self.w_max)   # clip angular velocity within limits
        self.cmd_pub.publish(twist)

#------------------------------------------- Visualisation ----------------------------------------------#
    
    # Publish visualisation markers for planned path
    def publish_path(self, path):
        m = Marker()
        m.header.frame_id = 'odom'
        m.header.stamp = rospy.Time.now()
        m.ns = 'path'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.1
        m.color = ColorRGBA(1, 0, 0, 1) # red
        m.pose.orientation.w = 1.0

        p = Point()
        p.x = self.current_pose[0]
        p.y = self.current_pose[1]
        m.points.append(p)

        for point in path:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            m.points.append(p)

        self.marker_pub.publish(m)

    # Publish visual marker for goal
    def publish_goal_marker(self, goal_point):
        m = Marker()
        m.header.frame_id = 'odom'
        m.header.stamp = rospy.Time.now()
        m.ns = 'goal'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = goal_point[0]
        m.pose.position.y = goal_point[1]
        m.pose.position.z = 0
        m.scale.x = 0.3
        m.scale.y = 0.3
        m.scale.z = 0.3
        m.color = ColorRGBA(0, 1, 0, 1) # green
        self.goal_marker_pub.publish(m)

    # Publish tree visualisation !!! currently empty, to do
    def publish_tree(self, tree):
        marker_array = MarkerArray()
        self.tree_marker_pub.publish(marker_array)
    
#------------------------------------------- Main ----------------------------------------------#

if __name__ == '__main__':
    # Initialise planner node
    rospy.init_node('turtlebot_online_path_planning_node')
    # Initialise OnlinePlanner object
    node = OnlinePlanner('/projected_map', '/turtlebot/kobuki/odom', '/turtlebot/kobuki/commands/velocity',
                        np.array([-10.0, 10.0, -10.0, 10.0]), 0.2)
    rospy.spin() # keep node running