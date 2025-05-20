#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

path_points = []

def odom_callback(msg):
    global path_points

    # Append current position
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    path_points.append((x, y))

    # Build a LINE_STRIP Marker
    marker = Marker()
    marker.header.frame_id = 'odom'  # or "odom" depending on your fixed frame
    marker.header.stamp = rospy.Time.now()
    marker.ns = 'actual_path'
    marker.id = 1
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.08
    marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)  # Blue
    marker.pose.orientation.w = 1.0

    # Add all points
    marker.points = []
    for x, y in path_points:
        p = Point()
        p.x = x
        p.y = y
        marker.points.append(p)

    marker_pub.publish(marker)

# Initialize node
rospy.init_node('odom_path_marker_node')

# Publisher
marker_pub = rospy.Publisher('~actual_path_marker', Marker, queue_size=10)

# Subscriber
rospy.Subscriber('/turtlebot/odom_ground_truth', Odometry, odom_callback)

rospy.spin()
