#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

class GoalManager:
    def __init__(self):
        # Initialise node
        rospy.init_node('goal_manager')

        # Initialise variables
        self.goal_queue = []
        self.goal_active = False

        # Publishers and subscribers
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        rospy.Subscriber('/new_goal', PoseStamped, self.new_goal_callback)
        rospy.Subscriber('/at_goal', Bool, self.goal_status_callback)

    def new_goal_callback(self, msg):
        self.goal_queue.append(msg)
        self.publish_new_goal()

    def goal_status_callback(self, msg):
        if msg.data: 
            self.goal_active = False
            self.publish_new_goal()

    def publish_new_goal(self):
        if not self.goal_active and self.goal_queue:
            new_goal = self.goal_queue.pop(0)
            self.goal_pub.publish(new_goal)
            self.goal_active = True

if __name__ == '__main__':
    GoalManager()
    rospy.spin()
