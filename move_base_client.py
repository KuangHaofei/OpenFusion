#!/usr/bin/env python
import rospy
import actionlib
from geometry_msgs.msg import PoseArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool


class MoveBaseClient:
    def __init__(self):
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.pose_subscriber = rospy.Subscriber('/dingo_viewpoints', PoseArray, self.pose_callback)
        self.client.wait_for_server()

        self.costmap_subscriber = rospy.Subscriber(
            '/move_base/global_costmap/costmap', OccupancyGrid, self.costmap_callback)
        self.costmap = OccupancyGrid()

        self.nv_pub = rospy.Publisher('/next_viewpoints', PoseArray, queue_size=10)

        # flag of navigation
        self.nav_completed_flag_pub = rospy.Publisher('/nav_completed', Bool, queue_size=10)

    def costmap_callback(self, data):
        self.costmap = data

    def pose_callback(self, pose_array):
        rospy.loginfo("Received %d poses", len(pose_array.poses))
        # check viewpoints are reachable or not
        valid_poses = PoseArray()
        valid_poses.header = pose_array.header
        for pose in pose_array.poses:
            if self.is_goal_reachable(pose.position.x, pose.position.y):
                valid_poses.poses.append(pose)
        self.nv_pub.publish(valid_poses)
        rospy.loginfo("Valid %d poses", len(valid_poses.poses))

        for pose in valid_poses.poses:
            self.send_goal(pose)

        self.nav_completed_flag_pub.publish(Bool(True))
        rospy.loginfo("Navigation completed")

    def send_goal(self, pose):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = pose
        self.client.send_goal(goal)
        wait = self.client.wait_for_result()
        if not wait:
            rospy.logerr("Action server not available!")
        else:
            return self.client.get_result()

    def is_goal_reachable(self, goal_x, goal_y):
        if self.costmap is None:
            rospy.logwarn("Costmap data is not yet available.")
            return False

        # Convert the goal coordinates to the costmap grid coordinates
        mx = int((goal_x - self.costmap.info.origin.position.x) / self.costmap.info.resolution)
        my = int((goal_y - self.costmap.info.origin.position.y) / self.costmap.info.resolution)

        # Check if the coordinates are within the costmap bounds
        if mx < 0 or my < 0 or mx >= self.costmap.info.width or my >= self.costmap.info.height:
            rospy.logwarn("Goal is outside the bounds of the costmap.")
            return False

        # Calculate the index in the costmap data array
        index = mx + my * self.costmap.info.width

        # Check if the goal location is free (typically, 0 is free space)
        return self.costmap.data[index] == 0


if __name__ == '__main__':
    rospy.init_node('move_base_client')
    try:
        client = MoveBaseClient()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
