#!/usr/bin/env python
import argparse
import os
import time

import numpy as np
import open3d as o3d

import rospy
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from open3d_ros_helper import open3d_ros_helper

from configs.build import get_config
from openfusion.slam import build_slam
from utils_ros import pose_to_transformation_matrix


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default="vlfusion", choices=["default", "cfusion", "vlfusion"])
    parser.add_argument('--vl', type=str, default="seem", help="vlfm to use")
    parser.add_argument('--data', type=str, default="dingo_house", help='Path to dir of dataset.')
    parser.add_argument('--scene', type=str, default="dingo_house", help='Name of the scene in the dataset.')
    parser.add_argument('--device', type=str, default="cuda:0", choices=["cpu:0", "cuda:0"])
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--host_ip', type=str, default="YOUR IP")  # for stream
    args = parser.parse_args()
    return args


class SLAMNode:
    def __init__(self, args):
        self.args = args
        self.params = get_config(args.data, args.scene)

        # transformations
        self.base2camera = np.array(
            [[0, 0, 1, 0.151],
             [-1, 0, 0, 0.018],
             [0, -1, 0, 1.282],
             [0, 0, 0, 1]])
        self.camera_mat = np.array(
            [[358.2839686653547, 0.0, 340.5],
             [0.0, 358.2839686653547, 240.5],
             [0.0, 0.0, 1.0]])

        pose = np.loadtxt('/home/ipbhk/OpenFusion/sample/dingo/dingo_house/poses.txt')[0]
        camera_origin = Pose()
        camera_origin.position.x, camera_origin.position.y, camera_origin.position.z =  pose[:3]
        camera_origin.orientation.x, camera_origin.orientation.y, camera_origin.orientation.z, camera_origin.orientation.w = pose[3:]
        self.global2camera = pose_to_transformation_matrix(camera_origin)

        # build slam system
        print("[*] building SLAM system...")
        data = np.load(f"results/{args.data}_{args.scene}/{args.algo}.npz")
        t = time.time()
        self.slam = build_slam(args, self.camera_mat, self.params)
        if args.load:
            if os.path.exists(f"results/{args.data}_{args.scene}/{args.algo}.npz"):
                print("[*] loading saved state...")
                self.slam.point_state.load(f"results/{args.data}_{args.scene}/{args.algo}.npz")
            else:
                print("[*] no saved state found, skipping...")
        print(f"[*] build SLAM system in {time.time() - t:.2f}s")

        self.count = 0

        rospy.init_node('slam_node')
        self.bridge = CvBridge()
        self.setup_subscribers()

        self.pcd_pub = rospy.Publisher('/dingo_pointcloud', PointCloud2, queue_size=10)

    def setup_subscribers(self):
        # 创建消息订阅器
        rgb_sub = message_filters.Subscriber('/realsense/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/realsense/depth/image_rect_raw', Image)
        pose_sub = message_filters.Subscriber('/dingo_gt/odom', Odometry)

        # 创建一个同步器（ApproximateTime或ExactTime）
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, pose_sub], 10, 0.1)
        ts.registerCallback(self.callback)

    def callback(self, rgb_msg: Image, depth_msg: Image, odom_msg: Odometry):
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        pose = odom_msg.pose.pose
        time_stamp = rgb_msg.header.stamp

        # pose to numpy matrix (transfer to global frame)
        pose = pose_to_transformation_matrix(pose) @ self.base2camera

        # SLAM
        self.update(rgb, depth, pose)

        # publish pointclouds
        curr_pcd = self.get_pc()
        print("Here")
        ros_pcd = open3d_ros_helper.o3dpc_to_rospc(curr_pcd, 'map', time_stamp)
        # # Publish the point cloud
        self.pcd_pub.publish(ros_pcd)

    def update(self, rgb, depth, pose):
        # preprocess data
        depth = (depth * self.params['depth_scale']).astype(np.uint16)
        pose = np.linalg.inv(self.global2camera) @ pose
        pose = np.linalg.inv(pose.astype(np.float64))

        self.slam.io.update(rgb, depth, pose)
        self.slam.vo()
        self.slam.compute_state(encode_image=self.count % 10 == 0)
        if self.count % 10 == 0:
            self.count = 0
        self.count += 1

    def get_pc(self):
        # in camera frame
        pcd = self.slam.point_state.world.extract_point_cloud().to_legacy()
        pcd.transform(self.global2camera)
        return pcd

    def get_occ_map(self):
        pass

    def run(self):
        rospy.spin()

    def stop(self):
        rospy.signal_shutdown("Shutting down")
        print("Done")


if __name__ == "__main__":
    args = get_args()
    node = SLAMNode(args)
    node.run()
    node.stop()
