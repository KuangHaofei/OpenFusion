#!/usr/bin/env python
import argparse
import os
import time

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

import rospy
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from open3d_ros_helper import open3d_ros_helper
from std_msgs.msg import Bool

from configs.build import get_config
from openfusion.slam import build_slam
from openfusion.utils import get_pcd
from utils_ros import pose_to_transformation_matrix, get_covered_viewpoints, get_sampling_based_viewpoints, ensure_dir


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
    parser.add_argument('--save', action='store_true')
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
        camera_origin.position.x, camera_origin.position.y, camera_origin.position.z = pose[:3]
        (camera_origin.orientation.x, camera_origin.orientation.y,
         camera_origin.orientation.z, camera_origin.orientation.w) = pose[3:]
        self.global2camera = pose_to_transformation_matrix(camera_origin)

        # build slam system
        print("[*] building SLAM system...")
        t = time.time()
        self.slam = build_slam(args, self.camera_mat, self.params)
        if args.load:
            if os.path.exists(f"results/{args.data}_{args.scene}/{args.algo}.npz"):
                print("[*] loading saved state...")
                self.slam.point_state.load(f"results/{args.data}_{args.scene}/{args.algo}.npz")
            else:
                print("[*] no saved state found, skipping...")
        print(f"[*] build SLAM system in {time.time() - t:.2f}s")

        # semantic-based searching
        self.total_classes = [
            'floor', 'wall', 'ceil', 'sofa', 'bed', 'table', 'cabinet',
            'home appliances', 'chair', 'ball', 'trash', 'tv', 'light', 'others']
        # self.exploration_objects = ['sofa', 'bed']
        self.exploration_objects = ['sofa']
        self.exploration_pcds = self.get_semantic()

        # start ROS node
        rospy.init_node('slam_node')

        self.count = 0
        self.nav_completed_flag = True
        self.robot_pose = Pose()

        # collecting data
        self.root_dir = f"results/{args.data}_{args.scene}/collect_data"
        ensure_dir(os.path.join(self.root_dir, 'rgb'))
        ensure_dir(os.path.join(self.root_dir, 'depth'))
        self.file_counter = 0
        self.callback_counter = 0
        self.save_frequency = 1

        self.bridge = CvBridge()
        self.setup_subscribers()

        # publisher
        self.pcd_pub = rospy.Publisher('/dingo_pointcloud', PointCloud2, queue_size=1)
        self.nv_pub = rospy.Publisher('/dingo_viewpoints', PoseArray, queue_size=1)
        self.sempcd_pub = rospy.Publisher('/object', PointCloud2, queue_size=1)

        # subscriber
        self.nav_completed_flag_sub = rospy.Subscriber('/nav_completed', Bool, self.nav_completed_callback)

        # semantic-based completion
        self.publish_3dmap(time_stamp=rospy.Time.now())
        self.semantic_based_completion()
        self.publish_3dmap(time_stamp=rospy.Time.now())

    def setup_subscribers(self):
        rgb_sub = message_filters.Subscriber('/realsense/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/realsense/depth/image_rect_raw', Image)
        # pose_sub = message_filters.Subscriber('/dingo_gt/odom', Odometry)
        pose_sub = message_filters.Subscriber('/amcl_pose', PoseWithCovarianceStamped)

        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, pose_sub], 10, 0.1)
        ts.registerCallback(self.sensor_callback)

    def sensor_callback(self, rgb_msg: Image, depth_msg: Image, pose_msg: PoseWithCovarianceStamped):
        self.callback_counter += 1
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        pose = pose_msg.pose.pose
        self.robot_pose = pose
        time_stamp = rgb_msg.header.stamp

        # pose to numpy matrix (transfer to global frame)
        pose = pose_to_transformation_matrix(pose) @ self.base2camera

        # SLAM
        self.update(rgb, depth, pose)

        # publish pointclouds
        # self.publish_3dmap(time_stamp)

        if self.callback_counter % self.save_frequency == 0:
            self.save_data(rgb, depth, pose)
            self.callback_counter = 0

    def nav_completed_callback(self, msg: Bool):
        self.nav_completed_flag = msg.data
        if self.nav_completed_flag:
            print("Navigation completed")

    def semantic_based_completion(self):
        for object_idx in range(len(self.exploration_pcds)):
            print(f"Collecting more views of {self.exploration_objects[object_idx]}......")
            objects_pcd = self.exploration_pcds[object_idx]
            o3d.io.write_point_cloud('ros_test.ply', objects_pcd)
            print(objects_pcd)

            labels = np.asarray(objects_pcd.cluster_dbscan(eps=0.2, min_points=50, print_progress=True))
            max_label = labels.max()
            print(f"point cloud has {max_label + 1} clusters")
            # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
            # colors[labels < 0] = 0

            # collect data
            for label in np.unique(labels):
                if label < 0:
                    continue    # Ignore noise points with label
                instance_pcd: o3d.geometry.PointCloud = \
                    objects_pcd.select_by_index(np.where(labels == label)[0])
                colors = np.array(instance_pcd.colors)
                colors[:] = np.array([255, 0, 0])
                instance_pcd.colors = o3d.utility.Vector3dVector(colors)
                print("Current groups is ", instance_pcd)

                # get viewpoints around pointclouds
                robot_position = [self.robot_pose.position.x, self.robot_pose.position.y]
                timestamp = rospy.Time.now()
                frame_id = 'map'
                # nvs = get_covered_viewpoints(instance_pcd, timestamp, frame_id)
                nvs = get_sampling_based_viewpoints(
                    instance_pcd, robot_position, timestamp, frame_id,
                    max_distance=3.0, sampling_interval=0.3)

                ros_pcd = open3d_ros_helper.o3dpc_to_rospc(instance_pcd, frame_id, timestamp)

                # Publish the point cloud
                # self.nav_completed_flag = False
                self.nv_pub.publish(nvs)
                self.nav_completed_flag = False

                rate = rospy.Rate(10)  # 10 Hz
                while not rospy.is_shutdown() and not self.nav_completed_flag:
                    self.sempcd_pub.publish(ros_pcd)
                    rate.sleep()
                print("================")

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

    def get_semantic(self):
        print("[*] Querying the Map by language to generate semantic map...")
        t = time.time()
        points, colors, semseg = self.slam.semantic_query(
            self.total_classes,
            save_file=f"results/{args.data}_{args.scene}/cmap.png")

        exploration_pcds = []
        if self.exploration_objects:
            for object in self.exploration_objects:
                object_class_id = self.total_classes.index(object)
                included_mask = np.isin(semseg, object_class_id)
                exploration_points = points[included_mask]
                exploration_colors = colors[included_mask]

                exploration_pcd = get_pcd(exploration_points, exploration_colors)
                exploration_pcd.transform(self.global2camera)

                exploration_pcds.append(exploration_pcd)

        print(f"[*] Querying the Map by language in {time.time() - t:.2f}s")
        return exploration_pcds

    def publish_3dmap(self, time_stamp):
        curr_pcd = self.get_pc()
        ros_pcd = open3d_ros_helper.o3dpc_to_rospc(curr_pcd, 'map', time_stamp)
        self.pcd_pub.publish(ros_pcd)

    def save_data(self, rgb, depth, pose):
        # File name formatting
        file_name = f"{self.file_counter:05d}"

        # Save RGB image and Depth
        cv2.imwrite(os.path.join(self.root_dir, f'rgb/{file_name}.png'), rgb)
        np.save(os.path.join(self.root_dir, f'depth/{file_name}.npy'), depth)

        # Save Pose
        trans = pose[:3, 3]
        rot = R.from_matrix(pose[:3, :3]).as_quat()
        with open(os.path.join(self.root_dir, 'poses.txt'), 'a') as f:
            f.write(f'{trans[0]} {trans[1]} {trans[2]} {rot[0]} {rot[1]} {rot[2]} {rot[3]}\n')

        self.file_counter += 1

    def run(self):
        rospy.spin()

    def stop(self):
        rospy.signal_shutdown("Shutting down")
        if self.args.save:
            file_name = f"results/{args.data}_{args.scene}/{args.algo}_completion.npz"
            self.slam.save(file_name)
            print(f"[*] saved state to {file_name}")
        print("Done")


if __name__ == "__main__":
    args = get_args()
    node = SLAMNode(args)
    node.run()
    node.stop()
