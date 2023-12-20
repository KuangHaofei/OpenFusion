#!/usr/bin/env python
import argparse
import os
import shutil
import time

import cv2
import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation as R

import rospy
from cv_bridge import CvBridge
import message_filters
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from open3d_ros_helper import open3d_ros_helper

from utils_ros import pose_to_transformation_matrix, get_sampling_based_viewpoints, ensure_dir, get_pcd, \
    find_closest_instance, find_closest_ray, pose2d_to_posemsg
from configs.build import get_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="hospital", help='Path to dir of dataset.')
    parser.add_argument('--scene', type=str, default="gazebo_hospital", help='Name of the scene in the dataset.')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    return args


class SLAMNode:
    def __init__(self, args):
        self.args = args

        # transformations
        self.base2camera = np.array(
            [[1, 0, 0, 0.151],
             [0, 1, 0, 0.018],
             [0, 0, 1, 1.282],
             [0, 0, 0, 1]])

        self.camera2realcamera = np.array(
            [[0.0, 0.0, 1.0, 0.0],
             [-1.0, 0.0, 0.0, 0.0],
             [0.0, -1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]])

        pose = np.loadtxt(f'/home/ipbhk/OpenFusion/sample/{args.data}/{args.scene}/poses.txt')[0]
        camera_origin = Pose()
        camera_origin.position.x, camera_origin.position.y, camera_origin.position.z = pose[:3]
        camera_origin.orientation.x, camera_origin.orientation.y, camera_origin.orientation.z, camera_origin.orientation.w = pose[
                                                                                                                             3:]
        self.global2camera = pose_to_transformation_matrix(camera_origin) @ self.camera2realcamera

        # semantic-based searching
        self.rgb_points = np.load(f"results/{args.data}_{args.scene}/rgb_points.npy")
        self.rgb_colors = np.load(f"results/{args.data}_{args.scene}/rgb_colors.npy")
        self.full_rgb_pcd = get_pcd(self.rgb_points, self.rgb_colors)
        self.full_rgb_pcd.transform(self.global2camera)

        self.sem_points = np.load(f"results/{args.data}_{args.scene}/sem_points.npy")
        self.sem_colors = np.load(f"results/{args.data}_{args.scene}/sem_colors.npy")
        self.semseg = np.load(f"results/{args.data}_{args.scene}/semseg.npy")
        self.full_sem_pcd = get_pcd(self.sem_points, self.sem_colors)
        self.full_sem_pcd.transform(self.global2camera)

        params = get_config(args.data, args.scene)
        self.total_classes = params['objects']

        self.exploration_objects = [
            "bed",
            "sofa",
            "chair",
        ]
        # clustering parameters
        self.clustering_params = yaml.load(open(f"sample/{args.data}/{args.scene}/clustering.yaml"),
                                           Loader=yaml.FullLoader)
        self.exploration_pcds = self.get_semantic()

        # start ROS node
        rospy.init_node('slam_node')

        self.count = 0
        self.nav_completed_flag = True
        self.robot_pose = Pose()

        # collecting data
        self.root_dir = f"sample/{args.data}/{args.scene}/collect_data"
        if os.path.exists(self.root_dir) and args.save:
            shutil.rmtree(self.root_dir)

        if args.save:
            ensure_dir(os.path.join(self.root_dir, 'rgb'))
            ensure_dir(os.path.join(self.root_dir, 'depth'))
        self.file_counter = 0
        self.callback_counter = 0
        self.save_frequency = 10

        self.bridge = CvBridge()
        self.setup_subscribers()

        # publisher
        self.rgb_pcd_pub = rospy.Publisher('/rgb_pointcloud', PointCloud2, queue_size=1)
        self.sem_pcd_pub = rospy.Publisher('/sem_pointcloud', PointCloud2, queue_size=1)
        self.total_views_pub = rospy.Publisher('/total_viewpoints', PoseArray, queue_size=1)
        self.next_views_pub = rospy.Publisher('/next_viewpoints', PoseArray, queue_size=1)
        self.sempcd_pub = rospy.Publisher('/object', PointCloud2, queue_size=1)

        # subscriber
        self.nav_completed_flag_sub = rospy.Subscriber('/nav_completed', Bool, self.nav_completed_callback)
        self.costmap_subscriber = rospy.Subscriber(
            '/move_base/global_costmap/costmap', OccupancyGrid, self.costmap_callback)
        self.costmap = OccupancyGrid()

        # semantic-based completion
        for i in range(10):
            self.publish_3dmap(time_stamp=rospy.Time.now())

        self.save_data_flag = True
        self.semantic_based_completion()

    def setup_subscribers(self):
        rgb_sub = message_filters.Subscriber('/realsense/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/realsense/depth/image_rect_raw', Image)
        # pose_sub = message_filters.Subscriber('/amcl_pose', PoseWithCovarianceStamped)
        pose_sub = message_filters.Subscriber('/dingo_gt/odom', Odometry)

        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, pose_sub], 10, 0.01)
        ts.registerCallback(self.sensor_callback)

    def sensor_callback(self, rgb_msg: Image, depth_msg: Image, pose_msg: Odometry):
        self.callback_counter += 1
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        pose = pose_msg.pose.pose
        self.robot_pose = pose

        # pose to numpy matrix (transfer to global frame)
        pose = pose_to_transformation_matrix(pose) @ self.base2camera

        # self.publish_3dmap(time_stamp=pose_msg.header.stamp)

        if self.args.save and self.callback_counter % self.save_frequency == 0:
            self.save_data(rgb, depth, pose)
            self.callback_counter = 0

    def nav_completed_callback(self, msg: Bool):
        self.nav_completed_flag = msg.data
        if self.nav_completed_flag:
            print("Navigation completed")

    def costmap_callback(self, data):
        self.costmap = data

    def semantic_based_completion(self):
        self.save_data_flag = True
        print("[*] Starting semantic-based completion...")

        for object_idx in range(len(self.exploration_pcds)):
            print(f"Collecting more views of [{self.exploration_objects[object_idx]}]......")
            objects_pcd = self.exploration_pcds[object_idx]
            print(objects_pcd)

            dbscan_params = self.clustering_params[self.exploration_objects[object_idx]]
            labels = np.asarray(
                objects_pcd.cluster_dbscan(
                    eps=dbscan_params["eps"], min_points=dbscan_params["min_points"], print_progress=False))
            if labels.size == 0:
                max_label = -1
            else:
                max_label = labels.max()
            print(f"point cloud has [{max_label + 1}] clusters")
            cluster_sizes = np.bincount(labels[labels >= 0])
            large_clusters = [i for i, size in enumerate(cluster_sizes) if size > dbscan_params["min_size"]]
            # Sort these clusters based on their sizes and select the top two
            large_clusters = sorted(large_clusters, key=lambda x: cluster_sizes[x], reverse=True)
            # large_clusters = large_clusters[:dbscan_params['max_instances']]
            filtered_labels = np.array([label if label in large_clusters else -1 for label in labels])

            # collect data
            instances = []
            for label in np.unique(filtered_labels):
                if label < 0:
                    continue  # Ignore noise points with label
                instance_pcd: o3d.geometry.PointCloud = \
                    objects_pcd.select_by_index(np.where(labels == label)[0])
                colors = np.array(instance_pcd.colors)
                colors[:] = np.array([1.0, 0, 0])
                instance_pcd.colors = o3d.utility.Vector3dVector(colors)
                instances.append(instance_pcd)

            while instances:
                # get viewpoints around pointclouds
                robot_position = [self.robot_pose.position.x, self.robot_pose.position.y]

                # find the closest instance to the robot
                instance_pcd, closest_index = find_closest_instance(robot_position, instances)
                print("Current groups is ", instance_pcd)
                instances.pop(closest_index)

                # nvs = get_covered_viewpoints(instance_pcd, timestamp, frame_id)
                viewpoints = get_sampling_based_viewpoints(
                    instance_pcd, self.costmap, max_distance=3.0, sampling_interval=0.3)

                while viewpoints:
                    # get viewpoints around pointclouds
                    robot_position = [self.robot_pose.position.x, self.robot_pose.position.y]
                    frame_id = 'map'
                    timestamp = rospy.Time.now()

                    # visualize viewpoints
                    flat_viewpoints = [vp for sublist in viewpoints for vp in sublist]
                    total_views = PoseArray()
                    total_views.header.stamp = timestamp
                    total_views.header.frame_id = frame_id
                    for viewpoint in flat_viewpoints:
                        nv = pose2d_to_posemsg(viewpoint)
                        total_views.poses.append(nv)

                    # visualize current instance
                    ros_pcd = open3d_ros_helper.o3dpc_to_rospc(instance_pcd, frame_id, timestamp)

                    # find the next closest rays
                    closest_ray_index = find_closest_ray(robot_position, viewpoints)
                    next_viewpoints = viewpoints[closest_ray_index]
                    viewpoints.pop(closest_ray_index)

                    # convert to ROS msgs
                    next_views = PoseArray()
                    next_views.header.stamp = timestamp
                    next_views.header.frame_id = frame_id
                    for viewpoint in next_viewpoints:
                        next_view = pose2d_to_posemsg(viewpoint)
                        next_views.poses.append(next_view)

                    # Publish the point cloud
                    self.nav_completed_flag = False
                    self.next_views_pub.publish(next_views)

                    rate = rospy.Rate(10)  # 10 Hz
                    while not rospy.is_shutdown() and not self.nav_completed_flag:
                        self.sempcd_pub.publish(ros_pcd)
                        self.total_views_pub.publish(total_views)
                        rate.sleep()
                print("================")

        print("[*] Finished the model completion!")
        self.save_data_flag = False

    def get_semantic(self):
        print("[*] Loading semantic pointcloud from pre-built files...")
        t = time.time()

        exploration_pcds = []
        if self.exploration_objects:
            for object in self.exploration_objects:
                object_class_id = self.total_classes.index(object)
                included_mask = np.isin(self.semseg, object_class_id)
                exploration_points = self.sem_points[included_mask]
                exploration_colors = self.sem_colors[included_mask]

                exploration_pcd = get_pcd(exploration_points, exploration_colors)
                exploration_pcd.transform(self.global2camera)

                exploration_pcds.append(exploration_pcd)

        print(f"[*] Loading the Map by pre-built files in {time.time() - t:.2f}s")
        return exploration_pcds

    def publish_3dmap(self, time_stamp):
        ros_rgb_pcd = open3d_ros_helper.o3dpc_to_rospc(self.full_rgb_pcd, 'map', time_stamp)
        ros_sem_pcd = open3d_ros_helper.o3dpc_to_rospc(self.full_sem_pcd, 'map', time_stamp)
        self.rgb_pcd_pub.publish(ros_rgb_pcd)
        self.sem_pcd_pub.publish(ros_sem_pcd)

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
        print("Done")


if __name__ == "__main__":
    args = get_args()
    node = SLAMNode(args)
    node.run()
    node.stop()
