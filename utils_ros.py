import copy
import os
import time

import numpy as np
from nav_msgs.msg import OccupancyGrid
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d

from geometry_msgs.msg import Pose, PoseArray

from astar_planner_bidirectional import Bidirectional_Astar_Planner
from astart_planner import Astar_Planner
from jmp_planner import Jps_Planner

import matplotlib.pyplot as plt


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_pcd(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def pose_to_transformation_matrix(pose):
    tx, ty, tz = pose.position.x, pose.position.y, pose.position.z
    t = np.array([tx, ty, tz])

    qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, \
        pose.orientation.z, pose.orientation.w
    q = np.array([qx, qy, qz, qw])
    r = Rot.from_quat(q)

    # 3x3 rotation matrix
    rot_matrix = r.as_matrix()

    # 4x4 transformation matrix
    trans_matrix = np.eye(4)
    trans_matrix[:3, :3] = rot_matrix
    trans_matrix[:3, 3] = t

    return trans_matrix


def find_closest_instance(robot_position, instances):
    # Initialize minimum distance and closest instance
    min_distance = float('inf')
    closest_instance = o3d.geometry.PointCloud()
    closest_index = -1

    # Iterate through each point cloud
    for index, instance in enumerate(instances):
        # Calculate the centroid of the point cloud
        centroid = instance.get_center()[:2]

        # Calculate 2D distance to the ce   ntroid (ignoring Z-coordinate)
        distance = np.linalg.norm(np.array(robot_position) - centroid[:2])

        # Update minimum distance and closest instance if necessary
        if distance < min_distance:
            min_distance = distance
            closest_instance = instance
            closest_index = index

    return closest_instance, closest_index


def generate_viewpoints(center, radius, interval_degrees=30):
    viewpoints = []
    for angle in range(0, 360, interval_degrees):
        radians = np.deg2rad(angle)
        # Position on the circle
        x = center[0] + radius * np.cos(radians)
        y = center[1] + radius * np.sin(radians)
        position = (x, y)
        # Direction towards the center
        yaw = np.arctan2(center[1] - y, center[0] - x)
        viewpoints.append([position[0], position[1], yaw])

    viewpoints = np.array(viewpoints)

    return viewpoints


def pose2d_to_posemsg(pose):
    # pose: [x, y, yaw]
    pose_msg = Pose()
    pose_msg.position.x = pose[0]
    pose_msg.position.y = pose[1]
    pose_msg.position.z = 0
    q = Rot.from_euler('z', pose[2]).as_quat()
    pose_msg.orientation.x = q[0]
    pose_msg.orientation.y = q[1]
    pose_msg.orientation.z = q[2]
    pose_msg.orientation.w = q[3]
    return pose_msg


def get_covered_viewpoints(pcd: o3d.geometry.PointCloud, time_stamp, frame_id='map'):
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_center = bbox.get_center()[:2]
    bbox_points = np.asarray(bbox.get_box_points())
    radius = 0.0
    for point in bbox_points:
        if np.linalg.norm(point[:2] - bbox_center) > radius:
            radius = np.linalg.norm(point[:2] - bbox_center)
    radius += 0.5
    viewpoints = generate_viewpoints(bbox_center, radius, 15)

    # covert viewpoint to Pose array
    nvs = PoseArray()
    nvs.header.stamp = time_stamp
    nvs.header.frame_id = frame_id
    for viewpoint in viewpoints:
        nv = pose2d_to_posemsg(viewpoint)
        nvs.poses.append(nv)

    return nvs


def find_closest_ray(robot_position, rays):
    min_distance = float('inf')
    closest_ray_index = -1

    for i, ray in enumerate(rays):
        for point in ray:
            distance = np.linalg.norm(np.array(robot_position[:2]) - np.array(point[:2]))
            if distance < min_distance:
                min_distance = distance
                closest_ray_index = i

    return closest_ray_index


def generate_sampling_based_viewpoints(
        center, bbox_2d, min_radius, max_distance, costmap,
        interval_degrees=30, sampling_interval=0.1):
    rays = []
    for angle in range(0, 360, interval_degrees):
        viewpints_along_ray = []
        radians = np.deg2rad(angle)
        for distance in np.arange(min_radius, min_radius + max_distance, sampling_interval):
            x = center[0] + distance * np.cos(radians)
            y = center[1] + distance * np.sin(radians)
            if (is_goal_reachable(costmap, x, y) and
                    is_goal_too_far(costmap, bbox_2d, center[0], center[1], x, y)):
                yaw = np.arctan2(center[1] - y, center[0] - x)
                viewpints_along_ray.append([x, y, yaw])

        if viewpints_along_ray:
            rays.append(viewpints_along_ray[::-1])  # from far to near

    return rays


def get_sampling_based_viewpoints(
        pcd: o3d.geometry.PointCloud, costmap: OccupancyGrid,
        max_distance=3.0, sampling_interval=0.3):
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_center = bbox.get_center()[:2]
    bbox_points = np.asarray(bbox.get_box_points()).astype(np.float32)
    bbox_points = np.unique(bbox_points[:, :2], axis=0)

    min_radius = 0.0
    for point in bbox_points:
        if np.linalg.norm(point[:2] - bbox_center) > min_radius:
            min_radius = np.linalg.norm(point[:2] - bbox_center)
    min_radius += 0.3

    viewpoints = generate_sampling_based_viewpoints(
        bbox_center, bbox_points, min_radius, max_distance, costmap, 45, sampling_interval)

    return viewpoints


def is_goal_reachable(costmap: OccupancyGrid, goal_x, goal_y):
    if costmap is None:
        print("Costmap data is not yet available.")
        return False

    # Convert the goal coordinates to the costmap grid coordinates
    mx = int((goal_x - costmap.info.origin.position.x) / costmap.info.resolution)
    my = int((goal_y - costmap.info.origin.position.y) / costmap.info.resolution)

    # Check if the coordinates are within the costmap bounds
    if mx < 0 or my < 0 or mx >= costmap.info.width or my >= costmap.info.height:
        print("Goal is outside the bounds of the costmap.")
        return False

    # Calculate the index in the costmap data array
    index = mx + my * costmap.info.width

    # Check if the goal location is free (typically, 0 is free space)
    return costmap.data[index] == 0


def is_goal_too_far(
        costmap: OccupancyGrid, bbox_2d, start_x, start_y, end_x, end_y):
    # set bbox are to free
    tmp_costmap = set_bbox_area_to_free(costmap, bbox_2d)
    tmp_costmap = increase_grid_resolution(tmp_costmap, 0.4)

    # convert start and end points to map coordinates
    start_x = int((start_x - tmp_costmap.info.origin.position.x) / tmp_costmap.info.resolution)
    start_y = int((start_y - tmp_costmap.info.origin.position.y) / tmp_costmap.info.resolution)
    end_x = int((end_x - tmp_costmap.info.origin.position.x) / tmp_costmap.info.resolution)
    end_y = int((end_y - tmp_costmap.info.origin.position.y) / tmp_costmap.info.resolution)

    # find the distance from bbox_center to viewpoint
    global_planner = Astar_Planner()
    # global_planner = Bidirectional_Astar_Planner()
    # global_planner = Jps_Planner()
    path, distance = find_the_astar_path(
        tmp_costmap, global_planner, start_x, start_y, end_x, end_y)

    # plt_costmap(tmp_costmap, bbox_2d, start_x, start_y, end_x, end_y, path)

    if distance > 5.0:
        return False

    return True


def find_the_astar_path(costmap: OccupancyGrid, global_planner,
                        start_x, start_y, goal_x, goal_y):
    # Initialize the start and goal positions
    start = (start_x, start_y)
    end = (goal_x, goal_y)

    data = np.array(costmap.data)
    occmap = np.reshape(data, (costmap.info.height, costmap.info.width))
    occmap = np.transpose(occmap)

    # Find the path
    # t = time.time()
    if isinstance(global_planner, Astar_Planner):
        path = global_planner.astar(
            occmap, costmap.info.width, costmap.info.height, start, end)
    elif isinstance(global_planner, Bidirectional_Astar_Planner):
        path = global_planner.bi_astar(
            occmap, costmap.info.width, costmap.info.height, start, end)
    elif isinstance(global_planner, Jps_Planner):
        path = global_planner.jps(
            occmap, costmap.info.width, costmap.info.height, start, end)
    else:
        raise ValueError("Invalid global planner type")

    # print(f"Find the path cost {time.time() - t} s")

    # Calculate the distance of the path
    path = np.array(path)
    dist = calculate_path_distance(path) * costmap.info.resolution

    return path, dist


def set_bbox_area_to_free(costmap: OccupancyGrid, bbox_2d: np.array):
    # Create a copy of the costmap to modify
    tmp_costmap = copy.copy(costmap)

    # plt_costmap(tmp_costmap, bbox_2d)

    # Convert tmp_costmap.data to a list for modification
    data_list = list(tmp_costmap.data)

    # Calculate min and max coordinates for the bounding box
    min_x = min(bbox_2d[:, 0])
    max_x = max(bbox_2d[:, 0])
    min_y = min(bbox_2d[:, 1])
    max_y = max(bbox_2d[:, 1])

    # Iterate over the bounding box area
    for x in np.arange(min_x, max_x, tmp_costmap.info.resolution):
        for y in np.arange(min_y, max_y, tmp_costmap.info.resolution):
            # Convert (x, y) to map coordinates
            mx = int((x - tmp_costmap.info.origin.position.x) / tmp_costmap.info.resolution)
            my = int((y - tmp_costmap.info.origin.position.y) / tmp_costmap.info.resolution)
            # Calculate the index and set the value to 0 (free)
            index = mx + my * tmp_costmap.info.width
            if 0 <= index < len(tmp_costmap.data):
                data_list[index] = 0

    # Convert the data back to a tuple if necessary
    tmp_costmap.data = tuple(data_list)

    # plt_costmap(tmp_costmap, bbox_2d)

    return tmp_costmap


def calculate_path_distance(path: list):
    """
    Calculate the distance of a path.
    :param path: The path as a list of points.
    :return: The distance of the path.
    """
    dist = 0
    for i in range(1, len(path)):
        dist += np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
    return dist


def increase_grid_resolution(costmap: OccupancyGrid, new_resolution: float):
    # Calculate scale factor
    scale = int(np.float32(new_resolution) / np.float32(costmap.info.resolution))

    # Calculate new dimensions
    new_width = costmap.info.width // scale
    new_height = costmap.info.height // scale

    # Convert original data to 2D array
    original_data = np.array(costmap.data).reshape((costmap.info.height, costmap.info.width))

    # Downsample the data
    new_data = original_data[::scale, ::scale]

    # Create a new OccupancyGrid
    new_costmap = OccupancyGrid()
    new_costmap.info = copy.deepcopy(costmap.info)
    new_costmap.info.width = new_width
    new_costmap.info.height = new_height
    new_costmap.info.resolution = new_resolution
    new_costmap.data = new_data.flatten()

    return new_costmap


def plt_costmap(costmap: OccupancyGrid, bbox_2d,
                start_x, start_y, end_x, end_y, path):
    bbox_2d = copy.copy(bbox_2d)
    plt.figure(figsize=(10, 10))
    data = np.array(costmap.data)

    occmap = np.reshape(data, (costmap.info.height, costmap.info.width))
    # convert occmap: free is light, occupied is dark
    occmap = 1 - occmap / 100.0
    occmap = occmap.astype(np.float32)

    # convert bbox_2d points to map coordinates
    bbox_2d[:, 0] = np.int32((bbox_2d[:, 0] - costmap.info.origin.position.x) / costmap.info.resolution)
    bbox_2d[:, 1] = np.int32((bbox_2d[:, 1] - costmap.info.origin.position.y) / costmap.info.resolution)

    plt.imshow(occmap, cmap='gray', origin='lower',
               extent=[0, costmap.info.width, 0, costmap.info.height])
    plt.plot(bbox_2d[:, 0], bbox_2d[:, 1], 'r-')
    plt.plot(start_x, start_y, 'ro')
    plt.plot(end_x, end_y, 'go')
    plt.plot(path[:, 0], path[:, 1], 'b-')
    plt.show()
    plt.close()