import time

import numpy as np
import open3d as o3d
from geometry_msgs.msg import Pose
from matplotlib import pyplot as plt

from openfusion.utils import show_pc
from utils_ros import get_pcd, pose_to_transformation_matrix


def get_semantic(sem_points, sem_colors, semseg,
                 total_classes, exploration_objects, global2camera):
    print("[*] Loading semantic pointcloud from pre-built files...")
    t = time.time()

    exploration_pcds = []
    if exploration_objects:
        for object in exploration_objects:
            object_class_id = total_classes.index(object)
            included_mask = np.isin(semseg, object_class_id)
            exploration_points = sem_points[included_mask]
            exploration_colors = sem_colors[included_mask]

            exploration_pcd = get_pcd(exploration_points, exploration_colors)
            exploration_pcd.transform(global2camera)

            exploration_pcds.append(exploration_pcd)

    print(f"[*] Loading the Map by pre-built files in {time.time() - t:.2f}s")
    return exploration_pcds


root_dir = '/home/ipbhk/OpenFusion/results/hospital_gazebo_hospital'
sem_points = np.load(root_dir + '/sem_points.npy')
sem_colors = np.load(root_dir + '/sem_colors.npy')
semseg = np.load(root_dir + '/semseg.npy')

total_classes = [
    "ceiling",
    "floor",
    "walls",
    "nurses station",
    "door",
    "chair",
    "trolley bed",
    "table",
    "sofa",
    "medical machine",
    "tv",
    "kitchen cabinet",
    "refrigerator",
    "toilet",
    "sink",
    "trash",
    "warehouse clusters",
    "others"
]

exploration_objects = [
    "chair",
]

camera2realcamera = np.array(
            [[0.0, 0.0, 1.0, 0.0],
             [-1.0, 0.0, 0.0, 0.0],
             [0.0, -1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]])

pose = np.loadtxt(f'/home/ipbhk/OpenFusion/sample/hospital/gazebo_hospital/poses.txt')[0]
camera_origin = Pose()
camera_origin.position.x, camera_origin.position.y, camera_origin.position.z = pose[:3]
camera_origin.orientation.x, camera_origin.orientation.y, camera_origin.orientation.z, camera_origin.orientation.w = pose[
                                                                                                                     3:]
global2camera = pose_to_transformation_matrix(camera_origin) @ camera2realcamera

exploration_pcds = get_semantic(sem_points, sem_colors, semseg,
                                total_classes, exploration_objects, global2camera)



print("[*] Starting semantic-based completion...")

for object_idx in range(len(exploration_pcds)):
    print(f"Collecting more views of [{exploration_objects[object_idx]}]......")
    objects_pcd = exploration_pcds[object_idx]
    print(objects_pcd)

    labels = np.asarray(objects_pcd.cluster_dbscan(eps=0.4, min_points=1000, print_progress=True))
    if labels.size == 0:
        max_label = -1
    else:
        max_label = labels.max()
    print(f"point cloud has [{max_label + 1}] clusters")
    cluster_sizes = np.bincount(labels[labels >= 0])
    large_clusters = [i for i, size in enumerate(cluster_sizes) if size > 50]
    # Sort these clusters based on their sizes and select the top two
    # large_clusters = sorted(large_clusters,
    #                                key=lambda x: cluster_sizes[x], reverse=True)[:2]
    filtered_labels = np.array([label if label in large_clusters else -1 for label in labels])

    for label in np.unique(filtered_labels):
        if label < 0:
            continue  # Ignore noise points with label
        instance_pcd: o3d.geometry.PointCloud = \
            objects_pcd.select_by_index(np.where(labels == label)[0])
        print("Current groups is ", instance_pcd)
        print("================")

    # visualization
    colors = plt.get_cmap("tab20")(filtered_labels / (max_label if max_label > 0 else 1))
    objects_pcd: o3d.geometry.PointCloud = \
        objects_pcd.select_by_index(np.where(filtered_labels >= 0)[0])
    colors = colors[np.where(filtered_labels >= 0)[0]]

    objects_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([objects_pcd])

