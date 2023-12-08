import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt

pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud("ros_test.ply")
o3d.visualization.draw_geometries([pcd])

eps = 0.2 # Spatial distance threshold for defining neighbors
min_points = 50  # Minimum number of points to form a cluster

labels = pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
labels = np.asarray(labels)
print(np.unique(labels))
print("Generate {:f} groups".format(np.unique(labels).size - 1))

# Extract sub-point clouds based on cluster indices
max_label = labels.max()
colors = plt.get_cmap("viridis")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # Ignore noise points with label -1

# Create a visualization object
vis = o3d.visualization.Visualizer()
sub_pointclouds = []
sub_pcds = o3d.geometry.PointCloud()

for label in np.unique(labels):
    # if label < 0:
    #     continue  # Ignore noise points with label 0
    sub_pcd = pcd.select_by_index(np.where(labels == label)[0])
    sub_pcd.colors = o3d.utility.Vector3dVector(colors[np.where(labels == label)[0], :3])
    sub_pcds += sub_pcd
    sub_pointclouds.append(sub_pcd)

print(sub_pointclouds)
o3d.visualization.draw_geometries([sub_pcds])
