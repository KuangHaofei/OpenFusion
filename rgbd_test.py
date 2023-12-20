import os

import cv2
import numpy as np
import open3d as o3d

TRIANGLE_MESH = o3d.geometry.TriangleMesh()
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from openfusion.datasets import Dataset, Robot


def get_pc(rgb, depth, pose, cam_mat):
    # generate pointclouds from RGBD images via open3d
    h, w = depth.shape
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, cam_mat)

    rgb = np.asarray(rgb, order='C')
    color = o3d.geometry.Image((rgb).astype(np.uint8))
    depth = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=1, depth_trunc=10.0, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    pcd = pcd.transform(pose)

    return pcd


def depth_image(depth_data, id):
    depth_data = np.nan_to_num(depth_data, copy=False, nan=10, posinf=0.0, neginf=0.0)
    grayscale_image = depth_data

    # Display the image
    # Save the image
    save_path = f'./depth_{id}.png'
    plt.imsave(save_path, grayscale_image, cmap='gray')
    plt.imshow(grayscale_image, cmap='gray')
    plt.show()


# data_path = '/home/ipbhk/OpenFusion/sample/hospital/gazebo_hospital'
data_path = '/home/ipbhk/OpenFusion/sample/hospital/gazebo_hospital/collect_data'
# data_path = '/home/ipbhk/OpenFusion/sample/house/gazebo_house'
# data_path = '/home/ipbhk/OpenFusion/sample/house/gazebo_house/collect_data'
# data_path = '/home/ipbhk/OpenFusion/sample/office/ipblab'
max_frames = 200
img_size = (640, 480),
input_size = (640, 480),
dataset = Robot(data_path, max_frames)
intrinsic = np.loadtxt(os.path.join(data_path, 'intrinsics.txt'))

all_pcds = o3d.geometry.PointCloud()
camera_poses = TRIANGLE_MESH

# ids = [0, 30, 90]
# for idx in ids:
#     print(idx)
#     rgb_path, depth_path, pose = dataset[idx]
for rgb_path, depth_path, pose in tqdm(dataset):
    rgb = np.array(Image.open(rgb_path))
    depth = np.load(depth_path)

    # save rgb and depth as images
    # depth_image(depth, idx)

    pose = np.linalg.inv(pose)

    pcd = get_pc(rgb, depth, pose, intrinsic)
    all_pcds += pcd

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    mesh.transform(pose)
    camera_poses += mesh

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
o3d.visualization.draw_geometries([all_pcds, camera_poses])
