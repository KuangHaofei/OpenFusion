import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d

from openfusion.slam import build_slam
from configs.build import get_config
from openfusion.utils import custom_intrinsic, get_pcd
from openfusion.utils import show_pc, save_pc

np.random.seed(42)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default="vlfusion", choices=["default", "cfusion", "vlfusion"])
    parser.add_argument('--vl', type=str, default="seem", help="vlfm to use")
    parser.add_argument('--data', type=str, default="dingo_house", help='Path to dir of dataset.')
    parser.add_argument('--scene', type=str, default="dingo_house", help='Name of the scene in the dataset.')
    parser.add_argument('--device', type=str, default="cuda:0", choices=["cpu:0", "cuda:0"])
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--load_completion', action='store_true')
    parser.add_argument('--host_ip', type=str, default="YOUR IP")  # for stream
    args = parser.parse_args()
    return args


args = get_args()
params = get_config(args.data, args.scene)
intrinsic_path = params['path'] + "/intrinsics.txt"
intrinsic = np.loadtxt(intrinsic_path)[:3, :3]
intrinsic = custom_intrinsic(intrinsic, *params["img_size"], *params["input_size"])
slam = build_slam(args, intrinsic, params)

# loading npz
results_root = f"results/{args.data}_{args.scene}"
if args.load_completion:
    results_root += "/completion"

if os.path.exists(f"{results_root}/{args.algo}.npz"):
    print("[*] loading saved state...")
    slam.point_state.load(f"{results_root}/{args.algo}.npz")
else:
    print("[*] no saved state found, skipping...")

rgb_points, rgb_colors = slam.point_state.get_pc()
np.save(f"{results_root}/rgb_points.npy", rgb_points)
np.save(f"{results_root}/rgb_colors.npy", rgb_colors)

total_classes = [
    'ceiling',
    'floor',
    'wall',
    'sink',
    'door',
    'oven',
    'garbage can',
    'whiteboard',
    'table',
    'desk',
    'sofa',
    'chair',
    'bookshelf',
    'cabinet',
    'extinguisher',
    'people',
    'others'
]

# visual-language querying
if args.algo in ["cfusion", "vlfusion"]:
    print("[*] semantic querying...")
    sem_points, sem_colors, semseg = slam.semantic_query(
        total_classes, rgb_points, rgb_colors, save_file=f"{results_root}/cmap.png")

    # save points, colors, and semseg to npy file
    np.save(f"{results_root}/sem_points.npy", sem_points)
    np.save(f"{results_root}/sem_colors.npy", sem_colors)
    np.save(f"{results_root}/semseg.npy", semseg)

    # show pcds
    excluded_classes = []
    excluded_classes.append(total_classes.index('ceiling'))
    excluded_classes.append(total_classes.index('floor'))
    excluded_classes.append(total_classes.index('people'))
    excluded_classes.append(total_classes.index('others'))
    excluded_mask = ~np.isin(semseg, excluded_classes)
    rgb_points = rgb_points[excluded_mask]
    rgb_colors = rgb_colors[excluded_mask]
    sem_points = sem_points[excluded_mask]
    sem_colors = sem_colors[excluded_mask]
    semseg = semseg[excluded_mask]

    show_pc(rgb_points, rgb_colors)
    show_pc(sem_points, sem_colors)

    object_class_id = total_classes.index('sofa')
    included_mask = np.isin(semseg, object_class_id)
    points = sem_points[included_mask]
    colors = sem_colors[included_mask]
    show_pc(points, colors)
    save_pc(points, colors, "test.ply")

    objects_pcd = get_pcd(points, colors)
    labels = np.array(objects_pcd.cluster_dbscan(eps=0.2, min_points=100, print_progress=True))
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    objects_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    objects_pcd = objects_pcd.select_by_index(np.where(labels >= 0)[0])
    o3d.visualization.draw_geometries([objects_pcd])
    print(f"point cloud has {max_label + 1} clusters")
