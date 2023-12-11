import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d

from openfusion.slam import build_slam
from configs.build import get_config
from openfusion.utils import custom_intrinsic, get_pcd
from openfusion.utils import show_pc, save_pc


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
if args.load_completion:
    if os.path.exists(f"results/{args.data}_{args.scene}/{args.algo}_completion.npz"):
        print("[*] loading saved state...")
        slam.point_state.load(f"results/{args.data}_{args.scene}/{args.algo}.npz")
    else:
        print("[*] no saved state found, skipping...")
else:
    if os.path.exists(f"results/{args.data}_{args.scene}/{args.algo}.npz"):
        print("[*] loading saved state...")
        slam.point_state.load(f"results/{args.data}_{args.scene}/{args.algo}.npz")
    else:
        print("[*] no saved state found, skipping...")

total_classes = [
    'floor', 'wall', 'ceil', 'sofa', 'bed', 'table', 'cabinet',
    'home appliances', 'chair', 'ball', 'trash', 'tv', 'others'
]

# visual-language querying
if args.algo in ["cfusion", "vlfusion"]:
    # points, colors = slam.semantic_query(params['objects'], save_file=f"results/{args.data}_{args.scene}/cmap.png")
    points, colors, semseg = slam.semantic_query(
        total_classes,
        save_file=f"results/{args.data}_{args.scene}/cmap.png")
    show_pc(points, colors, slam.point_state.poses)
    object_class_id = total_classes.index('sofa')
    included_mask = np.isin(semseg, object_class_id)
    points = points[included_mask]
    colors = colors[included_mask]
    show_pc(points, colors, slam.point_state.poses)
    save_pc(points, colors, "test.ply")

    objects_pcd = get_pcd(points, colors)
    labels = np.array(objects_pcd.cluster_dbscan(eps=0.2, min_points=100, print_progress=True))
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    print(f"point cloud has {max_label + 1} clusters")

    labels = np.unique(labels)
    for label in labels:
        if label < 0:
            continue
        instance_pcd: o3d.geometry.PointCloud = objects_pcd.select_by_index(labels == label)
        points = np.array(instance_pcd.points)
        colors = np.array(instance_pcd.colors)
        colors[:] = np.array([255, 0, 0])
        instance_pcd.colors = o3d.utility.Vector3dVector(colors)

        show_pc(points, colors, slam.point_state.poses)
