import argparse
import os

import numpy as np

from openfusion.slam import build_slam
from configs.build import get_config
from openfusion.utils import custom_intrinsic
from openfusion.utils import show_pc, save_pc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default="vlfusion", choices=["default", "cfusion", "vlfusion"])
    parser.add_argument('--vl', type=str, default="seem", help="vlfm to use")
    parser.add_argument('--data', type=str, default="dingo", help='Path to dir of dataset.')
    parser.add_argument('--scene', type=str, default="dingo_gazebo", help='Name of the scene in the dataset.')
    parser.add_argument('--device', type=str, default="cuda:0", choices=["cpu:0", "cuda:0"])
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--stream', action='store_true')
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
if os.path.exists(f"results/{args.data}_{args.scene}/{args.algo}.npz"):
    print("[*] loading saved state...")
    slam.point_state.load(f"results/{args.data}_{args.scene}/{args.algo}.npz")
else:
    print("[*] no saved state found, skipping...")

# visual-language querying
if args.algo in ["cfusion", "vlfusion"]:
    # points, colors = slam.semantic_query(params['objects'], save_file=f"results/{args.data}_{args.scene}/cmap.png")
    points, colors = slam.semantic_query(
        ['floor', 'wall', 'ceil', 'sofa', 'bed', 'table', 'cabinet',
         'home appliances', 'chair', 'ball', 'trash', 'tv', 'others'],
        save_file=f"results/{args.data}_{args.scene}/cmap.png")
    show_pc(points, colors, slam.point_state.poses)
