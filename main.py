import argparse
import os
import time
import numpy as np
from tqdm import tqdm
import open3d as o3d
from openfusion.slam import build_slam, BaseSLAM
from openfusion.datasets import Dataset
from openfusion.utils import (
    show_pc, save_pc, get_cmap_legend
)
from configs.build import get_config

def stream_loop(args, slam: BaseSLAM):
    if args.save:
        slam.export_path = f"{args.data}_live/{args.algo}.npz"

    slam.start_thread()
    if args.live:
        slam.start_monitor_thread()
        slam.start_query_thread()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        slam.stop_thread()
        if args.live:
            slam.stop_query_thread()
            slam.stop_monitor_thread()


def dataset_loop(args, slam: BaseSLAM, dataset: Dataset):
    if args.save:
        slam.export_path = f"results/{args.data}_{args.scene}_{args.algo}.npz"

    if args.live:
        slam.start_monitor_thread()
        slam.start_query_thread()
    i = 0
    for rgb_path, depth_path, extrinsics in tqdm(dataset):
        rgb, depth = slam.io.from_file(
            rgb_path, depth_path, depth_format='npy', depth_scale=slam.point_state.depth_scale)
        slam.io.update(rgb, depth, extrinsics)
        slam.vo()
        slam.compute_state(encode_image=i % 10 == 0)
        i += 1
    if args.live:
        slam.stop_query_thread()
        slam.stop_monitor_thread()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default="vlfusion", choices=["default", "cfusion", "vlfusion"])
    parser.add_argument('--vl', type=str, default="seem", help="vlfm to use")
    parser.add_argument('--data', type=str, default="kobuki", help='Path to dir of dataset.')
    parser.add_argument('--scene', type=str, default="icra", help='Name of the scene in the dataset.')
    parser.add_argument('--frames', type=int, default=-1, help='Total number of frames to use. If -1, use all frames.')
    parser.add_argument('--device', type=str, default="cuda:0", choices=["cpu:0", "cuda:0"])
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--model_completion', action='store_true')
    parser.add_argument('--host_ip', type=str, default="YOUR IP")  # for stream
    args = parser.parse_args()

    params = get_config(args.data, args.scene)
    # dataset: Dataset = params["dataset"](params["path"], args.frames, args.stream)
    dataset: Dataset = params["dataset"](params["path"], -1, args.stream)
    if args.model_completion:
        dataset_completion: Dataset = params["dataset"](params["path"], args.frames, args.stream, args.model_completion)
        # dataset_completion.current = 290
    intrinsic = dataset.load_intrinsics(params["img_size"], params["input_size"])
    slam = build_slam(args, intrinsic, params)

    if args.stream:
        args.scene = "live"

    # NOTE: real-time semantic map construction
    results_path = f"results/{args.data}_{args.scene}"
    save_path = results_path
    if args.model_completion:
        save_path = results_path + "/completion"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if args.load:
        if os.path.exists(f"{results_path}/{args.algo}.npz"):
            print("[*] loading saved state...")
            slam.point_state.load(f"{results_path}/{args.algo}.npz")
        else:
            print("[*] no saved state found, skipping...")

    if args.stream:
        stream_loop(args, slam)
    else:
        dataset_loop(args, slam, dataset)
        if args.model_completion:
            dataset_loop(args, slam, dataset_completion)
        if args.save:
            slam.save(f"{save_path}/{args.algo}.npz")
            print(f"[*] saved state to {f'{save_path}/{args.algo}.npz'}")

    # NOTE: save point cloud
    rgb_points, rgb_colors = slam.point_state.get_pc()
    np.save(f"{save_path}/rgb_points.npy", rgb_points)
    np.save(f"{save_path}/rgb_colors.npy", rgb_colors)
    print(f"[*] saved point cloud to {f'{save_path}/rgb_points.npy'} and {f'{save_path}/rgb_colors.npy'}")
    show_pc(rgb_points, rgb_colors)
    # save_pc(rgb_points, rgb_colors, f"{save_path}/color_pc.ply")
    # print(f"[*] saved color point cloud to {f'{save_path}/color_pc.ply'}")

    # NOTE: save colorized mesh
    # mesh = slam.point_state.get_mesh()
    # o3d.io.write_triangle_mesh(f"{save_path}/color_mesh.ply", mesh)
    # o3d.io.write_triangle_mesh(f"{save_path}/color_mesh.glb", mesh)
    # print(f"[*] saved color mesh to {f'{save_path}/color_mesh.ply'}")
    # print(f"[*] saved color mesh to {f'{save_path}/color_mesh.glb'}")

    # NOTE: modify below to play with query
    if args.algo in ["cfusion", "vlfusion"]:
        sem_points, sem_colors, semseg = slam.semantic_query(
            params['objects'], rgb_points, rgb_colors, save_file=f"{save_path}/cmap.png")

        # save points, colors, and semseg to npy file
        np.save(f"{save_path}/sem_points.npy", sem_points)
        np.save(f"{save_path}/sem_colors.npy", sem_colors)
        np.save(f"{save_path}/semseg.npy", semseg)
        print(f"[*] saved semantic point cloud to "
              f"{f'{save_path}/sem_points.npy'}, "
              f"{save_path}/sem_colors.npy and {save_path}/semseg.npy")
        show_pc(sem_points, sem_colors)
        # save_pc(sem_points, sem_colors, f"{save_path}/semantic_pc.ply")
        # print(f"[*] saved semantic point cloud to {f'{save_path}/semantic_pc.ply'}")


if __name__ == "__main__":
    main()
