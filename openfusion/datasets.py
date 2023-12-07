import glob
import numpy as np
import os
from io import StringIO
from scipy.spatial.transform import Rotation as Rot
from openfusion.utils import preprocess_extrinsics, kobuki_pose2rgbd, custom_intrinsic


def pose_to_transformation_matrix(pose):
    # Translation vector
    t = np.array(pose[:3])

    # Rotation quaternion
    q = np.array(pose[3:])
    r = Rot.from_quat(q)

    # 3x3 rotation matrix
    rot_matrix = r.as_matrix()

    rot_ro_cam = np.eye(3)
    rot_ro_cam[1, 1] = -1
    rot_ro_cam[2, 2] = -1
    rot_matrix = rot_matrix @ rot_ro_cam

    # 4x4 transformation matrix
    trans_matrix = np.eye(4)
    trans_matrix[:3, :3] = rot_matrix
    trans_matrix[:3, 3] = t

    return trans_matrix


class Dataset(object):
    def __init__(self, data_path, max_frames, stream=False) -> None:
        self.data_path = data_path
        self.current = 0
        self.max_frames = max_frames
        if not stream:
            self.rgbs_list = self.load_color()
            self.depths_list = self.load_depth()
            self.pose_list = self.load_pose()
            if max_frames == -1:
                self.max_frames = len(self.rgbs_list)

    def __len__(self):
        return self.max_frames

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.max_frames:
            rgb = self.rgbs_list[self.current]
            depth = self.depths_list[self.current]
            pose = self.pose_list[self.current]
            self.current += 1
            return rgb, depth, pose
        raise StopIteration

    def scenes(self):
        return []

    def load_intrinsics(self, intrinsic_path, img_size, input_size):
        intrinsic = np.loadtxt(intrinsic_path)[:3, :3]
        return custom_intrinsic(intrinsic, *img_size, *input_size)

    def load_pose(self):
        pass

    def load_color(self):
        rgbs_list = sorted(
            glob.glob(self.data_path + '/rgb/*.jpg'),
            key=lambda p: int(p.split("/")[-1].rstrip('.jpg'))
        )
        if len(rgbs_list) == 0:
            rgbs_list = sorted(
                glob.glob(self.data_path + '/rgb/*.png'),
                key=lambda p: int(p.split("/")[-1].rstrip('.png'))
            )
        return rgbs_list

    def load_depth(self):
        depths_list = sorted(
            glob.glob(self.data_path + '/depth/*.jpg'),
            key=lambda p: int(p.split("/")[-1].rstrip('.jpg'))
        )
        if len(depths_list) == 0:
            depths_list = sorted(
                glob.glob(self.data_path + '/depth/*.png'),
                key=lambda p: int(p.split("/")[-1].rstrip('.png'))
            )
        return depths_list


class ICL(Dataset):
    def scenes(self):
        return ["kt0", "kt1", "kt2", "kt3"]

    def load_intrinsics(self, img_size, input_size):
        return super().load_intrinsics(self.data_path + "/../intrinsics.txt", img_size, input_size)

    def load_pose(self):
        with open(self.data_path + "/livingRoom.gt.sim", "r") as f:
            lines = f.readlines()

        pose_arr = []
        for line in lines:
            line = line.strip().split()
            if len(line) == 0:
                continue
            pose_arr.append(
                np.asarray(
                    [float(line[0]), float(line[1]), float(line[2]), float(line[3])]
                )
            )
        pose_arr = np.stack(pose_arr)

        extrinsics = []
        for pose_line_idx in range(0, pose_arr.shape[0], 3):
            curpose = np.zeros((4, 4))
            curpose[3, 3] = 1
            curpose[0] = pose_arr[pose_line_idx]
            curpose[1] = pose_arr[pose_line_idx + 1]
            curpose[2] = pose_arr[pose_line_idx + 2]
            extrinsics.append(curpose)

        return [np.linalg.inv(e.astype(np.float64)) for e in
                preprocess_extrinsics(extrinsics)]


class Replica(Dataset):
    def scenes(self):
        return ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]

    def load_intrinsics(self, img_size, input_size):
        intrinsic = np.zeros((3, 3))
        intrinsic[0, 0] = 600.0
        intrinsic[1, 1] = 600.0
        intrinsic[0, 2] = 599.5
        intrinsic[1, 2] = 339.5
        return custom_intrinsic(intrinsic, *img_size, *input_size)

    def load_pose(self):
        with open(self.data_path + "/traj.txt", "r") as f:
            lines = f.readlines()

        extrinsics = []
        for line in lines:
            c = StringIO(line)
            curpose = np.loadtxt(c).reshape(4, 4)
            extrinsics.append(curpose)

        return [np.linalg.inv(e.astype(np.float64)) for e in
                preprocess_extrinsics(extrinsics)]


class ScanNet(Dataset):
    def load_intrinsics(self, img_size, input_size):
        return super().load_intrinsics(self.data_path + "/intrinsic/intrinsic_depth.txt", img_size, input_size)

    def load_pose(self):
        poses = sorted(
            glob.glob(self.data_path + "/pose/*.txt"),
            key=lambda x: int(os.path.basename(x).split(".")[0]),
        )
        poses = [np.loadtxt(p) for p in poses]
        return [np.linalg.inv(p.astype(np.float64)) for p in poses]


class Kobuki(Dataset):
    def load_intrinsics(self, img_size, input_size):
        return super().load_intrinsics(self.data_path + "/../intrinsics.txt", img_size, input_size)

    def load_pose(self):
        npz_lis = sorted(glob.glob(self.data_path + '/all/*.npz'), key=lambda p: int(p.split("/")[-1].rstrip('.npz')))
        pose2rgbd_matrix = kobuki_pose2rgbd()
        extrinsics = []
        for file in npz_lis:
            array = np.load(file)
            pose_quat = array["pose"]
            r = Rot.from_quat(pose_quat[3:]).as_matrix()
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = r
            pose_matrix[:3, 3] = pose_quat[:3]
            extrinsics.append(pose_matrix)
        return [np.linalg.inv(e.astype(np.float64) @ pose2rgbd_matrix) for e in
                preprocess_extrinsics(extrinsics)]


class Live(Dataset):
    def load_intrinsics(self, img_size, input_size):
        return super().load_intrinsics(self.data_path + "/../intrinsics.txt", img_size, input_size)

    def load_pose(self):
        pose_list = sorted(glob.glob(self.data_path + '/pose/*.txt'),
                           key=lambda p: int(p.split("/")[-1].rstrip('.txt')))
        extrinsics = []
        for file in pose_list:
            pose_matrix = np.loadtxt(file)
            extrinsics.append(pose_matrix)
        return extrinsics


class HabitatSim(Dataset):
    def scenes(self):
        return ["test"]

    def load_intrinsics(self, img_size, input_size):
        return super().load_intrinsics(self.data_path + "/intrinsics.txt", img_size, input_size)

    def load_pose(self):
        pose_arr = np.loadtxt(os.path.join(self.data_path, 'poses.txt')).tolist()

        extrinsics = []
        for pose in pose_arr:
            curpose = pose_to_transformation_matrix(pose)
            extrinsics.append(curpose)

        return [np.linalg.inv(e.astype(np.float64)) for e in
                preprocess_extrinsics(extrinsics)]

    def load_depth(self):
        depths_list = sorted(
            glob.glob(self.data_path + '/depth/*.npy'),
            key=lambda p: int(p.split("/")[-1].rstrip('.npy'))
        )
        return depths_list


class Dingo(Dataset):
    def scenes(self):
        return ["dingo_hospital", "dingo_hospital_large", "dingo_hospital_small",
                "dingo_house"]

    def load_intrinsics(self, img_size, input_size):
        return super().load_intrinsics(self.data_path + "/intrinsics.txt", img_size, input_size)

    def load_pose(self):
        pose_arr = np.loadtxt(os.path.join(self.data_path, 'poses.txt')).tolist()

        camera2realcamera = np.array([[1.0, 0.0, 0.0, 0.0],
                                      [0.0, -1.0, 0.0, 0.0],
                                      [0.0, 0.0, -1.0, 0.0],
                                      [0.0, 0.0, 0.0, 1.0]])
        extrinsics = []
        for pose in pose_arr:
            curpose = pose_to_transformation_matrix(pose)
            extrinsics.append(curpose @ camera2realcamera)

        return [np.linalg.inv(e.astype(np.float64)) for e in
                preprocess_extrinsics(extrinsics)]

    def load_depth(self):
        depths_list = sorted(
            glob.glob(self.data_path + '/depth/*.npy'),
            key=lambda p: int(p.split("/")[-1].rstrip('.npy'))
        )
        return depths_list
