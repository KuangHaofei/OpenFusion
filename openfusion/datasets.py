import glob
import numpy as np
import os
from io import StringIO
from scipy.spatial.transform import Rotation as Rot
from openfusion.utils import preprocess_extrinsics, kobuki_pose2rgbd, custom_intrinsic


def pose_to_transformation_matrix_habitat(pose):
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


def pose_to_transformation_matrix(pose):
    # Translation vector
    t = np.array(pose[:3])

    # Rotation quaternion
    q = np.array(pose[3:])
    r = Rot.from_quat(q)

    # 3x3 rotation matrix
    rot_matrix = r.as_matrix()

    # 4x4 transformation matrix
    trans_matrix = np.eye(4)
    trans_matrix[:3, :3] = rot_matrix
    trans_matrix[:3, 3] = t

    return trans_matrix


class Dataset(object):
    def __init__(self, data_path, max_frames, stream=False, model_completion=False) -> None:
        self.data_path = data_path
        self.current = 0
        self.max_frames = max_frames

        self.model_completion = model_completion
        if self.model_completion:
            self.data_path = os.path.join(self.data_path, "collect_data")

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

    def __getitem__(self, current):
        if current < self.max_frames:
            rgb = self.rgbs_list[current]
            depth = self.depths_list[current]
            pose = self.pose_list[current]
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


class HabitatSim(Dataset):
    def scenes(self):
        return ["test"]

    def load_intrinsics(self, img_size, input_size):
        return super().load_intrinsics(self.data_path + "/intrinsics.txt", img_size, input_size)

    def load_pose(self):
        pose_arr = np.loadtxt(os.path.join(self.data_path, 'poses.txt')).tolist()

        extrinsics = []
        for pose in pose_arr:
            curpose = pose_to_transformation_matrix_habitat(pose)
            extrinsics.append(curpose)

        return [np.linalg.inv(e.astype(np.float64)) for e in
                preprocess_extrinsics(extrinsics)]

    def load_depth(self):
        depths_list = sorted(
            glob.glob(self.data_path + '/depth/*.npy'),
            key=lambda p: int(p.split("/")[-1].rstrip('.npy'))
        )
        return depths_list


class Robot(Dataset):
    def scenes(self):
        return ["gazebo_hospital", "gazebo_house",
                "ipblab1", "ipblab2"]

    def load_intrinsics(self, img_size, input_size):
        return super().load_intrinsics(self.data_path + "/intrinsics.txt", img_size, input_size)

    def load_pose(self):
        pose_arr = np.loadtxt(os.path.join(self.data_path, 'poses.txt')).tolist()

        camera2realcamera = np.array([[0.0, 0.0, 1.0, 0.0],
                                      [-1.0, 0.0, 0.0, 0.0],
                                      [0.0, -1.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 1.0]])

        extrinsics = []
        for pose in pose_arr:
            curpose = pose_to_transformation_matrix(pose)
            extrinsics.append(curpose @ camera2realcamera)

        if self.model_completion:
            global2camera = np.loadtxt('/'.join(self.data_path.split("/")[:-1]) +'/poses.txt').tolist()[0]
            global2camera = pose_to_transformation_matrix(global2camera) @ camera2realcamera

            return [np.linalg.inv(e.astype(np.float64)) for e in
                    preprocess_extrinsics(extrinsics, global2camera=global2camera)]

        return [np.linalg.inv(e.astype(np.float64)) for e in
                preprocess_extrinsics(extrinsics)]

    def load_depth(self):
        depths_list = sorted(
            glob.glob(self.data_path + '/depth/*.npy'),
            key=lambda p: int(p.split("/")[-1].rstrip('.npy'))
        )
        return depths_list
