import numpy as np
from scipy.spatial.transform import Rotation as Rot


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