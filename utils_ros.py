import numpy as np
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d

from geometry_msgs.msg import Pose, PoseArray


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


def generate_viewpoints(center, radius, interval_degrees=30):
    viewpoints = []
    for angle in range(0, 360, interval_degrees):
        radians = np.deg2rad(angle)
        # Position on the circle
        x = center[0] + radius * np.cos(radians)
        y = center[1] + radius * np.sin(radians)
        position = (x, y)
        # Direction towards the center
        yaw = np.arctan2(center[1] - y, center[0] - x)
        viewpoints.append([position[0], position[1], yaw])

    viewpoints = np.array(viewpoints)

    return viewpoints


def pose2d_to_posemsg(pose):
    # pose: [x, y, yaw]
    pose_msg = Pose()
    pose_msg.position.x = pose[0]
    pose_msg.position.y = pose[1]
    pose_msg.position.z = 0
    q = Rot.from_euler('z', pose[2]).as_quat()
    pose_msg.orientation.x = q[0]
    pose_msg.orientation.y = q[1]
    pose_msg.orientation.z = q[2]
    pose_msg.orientation.w = q[3]
    return pose_msg


def get_covered_viewpoints(pcd: o3d.geometry.PointCloud, time_stamp, frame_id='map'):
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_center = bbox.get_center()[:2]
    bbox_points = np.asarray(bbox.get_box_points())
    radius = 0.0
    for point in bbox_points:
        if np.linalg.norm(point[:2] - bbox_center) > radius:
            radius = np.linalg.norm(point[:2] - bbox_center)
    radius += 0.5
    viewpoints = generate_viewpoints(bbox_center, radius, 15)

    # covert viewpoint to Pose array
    nvs = PoseArray()
    nvs.header.stamp = time_stamp
    nvs.header.frame_id = frame_id
    for viewpoint in viewpoints:
        nv = pose2d_to_posemsg(viewpoint)
        nvs.poses.append(nv)

    return nvs
