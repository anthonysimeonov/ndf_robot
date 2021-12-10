import os, os.path as osp
import sys
from PIL import Image
import numpy as np
from yacs.config import CfgNode
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import math

class AttrDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__


# Estimate rigid transform with SVD (from Nghia Ho)
def register_corresponding_points(A, B, return_error=True):
    assert len(A) == len(B)
    N = A.shape[0]; # Total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - np.tile(centroid_A, (N, 1)) # Centre the points
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.dot(np.transpose(AA), BB) # Dot is matrix multiplication for array
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0: # Special reflection case
        Vt[2,:] *= -1
        R = np.dot(Vt.T, U.T)
    t = np.dot(-R, centroid_A.T) + centroid_B.T
    # return R, t
    T = np.eye(4)
    T[:-1, :-1] = R
    T[:-1, -1] = t

    if return_error:
        registered_pts = transform_pcd(B, T)
        # error = np.transpose(registered_pts) - A
        error = registered_pts - A
        error = np.sum(np.multiply(error,error))
        rmse = np.sqrt(error/A.shape[0])
    else:
        rmse = None
    return T, rmse


def np2img(np_array, img_file):
    im = Image.fromarray(np_array)
    im.save(img_file)



def safe_makedirs(dirname):
    if not osp.exists(dirname):
        os.makedirs(dirname)


def signal_handler(sig, frame):
    """
    Capture exit signal from keyboard
    """
    print('Exit')
    sys.exit(0)


def cn2dict(config):
    """
    Convert a YACS CfgNode config object into a
    dictionary

    Args:
        config (CfgNode): Config object

    Returns:
        dict: Dictionary version of config
    """
    out_dict = {}
    items = config.items()
    for key, val in items:
        if isinstance(val, CfgNode):
            ret = cn2dict(val)
        else:
            ret = val
        out_dict[key] = ret
    return out_dict


def crop_pcd(raw_pts, x=[0.0, 0.7], y=[-0.4, 0.4], z=[0.9, 1.5]):
    npw = np.where(
            (raw_pts[:, 0] > min(x)) & (raw_pts[:, 0] < max(x)) &
            (raw_pts[:, 1] > min(y)) & (raw_pts[:, 1] < max(y)) &
            (raw_pts[:, 2] > min(z)) & (raw_pts[:, 2] < max(z)))
    return raw_pts[npw[0], :]


class Position:
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.z = 0.


class Orientation:
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.z = 0.
        self.w = 0.


class Pose:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation


class Header:
    def __init__(self):
        self.frame_id = "world"


class PoseStamped():
    def __init__(self):
        position = Position()
        orientation = Orientation()
        pose = Pose(position, orientation)
        header = Header()
        self.pose = pose
        self.header = header


def get_2d_pose(pose3d):
    #1. extract rotation about z-axis
    T = matrix_from_pose(pose3d)
    # euler_angles_list = tf.transformations.euler_from_matrix(T, 'rxyz')
    r = R.from_matrix(T[:3, :3])
    euler_angles_list = r.as_euler('XYZ')
    pose2d = np.array([pose3d.pose.position.x,
                       pose3d.pose.position.y,
                       euler_angles_list[2],
                       ])

    return pose2d


def C3_2d(theta):
    C = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]]
                 )

    return C


def C3(theta):
    C = np.array([[np.cos(theta), np.sin(theta), 0],
                  [-np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]]
                 )
    return C


def unwrap(angles, min_val=-np.pi, max_val=np.pi):
    if type(angles) is not 'ndarray':
        angles = np.array(angles)
    angles_unwrapped = []
    for counter in range(angles.shape[0]):
        angle = angles[counter]
        if angle < min_val:
            angle += 2 * np.pi
        if angle > max_val:
            angle -= 2 * np.pi
        angles_unwrapped.append(angle)
    return np.array(angles_unwrapped)


def angle_from_3d_vectors(u, v):
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    u_dot_v = np.dot(u, v)
    return np.arccos(u_dot_v) / (u_norm * v_norm)


def pose_from_matrix(matrix, frame_id="world"):
    # trans = tf.transformations.translation_from_matrix(matrix)
    # quat = tf.transformations.quaternion_from_matrix(matrix)
    quat = R.from_matrix(matrix[:3, :3]).as_quat()
    trans = matrix[:-1, -1]
    pose = list(trans) + list(quat)
    pose = list2pose_stamped(pose, frame_id=frame_id)
    return pose


def list2pose_stamped(pose, frame_id="world"):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = pose[0]
    msg.pose.position.y = pose[1]
    msg.pose.position.z = pose[2]
    msg.pose.orientation.x = pose[3]
    msg.pose.orientation.y = pose[4]
    msg.pose.orientation.z = pose[5]
    msg.pose.orientation.w = pose[6]
    return msg


def unit_pose():
    return list2pose_stamped([0, 0, 0, 0, 0, 0, 1])


def convert_reference_frame(pose_source, pose_frame_target, pose_frame_source, frame_id="yumi_body"):
    T_pose_source = matrix_from_pose(pose_source)
    pose_transform_target2source = get_transform(
        pose_frame_source, pose_frame_target)
    T_pose_transform_target2source = matrix_from_pose(
        pose_transform_target2source)
    T_pose_target = np.matmul(T_pose_transform_target2source, T_pose_source)
    pose_target = pose_from_matrix(T_pose_target, frame_id=frame_id)
    return pose_target


def convert_reference_frame_list(pose_source_list, pose_frame_target, pose_frame_source, frame_id="yumi_body"):
    pose_target_list = []
    for pose_source in pose_source_list:
        pose_target_list.append(convert_reference_frame(pose_source,
                                                        pose_frame_target,
                                                        pose_frame_source,
                                                        frame_id))
    return pose_target_list


def pose_stamped2list(msg):
    return [float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
            float(msg.pose.orientation.w),
            ]


def pose_stamped2np(msg):
    return np.asarray(pose_stamped2list(msg))


def get_transform(pose_frame_target, pose_frame_source):
    """
    Find transform that transforms pose source to pose target
    :param pose_frame_target:
    :param pose_frame_source:
    :return:
    """
    #both poses must be expressed in same reference frame
    T_target_world = matrix_from_pose(pose_frame_target)
    T_source_world = matrix_from_pose(pose_frame_source)
    T_relative_world = np.matmul(T_target_world, np.linalg.inv(T_source_world))
    pose_relative_world = pose_from_matrix(
        T_relative_world, frame_id=pose_frame_source.header.frame_id)
    return pose_relative_world


def matrix_from_pose(pose):
    pose_list = pose_stamped2list(pose)
    trans = pose_list[0:3]
    quat = pose_list[3:7]
    # T = tf.transformations.quaternion_matrix(quat)

    T = np.zeros((4, 4))
    T[-1, -1] = 1
    r = R.from_quat(quat)
    T[:3, :3] = r.as_matrix()
    # print("matrix from quat: ")
    # print(T)
    T[0:3, 3] = trans
    return T


def scale_matrix(factor, origin=None):
    """Return matrix to scale by factor around origin in direction.
    Use factor -1 for point symmetry.
    """
    if not isinstance(factor, list) and not isinstance(factor, np.ndarray):
        M = np.diag([factor, factor, factor, 1.0])
    else:
        assert len(factor) == 3, 'If applying different scaling per dimension, must pass in 3-element list or array'
        #M = np.diag([factor[0], factor[1], factor[2], 1.0])
        M = np.eye(4)
        M[0, 0] = factor[0]
        M[1, 1] = factor[1]
        M[2, 2] = factor[2]
    if origin is not None:
        M[:3, 3] = origin[:3]
        M[:3, 3] *= 1.0 - factor
    return M


def euler_from_pose(pose):
    T_transform = matrix_from_pose(pose)
    # euler = tf.transformations.euler_from_matrix(T_transform, 'rxyz')
    r = R.from_matrix(T_transform[:3, :3])
    euler = r.as_euler('XYZ')
    return euler


def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> np.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True

    """
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


def mat2quat(orient_mat_3x3):
    orient_mat_4x4 = [[orient_mat_3x3[0][0], orient_mat_3x3[0][1], orient_mat_3x3[0][2], 0],
                      [orient_mat_3x3[1][0], orient_mat_3x3[1]
                          [1], orient_mat_3x3[1][2], 0],
                      [orient_mat_3x3[2][0], orient_mat_3x3[2]
                          [1], orient_mat_3x3[2][2], 0],
                      [0, 0, 0, 1]]

    orient_mat_4x4 = np.array(orient_mat_4x4)
    quat = quaternion_from_matrix(orient_mat_4x4)
    return quat


def interpolate_pose(pose_initial, pose_final, N, frac=1):
    frame_id = pose_initial.header.frame_id
    pose_initial_list = pose_stamped2list(pose_initial)
    pose_final_list = pose_stamped2list(pose_final)
    trans_initial = pose_initial_list[0:3]
    quat_initial = pose_initial_list[3:7]
     # onvert to pyquaterion convertion (w,x,y,z)
    trans_final = pose_final_list[0:3]
    quat_final = pose_final_list[3:7]

    trans_interp_total = [np.linspace(trans_initial[0], trans_final[0], num=N),
                          np.linspace(trans_initial[1], trans_final[1], num=N),
                          np.linspace(trans_initial[2], trans_final[2], num=N)]
    
    key_rots = R.from_quat([quat_initial, quat_final])
    slerp = Slerp(np.arange(2), key_rots)
    interp_rots = slerp(np.linspace(0, 1, N))
    quat_interp_total = interp_rots.as_quat()    

    pose_interp = []
    for counter in range(int(frac * N)):
        pose_tmp = [
            trans_interp_total[0][counter],
            trans_interp_total[1][counter],
            trans_interp_total[2][counter],
            quat_interp_total[counter][0], #return in ROS ordering w,x,y,z
            quat_interp_total[counter][1],
            quat_interp_total[counter][2],
            quat_interp_total[counter][3],
        ]
        pose_interp.append(list2pose_stamped(pose_tmp, frame_id=frame_id))
    return pose_interp


def transform_pose(pose_source, pose_transform):
    T_pose_source = matrix_from_pose(pose_source)
    T_transform_source = matrix_from_pose(pose_transform)
    T_pose_final_source = np.matmul(T_transform_source, T_pose_source)
    pose_final_source = pose_from_matrix(
        T_pose_final_source, frame_id=pose_source.header.frame_id)
    return pose_final_source


def transform_body(pose_source_world, pose_transform_target_body):
    #convert source to target frame
    pose_source_body = convert_reference_frame(pose_source_world,
                                               pose_source_world,
                                               unit_pose(),
                                               frame_id="body_frame")
    #perform transformation in body frame
    pose_source_rotated_body = transform_pose(pose_source_body,
                                              pose_transform_target_body)
    # rotate back
    pose_source_rotated_world = convert_reference_frame(pose_source_rotated_body,
                                                        unit_pose(),
                                                        pose_source_world,
                                                        frame_id="yumi_body")
    return pose_source_rotated_world


def vec_from_pose(pose):
    #get unit vectors of rotation from pose
    quat = pose.pose.orientation
    # T = tf.transformations.quaternion_matrix([quat.x, quat.y, quat.z, quat.w])
    T = np.zeros((4, 4,))
    T[-1, -1] = 1
    T[:3, :3] = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()

    x_vec = T[0:3, 0]
    y_vec = T[0:3, 1]
    z_vec = T[0:3, 2]
    return x_vec, y_vec, z_vec


def list_to_pose(pose_list):
    msg = Pose()
    msg.position.x = pose_list[0]
    msg.position.y = pose_list[1]
    msg.position.z = pose_list[2]
    msg.orientation.x = pose_list[3]
    msg.orientation.y = pose_list[4]
    msg.orientation.z = pose_list[5]
    msg.orientation.w = pose_list[6]
    return msg


def pose_to_list(pose):
    pose_list = []
    pose_list.append(pose.position.x)
    pose_list.append(pose.position.y)
    pose_list.append(pose.position.z)
    pose_list.append(pose.orientation.x)
    pose_list.append(pose.orientation.y)
    pose_list.append(pose.orientation.z)
    pose_list.append(pose.orientation.w)
    return pose_list


def quat_multiply(quat1, quat2):
    """
    Quaternion mulitplication.

    Args:
        quat1 (list or np.ndarray): first quaternion [x,y,z,w]
            (shape: :math:`[4,]`).
        quat2 (list or np.ndarray): second quaternion [x,y,z,w]
            (shape: :math:`[4,]`).

    Returns:
        np.ndarray: quat1 * quat2 (shape: :math:`[4,]`).
    """
    r1 = R.from_quat(quat1)
    r2 = R.from_quat(quat2)
    r = r1 * r2
    return r.as_quat()


def quat_inverse(quat):
    """
    Return the quaternion inverse.

    Args:
        quat (list or np.ndarray): quaternion [x,y,z,w] (shape: :math:`[4,]`).

    Returns:
        np.ndarray: inverse quaternion (shape: :math:`[4,]`).
    """
    r = R.from_quat(quat)
    return r.inv().as_quat()


def pose_difference_np(pose, pose_ref, rs=False):
    """
    Compute the approximate difference between two poses, by comparing
    the norm between the positions and using the quaternion difference
    to compute the rotation similarity

    Args:
        pose (np.ndarray): pose 1, in form [pos, ori], where
            pos (shape: [3,]) is of the form [x, y, z], and ori (shape: [4,])
            if of the form [x, y, z, w]
        pose_ref (np.ndarray): pose 2, in form [pos, ori], where
            pos (shape: [3,]) is of the form [x, y, z], and ori (shape: [4,])
            if of the form [x, y, z, w]
        rs (bool): If True, use rotation_similarity metric for orientation error.
            Otherwise use geodesic distance. Defaults to False

    Returns:
        2-element tuple containing:
        - np.ndarray: Euclidean distance between positions
        - np.ndarray: Quaternion difference between the orientations
    """
    pos_1, pos_2 = pose[:3], pose_ref[:3]
    ori_1, ori_2 = pose[3:], pose_ref[3:]

    pos_diff = pos_1 - pos_2
    pos_error = np.linalg.norm(pos_diff)

    quat_diff = quat_multiply(quat_inverse(ori_1), ori_2)
    rot_similarity = np.abs(quat_diff[3])

    # dot_prod = np.dot(ori_1, ori_2)
    dot_prod1 = np.clip(np.dot(ori_1, ori_2), 0, 1)
    angle_diff1 = np.arccos(2*dot_prod1**2 - 1)

    dot_prod2 = np.clip(np.dot(ori_1, -ori_2), 0, 1)
    angle_diff2 = np.arccos(2*dot_prod2**2 - 1)    

    if rs:
        angle_diff1 = 1 - rot_similarity
        angle_diff2 = np.inf
    return pos_error, min(angle_diff1, angle_diff2)


def ori_difference(ori_1, ori_2):
    dot_prod1 = np.clip(np.dot(ori_1, ori_2), 0, 1)
    angle_diff1 = np.arccos(2*dot_prod1**2 - 1)

    dot_prod2 = np.clip(np.dot(ori_1, -ori_2), 0, 1)
    angle_diff2 = np.arccos(2*dot_prod2**2 - 1)    
    return min(angle_diff1, angle_diff2)


def pose_from_vectors(x_vec, y_vec, z_vec, trans, frame_id="yumi_body"):
    # Normalized frame
    hand_orient_norm = np.vstack((x_vec, y_vec, z_vec))
    hand_orient_norm = hand_orient_norm.transpose()
    quat = mat2quat(hand_orient_norm)
    # define hand pose
    pose = convert_pose_type(list(trans) + list(quat),
                             type_out="PoseStamped",
                             frame_out=frame_id)
    return pose

def transform_vectors(vectors, pose_transform):
    """Transform a set of vectors

    Args:
        vectors (np.ndarray): Numpy array of vectors, size
            [N, 3], where each row is a vector [x, y, z]
        pose_transform (PoseStamped): PoseStamped object defining the transform

    Returns:
        np.ndarray: Size [N, 3] with transformed vectors in same order as input
    """
    vectors_homog = np.ones((4, vectors.shape[0]))
    vectors_homog[:-1, :] = vectors.T

    T_transform = matrix_from_pose(pose_transform)

    vectors_trans_homog = np.matmul(T_transform, vectors_homog)
    vectors_trans = vectors_trans_homog[:-1, :].T
    return vectors_trans

def sample_orthogonal_vector(reference_vector):
    """Sample a random unit vector that is orthogonal to the specified reference

    Args:
        reference_vector (np.ndarray): Numpy array with
            reference vector, [x, y, z]. Cannot be all zeros

    Return:
        np.ndarray: Size [3,] that is orthogonal to specified vector
    """
    # y_unnorm = np.zeros(reference_vector.shape)

    # nonzero_inds = np.where(reference_vector)[0]
    # ind_1 = random.sample(nonzero_inds, 1)[0]
    # while True:
    #     ind_2 = np.random.randint(3)
    #     if ind_1 != ind_2:
    #         break

    # y_unnorm[ind_1] = reference_vector[ind_2]
    # y_unnorm[ind_2] = -reference_vector[ind_1]
    # y = y_unnorm / np.linalg.norm(y_unnorm)
    rand_vec = np.random.rand(3) * 2 - 1
    # rand_vec = np.random.rand(3) * -1.0
    y_unnorm = project_point2plane(rand_vec, reference_vector, [0, 0, 0])[0]
    y = y_unnorm / np.linalg.norm(y_unnorm)
    return y


def project_point2plane(point, plane_normal, plane_points):
    '''project a point to a plane'''
    point_plane = plane_points[0]
    w = point - point_plane
    dist = (np.dot(plane_normal, w) / np.linalg.norm(plane_normal))
    projected_point = point - dist * plane_normal / np.linalg.norm(plane_normal)
    return projected_point, dist


def body_world_yaw(current_pose, theta=None):
    """Given some initial pose, sample a new pose that is
    a pure yaw about the world frame orientation, with
    the origin at the current pose position

    Args:
        current_pose (PoseStamped): Current pose, to be yawed
        theta (float): Angle by which to yaw. If None, random
            theta in range [0, 2*pi] will be sampled

    Returns:
        PoseStamped: New pose, after in place yaw applied
    """
    current_pose_list = pose_stamped2list(current_pose)

    trans_to_origin = np.asarray(current_pose_list[:3])
    if theta is None:
        theta = np.random.random() * 2 * np.pi
    yaw = R.from_euler('xyz', [0, 0, theta]).as_matrix()

    # translate the source to the origin
    T_0 = np.eye(4)
    T_0[:-1, -1] = -trans_to_origin

    # apply pure rotation in the world frame
    T_1 = np.eye(4)
    T_1[:-1, :-1] = yaw

    # translate in [x, y] back away from origin
    T_2 = np.eye(4)
    T_2[0, -1] = trans_to_origin[0]
    T_2[1, -1] = trans_to_origin[1]
    T_2[2, -1] = trans_to_origin[2]
    yaw_trans = np.matmul(T_2, np.matmul(T_1, T_0))
    yaw_trans_pose = pose_from_matrix(yaw_trans)

    new_pose = transform_pose(current_pose, yaw_trans_pose)
    # new_pose_list = pose_stamped2list(new_pose)
    return new_pose


def rand_body_yaw_transform(pos, min_theta=0.0, max_theta=2*np.pi):
    """Given some initial position, sample a Transform that is
    a pure yaw about the world frame orientation, with
    the origin at the current pose position

    Args:
        pos (np.ndarray): Current position in the world frame 
        min (float, optional): Minimum boundary for sample
        max (float, optional): Maximum boundary for sample

    Returns:
        np.ndarray: Transformation matrix
    """
    if isinstance(pos, list):
        pos = np.asarray(pos)    
    trans_to_origin = pos
    theta = np.random.random() * (max_theta - min_theta) + min_theta
    yaw = R.from_euler('xyz', [0, 0, theta]).as_matrix()[:3, :3]

    # translate the source to the origin
    T_0 = np.eye(4)
    T_0[:-1, -1] = -trans_to_origin

    # apply pure rotation in the world frame
    T_1 = np.eye(4)
    T_1[:-1, :-1] = yaw

    # translate in [x, y] back away from origin
    T_2 = np.eye(4)
    T_2[0, -1] = trans_to_origin[0]
    T_2[1, -1] = trans_to_origin[1]
    T_2[2, -1] = trans_to_origin[2]
    yaw_trans = np.matmul(T_2, np.matmul(T_1, T_0))
    return yaw_trans


def get_base_pose_pb(obj_id, pb_client_id=0):
    import pybullet as p
    pose = p.getBasePositionAndOrientation(obj_id, physicsClientId=pb_client_id)
    pos, ori = list(pose[0]), list(pose[1])
    pose = list2pose_stamped(pos + ori)
    return pose


def transform_pcd(pcd, transform):
    if pcd.shape[1] != 4:
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    pcd_new = np.matmul(transform, pcd.T)[:-1, :].T
    return pcd_new


# import healpy as hp
# from https://github.com/google-research/google-research/blob/3ed7475fef726832c7288044c806481adc6de827/implicit_pdf/models.py#L381

def generate_healpix_grid(recursion_level=None, size=None):
  """Generates an equivolumetric grid on SO(3) following Yershova et al. (2010).
  Uses a Healpix grid on the 2-sphere as a starting point and then tiles it
  along the 'tilt' direction 6*2**recursion_level times over 2pi.
  Args:
    recursion_level: An integer which determines the level of resolution of the
      grid.  The final number of points will be 72*8**recursion_level.  A
      recursion_level of 2 (4k points) was used for training and 5 (2.4M points)
      for evaluation.
    size: A number of rotations to be included in the grid.  The nearest grid
      size in log space is returned.
  Returns:
    (N, 3, 3) array of rotation matrices, where N=72*8**recursion_level.
  """
  import healpy as hp  # pylint: disable=g-import-not-at-top
  from airobot.utils import common

  assert not(recursion_level is None and size is None)
  if size:
    recursion_level = max(int(np.round(np.log(size/72.)/np.log(8.))), 0)
  number_per_side = 2**recursion_level
  number_pix = hp.nside2npix(number_per_side)
  s2_points = hp.pix2vec(number_per_side, np.arange(number_pix))
  s2_points = np.stack([*s2_points], 1)

  # Take these points on the sphere and
  azimuths = np.arctan2(s2_points[:, 1], s2_points[:, 0])
  tilts = np.linspace(0, 2*np.pi, 6*2**recursion_level, endpoint=False)
  polars = np.arccos(s2_points[:, 2])
  grid_rots_mats = []
  for tilt in tilts:
    # Build up the rotations from Euler angles, zyz format
    # rot_mats = tfg.rotation_matrix_3d.from_euler(
    #     np.stack([azimuths,
    #               np.zeros(number_pix),
    #               np.zeros(number_pix)], 1))
    # rot_mats = rot_mats @ tfg.rotation_matrix_3d.from_euler(
    #     np.stack([np.zeros(number_pix),
    #               np.zeros(number_pix),
    #               polars], 1))
    # rot_mats = rot_mats @ tf.expand_dims(
    #     tfg.rotation_matrix_3d.from_euler([tilt, 0., 0.]), 0)

    euler = np.stack([azimuths, np.zeros(number_pix), np.zeros(number_pix)], 1)
    rot_mats = common.euler2rot(euler)

    euler2 = np.stack([np.zeros(number_pix), np.zeros(number_pix), polars], 1)
    rot_mats = rot_mats @ common.euler2rot(euler2)

    euler3 = [tilt, 0, 0]
    rot_mats = rot_mats @ common.euler2rot(euler3)

    grid_rots_mats.append(rot_mats)

  grid_rots_mats = np.concatenate(grid_rots_mats, 0)
  return grid_rots_mats
