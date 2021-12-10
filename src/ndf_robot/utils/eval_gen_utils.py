import os, os.path as osp
import numpy as np
import trimesh
from scipy.spatial import KDTree
import time
import pybullet as p
import copy

from ndf_robot.utils import util, trimesh_util

# some helpers
def soft_grasp_close(robot, joint_id2, force=100):
    p.setJointMotorControl2(robot.arm.robot_id, joint_id2, p.VELOCITY_CONTROL, targetVelocity=-1, force=force)
    # p.setJointMotorControl2(robot.arm.robot_id, joint_id2, p.VELOCITY_CONTROL, targetVelocity=-1, force=100)
    time.sleep(0.2)        


def constraint_grasp_close(robot, obj_id):
    obj_pose_world = p.getBasePositionAndOrientation(obj_id)
    obj_pose_world = util.list2pose_stamped(list(obj_pose_world[0]) + list(obj_pose_world[1]))

    ee_link_id = robot.arm.ee_link_id
    ee_pose_world = np.concatenate(robot.arm.get_ee_pose()[:2]).tolist()
    ee_pose_world = util.list2pose_stamped(ee_pose_world)

    obj_pose_ee = util.convert_reference_frame(
        pose_source=obj_pose_world,
        pose_frame_target=ee_pose_world,
        pose_frame_source=util.unit_pose()
    )
    obj_pose_ee_list = util.pose_stamped2list(obj_pose_ee)

    cid = p.createConstraint(
        parentBodyUniqueId=robot.arm.robot_id,
        parentLinkIndex=ee_link_id,
        childBodyUniqueId=obj_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=obj_pose_ee_list[:3],
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=obj_pose_ee_list[3:])
    return cid

def constraint_obj_world(obj_id, pos, ori):
    o_cid = p.createConstraint(
        obj_id,
        -1,
        -1,
        -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        [0, 0, 0],
        pos, childFrameOrientation=ori,
    )
    return o_cid


def constraint_grasp_open(cid=None):
    if cid is not None:
        p.removeConstraint(cid)


def put_on_angle_rack(robot, z=0.06, angle=40):
    y = np.abs(z*np.tan(np.deg2rad(angle)))
    robot.arm.move_ee_xyz([0, y, -z])


def take_off_angle_rack(robot, z=0.06, angle=40):
    y = np.abs(z*np.tan(np.deg2rad(angle)))
    robot.arm.move_ee_xyz([0, -y, z])


def safeCollisionFilterPair(bodyUniqueIdA, bodyUniqueIdB, linkIndexA, linkIndexB, enableCollision, *args, **kwargs):
    if bodyUniqueIdA is not None and bodyUniqueIdB is not None and linkIndexA is not None and linkIndexB is not None:
        p.setCollisionFilterPair(bodyUniqueIdA=bodyUniqueIdA, bodyUniqueIdB=bodyUniqueIdB, linkIndexA=linkIndexA, linkIndexB=linkIndexB, enableCollision=enableCollision)

def safeRemoveConstraint(cid):
    if cid is not None:
        p.removeConstraint(cid)

def object_is_still_grasped(robot, obj_id, right_pad_id, left_pad_id):
    obj_finger_right_info = p.getClosestPoints(bodyA=obj_id, bodyB=robot.arm.robot_id, distance=0.002,
                                            linkIndexA=-1, linkIndexB=right_pad_id)
    obj_finger_left_info = p.getClosestPoints(bodyA=obj_id, bodyB=robot.arm.robot_id, distance=0.002,
                                            linkIndexA=-1, linkIndexB=left_pad_id)
    obj_still_in_grasp = len(obj_finger_left_info) > 0 or len(obj_finger_right_info) > 0
    return obj_still_in_grasp


def get_ee_offset(ee_pose):
    """
    Gets the updated world frame normal direction of the palms
    """
    dist = 0.1
    normal_x = util.list2pose_stamped([dist, 0, 0, 0, 0, 0, 1])
    normal_y = util.list2pose_stamped([0, dist, 0, 0, 0, 0, 1])
    normal_z = util.list2pose_stamped([0, 0, dist, 0, 0, 0, 1])

    normal_x = util.transform_pose(normal_x, util.list2pose_stamped(ee_pose))
    normal_y = util.transform_pose(normal_y, util.list2pose_stamped(ee_pose))
    normal_z = util.transform_pose(normal_z, util.list2pose_stamped(ee_pose))

    # dx_vec = util.pose_stamped2np(normal_x)[:3] - np.asarray(ee_pose)[:3]
    # dy_vec = util.pose_stamped2np(normal_y)[:3] - np.asarray(ee_pose)[:3]
    # dz_vec = util.pose_stamped2np(normal_z)[:3] - np.asarray(ee_pose)[:3]
    dx_vec = np.asarray(ee_pose)[:3] - util.pose_stamped2np(normal_x)[:3]
    dy_vec = np.asarray(ee_pose)[:3] - util.pose_stamped2np(normal_y)[:3]
    dz_vec = np.asarray(ee_pose)[:3] - util.pose_stamped2np(normal_z)[:3]

    # return dx_vec.tolist() + [0, 0, 0, 1]
    # return dx_vec.tolist() + [0, 0, 0, 1]
    return dz_vec.tolist() + [0, 0, 0, 1]


def process_demo_data_rack(grasp_data, place_data, cfg, gaussian_scale=0.1):
    data = grasp_data
    demo_obj_pts = data['object_pointcloud']  # observed shape point cloud at start
    demo_pts_mean = np.mean(demo_obj_pts, axis=0)
    inliers = np.where(np.linalg.norm(demo_obj_pts - demo_pts_mean, 2, 1) < 0.2)[0]
    demo_obj_pts = demo_obj_pts[inliers]

    demo_gripper_pts_rs = data['gripper_pts']  # points we use to represent the gripper at their canonical pose position shown in the demonstration
    demo_gripper_pcd_rs = trimesh.PointCloud(demo_gripper_pts_rs)
    demo_ee_mat = util.matrix_from_pose(util.list2pose_stamped(data['ee_pose_world']))  # end-effector pose before grasping
    demo_gripper_pcd_rs.apply_transform(demo_ee_mat)
    demo_gripper_pts_rs = np.asarray(demo_gripper_pcd_rs.vertices) # points we use to represent the gripper at their canonical pose position shown in the demonstration

    # demo_gripper_pts = data['gripper_pts_gaussian'] * gaussian_scale # query points for the gripper (Gaussian distributed)
    demo_gripper_pts = data['gripper_pts_uniform'] # query points for the gripper (Uniform distributed)
    demo_gripper_pcd = trimesh.PointCloud(demo_gripper_pts)
    demo_ee_mat = util.matrix_from_pose(util.list2pose_stamped(data['ee_pose_world']))  # end-effector pose before grasping
    demo_gripper_pcd.apply_transform(demo_ee_mat)
    demo_gripper_pts = np.asarray(demo_gripper_pcd.vertices) # points we use to represent the gripper at their canonical pose position shown in the demonstration

    # place_data = np.load(place_demo_fn, allow_pickle=True)
    place_demo_obj_pts = place_data['object_pointcloud']  # observed shape points at start
    place_demo_pts_mean = np.mean(place_demo_obj_pts, axis=0)
    inliers = np.where(np.linalg.norm(place_demo_obj_pts - place_demo_pts_mean, 2, 1) < 0.2)[0]
    place_demo_obj_pts = place_demo_obj_pts[inliers]
    
    place_demo_obj_pcd = trimesh.PointCloud(place_demo_obj_pts)
    pick_demo_obj_pose = data['obj_pose_world']
    place_demo_obj_pose = place_data['obj_pose_world']
    place_demo_obj_pose_rel_mat = util.matrix_from_pose(
        util.get_transform(
            util.list2pose_stamped(place_demo_obj_pose), 
            util.list2pose_stamped(pick_demo_obj_pose)))  # ground truth relative transformation in demo
    place_demo_obj_pcd.apply_transform(place_demo_obj_pose_rel_mat)  # start shape points transformed into goal configuration
    place_demo_obj_pts = np.asarray(place_demo_obj_pcd.vertices)  # shape points at goal

    place_demo_rack_pts_rs = place_data['rack_pointcloud_gt']  # points used to represent the rack in canonical pose
    place_demo_rack_pcd_rs = trimesh.PointCloud(place_demo_rack_pts_rs)
    place_demo_rack_pose_mat = util.matrix_from_pose(util.list2pose_stamped(place_data['rack_pose_world']))
    place_demo_rack_pcd_rs.apply_transform(place_demo_rack_pose_mat)  # points used to represent the rack in demo pose
    place_demo_rack_pts_rs = np.asarray(place_demo_rack_pcd_rs.vertices)

    place_demo_rack_pts_obs = np.concatenate(place_data['rack_pointcloud_observed'], 0)  # points that we observed on the rack
    rndperm = np.random.permutation(place_demo_rack_pts_obs.shape[0])
    place_demo_rack_pts_obs = place_demo_rack_pts_obs[rndperm[:int(place_demo_rack_pts_obs.shape[0]/2)]]

    gaussian_place_demo_rack_pts = place_data['rack_pointcloud_gaussian'] * gaussian_scale # points used to represent the rack in canonical pose
    gaussian_place_demo_rack_pcd = trimesh.PointCloud(gaussian_place_demo_rack_pts)
    gaussian_place_demo_rack_pose_mat = util.matrix_from_pose(util.list2pose_stamped(place_data['rack_contact_pose']))
    gaussian_place_demo_rack_pcd.apply_transform(gaussian_place_demo_rack_pose_mat)  # points used to represent the rack in demo pose
    gaussian_place_demo_rack_pts = np.asarray(gaussian_place_demo_rack_pcd.vertices)

    uniform_place_demo_rack_pts = place_data['rack_pointcloud_uniform']
    uniform_place_demo_rack_pcd = trimesh.PointCloud(uniform_place_demo_rack_pts)
    uniform_place_demo_rack_pose_mat = util.matrix_from_pose(util.list2pose_stamped(place_data['rack_pose_world']))
    uniform_place_demo_rack_pcd.apply_transform(uniform_place_demo_rack_pose_mat)  # points used to represent the rack in demo pose
    uniform_place_demo_rack_pts = np.asarray(uniform_place_demo_rack_pcd.vertices)
    
    # place_demo_rack_pts = place_demo_rack_bb_pts
    place_demo_rack_pts = uniform_place_demo_rack_pts

    target_info = dict(
        demo_query_pts=demo_gripper_pts, 
        demo_query_pts_real_shape=demo_gripper_pts_rs,
        demo_obj_pts=demo_obj_pts, 
        demo_ee_pose_world=data['ee_pose_world'],
        demo_query_pt_pose=data['gripper_contact_pose'],
        demo_obj_rel_transform=np.eye(4))

    rack_target_info = dict(
        demo_query_pts=place_demo_rack_pts, 
        demo_query_pts_real_shape=place_demo_rack_pts_rs,
        demo_obj_pts=place_demo_obj_pts, 
        demo_ee_pose_world=place_data['ee_pose_world'],
        demo_query_pt_pose=place_data['rack_contact_pose'],
        demo_obj_rel_transform=place_demo_obj_pose_rel_mat)
        
    shapenet_id = data['shapenet_id'].item()

    return target_info, rack_target_info, shapenet_id


def process_demo_data_shelf(grasp_data, place_data, cfg):
    data = grasp_data
    demo_obj_pts = data['object_pointcloud']  # observed shape point cloud at start
    demo_pts_mean = np.mean(demo_obj_pts, axis=0)
    inliers = np.where(np.linalg.norm(demo_obj_pts - demo_pts_mean, 2, 1) < 0.2)[0]
    demo_obj_pts = demo_obj_pts[inliers]

    demo_gripper_pts_rs = data['gripper_pts']  # points we use to represent the gripper at their canonical pose position shown in the demonstration
    demo_gripper_pcd_rs = trimesh.PointCloud(demo_gripper_pts_rs)
    demo_ee_mat = util.matrix_from_pose(util.list2pose_stamped(data['ee_pose_world']))  # end-effector pose before grasping
    demo_gripper_pcd_rs.apply_transform(demo_ee_mat)
    demo_gripper_pts_rs = np.asarray(demo_gripper_pcd_rs.vertices) # points we use to represent the gripper at their canonical pose position shown in the demonstration

    # demo_gripper_pts = data['gripper_pts_gaussian'] * gaussian_scale # query points for the gripper (Gaussian distributed)
    demo_gripper_pts = data['gripper_pts_uniform'] # query points for the gripper (Uniform distributed)
    demo_gripper_pcd = trimesh.PointCloud(demo_gripper_pts)
    demo_ee_mat = util.matrix_from_pose(util.list2pose_stamped(data['ee_pose_world']))  # end-effector pose before grasping
    demo_gripper_pcd.apply_transform(demo_ee_mat)
    demo_gripper_pts = np.asarray(demo_gripper_pcd.vertices) # points we use to represent the gripper at their canonical pose position shown in the demonstration

    # place_data = np.load(place_demo_fn, allow_pickle=True)
    place_demo_obj_pts = place_data['object_pointcloud']  # observed shape points at start
    place_demo_pts_mean = np.mean(place_demo_obj_pts, axis=0)
    inliers = np.where(np.linalg.norm(place_demo_obj_pts - place_demo_pts_mean, 2, 1) < 0.2)[0]
    place_demo_obj_pts = place_demo_obj_pts[inliers]
    
    place_demo_obj_pcd = trimesh.PointCloud(place_demo_obj_pts)
    pick_demo_obj_pose = data['obj_pose_world']
    place_demo_obj_pose = place_data['obj_pose_world']
    place_demo_obj_pose_rel_mat = util.matrix_from_pose(
        util.get_transform(
            util.list2pose_stamped(place_demo_obj_pose), 
            util.list2pose_stamped(pick_demo_obj_pose)))  # ground truth relative transformation in demo
    place_demo_obj_pcd.apply_transform(place_demo_obj_pose_rel_mat)  # start shape points transformed into goal configuration
    place_demo_obj_pts = np.asarray(place_demo_obj_pcd.vertices)  # shape points at goal

    place_demo_shelf_pts_rs = place_data['shelf_pointcloud_gt']  # points used to represent the rack in canonical pose
    place_demo_shelf_pcd_rs = trimesh.PointCloud(place_demo_shelf_pts_rs)
    place_demo_shelf_pose_mat = util.matrix_from_pose(util.list2pose_stamped(place_data['shelf_pose_world']))
    place_demo_shelf_pcd_rs.apply_transform(place_demo_shelf_pose_mat)  # points used to represent the rack in demo pose
    place_demo_shelf_pts_rs = np.asarray(place_demo_shelf_pcd_rs.vertices)

    place_demo_rack_pts_obs = np.concatenate(place_data['shelf_pointcloud_observed'], 0)  # points that we observed on the rack
    rndperm = np.random.permutation(place_demo_rack_pts_obs.shape[0])
    place_demo_rack_pts_obs = place_demo_rack_pts_obs[rndperm[:int(place_demo_rack_pts_obs.shape[0]/2)]]

    uniform_place_demo_shelf_pts = place_data['shelf_pointcloud_uniform']
    uniform_place_demo_shelf_pcd = trimesh.PointCloud(uniform_place_demo_shelf_pts)
    uniform_place_demo_shelf_pose_mat = util.matrix_from_pose(util.list2pose_stamped(place_data['shelf_pose_world']))
    uniform_place_demo_shelf_pcd.apply_transform(uniform_place_demo_shelf_pose_mat)  # points used to represent the rack in demo pose
    uniform_place_demo_shelf_pts = np.asarray(uniform_place_demo_shelf_pcd.vertices)
    
    place_demo_shelf_pts = uniform_place_demo_shelf_pts 

    target_info = dict(
        demo_query_pts=demo_gripper_pts, 
        demo_query_pts_real_shape=demo_gripper_pts_rs,
        demo_obj_pts=demo_obj_pts, 
        demo_ee_pose_world=data['ee_pose_world'],
        demo_query_pt_pose=data['gripper_contact_pose'],
        demo_obj_rel_transform=np.eye(4))

    rack_target_info = dict(
        demo_query_pts=place_demo_shelf_pts,
        demo_query_pts_real_shape=place_demo_shelf_pts_rs,
        demo_obj_pts=place_demo_obj_pts, 
        demo_ee_pose_world=place_data['ee_pose_world'],
        demo_query_pt_pose=place_data['rack_contact_pose'],
        demo_obj_rel_transform=place_demo_obj_pose_rel_mat)
        
    shapenet_id = data['shapenet_id'].item()

    return target_info, rack_target_info, shapenet_id


def process_xq_data(grasp_data, place_data, shelf=True):
    optimizer_gripper_pts = grasp_data['gripper_pts_uniform']

    uniform_place_demo_rack_pts = place_data['rack_pointcloud_uniform']
    uniform_place_demo_rack_pcd = trimesh.PointCloud(uniform_place_demo_rack_pts)
    uniform_place_demo_rack_pose_mat = util.matrix_from_pose(util.list2pose_stamped(place_data['rack_pose_world']))
    uniform_place_demo_rack_pcd.apply_transform(uniform_place_demo_rack_pose_mat)  # points used to represent the rack in demo pose
    uniform_place_demo_rack_pts = np.asarray(uniform_place_demo_rack_pcd.vertices)

    if shelf:
        uniform_place_demo_shelf_pts = place_data['shelf_pointcloud_uniform']
        uniform_place_demo_shelf_pcd = trimesh.PointCloud(uniform_place_demo_shelf_pts)
        uniform_place_demo_shelf_pose_mat = util.matrix_from_pose(util.list2pose_stamped(place_data['shelf_pose_world']))
        uniform_place_demo_shelf_pcd.apply_transform(uniform_place_demo_shelf_pose_mat)  # points used to represent the rack in demo pose
        uniform_place_demo_shelf_pts = np.asarray(uniform_place_demo_shelf_pcd.vertices)

        return optimizer_gripper_pts, uniform_place_demo_rack_pts, uniform_place_demo_shelf_pts
    else:
        return optimizer_gripper_pts, uniform_place_demo_rack_pts, None 


def process_xq_rs_data(grasp_data, place_data, shelf=True):
    optimizer_gripper_pts_rs = grasp_data['gripper_pts']

    place_demo_rack_pts_rs = place_data['rack_pointcloud_gt']  # points used to represent the rack in canonical pose
    place_demo_rack_pcd_rs = trimesh.PointCloud(place_demo_rack_pts_rs)
    place_demo_rack_pose_mat = util.matrix_from_pose(util.list2pose_stamped(place_data['rack_pose_world']))
    place_demo_rack_pcd_rs.apply_transform(place_demo_rack_pose_mat)  # points used to represent the rack in demo pose
    place_demo_rack_pts_rs = np.asarray(place_demo_rack_pcd_rs.vertices)

    if shelf:
        place_demo_shelf_pts_rs = place_data['shelf_pointcloud_gt']  # points used to represent the rack in canonical pose
        place_demo_shelf_pcd_rs = trimesh.PointCloud(place_demo_shelf_pts_rs)
        place_demo_shelf_pose_mat = util.matrix_from_pose(util.list2pose_stamped(place_data['shelf_pose_world']))
        place_demo_shelf_pcd_rs.apply_transform(place_demo_shelf_pose_mat)  # points used to represent the rack in demo pose
        place_demo_shelf_pts_rs = np.asarray(place_demo_shelf_pcd_rs.vertices)

        return optimizer_gripper_pts_rs, place_demo_rack_pts_rs, place_demo_shelf_pts_rs
    else:
        return optimizer_gripper_pts_rs, place_demo_rack_pts_rs, None

def post_process_grasp_point(pre_grasp_ee_pose, target_obj_pcd, thin_feature=True, grasp_viz=False, grasp_dist_thresh=0.0025):
    grasp_pt = pre_grasp_ee_pose[:3]

    # o3d_obj_pcd = open3d.geometry.PointCloud()
    # o3d_obj_pcd.points  = open3d.geometry.Vector3dVector(target_obj_pcd)

    # down_o3d_pcd = o3d_obj_pcd.voxel_down_sample(0.005)
    # target_obj_voxel_down = np.asarray(down_o3d_pcd.points)
    rix = np.random.permutation(target_obj_pcd.shape[0])
    target_obj_voxel_down = target_obj_pcd[rix[:int(target_obj_pcd.shape[0]/5)]]

    target_obj_tree = KDTree(target_obj_pcd)
    target_obj_down_tree = KDTree(target_obj_voxel_down)
    grasp_close_idxs = target_obj_tree.query(grasp_pt, k=100)[1]
    grasp_close_pts = target_obj_pcd[grasp_close_idxs].squeeze()

    n_pts_within_ball = 0
    k = 0
    while True:
        # get the mean of the nearest neighbors, and check how many points are inside the ball with size roughly the contact patch of the fingers
        new_grasp_pt = np.mean(grasp_close_pts, axis=0)
        new_idxs_within_ball = target_obj_down_tree.query_ball_point(new_grasp_pt, r=0.015)
        new_idxs_within_larger_ball = target_obj_down_tree.query_ball_point(new_grasp_pt, r=0.02)
        pts_within_ball = target_obj_voxel_down[new_idxs_within_ball].squeeze()
        pts_within_larger_ball = target_obj_voxel_down[new_idxs_within_larger_ball].squeeze()
        n_pts_within_ball = len(new_idxs_within_ball)

        if n_pts_within_ball > 75:
            break
        else:
            grasp_pt = np.mean(pts_within_larger_ball, axis=0) 
            grasp_close_idxs = target_obj_tree.query(grasp_pt, k=100)[1]
            grasp_close_pts = target_obj_pcd[grasp_close_idxs].squeeze()
        
        k += 1
        if k > 5:
            break

    # get antipodal point
    # local_grasp_normal = np.mean(np.asarray(close_o3d.normals), axis=0)
    local_grasp_normal = util.vec_from_pose(util.list2pose_stamped(pre_grasp_ee_pose))[1]

    if not thin_feature:
        # sample along this direction to find an antipodal point
        search_vec = -1.0 * local_grasp_normal
        search_final_pt = grasp_pt + search_vec
        search_pts = np.linspace(grasp_pt, search_final_pt, 250)
        min_dist = np.inf
        for pt in search_pts:
            a_close_idx = target_obj_tree.query(pt, k=1)[1]
            a_close_pt = target_obj_pcd[a_close_idx].squeeze()
            dist = np.linalg.norm(a_close_pt - pt)
            if dist < min_dist and (np.linalg.norm(a_close_pt - grasp_pt) > grasp_dist_thresh):
                antipodal_close_idx = a_close_idx
                antipodal_close_pt = a_close_pt
                min_dist = dist
        # antipodal_close_idx = target_obj_tree.query(search_final_pt, k=1)[1]
        # antipodal_close_pt = target_obj_pcd_obs[antipodal_close_idx].squeeze()
        detected_pt = copy.deepcopy(grasp_pt)
        grasp_pt = (antipodal_close_pt + grasp_pt) / 2.0              

    if grasp_viz:
    # if True:
        # scene = trimesh_util.trimesh_show(
        #     [target_obj_pcd_obs[::5], grasp_close_pts, pts_within_ball], show=False)
        scene = trimesh_util.trimesh_show(
            [target_obj_voxel_down, grasp_close_pts, pts_within_ball], show=False)

        scale = 0.1
        grasp_sph = trimesh.creation.uv_sphere(0.005)
        grasp_sph.visual.face_colors = np.tile([40, 40, 40, 255], (grasp_sph.faces.shape[0], 1))
        grasp_sph.apply_translation(grasp_pt)

        new_grasp_sph = trimesh.creation.uv_sphere(0.005)
        new_grasp_sph.visual.face_colors = np.tile([40, 255, 40, 255], (new_grasp_sph.faces.shape[0], 1))
        new_grasp_sph.apply_translation(new_grasp_pt)

        # detected_sph = trimesh.creation.uv_sphere(0.005)
        # detected_sph.visual.face_colors = np.tile([40, 40, 40, 255], (detected_sph.faces.shape[0], 1))
        # detected_sph.apply_translation(grasp_pt.squeeze())

        # antipodal_sph = trimesh.creation.uv_sphere(0.005)
        # antipodal_sph.visual.face_colors = np.tile([40, 40, 40, 255], (antipodal_sph.faces.shape[0], 1))
        # antipodal_sph.apply_translation(antipodal_close_pt.squeeze())
        
        # grasp_x_pt = (grasp_pt + x_vec * scale).squeeze()
        # grasp_y_pt = (grasp_pt + y_vec * scale).squeeze()
        # grasp_z_pt = (grasp_pt + z_vec * scale).squeeze()
        # x_sph = trimesh.creation.uv_sphere(0.005)
        # y_sph = trimesh.creation.uv_sphere(0.005)
        # z_sph = trimesh.creation.uv_sphere(0.005)
        # x_sph.visual.face_colors = np.tile([255, 0, 0, 255], (x_sph.faces.shape[0], 1))
        # y_sph.visual.face_colors = np.tile([255, 255, 0, 255], (y_sph.faces.shape[0], 1))
        # z_sph.visual.face_colors = np.tile([0, 0, 255, 255], (z_sph.faces.shape[0], 1))
        # x_sph.apply_translation(grasp_x_pt)
        # y_sph.apply_translation(grasp_y_pt)
        # z_sph.apply_translation(grasp_z_pt)

        # pt1_sph = trimesh.creation.uv_sphere(0.0025)
        # pt2_sph = trimesh.creation.uv_sphere(0.0025)
        # pt1_sph.visual.face_colors = np.tile([255, 0, 255, 255], (pt1_sph.faces.shape[0], 1))
        # pt2_sph.visual.face_colors = np.tile([128, 128, 128, 255], (pt2_sph.faces.shape[0], 1))
        # pt1_sph.apply_translation(best_pt1.squeeze())
        # pt2_sph.apply_translation(best_pt2.squeeze())

        scene.add_geometry([grasp_sph, new_grasp_sph])
        scene.show()
    return new_grasp_pt