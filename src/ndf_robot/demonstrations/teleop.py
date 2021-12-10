import os, os.path as osp
import random
import numpy as np
import time
from multiprocessing import Process, Pipe, Queue, Manager
import signal
import copy
from pynput import keyboard
import trimesh
import pybullet as p

from airobot import Robot
from airobot.utils.common import euler2quat

from ndf_robot.utils import util, path_util
from ndf_robot.utils.eval_gen_utils import safeCollisionFilterPair
from ndf_robot.robot.multicam import MultiCams

from ndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from ndf_robot.config.default_obj_cfg import get_obj_cfg_defaults


def hide_link(obj_id, link_id): 
    if link_id is not None:
        p.changeVisualShape(obj_id, link_id, rgbaColor=[0, 0, 0, 0])


def show_link(obj_id, link_id, color):
    if link_id is not None:
        p.changeVisualShape(obj_id, link_id, rgbaColor=color)


KEY_MSG_MAP = {
    'a': 'Y+',
    'd': 'Y-',
    's': 'X+',
    'x': 'X-',
    'e': 'Z+',
    'q': 'Z-',
    'r': 'G+',
    'f': 'G-',
    'u': 'rX+',
    'o': 'rX-',
    'i': 'rY+',
    'k': 'rY-',
    'j': 'rZ+',
    'l': 'rZ-',
    'z': 'OPEN',
    'c': 'CLOSE',
    '0': 'END',
    '9': 'RESET',
    '1': 'DEMO_PICK',
    '2': 'DEMO_PLACE',
    '3': 'SKIP',
    '4': 'ON_RACK',
    '5': 'OFF_RACK',
}


def worker_robot(child_conn, work_queue, result_queue, global_dict, worker_flag_dict, worker_id):
    np.random.seed(global_dict['seed'])
    random.seed(global_dict['seed'])
    obj_class = global_dict['obj_class']
    save_dir = global_dict['save_dir']
    while True:
        try:
            if not child_conn.poll(0.0001):
                continue
            msg = child_conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if msg == "RESET":
            robot = Robot('franka', pb_cfg={'gui': True})
            new_home = [
                -0.2798878477975077, 
                -0.23823885657833854, 
                0.28537688039025716, 
                -2.081827496447527, 
                0.10717202097307935, 
                1.8621456957353935, 
                0.8129974299835407                
            ]

            # general experiment + environment setup/scene generation configs
            cfg = get_eval_cfg_defaults()
            config_fname = osp.join(path_util.get_ndf_config(), 'eval_cfgs', global_dict['config'] + '.yaml')
            if osp.exists(config_fname):
                cfg.merge_from_file(config_fname)
            else:
                print('Config file %s does not exist, using defaults' % config_fname)
            cfg.freeze()

            # object specific configs
            obj_cfg = get_obj_cfg_defaults()
            obj_config_name = osp.join(path_util.get_ndf_config(), '%s_obj_cfg.yaml' % obj_class)
            obj_cfg.merge_from_file(obj_config_name)
            obj_cfg.freeze()

            p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=robot.pb_client.get_client_id())

            finger_joint_id = 9
            left_pad_id = 9
            right_pad_id = 10

            finger_force = 20

            delta = 0.01
            delta_angle = np.pi/24
            angle_N = 10

            # set up possible gripper query points that are used in optimization
            gripper_mesh_file = osp.join(path_util.get_ndf_descriptions(), 'franka_panda/meshes/collision/hand.obj')
            gripper_mesh = trimesh.load_mesh(gripper_mesh_file)
            gripper_pts = gripper_mesh.sample(500)
            gripper_pts_gaussian = np.random.normal(size=(500,3))
            gripper_pts_pcd = trimesh.PointCloud(gripper_pts)
            gripper_pts_bb = gripper_mesh.bounding_box_oriented
            gripper_pts_uniform = gripper_pts_bb.sample_volume(500)

            # set up possible rack query points that are used in optimization
            rack_mesh_file = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/simple_rack.obj')
            rack_mesh = trimesh.load_mesh(rack_mesh_file)
            rack_pts_gt = rack_mesh.sample(500)
            rack_pts_gaussian = np.random.normal(size=(500,3))
            rack_pts_pcd_gt = trimesh.PointCloud(rack_pts_gt)
            rack_pts_bb = rack_mesh.bounding_box_oriented
            rack_pts_uniform = rack_pts_bb.sample_volume(500)

            # set up possible shelf query points that are used in optimization
            shelf_mesh_file = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/shelf_back.stl')
            shelf_mesh = trimesh.load_mesh(shelf_mesh_file)
            shelf_pts_gt = shelf_mesh.sample(500)
            shelf_mesh_bb = shelf_mesh.bounding_box_oriented
            shelf_pts_uniform = shelf_mesh_bb.sample_volume(500)

            have_rack = global_dict['have_rack']
            have_shelf = global_dict['have_shelf']

            if have_rack:
                table_urdf_file = 'table_rack.urdf'
            else:
                table_urdf_file = 'table_shelf.urdf'
            table_urdf = open(osp.join(path_util.get_ndf_descriptions(), 'hanging/table/%s' % table_urdf_file), 'r').read()

            table_ori = euler2quat([0, 0, np.pi / 2])
            table_id = robot.pb_client.load_urdf(osp.join(path_util.get_ndf_descriptions(), 'hanging/table', table_urdf_file),
                                    cfg.TABLE_POS, 
                                    table_ori,
                                    scaling=cfg.TABLE_SCALING)

            z = 0.06; y = np.abs(z*np.tan(np.deg2rad(40)))
            on_rack_offset = np.array([0, y, -z])
            on_shelf_offset = np.array([0, 0, -0.05])
            off_rack_offset = -1.0 * on_rack_offset
            off_shelf_offset = -1.0 * on_shelf_offset

            obj_id = None
            cid = None

            x_low, x_high = cfg.OBJ_SAMPLE_X_HIGH_LOW
            y_low, y_high = cfg.OBJ_SAMPLE_Y_HIGH_LOW
            table_z = cfg.TABLE_Z

            # check if we have list of skipped ids
            if osp.exists(osp.join(save_dir, 'demo_skipped_ids.npz')):
                skipped_ids = np.load(osp.join(save_dir, 'demo_skipped_ids.npz'))['ids'].tolist()
            else:
                skipped_ids = []

            continue
        if msg == "X+":
            robot.arm.move_ee_xyz([delta, 0, 0])
            continue
        if msg == "X-":
            robot.arm.move_ee_xyz([-delta, 0, 0])
            continue
        if msg == "Y+":
            robot.arm.move_ee_xyz([0, delta, 0])
            continue
        if msg == "Y-":
            robot.arm.move_ee_xyz([0, -delta, 0])
            continue
        if msg == "Z+":
            robot.arm.move_ee_xyz([0, 0, delta])
            continue
        if msg == "Z-":
            robot.arm.move_ee_xyz([0, 0, -delta])
            continue
        if msg == "rX+":
            robot.arm.rot_ee_xyz(delta_angle, 'x', N=angle_N)
            continue
        if msg == "rX-":
            robot.arm.rot_ee_xyz(-delta_angle, 'x', N=angle_N)
            continue
        if msg == "rY+":
            robot.arm.rot_ee_xyz(delta_angle, 'y', N=angle_N)
            continue
        if msg == "rY-":
            robot.arm.rot_ee_xyz(-delta_angle, 'y', N=angle_N)
            continue
        if msg == "rZ+":
            robot.arm.rot_ee_xyz(delta_angle, 'z', N=angle_N)
            continue
        if msg == "rZ-":
            robot.arm.rot_ee_xyz(-delta_angle, 'z', N=angle_N)
            continue
        if msg == "OPEN":
            p.setJointMotorControl2(robot.arm.robot_id, finger_joint_id, p.VELOCITY_CONTROL, targetVelocity=1, force=finger_force)
            p.setJointMotorControl2(robot.arm.robot_id, finger_joint_id+1, p.VELOCITY_CONTROL, targetVelocity=1, force=finger_force)

            if cid is not None:
                p.removeConstraint(cid)
            continue
        if msg == "CLOSE":
            if obj_id is not None:
                for i in range(p.getNumJoints(robot.arm.robot_id)):
                    p.setCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=False)

            p.setJointMotorControl2(robot.arm.robot_id, finger_joint_id, p.VELOCITY_CONTROL, targetVelocity=-1, force=finger_force)
            p.setJointMotorControl2(robot.arm.robot_id, finger_joint_id+1, p.VELOCITY_CONTROL, targetVelocity=-1, force=finger_force)

            if obj_id is not None:
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
            continue
        if msg == "ON_RACK":
            safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id, linkIndexA=-1, linkIndexB=rack_link_id, enableCollision=False)
            robot.arm.move_ee_xyz(on_rack_offset)
            time.sleep(1.0)
            continue
        if msg == "OFF_RACK":
            safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id, linkIndexA=-1, linkIndexB=rack_link_id, enableCollision=False)
            robot.arm.move_ee_xyz(off_rack_offset)
            time.sleep(1.0)
            continue
        if msg == "ON_SHELF":
            safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id, linkIndexA=-1, linkIndexB=shelf_link_id, enableCollision=False)
            robot.arm.move_ee_xyz(on_shelf_offset)
            time.sleep(1.0)
            continue
        if msg == "OFF_SHELF":
            safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id, linkIndexA=-1, linkIndexB=shelf_link_id, enableCollision=False)
            robot.arm.move_ee_xyz(off_shelf_offset)
            time.sleep(1.0)
            continue
        if msg == "SAMPLE":
            worker_flag_dict[worker_id] = False
            time.sleep(1.0)
            robot.arm.reset(force_reset=True)
            robot.pb_client.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, False)

            print('Resetting robot!')
            p.changeDynamics(robot.arm.robot_id, right_pad_id, lateralFriction=1.0)
            p.changeDynamics(robot.arm.robot_id, left_pad_id, lateralFriction=1.0)

            cams = MultiCams(cfg.CAMERA, robot.pb_client, n_cams=cfg.N_CAMERAS)
            cam_info = {}
            cam_info['pose_world'] = []
            cam_info['pose_world_mat'] = []
            for cam in cams.cams:
                cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))
                cam_info['pose_world_mat'].append(cam.cam_ext_mat)

            # put table at right spot
            table_ori = euler2quat([0, 0, np.pi / 2])
            table_id = robot.pb_client.load_urdf(osp.join(path_util.get_ndf_descriptions(), 'hanging/table', table_urdf_file),
                                    cfg.TABLE_POS,
                                    table_ori,
                                    scaling=cfg.TABLE_SCALING)

            shapenet_id = global_dict['shapenet_id']
            print('\n\nUsing shapenet id: %s\n\n' % shapenet_id)
            upright_orientation = global_dict['upright_ori']
            obj_obj_file = global_dict['object_obj_file']
            obj_class = global_dict['object_class']
            have_rack = global_dict['have_rack']
            have_shelf = global_dict['have_shelf']

            table_link_id = -1
            if have_rack:
                rack_link_id = 0
                shelf_link_id = 1
                place_color = p.getVisualShapeData(table_id)[rack_link_id][7]
                place_link_id = rack_link_id
            else:
                rack_link_id = 0
                shelf_link_id = 0
                place_color = p.getVisualShapeData(table_id)[shelf_link_id][7]
                place_link_id = shelf_link_id

            # convert mesh with vhacd
            obj_obj_file_dec = obj_obj_file.split('.obj')[0] + '_dec.obj'
            if not osp.exists(obj_obj_file_dec):
                print('converting via VHACD')
                p.vhacd(
                    obj_obj_file,
                    obj_obj_file_dec,
                    'log.txt',
                    concavity=0.0025,
                    alpha=0.04,
                    beta=0.05,
                    gamma=0.00125,
                    minVolumePerCH=0.0001,
                    resolution=1000000,
                    depth=20,
                    planeDownsampling=4,
                    convexhullDownsampling=4,
                    pca=0,
                    mode=0,
                    convexhullApproximation=1
                )

            mesh_scale=[cfg.MESH_SCALE_DEFAULT] * 3
            pos = [np.random.random() * (x_high - x_low) + x_low, np.random.random() * (y_high - y_low) + y_low, table_z]
            if global_dict['fixed_angle']:
                ori = upright_orientation
            else:
                pose = util.list2pose_stamped(pos + upright_orientation)
                rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi, max_theta=np.pi)
                pose_w_yaw = util.transform_pose(pose, util.pose_from_matrix(rand_yaw_T))
                pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]

            print('loading from: ' + str(obj_obj_file_dec))
            obj_id = robot.pb_client.load_geom(
                'mesh',
                mass=0.01,
                mesh_scale=mesh_scale,
                visualfile=obj_obj_file_dec,
                collifile=obj_obj_file_dec,
                base_pos=pos,
                base_ori=ori)

            time.sleep(1.5)
            p.changeDynamics(obj_id, -1, lateralFriction=1.0, linearDamping=5, angularDamping=5)

            # get object point cloud
            rgb_imgs = []
            depth_imgs = []
            seg_depth_imgs = []
            seg_idxs = []
            obj_poses = []
            obj_pcd_pts = []
            table_pcd_pts = []
            rack_pcd_pts = []
            shelf_pcd_pts = []
            cam_poses = []
            pcd_raw = []

            obj_pose_world = p.getBasePositionAndOrientation(obj_id)
            obj_pose_world = util.list2pose_stamped(list(obj_pose_world[0]) + list(obj_pose_world[1]))
            print('object pose world: ', util.pose_stamped2list(obj_pose_world))
            cam_intrinsics = []

            # first go through and get unoccluded object observations
            hide_link(table_id, place_link_id)
            for i, cam in enumerate(cams.cams): 
                cam_int = cam.cam_int_mat
                cam_ext = cam.cam_ext_mat
                cam_intrinsics.append(cam_int)
                cam_poses.append(cam_ext)

                cam_pose_world = cam_info['pose_world'][i]
                # cam_poses.append(cam_info['pose_world_mat'][i])
                obj_pose_camera = util.convert_reference_frame(
                    pose_source=obj_pose_world,
                    pose_frame_source=util.unit_pose(),
                    pose_frame_target=cam_pose_world
                )
                obj_pose_camera_np = util.pose_stamped2np(obj_pose_camera)

                rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
                pts_raw, _ = cam.get_pcd(in_world=True, rgb_image=rgb, depth_image=depth, depth_min=0.0, depth_max=np.inf)
                pcd_raw.append(pts_raw)

                flat_seg = seg.flatten()
                flat_depth = depth.flatten()
                obj_inds = np.where(flat_seg == obj_id)

                table_val = table_id + ((table_link_id+1) << 24)
                table_inds = np.where(flat_seg == table_val)
                seg_depth = flat_depth[obj_inds[0]]  

                obj_pts = pts_raw[obj_inds[0], :]
                table_pts = pts_raw[table_inds[0], :]
                obj_pcd_pts.append(util.crop_pcd(obj_pts))
                table_pcd_pts.append(table_pts)
           
                # save sample
                obj_poses.append(obj_pose_camera_np)
                depth_imgs.append(depth)
                seg_depth_imgs.append(seg_depth)
                rgb_imgs.append(rgb)
                seg_idxs.append(obj_inds)
            
            # now go back through and also get point clouds representing your environment objects 
            print('Have shelf: %s' % have_shelf)
            if have_shelf:
                shelf_pose_world = np.concatenate(p.getLinkState(table_id, shelf_link_id)[:2]).tolist()
                print('Shelf pose world: ', shelf_pose_world)

            show_link(table_id, place_link_id, place_color)
            for i, cam in enumerate(cams.cams): 
                rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
                pts_raw, _ = cam.get_pcd(in_world=True, rgb_image=rgb, depth_image=depth, depth_min=0.0, depth_max=np.inf)

                flat_seg = seg.flatten()
                flat_depth = depth.flatten()

                table_val = table_id + ((table_link_id+1) << 24)
                rack_val = table_id + ((rack_link_id+1) << 24)
                shelf_val = table_id + ((shelf_link_id+1) << 24)
                table_inds = np.where(flat_seg == table_val)
                rack_inds = np.where(flat_seg == rack_val)
                shelf_inds = np.where(flat_seg == shelf_val)

                table_pts = pts_raw[table_inds[0], :]
                rack_pts = pts_raw[rack_inds[0], :]
                shelf_pts = pts_raw[shelf_inds[0], :]

                table_pcd_pts.append(table_pts)
                rack_pcd_pts.append(rack_pts)
                shelf_pcd_pts.append(shelf_pts)

            pix_3d = np.concatenate(obj_pcd_pts, axis=0)
            table_pix_3d = np.concatenate(table_pcd_pts, axis=0)
            shelf_pix_3d = np.concatenate(shelf_pcd_pts, axis=0)

            #robot.arm.go_home(ignore_physics=True)
            robot.arm.set_jpos(new_home, ignore_physics=True)
            work_queue.get()
            time.sleep(1.0)
            continue
        if msg == "DEMO_PICK":
            worker_flag_dict[worker_id] = False

            # save current pose
            ee_pose_world = np.concatenate(robot.arm.get_ee_pose()[:2]).tolist()
            robot_joints = robot.arm.get_jpos()
            obj_pose_world = np.concatenate(p.getBasePositionAndOrientation(obj_id)[:2]).tolist()
            gripper_closest_points = p.getClosestPoints(
                bodyA=obj_id, 
                bodyB=robot.arm.robot_id, 
                distance=0.0025,
                linkIndexA=-1, 
                linkIndexB=right_pad_id)

            # sort by distance to the object
            sorted(gripper_closest_points, key=lambda pt_info: pt_info[8])
            gripper_contact_pose = copy.deepcopy(ee_pose_world)
            if len(gripper_closest_points):
                for i, pt in enumerate(gripper_closest_points):
                    print(pt[8])
                gripper_contact_pose[:3] = np.asarray(gripper_closest_points[0][5])

            grasp_save_path = osp.join(save_dir, 'grasp_demo_' + str(shapenet_id) + '.npz')
            cur_demo_iter = 0
            while True:
                if osp.exists(grasp_save_path):
                    grasp_save_path = osp.join(save_dir, 'grasp_demo_' + str(shapenet_id) + '_%d.npz' % cur_demo_iter)
                    cur_demo_iter += 1
                else:
                    break
            print('saving to: %s' % grasp_save_path)
            np.savez(
                grasp_save_path,
                shapenet_id=shapenet_id,
                ee_pose_world=np.asarray(ee_pose_world),
                robot_joints=np.asarray(robot_joints),
                obj_pose_world=np.asarray(obj_pose_world),
                obj_pose_camera=obj_pose_camera_np,
                object_pointcloud=pix_3d,
                rgb=rgb_imgs,
                depth_full=depth_imgs,
                depth=seg_depth_imgs,
                seg=seg_idxs,
                camera_poses=cam_poses,
                obj_model_file=obj_obj_file,
                obj_model_file_dec=obj_obj_file_dec,
                gripper_pts=gripper_pts,
                gripper_pts_gaussian=gripper_pts_gaussian,
                gripper_pts_uniform=gripper_pts_uniform,
                gripper_contact_pose=gripper_contact_pose,
                table_urdf=table_urdf,
                pcd_raw=pcd_raw,
                cam_intrinsics=cam_intrinsics
            )

            time.sleep(1.0)
            continue
        if msg == "DEMO_PLACE":
            worker_flag_dict[worker_id] = False

            # save current pose
            ee_pose_world = np.concatenate(robot.arm.get_ee_pose()[:2]).tolist()
            robot_joints = robot.arm.get_jpos()
            obj_pose_world = np.concatenate(p.getBasePositionAndOrientation(obj_id)[:2]).tolist()
            try:
                rack_pose_world = np.concatenate(p.getLinkState(table_id, rack_link_id)[:2]).tolist()
            except:
                print('Couldn"t get rack pose, saving "None" for rack_pose_world. Please check URDF')
                rack_pose_world = None

            try:
                shelf_pose_world = np.concatenate(p.getLinkState(table_id, shelf_link_id)[:2]).tolist()
            except:
                print('Couldn"t get shelf pose, saving "None" for shelf_pose_world. Please check URDF')
                shelf_pose_world = None

            # get contact point / closest point position
            rack_closest_points = p.getClosestPoints(
                bodyA=obj_id, 
                bodyB=table_id, 
                distance=0.005,
                linkIndexA=-1,
                linkIndexB=rack_link_id)

            sorted(rack_closest_points, key=lambda pt_info: pt_info[8])
            rack_contact_pose = copy.deepcopy(rack_pose_world)
            if len(rack_closest_points):
                for i, pt in enumerate(rack_closest_points):
                    print(pt[8])
                rack_contact_pose[:3] = np.asarray(rack_closest_points[0][5])

            place_save_path = osp.join(save_dir, 'place_demo_' + str(shapenet_id) + '.npz')
            cur_demo_iter = 0
            while True:
                if osp.exists(place_save_path):
                    place_save_path = osp.join(save_dir, 'place_demo_' + str(shapenet_id) + '_%d.npz' % cur_demo_iter)
                    cur_demo_iter += 1
                else:
                    break
            print('saving to: %s' % place_save_path)
            np.savez(
                place_save_path,
                shapenet_id=shapenet_id,
                ee_pose_world=np.asarray(ee_pose_world),
                robot_joints=np.asarray(robot_joints),
                obj_pose_world=np.asarray(obj_pose_world),
                obj_pose_camera=obj_pose_camera_np,
                object_pointcloud=pix_3d,
                rgb=rgb_imgs,
                depth_full=depth_imgs,
                depth=seg_depth_imgs,
                seg=seg_idxs,
                camera_poses=cam_poses,
                obj_model_file=obj_obj_file,
                obj_model_file_dec=obj_obj_file_dec,
                gripper_pts=gripper_pts,
                rack_pointcloud_observed=rack_pcd_pts,
                rack_pointcloud_gt=rack_pts_gt,
                rack_pointcloud_gaussian=rack_pts_gaussian,
                rack_pointcloud_uniform=rack_pts_uniform,
                rack_pose_world=rack_pose_world,
                rack_contact_pose=rack_contact_pose,
                shelf_pose_world=shelf_pose_world,
                shelf_pointcloud_observed=shelf_pcd_pts,
                shelf_pointcloud_uniform=shelf_pts_uniform,
                shelf_pointcloud_gt=shelf_pts_gt,
                table_urdf=table_urdf,
                pcd_raw=pcd_raw,
                cam_intrinsics=cam_intrinsics
            )

            worker_flag_dict[worker_id] = True
            time.sleep(1.0)
            continue
        if msg == "SKIP":
            skipped_ids.append(shapenet_id)
            np.savez(osp.join(save_dir, 'demo_skipped_ids.npz'), ids=skipped_ids)
            worker_flag_dict[worker_id] = True
            time.sleep(1.0)
            continue
        if msg == "END":
            break
        time.sleep(0.001)
    print('Breaking Worker ID: ' + str(worker_id))
    child_conn.close()


class RobotTeleop:
    def __init__(self, work_queue, result_queue, global_manager, model_path, obj_class, seed=0):

        # thread/process for sending commands to the robot
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.global_manager = global_manager
        self.global_dict = self.global_manager.dict()
        self.global_dict['checkpoint_path'] = model_path
        np.random.seed(seed)
        random.seed(seed)
        self.global_dict['seed'] = seed
        self.global_dict['obj_class'] = obj_class
        self.worker_flag_dict = self.global_manager.dict()

        self.setup_workers(1)

        # main thread for accepting inputs from the keyboard
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            if key.char in KEY_MSG_MAP.keys():
                msg = KEY_MSG_MAP[key.char]
                for i, worker_id in enumerate(self._worker_ids):
                    self._pipes[worker_id]['parent'].send(msg)
            else:
                pass
        except AttributeError:
            pass

    def on_release(self, key):
        if key == keyboard.Key.esc:
            return False

    def setup_workers(self, num_workers):
        """Setup function to instantiate the desired number of
        workers. Pipes and Processes set up, stored internally,
        and started.
        Args:
            num_workers (int): Desired number of worker processes
        """
        worker_ids = np.arange(num_workers, dtype=np.int64).tolist()

        self._worker_ids = worker_ids

        self._pipes = {}
        self._processes = {}
        for i, worker_id in enumerate(self._worker_ids):
            parent, child = Pipe(duplex=True)
            self.worker_flag_dict[worker_id] = False
            proc = Process(
                target=worker_robot,
                args=(
                    child,
                    self.work_queue,
                    self.result_queue,
                    self.global_dict,
                    self.worker_flag_dict,
                    worker_id,
                )
            )
            pipe = {}
            pipe['parent'] = parent
            pipe['child'] = child

            self._pipes[worker_id] = pipe
            self._processes[worker_id] = proc

        for i, worker_id in enumerate(self._worker_ids):
            self._processes[worker_id].start()
            self._pipes[worker_id]['parent'].send('RESET')
            print('RESET WORKER ID: ' + str(worker_id))
        print('FINISHED WORKER SETUP')

    def sample_object(self):
        done = [False] * len(self._worker_ids)
        for i, worker_id in enumerate(self._worker_ids):
            self.work_queue.put(True)
            self._pipes[worker_id]['parent'].send('SAMPLE')

        # wait until all are done, which is when work queue is empty
        while True:
            time.sleep(0.001)
            if self.work_queue.empty():
                break

    def set_workers_ready(self):
        for worker_id in self._worker_ids:
            self.set_worker_ready(worker_id)

    def all_done(self):
        done = True
        for worker_id in self._worker_ids:
            done = done and self.get_worker_ready(worker_id)
        return done

    def get_pipes(self):
        return self._pipes

    def get_processes(self):
        return self._processes

    def get_worker_ids(self):
        return self._worker_ids

    def get_worker_ready(self, worker_id):
        return self.worker_flag_dict[worker_id]

    def set_worker_ready(self, worker_id):
        self.worker_flag_dict[worker_id] = True

    def get_global_info_dict(self):
        return self.global_dict

    def get_trial_number(self):
        return self.global_dict['trial']

    def get_obj_trial_number(self):
        return self.global_dict['trial_object']


if __name__ == "__main__":
    signal.signal(signal.SIGINT, util.signal_handler)
    work_queue = Queue()
    result_queue = Queue()
    mp_manager = Manager()

    teleop = RobotTeleop(work_queue, result_queue, mp_manager)
    print(KEY_MSG_MAP)

    k = 0

    time.sleep(1.0)
    while True:
        time.sleep(0.001)
    print('done')
