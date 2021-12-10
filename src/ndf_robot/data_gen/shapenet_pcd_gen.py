import os, os.path as osp
import logging
import random
import numpy as np
import time
import argparse
import copy
import signal
from multiprocessing import Process, Pipe, Manager
import psutil

import trimesh
import pybullet as p

from airobot.utils.pb_util import create_pybullet_client
from airobot.utils import common

from ndf_robot.config.default_data_gen_cfg import get_data_gen_cfg_defaults
from ndf_robot.utils import util, path_util
from ndf_robot.robot.multicam import MultiCams
from ndf_robot.utils.experiment_utils import DistractorSampler, DistractorObjManager, DistractorObjectEnv
from ndf_robot.share.globals import (
    SHAPENET_ID_DICT, bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list)


def worker_gen(child_conn, global_dict, worker_flag_dict, seed, worker_id):
    while True:
        try:
            if not child_conn.poll(0.0001):
                continue
            msg = child_conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if msg == "INIT":
            np.random.seed(seed)
            pb_client = create_pybullet_client(gui=False, opengl_render=True, realtime=True)

            # we need to turn off file caching so memory doesn't keep growing
            p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pb_client.get_client_id())

            cfg = get_data_gen_cfg_defaults()
            x_low = min(cfg.OBJ_SAMPLE_X_HIGH_LOW)
            x_high = max(cfg.OBJ_SAMPLE_X_HIGH_LOW)
            y_low = min(cfg.OBJ_SAMPLE_Y_HIGH_LOW)
            y_high = max(cfg.OBJ_SAMPLE_Y_HIGH_LOW)
            table_z = cfg.TABLE_Z
            
            cam_cfg = cfg.CAMERA
            cams = MultiCams(cam_cfg, pb_client, n_cams=args.n_cams)

            local_trial = 1

            proc = psutil.Process(os.getpid())
            continue
        if msg == "RESET":
            pb_client.resetSimulation()
    
            # put table at right spot
            table_ori = common.euler2quat([0, 0, np.pi / 2])
            table_id = pb_client.load_urdf('table/table.urdf',
                                    cfg.TABLE_POS,
                                    table_ori,
                                    scaling=cfg.TABLE_SCALING)

            continue
        if msg == "SAMPLE":
            pb_client.resetSimulation()

            # initialize cameras
            cams = MultiCams(cam_cfg, pb_client, n_cams=args.n_cams)
            cam_info = {}
            cam_info['pose_world'] = []
            for cam in cams.cams:
                cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))

            # put table at right spot
            table_ori = common.euler2quat([0, 0, np.pi / 2])
            table_id = pb_client.load_urdf('table/table.urdf',
                                    cfg.TABLE_POS,
                                    table_ori,
                                    scaling=cfg.TABLE_SCALING)  
            worker_flag_dict[worker_id] = False

            # get information about object that will be imported and rendered
            obj_class = global_dict['obj_class']
            shapenet_cat_id = SHAPENET_ID_DICT[obj_class]
            shapenet_id = global_dict['shapenet_id']
            upright_orientation = global_dict['upright_ori']
            obj_file_to_load = global_dict['object_load_obj_file']
            save_dir = global_dict['save_dir']
            
            # sample mesh position and orientation on the table
            scale_high, scale_low = cfg.MESH_SCALE_HIGH, cfg.MESH_SCALE_LOW
            rand_val = lambda high, low: np.random.random() * (high - low) + low
            if args.rand_scale:
                # can have non-isotropic scaling
                val1 = rand_val(scale_high, scale_low)
                val2 = rand_val(scale_high, scale_low)
                val3 = rand_val(scale_high, scale_low)
                sample = np.random.randint(5)
                if sample == 0:
                    mesh_scale = [val1] * 3
                elif sample == 1:
                    mesh_scale = [val1] * 2 + [val2] 
                elif sample == 2:
                    mesh_scale = [val1] + [val2] * 2
                elif sample == 3:
                    mesh_scale = [val1, val2, val1]
                elif sample == 4:
                    mesh_scale = [val1, val2, val3]
            else:
                mesh_scale=[cfg.MESH_SCALE_DEFAULT] * 3

            if args.same_pose and not args.any_pose:
                # if we should always have a canonical orientation
                pos = [np.mean([x_high, x_low]), np.mean([y_high, y_low]), table_z]
                ori = upright_orientation
            else:
                pos = [np.random.random() * (x_high - x_low) + x_low, np.random.random() * (y_high - y_low) + y_low, table_z]
                rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi, max_theta=np.pi)
                if args.any_pose:
                    # first compute all stable poses and then sample one of them
                    mesh = trimesh.load(obj_file_to_load)
                    stable_pose = random.sample(list(mesh.compute_stable_poses()[0]), 1)[0]
                    ori = util.pose_stamped2list(util.pose_from_matrix(stable_pose))[3:]
                    pose = util.list2pose_stamped(pos + ori)
                else:
                    pose = util.list2pose_stamped(pos + upright_orientation)
                pose_w_yaw = util.transform_pose(pose, util.pose_from_matrix(rand_yaw_T))
                pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]

            # in case we want to add extra objects that can act as occluders
            distractor_sampler = DistractorSampler(pb_client)
            distractor_objs_path = osp.join(path_util.get_ndf_obj_descriptions(), 'distractors/cuboids')
            distractor_manager = DistractorObjManager(distractor_objs_path, distractor_sampler, None, table_id)
            distractor_env = DistractorObjectEnv(cfg.DISTRACTOR, pb_client, None, distractor_manager, distractor_sampler)

            if args.occlude and np.random.random() > 0.5:
                n_occ = np.random.randint(1, args.max_occluders)
                distractor_env.sample_objects(n=n_occ)
                distractor_env.initialize_object_states(keep_away_region=pos[:-1])
                time.sleep(1.0)

            # load the object and change dynamics so it doesn't move as much 
            obj_id = pb_client.load_geom(
                'mesh', 
                mass=0.01, 
                mesh_scale=mesh_scale,
                visualfile=obj_file_to_load, 
                collifile=obj_file_to_load,
                base_pos=pos,
                base_ori=ori,
                rgba = [0.5, 0.2, 1, 1]) 
            p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)

            time.sleep(1.5)

            # get object pose with respect to the camera
            cam_poses = []
            cam_intrinsics = []
            depth_imgs = []
            seg_idxs = []
            obj_poses = []
            obj_pcd_pts = []
            uncropped_obj_pcd_pts = []
            table_pcd_pts = []

            obj_pose_world = p.getBasePositionAndOrientation(obj_id)
            obj_pose_world = util.list2pose_stamped(list(obj_pose_world[0]) + list(obj_pose_world[1]))
            obj_pose_world_np = util.pose_stamped2np(obj_pose_world)
            obj_velocity = p.getBaseVelocity(obj_id)
            for i, cam in enumerate(cams.cams): 
                cam_pose_world = cam_info['pose_world'][i]
                cam_poses.append(util.matrix_from_pose(cam_pose_world))
                cam_intrinsics.append(cam.cam_int_mat)
                obj_pose_camera = util.convert_reference_frame(
                    pose_source=obj_pose_world,
                    pose_frame_source=util.unit_pose(),
                    pose_frame_target=cam_pose_world
                )
                obj_pose_camera_np = util.pose_stamped2np(obj_pose_camera)

                rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
                pts_raw, _ = cam.get_pcd(in_world=True, rgb_image=rgb, depth_image=depth, depth_min=0, depth_max=np.inf)

                flat_seg = seg.flatten()
                flat_depth = depth.flatten()
                obj_inds = np.where(flat_seg == obj_id)
                table_inds = np.where(flat_seg == table_id)
                seg_depth = flat_depth[obj_inds[0]]  
                table_seg_depth = flat_depth[table_inds[0]]
                
                obj_pts = pts_raw[obj_inds[0], :]
                table_pts = pts_raw[table_inds[0], :][::int(table_inds[0].shape[0]/500)]
                obj_pcd_pts.append(util.crop_pcd(obj_pts))
                uncropped_obj_pcd_pts.append(obj_pts)
                table_pcd_pts.append(table_pts)
           
                obj_poses.append(obj_pose_camera_np)
                depth_imgs.append(seg_depth)
                seg_idxs.append(obj_inds)
            
            pix_3d = np.concatenate(obj_pcd_pts, axis=0)
            table_pix_3d = np.concatenate(table_pcd_pts, axis=0)

            if local_trial % 5 == 0:
                print('Local trial: {} for object index: {}\n\n'.format(local_trial, global_dict['object_loop_index']))

            if not (np.abs(obj_velocity) > 0.01).any():
                # only save the sample if the object wasn't moving, else the fused point clouds can be messed up
                num_samples = copy.deepcopy(global_dict['trial'])
                global_dict['trial'] += 1
                global_dict['trial_object'] += 1
                if global_dict['local_trial_start'] > 0:
                    local_trial = global_dict['local_trial_start']
                    global_dict['local_trial_start'] = 0
                local_trial += 1

                save_path = osp.join(save_dir, '{}_{}_{}.npz'.format(worker_id, local_trial, num_samples))

                np.savez(
                    save_path,
                    mesh_scale=mesh_scale,
                    vertex_offset=None, # legacy
                    object_pose_cam_frame=obj_poses,
                    depth_observation=depth_imgs,
                    object_segmentation=seg_idxs,
                    shapenet_id=shapenet_id,
                    shapenet_category_id=shapenet_cat_id,
                    point_cloud=None, # can save pix_3d if you want the 3D point cloud, not saving saves space
                    table_point_cloud=None, # can save table_pix_3d if you want the table point cloud, not saving saves space
                    obj_pose_world=obj_pose_world_np,
                    cam_pose_world=cam_poses,
                    cam_intrinsics=cam_intrinsics
                )

            else:
                print('\n\n\nobject was moving!!!\n\n\n')
                print(obj_velocity)
                time.sleep(2.0)

            pb_client.remove_body(obj_id)                         
            pb_client.remove_body(table_id)

            worker_flag_dict[worker_id] = True

            # pybullet has a weird bug that makes the RAM usage grow as this runs, we catch it by monitoring the RAM. Manually restart and resume the script when this breaks the process
            mem_usage_gb = proc.memory_info().rss / (1024.0**3)
            if mem_usage_gb > 1.8:
                logging.critical(f"\n\n\nMemory consumption too large, breaking at object {global_dict['object_loop_index']}, total samples {num_samples}, worker id {worker_id}\n\n\n")
                break
            child_conn.send('DONE')
            continue
        if msg == "END":
            break        
        time.sleep(0.001)
    print('Breaking Worker ID: ' + str(worker_id))
    child_conn.close()


class DataGenWorkerManager:
    def __init__(self, global_manager, num_workers=1):

        # thread/process for sending commands to the robot
        self.global_manager = global_manager
        self.global_dict = self.global_manager.dict()
        self.global_dict['trial'] = 0
        self.worker_flag_dict = self.global_manager.dict()        

        self.np_seed_base = 1
        self.setup_workers(num_workers)

    def setup_workers(self, num_workers):
        """Setup function to instantiate the desired number of
        workers. Pipes and Processes set up, stored internally,
        and started.
        Args:
            num_workers (int): Desired number of worker processes
        """
        worker_ids = np.arange(num_workers, dtype=np.int64).tolist()
        seeds = np.arange(self.np_seed_base, self.np_seed_base + num_workers, dtype=np.int64).tolist()

        self._worker_ids = worker_ids
        self.seeds = seeds

        self._pipes = {}
        self._processes = {}
        for i, worker_id in enumerate(self._worker_ids):
            parent, child = Pipe(duplex=True)
            self.worker_flag_dict[worker_id] = True
            proc = Process(
                target=worker_gen,
                args=(
                    child,
                    self.global_dict,
                    self.worker_flag_dict,
                    seeds[i],
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
            self._pipes[worker_id]['parent'].send('INIT')
            print('RESET WORKER ID: ' + str(worker_id))
        print('FINISHED WORKER SETUP')

    def sample_trials(self, total_num_trials):
        num_trials = self.get_obj_trial_number()
        while num_trials < total_num_trials:
            num_trials = self.get_obj_trial_number()
            for i, worker_id in enumerate(self._worker_ids):
                if self.get_worker_ready(worker_id):
                    self._pipes[worker_id]['parent'].send('SAMPLE')
                    self.worker_flag_dict[worker_id] = False
                    # self.work_queue.put(True)
            time.sleep(0.001)
        print('\n\n\n\nDone!\n\n\n\n')

    def get_pipes(self):
        return self._pipes

    def get_processes(self):
        return self._processes

    def get_worker_ids(self):
        return self._worker_ids

    def get_worker_ready(self, worker_id):
        return self.worker_flag_dict[worker_id]

    def get_global_info_dict(self):
        """Returns the globally shared dictionary of data
        generation information, including success rate and
        trial number

        Returns:
            dict: Dictionary of global information shared
                between workers
        """
        return self.global_dict

    def get_trial_number(self):
        return self.global_dict['trial']

    def get_obj_trial_number(self):
        return self.global_dict['trial_object']


def main(args):
    signal.signal(signal.SIGINT, util.signal_handler)
    obj_class = args.object_class
    shapenet_centered_models_dir = osp.join(path_util.get_ndf_obj_descriptions(), obj_class + '_centered_obj')

    save_dir = osp.join(path_util.get_ndf_data(), 'training_data', args.save_dir)
    util.safe_makedirs(save_dir)

    if 'bowl' in obj_class:
        bad_shapenet_ids_list = bad_shapenet_bowls_ids_list
    elif 'mug' in obj_class:
        bad_shapenet_ids_list = bad_shapenet_mug_ids_list
    elif 'bottle' in obj_class:
        bad_shapenet_ids_list = bad_shapenet_bottles_ids_list
    else:
        bad_shapenet_ids_list = []

    # get train samples
    objects_raw = os.listdir(shapenet_centered_models_dir) 
    objects_filtered = [fn for fn in objects_raw if fn not in bad_shapenet_ids_list]
    total_filtered = len(objects_filtered)
    train_n = int(total_filtered * 0.9)

    train_objects = sorted(objects_filtered)[:train_n]
    test_objects = sorted(objects_filtered)[train_n:]

    print('\n\n\nTest objects: ')
    print(test_objects)
    print('\n\n\n')

    # get specific number of samples to obtain
    if args.samples_per_object == 0:
        samples_per_object = args.total_samples / len(train_objects)
    else:
        samples_per_object = args.samples_per_object

    # write out these splits
    train_split_str = '\n'.join(train_objects)
    test_split_str = '\n'.join(test_objects)
    open(osp.join(path_util.get_ndf_share(), '%s_train_object_split.txt' % obj_class), 'w').write(train_split_str)
    open(osp.join(path_util.get_ndf_share(), '%s_test_object_split.txt' % obj_class), 'w').write(test_split_str)
    
    # set up processes
    mp_manager = Manager()
    manager = DataGenWorkerManager(mp_manager, num_workers=args.num_workers)

    manager.global_dict['shapenet_load_obj_dir'] = shapenet_centered_models_dir
    manager.global_dict['save_dir'] = save_dir
    manager.global_dict['trial'] = 0
    manager.global_dict['local_trial_start'] = 0
    manager.global_dict['obj_class'] = obj_class

    if args.resume_i > 0:
        train_objects = train_objects[args.resume_i:]

        files_int = [int(fname.split('.npz')[0].split('_')[-1]) for fname in os.listdir(save_dir)]
        start_trial = max(files_int)
        manager.global_dict['trial'] = start_trial
        manager.global_dict['local_trial_start'] = start_trial

    if args.end_i > 0:
        stop_object_idx = args.end_i
    else:
        stop_object_idx = len(train_objects)

    for i, train_object in enumerate(train_objects):
        print('object: ', train_object)
        print('i: ', i + args.resume_i)

        if obj_class in ['bottle', 'jar', 'bowl', 'mug']:
            upright_orientation = common.euler2quat([np.pi/2, 0, 0]).tolist()
        else:
            upright_orientation = common.euler2quat([0, 0, 0]).tolist()

        manager.global_dict['object_loop_index'] = i + args.resume_i
        manager.global_dict['shapenet_id'] = train_object
        manager.global_dict['upright_ori'] = upright_orientation
        manager.global_dict['object_load_obj_file'] = osp.join(shapenet_centered_models_dir, train_object, 'models/model_128_df.obj')
        manager.global_dict['trial_object'] = 0
        manager.sample_trials(samples_per_object)

        if i >= stop_object_idx:
            print('Stopping on object index: ', i)
            break

    for i, worker_id in enumerate(manager._worker_ids):
        manager.get_pipes()[worker_id]['parent'].send('END')    

    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_samples', type=int, default=10000)
    parser.add_argument('--object_class', type=str, default='mug')
    parser.add_argument('--save_dir', type=str, default='data/debug_save')
    parser.add_argument('--metadata_dir', type=str, default='metadata')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--resume_i', type=int, default=0)
    parser.add_argument('--end_i', type=int, default=0)
    parser.add_argument('--occlude', action='store_true')
    parser.add_argument('--same_pose', action='store_true')
    parser.add_argument('--rand_scale', action='store_true')
    parser.add_argument('--n_cams', type=int, default=4)
    parser.add_argument('--max_occluders', type=int, default=4)
    parser.add_argument('--samples_per_object', type=int, default=0)
    parser.add_argument('--any_pose', action='store_true')

    args = parser.parse_args()
    main(args)
