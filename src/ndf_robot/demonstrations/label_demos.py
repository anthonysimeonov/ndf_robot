import os, os.path as osp
import random
import numpy as np
import time
import argparse
from multiprocessing import Queue, Manager
import signal

from airobot.utils import common

from ndf_robot.utils import util, path_util
from ndf_robot.demonstrations.teleop import RobotTeleop


def main(args):
    np.random.seed(args.np_seed)
    random.seed(args.np_seed)
    signal.signal(signal.SIGINT, util.signal_handler)

    obj_class = args.object_class
    shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(), obj_class + '_centered_obj_normalized')

    save_dir = osp.join(path_util.get_ndf_data(), 'demos', obj_class, args.exp)
    util.safe_makedirs(save_dir)

    # use the training split we saved from data generation
    train_objects = np.loadtxt(osp.join(path_util.get_ndf_share(), '%s_train_object_split.txt' % obj_class), dtype=str).tolist()

    if args.single_instance:
        args.shapenet_id = random.sample(train_objects, 1)[0]
        args.force_shapenet_id = True
    num_samples = args.num_samples
    sample_objects = random.sample(train_objects, num_samples)

    # setup teleop and robot interface
    work_queue = Queue()
    result_queue = Queue()
    mp_manager = Manager()

    manager = RobotTeleop(work_queue, result_queue, mp_manager, None, obj_class=obj_class, seed=args.np_seed)

    manager.global_dict['shapenet_obj_dir'] = shapenet_obj_dir
    manager.global_dict['save_dir'] = save_dir
    manager.global_dict['trial'] = 0
    manager.global_dict['object_class'] = obj_class
    manager.global_dict['num_samples'] = 0
    manager.global_dict['config'] = args.config

    have_rack = args.with_rack
    have_shelf = args.with_shelf
    if (have_rack and have_shelf) or (not have_rack and not have_shelf):
        print('"have_rack" and "have_shelf" cannot both be True/False. Defaulting to "have_rack" equal True')
        have_rack = True

    manager.global_dict['have_rack'] = have_rack
    manager.global_dict['have_shelf'] = have_shelf
    manager.global_dict['fixed_angle'] = args.fixed_angle

    if osp.exists(osp.join(save_dir, 'demo_skipped_ids.npz')):
        skipped_ids = np.load(osp.join(save_dir, 'demo_skipped_ids.npz'))['ids'].tolist()
    else:
        skipped_ids = []

    for i, train_object in enumerate(sample_objects):
        print('object: ', train_object)
        print('i: ', i)

        already_saved_samples = os.listdir(save_dir)
        skip_obj = False
        for fname in already_saved_samples:
            if train_object in fname:
                print('already got sample for this object -- fname: %s, skipping to next one' % fname)
                skip_obj = True 
                break
        if train_object in skipped_ids:
            skip_obj = True

        if skip_obj:
            continue

        # sample the metadata

        if obj_class in ['bowl', 'bottle', 'mug']:
            upright_orientation = common.euler2quat([np.pi/2, 0, 0])
        else:
            upright_orientation = common.euler2quat([0, 0, 0])

        if args.force_shapenet_id:
            demo_shapenet_id = args.shapenet_id
        else:
            demo_shapenet_id = train_object
        manager.global_dict['shapenet_id'] = demo_shapenet_id
        manager.global_dict['upright_ori'] = upright_orientation.tolist()
        manager.global_dict['object_obj_file'] = osp.join(shapenet_obj_dir, demo_shapenet_id, 'models/model_normalized.obj')
        manager.global_dict['trial_object'] = 0

        time.sleep(3.0)
        # load mesh into scene
        manager.sample_object()
        time.sleep(1.0)

        # teleop the robot to the grasp location, run demo, and save sample
        print('Perform the task using the keyboard')
        running = True
        while running:
            if manager.all_done():
                break
            time.sleep(0.001)

        if manager.global_dict['num_samples'] > num_samples:
            break
    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--exp', type=str, default='debug_label')
    parser.add_argument('--metadata_dir', type=str, default='metadata')
    parser.add_argument('--object_class', type=str, default='mug')
    parser.add_argument('--np_seed', type=int, default=0)
    parser.add_argument('--config', type=str, default='base_config')
    parser.add_argument('--force_shapenet_id', action='store_true')
    parser.add_argument('--shapenet_id', type=str, default='none')
    parser.add_argument('--single_instance', action='store_true')
    parser.add_argument('--fixed_angle', action='store_true')
    parser.add_argument('--with_rack', action='store_true')
    parser.add_argument('--with_shelf', action='store_true')

    args = parser.parse_args()
    main(args)
