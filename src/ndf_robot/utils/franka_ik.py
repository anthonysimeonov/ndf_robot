#!/usr/bin/env python

from __future__ import print_function
import os, os.path as osp
import copy
import sys
pb_planning_src = os.environ['PB_PLANNING_SOURCE_DIR']
sys.path.append(pb_planning_src)
import pybullet as p

from pybullet_tools.utils import add_data_path, connect, dump_body, disconnect, wait_for_user, \
    get_movable_joints, get_sample_fn, set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, pairwise_collision, set_client, get_client, pairwise_link_collision, \
    plan_joint_motion

from pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
FRANKA_URDF = osp.join(pb_planning_src, FRANKA_URDF)
print('FRANKA URDF: ', FRANKA_URDF)
from pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver

from airobot.utils import common
from ndf_robot.utils import util, path_util

class FrankaIK:
    def __init__(self, gui=True, base_pos=[0, 0, 1]):
        if gui:
            set_client(0)
        else:
            set_client(1)
        connect(use_gui=gui)
        self.pb_client = get_client()
        add_data_path()
        draw_pose(Pose(), length=1.)
        set_camera_pose(camera_point=[1, -1, 1])

        with LockRenderer():
            with HideOutput(True):
                self.robot = load_pybullet(FRANKA_URDF, base_pos=base_pos, fixed_base=True)
                assign_link_colors(self.robot, max_colors=3, s=0.5, v=1.)

        dump_body(self.robot)

        self.info = PANDA_INFO
        self.tool_link = link_from_name(self.robot, 'panda_hand')
        draw_pose(Pose(), parent=self.robot, parent_link=self.tool_link)
        self.movable_joints = get_movable_joints(self.robot)
        print('Joints', [get_joint_name(self.robot, joint) for joint in self.movable_joints])
        check_ik_solver(self.info)

        self.ik_joints = get_ik_joints(self.robot, self.info, self.tool_link)

        self.home_joints = [-0.19000000996229238,
                            0.0799999292867887,
                            0.22998567421354038,
                            -2.4299997925910426,
                            0.030000057559800147,
                            2.519999744925224,
                            0.859999719845722]

        self.panda_ignore_pairs_initial = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (6, 8),
            (7, 8), (7, 9), (7, 10), (7, 11),
            (8, 9), (8, 10), (8, 11),
            (9, 10), (9, 11),
            (10, 11)
        ]

        # symmetric
        self.panda_ignore_pairs = []
        for (i, j) in self.panda_ignore_pairs_initial:
            self.panda_ignore_pairs.append((i, j))
            self.panda_ignore_pairs.append((j, i))
        self._setup_self(ignore_link_pairs=self.panda_ignore_pairs)

        set_joint_positions(self.robot, self.ik_joints, self.home_joints)
        self.obstacle_dict = {} 

        self._grasp_target_to_ee = [0, 0, -0.105, 0, 0, 0, 1]
        self._ee_to_grasp_target = [0, 0, 0.105, 0, 0, 0, 1]

    def set_jpos(self, jnts):
        set_joint_positions(self.robot, self.ik_joints, jnts)

    def load_urdf(self, urdf_path, pos, ori, scale=1.0, collision=True, name=None):
        body_id = load_pybullet(urdf_path, base_pos=pos, base_ori=ori, scale=scale)
        if collision:
            if name is None:
                name = str(len(self.obstacle_dict) + 1)
            self.add_collision_bodies({name: body_id})
        return body_id

    def add_collision_bodies(self, bodies={}):
        if not len(bodies):
            return

        for k, v in bodies.items():
            self.obstacle_dict[k] = v
    
    def remove_collision_bodies(self, names):
        for name in names:
            if name in self.obstacle_dict.keys() and name != 'self':
                del self.obstacle_dict[name]

    def clear_collision_bodies(self):
        self.obstacle_dict = {}

    def _setup_self(self, ignore_link_pairs=[]):
        """Setup the internal information regarding the robot joints, links to
        consider for self-collision checking, and

        Args:
            ignore_link_pairs (list, optional): List of tuples. Each tuple indicates
                a pair of robot links that should NOT be considered when checking
                self collisions . Defaults to [].
        """
        # setup self-collision link pairs
        self.check_self_coll_pairs = []
        for i in range(p.getNumJoints(self.robot, physicsClientId=self.pb_client)):
            for j in range(p.getNumJoints(self.robot, physicsClientId=self.pb_client)):
                # don't check link colliding with itself, and ignore specified links
                if i != j and (i, j) not in ignore_link_pairs:
                    self.check_self_coll_pairs.append((i, j))
        
    def check_self_collision(self):
        # * self-collision link check
        for link1, link2 in self.check_self_coll_pairs:
            if pairwise_link_collision(self.robot, link1, self.robot, link2):
                return True, 1
        return False, 0

    def check_collision(self):
        # self_collision = any_link_pair_collision(self.robot, None, self.robot, None)
        self_collision = self.check_self_collision()[0]
        if self_collision:
            return True, 'self'
        
        for name, obstacle in self.obstacle_dict.items():
            collision = pairwise_collision(self.robot, obstacle)
            if collision:
                return True, name
        return False, None

    def _convert_to_ee(self, pose_list):
        pose_list = util.pose_stamped2list(util.convert_reference_frame(
            pose_source=util.list2pose_stamped(self._grasp_target_to_ee),
            pose_frame_target=util.unit_pose(),
            pose_frame_source=util.list2pose_stamped(pose_list)
        ))
        return pose_list

    def get_ik(self, pose_list, execute=False, target_link=True, *args, **kwargs):
        """ASSUMES THE POSE IS OF THE LINK "PANDA_GRASPTARGET"!!!
        We do an internal conversion here to express this as the pose of 
        the end effector link that this IK can use, which is panda_hand

        Args:
            pose_list (list): [x, y, z, x, y, z, w]
            execute (bool, optional): If yes, set the joints to this value. Defaults to False.

        Returns:
            list: Joint values
        """
        if target_link:
            # convert the pose we get to the pose of our EE
            old_pose_list = copy.deepcopy(pose_list)
            pose_list = self._convert_to_ee(pose_list)

        pos, ori = pose_list[:3], pose_list[3:] # check quat convention
        pose = (tuple(pos), tuple(ori))
        conf = next(either_inverse_kinematics(self.robot, self.info, self.tool_link, pose, 
                                              max_distance=None, max_time=0.5, max_candidates=250), None)
        if conf is None:
            print('Failure!')
            return None 
        
        if execute:
            set_joint_positions(self.robot, self.ik_joints, conf)
        return conf

    def get_feasible_ik(self, pose_list, max_attempts=100, verbose=False, target_link=True):
        if target_link:
            pose_list = self._convert_to_ee(pose_list)
        
        # and iterate over output of IK
        pos, ori = pose_list[:3], pose_list[3:] 
        pose = (tuple(pos), tuple(ori))
        confs = either_inverse_kinematics(self.robot, self.info, self.tool_link, pose, 
                                                    max_distance=None, max_time=0.5, max_candidates=250)
        for conf in confs:
            set_joint_positions(self.robot, self.ik_joints, conf)
            collision_info = self.check_collision()
            if not collision_info[0]:
                return conf 
            else:
                if verbose:
                    print('Collision with body: %s' % collision_info[1])
        print('Failed to get feasible IK')
        return None
    
    def plan_joint_motion(self, start, goal, alg='rrt_star', max_time=5.0):
        self.set_jpos(start)
        plan = plan_joint_motion(
            self.robot, self.ik_joints, goal, obstacles=self.obstacle_dict.values(), self_collisions=True, 
            disabled_collisions=set(self.panda_ignore_pairs), algorithm=alg, max_time=max_time)
        return plan

    def _retract(self):
        sample_fn = get_sample_fn(self.robot, self.movable_joints)
        for i in range(10):
            print('Iteration:', i)
            conf = sample_fn()
            set_joint_positions(self.robot, self.movable_joints, conf)
            self._test_retraction(use_pybullet=False, max_distance=0.1, max_time=0.5, max_candidates=100)

    def _test_retraction(self, distance=0.1, **kwargs):
        ik_joints = get_ik_joints(self.robot, self.info, self.tool_link)
        start_pose = get_link_pose(self.robot, self.tool_link)
        end_pose = multiply(start_pose, Pose(Point(z=-distance)))
        handles = [add_line(point_from_pose(start_pose), point_from_pose(end_pose), color=BLUE)]
        path = []
        pose_path = list(interpolate_poses(start_pose, end_pose, pos_step_size=0.01))
        for i, pose in enumerate(pose_path):
            print('Waypoint: {}/{}'.format(i+1, len(pose_path)))
            handles.extend(draw_pose(pose))
            conf = next(either_inverse_kinematics(self.robot, self.info, self.tool_link, pose, **kwargs), None)
            if conf is None:
                print('Failure!')
                path = None
                wait_for_user()
                break
            set_joint_positions(self.robot, ik_joints, conf)
            path.append(conf)
            wait_for_user()
        remove_handles(handles)
        return path


