import os, os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import trimesh

from airobot import log_info, log_warn, log_debug, log_critical

from ndf_robot.utils import util, torch_util, trimesh_util, torch3d_util
from ndf_robot.utils.plotly_save import plot3d


class OccNetOptimizer:
    def __init__(self, model, query_pts, query_pts_real_shape=None, opt_iterations=250, 
                 noise_scale=0.0, noise_decay=0.5, single_object=False):
        self.model = model
        self.model_type = self.model.model_type
        self.query_pts_origin = query_pts 
        if query_pts_real_shape is None:
            self.query_pts_origin_real_shape = query_pts
        else:
            self.query_pts_origin_real_shape = query_pts_real_shape

        self.loss_fn =  torch.nn.L1Loss()
        if torch.cuda.is_available():
            self.dev = torch.device('cuda:0')
        else:
            self.dev = torch.device('cpu')

        if self.model is not None:
            self.model = self.model.to(self.dev)
            self.model.eval()

        self.opt_iterations = opt_iterations

        self.noise_scale = noise_scale
        self.noise_decay = noise_decay

        # if this is true, we will use the activations from the demo with the same shape as in test time
        # defaut is false, because we want to test optimizing with the set of demos that don't include
        # the test shape
        self.single_object = single_object 
        self.target_info = None
        self.demo_info = None
        if self.single_object:
            log_warn('\n\n**** SINGLE OBJECT SET TO TRUE, WILL *NOT* USE A NEW SHAPE AT TEST TIME, AND WILL EXPECT TARGET INFO TO BE SET****\n\n')

        self.debug_viz_path = 'debug_viz'
        self.viz_path = 'visualization'
        util.safe_makedirs(self.debug_viz_path)
        util.safe_makedirs(self.viz_path)
        self.viz_files =  []

        self.rot_grid = util.generate_healpix_grid(size=1e6)
        # self.rot_grid = None

    def _scene_dict(self):
        self.scene_dict = {}
        plotly_scene = {
            'xaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
            'yaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
            'zaxis': {'nticks': 16, 'range': [-0.5, 0.5]}
        }
        self.scene_dict['scene'] = plotly_scene

    def set_demo_info(self, demo_info):
        """Function to set the information for a set of multiple demonstrations

        Args:
            demo_info (list): Contains the information for the demos
        """
        self.demo_info = demo_info

    def set_target(self, target_info):
        """
        Function to set the information about the task via the target activations
        """
        self.target_info = target_info

    def _get_query_pts_rs(self):
        # convert query points to camera frame
        query_pts_world_rs = torch.from_numpy(self.query_pts_origin_real_shape).float().to(self.dev)

        # convert query points to centered camera frame
        query_pts_world_rs_mean = query_pts_world_rs.mean(0)

        # center the query based on the it's shape mean, so that we perform optimization starting with the query at the origin
        query_pts_cam_cent_rs = query_pts_world_rs - query_pts_world_rs_mean
        query_pts_tf = np.eye(4)
        query_pts_tf[:-1, -1] = -query_pts_world_rs_mean.cpu().numpy()

        query_pts_tf_rs = query_pts_tf
        return query_pts_cam_cent_rs, query_pts_tf_rs

    def optimize_transform_implicit(self, shape_pts_world_np, ee=True, *args, **kwargs):
        """
        Function to optimzie the transformation of our query points, conditioned on
        a set of shape points observed in the world

        Args:
            shape_pts_world (np.ndarray): N x 3 array representing 3D point cloud of the object
                to be manipulated, expressed in the world coordinate system
        """
        dev = self.dev
        n_pts = 1500
        opt_pts = 500 
        perturb_scale = self.noise_scale
        perturb_decay = self.noise_decay

        if self.single_object:
            assert self.target_info is not None, 'Target info not set! Need to set the targets for single object optimization'

        ##### obtain the activations from the demos ####
        demo_feats_list = []
        demo_latents_list = []
        for i in range(len(self.demo_info)):
            # load in information from target
            demo_shape_pts_world = self.demo_info[i]['demo_obj_pts']
            demo_query_pts_world = self.demo_info[i]['demo_query_pts']
            demo_shape_pts_world = torch.from_numpy(demo_shape_pts_world).float().to(self.dev)
            demo_query_pts_world = torch.from_numpy(demo_query_pts_world).float().to(self.dev)

            demo_shape_pts_mean = demo_shape_pts_world.mean(0)
            demo_shape_pts_cent = demo_shape_pts_world - demo_shape_pts_mean
            demo_query_pts_cent = demo_query_pts_world - demo_shape_pts_mean
            demo_query_pts_cent_perturbed = demo_query_pts_cent + (torch.randn(demo_query_pts_cent.size()) * perturb_scale).to(dev)

            rndperm = torch.randperm(demo_shape_pts_cent.size(0))
            demo_model_input = dict(
                point_cloud=demo_shape_pts_cent[None, rndperm[:n_pts], :], 
                coords=demo_query_pts_cent_perturbed[None, :opt_pts, :])
            out = self.model(demo_model_input)
            # target_act_hat = out['features'].detach()
            target_latent = self.model.extract_latent(demo_model_input).detach()
            target_act_hat = self.model.forward_latent(target_latent, demo_model_input['coords']).detach()

            demo_feats_list.append(target_act_hat.squeeze())
            demo_latents_list.append(target_latent.squeeze())
        target_act_hat_all = torch.stack(demo_feats_list, 0)
        target_act_hat = torch.mean(target_act_hat_all, 0)

        ######################################################################

        # convert shape pts to camera frame
        shape_pts_world = torch.from_numpy(shape_pts_world_np).float().to(self.dev)
        shape_pts_mean = shape_pts_world.mean(0)
        shape_pts_cent = shape_pts_world - shape_pts_mean

        # convert query points to camera frame, and center the query based on the it's shape mean, so that we perform optimization starting with the query at the origin
        query_pts_world = torch.from_numpy(self.query_pts_origin).float().to(self.dev)
        query_pts_mean = query_pts_world.mean(0)
        query_pts_cent = query_pts_world - query_pts_mean

        query_pts_tf = np.eye(4)
        query_pts_tf[:-1, -1] = -query_pts_mean.cpu().numpy()

        if 'dgcnn' in self.model_type:
            full_opt = 5   # dgcnn can't fit 10 initialization in memory
        else:
            full_opt = 10
        best_loss = np.inf
        best_tf = np.eye(4)
        best_idx = 0
        tf_list = []
        M = full_opt

        trans = (torch.rand((M, 3)) * 0.1).float().to(dev)
        rot = torch.rand(M, 3).float().to(dev)
        # rot_idx = np.random.randint(self.rot_grid.shape[0], size=M)
        # rot = torch3d_util.matrix_to_axis_angle(torch.from_numpy(self.rot_grid[rot_idx])).float()

        # rand_rot_init = (torch.rand((M, 3)) * 2*np.pi).float().to(dev)
        rand_rot_idx = np.random.randint(self.rot_grid.shape[0], size=M)
        rand_rot_init = torch3d_util.matrix_to_axis_angle(torch.from_numpy(self.rot_grid[rand_rot_idx])).float()
        rand_mat_init = torch_util.angle_axis_to_rotation_matrix(rand_rot_init)
        rand_mat_init = rand_mat_init.squeeze().float().to(dev)

        query_pts_cam_cent_rs, query_pts_tf_rs = self._get_query_pts_rs()
        X_rs = query_pts_cam_cent_rs[:opt_pts][None, :, :].repeat((M, 1, 1))

        # set up optimization
        X = query_pts_cent[:opt_pts][None, :, :].repeat((M, 1, 1))
        X = torch_util.transform_pcd_torch(X, rand_mat_init)
        X_rs = torch_util.transform_pcd_torch(X_rs, rand_mat_init)

        mi_point_cloud = []
        for ii in range(M):
            rndperm = torch.randperm(shape_pts_cent.size(0))
            mi_point_cloud.append(shape_pts_cent[rndperm[:n_pts]])
        mi_point_cloud = torch.stack(mi_point_cloud, 0)
        mi = dict(point_cloud=mi_point_cloud)
        shape_mean_trans = np.eye(4)
        shape_mean_trans[:-1, -1] = shape_pts_mean.cpu().numpy()
        shape_pts_world_np = shape_pts_world.cpu().numpy()

        rot.requires_grad_()
        trans.requires_grad_()
        full_opt = torch.optim.Adam([trans, rot], lr=1e-2)
        full_opt.zero_grad()

        loss_values = []

        # set up model input with shape points and the shape latent that will be used throughout
        mi['coords'] = X
        latent = self.model.extract_latent(mi).detach()

        # run optimization
        pcd_traj_list = {}
        for jj in range(M):
            pcd_traj_list[jj] = []
        for i in range(self.opt_iterations):
            T_mat = torch_util.angle_axis_to_rotation_matrix(rot).squeeze()
            noise_vec = (torch.randn(X.size()) * (perturb_scale / ((i+1)**(perturb_decay)))).to(dev)
            X_perturbed = X + noise_vec
            X_new = torch_util.transform_pcd_torch(X_perturbed, T_mat) + trans[:, None, :].repeat((1, X.size(1), 1))

            ######################### visualize the reconstruction ##################33

            # for jj in range(M):
            if i == 0:
                jj = 0
                shape_mi = {}
                shape_mi['point_cloud'] = mi['point_cloud'][jj][None, :, :].detach()
                shape_np = shape_mi['point_cloud'].cpu().numpy().squeeze()
                shape_mean = np.mean(shape_np, axis=0)
                inliers = np.where(np.linalg.norm(shape_np - shape_mean, 2, 1) < 0.2)[0]
                shape_np = shape_np[inliers]
                shape_pcd = trimesh.PointCloud(shape_np)
                bb = shape_pcd.bounding_box
                bb_scene = trimesh.Scene(); bb_scene.add_geometry([shape_pcd, bb]) 

                eval_pts = bb.sample_volume(10000)
                shape_mi['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(self.dev).detach()
                out = self.model(shape_mi)
                thresh = 0.3
                in_inds = torch.where(out['occ'].squeeze() > thresh)[0].cpu().numpy()

                in_pts = eval_pts[in_inds]
                self._scene_dict()
                plot3d(
                    [in_pts, shape_np],
                    ['blue', 'black'], 
                    osp.join(self.debug_viz_path, 'recon_overlay.html'),
                    scene_dict=self.scene_dict,
                    z_plane=False)

            ###############################################################################

            act_hat = self.model.forward_latent(latent, X_new)
            t_size = target_act_hat.size()

            losses = [self.loss_fn(act_hat[ii].view(t_size), target_act_hat) for ii in range(M)]
            loss = torch.mean(torch.stack(losses))
            if i % 100 == 0:
                losses_str = ['%f' % val.item() for val in losses]
                loss_str = ', '.join(losses_str)
                log_debug(f'i: {i}, losses: {loss_str}')
            loss_values.append(loss.item())
            full_opt.zero_grad()
            loss.backward()
            full_opt.step()

        best_idx = torch.argmin(torch.stack(losses)).item()
        best_loss = losses[best_idx]
        log_debug('best loss: %f, best_idx: %d' % (best_loss, best_idx))

        for j in range(M):
            trans_j, rot_j = trans[j], rot[j]
            transform_mat_np = torch_util.angle_axis_to_rotation_matrix(rot_j.view(1, -1)).squeeze().detach().cpu().numpy()
            transform_mat_np[:-1, -1] = trans_j.detach().cpu().numpy()

            rand_query_pts_tf = np.matmul(rand_mat_init[j].detach().cpu().numpy(), query_pts_tf)
            transform_mat_np = np.matmul(transform_mat_np, rand_query_pts_tf)
            transform_mat_np = np.matmul(shape_mean_trans, transform_mat_np)

            ee_pts_world = util.transform_pcd(self.query_pts_origin_real_shape, transform_mat_np)

            all_pts = [ee_pts_world, shape_pts_world_np]
            opt_fname = 'ee_pose_optimized_%d.html' % j if ee else 'rack_pose_optimized_%d.html' % j
            plot3d(
                all_pts, 
                ['black', 'purple'], 
                osp.join('visualization', opt_fname), 
                z_plane=False)
            self.viz_files.append(osp.join('visualization', opt_fname))

            if ee:
                T_mat = transform_mat_np
            else:
                T_mat = np.linalg.inv(transform_mat_np)
            tf_list.append(T_mat)

        return tf_list, best_idx
