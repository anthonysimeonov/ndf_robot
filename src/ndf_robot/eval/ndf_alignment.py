import os, os.path as osp
import torch
import numpy as np
import trimesh
import random
import copy
import plotly.graph_objects as go

from ndf_robot.utils import torch_util, trimesh_util
from ndf_robot.utils.plotly_save import plot3d


class NDFAlignmentCheck:
    def __init__(self, model, pcd1, pcd2, model_type='pointnet', opt_iterations=500, sigma=0.025, trimesh_viz=False):
        self.model = model
        self.model_type = model_type
        self.opt_iterations = opt_iterations
        self.sigma = sigma
        self.trimesh_viz = trimesh_viz

        self.perturb_scale = 0.001
        self.perturb_decay = 0.5
        self.n_pts =  1500
        self.n_opt_pts = 500
        self.prepare_inputs(pcd1, pcd2)

        self.loss_fn = torch.nn.L1Loss()
        if torch.cuda.is_available():
            self.dev = torch.device('cuda:0')
        else:
            self.dev = torch.device('cpu')
        self.model = self.model.to(self.dev)
        self.model.eval()

        self.viz_path = 'visualization'
        if not osp.exists(self.viz_path):
            os.makedirs(self.viz_path)

        self.video_viz_path = 'vid_visualization'
        if not osp.exists(self.video_viz_path):
            os.makedirs(self.video_viz_path)

        self._cam_frame_scene_dict()

    def _cam_frame_scene_dict(self):
        self.cam_frame_scene_dict = {}
        cam_up_vec = [0, 1, 0]
        plotly_camera = {
            'up': {'x': cam_up_vec[0], 'y': cam_up_vec[1],'z': cam_up_vec[2]},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'eye': {'x': -0.6, 'y': -0.6, 'z': 0.4},
        }

        plotly_scene = {
            'xaxis': 
                {
                    'backgroundcolor': 'rgb(255, 255, 255)',
                    'gridcolor': 'white',
                    'zerolinecolor': 'white',
                    'tickcolor': 'rgb(255, 255, 255)',
                    'showticklabels': False,
                    'showbackground': False,
                    'showaxeslabels': False,
                    'visible': False,
                    'range': [-0.5, 0.5]},
            'yaxis': 
                {
                    'backgroundcolor': 'rgb(255, 255, 255)',
                    'gridcolor': 'white',
                    'zerolinecolor': 'white',
                    'tickcolor': 'rgb(255, 255, 255)',
                    'showticklabels': False,
                    'showbackground': False,
                    'showaxeslabels': False,
                    'visible': False,
                    'range': [-0.5, 0.5]},
            'zaxis': 
                {
                    'backgroundcolor': 'rgb(255, 255, 255)',
                    'gridcolor': 'white',
                    'zerolinecolor': 'white',
                    'tickcolor': 'rgb(255, 255, 255)',
                    'showticklabels': False,
                    'showbackground': False,
                    'showaxeslabels': False,
                    'visible': False,
                    'range': [-0.5, 0.5]},
        }
        self.cam_frame_scene_dict['camera'] = plotly_camera
        self.cam_frame_scene_dict['scene'] = plotly_scene

    def plotly_create_local_frame(self, transform=None, length=0.03):
        if transform is None:
            transform = np.eye(4)

        x_vec = transform[:-1, 0] * length
        y_vec = transform[:-1, 1] * length
        z_vec = transform[:-1, 2] * length

        origin = transform[:-1, -1]

        lw = 8
        x_data = go.Scatter3d(
            x=[origin[0], x_vec[0] + origin[0]], y=[origin[1], x_vec[1] + origin[1]], z=[origin[2], x_vec[2] + origin[2]],
            line=dict(
                color='red',
                width=lw
            ),
            marker=dict(
                size=0.0
            )
        )
        y_data = go.Scatter3d(
            x=[origin[0], y_vec[0] + origin[0]], y=[origin[1], y_vec[1] + origin[1]], z=[origin[2], y_vec[2] + origin[2]],
            line=dict(
                color='green',
                width=lw
            ),
            marker=dict(
                size=0.0
            )
        )
        z_data = go.Scatter3d(
            x=[origin[0], z_vec[0] + origin[0]], y=[origin[1], z_vec[1] + origin[1]], z=[origin[2], z_vec[2] + origin[2]],
            line=dict(
                color='blue',
                width=lw
            ),
            marker=dict(
                size=0.0
            )
        )
        # fig = go.Figure(data=[x_data, y_data, z_data])
        # fig.show()

        data = [x_data, y_data, z_data]
        return data

    def prepare_inputs(self, pcd1, pcd2):
        pcd1 = pcd1 - np.mean(pcd1, axis=0)
        pcd2 = pcd2 - np.mean(pcd2, axis=0)
        self.pcd1 = pcd1
        self.pcd2 = pcd2

        tpcd1 = trimesh.PointCloud(self.pcd1[:self.n_pts])
        tpcd2 = trimesh.PointCloud(self.pcd2[:self.n_pts])
        # tpcd1.show()
        # tpcd2.show()

    def sample_pts(self, show_recon=False, return_scene=False, visualize_all_inits=False, render_video=False):
        # sample query points
        query_pts = np.random.normal(0.0, self.sigma, size=(self.n_opt_pts, 3))
        
        # put the query points at one of the points in the point cloud
        q_offset_ind = np.random.randint(self.pcd1.shape[0])
        q_offset = self.pcd1[q_offset_ind]
        q_offset *= 1.2
        reference_query_pts = query_pts + q_offset

        reference_model_input = {}
        ref_query_pts = torch.from_numpy(reference_query_pts[:self.n_opt_pts]).float().to(self.dev)
        ref_shape_pcd = torch.from_numpy(self.pcd1[:self.n_pts]).float().to(self.dev)
        reference_model_input['coords'] = ref_query_pts[None, :, :]
        reference_model_input['point_cloud'] = ref_shape_pcd[None, :, :]

        # get the descriptors for these reference query points
        reference_latent = self.model.extract_latent(reference_model_input).detach()
        reference_act_hat = self.model.forward_latent(reference_latent, reference_model_input['coords']).detach()

        # set up the optimization
        if 'dgcnn' in self.model_type:
            full_opt = 5   # dgcnn can't fit 10 initialization in memory
        else:
            full_opt = 10
        best_loss = np.inf
        best_tf = np.eye(4)
        best_idx = 0
        tf_list = []
        M = full_opt

        trans = (torch.rand((M, 3)) * 0.1).float().to(self.dev)
        rot = torch.rand(M, 3).float().to(self.dev)
        trans.requires_grad_()
        rot.requires_grad_()
        opt = torch.optim.Adam([trans, rot], lr=1e-2)

        rand_rot_init = (torch.rand((M, 3)) * 2*np.pi).float().to(self.dev)
        rand_mat_init = torch_util.angle_axis_to_rotation_matrix(rand_rot_init)
        rand_mat_init = rand_mat_init.squeeze().float().to(self.dev)

        # now randomly initialize a copy of the query points
        opt_query_pts = torch.from_numpy(query_pts).float().to(self.dev)
        opt_query_pts = opt_query_pts[:self.n_opt_pts][None, :, :].repeat((M, 1, 1))
        X = torch_util.transform_pcd_torch(opt_query_pts, rand_mat_init)

        opt_model_input = {}
        opt_model_input['coords'] = X

        mi_point_cloud = []
        for ii in range(M):
            mi_point_cloud.append(torch.from_numpy(self.pcd2[:self.n_pts]).float().to(self.dev))
        mi_point_cloud = torch.stack(mi_point_cloud, 0)
        opt_model_input['point_cloud'] = mi_point_cloud
        opt_latent = self.model.extract_latent(opt_model_input).detach()

        loss_values = []
        vid_plot_idx = None  # we will set this during the optimization

        # run optimization
        pcd_traj_list = {}
        for jj in range(M):
            pcd_traj_list[jj] = []
            pcd_traj_list[jj].append(np.mean(X[jj].detach().cpu().numpy(), axis=0))
        for i in range(self.opt_iterations):
            T_mat = torch_util.angle_axis_to_rotation_matrix(rot).squeeze()
            noise_vec = (torch.randn(X.size()) * (self.perturb_scale / ((i+1)**(self.perturb_decay)))).to(self.dev)
            X_perturbed = X + noise_vec
            X_new = torch_util.transform_pcd_torch(X_perturbed, T_mat) + trans[:, None, :].repeat((1, X.size(1), 1))

            ######################### stuff for visualizing the reconstruction ##################33

            for jj in range(M):
                X_np = X_new[jj].detach().cpu().numpy()
                centroid = np.mean(X_np, axis=0)
                pcd_traj_list[jj].append(centroid)

            if i == 0 and show_recon:
                jj = 0
                shape_mi = {}
                shape_mi['point_cloud'] = opt_model_input['point_cloud'][jj][None, :, :].detach()
                shape_np = shape_mi['point_cloud'].cpu().numpy().squeeze()
                shape_mean = np.mean(shape_np, axis=0)
                inliers = np.where(np.linalg.norm(shape_np - shape_mean, 2, 1) < 0.2)[0]
                shape_np = shape_np[inliers]
                shape_pcd = trimesh.PointCloud(shape_np)
                bb = shape_pcd.bounding_box
                bb_scene = trimesh.Scene(); bb_scene.add_geometry([shape_pcd, bb]) 
                # bb_scene.show()

                eval_pts = bb.sample_volume(10000)
                shape_mi['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(self.dev).detach()
                out = self.model(shape_mi)
                thresh = 0.1
                in_inds = torch.where(out['occ'].squeeze() > thresh)[0].cpu().numpy()
                out_inds = torch.where(out['occ'].squeeze() < thresh)[0].cpu().numpy()

                in_pts = eval_pts[in_inds]
                out_pts = eval_pts[out_inds]
                if self.trimesh_viz:
                    scene = trimesh_util.trimesh_show([in_pts])
                    in_scene = trimesh_util.trimesh_show([in_pts, shape_np])
                self._cam_frame_scene_dict()
                plot3d(
                    [in_pts, shape_np],
                    ['blue', 'black'], 
                    osp.join(self.viz_path, 'recon_overlay_test.html'),
                    scene_dict=self.cam_frame_scene_dict,
                    z_plane=False)

            ###############################################################################

            act_hat = self.model.forward_latent(opt_latent, X_new)
            t_size = reference_act_hat.size()

            losses = [self.loss_fn(act_hat[ii].view(t_size), reference_act_hat) for ii in range(M)]

            loss = torch.mean(torch.stack(losses))
            if i % 100 == 0:
                losses_str = ['%f' % val.item() for val in losses]
                loss_str = ', '.join(losses_str)
                print(f'i: {i}, losses: {loss_str}')
            loss_values.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i > 5 and (vid_plot_idx is None):
                # try to guess which run in the batch will lead to lowest cost
                vid_plot_idx = torch.argmin(torch.stack(losses)).item()
                print('vid plot idx: ', vid_plot_idx)
                plot_rand_mat_np = rand_mat_init[vid_plot_idx].detach().cpu().numpy() 
            
            if i < 200:
                render_iter = (i % 4 == 0)
            else:
                render_iter = (i % 8 == 0)

            if render_video and render_iter and (vid_plot_idx is not None):
                # save image for each iteration
                transform_mat_np = T_mat[vid_plot_idx].detach().cpu().numpy()
                transform_mat_np[:-1, -1] = trans[vid_plot_idx].detach().cpu().numpy()
                frame_tf = np.matmul(transform_mat_np, plot_rand_mat_np)

                frame_data = self.plotly_create_local_frame(transform=frame_tf)
                plot3d(
                    [self.pcd2, X_new[vid_plot_idx].detach().cpu().numpy(), np.asarray(pcd_traj_list[vid_plot_idx])],
                    ['blue', 'black', 'red'], 
                    osp.join(self.video_viz_path, 'opt_iter_%d.png' % i),
                    scene_dict=self.cam_frame_scene_dict,
                    z_plane=False,
                    extra_data=frame_data)

        best_idx = torch.argmin(torch.stack(losses)).item()
        best_loss = losses[best_idx]
        print('best loss: %f, best_idx: %d' % (best_loss, best_idx))

        best_X = X_new[best_idx].detach().cpu().numpy()

        offset = np.array([0.4, 0, 0])
        vpcd1 = copy.deepcopy(self.pcd1)
        vquery1 = copy.deepcopy(reference_query_pts)

        vpcd1 += offset
        vquery1 += offset

        trans_best, rot_best = trans[best_idx], rot[best_idx]
        transform_mat_np = torch_util.angle_axis_to_rotation_matrix(rot_best.view(1, -1)).squeeze().detach().cpu().numpy()
        transform_mat_np[:-1, -1] = trans_best.detach().cpu().numpy()
        rand_mat_np = rand_mat_init[best_idx].detach().cpu().numpy() 

        frame1_tf = np.eye(4)
        frame1_tf[:-1, -1] = (q_offset + offset)
        frame2_tf = np.matmul(transform_mat_np, rand_mat_np)

        frame1 = self.plotly_create_local_frame(transform=frame1_tf)
        frame2 = self.plotly_create_local_frame(transform=rand_mat_np)
        frame3 = self.plotly_create_local_frame(transform=frame2_tf)
        frame_data = frame1 + frame2 + frame3
        plot3d(
            [vpcd1, vquery1, self.pcd2, best_X, np.asarray(pcd_traj_list[best_idx])],
            ['purple', 'black', 'blue', 'black', 'red'], 
            osp.join(self.viz_path, 'best_alignment.html'),
            scene_dict=self.cam_frame_scene_dict,
            z_plane=False,
            extra_data=frame_data)

        best_scene = None
        if self.trimesh_viz:
            local_frame_1 = trimesh.creation.axis(origin_size=0.002, transform=frame1_tf, origin_color=None, axis_radius=None, axis_length=0.03)
            local_frame_2 = trimesh.creation.axis(origin_size=0.002, transform=rand_mat_np, origin_color=None, axis_radius=None, axis_length=0.03)
            local_frame_3 = trimesh.creation.axis(origin_size=0.002, transform=frame2_tf, origin_color=None, axis_radius=None, axis_length=0.03)
             
            best_scene = trimesh_util.trimesh_show([vpcd1, vquery1 , self.pcd2, best_X, pcd_traj_list[best_idx]], show=False)
            best_scene.add_geometry([local_frame_1, local_frame_2, local_frame_3])
            best_scene.show()

            # all the runs from different initializations
        if visualize_all_inits:
            for j in range(M):
                trans_j, rot_j = trans[j], rot[j]
                transform_mat_np = torch_util.angle_axis_to_rotation_matrix(rot_j.view(1, -1)).squeeze().detach().cpu().numpy()
                transform_mat_np[:-1, -1] = trans_j.detach().cpu().numpy()

                rand_mat_np = rand_mat_init[j].detach().cpu().numpy() 

                frame1_tf = np.eye(4)
                frame1_tf[:-1, -1] = (q_offset + offset)
                frame2_tf = np.matmul(transform_mat_np, rand_mat_np)

                X_show = X_new[j].detach().cpu().numpy()

                frame1 = self.plotly_create_local_frame(transform=frame1_tf)
                frame2 = self.plotly_create_local_frame(transform=rand_mat_np)
                frame3 = self.plotly_create_local_frame(transform=frame2_tf)
                frame_data = frame1 + frame2 + frame3
                plot3d(
                    [vpcd1, vquery1, self.pcd2, X_show, np.asarray(pcd_traj_list[j])],
                    ['purple', 'black', 'blue', 'black', 'red'], 
                    osp.join(self.viz_path, 'alignment_%d.html' % j),
                    scene_dict=self.cam_frame_scene_dict,
                    z_plane=False,
                    extra_data=frame_data)

                if self.trimesh_viz:
                    local_frame_1 = trimesh.creation.axis(origin_size=0.002, transform=frame1_tf, origin_color=None, axis_radius=None, axis_length=0.03)
                    local_frame_2 = trimesh.creation.axis(origin_size=0.002, transform=rand_mat_np, origin_color=None, axis_radius=None, axis_length=0.03)
                    local_frame_3 = trimesh.creation.axis(origin_size=0.002, transform=frame2_tf, origin_color=None, axis_radius=None, axis_length=0.03)
                    
                    scene = trimesh_util.trimesh_show([vpcd1, vquery1 , self.pcd2, X_show, pcd_traj_list[j]], show=False)
                    scene.add_geometry([local_frame_1, local_frame_2, local_frame_3])
                    scene.show()
        if return_scene:
            return best_scene