import os.path as osp
import torch
import numpy as np
import trimesh
import random
import argparse
import copy
from scipy.spatial.transform import Rotation

from ndf_robot.utils import path_util
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.eval.ndf_alignment import NDFAlignmentCheck


if __name__ == '__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_recon', action='store_true')
    parser.add_argument('--sigma', type=float, default=0.025)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)

    # see the demo object descriptions folder for other object models you can try
    obj_model1 = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/28f1e7bc572a633cb9946438ed40eeb9/models/model_normalized.obj')
    obj_model2 = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/586e67c53f181dc22adf8abaa25e0215/models/model_normalized.obj')
    model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_demo_mug_weights.pth')  

    scale1 = 0.25
    scale2 = 0.4
    mesh1 = trimesh.load(obj_model1, process=False)
    mesh1.apply_scale(scale1)
    mesh2 = trimesh.load(obj_model2, process=False) # different instance, different scaling
    mesh2.apply_scale(scale2)
    # mesh2 = trimesh.load(obj_model1, process=False)  # use same object model to debug SE(3) equivariance
    # mesh2.apply_scale(scale1)

    # apply a random initial rotation to the new shape
    quat = np.random.random(4)
    quat = quat / np.linalg.norm(quat)
    rot = np.eye(4)
    rot[:-1, :-1] = Rotation.from_quat(quat).as_matrix()
    mesh2.apply_transform(rot)

    if args.visualize:
        show_mesh1 = mesh1.copy()
        show_mesh2 = mesh2.copy()

        offset = 0.1
        show_mesh1.apply_translation([-1.0 * offset, 0, 0])
        show_mesh2.apply_translation([offset, 0, 0])

        scene = trimesh.Scene()
        scene.add_geometry([show_mesh1, show_mesh2])
        scene.show()

    pcd1 = mesh1.sample(5000)
    pcd2 = mesh2.sample(5000)  # point cloud representing different shape
    # pcd2 = copy.deepcopy(pcd1)  # debug with the exact same point cloud
    # pcd2 = mesh1.sample(5000)  # debug with same shape but different sampled points

    model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=True).cuda()
    model.load_state_dict(torch.load(model_path))

    ndf_alignment = NDFAlignmentCheck(model, pcd1, pcd2, sigma=args.sigma, trimesh_viz=args.visualize)
    ndf_alignment.sample_pts(show_recon=args.show_recon, render_video=args.video)
