import numpy as np
import torchvision
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt


def occupancy(model, model_input, ground_truth, model_output, writer, iter, prefix=""):
    """Writes tensorboard summaries using tensorboardx api.

    :param writer: tensorboardx writer object.
    :param predictions: Output of forward pass.
    :param ground_truth: Ground truth.
    :param iter: Iteration number.
    :param prefix: Every summary will be prefixed with this string.
    """
    pred_occ = model_output['occ']
    gt_occ = ground_truth['occ']
    coords = model_input['coords'][:, 0]
    depth = model_input['depth']
    # pixel_coords = model_output['pixel_coords']

    writer.add_image(prefix + "gt_depth",
                    torchvision.utils.make_grid(depth[:, :1].repeat(1, 3, 1, 1),
                                                 scale_each=True,
                                                 normalize=True).cpu().detach().numpy(),
                     iter)

    writer.add_scalar(prefix + "out_min", pred_occ.min(), iter)
    writer.add_scalar(prefix + "out_max", pred_occ.max(), iter)

    writer.add_scalar(prefix + "trgt_min", gt_occ.min(), iter)
    writer.add_scalar(prefix + "trgt_max", gt_occ.max(), iter)

    writer.add_scalar(prefix + "depth_min", depth.min(), iter)
    writer.add_scalar(prefix + "depth_max", depth.max(), iter)

    # writer.add_scalar(prefix + "pixel_coords_min", pixel_coords.min(), iter)
    # writer.add_scalar(prefix + "pixel_coords_max", pixel_coords.max(), iter)
    input_coords = coords[:1].detach().cpu().numpy()
    gt_occ_coords = coords[:1, gt_occ[0].squeeze(-1) > 0, :].detach().cpu().numpy()
    pred_occ_coords = coords[:1, pred_occ[0].squeeze(-1) > 0.5, :].detach().cpu().numpy()

    # Compute colors for point predictions
    all_colors = np.ones_like(input_coords)
    all_colors[:, :, 1:] = 0.
    corr_mask = (gt_occ[0].squeeze(-1) > 0) == (pred_occ[0].squeeze(-1) > 0.5)
    all_colors[0, corr_mask.detach().cpu().numpy()] = np.array([[0., 1., 0.]])
    pred_occ_colors = all_colors[:1, pred_occ[0].squeeze(-1).detach().cpu().numpy() > 0]

    # point_cloud(writer, iter, 'input_coords', input_coords)
    point_cloud(writer, iter, 'ground_truth_coords', gt_occ_coords)
    point_cloud(writer, iter, 'predicted_coords', pred_occ_coords, colors=pred_occ_colors)


def occupancy_net(model, model_input, ground_truth, model_output, writer, iter, prefix=""):
    """Writes tensorboard summaries using tensorboardx api.

    :param writer: tensorboardx writer object.
    :param predictions: Output of forward pass.
    :param ground_truth: Ground truth.
    :param iter: Iteration number.
    :param prefix: Every summary will be prefixed with this string.
    """
    pred_occ = model_output['occ'][:, :, None]
    gt_occ = ground_truth['occ']
    coords = model_input['coords']

    writer.add_scalar(prefix + "out_min", pred_occ.min(), iter)
    writer.add_scalar(prefix + "out_max", pred_occ.max(), iter)

    writer.add_scalar(prefix + "trgt_min", gt_occ.min(), iter)
    writer.add_scalar(prefix + "trgt_max", gt_occ.max(), iter)

    # writer.add_scalar(prefix + "pixel_coords_min", pixel_coords.min(), iter)
    # writer.add_scalar(prefix + "pixel_coords_max", pixel_coords.max(), iter)
    input_coords = coords[:1].detach().cpu().numpy()
    gt_occ_coords = coords[:1, gt_occ[0].squeeze(-1) > 0, :].detach().cpu().numpy()
    pred_occ_coords = coords[:1, pred_occ[0].squeeze(-1) > 0.5, :].detach().cpu().numpy()

    # Compute colors for point predictions
    all_colors = np.ones_like(input_coords)
    all_colors[:, :, 1:] = 0.
    corr_mask = (gt_occ[0].squeeze(-1) > 0) == (pred_occ[0].squeeze(-1) > 0.5)
    all_colors[0, corr_mask.detach().cpu().numpy()] = np.array([[0., 1., 0.]])
    pred_occ_colors = all_colors[:1, pred_occ[0].squeeze(-1).detach().cpu().numpy() > 0]

    # point_cloud(writer, iter, 'input_coords', input_coords)
    point_cloud(writer, iter, 'ground_truth_coords', gt_occ_coords)
    point_cloud(writer, iter, 'predicted_coords', pred_occ_coords, colors=pred_occ_colors)


def distance_net(model, model_input, ground_truth, model_output, writer, iter, prefix=""):
    """Writes tensorboard summaries using tensorboardx api.

    :param writer: tensorboardx writer object.
    :param predictions: Output of forward pass.
    :param ground_truth: Ground truth.
    :param iter: Iteration number.
    :param prefix: Every summary will be prefixed with this string.
    """
    pred_occ = model_output['occ'][:, :, None]
    gt_occ = ground_truth['occ']
    coords = model_input['coords']

    writer.add_scalar(prefix + "out_min", pred_occ.min(), iter)
    writer.add_scalar(prefix + "out_max", pred_occ.max(), iter)

    writer.add_scalar(prefix + "trgt_min", gt_occ.min(), iter)
    writer.add_scalar(prefix + "trgt_max", gt_occ.max(), iter)

    # writer.add_scalar(prefix + "pixel_coords_min", pixel_coords.min(), iter)
    # writer.add_scalar(prefix + "pixel_coords_max", pixel_coords.max(), iter)
    input_coords = coords[:1].detach().cpu().numpy()
    gt_occ_coords = coords[:1, gt_occ[0].squeeze(-1) < 0.02, :].detach().cpu().numpy()
    pred_occ_coords = coords[:1, pred_occ[0].squeeze(-1) < 0.02, :].detach().cpu().numpy()

    # Compute colors for point predictions
    all_colors = np.ones_like(input_coords)
    all_colors[:, :, 1:] = 0.
    corr_mask = (gt_occ[0].squeeze(-1) < 0.1) == (pred_occ[0].squeeze(-1) < 0.02)
    all_colors[0, corr_mask.detach().cpu().numpy()] = np.array([[0., 1., 0.]])
    pred_occ_colors = all_colors[:1, pred_occ[0].squeeze(-1).detach().cpu().numpy() < 0.02]

    # point_cloud(writer, iter, 'input_coords', input_coords)
    point_cloud(writer, iter, 'ground_truth_coords', gt_occ_coords)
    point_cloud(writer, iter, 'predicted_coords', pred_occ_coords, colors=pred_occ_colors)


def semantic(model, model_input, ground_truth, model_output, writer, iter, prefix=""):
    pred_occ = model_output['occ']
    gt_occ = ground_truth['occ']
    coords = model_input['coords'][:, 0]
    depth = model_input['depth']

    writer.add_image(prefix + "input_depth",
                torchvision.utils.make_grid(depth[:, :1].repeat(1, 3, 1, 1),
                                                 scale_each=True,
                                                 normalize=True).cpu().detach().numpy(),
                     iter)

    writer.add_scalar(prefix + "out_min", pred_occ.min(), iter)
    writer.add_scalar(prefix + "out_max", pred_occ.max(), iter)

    writer.add_scalar(prefix + "trgt_min", gt_occ.min(), iter)
    writer.add_scalar(prefix + "trgt_max", gt_occ.max(), iter)

    writer.add_scalar(prefix + "depth_min", depth.min(), iter)
    writer.add_scalar(prefix + "depth_max", depth.max(), iter)

    input_coords = coords[0].detach().cpu().numpy()

    if len(gt_occ.size()) == 2:
        gt_occ_coords = coords[0, gt_occ[0].squeeze(-1) > 0, :].detach().cpu().numpy()
    else:
        gt_occ_coords = coords[0, :, :].detach().cpu().numpy()

    pred_occ_coords = coords[0, pred_occ[0].squeeze(-1) > 0.5, :].detach().cpu().numpy()

    plt.switch_backend('agg')
    fig = plt.figure()
    min_coords = input_coords.min(axis=0)
    max_coords = input_coords.max(axis=0)

    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(input_coords[..., 0], input_coords[..., 1], input_coords[..., 2])

    ax.set_xlim([min_coords[0], max_coords[0]])
    ax.set_ylim([min_coords[1], max_coords[1]])
    ax.set_zlim([min_coords[2], max_coords[2]])

    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(gt_occ_coords[..., 0], gt_occ_coords[..., 1], gt_occ_coords[..., 2])

    ax.set_xlim([min_coords[0], max_coords[0]])
    ax.set_ylim([min_coords[1], max_coords[1]])
    ax.set_zlim([min_coords[2], max_coords[2]])

    ax = fig.add_subplot(133, projection='3d')
    ax.scatter(pred_occ_coords[..., 0], pred_occ_coords[..., 1], pred_occ_coords[..., 2])

    ax.set_xlim([min_coords[0], max_coords[0]])
    ax.set_ylim([min_coords[1], max_coords[1]])
    ax.set_zlim([min_coords[2], max_coords[2]])

    writer.add_figure(prefix + 'pred_occupancies', fig, global_step=iter)


def semantic_occupancy(model, model_input, ground_truth, model_output, writer, iter, prefix=""):
    pred_occ = model_output['occ']
    gt_occ = ground_truth['occ']
    coords = model_input['coords']

    writer.add_scalar(prefix + "out_min", pred_occ.min(), iter)
    writer.add_scalar(prefix + "out_max", pred_occ.max(), iter)

    writer.add_scalar(prefix + "trgt_min", gt_occ.min(), iter)
    writer.add_scalar(prefix + "trgt_max", gt_occ.max(), iter)

    input_coords = coords[0].detach().cpu().numpy()

    if len(gt_occ.size()) == 2:
        gt_occ_coords = coords[0, gt_occ[0].squeeze(-1) > 0, :].detach().cpu().numpy()
    else:
        gt_occ_coords = coords[0, :, :].detach().cpu().numpy()

    # if prefix == "val_":
    #     import pdb
    #     pdb.set_trace()
    #     print((pred_occ > 0.5).float().mean())

    pred_occ_coords = coords[0, pred_occ[0].squeeze(-1) > 0.5, :].detach().cpu().numpy()

    plt.switch_backend('agg')
    fig = plt.figure()
    min_coords = input_coords.min(axis=0)
    max_coords = input_coords.max(axis=0)

    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(input_coords[..., 0], input_coords[..., 1], input_coords[..., 2])

    ax.set_xlim([min_coords[0], max_coords[0]])
    ax.set_ylim([min_coords[1], max_coords[1]])
    ax.set_zlim([min_coords[2], max_coords[2]])

    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(gt_occ_coords[..., 0], gt_occ_coords[..., 1], gt_occ_coords[..., 2])

    ax.set_xlim([min_coords[0], max_coords[0]])
    ax.set_ylim([min_coords[1], max_coords[1]])
    ax.set_zlim([min_coords[2], max_coords[2]])

    ax = fig.add_subplot(133, projection='3d')
    ax.scatter(pred_occ_coords[..., 0], pred_occ_coords[..., 1], pred_occ_coords[..., 2])

    ax.set_xlim([min_coords[0], max_coords[0]])
    ax.set_ylim([min_coords[1], max_coords[1]])
    ax.set_zlim([min_coords[2], max_coords[2]])

    writer.add_figure(prefix + 'pred_occupancies', fig, global_step=iter)

def point_cloud(writer, iter, name, points_xyz, colors=None):
    point_size_config = {
        'material': {
            'cls': 'PointsMaterial',
            'size': 0.05
        }
    }

    if colors is None:
       colors = np.zeros_like(points_xyz)

    writer.add_mesh(name, vertices=points_xyz, colors=colors,
                     config_dict={"material": point_size_config}, global_step=iter)
