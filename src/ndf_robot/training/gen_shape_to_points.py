import os.path as osp
import numpy as np
from scipy.io import loadmat
import os.path as osp
import glob
from tqdm import tqdm
from scipy.ndimage import convolve
import pickle
import trimesh

from ndf_robot.utils import path_util

obj_class = 'mug'
#obj_class = 'bowl'
#obj_class = 'bottle'

if __name__ == "__main__":
    if obj_class == 'mug':
        # For mugs
        base_path = osp.join(path_util.get_ndf_data(), 'training/ShapeNetCore.v2/03797390')
    elif obj_class == 'bowl':
        # For bowls
        base_path = osp.join(path_util.get_ndf_data(), 'training/ShapeNetCore.v2/02880940')
    elif obj_class == 'bottle':
        # For bottle
        base_path = osp.join(path_util.get_ndf_data(), 'training/ShapeNetCore.v2/02876657')
    search_path = osp.join(base_path + "/*/", "models/model_normalized_128.mat")
    shapenet_paths = glob.glob(search_path)

    block = 128
    bs = 1 / block
    hbs = bs * 0.5
    y, z, x = np.meshgrid(np.linspace(-.5+hbs, .5-hbs, block), np.linspace(-.5+hbs, .5-hbs, block), np.linspace(-.5+hbs, .5-hbs, block))

    voxel = np.stack([z, x, -y], axis=-1)
    voxel = voxel.reshape((-1, 3))

    conv_weight = np.ones((5, 5, 5))

    parse_dict = {}

    for shapenet_path in tqdm(shapenet_paths):
        path = shapenet_path.replace("model_normalized_128.mat", "model_128_df.obj")

        voxel_bool = loadmat(shapenet_path)['voxel'].astype(np.bool)
        voxel_full = voxel_bool.astype(np.float32)

        voxel_surface_bool = voxel_bool.flatten()
        voxel_surface_coord = voxel

        voxel_full = voxel_bool.flatten()
        rix = np.random.permutation(voxel_full.shape[0])[:10000]

        voxel_pos = voxel[voxel_bool.flatten()]

        obj = trimesh.load(path)
        obj.vertices[:] = obj.vertices - obj.centroid
        point_min, point_max = obj.bounds

        pos_bound = voxel_pos.max(axis=0) - voxel_pos.min(axis=0)
        obj_bound = point_max - point_min

        scale_factor = obj_bound / pos_bound

        voxel_pos = voxel_pos * scale_factor[None, :]
        max_voxel_pos = voxel_pos.max(axis=0)

        pos_diff = point_max - max_voxel_pos

        voxel_pos = voxel_pos + pos_diff

        voxel_surface_coord = voxel_surface_coord * scale_factor[None, :] + pos_diff

        rix = np.random.permutation(voxel_surface_coord.shape[0])[:100000]
        coord = np.concatenate([voxel_surface_coord], axis=0)[rix]
        voxel_bool = np.concatenate([voxel_surface_bool[:, None]], axis=0)[rix]

        dict_key = shapenet_path.split('ShapeNetCore.v2')[-1]
        parse_dict[dict_key] = (coord, voxel_bool, voxel_pos)

    pickle.dump(parse_dict, open("shapenet_%s.p" % obj_class, "wb"))
