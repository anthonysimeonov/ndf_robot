import numpy as np
import torch
from torch.utils.data import Dataset
import random
import glob
import os.path as osp
from scipy.spatial.transform import Rotation
import pickle

from ndf_robot.utils import path_util, geometry


class JointOccTrainDataset(Dataset):
    def __init__(self, sidelength, depth_aug=False, multiview_aug=False, phase='train', obj_class='all'):

        # Path setup (change to folder where your training data is kept)
        ## these are the names of the full dataset folders
        mug_path = osp.join(path_util.get_ndf_data(), 'training_data/mug_table_all_pose_4_cam_half_occ_full_rand_scale')
        bottle_path = osp.join(path_util.get_ndf_data(), 'training/bottle_table_all_pose_4_cam_half_occ_full_rand_scale')
        bowl_path = osp.join(path_util.get_ndf_data(), 'training/bowl_table_all_pose_4_cam_half_occ_full_rand_scale')

        ## these are the names of the mini-dataset folders, to ensure everything is up and running
        # mug_path = osp.join(path_util.get_ndf_data(), 'training_data/test_mug')
        # bottle_path = osp.join(path_util.get_ndf_data(), 'training_data/test_bottle')
        # bowl_path = osp.join(path_util.get_ndf_data(), 'training_data/test_bowl')

        if obj_class == 'all':
            paths = [mug_path, bottle_path, bowl_path]
        else:
            paths = []
            if 'mug' in obj_class:
                paths.append(mug_path)
            if 'bowl' in obj_class:
                paths.append(bowl_path)
            if 'bottle' in obj_class:
                paths.append(bottle_path)

        print('Loading from paths: ', paths)

        files_total = []
        for path in paths:
            files = list(sorted(glob.glob(path+"/*.npz")))
            n = len(files)
            idx = int(0.9 * n)

            if phase == 'train':
                files = files[:idx]
            else:
                files = files[idx:]

            files_total.extend(files)

        self.files = files_total

        self.sidelength = sidelength
        self.depth_aug = depth_aug
        self.multiview_aug = multiview_aug

        block = 128 
        bs = 1 / block
        hbs = bs * 0.5
        self.bs = bs
        self.hbs = hbs

        self.shapenet_mug_dict = pickle.load(open(osp.join(path_util.get_ndf_data(), 'training_data/occ_shapenet_mug.p'), 'rb'))
        self.shapenet_bowl_dict = pickle.load(open(osp.join(path_util.get_ndf_data(), 'training_data/occ_shapenet_bowl.p'), "rb"))
        self.shapenet_bottle_dict = pickle.load(open(osp.join(path_util.get_ndf_data(), 'training_data/occ_shapenet_bottle.p'), "rb"))

        self.shapenet_dict = {'03797390': self.shapenet_mug_dict, '02880940': self.shapenet_bowl_dict, '02876657': self.shapenet_bottle_dict}

        self.projection_mode = "perspective"

        self.cache_file = None
        self.count = 0

        print("files length ", len(self.files))

    def __len__(self):
        return len(self.files)

    def get_item(self, index):
        try:
            data = np.load(self.files[index], allow_pickle=True)
            posecam =  data['object_pose_cam_frame']  # legacy naming, used to use pose expressed in camera frame. global reference frame doesn't matter though

            idxs = list(range(posecam.shape[0]))
            random.shuffle(idxs)
            select = random.randint(1, 4)

            if self.multiview_aug:
                idxs = idxs[:select]

            poses = []
            quats = []
            for i in idxs:
                pos = posecam[i, :3]
                quat = posecam[i, 3:]

                poses.append(pos)
                quats.append(quat)

            shapenet_id = str(data['shapenet_id'].item())
            category_id = str(data['shapenet_category_id'].item())

            depths = []
            segs = []
            cam_poses = []

            for i in idxs:
                seg = data['object_segmentation'][i, 0]
                depth = data['depth_observation'][i]
                rix = np.random.permutation(depth.shape[0])[:1000]
                seg = seg[rix]
                depth = depth[rix]

                if self.depth_aug:
                    depth = depth + np.random.randn(*depth.shape) * 0.1

                segs.append(seg)
                depths.append(torch.from_numpy(depth))
                cam_poses.append(data['cam_pose_world'][i])

            # change these values depending on the intrinsic parameters of camera used to collect the data. These are what we used in pybullet
            y, x = torch.meshgrid(torch.arange(480), torch.arange(640))

            # Compute native intrinsic matrix
            sensor_half_width = 320
            sensor_half_height = 240

            vert_fov = 60 * np.pi / 180

            vert_f = sensor_half_height / np.tan(vert_fov / 2)
            hor_f = sensor_half_width / (np.tan(vert_fov / 2) * 320 / 240)

            intrinsics = np.array(
                [[hor_f, 0., sensor_half_width, 0.],
                [0., vert_f, sensor_half_height, 0.],
                [0., 0., 1., 0.]]
            )

            # Rescale to new sidelength
            intrinsics = torch.from_numpy(intrinsics)

            # build depth images from data
            dp_nps = []
            for i in range(len(segs)):
                seg_mask = segs[i]
                dp_np = geometry.lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depths[i].flatten(), intrinsics[None, :, :])
                dp_np = torch.cat([dp_np, torch.ones_like(dp_np[..., :1])], dim=-1)
                dp_nps.append(dp_np)

            # load in voxel occupancy data
            voxel_path = osp.join(category_id, shapenet_id, 'models', 'model_normalized_128.mat')
            coord, voxel_bool, _ = self.shapenet_dict[category_id][voxel_path]

            rix = np.random.permutation(coord.shape[0])

            coord = coord[rix[:1500]]
            label = voxel_bool[rix[:1500]]

            offset = np.random.uniform(-self.hbs, self.hbs, coord.shape)
            coord = coord + offset
            coord = coord * data['mesh_scale']

            coord = torch.from_numpy(coord)

            # transform everything into the same frame
            transforms = []
            for quat, pos in zip(quats, poses):
                quat_list = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
                rotation_matrix = Rotation.from_quat(quat_list)
                rotation_matrix = rotation_matrix.as_matrix()

                transform = np.eye(4)
                transform[:3, :3] = rotation_matrix
                transform[:3, -1] = pos
                transform = torch.from_numpy(transform)
                transforms.append(transform)


            transform = transforms[0]
            coord = torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)
            coord = torch.sum(transform[None, :, :] * coord[:, None, :], dim=-1)
            coord = coord[..., :3]

            points_world = []

            for i, dp_np in enumerate(dp_nps):
                point_transform = torch.matmul(transform, torch.inverse(transforms[i]))
                dp_np = torch.sum(point_transform[None, :, :] * dp_np[:, None, :], dim=-1)
                points_world.append(dp_np[..., :3])

            point_cloud = torch.cat(points_world, dim=0)

            rix = torch.randperm(point_cloud.size(0))
            point_cloud = point_cloud[rix[:1000]]

            if point_cloud.size(0) != 1000:
                return self.get_item(index=random.randint(0, self.__len__() - 1))

            label = (label - 0.5) * 2.0

            # translate everything to the origin based on the point cloud mean
            center = point_cloud.mean(dim=0)
            coord = coord - center[None, :]
            point_cloud = point_cloud - center[None, :]

            labels = label

            # at the end we have 3D point cloud observation from depth images, voxel occupancy values and corresponding voxel coordinates
            res = {'point_cloud': point_cloud.float(),
                   'coords': coord.float(),
                   'intrinsics': intrinsics.float(),
                   'cam_poses': np.zeros(1)}  # cam poses not used
            return res, {'occ': torch.from_numpy(labels).float()}

        except Exception as e:
           print(e)
        #    print(file)
           return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)
