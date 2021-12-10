import os.path as osp
import numpy as np

from ndf_robot.utils.path_util import get_ndf_assets


SHAPENET_ID_DICT = {}
SHAPENET_ID_DICT['mug'] = '03797390'
SHAPENET_ID_DICT['bottle'] = '02876657'
SHAPENET_ID_DICT['jar'] = '03593526'
SHAPENET_ID_DICT['bowl'] = '02880940'

bad_shapenet_mugs_data = np.load(osp.join(get_ndf_assets(), 'bad_mugs_all.npz'))
bad_shapenet_bowls_data = np.load(osp.join(get_ndf_assets(), 'bad_bowls.npz'))
bad_shapenet_bottles_data = np.load(osp.join(get_ndf_assets(), 'bad_bottles_all.npz'))

bad_shapenet_mug_ids_list = bad_shapenet_mugs_data['bad_ids'].tolist()
bad_shapenet_bowls_ids_list = bad_shapenet_bowls_data['bad_ids'].tolist()
bad_shapenet_bottles_ids_list = bad_shapenet_bottles_data['bad_ids'].tolist()


