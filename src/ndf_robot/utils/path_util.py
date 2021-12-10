import os, os.path as osp


def get_ndf_src():
    return os.environ['NDF_SOURCE_DIR']


def get_ndf_config():
    return osp.join(get_ndf_src(), 'config')


def get_ndf_share():
    return osp.join(get_ndf_src(), 'share')


def get_ndf_data():
    return osp.join(get_ndf_src(), 'data')


def get_ndf_recon_data():
    return osp.join(get_ndf_src(), 'data_gen/data')


def get_ndf_eval_data():
    return osp.join(get_ndf_src(), 'eval_data')


def get_ndf_descriptions():
    return osp.join(get_ndf_src(), 'descriptions')


def get_ndf_obj_descriptions():
    return osp.join(get_ndf_descriptions(), 'objects')


def get_ndf_demo_obj_descriptions():
    return osp.join(get_ndf_descriptions(), 'demo_objects')


def get_ndf_assets():
    return osp.join(get_ndf_src(), 'assets')


def get_ndf_model_weights():
    return osp.join(get_ndf_src(), 'model_weights')
