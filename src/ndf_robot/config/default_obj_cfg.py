from yacs.config import CfgNode as CN 

_C = CN()

# general configs
_C.AVOID_ORIS = [None]

def get_obj_cfg_defaults():
    return _C.clone()


