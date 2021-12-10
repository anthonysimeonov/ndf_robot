import numpy as np
import trimesh

from airobot.utils import common

# hand = trimesh.load('visual/hand.obj')
# lf = trimesh.load('visual/finger.obj')
# rf = trimesh.load('visual/finger.obj')
hand = trimesh.load('collision/hand.obj')
lf = trimesh.load('collision/finger.obj')
rf = trimesh.load('collision/finger.obj')

# offset
offset_z = 0.0584

l_offset = [0, 0.015, offset_z]
r_offset = [0, -0.015, offset_z]

# rotate
rf_tf = np.eye(4)
rf_tf[:-1, :-1] = common.euler2rot([0, 0, np.pi])
rf_tf[:-1, -1] = r_offset

lf_tf = np.eye(4)
lf_tf[:-1, -1] = l_offset

lf.apply_transform(lf_tf)
rf.apply_transform(rf_tf)


# scene = trimesh.Scene()
# scene.add_geometry([hand, lf, rf])
# scene.show()


# combine
combined_hand = trimesh.util.concatenate([hand, rf, lf])
tf = np.eye(4)
tf[:-1, -1] = [0, 0, -0.105]
tf[:-1, :-1] = common.euler2rot([0, 0, 0.785398163397])
combined_hand.apply_translation([0, 0, -0.105])
combined_hand.show()


from IPython import embed; embed()
