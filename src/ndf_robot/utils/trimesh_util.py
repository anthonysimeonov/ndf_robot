import numpy as np
import trimesh
from matplotlib import cm


def trimesh_show(np_pcd_list, color_list=None, rand_color=False, show=True):
    colormap = cm.get_cmap('brg', len(np_pcd_list))
    # colormap = cm.get_cmap('gist_ncar_r', len(np_pcd_list))
    colors = [
        (np.asarray(colormap(val)) * 255).astype(np.int32) for val in np.linspace(0.05, 0.95, num=len(np_pcd_list))
    ]
    if color_list is None:
        if rand_color:
            color_list = []
            for i in range(len(np_pcd_list)):
                color_list.append((np.random.rand(3) * 255).astype(np.int32).tolist() + [255])
        else:
            color_list = colors
    
    tpcd_list = []
    for i, pcd in enumerate(np_pcd_list):
        tpcd = trimesh.PointCloud(pcd)
        tpcd.colors = np.tile(color_list[i], (tpcd.vertices.shape[0], 1))

        tpcd_list.append(tpcd)
    
    scene = trimesh.Scene()
    scene.add_geometry(tpcd_list)
    if show:
        scene.show() 

    return scene