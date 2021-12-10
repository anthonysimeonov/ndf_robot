import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
from io import BytesIO

msz = 1.5
op = 1.0
cscale = 'Viridis'
black_marker = {
        'size': msz,
        'color': 'black',
        'colorscale': cscale, 
        'opacity': op 
        }
blue_marker = {
        'size': msz,
        'color': 'blue',
        'colorscale': cscale, 
        'opacity': op
        }
red_marker = {
        'size': msz,
        'color': 'red',
        'colorscale': cscale, 
        'opacity': op
        }
purple_marker = {
        'size': msz,
        'color': 'purple',
        'colorscale': cscale,
        'opacity': op
        }
green_marker = {
        'size': msz,
        'color': 'green',
        'colorscale': cscale,
        'opacity': op
        }
orange_marker = {
        'size': msz,
        'color': 'orange',
        'colorscale': cscale,
        'opacity': op
        }

marker_dict = {
        'black': black_marker,
        'blue': blue_marker,
        'red': red_marker,
        'purple': purple_marker,
        'orange': orange_marker,
        'green': green_marker
        }

def plot3d(pts_list, colors=['black'], fname='default_3d.html', 
           auto_scene=False, scene_dict=None, z_plane=True, write=True,
           extra_data=None):
    '''
    Function to create a 3D scatter plot in plotly

    Args:
        pts_list (list): list of numpy arrays, each containing a separate point cloud
        colors (list): list of color names corresponding to each point cloud in pts. If this is
            not a list, or there's only one element in the list, we will assume to use the 
            specified colors for each point cloud
        fname (str): name of file to save
        auto_scene (bool): If true, let plotly autoconfigure the scene camera / boundaries / etc.
        scene_dict (dict): If we include this, this contains the scene parameters we want. If this
            is left as None, we have a default scene setting used within the function. Expects
            keys '
        z_plane (bool): If True, then a gray horizontal plane will be drawn below all the point clouds
        write (bool): If True, then html file with plot will be saved
        extra_data (list): Additional plotly data that we might want to plot, which is created externally
    '''
    fig_data = []
    if not isinstance(pts_list, list):
        pts_list = [pts_list]
    if not isinstance(colors, list):
        colors = [colors]
    if len(colors) == 1:
        colors = colors * len(pts_list)

    all_pts = np.concatenate(pts_list, axis=0)

    for i, pts in enumerate(pts_list):
        pcd_data = {
                'type': 'scatter3d',
                'x': pts[:, 0],
                'y': pts[:, 1],
                'z': pts[:, 2],
                'mode': 'markers',
                'marker': marker_dict[colors[i]]}
        fig_data.append(pcd_data)

    z_height = min(all_pts[:, 2])
    plane_data = {
       'type': 'mesh3d',
       'x': [-1, 1, 1, -1],
       'y': [-1, -1, 1, 1],
       'z': [z_height]*4,
       'color': 'gray',
       'opacity': 0.5,
       'delaunayaxis': 'z'}
    
    if z_plane:
        fig_data.append(plane_data)

    if extra_data is not None:
        fig_data = fig_data + extra_data
    fig = go.Figure(data=fig_data)

    default_camera = {
        'up': {'x': 0, 'y': 0,'z': 1},
        'center': {'x': 0.45, 'y': 0, 'z': 0.0},
        'eye': {'x': -1.0, 'y': 0.0, 'z': 0.01}
    }
    default_scene = {
        'xaxis': {'nticks': 10, 'range': [-0.1, 0.9]},
        'yaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
        'zaxis': {'nticks': 8, 'range': [-0.01, 1.5]}
    }
    default_width = 1100
    default_margin = {'r': 20, 'l': 10, 'b': 10, 't': 10}
    default_scene_dict = dict(
        scene=default_scene,
        camera=default_camera,
        width=default_width,
        margin=default_margin
    )

    if scene_dict is None:
        scene_dict = default_scene_dict
    else:
        for key in default_scene_dict.keys():
            if key not in scene_dict.keys():
                scene_dict[key] = default_scene_dict[key]

    if not auto_scene:
        fig.update_layout(
            scene=scene_dict['scene'],
            scene_camera=scene_dict['camera'],
            width=scene_dict['width'],
            margin=scene_dict['margin']
        )

    #png_renderer = pio.renderers['png']
    #png_renderer.width = 500
    #png_renderer.height = 500
    #pio.renderers.default = 'png'

    if write:
        #fig.show()
        if fname.endswith('html'):
            fig.write_html(fname)
        else:
            fig.write_image(fname)
    return fig

