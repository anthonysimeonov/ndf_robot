# NDF Dataset
```
cd src/ndf_robot/data_gen
python shapenet_pcd_gen.py \
    --total_samples 100 \
    --object_class mug \
    --save_dir test_mug \
    --rand_scale \
    --num_workers 2
```

## Notes on dataset and dataset generation
Our dataset is primarily composed pairs of object point clouds and ground truth occupancy values of a large number of points sampled in the volume near the shape. These are obtained by running the script at [`data_gen/shapenet_pcd_gen.py`](../src/ndf_robot/data_gen/shapenet_pcd_gen.py) after downloading the object models (see [main README](../README.md)).

The script runs by placing the object meshes in a PyBullet simulation and rendering depth images using simulated cameras at different poses. The objects are randomly scaled and posed to create more diversity in the dataset to help the point cloud encoder generalize. The points at which the occupancy is evaluated are similarly scaled/transformed based on how the object is adjusted when we load it into the simulator. 

We also have the option of adding other random shapes into the simulator so that the shape is partially occluded in some of the samples (use the `--occlude` flag with the above command).