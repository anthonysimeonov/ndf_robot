# Training NDFs

**Download all data assets**

If you want the full dataset (~150GB for 3 object classes):
```
./scripts/download_training_data.sh 
```
If you want just the mug dataset (~50 GB -- other object class data can be downloaded with the according scripts):
```
./scripts/download_mug_training_data.sh 
```

If you want to recreate your own dataset, see Data Generation section

**Run training**
```
cd src/ndf_robot/training
python train_vnn_occupancy_net.py --obj_class all --experiment_name  ndf_training_exp
```

## Notes on training

We utilize the intermediate activations of a pre-trained [occupancy network](https://arxiv.org/abs/1812.03828) (with [vector neuron](https://arxiv.org/abs/2104.12229) layers in the point cloud encoder) as our 3D spatial descriptors. Therefore, our training procedure directly mimics the procedure for training an SO(3) equivariant occupancy network for 3D reconstruction.

The file [`ndf_robot/src/ndf_robot/training/dataio.py`](../src/ndf_robot/training/dataio.py) contains ourcustom dataset class. The path of the directory containing the training data is set in the constructor of the dataset class. Each sample contains rendered depth images of the object (we use ShapeNet objects for all our experiments), ground truth occupancy for the object, and camera poses that are used to reconstruct the 3D point cloud from the depth images. More information on the dataset can be found [here](dataset.md).

The rest of the training code is adapted from the training code in the [occupancy network repo](https://github.com/autonomousvision/occupancy_networks). We have found that training the network for 50-100 epochs leads to good performance.

Checkpoints are saved in the folder `ndf_robot/src/ndf_robot/model_weights/$LOGGING_ROOT/$EXPERIMENT_NAME/checkpoints`, where `$LOGGING_ROOT` and `$EXPERIMENT_NAME` are set via the training script `argparse` args.  

If you want to make sure training is working correctly without waiting for the full dataset to download, we provide a smaller dataset with the same format that can be quickly downloaded. To use this mini-training set for verifying the pipeline works, perform the following steps:

- From the root directory (after sourcding `ndf_env.sh`) run `./scripts/download_mini_training_data.sh`
- Navigate to the folder  [`ndf_robot/src/ndf_robot/training`](../src/ndf_robot/training)
- Comment out the lines in the `JointOccTrainDataset` constructor that specify the paths to the full dataset, and uncomment the lines that specify the path to the mini-dataset
- Save `dataio.py` and run the training command specified above