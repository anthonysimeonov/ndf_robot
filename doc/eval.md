# NDF evaluation with simulated robot
Make sure you have set up the additional inverse kinematics tools (see Setup section in [README](../README.md))

**Download all the object data assets**
```
./scripts/download_obj_data.sh
```

**Download pretrained weights**
```
./scripts/download_demo_weights.sh
```

**Download demonstrations**
```
./scripts/download_demo_demonstrations.sh
```

**Run evaluation**

If you are running this command on a remote machine, be sure to remove the `--pybullet_viz` flag!
```
cd src/ndf_robot/eval
CUDA_VISIBLE_DEVICES=0 python evaluate_ndf.py \
        --demo_exp grasp_rim_hang_handle_gaussian_precise_w_shelf \
        --object_class mug \
        --opt_iterations 500 \
        --only_test_ids \
        --rand_mesh_scale \
        --model_path multi_category_weights \
        --save_vis_per_model \
        --config eval_mug_gen \
        --exp test_mug_eval \
        --pybullet_viz
```

## Notes on simulated robot setup

We set up an experiment with a robot simulated in PyBullet to evaluate how NDF enables learning pick-and-place behaviors from a demonstrations. [`eval/evaluate_ndf.py`](../src/ndf_robot/eval/evaluate_ndf.py) contains the code for running this experiment. The basic procedure that it runs through to set up the experiment is as follows:
- Set up trained occupancy network with vector neuron layers by loading network weights
- Load in data obtained from demonstrations, and encode this into a reference pose descriptor to match for both picking and placing.
- Set up objects to be used in test, making sure that these are objects that were neither included in training nor in the demonstrations.
- Set up the PyBullet robot simulation (using our [airobot library](https://github.com/Improbable-AI/airobot)) and the IKFast inverse kinematics solver + motion planning interfaces (from [Caelan Garret's pybullet-planning repo](https://github.com/caelan/pybullet-planning)). 


We then iterate through the test objects. Each object has a random scaling and random pose applied to it (pose distribution depends on whether experiment is run using flag `--any_pose`), and our energy optimization procedure is run twice on the point cloud observation of this object to obtain a grasping and placing pose. These are executed and their success/failure is tracked to obtain the experimental results. These results are saved in the folder `eval_data` for post-processing.

### Metrics explanation
Our main experimental metrics are three success rate numbers: grasping success, placing success, and overall success. In this experiment, we compute grasping success and placing success independently, and then compute overall success depending on whether the full simulated execution results in a successful placement. The independent grasp/placement success checks work as follows:
- Grasping requires the gripper to move to the grasp pose, close the fingers, and hold the object without dropping it when collisions between the object and the table are turned off.
- Placing requires that resetting the object state to the predicted placement pose, and then turning on the physics (i.e., gravity and collisions) leads the object to stay supported upright in the desired orientation without falling down onto the ground (collisions with the table are turned off to enable this if the placement is off).

To prevent the effects of motion planning and inverse kinematics from affecting our quantitative results, the "overall success" numbers presented in the paper are computed based on checking which samples were independently successful in both grasping and placing (rather than using the success of the full robot execution).

### Commands for running experiments on other categories (bowls, bottles) with our pretrained weights

***Remember to download all the necessary object/demonstration/weight files! See the main [README](../README.md)***

Bottles:
```
cd src/ndf_robot/eval
CUDA_VISIBLE_DEVICES=0 python evaluate_ndf.py \
        --demo_exp grasp_side_place_shelf_start_upright_all_methods_multi_instance \
        --object_class bottle \
        --opt_iterations 500 \
        --only_test_ids \
        --rand_mesh_scale \
        --model_path multi_category_weights \
        --save_vis_per_model \
        --config eval_bottle_gen \
        --exp test_bottle_eval \
        --non_thin_feature \
        --pybullet_viz
```

Bowls:
```
cd src/ndf_robot/eval
CUDA_VISIBLE_DEVICES=0 python evaluate_ndf.py \
        --demo_exp grasp_rim_anywhere_place_shelf_all_methods_multi_instance \
        --object_class bowl \
        --opt_iterations 500 \
        --only_test_ids \
        --rand_mesh_scale \
        --model_path multi_category_weights \
        --save_vis_per_model \
        --config eval_bowl_gen \
        --exp test_bowl_eval \
        --pybullet_viz
```
