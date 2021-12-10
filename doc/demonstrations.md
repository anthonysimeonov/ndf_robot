# Collecting robot demonstrations and encoding with NDF
```
cd src/ndf_robot/demonstrations
python label_demos.py --exp test_bottle --object_class bottle --with_shelf
```

## Notes on teleoperation for collecting demonstrations
The script at [`demonstrations/label_demos.py`](../src/ndf_robot/demonstrations/label_demos.py) runs a super simple teleoperation setup for moving around a simulated Panda robot in PyBullet. We use it to provide grasping and placing demonstrations in our table+rack or table+shelf environments. 

The keys `Q|E|A|D|S|X` are used to move `down|up|left|right|forward|back`. The keys `Z|C` are used to `open|close` the gripper. The keys `U|O|I|K|J|L` are used to rotate about the `X|Y|Z` axes of the gripper. 

At each step, a random object will be sampled and initialized in the environment. The `1` key can be used to save the current gripper pose as a **grasp** and the `2` key can be used to save the current gripper pose as a **place**. 

Note that this pipeline is also where the query points that are used in the pose optimization are sampled and saved.