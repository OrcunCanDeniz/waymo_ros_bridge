Added PointCloud2 messages with Intensity values over this [work](https://github.com/biomotion/waymo-rosbag-record)


```bash
catkin build
```

After the build, run the node.
```bash
source devel/setup.bash
roslaunch waymo_ros_bridge recorder.launch
```

Launch file has 3 params;
- `lib_path` : Waymo SDK path. It is in this package by default, this param can be left as is.
- `folderpath` : Path to directory where .tfrecords reside.
- `bag_path` : Path to directory where Rosbags will be saved in.

