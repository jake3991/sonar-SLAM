## SLAM
This readme covers the details of our SLAM system. Consider this as a high-level user manual for the SLAM code specifically. A flow chart overview is shown below. 

<img src="images/SLAM_overview.png " width="500"/>

This SLAM system uses a factor graph to derive pose estimates from several factors. We use odometry factors, sequential scan matching factors (SSM), and non-sequential scan matching factors (NSSM a.k.a loop closures). Please note that this system requires odometry, in our case we use a DVL, IMU, and pressure sensor. 

As the vehicle moves along the trajectory, the dead reckoning (odometry) node provides pose estimates to the SLAM node. Concurrently, the feature extraction node identifies targets in the sonar imagery and publishes them as a planer PointCloud2. Note that the SLAM note subscribes to both the odometry and the point cloud. 

## Feature Extraction Node
This node subscribes to sonar images and outputs an IN PLANE point cloud. To convert the image to a point cloud, first, we must identify contacts in the sonar image. Here we use constant-false-alarm-rate (CFAR) which is written in C++ and built using pybind11. CFAR is governed by the following parameters in config/feature.yaml. We recommend this as the primary tuning point in this whole system. These points will be used as a basis for ICP scan matching, so if they are poor, your results will be poor. 

CFAR:
 - Ntc: number of training cells
 - Ngc: number of guard cells
 - Pfa: false alarm rate
 - rank:matrix rank
 - alg: method of CFAR, see cfar.cpp for options. SOCA is best.

After contacts are identified, they must be converted to a point cloud. This is done by using the knowledge of the sonar image geometry provided in the sonar image message. Once we have converted from pixels to meters, we apply some simple point cloud processing. In this case voxel downsampling and radius outlier filtering. See below. 

filter:
  - threshold: the min CFAR response to be added to the point cloud
  - resolution:  voxel downsampling res
  - radius: outlier rejection radius
  - min_points: outlier rejection min points
  - skip: how often to skip a point cloud, use 1 recommended do not change

Once the cloud is filtered, it is packaged and published with the same timestamp as the incoming sonar image. 

## Gyroscope Node
This node subscribes to our fiber-optic-gyroscope (FOG) and converts the change in angle to an angle estimate. Note that in our framework, while we have a 3-axis FOG, we only use the yaw axis. This node is still under active development, but as of now it simply integrates the gyro measurements to output Euler angles. So far there is no filtering required for this device. More TBD.

## Dead Reckoning Node
This node combines IMU, DVL, pressure sensor, and FOG (if used) into a 6DOF state estimate. Note that while we get a 6DOF estimate, the SLAM solution only needs x,y, and theta. Note that ROS may need 6DOF for transform/vis purposes. 

This node uses a single callback to subscribe to the IMU, DVL, and FOG (if used). The vectornav IMU outputs an attitude estimate and this is used to update the dead reckoning attitude estimate. Then using delta_t and the DVL's speed measurements (in x/y only) we propagate our state forward. Note that the IMU comes into this node filtered and the DVL is not filtered. Future development will be considered. Recall that dead reckoning is used as an initial guess between keyframes only. 

Some parameters below
dvl_max_velocity: max DVL velocity
use_gyro:  should we use the FOG? Do NOT set it to true if there is no FOG sensor. 
imu_version: which IMU driver are we using 1 is the heavy and 2 is heavy MKII. The provided data file is 1.

## SLAM Node 
This node subscribes to the odometry topic and the sonar topic. The callback in this node (SLAM_callback) executes the SLAM implementation.

If at any point the vehicle has moved more than keyframe_rotation or keyframe_translation, a new keyframe is instantiated. These keyframes are the time-discrete poses that are entered into the factor graph. As each keyframe is instantiated using an initial odometry factor, we also attempt to add an SSM and NSSM factor. SSM pulls the point cloud from the current frame and uses a global initialized ICP to compare it to the previous SSM/target_frames. If this ICP call passes the following checks, then it is entered as a factor in the robot factor graph. 

SSM Checks 
- min_points, min number of points overlapping between the two clouds
- max_translation,  max translation allowed relative to the initial guess
- max_rotation, max rotation allowed relative to the initial guess

Once the SSM system is complete we move on to the NSSM system. Here we search for loop closures by comparing the newest keyframe to all previous keyframes. We first create an aggregated point cloud about the newest keyframe, this cloud consists of NSSM/source_frames. This cloud is provided with an initial guess as per the most recent SLAM update (performed after the addition of the SSM factor as above). We then aggregate all previous frames, which includes an exclusion zone of frames so as not to have the same data in both clouds, here we consider a range of keyframes from current_keyframe - NSSM/min_st_sep to zero. We then eliminate keyframes that do not have an overlapping sensor area with the source cloud. Again a globally initialized ICP is used and if the following checks are successful, the loop closure is added to the PCM queue for further verification. 

NSSM Checks
- min_points, min number of points overlapping between the two clouds
- max_translation,  max translation allowed relative to the initial guess
- max_rotation, max rotation allowed relative to the initial guess

For more details on PCM (Pairwise Consistent Measurement Set Maximization for Robust Multi-robot Map Merging), please see the original paper. We use it to reject outliers in loop closures. We use two parameters to govern its behavior. 

PCM
- pcm_queue_size, the sliding window size for PCM
- min_pcm, the min number of pairwise consistent loop closures when rejecting outliers

