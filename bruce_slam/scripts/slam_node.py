#!/usr/bin/env python
import rospy
import numpy as np
from bruce_slam.utils.io import *
from bruce_slam.slam_ros import SLAMNode
from bruce_slam.utils.topics import *

def offline(args)->None:
    """run the SLAM system offline

    Args:
        args (Any): the arguments to run the system
    """

    # pull in the extra imports required
    from rosgraph_msgs.msg import Clock
    from dead_reckoning_node import DeadReckoningNode
    from feature_extraction_node import FeatureExtraction
    from gyro_node import GyroFilter
    from bruce_slam.utils import io

    # set some params
    io.offline = True
    node.save_fig = False
    node.save_data = False

    # instaciate the nodes required
    dead_reckoning_node = DeadReckoningNode()
    dead_reckoning_node.init_node(SLAM_NS + "localization/")
    feature_extraction_node = FeatureExtraction()
    feature_extraction_node.init_node(SLAM_NS + "feature_extraction/")
    gyro_node = GyroFilter()
    gyro_node.init_node(SLAM_NS + "gyro/")
    '''stereo_sonar_node = stereoSonar("")
    stereo_sonar_node.scene = args.scene
    stereo_sonar_node.keyframe_translation = args.translation
    stereo_sonar_node.keyframe_rotation = np.radians(args.rotation)
    bayes_mapping_node = BaysianMappingNode()
    bayes_mapping_node.scene = args.scene 
    bayes_mapping_node.keyframe_translation = args.translation
    bayes_mapping_node.keyframe_rotation = np.radians(args.rotation)'''
    #bayes_mapping_node.vis_3D = False
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=100)

    # loop over the entire rosbag
    for topic, msg in read_bag(args.file, args.start, args.duration, progress=True):
        while not rospy.is_shutdown():
            if callback_lock_event.wait(1.0):
                break

        if rospy.is_shutdown():
            break

        if topic == IMU_TOPIC or topic == IMU_TOPIC_MK_II:
            dead_reckoning_node.imu_sub.callback(msg)
        elif topic == DVL_TOPIC:
            dead_reckoning_node.dvl_sub.callback(msg)
        elif topic == DEPTH_TOPIC:
            dead_reckoning_node.depth_sub.callback(msg)
        elif topic == SONAR_TOPIC:
            feature_extraction_node.sonar_sub.callback(msg)
        elif topic == GYRO_TOPIC:
            gyro_node.gyro_sub.callback(msg)

        '''if topic == SONAR_TOPIC:
            stereo_sonar_node.horizontalSonarSub.callback(msg)
            bayes_mapping_node.image_sub.callback(msg)
        elif topic == VERTICAL_SONAR_TOPIC:
            stereo_sonar_node.verticalSonarSub.callback(msg)  '''          

        # use the IMU to drive the clock
        if topic == IMU_TOPIC or topic == IMU_TOPIC_MK_II:

            clock_pub.publish(Clock(msg.header.stamp))

            # Publish map to world so we can visualize all in a z-down frame in rviz.
            node.tf.sendTransform((0, 0, 0), [1, 0, 0, 0], msg.header.stamp, "map", "world")
            #node.tf.sendTransform((1.15, 0, 0), [1, 0, 0, 0], msg.header.stamp, "sonar_link", "base_link")
    

if __name__ == "__main__":

    #init the node
    rospy.init_node("slam", log_level=rospy.INFO)

    #call the class constructor
    node = SLAMNode()
    node.init_node()

    #parse and start
    args, _ = common_parser().parse_known_args()

    # push through any changes in the keyframe density from command line
    if args.translation is not None and args.translation != -1:
        node.keyframe_translation = args.translation
    if args.rotation is not None and args.rotation != -1:
        node.keyframe_rotation = np.radians(args.rotation)
    if args.scene is not None:
        node.scene = args.scene 
        #node.vis_3D = False

    if not args.file:
        loginfo("Start online slam...")
        rospy.spin()
    else:
        loginfo("Start offline slam...")
        offline(args)
