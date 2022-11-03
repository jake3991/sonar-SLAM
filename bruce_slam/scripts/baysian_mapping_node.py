#!/usr/bin/env python
import rospy
from bruce_slam.baysian_mapping import *

if __name__ == "__main__":

    #init the node
    rospy.init_node("Mapping", log_level=rospy.INFO)

    #call the class constructor for stereo sonar
    node = BaysianMappingNode()

    #log info and spin
    rospy.loginfo("Start 3D Mapping...")
    rospy.spin()