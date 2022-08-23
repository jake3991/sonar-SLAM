#!/usr/bin/env python

# python imports
import rospy

# pull in the dead reckoning code
from bruce_slam.utils.io import *
from bruce_slam.kalman import KalmanNode


if __name__ == "__main__":
    rospy.init_node("kalman", log_level=rospy.INFO)

    node = KalmanNode()
    node.init_node()

    args, _ = common_parser().parse_known_args()
    if not args.file:
        loginfo("Start online Kalman...")
        rospy.spin()
    else:
        loginfo("Start offline Kalman...")
        offline(args)
