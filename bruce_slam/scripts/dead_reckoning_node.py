#!/usr/bin/env python

# python imports
import rospy

# pull in the dead reckoning code
from bruce_slam.utils.io import *
from bruce_slam.dead_reckoning import DeadReckoningNode


if __name__ == "__main__":
    rospy.init_node("localization", log_level=rospy.INFO)

    node = DeadReckoningNode()
    node.init_node()

    args, _ = common_parser().parse_known_args()
    if not args.file:
        loginfo("Start online localization...")
        rospy.spin()
    else:
        loginfo("Start offline localization...")
        offline(args)