#!/usr/bin/env python
import cv2
import sys
import rospy
import cv_bridge
import numpy as np
from sensor_msgs.msg import Image, CompressedImage


class CompressorNode():

    def __init__(self, sonar_id: int) -> None:
        """Class constructor for the compressor node, get the raw image and republish as 
        as compressed node to make data logging cheaper

        Args:
            sonar_id (int): vertical or horizontal sonar
        """

        print(sonar_id)
        self.BridgeInstance = cv_bridge.CvBridge()

        # define the pub/sub objects based on which sonar we are using
        if sonar_id == 1:
            rospy.Subscriber("/rexrov/forward_sonar_horiz/raw_sonar", Image, self.callback)
            self.publisher = rospy.Publisher("/rexrov/forward_sonar_horiz/raw_sonar_compressed",CompressedImage,queue_size=10)
        elif sonar_id == 2:
            rospy.Subscriber("/rexrov/forward_sonar_vert/raw_sonar", Image, self.callback)
            self.publisher = rospy.Publisher("/rexrov/forward_sonar_vert/raw_sonar_compressed",CompressedImage,queue_size=10)
        else:
            raise(NotImplemented)

    def callback(self, msg: Image) -> None:
        """Callback to compress the simulation sonar images

        Args:
            msg (Image): the uncompressed sonar image
        """

        # handle the incoming image
        img = 255. * np.array(self.BridgeInstance.imgmsg_to_cv2(msg)).astype(float)
        img = np.array(img).astype(int)

        # publish the uncompressed image as a compressed one
        msg_out = CompressedImage()
        msg_out.header = msg.header
        msg_out.format = "jpeg"
        msg_out.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
        self.publisher.publish(msg_out)
        
if __name__ == '__main__':

    rospy.init_node('listener', anonymous=True)
    node = CompressorNode(int(sys.argv[1]))
    rospy.spin()
    