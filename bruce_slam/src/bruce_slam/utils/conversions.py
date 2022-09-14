from ctypes import Union
from typing import Any
import numpy as np
import gtsam
import cv2
import cv_bridge
import rospy
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Pose
import ros_numpy
import struct


from .topics import *


def X(x:int) -> gtsam.symbol:
    """convert an integer to a gtsam symbol

    Args:
        x (int): the index of the symbol

    Returns:
        gtsam.symbol: gtsam symbol x_n
    """

    return gtsam.symbol("x", x)

def pose322(pose:gtsam.Pose3) -> gtsam.Pose2:
    """Convert a gtsam.Pose3 to a gtsam.Pose2

    Args:
        pose (gtsam.Pose3): the input 3D pose

    Returns:
        gtsam.Pose2: the 2D pose
    """

    return gtsam.Pose2(pose.x(), pose.y(), pose.rotation().yaw())


def pose223(pose:gtsam.Pose2) -> gtsam.Pose3:
    """convert a gtsam.Pose2 to a gtsam.Pose3

    Args:
        pose (gtsam.Pose2): the input 2D pose

    Returns:
        gtsam.Pose3: the 3D pose with zeros for the unkown values
    """

    return gtsam.Pose3(
        gtsam.Rot3.Yaw(pose.theta()), gtsam.Point3(pose.x(), pose.y(), 0)
    )


def n2g(numpy_arr:np.array, obj:str) -> any:
    """convert a numpy array to gtsam

    Args:
        numpy_arr (np.array): the input numpy array
        obj (str): the output object

    Raises:
        NotImplementedError: _description_

    Returns:
        any: the desired gtsam object
    """

    if obj == "Quaternion":
        x, y, z, w = numpy_arr
        return gtsam.Rot3.Quaternion(w, x, y, z)
    elif obj == "Euler":
        roll, pitch, yaw = numpy_arr
        return gtsam.Rot3.Ypr(yaw, pitch, roll)
    elif obj == "Point2":
        x, y = numpy_arr
        return gtsam.Point2(x, y)
    elif obj == "Pose2":
        x, y, yaw = numpy_arr
        return gtsam.Pose2(x, y, yaw)
    elif obj == "Point3":
        x, y, z = numpy_arr
        return gtsam.Point3(x, y, z)
    elif obj == "Pose3":
        x, y, z, roll, pitch, yaw = numpy_arr
        return gtsam.Pose3(gtsam.Rot3.Ypr(yaw, pitch, roll), gtsam.Point3(x, y, z))
    elif obj == "imuBiasConstantBias":
        imu_bias = numpy_arr
        return gtsam.imuBias_ConstantBias(
            np.array(imu_bias[:3]), np.array(imu_bias[3:])
        )
    elif obj == "Vector":
        return np.array(numpy_arr)
    else:
        raise NotImplementedError("Not implemented from numpy to " + obj)


def g2n(gtsam_obj:gtsam) -> np.array:
    """converts a gtsam object to a numpy array

    Args:
        gtsam_obj (gtsam): the input gtsam object, we can accept several types

    Raises:
        NotImplementedError: if we don't have a conversion for that object raise an exception

    Returns:
        np.array: the input data in numpy format
    """

    if isinstance(gtsam_obj, type(gtsam.Point2())) and gtsam_obj.shape == (2,):
        point = gtsam_obj
        return np.array([point[0], point[1]])
    elif isinstance(gtsam_obj, type(gtsam.Point3())) and gtsam_obj.shape == (3,):
        point = gtsam_obj
        return np.array([point[0], point[1], point[2]])
    elif isinstance(gtsam_obj, gtsam.Rot3):
        rot = gtsam_obj
        return np.array([rot.roll(), rot.pitch(), rot.yaw()])
    elif isinstance(gtsam_obj, gtsam.Pose2):
        pose = gtsam_obj
        return np.array([pose.x(), pose.y(), pose.theta()])
    elif isinstance(gtsam_obj, gtsam.Pose3):
        pose = gtsam_obj
        return np.array(
            [
                pose.x(),
                pose.y(),
                pose.z(),
                pose.rotation().roll(),
                pose.rotation().pitch(),
                pose.rotation().yaw(),
            ]
        )
    elif isinstance(gtsam_obj, gtsam.imuBias_ConstantBias):
        bias = gtsam_obj
        return np.r_[bias.accelerometer(), bias.gyroscope()]
    elif isinstance(gtsam_obj, np.ndarray):
        return gtsam_obj
    else:
        raise NotImplementedError(
            "Not implemented from {} to numpy".format(str(type(gtsam_obj)))
        )


def r2g(ros_msg) -> gtsam.Pose3:
    """convert a ros message to a 3D pose in gtsam

    Args:
        ros_msg (geometry_msgs.msg ): the input geometry message

    Raises:
        NotImplementedError: if unknown type raise exception

    Returns:
        gtsam.Pose3: the input data packaged as a gtsam 3D pose
    """

    if ros_msg._type == "geometry_msgs/Pose":
        x = ros_msg.position.x
        y = ros_msg.position.y
        z = ros_msg.position.z
        qx = ros_msg.orientation.x
        qy = ros_msg.orientation.y
        qz = ros_msg.orientation.z
        qw = ros_msg.orientation.w
        return gtsam.Pose3(
            n2g([qx, qy, qz, qw], "Quaternion"), n2g([x, y, z], "Point3")
        )
    elif ros_msg._type == "geometry_msgs/PoseStamped":
        return r2g(ros_msg.pose)
    elif ros_msg._type == "geometry_msgs/Quaternion":
        return n2g([ros_msg.x, ros_msg.y, ros_msg.z, ros_msg.w], "Quaternion")
    else:
        raise NotImplementedError(
            "Not implemented from {} to gtsam".format(str(type(ros_msg)))
        )


def g2r(gtsam_obj:gtsam.Pose3) -> Pose:
    """convert a gtsam.Pose3 to a ros pose message

    Args:
        gtsam_obj (gtsam.Pose3): the input pose

    Raises:
        NotImplementedError: if not a gtsam.Pose3 raise an execption

    Returns:
        Pose: the poise message in ros
    """

    if isinstance(gtsam_obj, gtsam.Pose3):
        pose = gtsam_obj
        pose_msg = Pose()
        pose_msg.position.x = pose.x()
        pose_msg.position.y = pose.y()
        pose_msg.position.z = pose.z()
        qw, qx, qy, qz = pose.rotation().quaternion()
        pose_msg.orientation.x = qx
        pose_msg.orientation.y = qy
        pose_msg.orientation.z = qz
        pose_msg.orientation.w = qw
        return pose_msg
    else:
        raise NotImplementedError(
            "Not implemented from {} to ros".format(str(type(gtsam_obj)))
        )


bridge = cv_bridge.CvBridge()


def r2n(ros_msg:OculusPing) -> np.array:
    """Convert a ros message of type OculusPing to a numpy array

    Args:
        ros_msg (OculusPing): the input sonar message

    Raises:
        NotImplementedError: catch the wrong types

    Returns:
        np.array: the image data in numpy array form
    """

    if ros_msg._type == "sonar_oculus/OculusPing":
        
        img = r2n(ros_msg.ping)
        img = np.clip(
            cv2.pow(img / 255.0, 255.0 / ros_msg.fire_msg.gamma) * 255.0, 0, 255
        )
        return np.float32(img)
    elif ros_msg._type == "sensor_msgs/Image":
        img = bridge.imgmsg_to_cv2(ros_msg, desired_encoding="passthrough")
        return np.array(img, "uint8")
    elif ros_msg._type == "sensor_msgs/PointCloud2":
        rows = ros_msg.width
        cols = sum(f.count for f in ros_msg.fields)
        return np.array([p for p in pc2.read_points(ros_msg)]).reshape(rows, cols)
    else:
        raise NotImplementedError(
            "Not implemented from {} to numpy".format(str(type(ros_msg)))
        )

def build_rgb_cloud(arr:np.array) -> pc2:
    """Convert an array of [xyz,rgb] to a ROS point cloud with colors

    Args:
        arr (np.array): the input array

    Returns:
        pc2: a ROS point cloud 2
    """

    # define the point cloud fields and header
    header = rospy.Header()
    fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField('rgb', 12, pc2.PointField.UINT32, 1),
            ]

    # parse out and convert the RGB values to RGBA
    points = []
    for row in arr:
        r, g, b = int(row[2]), int(row[3]), int(row[4])
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
        points.append( [row[0], row[1], 0., rgb] )
    return pc2.create_cloud(header, fields, points)
    

def n2r(numpy_arr:np.array, msg:any) -> any:
    """Package a nump array as the target ros message type in msg

    Args:
        numpy_arr (np.array): the data to be entered into a message
        msg (any): the target message type

    Raises:
        NotImplementedError: catch an unkown type

    Returns:
        any: the output message
    """
    
    if msg == "Image":
        if numpy_arr.ndim == 2 or numpy_arr.shape[2] == 1:
            return bridge.cv2_to_imgmsg(numpy_arr, encoding="8U")
        else:
            return bridge.cv2_to_imgmsg(numpy_arr, encoding="rgb8")
    elif msg == "ImageBGR":
        return bridge.cv2_to_imgmsg(numpy_arr, encoding="bgr8")
    elif msg == "PointCloudXYZ":
        header = rospy.Header()
        return pc2.create_cloud_xyz32(header, np.array(numpy_arr))
    elif msg == "PointCloudXYZI":
        header = rospy.Header()
        fields = [
            pc2.PointField("x", 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField("y", 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField("z", 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField("i", 12, pc2.PointField.FLOAT32, 1),
        ]
        return pc2.create_cloud(header, fields, np.array(numpy_arr))
    elif msg == "PointCloudXYZRGB":
        return build_rgb_cloud(numpy_arr)
    else:
        raise NotImplementedError("Not implemented from numpy array to {}".format(msg))
