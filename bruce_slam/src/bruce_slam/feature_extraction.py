#!/usr/bin/env python
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import PointCloud2, Image
import cv_bridge
import ros_numpy

from bruce_slam.utils.io import *
from bruce_slam.utils.topics import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.visualization import apply_custom_colormap
#from bruce_slam.feature import FeatureExtraction
from bruce_slam import pcl
import matplotlib.pyplot as plt
from sonar_oculus.msg import OculusPing, OculusPingUncompressed
from scipy.interpolate import interp1d

from .utils import *
from .sonar import *

from bruce_slam.CFAR import CFAR

#from bruce_slam.bruce_slam import sonar

class FeatureExtraction(object):
    '''Class to handle extracting features from Sonar images using CFAR
    subsribes to the sonar driver and publishes a point cloud
    '''

    def __init__(self):
        '''Class constructor, no args required all read from yaml file
        '''

        #oculus info
        self.oculus = OculusProperty()

        #default parameters for CFAR
        self.Ntc = 40
        self.Ngc = 10
        self.Pfa = 1e-2
        self.rank = None
        self.alg = "SOCA"
        self.detector = None
        self.threshold = 0
        self.cimg = None

        #default parameters for point cloud 
        self.colormap = "RdBu_r"
        self.pub_rect = True
        self.resolution = 0.5
        self.outlier_filter_radius = 1.0
        self.outlier_filter_min_points = 5
        self.skip = 5

        # for offline visualization
        self.feature_img = None

        #for remapping from polar to cartisian
        self.res = None
        self.height = None
        self.rows = None
        self.width = None
        self.cols = None
        self.map_x = None
        self.map_y = None
        self.f_bearings = None
        self.to_rad = lambda bearing: bearing * np.pi / 18000
        self.REVERSE_Z = 1
        self.maxRange = None

        #which vehicle is being used
        self.compressed_images = True

        # place holder for the multi-robot system
        self.rov_id = ""

    def configure(self):
        '''Calls the CFAR class constructor for the featureExtraction class
        '''
        self.detector = CFAR(self.Ntc, self.Ngc, self.Pfa, self.rank)

    def init_node(self, ns="~"):

        #read in CFAR parameters
        self.Ntc = rospy.get_param(ns + "CFAR/Ntc")
        self.Ngc = rospy.get_param(ns + "CFAR/Ngc")
        self.Pfa = rospy.get_param(ns + "CFAR/Pfa")
        self.rank = rospy.get_param(ns + "CFAR/rank")
        self.alg = rospy.get_param(ns + "CFAR/alg", "SOCA")
        self.threshold = rospy.get_param(ns + "filter/threshold")

        #read in PCL downsampling parameters
        self.resolution = rospy.get_param(ns + "filter/resolution")
        self.outlier_filter_radius = rospy.get_param(ns + "filter/radius")
        self.outlier_filter_min_points = rospy.get_param(ns + "filter/min_points")

        #parameter to decide how often to skip a frame
        self.skip = rospy.get_param(ns + "filter/skip")

        #are the incoming images compressed?
        self.compressed_images = rospy.get_param(ns + "compressed_images")

        #cv bridge
        self.BridgeInstance = cv_bridge.CvBridge()
        
        #read in the format
        self.coordinates = rospy.get_param(
            ns + "visualization/coordinates", "cartesian"
        )

        #vis parameters
        self.radius = rospy.get_param(ns + "visualization/radius")
        self.color = rospy.get_param(ns + "visualization/color")

        #sonar subsciber
        if self.compressed_images:
            self.sonar_sub = rospy.Subscriber(
                SONAR_TOPIC, OculusPing, self.callback, queue_size=10)
        else:
            self.sonar_sub = rospy.Subscriber(
                SONAR_TOPIC_UNCOMPRESSED, OculusPingUncompressed, self.callback, queue_size=10)

        #feature publish topic
        self.feature_pub = rospy.Publisher(
            SONAR_FEATURE_TOPIC, PointCloud2, queue_size=10)

        #vis publish topic
        self.feature_img_pub = rospy.Publisher(
            SONAR_FEATURE_IMG_TOPIC, Image, queue_size=10)

        self.configure()

    def generate_map_xy(self, ping):
        '''Generate a mesh grid map for the sonar image, this enables converison to cartisian from the 
        source polar images

        ping: OculusPing message
        '''

        #get the parameters from the ping message
        _res = ping.range_resolution
        _height = ping.num_ranges * _res
        _rows = ping.num_ranges
        _width = np.sin(
            self.to_rad(ping.bearings[-1] - ping.bearings[0]) / 2) * _height * 2
        _cols = int(np.ceil(_width / _res))

        #check if the parameters have changed
        if self.res == _res and self.height == _height and self.rows == _rows and self.width == _width and self.cols == _cols:
            return

        #if they have changed do some work    
        self.res, self.height, self.rows, self.width, self.cols = _res, _height, _rows, _width, _cols

        #generate the mapping
        bearings = self.to_rad(np.asarray(ping.bearings, dtype=np.float32))
        f_bearings = interp1d(
            bearings,
            range(len(bearings)),
            kind='linear',
            bounds_error=False,
            fill_value=-1,
            assume_sorted=True)

        #build the meshgrid
        XX, YY = np.meshgrid(range(self.cols), range(self.rows))
        x = self.res * (self.rows - YY)
        y = self.res * (-self.cols / 2.0 + XX + 0.5)
        b = np.arctan2(y, x) * self.REVERSE_Z
        r = np.sqrt(np.square(x) + np.square(y))
        self.map_y = np.asarray(r / self.res, dtype=np.float32)
        self.map_x = np.asarray(f_bearings(b), dtype=np.float32)

    def publish_features(self, ping, points):
        '''Publish the feature message using the provided parameters in an OculusPing message
        ping: OculusPing message
        points: points to be converted to a ros point cloud, in cartisian meters
        '''

        #shift the axis
        points = np.c_[points[:,0],np.zeros(len(points)),  points[:,1]]

        #convert to a pointcloud
        feature_msg = n2r(points, "PointCloudXYZ")

        #give the feature message the same time stamp as the source sonar image
        #this is CRITICAL to good time sync downstream
        feature_msg.header.stamp = ping.header.stamp
        feature_msg.header.frame_id = "base_link"

        #publish the point cloud, to be used by SLAM
        self.feature_pub.publish(feature_msg)

    #@add_lock
    def callback(self, sonar_msg):
        '''Feature extraction callback
        sonar_msg: an OculusPing messsage, in polar coordinates
        '''

        if sonar_msg.ping_id % self.skip != 0:
            self.feature_img = None
            # Don't extract features in every frame.
            # But we still need empty point cloud for synchronization in SLAM node.
            nan = np.array([[np.nan, np.nan]])
            self.publish_features(sonar_msg, nan)
            return

        #decode the compressed image
        if self.compressed_images == True:
            img = np.frombuffer(sonar_msg.ping.data,np.uint8)
            img = np.array(cv2.imdecode(img,cv2.IMREAD_COLOR)).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        #the image is not compressed, just use the ros numpy package
        else:
            img = ros_numpy.image.image_to_numpy(sonar_msg.ping)

        #generate a mesh grid mapping from polar to cartisian
        self.generate_map_xy(sonar_msg)

        # Detect targets and check against threshold using CFAR (in polar coordinates)
        peaks = self.detector.detect(img, self.alg)
        peaks &= img > self.threshold

        vis_img = cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR)
        vis_img = cv2.applyColorMap(vis_img, 2)
        self.feature_img_pub.publish(ros_numpy.image.numpy_to_image(vis_img, "bgr8"))

        #convert to cartisian
        peaks = cv2.remap(peaks, self.map_x, self.map_y, cv2.INTER_LINEAR)        
        locs = np.c_[np.nonzero(peaks)]

        #convert from image coords to meters
        x = locs[:,1] - self.cols / 2.
        x = (-1 * ((x / float(self.cols / 2.)) * (self.width / 2.))) #+ self.width
        y = (-1*(locs[:,0] / float(self.rows)) * self.height) + self.height
        points = np.column_stack((y,x))

        #filter the cloud using PCL
        if len(points) and self.resolution > 0:
            points = pcl.downsample(points, self.resolution)

        #remove some outliars
        if self.outlier_filter_min_points > 1 and len(points) > 0:
            # points = pcl.density_filter(points, 5, self.min_density, 1000)
            points = pcl.remove_outlier(
                points, self.outlier_filter_radius, self.outlier_filter_min_points
            )

        #publish the feature message
        self.publish_features(sonar_msg, points)
