# python imports
import cv2
import time
import gtsam
import rospy
import pickle
import ros_numpy
import cv_bridge
import numpy as np
import tensorflow as tf
from typing import Tuple
from tensorflow import keras
from sklearn.cluster import DBSCAN
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

# ros imports
from std_msgs.msg import Header
from message_filters import Subscriber
import sensor_msgs.point_cloud2 as pc2
from message_filters import TimeSynchronizer, ApproximateTimeSynchronizer, Cache
from sensor_msgs.msg import Image, PointCloud2, PointField, CompressedImage

# bruce imports
from bruce_slam import pcl
from bruce_slam.utils.io import *
from bruce_slam.CFAR import CFAR
from bruce_slam.utils.topics import *
from bruce_msgs.msg import PoseHistory

# Argonaut imports
from sonar_oculus.msg import OculusPing, OculusPingUncompressed

# set tensorflow memeory growth, this prevents error on laptop GPU
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = keras.models.load_model("/home/jake/Desktop/open_source/src/sonar-SLAM/bruce_slam/models/real_data_model.h5")
model.make_predict_function()

class keyframe():
    '''class to be used as a sub class for 3D mapping
    '''
    def __init__(self, pose_, image_, fusedCloud_):

         self.pose = pose_
         self.image = image_
         self.segInfo = None
         self.segCloud = None
         self.ID = None
         self.width = float(image_.shape[1])
         self.height = float(image_.shape[0])
         self.maxRange = 30.
         self.FOV = 130.
         self.xRange = self.maxRange * np.cos(np.radians(90. - self.FOV / 2))
         self.fusedCloud = fusedCloud_
         self.fusedCloudDiscret = self.real2pix(fusedCloud_)
         self.matchImage = None
         self.rot = None
         self.rerun = True
         self.constructedCloud = None
         self.rerun = [True, True]
         self.containsPoints = False
         self.segcontainsPoints = False
         self.fusedCloudReg = None
         self.constructedCloudReg = None
         self.segCloudReg = None
         
    def real2pix(self,points):
        '''convert from meters to pixels
        '''
        x = ( - self.width / 2 * (points[:,2] / self.xRange)) + self.width/2
        y = self.height - (self.height * (points[:,0] / self.maxRange))
        
        return np.column_stack((x,y))

class BaysianMappingNode():
    '''Class to handle the 3D mapping problem using semantic inference
    '''

    def __init__(self):

        #list of keyframes
        self.keyframes = []
        self.numKeyframes = 0

        #subscibers
        self.image_sub = Subscriber(SONAR_TOPIC, OculusPing, queue_size = 1000)
        self.cloud_sub = Subscriber("/bruce/slam/SonarCloud", PointCloud2, queue_size = 1000)
        self.pose_step_sub = Subscriber("/bruce/slam/step", PoseHistory, queue_size = 1000)
        self.test_sub = Subscriber("/bruce/slam/step",PoseHistory,queue_size = 1000)
        self.pose_history_sub = rospy.Subscriber("/bruce/slam/pose_history",PoseHistory,self.pose_callback,queue_size=1000)
        '''self.image_cache = Cache(self.image_sub,cache_size=50)
        self.cloud_cache = Cache(self.cloud_sub,cache_size=50)'''

        #publishers
        self.mapPublisher = rospy.Publisher("sonarMap",PointCloud2,queue_size = 5)
        self.segPublisher = rospy.Publisher("sonarSegmented",PointCloud2,queue_size = 5)
        self.mapSimplePublisher = rospy.Publisher("sonarMapSimple",PointCloud2,queue_size = 5)
        self.object_1 = rospy.Publisher("object/one",Image,queue_size = 5)
        self.object_2 = rospy.Publisher("object/two",Image,queue_size = 5)

        # define time sync object
        self.timeSync = ApproximateTimeSynchronizer(
            [self.image_sub,self.cloud_sub,self.pose_step_sub],slop=0.5,queue_size=10000)

        #register callback
        self.timeSync.registerCallback(self.mappingCallback)

        #set up cv_bridge
        self.CVbridge = cv_bridge.CvBridge()

        #pose history
        self.poses = None

        #build the sonar image mask
        self.blank = np.zeros((600,1106))
        self.buildMask()

        #set up the CFAR detector
        self.detector = CFAR(20, 10, 0.5, None)
        self.thresholdCFAR = 65

        #define the classes
        self.classes = [0,1]
        
        #define a container for the grid regressions
        self.grids = {0: None,
                     1: None}

        self.guassianGrids = {0: None,
                     1: None}

        #define grid res
        self.gridRes = .1

        #define the number of required keyframes to declare an object simple
        self.minFrames = 1

        #define some image parameters
        self.width = 1106.
        self.height = 600.
        self.maxRange = 30.
        self.FOV = 130.
        self.xRange = self.maxRange * np.cos(np.radians(90. - self.FOV / 2))

        #define some parameters for baysian update
        self.maxDepth = 6.
        self.minDepth = -2.
        self.baysianRes = 30 / 600.
        self.baysianBins = int((self.maxDepth - self.minDepth) / self.baysianRes)
        self.baysianX = np.linspace(self.minDepth, self.maxDepth, self.baysianBins)
        self.measSigma = .08

        #ICP for object registration
        self.icp = pcl.ICP()
        self.icp.loadFromYaml("/home/jake/Desktop/open_source/src/sonar-SLAM/bruce_slam/config/object_icp.yaml")

        #define laser fields for fused point cloud
        self.laserFields = [
                            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
                            ]

        #define laser fields for segmented point cloud
        self.segFields = [
                            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
                            ]


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
        #self.maxRange = None
        self.predCount = 0

        #ablation study params, for logging purposes only
        self.scene = None
        self.keyframe_translation = None
        self.keyframe_rotation = None
        self.time_log = []
        self.vis_3D = True

    def generate_map_xy(self,ping):
        '''Generate a mesh grid map for the sonar image
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

    
    def buildMask(self):
        '''Build an image mask to determine if a point is outside the sonar block
        '''

        self.blank = cv2.circle(self.blank, (553, 600), 600, (255,255,255),-1)
        pts = np.array([(553,600),(1106,343),(1106,600)], np.int32)
        pts = pts.reshape((-1,1,2))
        self.blank = cv2.fillPoly(self.blank,[pts],(0,0,0))
        pts = np.array([(553,600),(0,343),(0,600)], np.int32)
        pts = pts.reshape((-1,1,2))
        self.blank = cv2.fillPoly(self.blank,[pts],(0,0,0))

    def pix2Real(self,points):
        '''Convert from pixel coords to meters (cartisian)
        '''

        #x and y in format for point cloud
        x = (-1 * ((points[:,1] / self.width) * (self.xRange * 2.))) + self.xRange
        y = (-1 * (points[:,0] / self.height) * self.maxRange) + self.maxRange

        return x, y
        
    def real2pix(self,points):
        '''convert from meters to pixels
        '''

        x = ( - self.width / 2 * (points[:,1] / self.xRange)) + self.width/2
        y = self.height - (self.height * (points[:,0] / self.maxRange))
        
        return np.column_stack((x,y))

    def extractFeatures(self,img,alg,threshold):
        '''Function to take a raw greyscale sonar image and apply CFAR
            img: raw greyscale sonar image
            detector: detector object
            alg: CFAR version to be used
            threshold: CFAR thershold
        '''

        #denoise the image
        img = cv2.fastNlMeansDenoising(img,None,10,7,21)
        
        #get raw detections
        peaks = self.detector.detect(img, alg)

        #check against threhold
        peaks &= img > threshold

        #convert to cartisian
        peaks = cv2.remap(peaks, self.map_x, self.map_y, cv2.INTER_LINEAR)

        #compile points
        points = np.c_[np.nonzero(peaks)]

        #return a numpy array
        return np.array(points), cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR)

    def queryBaysianObject(self,points,classNum):
        '''Given some points from an object and its class label, generate a 3D point cloud from 2D points
            points: 2D observations from the horizontal sonar
            classNum: the class id
            returns: 3D point cloud for this object
        '''

        #if the grid for this class exists
        if self.guassianGrids.get(classNum) != None:

            #containter for output points
            outputPoints = None

            #x and y in format for point cloud
            #x = (-1 * ((points[:,1] / self.width) * (self.xRange * 2.))) + self.xRange
            #y = (-1*(points[:,0] / self.height) * self.maxRange) + self.maxRange

            #convert to meters
            x = points[:,1] - self.cols / 2.
            x = (-1 * ((x / float(self.cols / 2.)) * (self.width / 2.)))
            y = (-1*(points[:,0] / float(self.rows)) * self.height) + self.height

            #get range and bearing
            r = np.sqrt(x**2 + y**2)
            b = np.degrees(np.arctan(x / y)) - 65.
            b_ = b + 65.
            r_ = r

            #convert the bearings to local image coordinates
            b = 256 * (b / 65.)
            b += 256
            b -= np.min(b)

            #convert spherical ranges to local image coordinates
            r = 600. * (r / 30.)
            r -= np.min(r)

            #Register the points to the grid
            sourcePoints = self.guassianGrids.get(classNum)[2]

            #set up target points, the newest set of observations
            targetPoints = np.column_stack(( np.array(b).astype(int), np.array(r).astype(int) ))
            targetPoints = self.getLeadingEdge(targetPoints)

            #compute ICP
            icpRes = self.icp.compute(targetPoints, sourcePoints, np.identity(3))
            icpStatus = icpRes[0]
            icpRes = icpRes[1]

            #call out the rotation matrix
            Rmtx = np.array([[icpRes[0][0], icpRes[0][1]],
                                [icpRes[1][0], icpRes[1][1]]])

            #get the rotation angle
            #eulerAngleeulerAngle = np.degrees(np.arctan(icpRes[0][0] / icpRes[1][0]))

            #register the new observations
            targetPoints = np.column_stack((b, r))
            targetPoints = np.array(targetPoints.dot(Rmtx.T) + np.array([ icpRes[0][2], icpRes[1][2]  ])).astype(int)

            #pull the grid 
            gridProb = self.guassianGrids.get(classNum)[0]
            gridImg = self.guassianGrids.get(classNum)[1]

            #parameters
            minConfidence = .5
            numberPeaks = 2
            
            #loop over the points
            for k in range(len(targetPoints)):

                #parse coords
                i = targetPoints[k][0]
                j = targetPoints[k][1]

                #protect for bounds
                if i < gridImg.shape[1] and i >= 0 and j < gridImg.shape[0] and j >= 0:

                    #check if we have any info at this coordinate
                    if gridImg[j][i] != 0:

                        #pull the distribution from the grid structure
                        distribution = gridProb[j][i]

                        #get n highest indexes
                        peaks = np.array(np.argpartition(distribution, -numberPeaks)[-numberPeaks:])

                        #get their probabilities
                        probabilities = np.take(distribution, peaks)

                        #convert peaks to meters
                        peaksMeters = peaks / 160.
                        peaksMeters = peaksMeters * 8.
                        peaksMeters = peaksMeters - 2.

                        #check if there is an opposing sign
                        if peaksMeters[0] < 0 and peaksMeters[1] >= 0:
                            meas = peaksMeters

                        #check if there is an opposing sign
                        elif peaksMeters[1] < 0 and peaksMeters[0] >= 0:
                            meas = peaksMeters

                        #no opposing sign, just take the argmax
                        else:
                            meas = [peaksMeters[np.argmax(probabilities)]]

                        #log the values
                        for i in range(len(meas)):

                            '''meas_i = meas[i]
                            arr = np.array([y[k], meas_i, x[k]])
                            outputPoints = np.row_stack((outputPoints, arr))'''

                            '''z = meas[i]
                            y_ = np.sqrt((r_[k]**2 - z**2) / (np.tan(np.radians(b_[k]))**2 + 1))
                            x_ = np.sign(b_[k]) * np.sqrt((r_[k]**2 - z**2) / ((1 / np.tan(np.radians(b_[k]))**2 + 1)))'''

                            z = meas[i]
                            elevationAngle = np.arccos(z / r_[k])
                            y_ = r_[k] * np.cos(np.radians(b_[k])) * np.sin(elevationAngle)
                            x_ = r_[k] * np.sin(np.radians(b_[k])) * np.sin(elevationAngle)

                            arr = np.array([y_, z, x_])
                            if outputPoints is None:
                                outputPoints = arr
                            else:
                                outputPoints = np.row_stack((outputPoints, arr))
                    
        return outputPoints

    def getLeadingEdge(self, points):
        '''Takes a set of points in polar coords and returns the closest point in each scan line
            points: 2D points from sonar, must be integers (bin numbers)
            returns: the closest point in each scan line
        '''

        #init output points
        outputPoints = np.ones((1,1))

        #get the possible bearings in the points
        bearings = list(set(points[:,0]))

        #loop over the bearings and get the closest range return
        for b in bearings:

            #get the points from this scan line
            points_i = points[points[:,0] == b]

            #get the min range
            r = np.min(points_i[:,1])

            #log the points
            if outputPoints.shape == (1,1):
                outputPoints = np.array([ b,  r ])
            else:
                outputPoints = np.row_stack((outputPoints, np.array([ b,  r ])))

        return outputPoints

    def guassianRegressObject(self,objectPoints,keyframe, box, classNum):
        '''Estimate object geometry using bayssian estimation
            keyframe: the frame to be analyzed, contains pointclouds from sensor fusion
            box: bounding box of object in qeuestion
            classNum: class label for object
            returns: nothing, updates class varibles
        '''

        #convert object points to range and bearing
        #objectX = (-1 * ((objectPoints[:,1] / self.width) * (self.xRange * 2.))) + self.xRange
        #objectY = (-1*(objectPoints[:,0] / self.height) * self.maxRange) + self.maxRange

        objectX = objectPoints[:,1] - self.cols / 2.
        objectX = (-1 * ((objectX / float(self.cols / 2.)) * (self.width / 2.)))
        objectY = (-1*(objectPoints[:,0] / float(self.rows)) * self.height) + self.height

        rangeObject = np.sqrt(objectX**2 + objectY**2)
        bearingObject = np.degrees(np.arctan2(objectY, objectX))

        #convert to image local 
        bearingObject = 256 * (bearingObject / 65.)
        bearingObject += 256
        bearingObject -= np.min(bearingObject)

        #convert spherical ranges to local image coordinates
        rangeObject = 600. * (rangeObject / 30.)
        rangeObject -= np.min(rangeObject)

        xMins = np.array([box[0][0] , box[1][0] ])
        yMins = np.array([box[0][1] , box[1][1] ])

        xMins = xMins - self.cols / 2.
        xMins = (-1 * ((xMins / float(self.cols / 2.)) * (self.width / 2.)))
        yMins = (-1*(yMins / float(self.rows)) * self.height) + self.height

        minX = xMins[0]
        maxX = xMins[1]
        minY = yMins[0]
        maxY = yMins[1]

        '''#get the box corners
        minY = box[0][1]
        minX = box[0][0]
        maxY = box[1][1]
        maxX = box[1][0]

        #convert box corners to meters
        minX = (-1 * ((minX / self.width) * (self.xRange * 2.))) + self.xRange
        minY = (-1*(minY / self.height) * self.maxRange) + self.maxRange
        maxX = (-1 * ((maxX / self.width) * (self.xRange * 2.))) + self.xRange
        maxY = (-1*(maxY / self.height) * self.maxRange) + self.maxRange'''

        #pull the point cloud from the keyframe
        x = keyframe.fusedCloud[:,0]
        z = keyframe.fusedCloud[:,1]
        y = keyframe.fusedCloud[:,2]

        #filter the 3D points based on the bounding box
        x = x[(keyframe.fusedCloud[:,0] < minY) & (keyframe.fusedCloud[:,0] > maxY) & (keyframe.fusedCloud[:,2] < minX) & (keyframe.fusedCloud[:,2] > maxX)]
        y = y[(keyframe.fusedCloud[:,0] < minY) & (keyframe.fusedCloud[:,0] > maxY) & (keyframe.fusedCloud[:,2] < minX) & (keyframe.fusedCloud[:,2] > maxX)]
        z = z[(keyframe.fusedCloud[:,0] < minY) & (keyframe.fusedCloud[:,0] > maxY) & (keyframe.fusedCloud[:,2] < minX) & (keyframe.fusedCloud[:,2] > maxX)]

        #Requires 3D points and 2D points
        if len(x) != 0 and len(bearingObject) != 0:

            #get the spherical range of each point
            rangeSpeherical = np.sqrt(x**2 + y**2 + z**2)

            #get the bearing of each point
            bearingPolar = np.degrees(np.arctan(y/x))

            #convert the bearings to local image coordinates
            bearingPolar = 256 * (bearingPolar / 65.)
            bearingPolar += 256
            bearingPolar -= np.min(bearingPolar)

            #convert spherical ranges to local image coordinates
            rangeSpeherical = 600. * (rangeSpeherical / 30.)
            rangeSpeherical -= np.min(rangeSpeherical)

            #check if we have not set up a grid before
            if self.guassianGrids.get(classNum) == None:

                #set up brand new grids
                gridProb = np.ones((int(np.max(rangeSpeherical)) + 1, int(np.max(bearingPolar)) + 1, 160)) / 160.
                gridImg = np.zeros((int(np.max(rangeSpeherical)) + 1, int(np.max(bearingPolar)) + 1))

                #get the first set of source points
                sourcePoints = np.column_stack((np.array(bearingObject).astype(int), np.array(rangeObject).astype(int)))
                sourcePoints = self.getLeadingEdge(sourcePoints)

            #if we have set up the grid before, pull the class objects
            else:

                #pull the grids
                gridProb = self.guassianGrids.get(classNum)[0]
                gridImg = self.guassianGrids.get(classNum)[1]

                #pull the source points
                sourcePoints = self.guassianGrids.get(classNum)[2]

                #set up target points, the newest set of observations
                targetPoints = np.column_stack((np.array(bearingObject).astype(int), np.array(rangeObject).astype(int)))
                targetPoints = self.getLeadingEdge(targetPoints)

                #compute ICP
                icpRes = self.icp.compute(targetPoints, sourcePoints, np.identity(3))
                icpStatus = icpRes[0]
                icpRes = icpRes[1]

                #call out the rotation matrix
                Rmtx = np.array([[icpRes[0][0], icpRes[0][1]],
                                    [icpRes[1][0], icpRes[1][1]]])

                #get the rotation angle
                #eulerAngle = np.degrees(np.arctan(icpRes[0][0] / icpRes[1][0]))

                #register the new observations
                targetPoints = targetPoints.dot(Rmtx.T)
                targetPoints += np.array([ icpRes[0][2], icpRes[1][2]  ])

                #concanate the point clouds for future timesteps
                sourcePoints = np.row_stack((sourcePoints, targetPoints))

                #transform the 3D measurnments
                targetPoints = np.column_stack((bearingPolar, rangeSpeherical)).dot(Rmtx.T) + np.array([ icpRes[0][2], icpRes[1][2] ])
                bearingPolar = targetPoints[:,0]
                rangeSpeherical = targetPoints[:,1]

                #grow the grids as required

                #if the data has more range than the grid
                if gridProb.shape[0] <= np.max(rangeSpeherical):

                    #grow the grid image
                    growth = np.zeros(( int(np.max(rangeSpeherical)) + 1, gridProb.shape[1]))
                    gridImg = np.row_stack((gridImg, growth))

                    #grow the probability grid
                    growth = np.ones(( int(np.max(rangeSpeherical)) + 1, gridProb.shape[1], 160)) / 160.
                    gridProb = np.row_stack((gridProb, growth))

                #if the data has a lower range than the grid
                '''if int(np.min(rangeSpeherical)) < 0:

                    #grow the grid image
                    growth = np.zeros(( int(abs(np.min(rangeSpeherical))), gridProb.shape[1]))
                    gridImg = np.row_stack((growth, gridImg))

                    #grow the probability grid
                    growth = np.ones(( int(abs(np.min(rangeSpeherical))), gridProb.shape[1], 160)) / 160.
                    gridProb = np.row_stack((growth, gridProb))

                    #shift the range observations as needed
                    rangeSpeherical -= growth.shape[0]'''

                #if the data has more bearing than the grid
                if gridProb.shape[1] <= np.max(bearingPolar):

                    #grow the grid image
                    growth = np.zeros(( gridProb.shape[0], int(np.max(bearingPolar)) + 1))
                    gridImg = np.column_stack((gridImg, growth))

                    #grow the probability grid
                    growth = np.ones(( gridProb.shape[0], int(np.max(bearingPolar)) + 1, 160)) / 160.
                    gridProb = np.column_stack(( gridProb, growth))

                #if the data has a lower bearing than the grid
                '''if int(np.min(bearingPolar)) < 0:

                    #grow the grid image
                    growth = np.zeros(( gridProb.shape[0], abs(int(np.min(bearingPolar))) ))
                    gridImg = np.column_stack((growth, gridImg))

                    #grow the probability grid
                    growth = np.ones(( gridProb.shape[0], abs(int(np.min(bearingPolar))), 160)) / 160.
                    gridProb = np.column_stack(( growth, gridProb))

                    #shift the bearings as needed
                    #bearingPolar -= abs(int(np.min(bearingPolar)))'''
            
            #loop over the bearings
            for bearing in list(set(bearingPolar)):

                #get the measurnments from this scan line
                z_at_bearing =  z[bearingPolar == bearing]
                range_at_bearing =  rangeSpeherical[bearingPolar == bearing]
                bearing_at_bearing = bearingPolar[bearingPolar == bearing]

                #loop over ranges
                for range_i in list(set(range_at_bearing)):

                    #get the measurnments for this range bin
                    z_at_range_bearing =  z_at_bearing[range_at_bearing == range_i]
                    bearing_at_range_bearing = bearing_at_bearing[range_at_bearing == range_i]
                    range_at_range_bearing =  range_at_bearing[range_at_bearing == range_i]

                    #dummy varible
                    p_m_z = np.ones((1,1)) 

                    #loop over all the z values here to make a max mixture
                    for zVal in z_at_range_bearing:

                        #generate the prob distrubtion for this single measurnment
                        p_m_z_i = np.exp( - np.power(self.baysianX - zVal, 2.) / (2 * np.power( self.measSigma, 2. )))

                        #if more than one measurnement at this bin record that
                        if p_m_z.shape == (1,1):
                            p_m_z = p_m_z_i
                        else:
                            p_m_z = np.column_stack((p_m_z, p_m_z_i))

                    #if there was more than one measunment take the max at that bin, a max mixture
                    if p_m_z.shape != (160,):
                        p_m_z = np.max(p_m_z, axis = 1)

                    if int(range_at_range_bearing[0]) >= 0:

                        #pull out the prior distrubtuion
                        prior = gridProb[int(range_at_range_bearing[0])][int(bearing_at_range_bearing[0])]

                        #get the post distrubtion
                        postiorior = p_m_z * prior
                        postiorior /= np.sum(postiorior)

                        #update the grids
                        gridProb[int(range_at_range_bearing[0])][int(bearing_at_range_bearing[0])] = postiorior
                        gridImg[int(range_at_range_bearing[0])][int(bearing_at_range_bearing[0])] = 255

            #push the new grids
            self.guassianGrids[classNum] = [gridProb, gridImg, sourcePoints]

    def segmentImage(self,keyframe):
        '''Segment a sonar image into classes
            sonarImg: greyscale sonar image from a keyframe
            matchImg: matches from orthoganal fusion as a sonar elevation image (float64)
        '''

        #get some info from the keyframe
        sonarImg = keyframe.image

        #get the CFAR points
        points, sonarImg = self.extractFeatures(sonarImg,"SOCA",self.thresholdCFAR)
        
        #output container
        segPoints = np.array([0, 0])
        segColors = [0]

        #outputs for downstream code
        boundingBoxes = []
        pointsBoxes = []
        probs = []

        #protect for an empty frame
        if len(points) > 0:
    
            #cluster the CFAR points
            clustering = DBSCAN(eps=5,min_samples=2).fit(points)
            labels = clustering.labels_

            #loop over the labels from the clusters
            for label in list(set(labels)):

                self.predCount +=1 
                #print self.predCount

                #if this label is not noise
                if label != -1:

                    #get the points in this cluster
                    pointsSubset = points[labels == label]

                    #get a bounding box
                    [x, y, w, h]  = cv2.boundingRect(pointsSubset)

                    #convert the bounding box to tuples 
                    refPt = [(y-5, x-5), (y + h + 5 ,x + w + 5)]

                    #get the query image
                    roi = sonarImg[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                    
                    #resize and package
                    roi = cv2.resize(roi,(40,40))
                    roi = np.array(roi).astype('uint8')

                    #Segment image for network query 
                    blur = cv2.GaussianBlur(roi,(5,5),0)
                    ret3,roi = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                    #normalize image
                    roi = roi / 255.

                    #check the blank mask
                    mask = self.blank[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

                    #if there are more than 30 CFAR points and the bounding box is not near an edge
                    if len(pointsSubset) > 30:

                        #copy the image 10 times
                        queries = []
                        for j in range(20):
                            queries.append(roi)

                        #package and predict
                        queries = np.expand_dims(queries,axis=3)
                        predictions = model.predict(queries)

                        #get the mean and variance of the predictions
                        avg = np.mean(predictions,axis=0)
                        var = np.var(predictions,axis=0)

                        #print avg
                        #print var

                        #if the confidence in the prediction is below 99% do nothing
                        if np.max(avg) > .99:

                            #record the points
                            segPoints = np.row_stack((segPoints, pointsSubset))

                            #get the class
                            pred = np.argmax(avg)

                            #color the points
                            if pred == 0:

                                #log colors for cloud
                                segColors += list(100 * np.ones(len(pointsSubset)))
                                
                                #update class zero
                                self.guassianRegressObject( pointsSubset, keyframe, refPt, 0)

                                #log for downstream code
                                boundingBoxes.append(refPt)
                                pointsBoxes.append(pointsSubset)
                                probs.append(0)

                                #plt.scatter(pointsSubset[:,1], pointsSubset[:,0], c = "r")

                            elif pred == 1:

                                #log colors for cloud
                                segColors += list(255 * np.ones(len(pointsSubset)))

                                #update class 1
                                #self.guassianRegressObject( pointsSubset, keyframe, refPt, 1)

                                #log for downstream code
                                boundingBoxes.append(refPt)
                                pointsBoxes.append(pointsSubset)
                                probs.append(1)

                                #plt.scatter(pointsSubset[:,1], pointsSubset[:,0], c = "b")

                            else:

                                #log colors for cloud
                                segColors += list(0 * np.ones(len(pointsSubset)))
                                boundingBoxes.append(refPt)
                                pointsBoxes.append(pointsSubset)
                                probs.append(2)

                                #plt.scatter(pointsSubset[:,1], pointsSubset[:,0], c = "g")
                            
                    else:

                        #record the points
                        segPoints = np.row_stack((segPoints, pointsSubset))
                        segColors += list(0 * np.ones(len(pointsSubset)))

        return segPoints, segColors, boundingBoxes, pointsBoxes, probs

    def buildCloud(self,boundingBoxes,pointsBoxes,probs):
        '''Takes a series of labeled bounding boxes and CFAR points, queries the
            grid models and creates the point cloud
            boundingBoxes: object boxes
            pointsBoxes: CFAR points in each box
            probs: class of each bounding box
        '''

        #start a cloud
        cloud = []
        #set rerun flag
        rerun = [True, True]
        status = False

        #for each detected object
        for i in range(len(pointsBoxes)):

            #get the ith values
            points = pointsBoxes[i]
            classLabel = probs[i]

            #if the classlabel is landmarks
            if probs[i] == 0 or probs[i] == 1:

                #check if we have info on this object
                if self.guassianGrids[classLabel] != None:

                    #record that we had enough info to render
                    rerun[classLabel] = False
                    status = True

                    #render this object
                    temp = self.queryBaysianObject(points, classLabel)
                    if temp is not None:
                        cloud.append(temp)
                    
        if len(cloud) > 0:
            return np.concatenate(cloud), rerun, status
        else:
            return None, rerun, status

    def publishUpdate(self) -> None:
        """Calls all publish functions, updates the map with any new transforms and/or clouds
        """

        #publish the map, seg and 3D
        if self.vis_3D:
            self.publishMap()
        else:
            self.updatePoses()

        #log the data
        if self.scene is not None:
            self.log()

        #publish the belif grids
        #self.publishGrids()

    def publishGrids(self) -> None:
        """Publish the object regression grids
        """

        #publish the belief grids
        '''if self.grids[0] != None:
            imgPub = np.array((self.grids[0][0] / np.max(self.grids[0][0])) * 255).astype(np.uint8)
            self.object_1.publish(self.CVbridge.cv2_to_imgmsg(imgPub, encoding="passthrough"))
        if self.grids[1] != None:
            imgPub = np.array((self.grids[1][0] / np.max(self.grids[1][0])) * 255).astype(np.uint8)
            self.object_2.publish(self.CVbridge.cv2_to_imgmsg(imgPub, encoding="passthrough"))'''

    def log(self) -> None:
        """Save the point clouds using pickle
        """

        inference_clouds = []
        fusion_clouds = []
        transforms = []

        for frame in self.keyframes:
            if frame.rot is not None:
                inference_clouds.append(frame.constructedCloud)
                fusion_clouds.append(frame.fusedCloud)
                transforms.append(gtsam.Pose3(gtsam.Rot3(frame.rot),gtsam.Point3(frame.pose[0], frame.pose[1], frame.pose[2])))

        file_name = str(int(self.keyframe_translation)) + "_" + str(int(np.round(np.degrees(self.keyframe_rotation))))
        path = "/home/jake/Desktop/open_source/src/sonar-SLAM/bruce_slam/notebooks/data_logs/" 
        path += self.scene + "/"

        with open(path + 'poses3D_'+file_name+'.pickle', 'wb') as handle:
            pickle.dump(transforms, handle)

        with open(path + 'inferenceclouds_'+file_name+'.pickle', 'wb') as handle:
            pickle.dump(inference_clouds, handle)

        with open(path + 'fusionclouds_'+file_name+'.pickle', 'wb') as handle:
            pickle.dump(fusion_clouds, handle)

        with open(path + 'bayesmaptime_'+file_name+'.pickle', 'wb') as handle:
            pickle.dump(self.time_log, handle)

    def updatePoses(self) -> None:
        """Update the poses, only use this function if 
        you are not running self.publishMap
        """

        # update all the old keyframes based on the new poses
        for i in range(len(self.keyframes)):
            self.keyframes[i].pose = self.poses[i]
            self.keyframes[i].rot = Rotation.from_euler('xyz', 
                    np.degrees([self.keyframes[i].pose[3], 
                    self.keyframes[i].pose[4], self.keyframes[i].pose[5]]), 
                    degrees = True).as_matrix()

    def publishMap(self) -> None:
        """Publish the maps
        """

        # create a blank cloud
        cloudOut = []
        simpleCloudOut = []
        segCloudOut = []
        segColors = []

        if self.poses is None:
            return

        # update all the old keyframes based on the new poses
        for i in range(min(len(self.keyframes),len(self.poses))):

            #update the rotation matrix and pose
            update = True
            self.keyframes[i].pose = self.poses[i]
            self.keyframes[i].rot = Rotation.from_euler('xyz', 
                    np.degrees([self.keyframes[i].pose[3], 
                    self.keyframes[i].pose[4], self.keyframes[i].pose[5]]), 
                    degrees = True).as_matrix()

            if self.keyframes[i].fusedCloudReg is None or update: 
                # register the point cloud to the global frame
                cloud = self.keyframes[i].fusedCloud.dot(self.keyframes[i].rot.T)
                cloud += np.array([self.keyframes[i].pose[0], self.keyframes[i].pose[1], self.keyframes[i].pose[2]])
                self.keyframes[i].fusedCloudReg = cloud
            #simpleCloudOut = np.row_stack((simpleCloudOut, self.keyframes[i].fusedCloudReg))
            simpleCloudOut.append(self.keyframes[i].fusedCloudReg)

            # if the cloud has any contents
            if self.keyframes[i].containsPoints == True:
                # register the point cloud to the global frame
                if self.keyframes[i].constructedCloudReg is None or update: 
                    cloud = self.keyframes[i].constructedCloud
                    cloud = cloud.dot(self.keyframes[i].rot.T)
                    cloud += np.array([self.keyframes[i].pose[0], 
                                        self.keyframes[i].pose[1], 
                                        self.keyframes[i].pose[2]])
                    self.keyframes[i].constructedCloudReg = cloud
                #cloudOut = np.row_stack((cloudOut,self.keyframes[i].constructedCloudReg))
                cloudOut.append(self.keyframes[i].constructedCloudReg)

            # check the segmented cloud for any contents
            if self.keyframes[i].segcontainsPoints == True:
                # register the segmented cloud to the global frame
                if self.keyframes[i].segCloudReg is None or update: 
                    cloud = self.keyframes[i].segCloud.dot(self.keyframes[i].rot.T)
                    cloud += np.array([self.keyframes[i].pose[0], 
                                        self.keyframes[i].pose[1], 
                                        self.keyframes[i].pose[2]])
                    self.keyframes[i].segCloudReg = cloud
                segCloudOut.append(self.keyframes[i].segCloudReg)
                segColors += self.keyframes[i].segInfo[1]

        # get a time stamp, common stamp for all three clouds
        time = rospy.get_rostime() 

        # protect from an empty cloud and publish the bayesian mapping cloud
        if len(cloudOut) > 0:
            cloudOut = np.concatenate(cloudOut)
            cloudOut = cloudOut[cloudOut[:,2] >= 0]
            header = Header()
            header.frame_id = 'map'
            header.stamp = time
            laserCloudOut = pc2.create_cloud(header, self.laserFields, cloudOut)
            self.mapPublisher.publish(laserCloudOut)

        # protect from an empty cloud and publish the segmentation cloud
        if len(segCloudOut) > 0:
            segCloudOut = np.concatenate(segCloudOut)
            header = Header()
            header.frame_id = 'map'
            header.stamp = time
            segCloudOut = pc2.create_cloud(header, self.segFields, np.column_stack((segCloudOut,np.array(segColors))))
            self.segPublisher.publish(segCloudOut)

        # protect from an empty cloud and publish the sonar fusion cloud
        if len(simpleCloudOut) > 0:
            simpleCloudOut = np.concatenate(simpleCloudOut)
            simpleCloudOut = simpleCloudOut[simpleCloudOut[:,2] >= 0]
            header = Header()
            header.frame_id = 'map'
            header.stamp = time
            laserCloudOut = pc2.create_cloud(header, self.laserFields, simpleCloudOut)
            self.mapSimplePublisher.publish(laserCloudOut)

    def pose_callback(self, pose_msg: PoseHistory) -> None:
        """Handle the incoming pose history data from the SLAM node

        Args:
            pose_msg (PoseHistory): list of updated pose information
        """
        
        pose_msg_data = list(pose_msg.data)
        self.poses = np.array(pose_msg_data).reshape(len(pose_msg_data) // 6 , 6)

    def mappingCallback(self,img_msg: OculusPing, cloud_msg: PointCloud2, dummy_msg:PoseHistory) -> None:
        """Handle the mapping proccess using the generated point cloud, the raw sonar image
        and a dummy message named pose_msg to enable time sync. 

        Args:
            img_msg (OculusPing): the raw sonar image from the newest keyframe
            cloud_msg (PointCloud2): the fused sonar image cloud from the newest keyframe
            dummy_msg (PoseHistory): a dummy message frmo the slam node with the same 
                                    timestamp as the newest keyframe. Used to sync up the 
                                    other topics
        """

        if (img_msg.header.stamp-img_msg.header.stamp).to_sec() > 0:
            print("WARNING BAD TIME")

        start_time = time.time()

        # decode the image and point cloud
        img = np.frombuffer(img_msg.ping.data,np.uint8)
        image = np.array(cv2.imdecode(img,cv2.IMREAD_GRAYSCALE)).astype(np.uint8)
        self.generate_map_xy(img_msg)
        cloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(cloud_msg)

        #create a new keyframe
        frame = keyframe(None, image, cloud)
        
        #segment the sonar image and push to the keyframe
        frame.segInfo = self.segmentImage(frame)

        #if there are points in the segcloud
        if len(frame.segInfo[0]) != 0 and frame.segInfo[0].shape != (2,):

            #compile the segmented point cloud
            pts = frame.segInfo[0]
            x = pts[:,1] - self.cols / 2.
            x = (-1 * ((x / float(self.cols / 2.)) * (self.width / 2.))) #+ self.width
            y = (-1*(pts[:,0] / float(self.rows)) * self.height) + self.height

            frame.segCloud = np.column_stack((y, x*0., x))
            frame.segcontainsPoints = True
        
        #build the cloud for this frame
        constructedCloud, frame.rerun, frame.containsPoints = self.buildCloud(frame.segInfo[2], frame.segInfo[3], frame.segInfo[4])

        #check if the constructed cloud has anything in it, if so combine the clouds
        if frame.containsPoints == True:
            frame.constructedCloud = np.row_stack((constructedCloud,frame.fusedCloud))
        elif frame.fusedCloud.shape[0] != 0:
            frame.constructedCloud = frame.fusedCloud
            frame.containsPoints = True

        #record the keyframe
        self.keyframes.append(frame)
        self.numKeyframes += 1

        # log time
        self.time_log.append(time.time() - start_time)

        #publish the new info
        self.publishUpdate()


