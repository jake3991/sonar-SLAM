#python imports
import tf
import rospy
import gtsam
import numpy as np
import rospy
from scipy.spatial.transform import Rotation


# standard ros message imports
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion

# import custom messages
from rti_dvl.msg import DVL
from bar30_depth.msg import Depth
from kvh_gyro.msg import gyro

# bruce imports
from bruce_slam.utils.topics import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.io import *

class KalmanNode(object):
	'''A class to support Kalman filtering using DVL, IMU, FOG and Depth readings.
	'''

	def __init__(self):

		#state vector = (x,y,z,roll, pitch, yaw, x_dot,y_dot,z_dot,roll_dot,pitch_dot,yaw_dot)
		self.state_vector= np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
		self.cov_matrix= np.diag([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
		self.yaw_gyro = 0.
		self.imu_yaw0 = None


	def init_node(self, ns="~")->None:
		"""Init the node, fetch all paramaters.

		Args:
			ns (str, optional): The namespace of the node. Defaults to "~".
		"""

		self.state_vector = rospy.get_param(ns + "state_vector")
		self.cov_matrix = rospy.get_param(ns + "cov_matrix")
		self.R_dvl = rospy.get_param(ns + "R_dvl")
		self.dt_dvl = rospy.get_param(ns + "dt_dvl")
		self.H_dvl = np.array(rospy.get_param(ns + "H_dvl"))
		self.R_imu = rospy.get_param(ns + "R_imu")
		self.dt_imu = rospy.get_param(ns + "dt_imu")
		self.H_imu = np.array(rospy.get_param(ns + "H_imu"))
		self.H_gyro = np.array(rospy.get_param(ns + "H_gyro"))
		self.R_gyro = rospy.get_param(ns + "R_gyro")
		self.dt_gyro = rospy.get_param(ns + "dt_gyro")
		self.H_depth = np.array(rospy.get_param(ns + "H_depth"))
		self.R_depth = rospy.get_param(ns + "R_depth")
		self.dt_depth = rospy.get_param(ns + "dt_depth")
		self.Q = rospy.get_param(ns + "Q") # Process Noise Uncertainty
		self.A_imu = rospy.get_param(ns + "A_imu") # State Transition Matrix
		x = rospy.get_param(ns + "offset/x") # gyroscope offset matrix
		y = rospy.get_param(ns + "offset/y")
		z = rospy.get_param(ns + "offset/z")
		self.offset_matrix = Rotation.from_euler("xyz",[x,y,z],degrees=True).as_matrix()
		self.dvl_max_velocity = rospy.get_param(ns + "dvl_max_velocity")
		self.use_gyro = rospy.get_param(ns + "use_gyro")
		self.imu_offset = np.radians(rospy.get_param(ns + "imu_offset"))

		# check which version of the imu we are using
		if rospy.get_param(ns + "imu_version") == 1:
			self.imu_sub = rospy.Subscriber(IMU_TOPIC, Imu,callback=self.imu_callback,queue_size=250)
		elif rospy.get_param(ns + "imu_version") == 2:
			self.imu_sub = rospy.Subscriber(IMU_TOPIC_MK_II, Imu, callback=self.imu_callback,queue_size=250)

		# define the other subcribers
		self.dvl_sub = rospy.Subscriber(DVL_TOPIC,DVL,callback=self.dvl_callback,queue_size=250)
		self.depth_sub = rospy.Subscriber(DEPTH_TOPIC, Depth,callback=self.pressure_callback,queue_size=250)
		self.odom_pub_kalman = rospy.Publisher(LOCALIZATION_ODOM_TOPIC, Odometry, queue_size=250)

		# define the transfor broadcaster
		self.tf1 = tf.TransformBroadcaster()

		# if we are using the gyroscope set up the subscribers
		if self.use_gyro:
			self.gyro_sub = rospy.Subscriber(GYRO_TOPIC, gyro, self.gyro_callback, queue_size=250)

		# define the initial pose, all zeros
		R_init = gtsam.Rot3.Ypr(0.,0.,0.)
		self.pose = gtsam.Pose3(R_init, gtsam.Point3(0, 0, 0))

		# log at the roslevel that we are done with init 
		loginfo("Kalman Node is initialized")


	def kalman_predict(self,previous_x:np.array,previous_P:np.array,A:np.array):
		"""Propagate the state and the error covariance ahead.

		Args:
			previous_x (np.array): value of the previous state vector
			previous_P (np.array): value of the previous covariance matrix
			A (np.array): State Transition Matrix

		Returns:
			predicted_x (np.array): predicted estimation
			predicted_P (np.array): predicted covariance matrix
		"""

		A = np.array(A)
		predicted_P = A @ previous_P @ A.T + self.Q
		predicted_x = A @ previous_x

		return predicted_x, predicted_P


	def kalman_correct(self, predicted_x:np.array, predicted_P:np.array, z:np.array, H:np.array, R:np.array):
		"""Measurement Update.

		Args:
			predicted_x (np.array): predicted state vector with kalman_predict()
			predicted_P (np.array): predicted covariance matrix with kalman_predict()
			z (np.array): Output Vector (measurement)
			H (np.array): Observation Matrix (H_dvl, H_imu, H_gyro, H_depth)
			R (np.array): Measurement Uncertainty (R_dvl, R_imu, R_gyro, R_depth)

		Returns:
			corrected_x (np.array): corrected estimation
			corrected_P (np.array): corrected covariance matrix

		"""

		K = predicted_P @ H.T @ np.linalg.inv(H @ predicted_P @ H.T + R)
		corrected_x = predicted_x + K @ (z - H @ predicted_x)
		corrected_P = predicted_P - K @ H @ predicted_P

		return corrected_x, corrected_P


	def gyro_callback(self,gyro_msg:gyro)->None:
		"""Handle the Kalman Filter using the FOG only.
		Args:
			gyro_msg (gyro): the euler angles from the gyro
		"""

		# parse message and apply the offset matrix
		arr = np.array(list(gyro_msg.delta))
		arr = arr.dot(self.offset_matrix) 
		delta_yaw_meas = np.array([[arr[0]],[0],[0]]) #Measurement of shape(3,1) to apply Kalman
		self.state_vector,self.cov_matrix = self.kalman_correct(self.state_vector, self.cov_matrix, delta_yaw_meas, self.H_gyro, self.R_gyro)
		self.yaw_gyro += self.state_vector[11][0]

	def dvl_callback(self, dvl_msg:DVL)->None:
		"""Handle the Kalman Filter using the DVL only.

		Args:
			dvl_msg (DVL): the message from the DVL
		"""

		# parse the dvl velocites 
		dvl_measurement = np.array([[dvl_msg.velocity.x], [dvl_msg.velocity.y], [dvl_msg.velocity.z]])

		# We do not do a kalman correction if the speed is high.
		if np.any(np.abs(dvl_measurement) > self.dvl_max_velocity):
			return
		else:
			self.state_vector,self.cov_matrix  = self.kalman_correct(self.state_vector, self.cov_matrix, dvl_measurement, self.H_dvl, self.R_dvl)


	def pressure_callback(self,depth_msg:Depth):
		"""Handle the Kalman Filter using the Depth.
		Args:
			depth_msg (Depth): pressure
		"""

		depth = np.array([[depth_msg.depth],[0],[0]]) # We need the shape(3,1) for the correction
		self.state_vector,self.cov_matrix = self.kalman_correct(self.state_vector, self.cov_matrix, depth, self.H_depth, self.R_depth)

	def imu_callback(self, imu_msg:Imu)->None:
		"""Handle the Kalman Filter using the VN100 only. Publish the state vector.

		Args:
			imu_msg (Imu): the message from VN100
		"""

		# Kalman prediction
		predicted_x, predicted_P = self.kalman_predict(self.state_vector, self.cov_matrix, self.A_imu)

		# parse the IMU measurnment
		roll_x, pitch_y, yaw_z = euler_from_quaternion((imu_msg.orientation.x,imu_msg.orientation.y,imu_msg.orientation.z,imu_msg.orientation.w))
		euler_angle = np.array([[self.imu_offset+roll_x], [pitch_y], [yaw_z]])

		#if we have no yaw yet, set this one as zero
		if self.imu_yaw0 is None:
			self.imu_yaw0 = yaw_z
		
		# make yaw relative to the first meas
		euler_angle[2] -= self.imu_yaw0

		# Kalman correction
		self.state_vector,self.cov_matrix = self.kalman_correct(predicted_x, predicted_P, euler_angle, self.H_imu, self.R_imu)

		# Use filtered velocity to update our x and y estimates
		trans_x = self.state_vector[6][0]*self.dt_imu # x update
		trans_y = self.state_vector[7][0]*self.dt_imu # y update
		local_point = gtsam.Point2(trans_x, trans_y)

		# check if we are using the FOG
		if self.use_gyro:
			R = gtsam.Rot3.Ypr(self.yaw_gyro,self.state_vector[4][0], self.state_vector[3][0]) 
			pose2 = gtsam.Pose2(self.pose.x(), self.pose.y(), self.yaw_gyro)
		else: # We are not using the gyro
			R = gtsam.Rot3.Ypr(self.state_vector[5][0], self.state_vector[4][0], self.state_vector[3][0])
			pose2 = gtsam.Pose2(self.pose.x(), self.pose.y(), self.pose.rotation().yaw())

		# update our pose estimate and send out the odometry message
		point = pose2.transformFrom(local_point)
		self.pose = gtsam.Pose3(R, gtsam.Point3(point[0], point[1], 0))
		self.send_odometry(imu_msg.header.stamp)

	def send_odometry(self,t:float):
		"""Publish the pose.
		Args:
			t (float): time from imu_msg
		"""
		
		header = rospy.Header()
		header.stamp = t
		header.frame_id = "odom"
		odom_msg = Odometry()
		odom_msg.header = header
		odom_msg.pose.pose = g2r(self.pose)
		odom_msg.child_frame_id = "base_link"
		odom_msg.twist.twist.linear.x = 0.
		odom_msg.twist.twist.linear.y = 0.
		odom_msg.twist.twist.linear.z = 0.
		odom_msg.twist.twist.angular.x = 0.
		odom_msg.twist.twist.angular.y = 0.
		odom_msg.twist.twist.angular.z = 0.
		self.odom_pub_kalman.publish(odom_msg)

		self.tf1.sendTransform(
			(odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z),
			(odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w),
			header.stamp, "base_link", "odom")
