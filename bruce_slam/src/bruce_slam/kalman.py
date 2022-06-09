#python imports
from typing import Tuple
import tf
import rospy
import gtsam
import numpy as np
import rospy

# ros-python imports
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Imu
from message_filters import ApproximateTimeSynchronizer, Cache, Subscriber

# import custom messages
from kvh_gyro.msg import gyro as GyroMsg
from rti_dvl.msg import DVL
from bar30_depth.msg import Depth

# bruce imports
from bruce_slam.utils.topics import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.io import *
from bruce_slam.utils.visualization import ros_colorline_trajectory

# new imports
from std_msgs.msg import String, Float32
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class KalmanNode(object):
	'''A class to support Kalman using DVL and IMU readings
	'''

	def __init__(self):
		#state vector = (x,y,z,roll, pitch, yaw, x_dot,y_dot,z_dot,roll_dot,pitch_dot,yaw_dot)
		self.state_vector= np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
		self.cov_matrix= np.diag([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		self.Q = np.eye(12,12) #proccess noise
		self.R = np.eye(3,3)
		self.dt_dvl = 0.2 # 5Hz
		self.H_dvl = np.array([
		[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,],
		[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,],
		[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,]])
		self.dt_imu = 0.005 # 200Hz
		self.H_imu = np.array([
		[0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
		[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
		[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])


	def init_node(self, ns="~")->None:
		"""Init the node, fetch all paramaters from ROS

		Args:
			ns (str, optional): The namespace of the node. Defaults to "~".
		"""
		self.dvl_sub = rospy.Subscriber(DVL_TOPIC,DVL,callback=self.dvl_callback,queue_size=10)
		self.pub = rospy.Publisher("state_vector_with_kalman",PoseStamped,queue_size=250)
		self.pubtheta = rospy.Publisher("yaw_kalman_topic",Float32,queue_size=250)

		if rospy.get_param(ns + "imu_version") == 1:
			self.imu_sub = rospy.Subscriber(IMU_TOPIC, Imu,callback=self.imu_callback,queue_size=250)
		elif rospy.get_param(ns + "imu_version") == 2:
			self.imu_sub = rospy.Subscriber(IMU_TOPIC_MK_II, Imu, callback=self.imu_callback,queue_size=250)

		loginfo("Kalman Node is initialized")


	def kalman_predict(self,previous_x:np.array,previous_P:np.array,dt:float):
		"""Project the state and the error covariance ahead.

		Args:
			previous_x (np.array): value of the previous state vector
			previous_P (np.array): value of the previous covariance matrix
			dt (float): may be dt_imu or dt_dvl

		Returns:
			predicted_x (np.array): predicted estimation
			predicted_P (np.array): predicted covariance matrix
		"""
		A = np.array([
		[1. , 0. , 0. , 0. , 0. , 0. , dt, 0. , 0. , 0. , 0. , 0. ],
		[0. , 1. , 0. , 0. , 0. , 0. , 0. , dt, 0. , 0. , 0. , 0. ],
		[0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , dt, 0. , 0. , 0. ],
		[0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , dt, 0. , 0. ],
		[0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , dt, 0. ],
		[0. , 0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , dt],
		[0. , 0. , 0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. ],
		[0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. ],
		[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. ],
		[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. , 0. , 0. ],
		[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. , 0. ],
		[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. ]])

		predicted_P = A @ previous_P @ A.T + self.Q
		predicted_x = A @ previous_x
		return predicted_x, predicted_P


	def kalman_correct(self,predicted_x:np.array,predicted_P:np.array,z:np.array,H:np.array):
		"""Measurement Update.

		Args:
			predicted_x (np.array): predicted state vector with kalman_predict()
			predicted_P (np.array): predicted covariance matrix with kalman_predict()
			z (np.array): measurement
			H (np.array): may be H_dvl or H_imu

		Returns:
			corrected_x (np.array): corrected estimation
			corrected_P (np.array): corrected covariance matrix

		"""
		K = predicted_P @ H.T @ inv(H @ predicted_P @ H.T + self.R)
		corrected_x = predicted_x + K @ (z - H @ predicted_x)
		corrected_P = predicted_P - K @ H @ predicted_P
		return corrected_x, corrected_P


	def dvl_callback(self, dvl_msg:DVL)->None:
		"""Handle the Kalman Filter using the DVL only. Publish the state vector.

		Args:
			dvl_msg (DVL): the message from the DVL
		"""
		vel = np.array([[dvl_msg.velocity.x], [dvl_msg.velocity.y], [dvl_msg.velocity.z]])

		predicted_x, predicted_P = self.kalman_predict(self.state_vector, self.cov_matrix, self.dt_dvl)
		corrected_x,corrected_P = self.kalman_correct(predicted_x, predicted_P, vel, self.H_dvl)
		self.state_vector, self.cov_matrix = corrected_x, corrected_P

		self.send_state_vector(self.state_vector,dvl_msg.header.stamp)


	def imu_callback(self, imu_msg:Imu)->None:
		"""Handle the Kalman Filter using the VN100 only.

		Args:
			imu_msg (Imu): the message from VN100
		"""
		quaternion = (imu_msg.orientation.x,imu_msg.orientation.y,imu_msg.orientation.z,imu_msg.orientation.w)
		roll_x, pitch_y, yaw_z = euler_from_quaternion(quaternion)
		euler_angle = np.array([[roll_x], [pitch_y], [yaw_z]])

		predicted_x, predicted_P = self.kalman_predict(self.state_vector, self.cov_matrix, self.dt_imu)
		corrected_x,corrected_P = self.kalman_correct(predicted_x, predicted_P, euler_angle, self.H_imu)
		self.state_vector, self.cov_matrix = corrected_x, corrected_P


	def send_state_vector(self,state_vector:np.array,t:float):
		"""Publish the state vector : pose (x,y,z) and orientation (x,y,z,w).

		Args:
			state_vector (np.array): value of the state vector
			dt (float): dvl_msg.header.stamp
		"""
		msg = PoseStamped()
		msg.header.stamp = t # took dvl_msg.header.stamp
		msg.header.frame_id = "map"
		msg.pose.position.x = state_vector[0,0]
		msg.pose.position.y = state_vector[1,0]
		msg.pose.position.z = state_vector[2,0]

		self.send_theta(self.state_vector[5,0])

		x,y,z,w = quaternion_from_euler(self.state_vector[3,0],self.state_vector[4,0],self.state_vector[5,0])
		msg.pose.orientation.x = x
		msg.pose.orientation.y = y
		msg.pose.orientation.z = z
		msg.pose.orientation.w = w

		self.pub.publish(msg)


	def send_theta(self,yaw:float):
		"""Publish the yaw.

		Args:
			yaw (float): theta
		"""
		msg = Float32()
		msg.data = yaw
		self.pubtheta.publish(msg)


	# def pressure_callback(self, depth_msg:Depth)->None:
	#
	#     #kalman prediction
	#     #kalman update
	#
	# def fiber_opitc_gyro(self,fog_msg):
	#     # FOG gives theta_dot
