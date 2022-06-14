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
from scipy.linalg import sqrtm,expm,logm,norm,block_diag
from matplotlib.cbook import flatten
from numpy.linalg import inv
from numpy import cos, sin


class KalmanNode(object):
	'''A class to support Kalman using DVL and IMU readings
	'''

	def __init__(self):
		#state vector = (x,y,z,roll, pitch, yaw, x_dot,y_dot,z_dot,roll_dot,pitch_dot,yaw_dot)
		self.state_vector= np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
		self.cov_matrix= np.diag([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

		self.σx, self.σy, self.σz = 0.00001, 0.00001, 0.00001
		self.σroll, self.σpitch, self.σyaw = 0.1, 0.1, 0.1
		self.σxd, self.σyd, self.σzd = 10, 10, 10
		self.σrolld, self.σpitchd, self.σyawd = 0.1, 0.1, 0.1

		self.Q = np.array([
		[self.σx, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
		[0., self.σy, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
		[0., 0., self.σz, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
		[0., 0., 0., self.σroll, 0., 0., 0., 0., 0., 0., 0., 0.],
		[0., 0., 0., 0., self.σpitch, 0., 0., 0., 0., 0., 0., 0.],
		[0., 0., 0., 0., 0., self.σyaw, 0., 0., 0., 0., 0., 0.],
		[0., 0., 0., 0., 0., 0., self.σxd, 0., 0., 0., 0., 0.],
		[0., 0., 0., 0., 0., 0., 0., self.σyd, 0., 0., 0., 0.],
		[0., 0., 0., 0., 0., 0., 0., 0., self.σzd, 0., 0., 0.],
		[0., 0., 0., 0., 0., 0., 0., 0., 0., self.σrolld, 0., 0.],
		[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., self.σpitchd, 0.],
		[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., self.σyawd]])

		self.R_dvl = np.eye(3,3)*0.00001
		self.dt_dvl = 0.2 # 5Hz
		self.H_dvl = np.array([
		[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,],
		[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,],
		[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,]])

		self.R_imu = np.eye(3,3)
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
		self.pub = rospy.Publisher("state_vector_kalman",PoseStamped,queue_size=250)
		self.pub_roll = rospy.Publisher("roll_kalman",Float32,queue_size=250)
		self.pub_pitch = rospy.Publisher("pitch_kalman",Float32,queue_size=250)
		self.pub_yaw = rospy.Publisher("yaw_kalman",Float32,queue_size=250)
		self.pub_x_vel = rospy.Publisher("xvel_kalman",Float32,queue_size=250)
		self.pub_y_vel = rospy.Publisher("yvel_kalman",Float32,queue_size=250)
		self.pub_z_vel = rospy.Publisher("zvel_kalman",Float32,queue_size=250)
		self.pub_x = rospy.Publisher("x_kalman",Float32,queue_size=250)
		self.pub_y = rospy.Publisher("y_kalman",Float32,queue_size=250)
		self.pub_z = rospy.Publisher("z_kalman",Float32,queue_size=250)

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


	def kalman_correct(self, predicted_x:np.array, predicted_P:np.array, z:np.array, H:np.array, R:np.array):
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
		K = predicted_P @ H.T @ inv(H @ predicted_P @ H.T + R)
		corrected_x = predicted_x + K @ (z - H @ predicted_x)
		corrected_P = predicted_P - K @ H @ predicted_P
		return corrected_x, corrected_P


	def rotation_matrix(self,φ:float,θ:float,ψ:float):
		"""Create the rotation matrix between the body frame and the global
		frame thanks to the three euler angles.

		Args:
			φ (float) : roll
			θ (float) : pitch
			ψ (float) : yaw
		"""
		R_roll = np.array([ [1,      0,       0],
							[0, cos(φ), -sin(φ)],
							[0, sin(φ), cos(φ)]])

		R_pitch = np.array([[cos(θ), 0, sin(θ)],
							[0,      1,      0],
							[-sin(θ), 0, cos(θ)]])

		R_yaw = np.array([  [cos(ψ), -sin(ψ), 0],
							[sin(ψ), cos(ψ),  0],
							[0,       0    , 1]])
		R = R_yaw@R_pitch@R_roll
		return R


	def dvl_callback(self, dvl_msg:DVL)->None:
		"""Handle the Kalman Filter using the DVL only. Publish the state vector.

		Args:
			dvl_msg (DVL): the message from the DVL
		"""
		dvl_measurement = np.array([[dvl_msg.velocity.x], [dvl_msg.velocity.y], [dvl_msg.velocity.z]])
		R = self.rotation_matrix(self.state_vector[3,0], self.state_vector[4,0], self.state_vector[5,0])
		dvl_global_frame = R @ dvl_measurement

		predicted_x, predicted_P = self.kalman_predict(self.state_vector, self.cov_matrix, self.dt_dvl)
		corrected_x,corrected_P = self.kalman_correct(predicted_x, predicted_P, dvl_global_frame, self.H_dvl, self.R_dvl)
		self.state_vector, self.cov_matrix = corrected_x, corrected_P


	def imu_callback(self, imu_msg:Imu)->None:
		"""Handle the Kalman Filter using the VN100 only.

		Args:
			imu_msg (Imu): the message from VN100
		"""
		quaternion = (imu_msg.orientation.x,imu_msg.orientation.y,imu_msg.orientation.z,imu_msg.orientation.w)
		roll_x, pitch_y, yaw_z = euler_from_quaternion(quaternion)
		euler_angle = np.array([[roll_x], [pitch_y], [yaw_z]])

		predicted_x, predicted_P = self.kalman_predict(self.state_vector, self.cov_matrix, self.dt_imu)
		corrected_x,corrected_P = self.kalman_correct(predicted_x, predicted_P, euler_angle, self.H_imu, self.R_imu)
		self.state_vector, self.cov_matrix = corrected_x, corrected_P

		self.send_state_vector(self.state_vector, imu_msg.header.stamp)


	def send_state_vector(self,state_vector:np.array,t:float):
		"""Publish the state vector : pose (x,y,z) and orientation (x,y,z,w).

		Args:
			state_vector (np.array): (x,y,z,roll, pitch, yaw, x_dot,y_dot,z_dot,roll_dot,pitch_dot,yaw_dot)
			dt (float): dvl_msg.header.stamp
		"""

		self.send_position(self.state_vector[0,0], self.state_vector[1,0], self.state_vector[2,0])
		self.send_angles(self.state_vector[3,0], self.state_vector[4,0], self.state_vector[5,0])
		self.send_velocities(self.state_vector[6,0], self.state_vector[7,0], self.state_vector[8,0])

		# msg = PoseStamped()
		# msg.header.stamp = t
		# msg.header.frame_id = "map"
		# msg.pose.position.x = state_vector[0,0]
		# msg.pose.position.y = state_vector[1,0]
		# msg.pose.position.z = state_vector[2,0]
		#
		# x,y,z,w = quaternion_from_euler(self.state_vector[3,0],self.state_vector[4,0],self.state_vector[5,0])
		# msg.pose.orientation.x = x
		# msg.pose.orientation.y = y
		# msg.pose.orientation.z = z
		# msg.pose.orientation.w = w
		#
		# self.pub.publish(msg)


	def send_position(self,x:float,y:float,z:float):

		msg_x = Float32()
		msg_x.data = x
		self.pub_x.publish(msg_x)

		msg_y = Float32()
		msg_y.data = y
		self.pub_y.publish(msg_y)

		msg_z = Float32()
		msg_z.data = z
		self.pub_z.publish(msg_z)


	def send_velocities(self,x_vel:float,y_vel:float,z_vel:float):
		"""Publish x,y and z velocities.

		Args:
			x_vel (float)
			y_vel (float)
			z_vel (float)
		"""
		msg_x = Float32()
		msg_x.data = x_vel
		self.pub_x_vel.publish(msg_x)

		msg_y = Float32()
		msg_y.data = y_vel
		self.pub_y_vel.publish(msg_y)

		msg_z = Float32()
		msg_z.data = z_vel
		self.pub_z_vel.publish(msg_z)


	def send_angles(self,roll:float, pitch:float, yaw:float):
		"""Publish roll, pitch and yaw.

		Args:
			roll (float)
			pitch (float)
			yaw (float)
		"""
		msg_r = Float32()
		msg_r.data = roll
		self.pub_roll.publish(msg_r)

		msg_p = Float32()
		msg_p.data = pitch
		self.pub_pitch.publish(msg_p)

		msg_y = Float32()
		msg_y.data = yaw
		self.pub_yaw.publish(msg_y)
