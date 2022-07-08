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
from kvh_gyro.msg import gyro
from scipy.spatial.transform import Rotation
from std_msgs.msg import String, Float32
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from scipy.linalg import sqrtm,expm,logm,norm,block_diag
from matplotlib.cbook import flatten
from numpy import cos, sin



class KalmanNode(object):
	'''A class to support Kalman using DVL, IMU, FOG and Depth readings.
	'''

	def __init__(self):
		#state vector = (x,y,z,roll, pitch, yaw, x_dot,y_dot,z_dot,roll_dot,pitch_dot,yaw_dot)
		self.state_vector= np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
		self.old_state_vector= np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
		self.cov_matrix= np.diag([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
		self.yaw_gyro = 90.

		self.prev_time = None #previous reading time
		self.dvl_error_timer = 0.0


	def init_node(self, ns="~")->None:
		"""Init the node, fetch all paramaters from ROS.

		Args:
			ns (str, optional): The namespace of the node. Defaults to "~".
		"""
		self.state_vector = rospy.get_param(ns + "state_vector")
		self.cov_matrix = rospy.get_param(ns + "cov_matrix")
		self.sigma_x = rospy.get_param(ns + "sigma_x")
		self.sigma_y = rospy.get_param(ns + "sigma_y")
		self.sigma_z = rospy.get_param(ns + "sigma_z")
		self.sigma_roll = rospy.get_param(ns + "sigma_roll")
		self.sigma_pitch = rospy.get_param(ns + "sigma_pitch")
		self.sigma_yaw = rospy.get_param(ns + "sigma_yaw")
		self.sigma_xd = rospy.get_param(ns + "sigma_xd")
		self.sigma_yd = rospy.get_param(ns + "sigma_yd")
		self.sigma_zd = rospy.get_param(ns + "sigma_zd")
		self.sigma_rolld = rospy.get_param(ns + "sigma_rolld")
		self.sigma_pitchd = rospy.get_param(ns + "sigma_pitchd")
		self.sigma_yawd = rospy.get_param(ns + "sigma_yawd")
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

		# Q is the Process Noise Uncertainty
		self.Q = np.array([
		[self.sigma_x, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
		[0., self.sigma_y, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
		[0., 0., self.sigma_z, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
		[0., 0., 0., self.sigma_roll, 0., 0., 0., 0., 0., 0., 0., 0.],
		[0., 0., 0., 0., self.sigma_pitch, 0., 0., 0., 0., 0., 0., 0.],
		[0., 0., 0., 0., 0., self.sigma_yaw, 0., 0., 0., 0., 0., 0.],
		[0., 0., 0., 0., 0., 0., self.sigma_xd, 0., 0., 0., 0., 0.],
		[0., 0., 0., 0., 0., 0., 0., self.sigma_yd, 0., 0., 0., 0.],
		[0., 0., 0., 0., 0., 0., 0., 0., self.sigma_zd, 0., 0., 0.],
		[0., 0., 0., 0., 0., 0., 0., 0., 0., self.sigma_rolld, 0., 0.],
		[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., self.sigma_pitchd, 0.],
		[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., self.sigma_yawd]])

		x = rospy.get_param(ns + "offset/x")
		y = rospy.get_param(ns + "offset/y")
		z = rospy.get_param(ns + "offset/z")
		self.offset_matrix = Rotation.from_euler("xyz",[x,y,z],degrees=True).as_dcm()
		self.latitude = np.radians(rospy.get_param(ns + "latitude"))
		self.earth_rate = -15.04107 * np.sin(self.latitude) / 3600.0
		self.sensor_rate = rospy.get_param(ns + "sensor_rate")

		if rospy.get_param(ns + "imu_version") == 1:
			self.imu_sub = rospy.Subscriber(IMU_TOPIC, Imu,callback=self.imu_callback,queue_size=250)
		elif rospy.get_param(ns + "imu_version") == 2:
			self.imu_sub = rospy.Subscriber(IMU_TOPIC_MK_II, Imu, callback=self.imu_callback,queue_size=250)

		self.dvl_sub = rospy.Subscriber(DVL_TOPIC,DVL,callback=self.dvl_callback,queue_size=250)
		self.depth_sub = rospy.Subscriber(DEPTH_TOPIC, Depth,callback=self.pressure_callback,queue_size=250)

		self.odom_pub = rospy.Publisher(LOCALIZATION_ODOM_TOPIC, Odometry, queue_size=250)

		self.tf1 = tf.TransformBroadcaster()
		self.dvl_max_velocity = rospy.get_param(ns + "dvl_max_velocity")
		self.use_gyro = rospy.get_param(ns + "use_gyro")

		if self.use_gyro:
			self.gyro_sub = rospy.Subscriber(GYRO_TOPIC, gyro, self.gyro_callback, queue_size=250)

		self.pose = None

		loginfo("Kalman Node is initialized")

		self.pub_yaw_state_vector = rospy.Publisher("yaw_without_gyro",Float32,queue_size=250)
		self.pub_delta_yaw_gyro = rospy.Publisher("yaw_with_gyro",Float32,queue_size=250)


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
		# State Transition Matrix A
		A = np.array([
		[1. , 0. , 0. , 0. , 0. , 0. , dt, 0. , 0. , 0. , 0. , 0. ],
		[0. , 1. , 0. , 0. , 0. , 0. , 0. , dt, 0. , 0. , 0. , 0. ],
		[0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
		[0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , dt, 0. , 0. ],
		[0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , dt, 0. ],
		[0. , 0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , 0.],
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
			z (np.array): Output Vector (measurement)
			H (np.array): Observation Matrix (H_dvl, H_imu, H_gyro, H_depth)
			R (np.array): Measurement Uncertainty (R_dvl, R_imu, R_gyro, R_depth)

		Returns:
			corrected_x (np.array): corrected estimation
			corrected_P (np.array): corrected covariance matrix

		"""
		K = predicted_P @ H.T @ inv(H @ predicted_P @ H.T + R)
		corrected_x = predicted_x + K @ (z - H @ predicted_x)
		corrected_P = predicted_P - K @ H @ predicted_P

		return corrected_x, corrected_P


	def gyro_callback(self,gyro_msg:gyro)->None:
		"""Handle the Kalman Filter using the FOG only.
		Args:
			gyro_msg (gyro): the euler angles from the gyro
		"""

		# parse message and apply the offset matrix
		dx,dy,dz = list(gyro_msg.delta)
		arr = np.array([dx,dy,dz])
		arr = arr.dot(self.offset_matrix)
		delta_roll, delta_pitch, delta_yaw = arr

		# subtract the rotation of the earth
		delta_yaw += (self.earth_rate / self.sensor_rate)
		delta_yaw_meas = np.array([[delta_yaw],[0],[0]]) #Measurement of shape(3,1) to apply Kalman

		predicted_x, predicted_P = self.state_vector, self.cov_matrix
		self.state_vector,self.cov_matrix = self.kalman_correct(predicted_x, predicted_P, delta_yaw_meas, self.H_gyro, self.R_gyro)

		self.yaw_gyro = self.yaw_gyro + self.state_vector[11][0]


	def dvl_callback(self, dvl_msg:DVL)->None:
		"""Handle the Kalman Filter using the DVL only.

		Args:
			dvl_msg (DVL): the message from the DVL
		"""

		vel = np.array([dvl_msg.velocity.x, dvl_msg.velocity.y, dvl_msg.velocity.z])

		dvl_measurement = np.array([[dvl_msg.velocity.x], [dvl_msg.velocity.y], [dvl_msg.velocity.z]])

		#if the DVL message has any velocity above the max threhold do some error handling
		if np.any(np.abs(vel) > self.dvl_max_velocity):
			if self.pose:
				self.dvl_error_timer += (dvl_msg.header.stamp - self.prev_time).to_sec()
				if self.dvl_error_timer > 5.0:
					logwarn(
						"DVL velocity ({:.1f}, {:.1f}, {:.1f}) exceeds max velocity {:.1f} for {:.1f} secs.".format(
							vel[0],
							vel[1],
							vel[2],
							self.dvl_max_velocity,
							self.dvl_error_timer,
						)
					)
				vel = self.prev_vel
				dvl_measurement = np.array([[vel[0]], [vel[1]], [vel[2]]])
				print('dvl correction')
			else:
				return
		else:
			self.dvl_error_timer = 0.0

		self.prev_time = dvl_msg.header.stamp
		self.prev_vel = vel

		predicted_x, predicted_P = self.kalman_predict(self.state_vector, self.cov_matrix, self.dt_dvl)
		corrected_x,corrected_P = self.kalman_correct(predicted_x, predicted_P, dvl_measurement, self.H_dvl, self.R_dvl)
		self.state_vector, self.cov_matrix = corrected_x, corrected_P


	def pressure_callback(self,depth_msg:Depth):
		"""Handle the Kalman Filter using the Depth.
		Args:
			depth_msg (Depth): pressure
		"""
		depth = np.array([[depth_msg.depth],[0],[0]]) #we need the shape(3,1)

		predicted_x, predicted_P = self.state_vector, self.cov_matrix
		corrected_x,corrected_P = self.kalman_correct(predicted_x, predicted_P, depth, self.H_depth, self.R_depth)
		self.state_vector, self.cov_matrix = corrected_x, corrected_P


	def imu_callback(self, imu_msg:Imu)->None:
		"""Handle the Kalman Filter using the VN100 only. Publish the state vector.

		Args:
			imu_msg (Imu): the message from VN100
		"""
		quaternion = (imu_msg.orientation.x,imu_msg.orientation.y,imu_msg.orientation.z,imu_msg.orientation.w)
		roll_x, pitch_y, yaw_z = euler_from_quaternion(quaternion)
		euler_angle = np.array([[roll_x], [pitch_y], [yaw_z]])

		predicted_x, predicted_P = self.state_vector, self.cov_matrix
		corrected_x,corrected_P = self.kalman_correct(predicted_x, predicted_P, euler_angle, self.H_imu, self.R_imu)
		self.state_vector, self.cov_matrix = corrected_x, corrected_P

		trans_x = self.state_vector[0][0] - self.old_state_vector[0][0]
		trans_y = self.state_vector[1][0] - self.old_state_vector[1][0]

		self.send_state_vector(self.state_vector[5][0])
		self.send_yaw(self.yaw_gyro)

		if self.use_gyro:
			R = gtsam.Rot3.Ypr(self.yaw_gyro, self.state_vector[4][0], self.state_vector[3][0])
		else:
			R = gtsam.Rot3.Ypr(self.state_vector[5][0], self.state_vector[4][0], self.state_vector[3][0])

		if self.pose:
			local_point = gtsam.Point2(trans_x, trans_y)
			if self.use_gyro:
				pose2 = gtsam.Pose2(self.pose.x(), self.pose.y(), self.yaw_gyro)
			else:
				pose2 = gtsam.Pose2(self.pose.x(), self.pose.y(), self.pose.rotation().yaw())
			point = pose2.transformFrom(local_point)
			self.pose = gtsam.Pose3(R, gtsam.Point3(point[0], point[1], 0))
		else:
			self.pose = gtsam.Pose3(R, gtsam.Point3(0, 0, 0))

		self.old_state_vector = self.state_vector

		self.send_odometry(imu_msg.header.stamp)


	def send_odometry(self,t:float):
		"""Publish the pose.

		Args:
			t (float): time from imu_msg
		"""

		if self.pose is None:
			return

		header = rospy.Header()
		header.stamp = t
		header.frame_id = "odom"

		odom_msg = Odometry()
		odom_msg.header = header
		odom_msg.pose.pose = g2r(self.pose)

		odom_msg.child_frame_id = "base_link"

		odom_msg.twist.twist.linear.x = self.state_vector[6][0]
		odom_msg.twist.twist.linear.y = self.state_vector[7][0]
		odom_msg.twist.twist.linear.z = self.state_vector[8][0]
		odom_msg.twist.twist.angular.x = self.state_vector[9][0]
		odom_msg.twist.twist.angular.y = self.state_vector[10][0]
		odom_msg.twist.twist.angular.z = self.state_vector[11][0]

		self.odom_pub.publish(odom_msg)

		pose_msg = odom_msg.pose.pose

		p = odom_msg.pose.pose.position
		q = odom_msg.pose.pose.orientation

		self.tf1.sendTransform(
			(p.x, p.y, p.z), (q.x, q.y, q.z, q.w), header.stamp, "base_link", "odom")


	def send_state_vector(self,yaw):
		msg = Float32()
		msg.data = yaw
		self.pub_yaw_state_vector.publish(msg)
	def send_yaw(self,yaw):
		msg = Float32()
		msg.data = yaw
		self.pub_delta_yaw_gyro.publish(msg)
