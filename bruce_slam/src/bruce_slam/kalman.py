#python imports
from typing import Tuple
import tf
import rospy
import gtsam
import numpy as np

import rospy
from std_msgs.msg import String, Float32

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
#---
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class KalmanNode(object):

	def __init__(self):
		#state vector = (x,y,z,roll, pitch, yaw, x_dot,y_dot,z_dot,roll_dot,pitch_dot,yaw_dot)
		# nice comment here, this makes it very easy to follow what you are doing
		self.state_vector= np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
		self.cov_matrix= np.diag([0,0,0,0,0,0,0,0,0,0,0,0])


	def init_node(self, ns="~")->None:
		self.dvl_sub = rospy.Subscriber(DVL_TOPIC,DVL,callback=self.dvl_callback,queue_size=10)
		self.pub = rospy.Publisher("state_vector_with_kalman",PoseStamped,queue_size=10) # change queue size to 250
		self.pubtheta = rospy.Publisher("yaw_kalman_topic",Float32,queue_size=250)

		if rospy.get_param(ns + "imu_version") == 1:
			self.imu_sub = rospy.Subscriber(IMU_TOPIC, Imu,callback=self.imu_callback,queue_size=10) # change queue size to 250
		elif rospy.get_param(ns + "imu_version") == 2:
			self.imu_sub = rospy.Subscriber(IMU_TOPIC_MK_II, Imu, callback=self.imu_callback,queue_size=10) # change queue size to 250
		# self.tf = tf.TransformBroadcaster()
		loginfo("Localization node is initialized") # change this to "kalman node is initialized"

	# not clear to me what the "up" in xup is supposed to mean
	# let's add type hints and docstring to all functions, I did this one so you have an idea
	def kalman_predict(self,xup:np.array,Pup:np.array,dt:float):
		"""Write a BREIF summary of what this function does

		Args:
			xup (np.array): _description_
			Pup (np.array): _description_
			dt (float): _description_
		"""
		# is this the A matrix we defined together I personally find this a bit confusing, mainly because of the syntax
		# can we make this a class varible and write it out super neat like you did the state vector?
		A = np.eye(12,12) + np.diag(np.ones(6)*dt,6) 
		P1 = A @ Pup @ A.T # this equation is not complete, we need to do P = A*P*A_transpose + Q. Q is proccess noise, defined by you I reccomend a diag matrix
		x1 = A @ xup 
		self.state_vector,self.cov_matrix = x1,P1

	# add type hints and doc string
	# fix spelling on kalman correc->kalman_correct
	def kalman_correc(self,x0,P,z,H):
		R = np.eye(3,3) # makes sense that this is diag matrix, but we want to be able to tune the values in this matrix. Let's make it a class varible that is easy to change the contents
		K = P @ H.T @ inv(H@P@H.T + R)
		xup = x0 + K @ (z-H@x0)
		Pup = (np.eye(len(x0))-K @ H) @ P # this equation is incomplete
		return(xup,Pup)

	# add doc string
	def dvl_callback(self, dvl_msg:DVL)->None:
		dt=0.2 #5Hz # move to class varible self.dvl_dt
		H = np.zeros((3,12)) # move to class varible, self.H_dvl
		H[0,6]=H[1,7]=H[2,8]=1
		vel = np.array([[dvl_msg.velocity.x], [dvl_msg.velocity.y], [dvl_msg.velocity.z]])

		self.state_vector[6,0]=dvl_msg.velocity.x # we are updating the state vector below using the kalman filter
		self.state_vector[7,0]=dvl_msg.velocity.y # no need for this, but you can comment your correction function
		self.state_vector[8,0]=dvl_msg.velocity.z # and use this code to test your prediction function

		# we want to do predict then correct
		xup,Pup = self.kalman_correc(self.state_vector,self.cov_matrix,vel,H) # once we need complete the kalman correct function we need to set our state to it
		# so here we would say
		# self.state_vector = xup
		# self.covariance_matrix = Pup
		self.kalman_predict(xup,Pup,dt)
		self.send_state_vector(self.state_vector,dvl_msg.header.stamp)

	# add docstring
	def imu_callback(self, imu_msg:Imu)->None:
		dt=0.005 #200Hz # move to class varible
		H = np.zeros((3,12)) # move to clas varible H_imu
		H[0,3]=H[1,4]=H[2,5]=1
		quaternion = (imu_msg.orientation.x,imu_msg.orientation.y,imu_msg.orientation.z,imu_msg.orientation.w)
		roll_x, pitch_y, yaw_z = euler_from_quaternion(quaternion)
		euler_angle = np.array([[roll_x], [pitch_y], [yaw_z]])

		self.state_vector[3,0] = roll_x # see note in dvl_callback, we don't need to do this
		self.state_vector[4,0] = pitch_y
		self.state_vector[5,0] = yaw_z

		# we want to do predict then correct
		xup,Pup = self.kalman_correc(self.state_vector,self.cov_matrix,euler_angle,H) 
		# self.state_vector = xup
		# self.covariance_matrix = Pup
		self.kalman_predict(xup,Pup,dt)
		# self.send_state_vector(self.state_vector) -> already send in dvl_callback

	# type hints and docstring
	def send_state_vector(self,state_vector,t):
		msg = PoseStamped()
		msg.header.stamp = t #not sure
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

	# type hints and docstring
	def send_theta(self,yaw):
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
