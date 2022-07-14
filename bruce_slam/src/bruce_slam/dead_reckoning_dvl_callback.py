#python imports
import tf
import rospy
import gtsam
import numpy as np

import rospy
from std_msgs.msg import String

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


class DeadReckoningNode(object):

	def __init__(self):
		#state vector = (x,y,z,roll, pitch, yaw, x_dot,y_dot,z_dot,roll_dot,pitch_dot,yaw_dot)
		self.state_vector= np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
		self.cov_matrix= np.diag([0,0,0,0,0,0,0,0,0,0,0,0])
		#self.start=0

	def init_node(self, ns="~")->None:
		self.dvl_sub = rospy.Subscriber(DVL_TOPIC,DVL,callback=self.dvl_callback,queue_size=10)
		self.pub = rospy.Publisher("state_vector_with_kalman",String,queue_size=10)

		# if rospy.get_param(ns + "imu_version") == 1:
		# 	self.imu_sub = rospy.Subscriber(IMU_TOPIC, Imu,callback=self.imu_callback,queue_size=10)
		# elif rospy.get_param(ns + "imu_version") == 2:
		# 	self.imu_sub = rospy.Subscriber(IMU_TOPIC_MK_II, Imu, callback=self.imu_callback,queue_size=10)

		loginfo("Localization node is initialized")

	def kalman_predict(self,xup,Pup,dt):
		A = np.eye(12,12) + np.diag(np.ones(6)*dt,6)
		P1 = A @ Pup @ A.T
		x1 = A @ xup
		self.state_vector,self.cov_matrix = x1,P1

	def kalman_correc(self,x0,P,z,H):
		R = np.eye(3,3)
		K = P @ H.T @ inv(H@P@H.T + R)
		xup = x0 + K @ (z-H@x0)
		Pup = (np.eye(len(x0))-K @ H) @ P
		return(xup,Pup)

	def dvl_callback(self, dvl_msg:DVL)->None:
		dt=0.2 #5Hz
		H = np.zeros((3,12))
		H[0,6]=H[1,7]=H[2,8]=1
		vel = np.array([[dvl_msg.velocity.x], [dvl_msg.velocity.y], [dvl_msg.velocity.z]])

		#if self.start < 3 :
		self.state_vector[6,0]=dvl_msg.velocity.x
		self.state_vector[7,0]=dvl_msg.velocity.y
		self.state_vector[8,0]=dvl_msg.velocity.z
			#self.start = self.start + 1

		xup,Pup = self.kalman_correc(self.state_vector,self.cov_matrix,vel,H)
		self.kalman_predict(xup,Pup,dt)

		self.send_sv(self.state_vector,vel)

	#
	# def imu_callback(self, imu_msg:Imu)->None:
	# 	H = np.zeros((3,12))
	# 	H[0,3]=H[1,4]=H[2,5]=1
	#
	#     #convert the imu message from msg to gtsam rotation object
	#     rot = r2g(imu_msg.orientation)
	# 	#print r2g etc



	def send_sv(self,state_vector,vel):
		msg = String()
		msg.data = str(state_vector)
		self.pub.publish(msg)

		# msg2 = String()
		# msg2.data = str(vel)
		# self.pub.publish(msg2)

	# def coord(self,state_vector):
	# 	x,y=state_vector[0,0],state_vector[1,0]
	# 	X.append(x)
	# 	Y.append(y)
	#
	# def init_figure(self,xmin,xmax,ymin,ymax):
	#     fig = plt.figure()
	#     ax = fig.add_subplot(111, aspect='equal')
	#     ax.xmin=xmin
	#     ax.xmax=xmax
	#     ax.ymin=ymin
	#     ax.ymax=ymax
	#     #clear(ax)
	#     return ax
	#
	# def draw_trajectory(self,state_vector):
	# 	ax=self.init_figure(-50,50,-60,60)
	# 	plt.plot([state_vector[0,0]],[state_vector[1,0]])
	# 	plt.show()




	# def pressure_callback(self, depth_msg:Depth)->None:
	#
	#     #kalman prediction
	#     #kalman update
	#
	# def fiber_opitc_gyro(self,fog_msg):
	#     # FOG gives theta_dot
