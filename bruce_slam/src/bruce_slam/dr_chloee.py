class DeadReckoningNode(object):

	def __init__(self):
        #state vector = (x,y,z,roll, pitch, yaw, x_dot,y_dot,z_dot,roll_dot,pitch_dot,yaw_dot)
        self.state_vector= np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])


    def init_node(self, ns="~")->None:
        #subscribers
        self.dvl_sub = Subscriber(DVL_TOPIC, DVL)
        self.gyro_sub = Subscriber(GYRO_INTEGRATION_TOPIC, Odometry)
		self.depth_sub = Subscriber(DEPTH_TOPIC, Depth)

		if rospy.get_param(ns + "imu_version") == 1:
			self.imu_sub = Subscriber(IMU_TOPIC, Imu)
		elif rospy.get_param(ns + "imu_version") == 2:
			self.imu_sub = Subscriber(IMU_TOPIC_MK_II, Imu)

        self.pub = rospy.Publisher("imu_sensors",float,queue_size=10)

        loginfo("Localization node is initialized")


    def imu_callback(self, imu_msg:Imu)->None:
		"""
		Args:
			imu_msg (Imu): the message from VN100
		"""
		#convert the imu message from msg to gtsam rotation object
		rot = r2g(imu_msg.orientation)
        #kalman prediction
        #kalman update



	def kalman_predict(self,state_vector,dt):
		#takes x(t-1) as argument
		#returns a prediction of x(t)
		I = np.eye(12,12)
		At = np.diag(np.ones(6)*dt,6)
		A = I + At
		return A@x
