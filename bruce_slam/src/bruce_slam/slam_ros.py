# python imports
import threading
import tf
import rospy
import cv_bridge
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from message_filters import  Subscriber
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseWithCovarianceStamped
from message_filters import ApproximateTimeSynchronizer

# bruce imports
from bruce_slam.utils.io import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.visualization import *
from bruce_slam.slam import SLAM, Keyframe
from bruce_msgs.msg import PoseHistory
from bruce_slam import pcl

# Argonaut imports
from sonar_oculus.msg import OculusPing


class SLAMNode(SLAM):
    """This class takes the functionality from slam.py and implments it in the ros
    environment. 
    """
    
    def __init__(self):
        super(SLAMNode, self).__init__()

        # the threading lock
        self.lock = threading.RLock()

    def init_node(self, ns="~")->None:
        """Configures the SLAM node

        Args:
            ns (str, optional): The namespace of the node. Defaults to "~".
        """

        #keyframe paramters, how often to add them
        self.keyframe_duration = rospy.get_param(ns + "keyframe_duration")
        self.keyframe_duration = rospy.Duration(self.keyframe_duration)
        self.keyframe_translation = rospy.get_param(ns + "keyframe_translation")
        self.keyframe_rotation = rospy.get_param(ns + "keyframe_rotation")

        #SLAM paramter, are we using SLAM or just dead reckoning
        self.enable_slam = rospy.get_param(ns + "enable_slam")
        print("SLAM STATUS: ", self.enable_slam)

        #noise models
        self.prior_sigmas = rospy.get_param(ns + "prior_sigmas")
        self.odom_sigmas = rospy.get_param(ns + "odom_sigmas")
        self.icp_odom_sigmas = rospy.get_param(ns + "icp_odom_sigmas")

        #max ICP run time
        self.max_icp_cov_run_time = rospy.get_param(ns + "max_icp_cov_run_time")

        #resultion for map downsampling
        self.point_resolution = rospy.get_param(ns + "point_resolution")

        #sequential scan matching parameters (SSM)
        self.ssm_params.enable = rospy.get_param(ns + "ssm/enable")
        self.ssm_params.min_points = rospy.get_param(ns + "ssm/min_points")
        self.ssm_params.max_translation = rospy.get_param(ns + "ssm/max_translation")
        self.ssm_params.max_rotation = rospy.get_param(ns + "ssm/max_rotation")
        self.ssm_params.target_frames = rospy.get_param(ns + "ssm/target_frames")
        print("SSM: ", self.ssm_params.enable)

        #non sequential scan matching parameters (NSSM) aka loop closures
        self.nssm_params.enable = rospy.get_param(ns + "nssm/enable")
        self.nssm_params.min_st_sep = rospy.get_param(ns + "nssm/min_st_sep")
        self.nssm_params.min_points = rospy.get_param(ns + "nssm/min_points")
        self.nssm_params.max_translation = rospy.get_param(ns + "nssm/max_translation")
        self.nssm_params.max_rotation = rospy.get_param(ns + "nssm/max_rotation")
        self.nssm_params.source_frames = rospy.get_param(ns + "nssm/source_frames")
        self.nssm_params.cov_samples = rospy.get_param(ns + "nssm/cov_samples")
        print("NSSM: ", self.nssm_params.enable)

        #pairwise consistency maximization parameters for loop closure 
        #outliar rejection
        self.pcm_queue_size = rospy.get_param(ns + "pcm_queue_size")
        self.min_pcm = rospy.get_param(ns + "min_pcm")

        # are we doing 3D submapping
        self.mapping_3d = rospy.get_param(ns + "mapping_3d")

        #mak delay between an incoming point cloud and dead reckoning
        self.feature_odom_sync_max_delay = 0.5

        #define the subsrcibing topics
        self.feature_sub = Subscriber(SONAR_FEATURE_TOPIC, PointCloud2)
        self.odom_sub = Subscriber(LOCALIZATION_ODOM_TOPIC, Odometry)
        self.odom_sub_repub = rospy.Subscriber(LOCALIZATION_ODOM_TOPIC_II, Odometry, self.odom_callback)
        if self.mapping_3d:
            self.sonar_fusion_sub = rospy.Subscriber("SonarCloud", PointCloud2,self.sonar_fusion_callback,queue_size=1000)

        #define the sync policy
        self.time_sync = ApproximateTimeSynchronizer(
            [self.feature_sub, self.odom_sub], 20, 
            self.feature_odom_sync_max_delay, allow_headerless = False)

        #register the callback in the sync policy
        self.time_sync.registerCallback(self.SLAM_callback)

        #pose publisher
        self.pose_pub = rospy.Publisher(
            SLAM_POSE_TOPIC, PoseWithCovarianceStamped, queue_size=10)

        #dead reckoning topic
        self.odom_pub = rospy.Publisher(SLAM_ODOM_TOPIC, Odometry, queue_size=10)

        #SLAM trajectory topic
        self.traj_pub = rospy.Publisher(
            SLAM_TRAJ_TOPIC, PointCloud2, queue_size=1, latch=True)

        #constraints between poses
        self.constraint_pub = rospy.Publisher(
            SLAM_CONSTRAINT_TOPIC, Marker, queue_size=1, latch=True)

        #point cloud publisher topic
        self.cloud_pub = rospy.Publisher(
            SLAM_CLOUD_TOPIC, PointCloud2, queue_size=1, latch=True)

        #point cloud publisher topic
        self.submap_pub = rospy.Publisher(
                    "submaps", PointCloud2, queue_size=1, latch=True)

        # pose history publisher
        self.pose_history_pub = rospy.Publisher(
                    "pose_history", PoseHistory, queue_size=1, latch=True)

        #tf broadcaster to show pose
        self.tf = tf.TransformBroadcaster()

        #cv bridge object
        self.CVbridge = cv_bridge.CvBridge()

        #get the ICP configuration from the yaml fukle
        icp_config = rospy.get_param(ns + "icp_config")
        self.icp.loadFromYaml(icp_config)

        # tf listener
        self.listener = tf.TransformListener()
        
        # define the robot ID this is not used here, extended in multi-robot SLAM
        self.rov_id = ""

        #call the configure function
        self.configure()
        loginfo("SLAM node is initialized")

    @add_lock
    def sonar_callback(self, ping:OculusPing)->None:
        """Subscribe once to configure Oculus property.
        Assume sonar configuration doesn't change much.

        Args:
            ping (OculusPing): The sonar message. 
        """
        
        self.oculus.configure(ping)
        self.sonar_sub.unregister()

    @add_lock
    def odom_callback(self, odom_msg: Odometry) -> None:
        """Handle an incoming odometry message. Here we append the odom
        pose to the current SLAM keyframe to get live pose publishing. 
        We do this here and not in the SLAM callback to avoid any long
        pauses in pose publishing. 

        Args:
            odom_msg (Odometry): The incoming pose from the odometry
        """

        if self.keyframes:
            time = odom_msg.header.stamp
            dr_pose3 = r2g(odom_msg.pose.pose)
            dr_pose3 = dr_pose3.compose(gtsam.Pose3(gtsam.Pose2(1.15,0,0))) # transform from base_link to sonar_link
            frame = Keyframe(False, time, dr_pose3)
            dr_odom = self.current_keyframe.dr_pose.between(frame.dr_pose)
            pose = self.current_keyframe.pose.compose(dr_odom)
            #set the frames twist
            frame.twist = odom_msg.twist.twist
            frame.update(pose)
            self.current_frame = frame
            self.publish_pose()

    @add_lock
    def sonar_fusion_callback(self, cloud_msg: PointCloud2) -> None:
        """Handle the incoming point cloud from the orthoganal sonar fusion system.
        We also lookup the odom transform to we can register this cloud later.
        Args:
            cloud_msg (PointCloud2): the point cloud from sonar fusion node
        """

        # check that we have init the keyframes
        # make sure that this cloud happened after or at the same time as the keyframe
        # we are about to log it to. 
        if self.keyframes: #and cloud_msg.header.stamp.secs >= self.keyframes[-1].time.secs:

            # decode the cloud message and fix the axis
            cloud_np = r2n(cloud_msg)
            cloud_np = np.c_[cloud_np[:,0] , -1 *  cloud_np[:,2], cloud_np[:,1]]

            try:

                # pull the transform from the timestamp of the incoming cloud message
                (trans,rot) = self.listener.lookupTransform('map', 'dead_reckoning', cloud_msg.header.stamp)

                #parse the transform
                roll,pitch,yaw = Rotation.from_quat(rot).as_euler("xyz")
                x,y,z = trans

                #package as a GTSAM pose
                pose = n2g([x,y,z,roll-1.5708,pitch,yaw],"Pose3")
                pose = pose.compose(gtsam.Pose3(gtsam.Pose2(1.15,0,0)))

                #log the cloud and transform into the system
                self.keyframes[-1].odom_tranforms.append(pose)
                self.keyframes[-1].sonar_fusion_clouds.append(cloud_np)

            # on exception release the lock and return the callback
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                return 

    @add_lock
    def SLAM_callback(self, feature_msg:PointCloud2, odom_msg:Odometry)->None:
        """SLAM call back. Subscibes to the feature msg point cloud and odom msg
            Handles the whole SLAM system and publishes map, poses and constraints

        Args:
            feature_msg (PointCloud2): the incoming sonar point cloud
            odom_msg (Odometry): the incoming DVL/IMU state estimate
        """

        #aquire the lock 
        self.lock.acquire()

        #get rostime from the point cloud
        time = feature_msg.header.stamp

        #get the dead reckoning pose from the odom msg, GTSAM pose object
        dr_pose3 = r2g(odom_msg.pose.pose)
        dr_pose3 = dr_pose3.compose(gtsam.Pose3(gtsam.Pose2(1.15,0,0))) # transform from base_link to sonar_link

        #init a new key frame
        frame = Keyframe(False, time, dr_pose3)

        #convert the point cloud message to a numpy array of 2D
        points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(feature_msg)
        points = np.c_[points[:,0] , -1 *  points[:,2]]

        # In case feature extraction is skipped in this frame
        if len(points) and np.isnan(points[0, 0]):
            frame.status = False
        else:
            frame.status = self.is_keyframe(frame)

        #set the frames twist
        frame.twist = odom_msg.twist.twist

        #update the keyframe with pose information from dead reckoning
        if self.keyframes:
            dr_odom = self.current_keyframe.dr_pose.between(frame.dr_pose)
            pose = self.current_keyframe.pose.compose(dr_odom)
            frame.update(pose)


        #check frame staus, are we actually adding a keyframe? This is determined based on distance 
        #traveled according to dead reckoning
        if frame.status:

            #add the point cloud to the frame
            frame.points = points

            #perform seqential scan matching
            #if this is the first frame do not
            if not self.keyframes:
                self.add_prior(frame)
            else:
                self.add_sequential_scan_matching(frame)

            #update the factor graph with the new frame
            self.update_factor_graph(frame)

            #if loop closures are enabled
            #nonsequential scan matching is True (a loop closure occured) update graph again
            if self.nssm_params.enable and self.add_nonsequential_scan_matching():
                self.update_factor_graph()

            # build a submap and publish info
            self.build_submap()
            self.publish_all()
            
        self.lock.release()

    def log(self) -> None:
        """Log the relevant data from the mission
        """

        # poses
        poses = []

        # submaps
        submaps = []

        # pull the submap and pose
        for index in range(len(self.keyframes)):
            submaps.append(self.keyframes[index].submap_3D)
            poses.append(pose223(self.keyframes[index].pose))

        

    def publish_all(self)->None:
        """Publish to all ouput topics
            trajectory, contraints, point cloud and the full GTSAM instance
        """
        if not self.keyframes:
            return

        # self.publish_pose()
        # if self.current_frame.status:
        self.publish_trajectory()
        self.publish_constraint()
        self.publish_point_cloud()
        self.publish_submaps()

    def publish_submaps(self)->None:
        """Pull the submaps from each keyframe and publish them all as one pointcloud message.
        """

        # container for the output, all the submaps in our system
        all_submaps= None
        
        # loop over each keyframe
        for index in range(len(self.keyframes)):

            # pull the submap and labels
            submap = self.keyframes[index].submap_3D
            
            # guard against a non map
            if submap is not None:

                # pull the keyframe pose, note this pose is only x,y,theta
                pose = pose223(self.keyframes[index].pose)

                # register the submap in the global frame
                H = pose.matrix().astype(np.float32)
                submap = submap.dot(H[:3, :3].T) + H[:3, 3]

                # concat this submap
                if all_submaps is None:
                    all_submaps = submap
                else:
                    all_submaps = np.row_stack((all_submaps,submap))

        # package and publish
        if all_submaps is not None:
            msg = n2r(all_submaps, "PointCloudXYZ")
            msg.header.frame_id = "map"
            self.submap_pub.publish(msg)

    def publish_pose(self)->None:
        """Append dead reckoning from Localization to SLAM estimate to achieve realtime TF.
        """

        #define a pose with covariance message 
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.current_frame.time
        if self.rov_id == "":
            pose_msg.header.frame_id = "map"
        else:
            pose_msg.header.frame_id = self.rov_id + "_map"

        # apply the transform from sonar_link to base_link
        current_pose = self.current_frame.pose3
        current_pose = current_pose.compose(gtsam.Pose3(gtsam.Pose2(-1.15,0,0)))
        pose_msg.pose.pose = g2r(current_pose)

        cov = 1e-4 * np.identity(6, np.float32)
        # FIXME Use cov in current_frame
        cov[np.ix_((0, 1, 5), (0, 1, 5))] = self.current_keyframe.transf_cov
        pose_msg.pose.covariance = cov.ravel().tolist()
        self.pose_pub.publish(pose_msg)

        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.pose.pose = pose_msg.pose.pose
        if self.rov_id == "":
            odom_msg.child_frame_id = "base_link"
        else:
            odom_msg.child_frame_id = self.rov_id + "_base_link"
        odom_msg.twist.twist = self.current_frame.twist
        self.odom_pub.publish(odom_msg)

        p = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
        self.tf.sendTransform(
            (p.x, p.y, p.z), (q.x, q.y, q.z, q.w), odom_msg.header.stamp, odom_msg.child_frame_id, pose_msg.header.frame_id
        )

    def publish_constraint(self)->None:
        """Publish constraints between poses in the factor graph,
        either sequential or non-sequential.
        """

        #define a list of all the constraints
        links = []

        #iterate over all the keframes
        for x, kf in enumerate(self.keyframes[1:], 1):

            #append each SSM factor in green
            p1 = self.keyframes[x - 1].pose3.x(), self.keyframes[x - 1].pose3.y(), self.keyframes[x - 1].dr_pose3.z()
            p2 = self.keyframes[x].pose3.x(), self.keyframes[x].pose3.y(), self.keyframes[x].dr_pose3.z()
            links.append((p1, p2, "green"))

            #loop over all loop closures in this keyframe and append them in red
            for k, _ in self.keyframes[x].constraints:
                p0 = self.keyframes[k].pose3.x(), self.keyframes[k].pose3.y(), self.keyframes[k].dr_pose3.z()
                links.append((p0, p2, "red"))

        #if nothing, do nothing
        if links:

            #conver this list to a series of multi-colored lines and publish
            link_msg = ros_constraints(links)
            link_msg.header.stamp = self.current_keyframe.time
            if self.rov_id != "":
                link_msg.header.frame_id = self.rov_id + "_map"
            self.constraint_pub.publish(link_msg)


    def publish_trajectory(self)->None:
        """Publish 3D trajectory as point cloud in [x, y, z, roll, pitch, yaw, index] format.
        """

        #get all the poses from each keyframe
        poses = np.array([g2n(kf.pose3) for kf in self.keyframes])

        # publish the whole timehistory for the baysian mapping system
        pose_history_msg = PoseHistory()
        pose_history_msg.header = Header()
        pose_history_msg.header.frame_id = ""
        pose_history_msg.header.stamp = self.keyframes[-1].time
        pose_history_msg.data = list(np.ravel(poses))
        self.pose_history_pub.publish(pose_history_msg)

        #convert to a ros color line
        traj_msg = ros_colorline_trajectory(poses)
        traj_msg.header.stamp = self.current_keyframe.time
        if self.rov_id == "":
            traj_msg.header.frame_id = "map"
        else:
            traj_msg.header.frame_id = self.rov_id + "_map"
        self.traj_pub.publish(traj_msg)

    def publish_point_cloud(self)->None:
        """Publish downsampled 3D point cloud with z = 0.
        The last column represents keyframe index at which the point is observed.
        """

        #define an empty array
        all_points = [np.zeros((0, 2), np.float32)]

        #list of keyframe ids
        all_keys = []

        #loop over all the keyframes, register 
        #the point cloud to the orign based on the SLAM estinmate
        for key in range(len(self.keyframes)):

            #parse the pose
            pose = self.keyframes[key].pose

            #get the resgistered point cloud
            transf_points = self.keyframes[key].transf_points

            #append
            all_points.append(transf_points)
            all_keys.append(key * np.ones((len(transf_points), 1)))

        all_points = np.concatenate(all_points)
        all_keys = np.concatenate(all_keys)

        #use PCL to downsample this point cloud
        sampled_points, sampled_keys = pcl.downsample(
            all_points, all_keys, self.point_resolution
        )

        #parse the downsampled cloud into the ros xyzi format
        sampled_xyzi = np.c_[sampled_points, np.zeros_like(sampled_keys), sampled_keys]
        
        #if there are no points return and do nothing
        if len(sampled_xyzi) == 0:
            return

        #convert the point cloud to a ros message and publish
        cloud_msg = n2r(sampled_xyzi, "PointCloudXYZI")
        cloud_msg.header.stamp = self.current_keyframe.time
        if self.rov_id == "":
            cloud_msg.header.frame_id = "map"
        else:
            cloud_msg.header.frame_id = self.rov_id + "_map"
        self.cloud_pub.publish(cloud_msg)
