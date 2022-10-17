from rti_dvl.msg import DVL
from sensor_msgs.msg import Imu
from bar30_depth.msg import Depth
from sonar_oculus.msg import OculusPing


"""
Topics for the bruce_slam project
"""
IMU_TOPIC = "/rexrov/imu"
DVL_TOPIC = "/rexrov/dvl_twist"
DEPTH_TOPIC = "/rexrov/pressure"
SONAR_TOPIC = "/rexrov/forward_sonar_horiz/raw_sonar"

SLAM_NS = "/bruce/slam/"
GYRO_INTEGRATION_TOPIC = SLAM_NS + "gyro_integrated"
SONAR_FUSION_TOPIC = SLAM_NS + "sonar_fusion"
LOCALIZATION_ODOM_TOPIC = SLAM_NS + "localization/odom"
LOCALIZATION_ODOM_TOPIC_II = SLAM_NS + "localization/odom_live"
LOCALIZATION_TRAJ_TOPIC = SLAM_NS + "localization/traj"
SLAM_POSE_TOPIC = SLAM_NS + "slam/pose"
SLAM_ODOM_TOPIC = SLAM_NS + "slam/odom"
SLAM_TRAJ_TOPIC = SLAM_NS + "slam/traj"
SLAM_CLOUD_TOPIC = SLAM_NS + "slam/cloud"
SLAM_CONSTRAINT_TOPIC = SLAM_NS + "slam/constraint"
SLAM_ISAM2_TOPIC = SLAM_NS + "slam/isam2"
SLAM_PREDICT_SLAM_UPDATE_SERVICE = SLAM_NS + "slam/predict_slam_update"
MAPPING_INTENSITY_TOPIC = SLAM_NS + "mapping/intensity"
MAPPING_OCCUPANCY_TOPIC = SLAM_NS + "mapping/occupancy"
MAPPING_GET_MAP_SERVICE = SLAM_NS + "mapping/get_map"
SONAR_FEATURE_TOPIC = SLAM_NS + "feature_extraction/feature"
SONAR_FEATURE_IMG_TOPIC = SLAM_NS + "feature_extraction/feature_img"
