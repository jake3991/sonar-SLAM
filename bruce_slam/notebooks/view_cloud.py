import sys
import pickle
import gtsam
import trimesh
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from utils import load_scene, aggragate_points, load_origin

scene = "penns_landing"

# open the SLAM data
file = open("data_logs/"+scene+"/poses_4_30.pickle",'rb')
poses = pickle.load(file)
file.close()
file = open("data_logs/"+scene+"/submaps_4_30.pickle",'rb')
submaps = pickle.load(file)
file.close()

# ground truth location, the starting point of the robot in gazebo
origin = load_origin(scene)

# build the submaps into one big point cloud
combined_map, _, _ = aggragate_points(submaps,poses,origin,coverage_rate=False,filter_surface=False)

# build some open3d vis objects
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(combined_map)
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
coord_frame = coord_frame.scale(5, center=coord_frame.get_center())

o3d.visualization.draw_geometries([coord_frame,point_cloud])