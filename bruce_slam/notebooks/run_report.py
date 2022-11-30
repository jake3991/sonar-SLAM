import sys
import pickle
import gtsam
import trimesh
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from bruce_slam import pcl

from utils import load_scene, aggragate_points, load_origin, get_ground_truth_map, run_numbers

# parse the arguments, do we want to visulize and do we want to run a quant study
scene, vis, quant = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

# load the scene
mesh = load_scene(scene)

# open the SLAM data
file = open("data_logs/"+scene+"/poses_5_60.pickle",'rb')
poses = pickle.load(file)
file.close()
file = open("data_logs/"+scene+"/submaps_5_60.pickle",'rb')
submaps = pickle.load(file)
file.close()

# ground truth location, the starting point of the robot in gazebo
origin = load_origin(scene)

# build the submaps into one big point cloud
combined_map, coverage_per_step, step = aggragate_points(submaps,poses,origin,coverage_rate=False,filter_surface=True)

# build some open3d vis objects
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(combined_map)
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
coord_frame = coord_frame.scale(5, center=coord_frame.get_center())

if vis == 1:
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([coord_frame,mesh,point_cloud])

if quant == 1:
    world = get_ground_truth_map(mesh, 10000000)
    mae, rmse, coverage_rate = run_numbers(poses,submaps,origin,world,True,False)
    print("MAE: ", mae)
    print("RMSE: ", rmse)
