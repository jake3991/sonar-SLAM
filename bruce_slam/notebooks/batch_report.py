import sys
import pickle
import gtsam
import glob
import trimesh
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

from utils import load_scene, load_origin, load_data_into_dict, run_numbers, get_ground_truth_map

# parse the arguments, do we want to visulize and do we want to run a quant study
scene, dist_metrics, coverage_metrics, cloud_type = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]

# load the scene
mesh = load_scene(scene)

# get the datafiles via glob
pose_paths = glob.glob("data_logs/"+scene+"/poses_*")
pose3D_paths = glob.glob("data_logs/"+scene+"/poses3D*")
submap_paths = glob.glob("data_logs/"+scene+"/submaps_*")
inference_cloud_paths = glob.glob("data_logs/"+scene+"/inferenceclouds*")
fusion_cloud_paths = glob.glob("data_logs/"+scene+"/fusionclouds*")
submap_time_paths = glob.glob("data_logs/"+scene+"/submap_times*")
inferences_time_paths = glob.glob("data_logs/"+scene+"/bayesmap_time*")

# load up the data files
pose_dict = load_data_into_dict(pose_paths)
pose3D_dict = load_data_into_dict(pose3D_paths)
submap_dict = load_data_into_dict(submap_paths)
inference_cloud_dict = load_data_into_dict(inference_cloud_paths)
fusion_cloud_dict = load_data_into_dict(fusion_cloud_paths)
submap_time_dict = load_data_into_dict(submap_time_paths)
inferences_time_dict = load_data_into_dict(inferences_time_paths)

# ground truth location, the starting point of the robot in gazebo
origin = load_origin(scene)

# get the ground truth comparsion object
world = get_ground_truth_map(mesh, 10000000)

# table of MAE and RMSE
data_table = np.ones((3,10)) * -1

# plotting
if coverage_metrics == 1:
    plt.figure(figsize=(20,10))
    legend_list = []
    plot_symbols = ["*","o","^","x","D"]

i = 0
k = 1
for translation in range(1,6):
    j = 0
    for rotation in range(30,120,30):
        if (translation,rotation) in pose_dict:
            print(translation,rotation)
            
            # get the point cloud we want
            if cloud_type == "submap":
                points = submap_dict[(translation,rotation)]
                poses = pose_dict[(translation,rotation)]
            elif cloud_type == "fusion":
                points = fusion_cloud_dict[(translation,rotation)]
                poses = pose3D_dict[(translation,rotation)]
            elif cloud_type == "infer":
                points = inference_cloud_dict[(translation,rotation)]
                poses = pose3D_dict[(translation,rotation)]
            else:
                raise NotImplemented
            
            mae, rmse, coverage_rate = run_numbers(poses,points,origin,world,dist_metrics==1,coverage_metrics==1)
            print(mae,rmse)
            data_table[j][i] = mae
            data_table[j][k] = rmse
            if coverage_metrics == 1:
                plt.plot(np.linspace(0,len(coverage_rate),len(coverage_rate)), coverage_rate,marker=plot_symbols[translation-1])
                plt.scatter(np.linspace(0,len(coverage_rate),len(coverage_rate)), coverage_rate,marker=plot_symbols[translation-1])
                legend_list.append(str(translation) + "," + str(rotation))
            j += 1
    i += 2
    k += 2

if dist_metrics == 1:
    data_table = np.round(data_table,3)
    df = pd.DataFrame(data_table)
    df = df.set_axis(["MAE 1", "RMSE 1", "MAE 2", "RMSE 2", "MAE 3", "RMSE 3", "MAE 4", "RMSE 4", "MAE 5", "RMSE 5"], axis=1)
    df = df.set_axis([30,60,90], axis=0)
    df.to_csv("csv/"+scene+"_"+cloud_type+"_distance_metrics.csv")

if coverage_metrics == 1:
    plt.ylim(0,150000)
    #plt.yticks(np.linspace(0,18000,10))
    plt.grid()
    plt.legend(legend_list)
    plt.savefig("csv/"+scene+"_"+cloud_type+".png")
    plt.show()

