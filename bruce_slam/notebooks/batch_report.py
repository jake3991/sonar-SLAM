import sys
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_scene, load_origin, load_data_into_dict, run_numbers, get_ground_truth_map, run_time_numbers

# parse the arguments, do we want to visulize and do we want to run a quant study
scene, dist_metrics, coverage_metrics = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

# load the scene
mesh = load_scene(scene)

# get the datafiles via glob
pose_paths = glob.glob("data_logs/"+scene+"/poses_*")
pose3D_paths = glob.glob("data_logs/"+scene+"/poses3D*")
submap_paths = glob.glob("data_logs/"+scene+"/submaps_*")
inference_cloud_paths = glob.glob("data_logs/"+scene+"/inferenceclouds*")
fusion_cloud_paths = glob.glob("data_logs/"+scene+"/fusionclouds*")
submap_time_paths = glob.glob("data_logs/"+scene+"/submaptimes*")
inferences_time_paths = glob.glob("data_logs/"+scene+"/bayesmaptime*")

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

coverage_by_keyframe = {}

# iterate over the three kinds of clouds
for cloud_type in ["fusion","infer","submap"]:

    # table of MAE and RMSE
    data_table = np.ones((3,10)) * -1
    data_time_table = np.ones((3,10)) * -1
    distance_by_keyframe = {}
    
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
                run_times = None
                if cloud_type == "submap":
                    points = submap_dict[(translation,rotation)]
                    poses = pose_dict[(translation,rotation)]
                    run_times = submap_time_dict[(translation,rotation)]
                elif cloud_type == "fusion":
                    points = fusion_cloud_dict[(translation,rotation)]
                    poses = pose3D_dict[(translation,rotation)]
                elif cloud_type == "infer":
                    points = inference_cloud_dict[(translation,rotation)]
                    poses = pose3D_dict[(translation,rotation)]
                    run_times = inferences_time_dict[(translation,rotation)]
                else:
                    raise NotImplemented
                
                # get the requested metrics and log
                mae, rmse, coverage_rate, distance = run_numbers(poses,points,origin,world,dist_metrics==1,coverage_metrics==1)
                distance_by_keyframe[(translation,rotation)] = distance
                data_table[j][i] = mae
                data_table[j][k] = rmse

                # log coverage
                coverage_by_keyframe[(cloud_type,translation,rotation)] = coverage_rate

                # log time
                if run_times is not None:
                    mean_time, std_time = run_time_numbers(run_times)
                    data_time_table[j][i] = mean_time
                    data_time_table[j][k] = std_time

                # plot the coverage data
                if coverage_metrics == 1:
                    plt.plot(np.linspace(0,len(coverage_rate),len(coverage_rate)), coverage_rate,marker=plot_symbols[translation-1])
                    plt.scatter(np.linspace(0,len(coverage_rate),len(coverage_rate)), coverage_rate,marker=plot_symbols[translation-1])
                    legend_list.append(str(translation) + "," + str(rotation))
                j += 1
        i += 2
        k += 2

    # write a CSV file of summary time metrics
    data_time_table = np.round(data_time_table,3)
    df = pd.DataFrame(data_time_table)
    df = df.set_axis(["Mean 1", "STD 1", "Mean 2", "STD 2", "Mean 3", "STD 3", "Mean 4", "STD 4", "Mean 5", "STD 5"], axis=1)
    df = df.set_axis([30,60,90], axis=0)
    df.to_csv("reports/"+scene+"/"+cloud_type+"_time_metrics.csv")

    if dist_metrics == 1:
        # write a CSV file of summary accuracy metrics
        data_table = np.round(data_table,3)
        df = pd.DataFrame(data_table)
        df = df.set_axis(["MAE 1", "RMSE 1", "MAE 2", "RMSE 2", "MAE 3", "RMSE 3", "MAE 4", "RMSE 4", "MAE 5", "RMSE 5"], axis=1)
        df = df.set_axis([30,60,90], axis=0)
        df.to_csv("reports/"+scene+"/"+cloud_type+"_distance_metrics.csv")

        # save the raw data so we can box plot it
        with open("reports/"+scene+"/"+cloud_type+"_distance_metrics.pickle", 'wb') as handle:
                pickle.dump(distance_by_keyframe, handle)

    if coverage_metrics == 1:
        # plot the coverage data
        
        with open("reports/"+scene+'/coverage.pickle', 'wb') as handle:
            pickle.dump(coverage_by_keyframe, handle)

        if scene == "suny":
            plt.ylim(0,150000)
        if scene == "plane":
            plt.ylim(0,50000)
        if scene == "rfal_land":
            plt.ylim(0,100000)
        if scene == "penns_landing":
            plt.ylim(0,700000)

        plt.grid()
        plt.legend(legend_list)
        plt.savefig("reports/"+scene+"/"+cloud_type+".png")
        #plt.show()
        plt.clf()
        plt.close()

