import sys
import pickle
import gtsam
import trimesh
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from utils import load_scene, aggragate_points, load_origin

# parse the arguments, do we want to visulize and do we want to run a quant study
scene, vis, quant = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

# load the scene
mesh = load_scene(scene)

# open the SLAM data
file = open("data_logs/poses.pickle",'rb')
poses = pickle.load(file)
file.close()
file = open("data_logs/submaps.pickle",'rb')
submaps = pickle.load(file)
file.close()

# ground truth location, the starting point of the robot in gazebo
origin = load_origin(scene)

# build the submaps into one big point cloud
combined_map, coverage_per_step, step = aggragate_points(submaps,poses,origin,coverage_rate=False)

# build some open3d vis objects
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(combined_map)
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
coord_frame = coord_frame.scale(5, center=coord_frame.get_center())

#@point_cloud = mesh.sample_points_uniformly(100000000)

if vis == 1:
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([coord_frame,mesh,point_cloud])

if quant == 1:
    tri_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
    _, distance, _ = trimesh.proximity.closest_point(tri_mesh,combined_map)

    # voxelize the point cloud to grade coverage
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud,voxel_size=0.1)

    # generate distance metrics
    print("mean absolute error: ", np.mean(abs(distance)))
    print("root mean square error : ", np.sqrt(np.mean(distance**2)))

    # generate coverage metrics
    print("voxel count: ", len(voxel_grid.get_voxels()))

    # generate coverage rate plot
    plt.figure(figsize=(20,10))
    plt.plot(step,coverage_per_step)
    plt.title(scene,fontsize=20)
    plt.show()
