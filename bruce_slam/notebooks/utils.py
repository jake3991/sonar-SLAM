from ast import Tuple
import gtsam
import numpy as np
import open3d as o3d
import pickle
import trimesh
from bruce_slam import pcl

def load_plane_scene() -> o3d.geometry.TriangleMesh:
    """Get the plane scene as an open3d triangle mesh

    Returns:
        o3d.geometry.TriangleMesh: return the gazebo scene in python
    """

    mesh = o3d.io.read_triangle_mesh("/home/jake/Desktop/uuv_sim_docker/uuv_simulator" +
                                    "/uuv_gazebo_worlds/models/plane_crash/meshes/B24.stl")
    mesh = mesh.scale(0.1,center=(0,0,0))
    R = mesh.get_rotation_matrix_from_xyz((np.radians(180),0,0))
    mesh.rotate(R, center=(0, 0, 0))

    return mesh

def load_penns_landing_scene() -> o3d.geometry.TriangleMesh:
    """Get the penns landing scene as an open3d triangle mesh

    Returns:
        o3d.geometry.TriangleMesh: return the gazebo scene in python
    """

    mesh = o3d.io.read_triangle_mesh("/home/jake/Desktop/uuv_sim_docker/uuv_simulator/uuv_gazebo_worlds" +
                                        "/models/penns_landing/meshes/main.stl")
    

    mesh_2 = o3d.io.read_triangle_mesh("/home/jake/Desktop/uuv_sim_docker/uuv_simulator/" +
                                        "uuv_gazebo_worlds/models/penns_landing/meshes/galleon.stl")
    mesh_2 = mesh_2.scale(0.25,center=(0,0,0))
    R = mesh_2.get_rotation_matrix_from_xyz((1.5708, 0, 0))
    mesh_2 = mesh_2.rotate(R, center=(0, 0, 0))
    mesh_2 = mesh_2.translate((-30, 45, -10))

    mesh_3 = o3d.io.read_triangle_mesh("/home/jake/Desktop/uuv_sim_docker/uuv_simulator/" +
                                        "uuv_gazebo_worlds/models/penns_landing/meshes/dread.stl")
    mesh_3 = mesh_3.scale(0.5,center=(0,0,0))
    R = mesh_3.get_rotation_matrix_from_xyz((3.14159,0, 1.5708))
    mesh_3 = mesh_3.rotate(R, center=(0, 0, 0))
    mesh_3 = mesh_3.translate((150, 35, 40))

    mesh = mesh + mesh_2 + mesh_3

    R = mesh.get_rotation_matrix_from_xyz((np.radians(180),0,0))
    mesh.rotate(R, center=(0, 0, 0))

    return mesh

def load_rfal_land_scene() -> o3d.geometry.TriangleMesh:
    """Get the RFAL land scene as an open3d triangle mesh

    Returns:
        o3d.geometry.TriangleMesh: return the gazebo scene in python
    """

    mesh = o3d.io.read_triangle_mesh("/home/jake/Desktop/uuv_sim_docker/uuv_simulator/uuv_gazebo_worlds/" +
                                       "models/RFAL_land/meshes/training_world_set_origin.stl" )


    # boat 1
    mesh_2 = o3d.io.read_triangle_mesh("/home/jake/Desktop/uuv_sim_docker/uuv_simulator/uuv_gazebo_worlds/" +
                                       "models/RFAL_land/meshes/boat.stl" )
    mesh_2 = mesh_2.translate((-12.5,-9,5))
    mesh += mesh_2

    # boat 2
    mesh_2 = o3d.io.read_triangle_mesh("/home/jake/Desktop/uuv_sim_docker/uuv_simulator/uuv_gazebo_worlds/" +
                                       "models/RFAL_land/meshes/boat.stl" )
    R = mesh_2.get_rotation_matrix_from_xyz((0,0,np.radians(90)))
    mesh_2 = mesh_2.rotate(R, center=(0, 0, 0))
    mesh_2 = mesh_2.translate((16,5,5))
    mesh += mesh_2 

    # boat 3
    mesh_2 = o3d.io.read_triangle_mesh("/home/jake/Desktop/uuv_sim_docker/uuv_simulator/uuv_gazebo_worlds/" +
                                       "models/RFAL_land/meshes/boat.stl" )
    R = mesh_2.get_rotation_matrix_from_xyz((0,0,-1.13))
    mesh_2 = mesh_2.rotate(R, center=(0, 0, 0))
    mesh_2 = mesh_2.translate((-15,15,5)) 
    mesh += mesh_2 

    # boat 4
    mesh_2 = o3d.io.read_triangle_mesh("/home/jake/Desktop/uuv_sim_docker/uuv_simulator/uuv_gazebo_worlds/" +
                                       "models/RFAL_land/meshes/boat.stl" )
    R = mesh_2.get_rotation_matrix_from_xyz((0,0,-1.13))
    mesh_2 = mesh_2.rotate(R, center=(0, 0, 0))
    mesh_2 = mesh_2.translate((-28,10,5)) 
    mesh += mesh_2 
    
    R = mesh.get_rotation_matrix_from_xyz((np.radians(180),0,0))
    mesh = mesh.rotate(R, center=(0, 0, 0))
    mesh = mesh.translate((0,0,5))

    return mesh

def load_suny_scene() -> o3d.geometry.TriangleMesh:
    """Get the suny scene as an open3d triangle mesh

    Returns:
        o3d.geometry.TriangleMesh: return the gazebo scene in python
    """

    mesh = o3d.io.read_triangle_mesh("/home/jake/Desktop/uuv_sim_docker/uuv_simulator/uuv_gazebo_worlds/models/suny_maritime/meshes/suny_maritime.stl" )
    R = mesh.get_rotation_matrix_from_xyz((0,0,-1.07))
    mesh = mesh.rotate(R, center=(0, 0, 0))
    mesh = mesh.translate((0,20,0)) 

    R = mesh.get_rotation_matrix_from_xyz((np.radians(180),0,0))
    mesh = mesh.rotate(R, center=(0, 0, 0))
    return mesh

def load_scene(scene: str) -> o3d.geometry.TriangleMesh:
    """Load the required scene by reading the STL files and 
    handling them as open3d triangle mesh objects

    Args:
        scene (str): the gazebo scene we are interested in

    Returns:
        o3d.geometry.TriangleMesh: return the gazebo scene in python
    """

    if scene == "plane":
        return load_plane_scene()
    elif scene == "penns_landing":
        return load_penns_landing_scene()
    elif scene == "rfal_land":
        return load_rfal_land_scene()
    elif scene == "suny":
        return load_suny_scene()
    else:
        raise NotImplementedError

def load_origin(scene: str) -> gtsam.Pose3:
    """Get the starting location of the robot based on the gazebo scene

    Args:
        scene (str): the gazebo scene we are interested in

    Returns:
        gtsam.Pose3: the starting location of the robot in the gazebo frame
    """

    if scene == "plane":
        return gtsam.Pose3(gtsam.Rot3(),[-10, 0, -7])
    elif scene == "penns_landing":
        return gtsam.Pose3(gtsam.Rot3(),[250, 0, 0])
    elif scene == "rfal_land":
        return gtsam.Pose3(gtsam.Pose2(-20,15,1.5708))
    elif scene == "suny":
        return gtsam.Pose3(gtsam.Pose2(0,0,0))
    else:
        raise NotImplementedError

def aggragate_points(clouds:list, poses: list, frame: gtsam.Pose3, coverage_rate: bool, filter_surface: bool=False) -> list:
    """Turn a list of point clouds and poses into a combined
    point cloud in the same frame

    Args:
        clouds (list): the list of np.array point clouds
        poses (list): list of gtsam.Pose3 for each cloud
        frame (gtsam.Pose3): the frame we want this cloud in
        coverage_rate (bool): if we are calculating coverage rate
        filter_surface (bool): if we need to filter the points above the surface

    Returns:
        list[np.array,list,list]: the combined points, the coverage at that step 
                                    and the index of the step
    """

    # tracker for coverage vs. time
    i = 0
    coverage_per_step = []
    step = []

    # build the submaps into one big point cloud
    combined_map = None
    for pose, cloud in zip(poses,clouds):
        if cloud is not None:
            pose = frame.compose(pose) # correct to be in gazebo frame
            H = pose.matrix().astype(np.float32)
            cloud = cloud.dot(H[:3, :3].T) + H[:3, 3]
            if combined_map is None:
                combined_map = cloud
            else:
                combined_map = np.row_stack((combined_map,cloud))

        if combined_map is not None and coverage_rate:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(combined_map)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud,voxel_size=0.1)
            coverage_per_step.append(len(voxel_grid.get_voxels()))
            step.append(i)
            i += 1
        
    if filter_surface:
        combined_map = combined_map[combined_map[:,2] >= 0]

    #use PCL to downsample this point cloud
    combined_map = pcl.downsample(combined_map, 0.2)

    return combined_map, coverage_per_step, step

def load_data_into_dict(paths: list) -> dict:
    """Given a list of file paths, load up the data and 
    save it into a dictionary with the keys (keyframe_translation,keyframe_rotation)

    Args:
        paths (list): the path list for the kind of data (poses,submaps, etc.)

    Returns:
        dict: a dictionary of the datasets
    """

    output_dict = {} # the output dict
    for path_i in paths: # loop over and parse
        temp = path_i.split("/")[2].split("_")
        trans = int(temp[1])
        rot = int(temp[2].split(".")[0])
        file = open(path_i,'rb')
        output_dict[(trans,rot)] = pickle.load(file) # load and push
        file.close()

    return output_dict

def get_ground_truth_map(mesh: o3d.geometry.TriangleMesh, sample_count: int) -> o3d.geometry.PointCloud:
    """Here we convert a mesh into a dense point cloud representation to enable quick
    comparison using open3D.

    Args:
        mesh (o3d.geometry.TriangleMesh): the triangle mesh of the env

    Returns:
        o3d.geometry.PointCloud: a dense point cloud of the env
    """

    sample_points = mesh.sample_points_uniformly(sample_count)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(sample_points,voxel_size=0.05)
    point_cloud_np = np.asarray([voxel_grid.origin + pt.grid_index*voxel_grid.voxel_size for pt in voxel_grid.get_voxels()])
    point_cloud_env = o3d.geometry.PointCloud()
    point_cloud_env.points = o3d.utility.Vector3dVector(point_cloud_np)

    return point_cloud_env


def run_numbers(poses: list, points: list, origin: gtsam.Pose3, world: o3d.geometry.PointCloud, accuracy: bool, coverage: bool) -> list:
    """Get the quant report for a set of points and poses given the 
    origin and mesh of the world. 

    Args:
        poses (list): the poses of each point cloud
        points (list): each point cloud
        origin (gtsam.Pose3): the origin of the solution in gazebo frame
        world (o3d.geometry.PointCloud): the cloud of the world
        accuracy (bool): do we want to proccess the distance to mesh metrics
        coverage (bool): do we want to proccess the coverage metrics

    Returns:
        list: mae, rmse, a list of the coverage per step
    """

    # aggragate some points and get the distance between each point and the mesh
    map, coverage_per_step, _ = aggragate_points(points,poses,origin,coverage_rate=coverage)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(map)

    mae, rmse = -1, -1
    distance = None
    if accuracy:
        # get the distance between the map and world
        distance = point_cloud.compute_point_cloud_distance(world)
        distance = np.asarray(distance)

        # reduce the distance to the numbers we care about
        mae = np.mean(abs(distance))
        rmse = np.sqrt(np.mean(distance**2))

    return mae, rmse, coverage_per_step, distance

def run_time_numbers(times: list) -> list:
    """Get the mean and standard deviation of the provided times

    Args:
        times (list): a list of run times

    Returns:
        list: [mean,STD]
    """

    return np.mean(times), np.std(times)