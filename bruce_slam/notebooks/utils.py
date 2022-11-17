from ast import Tuple
import gtsam
import numpy as np
import open3d as o3d


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
    """Get the penns landing scene as an open3d triangle mesh

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
    else:
        raise NotImplementedError

def aggragate_points(clouds:list, poses: list, frame: gtsam.Pose3, coverage_rate: bool) -> list:
    """Turn a list of point clouds and poses into a combined
    point cloud in the same frame

    Args:
        clouds (list): the list of np.array point clouds
        poses (list): list of gtsam.Pose3 for each cloud
        frame (gtsam.Pose3): the frame we want this cloud in
        coverage_rate (bool): if we are calculating coverage rate

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

    return combined_map, coverage_per_step, step