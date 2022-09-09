import gtsam
import numpy as np

from ctypes import Union
from typing import Any
from numpy import True_
from scipy.optimize import shgo
from itertools import combinations
from collections import defaultdict
from sklearn.covariance import MinCovDet
import time as time_pkg

from .sonar import OculusProperty
from .utils.conversions import *
from .utils.visualization import *
from .utils.io import *
from . import pcl

from bruce_slam.slam_objects import (
    STATUS,
    Keyframe,
    InitializationResult,
    ICPResult,
    SMParams,
)


class SLAM(object):
    """The class to run underwater sonar based SLAM"""

    def __init__(self):
        """Class constructor for the SLAM class, note we do not feed arguments in in the pythonic way
        we use the ros param system to get the params. Note that almost everything is eligible for
        overwrite when the yaml file is called. See config/slam.yaml."""

        # configure sonar info
        self.oculus = OculusProperty()

        # Create a new factor when
        # - |ti - tj| > min_duration and
        # - |xi - xj| > max_translation or
        # - |ri - rj| > max_rotation
        self.keyframe_duration = None
        self.keyframe_translation = None
        self.keyframe_rotation = None

        # List of keyframes, a keyframe is a step in the SLAM solution
        self.keyframes = []

        # Current (non-key)frame with real-time pose update
        # TODO propagate cov from previous keyframe
        self.current_frame = None

        # init isam the graph optimization tool
        self.isam_params = gtsam.ISAM2Params()
        self.isam = gtsam.ISAM2(self.isam_params)

        # define the graph and initial guess matrix, values. Use these to push info into isam
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()

        # initial location noise model [x, y, theta]
        self.prior_sigmas = None  # place holder

        # Noise model without ICP, just dead reckoning
        # [x, y, theta]
        self.odom_sigmas = None  # place holder

        # Downsample paramter for point cloud for ICP and publishing
        self.point_resolution = 0.5

        # Noise radius in overlap estimation
        self.point_noise = 0.5

        # paramters for sequnetial scan matching (SSM)
        self.ssm_params = SMParams()  # object to hold all the params
        self.ssm_params.initialization = True  # flag to indicate if we did this step
        self.ssm_params.initialization_params = 50, 1, 0.01
        self.ssm_params.min_st_sep = 1
        self.ssm_params.min_points = 50
        self.ssm_params.max_translation = 2.0
        self.ssm_params.max_rotation = np.pi / 6
        self.ssm_params.target_frames = 3
        # Don't use ICP covariance
        self.ssm_params.cov_samples = 0

        # paramters for loop closures (NSSM)
        self.nssm_params = SMParams()
        self.nssm_params.initialization = True
        self.nssm_params.initialization_params = 100, 5, 0.01
        self.nssm_params.min_st_sep = 10
        self.nssm_params.min_points = 100
        self.nssm_params.max_translation = 6.0
        self.nssm_params.max_rotation = np.pi / 2
        self.nssm_params.source_frames = 5
        self.nssm_params.cov_samples = 30

        # define ICP
        self.icp = pcl.ICP()
        self.icp_ssm = pcl.ICP()

        # Pairwise consistent measurement, for loop closure outlier rejection
        self.nssm_queue = []  # the loop closur queue
        self.pcm_queue_size = 5  # default val
        self.min_pcm = 3  # default val

        # Use fixed noise model in two cases
        # - Sequential scan matching
        # - ICP cov is too small in non-sequential scan matching
        # [x, y, theta]
        self.icp_odom_sigmas = None

        # Can't save fig in online mode
        # TODO remove this
        self.save_fig = False
        self.save_data = False

    @property
    def current_keyframe(self) -> Keyframe:
        """Get the current keyframe from the SLAM solution

        Returns:
            Keyframe: the current keyframe (most recent keyframe) in the system
        """

        # the current keyframe
        return self.keyframes[-1]

    @property
    def current_key(self) -> int:
        """Get the length of the list that stores the keyframes

        Returns:
            int: the length of self.keyframes
        """

        return len(self.keyframes)

    def configure(self) -> None:
        """Configure SLAM"""

        # check nssm covariance params
        assert (
            self.nssm_params.cov_samples == 0
            or self.nssm_params.cov_samples
            < self.nssm_params.initialization_params[0]
            * self.nssm_params.initialization_params[1]
        )

        # check ssm covariance params
        assert (
            self.ssm_params.cov_samples == 0
            or self.ssm_params.cov_samples
            < self.ssm_params.initialization_params[0]
            * self.ssm_params.initialization_params[1]
        )

        assert self.nssm_params.source_frames < self.nssm_params.min_st_sep

        # create noise models
        self.prior_model = self.create_noise_model(self.prior_sigmas)
        self.odom_model = self.create_noise_model(self.odom_sigmas)
        self.icp_odom_model = self.create_noise_model(self.icp_odom_sigmas)

    def get_states(self) -> np.array:
        """Retrieve all states as array which are represented as
            [time, pose2, dr_pose3, cov]
            - pose2: [x, y, yaw]
            - dr_pose3: [x, y, z, roll, pitch, yaw]
            - cov: 3 x 3

        Returns:
            np.array: the state array
        """

        # build the state array
        states = np.zeros(
            self.current_key,
            dtype=[
                ("time", np.float64),
                ("pose", np.float32, 3),
                ("dr_pose3", np.float32, 6),
                ("cov", np.float32, 9),
            ],
        )

        # Update all
        values = self.isam.calculateEstimate()
        for key in range(self.current_key):
            pose = values.atPose2(X(key))
            cov = self.isam.marginalCovariance(X(key))
            self.keyframes[key].update(pose, cov)

        # pull the state
        t_zero = self.keyframes[0].time
        for key in range(self.current_key):
            keyframe = self.keyframes[key]
            states[key]["time"] = (keyframe.time - t_zero).to_sec()
            states[key]["pose"] = g2n(keyframe.pose)
            states[key]["dr_pose3"] = g2n(keyframe.dr_pose3)
            states[key]["cov"] = keyframe.transf_cov.ravel()
        return states

    @staticmethod
    def sample_pose(pose: gtsam.Pose2, covariance: np.array) -> gtsam.Pose2:
        """Generate a random pose using the covariance matrix to define a normal dist.

        Args:
            pose (gtsam.Pose2): The pose we wish to add random to
            covariance (np.array): The covariance associated with this pose

        Returns:
            gtsam.Pose2: the provided pose with some random noise added
        """

        # get the random noise and add it to the provided pose
        delta = np.random.multivariate_normal(np.zeros(3), covariance)
        return pose.compose(n2g(delta, "Pose2"))

    def sample_current_pose(self) -> gtsam.Pose2:
        """Add random noise to self.current_keyframe.pose using self.sample_pose()

        Returns:
            gtsam.Pose2: The self.current_keyframe.pose with noise added
        """

        return self.sample_pose(self.current_keyframe.pose, self.current_keyframe.cov)

    def get_points(
        self, frames: list = None, ref_frame: Any = None, return_keys: bool = False
    ) -> np.array:
        """Get a point cloud, doing the following steps
            - Accumulate points in frames
            - Transform them to reference frame
            - Downsample points
            - Return the corresponding keys for every point

        Args:
            frames (list, optional): The list of indexes for the frames we care about. Defaults to None.
            ref_frame (Any, optional): The frame we want the points relative to, can be gtsam.Pose2 or int index. Defaults to None.
            return_keys (bool, optional): Do we want to return the keys?. Defaults to False.

        Returns:
            np.array: the point cloud array, maybe with keys for each point
        """

        # if there are no frames speced just get them all
        if frames is None:
            frames = range(self.current_key)

        # check if the ref frame is a gtsam.Pose2, if it is not we assume it's an index in the list of self.keyframes
        if ref_frame is not None:
            if isinstance(ref_frame, gtsam.Pose2):
                ref_pose = ref_frame
            else:
                ref_pose = self.keyframes[ref_frame].pose

        # Define a blank array to add our points to
        if return_keys:
            all_points = [np.zeros((0, 3), np.float32)]
        else:
            all_points = [np.zeros((0, 2), np.float32)]

        # Loop over the provided keyframe indexes
        for key in frames:
            # if we have a reference frame then use that, otherwise use the SLAM frame
            if ref_frame is not None:
                # transform to the reference frame provided
                points = self.keyframes[key].points
                pose = self.keyframes[key].pose
                transf = ref_pose.between(pose)
                transf_points = Keyframe.transform_points(points, transf)
            else:
                transf_points = self.keyframes[key].transf_points

            # if we want the key with each point, get those here
            if return_keys:
                transf_points = np.c_[
                    transf_points, key * np.ones((len(transf_points), 1))
                ]
            all_points.append(transf_points)

        # combine the points into a numpy array
        all_points = np.concatenate(all_points)

        # apply voxel downsampling and return
        if return_keys:
            return pcl.downsample(
                all_points[:, :2], all_points[:, (2,)], self.point_resolution
            )
        else:
            return pcl.downsample(all_points, self.point_resolution)

    def compute_icp(
        self,
        source_points: np.array,
        target_points: np.array,
        guess: np.array = gtsam.Pose2(),
    ) -> Union:
        """Compute standard ICP

        Args:
            source_points (np.array): source point cloud [x,y]
            target_points (np.array): target point cloud [x,y]
            guess (np.array, optional): the inital guess, if not provided we use identity. Defaults to gtsam.Pose2().

        Returns:
            Union[str,gtsam.Pose2]: returns the status message and the result as a gtsam.Pose2
        """

        # setup the points
        source_points = np.array(source_points, np.float32)
        target_points = np.array(target_points, np.float32)

        # convert the guess to a matrix and apply ICP
        guess = guess.matrix()
        message, T = self.icp.compute(source_points, target_points, guess)

        # parse the ICP output
        x, y = T[:2, 2]
        theta = np.arctan2(T[1, 0], T[0, 0])

        return message, gtsam.Pose2(x, y, theta)

    def compute_icp_with_cov(
        self, source_points: np.array, target_points: np.array, guesses: list
    ) -> Union:
        """Compute ICP with a covariance matrix

        Args:
            source_points (np.array): source point cloud [x,y]
            target_points (np.array): target point cloud [x,y]
            guesses (list): list of initial guesses

        Returns:
            Union[str,gtsam.Pose2,np.array,np.array]: status message,transform,covariance matrix,transforms tested
        """

        # parse the points
        source_points = np.array(source_points, np.float32)
        target_points = np.array(target_points, np.float32)

        # check each of the provided guesses with ICP
        sample_transforms = []
        start = time_pkg.time()
        for g in guesses:
            g = g.matrix()
            message, T = self.icp.compute(source_points, target_points, g)

            # only keep what works
            if message == "success":
                x, y = T[:2, 2]
                theta = np.arctan2(T[1, 0], T[0, 0])
                sample_transforms.append((x, y, theta))

            # enforce a max run time for this loop
            if time_pkg.time() - start >= 2.0:
                break

        # check if we have enough transforms to get a covariance
        sample_transforms = np.array(sample_transforms)
        if len(sample_transforms) < 5:
            return "Too few samples for covariance computation", None, None, None

        # Can't use np.cov(). Too many outliers
        try:
            fcov = MinCovDet(store_precision=False, support_fraction=0.8).fit(
                sample_transforms
            )
        except ValueError as e:
            return "Failed to calculate covariance", None, None, None

        # parse the result
        m = n2g(fcov.location_, "Pose2")
        cov = fcov.covariance_

        # unrotate to local frame
        R = m.rotation().matrix()
        cov[:2, :] = R.T.dot(cov[:2, :])
        cov[:, :2] = cov[:, :2].dot(R)

        # check if the default covariance for ICP is bigger than the one we just estimated
        default_cov = np.diag(self.icp_odom_sigmas) ** 2
        if np.linalg.det(cov) < np.linalg.det(default_cov):
            cov = default_cov

        return "success", m, cov, sample_transforms

    def get_overlap(
        self,
        source_points: np.array,
        target_points: np.array,
        source_pose: gtsam.Pose2 = None,
        target_pose: gtsam.Pose2 = None,
        return_indices: bool = False,
    ) -> int:
        """Get the overlap between the provided clouds, the count of points with a nearest neighbor

        Args:
            source_points (np.array): source point cloud
            target_points (np.array): target point cloud
            source_pose (gtsam.Pose2, optional): pose for the source points. Defaults to None.
            target_pose (gtsam.Pose2, optional): pose for the target points. Defaults to None.
            return_indices (bool, optional): if we want the cloud indexes. Defaults to False.

        Returns:
            int: the number of points with a nearest neighbor
        """

        # transform the points if we have a pose
        if source_pose:
            source_points = Keyframe.transform_points(source_points, source_pose)
        if target_pose:
            target_points = Keyframe.transform_points(target_points, target_pose)

        # match the points using nearest neigbor with PCL
        # note that un-matched points get a -1 in indices
        indices, dists = pcl.match(target_points, source_points, 1, self.point_noise)

        # if we want the indices, send those
        if return_indices:
            return np.sum(indices != -1), indices
        else:
            return np.sum(indices != -1)

    def add_prior(self, keyframe: Keyframe) -> None:
        """Add the prior factor for the first pose in the SLAM solution. This is the starting frame.

        Args:
            keyframe (Keyframe): the keyframe object for the initial frame
        """

        pose = keyframe.pose
        factor = gtsam.PriorFactorPose2(X(0), pose, self.prior_model)
        self.graph.add(factor)
        self.values.insert(X(0), pose)

    def add_odometry(self, keyframe: Keyframe) -> None:
        """Add the odometry factor between provided keyframe and the last keyframe

        Args:
            keyframe (Keyframe): the incoming keyframe, basically keyframe_t
        """

        # get the time a pose differnce between the provided keyframe and the last logged one
        dt = (keyframe.time - self.keyframes[-1].time).to_sec()
        dr_odom = self.keyframes[-1].pose.between(keyframe.pose)

        # build a factor and insert it into the graph, providing an initial guess as well
        factor = gtsam.BetweenFactorPose2(
            X(self.current_key - 1), X(self.current_key), dr_odom, self.odom_model
        )
        self.graph.add(factor)
        self.values.insert(X(self.current_key), keyframe.pose)

    def get_map(self, frames, resolution=None):
        # Implemented in slam_node
        # TODO remove this code
        raise NotImplementedError

    def get_matching_cost_subroutine1(
        self,
        source_points: np.array,
        source_pose: gtsam.Pose2,
        target_points: np.array,
        target_pose: gtsam.Pose2,
        source_pose_cov: np.array = None,
    ) -> Union:
        """Perform global cost point cloud alignment. Here we transform source points to target points.

        Args:
            source_points (np.array): source point cloud
            source_pose (gtsam.Pose2): pose for the source_points
            target_points (np.array): target point cloud
            target_pose (gtsam.Pose2): pose for the target_points
            source_pose_cov (np.array, optional): Covariance for the source points. Defaults to None.

        Returns:
            Union[function,list]: the function to be optimized by scipy.shgo and a list of poses
        """
        # pose_samples = []
        # target_tree = KDTree(target_points)

        # def subroutine(x):
        #     # x = [x, y, theta]
        #     delta = n2g(x, "Pose2")
        #     sample_source_pose = source_pose.compose(delta)
        #     sample_transform = target_pose.between(sample_source_pose)

        #     points = Keyframe.transform_points(source_points, sample_transform)
        #     dists, indices = target_tree.query(
        #         points, distance_upper_bound=self.point_noise
        #     )

        #     cost = -np.sum(indices != len(target_tree.data))

        #     pose_samples.append(np.r_[g2n(sample_source_pose), cost])
        #     return cost

        # return subroutine, pose_samples

        # maintain a list of poses we try
        pose_samples = []

        # create a grid for the target points
        xmin, ymin = np.min(target_points, axis=0) - 2 * self.point_noise
        xmax, ymax = np.max(target_points, axis=0) + 2 * self.point_noise
        resolution = self.point_noise / 10.0
        xs = np.arange(xmin, xmax, resolution)
        ys = np.arange(ymin, ymax, resolution)
        target_grids = np.zeros((len(ys), len(xs)), np.uint8)

        # populate the grid for the target points
        r = np.int32(np.round((target_points[:, 1] - ymin) / resolution))
        c = np.int32(np.round((target_points[:, 0] - xmin) / resolution))
        r = np.clip(r, 0, target_grids.shape[0] - 1)
        c = np.clip(c, 0, target_grids.shape[1] - 1)
        target_grids[r, c] = 255

        # dilate the grid
        dilate_hs = int(np.ceil(self.point_noise / resolution))
        dilate_size = 2 * dilate_hs + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_size, dilate_size), (dilate_hs, dilate_hs)
        )
        target_grids = cv2.dilate(target_grids, kernel)

        # # Calculate distance to the nearest points
        # target_grids = cv2.bitwise_not(target_grids)
        # target_grids = cv2.distanceTransform(target_grids, cv2.DIST_L2, 3)
        # target_grids = 1.0 - 0.2 * target_grids / self.point_noise
        # target_grids = np.clip(target_grids, 0.2, 1.0)

        source_pose_info = np.linalg.inv(source_pose_cov)

        def subroutine(x: np.array) -> float:
            """The optimization subroutine, called iterativly by scipy.shgo

            Args:
                x (gtsam.Pose2): the source pose as an array. [x, y, theta]

            Returns:
                float: cost of this step
            """

            # package the incoming pose as a gtsam.Pose2
            # apply this pose to the source_pose and get the transform between source and target
            delta = n2g(x, "Pose2")
            sample_source_pose = source_pose.compose(delta)
            sample_transform = target_pose.between(sample_source_pose)

            # apply this new transform to the source points
            # then limit the points to only points that fit inside the target grid
            points = Keyframe.transform_points(source_points, sample_transform)
            r = np.int32(np.round((points[:, 1] - ymin) / resolution))
            c = np.int32(np.round((points[:, 0] - xmin) / resolution))
            inside = (
                (0 <= r)
                & (r < target_grids.shape[0])
                & (0 <= c)
                & (c < target_grids.shape[1])
            )

            # get the number of cells that overlap and log the pose
            cost = -np.sum(target_grids[r[inside], c[inside]] > 0)
            pose_samples.append(np.r_[g2n(sample_source_pose), cost])

            return cost

        return subroutine, pose_samples

    def get_matching_cost_subroutine2(self, source_points, source_pose, occ):
        # TODO remove this code
        """
        Ceres scan matching

        Cost = - sum_i  ||1 - M_nearest(Tx s_i)||^2,
                given transform Tx, source points S, occupancy map M
        """
        pose_samples = []
        x0, y0, resolution, occ_arr = occ

        def subroutine(x):
            # x = [x, y, theta]
            delta = n2g(x, "Pose2")
            sample_pose = source_pose.compose(delta)

            xy = Keyframe.transform_points(source_points, sample_pose)
            r = np.int32(np.round((xy[:, 1] - y0) / resolution))
            c = np.int32(np.round((xy[:, 0] - x0) / resolution))

            sel = (r >= 0) & (c >= 0) & (r < occ_arr.shape[0]) & (c < occ_arr.shape[1])
            hit_probs_inside_map = occ_arr[r[sel], c[sel]]
            num_hits_outside_map = len(xy) - np.sum(sel)

            cost = (
                np.sum((1.0 - hit_probs_inside_map) ** 2)
                + num_hits_outside_map * (1.0 - 0.5) ** 2
            )
            cost = np.sqrt(cost / len(source_points))

            pose_samples.append(np.r_[g2n(sample_pose), cost])
            return cost

        return subroutine, pose_samples

    def initialize_sequential_scan_matching(
        self, keyframe: Keyframe
    ) -> InitializationResult:
        """Init a sequential scan matching call by using global ICP.

        Args:
            keyframe (Keyframe): the keyframe we want to register

        Returns:
            InitializationResult: the results of the the initilization
        """

        # instanciate an ICP InitializationResult object
        ret = InitializationResult()
        ret.status = STATUS.SUCCESS
        ret.status.description = None

        # Match current keyframe to previous k frames
        ret.source_key = self.current_key
        ret.target_key = self.current_key - 1
        ret.source_pose = keyframe.pose
        ret.target_pose = self.current_keyframe.pose

        # Accumulate reference points from previous k (self.ssm_params.target_frames) frames
        ret.source_points = keyframe.points
        target_frames = range(self.current_key)[-self.ssm_params.target_frames :]
        ret.target_points = self.get_points(target_frames, ret.target_key)
        ret.cov = np.diag(self.odom_sigmas)

        """if True:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "source points {}".format(len(ret.source_points))
            return ret"""

        """if len(self.keyframes) % 2 == 0:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "source points {}".format(len(ret.source_points))
            return ret"""

        # Only continue with this if it is enabled in slam.yaml
        if self.ssm_params.enable == False:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "source points {}".format(len(ret.source_points))
            return ret

        # check the source points for a minimum count
        if len(ret.source_points) < self.ssm_params.min_points:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "source points {}".format(len(ret.source_points))
            return ret

        # check the target points for a minimum count
        if len(ret.target_points) < self.ssm_params.min_points:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "target points {}".format(len(ret.target_points))
            return ret

        # check if we have initialized the ICP params
        if not self.ssm_params.initialization:
            return ret

        with CodeTimer("SLAM - sequential scan matching - sampling"):

            # define the search space for ICP global init
            pose_stds = np.array([self.odom_sigmas]).T
            pose_bounds = 5.0 * np.c_[-pose_stds, pose_stds]

            # TODO remove
            # ret.occ = self.get_map(target_frames)
            # subroutine, pose_samples = self.get_matching_cost_subroutine2(
            #     ret.source_points,
            #     ret.source_pose,
            #     ret.occ,
            # )

            # build the global ICP subroutine
            subroutine, pose_samples = self.get_matching_cost_subroutine1(
                ret.source_points,
                ret.source_pose,
                ret.target_points,
                ret.target_pose,
                ret.cov,
            )

            # optimize the subroutine using scipy.shgo
            result = shgo(
                func=subroutine,
                bounds=pose_bounds,
                n=self.ssm_params.initialization_params[0],
                iters=self.ssm_params.initialization_params[1],
                sampling_method="sobol",
                minimizer_kwargs={
                    "options": {"ftol": self.ssm_params.initialization_params[2]}
                },
            )

        # if the optimizer indicate success package results for return
        if result.success:
            ret.source_pose_samples = np.array(pose_samples)
            ret.estimated_source_pose = ret.source_pose.compose(n2g(result.x, "Pose2"))
            ret.status.description = "matching cost {:.2f}".format(result.fun)

            # TODO remove
            if self.save_data:
                ret.save("step-{}-ssm-sampling.npz".format(self.current_key))
        else:
            ret.status = STATUS.INITIALIZATION_FAILURE
            ret.status.description = result.message

        return ret

    def add_sequential_scan_matching(self, keyframe: Keyframe) -> None:
        """Add the sequential scan matching factor to the graph. Here we use the global ICP as an inital
        guess for standard ICP. We then perform some simple checks to catch silly outliers. If those
        checks pass we add the ICP result to the pose graph.

        Args:
            keyframe (Keyframe): The keyframe we are evaluating, this contains all the relevant info.
        """

        # call the global-ICP
        ret = self.initialize_sequential_scan_matching(keyframe)

        # TODO remove this
        if self.save_fig:
            ret.plot("step-{}-ssm-sampling.png".format(self.current_key))

        # check the status of the global-ICP call, if the result is a failure.
        # simply add the odometry factor and return
        if not ret.status:
            self.add_odometry(keyframe)
            return

        # copy the global-ICP into an ICPResult
        ret2 = ICPResult(ret, self.ssm_params.cov_samples > 0)

        # Compute ICP here with a timer
        with CodeTimer("SLAM - sequential scan matching - ICP"):

            # if possible compute ICP with covariance estimation
            if self.ssm_params.initialization and self.ssm_params.cov_samples > 0:
                message, odom, cov, sample_transforms = self.compute_icp_with_cov(
                    ret2.source_points,
                    ret2.target_points,
                    ret2.initial_transforms[: self.ssm_params.cov_samples],
                )

                # if ICP fails, push that into the ret2 object
                if message != "success":
                    ret2.status = STATUS.NOT_CONVERGED
                    ret2.status.description = message
                # Else push the ICP info into ret2
                else:
                    ret2.estimated_transform = odom
                    ret2.cov = cov
                    ret2.sample_transforms = sample_transforms
                    ret2.status.description = "{} samples".format(
                        len(ret2.sample_transforms)
                    )

            # Else call standard ICP
            else:
                message, odom = self.compute_icp(
                    ret2.source_points, ret2.target_points, ret2.initial_transform
                )

                # check for failure
                if message != "success":
                    ret2.status = STATUS.NOT_CONVERGED
                    ret2.status.description = message
                else:
                    ret2.estimated_transform = odom
                    ret2.status.description = ""

        # The transformation compared to dead reckoning can't be too large
        if ret2.status:
            delta = ret2.initial_transform.between(ret2.estimated_transform)
            delta_translation = np.linalg.norm(delta.translation())
            delta_rotation = abs(delta.theta())
            if (
                delta_translation > self.ssm_params.max_translation
                or delta_rotation > self.ssm_params.max_rotation
            ):
                ret2.status = STATUS.LARGE_TRANSFORMATION
                ret2.status.description = "trans {:.2f} rot {:.2f}".format(
                    delta_translation, delta_rotation
                )

        # There must be enough overlap between two point clouds.
        if ret2.status:
            overlap = self.get_overlap(
                ret2.source_points, ret2.target_points, ret2.estimated_transform
            )
            if overlap < self.ssm_params.min_points:
                ret2.status = STATUS.NOT_ENOUGH_OVERLAP
            ret2.status.description = "overlap {}".format(overlap)

        if ret2.status:

            # if we used ICP with covariance then we don't need a boilerplate noise model
            if ret2.cov is not None:
                icp_odom_model = self.create_full_noise_model(ret2.cov)
            else:
                icp_odom_model = self.icp_odom_model

            # package a factor to be added to the graph
            factor = gtsam.BetweenFactorPose2(
                X(ret2.target_key),
                X(ret2.source_key),
                ret2.estimated_transform,
                icp_odom_model,
            )

            # Add the factor and the initial guess for this new pose
            self.graph.add(factor)
            self.values.insert(
                X(ret2.source_key), ret2.target_pose.compose(ret2.estimated_transform)
            )
            ret2.inserted = True  # log as added

            # TODO remove
            if self.save_data:
                ret2.save("step-{}-ssm-icp.npz".format(self.current_key))

        # If ICP was a failure, then just push in the dead reckoning info
        else:
            self.add_odometry(keyframe)

        # TODO remove
        if self.save_fig:
            ret2.plot("step-{}-ssm-icp.png".format(self.current_key))

    def initialize_nonsequential_scan_matching(self) -> InitializationResult:
        """Initialize a nonsequential scan matching call. Here we use global ICP to check for loop closures with the
        most recent keyframe and the rest of the map.

        Returns:
            InitializationResult: The global-ICP outcome
        """

        # instanciate an object to capute the results
        ret = InitializationResult()
        ret.status = STATUS.SUCCESS
        ret.status.description = None

        # get the indices we care about for loop closure search
        ret.source_key = self.current_key - 1
        ret.source_pose = self.current_frame.pose
        ret.estimated_source_pose = ret.source_pose
        # aggratgate the source cloud, here we want k frames (self.nssm_params.source_frames)
        source_frames = range(
            ret.source_key, ret.source_key - self.nssm_params.source_frames, -1
        )
        ret.source_points = self.get_points(source_frames, ret.source_key)

        # gate loop closure search to those who have sufficent points
        if len(ret.source_points) < self.nssm_params.min_points:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "source points {}".format(len(ret.source_points))
            return ret

        # Find target points for matching
        # Limit searching keyframes. Here we want ALL the keyframes minus k (self.nssm_params.min_st_sep)
        target_frames = range(self.current_key - self.nssm_params.min_st_sep)

        # Target points in global frame
        target_points, target_keys = self.get_points(target_frames, None, True)

        # Loop over the source frames
        # Eliminate frames that do not have points in the same field of view
        sel = np.zeros(len(target_points), np.bool)
        for source_frame in source_frames:

            # pull the pose and covariance info
            pose = self.keyframes[source_frame].pose
            cov = self.keyframes[source_frame].cov

            # parse the covariance
            translation_std = np.sqrt(np.max(np.linalg.eigvals(cov[:2, :2])))
            rotation_std = np.sqrt(cov[2, 2])
            range_bound = translation_std * 5.0 + self.oculus.max_range
            bearing_bound = rotation_std * 5.0 + self.oculus.horizontal_aperture * 0.5

            # figure out the uncertain points
            local_points = Keyframe.transform_points(target_points, pose.inverse())
            ranges = np.linalg.norm(local_points, axis=1)
            bearings = np.arctan2(local_points[:, 1], local_points[:, 0])
            sel_i = (ranges < range_bound) & (abs(bearings) < bearing_bound)
            sel |= sel_i

        # only keep the certain points
        target_points = target_points[sel]
        target_keys = target_keys[sel]

        # Check which frame has the most points nearby
        target_frames, counts = np.unique(np.int32(target_keys), return_counts=True)
        target_frames = target_frames[counts > 10]
        counts = counts[counts > 10]

        # check the aggragate cloud for num of points
        if len(target_frames) == 0 or len(target_points) < self.nssm_params.min_points:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "target points {}".format(len(target_points))
            return ret

        # populate the initilization object with some info
        ret.target_key = target_frames[
            np.argmax(counts)
        ]  # this is critical, the one with the most points overlapping
        ret.target_pose = self.keyframes[ret.target_key].pose
        ret.target_points = Keyframe.transform_points(
            target_points, ret.target_pose.inverse()
        )
        ret.cov = self.keyframes[ret.source_key].cov

        # check if we have the params for global ICP
        if not self.nssm_params.initialization:
            return ret

        with CodeTimer("SLAM - nonsequential scan matching - sampling"):

            # set bounds for global ICP
            translation_std = np.sqrt(np.max(np.linalg.eigvals(cov[:2, :2])))
            rotation_std = np.sqrt(cov[2, 2])
            pose_stds = np.array([[translation_std, translation_std, rotation_std]]).T
            pose_bounds = 5.0 * np.c_[-pose_stds, pose_stds]

            # TODO remove
            # ret.occ = self.get_map(target_frames)
            # subroutine, pose_samples = self.get_matching_cost_subroutine2(
            #     ret.source_points,
            #     ret.source_pose,
            #     ret.occ,
            # )

            # build the subroutine
            subroutine, pose_samples = self.get_matching_cost_subroutine1(
                ret.source_points,
                ret.source_pose,
                ret.target_points,
                ret.target_pose,
                ret.cov,
            )

            # optimize with scipy.shgo
            result = shgo(
                func=subroutine,
                bounds=pose_bounds,
                n=self.nssm_params.initialization_params[0],
                iters=self.nssm_params.initialization_params[1],
                sampling_method="sobol",
                minimizer_kwargs={
                    "options": {"ftol": self.nssm_params.initialization_params[2]}
                },
            )

        # check the shgo result
        if not result.success:
            ret.status = STATUS.INITIALIZATION_FAILURE
            ret.status.description = result.message
            return ret

        # parse the result
        delta = n2g(result.x, "Pose2")
        ret.estimated_source_pose = ret.source_pose.compose(delta)
        ret.source_pose_samples = np.array(pose_samples)
        ret.status.description = "matching cost {:.2f}".format(result.fun)

        # Refine target key by searching for the pose with maximum overlap
        # with current source points
        estimated_source_points = Keyframe.transform_points(
            ret.source_points, ret.estimated_source_pose
        )
        overlap, indices = self.get_overlap(
            estimated_source_points, target_points, return_indices=True
        )
        target_frames1, counts1 = np.unique(
            np.int32(target_keys[indices[indices != -1]]), return_counts=True
        )
        if len(counts1) == 0:
            ret.status = STATUS.NOT_ENOUGH_OVERLAP
            ret.status.description = "0"
            return ret

        # TODO remove
        if self.save_data:
            ret.save("step-{}-nssm-sampling.npz".format(self.current_key - 1))

        # log the target key and
        # recalculate target points with new target key in target frame
        ret.target_key = target_frames1[np.argmax(counts1)]
        ret.target_pose = self.keyframes[ret.target_key].pose
        ret.target_points = self.get_points(target_frames, ret.target_key)

        return ret

    def add_nonsequential_scan_matching(self) -> ICPResult:
        """Run a loop closure search. Here we compare the most recent keyframe to the
        previous frames. If a loop is found it is subject to geometric verification via PCM.

        Returns:
            ICPResult: the loop we have found, returns for debugging perposes
        """

        # if we do not have enough keyframes to aggratgate a submap return
        if self.current_key < self.nssm_params.min_st_sep:
            return

        # init the search with a global ICP call
        ret = self.initialize_nonsequential_scan_matching()

        # if the global ICP call did not work, return
        if not ret.status:
            return

        # package the global ICP call result
        ret2 = ICPResult(ret, self.nssm_params.cov_samples > 0)

        # Compute ICP here with a timer
        with CodeTimer("SLAM - nonsequential scan matching - ICP"):

            

            # if possible, compute ICP with a covariance matrix
            if self.nssm_params.initialization and self.nssm_params.cov_samples > 0:
                message, odom, cov, sample_transforms = self.compute_icp_with_cov(
                    ret2.source_points,
                    ret2.target_points,
                    ret2.initial_transforms[: self.nssm_params.cov_samples],
                )

                # check the status
                if message != "success":
                    ret2.status = STATUS.NOT_CONVERGED
                    ret2.status.description = message
                else:
                    ret2.estimated_transform = odom
                    ret2.cov = cov
                    ret2.sample_transforms = sample_transforms
                    ret2.status.description = "{} samples".format(
                        len(ret2.sample_transforms)
                    )

            # otherwise use standard ICP
            else:
                message, odom = self.compute_icp(
                    ret2.source_points, ret2.target_points, ret2.initial_transform
                )

                # check status
                if message != "success":
                    ret2.status = STATUS.NOT_CONVERGED
                    ret2.status.description = message
                else:
                    ret2.estimated_transform = odom
                    ret.status.description = ""

        # Add some failure detections
        # The transformation compared to initial guess can't be too large
        if ret2.status:
            delta = ret2.initial_transform.between(ret2.estimated_transform)
            delta_translation = np.linalg.norm(delta.translation())
            delta_rotation = abs(delta.theta())
            if (
                delta_translation > self.nssm_params.max_translation
                or delta_rotation > self.nssm_params.max_rotation
            ):
                ret2.status = STATUS.LARGE_TRANSFORMATION
                ret2.status.description = "trans {:.2f} rot {:.2f}".format(
                    delta_translation, delta_rotation
                )

        # There must be enough overlap between two point clouds.
        if ret2.status:
            overlap = self.get_overlap(
                ret2.source_points, ret2.target_points[:, :2], ret2.estimated_transform
            )
            if overlap < self.nssm_params.min_points:
                ret2.status = STATUS.NOT_ENOUGH_OVERLAP
            ret2.status.description = str(overlap)
            
        # apply geometric verification, in this case PCM
        if ret2.status:

            # update the pcm queue
            while (
                self.nssm_queue
                and ret2.source_key - self.nssm_queue[0].source_key
                > self.pcm_queue_size
            ):
                self.nssm_queue.pop(0)

            # log the newest loop closure into the pcm queue and check PCM
            self.nssm_queue.append(ret2)
            pcm = self.verify_pcm(self.nssm_queue,self.min_pcm)

            # if the PCM result has no loop closures for us, the list pcm will be empty
            # loop over any results and add them to the graph
            for m in pcm:

                # pull the loop closure from the pcm queue
                ret2 = self.nssm_queue[m]

                # check if the loop has been added to the graph
                if not ret2.inserted:

                    # get a noise model
                    if ret2.cov is not None:
                        icp_odom_model = self.create_full_noise_model(ret2.cov)
                    else:
                        icp_odom_model = self.icp_odom_model

                    # build the factor and add it to the graph
                    factor = gtsam.BetweenFactorPose2(
                        X(ret2.target_key),
                        X(ret2.source_key),
                        ret2.estimated_transform,
                        icp_odom_model,
                    )
                    self.graph.add(factor)
                    self.keyframes[ret2.source_key].constraints.append(
                        (ret2.target_key, ret2.estimated_transform)
                    )
                    ret2.inserted = True  # update the status of this loop closure, don't add a loop twice

        return ret2

    def is_keyframe(self, frame: Keyframe) -> bool:
        """Check if a Keyframe object meets the conditions to be a SLAM keyframe.
        If the vehicle has moved enough. Either rotation or translation.

        Args:
            frame (Keyframe): the keyframe we want to check.

        Returns:
            bool: a flag indicating if we need to add this frame to the SLAM soltuion
        """

        # if there are no keyframes in our SLAM solution, this is the first one
        if not self.keyframes:
            return True

        # check for time
        duration = frame.time - self.current_keyframe.time
        if duration < self.keyframe_duration:
            return False

        # check for rotation and translation
        dr_odom = self.keyframes[-1].dr_pose.between(frame.dr_pose)
        translation = np.linalg.norm(dr_odom.translation())
        rotation = abs(dr_odom.theta())

        return (
            translation > self.keyframe_translation or rotation > self.keyframe_rotation
        )

    def create_full_noise_model(
        self, cov: np.array
    ) -> gtsam.noiseModel.Gaussian.Covariance:
        """Create a noise model from a numpy array using the gtsam api.

        Args:
            cov (np.array): numpy array of the covariance matrix.

        Returns:
            gtsam.noiseModel.Gaussian.Covariance: gtsam version of the input
        """

        return gtsam.noiseModel.Gaussian.Covariance(cov)

    def create_robust_full_noise_model(self, cov: np.array) -> gtsam.noiseModel.Robust:
        """Create a robust gtsam noise model from a numpy array

        Args:
            cov (np.array): numpy array of the covariance matrix

        Returns:
            gtsam.noiseModel.Robust: gtsam version of input
        """

        model = gtsam.noiseModel.Gaussian.Covariance(cov)
        robust = gtsam.noiseModel.mEstimator.Cauchy.Create(1.0)
        return gtsam.noiseModel.Robust.Create(robust, model)

    def create_noise_model(self, *sigmas: list) -> gtsam.noiseModel.Diagonal:
        """Create a noise model from a list of sigmas, treated like a diagnal matrix.

        Returns:
            gtsam.noiseModel.Diagonal: gtsam version of input
        """
        return gtsam.noiseModel.Diagonal.Sigmas(np.r_[sigmas])

    def create_robust_noise_model(self, *sigmas: list) -> gtsam.noiseModel.Robust:
        """Create a robust noise model from a list of sigmas

        Returns:
            gtsam.noiseModel.Robust: gtsam verison of input
        """

        model = gtsam.noiseModel.Diagonal.Sigmas(np.r_[sigmas])
        robust = gtsam.noiseModel.mEstimator.Cauchy.Create(1.0)
        return gtsam.noiseModel.Robust.Create(robust, model)

    def update_factor_graph(self, keyframe: Keyframe = None) -> None:
        """Update the internal SLAM estimate

        Args:
            keyframe (Keyframe, optional): The keyframe that needs to be added to the SLAM solution. Defaults to None.
        """

        # if we have a keyframe add it to our list of keyframes
        if keyframe:
            self.keyframes.append(keyframe)

        # push the newest factors into the ISAM2 instance
        self.isam.update(self.graph, self.values)
        self.graph.resize(0)  # clear the graph and values once we push it to ISAM2
        self.values.clear()

        # Update the whole trajectory
        values = self.isam.calculateEstimate()
        for x in range(values.size()):
            pose = values.atPose2(X(x))
            self.keyframes[x].update(pose)

        # Only update latest cov
        cov = self.isam.marginalCovariance(X(values.size() - 1))
        self.keyframes[-1].update(pose, cov)

        # Update the poses in pending loop closures for PCM
        for ret in self.nssm_queue:
            ret.source_pose = self.keyframes[ret.source_key].pose
            ret.target_pose = self.keyframes[ret.target_key].pose
            if ret.inserted:
                ret.estimated_transform = ret.target_pose.between(ret.source_pose)

    def verify_pcm(self, queue: list, min_pcm_value: int) -> list:
        """Get the pairwise consistent measurements.

        Args:
            queue (list): the list of loop closures being checked.
            min_pcm_value (int): the min pcm value we want

        Returns:
            list: returns any pairwise consistent loops. We return a list of indexes in the provided queue.
        """

        # check if we have enough loops to bother
        if len(queue) < min_pcm_value:
            return []

        # convert the loops to a consistentcy graph=
        G = defaultdict(list)
        for (a, ret_il), (b, ret_jk) in combinations(zip(range(len(queue)), queue), 2):
            pi = ret_il.target_pose
            pj = ret_jk.target_pose
            pil = ret_il.estimated_transform
            plk = ret_il.source_pose.between(ret_jk.source_pose)
            pjk1 = ret_jk.estimated_transform
            pjk2 = pj.between(pi.compose(pil).compose(plk))

            error = gtsam.Pose2.Logmap(pjk1.between(pjk2))
            md = error.dot(np.linalg.inv(ret_jk.cov)).dot(error)
            # chi2.ppf(0.99, 3) = 11.34
            if md < 11.34:  # this is not a magic number
                G[a].append(b)
                G[b].append(a)

        # find the sets of consistent loops
        maximal_cliques = list(self.find_cliques(G))

        # if we got nothing, return nothing
        if not maximal_cliques:
            return []

        # sort and return only the largest set, also checking that the set is large enough
        maximum_clique = sorted(maximal_cliques, key=len, reverse=True)[0]
        if len(maximum_clique) < min_pcm_value:
            return []

        return maximum_clique

    def find_cliques(self, G: defaultdict):
        """Returns all maximal cliques in an undirected graph.

        Args:
            G (defaultdict): consicentcy graph
        """

        if len(G) == 0:
            return

        adj = {u: {v for v in G[u] if v != u} for u in G}
        Q = [None]

        subg = set(G)
        cand = set(G)
        u = max(subg, key=lambda u: len(cand & adj[u]))
        ext_u = cand - adj[u]
        stack = []

        try:
            while True:
                if ext_u:
                    q = ext_u.pop()
                    cand.remove(q)
                    Q[-1] = q
                    adj_q = adj[q]
                    subg_q = subg & adj_q
                    if not subg_q:
                        yield Q[:]
                    else:
                        cand_q = cand & adj_q
                        if cand_q:
                            stack.append((subg, cand, ext_u))
                            Q.append(None)
                            subg = subg_q
                            cand = cand_q
                            u = max(subg, key=lambda u: len(cand & adj[u]))
                            ext_u = cand - adj[u]
                else:
                    Q.pop()
                    subg, cand, ext_u = stack.pop()
        except IndexError:
            pass
