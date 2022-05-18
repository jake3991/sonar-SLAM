#!/usr/bin/env python
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from bruce_slam import pcl

plt.style.use("classic")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

"""
This script is to load logs (npz) from ros LOG_DIR
that are produced by slam_node and then plot estimated
trajectory and point cloud from multiple trials.

"""

cmap = plt.get_cmap("jet")

bag_runs = defaultdict(list)
for filename in os.listdir("."):
    if "run" in filename and filename.endswith(".npz"):
        print(filename)
        data = np.load(filename)

        poses = data["poses"]["pose"]
        points = data["points"]

        bagname = filename.split("@")[0]
        bag_runs[bagname].append((poses, points))

for bagname, runs in bag_runs.items():
    plt.figure()
    total_points = []

    for i, (poses, points) in enumerate(runs):
        total_points.append(np.c_[points[:, :2], np.ones((len(points), 1)) * i])
        plt.plot(poses[:, 0], poses[:, 1], color=cmap(i / float(len(runs))))

    total_points = np.concatenate(total_points)
    points, indices = pcl.downsample(total_points[:, :2], total_points[:, (2)], 0.5)
    plt.scatter(
        points[:, 0], points[:, 1], c=indices.ravel(), s=1, cmap="jet", alpha=0.5
    )

    plt.xlabel("y (m)")
    plt.ylabel("x (m)")
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("{}.pdf".format(bagname), dpi=200, bbox_inches="tight")
