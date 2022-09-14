import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.patches import Ellipse
from shapely import geometry

import cv2
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def apply_custom_colormap(image_gray, cmap=plt.get_cmap("seismic")):
    """
    Implementation of applyColorMap in OpenCV using colormaps in Matplotlib.
    """

    assert image_gray.dtype == np.uint8, "must be np.uint8 image"
    if image_gray.ndim == 3:
        image_gray = image_gray.squeeze(-1)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256))[:, 0:3]  # color range RGBA => RGB
    color_range = (color_range * 255.0).astype(np.uint8)  # [0,1] => [0,255]
    color_range = np.squeeze(
        np.dstack([color_range[:, 2], color_range[:, 1], color_range[:, 0]]), 0
    )  # RGB => BGR

    # Apply colormap for each channel individually
    channels = [cv2.LUT(image_gray, color_range[:, i]) for i in range(3)]
    return np.dstack(channels)


def colorline(
    x, y, z=None, cmap="jet", norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0
):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(
        segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha
    )

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_cov_ellipse(pos, cov, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def ros_colorline(xyz):
    fields = [
        pc2.PointField("x", 0, pc2.PointField.FLOAT32, 1),
        pc2.PointField("y", 4, pc2.PointField.FLOAT32, 1),
        pc2.PointField("z", 8, pc2.PointField.FLOAT32, 1),
        pc2.PointField("i", 12, pc2.PointField.FLOAT32, 1),
    ]

    xyzi = np.c_[xyz, np.array([[i] for i in range(len(xyz))])]

    header = rospy.Header()
    return pc2.create_cloud(header, fields, xyzi)


def ros_colorline_trajectory(traj):
    fields = [
        pc2.PointField("x", 0, pc2.PointField.FLOAT32, 1),
        pc2.PointField("y", 4, pc2.PointField.FLOAT32, 1),
        pc2.PointField("z", 8, pc2.PointField.FLOAT32, 1),
        pc2.PointField("roll", 12, pc2.PointField.FLOAT32, 1),
        pc2.PointField("pitch", 16, pc2.PointField.FLOAT32, 1),
        pc2.PointField("yaw", 20, pc2.PointField.FLOAT32, 1),
        pc2.PointField("i", 24, pc2.PointField.FLOAT32, 1),
    ]

    traji = np.c_[traj, np.mgrid[0 : len(traj)]]

    header = rospy.Header()
    return pc2.create_cloud(header, fields, traji)


colors = {
    "red": ColorRGBA(1.0, 0.0, 0.0, 1.0),
    "blue": ColorRGBA(0.0, 0.0, 1.0, 1.0),
    "green": ColorRGBA(0.0, 1.0, 0.0, 1.0),
    "white": ColorRGBA(1.0,1.0,1.0,1.0),
    "yellow": ColorRGBA(1.0,1.0,0.0,1.0),
    "light_blue":ColorRGBA(.44,.62,.8118,1.0)
}


def ros_constraints(links):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.type = Marker.LINE_LIST
    marker.ns = "constraints"
    marker.scale.x = 0.2
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    for point1, point2, color in links:
        point1 = Point(point1[0], point1[1], point1[2])
        point2 = Point(point2[0], point2[1], point2[2])
        marker.points.append(point1)
        marker.points.append(point2)
        marker.colors.append(colors[color])
        marker.colors.append(colors[color])

    return marker


def plot_polygon(shape, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if isinstance(shape, geometry.MultiPolygon):
        for polygon in shape.geoms:
            if not isinstance(polygon, geometry.GeometryCollection):
                x, y = polygon.exterior.xy
                ax.plot(x, y, **kwargs)
    elif isinstance(shape, geometry.Polygon):
        x, y = shape.exterior.xy
        ax.plot(x, y, **kwargs)
