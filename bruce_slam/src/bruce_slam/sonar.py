import numpy as np
from scipy.interpolate import interp1d
import cv2
import rospy

from .utils.topics import *
from .utils.conversions import r2n


class OculusFireMsg(object):
    """Oculus Fire Message

    uint8_t masterMode;           // mode 0 is flexi mode, needs full fire message (not available for third party developers)
                                  // mode 1 - Low Frequency Mode (wide aperture, navigation)
                                  // mode 2 - High Frequency Mode (narrow aperture, target identification)
    PingRateType pingRate;        // Sets the maximum ping rate.
    uint8_t networkSpeed;         // Used to reduce the network comms speed (useful for high latency shared links)
    uint8_t gammaCorrection;      // 0 and 0xff = gamma correction = 1.0
                                  // Set to 127 for gamma correction = 0.5
    uint8_t flags;                // bit 0: 0 = interpret range as percent, 1 = interpret range as meters
                                  // bit 1: 0 = 8 bit data, 1 = 16 bit data
                                  // bit 2: 0 = wont send gain, 1 = send gain
                                  // bit 3: 0 = send full return message, 1 = send simple return message
                                  // bit 4: 0 = gain assistance off, 1 = gain assistance on
                                  // bit 5: 0 = low power mode off, 1 = low power mode on
    double range;                 // The range demand in percent or meters depending on flags
    double gainPercent;           // The gain demand if gain assistance is off or intensity demand if gain assistance is on
    double speedOfSound;          // meters/second, if set to zero then internal calc will apply using salinity
    double salinity;              // ppt, set to zero if we are in fresh water and 35.0 if we are in salt water

    """

    def __init__(self):
        self.mode = None

        self.gamma = None
        self.flags = None
        self.range = None
        self.gain = None
        self.speed_of_sound = None
        self.salinity = None

    def configure(self, ping):
        self.mode = ping.fire_msg.mode
        self.gamma = ping.fire_msg.gamma / 255.0
        self.flags = ping.fire_msg.flags
        self.range = ping.fire_msg.range
        self.gain = ping.fire_msg.gain
        self.speed_of_sound = ping.fire_msg.speed_of_sound
        self.salinity = ping.fire_msg.salinity

    def __str__(self):
        return (
            "\n=========================\n"
            "   Oculus Fire Message\n"
            "=========================\n"
            "Mode: {mode:>19d}\n"
            "Gamma: {gamma:>18.1f}\n"
            "Flags: {flags:>18b}\n"
            "Range: {range:17.1f}m\n"
            "Gain: {gain:>19.1f}\n"
            "Speed of sound: {speed_of_sound:5.1f}m/s\n"
            "Salinity: {salinity:>12.1f}ppt\n"
            "=========================\n".format(**self.__dict__)
        )


class OculusProperty(object):
    OCULUS_VERTICAL_APERTURE = {1: np.deg2rad(20), 2: np.deg2rad(12)}
    OCULUS_PART_NUMBER = {1042: "M1200d", 1032: "M750d"}

    noise = 0.01
    # fmt: off
    psf = np.array([[0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.0005, 0.0005, 0.0005, 0.0005, 0.    , 0.0005, 0.0005, 0.0005,
                     0.0005, 0.    , 0.    , 0.0005, 0.0005, 0.    , 0.    , 0.    ,
                     0.001 , 0.001 , 0.001 , 0.001 , 0.    , 0.    , 0.001 , 0.001 ,
                     0.001 , 0.    , 0.    , 0.001 , 0.0015, 0.002 , 0.0015, 0.0005,
                     0.    , 0.001 , 0.002 , 0.0025, 0.002 , 0.001 , 0.001 , 0.002 ,
                     0.003 , 0.003 , 0.0015, 0.    , 0.0025, 0.005 , 0.005 , 0.0035,
                     0.002 , 0.0105, 0.022 , 0.0355, 0.049 , 0.0615, 0.071 , 0.076 ,
                     0.076 , 0.071 , 0.0615, 0.049 , 0.0355, 0.022 , 0.0105, 0.002 ,
                     0.0035, 0.005 , 0.005 , 0.0025, 0.    , 0.0015, 0.003 , 0.003 ,
                     0.002 , 0.001 , 0.001 , 0.002 , 0.0025, 0.002 , 0.001 , 0.    ,
                     0.0005, 0.0015, 0.002 , 0.0015, 0.001 , 0.    , 0.    , 0.001 ,
                     0.001 , 0.001 , 0.    , 0.    , 0.001 , 0.001 , 0.001 , 0.001 ,
                     0.    , 0.    , 0.    , 0.0005, 0.0005, 0.    , 0.    , 0.0005,
                     0.0005, 0.0005, 0.0005, 0.    , 0.0005, 0.0005, 0.0005, 0.0005,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                     0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ]])
    # fmt: on

    def __init__(self):
        # model
        self.model = ""
        self.fire_msg = OculusFireMsg()

        # range bins: [r1, ..., rn]
        self.num_ranges = None
        self.ranges = None
        # r[i] - r[i - 1]
        self.range_resolution = None
        # n * resolution
        self.max_range = 30.

        # bearings: [b1, ..., bm]
        self.num_bearings = None
        # rad
        self.bearings = None
        # b[m] - b[1]
        self.horizontal_aperture = np.radians(130.)
        # mean: (b[m] - b[1]) / m
        self.angular_resolution = None
        # rad
        self.vertical_aperture = None

        ##################################################
        # polar <-> Cartesian
        ##################################################
        # functions to transform between bearings and cols
        # col -> bearing
        self.c2b = None
        # col <- bearing
        self.b2c = None
        # row -> range
        self.ro2ra = None
        # row <- range
        self.ra2ro = None

        # parameters for remapping from polar to Cartesian
        self.remap_x = None
        self.remap_y = None

    def configure(self, ping):
        self.fire_msg.configure(ping)

        if "part_number" not in ping.__slots__:
            # backward compatibility
            self.model = "M750d"
        else:
            self.model = OculusProperty.OCULUS_PART_NUMBER[ping.part_number]

        changed = False
        if (
            ping.num_ranges != self.num_ranges
            or ping.range_resolution != self.range_resolution
        ):
            self.num_ranges = ping.num_ranges
            self.range_resolution = ping.range_resolution
            self.ranges = self.range_resolution * (1 + np.arange(self.num_ranges))
            self.max_range = self.ranges[-1]

            self.ro2ra = lambda ro: (ro + 1) * self.range_resolution
            self.ra2ro = lambda ra: np.round(ra / self.range_resolution - 1)
            changed = True

        if len(ping.bearings) != self.num_bearings:
            self.num_bearings = len(ping.bearings)
            self.bearings = np.deg2rad(np.array(ping.bearings, np.float32) / 100)
            self.horizontal_aperture = abs(self.bearings[-1] - self.bearings[0])
            self.angular_resolution = self.horizontal_aperture / self.num_bearings
            self.vertical_aperture = OculusProperty.OCULUS_VERTICAL_APERTURE[
                self.fire_msg.mode
            ]

            self.b2c = interp1d(
                self.bearings,
                np.arange(self.num_bearings),
                kind="cubic",
                bounds_error=False,
                fill_value=-1,
                assume_sorted=True,
            )
            self.c2b = interp1d(
                np.arange(self.num_bearings),
                self.bearings,
                kind="cubic",
                bounds_error=False,
                fill_value=-1,
                assume_sorted=True,
            )
            changed = True

        if changed:
            height = self.max_range
            rows = self.num_ranges
            width = np.sin((self.bearings[-1] - self.bearings[0]) / 2) * height * 2
            cols = int(np.ceil(width / self.range_resolution))

            XX, YY = np.meshgrid(range(cols), range(rows))
            x = self.range_resolution * (rows - YY)
            y = self.range_resolution * (-cols / 2.0 + XX + 0.5)
            b = np.arctan2(y, x)
            r = np.sqrt(x ** 2 + y ** 2)
            self.remap_y = np.asarray(self.ra2ro(r), dtype=np.float32)
            self.remap_x = np.asarray(self.b2c(b), dtype=np.float32)

        return changed

    def remap(self, ping=None, img=None):
        if img is None:
            img = r2n(ping)
        img = np.array(img, dtype=img.dtype, order="F")

        if self.remap_x.shape[1] > img.shape[1]:
            img.resize(*self.remap_x.shape)
        # Not too much difference between cubic and nearest
        img = cv2.remap(img, self.remap_x, self.remap_y, cv2.INTER_NEAREST)
        return img

    @staticmethod
    def adjust_gamma(img, gamma=1.0):
        return cv2.pow(img / 255.0, gamma) * 255.0

    def deconvolve(self, img):
        """Remove impulse response function from ping

        Copy from https://github.com/pvazteixeira/multibeam
        """
        img = np.float32(img)

        img_f = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        psf_padded = np.zeros_like(img)
        kh, kw = self.psf.shape
        psf_padded[:kh, :kw] = self.psf

        # compute (padded) psf's DFT
        psf_f = cv2.dft(psf_padded, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows=kh)

        psf_f_2 = (psf_f ** 2).sum(-1)
        ipsf_f = psf_f / (psf_f_2 + self.noise)[..., np.newaxis]

        result_f = cv2.mulSpectrums(img_f, ipsf_f, 0)
        result = cv2.idft(result_f, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        result = np.roll(result, -kh // 2, 0)
        result = np.roll(result, -kw // 2, 1)

        # clip to 0-1 range
        result[result < 0] = 0
        result = (np.max(img) / np.max(result)) * result

        return result.astype(np.float32)

    def polygon(self, origin=(0, 0, 0), angular_resolution=0.2):
        from shapely.geometry import Polygon
        from shapely.affinity import affine_transform

        points = [(0, 0)]
        for bearing in np.arange(
            self.bearings[0], self.bearings[-1], angular_resolution
        ):
            c, s = np.cos(bearing), np.sin(bearing)
            points.append((self.max_range * c, self.max_range * s))
        poly = Polygon(points)

        c, s = np.cos(origin[2]), np.sin(origin[2])
        params = c, -s, s, c, origin[0], origin[1]
        poly = affine_transform(poly, params)
        return poly

    def plot(self, origin=(0, 0, 0), ax=None, zdown=True, **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge
        if ax is None:
            ax = plt.gca()

        x, y, theta = origin
        min_bearing = self.bearings[0] + theta
        max_bearing = self.bearings[-1] + theta
        if zdown:
            x, y = y, x
            min_bearing, max_bearing = np.pi / 2 - max_bearing, np.pi / 2 - min_bearing
        fov = Wedge(
            (x, y),
            self.max_range,
            np.rad2deg(min_bearing),
            np.rad2deg(max_bearing),
            fill=False,
            **kwargs
        )
        ax.add_artist(fov)

    def __str__(self):
        fire_msg = str(self.fire_msg)
        d = dict(self.__dict__)
        d["angular_resolution"] = np.degrees(d["angular_resolution"])
        d["horizontal_aperture"] = np.degrees(d["horizontal_aperture"])
        d["vertical_aperture"] = np.degrees(d["vertical_aperture"])
        return (
            "\n===============================\n"
            "         Oculus Property\n"
            "===============================\n"
            "Model: {model:>24}\n"
            "#Ranges: {num_ranges:>22.0f}\n"
            "Range resolution: {range_resolution:>12.2f}m\n"
            "#Bemas: {num_bearings:>23}\n"
            "Angular resolution: {angular_resolution:>8.1f}deg\n"
            "Horizontal aperture: {horizontal_aperture:>7.1f}deg\n"
            "Vertical aperture: {vertical_aperture:>9.1f}deg\n"
            "===============================\n".format(**d) + fire_msg
        )
