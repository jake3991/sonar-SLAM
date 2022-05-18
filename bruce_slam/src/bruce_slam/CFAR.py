import math
import numpy as np
from scipy.optimize import root

from .utils import * 
from .sonar import *
from . import cfar

class CFAR(object):
    """
    Constant False Alarm Rate (CFAR) detection with several variants
        - Cell averaging (CA) CFAR
        - Greatest-of cell-averaging (GOCA) CFAR
        - Order statistic (OS) CFAR
    """

    def __init__(self, Ntc, Ngc, Pfa, rank=None):
        self.Ntc = Ntc #number of training cells
        assert self.Ntc % 2 == 0
        self.Ngc = Ngc #number of guard cells
        assert self.Ngc % 2 == 0
        self.Pfa = Pfa #false alarm rate
        if rank is None: #matrix rank
            self.rank = self.Ntc / 2
        else:
            self.rank = rank
            assert 0 <= self.rank < self.Ntc

        #threshold factor calculation for the 4 variants of CFAR
        self.threshold_factor_CA = self.calc_WGN_threshold_factor_CA()
        self.threshold_factor_SOCA = self.calc_WGN_threshold_factor_SOCA()
        self.threshold_factor_GOCA = self.calc_WGN_threshold_factor_GOCA()
        self.threshold_factor_OS = self.calc_WGN_threshold_factor_OS()

        self.params = {
            "CA": (self.Ntc // 2, self.Ngc // 2, self.threshold_factor_CA),
            "SOCA": (self.Ntc // 2, self.Ngc // 2, self.threshold_factor_SOCA),
            "GOCA": (self.Ntc // 2, self.Ngc // 2, self.threshold_factor_GOCA),
            "OS": (self.Ntc // 2, self.Ngc // 2, self.rank, self.threshold_factor_OS),
        }
        self.detector = {
            "CA": cfar.ca,
            "SOCA": cfar.soca,
            "GOCA": cfar.goca,
            "OS": cfar.os,
        }
        self.detector2 = {
            "CA": cfar.ca2,
            "SOCA": cfar.soca2,
            "GOCA": cfar.goca2,
            "OS": cfar.os2,
        }

    def __str__(self):
        return "".join(
            [
                "CFAR Detector Information\n",
                "=========================\n",
                "Number of training cells: {}\n".format(self.Ntc),
                "Number of guard cells: {}\n".format(self.Ngc),
                "Probability of false alarm: {}\n".format(self.Pfa),
                "Order statictics rank: {}\n".format(self.rank),
                "Threshold factors:\n",
                "      CA-CFAR: {:.3f}\n".format(self.threshold_factor_CA),
                "    SOCA-CFAR: {:.3f}\n".format(self.threshold_factor_SOCA),
                "    GOCA-CFAR: {:.3f}\n".format(self.threshold_factor_GOCA),
                "    OSCA-CFAR: {:.3f}\n".format(self.threshold_factor_OS),
            ]
        )

    def calc_WGN_threshold_factor_CA(self):
        return self.Ntc * (self.Pfa ** (-1.0 / self.Ntc) - 1)

    def calc_WGN_threshold_factor_SOCA(self):
        x0 = self.calc_WGN_threshold_factor_CA()
        for ratio in np.logspace(-2, 2, 10):
            ret = root(self.calc_WGN_pfa_SOCA, x0 * ratio)
            if ret.success:
                return ret.x[0]
        raise ValueError("Threshold factor of SOCA not found")

    def calc_WGN_threshold_factor_GOCA(self):
        x0 = self.calc_WGN_threshold_factor_CA()
        for ratio in np.logspace(-2, 2, 10):
            ret = root(self.calc_WGN_pfa_GOCA, x0 * ratio)
            if ret.success:
                return ret.x[0]
        raise ValueError("Threshold factor of GOCA not found")

    def calc_WGN_threshold_factor_OS(self):
        x0 = self.calc_WGN_threshold_factor_CA()
        for ratio in np.logspace(-2, 2, 10):
            ret = root(self.calc_WGN_pfa_OS, x0 * ratio)
            if ret.success:
                return ret.x[0]
        raise ValueError("Threshold factor of OS not found")

    def calc_WGN_pfa_GOSOCA_core(self, x):
        x = float(x)
        temp = 0.0
        for k in range(int(self.Ntc / 2)):
            l1 = math.lgamma(self.Ntc / 2 + k)
            l2 = math.lgamma(k + 1)
            l3 = math.lgamma(self.Ntc / 2)
            temp += math.exp(l1 - l2 - l3) * (2 + x / (self.Ntc / 2)) ** (-k)
        return temp * (2 + x / (self.Ntc / 2)) ** (-self.Ntc / 2)

    def calc_WGN_pfa_SOCA(self, x):
        return self.calc_WGN_pfa_GOSOCA_core(x) - self.Pfa / 2

    def calc_WGN_pfa_GOCA(self, x):
        x = float(x)
        temp = (1.0 + x / (self.Ntc / 2)) ** (-self.Ntc / 2)
        return temp - self.calc_WGN_pfa_GOSOCA_core(x) - self.Pfa / 2

    def calc_WGN_pfa_OS(self, x):
        l1 = math.lgamma(self.Ntc + 1)
        l2 = math.lgamma(self.Ntc - self.rank + 1)
        l4 = math.lgamma(x + self.Ntc - self.rank + 1)
        l6 = math.lgamma(x + self.Ntc + 1)
        return math.exp(l1 - l2 + l4 - l6) - self.Pfa

    def detect(self, mat, alg="CA"):
        """
        Return target mask array.
        """
        return self.detector[alg](mat, *self.params[alg])

    def detect2(self, mat, alg="CA"):
        """
        Return target mask array and threshold array.
        """
        return self.detector2[alg](mat, *self.params[alg])