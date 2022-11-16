import numpy as np
import cv2
import os

from mpl_toolkits import mplot3d
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.ndimage import measurements

class Event:
    def __init__(self, x, y, polarity, t=0) -> None:
        self.x = int(x)
        self.y = int(y)
        self.polarity = polarity
        self.t = t/1e+8

class HeatMap:
    def __init__(self, w, h, alpha=12, sigma=3, tau=18, ho=0.9) -> None:
        self.w = w
        self.h = h
        self.map = np.zeros((self.w, self.h),dtype=np.float32)

        self.alpha = alpha
        self.sigma = sigma
        self.tau = tau
        self.ho = ho

        self.t = None

    def addEvent(self, e):

        if self.t is None:
            self.t = e.t
        else:
            # print(np.exp(-self.tau * (e.t - self.t)), self.t, e.t)
            if (e.t - self.t) < 0:
                # print("################################################")
                self.t = e.t

            self.map *= np.exp(-self.tau * (e.t - self.t))
            self.t = e.t

        event_map = np.zeros((self.w,self.h))
        event_map[e.x, e.y] = 1
        gaussian_peak = gaussian_filter(event_map, sigma=self.sigma)

        self.map += self.alpha * gaussian_peak

    def getMap(self):
        # a = self.map.copy()
        # b  = a.flatten()

        # b.sort()
        # b = b[::-1]
        # print(b[:10])
        return self.map

    def getCornerMap(self):
        return np.where(self.map > self.ho, 1, 0)

    def getCenter(self):

        x, y = np.where(self.getCornerMap() == 1)

        if x.size == 0:
            return None

        x_mean, y_mean, x_var, y_var = x.mean(), y.mean(), x.var(), y.var()

        return x_mean, y_mean

    def getCornerClusters(self):
        cmap = self.getCornerMap()
        lw, num = measurements.label(cmap)
        return lw, num

    def getCorners(self):
        lw, num = self.getCornerClusters()

        coord_x, coord_y = [], []
        for i in range(num):
            x, y = np.where(lw == i+1)

            if x.size == 0:
                return None

            x_mean, y_mean, x_var, y_var = x.mean(), y.mean(), x.var(), y.var()
            coord_x.append(int(x_mean))
            coord_y.append(int(y_mean))

        # print(coord_x)

        corner_map = np.zeros((self.w,self.h))
        corner_map[coord_x, coord_y] = 1
        
        return corner_map




# hm = HeatMap(64,64)

# for i in range(100):
#     hm.addEvent(Event((5 + 3*i)%62, 32,1, 1 * i*1e8))

# # hm.addEvent(e)

# img = hm.getMap()
# plt.imshow(img)
# plt.show()